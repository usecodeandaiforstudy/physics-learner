"""A v2 神经残余·IB 曲线代理（M3，2026-09-29）。

学习目标（design doc §2.2）：IB 曲线 I_XT(β)/I_TY(β)——慢路径是 Blahut–Arimoto
（每曲线 32×BA，推理期太慢），代理在推理期替代。**解析背板输出作特征**
（节点 Fisher + 边 MI + 族 + 扫描范围），不碰慢路径、不碰扫描值：
    features = [fam_idx, scan_lo, scan_hi, trace_1..3, det_1..3, mi_1..2]（11 维）
    target   = ln(1 + I_XT(β)), ln(1 + I_TY(β))（32+32 点，ln(1+I) 空间）

**DPI 单调硬构造**（M2 遗留项在此落地）：曲线用「起点 + 非负增量累加」参数化
（softplus 增量），I_XT/I_TY 随 β 单调非降**由构造保证**——顺带正则化掉
第 34 条发现的 BA 大 β 尾部收敛伪影（代理曲线不会出现 β>226 的下降）。

**双条件门控**（design doc §2.5，第 32 条）：不确定度 = 3 模型集成方差；
阈值只在训练系统上标定（防循环论证）；`covered=False` 无条件拒绝。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

__all__ = ["MonotonicCurveNet", "build_dataset", "train_ensemble",
           "predict_ensemble", "calibrate_gate", "apply_gate",
           "features_from_backbone", "FEATURE_DIM", "N_POINTS"]

N_POINTS = 32                 # IB beta 网格点数（conventions：logspace(-2,3,32)）
FEATURE_DIM = 11
_MAX_NODES = 3
_MAX_EDGES = 2


def _family_index(fam: str, families: list[str]) -> int:
    return families.index(fam) if fam in families else 0


def features_from_backbone(fam: str, lo: float, hi: float,
                           node_fisher: dict, mi_list: list,
                           families: list[str]) -> list[float]:
    """从解析背板输出组装 11 维特征（环境推理路径与训练路径共用同一布局）。"""
    feats: list[float] = [float(_family_index(fam, families)),
                          float(lo), float(hi)]
    for nid in sorted(node_fisher)[:_MAX_NODES]:
        f = node_fisher[nid]
        feats += [float(f["fisher_trace"]), float(f["fisher_det"])]
    while len(feats) < 3 + 2 * _MAX_NODES:
        feats.append(0.0)
    for k in range(_MAX_EDGES):
        mi = mi_list[k] if k < len(mi_list) else None
        feats.append(float(mi) if mi is not None else 0.0)
    return feats


def _featurize(row: dict, entry: dict, families: list[str]) -> list[float]:
    """DB2 行 + 条目 → 11 维特征（全部来自解析背板可算的廉价量）。"""
    lo = hi = 0.0
    sp = entry.get("scan_param")
    if sp:
        for p in entry.get("parameters", []):
            if p.get("name") == sp:
                rng = p.get("range") or [0, 0]
                lo, hi = float(rng[0]), float(rng[1])
    fps = row.get("fingerprints", {})
    mi = list(row.get("mi_profile") or [])
    return features_from_backbone(row["family"], lo, hi, fps, mi, families)


def build_dataset(db2_rows: list[dict], entries: dict[str, dict],
                  families: list[str]) -> dict:
    """从 DB2 组装 (特征, 目标) 数据集；只取 IB 曲线非空的行走。"""
    xs, ys, sids = [], [], []
    n_skipped = 0
    for row in db2_rows:
        ic = row.get("ib_curve") or {}
        if len(ic.get("beta") or []) != N_POINTS:
            n_skipped += 1
            continue
        if len(ic.get("I_XT") or []) != N_POINTS or len(ic.get("I_TY") or []) != N_POINTS:
            n_skipped += 1
            continue
        entry = entries.get(row["system_id"])
        if entry is None:
            n_skipped += 1
            continue
        x = _featurize(row, entry, families)
        y = np.concatenate([
            np.log1p(np.asarray(ic["I_XT"], dtype=float)),
            np.log1p(np.asarray(ic["I_TY"], dtype=float)),
        ])
        xs.append(x)
        ys.append(y)
        sids.append(row["system_id"])
    return {"X": np.asarray(xs, dtype=np.float32),
            "Y": np.asarray(ys, dtype=np.float32),
            "system_ids": sids, "n_skipped": n_skipped}


class MonotonicCurveNet(nn.Module):
    """单调 IB 曲线代理：起点 + 非负增量累加（I_XT/I_TY 随 β 单调非降）。"""

    def __init__(self, in_dim: int = FEATURE_DIM, n_points: int = N_POINTS,
                 hidden: int = 64):
        super().__init__()
        self.n_points = n_points
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.start = nn.Linear(hidden, 2)                  # (I_XT, I_TY) 起点
        self.inc = nn.Linear(hidden, 2 * (n_points - 1))   # 每段非负增量

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        s = self.start(h)                                  # (B, 2)
        inc = F.softplus(self.inc(h)).view(x.shape[0], 2, self.n_points - 1)
        curve = torch.cat([s.unsqueeze(-1),
                           s.unsqueeze(-1) + torch.cumsum(inc, dim=-1)],
                          dim=-1)                          # (B, 2, N) 单调非降
        return curve


def _standardize(X: np.ndarray, mu: np.ndarray | None = None,
                 sd: np.ndarray | None = None):
    if mu is None:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-6] = 1.0
    return (X - mu) / sd, mu, sd


def train_ensemble(Xtr: np.ndarray, Ytr: np.ndarray, seeds=(0, 1, 2),
                   epochs: int = 400, lr: float = 1e-3,
                   point_weights: np.ndarray | None = None) -> tuple:
    """训练 3 模型集成（不同 seed 初始化 + 数据顺序）。

    point_weights : (2, N) 逐点损失权重（如操作区 β 加权）；None = 均匀。
    """
    Xs, mu, sd = _standardize(Xtr)
    Xt = torch.tensor(Xs)
    Yt = torch.tensor(Ytr)
    Wt = None
    if point_weights is not None:
        Wt = torch.tensor(np.asarray(point_weights, dtype=np.float32))
    models = []
    for seed in seeds:
        torch.manual_seed(seed)
        net = MonotonicCurveNet()
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        n = Xt.shape[0]
        order = np.random.RandomState(seed).permutation(n)
        bs = 128
        for ep in range(epochs):
            net.train()
            for i in range(0, n, bs):
                idx = order[i:i + bs]
                opt.zero_grad()
                out = net(Xt[idx])                       # (B, 2, N)
                sq = (out - Yt[idx].view(-1, 2, N_POINTS)) ** 2
                if Wt is not None:
                    sq = sq * Wt
                loss = sq.mean()
                loss.backward()
                opt.step()
        net.eval()
        models.append(net)
    return models, mu, sd


def predict_ensemble(models, mu, sd, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """集成预测：均值 + 标准差（ln(1+I) 空间，形状 (n, 2, N)）。"""
    Xs = (X - mu) / sd
    Xt = torch.tensor(Xs, dtype=torch.float32)
    outs = []
    with torch.no_grad():
        for net in models:
            outs.append(net(Xt).numpy())
    outs = np.stack(outs)                                # (M, n, 2, N)
    return outs.mean(axis=0), outs.std(axis=0)


def calibrate_gate(train_std: np.ndarray, quantile: float = 0.90) -> float:
    """在训练系统上标定不确定度阈值（防循环论证：绝不用评估数据标定）。"""
    return float(np.quantile(train_std, quantile))


def apply_gate(covered: np.ndarray, std_sys: np.ndarray,
               threshold: float) -> np.ndarray:
    """双条件门控（design doc §2.5）：covered=False 或 高不确定度 → 拒绝。

    返回接受掩码（True=接受）。covered=False 无条件拒绝；covered=True 时
    系统级不确定度超过训练集标定阈值 → 拒绝。
    """
    return np.asarray(covered) & (std_sys <= threshold)
