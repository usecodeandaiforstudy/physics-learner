# -*- coding: utf-8 -*-
"""B v1 置信度标定（温度缩放，2026-10-02，第 78 条下一步 ③）。

动机：held-out 新表述集实测显示 **v1 置信度系统性偏低**（预测常对但 conf
低，如 PARA-10 conf=0.07 / PARA-12 conf=0.15），导致「弃权→人工」的自动
覆盖率被压缩。温度缩放（dividing logits by T）是标准的事后标定方法：
不改变预测（T>0 保序），只让置信度与正确率对齐。

协议（防泄漏）：
  - **5 折 CV**：每折在 4/5 锚点上训练模型、在留出折上**拟合 T**
    （最小化属性头 BCE + 方向头 CE 的加权 NLL）；
  - 指标：**ECE**（10 桶期望标定误差，正确性 = 属性集合完全匹配）
    与**覆盖率@目标精度**（选 τ 使自动段精度 ≥ 95%，看覆盖率）；
  - 产出：`bv1_model_cal.pt`（含 `temperature` = 各折 T 均值），
    供 `Bv1Parser(..., temperature=T)` 使用。

用法（项目根目录）：python b_network/bv1_calibrate.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from b_network.bv1_dataset import PROPERTIES, load_anchors
from b_network.bv1_parse import Bv1Model, build_vocab, featurize, input_dim
from b_network.bv1_train_eval import train_model

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
MODEL_IN = ROOT / "b_network" / "bv1_model.pt"
MODEL_OUT = ROOT / "b_network" / "bv1_model_cal.pt"


def ece(confs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """期望标定误差（等宽分桶；correct 为该条属性集合是否完全匹配）。"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confs)
    err = 0.0
    for i in range(n_bins):
        m = (confs > bins[i]) & (confs <= bins[i + 1])
        if not m.any():
            continue
        acc = float(correct[m].mean())
        conf = float(confs[m].mean())
        err += (m.sum() / total) * abs(acc - conf)
    return err


def coverage_at_precision(confs: np.ndarray, correct: np.ndarray,
                          target: float = 0.95) -> tuple[float, float]:
    """选 (τ) 使自动段精度 ≥ target，返回 (覆盖率, 实际精度, τ)。"""
    best = (0.0, 1.0, 1.01)
    for tau in np.linspace(0.05, 0.99, 95):
        m = confs >= tau
        if m.sum() == 0:
            continue
        prec = float(correct[m].mean())
        cov = float(m.mean())
        if prec >= target and cov > best[0]:
            best = (cov, prec, float(tau))
    return best


def fit_platt_on_maxprob(confs: np.ndarray,
                         correct: np.ndarray) -> tuple[float, float]:
    """Platt 缩放：直接标定**弃权统计量**（最大概率 → 正确率）。

    sigmoid(a·logit(c) + b) ≈ P(属性集合完全匹配 | c)。
    与温度缩放的差别：温度优化逐标签 BCE（**目标错配**——弃权用的是最大
    概率）；Platt 直接优化「最大概率是否可信」，与生产判据一致。
    """
    eps = 1e-6
    c = np.clip(confs, eps, 1 - eps)
    x = np.log(c / (1 - c))            # logit
    a, b = 1.0, 0.0
    for _ in range(300):               # 简单梯度下降（样本极少，够用）
        z = a * x + b
        p = 1.0 / (1.0 + np.exp(-z))
        g_a = float(np.mean((p - correct) * x))
        g_b = float(np.mean(p - correct))
        a -= 0.5 * g_a
        b -= 0.5 * g_b
    return float(a), float(b)


def fit_temperature(logits_p: np.ndarray, y_p: np.ndarray,
                    logits_d: np.ndarray, y_d: np.ndarray) -> float:
    """拟合单一温度 T（属性头 BCE + 方向头 CE 联合 NLL 最小化）。"""
    lp = torch.tensor(logits_p, dtype=torch.float32)
    yp = torch.tensor(y_p, dtype=torch.float32)
    ld = torch.tensor(logits_d, dtype=torch.float32)
    yd = torch.tensor(y_d, dtype=torch.long)

    def nll(t: float) -> float:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            lp / t, yp).item()
        ce = torch.nn.functional.cross_entropy(ld / t, yd).item()
        return bce + 0.5 * ce

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(nll, bounds=(0.2, 10.0), method="bounded")
    return float(res.x)


def main() -> int:
    anchors = load_anchors()
    n = len(anchors)
    vocab = build_vocab([a["text"] for a in anchors])
    in_dim = input_dim(vocab)   # 含线索块（第 193 条）
    Xs = np.stack([featurize(a["text"], vocab) for a in anchors])
    y_prop = np.zeros((n, len(PROPERTIES)), dtype=np.float32)
    for i, a in enumerate(anchors):
        for p in a["properties"]:
            y_prop[i, PROPERTIES.index(p)] = 1.0
    y_dir = np.array([0 if a["direction"] in (None, "bound_below") else 1
                      for a in anchors], dtype=np.int64)

    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)

    Ts, rows_before, rows_after, rows_platt = [], [], [], []
    rows_margin: list[tuple[float, float, float]] = []
    rows_neg_ent: list[tuple[float, float, float]] = []
    Platt: list[tuple[float, float]] = []
    print("================ B v1 温度标定（5 折 CV） ================")
    for k, val_idx in enumerate(folds):
        tr_idx = np.setdiff1d(np.arange(n), val_idx)
        model = train_model(Xs[tr_idx], y_prop[tr_idx], y_dir[tr_idx], in_dim)
        with torch.no_grad():
            lp, ld = model(torch.tensor(Xs[val_idx]))
        lp_np, ld_np = lp.numpy(), ld.numpy()
        probs = 1.0 / (1.0 + np.exp(-lp_np))
        confs = probs.max(axis=1)
        preds = (probs > 0.5).astype(np.float32)
        correct = np.array([set(np.array(PROPERTIES)[preds[i] > 0])
                            == set(anchors[j]["properties"])
                            for i, j in enumerate(val_idx)])
        T = fit_temperature(lp_np, y_prop[val_idx], ld_np, y_dir[val_idx])
        Ts.append(T)
        probs_c = 1.0 / (1.0 + np.exp(-lp_np / T))
        confs_c = probs_c.max(axis=1)
        # Platt（直接标定弃权统计量）
        a, b = fit_platt_on_maxprob(confs, correct)
        confs_p = 1.0 / (1.0 + np.exp(-(a * np.log(
            np.clip(confs, 1e-6, 1 - 1e-6) / (1 - np.clip(
                confs, 1e-6, 1 - 1e-6))) + b)))
        Platt.append((a, b))
        rows_before.append((ece(confs, correct),
                            coverage_at_precision(confs, correct)))
        rows_after.append((ece(confs_c, correct),
                           coverage_at_precision(confs_c, correct)))
        rows_platt.append((ece(confs_p, correct),
                           coverage_at_precision(confs_p, correct)))
        # 备选置信度统计量（**单调标定不改排序 → 只能靠换统计量提升覆盖**）
        srt = np.sort(probs, axis=1)
        margin = srt[:, -1] - srt[:, -2]                 # top1−top2 间隔
        p = np.clip(probs, 1e-9, 1 - 1e-9)
        # 负熵统计量归一化到 [0,1]（原实现落在负区间 → 阈值法全不通过，
        # 2026-10-02 修正）：1 + mean(p ln p + (1-p) ln(1-p))/ln2
        neg_ent = 1.0 + (p * np.log(p) + (1 - p) * np.log(1 - p)).mean(
            axis=1) / np.log(2.0)
        rows_margin.append(coverage_at_precision(margin, correct))
        rows_neg_ent.append(coverage_at_precision(neg_ent, correct))
        print("  折 %d：T=%.3f  Platt(a=%.2f,b=%.2f)  ECE %.3f→%.3f→%.3f  "
              "覆盖@95%%精度 %.0f%%→%.0f%%→%.0f%%"
              % (k + 1, T, a, b, rows_before[-1][0], rows_after[-1][0],
                 rows_platt[-1][0], 100 * rows_before[-1][1][0],
                 100 * rows_after[-1][1][0], 100 * rows_platt[-1][1][0]))

    T_mean = float(np.mean(Ts))
    e_b = float(np.mean([r[0] for r in rows_before]))
    e_a = float(np.mean([r[0] for r in rows_after]))
    e_p = float(np.mean([r[0] for r in rows_platt]))
    c_b = float(np.mean([r[1][0] for r in rows_before]))
    c_a = float(np.mean([r[1][0] for r in rows_after]))
    c_p = float(np.mean([r[1][0] for r in rows_platt]))
    a_mean = float(np.mean([x[0] for x in Platt]))
    b_mean = float(np.mean([x[1] for x in Platt]))
    print("  均值：T=%.3f（折间 %.2f–%.2f，**极不稳定 → 9 条验证样本不足以"
          "拟合温度**）" % (T_mean, min(Ts), max(Ts)))
    print("    ECE：未标定 %.3f | 温度 %.3f | **Platt %.3f**"
          % (e_b, e_a, e_p))
    print("    覆盖@95%%精度：未标定 %.0f%% | 温度 %.0f%% | **Platt %.0f%%**"
          % (100 * c_b, 100 * c_a, 100 * c_p))
    print("    Platt 参数均值：a=%.2f, b=%.2f（a>1 = 原置信度区分度不足，"
          "标定后拉开）" % (a_mean, b_mean))
    c_m = float(np.mean([r[0] for r in rows_margin]))
    c_n = float(np.mean([r[0] for r in rows_neg_ent]))
    print("    **统计量对比（覆盖@95%%精度；标定不改排序故与未标定同值）**："
          "max_prob %.0f%% | top1−top2 间隔 %.0f%% | 负熵 %.0f%%"
          % (100 * c_b, 100 * c_m, 100 * c_n))

    # 产出标定模型（沿用既有全量模型权重 + CV 平均 Platt 参数）
    ckpt = torch.load(MODEL_IN, map_location="cpu")
    # **温度标定不采纳**（第 79 条：目标错配使 ECE 反向恶化 → 置 1.0 不生效）；
    # 只采纳 Platt——它直接标定弃权统计量，且 Platt 参数是在**未缩放**的
    # max_prob 上拟合的，叠加温度会使拟合失效（2026-10-02 发现的双重标定坑）
    ckpt["temperature"] = 1.0
    ckpt["platt"] = {"a": a_mean, "b": b_mean}
    ckpt["calibration"] = {"method": "platt_on_maxprob(5fold CV); temperature "
                                     "measured but NOT adopted (see item 79)",
                           "fold_T": Ts, "fold_platt": Platt,
                           "ece": {"raw": e_b, "temp": e_a, "platt": e_p},
                           "coverage_at_95": {"raw": c_b, "temp": c_a,
                                              "platt": c_p}}
    torch.save(ckpt, MODEL_OUT)
    print("  标定模型已写入 %s（platt a=%.2f b=%.2f；temperature=1.0 不采纳）"
          % (MODEL_OUT, a_mean, b_mean))
    return 0


if __name__ == "__main__":
    sys.exit(main())
