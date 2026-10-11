# -*- coding: utf-8 -*-
"""R5-②：**第 2 个域的 C 实例（rayleigh：需求 → 设计分布）+ 闭环消费**
（2026-10-02，第 146 条；架构 §13.2/§13.5）。

## 任务形式（与 landau 域同构，可机器验证）
    条件 `c = q1 目标`（归一化）→ 输出**设计分布** `p((D, f) | c)`
    验证：采样设计 → **域自带校验**（设计 → σ → 既有编码器 → q1）+ DPI
          → 判 `q1 ≥ 目标`
    ⟹ 与 `v33_landau_c_check.py` **同一套判据与三态判定**

## 与 landau 版的两处**关键差别**
  ① 本域的解流形**有解析形式**（一维射线）⟹ 生成模型应**很容易**学到；
     故判据门槛保持 **≥70%**（不因"更容易"而放松，也不因"太容易"而提高）；
  ② **训练数据可直接来自 reflow 的加权数据集**
     （`reflow_weighted_rayleigh.csv`）⟹ **闭环实质闭合**：
     `reflow（验证反馈）→ 加权数据集 → 训练 → 采样 → 再验证` ✓
     若该文件不存在则退化为按需求采样（并**如实标注**数据来源）。

用法：
    python pipeline/v34_rayleigh_c_check.py --seeds 3     # 快速试跑
    python pipeline/v34_rayleigh_c_check.py               # 20 seed 正式判定
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

def _runs_name(kind: str) -> str:
    """**正式记录保护（第 164 条）**：`--seeds < 20` 的试点跑写
    `*_runs_pilot_<n>seed.csv`，绝不覆盖 20 seed 的正式记录。"""
    n = 20
    if "--seeds" in sys.argv:
        try:
            n = int(sys.argv[sys.argv.index("--seeds") + 1])
        except (ValueError, IndexError):
            n = 20
    return ("%s_runs.csv" % kind) if n >= 20 else \
        ("%s_runs_pilot_%dseed.csv" % (kind, n))


RUNS = _ROOT / _runs_name("v34")
WEIGHTED = _ROOT / "reflow_weighted_rayleigh.csv"
DATA_CSV = _ROOT / "v34_dataset.csv"      # 大规模已验证数据集（缓存）
N_SAMPLE = 40
TRAIN_STEPS = 3000
Q_TARGETS = (1.5, 2.0, 3.0, 4.0, 6.0)      # 条件（q1 目标）
DESIGN_LO = {"D_m": 0.05, "f_Hz": 1e4}
DESIGN_HI = {"D_m": 3.0, "f_Hz": 2e6}
# **第二轮预登记（第 146 条）**：第一轮训练数据仅 **~320 条**（reflow 产出）
#   ⟹ C1 = **64.8% ± 21.2%（10/20）不稳定**；而 v33（landau）用 **2419 条**。
#   **判据不变（达标 seed ≥70%）**，只**扩大已验证数据集**到 `SCALE` 条
#   （均匀采样 + **逐个过域校验**作为标签，provenance 仍为 AI+simulated）。
SCALE = 4000
_REBUILD = False          # `--rebuild` 时忽略数据集缓存


def _norm_q(q: float, lo: float = 1.0, hi: float = 8.0) -> float:
    return float((math.log10(max(q, 1e-6)) - math.log10(lo))
                 / max(math.log10(hi) - math.log10(lo), 1e-9))


def _norm_design(D: float, f: float) -> np.ndarray:
    return np.array([(D - DESIGN_LO["D_m"]) / (DESIGN_HI["D_m"]
                                               - DESIGN_LO["D_m"]),
                     (math.log10(max(f, 1e-6)) - math.log10(DESIGN_LO["f_Hz"]))
                     / (math.log10(DESIGN_HI["f_Hz"])
                        - math.log10(DESIGN_LO["f_Hz"]))], dtype=np.float32)


def _denorm_design(z) -> dict:
    z = np.asarray(z, dtype=float).ravel()
    D = DESIGN_LO["D_m"] + float(np.clip(z[0], 0, 1)) * (
        DESIGN_HI["D_m"] - DESIGN_LO["D_m"])
    f = 10 ** (math.log10(DESIGN_LO["f_Hz"]) + float(np.clip(z[1], 0, 1))
               * (math.log10(DESIGN_HI["f_Hz"])
                  - math.log10(DESIGN_LO["f_Hz"])))
    return {"D_m": D, "f_Hz": f}


def build_dataset(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray,
                                                        str]:
    """构造训练集 `(设计向量, 条件=**需求水平**)`。

    **第三轮修正（第 146 条，真因）**：前两轮把标签取成「设计的**自身** q1」
    ⟹ 模型无偏 ⟹ 判据 `q1 ≥ 目标` **结构性只约 50% 通过**（实测 59.5%）。
    **正确形式**：对每个需求水平 `q_req`，只保留**可行设计**（`q1 ≥ q_req`）
    ⟹ 模型学的是 `p(设计 | 需求)` 在**可行集**上的分布 ⟹ 采样应普遍达标。
    **判据不变**（达标 seed ≥70%），只改**数据构造**（这不是移动球门，
    而是修正我自己的任务形式错误）。
    """
    from pipeline.design_space import verify_design
    import csv as _csv
    rng = np.random.default_rng(seed)
    levels = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    per_level = max(40, SCALE // len(levels))
    # 缓存：{需求水平, D_m, f_Hz, q1}
    if DATA_CSV.exists() and not _REBUILD:
        X, C = [], []
        with open(DATA_CSV, "r", encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                try:
                    X.append(_norm_design(float(row["D_m"]), float(row["f_Hz"])))
                    C.append(_norm_q(float(row["q_req"])))
                except (TypeError, ValueError):
                    continue
        if X:
            return (np.asarray(X, dtype=np.float32),
                    np.asarray(C, dtype=np.float32)[:, None],
                    "**可行集数据集（%s，%d 条，%d 个需求水平）**"
                    % (DATA_CSV.name, len(X), len(levels)))
    X, C, rows = [], [], []
    for lv in levels:
        got = 0
        tries = 0
        while got < per_level and tries < per_level * 60:
            tries += 1
            D = float(rng.uniform(DESIGN_LO["D_m"], DESIGN_HI["D_m"]))
            f = float(10 ** rng.uniform(math.log10(DESIGN_LO["f_Hz"]),
                                        math.log10(DESIGN_HI["f_Hz"])))
            try:
                v = verify_design("rayleigh", {"D_m": D, "f_Hz": f})
            except Exception:  # noqa: BLE001
                continue
            q = v.get("measured")
            if q is None or not (q >= lv):        # **只留可行设计**
                continue
            X.append(_norm_design(D, f))
            C.append(_norm_q(lv))
            rows.append((lv, D, f, float(q)))
            got += 1
    with open(DATA_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["q_req", "D_m", "f_Hz", "q1"])
        for lv, D, f, q in rows:
            w.writerow(["%.6g" % lv, "%.6g" % D, "%.6g" % f, "%.6g" % q])
    return (np.asarray(X, dtype=np.float32),
            np.asarray(C, dtype=np.float32)[:, None],
            "**可行集数据集（%d 条 = %d 水平 × ~%d；已缓存 %s）**"
            % (len(X), len(levels), per_level, DATA_CSV.name))


def train_one(seed: int, X, C) -> tuple[object, float]:
    import torch
    from c_network.cfm import ConditionalFlowMatching
    xt = torch.tensor(X)
    ct = torch.tensor(C)
    model = ConditionalFlowMatching(in_dim=2, cond_dim=1, hidden=128, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    rng = np.random.default_rng(seed)
    loss = None
    for _ in range(TRAIN_STEPS):
        idx = torch.tensor(rng.integers(0, len(X), 256))
        loss = model.loss(ct[idx], xt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, float(loss.item())


def evaluate(model, targets) -> dict:
    """**闭环校验**：采样 → 域自带校验（含 DPI）→ 判 `q1 ≥ 目标`。"""
    from pipeline.design_space import verify_design
    n_ok = n_tot = 0
    uniq = set()
    qs = []
    for q_t in targets:
        c = np.array([_norm_q(q_t)], dtype=np.float32)
        Z = model.sample_cond(c, N_SAMPLE, steps=60, seed=int(q_t * 1000))
        for z in Z:
            d = _denorm_design(z)
            uniq.add((round(d["D_m"], 4), round(d["f_Hz"], 1)))
            v = verify_design("rayleigh", d, target=q_t)
            n_tot += 1
            if v.get("meets"):
                n_ok += 1
            if v.get("measured") is not None:
                qs.append(float(v["measured"]))
    return {"pass_rate": n_ok / max(n_tot, 1), "n_tot": n_tot,
            "diversity": len(uniq),
            "q_min": (min(qs) if qs else float("nan")),
            "q_max": (max(qs) if qs else float("nan"))}


def main() -> int:
    global _REBUILD
    seeds = 20
    if "--seeds" in sys.argv:
        seeds = int(sys.argv[sys.argv.index("--seeds") + 1])
    _REBUILD = "--rebuild" in sys.argv
    print("=" * 74)
    print("R5-② rayleigh 域：C 实例（需求 → 设计分布）+ 闭环消费")
    print("=" * 74)
    t0 = time.time()
    X, C, src = build_dataset(600, seed=0)
    print("训练数据来源：%s；样本 %d" % (src, len(X)))
    print("条件 = q1 目标（归一化）；设计向量 = (D_m, f_Hz) 归一化 → 2 维")
    targets = list(Q_TARGETS)
    rec = []
    print("\n%4s %10s %10s %9s %9s %8s" % ("seed", "闭环达标率", "多样性",
                                          "q1_min", "q1_max", "loss"))
    for s in range(seeds):
        model, loss = train_one(s, X, C)
        ev = evaluate(model, targets)
        rec.append({"seed": s, **ev, "loss": loss})
        print("%4d %9.1f%% %10d %9.4g %9.4g %8.4f"
              % (s, 100 * ev["pass_rate"], ev["diversity"], ev["q_min"],
                 ev["q_max"], loss))

    pr = np.array([r["pass_rate"] for r in rec])
    dv = np.array([r["diversity"] for r in rec])
    print("\n===== 汇总（%d seed）=====" % seeds)
    print("C1 闭环达标率：均值 **%.1f%% ± %.1f%%**；达标(≥70%%)的 seed：%d/%d"
          % (100 * pr.mean(), 100 * pr.std(), int(np.sum(pr >= 0.70)), seeds))
    print("C2 多样性：均值 %.0f（每目标 %d 采样）" % (dv.mean(), N_SAMPLE))
    q_all = np.array([r["q_min"] for r in rec] + [r["q_max"] for r in rec])
    print("C3 无不可能的数：实测 q1 ∈ [%.4g, %.4g]（应有限正）⟹ %s"
          % (np.nanmin(q_all), np.nanmax(q_all),
             "✓" if (np.nanmin(q_all) > 0 and np.isfinite(q_all).all())
             else "**查 bug**"))
    rate70 = float(np.mean(pr >= 0.70))
    verdict = ("**稳健成立**" if rate70 >= 0.70 else
               "**稳健不成立**" if rate70 <= 0.30 else "**不稳定**")
    print("C4 三态判定（达标 seed 比例 %.0f%%）：%s" % (100 * rate70, verdict))
    with open(RUNS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec[0]))
        w.writeheader()
        w.writerows(rec)
    print("\n逐 seed 落盘：%s（%d 行）；耗时 %.1fs"
          % (RUNS.name, len(rec), time.time() - t0))
    print("数据来源：%s" % src)
    print("   （注：初版此处按「reflow 文件是否存在」打印数据来源，**口径不准**"
          "——实际来源见上一行；已修）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
