# -*- coding: utf-8 -*-
"""任务 A：L2 重做——用**路径相关目标量**检验变分特征的增量价值
（2026-10-02，第 119 条）。

**为什么要重做**：第 117 条的 L2 用「逐点静态量 `I_TY@β=4`」作目标，而变分
特征描述的是**沿参数路径的度规形状**——**用路径特征预测点值结构上注定无
增量**（基线 R² 已达 0.9706，天花板效应；实测 Δ=−0.0018，0/20）。

**本次的目标量（路径相关）**：**跨族迁移差距**——
  对每行，用**留一族**的岭回归预测其 `I_TY@4`，取**绝对残差**作目标
  `gap_i`（"该设计在族外模型下的可预测性缺口"）。
  ⟹ 问题变成：**信息几何距离能否解释这个缺口？**
  这正是架构关心的「C 的 ×N 与跨域迁移」在数据侧的对应量。

**预登记判据**：
  W1 **解释力**：`gap` 与变分特征（测地线长度 L、极值作用量 S*）的
     Spearman ρ ≥ **0.30**，且 **p < 0.05**；
  W2 **非伪相关**：对 `gap` 做 **1000 次打乱**的零假设检验，
     真实 ρ 须超过零分布的 **95 分位**；
  W3 **方向**：ρ **为正**（离族基准越"远"的设计，迁移缺口越大）——
     若显著为负，同样如实报告（说明方向假设错了）；
  W4 **20 bootstrap seed** 给出 ρ 的置信区间（逐点落盘）。

用法（项目根目录）：
  python a_network/v28_transfer_gap_check.py
  python a_network/v28_transfer_gap_check.py --report v28_runs.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

N_SEED = int(os.environ.get("V28_SEEDS", "20"))
S0 = int(os.environ.get("V28_S0", "0"))
CSV_PATH = os.environ.get("V28_CSV", "v28_runs.csv")
N_PERM = 1000


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def _p_value_perm(a: np.ndarray, b: np.ndarray, rho: float,
                  rng: np.random.Generator) -> float:
    """置换检验：打乱 b，统计 |ρ| ≥ |rho| 的比例。"""
    cnt = 0
    for _ in range(N_PERM):
        bp = rng.permutation(b)
        if abs(_spearman(a, bp)) >= abs(rho) - 1e-12:
            cnt += 1
    return (cnt + 1) / (N_PERM + 1)


def main() -> int:
    from a_network.v27_variational_value_check import (_std, load_rows,
                                                       prepare)
    if "--report" in sys.argv:
        i = sys.argv.index("--report")
        path = sys.argv[i + 1] if len(sys.argv) > i + 1 else CSV_PATH
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        if not rows:
            print("（CSV 为空）")
            return 1
        rho = np.array([float(r["rho_L"]) for r in rows])
        pv = np.array([float(r["p_L"]) for r in rows])
        n_sig = int(np.sum((rho >= 0.30) & (pv < 0.05)))
        print("========== 任务 A：迁移缺口 vs 信息几何距离（%d seed）=========="
              % len(rows))
        print("  ρ(L, gap)：均值 %+.4f（std %.4f，范围 %+.4f…%+.4f）"
              % (rho.mean(), rho.std(ddof=1), rho.min(), rho.max()))
        print("  p 值中位 %.4g；满足 (ρ≥0.30 ∧ p<0.05) 的 seed：%d/%d"
              % (float(np.median(pv)), n_sig, len(rows)))
        print("  判据 W1：%s" % ("**满足**" if n_sig >= 0.7 * len(rows)
                                 else "不满足"))
        return 0

    fam_rows = load_rows()
    prep = prepare(fam_rows)
    # ---- 目标量：留一族预测的绝对残差（跨族迁移缺口）----
    from a_network.v27_variational_value_check import _ridge_cv_r2  # noqa: F401
    Xb = _std(prep["Xb"])
    y = prep["y"]
    fams = prep["fams"]
    gap = np.zeros_like(y)
    for fam in np.unique(fams):
        te = fams == fam
        tr = ~te
        if tr.sum() < 3:
            gap[te] = np.abs(y[te] - y[tr].mean()) if tr.sum() else 0.0
            continue
        Xtr = np.c_[np.ones(tr.sum()), Xb[tr]]
        Xte = np.c_[np.ones(te.sum()), Xb[te]]
        A = Xtr.T @ Xtr + 1e-3 * np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, Xtr.T @ y[tr])
        gap[te] = np.abs(y[te] - Xte @ w)
    L_feat = prep["Xv"][:, 2]          # 测地线长度
    S_feat = prep["Xv"][:, 3]          # 极值作用量
    print("========== 任务 A：迁移缺口 vs 信息几何距离 ==========")
    print("行 %d / 族 %d；gap 中位 %.4g（范围 %.3g…%.3g）"
          % (len(y), len(np.unique(fams)), float(np.median(gap)),
             gap.min(), gap.max()))
    new = not Path(CSV_PATH).exists()
    f = open(CSV_PATH, "a", encoding="utf-8", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["seed", "rho_L", "p_L", "rho_S", "p_S", "rho_null_p95"])
    for seed in range(S0, N_SEED):
        rng = np.random.default_rng(seed)
        # bootstrap 重采样（逐族）
        idx = []
        for fam in np.unique(fams):
            ii = np.where(fams == fam)[0]
            idx.extend(rng.choice(ii, size=len(ii), replace=True))
        idx = np.asarray(idx)
        rho_L = _spearman(L_feat[idx], gap[idx])
        p_L = _p_value_perm(L_feat[idx], gap[idx], rho_L, rng)
        rho_S = _spearman(S_feat[idx], gap[idx])
        p_S = _p_value_perm(S_feat[idx], gap[idx], rho_S, rng)
        # W2：零分布 95 分位（打乱 gap）
        null = np.array([abs(_spearman(L_feat[idx], rng.permutation(gap[idx])))
                         for _ in range(200)])
        p95 = float(np.percentile(null, 95))
        print("  seed %2d：ρ(L) %+0.3f（p=%.3g）| ρ(S*) %+0.3f（p=%.3g）| "
              "零分布 p95 %.3f" % (seed, rho_L, p_L, rho_S, p_S, p95))
        w.writerow([seed, "%.5f" % rho_L, "%.5g" % p_L, "%.5f" % rho_S,
                    "%.5g" % p_S, "%.5f" % p95])
        f.flush()
    f.close()
    print("\n已写入 %s（用 --report %s 汇总）" % (CSV_PATH, CSV_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
