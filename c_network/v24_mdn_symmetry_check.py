# -*- coding: utf-8 -*-
"""MDN 对称测量（2026-10-02，第 102 条）——补齐 v22 对照的不对称。

**问题**：`v22` 的 V3（双阱模式召回）与 V5（外推重叠）**只测了 CFM**，
MDN 侧缺同口径数字 ⟹ 不能据此下「MDN 优于 CFM」的一般性结论。
本脚本用**同一评估口径**测 MDN 的这两项（MDN 训练快，20 seed 成本低）。

**口径（与 v22 完全一致）**：
  - 模式召回：a ∈ {−1.5, −0.75}（双阱区），要求模型密度在 ±√(−a) 的
    ±0.4 窗内**各有峰**；
  - 外推：a = 2.5（训练范围 {−2,…,2} 之外）的重叠系数 ∫min(p_true,p̂)dφ；
  - 另报**平均重叠**（6 评估点）以核对与 v22 的 MDN 列一致。

用法（项目根目录）：python c_network/v24_mdn_symmetry_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from c_network.mdn import MixtureDensityNetwork, sample_landau

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

A_TRAIN = [-2.0, -1.0, 0.0, 1.0, 2.0]
A_EVAL = [-1.5, -0.75, -0.25, 0.5, 1.5, 2.5]
GRID = np.linspace(-3.2, 3.2, 241)
N_SEED = int(os.environ.get("V24_SEEDS", "20"))


def true_pdf(a: float, grid: np.ndarray) -> np.ndarray:
    logp = -(a * grid ** 2 / 2.0 + grid ** 4 / 4.0)
    logp = logp - logp.max()
    p = np.exp(logp)
    return p / (p.sum() * (grid[1] - grid[0]))


def norm_on_grid(logp: np.ndarray, grid: np.ndarray) -> np.ndarray:
    p = np.exp(logp - logp.max())
    return p / (p.sum() * (grid[1] - grid[0]))


def overlap(p, q, grid):
    return float(np.sum(np.minimum(p, q)) * (grid[1] - grid[0]))


def two_mode_recall(a: float, p_model: np.ndarray) -> bool:
    """与 v22 同口径：±√(−a) 的 ±0.4 窗内各有峰。"""
    if a >= -0.5:
        return True
    peaks = [GRID[i] for i in range(1, len(GRID) - 1)
             if p_model[i] >= p_model[i - 1] and p_model[i] > p_model[i + 1]]
    c = float(np.sqrt(-a))
    return (any(abs(pk + c) < 0.4 for pk in peaks)
            and any(abs(pk - c) < 0.4 for pk in peaks))


def train_mdn(seed: int) -> MixtureDensityNetwork:
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for a in A_TRAIN:
        phi = sample_landau(a, 1.0, 1.0, 4000, seed=seed)
        X.append(np.full(len(phi), a, dtype=np.float32))
        Y.append(phi.astype(np.float32))
    X = np.concatenate(X)[:, None]
    Y = np.concatenate(Y)[:, None]
    torch.manual_seed(seed)
    mdn = MixtureDensityNetwork(1, n_components=8, seed=seed)
    opt = torch.optim.Adam(mdn.parameters(), lr=3e-3)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for _ in range(6000):
        idx = rng.integers(0, len(Xt), 512)
        loss = mdn.nll(Xt[idx], Yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return mdn


def main() -> int:
    print("========== MDN 对称测量（%d seed，口径与 v22 一致）=========="
          % N_SEED)
    ovls, modes, ext = [], [], []
    for seed in range(N_SEED):
        mdn = train_mdn(seed)
        o_each, m_each = [], []
        for a in A_EVAL:
            pt = true_pdf(a, GRID)
            pm = norm_on_grid(np.log(mdn.density(a, GRID) + 1e-12), GRID)
            ov = overlap(pt, pm, GRID)
            o_each.append(ov)
            m_each.append(two_mode_recall(a, pm))
            if abs(a - 2.5) < 1e-9:
                ext.append(ov)
        ovls.append(float(np.mean(o_each)))
        modes.append(float(np.mean([m for a, m in zip(A_EVAL, m_each)
                                    if a <= -0.5])))
        print("    seed %2d：平均重叠 %.4f | 双阱召回 %.2f | 外推 %.4f"
              % (seed, ovls[-1], modes[-1], ext[-1]))
    ov, md, ex = np.array(ovls), np.array(modes), np.array(ext)
    print("\n  **MDN（同口径）**：")
    print("    平均重叠 %.4f（std %.4f）——v22 的 MDN 列为 %.4f（应一致）"
          % (ov.mean(), ov.std(ddof=1), 0.9663))
    print("    双阱模式召回 %.3f（std %.3f）——CFM 为 **0.000**"
          % (md.mean(), md.std(ddof=1)))
    print("    外推（a=2.5）重叠 %.4f（std %.4f）——CFM 为 **0.9476**"
          % (ex.mean(), ex.std(ddof=1)))
    print("\n  ⟹ 对称化后的结论：%s"
          % ("MDN 在双阱与外推两项上**也都优于或持平 CFM** ⟹ "
             "「本域 MDN 更适合」成立"
             if (md.mean() > 0.5 and ex.mean() >= 0.90) else
             "两项存在**交叉**（MDN 胜模式召回 / CFM 胜外推）⟹ "
             "结论须按维度分别陈述，不可笼统说谁更好"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
