# -*- coding: utf-8 -*-
"""可信度复检 · 第三批（第 98 条，2026-10-02）——**M5 双模覆盖率，预登记 + 20 seed**。

被复检的原结论（第 53 条 / `v5_gfn_check.py` TaskB）：
  **GFlowNet 在双峰需求（±1.5）下双模各覆盖 34% / 34.5%**——但原运行是
  **单 seed（seed=1）、600 步、未播种全局 RNG**。

**预登记判据（运行前写死，机械套用；与第 96/97 条同一标准）**：
  原判据「+1.5 模 ≥25% 且 −1.5 模 ≥25%」在 seed 上的**达标率**：
    ≥70% → **稳健成立**；≤30% → **稳健不成立**；其余 → **不稳定**。
  另报两模覆盖率与**和**（多样性的直接度量）的均值 ± std / 范围。

用法（项目根目录）：python a_network/v21_m5_credibility_check.py
           环境变量 V21_SEEDS 可改 seed 数（默认 20）。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from a_network.gflownet import TrajectoryBalanceGFN
from a_network.rl_env import DesignTaskEnv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

N_SEED = int(os.environ.get("V21_SEEDS", "20"))
S0 = int(os.environ.get("V21_S0", "0"))     # 断点续跑起始 seed（默认 0）


def verdict(rate: float) -> str:
    if rate >= 0.70:
        return "稳健成立"
    if rate <= 0.30:
        return "稳健不成立"
    return "不稳定（依赖 seed）"


def one(seed: int) -> dict:
    """原协议单次运行（唯一差别：逐 seed 播种 + 逐点报告）。"""
    np.random.seed(seed)
    torch.manual_seed(600 + seed)
    env = DesignTaskEnv(task="bimodal", mode_a=1.5, sigma_r=0.25)
    gfn = TrajectoryBalanceGFN(env.n_states, env.n_actions, seed=seed)
    for _ in range(600):
        gfn.train_step(env, horizon=8, batch=64)
    terms = gfn.sample_terminals(env, 800, horizon=8)
    a = env.a_grid[terms]
    fp = float(np.mean(np.abs(a - 1.5) < 0.3))
    fn = float(np.mean(np.abs(a + 1.5) < 0.3))
    return {"pos": fp, "neg": fn, "both": fp + fn,
            "ok": (fp >= 0.25 and fn >= 0.25)}


def main() -> int:
    print("========== 可信度复检 · 第三批：M5 双模覆盖"
          "（预登记，%d seed）==========" % N_SEED)
    print("  预登记判据：达标率 ≥70% → 稳健成立；≤30% → 稳健不成立；"
          "其余 → 不稳定\n")
    rows = []
    for seed in range(S0, N_SEED):
        r = one(seed)
        rows.append(r)
        print("    seed %2d：+1.5 模 %.3f  −1.5 模 %.3f  合计 %.3f  %s"
              % (seed, r["pos"], r["neg"], r["both"],
                 "OK" if r["ok"] else "--"))
    pos = np.array([r["pos"] for r in rows])
    neg = np.array([r["neg"] for r in rows])
    both = np.array([r["both"] for r in rows])
    rate = float(np.mean([r["ok"] for r in rows]))
    print("\n  +1.5 模覆盖：均值 %.3f（std %.3f，范围 %.3f…%.3f）"
          % (pos.mean(), pos.std(ddof=1), pos.min(), pos.max()))
    print("  −1.5 模覆盖：均值 %.3f（std %.3f，范围 %.3f…%.3f）"
          % (neg.mean(), neg.std(ddof=1), neg.min(), neg.max()))
    print("  两模合计（多样性度量）：均值 %.3f（std %.3f）"
          % (both.mean(), both.std(ddof=1)))
    print("  原判据（两模均 ≥25%%）达标率 %d/%d → **%s**"
          % (round(rate * N_SEED), N_SEED, verdict(rate)))
    print("  与第 53 条原值（+34.0%% / −34.5%%，单 seed seed=1）对照：%s"
          % ("原值在分布范围内" if (pos.max() >= 0.34 and neg.max() >= 0.345)
             else "**原值高于本次分布上界 ⟹ 有利 seed 的极值**"))
    print("\n  说明：本脚本与 v5_gfn_check 的 TaskB 唯一差别 = seed 数与"
          "逐 seed 报告；协议、超参、判据阈值均未改动。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
