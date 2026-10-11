# -*- coding: utf-8 -*-
"""RL 侧可信度复检（第 96 条，2026-10-02）——**预登记 + 20 seed**。

**动机**：第 95 条证明「3–5 seed 的显著性结论不可靠」（R1 注入 +0.82→−0.52）。
而 M5/M6/M6.2/M7 的关键判据当年都是 **1–12 seed** 的单点结论。本脚本用与
第 95 条**同一标准**复检两个最关键的 RL 结论：

  **A. M6.2 结构空间能力曲线**（`v6c_m6_2_check.py` 原协议：MultiFamilyEnv
     + SubTB GFN，2000 步，terminate 动作）——复检 5 条判据（M1–M5）在
     20 seed 下的分布；
  **B. M7 SAC 单峰寻优**（`v7_sac_check.py` 的 S1/S3）——20 seed 命中率分布。

**预登记判据（运行前写死）**：
  对每条原判据，按「该判据在 seed 上的**达标率**」判定：
    - **稳健成立**：达标率 ≥ 70%（≥14/20）——原结论可信；
    - **稳健不成立**：达标率 ≤ 30%（≤6/20）——原结论应改述为「不成立」；
    - **不稳定**：其余——原结论应改述为「依赖 seed，不可作为结论」。
  （**不设效应量门槛**：这里复检的是「原判据是否稳健成立」，判据本身沿用
  原脚本的阈值，以免改动协议。）

用法（项目根目录）：python a_network/v19_rl_credibility_check.py
           环境变量 V19_SEEDS 可改 seed 数（默认 20）。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from a_network.gflownet import TrajectoryBalanceGFN
from a_network.rl_env import DesignTaskEnv, MultiFamilyEnv
from a_network.sac import DiscreteSAC

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

N_SEED = int(os.environ.get("V19_SEEDS", "20"))


def verdict(rate: float) -> str:
    if rate >= 0.70:
        return "稳健成立"
    if rate <= 0.30:
        return "稳健不成立"
    return "不稳定（依赖 seed）"


def m62_once(seed: int) -> dict:
    """M6.2 原协议单次运行（seed 控制全局 RNG）。"""
    np.random.seed(seed)
    torch.manual_seed(300 + seed)
    env = MultiFamilyEnv()
    gfn = TrajectoryBalanceGFN(env.n_states, env.n_actions, seed=seed)
    for _ in range(2000):
        gfn.train_step_subtb(env, horizon=16, batch=128, lr=2e-2,
                             entropy_bonus=0.05, lam=0.9, use_replay=0.5)
    states, acts, _r, _l, _s, _t, *_ = gfn.sample_trajectories(
        env, 800, horizon=16)
    last_live = np.array([
        next((int(states[k][i]) for k in range(len(states) - 1, -1, -1)
              if int(states[k][i]) < env.absorbing), env.absorbing)
        for i in range(800)])
    fams = np.array([env._decode(int(s))[0] if s < env.absorbing else -1
                     for s in last_live])
    hit = np.zeros(800, dtype=bool)
    for k, s in enumerate(last_live):
        if s < env.absorbing:
            f, b = env._decode(int(s))
            hit[k] = abs(env.i_tables[f][b] - 0.4) < 0.05
    logr = float(np.mean(np.log(
        np.array([max(env.reward(int(s)), 1e-6) for s in last_live]))))
    rng = np.random.default_rng(seed)
    logr_rand = float(np.mean(np.log(
        np.array([env.reward(int(s))
                  for s in rng.integers(0, env.absorbing, 800)]))))
    return {
        "M1 landau 解区 ≥25%": float(np.mean(hit & (fams == 0))) >= 0.25,
        "M2 elastic 解区 ≥25%": float(np.mean(hit & (fams == 1))) >= 0.25,
        "M3 ΔlogR ≥0.8": (logr - logr_rand) >= 0.8,
        "M4 族切换 ≥30%": (sum(any(int(a) in (5, 6) for a in seq.tolist())
                               for seq in acts) / 800) >= 0.30,
        "M5 终止轨迹 ≥50%": (sum(any(int(a) == 7 for a in seq.tolist())
                                for seq in acts) / 800) >= 0.50,
        "_dlogr": logr - logr_rand,
        "_landau": float(np.mean(hit & (fams == 0))),
        "_elastic": float(np.mean(hit & (fams == 1))),
    }


def m7_once(seed: int) -> dict:
    """M7 S1/S3 原协议单次运行（SAC 单峰寻优命中率）。"""
    np.random.seed(seed)
    torch.manual_seed(400 + seed)
    env = DesignTaskEnv(task="info_target", beta=4.0, mode_a=-1.9, sigma_r=0.3)
    sac = DiscreteSAC(env.n_states, env.n_actions, seed=seed)
    for _ in range(300):
        s = int(np.random.randint(0, env.n_states))
        for t in range(12):
            a = sac.act(s)
            s2, r = env.step(s, a)
            sac.store(s, a, float(r), s2, t == 11)
            s = s2
            sac.train_step(batch=64)
    hits = 0
    for _ in range(500):
        s = int(np.random.randint(0, env.n_states))
        for _t in range(12):
            pi = sac.policy(torch.tensor([s], dtype=torch.long))[0]
            s, _ = env.step(s, int(torch.argmax(pi).item()))
        hits += abs(env.a_grid[s] + 1.9) < 0.3
    frac = hits / 500
    return {"S1 命中 ≥60%": frac >= 0.60, "S3 命中 ≥62.4%": frac >= 0.624,
            "_frac": frac}


def main() -> int:
    print("========== RL 侧可信度复检（预登记，%d seed）==========" % N_SEED)
    print("  预登记判据：达标率 ≥70% → 稳健成立；≤30% → 稳健不成立；"
          "其余 → 不稳定\n")

    a_rows, b_rows = [], []
    for seed in range(N_SEED):
        r = m62_once(seed)
        a_rows.append(r)
        print("  [M6.2] seed %2d：ΔlogR=%+6.2f landau=%.3f elastic=%.3f"
              % (seed, r["_dlogr"], r["_landau"], r["_elastic"]))
    for seed in range(N_SEED):
        r = m7_once(seed)
        b_rows.append(r)
        print("  [M7]   seed %2d：SAC 命中=%.3f" % (seed, r["_frac"]))

    print("\n  **A. M6.2 判据复检（%d seed）**" % N_SEED)
    for key in ("M1 landau 解区 ≥25%", "M2 elastic 解区 ≥25%",
                "M3 ΔlogR ≥0.8", "M4 族切换 ≥30%", "M5 终止轨迹 ≥50%"):
        rate = float(np.mean([r[key] for r in a_rows]))
        print("    %-22s 达标率 %2d/%d（%3.0f%%）→ **%s**"
              % (key, round(rate * N_SEED), N_SEED, 100 * rate,
                 verdict(rate)))
    dl = np.array([r["_dlogr"] for r in a_rows])
    print("    ΔlogR 分布：均值 %+.2f（std %.2f，范围 %+.2f…%+.2f）"
          % (dl.mean(), dl.std(ddof=1), dl.min(), dl.max()))

    print("\n  **B. M7 判据复检（%d seed）**" % N_SEED)
    for key in ("S1 命中 ≥60%", "S3 命中 ≥62.4%"):
        rate = float(np.mean([r[key] for r in b_rows]))
        print("    %-22s 达标率 %2d/%d（%3.0f%%）→ **%s**"
              % (key, round(rate * N_SEED), N_SEED, 100 * rate,
                 verdict(rate)))
    fr = np.array([r["_frac"] for r in b_rows])
    print("    命中率分布：均值 %.3f（std %.3f，范围 %.3f…%.3f）"
          % (fr.mean(), fr.std(ddof=1), fr.min(), fr.max()))

    print("\n  说明：本脚本只做「原判据的稳健性复检」，不改协议与阈值；"
          "结论须按预登记判据机械套用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
