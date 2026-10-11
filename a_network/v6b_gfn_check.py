# -*- coding: utf-8 -*-
"""M6.1 决定性诊断（2026-09-29）：边界吸引子假说。

主实验（v6 判据）八轮不变量 = 族切换使用率恒 1.50%。假说：landau 解区位于
裁剪边界（bin 0–2，左移被 clip 吸收 = 天然吸引子），elastic 解区位于族
内部（bin 1–3，随机游走漂失）——固定 horizon 无终止动作下，边界解天然
「粘住」、内部解天然「扩散丢失」，所有算法收敛到同一全局最优。

本实验：需求窗口移到 elastic 边界 bin 0（I_TY=0.3，k=0.1）——按假说，
elastic（新边界解）覆盖应跃升、landau（变内部解）应下降。

用法（项目根目录）：python a_network/v6b_gfn_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from a_network.gflownet import TrajectoryBalanceGFN
from a_network.rl_env import MultiFamilyEnv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def main() -> int:
    # 需求 I_TY≈0.3：elastic bin0（k=0.1，I=0.3）为边界解；landau 的
    # I(a)=0.3 落在 a≈−1.63（内部）
    env = MultiFamilyEnv(target_i=0.3, sigma_i=0.04)
    gfn = TrajectoryBalanceGFN(env.n_states, env.n_actions, seed=0)
    for step in range(2000):
        loss = gfn.train_step_subtb(env, horizon=12, batch=128, lr=2e-2,
                                    entropy_bonus=0.05, lam=0.9,
                                    use_replay=0.5)
    _, acts, _, _, _, terms, *_ = gfn.sample_trajectories(env, 800,
                                                         horizon=12)
    fams = np.array([env._decode(int(s))[0] for s in terms.numpy()])
    i_ty = np.array([env.i_tables[f][b] for f, b in
                     [env._decode(int(s)) for s in terms.numpy()]])
    hit = np.abs(i_ty - 0.3) < 0.04
    frac_landau = float(np.mean(hit & (fams == 0)))
    frac_elastic = float(np.mean(hit & (fams == 1)))
    switch_used = sum(any(int(a) >= 5 for a in seq.tolist()) for seq in acts)
    print("[v6b 边界诊断] 需求窗口 I_TY∈[0.26,0.34]：")
    print("  landau（内部解）覆盖 %.2f%%；elastic（边界解）覆盖 %.2f%%；"
          "族切换轨迹 %.2f%%" % (100 * frac_landau, 100 * frac_elastic,
                                100 * switch_used / 800))
    print("  假说判定：elastic 覆盖 ≥ landau 且 elastic ≥ 20%%"
          " = 边界吸引子假说成立（解的『粘性』取决于几何，不取决于算法）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
