# -*- coding: utf-8 -*-
"""M6 验证（2026-09-29）：跨族结构编辑——GFlowNet 多解发现。

任务：需求 I_TY ≈ 0.4 nats（精密测量工况）。三个族：
  - landau（a 扫描）：深双阱边界区 I_TY≈0.389 ✓ 解区；
  - elastic（k 扫描）：k≈0.117–0.15 区间 ✓ 解区；
  - two_level（p 扫描）：I_TY ≥ 4 ✗ 无解（负样本族）。
动作含**族切换**（结构编辑）。

判据：
  M1 landau 解区覆盖 ≥ 25%（|I_TY−0.4|<0.05 且族=landau）；
  M2 elastic 解区覆盖 ≥ 25%（同窗且族=elastic）——跨族多解，不塌缩单族；
  M3 GFN 平均 log 奖励 ≥ 随机基线 + 0.8 nats；
  M4 结构编辑被使用：≥30% 采样轨迹含族切换动作。

用法（项目根目录）：python a_network/v6_gfn_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from a_network.gflownet import TrajectoryBalanceGFN
from a_network.rl_env import MultiFamilyEnv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def main() -> int:
    fails = 0
    env = MultiFamilyEnv()
    table = env.table()
    print("[M6] 需求 I_TY≈0.4；landau 峰值 %.3f @ a=%.3f；"
          "elastic 解区 k∈[0.117,0.15]；two_level I_TY≥%.2f（无解）"
          % (max(table["i_tables"][0]), -2.0,
             min(table["i_tables"][2])))

    gfn = TrajectoryBalanceGFN(env.n_states, env.n_actions, seed=0)
    # M6.1：SubTB + 回放缓冲——组合路径「先走位再切换」在 on-policy 下
    # 只在训练早期被采样一次，回放防止其被遗忘（探索侧塌缩的最后标准武器）
    for step in range(2000):
        loss = gfn.train_step_subtb(env, horizon=12, batch=128, lr=2e-2,
                                    entropy_bonus=0.05, lam=0.9,
                                    use_replay=0.5)
        if step % 500 == 0:
            print(f"  step {step}: SubTB loss={loss:.3f}")

    # 终止态 + 族切换使用统计
    states, acts, _, _, _, terms, *_ = gfn.sample_trajectories(env, 800,
                                                              horizon=12)
    fams = np.array([env._decode(int(s))[0] for s in terms.numpy()])
    i_ty = np.array([env.i_tables[f][b] for f, b in
                     [env._decode(int(s)) for s in terms.numpy()]])
    hit = np.abs(i_ty - 0.4) < 0.05
    frac_landau = float(np.mean(hit & (fams == 0)))
    frac_elastic = float(np.mean(hit & (fams == 1)))
    frac_two = float(np.mean(fams == 2))

    switch_used = 0
    for a_seq in acts:
        if any(int(a) >= 5 for a in a_seq.tolist()):
            switch_used += 1
    frac_switch = switch_used / 800

    logr_gfn = float(np.mean(np.log(
        np.array([env.reward(int(s)) for s in terms.numpy()]))))
    rng = np.random.default_rng(0)
    terms_rand = rng.integers(0, env.n_states, 800)
    logr_rand = float(np.mean(np.log(
        np.array([env.reward(int(s)) for s in terms_rand]))))

    print(f"[M6] 采样分布：landau 解区 {frac_landau:.2%}、"
          f"elastic 解区 {frac_elastic:.2%}、two_level {frac_two:.2%}；"
          f"族切换轨迹 {frac_switch:.2%}；"
          f"ΔlogR={logr_gfn - logr_rand:.2f}")
    fails += check(f"M1 landau 解区覆盖 {frac_landau:.2%} ≥ 25%",
                   frac_landau >= 0.25)
    fails += check(f"M2 elastic 解区覆盖 {frac_elastic:.2%} ≥ 25%",
                   frac_elastic >= 0.25)
    fails += check(f"M3 奖励提升 ΔlogR={logr_gfn - logr_rand:.2f} ≥ 0.8",
                   logr_gfn >= logr_rand + 0.8)
    fails += check(f"M4 族切换轨迹 {frac_switch:.2%} ≥ 30%",
                   frac_switch >= 0.30)

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
