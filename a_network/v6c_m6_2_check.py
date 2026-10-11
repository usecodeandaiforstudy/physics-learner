# -*- coding: utf-8 -*-
"""M6.2 验证（2026-10-02）：显式终止动作下的跨族多解发现。

M6 失败模式（第 54–55 条）：固定 horizon 无终止 → 解区「粘性」由游走
几何决定 + on-policy 首模塌缩。M6.2 对策：**显式终止动作**（吸收态，
任意状态可停下取奖励）——「起点在解区 → 立即终止」成为一步解，瓦解
游走漂移；配合 SubTB + 回放（M6.1 机制）。

判据（与 v6 同口径 + 终止使用）：
  M1 landau 解区覆盖 ≥ 25%；M2 elastic 解区覆盖 ≥ 25%；
  M3 ΔlogR ≥ 0.8；M4 族切换轨迹 ≥ 30%；M5 终止轨迹 ≥ 50%。

用法（项目根目录）：python a_network/v6c_m6_2_check.py
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


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def main() -> int:
    fails = 0
    env = MultiFamilyEnv()
    print("[M6.2] 显式终止 + SubTB；需求 I_TY≈0.4；解区：landau 边界 / "
          "elastic 软弹簧；two_level 无解")

    gfn = TrajectoryBalanceGFN(env.n_states, env.n_actions, seed=0)
    for step in range(2000):
        loss = gfn.train_step_subtb(env, horizon=16, batch=128, lr=2e-2,
                                    entropy_bonus=0.05, lam=0.9,
                                    use_replay=0.5)
        if step % 500 == 0:
            print(f"  step {step}: SubTB loss={loss:.3f}")

    states, acts, _, _, _, terms, *_ = gfn.sample_trajectories(
        env, 800, horizon=16)
    terms_np = terms.numpy()
    # 评估口径：终止轨迹取「终止发生时的状态」（最后一个非吸收态），
    # 未终止取末态（吸收态 reward=0，直接统计会出 nan/log0）
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
    frac_landau = float(np.mean(hit & (fams == 0)))
    frac_elastic = float(np.mean(hit & (fams == 1)))
    switch_used = sum(any(int(a) in (5, 6) for a in seq.tolist())
                      for seq in acts)
    term_used = sum(any(int(a) == 7 for a in seq.tolist()) for seq in acts)
    frac_switch = switch_used / 800
    frac_term = term_used / 800

    logr_gfn = float(np.mean(np.log(
        np.array([max(env.reward(int(s)), 1e-6) for s in last_live]))))
    rng = np.random.default_rng(0)
    terms_rand = rng.integers(0, env.absorbing, 800)
    logr_rand = float(np.mean(np.log(
        np.array([env.reward(int(s)) for s in terms_rand]))))

    print(f"[M6.2] landau {frac_landau:.2%}、elastic {frac_elastic:.2%}、"
          f"切换 {frac_switch:.2%}、终止 {frac_term:.2%}、"
          f"ΔlogR={logr_gfn - logr_rand:.2f}")
    fails += check(f"M1 landau 解区 {frac_landau:.2%} ≥ 25%",
                   frac_landau >= 0.25)
    fails += check(f"M2 elastic 解区 {frac_elastic:.2%} ≥ 25%",
                   frac_elastic >= 0.25)
    fails += check(f"M3 ΔlogR={logr_gfn - logr_rand:.2f} ≥ 0.8",
                   logr_gfn >= logr_rand + 0.8)
    fails += check(f"M4 族切换 {frac_switch:.2%} ≥ 30%",
                   frac_switch >= 0.30)
    fails += check(f"M5 终止轨迹 {frac_term:.2%} ≥ 50%",
                   frac_term >= 0.50)

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
