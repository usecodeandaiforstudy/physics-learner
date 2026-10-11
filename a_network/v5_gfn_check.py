# -*- coding: utf-8 -*-
"""M5 验证（2026-09-29）：GFlowNet 生产管线 v0 双任务检查。

任务 A（info_target，单峰）：R = exp(β·I_TY − I_XT − ((a−a*)²/σ²))——
  信息作用量 × 需求邻近窗（a*=−1.9 深双阱信息峰值邻近）→ 采样应集中于
  需求区。（首跑纯 I_TY 奖励景观平坦不可学，已修任务设计，见文件内注释。）
任务 B（bimodal，双峰）：双高斯需求 ±1.5——
  GFlowNet 的标志性能力：**两模都要发现**（多样性，不是单点最优）。

判据：
  A1 采样态 |a+1.9|<0.3 覆盖 ≥ 40%；
  A2 GFN 平均 log 奖励 ≥ 随机基线 + 0.8 nats；
  B1 两模各自覆盖 ≥ 25%（±0.3 窗内）。

用法（项目根目录）：python a_network/v5_gfn_check.py
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


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def main() -> int:
    fails = 0

    # ---------- 任务 A：单峰（信息峰值邻近需求） ----------
    # 首跑教训：纯 I_TY 奖励的景观近平坦（32/33 箱 ≈0.18）→ 集中度判据
    # 对完美 GFN 都不可达；改用「信息作用量 × 需求邻近窗」的尖锐任务
    # （v5 修正，2026-09-29）
    env_a = DesignTaskEnv(task="info_target", beta=4.0,
                          mode_a=-1.9, sigma_r=0.3)
    table_a = env_a.table()
    i_ty_max = float(table_a["I_TY"].max())
    a_peak = float(table_a["a"][int(np.argmax(table_a["I_TY"]))])
    print(f"[TaskA] I_TY 峰值 {i_ty_max:.3f} @ a={a_peak:.3f}"
          f"（深双阱边界）；需求目标 a*=−1.9（信息峰值邻近）")

    gfn_a = TrajectoryBalanceGFN(env_a.n_states, env_a.n_actions, seed=0)
    for step in range(600):
        loss = gfn_a.train_step(env_a, horizon=10, batch=128, lr=2e-2)
        if step % 150 == 0:
            print(f"  step {step}: TB loss={loss:.3f}")
    terms_a = gfn_a.sample_terminals(env_a, 800, horizon=10)
    a_sampled_a = env_a.a_grid[terms_a]
    frac_a = float(np.mean(np.abs(a_sampled_a + 1.9) < 0.3))
    logr_gfn = float(np.mean(np.log(
        np.array([env_a.reward(int(s)) for s in terms_a]))))
    # 随机基线（均匀终止态）
    rng = np.random.default_rng(0)
    terms_rand = rng.integers(0, env_a.n_states, 800)
    logr_rand = float(np.mean(np.log(
        np.array([env_a.reward(int(s)) for s in terms_rand]))))

    fails += check(f"TaskA 采样集中于需求区（|a+1.9|<0.3 覆盖 "
                   f"{frac_a:.2%} ≥ 40%）",
                   frac_a >= 0.40, f"frac={frac_a:.3f}")
    fails += check(f"TaskA 奖励提升（GFN {logr_gfn:.2f} ≥ 随机 "
                   f"{logr_rand:.2f} + 0.8）",
                   logr_gfn >= logr_rand + 0.8,
                   f"Δ={logr_gfn - logr_rand:.2f}")

    # ---------- 任务 B：双峰（多样解采样） ----------
    env_b = DesignTaskEnv(task="bimodal", mode_a=1.5, sigma_r=0.25)
    gfn_b = TrajectoryBalanceGFN(env_b.n_states, env_b.n_actions, seed=1)
    for step in range(600):
        gfn_b.train_step(env_b, horizon=8, batch=64)
    terms_b = gfn_b.sample_terminals(env_b, 800, horizon=8)
    a_sampled = env_b.a_grid[terms_b]
    frac_pos = float(np.mean(np.abs(a_sampled - 1.5) < 0.3))
    frac_neg = float(np.mean(np.abs(a_sampled + 1.5) < 0.3))
    print(f"[TaskB] 采样分布：+1.5 模覆盖 {frac_pos:.2%}，"
          f"−1.5 模覆盖 {frac_neg:.2%}")
    fails += check(f"TaskB 双模覆盖（+{frac_pos:.2%} ≥ 25% 且 "
                   f"{frac_neg:.2%} ≥ 25%）",
                   frac_pos >= 0.25 and frac_neg >= 0.25)

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
