# -*- coding: utf-8 -*-
"""接线验证（2026-10-02）：§2.6.2 两条设计意图的兑现。

W1 **DB4 作离策略训练源**（§2.6.2 (i)）：DB4 记录 → 结构空间状态映射；
W2 **A 的表示作智能体状态**（§2.6.2 第 2 条）：用 A 的编码+指纹管线的
   节点特征（`struct_embed`）替代人工槽位特征，GFN 在结构空间可学习；
W3 离线预训练对照：DB4 离线流预训练 + 在线 SubTB vs 冷启动（配对 3 seed），
   判据 = 离线臂不劣于冷启动（均值 ≥ 冷启动 − 0.3 nats）。

用法（项目根目录）：python a_network/v15_wiring_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from a_network.gflownet import TrajectoryBalanceGFN
from a_network.offline_db4 import db4_to_states
from a_network.structure_env import StructureChainEnv
from a_network.struct_embed import CKPT_PATH, state_features

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

H = 20


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def random_policy_logr(env, n: int = 400, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        s = int(rng.integers(0, env.absorbing))
        for _ in range(H):
            if rng.random() < 0.1:
                break
            s, _ = env.step(s, int(rng.integers(0, env.n_actions)))
            if s == env.absorbing:
                break
        vals.append(np.log(max(env.reward(s), 1e-6)))
    return float(np.mean(vals))


def eval_logr(env, gfn, n: int = 400) -> float:
    states, _a, *_r = gfn.sample_trajectories(env, n, horizon=H)
    last = [next((int(states[k][i]) for k in range(len(states) - 1, -1, -1)
                  if int(states[k][i]) < env.absorbing), env.absorbing)
            for i in range(n)]
    vals = [np.log(max(env.reward(s), 1e-6)) for s in last
            if s != env.absorbing]
    return float(np.mean(vals)) if vals else -13.8


def main() -> int:
    fails = 0
    env = StructureChainEnv(n_slots=2)

    # ---- W1：DB4 → 状态映射 ----
    pairs = db4_to_states(env)
    print(f"  DB4 映射：{len(pairs)} 条记录 → 结构空间状态"
          f"（源记录 {len(__import__('db4.db4_writer', fromlist=['DB4Writer']).DB4Writer().load())} 条）")
    fails += check(f"W1 DB4 离策略源映射（{len(pairs)} 条 ≥ 20）",
                   len(pairs) >= 20, f"n={len(pairs)}")

    # ---- W2：A 表示特征 ----
    print(f"  A 表示模式：{'冻结骨干嵌入（检查点存在）' if CKPT_PATH.exists() else 'A 输入表示（编码+指纹管线节点特征；骨干嵌入需 --save-checkpoint）'}")
    F = state_features(env)
    ok_feat = F.shape[0] == env.n_states and np.isfinite(F).all()
    fails += check(f"W2a A 表示特征矩阵（{F.shape}，全有限）", ok_feat)

    def run_arm(seed: int, feat, offline: bool, steps: int = 2000):
        gfn = TrajectoryBalanceGFN(env.n_states, env.n_actions, seed=seed,
                                   in_dim=(None if feat is None
                                           else feat.shape[1]),
                                   feat_fn=(None if feat is None
                                            else (lambda s: feat[s])))
        if offline:
            gfn.pretrain_flow(pairs, steps=400, lr=1e-2)
        for _ in range(steps):
            gfn.train_step_subtb(env, horizon=H, batch=128, lr=2e-2,
                                 entropy_bonus=0.05, lam=0.9, use_replay=0.5)
        return eval_logr(env, gfn)

    rnd = random_policy_logr(env)
    a_feat_logr = run_arm(seed=5, feat=F, offline=False)
    print(f"  A 表示臂 logR {a_feat_logr:+.2f}（随机策略 {rnd:+.2f}）")
    fails += check(f"W2b A 表示臂奖励提升 ΔlogR={a_feat_logr - rnd:+.2f} ≥ 0.8",
                   a_feat_logr >= rnd + 0.8)

    # ---- W3：离线预训练对照（配对）----
    cold, off = [], []
    for seed in (0, 1, 2):
        cold.append((seed, run_arm(seed=seed, feat=F, offline=False)))
        off.append((seed, run_arm(seed=seed, feat=F, offline=True)))
        print(f"  seed {seed}: 冷启动 {cold[-1][1]:+.2f} | "
              f"离线预训练 {off[-1][1]:+.2f}")
    mc = float(np.mean([v for _, v in cold]))
    mo = float(np.mean([v for _, v in off]))
    print(f"  均值：冷启动 {mc:+.2f} | 离线 {mo:+.2f}（Δ{mo - mc:+.2f}）")
    fails += check(f"W3 离线预训练不劣（{mo:+.2f} ≥ {mc:+.2f} − 0.3）",
                   mo >= mc - 0.3, f"Δ={mo - mc:+.2f}")

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
