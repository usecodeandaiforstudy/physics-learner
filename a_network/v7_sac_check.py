# -*- coding: utf-8 -*-
"""M7 验证（2026-09-29）：SAC 生产管线（单峰寻优回退路径）。

任务复用 M5/M6 环境：
  S1（DesignTaskEnv·info_target，单峰）：SAC 收敛后，贪心策略终端态
     落在需求区（|a+1.9|<0.3）的比例 ≥ 60%（单峰寻优 = SAC 主场景）；
  S2（MultiFamilyEnv·requirement）：SAC 找到一个解（终端态命中任一
     需求窗口的比例 ≥ 25%）+ 平均 logR ≥ 随机基线 + 0.8——不要求双模
     （多样性是 GFN 的职责，SAC 回退只承诺「找到一个好解」）；
  S3 对照：SAC 的峰值浓度 ≥ GFN（M5 的 78% 为参照，SAC 单峰应更高或
     相当——寻优 vs 采样各有分工）。

用法（项目根目录）：python a_network/v7_sac_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from a_network.rl_env import DesignTaskEnv, MultiFamilyEnv
from a_network.sac import DiscreteSAC

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def run_episodes(sac, env, episodes: int, horizon: int,
                 log_reward: bool = False) -> float:
    """训练 episodes 轮（固定 horizon 的回合式交互 + 回放训练）。

    log_reward=True：以 log(奖励) 训练（稀疏/窄峰奖励的线性 Q 学习信号
    全被压缩在零附近——log 空间是可学的形状，标准奖励塑形）。
    """
    for ep in range(episodes):
        s = int(np.random.randint(0, env.n_states))
        for t in range(horizon):
            a = sac.act(s)
            s2, r = env.step(s, a)
            done = (t == horizon - 1)
            rr = float(np.log(max(r, 1e-6))) if log_reward else float(r)
            sac.store(s, a, rr, s2, done)
            s = s2
            sac.train_step(batch=64)
    return 0.0


def main() -> int:
    fails = 0

    # ---------- S1/S3：单峰寻优（**3-seed 均值**）----------
    # **2026-10-02 修复**：原实现用全局 numpy RNG 却从未播种（只有 SAC 内部
    # 有 seed）→ 每次进程运行给出不同数值（实测 100% / 61.8% / 56.8%），
    # 历史「4/4」实为**单次幸运抽样**。改为逐 seed 播种 + 3 seed 均值。
    fracs = []
    for seed in (0, 1, 2):
        np.random.seed(seed)
        torch.manual_seed(100 + seed)
        env_a = DesignTaskEnv(task="info_target", beta=4.0,
                              mode_a=-1.9, sigma_r=0.3)
        sac_a = DiscreteSAC(env_a.n_states, env_a.n_actions, seed=seed)
        run_episodes(sac_a, env_a, 300, horizon=12)

        n_hit = 0
        n_test = 500
        for _ in range(n_test):
            s = int(np.random.randint(0, env_a.n_states))
            for _t in range(12):
                pi = sac_a.policy(torch.tensor([s], dtype=torch.long))[0]
                a = int(torch.argmax(pi).item())   # 贪心评估
                s, _ = env_a.step(s, a)
            if abs(env_a.a_grid[s] + 1.9) < 0.3:
                n_hit += 1
        fracs.append(n_hit / n_test)
        print(f"    seed {seed}: S1 命中 {fracs[-1]:.2%}")
    frac = float(np.mean(fracs))
    spread = max(fracs) - min(fracs)
    print(f"    **seed 间方差**：{min(fracs):.1%} – {max(fracs):.1%}"
          f"（极差 {spread:.1%}）——SAC 单峰寻优**不稳定**，"
          f"{'均值仅勉强过线' if frac < 0.75 else '均值稳健'}")
    if min(fracs) < 0.5:
        print("    [WARN] 存在 seed < 50%：判据对**单次运行**不可靠——"
              "任何「通过」都必须以多 seed 均值为准（第 91 条教训）")
    fails += check(f"S1 单峰寻优命中需求区（3-seed 均值 {frac:.2%} ≥ 60%）",
                   frac >= 0.60, f"fracs={[f'{x:.3f}' for x in fracs]}")
    fails += check(f"S3 峰值浓度对照（SAC 均值 {frac:.2%} ≥ GFN M5 参照 "
                   f"78% × 0.8 = 62.4%）", frac >= 0.624,
                   f"fracs={[f'{x:.3f}' for x in fracs]}")

    # ---------- S2：跨族环境找到一个解 ----------
    env_m = MultiFamilyEnv()
    sac_m = DiscreteSAC(env_m.n_states, env_m.n_actions, seed=1, lr=1e-2)
    run_episodes(sac_m, env_m, 600, horizon=12, log_reward=True)

    terms = []
    for _ in range(500):
        # 采样须排除吸收态（**2026-10-02 修潜伏 bug**：原用 n_states 会把
        # 索引 99=absorbing 采进来 → _decode 得 fam=3 → i_tables 越界；
        # M6.2 引入吸收态后此 bug 潜伏，v11 已用 absorbing 上界，v7 未同步）
        s = int(np.random.randint(0, env_m.absorbing))
        last_valid = s
        for t in range(12):
            pi = sac_m.policy(torch.tensor([s], dtype=torch.long))[0]
            a = int(torch.argmax(pi).item())
            s, _ = env_m.step(s, a)
            if s == env_m.absorbing:
                break          # 终止动作（M6.2）：后续用最后非吸收态评估
            last_valid = s
        # **用「最后一个非吸收态」评估**（第 61 条同一教训：有终止动作时
        # 不能用吸收态做评估口径——absorbing 的 _decode 会得到 fam=n_families）
        terms.append(last_valid)
    i_ty = np.array([env_m.i_tables[f][b] for f, b in
                     [env_m._decode(s) for s in terms]])
    hit = float(np.mean(np.abs(i_ty - 0.4) < 0.05))
    logr_sac = float(np.mean(np.log(
        np.array([env_m.reward(s) for s in terms]))))
    rng = np.random.default_rng(0)
    terms_rand = rng.integers(0, env_m.absorbing, 500)
    logr_rand = float(np.mean(np.log(
        np.array([env_m.reward(s) for s in terms_rand]))))
    fails += check(f"S2a 找到一个解（{hit:.2%} ≥ 25%）",
                   hit >= 0.25, f"hit={hit:.3f}")
    fails += check(f"S2b 奖励提升（SAC {logr_sac:.2f} ≥ 随机 "
                   f"{logr_rand:.2f} + 0.8）",
                   logr_sac >= logr_rand + 0.8,
                   f"Δ={logr_sac - logr_rand:.2f}")

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
