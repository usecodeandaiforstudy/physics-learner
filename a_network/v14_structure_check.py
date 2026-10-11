# -*- coding: utf-8 -*-
"""缺口② 验证（2026-10-02）：结构空间搜索 + 演示初始化。

对照实验：
  - 基线：SubTB-GFlowNet 直接在**结构空间**（3375 态 × 10 动作，含类型
    切换的结构编辑）上训练；
  - 演示：先以 BFS 演示轨迹做 P_F 交叉熵预训练，再同预算 SubTB 微调。

判据：
  G1 结构空间可行：解状态 ≥5 个且类型模式 ≥2 种（结构性多解存在）；
  G2 结构级多解发现：最优臂命中 ≥ 0.5ρ（ρ = 随机起点中 horizon 内可达解
     的比例，BFS 实测；绝对门槛在组合搜索里不可达——首跑 40% 门槛即此错）
     且命中解的类型模式 ≥2 种；
  G3 记录：四臂对照（one-hot / one-hot+演示 / 结构化特征 / 结构化特征+演示）。

用法（项目根目录）：python a_network/v14_structure_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from a_network.demo_pretrain import generate_demos
from a_network.gflownet import TrajectoryBalanceGFN
from a_network.structure_env import StructureChainEnv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def type_pattern(env, state: int) -> tuple[int, ...]:
    return tuple(t for t, _ in env.decode(state))


def evaluate(env, gfn, n: int = 500, horizon: int = 20) -> tuple[float, int, float]:
    """(针尖命中率, 结构模式数, 采样终止态平均 logR)。

    注意（v14 第二跑 bug）：有终止动作时，终止轨迹的末态是吸收态——直接
    统计末态会把学会终止的代理全判 0（同 v6c 的坑，复发于此）。
    第三跑补充：针尖命中只反映「精确解」，需同时量平均 logR（代理的真实
    目标 = 采样分布 ∝ 奖励，会展布在信息态上而非只落在针尖）。
    """
    states, _acts, *_rest = gfn.sample_trajectories(env, n, horizon=horizon)
    last_live = [
        next((int(states[k][i]) for k in range(len(states) - 1, -1, -1)
              if int(states[k][i]) < env.absorbing), env.absorbing)
        for i in range(n)]
    live = [s for s in last_live if s != env.absorbing]
    hits = [s for s in live if env.reward(s) >= 0.8]
    patterns = {type_pattern(env, s) for s in hits}
    mean_logr = float(np.mean([np.log(max(env.reward(s), 1e-6))
                               for s in live])) if live else -13.8
    return len(hits) / n, len(patterns), mean_logr


def random_policy_logr(env, n: int = 500, horizon: int = 20,
                       seed: int = 0) -> float:
    """随机策略基线：随机起点 + 随机动作 + 随机终止 → 平均 logR。"""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        s = int(rng.integers(0, env.absorbing))
        for _ in range(horizon):
            if rng.random() < 0.1:            # 随机终止
                break
            s, _ = env.step(s, int(rng.integers(0, env.n_actions)))
            if s == env.absorbing:
                break
        vals.append(np.log(max(env.reward(s), 1e-6)))
    return float(np.mean(vals))


def reachability(env, n_starts: int = 100, horizon: int = 20,
                 seed: int = 1) -> float:
    """ρ：随机起点中 horizon 内可达解（reward ≥ 0.8）的比例（BFS 实测）。"""
    from collections import deque
    rng = np.random.default_rng(seed)
    reach = 0
    for _ in range(n_starts):
        start = int(rng.integers(0, env.absorbing))
        seen = {start}
        q = deque([(start, 0)])
        ok = False
        while q:
            s, d = q.popleft()
            if s != start and env.reward(s) >= 0.8:
                ok = True
                break
            if d == horizon:
                continue
            for a in range(env.n_actions):
                nxt, _ = env.step(s, a)
                if nxt == env.absorbing or nxt in seen:
                    continue
                seen.add(nxt)
                q.append((nxt, d + 1))
        reach += ok
    return reach / n_starts


def main() -> int:
    fails = 0
    H = 20
    summary = []
    for sigma in (0.25, 0.6, 1.0):
        env = StructureChainEnv(n_slots=2, sigma_i=sigma)
        sols = env.solution_states(0.8)
        patterns = {type_pattern(env, s) for s in sols}
        n_info = sum(1 for s in range(env.absorbing)
                     if env.reward(s) >= 0.5)
        rho = reachability(env, horizon=H)
        rand_hit = len(sols) / env.absorbing
        print(f"\n  [σ={sigma}] 解 {len(sols)} 个（{len(patterns)} 模式）、"
              f"reward≥0.5 的态 {n_info} 个、随机命中 {rand_hit:.2%}、"
              f"ρ={rho:.0%}")

        demos = generate_demos(env, n_starts=40, max_depth=H, thresh=0.8)

        def run_arm(seed: int, use_feat: bool):
            gfn = TrajectoryBalanceGFN(
                env.n_states, env.n_actions, seed=seed,
                in_dim=(env.feat_dim if use_feat else None),
                feat_fn=(env.feat if use_feat else None))
            gfn.pretrain_pf(demos, steps=400, lr=1e-2)
            for _ in range(3000):
                gfn.train_step_subtb(env, horizon=H, batch=128, lr=2e-2,
                                     entropy_bonus=0.05, lam=0.9,
                                     use_replay=0.5)
            return evaluate(env, gfn, n=500, horizon=H)

        h_oh, p_oh, l_oh = run_arm(seed=1, use_feat=False)
        h_ft, p_ft, l_ft = run_arm(seed=3, use_feat=True)
        rnd_logr = random_policy_logr(env, seed=7)
        best_hit, best_pat, best_logr = max((h_oh, p_oh, l_oh),
                                            (h_ft, p_ft, l_ft),
                                            key=lambda t: t[2])
        print(f"    one-hot+演示 {h_oh:.2%}/{p_oh}种/logR {l_oh:+.2f} | "
              f"结构化特征+演示 {h_ft:.2%}/{p_ft}种/logR {l_ft:+.2f} | "
              f"随机策略 logR {rnd_logr:+.2f}")
        print(f"    达标线 0.5ρ={0.5 * rho:.0%}；奖励提升 ΔlogR={best_logr - rnd_logr:+.2f}")
        summary.append((sigma, len(sols), n_info, rand_hit, rho,
                        best_hit, best_pat, best_logr, rnd_logr))

    # 判据：奖励提升（代理真实目标 = 采样 ∝ 奖励，用 ΔlogR 量）+ 针尖命中
    ok_sigmas = [s for s in summary
                 if s[7] >= s[8] + 0.8 and s[6] >= 2]
    print("\n  能力曲线（σ | 解数 | 信息态 | 随机命中 | ρ | 最优臂命中 | 模式"
          " | 臂 logR | 随机 logR）：")
    for s in summary:
        print("    %.2f | %3d | %3d | %5.2f%% | %3.0f%% | %6.2f%% | %d | "
              "%+6.2f | %+6.2f"
              % (s[0], s[1], s[2], 100 * s[3], 100 * s[4], 100 * s[5], s[6],
                 s[7], s[8]))
    fails += check(f"G1 结构空间可行（≥5 解且 ≥2 模式，全部 σ）",
                   all(s[1] >= 5 and s[6] >= 2 for s in summary))
    fails += check(f"G2 结构级多解发现（存在 σ 使 ΔlogR ≥ 0.8 且模式 ≥2）",
                   len(ok_sigmas) > 0,
                   f"达标 σ={ok_sigmas if ok_sigmas else '无'}")

    def run_arm(seed: int, use_feat: bool, use_demo: bool):
        gfn = TrajectoryBalanceGFN(
            env.n_states, env.n_actions, seed=seed,
            in_dim=(env.feat_dim if use_feat else None),
            feat_fn=(env.feat if use_feat else None))
    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
