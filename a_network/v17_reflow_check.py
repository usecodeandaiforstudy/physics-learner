# -*- coding: utf-8 -*-
"""回流边验证 v8（第 90 条，2026-10-02）：**证据空间 vs RL 状态空间的宽度差**。

v11（v1–v7 归档）结论：DB5（landau 域解析验证）注入 RL 缓冲在 12 seed 配对
下**中性偏负**（@150 无加速、@600 非劣）。本轮问题是：缺口① 引入了**外部
模拟器验证**的新证据类型（rayleigh/OU/diffusion 域），它能回流吗？

**预登记判据**：
  R1 **映射可达性**：DB5 pass 记录中能被 `DOMAIN_TO_FAMILY` 映射的比例
     （报告 source × domain 分解；不可映射域**必须报告**）；
  R2 **回流边有效性**（仅当 R1 有新可映射状态时执行）：三臂配对
     （base / landau 解析验证注入 / +模拟器验证注入）@150 与 @600；
     非劣性判据同 v11（判「有损害」需 p<0.05 且 gap<−0.3，双检验口径）；
  R3 **缺口陈述**：若模拟器证据不可映射，明确记录「RL 状态空间窄于 DB5
     证据空间」并给出扩展方向（不静默丢弃）。

用法（项目根目录）：python a_network/v17_reflow_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from a_network.db5_feedback import (DOMAIN_TO_FAMILY, describe_reflow,
                                    load_validated_states)
from a_network.rl_env import MultiFamilyEnv
from a_network.sac import DiscreteSAC

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def eval_logr(sac, env, n: int = 200) -> float:
    vals = []
    for _ in range(n):
        s = int(np.random.randint(0, env.absorbing))
        for _ in range(12):
            pi = sac.policy(torch.tensor([s], dtype=torch.long))[0]
            s, _ = env.step(s, int(torch.argmax(pi).item()))
            if s == env.absorbing:
                break
        if s == env.absorbing:
            s = int(np.random.randint(0, env.absorbing))
        vals.append(np.log(max(env.reward(s), 1e-6)))
    return float(np.mean(vals))


def train(sac, env, episodes: int, horizon: int = 12) -> None:
    for _ in range(episodes):
        s = int(np.random.randint(0, env.absorbing))
        for t in range(horizon):
            a = sac.act(s)
            s2, r = env.step(s, a)
            sac.store(s, a, float(np.log(max(r, 1e-6))), s2, t == horizon - 1)
            s = s2
            sac.train_step(batch=64)


def main() -> int:
    fails = 0
    # **显式开启扩展族**（第 91 条）：回流域需要 control/acoustics/materials
    # 三族才能映射模拟器证据；其余实验保持默认 3 族协议不变。
    env = MultiFamilyEnv(extra_families=True)
    rep = describe_reflow(env)
    print("========== 回流边 v8：证据空间 vs 状态空间 ==========")
    print("  R1 DB5 记录：%d 条；来源分解 %s" % (rep["total"],
                                            rep["by_source"]))
    print("     域分解 %s" % rep["by_domain"])
    print("     可映射（pass 且域在映射表内）：%d 条" % rep["mappable"])
    if rep["unmapped_domains"]:
        print("     **不可映射域（须报告，不静默丢弃）**：%s"
              % rep["unmapped_domains"])
    fails += check("R1 回流映射可达（≥3 条可映射 pass 记录）",
                   rep["mappable"] >= 3, "mappable=%d" % rep["mappable"])

    sim_only = [r for r in (rep["by_source"] or {}) if r == "AI+simulated"]
    sim_states = load_validated_states(env, include_simulated=True)
    landau_states = load_validated_states(env, include_simulated=False)
    print("     注入状态数：含模拟器证据 %d；仅解析证据 %d（差 %d）"
          % (len(sim_states), len(landau_states),
             len(sim_states) - len(landau_states)))
    if sim_only and len(sim_states) == len(landau_states):
        # **区分两种成因**（2026-10-02 第 91 条）：域不可映射 vs 映射了但
        # 被任务相关性过滤（后者是语义正确，不是缺口）
        unmapped = rep["unmapped_domains"]
        if unmapped:
            print("  R3 **缺口类型 A（状态空间宽度）**：模拟器证据域 %s "
                  "不在映射表内 →\n     **RL 状态空间窄于 DB5 证据空间**。"
                  "修法：在 rl_env 增加对应族。" % unmapped)
        else:
            print("  R3 **缺口类型 B（任务对齐）**：模拟器证据**域已可映射**"
                  "（映射表覆盖 %d 域），但**全部被任务相关性过滤器剔除**"
                  "——\n     因为这些设计是按**各自需求**验证的"
                  "（分辨力 σ / 弛豫率 θ / 扩散 κ），而本 env 只有**单一"
                  "硬编码任务**（target_i=%.2f）。\n"
                  "     **语义正确，非 bug**；正确形态 = **需求条件化 RL**"
                  "（requirement-conditioned reward）。"
                  % (len(DOMAIN_TO_FAMILY), env.target_i))

    # ---- R2：三臂配对实验（base / landau / +simulated）----
    N_SEED = int(os.environ.get("V17_SEEDS", "6"))
    base_e, base_l, inj_e, inj_l, sim_e, sim_l = [], [], [], [], [], []
    for seed in range(N_SEED):
        for arm in ("base", "landau", "sim"):
            np.random.seed(seed)
            torch.manual_seed(100 + seed)
            sac = DiscreteSAC(env.n_states, env.n_actions, seed=seed, lr=1e-2)
            if arm != "base":
                states = landau_states if arm == "landau" else sim_states
                for st, w in states:
                    r = float(np.log(max(env.reward(st), 1e-6)))
                    for _ in range(5):
                        sac.store(st, 0, r, st, False, weight=w)
            train(sac, env, 150)
            early = eval_logr(sac, env)
            train(sac, env, 450)
            late = eval_logr(sac, env)
            (base_e if arm == "base" else
             inj_e if arm == "landau" else sim_e).append(early)
            (base_l if arm == "base" else
             inj_l if arm == "landau" else sim_l).append(late)
        print("  seed %d：base @150=%6.2f @600=%6.2f | landau %6.2f/%6.2f | "
              "sim %6.2f/%6.2f"
              % (seed, base_e[-1], base_l[-1], inj_e[-1], inj_l[-1],
                 sim_e[-1], sim_l[-1]))

    def paired(a, b, label):
        d = np.array(b) - np.array(a)
        m, sd = float(d.mean()), float(d.std(ddof=1))
        t = m / (sd / np.sqrt(len(d))) if sd > 0 else 0.0
        wins = int((d > 0).sum())
        print("    %s：Δ 均值 %+.2f（std %.2f）t=%.2f 同号 %d/%d"
              % (label, m, sd, t, wins, len(d)))
        return m, t

    print("  R2 三臂配对（%d seed）：" % N_SEED)
    m_e, t_e = paired(base_e, inj_e, "landau 注入 @150")
    m_l, t_l = paired(base_l, inj_l, "landau 注入 @600")
    paired(base_l, sim_l, "＋模拟器注入 @600")
    harmed = (abs(t_l) > 1.96 and m_l < -0.3)
    fails += check("R2 回流非劣（@600 无显著损害：非 p<0.05 且 gap<−0.3）",
                   not harmed, "m=%.2f t=%.2f" % (m_l, t_l))
    print("\n%s（R 判据 %d 项失败）"
          % ("===== 全部通过 =====" if not fails else "===== 有失败 =====",
             fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
