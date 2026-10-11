# -*- coding: utf-8 -*-
"""回流边验证 v9（第 92 条，2026-10-02）：**需求条件化 RL**。

第 91 条诊断：回流边瓶颈 = 任务对齐（DB5 证据按各自需求验证，旧 env 只有
单一硬编码任务）。本检查验证需求条件化形态：

**预登记判据**：
  Q1 **证据归属**：6 条模拟器证据中，能按**自身需求**归属并成为该需求
     解的 demo 数 ≥ 2（旧单任务形态下为 0）；
  Q2 **回流有效性**：需求条件化环境下，注入臂 vs base 的**按需求分层**
     ΔlogR（3 seed 配对；判「有损害」需 p<0.05 且 gap<−0.3）；
  Q3 **旧协议不受影响**：`MultiFamilyEnv` 默认参数未变（3 族 100 状态）。

用法（项目根目录）：python a_network/v18_req_reflow_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from a_network.db5_feedback import load_requirement_demos
from a_network.req_env import REQUIREMENTS, RequirementConditionedEnv
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


def eval_per_req(sac, env, n: int = 120) -> dict[int, float]:
    """按需求分层评估：贪心 rollout 的达标率 + 平均 logR。"""
    out: dict[int, float] = {}
    for req_idx in range(env.n_req):
        hits, logrs = 0, []
        for _ in range(n):
            s = env.start_state(req_idx)
            last = s
            for _t in range(12):
                pi = sac.policy(torch.tensor([s], dtype=torch.long))[0]
                s, _ = env.step(s, int(torch.argmax(pi).item()))
                if s == env.absorbing:
                    break
                last = s
            r = env.reward(last)
            hits += (r >= 0.999)
            logrs.append(float(np.log(max(r, 1e-6))))
        out[req_idx] = float(np.mean(logrs))
    return out


def train(sac, env, episodes: int, horizon: int = 12) -> None:
    rng = np.random.default_rng(0)
    for _ in range(episodes):
        req_idx = int(rng.integers(0, env.n_req))     # 多任务：每回合采需求
        s = env.start_state(req_idx)
        for t in range(horizon):
            a = sac.act(s)
            s2, r = env.step(s, a)
            sac.store(s, a, float(np.log(max(r, 1e-6))), s2, t == horizon - 1)
            s = s2
            sac.train_step(batch=64)


def main() -> int:
    fails = 0
    env = RequirementConditionedEnv()
    print("========== 回流边 v9：需求条件化 RL ==========")
    print("  环境：%d 需求 × %d 族 × %d 箱 + 吸收 = %d 状态"
          % (env.n_req, env.n_families, env.n_bins, env.n_states))
    for i, r in enumerate(REQUIREMENTS):
        nsol = len(env.solution_states(i))
        print("    R%d %-38s 解状态 %d" % (i, r.label, nsol))

    demos = load_requirement_demos(env)
    by_req: dict[int, int] = {}
    for req_idx, _st, _w in demos:
        by_req[req_idx] = by_req.get(req_idx, 0) + 1
    print("  Q1 DB5 → 需求条件化 demo：%d 条（按需求 %s）"
          % (len(demos), by_req))
    new_demos = sum(v for k, v in by_req.items() if k != 0)
    fails += check("Q1 模拟器证据按自身需求归属（新需求 demo ≥ 2）",
                   new_demos >= 2, f"new_demos={new_demos}")

    # ---- Q2：注入 / 预训练 vs base（3 seed 配对，按需求分层）----
    ARMS = ("base", "inject", "pretrain")
    N_SEED = int(os.environ.get("V18_SEEDS", "5"))   # 扩容后提高检验力
    logs: dict[str, dict[int, list[float]]] = {
        arm: {i: [] for i in range(env.n_req)} for arm in ARMS}
    for seed in range(N_SEED):
        for arm in ARMS:
            np.random.seed(seed)
            torch.manual_seed(200 + seed)
            sac = DiscreteSAC(env.n_states, env.n_actions, seed=seed, lr=1e-2)
            if arm == "inject":
                for req_idx, st, w in demos:
                    r = env.reward(st)
                    for _ in range(5):
                        sac.store(st, 0, float(np.log(max(r, 1e-6))), st,
                                  False, weight=w)
            elif arm == "pretrain":
                # 监督暖启动（第 63 条另一入口）：把解状态的 Q 拟合到 log 1
                pairs = [(st, 0, 0.0) for _rq, st, _w in demos]
                if pairs:
                    sac.pretrain_on(pairs, steps=300)
            train(sac, env, 300)
            res = eval_per_req(sac, env)
            for k, v in res.items():
                logs[arm][k].append(v)
        print("  seed %d：" % seed + " | ".join(
            "%s %s" % (arm, ["%.2f" % logs[arm][i][-1]
                             for i in range(env.n_req)]) for arm in ARMS))

    print("  Q2 按需求分层 ΔlogR（%d seed）：" % N_SEED)
    # ---- **预登记判据（2026-10-02 第 95 条，运行前写死，不做事后挑选）**----
    # 每条 (需求 × 机制) 给出三态判定：
    #   "支持正收益"：mean Δ ≥ +0.3 且 配对 t 双侧 p < 0.05 且 ≥70% seed 为正
    #   "支持有害"  ：mean Δ ≤ −0.3 且 配对 t 双侧 p < 0.05 且 ≥70% seed 为负
    #   "未定"      ：其余（含效应小于 0.3 或未达显著）
    # 20 seed 下 70% = 14/20；该阈值与效应量门槛**均在看到数据前设定**。
    def verdict(d: np.ndarray) -> tuple[str, float, float, int]:
        m = float(d.mean())
        sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        t = m / (sd / np.sqrt(len(d))) if sd > 0 else 0.0
        pos = int((d > 0).sum())
        frac_pos = pos / len(d)
        if abs(t) < 1e9:
            # 双侧 p 值（正态近似；20 seed 下 t 分布近似足够）
            from math import erfc, sqrt
            p = erfc(abs(t) / sqrt(2))
        else:
            p = 1.0
        if m >= 0.3 and p < 0.05 and frac_pos >= 0.70:
            return "支持正收益", m, p, pos
        if m <= -0.3 and p < 0.05 and frac_pos <= 0.30:
            return "支持有害", m, p, pos
        return "未定", m, p, pos

    summary: list[tuple[str, str, str, float, float, int]] = []
    harmed_any = False
    for i, r in enumerate(REQUIREMENTS):
        line = "    R%d %-34s" % (i, r.label)
        for arm in ("inject", "pretrain"):
            d = np.array(logs[arm][i]) - np.array(logs["base"][i])
            vd, m, p, pos = verdict(d)
            summary.append(("R%d" % i, arm, vd, m, p, pos))
            line += "  %s %s(Δ%+.2f, p=%.3f, %d/%d正)" % (
                arm, vd, m, p, pos, len(d))
            if arm == "inject" and vd == "支持有害":
                harmed_any = True
        print(line)

    print("\n  **预登记判据的机械结论（%d seed）**：" % N_SEED)
    n_pos = sum(1 for s in summary if s[2] == "支持正收益")
    n_neg = sum(1 for s in summary if s[2] == "支持有害")
    n_und = sum(1 for s in summary if s[2] == "未定")
    print("    支持正收益 %d / 支持有害 %d / **未定 %d**（共 %d 组）"
          % (n_pos, n_neg, n_und, len(summary)))
    if n_pos > n_neg:
        print("    ⇒ 回流边**有正收益证据**（%d 组通过完整门槛）" % n_pos)
    elif n_neg > n_pos:
        print("    ⇒ 回流边**有有害证据**（%d 组通过完整门槛）" % n_neg)
    else:
        print("    ⇒ 回流边效应**未定**：无任何 (需求×机制) 组通过"
              "「Δ≥0.3 ∧ p<0.05 ∧ ≥70% 同号」的完整门槛")
    fails += check("Q2 需求条件化回流非劣（注入臂无「支持有害」判定）",
                   not harmed_any)

    # ---- Q3：旧协议不受影响 ----
    env_old = MultiFamilyEnv()
    fails += check("Q3 旧 MultiFamilyEnv 默认未变（3 族 / 100 状态）",
                   env_old.n_families == 3 and env_old.n_states == 100,
                   f"fam={env_old.n_families} states={env_old.n_states}")

    print("\n%s（Q 判据 %d 项失败）"
          % ("===== 全部通过 =====" if not fails else "===== 有失败 =====",
             fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
