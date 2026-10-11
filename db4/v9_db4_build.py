# -*- coding: utf-8 -*-
"""DB4 建库（v0，2026-10-02）：用 RL 双路径的实际采样做种子经验。

来源（真实代理运行，非合成占位）：
  - GFN（M5 双峰任务）：200 条终端结构（agent=gfn-tb-v0）；
  - SAC（M7 跨族任务）：200 条终端结构（agent=sac-v0）；
  - 随机基线：100 条（agent=random，未来离策略训练的负样本）。

全部记录 source=AI+verified、anchor_eligible=false（红线）；校验 +
幂等去重由 DB4Writer 承担。

用法（项目根目录）：python db4/v9_db4_build.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from a_network.gflownet import TrajectoryBalanceGFN
from a_network.rl_env import DesignTaskEnv, MultiFamilyEnv
from a_network.sac import DiscreteSAC
from db4.db4_writer import DB4Writer, build_record

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def main() -> int:
    writer = DB4Writer()
    n_new = n_dup = n_err = 0

    # ---- GFN：M5 双峰任务（DesignTaskEnv bimodal）----
    env_b = DesignTaskEnv(task="bimodal", mode_a=1.5, sigma_r=0.25)
    gfn = TrajectoryBalanceGFN(env_b.n_states, env_b.n_actions, seed=1)
    for _ in range(300):
        gfn.train_step(env_b, horizon=8, batch=64, entropy_bonus=0.1)
    terms = gfn.sample_terminals(env_b, 200, horizon=8)
    for s in terms:
        a_val = float(env_b.a_grid[int(s)])
        rec = build_record("landau", {"a": a_val, "b": 1.0, "kT": 1.0},
                           float(env_b.table()["I_TY"][int(s)]),
                           float(env_b.reward(int(s))),
                           float(np.log(max(env_b.reward(int(s)), 1e-6))),
                           "gfn-tb-v0", "bimodal")
        st = writer.append(rec)
        n_new += st.get("ok", False) and not st.get("duplicate")
        n_dup += st.get("duplicate", False)
        n_err += not st.get("ok", False)

    # ---- SAC：M7 跨族任务（MultiFamilyEnv requirement）----
    env_m = MultiFamilyEnv()
    sac = DiscreteSAC(env_m.n_states, env_m.n_actions, seed=1, lr=1e-2)
    for ep in range(300):
        s = int(np.random.randint(0, env_m.n_states))
        for t in range(12):
            a = sac.act(s)
            s2, r = env_m.step(s, a)
            sac.store(s, a, float(np.log(max(r, 1e-6))), s2,
                      t == 11)
            s = s2
            sac.train_step(batch=64)
    terms_m = []
    for _ in range(200):
        s = int(np.random.randint(0, env_m.n_states))
        for t in range(12):
            import torch
            pi = sac.policy(torch.tensor([s], dtype=torch.long))[0]
            s, _ = env_m.step(s, int(torch.argmax(pi).item()))
        terms_m.append(s)
    for s in terms_m:
        fam, b = env_m._decode(int(s))
        domain = env_m.family_names[fam]
        params = {env_m.param_names[fam]: float(env_m.grids[fam][b])}
        i_ty = float(env_m.i_tables[fam][b])
        r = float(env_m.reward(int(s)))
        rec = build_record(domain, params, i_ty, r,
                           float(np.log(max(r, 1e-6))), "sac-v0",
                           "requirement_m6")
        st = writer.append(rec)
        n_new += st.get("ok", False) and not st.get("duplicate")
        n_dup += st.get("duplicate", False)
        n_err += not st.get("ok", False)

    # ---- 随机基线（负样本经验）----
    rng = np.random.default_rng(0)
    for _ in range(100):
        s = int(rng.integers(0, env_m.n_states))
        fam, b = env_m._decode(s)
        domain = env_m.family_names[fam]
        params = {env_m.param_names[fam]: float(env_m.grids[fam][b])}
        i_ty = float(env_m.i_tables[fam][b])
        r = float(env_m.reward(s))
        rec = build_record(domain, params, i_ty, r,
                           float(np.log(max(r, 1e-6))), "random",
                           "baseline")
        st = writer.append(rec)
        n_new += st.get("ok", False) and not st.get("duplicate")
        n_dup += st.get("duplicate", False)
        n_err += not st.get("ok", False)

    # ---- 汇总 ----
    recs = writer.load()
    by_agent = {}
    for r in recs:
        by_agent.setdefault(r["agent"], []).append(r["reward"])
    print("DB4 建库汇总：新增 %d、去重跳过 %d、校验失败 %d（总 %d 条）"
          % (n_new, n_dup, n_err, len(recs)))
    for ag, rs in sorted(by_agent.items()):
        print("  %-10s n=%3d  平均 reward=%.3f  最大=%.3f"
              % (ag, len(rs), float(np.mean(rs)), float(np.max(rs))))
    top = sorted(recs, key=lambda r: -r["reward"])[:5]
    print("  Top-5 结构：")
    for r in top:
        print("    %s %s reward=%.3f agent=%s"
              % (r["structure_id"], r["params"], r["reward"], r["agent"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
