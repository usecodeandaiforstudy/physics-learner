# -*- coding: utf-8 -*-
"""DB4 离策略接线（文档 §2.6.2 第 (i) 条，2026-10-02）。

文档原话：选 GFlowNet 而非 PPO/SAC 的理由之一是「**离策略——DB4 经验库成为
主要训练源**」。本模块把 DB4 记录映射回目标环境的状态 + 对数奖励，作为
**离线流预训练**（`TrajectoryBalanceGFN.pretrain_flow`）的数据源，再进入
在线 SubTB 微调。

映射规则（v0）：
  - DB4 记录 (domain=族, params) → 该族在结构空间的槽内 (类型, 箱)，
    单组件设计 = 槽 0 设该组件、其余槽置空；
  - logR = log(env.reward(状态))。

v0 边界：DB4 现只存单族设计（RL 运行产物），多组件结构记录属 DB4 v1 扩展。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__all__ = ["db4_to_states"]

_TYPE_OF = {"landau": 1, "elastic": 2}
_PARAM_OF = {"landau": "a", "elastic": "k"}


def db4_to_states(env) -> list[tuple[int, float]]:
    """DB4 记录 → [(结构空间状态, logR)]（仅族/参数可映射者）。"""
    from db4.db4_writer import DB4Writer

    out: list[tuple[int, float]] = []
    for rec in DB4Writer().load():
        t = _TYPE_OF.get(rec["domain"])
        if t is None:
            continue
        key = _PARAM_OF[rec["domain"]]
        if key not in rec.get("params", {}):
            continue
        val = float(rec["params"][key])
        grid = np.asarray(env.grid_for(t))
        b = int(np.argmin(np.abs(grid - val)))
        slots = [(t, b)] + [(0, 0)] * (env.n_slots - 1)
        state = env.encode(slots)
        out.append((state, float(np.log(max(env.reward(state), 1e-6)))))
    return out
