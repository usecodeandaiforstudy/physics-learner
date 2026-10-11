# -*- coding: utf-8 -*-
"""专家演示生成（缺口②，2026-10-02）：结构空间中求解轨迹。

用途：为 P_F 提供初始化演示（M6/M6.2 诊断的处方——组合/结构动作在 on-policy
训练早期会因「用错就负反馈」而死亡，需演示先给正概率）。

方法：随机起点 + 宽度优先（BFS）在动作图上搜索到解状态（reward ≥ thresh）
的最短路径；演示对 = 路径上全部 (state, action)。
"""

from __future__ import annotations

from collections import deque

__all__ = ["generate_demos"]


def generate_demos(env, n_starts: int = 40, max_depth: int = 10,
                   thresh: float = 0.8, seed: int = 0) -> list[tuple[int, int]]:
    import numpy as np
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    n_solved = 0
    for _ in range(n_starts):
        start = int(rng.integers(0, env.absorbing))
        # BFS
        prev: dict[int, tuple[int, int]] = {start: (-1, -1)}
        q = deque([start])
        goal = None
        while q:
            s = q.popleft()
            depth = 0
            t = s
            while prev[t][0] != -1:
                t = prev[t][0]
                depth += 1
            if depth >= max_depth:
                continue
            if s != start and env.reward(s) >= thresh:
                goal = s
                break
            for a in range(env.n_actions):
                nxt, _ = env.step(s, a)
                if nxt == env.absorbing or nxt in prev:
                    continue
                prev[nxt] = (s, a)
                q.append(nxt)
        if goal is None:
            continue
        n_solved += 1
        # 回溯路径（含终止动作）
        path = []
        cur = goal
        while prev[cur][0] != -1:
            p, a = prev[cur]
            path.append((p, a))
            cur = p
        path.reverse()
        path.append((goal, env.term_action))   # 解状态上终止（动作编号随槽数变）
        pairs.extend(path)
    return pairs
