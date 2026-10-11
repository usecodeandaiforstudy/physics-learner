# -*- coding: utf-8 -*-
"""缺口③ 第 2 步：**图编辑环境** `GraphEditEnv`（2026-10-02，第 125 条）。

**接口镜像既有环境**（`a_network/structure_env.py` / `req_env.py`）：
    `n_states`（动态增长的状态表）、`n_actions`（**固定 42**）、
    `step(state, action) -> (next_state, reward)`、`absorbing`
⟹ **既有 GFN 训练器（`gflownet.train_step_subtb`）可直接复用**
（GFN 天然支持变长轨迹：`stop` 即终止）。

**状态**：`StructSpec` 的规范键 → 整数 id（`_key_to_id`，按需增长）。
**动作（固定 42 个，非法者用掩码屏蔽）**：
    0..9   `perturb(i, ±δ)`：i∈0..4（MAX_NODES−1），± 两个方向
    10..15 `add(comp)`：6 个组件（值取组件范围中点）
    16..20 `del(i)`：i∈0..4
    21..30 `link(i,j)`：i<j 的 10 个索引对
    31..40 `unlink(i,j)`：同 10 对
    41     `stop`：评估当前结构并终止
**奖励**（与既有 M4 环境同形）：
    `R = β·I_TY(β_op) − I_XT(β_op)`，取 `compute_backbone` + IB 曲线；
    **DPI 违规 → 0 并终止**（与既有环境的门禁语义一致）；
    终态按**规范键缓存**（G1 实测单次评估 0.00–0.26 s）。

**用法**：
    env = GraphEditEnv(beta=4.0)
    s = env.reset()
    s2, r = env.step(s, env.action_for("add", comp="gamma_k"))
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from a_network.structure_space import (COMPONENTS, MAX_NODES,  # noqa: E402
                                       StructSpec, add_edge, add_node,
                                       canonical_key, chain_is_dpi_legal,
                                       del_edge, del_node, node_sigma,
                                       perturb_param, swap_nodes,
                                       to_encoded_graph)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

DELTA = 0.25          # 参数扰动步长（占范围比例）
BETA_OP = 4.0

__all__ = ["GraphEditEnv"]


class GraphEditEnv:
    """变拓扑信息结构空间的**图编辑 MDP**。

    **两种任务模式（第 126 条）**：
      `task="ib"`（默认）：`R = β·I_TY − I_XT`（IB 型）——**注意**：DPI 下
        链越长信息越少 ⟹ 该奖励的最优策略是**停在单节点**（第 125 条实测），
        故它**不适合**评估组合能力；
      `task="constraint"`：**多约束满足 + 达标型奖励**（第 114 条口径）：
        `节点数 ≥ k ∧ I_TY ≥ a ∧ I_XT ≤ b` → `R=1`；否则按违反量软惩罚。
        **这是 G3（组合编辑）的正确任务形式**。
    """

    def __init__(self, beta: float = BETA_OP, max_nodes: int = MAX_NODES,
                 reward_cache: bool = True, task: str = "ib",
                 min_nodes: int = 3, i_ty_min: float = 0.0,
                 i_xt_max: float = 1e9):
        self.task = task
        self.min_nodes = int(min_nodes)
        self.i_ty_min = float(i_ty_min)
        self.i_xt_max = float(i_xt_max)
        self.beta = float(beta)
        self.max_nodes = int(max_nodes)
        # **动作布局用命名常量（2026-10-02 修）**：前面 7 个缺陷里有 3 个是
        # "索引/顺序"类（硬编码 21/31 等）⟹ 改为常量，避免漂移。
        self._n_comp = len(COMPONENTS)
        self._n_pairs = len([(i, j) for i in range(MAX_NODES)
                             for j in range(i + 1, MAX_NODES)])
        self.A_PERTURB = 0
        self.A_ADD = self.A_PERTURB + 2 * MAX_NODES
        self.A_DEL = self.A_ADD + self._n_comp
        self.A_LINK = self.A_DEL + MAX_NODES
        self.A_UNLINK = self.A_LINK + self._n_pairs
        self.A_SWAP = self.A_UNLINK + self._n_pairs
        self.A_STOP = self.A_SWAP + self._n_pairs
        self.n_actions = self.A_STOP + 1
        self._specs: list[StructSpec] = []
        self._key_to_id: dict[str, int] = {}
        self.absorbing = 0         # 0 号状态保留为"吸收/终止"哨兵
        self._specs.append(StructSpec((), ()))     # 占位（不会被使用）
        self._r_cache: dict[str, tuple] = {}
        self._use_cache = reward_cache
        self._link_pairs = [(i, j) for i in range(MAX_NODES)
                            for j in range(i + 1, MAX_NODES)]

    # ------------------------------------------------------------ 状态表
    def _intern(self, spec: StructSpec) -> int:
        k = canonical_key(spec)
        i = self._key_to_id.get(k)
        if i is None:
            i = len(self._specs)
            self._key_to_id[k] = i
            self._specs.append(spec)
        return i

    def spec_of(self, state: int) -> StructSpec:
        return self._specs[state]

    @property
    def n_states(self) -> int:
        return len(self._specs)

    def reset(self, rng: np.random.Generator | None = None,
              comp: str = "gauss_sigma") -> int:
        """初始状态：1 个**高斯族**节点（连续参数随机采样）。

        **为什么默认用 `gauss_sigma`（2026-10-02 修）**：`compute_ib_curve`
        要求 X 源节点为 `DIST-GAUSS`；若初始节点是非高斯族，则该 episode 的
        所有结构都无 IB 曲线（奖励恒 0，**死区**）⟹ 初版随机选组件会让
        GFN 在死区里空转。现默认从高斯族起步；`comp` 可显式改。
        """
        rng = rng or np.random.default_rng()
        if comp not in COMPONENTS:
            comp = "gauss_sigma"
        lo, hi = COMPONENTS[comp][2]
        return self._intern(StructSpec(((comp, float(rng.uniform(lo, hi))),), ()))

    # ------------------------------------------------------------ 动作
    def action_for(self, kind: str, **kw) -> int:
        """把语义动作映射到动作 id（便于外部构造/测试）。"""
        if kind == "perturb":
            i = int(kw["node"]); sign = 1 if kw.get("sign", 1) > 0 else 0
            return self.A_PERTURB + 2 * i + (0 if sign else 1)
        if kind == "add":
            return self.A_ADD + list(COMPONENTS).index(kw["comp"])
        if kind == "del":
            return self.A_DEL + int(kw["node"])
        if kind in ("link", "unlink"):
            i, j = int(kw["i"]), int(kw["j"])
            if i > j:
                i, j = j, i
            base = self.A_LINK if kind == "link" else self.A_UNLINK
            return base + self._link_pairs.index((i, j))
        if kind == "swap":
            i, j = int(kw["i"]), int(kw["j"])
            if i > j:
                i, j = j, i
            return self.A_SWAP + self._link_pairs.index((i, j))
        if kind == "stop":
            return self.A_STOP
        raise KeyError(kind)

    def valid_actions(self, state: int) -> np.ndarray:
        """合法动作掩码（**环境层强制**：非法动作不进入搜索空间）。"""
        m = np.zeros(self.n_actions, dtype=bool)
        spec = self.spec_of(state)
        n = spec.n_nodes
        for i in range(min(n, MAX_NODES)):
            m[self.A_PERTURB + 2 * i] = m[self.A_PERTURB + 2 * i + 1] = True
        if n < self.max_nodes:
            for c_i in range(self._n_comp):
                m[self.A_ADD + c_i] = True                  # add
        if n > 1:
            for i in range(min(n, MAX_NODES)):
                m[self.A_DEL + i] = True                    # del
        for idx, (i, j) in enumerate(self._link_pairs):
            # **只允许链式边 (i, i+1) 且连上后 σ 仍严格递增**（2026-10-02 修）：
            #   ① `_end_to_end_channel` 按边顺序复合 ⟹ 非链式边会让 IB 的
            #      σ_src 与噪声无关、MI 爆炸（实测 I_TY=3.325）；
            #   ② 链上 σ 必须递增（下游 Fisher 递减）否则 A1 判 DPI 违规。
            #   ⟹ **把 DPI 条件提到搜索空间层**（掩码），使搜索只在合法区进行。
            if j != i + 1:
                continue
            if i < n and j < n:
                if (i, j) in spec.edges:
                    m[self.A_UNLINK + idx] = True           # unlink
                else:
                    try:
                        trial = add_edge(spec, i, j)
                        m[self.A_LINK + idx] = bool(
                            chain_is_dpi_legal(trial))
                    except ValueError:
                        m[self.A_LINK + idx] = False
            # **swap（候选①）**：交换链序——让"高 F 节点在源位"可达。
            #   不做合法性预判（交换本身总是允许的；合法性由奖励门禁判）。
            if i < n and j < n:
                m[self.A_SWAP + idx] = True
        m[self.A_STOP] = True                               # stop
        return m

    def _apply(self, spec: StructSpec, action: int,
               rng: np.random.Generator) -> StructSpec | None:
        """执行编辑算子；**任何非法的编辑一律返回 None（不得抛异常）**。

        修因（2026-10-02）：初版直接调 `del_node` 等算子，遇到 n_nodes==1 的
        删除会 `ValueError: 至少保留 1 个节点` ⟹ 掩码一致性检查（**故意对
        被屏蔽的动作调用 `_apply`**）会崩。非法 → None 是统一契约。
        """
        try:
            return self._apply_inner(spec, action, rng)
        except (ValueError, IndexError, KeyError):
            return None

    def _apply_inner(self, spec: StructSpec, action: int,
                     rng: np.random.Generator) -> StructSpec | None:
        a = int(action)
        if self.A_PERTURB <= a < self.A_ADD:                 # perturb
            i, sign = (a - self.A_PERTURB) // 2, (
                1 if (a - self.A_PERTURB) % 2 == 0 else -1)
            if i >= spec.n_nodes:
                return None
            comp = spec.nodes[i][0]
            lo, hi = COMPONENTS[comp][2]
            return perturb_param(spec, i, sign * DELTA * (hi - lo))
        if self.A_ADD <= a < self.A_DEL:                     # add
            c = list(COMPONENTS)[a - self.A_ADD]
            lo, hi = COMPONENTS[c][2]
            return add_node(spec, c, 0.5 * (lo + hi))
        if self.A_DEL <= a < self.A_LINK:                    # del
            i = a - self.A_DEL
            return None if i >= spec.n_nodes else del_node(spec, i)
        if self.A_LINK <= a < self.A_UNLINK:                 # link
            idx = a - self.A_LINK
            if idx >= len(self._link_pairs):
                return None
            i, j = self._link_pairs[idx]
            if i >= spec.n_nodes or j >= spec.n_nodes:
                return None
            try:
                return add_edge(spec, i, j)
            except ValueError:
                return None
        if self.A_UNLINK <= a < self.A_SWAP:                 # unlink
            idx = a - self.A_UNLINK
            if idx >= len(self._link_pairs):
                return None
            i, j = self._link_pairs[idx]
            if i >= spec.n_nodes or j >= spec.n_nodes:
                return None
            try:
                return del_edge(spec, i, j)
            except ValueError:
                return None
        if self.A_SWAP <= a < self.A_STOP:                   # **swap（候选①）**
            idx = a - self.A_SWAP
            if idx >= len(self._link_pairs):
                return None
            i, j = self._link_pairs[idx]
            if i >= spec.n_nodes or j >= spec.n_nodes:
                return None
            return swap_nodes(spec, i, j)
        return None                                          # stop 由 step 处理

    # ------------------------------------------------------------ 奖励
    def _reward_of(self, spec: StructSpec) -> tuple[float, dict]:
        """终态奖励 + 诊断信息。**缓存也保留完整 info**（含 `dpi_ok`）——
        修因：初版缓存只存奖励值，缓存命中时 `info` 缺 `dpi_ok` ⟹ 统计 DPI
        合法率时被算成 0（实测 0/40 是**统计 bug**，非环境问题）。"""
        k = canonical_key(spec)
        if self._use_cache and k in self._r_cache:
            return self._r_cache[k]
        from a_network.analytic_backbone import compute_backbone
        from b_network.fingerprint_computer import FingerprintComputer
        g = to_encoded_graph(spec)
        # **未连边的多节点结构 = 不是信道 ⟹ 奖励 0**（2026-10-02 修）：
        #   无边图的 `_end_to_end_channel` 返回 `(1.0, 0.0)` ⟹ 信道无噪 ⟹
        #   `MI = H(X) ≈ 3.325 nats`（超容量上界，属"不可能的数字"）。
        if (g.metadata or {}).get("no_edges"):
            info = {"dpi_ok": True, "reason": "no_edges", "cached": False}
            out = (0.0, info)
            if self._use_cache:
                self._r_cache[k] = out
            return out
        bb = compute_backbone(g)
        info = {"dpi_ok": bool(bb["dpi"]["ok"]), "cached": False}
        if not bb["dpi"]["ok"]:
            r = 0.0
            info["reason"] = "dpi_violation"
        else:
            c = FingerprintComputer().compute_ib_curve(g)
            b = np.asarray(c.get("beta", []), dtype=float)
            if len(b) == 0:
                r = 0.0
                info["reason"] = "empty_curve"
            else:
                i_xt = float(np.interp(self.beta, b, np.asarray(c["I_XT"])))
                i_ty = float(np.interp(self.beta, b, np.asarray(c["I_TY"])))
                info.update({"I_XT": i_xt, "I_TY": i_ty})
                # **默认 IB 型奖励**（`task="ib"`）；约束型分支会覆盖它。
                #   修因：改约束分支时误删了这行 ⟹ `task="ib"` 下 `r` 未赋值
                #   （`UnboundLocalError`）。
                r = self.beta * i_ty - i_xt
                if self.task == "constraint":
                    # **达标型奖励（第 114/126 条）**：三项约束全满足 → R=1；
                    #   否则按"违反量"给 0–1 的连续软惩罚（利于 GFN 学习，
                    #   且**不使用 β**——阈值/约束型需求本就不该用 β）。
                    v_nodes = max(0.0, (self.min_nodes - spec.n_nodes)
                                  / max(self.min_nodes, 1))
                    v_ity = max(0.0, (self.i_ty_min - i_ty)
                                / max(abs(self.i_ty_min), 1e-6))
                    v_ixt = max(0.0, (i_xt - self.i_xt_max)
                                / max(abs(self.i_xt_max), 1e-6))
                    viol = max(v_nodes, v_ity, v_ixt)
                    r = 1.0 if viol <= 0 else max(0.0, 1.0 - viol)
                    info["task"] = "constraint"
                    info["violations"] = {"nodes": round(v_nodes, 4),
                                          "i_ty": round(v_ity, 4),
                                          "i_xt": round(v_ixt, 4)}
                    info["satisfied"] = bool(viol <= 0)
        out = (r, info)
        if self._use_cache:
            self._r_cache[k] = out
        return out

    def step(self, state: int, action: int,
             rng: np.random.Generator | None = None) -> tuple[int, float]:
        """执行动作；`stop`（或非法）→ 终止并给奖励。"""
        rng = rng or np.random.default_rng()
        spec = self.spec_of(state)
        if int(action) == self.A_STOP:                       # stop
            r, _info = self._reward_of(spec)
            return self.absorbing, r
        nxt = self._apply(spec, action, rng)
        if nxt is None:                                      # 非法 → 终止
            return self.absorbing, 0.0
        return self._intern(nxt), 0.0

    def reward(self, state: int) -> float:
        """查询某状态的终态奖励（供评估用）。"""
        return self._reward_of(self.spec_of(state))[0]


if __name__ == "__main__":
    import time
    env = GraphEditEnv()
    rng = np.random.default_rng(0)
    print("========== GraphEditEnv 冒烟（第 125 条 + 候选① swap） ==========")
    print("动作空间 %d（perturb 10 / add %d / del %d / link 10 / unlink 10 / "
          "**swap 10** / stop 1）"
          % (env.n_actions, len(COMPONENTS), MAX_NODES))

    # ① 随机编辑轨迹的奖励分布 + DPI 合法率
    rs, dpi_ok, lens, n_illegal = [], 0, [], 0
    t0 = time.time()
    for _ in range(40):
        s = env.reset(rng)
        for t in range(6):
            m = env.valid_actions(s)
            acts = np.flatnonzero(m)
            a = int(rng.choice(acts))
            a = env.action_for("stop") if (t == 5 or rng.random() < 0.2) else a
            s2, r = env.step(s, a, rng)
            if s2 == env.absorbing:
                rs.append(r)
                info = env._reward_of(env.spec_of(s))[1]
                dpi_ok += int(bool(info.get("dpi_ok")))
                lens.append(t + 1)
                break
            s = s2
    dt = time.time() - t0
    print("随机轨迹 40 条：状态表 %d；DPI 合法 %d/40；奖励 均值 %+.4f "
          "（范围 %+.4f…%+.4f）；平均步数 %.1f；总耗时 %.1fs"
          % (env.n_states, dpi_ok, float(np.mean(rs)), min(rs), max(rs),
             float(np.mean(lens)), dt))

    # ② 掩码正确性：非法动作必须被屏蔽
    s = env.reset(rng)
    m = env.valid_actions(s)
    print("初始状态合法动作数 %d/%d（应含 perturb/add/stop，不含 del/unlink）"
          % (int(m.sum()), env.n_actions))
    bad = [a for a in range(env.n_actions) if not m[a] and
           env._apply(env.spec_of(s), a, rng) is not None]
    print("掩码一致性：被屏蔽但实际可执行的动作 %d 个（应为 0）" % len(bad))

    # ③ 组合编辑可行性：**只走合法动作**（2026-10-02 修：初版直接 step 非法
    #    link ⟹ 落到吸收哨兵（空 spec）⟹ 打印成 "0 节点 / no_edges"，
    #    那是"非法动作被拒"而非"组合不可行"）。
    s = env.reset(rng)
    s, _ = env.step(s, env.action_for("add", comp="gamma_k"), rng)
    s, _ = env.step(s, env.action_for("add", comp="two_level_p"), rng)
    mm = env.valid_actions(s)
    links = [a for a in range(env.n_actions)
             if mm[a] and env.A_LINK <= a < env.A_UNLINK]
    print("组合后（3 节点）合法 link 动作数 %d（若为 0 ⟹ 该组合无法合法连边）"
          % len(links))
    for a in links[:2]:
        s, _ = env.step(s, a, rng)
    spec = env.spec_of(s) if s != env.absorbing else None
    if spec is None:
        print("组合编辑：动作被拒（落到哨兵）—— 说明该序列非法")
    else:
        r, info = env._reward_of(spec)
        print("组合编辑（add×2 + 合法 link×%d）→ %d 节点 / %d 边；奖励 %+.4f；"
              "DPI %s；构成 %s"
              % (len(links[:2]), spec.n_nodes, len(spec.edges), r,
                 info.get("dpi_ok"),
                 {k: round(v, 4) for k, v in info.items()
                  if k in ("I_TY", "I_XT")} or info.get("reason", "")))
    # 单节点高斯基线（奖励尺度参照，用于 G2 的任务对齐）
    s0 = env.reset(rng)
    r0, i0 = env._reward_of(env.spec_of(s0))
    print("单节点高斯基线：奖励 %+.4f；构成 %s"
          % (r0, {k: round(v, 4) for k, v in i0.items()
                  if k in ("I_TY", "I_XT")}))
    # ④ **诊断模式**：按结构规模看奖励分布 + 标记"不可能/不可信"的奖励
    print("\n诊断：按结构规模统计（%d 条随机终态）" % len(rs))
    by_size: dict[int, list[tuple[float, float]]] = {}
    for _ in range(60):
        s = env.reset(rng)
        for t in range(8):
            m = env.valid_actions(s)
            a = int(rng.choice(np.flatnonzero(m)))
            if t >= 2 and rng.random() < 0.3:
                a = env.action_for("stop")
            s2, _r = env.step(s, a, rng)
            if s2 == env.absorbing:
                spec_t = env.spec_of(s)
                rr, ii = env._reward_of(spec_t)
                by_size.setdefault(spec_t.n_nodes, []).append(
                    (rr, float(ii.get("I_TY", 0.0))))
                break
            s = s2
    CAP_MAX = 1.83          # 已知族容量上界（第 120 条 T2 实测）
    for n in sorted(by_size):
        v = [x[0] for x in by_size[n]]
        ity = [x[1] for x in by_size[n]]
        n_over = sum(1 for x in ity if x > CAP_MAX)
        print("  %d 节点：%2d 条；奖励 均值 %+.3f（最大 %+.3f）；"
              "I_TY 最大 %.3f；**I_TY 超容量上界 %.2f 的条数 %d**"
              % (n, len(v), float(np.mean(v)), max(v), max(ity), CAP_MAX,
                 n_over))
    print("  判读：I_TY 超上界 = **不可信**（噪声下限缺失会让 MI 爆炸）——"
          "已在 structure_space 加 η=5%% 噪声下限（MI 上界 ≈1.52 nats）")

    # ④ **候选① 验证：swap 能否把"σ 递减（非法）"链修成"σ 递增（合法）"**
    print("\n候选① 验证（swap 打开合法区）：")
    s = env.reset(rng)                                   # σ ≈ 0.707（θ=1 附近）
    s, _ = env.step(s, env.action_for("add", comp="ho_omega"), rng)
    s, _ = env.step(s, env.action_for("link", i=0, j=1), rng)
    spec_a = env.spec_of(s)
    ra, ia = env._reward_of(spec_a)
    sig_a = [None if node_sigma(spec_a, k) is None else
             round(float(node_sigma(spec_a, k)), 4)
             for k in range(spec_a.n_nodes)]
    print("  交换前：σ 序列 %s → 奖励 %+.4f（%s）"
          % (sig_a, ra, ia.get("reason", "ok")))
    m = env.valid_actions(s)
    sw = env.action_for("swap", i=0, j=1)
    print("  swap 动作在掩码中合法？%s" % bool(m[sw]))
    s2, _ = env.step(s, sw, rng)
    spec_b = env.spec_of(s2) if s2 != env.absorbing else None
    if spec_b is not None:
        rb, ib = env._reward_of(spec_b)
        sig_b = [None if node_sigma(spec_b, k) is None else
                 round(float(node_sigma(spec_b, k)), 4)
                 for k in range(spec_b.n_nodes)]
        print("  交换后：σ 序列 %s → 奖励 %+.4f（%s）"
              % (sig_b, rb, ib.get("reason", "ok")))
        print("  ⟹ %s" % ("**候选① 有效**（swap 打开了合法区）"
                          if rb > ra else
                          "swap 未改善（该组合互换后仍非法，需 perturb 推参数）"))
    else:
        print("  交换后被掩码拒绝（应不会发生：swap 不做合法性预判）")

    # ⑤ **合法区宽度（新，量化）**：枚举 2 节点链的组件组合（9×9=81），
    #    统计其中 **DPI 合法**（σ 递增）且**有边**的比例 —— 这是"合法区多宽"
    #    的直接数字，也是 G2/G3 能否训练的可行性依据。
    print("\n合法区宽度（2 节点链，枚举 9×9 组件组合）：")
    mid = {c: 0.5 * (COMPONENTS[c][2][0] + COMPONENTS[c][2][1])
           for c in COMPONENTS}
    n_total = n_legal = n_pos = 0
    best = (None, -1e9)
    for c0 in COMPONENTS:
        for c1 in COMPONENTS:
            n_total += 1
            sp = StructSpec(((c0, mid[c0]), (c1, mid[c1])), ((0, 1),))
            if not chain_is_dpi_legal(sp):
                continue
            n_legal += 1
            rr, _ii = env._reward_of(sp)
            if rr > 0:
                n_pos += 1
            if rr > best[1]:
                best = (("%s→%s" % (c0, c1)), rr)
    print("  组件对 %d 个：**σ 递增（DPI 合法）%d 个（%.0f%%）**；"
          "其中奖励>0 的 %d 个"
          % (n_total, n_legal, 100.0 * n_legal / n_total, n_pos))
    if best[0]:
        print("  最佳 2 节点链：%s → 奖励 %+.4f" % (best[0], best[1]))
    print("  判读：合法区窄则随机探索无信号（实测随机轨迹全 0），"
          "但**可达**（swap 演示 +0.2509）⟹ 需 RL/课程引导")

    print("\n结论：环境层可用（接口与既有 env 同形 ⟹ 可复用 GFN 训练器）")
