# -*- coding: utf-8 -*-
"""缺口③（结构空间完整形态）：**变拓扑信息结构空间**（2026-10-02，第 124 条）。

**问题**：现结构空间 = **2 槽定长链 + 离散参数**（226 态，v0.1 容量边界 3376 态
即超），文档语义要求的是**变拓扑图**（增删节点/边 + 连续参数 + 条件注入）。
20-seed 复检已证：**定长链上的 GFN 无法做组合结构编辑**（M6.2 四项 0/20）。

**本模块提供（第 1 步：表示层）**：
  `StructSpec`  —— 可哈希的变拓扑规范表示（节点 = 组件+连续参数，边 = 索引对）
  `to_encoded_graph(spec)` —— 转成 `core.fingerprint` 管线可消费的 `EncodedGraph`
  `perturb / add_node / del_node / add_edge / del_edge / stop` —— 编辑算子
  `random_spec / all_specs` —— 采样与（小规模）穷举

**兼容性设计（关键）**：
  - **单节点** spec → **委托 `SystemEncoder`**（用合成 entry）⟹ 与 DB1/DB2/A/C
    全链**完全同构**（同一编码器、同一指纹、同一 IB 曲线口径）；
  - **多节点** spec → 直接构造：源节点（X 角色）+ 表示节点 + 边按
    **加性噪声高斯信道**（`cov[0,1] == cov[0,0]`，与 `analytic_backbone` 的
    `_ADDNOISE_TOL` 判据一致）⟹ DPI 的 A1/A2 检查可通过。

**用法**：
    from a_network.structure_space import COMPONENTS, StructSpec, to_encoded_graph
    spec = StructSpec(nodes=(("gauss_sigma", 1.0), ("gamma_k", 2.0)),
                      edges=((0, 1),))
    g = to_encoded_graph(spec)
    from a_network.analytic_backbone import compute_backbone
    bb = compute_backbone(g)      # → node_fisher / edge_mi / dpi
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from core.channels import (EncodedGraph, Edge, GaussianChannel,  # noqa: E402
                           Node)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

__all__ = ["COMPONENTS", "StructSpec", "to_encoded_graph", "canonical_key",
           "canonical_spec", "perturb_param", "add_node", "del_node",
           "add_edge", "del_edge", "random_spec", "enumerate_small",
           "MAX_NODES", "node_sigma", "chain_is_dpi_legal",
           "SIGMA_COMPONENTS"]

MAX_NODES = 5          # 变拓扑上限（控制搜索空间；v0.1 为 2 槽）

# **组件库**：节点 = 某一族的"可调主参数"。**参数名与范围不硬编码**——
# 从该族的**真实 DB1 条目模板**自动派生（初版硬编码 `sigma`/`theta` 等 ⟹
# `KeyError: 'theta'`，因为 `ou` 编码器要 `theta` 而组件表给的是 `sigma`）。
# 值 = (DB1 family 名, 兜底范围)；参数名与真实范围由模板给出。
COMPONENTS_SPEC: dict[str, tuple[str, tuple[float, float]]] = {
    "gauss_sigma":    ("ou", (0.2, 5.0)),
    "decay_rate":     ("radioactive_decay", (0.1, 5.0)),
    "gamma_k":        ("gamma", (0.5, 8.0)),
    "weibull_shape":  ("weibull", (0.5, 5.0)),
    "two_level_p":    ("two_level", (0.05, 0.95)),
    "landau_a":       ("landau", (-2.0, 2.0)),
    # **σ 型（高斯族）组件——多节点建边需要它们**（2026-10-02 加）：
    #   边的噪声方差须与**节点的 σ** 同尺度（见 `to_encoded_graph` 的推导），
    #   故只有带 σ 的组件才能参与建边；这三族提供额外的 σ 自由度。
    "elastic_k":      ("elastic", (0.1, 10.0)),
    "ho_omega":       ("harmonic_oscillator", (0.2, 5.0)),
    "truncnorm_sigma": ("truncated_gaussian", (0.2, 3.0)),
}

# σ 型组件（可参与建边）：其编码器产出的节点 params 含 "sigma"
SIGMA_COMPONENTS = ("gauss_sigma", "elastic_k", "ho_omega", "truncnorm_sigma")

# 被跳过的边计数（诊断用：非链式边或非 σ 型节点）
LOG_SKIPPED: list[int] = []

_TEMPLATE_CACHE: dict[str, dict] = {}


def family_template(family: str) -> dict:
    """取该族的**真实 DB1 条目**作模板（缓存）——保证编码器所需参数齐全。"""
    if family in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[family]
    d = _ROOT / "db1" / "entries"
    for p in sorted(d.glob("*.json")):
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if e.get("family") == family:
            _TEMPLATE_CACHE[family] = e
            return e
    raise KeyError("DB1 中无 family=%s 的模板条目" % family)


def component_param(comp: str) -> tuple[str, tuple[float, float]]:
    """该组件的**可调参数名**与范围——取自族模板的 `scan_param`（或首参）。"""
    fam, fb = COMPONENTS_SPEC[comp]
    tpl = family_template(fam)
    pname = tpl.get("scan_param")
    params = tpl.get("parameters") or []
    if not pname:
        pname = params[0]["name"] if params else "x"
    for pr in params:
        if pr.get("name") == pname and pr.get("range"):
            r = pr["range"]
            return pname, (float(r[0]), float(r[1]))
    return pname, fb


def _component_table() -> dict[str, tuple[str, str, tuple[float, float]]]:
    out = {}
    for comp, (fam, fb) in COMPONENTS_SPEC.items():
        pname, rng = component_param(comp)
        out[comp] = (fam, pname, rng)
    return out


COMPONENTS = _component_table()

# 单节点/多节点统一走族模板 ⟹ `COMPONENT_TO_FAMILY` 即 spec 里的族名
COMPONENT_TO_FAMILY = {c: f for c, (f, _fb) in COMPONENTS_SPEC.items()}


@dataclass(frozen=True)
class StructSpec:
    """变拓扑结构规范（**不可变、可哈希** ⟹ 可作 RL 状态 id）。"""
    nodes: tuple[tuple[str, float], ...]          # (组件名, 参数值)
    edges: tuple[tuple[int, int], ...]            # (src_idx, dst_idx)
    beta: float = 1.0

    def canonical(self) -> str:
        return canonical_key(self)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)


def node_sigma(spec: StructSpec, i: int) -> float | None:
    """该节点的 σ（用于判断链上 σ 是否递增 ⟹ DPI 是否合法）。

    带缓存（编码一个节点很快，但 RL 里会被高频调用）。
    """
    key = (COMPONENT_TO_FAMILY[spec.nodes[i][0]], spec.nodes[i][0],
           round(float(spec.nodes[i][1]), 8))
    if key in _SIGMA_CACHE:
        return _SIGMA_CACHE[key]
    from b_network.system_encoder import SystemEncoder
    sub = SystemEncoder().encode(_synthetic_entry(spec, i))
    nd = sub.nodes[0]
    p = nd.params or {}
    val = None
    for k in ("sigma", "sigma_stat", "scale"):
        if k in p:
            try:
                v = float(p[k])
                if v > 0:
                    val = v
                    break
            except (TypeError, ValueError):
                continue
    _SIGMA_CACHE[key] = val
    return val


_SIGMA_CACHE: dict[tuple, float | None] = {}


def chain_is_dpi_legal(spec: StructSpec) -> bool:
    """链上 σ 是否**严格递增**（⟹ 下游 Fisher 递减 ⟹ A1 合法）。

    这是本空间里唯一需要"预判"的 DPI 条件；把提到**动作掩码层**可让搜索
    只在合法区进行（否则大量动作落在 DPI 违规的奖励 0 区）。
    """
    prev = None
    for i in range(spec.n_nodes):
        s = node_sigma(spec, i)
        if s is None:
            return False
        if prev is not None and s <= prev:
            return False
        prev = s
    return True


def swap_nodes(spec: StructSpec, i: int, j: int) -> StructSpec:
    """**交换两个节点的位置（2026-10-02 加，候选①）**。

    为什么需要它：DPI 要求链上 **σ 严格递增**（下游 Fisher 递减），而
    `node 0` 恒为 X（源）。**链序本身就是结构的一部分**——把高 F（小 σ）
    的组件放到源位才能合法。没有 swap 时，随机组合几乎无法到达合法区
    （实测 2–5 节点奖励**全为 0**）。
    交换节点位置并**同步重映射边**（保持图的拓扑语义）。
    """
    if not (0 <= i < spec.n_nodes and 0 <= j < spec.n_nodes) or i == j:
        raise ValueError("非法交换 (%d, %d)" % (i, j))
    nodes = list(spec.nodes)
    nodes[i], nodes[j] = nodes[j], nodes[i]

    def _remap(k: int) -> int:
        return j if k == i else (i if k == j else k)

    edges = tuple(sorted(tuple(sorted((_remap(a), _remap(b))))
                         for a, b in spec.edges))
    return StructSpec(tuple(nodes), edges, spec.beta)


def canonical_spec(spec: StructSpec) -> StructSpec:
    """**规范化 spec = 保序**（2026-10-02 修，概念修正）。

    **为什么不再排序节点**：本空间是**有向链**（`X → T1 → …`），
    **链序本身就是结构**——`[A,B]` 与 `[B,A]` 是**不同的信道**（源不同、
    DPI 合法性也可能不同）。初版按 `(组件, 参数)` **排序节点**做"同构去重"
    ⟹ 排序把 `swap` 撤销、并把 agent 对"谁是源"的选择**抹掉**
    （实测：swap 后 σ 变递增，却因排序回到原序而报 `no_edges`）。
    ⟹ 现规则：**保序**（节点序列即链序），只对**边集**排序。
    状态去重的口径随之改为「**有序链 + 边集**」。
    """
    return StructSpec(spec.nodes, tuple(sorted(spec.edges)), spec.beta)


def canonical_key(spec: StructSpec) -> str:
    """规范化键：由 `canonical_spec`（保序）单一真源导出。

    **保序** ⟹ `[A,B]` 与 `[B,A]` 是**不同状态**（不同源）——这是正确的，
    因为链的方向有语义。
    """
    cs = canonical_spec(spec)
    s = json.dumps({"n": [[c, round(p, 6)] for c, p in cs.nodes],
                    "e": [list(e) for e in cs.edges],
                    "b": round(cs.beta, 6)}, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------- 编辑算子
def add_node(spec: StructSpec, comp: str, value: float) -> StructSpec:
    if comp not in COMPONENTS:
        raise KeyError("未知组件 %s" % comp)
    if spec.n_nodes >= MAX_NODES:
        raise ValueError("超出节点上限 %d" % MAX_NODES)
    lo, hi = COMPONENTS[comp][2]
    v = float(np.clip(value, lo, hi))
    return StructSpec(spec.nodes + ((comp, v),), spec.edges, spec.beta)


def del_node(spec: StructSpec, idx: int) -> StructSpec:
    if not (0 <= idx < spec.n_nodes):
        raise IndexError(idx)
    if spec.n_nodes <= 1:
        raise ValueError("至少保留 1 个节点")
    nodes = tuple(n for i, n in enumerate(spec.nodes) if i != idx)
    remap = {i: (i if i < idx else i - 1) for i in range(spec.n_nodes)
             if i != idx}
    edges = tuple(sorted((remap[a], remap[b]) for a, b in spec.edges
                         if a != idx and b != idx))
    return StructSpec(nodes, edges, spec.beta)


def add_edge(spec: StructSpec, i: int, j: int) -> StructSpec:
    if i == j or not (0 <= i < spec.n_nodes and 0 <= j < spec.n_nodes):
        raise ValueError("非法边 %d→%d" % (i, j))
    if (i, j) in spec.edges:
        raise ValueError("边已存在")
    return StructSpec(spec.nodes, tuple(sorted(spec.edges + ((i, j),))),
                      spec.beta)


def del_edge(spec: StructSpec, i: int, j: int) -> StructSpec:
    if (i, j) not in spec.edges:
        raise ValueError("边不存在")
    return StructSpec(spec.nodes,
                      tuple(e for e in spec.edges if e != (i, j)), spec.beta)


def perturb_param(spec: StructSpec, idx: int, delta: float) -> StructSpec:
    """对第 idx 个节点的参数施加连续扰动（裁剪到组件范围）。"""
    if not (0 <= idx < spec.n_nodes):
        raise IndexError(idx)
    comp, val = spec.nodes[idx]
    lo, hi = COMPONENTS[comp][2]
    new = float(np.clip(val + delta, lo, hi))
    nodes = tuple((c, (new if i == idx else v))
                  for i, (c, v) in enumerate(spec.nodes))
    return StructSpec(nodes, spec.edges, spec.beta)


# --------------------------------------------------------- 采样 / 穷举
def random_spec(rng: np.random.Generator, n_nodes: int | None = None,
                edge_prob: float = 0.5) -> StructSpec:
    comps = list(COMPONENTS)
    n = int(n_nodes or rng.integers(1, MAX_NODES + 1))
    n = max(1, min(n, MAX_NODES))
    nodes = []
    for _ in range(n):
        c = comps[int(rng.integers(0, len(comps)))]
        lo, hi = COMPONENTS[c][2]
        nodes.append((c, float(rng.uniform(lo, hi))))
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                edges.append((i, j))
    return StructSpec(tuple(nodes), tuple(sorted(edges)))


def enumerate_small(max_nodes: int = 2, n_levels: int = 2
                    ) -> list[StructSpec]:
    """小规模穷举（用于 G1 的"可达规模"对比：v0.1 定长链 = 226 态）。"""
    out: list[StructSpec] = []
    comps = list(COMPONENTS)
    grids = {c: np.linspace(COMPONENTS[c][2][0], COMPONENTS[c][2][1], n_levels)
             for c in comps}

    def rec(nodes, edges):
        if len(nodes) >= 1:
            out.append(StructSpec(tuple(nodes), tuple(sorted(edges))))
        if len(nodes) >= max_nodes:
            return
        for c in comps:
            for v in grids[c]:
                nodes.append((c, float(v)))
                rec(nodes, edges)
                nodes.pop()

    rec([], [])
    return out


# ------------------------------------------------- 转 EncodedGraph（管线兼容）
def _synthetic_entry(spec: StructSpec, i: int) -> dict:
    """由**族模板深拷贝**构造合成 entry（2026-10-02 修）。

    初版自建 entry 只声明一个参数 ⟹ 编码器读其他参数时 `KeyError`
    （`ou` 要 `theta`、`landau` 要 `b`/`kT`）。改为**以该族真实 DB1 条目为
    模板**，只覆盖被调参数的 `default`（`scan_param` 同步改），
    其余参数原样保留 ⟹ **编码器所需参数永远齐全**。
    """
    import copy
    comp, val = spec.nodes[i]
    fam = COMPONENT_TO_FAMILY[comp]
    # COMPONENTS[comp] = (family, pname, (lo, hi)) —— **三元组**（初版此处误按
    # 二元组解包 ⟹ `ValueError: too many values to unpack`）
    _fam, pname, (lo, hi) = COMPONENTS[comp]
    e = copy.deepcopy(family_template(fam))
    e["system_id"] = "SYNTH-%s-%d" % (canonical_key(spec), i)
    e["domain"] = "synthetic"
    e["scan_param"] = pname
    for pr in e.get("parameters", []):
        if pr.get("name") == pname:
            pr["default"] = float(np.clip(val, lo, hi))
            pr["range"] = [lo, hi]
    e["provenance"] = {"source": "AI+verified",
                       "human_review_status": "pending",
                       "note": "结构空间合成节点（第 124 条）"}
    return e


def to_encoded_graph(spec: StructSpec) -> EncodedGraph:
    """`StructSpec` → `EncodedGraph`（可被 `compute_backbone` 消费）。

    **图必须是规范键的函数（2026-10-02 修，关键）**：`canonical_key` 会
    **排序节点**以做无序去重，而初版 `to_encoded_graph` 用 **spec 原始顺序**
    建图（X = node 0）⟹ 两个同构但顺序不同的 spec 会得到**同一状态键**却
    **不同的图**（不同 X 角色）⟹ **同一状态奖励不一致**（RL 的致命问题）。
    现统一：**先规范化（排序节点 + 重映射边），再按规范化顺序建图**，
    使 `图 = f(键)`。
    """
    spec = canonical_spec(spec)
    from b_network.system_encoder import SystemEncoder
    enc = SystemEncoder()

    if spec.n_nodes == 1:
        g = enc.encode(_synthetic_entry(spec, 0))
        g.beta = spec.beta
        g.metadata = dict(g.metadata or {})
        g.metadata["synthetic_spec"] = canonical_key(spec)
        return g

    # **不做节点重排（2026-10-02 修，关键）**：初版把"第一个高斯节点"挪到
    #   首位当 X，但**边仍用 spec 索引**去索引重排后的列表 ⟹ 边连错节点 ⟹
    #   IB 的 σ_src 与边噪声无关 ⟹ `MI` 爆炸（实测 I_TY=3.325）且 A1/A2 违规。
    #   现规则：**X 恒为 node 0**（由环境保证 node 0 是 σ 型组件）。
    nodes: list[Node] = []
    for i, (comp, _val) in enumerate(spec.nodes):
        sub = enc.encode(_synthetic_entry(spec, i))
        src_node = sub.nodes[0]
        role = "X" if i == 0 else "T"
        nodes.append(Node(node_id="%s__%s__%d" % (comp, role, i),
                          distribution_family=src_node.distribution_family,
                          params=dict(src_node.params),
                          noise_type=src_node.noise_type,
                          dimension=src_node.dimension))

    # **边的方向与噪声按「节点 σ」定（2026-10-02 修，见下）**；
    # 不再需要预计算 F_trace（初版用 `1/F` 定标噪声 ⟹ MI 爆炸）。

    # **拓扑约束：只支持「以 node 0 为根的链」（2026-10-02 定，重要）**
    #   `FingerprintComputer._end_to_end_channel` 把边**按列表顺序**复合为
    #   (slope=∏slope_k, var_cond=Σ var_k)，**没有拓扑排序**；而 IB 用的源方差
    #   是 **X 节点（`__X__`）的 σ²**。若 X 不在边链上、或边不是链式，
    #   则 σ_src 与噪声方差**无关** ⟹ `MI = ½ln(1+σ_src²/Σvar_n)` **爆炸**
    #   （实测 I_TY = **3.325 nats**，超所有已知族容量上界 1.83）。
    #   ⟹ 本空间的**边只允许 (i, i+1) 形式的链式边**（X = node 0），
    #   其它边**跳过并计数**（不静默丢弃）。任意 DAG 拓扑需扩展
    #   `_end_to_end_channel`（core 改动，列为后续）。
    NOISE_FLOOR_ETA = 0.05

    def _sigma_of(nd: Node) -> float | None:
        p = nd.params or {}
        for k in ("sigma", "sigma_stat", "scale"):
            if k in p:
                try:
                    v = float(p[k])
                    if v > 0:
                        return v
                except (TypeError, ValueError):
                    continue
        return None

    sigmas = [_sigma_of(nd) for nd in nodes]
    edges: list[Edge] = []
    n_skipped = 0
    for (a, b) in sorted(spec.edges):
        if b != a + 1:                     # 非链式边 → 跳过（见上）
            n_skipped += 1
            continue
        sa, sb = sigmas[a], sigmas[b]
        if sa is None or sb is None:       # 非 σ 型节点 → 无法建边
            n_skipped += 1
            continue
        # **不再"兜底"伪造边（2026-10-02 修）**：初版在 σ 递减时令 `sb = sa`
        # ⟹ 造出一条**假的**合法边，掩盖了 A1 违规。现**如实建边**：
        #   `var_n = max(σ_v² − σ_u², η·σ_u²)`；σ 递减时该项为负 → 取 η 下限，
        #   而 A1 会如实判其 **DPI 违规**（环境奖励 0）——这正是"约束"语义。
        var_u = sa ** 2
        var_n = max(sb ** 2 - var_u, NOISE_FLOOR_ETA * var_u)
        cov = np.array([[var_u, var_u], [var_u, var_u + var_n]])
        edges.append(Edge(src=nodes[a].node_id, dst=nodes[b].node_id,
                          channel=GaussianChannel(cov=cov,
                                                  block_shape=(1, 1))))
    if n_skipped:
        LOG_SKIPPED.append(n_skipped)
    g = EncodedGraph(nodes=nodes, edges=edges, beta=spec.beta,
                     max_nodes=MAX_NODES)
    g.metadata = {"synthetic_spec": canonical_key(spec),
                  "derivation": "变拓扑结构空间合成图（加性噪声高斯边）",
                  "citation": "本仓库 a_network/structure_space.py（第 124 条）"}
    # **无边的多节点图不可评估（2026-10-02 修，关键）**：
    #   `_end_to_end_channel` 对**无边图**返回 `(slope=1, var_cond=0)`
    #   ⟹ 信道变成**无噪**（Y = X）⟹ `MI = H(X)` 被算成 **3.325 nats**
    #   （超所有已知族容量上界 1.83，属"不可能的数字"）。
    #   这不是 η/σ 的问题，而是**"未连边的结构不是信道"**——
    #   故显式标记，由环境返回奖励 0（不算曲线）。
    if spec.n_nodes > 1 and not edges:
        g.metadata["no_edges"] = True
    if n_skipped:
        g.metadata["edges_skipped"] = int(n_skipped)
    return g


if __name__ == "__main__":
    import time
    from a_network.analytic_backbone import compute_backbone

    print("组件库（参数名/范围由族模板自动派生）：")
    for c, (fam, pname, rng) in COMPONENTS.items():
        print("  %-16s family=%-20s param=%-8s range=[%.3g, %.3g]"
              % (c, fam, pname, rng[0], rng[1]))

    # **确定性样例（带边；用 σ 型组件构链）**：1 / 2 / 3 节点链
    cases = [
        StructSpec((("gauss_sigma", 1.0),), ()),
        StructSpec((("gauss_sigma", 1.0), ("ho_omega", 2.0)), ((0, 1),)),
        StructSpec((("gauss_sigma", 1.0), ("ho_omega", 2.0),
                     ("elastic_k", 3.0)), ((0, 1), (1, 2))),
        # 反例：非链式边 (0,2) 应被**跳过并计数**
        StructSpec((("gauss_sigma", 1.0), ("ho_omega", 2.0),
                     ("elastic_k", 3.0)), ((0, 1), (0, 2))),
    ]
    print("\nG1 表示能力（确定性样例，含边）：")
    ok_all = 0
    for s in cases:
        t0 = time.time()
        g = to_encoded_graph(s)
        bb = compute_backbone(g)
        dt = time.time() - t0
        n_nodes = len(g.nodes)
        n_cov = sum(1 for v in bb["node_fisher"].values() if v["covered"])
        mi = [round(float(e["mi"]), 6) for e in bb["edge_mi"]]
        sig = [None if node_sigma(s, i) is None else
               round(float(node_sigma(s, i)), 4) for i in range(s.n_nodes)]
        legal = bool(bb["dpi"]["ok"]) and n_cov == n_nodes
        ok_all += int(legal)
        print("  spec %d 节点 / %d 边 → 产出图 %d 节点 / %d 边："
              "Fisher 覆盖 %d/%d；边 MI %s；**DPI ok=%s**"
              "；σ 序列 %s；跳边 %s；耗时 %.2fs"
              % (s.n_nodes, len(s.edges), n_nodes, len(g.edges),
                 n_cov, n_nodes, mi, bb["dpi"]["ok"], sig,
                 (g.metadata or {}).get("edges_skipped", 0), dt))
        if not bb["dpi"]["ok"]:
            print("     A1 违规 %d 条 / A2 违规 %d 条 ⟹ **该结构 DPI 非法**"
                  "（环境会给奖励 0；这属'约束生效'而非 bug）"
                  % (len(bb["dpi"]["a1_violations"]),
                     len(bb["dpi"]["a2_violations"])))
    print("\nG1 判据（修订）：**存在 ≥1 条 DPI 合法的链** 且 "
          "**非链式边被确实跳过** ⟹ 通过（σ 递减的链非法是'约束生效'）")
    # **保序验证（2026-10-02 修正）**：链序有语义 ⟹ `[A,B]` 与 `[B,A]`
    #   **必须**是不同的键（初版"排序去重"把它们并成一个，导致 swap 失效）。
    a = StructSpec((("gamma_k", 2.0), ("gauss_sigma", 1.0)), ((0, 1),))
    b = StructSpec((("gauss_sigma", 1.0), ("gamma_k", 2.0)), ((0, 1),))
    same_edges = StructSpec((("gauss_sigma", 1.0), ("ho_omega", 2.0)), ((0, 1),))
    same_edges2 = StructSpec((("gauss_sigma", 1.0), ("ho_omega", 2.0)), ((0, 1),))
    print("\n保序验证：**链序不同 ⟹ 键不同**？%s（期望 True）"
          % (a.canonical() != b.canonical()))
    print("同一结构重复构造 ⟹ 键相同？%s（期望 True）"
          % (same_edges.canonical() == same_edges2.canonical()))
    print("\n小规模穷举（max_nodes=2, 2 档）：%d 个结构（对比 v0.1 定长链 226 态）"
          % len(enumerate_small(max_nodes=2, n_levels=2)))
    print("\nG1 结论：%s" % ("**通过**（变拓扑 ⟹ 合法指纹通路已建立）"
                            if ok_all == len(cases) else
                            "**未通过**（%d/%d）" % (ok_all, len(cases))))
