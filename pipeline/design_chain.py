# -*- coding: utf-8 -*-
"""R5-③：**组合链设计域**——高维设计域的载体（2026-10-02，第 141 条）。

## 定位（与缺口③ 的关系）
第 126 条已证：**在信息量目标下，组合结构无价值**（DPI ⟹ 链越长信息越少）
⟹ 变拓扑空间**不能**当"搜索空间"用。
**但可以当「设计域载体」**：设计 = 一条链（组件选择 + 连续参数 + 连接），
信息侧只给 **2 个标量**（X 节点 Fisher `q1`、边 MI）
⟹ **4–8 dof → 2 标量** ⟹ **多对一（非退化）** ✓
**⟹ 缺口③ 的价值从"搜索更好的结构"改为"承载高维设计域"**。

## 设计变量
    `nodes`：1–3 个组件（**σ 型**：gauss_sigma / ho_omega / elastic_k /
             truncnorm_sigma），每个带连续参数
    `link`：是否连边（链式，DPI 合法才可评估）
    ⟹ dof = 2 × n_nodes（+1 若允许选择连接）

## 预登记判据（**先写死**）
  | 判据 | 内容 | 不过的含义 |
  |---|---|---|
  | **C1 可评估** | 采样设计中 **≥30%** 能算出信息量（非 `no_edges`、非 DPI 违规） | 域太"死"，无效 |
  | **C2 非退化** | 同一 `(q1, MI)` 桶内设计中位 **≥5** | 一对一 ⟹ C 无用武之地 |
  | **C3 维度** | 有效 dof **≥4** | 否则不算"高维域" |
  | **C4 无不可能的数** | q1/MI 有限且 ≤ 已知容量上界（1.83 nats） | 实现有 bug |

用法：
    python pipeline/design_chain.py            # 预检（C1–C4）
    python pipeline/design_chain.py --n 400    # 更多采样
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

SIGMA_COMPS = ("gauss_sigma", "ho_omega", "elastic_k", "truncnorm_sigma")
CAP_MAX = 1.83          # 已知族容量上界（第 120 条）

# **向量编码（第 144 条，任务 4）**：链域是**混合型**（离散组件 + 连续参数），
#   生成模型需要一个**定长实向量**表示。约定 7 维：
#     `[comp0, val0, comp1, val1, comp2, val2, link]`
#   其中 `comp_k ∈ [0,1]` 线性映射到 `SIGMA_COMPS` 的索引（四舍五入）；
#   `val_k ∈ [0,1]` 映射到该组件的参数范围；**缺席槽位用 −0.5 作哨兵**
#   （两个字段都 <0 ⟹ 该槽位无节点）。`link ∈ {0,1}`。
VEC_DIM = 7
SENTINEL = -0.5


def to_vector(design: dict) -> np.ndarray:
    """链设计 → **定长 7 维向量**（供生成模型使用；缺席槽位 −0.5）。"""
    from a_network.structure_space import COMPONENTS
    z = np.full(VEC_DIM, SENTINEL, dtype=float)
    nodes = list(design.get("nodes") or [])
    for i, (comp, val) in enumerate(nodes[:3]):
        idx = SIGMA_COMPS.index(comp) if comp in SIGMA_COMPS else 0
        z[2 * i] = idx / max(len(SIGMA_COMPS) - 1, 1)
        lo, hi = COMPONENTS[comp][2]
        z[2 * i + 1] = float(np.clip((float(val) - lo) / max(hi - lo, 1e-12),
                                     0.0, 1.0))
    z[6] = 1.0 if design.get("link") else 0.0
    return z


def from_vector(z) -> dict:
    """**定长 7 维向量 → 链设计**（`to_vector` 的逆；越界自动夹取）。"""
    from a_network.structure_space import COMPONENTS
    z = np.asarray(z, dtype=float).ravel()
    if len(z) != VEC_DIM:
        raise ValueError("维度不符：%d vs %d" % (len(z), VEC_DIM))
    nodes = []
    for i in range(3):
        c_raw, v_raw = z[2 * i], z[2 * i + 1]
        if c_raw < 0.0 or v_raw < 0.0:          # 哨兵 ⟹ 该槽位为空
            continue
        idx = int(round(np.clip(c_raw, 0.0, 1.0) * (len(SIGMA_COMPS) - 1)))
        comp = SIGMA_COMPS[idx]
        lo, hi = COMPONENTS[comp][2]
        nodes.append((comp, float(lo + np.clip(v_raw, 0.0, 1.0) * (hi - lo))))
    if not nodes:                               # 至少 1 个节点（域约束）
        nodes = [(SIGMA_COMPS[0], 0.5 * (COMPONENTS[SIGMA_COMPS[0]][2][0]
                                         + COMPONENTS[SIGMA_COMPS[0]][2][1]))]
    return {"nodes": tuple(nodes), "link": bool(z[6] >= 0.5)}


def sample_chain(rng, n_nodes: int | None = None) -> dict:
    """采样一个链式设计（**只含 σ 型组件**，保证可建边）。"""
    from a_network.structure_space import COMPONENTS
    k = int(n_nodes or rng.integers(2, 4))          # 2–3 节点（dof 4–6）
    comps = [SIGMA_COMPS[int(rng.integers(0, len(SIGMA_COMPS)))]
             for _ in range(k)]
    vals = []
    for c in comps:
        lo, hi = COMPONENTS[c][2]
        vals.append(float(rng.uniform(lo, hi)))
    return {"nodes": tuple(zip(comps, vals)),
            "link": bool(rng.random() < 0.7)}


def measure_chain(design: dict) -> dict | None:
    """链设计 → 信息量：**X 节点 Fisher `q1`** 与 **边 MI**（经既有编码器）。"""
    from a_network.analytic_backbone import compute_backbone
    from a_network.structure_space import StructSpec, to_encoded_graph
    nodes = tuple(design["nodes"])
    edges = tuple((i, i + 1) for i in range(len(nodes) - 1)) \
        if design.get("link") else ()
    sp = StructSpec(nodes, edges)
    try:
        g = to_encoded_graph(sp)
        bb = compute_backbone(g)
    except Exception:  # noqa: BLE001
        return None
    if (g.metadata or {}).get("no_edges") and design.get("link"):
        return None
    q1 = None
    for nid, v in bb["node_fisher"].items():
        if v.get("covered") and v.get("fisher_trace"):
            q1 = float(v["fisher_trace"])
            break
    mi = [float(e["mi"]) for e in bb["edge_mi"] if e.get("mi") is not None]
    if q1 is None:
        return None
    return {"q1": q1, "mi": (float(np.sum(mi)) if mi else 0.0),
            # **单边最大 MI**（用于容量上界判据；**ΣMI 不可与单边上界比较**）
            "max_mi": (float(np.max(mi)) if mi else 0.0),
            "dpi_ok": bool(bb["dpi"]["ok"]), "n_edges": len(bb["edge_mi"])}


def main() -> int:
    n = 400
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    rng = np.random.default_rng(0)
    print("=" * 74)
    print("R5-③ 组合链设计域：可行性 + 非退化性预检")
    print("=" * 74)
    print("组件（σ 型，可建边）：%s" % list(SIGMA_COMPS))
    print("设计 dof：2 × 节点数（2–3）⟹ **4–6 dof**；信息侧 2 标量 (q1, ΣMI)")

    rows = []
    n_dead = 0
    for _ in range(n):
        d = sample_chain(rng)
        m = measure_chain(d)
        if m is None:
            n_dead += 1
            continue
        m.update({"design": d})
        rows.append(m)
    print("\n采样 %d：**可评估 %d（%.0f%%）**；死区 %d"
          % (n, len(rows), 100 * len(rows) / max(n, 1), n_dead))
    if not rows:
        print("**全部死区 ⟹ 该域不可用**（如实记录）")
        return 1

    q1 = np.array([r["q1"] for r in rows])
    mi = np.array([r["mi"] for r in rows])
    mmi = np.array([r["max_mi"] for r in rows])
    legal = [r for r in rows if r["dpi_ok"]]
    print("q1 范围 [%.4g, %.4g]（中位 %.4g）；ΣMI 范围 [%.4g, %.4g]；"
          "**单边最大 MI** 范围 [%.4g, %.4g]"
          % (q1.min(), q1.max(), float(np.median(q1)), mi.min(), mi.max(),
             mmi.min(), mmi.max()))
    print("**DPI 合法 %d/%d（%.0f%%）**——非法者（σ 递减）**不是有效设计**"
          % (len(legal), len(rows), 100 * len(legal) / max(len(rows), 1)))

    # ---- C1 可评估率 + **合法率**（判据用**合法子集**）----
    c1 = len(legal) / max(n, 1) >= 0.30
    print("\nC1 **可评估 %d/%d** ＋ **DPI 合法子集 %d（%.0f%% 的采样）**"
          "（判据 ≥30%%）⟹ %s"
          % (len(rows), n, len(legal), 100 * len(legal) / max(n, 1),
             "**通过** ✓" if c1 else "**未通过**"))

    # ---- C2 非退化性（**在合法子集上**做 2D 桶）----
    if not legal:
        print("C2 **无非有效设计 ⟹ 不适用**")
        c2 = False
        counts = {}
        med = 0.0
    else:
        q1l = np.array([r["q1"] for r in legal])
        mil = np.array([r["mi"] for r in legal])
        nbin = 5
        qe = np.quantile(q1l, np.linspace(0, 1, nbin + 1))
        me = np.quantile(mil, np.linspace(0, 1, nbin + 1))
        counts = {}
        for a, b in zip(q1l, mil):
            kq = int(np.clip(np.searchsorted(qe, a, side="right") - 1, 0,
                             nbin - 1))
            km = int(np.clip(np.searchsorted(me, b, side="right") - 1, 0,
                             nbin - 1))
            counts[(kq, km)] = counts.get((kq, km), 0) + 1
        med = float(np.median(list(counts.values()))) if counts else 0.0
        c2 = med >= 5
    print("C2 **非退化性**（**仅合法设计**）：非空桶 %d 个（%d×%d）；"
          "桶内设计中位 **%.1f**（最大 %d）（判据 ≥5）⟹ %s"
          % (len(counts), 5, 5, med,
             max(counts.values()) if counts else 0,
             "**通过** ✓" if c2 else "**未通过**"))

    # ---- C3 维度 ----
    dofs = sorted({2 * len(r["design"]["nodes"]) for r in rows})
    c3 = max(dofs) >= 4
    print("C3 **维度**：出现的 dof = %s（判据 ≥4）⟹ %s"
          % (dofs, "**通过** ✓" if c3 else "**未通过**"))

    # ---- C4 无不可能的数（**口径修正**：单边上界，不是 ΣMI；q1 无此上界）----
    #   **修因（2026-10-02）**：初版把 **Σ边 MI** 与「**单信道**容量上界 1.83」
    #   比较 ⟹ 3 节点链有 2 条边（各 ≤1.52）⟹ ΣMI 必然超界（实测 319 条"超界"）。
    #   同理 **q1 是 Fisher（非 nats）**，本就无 1.83 上界（反解表里 q1 达 55+）。
    #   ⟹ 正解：**单边 MI ≤ 1.83**（nats 口径）+ q1 有限正。
    over = int(np.sum(mmi > CAP_MAX))
    finite = bool(np.all(np.isfinite(q1)) and np.all(np.isfinite(mmi))
                  and np.all(q1 > 0))
    c4 = finite and over == 0
    print("C4 **无不可能的数**：q1 有限正=%s；**单边 MI** 超容量上界（%.2f nats）"
          "的条目 **%d** ⟹ %s（**注**：q1 是 Fisher、无 1.83 上界；ΣMI 是"
          "多边之和、亦不适用单边上界）"
          % (finite, CAP_MAX, over, "**通过** ✓" if c4 else "**未通过**"))

    ok = c1 and c2 and c3 and c4
    print("\n===== 预检结论 =====")
    print("该域可作为**高维设计域载体**：%s" % ("**是** ✓" if ok else "**否**（见上）"))
    if ok:
        print("下一步：登记进 `pipeline/design_space.DOMAINS`（dof=4–6，"
              "验证走既有编码器 + DPI），并按 §13.7 的 R1/R2 补 20-seed 闭环。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
