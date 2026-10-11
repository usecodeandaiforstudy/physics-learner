"""A v2 解析背板（M1，2026-09-29）。

把 core 现有解析量组装成环境的「确定性计算层」：
    EncodedGraph → {节点 Fisher(解析), 边 MI(解析), DPI 硬判据, 覆盖标记, 可微性标注}

用途（`environment_a_v2_design.md` §2.1）：A v2 环境里 RL 只消费背板的确定
分量；神经残余只负责 covered=False 的部分。本模块是**精确计算器**——所有值
直接复用 DB2 的生成路径（`core.fingerprint.compute_graph_fingerprint`），
V1 验收 = 与 DB2 全量逐位一致。

M1 范围（与设计文档 M1 对齐，torch 可微移植留 M2）：
- 节点 Fisher：`core.fingerprint.compute_node_fisher`（17 族解析式；landau 为
  数值求积——标记 `closed_form=False`，M2 用固定网格 autograd 移植）；
- 边 MI：高斯解析 `mi_gaussian_analytic`；
- **DPI 硬判据**（第 26 条约束 A 的精确版，离线验证过 A1 33/33、A2 21/21）：
  A1 沿信息流 log F_trace 单调非增；A2 加性噪声边闭式
  MI = −½ln(1 − F_v/F_u)。解析数据上违规 = 结构非法（硬拒绝依据）；
- 覆盖标记：Fisher/MI 是否解析可算（未覆盖 → 神经残余 + 门控，见 §2.5）；
- 可微性标注：`closed_form`（闭式，易移植 torch）/ `torch_ready`（M1 恒 False，
  M2 移植后置 True）。

用法：
    from a_network.analytic_backbone import compute_backbone
    result = compute_backbone(graph)   # graph = SystemEncoder.encode(entry)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.fingerprint import compute_graph_fingerprint
from core.mutual_info import mi_gaussian_analytic

__all__ = ["compute_backbone", "DPI_A1_TOL", "DPI_A2_TOL"]

# DPI 判据容差（解析数据应机器精度成立；留余量防浮点噪声）
DPI_A1_TOL = 1e-9          # A1：log F 单调非增（log 空间）
DPI_A2_TOL = 1e-6          # A2：加性噪声边闭式（MI 绝对容差）
_ADDNOISE_TOL = 1e-9       # cov[0,1] == cov[0,0] 判据（相对容差）

# 闭式 Fisher 的分布族（landau 为数值求积，其余 16 族均为解析闭式）
_LANDAU = "DIST-LANDAU"


def compute_backbone(graph) -> dict[str, Any]:
    """解析背板：在信息结构图上做确定性计算。

    参数
    ----
    graph : EncodedGraph
        SystemEncoder 编码出的信息结构图。

    返回
    ----
    dict
        {
          "node_fisher": {node_id: {"fisher_trace", "fisher_det", "covered",
                                    "closed_form", "torch_ready", "family"}},
          "edge_mi": [{"src", "dst", "mi", "covered", "is_additive_noise"}],
          "dpi": {"ok", "n_a1_checks", "n_a2_checks", "a1_violations",
                  "a2_violations"},
          "action_s": None,   # M2（作用量奖励接入时启用）
        }
    """
    fp = compute_graph_fingerprint(graph)

    # 1) 节点 Fisher + 覆盖/可微性标注
    node_fisher: dict[str, dict] = {}
    for node in graph.nodes:
        f = fp["nodes"].get(node.node_id) or {}
        trace = f.get("fisher_trace")
        det = f.get("fisher_det")
        covered = trace is not None and det is not None
        node_fisher[node.node_id] = {
            "fisher_trace": trace if covered else None,
            "fisher_det": det if covered else None,
            "covered": covered,
            "closed_form": covered and node.distribution_family != _LANDAU,
            "torch_ready": False,   # M1：torch 可微移植留 M2
            "family": node.distribution_family,
        }

    # 2) 边 MI + 加性噪声标记 + DPI 硬判据
    edge_mi: list[dict] = []
    a1_violations: list[dict] = []
    a2_violations: list[dict] = []
    n_a1 = n_a2 = 0
    for e in graph.edges:
        cov = getattr(e.channel, "cov", None) if e.channel is not None else None
        mi = e.mutual_information
        covered = mi is not None
        is_noise = False
        if cov is not None:
            cov_a = np.asarray(cov, dtype=float)
            if cov_a.size >= 4 and cov_a[0, 0] > 0:
                is_noise = bool(abs(cov_a[0, 1] - cov_a[0, 0])
                                <= _ADDNOISE_TOL * abs(cov_a[0, 0]))
        edge_mi.append({
            "src": e.src, "dst": e.dst, "mi": mi,
            "covered": covered, "is_additive_noise": is_noise,
        })

        # A1：沿信息流 u→v，log F_trace(v) ≤ log F_trace(u)
        fu = node_fisher.get(e.src, {}).get("fisher_trace")
        fv = node_fisher.get(e.dst, {}).get("fisher_trace")
        if fu is not None and fv is not None and fu > 0 and fv > 0:
            n_a1 += 1
            lu, lv = float(np.log(fu)), float(np.log(fv))
            if lv > lu + DPI_A1_TOL:
                a1_violations.append({"edge": [e.src, e.dst],
                                      "logF_u": lu, "logF_v": lv})

            # A2：加性噪声边闭式 MI = −½ln(1 − F_v/F_u)
            if is_noise and mi is not None:
                n_a2 += 1
                if fv >= fu:
                    a2_violations.append({"edge": [e.src, e.dst],
                                          "note": "F_v ≥ F_u，闭式无定义"})
                else:
                    expect = -0.5 * float(np.log(1.0 - fv / fu))
                    if abs(float(mi) - expect) > DPI_A2_TOL:
                        a2_violations.append({"edge": [e.src, e.dst],
                                              "mi": float(mi),
                                              "expect": expect})

    return {
        "node_fisher": node_fisher,
        "edge_mi": edge_mi,
        "dpi": {
            "ok": not a1_violations and not a2_violations,
            "n_a1_checks": n_a1,
            "n_a2_checks": n_a2,
            "a1_violations": a1_violations,
            "a2_violations": a2_violations,
        },
        "action_s": None,   # M2
    }
