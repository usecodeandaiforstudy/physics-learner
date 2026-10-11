"""A v2 作用量奖励（M2，2026-09-29）。

信息作用量 = 环境的奖励本体（design doc §2.4）：
    A[q] = Σ_k [ ½ g_ij Δq_i Δq_j − (I(X;T) − β·I(T;Y)) ]_k · dt
RL 的结构编辑轨迹 q(t)（节点/参数/容量编辑序列）上离散求和；dt=1。
`core/action.py:info_action` 从零调用升级为核心调用（第 26 条缺口闭合）——
本模块是它的「轨迹级」包装：info_action 管单点拉格朗日量，本模块管
整条轨迹的作用量与合法性。

v0 边界（文档化，后续里程碑收紧）：
- 动能项的度量 g 用节点 `fisher_trace` 近似：单参数族编辑该参数时精确
  （trace=I(θ)）；双参数族用 trace 高估（待背板提供逐参数 Fisher 分量）；
- β 规则 v0 占位（strictness→β 表），正式标定属 B v1 / DB3 B 角色复核；
- 非法结构（DPI 违反）→ `legal=False`，奖励惩罚由调用方施加（设计 §2.3）。
"""

from __future__ import annotations

from typing import Any

__all__ = ["kinetic_step", "beta_from_requirement", "trajectory_action"]

# v0 β 规则：DB3 对的 strictness → IB 权衡（占位；正式标定属 B v1 / DB3 复核）
_BETA_V0: dict[str, float] = {"定理": 4.0, "经验": 1.0}
_DEFAULT_BETA = 1.0


def kinetic_step(fisher_trace: float, dtheta: float) -> float:
    """单参数编辑的动能增量 ½ g·Δθ²（g = fisher_trace，v0 近似）。"""
    return 0.5 * float(fisher_trace) * float(dtheta) * float(dtheta)


def beta_from_requirement(pair: dict) -> dict:
    """DB3 需求对 → β（v0 占位规则：strictness 映射，保证 β>0 与接口连通）。

    正式标定属 B v1 / DB3 B 角色复核（需求方向/目标值 → IB 权衡的真实映射），
    本规则只是让「需求通路 → 奖励参数」先跑通。
    """
    strictness = str(pair.get("strictness") or "经验")
    beta = float(_BETA_V0.get(strictness, _DEFAULT_BETA))
    return {"beta": beta, "rule": "v0-strictness", "strictness": strictness}


def trajectory_action(states: list[dict], beta: float,
                      dt: float = 1.0) -> dict[str, Any]:
    """轨迹作用量 + 合法性。

    参数
    ----
    states : list[dict]，按时间序，每项 = {
        "bb": compute_backbone(graph) 的输出,
        "I_XT": float,     # 该状态的信息压缩量（nats）
        "I_TY": float,     # 该状态的任务相关量（nats）
        "edit": (node_id, dtheta) 或 None,   # 从上一状态到本状态的编辑
    }
    beta : float
        IB 权衡参数。
    dt : float
        每步时间间隔（离散轨迹默认 1）。

    返回
    ----
    {"action", "kinetic", "potential_sum", "legal", "illegal_states", "per_step"}

    约定：动能项在**编辑发生前**的状态度量上求值（g(θ_{k-1})），势能项逐状态
    求和；DPI 非法的状态计入 illegal_states 并置 legal=False。
    """
    kinetic = 0.0
    potential_sum = 0.0
    per_step: list[dict] = []
    illegal: list[int] = []
    for k, st in enumerate(states):
        bb = st["bb"]
        if not bb.get("dpi", {}).get("ok", True):
            illegal.append(k)
        pot = float(st["I_XT"]) - beta * float(st["I_TY"])
        potential_sum += pot * dt
        kin_k = 0.0
        edit = st.get("edit")
        if edit is not None and k > 0:
            node_id, dtheta = edit
            pre = states[k - 1]["bb"]["node_fisher"].get(node_id, {})
            g = pre.get("fisher_trace")
            if g is not None:
                kin_k = kinetic_step(g, dtheta)
            else:
                kin_k = float("nan")   # 无解析 Fisher：由调用方经门控处理
        kinetic += (kin_k if kin_k == kin_k else 0.0) * dt
        per_step.append({"index": k, "kinetic": kin_k, "potential": pot})
    action = kinetic - potential_sum
    return {
        "action": action,
        "kinetic": kinetic,
        "potential_sum": potential_sum,
        "legal": not illegal,
        "illegal_states": illegal,
        "per_step": per_step,
    }
