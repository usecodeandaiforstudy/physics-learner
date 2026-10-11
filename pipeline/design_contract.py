# -*- coding: utf-8 -*-
"""设计解码契约 + 需求反解（2026-10-02，第 106 条）——接上断点①②。

**背景**（第 104 条跑通审计暴露的两个断点）：
  ① RL 的 G* 是**离散状态**（需求×族×箱），C 需要**领域参数**——缺统一契约；
  ② C 以**领域参数**为条件，而需求给的是**信息指标 q**——缺 q→参数反解。

**本模块给出两件事**：
  **A. 统一解码契约** `state_to_design(env, state) -> DesignSpec`
     把 RL 状态解成「领域 + 族 + 参数向量 + 度量值 + 是否达标」的规范结构。
  **B. 需求反解** `param_bound_for(metric, q_req, family) -> (param, bound_kind, value)`
     由**已登记的桥接式**反解出参数界（只对**可解析反解**的族生效，
     其余**如实报告为「无反解」**，不猜）：
       - rayleigh：      q1 = 4/σ²        ⟹ σ_max = √(4/q1_req)（σ 越小越好）
       - control_ou：    q9 = θ           ⟹ θ_min = q9_req
       - materials_diff：q7 = κ/L         ⟹ κ_min = q7_req·L
       - two_level：     q1 = 1/(p(1−p))  ⟹ p ∈ ]0, 0.5−√(0.25−1/q1)]（双解区间）
     反解式的**依据**：第 88 条 `q_definitions` 登记册 + DB2 族表。

**红线**：本模块只做**解码与判定**，不产生新数据；输出可用于演示与验证，
  产物入库仍走 DB5 的既有校验与 provenance 规则。

用法：
    from pipeline.design_contract import state_to_design, param_bound_for
    spec = state_to_design(env, 332)
    pb = param_bound_for("q1", 3.0, "acoustics_rayleigh")
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

try:                                   # 包内导入（正常路径）
    from pipeline.family_aliases import to_db1
except ImportError:                    # 脚本直跑（sys.path 已在根目录）
    from family_aliases import to_db1  # type: ignore

__all__ = ["DesignSpec", "state_to_design", "param_bound_for",
           "check_spec_against_bound", "INVERTIBLE", "expand_nondegenerate"]

KAPPA_REF = 1e-4       # materials 族参考扩散率
L_DIFF = 0.01          # 特征长度（与 req_env 一致）


@dataclass
class DesignSpec:
    """统一解码契约：一个设计状态的规范结构。"""
    state: int
    req_idx: int
    requirement: str
    family: str
    param_name: str
    param_value: float
    metric_name: str
    metric_value: float
    reward: float
    feasible: bool            # 该族是否支持该需求的度量
    satisfies_req: bool       # 是否达标（reward ≈ 1）
    notes: list[str] = field(default_factory=list)
    # **非退化设计参数（2026-10-02，第 130 条）**：当该族存在**非退化设计域**
    #   时（设计 dof > 状态 dof），这里存放**真实物理设计参数**（如
    #   two_level 的 `{dF_eV, T_K}`）与解流形规模；退化族保持为空 {}。
    design_params: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"state": self.state, "req_idx": self.req_idx,
                "requirement": self.requirement, "family": self.family,
                "param": {self.param_name: self.param_value},
                "metric": {self.metric_name: self.metric_value},
                "reward": self.reward, "feasible": self.feasible,
                "satisfies_req": self.satisfies_req, "notes": self.notes,
                "design_params": self.design_params}


def state_to_design(env, state: int) -> DesignSpec:
    """断点①的解法：RL 离散状态 → 规范设计结构（含参数与度量）。"""
    req_idx, fam, b = env._decode(int(state))
    req = env.requirements[req_idx]
    fam_name = env.family_names[fam]
    pname = env.param_names[fam]
    pval = float(env.grids[fam][b])
    v = env.metric_value(int(state))
    r = env.reward(int(state))
    notes: list[str] = []
    feasible = v is not None
    if not feasible:
        notes.append("族 %s 不支持度量 %s（跨族不可比）" % (fam_name, req.metric))
    return DesignSpec(state=int(state), req_idx=req_idx, requirement=req.label,
                      family=fam_name, param_name=pname, param_value=pval,
                      metric_name=req.metric,
                      metric_value=(float(v) if feasible else float("nan")),
                      reward=float(r), feasible=feasible,
                      satisfies_req=bool(r >= 0.999), notes=notes)


# ---------------------------------------------------------------------------
# 需求反解表：metric → {family: (参数名, 反解函数, 界类型)}
#   反解函数返回「满足 q ≥ q_req 的参数可行域」的界（下界或上界）
# ---------------------------------------------------------------------------
def _rayleigh_sigma_max(q_req: float) -> float:
    return math.sqrt(4.0 / q_req)


def _ou_theta_min(q_req: float) -> float:
    return q_req


def _diff_kappa_min(q_req: float) -> float:
    return q_req * L_DIFF


def _two_level_p_band(q_req: float) -> tuple[float, float]:
    """q1 = 1/(p(1−p)) ≥ q_req ⟹ p ≤ 0.5−√(0.25−1/q_req) 或 p ≥ 0.5+√(...)"""
    if q_req < 4.0:
        return (float("nan"), float("nan"))     # 该族 q1 最小值为 4
    d = math.sqrt(0.25 - 1.0 / q_req)
    return (0.5 - d, 0.5 + d)


INVERTIBLE: dict[str, dict[str, tuple]] = {
    "q1": {
        "acoustics_rayleigh": ("sigma", _rayleigh_sigma_max, "upper"),
        "two_level": ("p", _two_level_p_band, "exclude_band"),
    },
    "q9": {
        "control_ou": ("theta", _ou_theta_min, "lower"),
    },
    "q7": {
        "materials_diffusion": ("kappa", _diff_kappa_min, "lower"),
    },
}


def param_bound_for(metric: str, q_req: float, family: str):
    """断点②的解法：由需求指标反解出该族的参数可行界。

    优先级（2026-10-02，第 129 条）：
      ① **手写解析桥接式**（`INVERTIBLE`）——已有独立校验
         （`rayleigh` 的 `q1=4/σ²` 与数值 Fisher 反解相对差 **0.07%**）；
      ② **数值 Fisher 反解**（`pipeline/inverse_fisher.py`，覆盖 **21 族**）
         ——**仅对 `q1`（Fisher 型度量）启用**：数值反解的对象是解析
         `F_trace(θ)`，它就是 q1；对 `q9`/`q7`（速率/输运量，**不是 Fisher**）
         **不得套用**，否则是"口径错配"（本会话已犯过一次：I3b 初版）；
      ③ 否则 **如实返回 None**（不猜）。

    返回 (param_name, bound_kind, value) 或 None。
    """
    fam_map = INVERTIBLE.get(metric, {})
    db1_fam = to_db1(family)          # **族名别名集中化**（第 129 条）
    key = family if family in fam_map else (
        db1_fam if db1_fam in fam_map else None)
    if key is not None:
        pname, fn, kind = fam_map[key]
        return pname, kind, fn(float(q_req))

    if metric != "q1":
        return None                   # 非 Fisher 度量：无数值反解可用（如实）

    try:
        from pipeline.inverse_fisher import invert
        r = invert(db1_fam, float(q_req))
    except Exception:  # noqa: BLE001
        return None
    if not r or r.get("kind") is None or r.get("value") is None:
        return None                   # 非单调 / 超上界 ⟹ 无界可给（如实）
    return r["param"], r["kind"], float(r["value"])


def expand_nondegenerate(spec: DesignSpec, q_req: float | None = None,
                         dF_min_eV: float = 0.15, T_min_K: float = 250.0,
                         T_max_K: float = 1200.0, D_max_m: float = 1.0,
                         f_min_Hz: float = 10e3, f_max_Hz: float = 2e6
                         ) -> dict | None:
    """**把（可能退化的）设计扩展为真实物理设计参数**（第 130/131 条）。

    **两个非退化域（机理同构：信息量定比值、绝对需求定标度）**：
      · `two_level`：`p` 定住 `x = ΔF/(k_B T)` ⟹ 设计 **`(ΔF, T)` 2 dof**
        ⟹ 解为 **T 上的射线**；
      · `rayleigh`：`σ` 定住 `λ/D` ⟹ 设计 **`(D, f)` 2 dof**
        ⟹ 解为 **f 上的射线**（`D = λ/(σ·θ_ref)`，取最小孔径支）。
    返回 `{"manifold":…, "representative":{…}, "n_designs":…}`；
    **无该族非退化模型时返回 None**（不猜、不假装）。
    """
    fam_db1 = to_db1(spec.family)

    if fam_db1 == "two_level":
        try:
            from pipeline.design_two_level import EV, design_manifold
        except ImportError:            # 脚本直跑
            from design_two_level import EV, design_manifold  # type: ignore
        m = design_manifold(float(spec.param_value), dF_min_J=dF_min_eV * EV,
                            T_min_K=T_min_K, T_max_K=T_max_K)
        if not m.get("feasible") or len(m["T"]) == 0:
            return {"manifold": m, "representative": None, "n_designs": 0,
                    "infeasible_reason": m.get("note", "")}
        i = len(m["T"]) // 2
        return {"manifold": m,
                "representative": {"dF_eV": float(m["dF_J"][i]) / EV,
                                   "T_K": float(m["T"][i])},
                "n_designs": int(len(m["T"])),
                "dF_range_eV": [float(m["dF_J"][0]) / EV,
                                float(m["dF_J"][-1]) / EV],
                "T_range_K": [float(m["T"][0]), float(m["T"][-1])]}

    if fam_db1 == "rayleigh":
        try:
            from pipeline.design_rayleigh import design_manifold as dm_r
        except ImportError:            # 脚本直跑
            from design_rayleigh import design_manifold as dm_r  # type: ignore
        # 目标 q1：优先用**需求**；否则用该状态已达成的 q1（≥ 需求 ⟹ 更严）
        q = float(q_req) if q_req is not None else float(spec.metric_value)
        m = dm_r(q, D_max_m=D_max_m, f_min_Hz=f_min_Hz, f_max_Hz=f_max_Hz)
        if not m.get("feasible") or len(m["f"]) == 0:
            return {"manifold": m, "representative": None, "n_designs": 0,
                    "infeasible_reason": m.get("note", "")}
        i = len(m["f"]) // 2
        return {"manifold": m,
                "representative": {"D_m": float(m["D"][i]),
                                   "f_Hz": float(m["f"][i])},
                "n_designs": int(len(m["f"])),
                "D_range_m": [float(m["D"][0]), float(m["D"][-1])],
                "f_range_Hz": [float(m["f"][0]), float(m["f"][-1])]}

    return None                        # 无该族非退化模型 ⟹ 如实返回 None


def check_spec_against_bound(spec: DesignSpec, metric: str, q_req: float):
    """用反解界判定设计是否满足需求（与奖励口径互为独立校验）。

    返回 (verdict, detail)：verdict ∈ {"pass", "fail", "no_inverse"}。
    """
    pb = param_bound_for(metric, q_req, spec.family)
    if pb is None:
        return "no_inverse", "族 %s 无登记反解（metric=%s）" % (spec.family,
                                                              metric)
    pname, kind, val = pb
    if pname != spec.param_name:
        return "no_inverse", "参数名不匹配（契约 %s vs 反解 %s）" % (
            spec.param_name, pname)
    x = spec.param_value
    if kind == "upper":
        ok = x <= val
        return ("pass" if ok else "fail"), "%s=%.4g vs 界 ≤ %.4g" % (
            pname, x, val)
    if kind == "lower":
        ok = x >= val
        return ("pass" if ok else "fail"), "%s=%.4g vs 界 ≥ %.4g" % (
            pname, x, val)
    if kind == "exclude_band":
        lo, hi = val
        ok = (x <= lo) or (x >= hi)
        return ("pass" if ok else "fail"), "%s=%.4g vs 禁带 (%.4g, %.4g)" % (
            pname, x, lo, hi)
    return "no_inverse", "未知界类型 %s" % kind
