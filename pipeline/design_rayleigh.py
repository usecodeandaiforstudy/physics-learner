# -*- coding: utf-8 -*-
"""中游 A：**第 2 个非退化设计域——声学阵列分辨力（rayleigh 族）**
（2026-10-02，第 131 条）。

**与 `two_level` 域同构的构造**：
    物理：角分辨力 `σ = (λ/D)/θ_ref`，其中 `λ = v/f`（波长）、`D`（孔径）、
          `θ_ref` 为**登记参考标度**（见下）
    ⟹ `σ` 只定住**比值** `λ/D` ⟹ **1 个约束、2 个自由度** `(D, f)`
    ⟹ 同一信息状态对应**一维解流形**（非退化）✓
    信息侧：`q1 = 4/σ²`（既有桥接式，与数值 Fisher 反解相对差 **0.07%**）

**为什么用"参考标度"而非绝对弧度**：DB3 的 `rayleigh` 需求（DES-RAY-01：
`q1 ≥ 3`）解出 `σ ≤ 1.1547`——**无量纲**。若直接当作弧度则物理上荒谬
（1.15 rad ≈ 66°）。故引入**登记参考标度** `θ_ref`（默认 **1 mrad**），
与第 108 条「有量纲量须除以文档化参考尺度」的机制**同源** ✓
⟹ `σ = 1.1547` 对应 `λ/D = 1.1547 mrad ≈ 0.066°` ✓ 物理合理。

**第二类需求（定标度）**：绝对约束——`D ≤ D_max`（尺寸上限）、
`f ∈ [f_min, f_max]`（频段/穿透要求）⟹ 只有绝对量能定住标度 ✓

**闭环校验（经既有编码器）**：设计 `(D, f)` → `σ` → `SystemEncoder`
（rayleigh 族）→ `q1`/DPI ⟹ 与 `4/σ²` 及需求对照。

用法：
    python pipeline/design_rayleigh.py          # 非退化性 + 需求互补 + 闭环
"""

from __future__ import annotations

import math
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

THETA_REF = 1e-3        # 参考角标度：1 mrad（**登记约定**）
V_SOUND = 1500.0        # 水中声速 m/s（登记约定；空气中 343 m/s）


def sigma_from_design(D_m: float, f_Hz: float, v: float = V_SOUND,
                      theta_ref: float = THETA_REF) -> float:
    """`σ = (λ/D)/θ_ref`，`λ = v/f`（无量纲 σ，单位 = θ_ref）。"""
    if D_m <= 0 or f_Hz <= 0:
        raise ValueError("D、f 须 > 0")
    lam = v / f_Hz
    return (lam / D_m) / theta_ref


def design_from_sigma(sigma: float, f_Hz: float, v: float = V_SOUND,
                      theta_ref: float = THETA_REF) -> float:
    """给定 σ 与频率，反解**最小孔径** `D = λ/(σ·θ_ref)`。"""
    lam = v / f_Hz
    return lam / (sigma * theta_ref)


def q1_from_sigma(sigma: float) -> float:
    """既有桥接式：`q1 = 4/σ²`（与数值 Fisher 反解相对差 0.07%）。"""
    return 4.0 / (sigma ** 2)


def design_manifold(q1_req: float, D_max_m: float = 1.0,
                    f_min_Hz: float = 10e3, f_max_Hz: float = 500e3,
                    n: int = 200, minimal: bool = True) -> dict:
    """解流形：`{ (D, f) : q1(D,f) ≥ q1_req ∧ D ≤ D_max ∧ f ∈ [f_min, f_max] }`。

    `minimal=True` 时取**最小孔径支**（`D = λ/(σ_max·θ_ref)`）⟹ **一维流形**
    （f 参数化）；否则解集是二维区域。
    可行性条件：`f ≥ v/(σ_max·θ_ref·D_max)`（否则即使最大孔径也不达标）。
    """
    if q1_req <= 0:
        raise ValueError("q1_req 须 > 0")
    sigma_max = 2.0 / math.sqrt(q1_req)          # q1 = 4/σ² ≥ q1_req
    f_need = V_SOUND / (sigma_max * THETA_REF * D_max_m)   # 达标所需的最低频
    f_lo = max(f_min_Hz, f_need)
    feas = f_lo <= f_max_Hz
    if not feas:
        return {"sigma_max": sigma_max, "feasible": False, "f_min_eff": f_lo,
                "f": np.array([]), "D": np.array([]), "q1": np.array([]),
                "note": ("不可行：σ_max=%.4f ⟹ 需要 f ≥ %.0f Hz（由 D_max=%.3g m "
                         "决定），超出频段上界 %.0f Hz ⟹ 要么放宽 D_max 到 "
                         "≥%.3g m，要么放宽 q1_req"
                         % (sigma_max, f_need, D_max_m, f_max_Hz,
                            V_SOUND / (sigma_max * THETA_REF * f_max_Hz)))}
    fs = np.linspace(f_lo, f_max_Hz, n)
    Ds = np.array([design_from_sigma(sigma_max, float(f)) for f in fs])
    q1s = np.array([q1_from_sigma(sigma_from_design(float(d), float(f)))
                    for d, f in zip(Ds, fs)])
    return {"sigma_max": sigma_max, "feasible": True, "f_min_eff": float(f_lo),
            "f": fs, "D": Ds, "q1": q1s,
            "note": ("f ∈ [%.0f, %.0f] Hz（最小孔径支）；D = λ/(σ_max·θ_ref)"
                     % (f_lo, f_max_Hz))}


def verify_design(D_m: float, f_Hz: float, q1_req: float | None = None,
                  tol: float = 1e-6) -> dict:
    """**经既有 B 编码器校验**：设计 → σ → `SystemEncoder`(rayleigh) → q1/DPI。

    **边界容差（2026-10-02 修）**：最小孔径支**按构造落在可行域边界上**
    （`σ = σ_max` ⟹ `q1 = q1_req` 恰好相等），浮点下会得到
    `2.9999999996 < 3.0` ⟹ 判"不达标"。故达标判据用**相对容差** `tol`
    （与项目既有 `DPI_A1_TOL` 等同类做法一致）。
    """
    from a_network.analytic_backbone import compute_backbone
    from b_network.system_encoder import SystemEncoder
    from pipeline.inverse_fisher import synthetic_entry
    sigma = sigma_from_design(D_m, f_Hz)
    e = synthetic_entry("rayleigh", "sigma", float(sigma))
    g = SystemEncoder().encode(e)
    bb = compute_backbone(g)
    q1 = None
    for v in bb["node_fisher"].values():
        if v.get("covered") and v.get("fisher_trace"):
            q1 = float(v["fisher_trace"])
            break
    out = {"sigma": sigma, "q1_from_encoder": q1,
           "q1_analytic": q1_from_sigma(sigma), "dpi_ok": bool(bb["dpi"]["ok"])}
    # **族登记范围检查（如实标注，不静默外推）**：设计给出的 σ 若落在
    #   DB1 该族 `sigma` 的登记范围之外，该设计属**范围外推**，须报告。
    try:
        from pipeline.inverse_fisher import family_param
        _pname, (lo, hi) = family_param("rayleigh")
        out["sigma_range"] = [lo, hi]
        out["in_documented_range"] = bool(lo <= sigma <= hi)
    except Exception:  # noqa: BLE001
        out["in_documented_range"] = None
    if q1 is not None:
        out["q1_rel_err"] = abs(q1 - out["q1_analytic"]) / out["q1_analytic"]
    if q1_req is not None:
        out["q1_meets_requirement"] = bool(
            q1 is not None and q1 >= q1_req * (1.0 - tol))
    return out


def main() -> int:
    print("=" * 74)
    print("中游 A：第 2 个非退化设计域——声学阵列分辨力（rayleigh）")
    print("=" * 74)
    q1_req = 3.0                                  # DES-RAY-01 的需求
    print("\n需求（与闭环演示任务一致）：q1 ≥ %.1f ⟹ σ ≤ %.4f（=2/√q1）"
          % (q1_req, 2.0 / math.sqrt(q1_req)))
    print("登记约定：θ_ref = %.0e rad；v_sound = %.0f m/s" % (THETA_REF,
                                                            V_SOUND))
    print("**量纲提示**：σ 的单位是 θ_ref ⟹ σ=1.1547 表示 λ/D = 1.1547 mrad "
          "（≈0.066°）——这是**很细**的分辨力，需要大孔径或高频：")
    print("   D=1.0 m 时需 λ ≤ 1.15 mm ⟹ f ≥ **1.30 MHz**（医用超声量级）；")
    print("   f ≤ 500 kHz 时需 D ≥ **2.60 m**（大孔径阵列）。")

    # ---- R1 非退化性 ----
    # **演示参数取物理相容的一组**：f_max 提到 2 MHz（医用超声量级），
    #   使 D ≤ 1 m 的阵列可满足 q1 ≥ 3。
    m = design_manifold(q1_req, D_max_m=1.0, f_min_Hz=10e3, f_max_Hz=2e6)
    print("\nR1 **非退化性**（同一状态 ⟹ 多少个设计？）：")
    print("  绝对约束：D ≤ 1.0 m；f ∈ [10 kHz, 2 MHz]")
    print("  可行=%s；有效频段下界 f ≥ %.0f Hz（由 D_max 决定）"
          % (m["feasible"], m["f_min_eff"]))
    if m["feasible"]:
        for frac in (0.0, 0.5, 1.0):
            i = int(frac * (len(m["f"]) - 1))
            print("   设计样例：f = %7.0f Hz → D = %.4f m（σ 恒为 %.4f，"
                  "q1 恒为 %.4f）"
                  % (m["f"][i], m["D"][i],
                     sigma_from_design(float(m["D"][i]), float(m["f"][i])),
                     float(m["q1"][i])))
        print("  ⟹ 同一状态对应 **%d 个不同设计** ⟹ **多对一（非退化）** ✓"
              % len(m["f"]))
    else:
        print("  ⟹ **无可行设计**；建设性诊断：%s" % m.get("note", ""))

    # ---- R2 两类需求互补 ----
    print("\nR2 **两类需求互补**（比值 vs 标度）：")
    print("  统计/信息型（q1 → σ）：只定住**比值 λ/D**（1 个约束）")
    print("  绝对型（D_max、频段）：定住**标度** ⟹ 给出 f 下界 %.0f Hz"
          % m["f_min_eff"])
    print("  ⟹ 与 two_level 域**完全同构**（比值定形状、绝对量定标度）✓")

    # ---- R3 闭环校验 ----
    print("\nR3 **闭环校验（经既有 B 编码器）**：")
    if m["feasible"]:
        for frac in (0.0, 0.5, 1.0):
            i = int(frac * (len(m["f"]) - 1))
            v = verify_design(float(m["D"][i]), float(m["f"][i]), q1_req)
            print("  [边界支] f=%7.0f Hz D=%.4f m → σ=%.4f；q1(编码器)=%.4f "
                  "vs 解析 %.4f（相对差 %.1e）；DPI=%s；需求达标=%s；族登记范围 %s"
                  % (m["f"][i], m["D"][i], v["sigma"],
                     v["q1_from_encoder"] or float("nan"), v["q1_analytic"],
                     v.get("q1_rel_err", float("nan")), v["dpi_ok"],
                     v.get("q1_meets_requirement"),
                     ("**范围外推**" if v.get("in_documented_range") is False
                      else "内 ✓")))
        # **严格内部的示例（2026-10-02 加）**：最小孔径支落在可行域**边界**上
        #   （σ 恒等于 σ_max）⟹ 用**更大孔径**得到严格满足需求的内部点。
        j = len(m["f"]) // 2
        D_in = float(m["D"][j]) * 1.2
        if D_in <= 1.0:
            v = verify_design(D_in, float(m["f"][j]), q1_req)
            print("  [内部点] f=%7.0f Hz D=%.4f m（=1.2×最小孔径）→ σ=%.4f；"
                  "q1(编码器)=%.4f vs 解析 %.4f（相对差 %.1e）；DPI=%s；"
                  "需求达标=%s（**q1 严格高于需求** ✓）"
                  % (m["f"][j], D_in, v["sigma"],
                     v["q1_from_encoder"] or float("nan"), v["q1_analytic"],
                     v.get("q1_rel_err", float("nan")), v["dpi_ok"],
                     v.get("q1_meets_requirement")))
        else:
            print("  [内部点] 跳过（1.2×最小孔径 = %.4f m 超出 D_max=1.0 m）"
                  % D_in)

    # ---- R4 两域对照 ----
    print("\nR4 **两个非退化域对照**：")
    print("  %-14s %-20s %-12s %s"
          % ("族", "设计变量", "约束", "解流形"))
    print("  %-14s %-20s %-12s %s"
          % ("two_level", "(ΔF, T)", "ΔF ≥ ΔF_min / T 范围", "T 上的一条射线"))
    print("  %-14s %-20s %-12s %s"
          % ("rayleigh", "(D, f)", "D ≤ D_max / 频段", "f 上的一条射线"))
    print("  ⟹ 两者机理一致：**信息量定比值、绝对需求定标度** ✓")

    print("\n结论：第 2 个非退化设计域建成（%s）"
          % ("可行且经编码器闭环校验" if m["feasible"] else "**不可行**"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
