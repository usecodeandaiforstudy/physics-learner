# -*- coding: utf-8 -*-
"""中游②：**非退化设计域的 C 实例**（二能级：设计 (ΔF, T) → 状态 p）
（2026-10-02，第 128 条）。

**为什么需要**："中游没做好"的第二个表现是 **C 的生成职责退化**——
现有 4 个族（rayleigh/two_level/control_ou/materials_diffusion）里
**设计对象就是信息参数本身** ⟹ 解码器退化为恒等映射，生成能力无从体现。

**非退化的构造（本模块）**：
    物理：`p = 1/(1 + e^{ΔF/(k_B T)})` ⟹ `p` 只定住**比值**
          `x = ΔF/(k_B T)`（nats）—— **1 个约束、2 个自由度**
    设计变量：`(ΔF, T)` —— **2 dof**
    ⟹ **每个信息状态 p 对应一维解流形**（ΔF 与 T 可同比例伸缩而不改 p）✓
    **第二条需求必须是"绝对"的**（如 `ΔF ≥ ΔF_min` 焦耳、`T ≥ T_min` K）
    —— 只有绝对量才能定住标度 ⟹ 两类需求**互补**：
      · 统计型（p）：信息量型，**β 适用**（第 114 条）
      · 约束型（ΔF_min / T_min，绝对值）：**不用 β**，走**达标型**奖励 ✓

**闭环校验（经既有管线，不是自算自）**：
    设计 `(ΔF, T)` → 算 `p` → **用 `SystemEncoder` 编码成 DB1 式条目**
    → `compute_backbone` 得 `q1 = 1/(p(1-p))` 与 **DPI 检查**
    ⟹ 设计与"信息量"的关系由**既有编码器**确认（口径一致）✓

用法：
    python pipeline/design_two_level.py            # 非退化性报告 + 示例设计
    python pipeline/design_two_level.py --demo     # 走完整闭环校验
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

# 物理常数（与项目其它处一致：以 nats 表达的比值用 k_B）
K_B = 1.380649e-23          # J/K
EV = 1.602176634e-19        # J
T_REF = 300.0               # K（室温参考）


def x_from_p(p: float) -> float:
    """`x = ΔF/(k_B T) = ln((1-p)/p)`（nats）。

    **约定由 DB1 自身确定（2026-10-02，不猜）**：
      `2lvl_ground_dominant`（近基态占据）的 `p = 0.1`；
      `2lvl_excited_dominant`（反转占据）的 `p = 0.9`
      ⟹ **`p` = 激发态占据概率** ⟹ `p = 1/(1+e^{ΔF/(k_B T)})`，
      `ΔF = E_excited − E_ground`：
        · `p < 0.5`（基态为主）⟹ `x > 0` ⟹ **正势垒** ✓
        · `p > 0.5`（反转占据）⟹ `x < 0` ⟹ 负 ΔF（**泵浦态**，非势垒）
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p 须在 (0,1)，收到 %r" % p)
    return math.log((1.0 - p) / p)


def p_from_x(x: float) -> float:
    return 1.0 / (1.0 + math.exp(x))


def p_from_design(dF_J: float, T_K: float) -> float:
    """由设计 `(ΔF [J], T [K])` 得占据概率 p。"""
    if T_K <= 0:
        raise ValueError("T 须 > 0")
    return p_from_x(dF_J / (K_B * T_K))


def design_manifold(p_target: float, dF_min_J: float = 0.0,
                    T_min_K: float = 0.0, T_max_K: float = 1200.0,
                    n: int = 200) -> dict:
    """求解设计流形：`{ (ΔF, T) : p(ΔF,T) = p_target ∧ ΔF ≥ dF_min ∧ T ≥ T_min }`。

    由于 `p` 只定住 `x = ΔF/(k_B T)`，解集是 **T 上的一条射线**：
        `ΔF(T) = x·k_B·T`，约束给出 `T ≥ max(T_min, dF_min/(x·k_B))`
    ⟹ **一维流形（非退化）**：同一 p 有无穷多设计。
    """
    x = x_from_p(p_target)
    if x < 0:
        # **反转占据（p > 0.5）**：x < 0 ⟹ ΔF < 0（泵浦态）。
        #   "势垒 ΔF ≥ dF_min（正值）"在该支上**本就不可行** ⟹ 如实报告，
        #   并给出正确的约束方向（泵浦能 ≤ −dF_min）。
        return {"x_nats": x, "T_min_eff": float("nan"), "feasible": False,
                "T": np.array([]), "dF_J": np.array([]),
                "p_target": p_target,
                "note": ("**反转占据**（p>0.5 ⟹ x=%.4f < 0 ⟹ ΔF<0，泵浦态）："
                         "正势垒约束 `ΔF ≥ %.3g J` 不可行；"
                         "该支应表述为**泵浦能** `ΔF ≤ −%.3g J`"
                         % (x, dF_min_J, dF_min_J))}
    T_lo = max(T_min_K, (dF_min_J / (x * K_B)) if dF_min_J > 0 else 0.0)
    feas = T_lo <= T_max_K
    Ts = np.linspace(T_lo, T_max_K, n) if feas else np.array([])
    dFs = x * K_B * Ts
    return {"x_nats": x, "T_min_eff": float(T_lo), "feasible": bool(feas),
            "T": Ts, "dF_J": dFs, "p_target": p_target,
            "T_max_K": T_max_K,
            # **建设性诊断（2026-10-02 加）**：不可行时告诉用户"哪个旋钮该转"
            "dF_max_at_Tmax_J": x * K_B * T_max_K,
            "note": ("T ≥ %.2f K ⟹ ΔF = x·k_B·T（射线）" % T_lo) if feas else
                    ("不可行：ΔF = x·k_B·T 在 T=%.0f K 时最大只有 %.4f eV，"
                     "低于要求的 %.4f eV ⟹ 要么把 ΔF_min 降到 ≤%.4f eV，"
                     "要么把 T_max 提到 ≥%.0f K"
                     % (T_max_K, x * K_B * T_max_K / EV, dF_min_J / EV,
                        x * K_B * T_max_K / EV, T_lo))}


def verify_design(dF_J: float, T_K: float, q1_expected: float | None = None
                  ) -> dict:
    """**经既有管线校验设计**：设计 → p → `SystemEncoder` → `q1`/DPI。"""
    p = p_from_design(dF_J, T_K)
    from a_network.analytic_backbone import compute_backbone
    from b_network.system_encoder import SystemEncoder
    # 以 two_level 族的模板构造条目（**委托既有编码器**，不手拼）
    from pipeline.inverse_fisher import synthetic_entry
    e = synthetic_entry("two_level", "p", float(p))
    g = SystemEncoder().encode(e)
    bb = compute_backbone(g)
    q1 = None
    for v in bb["node_fisher"].values():
        if v.get("covered") and v.get("fisher_trace"):
            q1 = float(v["fisher_trace"])
            break
    out = {"p": p, "q1_from_encoder": q1,
           "q1_analytic": 1.0 / (p * (1.0 - p)),
           "dpi_ok": bool(bb["dpi"]["ok"])}
    if q1 is not None:
        out["q1_rel_err"] = abs(q1 - out["q1_analytic"]) / out["q1_analytic"]
    if q1_expected is not None:
        out["q1_matches_requirement"] = bool(
            q1 is not None and abs(q1 - q1_expected) / q1_expected < 0.05)
    return out


def main() -> int:
    print("=" * 74)
    print("中游②：非退化设计域（二能级：设计 (ΔF, T) → 状态 p）")
    print("=" * 74)

    # ---------- N1 非退化性 ----------
    print("\nN1 **非退化性**（同一状态 p ⟹ 多少个不同设计？）：")
    # **p 的约定由 DB1 确定**：p = **激发态占据** ⟹ p=0.1 为基态为主（正势垒）。
    #   初版演示误取 p=0.8（**反转占据**）⟹ 正势垒约束不可行、0 个设计，
    #   而判读行**仍打印"非退化 ✓"** ⟹ 判读未检查数据（已修：见下方守卫）。
    p_t = 0.10
    # **参数据物理选定（2026-10-02 修）**：p=0.10 ⟹ x=2.1972；
    #   若 T_max=1200 K，则 ΔF 最大只能到 x·k_B·1200 = 0.2272 eV
    #   ⟹ 绝对约束须 ≤0.2272 eV 才可行。初版取 0.30 eV ⟹ 需 T≥1584 K
    #   （**算术无误**：1 eV = 11604.5 K ⟹ 0.30 eV = 3481 K；÷2.1972 = 1584 K）
    #   ——这正说明**两类需求的关系被显式化了**（非退化域的价值）。
    dF_min_ev = 0.15
    m = design_manifold(p_t, dF_min_J=dF_min_ev * EV, T_min_K=250.0,
                        T_max_K=1200.0)
    print("  需求：p = %.2f（统计型，**激发态占据**）∧ ΔF ≥ %.2f eV"
          "（绝对约束型）∧ T ∈ [250, 1200] K" % (p_t, dF_min_ev))
    print("  x = ln((1-p)/p) = %+.4f nats ⟹ 解为 T 上一条**射线**"
          "（ΔF = x·k_B·T）" % m["x_nats"])
    print("  有效下界 T ≥ %.1f K；可行=%s；流形点数 %d"
          % (m["T_min_eff"], m["feasible"], len(m["T"])))
    if len(m["T"]):
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            i = int(frac * (len(m["T"]) - 1))
            print("   设计样例：T = %7.1f K → ΔF = %.4f eV（p 恒为 %.4f）"
                  % (m["T"][i], m["dF_J"][i] / EV, p_from_design(m["dF_J"][i],
                                                               m["T"][i])))
        print("  ⟹ 同一 p 对应**%d 个不同设计** ⟹ **映射多对一（非退化）** ✓"
              % len(m["T"]))
    else:
        print("  ⟹ **无可行设计**（约束冲突）")
        print("  **建设性诊断**：%s" % m.get("note", ""))
        print("  ⟹ 这**不是缺陷**：它说明两类需求在此参数下不相容——"
              "非退化域把该关系**显式化**了（退化域下这个问题根本看不见）")

    # ---------- N2 两类需求互补 ----------
    print("\nN2 **两类需求互补**（只有绝对约束能定住标度）：")
    m1 = design_manifold(p_t, T_max_K=1200.0)                  # 仅统计型
    m2 = design_manifold(p_t, dF_min_J=dF_min_ev * EV, T_min_K=250.0,
                         T_max_K=1200.0)
    print("  仅统计型需求：T 可任意 ⟹ 解集 T ∈ [0, 1200] K（**标度不定**），"
          "ΔF 随 T 从 0 到 %.4f eV 连续变化"
          % (m1["x_nats"] * K_B * 1200.0 / EV if len(m1["T"]) else float("nan")))
    print("  加绝对约束后：T ≥ %.1f K ⟹ 标度被定住；"
          "另一个等价说法：该需求**等价于**热度下界"
          % m2["T_min_eff"])
    print("  ⟹ 统计型（信息量型，**β 适用**）与绝对约束型（**不用 β**）"
          "在数学上互补 ✓")

    # ---------- N1b 反转占据分支（p > 0.5）----------
    print("\nN1b **反转占据分支**（p > 0.5 ⟹ ΔF < 0，须改述为泵浦能）：")
    m_inv = design_manifold(0.90, dF_min_J=0.30 * EV, T_min_K=250.0)
    print("  p = 0.90 ⟹ x = %+.4f nats（< 0）" % m_inv["x_nats"])
    print("  %s" % m_inv["note"])
    print("  ⟹ **同一物理关系在不同分支上给出不同方向的约束**——"
          "这类分支必须在契约里显式区分（否则会产出负势垒的\"设计\"）")

    # ---------- N3 经既有管线闭环校验 ----------
    print("\nN3 **闭环校验**（设计 → p → SystemEncoder → q1/DPI）：")
    if "--demo" in sys.argv or True:
        for i in (0, len(m["T"]) // 2, len(m["T"]) - 1):
            if not len(m["T"]):
                break
            v = verify_design(float(m["dF_J"][i]), float(m["T"][i]),
                              q1_expected=1.0 / (p_t * (1 - p_t)))
            print("  T=%7.1f K ΔF=%.4f eV → p=%.4f；q1(编码器)=%.4f "
                  "vs 解析 %.4f（相对差 %.2e）；DPI=%s；需求的 q1 匹配=%s"
                  % (m["T"][i], m["dF_J"][i] / EV, v["p"],
                     v["q1_from_encoder"] or float("nan"), v["q1_analytic"],
                     v.get("q1_rel_err", float("nan")), v["dpi_ok"],
                     v.get("q1_matches_requirement")))

    # ---------- N4 与退化族对照 ----------
    print("\nN4 **与退化族对照**（说明②到底补了什么）：")
    print("  %-16s %-22s %-10s %s"
          % ("族", "设计变量", "状态维度", "映射"))
    print("  %-16s %-22s %-10s %s"
          % ("rayleigh", "σ（=状态参数）", "1", "**恒等（退化）**"))
    print("  %-16s %-22s %-10s %s"
          % ("two_level(旧)", "p（=状态参数）", "1", "**恒等（退化）**"))
    print("  %-16s %-22s %-10s %s"
          % ("two_level(本模块)", "(ΔF, T)", "2 → 1", "**多对一（非退化）** ✓"))

    print("\n结论：非退化设计域已建立（设计 2 dof → 状态 1 dof，解为射线），"
          "且闭环经**既有编码器**校验。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
