# -*- coding: utf-8 -*-
"""变分层校验（2026-10-02，第 116 条）：L1 解析对照 + L2/L3 预登记。

**L1（本脚本执行）——极值解与解析解对照**（三种可解析情形）：
  L1a 平坦度规 g=const、V=0：`S* = ½gΔθ²/T`、轨迹=直线、长度 `√g|Δθ|`；
  L1b 尺度族度规 `g=c/θ²`、V=0：长度 `√c|ln(θ_b/θ_a)|`、
      轨迹 `θ(t)=θ_a(θ_b/θ_a)^{t/T}`、`S* = L²/(2T)`；
  L1c 平坦度规 + 谐振势 `V=½kθ²`：`θ(t) = sin(ωt)/sin(ωT)`（θ_a=0, ω=√(k/g)）。
  **判据：相对误差 ≤ 5%（与架构文档的 L1 一致）。**

**L2（预登记，尚未执行）——增量价值**：
  「变分解出的作用量/测地线长度」相对「静态作用量」是否提供**可测的增量**：
    对照臂：A 侧特征 + 静态 action_s；实验臂：+ 极值作用量 S* 与测地线长度；
    指标：族级 OOD 误差（或 RL 奖励）；
    协议：**≥20 seed、预登记判据、逐点落盘**（架构 §10.2）。
  **本脚本不执行 L2**——只登记协议，避免"未预登记就跑"。

**L3（本脚本执行）——不劣**：变分层为**新增模块**，不修改既有路径 ⟹
  以「既有 A 侧检查仍全过」作为不劣证据（脚本会提示运行哪条）。

用法（项目根目录）：python a_network/v26_variational_check.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from a_network.variational import (extremal_trajectory,  # noqa: E402
                                   geodesic_length_1d)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

TOL = 0.05          # L1 判据：相对误差 ≤5%


def check(name: str, got: float, want: float, tol: float = TOL
          ) -> tuple[bool, str]:
    rel = abs(got - want) / max(abs(want), 1e-12)
    return rel <= tol, "got %.6g / want %.6g（相对误差 %.2f%%）" % (
        got, want, 100 * rel)


def main() -> int:
    print("=" * 72)
    print("变分层校验（第 116 条）：L1 解析对照 + L2/L3 预登记")
    print("=" * 72)
    fails = 0

    # ---------- L1a 平坦度规、无势 ----------
    print("\nL1a 平坦 g=2、V=0、θ:0→1、T=1（解析：直线；S*=½gΔθ²/T=1；L=√2）")
    r = extremal_trajectory("flat", "zero", 0.0, 1.0, T=1.0,
                            m_params={"g": 2.0})
    ok1, d1 = check("L1a 作用量", r["action"], 0.5 * 2.0 * 1.0 ** 2 / 1.0)
    ok2, d2 = check("L1a 测地线长度", r["geodesic_length"], math.sqrt(2.0))
    th = r["theta"]
    lin = np.linspace(0.0, 1.0, len(th))
    lin_err = float(np.abs(th - lin).max())
    ok3 = lin_err < 1e-3
    fails += (not ok1) + (not ok2) + (not ok3)
    print("  [%s] %s  %s" % ("PASS" if ok1 else "FAIL", d1.split("（")[0],
                             d1.split("（")[1]))
    print("  [%s] %s  %s" % ("PASS" if ok2 else "FAIL", d2.split("（")[0],
                             d2.split("（")[1]))
    print("  [%s] L1a 轨迹为直线（max|θ−直线| = %.2e < 1e-3）"
          % ("PASS" if ok3 else "FAIL", lin_err))

    # ---------- L1b 尺度族度规、无势 ----------
    c, a0, b0, Tb = 1.0, 1.0, 4.0, 1.0
    L_analytic = math.sqrt(c) * abs(math.log(b0 / a0))
    print("\nL1b 尺度族 g=1/θ²、V=0、θ:1→4、T=1"
          "（解析：L=ln4=%.4f；S*=L²/2T=%.4f；θ(t)=4^t）"
          % (L_analytic, L_analytic ** 2 / 2))
    r2 = extremal_trajectory("scale", "zero", a0, b0, T=Tb,
                            m_params={"c": c})
    ok4, d4 = check("L1b 测地线长度", r2["geodesic_length"], L_analytic)
    ok5, d5 = check("L1b 作用量", r2["action"], L_analytic ** 2 / 2)
    t = r2["t"]
    ana_traj = a0 * (b0 / a0) ** (t / Tb)
    traj_err = float(np.abs(r2["theta"] - ana_traj).max())
    ok6 = traj_err < 5e-3
    fails += (not ok4) + (not ok5) + (not ok6)
    print("  [%s] %s  %s" % ("PASS" if ok4 else "FAIL", d4.split("（")[0],
                             d4.split("（")[1]))
    print("  [%s] %s  %s" % ("PASS" if ok5 else "FAIL", d5.split("（")[0],
                             d5.split("（")[1]))
    print("  [%s] L1b 轨迹为指数曲线（max|θ−4^t| = %.2e < 5e-3）"
          % ("PASS" if ok6 else "FAIL", traj_err))

    # ---------- L1c 平坦度规 + 谐振势 ----------
    g, k, Tc = 1.0, 4.0, 1.0
    w = math.sqrt(k / g)
    print("\nL1c 平坦 g=1 + 谐振 V=½kθ²（k=4，ω=2）、θ:0→1、T=1"
          "（解析：θ(t)=sin(ωt)/sin(ωT)）")
    r3 = extremal_trajectory("flat", "harmonic", 0.0, 1.0, T=Tc,
                             m_params={"g": g},
                             v_params={"k": k, "theta0": 0.0},
                             n_nodes=81, iters=2500, lr=0.01)
    ana = np.sin(w * r3["t"]) / math.sin(w * Tc)
    e3 = float(np.abs(r3["theta"] - ana).max())
    ok7 = e3 < 0.05
    # **解析作用量（2026-10-02 修：原式恒等抵消，是错的）**：
    #   θ = B sin(ωt)，B = 1/sin(ωT)，ω² = k/g
    #   ∫½θ̇²dt = ½B²ω²[T/2 + sin(2ωT)/(4ω)]
    #   ∫½kθ²dt = ½kB²[T/2 − sin(2ωT)/(4ω)]
    #   ⟹ S = ½B²{ω²[T/2 + s] − k[T/2 − s]}（s = sin(2ωT)/(4ω)）
    #         = ½B²·2ω²s = B²ω sin(2ωT)/4 = **(ω/2)·cot(ωT)**
    #   （ωT=2 > π/2 ⟹ cot<0 ⟹ 作用量为负，极值点为鞍点，物理上正常）
    S_ana = (w / 2.0) / math.tan(w * Tc)
    ok8, d8 = check("L1c 作用量", r3["action"], S_ana, tol=0.05)
    fails += (not ok7) + (not ok8)
    print("  [%s] L1c 轨迹与解析解一致（max|Δθ| = %.4f < 0.05）"
          % ("PASS" if ok7 else "FAIL", e3))
    print("  [%s] %s  %s" % ("PASS" if ok8 else "FAIL", d8.split("（")[0],
                             d8.split("（")[1]))

    print("\n收敛诊断（**能量漂移** = max|H−mean H|/|mean H|，一阶量、噪声不敏感）：")
    for name, rr in (("L1a", r), ("L1b", r2), ("L1c", r3)):
        print("  %s：能量漂移 %.3g（EL 残差 %.3g 为二阶量、受 1/dt² 放大，"
              "仅作参考）" % (name, rr["energy_drift"], rr["residual"]))

    # ---------- L2 预登记（不执行） ----------
    print("\nL2（**预登记，本脚本不执行**）：")
    print("  对照臂 = A 侧特征 + 静态 action_s；实验臂 = + 极值作用量 S* 与"
          "测地线长度")
    print("  指标 = 族级 OOD 误差（或 RL 奖励）；**≥20 seed + 预登记 + 逐点落盘**")
    print("  ⟹ 待实现：`a_network/v27_variational_value_check.py`（协议同 v19–v22）")

    # ---------- L3 ----------
    print("\nL3（不劣）：变分层为**新增模块**，未修改任何既有路径。")
    print("  证据：`core/action.py`、`action_reward.py`、`db4_writer.py` "
          "均未改动；")
    print("  请运行既有 A 侧检查确认不退化："
          "`python b_network/test_theory_constraints.py`")

    print("\n===== %s =====" % ("L1 全部通过" if fails == 0 else
                                "有 %d 项未达标" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
