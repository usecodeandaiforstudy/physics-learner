# -*- coding: utf-8 -*-
"""**波前/输运速度适配器（FrontAdapter）**（2026-10-02，第 158 条）。

## 缺口
`q7`（信息传播速度）此前只有**扩散**机制 `q7 = κ/L`；第 156 条扩展后能力可达
`[1e-17, 1e5] m/s`，仍剩 **1 条**（`DB3-0158` astro，`1.67e5 m/s`）——
该量级属**波前**机制，扩散模型**表达不了**。

## 本模块（两种波前机制）
| 族 | 波速 `v`（解析） | 物理 |
|---|---|---|
| `harmonic_oscillator` | `v = ω·a` | 一维晶格**声子群速度**（`ω=√(k/m)` 为振子频率，`a` 为晶格常数）；长波极限 `v_g(q→0)=a·√(k/m)` |
| `lc_circuit` | `v = 1/√(L·C)` | **LC 梯形网络**（传输线）信号速度（每节电感 `L`、电容 `C`） |

**几何参数**（可选，登记在 `optional`，缺省注入登记约定值）：
`a`（晶格常数，默认 **5e-10 m**）、`inductance`（每节电感，默认 **1e-9 H**）。
⟹ 可达范围：`v = ω·a ∈ [0.1×5e-10, 5×5e-10]`… 加上几何可放大到 `1e6 m/s` 量级；
`1/√(LC)` 可覆盖 **`1e4 … 1e9 m/s`** ✓（含 astro 的 `1.67e5`）。

## 数值估计（真值自检用；与解析式对照）
模拟**一维晶格/LC 梯形**上的脉冲传播，测**前缘到达时间** `t`（阈值：振幅达
峰值的 5%）⟹ `v̂ = d/t`（`d` = 传播距离）。离散色散使脉冲会展宽，
故自检判据取 **≤35%**（**如实标注这是色散导致**，不是"精确一致"）。

用法：
    python sim/front_adapter.py          # 真值自检
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

__all__ = ["FrontAdapter", "FRONT_MODELS", "A_DEFAULT", "L_IND_DEFAULT"]

A_DEFAULT = 5e-10          # 晶格常数默认（m）—— 登记约定
L_IND_DEFAULT = 1e-9       # 每节电感默认（H）—— 登记约定


class FrontAdapter:
    """波前速度适配器：**数值估计** `v̂`（真值为解析波速）。"""

    name: str = "front"

    def run(self, spec: dict, n_samples: int = 20000, seed: int = 0,
            **kw) -> dict:
        fam = spec.get("family") or spec.get("kind")
        if fam == "harmonic_oscillator":
            omega = float(spec["omega"])
            a = float(spec.get("a", A_DEFAULT))
            v_true = omega * a
            v_hat = self._lattice_front(omega, a, seed, n=int(min(
                max(n_samples // 100, 200), 2000)))
        elif fam == "lc_circuit":
            C = float(spec["capacitance"])
            L = float(spec.get("inductance", L_IND_DEFAULT))
            v_true = 1.0 / math.sqrt(max(L * C, 1e-30))
            v_hat = self._ladder_front(L, C, seed, n=int(min(
                max(n_samples // 100, 200), 2000)))
        else:
            raise KeyError("未支持的波前族：%s" % fam)
        return {"obs": None, "family": fam, "v_true": v_true,
                "v_hat": v_hat, "q7_info_propagation_speed": v_hat}

    # ---------------------------------------------------------------- 机制
    @staticmethod
    def _front_speed(hist: list, t_hist: list, d1: float, d2: float,
                     i1: int, i2: int) -> float:
        """**两点测速（2026-10-02 修）**：`v̂ = (d2−d1)/(t2−t1)`，阈值取
        **每个测点自身最终峰值的 5%**（两遍判定）。

        初版用"运行最大值"的单点阈值 ⟹ 早期**数值噪声**即误触发
        ⟹ 四项自检恒定偏大 **3.33×**（恒定因子 = 实现/判据错误，非物理）。
        """
        a1 = np.abs(np.asarray(hist)[:, i1])
        a2 = np.abs(np.asarray(hist)[:, i2])
        t = np.asarray(t_hist)
        m1, m2 = float(a1.max()), float(a2.max())
        if m1 <= 0 or m2 <= 0:
            return float("nan")
        k1 = int(np.argmax(a1 > 0.05 * m1))
        k2 = int(np.argmax(a2 > 0.05 * m2))
        if k2 <= k1:
            return float("nan")
        dt = float(t[k2] - t[k1])
        if dt <= 0:
            return float("nan")
        return float((d2 - d1) / dt)

    @staticmethod
    def _lattice_front(omega: float, a: float, seed: int,
                       n: int = 800) -> float:
        """一维晶格脉冲传播 ⟹ **前缘速度** `v̂`（两点测速）。

        链：`N` 个质量 `m=1`、最近邻弹簧 `k=ω²`（`ω=√(k/m)`）；两端固定。
        测点：节点 100 与 300（距离 `Δd = 200·a`）。
        """
        N = 400
        k = omega ** 2
        dt = 0.02 / max(omega, 1e-9)
        u = np.zeros(N)
        v = np.zeros(N)
        src = 10
        i1, i2 = 100, 300
        hist, thist = [], []
        steps = int(min(n * 400, 400000))
        for step in range(steps):
            if step < src:
                u[1] = math.sin(math.pi * step / src) * 1e-3
            acc = k * (np.r_[0.0, u[:-1]] + np.r_[u[1:], 0.0] - 2 * u)
            v += acc * dt
            u += v * dt
            hist.append((u[i1], u[i2]))
            thist.append(step * dt)
        return FrontAdapter._front_speed(hist, thist, (i1 - 1) * a,
                                         (i2 - 1) * a, 0, 1)

    @staticmethod
    def _ladder_front(L: float, C: float, seed: int, n: int = 800) -> float:
        """LC 梯形网络脉冲传播 ⟹ **前缘速度** `v̂`（两点测速；每节 1 m）。"""
        N = 400
        dt = 0.02 * math.sqrt(max(L * C, 1e-30))
        V = np.zeros(N)
        I = np.zeros(N)
        src = 10
        i1, i2 = 100, 300
        hist, thist = [], []
        steps = int(min(n * 400, 400000))
        for step in range(steps):
            if step < src:
                V[0] = math.sin(math.pi * step / src) * 1e-3
            V[1:] += (-np.diff(I) / max(C, 1e-30)) * dt
            I[:-1] += (-np.diff(V[:N]) / max(L, 1e-30)) * dt
            hist.append((V[i1], V[i2]))
            thist.append(step * dt)
        return FrontAdapter._front_speed(hist, thist, float(i1), float(i2),
                                         0, 1)


def main() -> int:
    """**真值自检**：数值 `v̂` vs 解析波速 `v`（判据 ≤35%，色散会展宽脉冲）。"""
    print("=" * 74)
    print("波前/输运速度适配器（FrontAdapter）真值自检")
    print("=" * 74)
    print("机制：① `harmonic_oscillator`：`v = ω·a`（声子群速度）；"
          "② `lc_circuit`：`v = 1/√(L·C)`（传输线）")
    print("几何默认（登记约定）：`a = %.1e m`、`inductance = %.1e H`"
          % (A_DEFAULT, L_IND_DEFAULT))
    cases = [
        ({"family": "harmonic_oscillator", "omega": 2.0}, "ho ω=2"),
        ({"family": "harmonic_oscillator", "omega": 5.0}, "ho ω=5"),
        ({"family": "lc_circuit", "capacitance": 1.0,
          "inductance": 1e-9}, "lc C=1, L=1nH"),
        ({"family": "lc_circuit", "capacitance": 0.1,
          "inductance": 1e-12}, "lc C=0.1, L=1pH"),
    ]
    fails = 0
    print("\n%-16s %14s %14s %9s %s" % ("情形", "解析 v (m/s)", "数值 v̂ (m/s)",
                                        "相对差", "判定"))
    for spec, tag in cases:
        est = []
        for s in range(3):
            out = FrontAdapter().run(spec, n_samples=200000, seed=s)
            est.append(out["v_hat"])
        v_true = FrontAdapter().run(spec, n_samples=1000, seed=0)["v_true"]
        e = float(np.nanmedian(est))
        rel = abs(e - v_true) / max(abs(v_true), 1e-12)
        ok = rel <= 0.35
        fails += int(not ok)
        print("%-16s %14.5g %14.5g %8.1f%% %s"
              % (tag, v_true, e, 100 * rel, "✓" if ok else "**✗ 超 35%**"))
    print("\n判据说明：离散色散使脉冲**展宽**（前缘速度 ≠ 相速度），"
          "故阈值取 **35%**——**如实标注，不假装精确一致**。")
    print("%s" % ("===== 全部通过 =====" if fails == 0
                  else "===== 有 %d 项失败 =====" % fails))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
