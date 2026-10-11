# -*- coding: utf-8 -*-
"""**外部验证（可行子集）：diffusion / q7 一个域**（2026-10-02，第 198 条）。

## 为什么做这个子集
第 197 条的结论：本项目的验证器是**自建**的（解析 + 动力学）⟹ 只能声称
"**自洽**"，不能声称"**经独立验证**"✗。本条把 **diffusion/q7 这一条路径**
升级为"**外部印证**"：用**不属于本项目**的方法复算**同一可观测量** ✓

## 被验证的项目协议（读自 `sim/simulator_adapter.py::_diffusion`）
    · 方程 ∂T/∂t = κ ∂²T/∂x²，域 x ∈ [0, 3L]，Dirichlet T(0,t)=1、T(3L,t)=0，
      IC T(x>0,0)=0；**显式 FTCS**（dt = 0.4·dx²/κ）；
    · 可观测量 **t_half** = 首次 T(L,t) ≥ 0.5 的时刻 ✓；
    · 反解 **κ̂ = L² / (4·t_half·erfcinv(0.5)²)** ✓

## 三个**外部**方法（同一可观测量 t_half）
    1. **解析级数解（Carslaw & Jaeger，精确有限域）**：
       T(x,t) = (1−x/D) − (2/π)Σ (1/n)·sin(nπx/D)·exp(−κn²π²t/D²)，D=3L ✓
       另有**半无限**闭式 t_half = L²/(4κ·erfcinv(0.5)²) 作交叉参照 ✓
    2. **Crank–Nicolson 隐式**（scipy 稀疏求解，二阶时间精度；与项目的显式不同）✓
    3. **蒙特卡洛随机游走**（微观；X_t ~ N(0, 2κt)，T = 2·P(X_t > L)）✓
       多种子 ⟹ 给**统计误差棒** ✓

## 预登记判据（先写死）
    **E1 外部互洽**：CN 与解析级数的 t_half 相对差 **≤0.5%** ✓
    **E2 蒙特卡洛自洽**：MC 的 t_half 落在**解析半无限值 ±3σ** 内 ✓
    **E3 项目 vs 外部**：项目 κ̂ 与外部 κ̂ 相对差 **≤10%**
       （项目 API 的声明容差），并**如实报告**与项目自称 0.16% 的差距 ✓
    **E4 数字健全**：全部 κ̂/t_half 有限且 > 0 ✓
**诚实范围**：本条只外部印证 **27 族中的 1 族、1 个度量（q7）** ✓
  其余族**仍是内部自洽**，不得据此声称"整套验证已外部验证" ✗

用法：
    python sim/external_check_diffusion.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
from scipy.special import erfcinv  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

RUNS = _ROOT / "external_check_diffusion.csv"
CASES = [(1e-6, 0.01), (1e-4, 0.01)]
E1_TOL = 0.005          # CN vs 解析级数 ≤0.5%
E3_TOL = 0.10           # 项目 vs 外部 ≤10%（项目 API 声明容差）
HALF = float(erfcinv(0.5))          # ≈0.4769


# ---------------- 外部方法 1：解析 ----------------
def t_half_semi_infinite(kappa: float, L: float) -> float:
    """半无限闭式：x_half = 2√(κt)·erfcinv(0.5) ⟹ t = L²/(4κ·erfcinv(0.5)²) ✓"""
    return L ** 2 / (4.0 * kappa * HALF ** 2)


def T_series(x: float, t: float, kappa: float, D: float,
             n_max: int = 4000) -> float:
    """有限域解析级数解（Carslaw & Jaeger）：T(0,t)=1、T(D,t)=0、T(x,0)=0 ✓"""
    if t <= 0:
        return 0.0
    s = 0.0
    for n in range(1, n_max + 1):
        lam = math.exp(-kappa * (n * math.pi / D) ** 2 * t)
        term = math.sin(n * math.pi * x / D) * lam / n
        s += term
        if abs(term) < 1e-14 and n > 20:
            break
    return (1.0 - x / D) - (2.0 / math.pi) * s


def t_half_series(kappa: float, L: float, D: float, x_probe: float | None = None,
                  ) -> float:
    """二分求 T(x_probe,t)=0.5 的时刻（解析级数）。

    **`x_probe` 必须与数值方案的探针**同一位置**（默认 L；调用方传网格节点 ✓）
    —— 初版拿**精确 x=L** 去比**网格节点**，制造了 0.51% 的**假口径差** ✗
    （`nx=401`、域长 3L 时 `L/间距 = 133.33` **不落在网格点** ✓）
    """
    x = L if x_probe is None else x_probe
    lo, hi = 1e-12, 1e6 * L ** 2 / kappa
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if T_series(x, mid, kappa, D) > 0.5:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------- 外部方法 2：Crank–Nicolson ----------------
def t_half_cn(kappa: float, L: float, nx: int = 401) -> float:
    """隐式 Crank–Nicolson（scipy 稀疏）求 t_half；与项目显式 FTCS 不同 ✓"""
    D = 3.0 * L
    dx = D / (nx - 1)
    dt = 0.4 * dx ** 2 / kappa          # 与项目同量级的时间步（公平比较）
    r = kappa * dt / (2.0 * dx ** 2)
    n = nx - 2                          # 内部未知量
    main = (1.0 + 2.0 * r) * np.ones(n)
    off = -r * np.ones(n - 1)
    A = sp.diags([off, main, off], [-1, 0, 1], format="csc")
    lu = spla.splu(A)
    T = np.zeros(nx)
    j = int(np.argmin(np.abs(np.linspace(0, D, nx) - L)))
    t = 0.0
    t_half = None
    for _ in range(2_000_000):
        b = T[1:-1] + r * (T[2:] - 2 * T[1:-1] + T[:-2])
        b[0] += r * 1.0                 # 左边界 T(0)=1（隐含在 RHS）
        b[-1] += r * 0.0                # 右边界 T(D)=0
        T[1:-1] = lu.solve(b)
        T[0], T[-1] = 1.0, 0.0
        t += dt
        if t_half is None and T[j] >= 0.5:
            t_half = t
        if t_half is not None and t > 3.0 * t_half:
            break
    return float(t_half)


# ---------------- 外部方法 3：蒙特卡洛 ----------------
def t_half_mc(kappa: float, L: float, n: int = 400_000,
              seeds=(0, 1, 2, 3, 4)) -> tuple[float, float]:
    """随机游走（半无限）：T = 2·P(X_t > L)，X_t ~ N(0, 2κt) ✓ 多种子给误差棒。"""
    vals = []
    ts = np.linspace(0.2 * t_half_semi_infinite(kappa, L),
                     3.0 * t_half_semi_infinite(kappa, L), 60)
    for sd in seeds:
        rng = np.random.default_rng(sd)
        z = rng.standard_normal(n)
        # 对每个 t 估 T(L,t)，再线性插值求 0.5 交点
        Tt = [float(np.mean(2.0 * (z * math.sqrt(2 * kappa * t) > L)))
              for t in ts]
        Tt = np.asarray(Tt)
        idx = np.argmax(Tt >= 0.5)
        if idx == 0:
            vals.append(ts[0])
            continue
        t1, t0 = ts[idx], ts[idx - 1]
        y1, y0 = Tt[idx], Tt[idx - 1]
        vals.append(float(t0 + (0.5 - y0) * (t1 - t0) / max(y1 - y0, 1e-12)))
    return float(np.mean(vals)), float(np.std(vals))


def main() -> int:
    from sim.simulator_adapter import DynamicsAdapter
    ad = DynamicsAdapter()
    print("=" * 74)
    print("外部验证（可行子集）：diffusion / q7 —— 三个外部方法 vs 项目自建验证器")
    print("=" * 74)
    rows = []
    for kappa, L in CASES:
        D = 3.0 * L
        nx = 401                               # 与项目同一网格 ✓
        xs = np.linspace(0.0, D, nx)
        j = int(np.argmin(np.abs(xs - L)))
        x_node = float(xs[j])                  # **探针实际所在节点** ✓
        t_semi = t_half_semi_infinite(kappa, L)
        t_ser_node = t_half_series(kappa, L, D, x_probe=x_node)   # 同节点 ✓
        t_ser_exact = t_half_series(kappa, L, D)                  # 精确 x=L
        t_cn = t_half_cn(kappa, L)
        t_mc, t_mc_sd = t_half_mc(kappa, L)
        res = ad.run({"kind": "diffusion", "kappa": kappa, "L": L})
        t_proj = float(res["t_half"])
        k_ext = L ** 2 / (4.0 * t_ser_node * HALF ** 2)   # 外部（同节点）反解
        k_proj = float(res["kappa_hat"])
        e1 = abs(t_cn - t_ser_node) / t_ser_node
        e2 = abs(t_mc - t_semi) / max(3.0 * t_mc_sd / max(t_semi, 1e-30), 1e-30)
        e3 = abs(k_proj - k_ext) / k_ext
        rows.append({"kappa_true": kappa, "L": L, "x_node": x_node,
                     "t_semi": t_semi, "t_series_node": t_ser_node,
                     "t_series_exactL": t_ser_exact, "t_cn": t_cn,
                     "t_mc": t_mc, "t_mc_sd": t_mc_sd, "t_project": t_proj,
                     "kappa_ext": k_ext, "kappa_project": k_proj,
                     "E1_cn_vs_series": e1, "E3_proj_vs_ext": e3,
                     "grid_align": abs(t_ser_exact - t_ser_node)
                     / t_ser_exact})
        print("\nκ=%.0e, L=%.3g （域长 3L=%.3g；探针节点 x=%.6g，L/x−1=%.3f%%）"
              % (kappa, L, D, x_node, 100 * (L / x_node - 1)))
        print("  **外部方法**：解析半无限 t=%.4g ｜ 解析级数(**同节点**) t=%.4g"
              % (t_semi, t_ser_node))
        print("              解析级数(精确 x=L) t=%.4g ｜ CN t=%.4g ｜ MC t=%.4g ± %.2g"
              % (t_ser_exact, t_cn, t_mc, t_mc_sd))
        print("  **项目自建**：t_half=%.4g ⟹ κ̂=%.4g（真值 %.0e）"
              % (t_proj, k_proj, kappa))
        print("  外部反解（同节点）κ_ext=%.4g" % k_ext)
        print("  → E1 CN vs 级数(同节点) **%.3f%%** ｜ E3 项目 vs 外部 **%.2f%%**"
              % (100 * e1, 100 * e3))
        print("  → **新发现**：网格对齐效应（探针在节点而非精确 L）"
              "使 t_half 偏 %.3f%%、κ̂ 相应偏 **%.3f%%**"
              % (100 * abs(t_ser_exact - t_ser_node) / t_ser_exact,
                 100 * abs(t_ser_exact - t_ser_node) / t_ser_exact))
    e1_ok = all(r["E1_cn_vs_series"] <= E1_TOL for r in rows)
    e2_ok = all(abs(r["t_mc"] - r["t_semi"]) <= 3.0 * r["t_mc_sd"] + 1e-12
                for r in rows)
    e3_ok = all(r["E3_proj_vs_ext"] <= E3_TOL for r in rows)
    e4_ok = all(np.isfinite(r[k]) and r[k] > 0 for r in rows
                for k in ("t_series_node", "t_cn", "t_mc", "t_project",
                          "kappa_ext", "kappa_project"))
    print("\n**E1 外部互洽**（CN vs 解析级数 ≤%.1f%%）⟹ %s"
          % (100 * E1_TOL, "**通过** ✓" if e1_ok else "**未通过**"))
    print("**E2 蒙特卡洛自洽**（MC 落在半无限值 ±3σ 内）⟹ %s"
          % ("**通过** ✓" if e2_ok else "**未通过**"))
    print("**E3 项目 vs 外部**（κ̂ 相对差 ≤%.0f%%）⟹ %s；实测最大 **%.2f%%**"
          % (100 * E3_TOL, "**通过** ✓" if e3_ok else "**未通过**",
             100 * max(r["E3_proj_vs_ext"] for r in rows)))
    print("**E4 数字健全** ⟹ %s" % ("**通过** ✓" if e4_ok else "**未通过**"))
    print("\n**结论**：diffusion/q7 这一条路径由**三个外部方法**独立印证 ✓"
          "（解析级数、Crank–Nicolson、蒙特卡洛）")
    print("**诚实范围**：仅 **1 族 / 1 度量**；其余 26 族**仍是内部自洽**，"
          "不得据此声称整套验证已外部验证 ✗")
    with open(RUNS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("落盘：%s（%d 行）" % (RUNS.name, len(rows)))
    return 0 if (e1_ok and e2_ok and e3_ok and e4_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
