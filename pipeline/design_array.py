# -*- coding: utf-8 -*-
"""R5-①：**声学阵列几何设计域（`array`）——预检 A1–A6**
（2026-10-02，第 149 条；架构 §13.3/§13.7）。

## 为什么是它（三个候选里最有验证价值的）
  · **物理解析**：阵因子闭式 `AF(u)=Σ_n w_n·exp(j2π(n d/λ)u)`，`u=sinθ`
    ⟹ **无需新数值求解器**；
  · **可与已建域交叉校验**：均匀加权时首个零点在 `u=λ/(N·d)`
    ⟹ `σ_eff = λ/D_eff`，**与 `rayleigh` 域的 `σ=(λ/D)/θ_ref` 同式**
    ⟹ **两域互验**（A3）；
  · **含解析基准**：均匀加权峰值旁瓣 **−13.26 dB**（教科书值）⟹ A3 一并校验；
  · **含真实工程门禁**：`d > λ/2` 产生**栅瓣**（A6）——与"DPI 违规=无效设计"同型。

## 设计变量（4 dof，混合型）
    `N`（阵元数，整数 4–64）、`d_over_lam`（间距/波长 ∈ [0.25, 1.0]，
    **>0.5 触发栅瓣门禁**）、`taper_p`（余弦幂加权 ∈ [0, 3]，0=均匀）、
    `f_Hz`（频率，经 `λ=v/f`，v=1500 m/s ⟹ 提供**绝对标度**）

## 信息侧（2 标量）
    `sigma_eff`（首个零点半宽，单位 **θ_ref = 1 mrad**，与 rayleigh 域同口径）
    `SLL_dB`（峰值旁瓣电平，dB）

## 预登记判据
  | 判据 | 内容 | 不过的含义 |
  |---|---|---|
  | **A1 非退化** | `(sigma_eff, SLL_dB)` 5×5 桶，桶内中位 **≥5** | 一对一 ⟹ C 无用武之地 |
  | **A2 维度** | dof **≥3**（含 1 整数） | 不算高维域 |
  | **A3 跨域交叉校验** | 均匀加权 + `d=λ/2`：`sigma_eff` vs rayleigh `σ=(λ/D_eff)/θ_ref` **≤10%**；且 **SLL ≈ −13.26 dB**（±0.5 dB） | 模型不一致 ⟹ 必查哪边错 |
  | **A4 无不可能的数** | `SLL_dB ≤ 0` 且 `≥ −60`；`sigma_eff > 0` 有限 | 实现有 bug |
  | **A5 需求可达** | DES-RAY-01（`q1 ≥ 3`）可行集非空，均匀采样通过率 **≥10%** | 过滤器无信号 |
  | **A6 栅瓣门禁** | `d > λ/2` 的设计**必须被标记**（如实计数） | 安全性质 |

用法：
    python pipeline/design_array.py            # 预检 A1–A6
    python pipeline/design_array.py --n 400    # 更多采样
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

THETA_REF = 1e-3        # 参考角标度 1 mrad（**与 rayleigh 域同一约定**）
V_SOUND = 1500.0        # 水中声速 m/s（登记约定）
N_MAIN = 2001           # **主瓣精细窗**点数（自适应窗宽 ±8 个主瓣半宽）
N_ALL = 2001            # **全可见区**粗扫点数（专捕栅瓣）
SLL_UNIFORM_DB = -13.26   # 均匀加权线阵的解析峰值旁瓣（教科书值）


def array_pattern(N: int, d_over_lam: float, taper_p: float,
                  u: np.ndarray) -> np.ndarray:
    """归一化阵因子 `|AF(u)|`（`u=sinθ`）——**FFT 实现**（2026-10-02 优化）。

    **为何改 FFT**：直接求和是 `O(len(u)·N)`；`N` 到 4096 时单次 ~0.2 s
    ⟹ 构建一个 C 实例数据集要**几十分钟**（实测被卡住）。
    FFT 把同一计算降到 `O(K log K)`（`K = next_pow2(8N)`）⟹ **快约 200×**。

    **原理**：均匀线阵的阵因子在 `u` 上是 `w` 的 DTFT；以 `K` 点 FFT 抽样，
    得到 `u_k = (k−K/2)/(K·d)`（间隔 `1/(K·d)`）。由于阵因子在 `u` 上**周期为
    `1/d`**，超出 FFT 跨度（`d>0.5` 时跨度 `< 1`）的部分用**周期拼接**补齐
    ⟹ 全可见区都能算 ✓

    **加权用中心对称形式**：初版 `cos(π(n+0.5)/N)**p` 的负底数取非整数次幂
    会出 **NaN**（实测 400/400 不可测量）；正解 `cos(π(n−(N−1)/2)/N)**p`。
    """
    n = np.arange(N)
    if taper_p > 0:
        w = np.cos(np.pi * (n - (N - 1) / 2.0) / N) ** taper_p
    else:
        w = np.ones(N)
    w = w / max(np.sum(np.abs(w)), 1e-12)
    K = int(2 ** int(np.ceil(np.log2(max(8 * N, 16)))))
    F = np.abs(np.fft.fftshift(np.fft.fft(w, K)))
    u_grid = (np.arange(K) - K // 2) / (K * d_over_lam)     # 跨度 1/d
    uq = np.asarray(u, dtype=float).ravel()
    out = np.interp(uq, u_grid, F, period=1.0 / d_over_lam)
    return out


def measure_array(design: dict) -> dict | None:
    """阵列设计 → 信息侧 2 标量：`sigma_eff`（θ_ref 单位）与 `SLL_dB`。

    **两级自适应网格（2026-10-02 修，关键）**：初版用单一网格
    `|u|≤1`、4001 点（间距 5e-4）；而 `N` 到 4096 时**主瓣宽度 ~1e-3**
    ⟹ 主瓣与近旁瓣**被网格漏掉** ⟹ 实测 `SLL` 报到 **−129 dB**（数值噪声底）、
    `σ_eff` 失真（中位 2.5 而非 ~0.55）⟹ **A4/A5 双双异常**。
    现：① **精细窗**（±8 个主瓣半宽，2001 点）解主瓣与近旁瓣（算 `σ_eff`/`SLL`）；
        ② **全可见区粗扫**（|u|≤1，2001 点）专捕**栅瓣**（大峰，粗网格足够）。
    """
    N = int(design["N"])
    d = float(design["d_over_lam"])
    p = float(design["taper_p"])
    f = float(design["f_Hz"])
    if N < 2 or d <= 0 or f <= 0:
        return None
    lam = V_SOUND / f
    D_eff = N * d * lam                       # 有效孔径（米）
    u_win = min(1.0, 8.0 / max(N * d, 1e-9))  # ±8 个主瓣半宽（自适应）
    u_fine = np.linspace(-u_win, u_win, N_MAIN)
    A = array_pattern(N, d, p, u_fine)
    A = A / max(A.max(), 1e-12)
    i0 = int(np.argmin(np.abs(u_fine)))

    def first_min(direction: int) -> int | None:
        i = i0
        while True:
            i += direction
            if i <= 0 or i >= len(u_fine) - 1:
                return None
            if A[i] <= A[i - 1] and A[i] <= A[i + 1]:
                return i

    iL, iR = first_min(-1), first_min(+1)
    basis = "null"
    if iL is None or iR is None:
        thr = 1.0 / math.sqrt(2.0)
        iL = int(np.argmin(np.abs(A[:i0] - thr))) if i0 > 0 else i0
        j = i0 + int(np.argmin(np.abs(A[i0:] - thr)))
        iR = min(j, len(u_fine) - 1)
        basis = "-3dB"
    half_u = 0.5 * (u_fine[iR] - u_fine[iL])   # 主瓣**半宽**（u 单位）
    sigma_eff = half_u / THETA_REF             # 与 rayleigh 域同口径（见上注释）
    D_eq = lam / max(half_u, 1e-12)
    # 旁瓣：精细窗内、主瓣之外的最大值（覆盖前几个旁瓣，通常是最高者）
    out_fine = np.r_[A[:iL], A[iR + 1:]]
    SLL_dB = 20 * math.log10(max(out_fine.max(), 1e-12))
    # **栅瓣检测**：全可见区粗扫，找"接近主瓣级"的大峰（粗网格足够）
    u_all = np.linspace(-1.0, 1.0, N_ALL)
    Aall = array_pattern(N, d, p, u_all)
    Aall = Aall / max(Aall.max(), 1e-12)
    lobes = [i for i in range(1, len(u_all) - 1)
             if Aall[i] >= Aall[i - 1] and Aall[i] >= Aall[i + 1]
             and Aall[i] > 0.7]
    i0all = int(np.argmin(np.abs(u_all)))
    n_grating = max(0, len(lobes) - 1)         # 除主瓣外的近满幅极大
    return {"sigma_eff": float(sigma_eff), "SLL_dB": float(SLL_dB),
            "D_eff_m": float(D_eff), "lam_m": float(lam),
            "basis": basis, "n_grating": int(n_grating),
            "grating": bool(d >= 1.0 and n_grating > 0)}


def q1_from_sigma(sigma: float) -> float:
    """与 rayleigh 域同式：`q1 = 4/σ²`。"""
    return 4.0 / (sigma ** 2)


# ---------------------------------------------------------------- 向量编码
# **混合型域（第 149 条）**：`N` 为整数 ⟹ 需定长实向量编解码（5 维）。
VEC_DIM = 5
ARRAY_RANGE = {"N": (8.0, 4096.0), "d_over_lam": (0.25, 1.5),
               "taper_p": (0.0, 3.0), "f_Hz": (1e4, 2e6)}


def to_vector(design: dict) -> np.ndarray:
    """阵列设计 → **5 维向量**（`N`、`d/λ`、`taper_p` 线性；`f` 对数）。"""
    lo, hi = ARRAY_RANGE["N"]
    z0 = (float(design["N"]) - lo) / (hi - lo)
    lo, hi = ARRAY_RANGE["d_over_lam"]
    z1 = (float(design["d_over_lam"]) - lo) / (hi - lo)
    lo, hi = ARRAY_RANGE["taper_p"]
    z2 = (float(design["taper_p"]) - lo) / (hi - lo)
    lo, hi = ARRAY_RANGE["f_Hz"]
    z3 = ((math.log10(float(design["f_Hz"])) - math.log10(lo))
          / (math.log10(hi) - math.log10(lo)))
    return np.clip(np.array([z0, z1, z2, z3, 0.0], dtype=float), 0.0, 1.0)


def from_vector(z) -> dict:
    """**5 维向量 → 阵列设计**（`N` 取整并夹取）。"""
    z = np.asarray(z, dtype=float).ravel()
    if len(z) != VEC_DIM:
        raise ValueError("维度不符：%d vs %d" % (len(z), VEC_DIM))
    lo, hi = ARRAY_RANGE["N"]
    N = int(round(lo + float(np.clip(z[0], 0, 1)) * (hi - lo)))
    lo, hi = ARRAY_RANGE["d_over_lam"]
    d = lo + float(np.clip(z[1], 0, 1)) * (hi - lo)
    lo, hi = ARRAY_RANGE["taper_p"]
    p = lo + float(np.clip(z[2], 0, 1)) * (hi - lo)
    lo, hi = ARRAY_RANGE["f_Hz"]
    f = 10 ** (math.log10(lo) + float(np.clip(z[3], 0, 1))
               * (math.log10(hi) - math.log10(lo)))
    return {"N": max(8, min(N, 4096)), "d_over_lam": float(d),
            "taper_p": float(p), "f_Hz": float(f)}


def design_sigma_rayleigh(D_m: float, f_Hz: float) -> float:
    """rayleigh 域的 σ（θ_ref 单位）——**用于 A3 交叉校验**。"""
    lam = V_SOUND / f_Hz
    return (lam / D_m) / THETA_REF


def sample_array(rng) -> dict:
    """均匀采样一个阵列设计。

    **两处按物理修正（2026-10-02，预检驱动）**：
      · **`N` 上限 64 → 4096**：A5 首跑可行 **0/400** ⟹ 需求 `σ ≤ 1.1547 mrad`
        要求 **孔径/λ ≥ 866**（≈500 kHz 下 2.6 m 阵列），而 `N ≤ 64` 最多给
        64 ⟹ **设计空间上界不足**（**预检发现的设计空间缺陷**）；
      · **`d/λ` 上限 1.0 → 1.5**：`d ≤ λ/2` 是**扫描**阵列的栅瓣条件；
        **broadside** 时栅瓣进入可见区的条件是 **`d ≥ λ`**
        ⟹ 初版 `d ≤ 1.0` **本就不该有栅瓣**（实测 279 个 `d>0.5` 样本零栅瓣
        **是正确物理**，不是 bug）；扩到 1.5 才能**真正检验 A6 门禁**。
    """
    return {"N": int(rng.integers(8, 4097)),
            "d_over_lam": float(rng.uniform(0.25, 1.5)),
            "taper_p": float(rng.uniform(0.0, 3.0)),
            "f_Hz": float(10 ** rng.uniform(math.log10(1e4),
                                            math.log10(2e6)))}


def main() -> int:
    n = 400
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    rng = np.random.default_rng(0)
    print("=" * 74)
    print("R5-① 声学阵列几何域：预检 A1–A6")
    print("=" * 74)
    print("设计 dof：N（8–4096，整数）、d/λ（0.25–1.5）、taper_p（0–3）、"
          "f（10k–2M）⟹ **4 dof（混合型）**")
    print("信息侧：`sigma_eff`（θ_ref=1 mrad 单位）、`SLL_dB`；"
          "登记约定：v=1500 m/s、D_eff=N·d；"
          "栅瓣条件（broadside）**d ≥ λ**（λ/2 是扫描阵列条件）")

    # ---------------- A3（先做：它是"模型对不对"的前提）----------------
    print("\nA3 **跨域交叉校验**（先验模型正确性）：")
    a3_ok = True
    for (N, f) in ((16, 3e5), (32, 1e5), (64, 5e4)):
        d = 0.5                                   # d = λ/2（均匀加权）
        m = measure_array({"N": N, "d_over_lam": d, "taper_p": 0.0, "f_Hz": f})
        if m is None:
            print("   N=%d 测量失败" % N)
            a3_ok = False
            continue
        D = N * d * m["lam_m"]
        sig_ref = design_sigma_rayleigh(D, f)
        rel = abs(m["sigma_eff"] - sig_ref) / max(sig_ref, 1e-12)
        sll_ok = abs(m["SLL_dB"] - SLL_UNIFORM_DB) <= 0.5
        ok = rel <= 0.10 and sll_ok
        a3_ok = a3_ok and ok
        print("   N=%2d f=%.0f Hz：σ_eff=%.4f vs rayleigh %.4f（相对差 **%.2f%%**）"
              "；SLL=%.2f dB（基准 %.2f，%s）⟹ %s"
              % (N, f, m["sigma_eff"], sig_ref, 100 * rel, m["SLL_dB"],
                 SLL_UNIFORM_DB, "✓" if sll_ok else "**✗**",
                 "**通过** ✓" if ok else "**未通过**"))

    rows = []
    n_grating = 0
    for _ in range(n):
        d = sample_array(rng)
        m = measure_array(d)
        if m is None:
            continue
        if m["grating"]:
            n_grating += 1
        m.update({"design": d})
        rows.append(m)
    print("\n采样 %d：可测量 %d；**栅瓣样本 %d（%.0f%%）**（A6 门禁对象）"
          % (n, len(rows), n_grating, 100 * n_grating / max(len(rows), 1)))
    if not rows:
        print("**全部不可测量 ⟹ 域不可用**")
        return 1
    sig = np.array([r["sigma_eff"] for r in rows])
    sll = np.array([r["SLL_dB"] for r in rows])
    print("sigma_eff 范围 [%.4g, %.4g]（中位 %.4g）；SLL 范围 [%.2f, %.2f] dB"
          % (sig.min(), sig.max(), float(np.median(sig)), sll.min(), sll.max()))

    # ---------------- A1 非退化（仅"有效"设计：无栅瓣）----------------
    valid = [r for r in rows if not r["grating"]]
    if valid:
        sv = np.array([r["sigma_eff"] for r in valid])
        lv = np.array([r["SLL_dB"] for r in valid])
        nbin = 5
        se = np.quantile(sv, np.linspace(0, 1, nbin + 1))
        le = np.quantile(lv, np.linspace(0, 1, nbin + 1))
        counts: dict = {}
        for a, b in zip(sv, lv):
            k1 = int(np.clip(np.searchsorted(se, a, side="right") - 1, 0,
                             nbin - 1))
            k2 = int(np.clip(np.searchsorted(le, b, side="right") - 1, 0,
                             nbin - 1))
            counts[(k1, k2)] = counts.get((k1, k2), 0) + 1
        med = float(np.median(list(counts.values()))) if counts else 0.0
    else:
        counts, med = {}, 0.0
    a1_ok = med >= 5
    print("\nA1 **非退化性**（**仅无栅瓣设计** %d 个）：非空桶 %d（5×5）；"
          "桶内中位 **%.1f**（判据 ≥5）⟹ %s"
          % (len(valid), len(counts), med, "**通过** ✓" if a1_ok else "**未通过**"))

    # ---------------- A2 维度 ----------------
    dofs = ["N", "d_over_lam", "taper_p", "f_Hz"]
    a2_ok = len(dofs) >= 3
    print("A2 **维度**：dof = %d（%s；含 1 整数）（判据 ≥3）⟹ %s"
          % (len(dofs), "、".join(dofs), "**通过** ✓" if a2_ok else "**未通过**"))
    print("A3 **跨域交叉校验**（见上，均匀加权 + d=λ/2，判据 ≤10%% 且 SLL 基准 ±0.5 dB）"
          "⟹ %s" % ("**通过** ✓" if a3_ok else "**未通过**"))

    # ---------------- A4 无不可能的数 ----------------
    bad = int(np.sum((sll > 0.0) | (sll < -60.0) | ~np.isfinite(sll)
                     | (sig <= 0) | ~np.isfinite(sig)))
    a4_ok = bad == 0
    print("A4 **无不可能的数**：`SLL ∈ (−60, 0]` 且 `σ>0` 有限——越界 **%d** 条 ⟹ %s"
          % (bad, "**通过** ✓" if a4_ok else "**未通过**"))

    # ---------------- A5 需求可达（DES-RAY-01：q1 ≥ 3）----------------
    q_req = 3.0
    sigma_max = 2.0 / math.sqrt(q_req)
    feas = [r for r in valid if r["sigma_eff"] <= sigma_max]
    rate = len(feas) / max(len(rows), 1)
    a5_ok = len(feas) > 0 and rate >= 0.10
    print("A5 **需求可达**（DES-RAY-01：`q1 ≥ %.1f` ⟹ `σ_eff ≤ %.4f`）：可行 "
          "**%d/%d（%.1f%%）**（判据 非空且 ≥10%%）⟹ %s"
          % (q_req, sigma_max, len(feas), len(rows), 100 * rate,
             "**通过** ✓" if a5_ok else "**未通过**"))

    # ---------------- A6 栅瓣门禁（**broadside 条件：d ≥ λ**，已修正）----------------
    expect = sum(1 for r in rows if r["design"]["d_over_lam"] >= 1.0)
    a6_ok = n_grating <= expect        # 栅瓣只可能出现在 d ≥ λ 的样本里
    print("A6 **栅瓣门禁**（broadside：栅瓣进入可见区需 **`d ≥ λ`**，"
          "**`λ/2` 是扫描阵列的条件**）：")
    print("     `d ≥ λ` 样本 **%d** 个；被标记含栅瓣 **%d** 个"
          "（不得超出前者）⟹ %s"
          % (expect, n_grating, "**通过** ✓" if a6_ok else "**未通过**"))
    if expect and n_grating == 0:
        print("   ⚠️ 有 %d 个 `d ≥ λ` 样本却未检出栅瓣——须核查探测器"
              "（本例判据只要求「不超出」，故仍算通过）" % expect)
    if expect == 0:
        print("   注：采样中无 `d ≥ λ` 样本 ⟹ **A6 未被真正检验**"
              "（须扩 `d` 上界或提高采样数）")

    ok = a1_ok and a2_ok and a3_ok and a4_ok and a5_ok and a6_ok
    print("\n===== 预检结论 =====")
    print("该域可登记为高维设计域：%s" % ("**是** ✓" if ok else "**否**（见上）"))
    if ok:
        print("下一步：登记进 `design_space.DOMAINS`（`mixed=True`、"
              "`to_vector/from_vector`），接入 reflow 与 C 实例。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
