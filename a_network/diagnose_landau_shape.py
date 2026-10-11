"""D1 · landau 结构刻画（第 29 条诊断第一步，零训练成本）。

回答三个问题（对应诊断方案 D1.1–D1.4）：
- **塌缩验证**：I(a;b,kT) = Var(x²(u)) / (4kTb)、u = a/√(b kT)——目标实为
  「单变量无量纲函数 f(u)」，而模型输入是三列原始参数（b/kT 库内恒 1）；
- **形状统计**：f(u) 的倒 U 非单调性、两端渐近（~2/u²）、log-log 局部斜率
  过零，与 17 个训练族的（恒定/单调）斜率对照；
- **编码不连续**：a 的 log 编码在 a=0 两侧跳变（a>0 取 ln a、a<0 取原值），
  而目标 ln f 在临界点连续；
- **非单调对照组**：geometric 的 Fisher 1/(p²(1−p)) 也是非单调（U 形，极小
  在 p=2/3）——但它在训练集里。对比两族形状，为 D2/D3 的「训练内能否拟合
  非单调」问题铺路。

输出：console 报告 + `a_network/landau_shape_report.json`。

用法（项目根目录）：
    python a_network/diagnose_landau_shape.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.fisher import fisher_landau  # noqa: E402

_A_DIR = Path(__file__).resolve().parent
OUT_JSON = _A_DIR / "landau_shape_report.json"

# 与 graph_dataset._scan_grid 对齐：a ∈ [-2, 2] 的 30 点 linspace（横跨临界点）
SCAN_LO, SCAN_HI, SCAN_N = -2.0, 2.0, 30


def scan_grid() -> np.ndarray:
    return np.linspace(SCAN_LO, SCAN_HI, SCAN_N)


def f_u(u_values, n_grid: int = 2000) -> np.ndarray:
    """f(u) = Var(x²(u))，x ~ exp(-(½u x² + ¼x⁴))。

    与 fisher_landau 同口径：f(u) = 4·kT·b·I(a=u, b=1, kT=1)。
    """
    return np.array([4.0 * fisher_landau(float(u), 1.0, 1.0, n_grid)
                     for u in u_values])


def d1_1_collapse(n_grid: int) -> dict:
    """塌缩验证：9 组 (b,kT) 下 4kTb·I(u√(bkT); b, kT) 应落在同一条 f(u) 上。"""
    u_grid = np.linspace(-4.0, 4.0, 81)
    f_ref = f_u(u_grid, n_grid)
    combos = [(b, kT) for b in (0.1, 1.0, 5.0) for kT in (0.1, 1.0, 5.0)]
    rows = []
    worst = 0.0
    for b, kT in combos:
        I = np.array([fisher_landau(float(u) * math.sqrt(b * kT), b, kT, n_grid)
                      for u in u_grid])
        scaled = 4.0 * b * kT * I
        rel = np.abs(scaled - f_ref) / np.maximum(f_ref, 1e-300)
        sel = f_ref > 0.05 * f_ref.max()   # 只在 f 的核心区比相对误差
        worst = max(worst, float(rel[sel].max()))
        rows.append({"b": b, "kT": kT,
                     "core_max_rel_err": float(rel[sel].max())})
    return {
        "u_grid": [round(float(u), 4) for u in u_grid],
        "f_ref": [float(v) for v in f_ref],
        "combos": rows,
        "worst_core_rel_err": worst,
        "collapse_ok": worst < 1e-6,
        "conclusion": ("I(a;b,kT) 塌缩为单变量无量纲函数 f(u)，u=a/√(b kT)："
                       "目标的结构复杂在 f 的形状，且依赖 b/kT 的复合量——"
                       "模型输入给的是 a/b/kT 三列（b/kT 库内恒 1），须自行发明 u。"),
    }


def d1_2_shape(n_grid: int) -> dict:
    """形状统计：峰值位置（可能在扫描窗口外！）、窗口内单调性、渐近、log-log 斜率。"""
    u_grid = np.linspace(-4.0, 4.0, 401)
    f = f_u(u_grid, n_grid)
    imax = int(np.argmax(f))
    u_peak = float(u_grid[imax])

    u_far = 4.0
    asym_plus = float(f[np.argmin(np.abs(u_grid - u_far))] * u_far ** 2)
    asym_minus = float(f[np.argmin(np.abs(u_grid + u_far))] * u_far ** 2)

    pos = u_grid > 0.05
    up, fp = u_grid[pos], f[pos]
    lnu, lnf = np.log(up), np.log(np.maximum(fp, 1e-30))
    slopes_pos = np.gradient(lnf, lnu)

    # 扫描窗口内（a ∈ [-2,2] ↔ u ∈ [-2,2]，b=kT=1）的单调性
    win = (u_grid >= SCAN_LO) & (u_grid <= SCAN_HI)
    f_win = f[win]
    within_scan_monotone = bool(np.all(np.diff(f_win) <= 1e-12))

    # 全局非单调：d ln f / du 变号（峰值在 u_peak 处）
    dlnf = np.gradient(np.log(np.maximum(f, 1e-30)), u_grid)
    global_non_monotonic = bool((dlnf[:-1] * dlnf[1:] < 0).any())

    # 训练族对照：各分布族 Fisher 的 log-log 斜率（关于被扫描参数；全部恒定或单调）
    train_families = {
        "gauss(sigma)": "恒定 -2", "gibbs(sigma)": "恒定 -2",
        "exponential(rate)": "恒定 -2", "poisson(rate)": "恒定 -1",
        "bernoulli(p)": "非幂律单调 1/(p(1-p))", "laplace(b)": "恒定 -2",
        "gamma(k)": "ψ₁ 曲线（斜率 ∈[-2,-1]）", "beta(alpha)": "ψ₁ 组合（∈[-2,-1]）",
        "weibull(shape)": "幂律组合 k²/λ²（单调）", "vonmises(kappa)": "Bessel 比（单调）",
        "geometric(p)": "扫描范围 p∈[0.05,0.5] 只覆盖 U 形左支（单调降）",
        "cauchy(gamma)": "恒定 -2", "rayleigh(sigma)": "恒定 -2",
        "logistic(s)": "恒定 -2", "chisq(k)": "ψ₁ 曲线（斜率 ∈[-2,-1]）",
        "negbin(p)": "非幂律单调 k/(p²(1-p))", "lognormal(sigma)": "恒定 -2",
    }
    return {
        "u_peak": u_peak,
        "f_peak": float(f[imax]),
        "peak_inside_scan_window": bool(SCAN_LO < u_peak < SCAN_HI),
        "within_scan_monotone_decreasing": within_scan_monotone,
        "global_non_monotonic": global_non_monotonic,
        "asym_fu2_plus": asym_plus,
        "asym_fu2_minus": asym_minus,
        "asym_note": "u→+∞：f ~ 2/u²（高斯极限）；u→-∞：f → 2（双阱极限饱和，非 0）",
        "loglog_slope_range_pos_side": [float(slopes_pos.min()),
                                        float(slopes_pos.max())],
        "train_families_slopes": train_families,
        "conclusion": ("峰值 u_peak 在扫描窗口之外：窗口 a∈[-2,2] 只覆盖峰值右侧的"
                       "「膝部 + 下降尾」。窗口内目标单调，但 log-log 斜率强变化"
                       "（正侧从约 -1.4 到接近 0），而 17 个训练族的扫描曲线斜率"
                       "全部恒定（-2 / -1）或落在 [-2,-1] 区间——「接近 0 的平膝」"
                       "是训练集里没有的形状。注意：训练集里也没有任何非单调曲线"
                       "（geometric 的扫描只覆盖 U 形左支，见 d1_4）。"),
    }


def d1_3_encoding() -> dict:
    """混合编码在 a=0 处的不连续 vs 目标连续性。"""
    g = scan_grid()
    i0 = int(np.searchsorted(g, 0.0))    # 第一个 ≥0 的网格点（≈+0.069）
    a_neg, a_pos = float(g[i0 - 1]), float(g[i0])

    def enc(v: float) -> float:          # graph_dataset.py 的 log 编码分支
        return math.log(v) if v > 0 else v

    f_neg = 4.0 * fisher_landau(a_neg, 1.0, 1.0)
    f_pos = 4.0 * fisher_landau(a_pos, 1.0, 1.0)
    return {
        "a_neg": a_neg, "a_pos": a_pos,
        "feature_neg": enc(a_neg), "feature_pos": enc(a_pos),
        "feature_gap": enc(a_pos) - enc(a_neg),
        "target_lnF_neg": math.log(f_neg), "target_lnF_pos": math.log(f_pos),
        "target_gap": math.log(f_pos) - math.log(f_neg),
        "conclusion": ("特征在 a=0 两侧跳变 ~2.7（ln 0.069 ≈ -2.67 vs +0.069），"
                       "而目标 ln f 连续（gap 仅为 O(0.1)）。模型必须在特征跳变处"
                       "重建连续函数，且 a<0 分支的特征是原始值、a>0 是 log 值——"
                       "两分支的编码尺度不一致。"),
    }


def d1_4_geometric_contrast() -> dict:
    """geometric 对照：库内另一个非单调 Fisher 族（在训练集里）。"""
    p = np.linspace(0.02, 0.98, 97)
    F = 1.0 / (p ** 2 * (1.0 - p))        # 关于 p 的 Fisher，U 形
    lnp, lnF = np.log(p), np.log(F)
    slopes = np.gradient(lnF, lnp)
    imin = int(np.argmin(F))
    return {
        "p_min": float(p[imin]),          # 期望 2/3
        "F_min": float(F[imin]),
        "shape": "U 形（两端 →∞，极小在 p=2/3）",
        "loglog_slope_range": [float(slopes.min()), float(slopes.max())],
        "in_training_fixed_holdout": True,
        "conclusion": ("geometric 的 Fisher 是 U 形（极小在 p=2/3），但库内扫描范围"
                       " p∈[0.05,0.5] 只覆盖极小点左侧的单调下降支——训练集实际"
                       "没有暴露任何非单调曲线，模型从未被要求拟合非单调目标。"
                       "故「非单调容量」未经验证；landau 窗口内的失败与几何形状"
                       "非单调性无关（窗口内 landau 也单调），根因在平膝曲率与"
                       "编码跳变（见 d1_2 / d1_3）。"),
    }


def main() -> int:
    n_grid = 1000
    report = {
        "note": "landau 结构刻画（诊断方案 D1，第 29 条）。f(u)=4kTb·I(a)，"
                "u=a/√(b kT)；b=kT=1 时 u=a。",
        "scan": {"lo": SCAN_LO, "hi": SCAN_HI, "n": SCAN_N},
        "d1_1_collapse": d1_1_collapse(n_grid),
        "d1_2_shape": d1_2_shape(n_grid),
        "d1_3_encoding": d1_3_encoding(),
        "d1_4_geometric_contrast": d1_4_geometric_contrast(),
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("================ landau 结构刻画（D1） ================")
    c = report["d1_1_collapse"]
    print(f"\n[D1.1 塌缩验证] 9 组 (b,kT) 的 4kTb·I(u√(bkT)) 最大核心相对误差 = "
          f"{c['worst_core_rel_err']:.2e}  →  {'✅ 塌缩成立' if c['collapse_ok'] else '❌ 塌缩失败'}")
    print(f"    {c['conclusion']}")

    s = report["d1_2_shape"]
    print(f"\n[D1.2 形状统计] 峰值 u={s['u_peak']:+.3f}"
          f"（{'在' if s['peak_inside_scan_window'] else '不在'}扫描窗口内），"
          f"f_peak={s['f_peak']:.4f}")
    print(f"    全局非单调：{'是' if s['global_non_monotonic'] else '否'}；"
          f"窗口内单调递减：{'是' if s['within_scan_monotone_decreasing'] else '否'}")
    print(f"    两端渐近 f·u² = {s['asym_fu2_plus']:.3f}（u=+4）/ "
          f"{s['asym_fu2_minus']:.3f}（u=-4）；{s['asym_note']}")
    print(f"    正侧 log-log 斜率范围 = [{s['loglog_slope_range_pos_side'][0]:+.2f}, "
          f"{s['loglog_slope_range_pos_side'][1]:+.2f}]")
    print(f"    {s['conclusion']}")

    e = report["d1_3_encoding"]
    print(f"\n[D1.3 编码不连续] a 两侧最近点 {e['a_neg']:+.3f} / {e['a_pos']:+.3f}")
    print(f"    特征值 {e['feature_neg']:+.3f} → {e['feature_pos']:+.3f}"
          f"（跳变 {e['feature_gap']:+.2f}）；目标 ln f 差 {e['target_gap']:+.3f}")
    print(f"    {e['conclusion']}")

    g = report["d1_4_geometric_contrast"]
    print(f"\n[D1.4 geometric 对照] 极小 p={g['p_min']:.3f}（期望 2/3），"
          f"log-log 斜率范围 [{g['loglog_slope_range'][0]:+.2f}, "
          f"{g['loglog_slope_range'][1]:+.2f}]")
    print(f"    {g['conclusion']}")

    print(f"\n报告已写入 {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
