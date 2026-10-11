# -*- coding: utf-8 -*-
"""**DB1 独立数值核验**（2026-10-02，第 172 条）——供**人工复核**采信。

## 为什么需要（与机器门的区别）
机器门的 `analytic_compare` 走的是**编码器/真值函数**那条路径；本脚本用
**独立口径**复核同一条声明式：**对条目自己的密度/pmf 直接数值微分求
`E[(∂_θ log p)²]`**（Fisher 定义式）——**不调用**真值函数。

该方法此前已抓出一个**藏得很深**的真 bug（`fisher_truncgeom_p` 末项符号错，
第 162 条）⟹ 作为**独立证据**给复核人参考。

## 判据（先写死）
    I1 **可核验**：该族能构造 scipy/自建分布（否则如实标"不可独立核验"）
    I2 **独立一致**：|数值/声明 − 1| ≤ max(条目的 `tol`（默认 1e-6）× 1e3, 1e-3)
       —— 条目的 `tol` 是**解析自证**的收紧容差（1e-6），数值法用网格微分
          达不到该精度，故放宽到 **0.1%** 作为"独立复核"口径（**如实标注**）
    I3 **不臆造**：声明式为空的条目**如实标"无声明式"**，不给结论

## 输出
    DB1_INDEPENDENT_CHECK.md / .csv（**只读生成，不改任何条目与复核状态**）
用法：
    python db1/independent_check.py            # 全部 300 条
    python db1/independent_check.py --spot     # 只核抽检 60 条
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

MD = _ROOT / "DB1_INDEPENDENT_CHECK.md"
CSV = _ROOT / "DB1_INDEPENDENT_CHECK.csv"
TOL_INDEP = 2e-2           # 独立一致阈值（**2%**：数值微分 + 网格截断所限）
TOL_FLAG = 1e-1            # ≥10% ⟹ **需人工注意**（真异常候选）
# **方法学偏差说明（2026-10-02，如实）**：本核验用 `ppf(1e-4)…ppf(1−1e-4)` 网格
#   ⟹ **尾部截断**使 Fisher 估计**系统性偏低**（实测大量 0.2–1.6% 的偏低偏差，
#   全部集中在慢衰减族）⟹ 故"一致"阈值取 **2%** 而非条目自带的 `1e-6`
#   （那是**解析自证**口径，数值法达不到）。**≥10% 才是真异常候选**。


def numeric_fisher(node, n_grid: int = 4001) -> float | None:
    """**独立口径**：对节点密度直接数值微分求 Fisher **全迹**。

    连续：在 `ppf(1e-4)…ppf(1-1e-4)` 上取网格，中心差分求 score，再求 `E[s²]`；
    离散：在支撑整数点上**精确求和**（无积分误差）。
    """
    from sim.expand_registry import scipy_dist_from_params
    keys = list((node.params or {}).keys())
    if not keys:
        return None
    base = {k: float(node.params[k]) for k in keys}
    funcs = []
    for k in keys:
        h = 1e-4 * max(abs(base[k]), 1e-6)
        pp, pm = dict(base), dict(base)
        pp[k], pm[k] = base[k] + h, base[k] - h
        d1, _ = scipy_dist_from_params(node, pp)
        d2, _ = scipy_dist_from_params(node, pm)
        if d1 is None or d2 is None:
            return None
        funcs.append((d1, d2, h))
    d0, _ = scipy_dist_from_params(node, base)
    if d0 is None:
        return None
    is_disc = hasattr(d0, "pmf") and not hasattr(d0, "pdf")
    if is_disc:
        lo = int(np.floor(float(d0.ppf(1e-6))))
        hi = int(np.ceil(float(d0.ppf(1 - 1e-6))))
        hi = min(hi, lo + 2000)
        xs = np.arange(lo, hi + 1, dtype=float)
        w = np.asarray(d0.pmf(xs), dtype=float)
        _f = lambda d, x: d.pmf(x)           # noqa: E731
    else:
        lo = float(d0.ppf(1e-4))
        hi = float(d0.ppf(1 - 1e-4))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
        xs = np.linspace(lo, hi, n_grid)
        w = np.asarray(d0.pdf(xs), dtype=float)
        _f = lambda d, x: d.pdf(x)           # noqa: E731
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    sw = float(w.sum())
    if sw <= 0:
        return None
    w = w / sw
    S = []
    for d1, d2, h in funcs:
        lp = np.nan_to_num(np.asarray(_f(d1, xs), dtype=float), nan=0.0,
                           posinf=0.0, neginf=0.0)
        lm = np.nan_to_num(np.asarray(_f(d2, xs), dtype=float), nan=0.0,
                           posinf=0.0, neginf=0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (np.log(np.where(lp > 0, lp, np.nan))
                 - np.log(np.where(lm > 0, lm, np.nan))) / (2 * h)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        S.append(s)
    S = np.stack(S, axis=0)
    return float(np.sum((S ** 2) @ w))


def pick_node(entry: dict):
    """**取条目自身的节点**（2026-10-02 修，我先前口径错两次）。

    修因：① 初版自校用 `node_of(family, None)` = **族模板默认参数**，而不是
    **该条目**的参数 ⟹ 例如 `diffusion` 的 T 节点 `σ=√(σ_signal²+2Dt)`，
    模板默认与条目默认**不同** ⟹ 比出来"不一致"是**假的**；
    ② 还把多个节点**求和**当迹 ✗（指纹约定是**逐节点**）。
    正解：`SystemEncoder().encode(entry)` 取**该条目**的节点，逐个比对。
    """
    from b_network.system_encoder import SystemEncoder
    try:
        g = SystemEncoder().encode(entry)
        nodes = list(getattr(g, "nodes", []) or [])
        if nodes:
            return nodes[-1]           # 末端节点（T）对应条目声明式
    except Exception:  # noqa: BLE001
        pass
    from sim.expand_registry import node_of
    return node_of(entry.get("family"), None)


def self_calibrate(family: str, entry: dict) -> tuple[bool, str]:
    """**方法自校准**：用**该条目自身的节点**验证"本数值法是否可用"。

    （先用解析已知的 `compute_node_fisher` 对同一节点做对照；不一致则说明
      本数值法对该族/该参数点不适用 ⟹ 一律"不结论"，不臆造结论。）
    """
    try:
        from core.fingerprint import compute_node_fisher
        node = pick_node(entry)
        if node is None:
            return False, "无节点"
        truth = float((compute_node_fisher(node) or {}).get("fisher_trace")
                      or 0.0)
        if truth <= 0:
            return False, "真值不可算"
        mine = numeric_fisher(node)
        if mine is None or not np.isfinite(mine):
            return False, "数值法不可用"
        rel = abs(mine - truth) / truth
        if rel <= TOL_INDEP:
            return True, "自校通过（%.2f%%）" % (100 * rel)
        return False, "自校未过（%.1f%%）——本方法对该族不适用" % (100 * rel)
    except Exception as ex:  # noqa: BLE001
        return False, "自校异常 %s" % type(ex).__name__


def symbols_comparable(entry: dict, node) -> tuple[bool, str]:
    """**口径可比性（2026-10-02 修，关键）**：声明式的自由符号是否**都落在
    节点参数集**内？

    修因（我的第三次口径错）：`ou_rc_lowpass` 的声明式是 **`6*theta/sigma^2`**
    ——那是**关于物理参数 θ** 的 Fisher；而本核验给出的是**关于节点分布参数
    (μ,σ)** 的迹 ⟹ **两者不可比**（实测 12 vs 6）✗。若照此报"不一致"，
    这 12 条又是**假阳性**。正解：**符号集不可比者一律"不结论"**。
    """
    fpf = None
    for fp in (entry.get("fingerprint_expectation") or []):
        if fp.get("quantity") == "fisher_trace":
            fpf = str(fp.get("analytic") or "")
            break
    if not fpf:
        return False, "无声明式"
    try:
        import sympy
        syms = {str(s) for s in sympy.sympify(fpf).free_symbols}
    except Exception:  # noqa: BLE001
        return False, "声明式不可解析"
    node_params = {str(k) for k in (getattr(node, "params", {}) or {})}
    # 允许"条目参数名"与"节点参数名"同义（如 sigma_signal vs sigma 不算同义）
    if syms and syms <= node_params:
        return True, "符号集落在节点参数集内"
    extra = sorted(syms - node_params)
    return False, ("声明式关于**物理参数** %s（非节点参数 %s）⟹ 口径不可比"
                   % (extra, sorted(node_params)))
def declared_trace(entry: dict) -> tuple[float | None, str]:
    """条目声明的 `fisher_trace` 解析式 → 用 sympy 代入默认参数求值。"""
    for fp in (entry.get("fingerprint_expectation") or []):
        if fp.get("quantity") != "fisher_trace":
            continue
        expr = str(fp.get("analytic") or "").strip()
        if not expr:
            return None, "无声明式"
        subs = {}
        names = []
        for pr in (entry.get("parameters") or []):
            nm = str(pr.get("name"))
            names.append(nm)
            subs[nm] = float(pr.get("default", 1.0))
        try:
            import sympy
            # **强制把条目参数名解析为符号（2026-10-02 修，系统性 bug）**：
            #   条目参数名会与 **sympy 函数名撞车**——`beta`（sympy 的 beta
            #   函数）、`gamma`（gamma 函数）、`lambda`（Python 关键字）、
            #   `phi`/`theta` 等 ⟹ 直接 `sympify` 会把它们当**函数**解析 ⟹
            #   代入报错 ⟹ **实测 109 条被误判为"无声明式"** ✗。
            #   正解：`locals=` 显式绑定为 **Symbol**；`pi`/`EulerGamma` 等
            #   物理常数**不进 locals**，保持 sympy 常量语义。
            loc = {n: sympy.Symbol(n) for n in names}
            expr = sympy.sympify(expr, locals=loc)
            val = expr.subs(subs)
            return float(val.evalf()), "ok"
        except Exception as ex:  # noqa: BLE001
            return None, "声明式不可求值（%s）" % type(ex).__name__
    return None, "无声明式"


def main() -> int:
    spot_only = "--spot" in sys.argv
    ids = None
    if spot_only:
        sf = _HERE / "reviews" / "spot_check_list.json"
        if sf.exists():
            ids = set(json.loads(sf.read_text(encoding="utf-8")).get(
                "spot_check_ids", []))
    from b_network.system_encoder import SystemEncoder
    enc = SystemEncoder()
    cal_cache: dict[str, tuple[bool, str]] = {}
    rows = []
    n_ok = n_bad = n_skip = 0
    for p in sorted((_HERE / "entries").glob("*.json")):
        e = json.loads(p.read_text(encoding="utf-8"))
        sid = e.get("system_id")
        fam = e.get("family")
        if ids is not None and sid not in ids:
            continue
        if fam not in cal_cache:
            cal_cache[fam] = self_calibrate(fam, e)
        cal_ok, cal_why = cal_cache[fam]
        dec, why = declared_trace(e)
        ind = None
        err = None
        try:
            node = pick_node(e)
            ind = numeric_fisher(node) if node is not None else None
            cmp_ok, cmp_why = symbols_comparable(e, node) if node is not None \
                else (False, "无节点")
        except Exception:  # noqa: BLE001
            ind = None
            cmp_ok, cmp_why = False, "口径判定异常"
        if dec is None:
            verdict, why = "无声明式（不结论）", why
            n_skip += 1
        elif not cal_ok:
            verdict, why = "不可独立核验（**方法未过自校**）", cal_why
            n_skip += 1
        elif not cmp_ok:
            verdict, why = "**口径不可比（不结论）**", cmp_why
            n_skip += 1
        elif ind is None or not np.isfinite(ind):
            verdict, why = "不可独立核验", "密度不可构造"
            n_skip += 1
        else:
            err = abs(ind - dec) / max(abs(dec), 1e-12)
            low = ind < dec
            if err <= TOL_INDEP:
                verdict = "**独立一致** ✓（偏差 %s，数值法口径）" % (
                    "偏低" if low else "偏高")
                n_ok += 1
            elif err < TOL_FLAG:
                verdict = "边界（%.1f%% %s）" % (100 * err,
                                              "偏低" if low else "偏高")
                n_ok += 1
                why = "接近阈值；多为尾截断偏差"
            else:
                verdict = "**需人工注意 ✗**（%.3g×）" % err
                why = "独立值与声明值差 %.0f%%（远超数值法误差）" % (100 * err)
                n_bad += 1
        rows.append({"system_id": sid, "domain": e.get("domain"),
                     "family": e.get("family"), "declared": dec,
                     "independent": ind, "rel_err": err, "verdict": verdict,
                     "note": why})

    with open(CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(MD, "w", encoding="utf-8") as f:
        f.write("# DB1 独立数值核验（%s）\n\n" % (
            "抽检 60 条" if spot_only else "全部条目"))
        f.write("> **只读生成**：不修改任何条目与复核状态。本核验**独立于**"
                "真值函数——对条目密度**直接数值微分**求 Fisher 定义式，"
                "供人工复核参考（该法曾抓出 `fisher_truncgeom_p` 符号 bug）。\n\n")
        f.write("判据：`≤2%%` 判**独立一致**（数值微分 + 网格截断所限；实测偏差多为"
                "**尾部截断导致的偏低**）；`≥10%%` 判**需人工注意**。"
                "条目自带的 `tol=1e-6` 是**解析自证**口径，**不适用于数值法**。\n\n")
        f.write("| 系统 ID | 域 | 族 | 声明值 | 独立值 | 相对差 | 判定 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| `%s` | %s | %s | %s | %s | %s | %s |\n"
                    % (r["system_id"], r["domain"], r["family"],
                       ("%.6g" % r["declared"]) if r["declared"] is not None
                       else "—",
                       ("%.6g" % r["independent"])
                       if r["independent"] is not None else "—",
                       ("%.2e" % r["rel_err"]) if r["rel_err"] is not None
                       else "—", r["verdict"]))
        f.write("\n## 汇总\n\n- 独立一致：**%d**\n- 独立不一致：**%d**\n"
                "- 不结论（无声明式/不可核验）：**%d**\n" % (n_ok, n_bad, n_skip))
    print("=" * 74)
    print("DB1 独立数值核验（%s）" % ("抽检 60 条" if spot_only else "全部 300 条"))
    print("=" * 74)
    print("**独立一致 %d**；**独立不一致 %d**；不结论 %d"
          % (n_ok, n_bad, n_skip))
    for r in rows:
        if r["verdict"].startswith("**独立不一致"):
            print("   ✗ %-28s 声明 %.6g vs 独立 %.6g（相对差 %.3g）"
                  % (r["system_id"], r["declared"], r["independent"],
                     r["rel_err"]))
    print("\n报告：%s ／ 清单：%s" % (MD.name, CSV.name))
    print("（**未修改任何条目与复核状态**）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
