# -*- coding: utf-8 -*-
"""**T2 负结果的诊断（S1+S2）：容量族外可预测性的口径与特征**（2026-10-02，第 200 条）。

## 背景（第 187–198 条）
`a_network/v29_cross_domain_transfer.py` 的 **T2**：用**每族 2 个特征**
（`median(fisher_trace)`、`median(I_XT)`）+ **岭回归（线性）**，留一族预测
**族容量 `I_TY_max`**，判据"相对误差中位 ≤20%" ⟹ **实测 95.9% ⟹ FAIL** ✗

**读代码后的四条根因假设**：
  1. **口径**：容量跨族差若干数量级 ⟹ 原始容量 + 相对误差 + **线性**模型，
     可预测出负值 ⟹ >100% 误差近乎"必然" ✗
  2. **特征**：容量本质上由**信道类型/支撑/噪声结构**决定，而特征只有两个**中位数**
     ⟹ **不可辨识** ✗
  3. **样本**：n≈27 族、2 特征、LOO ⟹ 方差极大
  4. **模型**：线性岭回归对跨族非线性映射不合适（但 n 小 ⟹ 不能简单放大模型）

## 本条做什么（**只诊断，不改 v29 的判据与结果** ✓）
**协议完全不变**（留一族、岭回归 λ=1e-3）——**只变两件**：
  · **S1 口径**：目标改 **log(容量)**（预测后**指数还原**，仍以**原始相对误差**报告
    ⟹ 各臂**同一把尺子** ✓）
  · **S2 特征**：加**客观可复现的结构/形状特征**（**不手工贴域标签** ✗）：
    `fisher_trace` 剖面(min/max/median/IQR)、`I_XT` max/median、行数、参数个数、
    离散/连续、支撑是否有界、偏度、超额峰度、参数区间对数宽度 ✓

**五个臂**（全部以"原始相对误差中位"报告）：
    A `raw_base`（**复现 v29**，全局标准化 ✓）｜ B `raw_base_fold`（同 A，但**折内**标准化）
    C `raw_rich` ｜ D `log_base`（S1）｜ E **`log_rich`（S1+S2，主臂）**
    ＋ **零模型 `null`**（预测训练集容量的中位数/几何均值 ⟹ **没有它就不知道有无信号** ✓）

## 预登记判据（先写死，未看数据）
    **C0 复现**：|median(A) − **95.9%**| ≤ 3 pp ⟹ 本 harness 与 v29 一致 ✓
    **C1 S1 有效**：median(D) ≤ **0.5 ×** median(B)（至少减半 ✓）
    **C2 S2 有效**：median(E) ≤ **0.8 ×** median(D)（再降 ≥20% ✓）
    **C3 达既有判据**：median(E) ≤ **20%**（**v29 的官方判据，未改** ✓）
    **C4 数字健全**：全部相对误差有限且 ≥ 0 ✓
    **C5 胜零模型**：median(E) < median(null) ⟹ 确有迁移信号 ✓
    ⟹ **C0 必须过**（否则说明我的 harness 与既有结果不可比，结论作废 ✓）

用法：
    python a_network/v36_capacity_transfer.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

RUNS = _ROOT / "v36_capacity_transfer.csv"
V29_MED = 0.959          # v29 实测（复现基准）
C0_TOL = 0.03            # 复现容差 3 pp
LAM = 1e-3               # 与 v29 相同


# ---------------- 特征：客观结构/形状（无手工域标签） ----------------
_DISCRETE = {"poisson", "poisson_count", "geometric", "truncated_geometric",
             "negbin", "binomial", "chisq"}


def _family_row_stats(rows: dict, fam: str) -> dict:
    ft = np.asarray([r["fisher_trace"] for r in rows[fam]], float)
    ix = np.asarray([r["I_XT"] for r in rows[fam]], float)
    return {"ft_min": float(np.min(ft)), "ft_max": float(np.max(ft)),
            "ft_med": float(np.median(ft)),
            "ft_iqr": float(np.percentile(ft, 75) - np.percentile(ft, 25)),
            "ix_max": float(np.max(ix)), "ix_med": float(np.median(ix)),
            "n_rows": float(len(ft))}


def _shape_features(fam: str) -> tuple[dict, bool]:
    """由**该族的分布对象**算形状特征（客观可算 ✓）。失败则回退并如实计数 ✓"""
    out = {"is_discrete": 1.0 if fam in _DISCRETE else 0.0,
           "sup_lower": 0.0, "sup_upper": 0.0, "skew": 0.0, "exkurt": 0.0,
           "n_params": 0.0, "log_range": 0.0}
    ok = False
    try:
        from scipy import stats
        d = None
        for cand in (fam, fam.replace("_count", ""), fam.replace("2", "")):
            if hasattr(stats, cand):
                d = getattr(stats, cand)
                break
        if d is not None:
            lo, hi = d.support()
            out["sup_lower"] = 1.0 if np.isfinite(lo) else 0.0
            out["sup_upper"] = 1.0 if np.isfinite(hi) else 0.0
            sk, ku = d.stats(moments="sk")
            # **重尾族的偏度/峰度会是 inf 或 nan**（如 cauchy）⟹ 必须先做
            #   `nan_to_num` 再裁剪，否则 **NaN 会污染整个特征矩阵** ✗
            #   （实测：初版 E 臂全 NaN ⟹ C2–C5 全部失去判定力 ✗）
            out["skew"] = float(np.clip(np.nan_to_num(sk, nan=0.0,
                                                      posinf=50.0,
                                                      neginf=-50.0),
                                        -50, 50))
            out["exkurt"] = float(np.clip(np.nan_to_num(ku, nan=0.0,
                                                        posinf=1e4,
                                                        neginf=0.0),
                                          -1, 1e4))
            ok = True
    except Exception:  # noqa: BLE001
        pass
    # 参数个数与区间宽度：取自 registry（**同一真源** ✓）
    try:
        reg = json.loads((_ROOT / "sim" / "registry.json").read_text(
            encoding="utf-8"))
        prm = (reg.get(fam) or {}).get("params") or {}
        out["n_params"] = float(len(prm))
        wid = []
        for _k, v in prm.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                lo, hi = float(v[0]), float(v[1])
                if lo > 0 and hi > lo:
                    wid.append(math.log(hi / lo))
        out["log_range"] = float(np.mean(wid)) if wid else 0.0
    except Exception:  # noqa: BLE001
        pass
    return out, ok


def _ridge(Xtr, ytr, Xte):
    A = Xtr.T @ Xtr + LAM * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ ytr)
    return Xte @ w


def main() -> int:
    import contextlib
    import io
    from a_network.v29_cross_domain_transfer import load_all, t1_t2
    rows, cap = load_all()
    fams = [f for f in sorted(cap) if f in rows]
    print("=" * 74)
    print("T2 诊断（S1 口径 + S2 特征）：容量族外可预测性")
    print("=" * 74)
    print("族数 **%d**（有行且有容量）；协议与 v29 相同（留一族 + 岭回归 λ=%g）"
          % (len(fams), LAM))
    # **C0 的参照改为「现在跑一次 v29」**（2026-10-02 修）：
    #   初版把 v29 的**历史记录 95.9%** 当参照 ⟹ 实测**本 harness 与 v29 现跑
    #   完全一致（64.0%）**，而 95.9% 是 **DB2 补齐曲线之前**的旧数字 ✗
    #   ⟹ **数据版本漂移**：DB2 41.5% → 99.7% 后 T2 由 95.9% 变为 **64.0%** ✗
    #   ⟹ 活参照可**彻底消除"过期常量"这一类隐患** ✓
    with contextlib.redirect_stdout(io.StringIO()):
        _g, v29_med, _n = t1_t2(rows, cap)
    print("**C0 活参照**：现在跑 v29 的 T2 中位 = **%.1f%%**（历史记录为 95.9%%，"
          "系 **DB2 补齐前**的旧数字 ⟹ **数据版本漂移** ✗）" % (100 * v29_med))
    base_cols, rich_extra = None, None
    feats, ok_shape, n_fallback = {}, 0, 0
    for f in fams:
        st = _family_row_stats(rows, f)
        sh, ok = _shape_features(f)
        ok_shape += int(ok)
        n_fallback += int(not ok)
        feats[f] = {**st, **sh}
    base_cols = ["ft_med", "ix_med"]
    rich_extra = [c for c in feats[fams[0]] if c not in base_cols]
    print("形状特征可算的族：**%d/%d**（回退 %d 个 ⟹ 如实计数 ✓）"
          % (ok_shape, len(fams), n_fallback))
    print("S2 新增特征（%d）：%s" % (len(rich_extra), ", ".join(rich_extra)))

    y_raw = np.asarray([cap[f] for f in fams], float)
    y_log = np.log(np.maximum(y_raw, 1e-12))
    Xb = np.asarray([[feats[f][c] for c in base_cols] for f in fams], float)
    Xr = np.asarray([[feats[f][c] for c in base_cols + rich_extra]
                     for f in fams], float)

    def run(X, y, log_target, fold_std=True, std_all=False):
        """留一族；返回逐族相对误差（**始终在原始尺度上算** ✓）。"""
        rels = []
        for i in range(len(fams)):
            te = np.zeros(len(fams), bool)
            te[i] = True
            tr = ~te
            if std_all:
                mu, sd = X.mean(0), X.std(0) + 1e-12
            else:
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
            Xs = (X - mu) / sd
            Xtr = np.c_[np.ones(tr.sum()), Xs[tr]]
            Xte = np.r_[1.0, Xs[i]][None, :]
            yy = y[tr]
            pred = float(_ridge(Xtr, yy, Xte)[0])
            if log_target:
                pred = float(np.exp(pred))
            rels.append(abs(pred - y_raw[i]) / max(y_raw[i], 1e-9))
        return np.asarray(rels)

    def null(y, log_target):
        rels = []
        for i in range(len(fams)):
            tr = np.ones(len(fams), bool)
            tr[i] = False
            p = (float(np.exp(np.mean(y[tr]))) if log_target
                 else float(np.median(y[tr])))
            rels.append(abs(p - y_raw[i]) / max(y_raw[i], 1e-9))
        return np.asarray(rels)

    arms = {
        "A raw_base（复现 v29）": run(Xb, y_raw, False, std_all=True),
        "B raw_base_fold": run(Xb, y_raw, False),
        "C raw_rich": run(Xr, y_raw, False),
        "D log_base（S1）": run(Xb, y_log, True),
        "E log_rich（S1+S2）": run(Xr, y_log, True),
        "null 零模型": null(y_log, True),
    }
    print("\n%-26s %12s %12s" % ("臂", "相对误差中位", "相对误差均值"))
    med = {}
    for k, v in arms.items():
        med[k] = float(np.median(v))
        print("%-26s %11.1f%% %11.1f%%" % (k, 100 * med[k], 100 * np.mean(v)))
    print("\n逐族（相对误差）：")
    print("  %-20s %10s %10s %10s %10s" % ("族", "容量", "A", "D", "E"))
    for i, f in enumerate(fams):
        print("  %-20s %10.4f %9.1f%% %9.1f%% %9.1f%%"
              % (f, y_raw[i], 100 * arms["A raw_base（复现 v29）"][i],
                 100 * arms["D log_base（S1）"][i],
                 100 * arms["E log_rich（S1+S2）"][i]))

    a = med["A raw_base（复现 v29）"]
    b = med["B raw_base_fold"]
    d = med["D log_base（S1）"]
    e = med["E log_rich（S1+S2）"]
    nl = med["null 零模型"]
    c0 = abs(a - v29_med) <= C0_TOL
    c1 = d <= 0.5 * b
    c2 = e <= 0.8 * d
    c3 = e <= 0.20
    c4 = all(np.isfinite(v).all() and (v >= 0).all() for v in arms.values())
    c5 = e < nl
    print("\n**C0 复现 v29（活参照）**：本 harness A=%.1f%% vs v29 现跑=%.1f%%"
          "（容差 %.0f pp）⟹ %s"
          % (100 * a, 100 * v29_med, 100 * C0_TOL,
             "**通过** ✓" if c0 else "**未通过 ⟹ 结论作废** ✗"))
    print("**C1 S1 口径有效**（D ≤ 0.5×B）：%.1f%% vs %.1f%% ⟹ %s"
          % (100 * d, 100 * b, "**通过** ✓" if c1 else "**未通过**"))
    print("**C2 S2 特征有效**（E ≤ 0.8×D）：%.1f%% vs %.1f%% ⟹ %s"
          % (100 * e, 100 * d, "**通过** ✓" if c2 else "**未通过**"))
    print("**C3 达 v29 官方判据**（E ≤ 20%%）：%.1f%% ⟹ %s"
          % (100 * e, "**通过** ✓" if c3 else "**未通过**"))
    print("**C4 数字健全** ⟹ %s" % ("**通过** ✓" if c4 else "**未通过**"))
    print("**C5 胜零模型**：E=%.1f%% vs null=%.1f%% ⟹ %s"
          % (100 * e, 100 * nl, "**通过** ✓" if c5 else "**未通过**"))
    verdict = ("**T2 在修正口径与扩充特征后达标** ✓（口径已变更，**两条结果都须保留**）"
               if (c0 and c3) else
               "**T2 仍未达标**——如实记为负结果（**v29 的判据未被改动** ✓）")
    print("\n⟹ **结论**：%s" % verdict)
    print("**纪律**：本条**只诊断**——`v29` 的判据、结果与运行器状态**均未改动** ✓")

    # ================= **探索性（post-hoc）稳健性检验** =================
    # **明确标注**：以下检验是在**已看到主结果之后**做的 ⟹ **不是确认性检验** ✗
    #   目的只有一个：判断"C 臂 19.8%"是**真实改善**还是 **n=27 的偶然** ✓
    if "--robust" in sys.argv:
        print("\n" + "─" * 74)
        print("**探索性（post-hoc）稳健性检验**——不是确认性检验 ✗")
        print("─" * 74)
        globals()["LAM"] = LAM
        # (a) λ 扫描
        print("(a) λ 扫描（中位相对误差）：", end="")
        lam0 = LAM                     # **先存原值**（下面会覆盖全局 LAM ✗）
        for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            globals()["LAM"] = lam
            m = float(np.median(run(Xr, y_raw, False)))
            print(" λ=%-6g %.1f%% ｜" % (lam, 100 * m), end="")
        globals()["LAM"] = lam0        # **还原为原值** ✓
        print()
        # (b) bootstrap：对"逐族误差"重采样，给中位的 95% 区间
        r_c = run(Xr, y_raw, False)
        rng = np.random.default_rng(0)
        bs = np.median(rng.choice(r_c, size=(2000, len(r_c)), replace=True),
                       axis=1)
        print("(b) bootstrap（2000 次，逐族误差重采样）：C 臂中位 **%.1f%%**，"
              "95%% 区间 **[%.1f%%, %.1f%%]**"
              % (100 * float(np.median(r_c)), 100 * float(np.percentile(bs, 2.5)),
                 100 * float(np.percentile(bs, 97.5))))
        # (c) 特征块消融：只加"剖面块" / 只加"结构块"
        prof = ["ft_min", "ft_max", "ft_iqr", "ix_max", "n_rows"]
        stru = ["is_discrete", "sup_lower", "sup_upper", "skew", "exkurt",
                "n_params", "log_range"]
        Xp = np.asarray([[feats[f][c] for c in base_cols + prof] for f in fams],
                        float)
        Xs2 = np.asarray([[feats[f][c] for c in base_cols + stru] for f in fams],
                         float)
        print("(c) 特征块消融：仅剖面块 **%.1f%%** ｜ 仅结构块 **%.1f%%** ｜ "
              "全块 **%.1f%%**"
              % (100 * float(np.median(run(Xp, y_raw, False))),
                 100 * float(np.median(run(Xs2, y_raw, False))),
                 100 * float(np.median(r_c))))
        print("    ⟹ 读法：若 95% 区间**跨过 20%%** 或 λ 敏感 ⟹ 19.8%% **不稳健** ✗；"
              "若两块都有贡献 ⟹ 改善不是单点偶然 ✓")
    with open(RUNS, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "capacity"] + list(arms))
        for i, fam in enumerate(fams):
            w.writerow([fam, y_raw[i]] + [float(arms[k][i]) for k in arms])
    print("落盘：%s（%d 行）" % (RUNS.name, len(fams)))
    return 0 if c0 else 1


if __name__ == "__main__":
    sys.exit(main())
