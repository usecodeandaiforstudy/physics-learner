# -*- coding: utf-8 -*-
"""**T-a：把样本量从 27 族提到 1002 行，收紧 T2 的置信区间**（2026-10-02，第 201 条）。

## 为什么要这样做（承第 200 条）
第 200 条结论：T2 的主因**不是口径**，而是**特征信息损失**（剖面 vs 中位数：
64.0% → 17.8% ✓），**但 n=27 族 ⟹ bootstrap 95% 区间 [13.1%, 35.9%] 跨过 20%
⟹ 无法认证** ✗。要认证，只有两条路：**扩族**（受 27 族上限限制 ✗）或
**把分析单元降到行级**（✓ 本条的 T-a）。

## 本条设计（**新判据，不改 v29/v36** ✓）
    · **单元**：每族逐行（`v27.load_rows()`：**1002 行 / 27 族** ✓），
      行含 `theta, fisher_trace, I_XT, I_TY` ✓；
    · **目标**：该行所属族的**容量**（由**同一份行数据**取 `max I_TY` ⟹ 自洽 ✓，
      并**与 DB2 的容量对照**作一致性检查 ✓）；
    · **协议**：**按族分组留一**（训练=其它族的全部行，测试=留出族的全部行 ✓）
      ⟹ 仍是**跨族**检验 ✓，但每折的测试点从 1 个变成几十个 ✓；
    · **两臂**：
      · **P1 归纳式**：只用**该行自身**特征（`log θ`、`log fisher_trace`、
        `log I_XT`）⟹ 无留出族的任何聚合信息 ✓；
      · **P2 传导式（主臂）**：P1 ＋ **族级剖面**（`fisher_trace` 的
        min/max/IQR/中位、`I_XT` 的 max/中位、行数 ✓）；
    · **指标**：**行级**中位相对误差 ＋ **族级**中位（每族取中位再跨族取中位，
      与 v29/v36 **可比** ✓）＋ 对**族**做 bootstrap 的 95% 区间 ✓。

## 预登记判据（先写死）
    **D0 可行**：全部折完成训练与评估 ✓
    **D1 主判据（认证）**：**族级中位 ≤ 20%**（v29 官方判据，未改 ✓）
       **且 bootstrap 95% 区间上界 < 20%** ✓
    **D2 行级**：P2 的行级中位 ≤ 20% ✓
    **D3 归纳式**：P1 的族级中位 ≤ 30%（无聚合信息，判据放宽并预登记 ✓）
    **D4 胜零模型**：P2 族级中位 < 零模型（训练族容量的中位）✓
    **D5 数字健全** ✓
    ⟹ **D1 是关键**：它把第 200 条的"点估计达标但区间跨门槛"升级为
       **"认证达标"或"认证不达标"** ✓

用法：
    python a_network/v37_row_capacity_transfer.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

RUNS = _ROOT / "v37_row_capacity_transfer.csv"
LAM = 1e-3


def _ridge(Xtr, ytr, Xte, lam=LAM):
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    return Xte @ np.linalg.solve(A, Xtr.T @ ytr)


def main() -> int:
    from a_network.v27_variational_value_check import load_rows
    from a_network.v29_cross_domain_transfer import load_all
    rows = load_rows()
    _r2, cap_db2 = load_all()
    fams = sorted(rows)
    print("=" * 74)
    print("T-a：行级样本 + 按族分组留一（跨族认证）")
    print("=" * 74)
    n_rows = sum(len(rows[f]) for f in fams)
    # **目标必须与 v29/v36 同一口径：DB2 的族容量（IB 曲线平台）** ✓
    #   初版用 `load_rows()` 里**逐行 I_TY 的最大值**作目标 ⟹ 那是
    #   **固定 β 下沿 θ 扫描**的最大值，**不是**容量 ✗（实测 27/27 族与 DB2
    #   不符，如 beta 0.160 vs 0.338 ≈ 2×）⟹ **目标量用错，结果无效** ✗
    cap = {f: float(cap_db2[f]) for f in fams if f in cap_db2}
    miss = [f for f in fams if f not in cap_db2]
    row_max = {f: float(max(r["I_TY"] for r in rows[f])) for f in fams}
    ratio = [cap[f] / max(row_max[f], 1e-12) for f in cap]
    print("族 **%d**；行 **%d**；容量取自 **DB2（IB 平台，与 v29/v36 同口径 ✓）**；"
          "缺容量的族：%d %s" % (len(fams), n_rows, len(miss),
                                "✓" if not miss else miss))
    print("口径说明（如实）：`load_rows()` 的逐行 I_TY 是**固定 β 沿 θ 扫描**的"
          "取值 ⟹ 其最大值 **≠** 族容量；两者比值中位 **%.2f×** ✓" %
          float(np.median(ratio)))

    # ---------------- 特征 ----------------
    def row_feats(f, r):
        return [np.log(max(r["theta"], 1e-30)),
                np.log(max(r["fisher_trace"], 1e-30)),
                np.log(max(r["I_XT"], 1e-30))]

    prof = {}
    for f in fams:
        ft = np.asarray([r["fisher_trace"] for r in rows[f]], float)
        ix = np.asarray([r["I_XT"] for r in rows[f]], float)
        prof[f] = [np.log(max(ft.min(), 1e-30)), np.log(max(ft.max(), 1e-30)),
                   np.log(max(np.percentile(ft, 75) - np.percentile(ft, 25),
                              1e-30)),
                   np.log(max(np.median(ft), 1e-30)),
                   np.log(max(ix.max(), 1e-30)),
                   np.log(max(np.median(ix), 1e-30)),
                   float(len(ft))]

    # ---------------- 按族分组留一 ----------------
    def run_transductive(extra_family: bool):
        rel_rows, rel_fam, preds = [], {}, {}
        for f in fams:
            tr_f = [g for g in fams if g != f]
            Xtr = np.asarray([row_feats(g, r) + (prof[g] if extra_family
                                                 else [])
                              for g in tr_f for r in rows[g]], float)
            ytr = np.asarray([cap[g] for g in tr_f for _ in rows[g]], float)
            Xte = np.asarray([row_feats(f, r) + (prof[f] if extra_family
                                                 else [])
                              for r in rows[f]], float)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
            pred = _ridge(np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd], ytr,
                          np.c_[np.ones(len(Xte)), (Xte - mu) / sd])
            rel = np.abs(pred - cap[f]) / max(cap[f], 1e-9)
            rel_rows.extend(rel.tolist())
            rel_fam[f] = float(np.median(rel))
            preds[f] = float(np.median(pred))
        return np.asarray(rel_rows), rel_fam

    def run_null():
        rel_fam = {}
        for f in fams:
            tr = [g for g in fams if g != f]
            p = float(np.median([cap[g] for g in tr]))
            rel_fam[f] = abs(p - cap[f]) / max(cap[f], 1e-9)
        return rel_fam

    rel_rows_p1, rel_fam_p1 = run_transductive(False)
    rel_rows_p2, rel_fam_p2 = run_transductive(True)
    rel_fam_null = run_null()

    def fam_med(d):
        return float(np.median(list(d.values())))

    m_p1, m_p2, m_null = fam_med(rel_fam_p1), fam_med(rel_fam_p2), \
        fam_med(rel_fam_null)
    print("\n%-24s %14s %14s" % ("臂", "行级中位", "族级中位（可比）"))
    for name, rr, rf in (("P1 归纳式（仅行特征）", rel_rows_p1, rel_fam_p1),
                         ("**P2 传导式（行+族剖面）**", rel_rows_p2,
                          rel_fam_p2),
                         ("null 零模型", None, rel_fam_null)):
        print("%-24s %13s %13.1f%%"
              % (name, ("%.1f%%" % (100 * float(np.median(rr)))
                        if rr is not None else "—"), 100 * fam_med(rf)))

    # 对**族**做 bootstrap（族级统计量的正确重采样单元 ✓）
    vals = np.asarray([rel_fam_p2[f] for f in fams], float)
    rng = np.random.default_rng(0)
    bs = np.median(rng.choice(vals, size=(4000, len(vals)), replace=True),
                   axis=1)
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    print("\n**P2 族级 bootstrap（4000 次，重采样族）**：中位 %.1f%%，"
          "95%% 区间 **[%.1f%%, %.1f%%]**" % (100 * m_p2, 100 * lo, 100 * hi))

    d0 = all(np.isfinite([m_p1, m_p2, m_null]) ) and len(rel_fam_p2) == len(fams)
    d1 = (m_p2 <= 0.20) and (hi < 0.20)
    d2 = float(np.median(rel_rows_p2)) <= 0.20
    d3 = m_p1 <= 0.30
    d4 = m_p2 < m_null
    d5 = (np.isfinite(rel_rows_p2).all() and (rel_rows_p2 >= 0).all()
          and np.isfinite(vals).all())
    print("\n**D0 可行**：%d 折完成 ⟹ %s" % (len(fams),
                                          "**通过** ✓" if d0 else "**未通过**"))
    print("**D1 主判据（族级 ≤20%% 且区间上界 <20%%）**：%.1f%% ／ 上界 %.1f%% ⟹ %s"
          % (100 * m_p2, 100 * hi, "**认证达标** ✓" if d1 else "**未认证** ✗"))
    print("**D2 行级 ≤20%%**：%.1f%% ⟹ %s"
          % (100 * float(np.median(rel_rows_p2)),
             "**通过** ✓" if d2 else "**未通过**"))
    print("**D3 归纳式 ≤30%%**：%.1f%% ⟹ %s"
          % (100 * m_p1, "**通过** ✓" if d3 else "**未通过**"))
    print("**D4 胜零模型**：%.1f%% vs %.1f%% ⟹ %s"
          % (100 * m_p2, 100 * m_null, "**通过** ✓" if d4 else "**未通过**"))
    print("**D5 数字健全** ⟹ %s" % ("**通过** ✓" if d5 else "**未通过**"))
    print("\n**结论**：%s"
          % ("**T2 类型命题在行级样本下认证达标** ✓"
             if d1 else
             "**仍未认证达标** ✗——如实记录（v29/v36 判据均未改动 ✓）"))
    print("**口径说明**：本条是**新的检验**（单元=行、分组=族 ✓），"
          "与 v29 的族级单元**不可混用**；两套结果都保留在案 ✓")
    with open(RUNS, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "capacity", "rel_fam_P1", "rel_fam_P2",
                    "rel_fam_null"])
        for fam in fams:
            w.writerow([fam, cap[fam], rel_fam_p1[fam], rel_fam_p2[fam],
                        rel_fam_null[fam]])
    print("落盘：%s（%d 行）" % (RUNS.name, len(fams)))
    return 0 if d1 else 1


if __name__ == "__main__":
    sys.exit(main())
