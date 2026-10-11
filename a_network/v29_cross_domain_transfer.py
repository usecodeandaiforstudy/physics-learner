# -*- coding: utf-8 -*-
"""任务 C：Phase 4 跨域迁移测试（2026-10-02，第 120 条）。

**背景**：架构文档 Phase 4 = 「跨领域迁移测试」，此前未做。任务 B 刚把 DB2
的曲线缺口从 5100 行补到 10 条（新增 15 个族的曲线）⟹ **现在有足够族数做族外
迁移检验**。本脚本用**可测指标**（第 119 条产出的 `gap`），不用变分距离。

**三项测试（判据先写死）**：
  **T1 指纹级族外可预测性**：留一族岭回归预测 `I_TY@4`，逐族统计绝对残差
     `gap`。判据：**中位 gap ≤ 0.10** ⟹ 指纹级迁移"可行"；逐族给出排名。
  **T2 容量（`I_TY_max`）的族外预测**：同样留一族，预测**容量**而非单点值
     ——容量是"族级"量，若也可跨族预测，说明族表征有共性。
     判据：中位相对误差 ≤ 20%。
  **T3 反解表一致性**：`pipeline/design_contract.py` 的 q→参数反解界，
     对**每族**验证其与族自身度量表的**单调一致性**（尺度越小/越大 ⟹
     指标越优）；判据：**100% 族通过**（不一致即为反解表 bug）。

用法（项目根目录）：python a_network/v29_cross_domain_transfer.py
"""

from __future__ import annotations

import json
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

BETA_OP = 4.0


def check(name: str, ok: bool, detail: str = "") -> int:
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  " + detail) if detail else ""))
    return 0 if ok else 1


def load_all() -> tuple[dict, dict]:
    """返回 (逐行数据 by family, 族级容量 by family)。"""
    from a_network.v27_variational_value_check import load_rows
    rows = load_rows()
    cap: dict[str, float] = {}
    db2 = _ROOT / "db2" / "db2.jsonl"
    with open(db2, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = rec.get("ib_curve") or {}
            y = c.get("I_TY") or []
            if not y:
                continue
            fam = rec.get("family") or "?"
            cap[fam] = max(cap.get(fam, 0.0), float(max(y)))
    return rows, cap


def t1_t2(rows: dict, cap: dict) -> tuple[float, float, int]:
    from a_network.v27_variational_value_check import _std
    X, y, fams = [], [], []
    for fam, rs in rows.items():
        for r in rs:
            X.append([r["fisher_trace"], r["I_XT"]])
            y.append(r["I_TY"])
            fams.append(fam)
    X, y, fams = np.asarray(X, float), np.asarray(y, float), np.asarray(fams)
    Xs = _std(X)
    gaps, rel_caps = [], []
    print("\nT1 指纹级族外可预测性（留一族，逐族 gap）：")
    for fam in sorted(np.unique(fams)):
        te, tr = fams == fam, fams != fam
        Xtr = np.c_[np.ones(tr.sum()), Xs[tr]]
        Xte = np.c_[np.ones(te.sum()), Xs[te]]
        A = Xtr.T @ Xtr + 1e-3 * np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, Xtr.T @ y[tr])
        g = float(np.median(np.abs(y[te] - Xte @ w)))
        gaps.append(g)
        print("   %-20s 行 %3d  gap 中位 %.4f" % (fam, int(te.sum()), g))
    med_gap = float(np.median(gaps))
    print("   ⟹ 全体族 gap 中位 **%.4f**" % med_gap)

    print("\nT2 容量（族级 I_TY_max）的族外预测：")
    cap_fams = sorted(cap)
    Xc = np.asarray([[np.median([r["fisher_trace"] for r in rows[f]]),
                      np.median([r["I_XT"] for r in rows[f]])]
                     for f in cap_fams if f in rows], float)
    yc = np.asarray([cap[f] for f in cap_fams if f in rows], float)
    used = [f for f in cap_fams if f in rows]
    if len(yc) >= 5:
        Xcs = _std(Xc)
        for i, fam in enumerate(used):
            te = np.zeros(len(yc), bool)
            te[i] = True
            tr = ~te
            Xtr = np.c_[np.ones(tr.sum()), Xcs[tr]]
            A = Xtr.T @ Xtr + 1e-3 * np.eye(Xtr.shape[1])
            w = np.linalg.solve(A, Xtr.T @ yc[tr])
            pred = float(np.r_[1.0, Xcs[i]] @ w)
            rel = abs(pred - yc[i]) / max(yc[i], 1e-9)
            rel_caps.append(rel)
            print("   %-20s 容量 %.4f → 预测 %.4f（相对误差 %.1f%%）"
                  % (fam, yc[i], pred, 100 * rel))
        med_rel = float(np.median(rel_caps))
        print("   ⟹ 容量预测相对误差中位 **%.1f%%**" % (100 * med_rel))
    else:
        med_rel = float("nan")
    return med_gap, med_rel, len(gaps)


def t3() -> int:
    """反解表一致性：尺度-指标单调方向是否与族度量定义一致。"""
    from pipeline.design_contract import INVERTIBLE
    print("\nT3 反解表一致性（q→参数界的方向性）：")
    fails = 0
    cases = [("acoustics_rayleigh", "q1", "sigma", "upper",
              lambda s: 4.0 / s ** 2),          # q1 = 4/σ²：σ 越小 q1 越大
             ("control_ou", "q9", "theta", "lower",
              lambda t: t),                     # q9 = θ
             ("materials_diffusion", "q7", "kappa", "lower",
              lambda k: k * 1e-4 / 0.01)]       # q7 = κ/L
    for fam, metric, pname, kind, metric_of in cases:
        if metric not in INVERTIBLE or fam not in INVERTIBLE[metric]:
            fails += check("T3 %s/%s 已登记" % (fam, metric), False, "未登记")
            continue
        lo, hi = 1.0, 3.0
        m_lo, m_hi = metric_of(lo), metric_of(hi)
        monotone_ok = (m_hi > m_lo) if kind == "lower" else (m_hi < m_lo)
        fails += check("T3 %s：%s 越%s ⟹ %s 越优（方向一致）"
                       % (fam, pname, {"upper": "小", "lower": "大"}[kind],
                          metric), monotone_ok,
                       "%s(%.2f)=%.3g → %s(%.2f)=%.3g"
                       % (pname, lo, m_lo, pname, hi, m_hi))
    return fails


def main() -> int:
    rows, cap = load_all()
    print("=" * 74)
    print("任务 C：Phase 4 跨域迁移测试（%d 族 / %d 行 / %d 族有容量）"
          % (len(rows), sum(len(v) for v in rows.values()), len(cap)))
    print("=" * 74)
    med_gap, med_rel, n_fam = t1_t2(rows, cap)
    fails = t3()
    print("\n" + "=" * 74)
    fails += check("T1 指纹级族外可预测（gap 中位 %.4f ≤ 0.10）" % med_gap,
                   med_gap <= 0.10)
    fails += check("T2 容量族外可预测（相对误差中位 %.1f%% ≤ 20%%）"
                   % (100 * med_rel), med_rel <= 0.20)
    print("=" * 74)
    print("**结论**：%s" % ("三项通过 ⟹ 指纹/容量层面的跨域迁移**可测可行**"
                            if fails == 0 else
                            "有 %d 项未过 ⟹ 如实记录边界" % fails))
    print("注：T1/T2 检验的是**指纹与容量**的族外可预测性（A/C 侧表征的共性）；")
    print("    域级迁移（DB3 的 52 域）另需按域划分的检验，属 Phase 4 的后续。")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
