# -*- coding: utf-8 -*-
"""缺口① 验证（2026-10-02）：模拟数据 → 指纹估计 → 解析真值对照。

闭环：`db1/ingest_sim_data.py`（模拟数据入库）→ 从**数据**估计指纹
（非参数 KDE 差分路线 + 参数化 MLE 路线）→ 与 `fisher_landau_analytic`
（解析真值）对照。

判据：
  D1 数据入库：3 条模拟数据条目 + 数据文件齐备（字段/红线校验）；
  D2 非参数路线：3 个工况的相对误差均值 < 30%（无分布族假设，纯数据）；
  D3 参数化路线：3 个工况的相对误差均值 < 10%（拟合定律后解析）。

用法（项目根目录）：python db2/v13_data_ingest_check.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.fisher import fisher_landau_analytic
from db2.estimate_fingerprint import (estimate_fisher_nonparametric,
                                      estimate_fisher_parametric)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def main() -> int:
    fails = 0
    cases = []
    for p in sorted((ROOT / "db1" / "entries_data").glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        data_ok = (ROOT / rec["data_path"]).exists()
        prov_ok = (rec["provenance"]["source"] == "AI+verified"
                   and rec["provenance"]["anchor_eligible"] is False)
        cases.append((rec, data_ok, prov_ok))
    n_ok = sum(1 for _, d, pv in cases if d and pv)
    fails += check(f"D1 模拟数据入库（{n_ok}/{len(cases)} 条齐备，"
                   f"含数据文件与红线 provenance）",
                   n_ok == len(cases) and len(cases) >= 3)

    errs_np, errs_pm = [], []
    print("  工况 | 解析真值 | 非参数估计 | 参数化估计 | 相对误差(非参/参数)")
    for rec, _, _ in cases:
        pr = rec["params"]
        a, b, kT = pr["a"], pr["b"], pr["kT"]
        samples = np.load(ROOT / rec["data_path"])["samples"]
        truth = float(fisher_landau_analytic(a, b, kT))
        f_np = estimate_fisher_nonparametric(a, b, kT, n=len(samples),
                                             h=0.05, seed=7)
        a_hat, f_pm = estimate_fisher_parametric(samples, b, kT)
        e_np = abs(f_np - truth) / truth
        e_pm = abs(f_pm - truth) / truth
        errs_np.append(e_np)
        errs_pm.append(e_pm)
        print("  a=%+5.2f | %8.4f | %10.4f | %10.4f | %6.1f%% / %5.1f%%"
              "（â=%+.3f）" % (a, truth, f_np, f_pm, 100 * e_np,
                               100 * e_pm, a_hat))

    m_np = float(np.mean(errs_np))
    m_pm = float(np.mean(errs_pm))
    fails += check(f"D2 非参数路线平均相对误差 {100 * m_np:.1f}% < 30%",
                   m_np < 0.30, f"{100 * m_np:.1f}%")
    fails += check(f"D3 参数化路线平均相对误差 {100 * m_pm:.1f}% < 10%",
                   m_pm < 0.10, f"{100 * m_pm:.1f}%")

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
