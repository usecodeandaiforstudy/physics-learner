# -*- coding: utf-8 -*-
"""**注册表测量 vs 节点真值的一致性扫描**（2026-10-02，第 206 条）。

## 它查什么
`verify_api` 对每个注册族给出的 `measured`，必须与**该族节点自身的真值**
（`compute_node_fisher(node)["fisher_trace"]`）一致 —— 这正是注册表
"校准通过"所声称的东西 ✓
⟹ 本脚本**逐族直接对照**，把"声称校准通过但实际不一致"的族**暴露出来** ✓

## 预登记判据（先写死）
    **R1 一致**：每族在**域中点**的相对误差 **≤ 10%**（= 注册表的登记门槛 ✓）
    **R2 健全**：全部数值有限且 > 0 ✓
    ⟹ **不符合 R1 的族逐条列出**（**不隐藏、不放宽** ✓）

## 背景（为什么会有这个脚本）
第 206 条核对 `landau` 口径时发现：**注册表测量是节点真值的 4.7–15.7 倍** ✗，
而注册表标"校准通过" ✗ ⟹ 遂扫描全部 27 族，查出 **3 族超限**
（`gaussian_mixture2` 1951.86%、`landau` 467.55%、`truncated_gaussian` 2404.91%），
**其余 24 族 0.00%–1.32% ✓**
⟹ 该阶段在运行器里登记为 **`已知待修`**（`stale=True`）：
   **缺陷可见、被追踪，但不伪装成通过** ✓

用法：
    python sim/registry_truth_scan.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

RUNS = _ROOT / "registry_truth_scan.csv"
R1_TOL = 0.10


def main() -> int:
    from core.fingerprint import compute_node_fisher
    from sim.expand_registry import measure_for_verify, node_of
    reg = json.loads((_ROOT / "sim" / "registry.json").read_text(
        encoding="utf-8"))
    print("=" * 74)
    print("注册表测量 vs 节点真值（逐族，域中点；判据 R1 相对误差 ≤%.0f%%）"
          % (100 * R1_TOL))
    print("=" * 74)
    rows, bad = [], []
    for f in sorted(reg):
        prm = reg[f]["params"]
        pname = next(iter(prm))
        lo, hi = prm[pname]
        mid = 0.5 * (lo + hi)
        try:
            node = node_of(f, mid)
            truth = float((compute_node_fisher(node) or {}).get(
                "fisher_trace") or 0.0)
            m = measure_for_verify(f, mid)
            m = float("nan") if m is None else float(m)
            err = abs(m - truth) / max(abs(truth), 1e-30)
        except Exception as ex:  # noqa: BLE001
            truth, m, err = float("nan"), float("nan"), float("nan")
            print("  %-20s 异常：%s: %s" % (f, type(ex).__name__,
                                          str(ex)[:40]))
        ok = (err == err) and err <= R1_TOL
        rows.append({"family": f, "param": pname, "mid": mid,
                     "truth": truth, "measured": m, "rel_err": err,
                     "ok": ok})
        if not ok:
            bad.append((f, err))
        print("  %-20s 真值 %12.5g ｜ 测量 %12.5g ｜ 误差 %9.2f%% %s"
              % (f, truth, m, 100 * err, "✓" if ok else "**✗ 超限**"))

    r1 = len(bad) == 0
    r2 = all(r["truth"] == r["truth"] and r["truth"] > 0
             and r["measured"] == r["measured"] for r in rows)
    print("\n**R1 逐族一致（≤%.0f%%）**：超限 **%d/%d 族** ⟹ %s"
          % (100 * R1_TOL, len(bad), len(rows),
             "**通过** ✓" if r1 else "**未通过** ✗"))
    if bad:
        for f, e in sorted(bad, key=lambda kv: -kv[1]):
            print("   · %-20s 误差 **%.2f%%**（测量/真值 ≈ %.1f×）"
                  % (f, 100 * e, (1 + e)))
        print("   ⟹ 这 3 族均为**自建分布**（`SPECIAL_DISTS` 系）✗ ⟹ "
              "其 `verify_api` 判定**不可直接采信**，须先修 ✓")
    print("**R2 数字健全** ⟹ %s" % ("**通过** ✓" if r2 else "**未通过**"))
    with open(RUNS, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n落盘：%s（%d 行）" % (RUNS.name, len(rows)))
    print("**语义**：本阶段登记为**已知待修**（`stale=True`）——"
          "缺陷可见、被追踪，**不伪装成通过** ✓")
    return 0 if r1 else 1


if __name__ == "__main__":
    sys.exit(main())
