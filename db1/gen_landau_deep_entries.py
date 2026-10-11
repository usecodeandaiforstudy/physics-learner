# -*- coding: utf-8 -*-
"""生成 landau 深阱条目（2026-10-02，第 113 条，任务 ② 扩参数范围）。

**目的**：任务 3 的 barrier 类需求要 `ΔF = a²/(4b·kT)` 覆盖 **3–81 nats**，
而现有条目 `a ∈ [−2,2]`（b=kT=1）只能给 **ΔF ≤ 1 nat** ⟹ 需深阱条目。

**关键设计决定（把 ② 与 ① 接起来）**：每个新条目的 `a` **声明窄范围**
（默认值 ±10% 作**容差带**），于是
  - **设计点** = 该条目的 `a` 默认值 → 决定能力 `ΔF`；
  - **表示代价** = 以该容差带为先验的信道 `I_XT`（`(channel, lo, hi)` 各不同
    ⟹ 每个条目一条**独立**曲线）⟹ 「能力 vs 代价」的配对**不再退化**
    （这正是第 112 条原因 A 的解法，也就是①「逐设计代价」的第一版落实）。

**红线遵从**：这些条目为 **AI 生成**（programmatic template fill），
`provenance.source = "AI+verified"`、`human_review_status = "pending"`、
**不得作为锚点**；须走既有 `db1/review_queue.py` 人工复核流程。

用法（项目根目录）：
  python db1/gen_landau_deep_entries.py --dry-run   # 只看将写入什么
  python db1/gen_landau_deep_entries.py             # 写入 db1/entries/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ENTRIES = _ROOT / "db1" / "entries"

# (system_id, a_default, b, kT, 名称)
#   ΔF = a²/(4·b·kT)  —— 覆盖 3…81 nats（需求区间 3–80）
SPECS = [
    ("landau_deep_a4", -4.0, 1.0, 1.0, "深阱 a=4（ΔF=4 nats）"),
    ("landau_deep_a5", -5.0, 1.0, 1.0, "深阱 a=5（ΔF=6.25 nats）"),
    ("landau_deep_a6", -6.0, 1.0, 1.0, "深阱 a=6（ΔF=9 nats）"),
    ("landau_deep_a8", -8.0, 1.0, 1.0, "深阱 a=8（ΔF=16 nats）"),
    ("landau_deep_a10", -10.0, 1.0, 1.0, "深阱 a=10（ΔF=25 nats）"),
    ("landau_deep_a12", -12.0, 1.0, 1.0, "深阱 a=12（ΔF=36 nats）"),
    ("landau_deep_a15", -15.0, 1.0, 1.0, "深阱 a=15（ΔF=56.25 nats）"),
    ("landau_deep_a18", -18.0, 1.0, 1.0, "深阱 a=18（ΔF=81 nats）"),
    # 同 |a| 但不同 b/kT：验证 ΔF 由三参数共同决定（能力轴去相关）
    ("landau_soft_b2_a12", -12.0, 2.0, 1.0, "软四次项 b=2（ΔF=18 nats）"),
    ("landau_hot_kt2_a12", -12.0, 1.0, 2.0, "高温 kT=2（ΔF=18 nats）"),
]


def _entry(sid: str, a0: float, b0: float, kt0: float, name: str) -> dict:
    lo, hi = a0 * 0.9, a0 * 1.1        # **窄范围 = 该设计的容差带（±10%）**
    return {
        "system_id": sid,
        "domain": "thermodynamics",
        "name": name,
        "family": "landau",
        "equations": {
            "latex": "F = \\tfrac12 a \\phi^2 + \\tfrac14 b \\phi^4",
            "sympy": "F = a*phi**2/2 + b*phi**4/4",
        },
        "parameters": [
            {"name": "a", "symbol": "a", "range": [lo, hi], "default": a0},
            {"name": "b", "symbol": "b", "range": [0.1, 5.0], "default": b0},
            {"name": "kT", "symbol": "k_B T", "range": [0.1, 5.0],
             "default": kt0, "unit": "energy", "is_bridge_constant": True},
        ],
        "channel_template": "CT-LANDAU-001",
        "fingerprint_expectation": [],
        "strictness_level": "定理",
        "source_ref": "Landau 相变理论（序参量涨落）；深阱限用于高势垒/长保持需求",
        "scan_param": "a",
        "provenance": {
            "source": "AI+verified",
            "generator_model": "db1/gen_landau_deep_entries.py "
                               "(programmatic template fill)",
            "verified_by": "analytic_compare",
            "human_review_status": "pending",
            "review_date": None,
            "note": "**AI 生成；不得作锚点；须走 db1/review_queue.py 人工复核**",
        },
        "convention_version": "1.0.0",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print("将生成 %d 个 landau 深阱条目（ΔF = a²/(4b·kT)）：" % len(SPECS))
    print("%-22s %8s %6s %6s %10s" % ("system_id", "a", "b", "kT", "ΔF(nats)"))
    for sid, a0, b0, kt0, _n in SPECS:
        print("%-22s %8.2f %6.1f %6.1f %10.2f"
              % (sid, a0, b0, kt0, a0 ** 2 / (4 * b0 * kt0)))
    if args.dry_run:
        print("\n(dry-run：未写入；去掉 --dry-run 落盘)")
        return 0
    n_new = n_skip = 0
    for sid, a0, b0, kt0, name in SPECS:
        p = ENTRIES / ("%s.json" % sid)
        if p.exists():
            n_skip += 1
            continue
        p.write_text(json.dumps(_entry(sid, a0, b0, kt0, name),
                                ensure_ascii=False, indent=2),
                     encoding="utf-8")
        n_new += 1
    print("\n写入 %d 个新条目（跳过已存在 %d 个）" % (n_new, n_skip))
    print("下一步：python db2/rebuild_family_curves.py --family landau --write")
    print("      → python pipeline/extra_curves.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
