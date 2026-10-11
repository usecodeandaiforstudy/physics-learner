# -*- coding: utf-8 -*-
"""**打印下一批待复核条目的证据要点**（只读，2026-10-02，第 186 条）。

把"复核时要看的东西"压缩成每条约 7 行的可读块：域/族、严格性、机器门、
方程、指纹声明、出处、独立核验结论，以及**可直接复制的 accept 命令**。
**不做任何判定**（红线：判定由人下）。

用法：
    python db1\next_batch.py            # 默认 5 条
    python db1\next_batch.py -n 8
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def main() -> int:
    n = 5
    if "-n" in sys.argv:
        n = int(sys.argv[sys.argv.index("-n") + 1])
    rev = "reviewer"
    if "--reviewer" in sys.argv:
        rev = sys.argv[sys.argv.index("--reviewer") + 1]
    from review_queue import brief, next_pending
    ids = [r["system_id"] for r in next_pending(n)]
    ind: dict[str, str] = {}
    csvp = _HERE.parent / "DB1_INDEPENDENT_CHECK.csv"
    if csvp.exists():
        with open(csvp, "r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                ind[r["system_id"]] = r.get("verdict", "")
    bs = {b["system_id"]: b for b in brief(ids, reviewer=rev)}
    for i, sid in enumerate(ids, 1):
        b = bs.get(sid)
        if not b:
            continue
        eq = (b.get("equations") or {}).get("sympy") or "—"
        fp = b["fingerprint"][0] if b["fingerprint"] else ("—", "—", "—")
        gate = "全过 ✓" if b["machine_gate"] == "全过 ✓" else str(
            b["machine_gate"])
        print("%d) %s ｜ %s" % (i, sid, b["name"]))
        print("   域/族：%s / %s ｜ 严格性：%s ｜ 机器门：%s"
              % (b["domain"], b["family"], b["strictness"], gate))
        print("   方程：%s" % eq[:100])
        print("   指纹：%s = %s" % (fp[0], fp[1][:90]))
        print("   出处：%s" % b["source_ref"])
        print("   独立核验：%s" % ind.get(sid, "（未跑）"))
        print("   记：%s" % b["apply_cmd"])
        print("   退：%s" % b["reject_cmd"])
        print()
    print("（只读输出；**未改动任何状态**。判定请用上面的 apply 命令）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
