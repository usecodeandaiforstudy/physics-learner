# -*- coding: utf-8 -*-
"""quantity_bound 人工确认（B v1 口径统一，2026-10-02）。

流程（红线合规：AI 提案 → 人类确认）：
  1. `migrate_direction_semantics.py` 干跑给出提案
     （`quantity_bound_hint` + `quantity_bound_review=pending:propose_*`）；
  2. **人类**运行本脚本确认：把 hint 写入 `quantity_bound`，
     `quantity_bound_source="human_confirmed"`、`review="confirmed"`；
     个别条目可用 `--override DB3-0001=lower` 改写（人类判定优先）。

用法（项目根目录）：
  python db3/confirm_quantity_bounds.py                      # 干跑
  python db3/confirm_quantity_bounds.py --apply              # 确认全部提案
  python db3/confirm_quantity_bounds.py --apply --override DB3-0001=lower
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

PAIRS = Path(__file__).resolve().parent / "pairs"


def main() -> int:
    argv = sys.argv[1:]
    apply = "--apply" in argv
    overrides: dict[str, str] = {}
    if "--override" in argv:
        for spec in argv[argv.index("--override") + 1:]:
            if "=" in spec:
                pid, val = spec.split("=", 1)
                overrides[pid.strip()] = val.strip()

    n_conf = n_skip = 0
    for path in sorted(PAIRS.glob("DB3-*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        rs = rec["requirement_structured"]
        pid = rec["pair_id"]
        review = str(rs.get("quantity_bound_review", ""))
        if review.startswith("pending"):
            val = overrides.get(pid, rs.get("quantity_bound_hint"))
            if val is None:
                n_skip += 1
                continue
            print(f"  {pid}: 提案 {val}"
                  + ("（人类覆盖）" if pid in overrides else " → 确认"))
            if apply:
                rs["quantity_bound"] = val
                rs["quantity_bound_source"] = "human_confirmed"
                rs["quantity_bound_review"] = "confirmed"
                rec["requirement_structured"] = rs
                path.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            n_conf += 1
        elif review.startswith("REVIEW"):
            val = overrides.get(pid)
            if val:
                print(f"  {pid}: 人工判定 {val}（原提案 "
                      f"{rs.get('quantity_bound_hint')}）")
                if apply:
                    rs["quantity_bound"] = val
                    rs["quantity_bound_source"] = "human_confirmed"
                    rs["quantity_bound_review"] = "confirmed"
                    rec["requirement_structured"] = rs
                    path.write_text(json.dumps(rec, ensure_ascii=False,
                                               indent=2), encoding="utf-8")
                n_conf += 1
            else:
                n_skip += 1
                print(f"  {pid}: [REVIEW] 待人工判定（提案 "
                      f"{rs.get('quantity_bound_hint')}）；用 "
                      f"--override {pid}=upper/lower 确认")

    print(f"{'已确认' if apply else '（干跑）'} {n_conf} 条，"
          f"跳过（待人工判定/不适用）{n_skip} 条")
    if not apply:
        print("（干跑结束；加 --apply 写入）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
