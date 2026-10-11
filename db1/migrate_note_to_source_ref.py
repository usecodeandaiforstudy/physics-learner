# -*- coding: utf-8 -*-
"""**DB1 数据卫生迁移：`provenance.note` → 并入 `source_ref`**（第 168 条）。

## 背景
机器门实测：DB1 **300 条中 10 条不合格**，全部是 `landau_deep_a*`，错误为
`provenance 不允许的字段: note`。查 `db1/validate_entry.py`：
`_ALLOWED_PROVENANCE_KEYS` 与 `_ALLOWED_TOP_LEVEL` **都是闭集**（`note` 两边都
不在其中）⟹ 不能简单移到顶层。

## 处置（最小侵入、**信息不丢**、不动任何复核状态）
把 `provenance.note` 的文本**并入 `source_ref`**（合法自由文本字段），并删除
非法的 `note` 键。**不修改** `human_review_status` / `review_date` / `reviewers`
——**人工复核结论仍由人做**（红线）。

用法：
    python db1/migrate_note_to_source_ref.py --dry-run   # 只看会改什么
    python db1/migrate_note_to_source_ref.py             # 执行
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

SEP = "；注："


def main() -> int:
    dry = "--dry-run" in sys.argv
    entries = sorted((_HERE / "entries").glob("*.json"))
    changed, skipped = [], 0
    for p in entries:
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped += 1
            continue
        prov = e.get("provenance")
        if not isinstance(prov, dict) or "note" not in prov:
            continue
        note = str(prov.pop("note") or "").strip()
        src = str(e.get("source_ref") or "").strip()
        if note and note not in src:
            e["source_ref"] = (src + SEP + note) if src else note
        if not dry:
            p.write_text(json.dumps(e, ensure_ascii=False, indent=2),
                         encoding="utf-8")
        changed.append(p.stem)

    print("扫描条目：**%d**；含非法 `provenance.note` 的：**%d**"
          % (len(entries), len(changed)))
    for sid in changed:
        print("   · %s" % sid)
    if dry:
        print("\n（--dry-run：**未写文件**）")
    else:
        print("\n已把说明并入 `source_ref` 并删除非法键；"
              "**未改动任何复核状态** ✓")
    if skipped:
        print("（跳过无法解析的条目 %d 个）" % skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
