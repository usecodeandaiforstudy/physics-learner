# -*- coding: utf-8 -*-
"""DB1 系统回滚脚本（第 51 条：rician 反噬回滚用）。

用法（项目根目录）：
    python db1/remove_systems.py <system_id> [<system_id> ...]

动作（三处同步清理，保持 DB1/DB2/复核记录一致）：
  1) 删除 db1/entries/<system_id>.json；
  2) 从 db2/db2.jsonl 删除该系统的全部扫描行；
  3) 删除 db1/reviews/<system_id>.yaml（若存在）。

红线：只删显式指定的系统；打印每步计数；不触碰其他数据。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENTRIES = _PROJECT_ROOT / "db1" / "entries"
_REVIEWS = _PROJECT_ROOT / "db1" / "reviews"
_DB2 = _PROJECT_ROOT / "db2" / "db2.jsonl"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def main(argv: list[str] | None = None) -> int:
    ids = argv or sys.argv[1:]
    if not ids:
        print("用法：python db1/remove_systems.py <system_id> [...]")
        return 2

    # 1) 条目
    for sid in ids:
        f = _ENTRIES / f"{sid}.json"
        if f.exists():
            f.unlink()
            print(f"[entry] 已删除 {sid}.json")
        else:
            print(f"[entry] 不存在 {sid}.json（跳过）")

    # 2) DB2 行
    if _DB2.exists():
        lines = _DB2.read_text(encoding="utf-8").splitlines()
        keep, removed = [], 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                keep.append(line)
                continue
            if rec.get("system_id") in ids:
                removed += 1
            else:
                keep.append(line)
        _DB2.write_text("\n".join(keep) + ("\n" if keep else ""),
                        encoding="utf-8")
        print(f"[db2] 删除 {removed} 行；剩余 {len(keep)} 行")
    else:
        print("[db2] db2.jsonl 不存在（跳过）")

    # 3) 复核记录
    for sid in ids:
        r = _REVIEWS / f"{sid}.yaml"
        if r.exists():
            r.unlink()
            print(f"[review] 已删除 {sid}.yaml")
        else:
            print(f"[review] 不存在 {sid}.yaml（跳过）")

    print("回滚完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
