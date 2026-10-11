# -*- coding: utf-8 -*-
"""需求对去重（2026-10-02）。

背景：表驱动生成器以 `max(existing)+1` 为起始编号，重跑时会**把整张规格表
重新生成一遍**，导致批次 1 的已复核对出现副本（DB3-0082..0117 ≡
DB3-0046..0081）。本脚本按 `requirement_text` 去重，**保留最小编号**
（= 已复核原件），删除后出现的副本，并备份被删文件。

用法（项目根目录）：
  python db3/dedupe_pairs.py            # 干跑
  python db3/dedupe_pairs.py --apply    # 执行（自动备份）
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

PAIRS = Path(__file__).resolve().parent / "pairs"
BACKUP = Path(__file__).resolve().parent / "pairs_backup_dedupe"


def main() -> int:
    apply = "--apply" in sys.argv[1:]
    seen: dict[str, str] = {}
    dups: list[Path] = []
    for p in sorted(PAIRS.glob("DB3-*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        key = rec["requirement_text"].strip()
        if key in seen:
            dups.append(p)
        else:
            seen[key] = rec["pair_id"]
    print("=== 需求对去重%s：重复 %d 条（保留最小编号原件）==="
          % ("（执行）" if apply else "（干跑）", len(dups)))
    for p in dups:
        rec = json.loads(p.read_text(encoding="utf-8"))
        print("  %s  ≡ %s  %s" % (rec["pair_id"], seen[rec["requirement_text"]
                                                       .strip()],
                                  rec["requirement_text"][:30]))
    if apply and dups:
        BACKUP.mkdir(parents=True, exist_ok=True)
        for p in dups:
            shutil.copy2(p, BACKUP / p.name)
            p.unlink()
        print("  已删除 %d 条；备份于 %s" % (len(dups), BACKUP))
    elif not apply:
        print("  （干跑结束；加 --apply 执行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
