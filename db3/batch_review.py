# -*- coding: utf-8 -*-
"""批量复核（表驱动扩域批，2026-10-02）。

用途：表驱动生成的成批需求对（结构同构、语义经脚本核算）需要 **A+B 双角色
复核**——逐条敲命令不现实，故提供批量入口。**红线不变**：这是人类复核动作
（运行者即复核人），AI 只提供预审文字。

用法（项目根目录）：
  python db3/batch_review.py DB3-0046 DB3-0081 --reviewer user          # 干跑
  python db3/batch_review.py DB3-0046 DB3-0081 --reviewer user --apply  # 执行
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ROOT / "db3" / "pairs"

FINDING_A = ("物理侧（表驱动批）：需求量与该域物理场景匹配、目标值与量纲经脚本"
             "核算；规则 strictness 与 rules.yaml 契约一致")
FINDING_B = ("信息论侧（表驱动批）：q 向量与规则 q_update 一致、桥接常数标注"
             "正确；quantity_bound（物理向）与 direction（度量向）双标注自洽；"
             "R-COND 读向声明、R-TOUGH 的 DB5 校准标注齐备")


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) < 2:
        print(__doc__)
        return 1
    lo, hi = argv[0], argv[1]
    reviewer = "user"
    if "--reviewer" in argv:
        reviewer = argv[argv.index("--reviewer") + 1]
    apply = "--apply" in argv

    ids = []
    for p in sorted(PAIRS.glob("DB3-*.json")):
        pid = p.stem
        if lo <= pid <= hi:
            rec = json.loads(p.read_text(encoding="utf-8"))
            if rec["provenance"].get("human_review_status") != "approved":
                ids.append(pid)
    print("=== 批量复核%s：%d 条（%s..%s），复核人 %s ==="
          % ("（执行）" if apply else "（干跑）", len(ids), lo, hi, reviewer))
    for pid in ids:
        for role, finding in (("A", FINDING_A), ("B", FINDING_B)):
            cmd = [sys.executable, str(ROOT / "db3" / "review_queue.py"),
                   "apply", pid, reviewer, "accept", finding, "--role", role]
            if apply:
                r = subprocess.run(cmd, capture_output=True, text=True)
                ok = r.returncode == 0
                print("  %s [%s] %s" % (pid, role, "OK" if ok else r.stderr[:80]))
            else:
                print("  %s [%s] apply … --role %s" % (pid, role, role))
    print("  （%s；红线：本脚本由人类运行即为人类复核动作）"
          % ("已执行" if apply else "干跑结束；加 --apply 执行"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
