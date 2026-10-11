# -*- coding: utf-8 -*-
"""DB3 round-trip 校验：RequirementParser+RuleMatcher 能否从种子对文本还原 rule_ids。

这是 B v0 需求侧三模块（parser/matcher/composer）的正确性证据：
对 db3/pairs/*.json 的 requirement_text 跑解析 → 匹配，
与种子对自带的 rule_ids 对比（组合对允许子集命中 + 报告差异）。
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from b_network.composer import Composer
from b_network.requirement_parser import RequirementParser
from b_network.rule_matcher import RuleMatcher

sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = RequirementParser()
    matcher = RuleMatcher()
    composer = Composer(matcher)

    paths = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "pairs", "DB3-*.json")))
    exact, partial, miss = [], [], []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        text = rec["requirement_text"]
        expect = set(rec["rule_ids"]) - {"R-NOMAP-000"}
        req = parser.parse(text)
        got_rules = set(matcher.match(req))
        spec = composer.compose(req)
        pid = rec["pair_id"]

        if not expect:  # UNMAPPED 对：期望解析器也拒绝映射
            if spec.unmapped:
                exact.append((pid, text, "UNMAPPED ✓（解析器同样拒绝硬映射）"))
            else:
                miss.append((pid, text, "期望 UNMAPPED，解析器却命中 %s" % spec.rule_ids))
            continue

        if got_rules == expect:
            exact.append((pid, text, "命中 %s" % sorted(expect)))
        elif got_rules and got_rules <= expect:
            partial.append((pid, text, "部分命中 %s ⊂ %s" % (sorted(got_rules), sorted(expect))))
        else:
            miss.append((pid, text, "期望 %s，实得 %s" % (sorted(expect), sorted(got_rules))))

    n = len(paths)
    print("===== DB3 round-trip（%d 对）=====" % n)
    print("完全命中：%d/%d（%.0f%%）" % (len(exact), n, 100.0 * len(exact) / n))
    print("部分命中：%d（组合需求抽到子集）" % len(partial))
    print("未命中/错判：%d" % len(miss))
    if partial:
        print("\n-- 部分命中明细 --")
        for pid, text, note in partial:
            print("  %s：%s\n      %s" % (pid, note, text[:60]))
    if miss:
        print("\n-- 未命中明细（B v0 词典的已知盲区）--")
        for pid, text, note in miss:
            print("  %s：%s\n      %s" % (pid, note, text[:60]))
    return 0 if not miss else 1


if __name__ == "__main__":
    sys.exit(main())
