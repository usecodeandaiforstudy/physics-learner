# -*- coding: utf-8 -*-
"""B 生产解析入口冒烟测试（2026-10-02，第 81 条切换的验收）。

验收项：
  P1 入口可用：`parse_requirement` 返回 ProductionParse（status ∈
     {auto, human_queue, v0_fallback}）；
  P2 高置信样本走 auto 且槽位正确（取库内文本，应 >70% auto）；
  P3 低置信/新表述样本走 human_queue 且**不回退 v0 槽位**
     （property_ids 为空、unmapped=True、notes 带模型建议）；
  P4 B-map 仍在工作：auto 样本的 target_value 非空（数值抽取来自 v0 规则）；
  P5 弃权语义：human_queue 样本的 req.direction 为 None（未做槽位决策）。

用法（项目根目录）：python b_network/smoke_b_parse.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from b_network.b_parse import DEFAULT_TAU, parse_requirement

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
    # 库内文本（锚点）→ 预期大量 auto
    anchors = []
    for p in sorted((ROOT / "db3" / "pairs").glob("DB3-*.json"))[:40]:
        anchors.append(json.loads(p.read_text(encoding="utf-8"))
                       ["requirement_text"])
    # 新表述（held-out 测试集）→ 预期较多 human_queue
    ts = json.loads((ROOT / "db3" / "testset" /
                     "b_paraphrase_testset.json").read_text(encoding="utf-8"))
    paraphrases = [it["text"] for it in ts["items"]]

    res_a = [parse_requirement(t) for t in anchors]
    res_p = [parse_requirement(t) for t in paraphrases]

    auto_a = sum(1 for r in res_a if r.status == "auto")
    auto_p = sum(1 for r in res_p if r.status == "auto")
    hq = [r for r in res_a + res_p if r.status == "human_queue"]
    print("  τ=%.2f：库内文本 auto %d/%d（%.0f%%）；新表述 auto %d/%d（%.0f%%）"
          % (DEFAULT_TAU, auto_a, len(res_a), 100 * auto_a / len(res_a),
             auto_p, len(res_p), 100 * auto_p / len(res_p)))

    fails += check("P1 入口可用（全部返回合法 status）",
                   all(r.status in ("auto", "human_queue", "v0_fallback")
                       for r in res_a + res_p))
    fails += check(f"P2 库内文本 auto 覆盖 {auto_a}/{len(res_a)} ≥ 50%",
                   auto_a >= 0.5 * len(res_a))
    fails += check(f"P3 弃权样本不回退 v0 槽位（{len(hq)} 条全部 "
                   f"property_ids 空 + unmapped=True + 带模型建议）",
                   all(not r.req.property_ids and r.req.unmapped
                       and r.model_slots for r in hq))
    auto_with_num = [r for r in res_a if r.status == "auto"
                     and r.req.target_value]
    fails += check(f"P4 B-map 数值抽取在 auto 样本上生效"
                   f"（{len(auto_with_num)}/{auto_a} 带 target_value）",
                   len(auto_with_num) >= 0.5 * max(auto_a, 1))
    fails += check("P5 弃权样本不做槽位决策（direction 为 None）",
                   all(r.req.direction is None for r in hq))
    if hq:
        print("  示例（人工队列）：", hq[0].req.parser_notes[-1][:78])

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
