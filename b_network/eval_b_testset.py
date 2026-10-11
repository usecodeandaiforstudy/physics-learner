# -*- coding: utf-8 -*-
"""B 侧新表述测试集评估（2026-10-02）：v0 / v1 / 混合三配置对比。

动机（HANDOVER 第 76 条）：锚点上 v0 规则与 v1 神经**双方都过拟合**
（v0 规则按锚点设计、v1 全量训练），无法比较真实泛化。本脚本在
**held-out 新表述**（`db3/testset/b_paraphrase_testset.json`，27 条，
措辞刻意偏离锚点）上比较：
  - **v0**：关键词规则解析器；
  - **v1（无回退）**：神经槽位预测（属性 + 方向），不看置信度；
  - **混合**：conf ≥ τ 用 v1 槽位，否则回退 v0（τ 扫描）。

指标：属性集合精确率（集合相等）/ 方向准确率（仅对文本给出界者）/
unmapped 判定一致率 / 混合的覆盖率-精确率权衡。

用法（项目根目录）：
  python b_network/eval_b_testset.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from b_network.bv1_dataset import PROPERTIES
from b_network.bv1_parse import Bv1Parser
from b_network.requirement_parser import RequirementParser

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "db3" / "testset" / "b_paraphrase_testset.json"
MODEL = ROOT / "b_network" / "bv1_model.pt"


def score(preds: dict, exp: dict) -> dict:
    """单条评分：属性集合精确、方向、unmapped 一致。"""
    p = set(preds["properties"])
    e = set(exp["properties"])
    exact = (p == e)
    unmapped_ok = (("unmapped" in p) == ("unmapped" in e))
    dir_ok = None
    if exp["direction"] is not None and "unmapped" not in e:
        dir_ok = (preds["direction"] == exp["direction"])
    return {"exact": exact, "dir_ok": dir_ok, "unmapped_ok": unmapped_ok,
            "conf": preds.get("conf")}


def summarize(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    ex = sum(r["exact"] for r in rows)
    uv = [r["dir_ok"] for r in rows if r["dir_ok"] is not None]
    um = sum(r["unmapped_ok"] for r in rows)
    print(f"  {name:16s} 属性精确 {ex:2d}/{n}（{100 * ex / n:5.1f}%）  "
          f"方向 {sum(uv):2d}/{len(uv)}（{100 * sum(uv) / max(len(uv), 1):5.1f}%）  "
          f"unmapped {um}/{n}（{100 * um / n:5.1f}%）")
    return {"exact": ex / n, "dir": sum(uv) / max(len(uv), 1),
            "unmapped": um / n}


def main() -> int:
    doc = json.loads(TESTSET.read_text(encoding="utf-8"))
    items = doc["items"]
    v0 = RequirementParser()
    v1 = Bv1Parser.load(str(MODEL)) if MODEL.exists() else None
    if v1 is None:
        print("缺 v1 模型（先跑 b_network/bv1_train_eval.py）")
        return 1

    print(f"================ B 侧新表述测试集（{len(items)} 条，held-out）"
          f" ================")
    rows_v0, rows_v1, rows_hyb = [], [], []
    detail = []
    for it in items:
        exp = it["expected"]
        r0 = v0.parse(it["text"])
        s0 = score({"properties": r0.property_ids, "direction": r0.direction},
                   exp)
        slots = v1.predict_slots(it["text"])
        s1 = score(slots, exp)
        # 混合（τ=0.7）：高置信走 v1，否则回退 v0
        use_v1 = slots["conf"] >= 0.7
        sh = score(slots if use_v1 else
                   {"properties": r0.property_ids, "direction": r0.direction},
                   exp)
        sh["used_v1"] = use_v1
        rows_v0.append(s0)
        rows_v1.append(s1)
        rows_hyb.append(sh)
        detail.append((it["test_id"], it["text"][:28], s0, s1, sh,
                       slots["conf"]))

    summarize("v0 规则", rows_v0)
    summarize("v1 神经（无回退）", rows_v1)
    summarize("混合（τ=0.7）", rows_hyb)

    print("\n  τ 扫描（混合）：")
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        sel = [r for r in rows_v1 if r["conf"] >= tau]
        if not sel:
            continue
        print("    τ=%.1f  覆盖 %2d/%d（%3.0f%%）  v1 段属性精确 %.1f%%"
              % (tau, len(sel), len(rows_v1),
                 100 * len(sel) / len(rows_v1),
                 100 * sum(r["exact"] for r in sel) / len(sel)))

    # 推荐配置：神经槽位 + 弃权转人工（**不回退 v0**）
    print("\n  -- 推荐配置：v1 神经 + abstain→人工队列（不回退 v0）--")
    for tau in (0.5, 0.7, 0.9):
        auto = [r for r in rows_v1 if r["conf"] >= tau]
        if not auto:
            continue
        print("    τ=%.1f  全自动 %2d/%d（%3.0f%%）自动段属性精确 %.1f%%、"
              "方向 %.1f%%；其余 %d 条转人工"
              % (tau, len(auto), len(rows_v1),
                 100 * len(auto) / len(rows_v1),
                 100 * sum(r["exact"] for r in auto) / len(auto),
                 100 * sum(1 for r in auto if r["dir_ok"]) /
                 max(sum(1 for r in auto if r["dir_ok"] is not None), 1),
                 len(rows_v1) - len(auto)))

    # **共识配置**（2026-10-02 第 86 条新增）：v1 与 v0 槽位一致 → 全自动；
    # 不一致 → 转人工。动机：67 条难集上「混合（回退 v0）」优于任一单模型，
    # 说明两者互补；共识把「不一致」显式路由给人工，而不是猜一个。
    agree = agree_ok = agree_dir = agree_dir_n = 0
    for it in items:
        exp = it["expected"]
        r0 = v0.parse(it["text"])
        s1 = v1.predict_slots(it["text"])
        if set(r0.property_ids) != set(s1["properties"]):
            continue
        agree += 1
        agree_ok += (set(r0.property_ids) == set(exp["properties"]))
        if exp["direction"] is not None and "unmapped" not in exp["properties"]:
            agree_dir_n += 1
            agree_dir += (s1["direction"] == exp["direction"])
    print("\n  -- **共识配置**：v1 与 v0 槽位一致 → 全自动，不一致 → 转人工 --")
    if agree:
        print("    一致 %d/%d（%.0f%%）→ 自动段属性精确 **%.1f%%**、方向 %.1f%%；"
              "不一致 %d 条转人工"
              % (agree, len(items), 100 * agree / len(items),
                 100 * agree_ok / agree,
                 100 * agree_dir / max(agree_dir_n, 1),
                 len(items) - agree))
    else:
        print("    无一致样本（两模型完全分歧）")

    print("\n  -- **三级策略**：共识一致 ∪ (v1 置信 ≥ τ) → 全自动 --")
    for tau in (0.7, 0.9):
        auto = 0
        ok = 0
        dnum = dnum_ok = 0
        for it in items:
            exp = it["expected"]
            r0 = v0.parse(it["text"])
            s1 = v1.predict_slots(it["text"])
            same = (set(r0.property_ids) == set(s1["properties"]))
            if not (same or s1["conf"] >= tau):
                continue
            auto += 1
            props = (set(s1["properties"]) if s1["conf"] >= tau
                     else set(r0.property_ids))
            ok += (props == set(exp["properties"]))
            if exp["direction"] is not None and "unmapped" not in exp["properties"]:
                dnum += 1
                dnum_ok += (s1["direction"] == exp["direction"])
        if auto:
            print("    τ=%.1f  全自动 %2d/%d（%3.0f%%）自动段属性精确 **%.1f%%**、"
                  "方向 %.1f%%；其余 %d 条转人工"
                  % (tau, auto, len(items), 100 * auto / len(items),
                     100 * ok / auto, 100 * dnum_ok / max(dnum, 1),
                     len(items) - auto))

    print("\n  -- 逐条（v0 / v1 / 混合 的 exact 与 conf）--")
    for tid, txt, s0, s1, sh, conf in detail:
        flag = "" if (s0["exact"] and s1["exact"]) else "  <-- 至少一方错"
        print("    %s conf=%.2f v0=%s v1=%s 混合=%s(%s)  %s%s"
              % (tid, conf, s0["exact"], s1["exact"], sh["exact"],
                 "v1" if sh["used_v1"] else "v0", txt, flag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
