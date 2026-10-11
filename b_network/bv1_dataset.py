# -*- coding: utf-8 -*-
"""B v1 数据集：DB3 锚点（39 对，已双人复核）→ (文本, 槽位标签)。

标签：属性集合（多标签，9 类含 unmapped）、方向（bound_below/above）。
组合对（0028/0029/0030）的属性集合由 rule_ids 推导；UNMAPPED 对（0031–0034）
属性集合 = {"unmapped"}。

**方向标签的口径修正（2026-10-02，第 70 条语义碰撞的落地修复）**：
锚点的 `direction` 字段是**信息度量的界方向**（q 有下界 → bound_below，
34 条老锚点因此全是 bound_below），而神经解析器要学的是**物理量的界方向**
（文本说「不超过」就是 bound_above）。直接用 `direction` 做标签会让方向
头学到「一律 bound_below」的标签先验（旧评估 86.7% 的准确率即此假象）。
修正后的标签来源（按优先级）：
  1. `quantity_bound`（upper→bound_above / lower→bound_below）——新对已标；
  2. 文本的物理方向（由 v0 规则解析器的方向规则判定；该规则是关键词级、
     零训练，用作标签来源不构成对神经头的循环依赖）；
  3. 两者皆无 → None（不计入方向指标）。
"""

from __future__ import annotations

import glob
import json
import os

__all__ = ["PROPERTIES", "RULE_TO_PROPERTY", "load_anchors"]

PROPERTIES = ["stiffness", "precision", "response_speed", "transport",
              "robustness", "stability", "adaptability", "toughness",
              "unmapped"]

RULE_TO_PROPERTY = {
    "R-STIFF-001": "stiffness",
    "R-PRECISE-001": "precision",
    "R-FAST-001": "response_speed",
    "R-COND-001": "transport",
    "R-ROBUST-001": "robustness",
    "R-STABLE-001": "stability",
    "R-ADAPT-001": "adaptability",
    "R-TOUGH-001": "toughness",
}

BASE = os.path.dirname(os.path.abspath(__file__))
PAIRS_DIR = os.path.join(BASE, os.pardir, "db3", "pairs")


def _physical_direction(rec: dict, rs: dict) -> str | None:
    """物理量界方向（标签口径，见文件头修正说明）。"""
    qb = rs.get("quantity_bound")
    if qb == "upper":
        return "bound_above"
    if qb == "lower":
        return "bound_below"
    # 回退：v0 规则解析器的关键词方向（零训练依赖）
    try:
        import sys
        sys.path.insert(0, os.path.dirname(BASE))
        from b_network.requirement_parser import RequirementParser
        return RequirementParser().parse(rec["requirement_text"]).direction
    except Exception:  # noqa: BLE001
        return None


def load_anchors() -> list[dict]:
    """读全部锚点对，返回 [{pair_id, text, properties, direction, ...}]。

    direction = **物理量界方向**（标签口径修正后；见文件头）。
    """
    records = []
    for path in sorted(glob.glob(os.path.join(PAIRS_DIR, "DB3-*.json"))):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        rs = rec["requirement_structured"]
        if "R-NOMAP-000" in rec["rule_ids"]:
            props = ["unmapped"]
        elif rs.get("property_id") is not None:
            props = [rs["property_id"]]
        else:  # 组合对：由 rule_ids 推导
            props = [RULE_TO_PROPERTY[r] for r in rec["rule_ids"]
                     if r in RULE_TO_PROPERTY]
        records.append({
            "pair_id": rec["pair_id"],
            "domain": rec.get("domain"),
            "text": rec["requirement_text"],
            "properties": sorted(props),
            "direction": _physical_direction(rec, rs),
            "direction_metric": rs.get("direction"),   # 原字段（度量向）留档
            "quantity_bound": rs.get("quantity_bound"),
            "rule_ids": rec["rule_ids"],
            "target_value": rs.get("target_value") or {},
        })
    return records
