# -*- coding: utf-8 -*-
"""RuleMatcher：StructuredRequirement → 命中规则列表（arch §2.3.9(a)）。

从 db3/rules/rules.yaml 规则注册表按 ontology_property_id 匹配；
可多规则命中（组合需求）；目标键做量纲声明检查（warn 不阻断——
数值抽取是 v0 已知局限，硬拒收会误伤纯方向性需求）。
"""

from __future__ import annotations

import os

import yaml

from b_network.metric_spec import StructuredRequirement

__all__ = ["RuleMatcher"]

_RULES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "db3", "rules", "rules.yaml")


class RuleMatcher:
    """规则本体匹配器（B v0，确定性查表）。"""

    def __init__(self, rules_path: str = _RULES_PATH):
        with open(rules_path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        self.rules: dict = doc["rules"]
        self.rules_version: str = doc["rules_version"]
        # ontology_property_id → rule_id（一一对应；组合在 Composer 层合成）
        self._by_property = {r["ontology_property_id"]: rid for rid, r in self.rules.items()
                             if rid != "R-NOMAP-000"}

    def match(self, req: StructuredRequirement) -> list[str]:
        """返回命中的 rule_id 列表；未命中返回 []（→ R-NOMAP-000）。"""
        if req.unmapped or not req.property_ids:
            return []
        matched = []
        for prop in req.property_ids:
            rid = self._by_property.get(prop)
            if rid and rid not in matched:
                matched.append(rid)
        return matched

    def check_target_keys(self, req: StructuredRequirement, rule_ids: list[str]) -> list[str]:
        """量纲声明检查：target_value 键须在命中规则 target_value_keys 并集内。"""
        allowed = set()
        for rid in rule_ids:
            allowed.update(self.rules[rid].get("target_value_keys", []))
        return [k for k in req.target_value if k not in allowed]
