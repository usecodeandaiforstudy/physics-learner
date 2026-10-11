# -*- coding: utf-8 -*-
"""B 网络需求侧核心数据类（arch §2.3.9(a)）。

RequirementParser / RuleMatcher / Composer 三个模块只允许经
StructuredRequirement 与 MetricSpec 两个数据类通信（禁止旁路），
保证"系统内无未经编码的物理知识"在代码层面可审计。

两个数据类均带 strictness 与 encoder_version 字段（架构约束）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StructuredRequirement:
    """槽位化结构需求（arch §2.3.6）。"""

    requirement_text: str
    property_ids: list[str] = field(default_factory=list)      # 本体属性，可多条（组合需求）
    direction: str | None = None                               # bound_below / bound_above
    target_value: dict = field(default_factory=dict)           # 数值目标（键对齐规则 target_value_keys）
    rv_hint: str | None = None                                  # 随机变量提示
    constraints: list[str] = field(default_factory=list)
    ambiguity_flags: list[str] = field(default_factory=list)
    unmapped: bool = False                                      # 规则库未命中
    parser_notes: list[str] = field(default_factory=list)       # 解析器备注（未解析出的槽位等）

    def to_dict(self) -> dict:
        return {
            "property_id": self.property_ids[0] if len(self.property_ids) == 1 else None,
            "direction": self.direction,
            "target_value": self.target_value,
            "rv_hint": self.rv_hint,
            "constraints": self.constraints,
            "ambiguity_flags": self.ambiguity_flags,
        }


@dataclass
class MetricSpec:
    """规则合成的信息指标规格：q 向量（方向 + 符号表达 + 可选数值）。"""

    requirement_text: str
    rule_ids: list[str] = field(default_factory=list)
    q: dict = field(default_factory=dict)        # {分量: {direction, expr, numeric?, reading?, note?}}
    strictness: str | None = None                # 命中规则最弱档
    ambiguity_flags: list[str] = field(default_factory=list)
    unmapped: bool = False
    nearest_rule_suggestions: list[str] = field(default_factory=list)
    requires_db5_calibration: bool = False
    encoder_version: str = "b-v0.1"

    def summary(self) -> str:
        if self.unmapped:
            sug = "; ".join(self.nearest_rule_suggestions) or "（无）"
            return "UNMAPPED（转人工队列；最近邻建议：%s）" % sug
        parts = []
        for comp, spec in self.q.items():
            num = spec.get("numeric")
            parts.append("%s %s %s%s" % (
                comp, spec["direction"], spec["expr"],
                "（数值 ≈ %.4g）" % num if isinstance(num, (int, float)) else ""))
        return "规则[%s]｜%s" % ("+".join(self.rule_ids), "；".join(parts))
