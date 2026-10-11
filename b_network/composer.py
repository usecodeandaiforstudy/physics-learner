# -*- coding: utf-8 -*-
"""Composer：命中规则 → MetricSpec（arch §2.3.4 encode_requirement 的规则线）。

职责：
1. 合并各规则 q_update 的方向/符号表达；
2. 数值代入（能算则算：R-PRECISE 1/(nσ²)、R-FAST 1/τ、R-ADAPT ln M、
   R-STABLE (Tc−T)/Tc 等天然无量纲量；物理单位量纲留符号形式——
   数值换算须桥接常数显式介入，属 DB5 验证职责，不在此处隐式折算）；
3. R-COND 读向过滤（按 ambiguity_flags 判读向 a/b，两读向方向相反）；
4. strictness 取最弱命中；R-TOUGH 标 requires_db5_calibration；
5. 未命中 → UNMAPPED + 最近邻规则建议（arch §2.3.3 R-NOMAP-000）。
"""

from __future__ import annotations

import math

from b_network.metric_spec import MetricSpec, StructuredRequirement
from b_network.rule_matcher import RuleMatcher

__all__ = ["Composer"]

STRICTNESS_ORDER = {
    "定理": 4, "定理（马尔可夫系统）": 3, "推导": 2, "推导/类比": 1, "类比": 0.5, "猜想": 0,
}


class Composer:
    """规则合成器（B v0，确定性）。"""

    def __init__(self, matcher: RuleMatcher):
        self.matcher = matcher
        self.rules = matcher.rules

    # ------------------------------------------------------------------
    def compose(self, req: StructuredRequirement) -> MetricSpec:
        rule_ids = self.matcher.match(req)
        if not rule_ids:
            return self._unmapped(req)
        spec = MetricSpec(
            requirement_text=req.requirement_text,
            rule_ids=rule_ids,
            ambiguity_flags=list(req.ambiguity_flags),
        )
        for rid in rule_ids:
            rule = self.rules[rid]
            self._merge_q(spec, req, rid, rule)
        # strictness = 最弱命中
        strictnesses = [self.rules[r]["strictness"] for r in rule_ids if self.rules[r].get("strictness")]
        spec.strictness = min(strictnesses, key=lambda s: STRICTNESS_ORDER[s])
        spec.requires_db5_calibration = "R-TOUGH-001" in rule_ids
        bad_keys = self.matcher.check_target_keys(req, rule_ids)
        if bad_keys:
            spec.ambiguity_flags.append("目标键量纲失配（%s）——须人工核对" % ", ".join(bad_keys))
        return spec

    # ------------------------------------------------------------------
    def _merge_q(self, spec: MetricSpec, req: StructuredRequirement, rid: str, rule: dict) -> None:
        reading_b = any("读向(b)" in f or "读向 b" in f for f in req.ambiguity_flags)
        for comp, qs in rule.get("q_update", {}).items():
            # R-COND：按读向只保留对应分量（两读向方向相反，禁止同时携带）
            if rid == "R-COND-001":
                is_b = qs.get("reading", "").startswith("b")
                if is_b != reading_b:
                    continue
            entry = {"direction": qs["direction"], "expr": qs["expr"]}
            if qs.get("reading"):
                entry["reading"] = qs["reading"]
            numeric = self._numeric(comp, req, rid)
            if numeric is not None:
                entry["numeric"] = numeric
            spec.q.setdefault(comp, entry)

    # ------------------------------------------------------------------
    def _numeric(self, comp: str, req: StructuredRequirement, rid: str):
        """天然无量纲量的数值代入；物理单位量返回 None（留符号形式）。"""
        tv = req.target_value
        try:
            if comp == "q1_fisher_directional" and rid == "R-PRECISE-001":
                n = tv.get("n_samples_count", 1)
                for key in ("sigma_K", "sigma_m", "sigma_dimensionless_fraction", "sigma_pH", "sigma_rad"):
                    if key in tv:
                        return 1.0 / (n * float(tv[key]) ** 2)
            if comp == "q1_fisher_directional" and rid == "R-ROBUST-001":
                for key in ("sigma_K", "sigma_m", "sigma_dimensionless_fraction", "sigma_pH", "sigma_rad"):
                    if key in tv:
                        return 1.0 / float(tv[key]) ** 2
            if comp == "q9_spectral_gap":
                if "tau_s" in tv:
                    return 1.0 / float(tv["tau_s"])
                if "settling_time_s" in tv:  # 2% 建立时间 ≈ 3.9τ
                    return 3.9 / float(tv["settling_time_s"])
            if comp == "q5_channel_capacity" and "n_environments_count" in tv:
                return math.log(int(tv["n_environments_count"]))
            if comp == "q8_stability_margin" and rid == "R-STABLE-001":
                if {"T_c_K", "T_work_K"} <= set(tv) and float(tv["T_c_K"]) != 0:
                    return (float(tv["T_c_K"]) - float(tv["T_work_K"])) / float(tv["T_c_K"])
                if "delta_dimensionless" in tv:
                    return float(tv["delta_dimensionless"])
                if "E_req_nats" in tv:
                    return float(tv["E_req_nats"])
            if comp == "q8_stability_margin" and rid == "R-TOUGH-001" and "E_req_nats" in tv:
                return float(tv["E_req_nats"])
            if comp == "q7_info_propagation_speed" and {"kappa_m2_per_s", "length_scale_m"} <= set(tv):
                return float(tv["kappa_m2_per_s"]) / float(tv["length_scale_m"])
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        return None

    # ------------------------------------------------------------------
    def _unmapped(self, req: StructuredRequirement) -> MetricSpec:
        suggestions = self._nearest_rules(req)
        return MetricSpec(
            requirement_text=req.requirement_text,
            rule_ids=["R-NOMAP-000"],
            unmapped=True,
            nearest_rule_suggestions=suggestions,
        )

    def _nearest_rules(self, req: StructuredRequirement, top_n: int = 3) -> list[str]:
        """最近邻规则建议（提示人工，禁止自动映射）。按谓词词面重叠粗排。"""
        text = req.requirement_text
        scored = []
        for rid, rule in self.rules.items():
            if rid == "R-NOMAP-000":
                continue
            pred = rule.get("physical_predicate", "")
            overlap = sum(1 for ch in pred if ch in text and not ch.isspace())
            scored.append((overlap, rid))
        scored.sort(reverse=True)
        return [rid for sc, rid in scored[:top_n] if sc > 0]
