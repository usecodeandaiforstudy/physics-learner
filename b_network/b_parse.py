# -*- coding: utf-8 -*-
"""B 生产解析入口（2026-10-02，第 81 条决策落地）。

**生产路径**（B-parse / B-map 分裂架构的兑现）：
  - **B-parse（神经）**：`Bv1Parser.predict_slots` 输出属性集合 + 方向；
  - **B-map（冻结规则，零神经容量）**：v0 `RequirementParser` 负责**数值/
    单位抽取与 rv 查找**，并在模型缺失时兜底；
  - **abstain → 人工队列**：置信度低于 τ 时**不回退 v0 槽位**，标记
    `status="human_queue"`（附模型建议槽位供人工核对）。

切换依据（第 81 条）：LOOCV 70.4%（≥60% 门槛）、held-out 新表述
属性/方向 74.1%/80.0%（v0 为 59.3%/60.0%）；τ=0.8 时在库内文本上
58% 覆盖 @97.9% 精度。

用法：
    from b_network.b_parse import parse_requirement
    r = parse_requirement("电池 SOC 估计须在 300 点后把相对不确定度压到 1% 以内")
    r.status      # "auto" | "human_queue" | "v0_fallback"（模型缺失时）
    r.req         # StructuredRequirement（property_ids/direction/target_value）
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from b_network.requirement_parser import RequirementParser  # noqa: E402

__all__ = ["parse_requirement", "ProductionParse", "MODEL_PATH",
           "DEFAULT_TAU"]

MODEL_PATH = _ROOT / "b_network" / "bv1_model_cal.pt"   # 标定模型（无则退 bv1_model.pt）
MODEL_FALLBACK = _ROOT / "b_network" / "bv1_model.pt"
# τ 权衡（Platt 标定后 conf ≈ P(槽位正确)；实测随数据增长而移动，
# **每次重训后应重跑 bv1_calibrate.py 并复核此表**）：
#   159 锚点：τ=0.7 → 95.6% @85% | τ=0.9 → 96.9% @81% | τ=0.95 → 98.4% @77%
#   120 锚点：τ=0.7 → 98.9% @75% | τ=0.9 → **100% @72%** | τ=0.95 → 100% @65%
# 默认取 0.9（「P(正确)≥0.9 才全自动」——实测精度 96.9% ≥ 承诺值，姿态安全；
# 需要更高精度可提到 0.95，代价是覆盖 -4 点）。
DEFAULT_TAU = 0.9


@dataclass
class ProductionParse:
    req: object                 # StructuredRequirement
    status: str                 # auto / human_queue / v0_fallback
    conf: float                 # 模型置信度（v0_fallback 时为 0）
    model_slots: dict | None    # 模型建议槽位（人工队列时供核对）


_PARSER = None


def _get_parser(tau: float) -> tuple[object | None, str]:
    """惰性加载 Bv1Parser（**取两个工件中较新者**；标定文件过期则警告）。"""
    global _PARSER
    if _PARSER is not None:
        return _PARSER, "ok"
    cands = [p for p in (MODEL_PATH, MODEL_FALLBACK) if p.exists()]
    if not cands:
        return None, "缺模型文件（先跑 b_network/bv1_train_eval.py）"
    path = max(cands, key=lambda p: p.stat().st_mtime)
    if MODEL_PATH.exists() and MODEL_FALLBACK.exists() \
            and MODEL_PATH.stat().st_mtime < MODEL_FALLBACK.stat().st_mtime:
        print("[警告] 标定文件早于模型权重（重训后未重标定）→ 采用较新权重 "
              f"{path.name}；建议重跑 b_network/bv1_calibrate.py")
    try:
        from b_network.bv1_parse import Bv1Parser
        _PARSER = Bv1Parser.load(str(path), tau_abstain=tau,
                                 abstain_action="human_queue")
        return _PARSER, "ok"
    except Exception as ex:  # noqa: BLE001
        return None, "%s: %s" % (type(ex).__name__, ex)


def parse_requirement(text: str, tau: float = DEFAULT_TAU) -> ProductionParse:
    """生产解析：神经槽位（B-parse）+ 冻结数值抽取（B-map）+ 弃权转人工。"""
    parser, why = _get_parser(tau)
    if parser is None:
        base = RequirementParser().parse(text)
        base.parser_notes.append("B 生产入口降级为 v0 纯规则（%s）" % why)
        return ProductionParse(base, "v0_fallback", 0.0, None)

    base = parser.v0.parse(text)              # B-map：数值/单位抽取（冻结）
    slots = parser.predict_slots(text)
    if slots["conf"] < tau:
        base.property_ids = []
        base.unmapped = True
        base.direction = None
        base.parser_notes.append(
            "B-parse 弃权（conf=%.2f < %.2f）→ 人工队列；模型建议 %s / %s"
            % (slots["conf"], tau, slots["properties"], slots["direction"]))
        return ProductionParse(base, "human_queue", slots["conf"], slots)

    if "unmapped" in slots["properties"]:
        base.unmapped = True
        base.property_ids = []
        base.direction = None
        base.parser_notes.append(
            "B-parse 判为 unmapped（conf=%.2f）→ R-NOMAP-000 流程"
            % slots["conf"])
    else:
        base.property_ids = slots["properties"]
        base.unmapped = False
        base.direction = slots["direction"]
        base.parser_notes.append(
            "B-parse 槽位 %s / %s（conf=%.2f）"
            % (slots["properties"], slots["direction"], slots["conf"]))
    return ProductionParse(base, "auto", slots["conf"], slots)
