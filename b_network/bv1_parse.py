# -*- coding: utf-8 -*-
"""B v1 · B-parse（神经解析器）+ B-map 冻结接口。

定位（HANDOVER §2.4、架构文档 §2.3.9(b)）：B v1 = 分裂架构——
  - **B-parse（本模块）**：只学语言可变性 → 输出槽位（属性集合/方向）+ abstain；
  - **B-map（冻结）**：RuleMatcher + Composer 规则引擎（零神经容量，规则变更
    即时生效），B v1 不改动。

设计诚实性（34 锚点的数据现实）：
  - 学习范围最小化：只学**属性集合**与**方向**两个槽位；数值/单位抽取沿用 v0
    正则（roundtrip 33/34 已证明其可靠）；rv_hint 用冻结查找表；
  - 特征 = v0 关键词命中（先验）+ 单位/数字 + 字符 3-gram/英文词（小词表，
    传导式，报告中注明）；
  - **abstain**：置信度（最大 sigmoid 概率）低于阈值时放弃模型预测、回退 v0
    规则槽位并记 note——「模型负责说得好的部分，说不好的交给规则/人工」。

接口与 v0 完全兼容：Bv1Parser.parse(text) → StructuredRequirement。
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import torch
import torch.nn as nn

from b_network.metric_spec import StructuredRequirement
from b_network.requirement_parser import (
    DIR_ABOVE,
    DIR_BELOW,
    PROPERTY_KEYWORDS,
    RequirementParser,
    SIGMA_CONTEXT_KW,
    UNIT_MAP,
)

from .bv1_dataset import PROPERTIES

__all__ = ["Bv1Model", "Bv1Parser", "build_vocab", "featurize"]

_DIRS = ["bound_below", "bound_above"]

# rv_hint 冻结查找（属性 → 随机变量提示；与锚点一致）
RV_LOOKUP = {
    "stiffness": "microstate_gibbs",
    "precision": "iid_samples",
    "response_speed": "markov_chain",
    "transport": "field_time_slices",
    "robustness": "joint_signal_noise_family",
    "stability": "landau_order_parameter",
    "adaptability": "environment_channel",
    "toughness": "fracture_path_landscape",
}


class Bv1Model(nn.Module):
    """B-parse 槽位分类器：属性集合（多标签 BCE）+ 方向（2 类 CE）。"""

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.prop_head = nn.Linear(hidden, len(PROPERTIES))   # 多标签 logits
        self.dir_head = nn.Linear(hidden, 2)                   # bound_below/above

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        return self.prop_head(h), self.dir_head(h)


def _char3(text: str) -> list[str]:
    t = text.lower()
    return [t[i:i + 3] for i in range(len(t) - 2) if not any(c in t[i:i + 3] for c in " \t\n")]


def _en_words(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text.lower())


def build_vocab(texts: list[str]) -> list[str]:
    """特征维度表：先验关键词/单位/数字 + 字符 3-gram/英文词（传导式，报告注明）。

    **注（第 193 条）**：`featurize()` 现在还会**拼接线索块**（见 `CUE_DIM`），
    因此**输入维度 = `input_dim(vocab)`**，**不是** `len(vocab)` ✗。
    """
    dims = ["n_numbers", "dir_below", "dir_above", "kw_soft_hard", "kw_sigma"]
    dims += ["kw_%s" % p for p in PROPERTIES if p != "unmapped"]
    dims += ["unit_%s" % u for u in UNIT_MAP]
    grams: set[str] = set()
    for t in texts:
        grams.update(_char3(t))
        grams.update(_en_words(t))
    dims += ["g_%s" % g for g in sorted(grams)]
    return dims


# ======================================================================
# **线索块（v2 特征，2026-10-02 第 191–193 条，方案 C 采纳）**
#   动因：v1 特征对**同义改写不敏感**——A/B 与留一候选（LOPO）实测：
#     加 1–2 条锚点只能"记住那几句话"（域内迁移 1/10）✗ ⟹ **瓶颈在特征层**。
#   做法：按 `db3/rules/rules.yaml` 的 **`physical_predicate`**（如 precision =
#     「可精密测量 / 可精密控制」）建立**双语通用属性线索词表**，统计
#     (命中数, 是否命中) 两个量 ⟹ 固定 **18 维**（9 属性 × 2），与词表**解耦**
#     （**归纳式**词表下依然可用 ✓）。
#   **依据**：规则谓词描述 + 标准物理词汇；**不是**域标签，也**不是**从测试
#     锚点抄来的 ✓
#   效果（严格 LODO 52 域，归纳式）：τ=0.50 时覆盖率 **84.9% → 87.4%**、
#     总体精确率 **81.1% → 84.9%**、选择性精确率 **95.6% → 97.1%** ✓
# ======================================================================
CUES: dict[str, list[str]] = {
    "stiffness": ["刚度", "挠度", "弹性模量", "硬度", "形变", "杨氏模量",
                  "stiffness", "deflection", "modulus", "rigid", "elastic"],
    "precision": ["精度", "分辨", "不确定度", "噪声", "误差", "漂移", "重复性",
                  "准确度", "precision", "resolution", "accuracy",
                  "uncertainty", "noise", "drift", "error", "repeatability"],
    "response_speed": ["响应", "建立时间", "上升时间", "延迟", "恢复", "弛豫",
                       "时间常数", "响应时间", "response", "settling",
                       "rise time", "latency", "recovery", "relaxation",
                       "time constant", "throughput"],
    "transport": ["导热", "热导率", "扩散", "输运", "传热", "热阻", "渗透",
                  "散热", "conductivity", "diffusion", "transport",
                  "thermal", "effusivity", "permeability", "heat"],
    "robustness": ["鲁棒", "抗扰", "抗噪", "容错", "丢包", "误码", "误帧",
                   "波动", "耐受", "robust", "immunity", "fault",
                   "packet loss", "bit error", "jitter", "disturbance",
                   "outage", "variation"],
    "stability": ["稳定", "相变", "保持时间", "退化", "老化", "滞回", "蠕变",
                  "stability", "phase transition", "retention",
                  "degradation", "aging", "hysteresis", "creep"],
    "adaptability": ["适应", "多工况", "多环境", "可重构", "通用", "柔性",
                     "adapt", "reconfigur", "multi-environment", "versatile",
                     "transferab", "scalab"],
    "toughness": ["韧性", "断裂", "疲劳", "冲击", "循环", "裂纹", "脆",
                  "toughness", "fracture", "fatigue", "impact", "cycle",
                  "crack", "damage"],
    "unmapped": ["成本", "价格", "环保", "可回收", "可持续", "合规", "法规",
                 "cost", "price", "environment", "recycl", "sustainab",
                 "compliance", "regulat"],
}
CUE_PROPS = list(CUES)
CUE_DIM = len(CUE_PROPS) * 2


def cue_block(text: str) -> np.ndarray:
    """线索块：9 个属性各给 (命中数, 是否命中) 两个统计量。"""
    low = text.lower()
    out = np.zeros(CUE_DIM, dtype=np.float32)
    for i, p in enumerate(CUE_PROPS):
        n = sum(1 for w in CUES[p] if w in low or w in text)
        out[2 * i] = float(n)
        out[2 * i + 1] = 1.0 if n > 0 else 0.0
    return out


def input_dim(vocab: list[str]) -> int:
    """**模型输入维度**（= 词表维度 + 线索块维度）。

    **凡是要构造 `Bv1Model` 的地方都必须用它**，不得再用 `len(vocab)` ✗
    （第 193 条：忘记加 `CUE_DIM` 会在 `load_state_dict`/前向时报维度错）。
    """
    return max(len(vocab), 1) + CUE_DIM


def featurize(text: str, vocab: list[str]) -> np.ndarray:
    """**v2 特征**：v1 特征 ⊕ **线索块**（维度 = `input_dim(vocab)`）。

    （第 193 条方案 C：线索块见本节 `CUES`；**维度计算请一律用 `input_dim`** ✓）
    """
    v = {d: 0.0 for d in vocab}
    v["n_numbers"] = float(len(re.findall(r"\d+(?:\.\d+)?", text)))
    v["dir_below"] = float(any(k in text for k in DIR_BELOW))
    v["dir_above"] = float(any(k in text for k in DIR_ABOVE))
    v["kw_soft_hard"] = float("硬" in text)
    v["kw_sigma"] = float(any(k in text for k in SIGMA_CONTEXT_KW))
    for p, kws in PROPERTY_KEYWORDS.items():
        v["kw_%s" % p] = float(any(re.search(k, text) for k in kws))
    for u in UNIT_MAP:
        v["unit_%s" % u] = float(u in text)
    for g in _char3(text):
        key = "g_%s" % g
        if key in v:
            v[key] = 1.0
    for w in _en_words(text):
        key = "g_%s" % w
        if key in v:
            v[key] = 1.0
    base = np.asarray([v[d] for d in vocab], dtype=np.float32)
    return np.concatenate([base, cue_block(text)])


class Bv1Parser:
    """B-parse v1：模型槽位预测 + abstain + 冻结数值抽取/rv 查找。

    **abstain_action（2026-10-02，第 77 条决策）**：
      - `"v0_fallback"`（默认，兼容旧行为）：弃权时用 v0 规则槽位；
      - `"human_queue"`（**推荐生产配置**）：弃权时**不回退 v0**，而是标记
        转人工队列（property_ids=[]、unmapped=True、parser_notes 记原因）。
    依据：held-out 新表述集实测——v1 在 OOD 文本上置信度系统偏低但预测常对，
    「回退 v0」会把对的预测换成错的（混合 48.1% < 纯 v1 55.6%、方向
    35% < 85%）。

    **τ 默认 0.7 → 0.50（2026-10-02，第 193 条，方案 C）**：
      依据 = **风险–覆盖曲线**（严格 LODO，52 域，归纳式词表，7 个工作点）：
        | τ | 覆盖率 | 选择性精确率 | 总体精确率 |
        |---|---|---|---|
        | **0.50** | **87.4%** | **97.1%** | **84.9%** |
        | 0.70（旧默认） | 83.0% | 98.5% | 81.8% |
        | 0.90 | 76.7% | 100.0% | 76.7% |
      ⟹ τ=0.50 在**覆盖率 +4.4 pp** 的同时**总体精确率 +3.1 pp**，
      选择性精确率仅降 1.4 pp ⟹ **帕累托更优** ✓
      **注**：τ 是**生产权衡**（弃权 ⟹ 转人工），本值由**用户决策**选定 ✓
    """

    def __init__(self, model: Bv1Model, vocab: list[str],
                 tau_abstain: float = 0.50,
                 abstain_action: str = "v0_fallback",
                 temperature: float = 1.0,
                 platt: tuple[float, float] | None = None):
        self.model = model
        self.vocab = vocab
        self.tau_abstain = tau_abstain
        self.abstain_action = abstain_action
        # 温度标定（第 79 条实测：目标错配 → **不采纳**，默认 T=1 不生效）；
        # Platt 缩放（第 79 条采纳项）直接标定弃权统计量
        self.temperature = max(float(temperature), 1e-3)
        self.platt = platt
        self.v0 = RequirementParser()
        self.model.eval()

    @classmethod
    def load(cls, model_path: str, tau_abstain: float | None = None,
             abstain_action: str = "v0_fallback") -> "Bv1Parser":
        ckpt = torch.load(model_path, map_location="cpu")
        vocab = ckpt["vocab"]
        # **必须用 `input_dim`（含线索块）**，否则 `load_state_dict` 维度不符 ✗
        model = Bv1Model(in_dim=input_dim(vocab))
        model.load_state_dict(ckpt["model"])
        tau = tau_abstain if tau_abstain is not None else float(
            ckpt.get("tau", 0.50))
        pl = ckpt.get("platt")
        platt = (float(pl["a"]), float(pl["b"])) if pl else None
        return cls(model, vocab, tau, abstain_action,
                   float(ckpt.get("temperature", 1.0)), platt)

    def predict_slots(self, text: str) -> dict:
        """模型槽位预测（含置信度；不落盘）。"""
        x = torch.tensor(featurize(text, self.vocab)[None, :])
        with torch.no_grad():
            lp, ld = self.model(x)
        probs = torch.sigmoid(lp / self.temperature)[0].numpy()
        raw_conf = float(probs.max())
        conf = raw_conf
        if self.platt is not None:      # Platt：max_prob → P(槽位正确)
            a, b = self.platt
            c = min(max(raw_conf, 1e-6), 1 - 1e-6)
            conf = float(1.0 / (1.0 + np.exp(-(a * np.log(c / (1 - c)) + b))))
        preds = [PROPERTIES[i] for i in range(len(PROPERTIES)) if probs[i] > 0.5]
        if not preds:
            preds = [PROPERTIES[int(probs.argmax())]]
        direction = _DIRS[int(torch.softmax(ld, dim=-1)[0].argmax())]
        return {"properties": preds, "direction": direction, "conf": conf}

    def parse(self, text: str) -> StructuredRequirement:
        base = self.v0.parse(text)     # 数值/单位抽取 + 默认槽位（冻结复用）
        slots = self.predict_slots(text)
        if slots["conf"] >= self.tau_abstain:
            if "unmapped" in slots["properties"]:
                base.unmapped = True
                base.property_ids = []
                base.parser_notes.append(
                    "Bv1 预测 unmapped（conf=%.2f）→ R-NOMAP-000 流程" % slots["conf"])
            else:
                base.property_ids = slots["properties"]
                base.unmapped = False
                base.direction = slots["direction"]
                base.parser_notes.append(
                    "Bv1 槽位：%s / %s（conf=%.2f）"
                    % (slots["properties"], slots["direction"], slots["conf"]))
        else:
            if self.abstain_action == "human_queue":
                # 推荐生产配置：不回退 v0，转人工队列（保留数值抽取供人工核对）
                base.property_ids = []
                base.unmapped = True
                base.direction = None
                base.parser_notes.append(
                    "Bv1 弃权（conf=%.2f < %.2f）→ **转人工队列**"
                    "（human_queue；不回退 v0 规则）。模型建议槽位：%s / %s"
                    % (slots["conf"], self.tau_abstain,
                       slots["properties"], slots["direction"]))
                return base
            base.parser_notes.append(
                "Bv1 abstain（conf=%.2f < %.2f）→ 回退 v0 规则槽位"
                % (slots["conf"], self.tau_abstain))
        # rv_hint：冻结查找（仅单属性命中时）
        if not base.unmapped and len(base.property_ids) == 1:
            base.rv_hint = RV_LOOKUP.get(base.property_ids[0])
        return base


def save_parser(model: Bv1Model, vocab: list[str], tau: float, path: str) -> None:
    torch.save({"model": model.state_dict(), "vocab": vocab, "tau": tau}, path)
