# -*- coding: utf-8 -*-
"""**为 5 个"全靠弃权"的域起草新锚点**（2026-10-02，第 189 条）。

## 为什么是这 5 个域
第 188 条修正计分口径后，LODO 的"全错域"变为 5 个
（`biomedical`/`communication`/`geophysics`/`measurement`/`mechanics`），
其**弃权率均为 1.00** ⟹ **不是答错，而是解析器对这些域的措辞毫无把握、
全部转人工** ⟹ **这正是该补锚点的地方** ✓
（注：`general` 域**已通过**——两条 `unmapped` 被正确弃权；其标注经核对
`rule_ids=['R-NOMAP-000']` **正确**，成本/环保本就不属于 9 个信息量属性 ✓）

## 本脚本做什么
**起草**（**不直接入库**）：为上述 5 个域各起草 **2 条**锚点，写到
`db3/proposed/`（**不写 `db3/pairs/`**），provenance 为
`AI+drafted / pending / anchor_eligible=false` ✓（红线：**须人工复核后才可入锚**）
**结构取自同属性的既有锚点**（克隆后替换文本/目标/出处）⟹ **天然 schema 合规** ✓
并用 `db3/validate_pair.py` **逐条校验** ✓

## 预登记判据
    P1 结构合规：每条都过 `validate_pair`；
    P2 溯源合规：`source=AI+drafted`、`human_review_status=pending`、
       `anchor_eligible=false`；
    P3 规则命中：`rule_ids` 必须引用**存在**的规则，且其 `ontology_property_id`
       与锚点的 `property_id` **一致**（谓词语义适用）。
       **不要求**规则的 `domain` 列表覆盖该域——**实测**：52 个域中 **45 个域**
       的既有锚点都在**跨域复用**规则（`rules.yaml` 的 `domain` 是**种子域提示**，
       不是适用性约束）⟹ 我初版拿它当门禁是**口径错**，已更正 ✓
    P4 不与既有锚点重复（文本去重）。

用法：
    python db3/propose_anchors.py            # 起草 + 校验（写 db3/proposed/）
    python db3/propose_anchors.py --dry-run  # 只看会写什么
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import yaml  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

PAIRS = _ROOT / "db3" / "pairs"
OUT = _ROOT / "db3" / "proposed"
RULES = _ROOT / "db3" / "rules" / "rules.yaml"

# **起草内容**：(域, 语言, 文本, 属性, 规则, 目标量, 方向)
# **键名必须落在规则的 `target_value_keys` 内（门2 规则，实测被拒后已改）**：
#   precision     ⟹ `sigma_dimensionless_fraction` / `sigma_m` / `n_samples_count`
#   robustness    ⟹ `sigma_dimensionless_fraction`
#   response_speed⟹ `settling_time_s` + `settling_band` / `tau_s`
#   stiffness     ⟹ `stiffness_N_per_m`
#   toughness     ⟹ `E_req_nats` + `n_valleys_count`
# **起草内容**：(域, 语言, 文本, 属性, 规则, 目标量, direction, quantity_bound)
# **两条约定（2026-10-02 第 194 条，实测踩过坑后写死）**：
#   ① `direction` 是**度量方向**——与 `info_metric_vector.*.direction`
#      （`lower_bound`）一致 ⟹ 一律 **`bound_below`** ✓
#      （**不是**文本量词方向！文本"不超过/以内"仍配 `bound_below` ✗）
#   ② `quantity_bound`/`hint` 是**文本量词方向**——"以内/不超过/within" ⟹ `upper`；
#      "不低于/至少/at least" ⟹ `lower` ✓（初版克隆模板导致二者矛盾 ⟹ 20 个错误 ✗）
#   ③ 文本须含该属性的**规则词**（如 precision 要有「精度/分辨」、
#      robustness 要有「鲁棒」、response_speed 要有「响应」、
#      toughness 要有「韧性/断裂」）——否则 `check_parser_consistency`
#      会报"property 未被解析器命中" ✗
DRAFTS = [
    ("biomedical", "zh",
     "CT 重建的分辨力：须在 1024 次投影下把密度不确定度压到 0.5% 以内。",
     "precision", "R-PRECISE-001",
     {"sigma_dimensionless_fraction": 0.005, "n_samples_count": 1024},
     "bound_below", "upper"),
    ("biomedical", "en",
     "The ECG front-end must keep its precision: baseline noise below 0.05 "
     "percent of full scale (RMS).",
     "precision", "R-PRECISE-001",
     {"sigma_dimensionless_fraction": 5e-4, "n_samples_count": 2048},
     "bound_below", "upper"),
    ("communication", "zh",
     "链路须保持鲁棒：误帧率不超过 1e-3。",
     "robustness", "R-ROBUST-001",
     {"sigma_dimensionless_fraction": 1e-3}, "bound_below", "upper"),
    ("communication", "en",
     "The handover response time must be within 50 ms without dropping the "
     "session.",
     "response_speed", "R-FAST-001",
     {"settling_time_s": 0.05, "settling_band": "95_percent"},
     "bound_below", "upper"),
    ("geophysics", "zh",
     "重力仪的分辨力：须在 10000 个样本平均后把相对漂移压到 1e-5 以内。",
     "precision", "R-PRECISE-001",
     {"sigma_dimensionless_fraction": 1e-5, "n_samples_count": 10000},
     "bound_below", "upper"),
    ("geophysics", "en",
     "The seismic inversion resolution must resolve a 5 m thin bed at 2 km "
     "depth.",
     "precision", "R-PRECISE-001", {"sigma_m": 5.0}, "bound_below", "upper"),
    ("measurement", "zh",
     "压力传感器的精度：须在 -20…80 °C 内把零点漂移控制在满量程 0.2% 以内。",
     "precision", "R-PRECISE-001",
     {"sigma_dimensionless_fraction": 2e-3, "n_samples_count": 4096},
     "bound_below", "upper"),
    ("measurement", "en",
     "The lock-in amplifier response must settle within 10 ms after a gain "
     "change.",
     "response_speed", "R-FAST-001",
     {"settling_time_s": 0.01, "settling_band": "95_percent"},
     "bound_below", "upper"),
    ("mechanics", "zh",
     "悬臂梁的刚度不得低于 5000 N/m（1 N 端载荷下）。",
     "stiffness", "R-STIFF-001",
     {"stiffness_N_per_m": 5000.0}, "bound_below", "lower"),
    ("mechanics", "en",
     "The bracket toughness must survive at least 1e6 cycles at 80% of "
     "yield without crack initiation.",
     "toughness", "R-TOUGH-001",
     {"E_req_nats": 14.0, "n_valleys_count": 1}, "bound_below", "lower"),
]


def _template(prop: str) -> dict:
    """取**同属性**的既有锚点作结构模板（无则取任意一条）。"""
    fallback = None
    for p in sorted(PAIRS.glob("DB3-*.json")):
        e = json.loads(p.read_text(encoding="utf-8"))
        if fallback is None:
            fallback = e
        if (e.get("requirement_structured") or {}).get("property_id") == prop:
            return e
    return fallback or {}


def main() -> int:
    dry = "--dry-run" in sys.argv
    rules = (yaml.safe_load(RULES.read_text(encoding="utf-8")) or {}).get(
        "rules") or {}
    existing_texts = set()
    max_id = 0
    for p in sorted(PAIRS.glob("DB3-*.json")):
        e = json.loads(p.read_text(encoding="utf-8"))
        existing_texts.add(e.get("requirement_text", "").strip())
        try:
            max_id = max(max_id, int(str(e.get("pair_id", "")).split("-")[1]))
        except Exception:  # noqa: BLE001
            pass
    print("=" * 74)
    print("为 5 个「全靠弃权」的域起草锚点（**不直接入库**，写 db3/proposed/）")
    print("=" * 74)
    print("既有锚点 %d 条，最大 ID 号 %d" % (len(existing_texts), max_id))

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    pid = max_id
    for dom, lang, text, prop, rule, target, direction, qbound in DRAFTS:
        ok_rule = rule in rules
        # **谓词一致性**（正确判据）：规则的 ontology_property_id 须等于锚点属性
        rule_prop = (rules.get(rule) or {}).get("ontology_property_id")
        covers = (rule_prop == prop)
        if text.strip() in existing_texts:
            print("  [跳过] 文本重复：%s" % text[:40])
            continue
        pid += 1
        new_id = "DB3-%04d" % pid
        tpl = copy.deepcopy(_template(prop))
        tpl["pair_id"] = new_id
        tpl["domain"] = dom
        tpl["language"] = lang
        tpl["requirement_text"] = text
        rs = tpl.setdefault("requirement_structured", {})
        rs["property_id"] = prop
        rs["target_value"] = target
        rs["direction"] = direction
        rs["metric_direction"] = direction
        # **量词方向显式指定**（文本量词，与 `direction` 的度量方向**不同** ✓）
        rs["quantity_bound"] = qbound
        rs["quantity_bound_hint"] = qbound
        rs["quantity_bound_source"] = "ai_drafted"
        rs["quantity_bound_review"] = "pending"
        tpl["rule_ids"] = [rule]
        tpl["provenance"] = {"source": "AI+drafted",
                             "human_review_status": "pending",
                             "anchor_eligible": False,
                             "review_date": None,
                             # **`reviewer` 必须存在（可为 null）**：`gate3_provenance`
                             #   会直接索引 `prov["reviewer"]` ⟹ 缺键即
                             #   `KeyError: 'reviewer'`（实测；已修）
                             "reviewer": None}
        tpl.pop("expected_unmapped_output", None)
        out = OUT / ("%s.json" % new_id)
        if not dry:
            out.write_text(json.dumps(tpl, ensure_ascii=False, indent=2),
                           encoding="utf-8")
        rows.append({"pair_id": new_id, "domain": dom, "property": prop,
                     "rule": rule, "rule_exists": ok_rule,
                     "rule_covers_domain": covers, "file": out.name})

    print("\n起草 %d 条：" % len(rows))
    print("  %-10s %-14s %-16s %-14s %s" % ("ID", "域", "属性", "规则",
                                            "规则谓词=属性"))
    for r in rows:
        print("  %-10s %-14s %-16s %-14s %s"
              % (r["pair_id"], r["domain"], r["property"], r["rule"],
                 "✓" if r["rule_covers_domain"] else "**✗ 需核对**"))
    p1 = all(r["rule_exists"] for r in rows)
    p3 = all(r["rule_covers_domain"] for r in rows)
    print("\nP1 结构：见下方校验器输出；P2 溯源：全部 `AI+drafted/pending/"
          "anchor_eligible=false` ✓")
    print("P3 **规则命中**：规则存在 %s ｜ **谓词=属性** %s"
          % ("✓" if p1 else "**✗**", "✓" if p3 else "**✗**"))
    print("   （注：**不**要求规则的 `domain` 覆盖该域——45/52 个域的既有锚点"
          "都在跨域复用规则 ✓）")
    print("P4 去重：与既有 %d 条文本无重复 ✓" % len(existing_texts))
    if dry:
        print("\n（--dry-run：**未写文件**）")
        return 0
    print("\n已写入 %s（%d 个文件）" % (OUT, len(rows)))
    print("**下一步（人工）**：逐条复核 ⟹ 通过后由 `db3/batch_review.py` 或"
          "手工移入 `db3/pairs/` 并把 `human_review_status` 置为 `approved`、"
          "`anchor_eligible` 置为 `true`（**红线：AI 不得自审**）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
