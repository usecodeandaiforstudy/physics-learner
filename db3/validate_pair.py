# -*- coding: utf-8 -*-
"""
DB3 种子对三道门校验器 + 配比报告
--------------------------------
门 1 schema：db3.schema.json（jsonschema）
门 2 规则一致性：rule_id 在注册表；property/target 键匹配；R-COND 必带歧义标志；
    UNMAPPED 禁止携带 q；R-TOUGH 必标 DB5 校准；桥接常数必带单位；strictness 取最弱
门 3 溯源：source=AI+drafted、status=pending、anchor_eligible=false；rationale 引用理论文档
配比报告：单领域 ≤40%；逐规则计数；语言配比；UNMAPPED 计数
用法：python validate_pair.py [pair.json ...]（缺省全量 db3/pairs/）
退出码：0 全部通过；1 有失败
"""
import glob
import json
import os
import re
import sys

import yaml

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(BASE, "db3.schema.json")
RULES_PATH = os.path.join(BASE, "rules", "rules.yaml")
PAIRS_DIR = os.path.join(BASE, "pairs")

DOMAIN_CAP = 0.40  # 单领域占比上限（arch §3.9 DB3 第 4 条）

# ---- schema 常量（与 db3.schema.json 保持同步；手写门，风格对齐 db1/validate_entry.py）----
TOP_REQUIRED = ["pair_id", "domain", "language", "requirement_text", "requirement_structured",
                "rule_ids", "info_metric_vector", "worked_example", "metric_rationale",
                "strictness", "provenance", "rules_version", "convention_version"]
TOP_ALLOWED = set(TOP_REQUIRED) | {"requires_db5_calibration", "expected_unmapped_output"}
DOMAINS = {"mechanics", "materials", "thermal", "control", "measurement",
           "electronics", "communication", "general", "plasma",
           "geophysics", "biomedical", "energy", "agriculture", "acoustics",
           "aerospace", "automotive", "civil", "chemical", "food", "textile",
           "optics", "robotics", "marine", "mining", "semiconductor",
           "sports",
           # 批次 2（2026-10-02）
           "nuclear", "meteorology", "hydrology", "forestry", "aquaculture",
           "pharmaceutical", "packaging", "coating", "railway", "traffic",
           "water_utility", "geotechnical", "metrology",
           # 批次 3（2026-10-02）
           "astro", "particle", "cryogenics", "vacuum", "printing", "glass",
           "cement", "pulp_paper", "hvac", "elevator", "fire_safety",
           "desalination", "smart_grid"}
PROPERTIES = {"stiffness", "precision", "response_speed", "transport", "robustness",
              "stability", "adaptability", "toughness", "unmapped"}
DIRECTIONS = {"bound_below", "bound_above"}
LANGS = {"zh", "en"}
STRICTNESSES = {"定理", "定理（马尔可夫系统）", "推导", "推导/类比", "类比", "猜想"}
Q_DIRECTIONS = {"lower_bound", "upper_bound", "increase", "decrease", "approx"}
WE_REQUIRED = ["setup", "given", "computation", "result", "reference_scale"]

STRICTNESS_ORDER = {
    "定理": 4, "定理（马尔可夫系统）": 3, "推导": 2, "推导/类比": 1, "类比": 0.5, "猜想": 0,
}

RATIONALE_REF_HINTS = (".md", "§")  # 至少一条要含文献引用


def load_registry():
    with open(RULES_PATH, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return doc["rules"], doc["rules_version"]


def gate1_schema(rec, errs):
    """手写结构门（与 db3.schema.json 同步维护，风格对齐 db1）。"""
    pid = rec.get("pair_id", "?")
    for k in TOP_REQUIRED:
        if k not in rec:
            errs.append("[门1 schema] %s: 缺必填字段 %s" % (pid, k))
    for k in rec:
        if k not in TOP_ALLOWED:
            errs.append("[门1 schema] %s: 未声明字段 %s" % (pid, k))
    if not re.match(r"^DB3-\d{4}$", str(rec.get("pair_id", ""))):
        errs.append("[门1 schema] %s: pair_id 格式应为 DB3-XXXX" % pid)
    if rec.get("domain") not in DOMAINS:
        errs.append("[门1 schema] %s: domain %r 不在枚举" % (pid, rec.get("domain")))
    if rec.get("language") not in LANGS:
        errs.append("[门1 schema] %s: language %r 不在枚举" % (pid, rec.get("language")))
    if not isinstance(rec.get("requirement_text"), str) or len(rec.get("requirement_text", "")) < 6:
        errs.append("[门1 schema] %s: requirement_text 过短" % pid)
    rs = rec.get("requirement_structured")
    if not isinstance(rs, dict):
        errs.append("[门1 schema] %s: requirement_structured 须为对象" % pid)
    else:
        for k in ["property_id", "direction", "target_value", "rv_hint", "constraints", "ambiguity_flags"]:
            if k not in rs:
                errs.append("[门1 schema] %s: requirement_structured 缺 %s" % (pid, k))
        if rs.get("property_id") not in PROPERTIES and rs.get("property_id") is not None:
            errs.append("[门1 schema] %s: property_id %r 不在枚举" % (pid, rs.get("property_id")))
        if rs.get("direction") not in DIRECTIONS and rs.get("direction") is not None:
            errs.append("[门1 schema] %s: direction %r 不在枚举" % (pid, rs.get("direction")))
        if not isinstance(rs.get("target_value"), dict):
            errs.append("[门1 schema] %s: target_value 须为对象" % pid)
        if not isinstance(rs.get("constraints"), list) or not isinstance(rs.get("ambiguity_flags"), list):
            errs.append("[门1 schema] %s: constraints/ambiguity_flags 须为列表" % pid)
    rids = rec.get("rule_ids")
    if not isinstance(rids, list) or not rids:
        errs.append("[门1 schema] %s: rule_ids 须为非空列表" % pid)
    else:
        for r in rids:
            if not re.match(r"^R-[A-Z]+-\d{3}$", str(r)):
                errs.append("[门1 schema] %s: rule_id %r 格式非法" % (pid, r))
    q = rec.get("info_metric_vector")
    if q is not None:
        if not isinstance(q, dict) or not q:
            errs.append("[门1 schema] %s: info_metric_vector 须为非空对象或 null" % pid)
        else:
            for k, v in q.items():
                if not isinstance(v, dict) or v.get("direction") not in Q_DIRECTIONS or not v.get("expr"):
                    errs.append("[门1 schema] %s: q 分量 %s 缺 direction/expr" % (pid, k))
    we = rec.get("worked_example")
    if we is not None:
        if not isinstance(we, dict):
            errs.append("[门1 schema] %s: worked_example 须为对象或 null" % pid)
        else:
            for k in WE_REQUIRED:
                if k not in we:
                    errs.append("[门1 schema] %s: worked_example 缺 %s" % (pid, k))
    if not isinstance(rec.get("metric_rationale"), list) or not rec.get("metric_rationale"):
        errs.append("[门1 schema] %s: metric_rationale 须为非空列表" % pid)
    if rec.get("strictness") is not None and rec.get("strictness") not in STRICTNESSES:
        errs.append("[门1 schema] %s: strictness %r 不在枚举" % (pid, rec.get("strictness")))
    prov = rec.get("provenance")
    if not isinstance(prov, dict) or "source" not in prov:
        errs.append("[门1 schema] %s: provenance 缺失" % pid)
    exp = rec.get("expected_unmapped_output")
    if exp is not None and (not isinstance(exp, dict) or exp.get("flag") != "UNMAPPED"
                           or exp.get("human_queue") is not True):
        errs.append("[门1 schema] %s: expected_unmapped_output 结构非法" % pid)


def gate2_rules(rec, rules, errs):
    pid = rec["pair_id"]
    rids = rec["rule_ids"]
    for r in rids:
        if r not in rules:
            errs.append("[门2 规则] %s: rule_id %s 不在注册表" % (pid, r))
            return
    is_unmapped = "R-NOMAP-000" in rids
    rs = rec["requirement_structured"]

    # --- UNMAPPED 红线 ---
    if is_unmapped:
        if rids != ["R-NOMAP-000"]:
            errs.append("[门2 规则] %s: R-NOMAP-000 只能单独出现" % pid)
        if rec["info_metric_vector"] is not None:
            errs.append("[门2 规则] %s: UNMAPPED 对禁止携带 q 向量" % pid)
        if rec["worked_example"] is not None:
            errs.append("[门2 规则] %s: UNMAPPED 对不得带数值算例" % pid)
        if rec["strictness"] is not None:
            errs.append("[门2 规则] %s: UNMAPPED 对 strictness 应为 null" % pid)
        exp = rec.get("expected_unmapped_output")
        if not exp or exp.get("flag") != "UNMAPPED" or exp.get("human_queue") is not True:
            errs.append("[门2 规则] %s: UNMAPPED 对须带 expected_unmapped_output(flag=UNMAPPED, human_queue=true)" % pid)
        return

    # --- 非 UNMAPPED ---
    if rec["info_metric_vector"] is None or not rec["info_metric_vector"]:
        errs.append("[门2 规则] %s: 非 UNMAPPED 对必须有非空 q 向量" % pid)
    if rec["worked_example"] is None:
        errs.append("[门2 规则] %s: 非 UNMAPPED 对必须有数值算例" % pid)

    # property_id / 组合规则
    if len(rids) == 1:
        expect_prop = rules[rids[0]].get("ontology_property_id")
        if rs["property_id"] != expect_prop:
            errs.append("[门2 规则] %s: property_id=%r 应为 %r（单规则命中）" % (pid, rs["property_id"], expect_prop))
    else:
        if rs["property_id"] is not None:
            errs.append("[门2 规则] %s: 多规则组合对 property_id 应为 null" % pid)

    # target_value 键量纲匹配（目标键须在命中规则的 target_value_keys 并集内）
    allowed_keys = set()
    for r in rids:
        allowed_keys.update(rules[r].get("target_value_keys", []))
    for k in rs["target_value"]:
        if k not in allowed_keys:
            errs.append("[门2 规则] %s: target_value 键 %r 不在命中规则 target_value_keys 内（量纲失配）" % (pid, k))

    # R-COND-001 歧义标志（含组合）
    if "R-COND-001" in rids and not rs["ambiguity_flags"]:
        errs.append("[门2 规则] %s: R-COND-001 对必须带非空 ambiguity_flags（禁止静默选择）" % pid)

    # R-TOUGH-001 DB5 校准标注
    if "R-TOUGH-001" in rids and rec.get("requires_db5_calibration") is not True:
        errs.append("[门2 规则] %s: R-TOUGH-001 对必须标 requires_db5_calibration=true" % pid)

    # strictness = 命中规则最弱
    strictnesses = [rules[r]["strictness"] for r in rids if rules[r].get("strictness")]
    if strictnesses:
        weakest = min(strictnesses, key=lambda s: STRICTNESS_ORDER[s])
        if rec["strictness"] != weakest:
            errs.append("[门2 规则] %s: strictness=%r 应为最弱命中 %r" % (pid, rec["strictness"], weakest))

    # 桥接常数必带单位（worked_example.given 中标 is_bridge_constant 的项）
    we = rec["worked_example"] or {}
    for k, v in (we.get("given") or {}).items():
        if isinstance(v, dict) and v.get("is_bridge_constant"):
            if not v.get("unit"):
                errs.append("[门2 规则] %s: 桥接常数 %s 缺单位" % (pid, k))


def gate3_provenance(rec, rules_version, errs):
    pid = rec["pair_id"]
    prov = rec["provenance"]
    if prov["source"] != "AI+drafted":
        errs.append("[门3 溯源] %s: 初始种子对 source 必须为 AI+drafted，得 %r" % (pid, prov["source"]))
    # 状态机一致性（复核流转合法化：pending → in_review → approved / rejected）
    status = prov["human_review_status"]
    if status == "pending":
        if prov["anchor_eligible"] is not False:
            errs.append("[门3 溯源] %s: pending 状态 anchor_eligible 必须为 false" % pid)
        if prov["review_date"] is not None or prov["reviewer"] is not None:
            errs.append("[门3 溯源] %s: 未复核不得填 review_date/reviewer" % pid)
    elif status == "in_review":
        if prov["anchor_eligible"] is not False:
            errs.append("[门3 溯源] %s: in_review（单角色通过）状态 anchor_eligible 必须为 false" % pid)
        if prov["review_date"] is None or prov["reviewer"] is None:
            errs.append("[门3 溯源] %s: in_review 状态必须已填 review_date/reviewer" % pid)
    elif status == "approved":
        if prov["anchor_eligible"] is not True:
            errs.append("[门3 溯源] %s: approved（双人通过）状态 anchor_eligible 必须为 true" % pid)
        if prov["review_date"] is None or prov["reviewer"] is None:
            errs.append("[门3 溯源] %s: approved 状态必须已填 review_date/reviewer" % pid)
    elif status == "rejected":
        if prov["anchor_eligible"] is not False:
            errs.append("[门3 溯源] %s: rejected 状态 anchor_eligible 必须为 false" % pid)
    else:
        errs.append("[门3 溯源] %s: human_review_status 非法值 %r" % (pid, status))
    if not rec["metric_rationale"]:
        errs.append("[门3 溯源] %s: metric_rationale 不得为空" % pid)
    elif not any(any(h in item for h in RATIONALE_REF_HINTS) for item in rec["metric_rationale"]):
        errs.append("[门3 溯源] %s: metric_rationale 必须引用理论文档章节" % pid)
    if rec["rules_version"] != rules_version:
        errs.append("[门3 溯源] %s: rules_version=%r 与注册表 %r 不一致" % (pid, rec["rules_version"], rules_version))


def ratio_report(records, rules):
    n = len(records)
    print("\n===== DB3 配比报告（%d 对） =====" % n)
    # 领域
    domains = {}
    for r in records:
        domains[r["domain"]] = domains.get(r["domain"], 0) + 1
    cap_n = int(DOMAIN_CAP * n)
    print("\n[领域分布]（红线：单领域 ≤ 40%% = %d 对）" % cap_n)
    for d, c in sorted(domains.items(), key=lambda x: -x[1]):
        flag = "  <-- 超限!" if c > DOMAIN_CAP * n else ""
        print("  %-14s %2d  (%4.1f%%)%s" % (d, c, 100.0 * c / n, flag))
    # 规则
    rule_counts = {}
    for r in records:
        for rid in r["rule_ids"]:
            rule_counts[rid] = rule_counts.get(rid, 0) + 1
    print("\n[逐规则命中计数]（含组合对）")
    for rid, c in sorted(rule_counts.items()):
        print("  %-14s %d" % (rid, c))
    # 语言
    langs = {}
    for r in records:
        langs[r["language"]] = langs.get(r["language"], 0) + 1
    print("\n[语言配比]（目标：中文 ~75% / 英文 ~25%）")
    for lg, c in sorted(langs.items()):
        print("  %s  %2d  (%4.1f%%)" % (lg, c, 100.0 * c / n))
    # 红线专项
    cond_pairs = [r for r in records if "R-COND-001" in r["rule_ids"]]
    cond_flagged = [r for r in cond_pairs if r["requirement_structured"]["ambiguity_flags"]]
    tough_pairs = [r for r in records if "R-TOUGH-001" in r["rule_ids"]]
    tough_cal = [r for r in tough_pairs if r.get("requires_db5_calibration")]
    unmapped = [r for r in records if "R-NOMAP-000" in r["rule_ids"]]
    unmapped_clean = [r for r in unmapped if r["info_metric_vector"] is None]
    print("\n[红线专项]")
    print("  R-COND-001 带歧义标志： %d/%d" % (len(cond_flagged), len(cond_pairs)))
    print("  R-TOUGH-001 带 DB5 校准标注： %d/%d" % (len(tough_cal), len(tough_pairs)))
    print("  UNMAPPED 无 q 向量： %d/%d" % (len(unmapped_clean), len(unmapped)))
    ok = (
        all(c <= DOMAIN_CAP * n for c in domains.values())
        and len(cond_flagged) == len(cond_pairs)
        and len(tough_cal) == len(tough_pairs)
        and len(unmapped_clean) == len(unmapped)
    )
    print("  配比结论： %s" % ("通过" if ok else "不通过"))
    return ok


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(PAIRS_DIR, "DB3-*.json")))
    if not paths:
        print("没有找到种子对文件")
        return 1
    rules, rules_version = load_registry()

    errs = []
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        records.append(rec)
        gate1_schema(rec, errs)
        gate2_rules(rec, rules, errs)
        gate3_provenance(rec, rules_version, errs)

    if errs:
        print("===== 校验失败（%d 项） =====" % len(errs))
        for e in errs:
            print(" ", e)
        return 1
    print("===== 校验通过：%d/%d =====" % (len(records), len(paths)))
    ratio_ok = ratio_report(records, rules)
    return 0 if ratio_ok else 1


if __name__ == "__main__":
    sys.exit(main())
