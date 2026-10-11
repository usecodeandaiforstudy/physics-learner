# -*- coding: utf-8 -*-
"""DB3 种子对人工复核队列管理（镜像 db1/review_queue.py，全量复核版）。

与 DB1 的差异：DB3 是 B 网络训练锚点层，锚点质量要求 100%——
无抽检路径，34 对全部走双人复核（Reviewer A 需求/物理语义，Reviewer B 信息论）。

状态机（与 db3/db3.schema.json 枚举对齐）：
    pending → in_review（单人 accept）→ approved（双人 accept，anchor_eligible=true）
    任意角色 reject → rejected（anchor_eligible=false，退回修订）

用法：
    python db3/review_queue.py build
    python db3/review_queue.py status
    python db3/review_queue.py apply <pair_id> <reviewer> <accept|reject> [findings] --role A|B
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import date
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from validate_pair import gate1_schema, gate2_rules, gate3_provenance, load_registry  # noqa: E402

PAIRS_DIR = _HERE / "pairs"
REVIEWS_DIR = _HERE / "reviews"


def _machine_gate(rec: dict, rules: dict, rules_version: str) -> list[str]:
    errs: list[str] = []
    gate1_schema(rec, errs)
    gate2_rules(rec, rules, errs)
    gate3_provenance(rec, rules_version, errs)
    return errs


def build_review_record(rec: dict, rules: dict, rules_version: str) -> dict:
    """机器门预填 + 复核呈现素材 + 人工字段留空。"""
    prov = rec["provenance"]
    return {
        "review_record": {
            "pair_id": rec["pair_id"],
            "rule_ids": rec["rule_ids"],
            "domain": rec["domain"],
            "strictness": rec["strictness"],
            "requirement_text": rec["requirement_text"],
            "q_summary": {
                comp: "%s %s" % (e["direction"], e["expr"])
                for comp, e in rec["info_metric_vector"].items()
            } if rec["info_metric_vector"] else None,
            "worked_example_result": (rec.get("worked_example") or {}).get("result"),
            "rationale_first_ref": rec["metric_rationale"][0] if rec["metric_rationale"] else None,
            "ambiguity_flags": rec["requirement_structured"].get("ambiguity_flags", []),
            "requires_db5_calibration": rec["requires_db5_calibration"],
            "provenance_source": prov["source"],
            "human_review_status": prov["human_review_status"],
            "anchor_eligible": prov["anchor_eligible"],
            "machine_gate": {"errors": _machine_gate(rec, rules, rules_version)},
            "reviewer_a": {"name": None, "decision": None, "findings": None, "date": None},
            "reviewer_b": {"name": None, "decision": None, "findings": None, "date": None},
        }
    }


def build_queue() -> dict:
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    rules, rules_version = load_registry()
    pairs = []
    for path in sorted(glob.glob(str(PAIRS_DIR / "DB3-*.json"))):
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        pairs.append(rec)
        out = REVIEWS_DIR / ("%s.yaml" % rec["pair_id"])
        out.write_text(yaml.safe_dump(build_review_record(rec, rules, rules_version),
                                      allow_unicode=True, sort_keys=False), encoding="utf-8")
    n_gate_fail = sum(1 for r in pairs
                      if _machine_gate(r, rules, rules_version))
    return {"pairs": len(pairs), "review_records_written": len(pairs),
            "machine_gate_fail": n_gate_fail}


def apply_review(pair_id: str, reviewer: str, decision: str,
                 findings: str = "", role: str = "A") -> dict:
    if decision not in ("accept", "reject"):
        raise ValueError("decision 须为 accept/reject")
    if role not in ("A", "B"):
        raise ValueError("role 须为 A/B")
    pair_file = PAIRS_DIR / ("%s.json" % pair_id)
    if not pair_file.exists():
        raise FileNotFoundError(pair_file)
    rec = json.loads(pair_file.read_text(encoding="utf-8"))
    rec_file = REVIEWS_DIR / ("%s.yaml" % pair_id)
    if rec_file.exists():
        rrec = yaml.safe_load(rec_file.read_text(encoding="utf-8"))
    else:
        rules, rv = load_registry()
        rrec = build_review_record(rec, rules, rv)

    today = date.today().isoformat()
    key = "reviewer_%s" % role.lower()
    rrec["review_record"][key] = {
        "name": reviewer, "decision": decision,
        "findings": findings or "无", "date": today,
    }

    ra = rrec["review_record"]["reviewer_a"]
    rb = rrec["review_record"]["reviewer_b"]
    if ra["decision"] == "accept" and rb["decision"] == "accept":
        status = "approved"
    elif decision == "reject":
        status = "rejected"
    elif ra["decision"] == "accept" or rb["decision"] == "accept":
        status = "in_review"
    else:
        status = "pending"

    rrec["review_record"]["human_review_status"] = status
    rrec["review_record"]["anchor_eligible"] = status == "approved"
    rec_file.write_text(yaml.safe_dump(rrec, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")

    # 回写种子对 provenance
    prov = rec["provenance"]
    prov["human_review_status"] = status
    prov["anchor_eligible"] = status == "approved"
    prov["review_date"] = today if status != "pending" else None
    names = [x["name"] for x in (ra, rb) if x["name"]]
    prov["reviewer"] = "+".join(sorted(set(names))) if names else None
    pair_file.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"pair_id": pair_id, "role": role, "decision": decision,
            "human_review_status": status, "anchor_eligible": prov["anchor_eligible"]}


def status() -> dict:
    counts = {"pending": 0, "in_review": 0, "approved": 0, "rejected": 0}
    records = sorted(REVIEWS_DIR.glob("*.yaml")) if REVIEWS_DIR.exists() else []
    for rf in records:
        rec = yaml.safe_load(rf.read_text(encoding="utf-8"))
        st = rec["review_record"].get("human_review_status") or "pending"
        counts[st] = counts.get(st, 0) + 1
    return {"review_records": len(records), "status_counts": counts}


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    parser = argparse.ArgumentParser(description="DB3 种子对复核队列管理")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="构建复核队列（34 对全量）")
    sub.add_parser("status", help="复核进度")
    p_apply = sub.add_parser("apply", help="记录复核结论")
    p_apply.add_argument("pair_id")
    p_apply.add_argument("reviewer")
    p_apply.add_argument("decision", choices=["accept", "reject"])
    p_apply.add_argument("findings", nargs="?", default="")
    p_apply.add_argument("--role", default="A", choices=["A", "B"])
    args = parser.parse_args(argv)

    if args.cmd == "build":
        r = build_queue()
        print("复核记录已写：%d 对（机器门失败 %d）" % (r["review_records_written"],
                                                     r["machine_gate_fail"]))
        return 0
    if args.cmd == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "apply":
        print(json.dumps(apply_review(args.pair_id, args.reviewer, args.decision,
                                      args.findings, role=args.role),
                         ensure_ascii=False, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
