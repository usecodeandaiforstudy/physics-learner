# -*- coding: utf-8 -*-
"""direction 语义迁移（B v1 口径统一，2026-10-02）——第 70 条立项①。

问题：`requirement_structured.direction` 在 34 条已复核锚点里指**信息度量 q
的界方向**（q 有下界 → bound_below），而解析器/物理语义要的是**物理量的界
方向**（文本「不超过」→ 上界）。同名字段两种语义 → 13 项历史分歧。

迁移（保守、可回溯、不改动已复核的既有取值）：
  - `metric_direction` = 原 `direction` 值（**语义保留**，含义显式化）；
  - `direction` 保留为兼容别名（值不变，标记 deprecated）；
  - `quantity_bound`：新对已有；老对由 **v0 关键词规则**推导
    （bound_above→"upper"，bound_below→"lower"），并标
    `quantity_bound_source="parser_derived"` + `quantity_bound_review="pending"`
    供人工核对；多约束文本（同时含上下界关键词）标 `ambiguous`。

用法（项目根目录）：
  python db3/migrate_direction_semantics.py            # 干跑（打印计划）
  python db3/migrate_direction_semantics.py --apply    # 写入（自动备份）
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ROOT / "db3" / "pairs"
BACKUP = ROOT / "db3" / "pairs_backup_pre_direction_migration"

_BOTH_ABOVE = ["不超过", "以内", "<=", "≤", "至多", "压回", "回稳到",
               "within", "at most", "up to", "no more than"]
_BOTH_BELOW = ["不低于", "不小于", "至少", ">=", "≥",
               "at least", "no less than", "more than"]

# 规则 → 目标量的期望界（由规则的 q_update 单调性推出，2026-10-02）
# 用于与解析器 hint 交叉核对：不符者 = 需人工判定的候选。
#   R-STIFF-001  q1 ∝ K → K 有下界        R-PRECISE-001 σ ≤ → 上界
#   R-FAST-001   q9 ∝ 1/t_settle → t 上界 R-COND-001    q7 ∝ κ/L → κ 下界
#   R-ROBUST-001 干扰 ≤ ε → 上界          R-STABLE-001 裕度 ≥ → 下界
#   R-ADAPT-001  M ≥ → 下界               R-TOUGH-001   E ≥ → 下界
RULE_EXPECTED_BOUND = {
    "R-STIFF-001": "lower",
    "R-PRECISE-001": "upper",
    "R-FAST-001": "upper",
    "R-COND-001": "lower",
    "R-ROBUST-001": "upper",
    "R-STABLE-001": "lower",
    "R-ADAPT-001": "lower",
    "R-TOUGH-001": "lower",
}


def _proposal_from_keys(rs: dict, rule_ids: list[str],
                        imv: dict | None) -> str | None:
    """键/规则语义推导（比文本关键词可靠，2026-10-02 干跑修正）。

    文本关键词只在**真的命中**时算证据；`resolve …` / `keep … at sigma_req`
    这类表述不含方向词，解析器的 lower 只是默认兜底。
    键语义：sigma_* 按定义是 σ 的上界；settling/tau 是时间上界；
    stiffness/kappa/E_req/环境数/裕度是下界。
    """
    tgt = rs.get("target_value") or {}
    if any(str(k).startswith("sigma_") for k in tgt):
        return "upper"
    if any(k in tgt for k in ("settling_time_s", "tau_s")):
        return "upper"
    if any(k in tgt for k in ("stiffness_N_per_m", "modulus_Pa",
                              "equivalent_modulus_Pa", "kappa_m2_per_s",
                              "E_req_nats", "delta_dimensionless",
                              "n_environments_count", "n_valleys_count")):
        return "lower"
    return _rule_expects(rule_ids, imv)


def _text_has_dir_kw(text: str) -> bool:
    """文本是否真的含方向关键词（否则解析器的方向是默认兜底，不算证据）。"""
    import re
    return any(re.search(k, text) for k in _BOTH_ABOVE + _BOTH_BELOW)


def _rule_expects(rule_ids: list[str],
                  info_metric_vector: dict | None = None) -> str | None:
    """规则/读法 → 目标量期望界。

    R-COND-001 有两种读法且界方向相反（干跑发现，2026-10-02）：
      - reading a_heat_front_fast：q7 = κ/L 有下界 → κ 下界；
      - reading b_fine_structure_fidelity：q6 = ℓ_req 有上界 → ℓ 上界。
    """
    for entry in (info_metric_vector or {}).values():
        if entry.get("reading") == "b_fine_structure_fidelity":
            return "upper"
        if entry.get("reading") == "a_heat_front_fast":
            return "lower"
    exp = {RULE_EXPECTED_BOUND[r] for r in rule_ids
           if r in RULE_EXPECTED_BOUND}
    return exp.pop() if len(exp) == 1 else None      # 组合/无规则 → 无期望


def _ambiguous(text: str) -> bool:
    return (any(k in text for k in _BOTH_ABOVE)
            and any(k in text for k in _BOTH_BELOW))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    apply = "--apply" in argv
    from b_network.requirement_parser import RequirementParser
    parser = RequirementParser()

    plan, n_new = [], 0
    for path in sorted(PAIRS.glob("DB3-*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        rs = rec["requirement_structured"]
        old_dir = rs.get("direction")
        qb = rs.get("quantity_bound")
        is_unmapped = "R-NOMAP-000" in rec["rule_ids"]
        if qb is not None:
            src, hint, flag = "declared", qb, "ok"
        elif is_unmapped:
            src, hint, flag = "not_applicable", None, "n/a"
            qb = None
        elif old_dir is None:      # 组合对：多约束，无单一界
            src, hint, flag = "multi_constraint", None, "n/a(组合对)"
            qb = None
        else:
            # 推导以**键/规则语义为主**（文本关键词只在真命中时算证据）：
            # 干跑实例：DB3-0001「变形不超过」→ 刚度实为下界（派生量反向）；
            # DB3-0007/0017 无方向词 → 解析器 lower 只是默认兜底；
            # sigma_req 按定义是 σ 的上界。
            text_phys = parser.parse(rec["requirement_text"]).direction
            text_hint = ("upper" if text_phys == "bound_above" else "lower") \
                if _text_has_dir_kw(rec["requirement_text"]) else None
            prop = _proposal_from_keys(rs, rec["rule_ids"],
                                       rec.get("info_metric_vector"))
            hint = prop if prop is not None else text_hint
            src = "unresolved"          # 不写入未经验证的界
            if text_hint is not None and prop is not None \
                    and text_hint != prop:
                flag = "REVIEW:文本词与该量期望冲突"
            elif prop is not None:
                flag = "pending:propose_%s" % prop
            elif text_hint is not None:
                flag = "pending:text_%s" % text_hint
            else:
                flag = "REVIEW:需人工判定"
            n_new += 1
        amb = _ambiguous(rec["requirement_text"])
        if src == "unresolved" and amb and flag == "pending":
            flag = "pending(多约束)"
        plan.append((rec["pair_id"], old_dir, qb, hint, src, flag))
        if apply:
            rec["requirement_structured"] = {
                **{k: v for k, v in rs.items() if k != "direction"},
                "metric_direction": old_dir,
                "direction": old_dir,
                "quantity_bound": qb,
                "quantity_bound_hint": hint,
                "quantity_bound_source": src,
                "quantity_bound_review": flag,
            }
            path.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    print("=== direction 语义迁移%s（%d 对）==="
          % ("（已写入）" if apply else "（干跑）", len(plan)))
    print("  pair       原 direction   quantity_bound  hint    来源         核对")
    for pid, od, qb, hint, src, flag in plan:
        print("  %-10s %-14s %-14s %-7s %-12s %s"
              % (pid, od, str(qb), str(hint), src, flag))
    n_rev = sum(1 for p in plan if str(p[5]).startswith("REVIEW"))
    n_pend = sum(1 for p in plan if str(p[5]).startswith("pending"))
    print(f"  需人工判定：{n_rev} 条（文本词与该量期望冲突 / 无推导依据）；"
          f"推导已提出待确认：{n_pend} 条；"
          f"UNMAPPED 不适用：{sum(1 for p in plan if p[5] == 'n/a')} 条")
    if apply:
        print(f"  原文件备份：{BACKUP}")
    else:
        print("  （干跑结束；加 --apply 写入）")
    return 0


def backup_once() -> None:
    if BACKUP.exists():
        return
    BACKUP.mkdir(parents=True, exist_ok=True)
    for p in PAIRS.glob("DB3-*.json"):
        shutil.copy2(p, BACKUP / p.name)


if __name__ == "__main__":
    if "--apply" in sys.argv[1:]:
        backup_once()
    sys.exit(main())
