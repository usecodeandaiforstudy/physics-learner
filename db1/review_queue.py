"""DB1 人工复核队列与复核记录管理。

配合 `db1/review_workflow.md` 的双人复核流程，提供三件事：

1. **复核队列构建**：为每个条目生成复核记录（机器门结果预填，人工字段留空），
   并按 review_workflow.md §4.3 的签核矩阵，为 DB2 路径选定 ≥20% 抽检名单。
2. **抽检选择**：确定性等距抽样（可复现），供 Reviewer A（物理）复核。
3. **复核回写**：记录复核结论后，同步更新条目 provenance 的
   human_review_status / review_date / reviewers 三个字段。

用法：
    python db1/review_queue.py build              # 构建复核队列 + 抽检名单
    python db1/review_queue.py status             # 查看复核进度
    python db1/review_queue.py apply <system_id> <reviewer> <accept|reject> [findings]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from validate_entry import validate_entry  # noqa: E402

ENTRIES_DIR = _HERE / "entries"
REVIEWS_DIR = _HERE / "reviews"
SPOT_CHECK_FILE = REVIEWS_DIR / "spot_check_list.json"

# DB2 路径的抽检比例（review_workflow.md §4.3：≥20%）
SPOT_CHECK_FRACTION = 0.2


# ---------------------------------------------------------------------------
# 抽检选择
# ---------------------------------------------------------------------------
def select_spot_check(system_ids: list[str], fraction: float = SPOT_CHECK_FRACTION) -> list[str]:
    """确定性等距抽检：排序后按步长取，保证可复现且比例达标。

    参数
    ----
    system_ids : list[str]
        全部条目 ID。
    fraction : float
        抽检比例（≥0.2）。

    返回
    ----
    list[str]
        抽检选中的条目 ID（保持排序）。
    """
    ids = sorted(set(system_ids))
    if not ids:
        return []
    # 向上取整：保证抽检比例恒 ≥ 指定值（如 51×0.2=10.2 → 11，而非 round 后的 10<20%）
    n_take = max(1, math.ceil(len(ids) * fraction))
    if n_take >= len(ids):
        return ids
    # 等距抽样：step = n / n_take，取 floor(i*step)
    step = len(ids) / n_take
    picked = [ids[min(int(i * step), len(ids) - 1)] for i in range(n_take)]
    # 去重并保持顺序
    seen: set[str] = set()
    out = []
    for x in picked:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------------------------------------------------------
# 复核记录
# ---------------------------------------------------------------------------
def build_review_record(entry: dict) -> dict:
    """为单条目构建复核记录：机器门结果预填，人工字段留空。

    复核记录结构对齐 review_workflow.md §6 的签核记录格式。
    """
    # 重跑机器门（确定性、廉价），把证据填进记录
    att = _load_attestation_for(entry)
    machine = validate_entry(entry, att)

    prov = entry.get("provenance", {})
    return {
        "review_record": {
            "entry_system_id": entry.get("system_id"),
            "family": entry.get("family"),
            "domain": entry.get("domain"),
            "scan_param": entry.get("scan_param"),
            "machine_gate": {
                "schema_valid": machine["passed"] or _no_schema_error(machine),
                "convention_check": machine["passed"] or _no_convention_error(machine),
                "computability": machine["passed"] or _no_computability_error(machine),
                "analytic_compare": machine["passed"],
                "errors": machine["errors"],
                "warnings": machine["warnings"],
            },
            "provenance_source": prov.get("source"),
            "anchor_eligible": False,  # AI 生成物恒为 false（review_workflow.md §4.2）
            "reviewer_a": {
                "name": None, "decision": None, "findings": None, "date": None,
            },
            "reviewer_b": {
                "name": None, "decision": None, "findings": None, "date": None,
            },
            "final_destination": None,   # DB2 / DB3
            "final_decision": None,
        }
    }


def _no_schema_error(m: dict) -> bool:
    return not any("缺少必填字段" in e or "不允许的字段" in e or "非法枚举" in e
                   for e in m["errors"])


def _no_convention_error(m: dict) -> bool:
    return not any("自证表" in e or "桥接" in e or "anchor" in e for e in m["errors"])


def _no_computability_error(m: dict) -> bool:
    return not any("无法解析" in e or "未声明" in e for e in m["errors"])


def _load_attestation_for(entry: dict) -> dict:
    """按条目桥接常数重建自证表（与 batch_generate_ingest 保持一致）。"""
    bridge = [p["name"] for p in entry.get("parameters", []) if p.get("is_bridge_constant")]
    return {"convention_attestation": {
        "convention_version": entry.get("convention_version"),
        "log_base": "natural",
        "log_base_unit": "nats",
        "sigma_parameterization": "sigma_as_std",
        "sde_calculus": "ito",
        "thermal_bath_kT": 1.0,
        "nondimensionalized": True,
        "bridge_constants": bridge,
        "mi_estimator_used": "analytic",
        "ib_beta_grid": "logspace(-2, 3, 32)",
        "anchor_formulas": [],
        "uncertainties": [],
    }}


# ---------------------------------------------------------------------------
# 队列构建
# ---------------------------------------------------------------------------
def build_queue() -> dict:
    """构建复核队列：为每个条目写复核记录，并生成抽检名单。

    **幂等性（2026-10-02 修，真缺陷，第 168 条）**：初版**无条件覆盖**已有
    复核记录 ⟹ 实测把 `lc_room_temp` 的 `reviewer_a`（accept/结论/日期）
    **清空**，而条目 provenance 仍写 `spot_checked` ⟹ **条目与记录不一致**、
    `status` 也退回全 pending。正解：**保留已有的人工结论**（`reviewer_a` /
    `reviewer_b` / `final_decision` / `final_destination` /
    `human_review_status`），只刷新**机器门**等自动字段。
    """
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    entry_files = sorted(ENTRIES_DIR.glob("*.json"))

    KEEP = ("reviewer_a", "reviewer_b", "final_decision",
            "final_destination", "human_review_status")
    system_ids = []
    preserved = 0
    for ef in entry_files:
        entry = json.loads(ef.read_text(encoding="utf-8"))
        sid = entry.get("system_id")
        system_ids.append(sid)
        rec = build_review_record(entry)
        out = REVIEWS_DIR / f"{sid}.yaml"
        if out.exists():
            try:
                old = yaml.safe_load(out.read_text(encoding="utf-8")) or {}
                old_rr = old.get("review_record") or {}
                new_rr = rec.get("review_record") or {}
                for k in KEEP:
                    if old_rr.get(k) not in (None, {}, []):
                        new_rr[k] = old_rr[k]
                        preserved += 1
            except Exception:  # noqa: BLE001
                pass
        out.write_text(yaml.safe_dump(rec, allow_unicode=True,
                                      sort_keys=False), encoding="utf-8")

    spot = select_spot_check(system_ids, SPOT_CHECK_FRACTION)
    SPOT_CHECK_FILE.write_text(json.dumps({
        "fraction": SPOT_CHECK_FRACTION,
        "total_entries": len(system_ids),
        "spot_check_count": len(spot),
        "spot_check_ids": spot,
        "note": "DB2 路径：Reviewer A 抽检这些条目（review_workflow.md §4.3）",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "entries": len(system_ids),
        "review_records_written": len(system_ids),
        "human_fields_preserved": preserved,
        "spot_check_count": len(spot),
        "spot_check_ids": spot,
    }


# ---------------------------------------------------------------------------
# 复核回写
# ---------------------------------------------------------------------------
def apply_review(system_id: str, reviewer: str, decision: str,
                 findings: str = "", role: str = "A") -> dict:
    """记录一次复核结论，并同步更新条目 provenance。

    参数
    ----
    system_id : str
        条目 ID。
    reviewer : str
        复核人姓名/标识。
    decision : str
        accept / reject。
    findings : str
        复核发现（可空）。
    role : str
        A（物理审查员）或 B（信息论审查员）。

    返回
    ----
    dict
        更新后的状态摘要。
    """
    if decision not in ("accept", "reject"):
        raise ValueError(f"decision 须为 accept/reject，收到 {decision!r}")
    if role not in ("A", "B"):
        raise ValueError(f"role 须为 A/B，收到 {role!r}")

    entry_file = ENTRIES_DIR / f"{system_id}.json"
    if not entry_file.exists():
        raise FileNotFoundError(f"条目不存在: {entry_file}")
    entry = json.loads(entry_file.read_text(encoding="utf-8"))

    rec_file = REVIEWS_DIR / f"{system_id}.yaml"
    rec = (yaml.safe_load(rec_file.read_text(encoding="utf-8"))
           if rec_file.exists() else build_review_record(entry))

    today = date.today().isoformat()
    rec["review_record"][f"reviewer_{role.lower()}"] = {
        "name": reviewer, "decision": decision,
        "findings": findings or "无", "date": today,
    }

    # 判定复核状态：
    # - 仅 Reviewer A 抽检 → spot_checked（DB2 路径）
    # - A + B 均 accept → fully_reviewed（可作 DB3 锚点候选）
    ra = rec["review_record"].get("reviewer_a", {})
    rb = rec["review_record"].get("reviewer_b", {})
    if decision == "reject":
        status = "pending"          # 被拒 → 退回，需重做
    elif ra.get("decision") == "accept" and rb.get("decision") == "accept":
        status = "fully_reviewed"
    elif ra.get("decision") == "accept" or rb.get("decision") == "accept":
        status = "spot_checked"
    else:
        status = "pending"

    rec["review_record"]["final_decision"] = decision
    rec["review_record"]["final_destination"] = "DB2"
    rec["review_record"]["human_review_status"] = status

    rec_file.write_text(yaml.safe_dump(rec, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")

    # 同步条目 provenance
    prov = entry.setdefault("provenance", {})
    prov["human_review_status"] = status
    if status != "pending":
        prov["review_date"] = today
        reviewers = sorted({x for x in [ra.get("name"), rb.get("name")] if x})
        prov["reviewers"] = reviewers
    entry_file.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"system_id": system_id, "role": role, "decision": decision,
            "human_review_status": status, "review_date": prov.get("review_date")}


# ---------------------------------------------------------------------------
# 进度查询
# ---------------------------------------------------------------------------
def worksheet(reviewer: str = "reviewer") -> Path:
    """**生成抽检复核工作表**（2026-10-02，第 183 条；只读 DB1，写根目录 MD）。

    每条给出：序号/ID/名称/域/族/严格性/机器门/**独立核验**/出处 +
    **两个打勾位**（判断项是否通过）+ **可直接粘贴的 apply 命令**（带 `reviewer`）。
    **不改任何复核状态**（红线：判定由人下）。
    """
    import csv as _csv
    ind: dict[str, str] = {}
    csvp = _PROJECT_ROOT / "DB1_INDEPENDENT_CHECK.csv"
    if csvp.exists():
        with open(csvp, "r", encoding="utf-8", newline="") as f:
            for r in _csv.DictReader(f):
                ind[r["system_id"]] = r.get("verdict", "")
    rows = next_pending(10 ** 6, spot_only=True, reviewer=reviewer)
    ids = [r["system_id"] for r in rows]
    # **改为基于 `brief()` 生成（2026-10-02 修）**：`next_pending` **不返回**
    #   `machine_gate`/`source_ref`/`status`/`reject_cmd` 字段 ⟹ 初版直接用它
    #   会把这几列打成 `None` ✗。`brief()` 才是"证据齐全"的那一份。
    det = {b["system_id"]: b for b in brief(ids, reviewer=reviewer)}
    out = _PROJECT_ROOT / "DB1_REVIEW_WORKSHEET.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# DB1 抽检复核工作表（%d 条，生成于 %s）\n\n"
                % (len(ids), time.strftime("%Y-%m-%d %H:%M")))
        f.write("> **只读生成**：本表不修改任何状态。**判定只能由人下**"
                "（红线：AI 不得自审 DB1）。\n>\n"
                "> 判据参考：机器门（schema/约定/可算性/解析对照）＋"
                "**独立数值核验**（对该条目密度直接数值微分求 Fisher 定义式；"
                "`口径不可比`/`方法未过自校` 表示**我的方法不适用**，"
                "**不是条目问题**）。\n\n")
        f.write("## 用法\n\n```\n")
        f.write("python db1\\review_queue.py brief <id>      # 看单条完整证据\n")
        f.write("python db1\\review_queue.py apply <id> %s "
                "accept \"结论\" --role A\n" % reviewer)
        f.write("python db1\\review_queue.py status          # 看进度\n```\n\n")
        f.write("状态语义：单角色 `accept` ⟹ `spot_checked`；"
                "**A+B 双 accept** ⟹ `fully_reviewed`；`reject` ⟹ 退回 `pending`。\n\n")
        f.write("## 清单\n\n")
        for i, sid in enumerate(ids, 1):
            b = det.get(sid) or {}
            gate = b.get("machine_gate")
            gate_s = "全过 ✓" if gate == "全过 ✓" else "**%s**" % gate
            f.write("### %d. `%s` ｜ %s\n\n" % (i, sid, b.get("name")))
            f.write("- 域/族：**%s / %s** ｜ 严格性：**%s** ｜ 状态：%s\n"
                    % (b.get("domain"), b.get("family"),
                       b.get("strictness"), b.get("status")))
            f.write("- 机器门：%s ｜ 独立核验：%s\n"
                    % (gate_s, b.get("independent", "（未跑）")))
            f.write("- 方程：%s\n" % (b.get("equations") or {}).get("latex"))
            fpv = b.get("fingerprint") or []
            if fpv:
                f.write("- 指纹声明：`%s = %s`\n" % (fpv[0][0], fpv[0][1]))
            f.write("- 出处：%s\n" % b.get("source_ref"))
            f.write("- [ ] ① 方程忠实描述该系统  "
                    "- [ ] ② 出处对得上  "
                    "- [ ] ③ 严格性标注恰当\n")
            f.write("- 记：`%s`\n" % b.get("apply_cmd"))
            f.write("- 退：`%s`\n\n" % b.get("reject_cmd"))
    return out


def brief(system_ids: list[str], reviewer: str = "reviewer") -> list[dict]:
    """**复核简报**（2026-10-02，第 181 条；只读）：把一条条目的
    **声明 + 机器证据 + 独立核验 + 待判断问题**一次给全。

    **不代替人工判断**（红线）：只把"人能直接采信的证据"摆在一起，
    并列出**必须由人回答**的判断项。
    """
    import csv as _csv
    ind: dict[str, dict] = {}
    csvp = _PROJECT_ROOT / "DB1_INDEPENDENT_CHECK.csv"
    if csvp.exists():
        with open(csvp, "r", encoding="utf-8", newline="") as f:
            for r in _csv.DictReader(f):
                ind[r["system_id"]] = r
    out = []
    for sid in system_ids:
        ef = ENTRIES_DIR / f"{sid}.json"
        if not ef.exists():
            continue
        e = json.loads(ef.read_text(encoding="utf-8"))
        rf = REVIEWS_DIR / f"{sid}.yaml"
        rr = {}
        if rf.exists():
            rr = (yaml.safe_load(rf.read_text(encoding="utf-8"))
                  or {}).get("review_record") or {}
        out.append({
            "system_id": sid, "name": e.get("name"),
            "domain": e.get("domain"), "family": e.get("family"),
            "strictness": e.get("strictness_level"),
            "equations": e.get("equations"),
            "parameters": [(p.get("name"), p.get("range"),
                            p.get("default"), p.get("unit"),
                            bool(p.get("is_bridge_constant")))
                           for p in (e.get("parameters") or [])],
            "channel_template": e.get("channel_template"),
            "source_ref": e.get("source_ref"),
            "fingerprint": [(f.get("quantity"), f.get("analytic"),
                             f.get("tol"))
                            for f in (e.get("fingerprint_expectation") or [])],
            "machine_gate": (rr.get("machine_gate") or {}).get("errors") or
                            "全过 ✓",
            "independent": ind.get(sid, {}).get("verdict", "（未跑独立核验）"),
            "status": (e.get("provenance") or {}).get("human_review_status"),
            "judge": [
                "① 方程是否**忠实描述**该系统的物理？（域/族是否相符）",
                "② `source_ref` 的出处是否**对得上**这条内容？",
                "③ 严格性等级（定理/约定）标注是否**恰当**？",
            ],
            "apply_cmd": (f'python db1\\review_queue.py apply {sid} '
                          f'{reviewer} accept "结论" --role A'),
            "reject_cmd": (f'python db1\\review_queue.py apply {sid} '
                           f'{reviewer} reject "问题描述" --role A'),
        })
    return out


def show(system_id: str) -> dict:
    """**打印一条 DB1 条目的复核所需字段（只读）**（2026-10-02，第 161 条）。

    复核人不必逐个打开 JSON：一条命令给出方程、参数、指纹期望、出处、
    复核状态，外加**机器可核项的预检结果**（人能直接采信的部分）。
    """
    entry_file = ENTRIES_DIR / f"{system_id}.json"
    if not entry_file.exists():
        raise FileNotFoundError(f"找不到条目：{entry_file}")
    e = json.loads(entry_file.read_text(encoding="utf-8"))
    prov = e.get("provenance") or {}
    rec_file = REVIEWS_DIR / f"{system_id}.yaml"
    rec = (yaml.safe_load(rec_file.read_text(encoding="utf-8"))
           if rec_file.exists() else {})
    # 机器可核项预检（与 review_workflow 的机器门一致：字段齐备 + 参数范围）
    issues: list[str] = []
    for pr in (e.get("parameters") or []):
        rng = pr.get("range")
        if not (isinstance(rng, list) and len(rng) == 2
                and float(rng[0]) < float(rng[1])):
            issues.append(f"参数 {pr.get('name')} 范围异常：{rng}")
        if pr.get("is_bridge_constant") and not pr.get("unit"):
            issues.append(f"桥接常数 {pr.get('name')} 缺单位")
    if not e.get("equations"):
        issues.append("缺 equations")
    return {"system_id": system_id, "name": e.get("name"),
            "domain": e.get("domain"), "family": e.get("family"),
            "strictness_level": e.get("strictness_level"),
            "equations": e.get("equations"),
            "parameters": e.get("parameters"),
            "fingerprint_expectation": e.get("fingerprint_expectation"),
            "channel_template": e.get("channel_template"),
            "source_ref": e.get("source_ref"),
            "human_review_status": prov.get("human_review_status"),
            "review_record": rec.get("review_record"),
            "machine_gate_issues": issues or "无（机器可核项全过）"}


def next_pending(n: int = 5, spot_only: bool = True,
                 reviewer: str = "reviewer") -> list[dict]:
    """**列出接下来该复核的 N 条**（2026-10-02，第 166 条；只读）。

    默认按**抽检名单顺序**（工作流 §4.3 要求的 ≥20%），跳过已复核者；
    每条给出：系统 ID / 名称 / 域 / 族 / 机器门结论 / **可直接粘贴的 apply 命令**。
    """
    ids: list[str] = []
    spot_file = REVIEWS_DIR / "spot_check_list.json"
    if spot_only and spot_file.exists():
        ids = json.loads(spot_file.read_text(encoding="utf-8")).get(
            "spot_check_ids", [])
    if not ids:
        ids = sorted(p.stem for p in ENTRIES_DIR.glob("*.json"))
    out: list[dict] = []
    for sid in ids:
        ef = ENTRIES_DIR / f"{sid}.json"
        if not ef.exists():
            continue
        e = json.loads(ef.read_text(encoding="utf-8"))
        prov = e.get("provenance") or {}
        if prov.get("human_review_status") not in (None, "pending"):
            continue
        rf = REVIEWS_DIR / f"{sid}.yaml"
        rr = {}
        if rf.exists():
            rr = (yaml.safe_load(rf.read_text(encoding="utf-8"))
                  or {}).get("review_record") or {}
        gate = rr.get("machine_gate") or {}
        ok = bool(gate.get("schema_valid", True)
                  and gate.get("computability", True)
                  and gate.get("analytic_compare", True)
                  and not gate.get("errors"))
        out.append({
            "system_id": sid, "name": e.get("name"), "domain": e.get("domain"),
            "family": e.get("family"),
            "strictness": e.get("strictness_level"),
            "machine_gate_ok": ok,
            "machine_gate_errors": gate.get("errors") or [],
            "show_cmd": f'python db1\\review_queue.py show {sid}',
            "apply_cmd": (f'python db1\\review_queue.py apply {sid} '
                          f'{reviewer} accept "结论" --role A'),
        })
        if len(out) >= n:
            break
    return out


def status() -> dict:
    """统计复核进度。"""
    records = sorted(REVIEWS_DIR.glob("*.yaml")) if REVIEWS_DIR.exists() else []
    counts = {"pending": 0, "spot_checked": 0, "fully_reviewed": 0}
    for rf in records:
        rec = yaml.safe_load(rf.read_text(encoding="utf-8"))
        st = rec["review_record"].get("human_review_status")
        if st is None:
            # 未复核的记录里没有该字段
            st = "pending"
        counts[st] = counts.get(st, 0) + 1
    spot = json.loads(SPOT_CHECK_FILE.read_text(encoding="utf-8")) if SPOT_CHECK_FILE.exists() else {}
    return {
        "review_records": len(records),
        "status_counts": counts,
        "spot_check_required": spot.get("spot_check_count", 0),
        "spot_check_ids": spot.get("spot_check_ids", []),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(description="DB1 人工复核队列管理")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="构建复核队列 + 抽检名单")
    sub.add_parser("status", help="查看复核进度")
    p_ws = sub.add_parser("worksheet", help="生成抽检复核工作表（只读）")
    p_ws.add_argument("--reviewer", default="reviewer")
    p_show = sub.add_parser("show", help="看一条条目的复核所需字段（只读）")
    p_show.add_argument("system_id")
    p_next = sub.add_parser("next", help="列出接下来该复核的 N 条（只读）")
    p_next.add_argument("-n", type=int, default=5)
    p_next.add_argument("--all", action="store_true",
                        help="不限抽检名单，按 ID 顺序")
    p_next.add_argument("--reviewer", default="reviewer",
                        help="复核人标识（默认 reviewer）")
    p_brief = sub.add_parser("brief", help="复核简报（声明+证据+待判断项，只读）")
    p_brief.add_argument("system_ids", nargs="*")
    p_brief.add_argument("--next", type=int, default=0,
                         help="改为取接下来 N 条")
    p_brief.add_argument("--reviewer", default="reviewer",
                         help="复核人标识（默认 reviewer）")

    p_apply = sub.add_parser("apply", help="记录一次复核结论")
    p_apply.add_argument("system_id")
    p_apply.add_argument("reviewer")
    p_apply.add_argument("decision", choices=["accept", "reject"])
    p_apply.add_argument("findings", nargs="?", default="")
    p_apply.add_argument("--role", default="A", choices=["A", "B"])

    args = parser.parse_args(argv)

    if args.cmd == "build":
        r = build_queue()
        print(f"已写复核记录：{r['review_records_written']} 条")
        print(f"抽检名单（{r['spot_check_count']}/{r['entries']}）：{r['spot_check_ids']}")
        return 0
    if args.cmd == "status":
        s = status()
        print(json.dumps(s, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "worksheet":
        p = worksheet(reviewer=getattr(args, "reviewer", "reviewer"))
        print("工作表已写入：%s" % p.name)
        print("（**未修改任何复核状态**；判定由人下）")
        return 0
    if args.cmd == "brief":
        ids = list(args.system_ids or [])
        rev = getattr(args, "reviewer", "reviewer")
        if args.next:
            ids = [r["system_id"] for r in next_pending(args.next, reviewer=rev)]
        if not ids:
            print("用法：brief <系统ID...> 或 brief --next 5")
            return 0
        for b in brief(ids, reviewer=rev):
            print("=" * 74)
            print("%s ｜ %s" % (b["system_id"], b["name"]))
            print("=" * 74)
            print("域/族：%s / %s    严格性：**%s**    状态：%s"
                  % (b["domain"], b["family"], b["strictness"], b["status"]))
            eq = b["equations"] or {}
            print("方程：%s" % eq.get("latex", "—"))
            print("      （sympy：%s）" % eq.get("sympy", "—"))
            print("参数：")
            for nm, rg, df, un, br in b["parameters"]:
                print("   %-16s 范围 %-14s 默认 %-8s %s%s"
                      % (nm, rg, df, ("单位 " + str(un)) if un else "",
                         "【桥接常数】" if br else ""))
            print("声道模板：%s" % b["channel_template"])
            print("指纹声明：")
            for q, a, t in b["fingerprint"]:
                print("   %-14s = %s   (tol %s)" % (q, a, t))
            print("出处：%s" % b["source_ref"])
            print("机器门：%s" % b["machine_gate"])
            print("独立核验：%s" % b["independent"])
            print("**待你判断（AI 不做此判断）**：")
            for q in b["judge"]:
                print("   [ ] %s" % q)
            print("记通过：%s" % b["apply_cmd"])
            print("记退回：%s" % b["reject_cmd"])
            print()
        return 0
    if args.cmd == "show":
        print(json.dumps(show(args.system_id), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "next":
        rows = next_pending(args.n, spot_only=not args.all,
                            reviewer=getattr(args, "reviewer", "reviewer"))
        if not rows:
            print("**没有待复核条目**（抽检名单已清空或全部已复核）✓")
            return 0
        print("接下来 %d 条（只读；状态未改动）：\n" % len(rows))
        for i, r in enumerate(rows, 1):
            gate = "机器门全过 ✓" if r["machine_gate_ok"] else \
                "**机器门有问题** %s" % r["machine_gate_errors"]
            print("%d) %s ｜ %s ｜ %s / %s ｜ %s ｜ %s"
                  % (i, r["system_id"], r["name"], r["domain"], r["family"],
                     r["strictness"], gate))
            print("   看： %s" % r["show_cmd"])
            print("   记： %s\n" % r["apply_cmd"])
        return 0
    if args.cmd == "apply":
        r = apply_review(args.system_id, args.reviewer, args.decision,
                         args.findings, role=args.role)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
