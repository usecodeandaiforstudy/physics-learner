# -*- coding: utf-8 -*-
"""**生成 DB1 抽检待办清单**（2026-10-02，第 162 条）。

只读：读取 `db1/entries/*.json` 与 `db1/reviews/`，产出人可读的
`DB1_REVIEW_CHECKLIST.md`（**不修改任何复核状态**）。

清单内容（60 条抽检，工作流 §4.3 要求 ≥20%）：
    · 每条：系统 ID / 名称 / 域 / 族 / 严格性 / 机器门结论 / 方程提示 /
      指纹期望式（首条）/ 当前状态，以及 **A、B 两个角色的打勾位**；
    · 顶部给**复核步骤**与**状态语义**（单角色 accept ⟹ spot_checked；
      A+B 双 accept ⟹ fully_reviewed；reject ⟹ 退回 pending）。

用法：
    python tools/db1_checklist.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

OUT = _ROOT / "DB1_REVIEW_CHECKLIST.md"


def main() -> int:
    import yaml
    entries_dir = _ROOT / "db1" / "entries"
    reviews_dir = _ROOT / "db1" / "reviews"
    spot_file = reviews_dir / "spot_check_list.json"
    ids: list[str] = []
    if spot_file.exists():
        ids = json.loads(spot_file.read_text(encoding="utf-8")).get(
            "spot_check_ids", [])
    if not ids:
        print("**未找到抽检名单**（先跑 `python db1/review_queue.py build`）")
        return 1

    rows = []
    n_machine_ok = 0
    for sid in ids:
        ef = entries_dir / ("%s.json" % sid)
        if not ef.exists():
            rows.append((sid, None))
            continue
        e = json.loads(ef.read_text(encoding="utf-8"))
        rf = reviews_dir / ("%s.yaml" % sid)
        rec = (yaml.safe_load(rf.read_text(encoding="utf-8"))
               if rf.exists() else {})
        rr = (rec or {}).get("review_record") or {}
        gate = rr.get("machine_gate") or {}
        gate_ok = bool(gate.get("schema_valid") and gate.get("computability")
                       and gate.get("analytic_compare")
                       and not gate.get("errors"))
        n_machine_ok += int(gate_ok)
        prov = e.get("provenance") or {}
        fp = (e.get("fingerprint_expectation") or [{}])[0]
        rows.append((sid, {
            "name": e.get("name"), "domain": e.get("domain"),
            "family": e.get("family"),
            "strictness": e.get("strictness_level"),
            "eq": (e.get("equations") or {}).get("sympy"),
            "fp": "%s = %s" % (fp.get("quantity"), fp.get("analytic")),
            "gate_ok": gate_ok,
            "gate_errors": gate.get("errors") or [],
            "status": prov.get("human_review_status"),
            "reviewers": prov.get("reviewers"),
            "ra": (rr.get("reviewer_a") or {}).get("decision"),
            "rb": (rr.get("reviewer_b") or {}).get("decision"),
        }))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# DB1 抽检待办清单（%d 条）\n\n" % len(ids))
        f.write("> **只读生成**；本清单不修改任何复核状态。人工复核由人执行"
                "（红线：AI 不得自审 DB1）。\n\n")
        f.write("## 复核步骤\n\n```\n")
        f.write("python db1\\review_queue.py show <id>        # 看内容（含机器门结论）\n")
        f.write("python db1\\review_queue.py apply <id> \"真实姓名\" accept \"结论\" --role A\n")
        f.write("python db1\\review_queue.py apply <id> \"真实姓名\" accept \"结论\" --role B\n")
        f.write("python db1\\review_queue.py status           # 看进度\n```\n\n")
        f.write("**状态语义**：单角色 `accept` ⟹ `spot_checked`；**A+B 双 accept** "
                "⟹ `fully_reviewed`；`reject` ⟹ 退回 `pending`。\n")
        f.write("**注意**：命令里填**真实值**——PowerShell 中 `<` 是保留运算符，"
                "照抄尖括号会报 `RedirectionNotSupported`。\n\n")
        f.write("## 机器门结论（%d/%d 全过）\n\n" % (n_machine_ok, len(ids)))
        f.write("机器门全过的条目，人工只需判**判断项**（物理建模合理性、"
                "文献出处、严格性等级标注）。\n\n")
        f.write("| # | 系统 ID | 名称 | 域 | 族 | 严格性 | 机器门 | 状态 | A | B |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for i, (sid, d) in enumerate(rows, 1):
            if d is None:
                f.write("| %d | %s | **缺条目** | | | | | | | |\n" % (i, sid))
                continue
            gate = "✓" if d["gate_ok"] else "**✗ %s**" % ("; ".join(
                str(x) for x in d["gate_errors"])[:40])
            f.write("| %d | `%s` | %s | %s | %s | %s | %s | %s | %s | %s |\n"
                    % (i, sid, d["name"], d["domain"], d["family"],
                       d["strictness"], gate, d["status"] or "pending",
                       d["ra"] or "☐", d["rb"] or "☐"))
        f.write("\n## 逐条参考信息\n\n")
        for i, (sid, d) in enumerate(rows, 1):
            if d is None:
                continue
            f.write("### %d. `%s`\n\n" % (i, sid))
            f.write("- 名称：%s ｜ 域：%s ｜ 族：%s ｜ 严格性：**%s**\n"
                    % (d["name"], d["domain"], d["family"], d["strictness"]))
            f.write("- 方程（sympy）：`%s`\n" % d["eq"])
            f.write("- 指纹期望（首条）：`%s`\n" % d["fp"])
            f.write("- 机器门：%s\n" % ("全过 ✓" if d["gate_ok"]
                                      else "**有问题** %s" % d["gate_errors"]))
            f.write("- 判断项：`[ ]` 建模合理 `[ ]` 出处可对 `[ ]` 严格性标注正确 "
                    "`[ ]` 参数范围合常识 `[ ]` 指纹式与方程一致\n\n")
        f.write("---\n\n生成命令：`python tools/db1_checklist.py`\n")

    print("抽检 **%d** 条；机器门全过 **%d** 条（%.0f%%）"
          % (len(ids), n_machine_ok, 100 * n_machine_ok / max(len(ids), 1)))
    print("清单已写入：%s" % OUT.name)
    print("（**未修改任何复核状态**）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
