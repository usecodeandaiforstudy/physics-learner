"""review_queue.py 自测：抽检选择 + 复核回写 + 状态统计。

用临时备份/还原的方式测试，不污染真实数据。
运行：python db1/test_review.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import review_queue as rq  # noqa: E402


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    lines = []

    # --- 1. 抽检比例与确定性 ---
    ids = [f"sys_{i:03d}" for i in range(30)]
    s1 = rq.select_spot_check(ids, 0.2)
    s2 = rq.select_spot_check(ids, 0.2)
    lines.append(
        ("PASS" if len(s1) == 6 else "FAIL")
        + f"  抽检数量 6/30（20%）：实际 {len(s1)}"
    )
    lines.append(
        ("PASS" if s1 == s2 else "FAIL")
        + f"  抽检确定性（两次调用结果一致）"
    )

    # --- 2. 抽检边界：小集合不退化为 0 ---
    s3 = rq.select_spot_check(["a", "b", "c"], 0.2)
    lines.append(
        ("PASS" if len(s3) >= 1 else "FAIL")
        + f"  小集合抽检至少 1 个：{s3}"
    )

    # --- 3. 真实条目上测 apply_review（备份后还原）---
    sid = "ou_rc_lowpass"
    entry_file = rq.ENTRIES_DIR / f"{sid}.json"
    rec_file = rq.REVIEWS_DIR / f"{sid}.yaml"
    if not entry_file.exists():
        lines.append(f"SKIP  条目 {sid} 不存在，跳过回写测试")
    else:
        backup_entry = entry_file.read_bytes()
        backup_rec = rec_file.read_bytes() if rec_file.exists() else None
        try:
            # apply_review 内部会在记录缺失时构建单条，无需 build_queue。
            # （build_queue 会重写全部 reviews/*.yaml，污染真实复核记录——已踩坑，
            #   见 HANDOVER 第 17 条，切勿在此调用。）
            res = rq.apply_review(sid, "TEST-REVIEWER-A", "accept",
                                  "机制自测", role="A")
            # 校验条目 provenance 被更新
            entry = json.loads(entry_file.read_text(encoding="utf-8"))
            prov = entry["provenance"]
            ok_status = prov.get("human_review_status") == "spot_checked"
            ok_date = bool(prov.get("review_date"))
            ok_reviewers = prov.get("reviewers") == ["TEST-REVIEWER-A"]
            lines.append(
                ("PASS" if (ok_status and ok_date and ok_reviewers) else "FAIL")
                + f"  回写 provenance：status={prov.get('human_review_status')}, "
                  f"date={prov.get('review_date')}, reviewers={prov.get('reviewers')}"
            )

            # 校验 review 记录里 reviewer_a 被填
            import yaml
            rec = yaml.safe_load(rec_file.read_text(encoding="utf-8"))
            ra = rec["review_record"]["reviewer_a"]
            lines.append(
                ("PASS" if ra.get("name") == "TEST-REVIEWER-A" and ra.get("decision") == "accept" else "FAIL")
                + f"  复核记录 reviewer_a 已填：{ra.get('name')}/{ra.get('decision')}"
            )

            # A+B 都 accept → fully_reviewed
            rq.apply_review(sid, "TEST-REVIEWER-B", "accept", "机制自测", role="B")
            entry = json.loads(entry_file.read_text(encoding="utf-8"))
            lines.append(
                ("PASS" if entry["provenance"].get("human_review_status") == "fully_reviewed" else "FAIL")
                + f"  A+B 均 accept → fully_reviewed：实际 {entry['provenance'].get('human_review_status')}"
            )

            # reject 应退回 pending
            rq.apply_review(sid, "TEST-REVIEWER-A", "reject", "机制自测", role="A")
            entry = json.loads(entry_file.read_text(encoding="utf-8"))
            lines.append(
                ("PASS" if entry["provenance"].get("human_review_status") == "pending" else "FAIL")
                + f"  reject → 退回 pending：实际 {entry['provenance'].get('human_review_status')}"
            )
        finally:
            # 还原
            entry_file.write_bytes(backup_entry)
            if backup_rec is not None:
                rec_file.write_bytes(backup_rec)
            lines.append("INFO  已还原测试前的条目与复核记录")

    n_pass = sum(1 for ln in lines if ln.startswith("PASS"))
    n_total = sum(1 for ln in lines if ln.startswith(("PASS", "FAIL")))
    lines.append(f"\n{n_pass}/{n_total} 通过")

    out = Path(r"d:\physics learn\review_test_result.txt")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
