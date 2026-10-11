# -*- coding: utf-8 -*-
"""multi-seed 记录清理脚本（第 51 条配套：rician 时代 d149 记录作废）。

用法（项目根目录）：
    python a_network/purge_records.py <token>

动作：
  1) multi_seed_verify.jsonl：删除 key/holdout_sig 含 token 的行；
  2) 删除 gine_metrics_family_*_seed*_*<token>*.json（种子指标文件）；
  3) 重写 multi_seed_verify_summary.json（丢弃同签名组）。

场景：数据集内容变了但系统数相同 → -d{n} 标签撞车（rician 回滚后
wrapnorm 同为 149 系统）；须先清除旧数据版本的记录再跑新验证。
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

_A_DIR = Path(__file__).resolve().parent
_RESULTS = _A_DIR / "multi_seed_verify.jsonl"
_SUMMARY = _A_DIR / "multi_seed_verify_summary.json"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def main(argv: list[str] | None = None) -> int:
    token = (argv or sys.argv[1:])[0] if (argv or sys.argv[1:]) else None
    if not token:
        print("用法：python a_network/purge_records.py <token>")
        return 2

    n_lines = 0
    if _RESULTS.exists():
        keep = []
        for line in _RESULTS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                keep.append(line)
                continue
            if token in rec.get("key", "") or token in rec.get(
                    "holdout_sig", ""):
                n_lines += 1
            else:
                keep.append(line)
        _RESULTS.write_text("\n".join(keep) + ("\n" if keep else ""),
                            encoding="utf-8")
        print(f"[jsonl] 删除 {n_lines} 条记录（token={token}）")

    n_files = 0
    for f in glob.glob(str(_A_DIR / f"*{token}*.json")):
        Path(f).unlink()
        n_files += 1
    print(f"[seed files] 删除 {n_files} 个文件")

    if _SUMMARY.exists():
        summary = json.loads(_SUMMARY.read_text(encoding="utf-8"))
        dropped = []
        for grp in list(summary.keys()):
            if token in str(grp):
                dropped.append(grp)
                del summary[grp]
        _SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"[summary] 丢弃组：{dropped}")

    print("清理完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
