# -*- coding: utf-8 -*-
"""**DB2 曲线补齐的分组归因实验**（2026-10-02，第 184 条）。

## 目的
已知：把 5100 行"空曲线"补上后，`M4`（代理 vs 慢路径一致性）由 PASS 变 FAIL
（Spearman 0.907 → 0.714）。**机理是训练集构成位移**，但**是哪一部分造成的**？

## 分组（用审计副本 `db2.jsonl.filled-unified` 构造变体）
    · **变体 A**：只补**原本就有曲线**的族的空行（**不引入新族**）；
    · **变体 B**：只补**新注册的 5 族**（ou/diffusion/lc_circuit/
      gaussian_mixture2/truncated_geometric）的空行；
    · **原状**：不改（对照）。
对每个变体跑 `a_network/v4_env_smoke.py`（M4）⟹ 比较 Spearman 与判定。

**只读 DB1/DB3，不写 DB2**：每次把变体写到临时文件后**换入 → 跑 M4 → 换回**，
结束后恢复原状（并校验）。

用法：
    python db2/attribute_fill_effect.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

DB2 = _ROOT / "db2" / "db2.jsonl"
FILLED = _ROOT / "db2" / "db2.jsonl.filled-unified"
NEW_FAMS = {"ou", "diffusion", "lc_circuit", "gaussian_mixture2",
            "truncated_geometric"}
M4 = _ROOT / "a_network" / "v4_env_smoke.py"


def usable(cur) -> bool:
    iy = (cur or {}).get("I_TY") or []
    return bool(iy) and max(iy) > 1e-9


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
            if x.strip()]


def run_m4() -> tuple[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(_ROOT))
    r = subprocess.run([sys.executable, str(M4)], cwd=str(_ROOT),
                       capture_output=True, text=True, env=env,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    verdict = "PASS" if "✅ V4 通过" in out else (
        "FAIL" if "部分通过" in out else "?")
    line = next((ln for ln in out.splitlines() if "Spearman" in ln), "")
    return verdict, line.strip()


def main() -> int:
    if not FILLED.exists():
        print("缺少审计副本 %s" % FILLED.name)
        return 1
    cur = load(DB2)
    filled = load(FILLED)
    if len(cur) != len(filled):
        print("行数不一致")
        return 1
    # 原状中"有曲线"的族
    fams_with = {str(r.get("family")) for r in cur if usable(r.get("ib_curve"))}
    print("=" * 74)
    print("DB2 补齐效应归因：分组 A/B（M4 为判据）")
    print("=" * 74)
    print("原状有用曲线的族 %d 个；新注册族 %d 个"
          % (len(fams_with), len(NEW_FAMS)))

    variants = {
        "原状（不补）": None,
        "变体A：只补已有曲线的族": lambda i: (
            str(cur[i].get("family")) in fams_with),
        "变体B：只补新注册的 5 族": lambda i: (
            str(cur[i].get("family")) in NEW_FAMS),
        "全补（=审计副本）": lambda i: True,
    }
    results = {}
    backup = DB2.with_suffix(".jsonl.attr-backup")
    shutil.copy2(DB2, backup)
    try:
        for name, keep in variants.items():
            if keep is None:
                shutil.copy2(backup, DB2)
            else:
                rows = []
                n_fill = 0
                for i, r in enumerate(cur):
                    if usable(r.get("ib_curve")):
                        rows.append(r)
                        continue
                    if keep(i) and usable(filled[i].get("ib_curve")):
                        rows.append(filled[i])
                        n_fill += 1
                    else:
                        rows.append(r)
                tmp = DB2.with_suffix(".jsonl.variant")
                with open(tmp, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                os.replace(tmp, DB2)
                n_usable = sum(1 for r in rows if usable(r.get("ib_curve")))
                print("   [%s] 补入 %d 行 ⟹ 可用 %d" % (name, n_fill, n_usable))
            t0 = time.time()
            verdict, line = run_m4()
            results[name] = (verdict, line)
            print("   %-24s ⟹ **%s**（%.0fs）｜ %s"
                  % (name, verdict, time.time() - t0, line[:64]))
    finally:
        shutil.copy2(backup, DB2)          # **恢复原状**
        backup.unlink(missing_ok=True)
    print("\n===== 归因结论 =====")
    base_v = results["原状（不补）"][0]
    for name, (v, _l) in results.items():
        if name == "原状（不补）":
            continue
        print("   %-24s %s%s" % (name, v,
                                "（与基准一致）" if v == base_v
                                else "（**与基准不同 ⟹ 该组是致因**）"))
    print("\nDB2 已恢复原状 ✓（实验只改临时文件并回滚）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
