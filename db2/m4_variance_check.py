# -*- coding: utf-8 -*-
"""**M4 判定方差核查**（2026-10-02，第 185 条）——判定到底稳不稳？

## 为什么要做
第 184 条的归因实验出现**自相矛盾**：
    · 同一份数据（`db2.jsonl.filled-unified`）**上次跑 FAIL（Spearman 0.714）**，
      **这次跑 PASS（0.951）** ⟹ **M4 判定有 run-to-run 方差** ✗
⟹ 若方差跨越通过阈值，则我**先前据 M4 回滚 DB2 的依据不成立**（把噪声当信号）。
   项目纪律：**监测门的判定必须可复现**（≥N 次，看分布），不得单次定论。

## 本脚本
对**指定 DB2 版本**重复跑 M4 `--repeat N` 次，报告判定分布与 Spearman 的
min/中位/max/标准差；可对两个版本各跑一遍做对照。

用法：
    python db2/m4_variance_check.py --repeat 4
    python db2/m4_variance_check.py --repeat 4 --variant FILLED
"""

from __future__ import annotations

import json
import os
import shutil
import statistics as st
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
M4 = _ROOT / "a_network" / "v4_env_smoke.py"


def run_m4() -> tuple[str, float]:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(_ROOT))
    r = subprocess.run([sys.executable, str(M4)], cwd=str(_ROOT),
                       capture_output=True, text=True, env=env,
                       encoding="utf-8", errors="replace")
    out = r.stdout or ""
    verdict = "PASS" if "✅ V4 通过" in out else (
        "FAIL" if "部分通过" in out else "?")
    sp = float("nan")
    for ln in out.splitlines():
        if "Spearman" in ln:
            try:
                sp = float(ln.split("Spearman=")[1].split("（")[0]
                           .split("；")[0].strip().replace("+", ""))
            except Exception:  # noqa: BLE001
                pass
    return verdict, sp


def main() -> int:
    n = 4
    if "--repeat" in sys.argv:
        n = int(sys.argv[sys.argv.index("--repeat") + 1])
    variant = "CUR"
    if "--variant" in sys.argv:
        variant = sys.argv[sys.argv.index("--variant") + 1].upper()
    src = FILLED if variant == "FILLED" else DB2
    if variant == "FILLED" and not FILLED.exists():
        print("缺少 %s" % FILLED.name)
        return 1
    print("=" * 74)
    print("M4 判定方差核查：版本 **%s**，重复 %d 次" % (variant, n))
    print("=" * 74)
    backup = DB2.with_suffix(".jsonl.var-backup")
    shutil.copy2(DB2, backup)
    try:
        if variant == "FILLED":
            shutil.copy2(FILLED, DB2)
        vers, sps = [], []
        for i in range(n):
            t0 = time.time()
            v, sp = run_m4()
            vers.append(v)
            sps.append(sp)
            print("   第 %d 次：**%s**  Spearman=%+.3f（%.0fs）"
                  % (i + 1, v, sp, time.time() - t0))
        good = [s for s in sps if s == s]
        print("\n判定分布：%s" % {v: vers.count(v) for v in sorted(set(vers))})
        if good:
            print("Spearman：min %+.3f ｜ 中位 %+.3f ｜ max %+.3f ｜ 标准差 %.3f"
                  % (min(good), st.median(good), max(good),
                     st.pstdev(good) if len(good) > 1 else 0.0))
            print("⟹ **方差结论**：%s"
                  % ("判定**稳定**（全部一致）" if len(set(vers)) == 1 else
                     "**判定不稳定**（同一数据下 PASS/FAIL 混现）⟹ "
                     "**单次判定不可作为信号**"))
    finally:
        shutil.copy2(backup, DB2)
        backup.unlink(missing_ok=True)
        print("\nDB2 已恢复原状 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
