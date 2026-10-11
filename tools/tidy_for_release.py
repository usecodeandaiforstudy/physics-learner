# -*- coding: utf-8 -*-
"""**开源前的仓库整理**（2026-10-02，第 208 条）——**显式清单、默认干跑**。

## 原则（本轮定死，避免我此前踩过的坑）
  1. **绝不用通配符删/移**（曾因 `DB3-019*.json` 误删 6 条锚点 ✗）
     ⟹ 本脚本**只按显式文件名清单**操作 ✓；通配符仅用于**只读的枚举** ✓
  2. **只移不删**：所有移出物到**仓库外**的归档目录，**保留不删** ✓
     （唯一例外：`__pycache__`（可再生字节码 ✓，先打印再删 ✓）
  3. **默认干跑**：`--apply` 才动手 ✓
  4. **第三方笔记只移"明确的"**（书名/论文名可辨 ✓）；**含糊的一律留在原地**
     并列入"待你确认" ✓（不替你做版权判断 ✓）

## 动作
  **A** 移出 **11 个明确第三方** 笔记 → `..\\physics-learn-notes\\`
  **B** 移出**体积大头**（DB2 备份 5 个 ≈107 MB、`db3/proposed/`、大 dump）
      → `..\\physics-learn-archive\\`
  **C** 归档脚本既定的根目录残渣 → 仓库内 `_archive/`（随后加进 `.gitignore` ✓）
  **D** 删除 `__pycache__`（可再生 ✓；打印清单后删 ✓）

用法：
    python tools/tidy_for_release.py            # 干跑（默认）
    python tools/tidy_for_release.py --apply    # 执行
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_NOTES = _ROOT.parent / "physics-learn-notes"
_ARCH = _ROOT.parent / "physics-learn-archive"

# ---- A：明确的第三方书籍/论文笔记（书名或论文名可辨 ✓；其余留在原地 ✓）----
THIRD_PARTY = [
    "Cover_Thomas_入门总结.md",
    "Elements of Causal Inference.md",
    "Information Geometry and Its Applications.md",
    "MacKay_入门总结.md",
    "Path Integrals in Quantum Mechanics.md",
    "Physics, Topology, Logic and Computation.md",
    "Spiking Neuron Models.md",
    "Statistical Physics of Fields.md",
    "The Free-Energy Principle.md",
    "Thermodynamics of Prediction.md",
    "Tishby & Zaslavsky — Deep Learning and the Information Bottleneck "
    "Principle (2015).md",
]
# ---- 含糊项：可能是教科书摘要 ⟹ **不动**，列入待确认 ✓ ----
AMBIGUOUS = [
    "quantum measurement.md", "four fundamental forces.md",
    "gravity and superconductivity.md", "light speed and entanglement.md",
    "dimensional analysis.md", "computational examples.md",
    "analysis theory.md", "information thermodynamics.md",
]
# ---- B：体积大头（仓库外归档，保留 ✓）----
BIG = [
    ("db2/db2.jsonl.bak-20261007-153254", "db2"),
    ("db2/db2.jsonl.bak-20261007-153801", "db2"),
    ("db2/db2.jsonl.bak-unify-161401", "db2"),
    ("db2/db2.jsonl.filled-unified", "db2"),
    ("db2/db2.jsonl.recompute-attempt", "db2"),
    ("a_network/landau_pred_dump.jsonl", "dumps"),
]
BIG_DIRS = [("db3/proposed", "db3-proposed")]
# ---- C：脚本既定的根目录残渣 → 仓库内 `_archive/` ----
JUNK_GLOBS = ["*.log", "io_test.*", "grid_check.*", "slow_check.*",
              "single_sys_test.*", "encoder_test_result.txt",
              "new_encoders_test.txt", "*_report.json",
              "v20_runs.csv", "v27_runs.csv", "v28_runs.csv"]
# ---- C2：第二轮定向清理（2026-10-02，第 208 条续）——**显式文件名** ✓ ----
#   这些是**探测/测试产物**（含我本轮留下的 `acl-report-*.jsonl` ✗）⟹ 归入 `_archive/` ✓
EXTRA_JUNK = [
    "acl-report-0feacdf7f8794e24ab8fcd8d8395d0d2.jsonl",
    "batch_summary.json", "full_test_result.txt",
    "pipeline_test_result.txt", "review_test_result.txt",
    "validate_test_result.txt", "laplace_resonator_landau_test.txt",
    "sandbox_probe.txt", "status.txt", "_gpu_probe.txt", "_gpu_time.txt",
    "cov_c.csv",
]


def main() -> int:
    apply = "--apply" in sys.argv
    print("=" * 74)
    print("开源前仓库整理（%s）" % ("**执行**" if apply else "干跑（默认）"))
    print("=" * 74)

    # ---------- A ----------
    print("\n【A】移出**明确的**第三方笔记 → %s" % _NOTES)
    n_a = 0
    for name in THIRD_PARTY:
        src = _ROOT / "md" / name
        if src.exists():
            n_a += 1
            print("   %s %s" % ("移动" if apply else "将移动", name))
            if apply:
                _NOTES.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(_NOTES / name))
        else:
            print("   （不存在，跳过）%s" % name)
    print("   ⟹ %d 个" % n_a)
    print("\n   **留在原地待你确认**（疑似教科书摘要，不替你做版权判断 ✓）：")
    for name in AMBIGUOUS:
        if (_ROOT / "md" / name).exists():
            print("     · md/%s" % name)

    # ---------- B ----------
    print("\n【B】移出体积大头 → %s（**保留不删** ✓）" % _ARCH)
    tot = 0.0
    for rel, sub in BIG:
        src = _ROOT / rel
        if src.exists():
            mb = src.stat().st_size / 1e6
            tot += mb
            print("   %s %s（%.1f MB）" % ("移动" if apply else "将移动",
                                        rel, mb))
            if apply:
                dst = _ARCH / sub
                dst.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst / src.name))
        else:
            print("   （不存在）%s" % rel)
    for rel, sub in BIG_DIRS:
        src = _ROOT / rel
        if src.exists():
            mb = sum(f.stat().st_size for f in src.rglob("*")
                     if f.is_file()) / 1e6
            tot += mb
            print("   %s %s/（%d 个文件，%.1f MB）"
                  % ("移动" if apply else "将移动", rel,
                     len(list(src.rglob("*"))), mb))
            if apply:
                dst = _ARCH / sub
                dst.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst / src.name))
    print("   ⟹ 合计约 **%.0f MB** 移出仓库 ✓" % tot)

    # ---------- C ----------
    print("\n【C】根目录残渣 → 仓库内 `_archive/`")
    junk: list[Path] = []
    for pat in JUNK_GLOBS:
        junk.extend(sorted(p for p in _ROOT.glob(pat) if p.is_file()))
    junk.extend(p for n in EXTRA_JUNK
                if (p := _ROOT / n).is_file())
    junk = sorted(set(junk))
    for p in junk:
        print("   %s %s（%.2f MB）"
              % ("归档" if apply else "将归档", p.name,
                 p.stat().st_size / 1e6))
    if apply and junk:
        (_ROOT / "_archive").mkdir(exist_ok=True)
        for p in junk:
            shutil.move(str(p), str(_ROOT / "_archive" / p.name))
    print("   ⟹ %d 个" % len(junk))

    # ---------- D ----------
    print("\n【D】删除 `__pycache__`（可再生字节码 ✓）")
    pyc = sorted(p for p in _ROOT.rglob("__pycache__") if p.is_dir())
    print("   共 %d 个目录（例如 %s）"
          % (len(pyc), ", ".join(str(p.relative_to(_ROOT))
                                for p in pyc[:4])))
    if apply:
        for p in pyc:
            shutil.rmtree(p, ignore_errors=True)
        print("   已删除 ✓")
    else:
        print("   （干跑：未删）")

    print("\n" + "=" * 74)
    if apply:
        print("整理完成 ✓ 建议随后：`python run_system.py`（应仍全绿）+ `python "
              "tests/test_invariants.py` ✓")
    else:
        print("（干跑结束：**未做任何改动** ✓；加 `--apply` 执行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
