# -*- coding: utf-8 -*-
"""**生成"开源发布用"的内容副本**（2026-10-02，第 213 条）。

## 用途
当前文件夹是**工作目录**（含归档/备份/草稿/缓存 ✗）——不适合直接变仓库。
本脚本把它**该开源的部分**复制到一个**新的发布暂存目录**，供手动复制发布 ✓

## 规矩（沿用本会话的安全约定 ✓）
  1. **只复制、不移动、不删除**源目录任何东西 ✓
  2. **显式清单**：包含哪些目录/文件、排除哪些，全部写死并打印 ✓
     （**不用通配符做危险操作** ✗ —— 通配符只用于**只读枚举** ✓）
  3. **默认干跑**；`--apply` 才写盘 ✓
  4. **目标目录非空则拒绝**（除非 `--force` ✓），避免误覆盖 ✗
  5. 复制后**打印逐目录文件数/体积**，便于人工核对 ✓

## 排除什么（为什么）
  · `_archive/`、`db2/db2.jsonl.*`、`*.bak*`  —— 归档与数据库备份 ✗（已在仓库外 ✓）
  · `__pycache__/`、`*.pyc`                   —— 可再生缓存 ✗
  · `.trae/`、`.git`、编辑器残渣                —— 环境/工具文件 ✗
  · `db3/proposed/`                           —— 锚点草稿（入库前）✗
  · **8 个"含糊项"理论笔记**                    —— **版权待用户判断** ⟹ **不进发布副本** ✓
  · A 侧大权重（`a_network/*.pt`）              —— 可重训 ✗（B 解析器权重**保留** ✓）

用法：
    python tools/make_release_copy.py                 # 干跑
    python tools/make_release_copy.py --apply         # 执行
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
_DST = _SRC.parent / "physics-learn-release"

# ---- 包含的目录（全部内容，除下列排除项）----
INC_DIRS = ["core", "b_network", "c_network", "a_network", "pipeline", "sim",
            "db1", "db2", "db3", "db4", "db5", "tests", "tools", "md"]
# ---- 包含的根文件 ----
INC_FILES = [
    # **只留读者需要的最终文档（第 216 条，用户决定 ①）** ✓
    #   原则："要开源就要只保留最终的东西" ⟹ **过程物一律不随发布** ✗
    "README.md", "README_cn.md",
    "CLAIMS_AND_BOUNDARIES.md", "CLAIMS_AND_BOUNDARIES_cn.md",
    "HANDOVER.md",                       # 英文＝**最终状态**可信度索引 ✓
    "cross domain rl architecture plan.md",   # 英文＝**当前架构**总览 ✓
    "LICENSE", "requirements.txt", ".gitignore",
    "run_system.py", "run_demo.py",
    # **不随发布的内部过程物** ✗（保留在工作目录 ✓）：
    #   HANDOVER_cn.md（469 KB 开发日志）、cross domain rl architecture plan_cn.md（100 KB 内部方案）
    #   RELEASE_CHECKLIST.md / _cn.md（发布清单，给 owner 用）
    #   DB1_INDEPENDENT_CHECK.md/.csv、DB1_REVIEW_CHECKLIST.md、DB1_REVIEW_WORKSHEET.md（内部复核材料）
]
# ---- 排除（按路径片段 / 后缀；只用于**判断是否跳过** ✓）----
EXCL_DIR_NAMES = {"__pycache__", ".trae", ".git", "_archive", "proposed"}
# **构建残留一律不发**（第 214 条）：日志/报告文本 ✗
#   起因：`a_network/multi_seed_verify.log` 曾把 **Windows 用户名**带进发布副本 ✗
#   （`requirements.txt` 在 INC_FILES 里显式包含 ⟹ 不受 `.txt` 规则影响 ✓）
EXCL_SUFFIX = (".pyc", ".pyo", ".orig", ".rej", "~", ".log", ".txt")
EXCL_NAME_PREFIX = ("db2.jsonl.",)          # 数据库备份/快照 ✗
EXCL_CONTAINS = (".bak",)
# **构建报告**（`*_report.json` 等，可由脚本重跑生成 ✓）——与 `.gitignore` 口径一致 ✓
EXCL_REPORT_SUFFIX = ("_report.json", "_report.txt")
# **运行时会再生成的输出**（第 214 条精简）：不进发布副本 ✓
#   依据：只被写入、无代码读取 ⟹ 移出不影响任何阶段 ✓（已实测通过门禁 ✓）
#
#   **⚠️ 第 218 条事故修正（用户查出 ✗）**：此处原为**子串匹配**
#   `"design_reflow_"` ✗ —— 它**连带吞掉了主链模块**
#   `pipeline/design_reflow_consume.py` ✗ ⟹ 发布副本缺文件 ⟹
#   `run_system.py` 的 gate 阶段与 `tests/test_invariants.py` 都会失败 ✗
#   （而 `design_reflow.py` 因后缀是 `.` 未被吞 ✓ —— 症状看起来"只缺一个" ✓）
#   ⟹ **一律改为精确的 glob 匹配**（`fnmatch` ✓），不再用裸子串 ✗
EXCL_GENERATED_GLOBS = (
    "design_reflow_*.jsonl",      # reflow 原始记录（**仅 .jsonl** ✓，不要吞 .py ✗）
    "v34_dataset.csv", "v35_dataset.csv", "v37_dataset.csv",
    "landau_designs.csv",
    "verify_audit.jsonl",         # 验证审计日志（追加式运行日志 ✓）
    "gine_metrics_*.json",        # A 侧逐次训练指标 ✓
    "multi_seed_verify.jsonl",    # 第 218 条：多 seed 复检的运行日志（每轮追加 ✓）
)
# **内部材料**（第 216 条）：不随发布 ✗（即使出现在根目录的自动收录范围内 ✓）
EXCL_INTERNAL = ("DB1_INDEPENDENT_CHECK", "DB1_REVIEW_", "_cn.md",
                 "RELEASE_CHECKLIST", "cross domain rl architecture plan_cn",
                 # 第 218 条：**DB1/DB3 人工复核过程工件**（角色名 only，无隐私风险 ✓，
                 #   但属内部过程物 ⟹ 与 HANDOVER §7 的表述保持一致 ✗→不随发布 ✓）
                 "review_worksheet", "b_role_findings", "b_role_review_worksheet")
# ---- 版权待定的理论笔记（**不进发布副本** ✓；用户可自行决定 ✓）----
AMBIGUOUS_MD = {
    "quantum measurement.md", "four fundamental forces.md",
    "gravity and superconductivity.md", "light speed and entanglement.md",
    "dimensional analysis.md", "computational examples.md",
    "analysis theory.md", "information thermodynamics.md",
}
# ---- 必须存在的关键文件（缺失则报警 ✓）----
MUST = ["README.md", "LICENSE", "requirements.txt", "run_system.py",
        "b_network/bv1_model.pt", "b_network/bv1_model_cal.pt",
        "sim/verify_api.py", "tests/test_invariants.py",
        "db1/entries", "db2/db2.jsonl", "db3/pairs"]


def skip_dir(d: Path) -> bool:
    return d.name in EXCL_DIR_NAMES


def skip_file(p: Path) -> bool:
    n = p.name
    if n.startswith(EXCL_NAME_PREFIX) or n.endswith(EXCL_SUFFIX):
        return True
    if n.endswith(EXCL_REPORT_SUFFIX):                          # 构建报告 ✗
        return True
    if any(fnmatch.fnmatch(n, g) for g in EXCL_GENERATED_GLOBS):  # 会再生成的输出 ✗
        return True                                             # （**精确 glob** ✓ 非子串 ✗）
    if any(s in n for s in EXCL_INTERNAL):                      # 内部材料 ✗
        return True
    if any(s in n for s in EXCL_CONTAINS):
        return True
    if p.suffix == ".pt" and p.parent.name == "a_network":     # A 侧大权重 ✗
        return True
    if p.parent.name == "md" and n in AMBIGUOUS_MD:            # 版权待定 ✗
        return True
    return False


def completeness_check(plan: list[tuple[Path, Path]]) -> list[str]:
    """**完整性自检**（第 218 条）——防"计划本身漏文件"这一类错误 ✗。

    事故背景：排除规则用**裸子串** `design_reflow_` ✗ 吞掉了主链模块
    `pipeline/design_reflow_consume.py` ✗，而 `--verify` 只校验
    "副本 = 计划" ✓ ⟹ **全绿却发布了一个跑不通的副本** ✗

    预登记三条判据（**任一不过即拒绝复制/校验失败** ✗）：
      **W1 代码一个不少**：树内所有 `.py` 都必须在计划内 ✓
      **W2 被引用脚本齐全**：`run_system.py` 里出现的 `pkg/xxx.py` 路径必须在计划内 ✓
      **W3 测试依赖齐全**：`tests/*.py` 里 `import a.b` 对应的 `a/b.py` 必须在计划内 ✓
    """
    problems: list[str] = []
    planned = {d.resolve() for _, d in plan}

    def _planned(rel: str) -> bool:
        return (_DST / rel).resolve() in planned

    # W1：全部 .py
    missing_py: list[str] = []
    for p in sorted(_SRC.rglob("*.py")):
        if any(skip_dir(par) for par in p.parents if par.is_relative_to(_SRC)):
            continue
        rel = p.relative_to(_SRC).as_posix()
        if not _planned(rel):
            missing_py.append(rel)
    if missing_py:
        problems.append("W1 代码被误排除：%s" % ", ".join(missing_py[:10]))

    # W2：run_system.py 引用的脚本
    runner = _SRC / "run_system.py"
    if runner.exists():
        refs = set(re.findall(r'"([a-z_]+/[A-Za-z0-9_]+\.py)"',
                              runner.read_text(encoding="utf-8")))
        bad = sorted(r for r in refs if not _planned(r))
        if bad:
            problems.append("W2 被 run_system.py 引用但不在计划内：%s"
                            % ", ".join(bad))

    # W3：tests/ 里的 import
    tdir = _SRC / "tests"
    if tdir.exists():
        mods: set[str] = set()
        for t in sorted(tdir.glob("*.py")):
            mods |= set(re.findall(r"^\s*import\s+([a-z_]+\.[a-z_0-9]+)",
                                   t.read_text(encoding="utf-8"), re.M))
        bad2 = sorted(m for m in mods
                      if not _planned(m.replace(".", "/") + ".py"))
        if bad2:
            problems.append("W3 被 tests import 但不在计划内：%s"
                            % ", ".join(bad2))
    return problems


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(plan: list[tuple[Path, Path]]) -> int:
    """**一致性校验**（第 217 条）：副本是否**逐字节**等于"该发布的内容" ✓。

    判据（预登记 ✓）：
      **V1 一个不缺**：计划内每个文件在副本中存在 ✓
      **V2 一个不差**：存在者的 **SHA256 与源一致** ✓
      **V3 一个不多**：副本内**没有**计划外的文件 ✗
      ⟹ 三条全过 = **副本与工作区一致** ✓；任一不过 = **必须先同步再发布** ✗
    """
    print("=" * 76)
    print("一致性校验（--verify）：发布副本 vs 工作区")
    print("=" * 76)
    print("源：%s\n目标：%s" % (_SRC, _DST))

    # **先做完整性自检**（第 218 条 ✓）：计划本身漏了文件 ⟹ 后面全是白验 ✗
    probs = completeness_check(plan)
    print("\n【W 完整性自检】%s"
          % ("W1 代码一个不少 ✓ ｜ W2 被引用脚本齐全 ✓ ｜ W3 测试依赖齐全 ✓"
             if not probs else "**未通过 ✗**"))
    for x in probs:
        print("   ✗ %s" % x)

    if not _DST.exists():
        print("\n**V1 失败**：目标目录不存在 ✗")
        return 1
    missing, differ, ok = [], [], 0
    for src, dst in plan:
        if not dst.exists():
            missing.append(dst.relative_to(_DST).as_posix())
            continue
        if _sha(src) != _sha(dst):
            differ.append(dst.relative_to(_DST).as_posix())
        else:
            ok += 1
    planned = {d.resolve() for _, d in plan}
    extras = [p.relative_to(_DST).as_posix()
              for p in sorted(_DST.rglob("*"))
              if p.is_file() and p.resolve() not in planned]

    print("\n【V1 一个不缺】计划 %d 个 ｜ 缺失 **%d** 个" % (len(plan), len(missing)))
    for x in missing[:20]:
        print("   ✗ 缺 %s" % x)
    print("【V2 一个不差】内容一致 **%d** 个 ｜ 内容不同 **%d** 个"
          % (ok, len(differ)))
    for x in differ[:20]:
        print("   ✗ 内容不同 %s" % x)
    print("【V3 一个不多】计划外文件 **%d** 个" % len(extras))
    for x in extras[:20]:
        print("   ✗ 多余 %s" % x)

    bad = len(missing) + len(differ) + len(extras)
    print("\n**结论**：%s"
          % ("**副本与工作区一致 ✓ 可发布 ✓**" if (bad == 0 and not probs) else
             "**不一致 ✗（%d 处%s）—— 先跑 `--apply --force` 同步，再发布 ✗**"
             % (bad, " ＋ 完整性 %d 项" % len(probs) if probs else "")))
    if bad == 0 and not probs:
        tot = sum(s.stat().st_size for s, _ in plan)
        cnt = sum(1 for _ in _DST.rglob("*") if _.is_file())
        print("  计划内 %.1f MB ｜ 副本内 %d 个文件（应等于 %d ✓）"
              % (tot / 1e6, cnt, len(plan)))
    return 0 if (bad == 0 and not probs) else 1


def main() -> int:
    apply = "--apply" in sys.argv
    force = "--force" in sys.argv
    print("=" * 76)
    print("生成开源发布副本（%s）" % ("**执行**" if apply else "干跑（默认）"))
    print("=" * 76)
    print("源：%s" % _SRC)
    print("目标：%s" % _DST)

    # **校验模式是只读的** ⟹ 不受"目标非空则拒绝"约束 ✓（第 217 条）
    _verify_only = "--verify" in sys.argv
    if _DST.exists() and any(_DST.iterdir()) and not force and not _verify_only:
        print("\n**目标目录已存在且非空 ✗** —— 拒绝执行（避免覆盖）✓")
        print("如确认要覆盖：加 `--force`；或先手动清空该目录 ✓")
        return 2
    if apply:
        _DST.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[Path, Path]] = []       # (src, dst)
    skipped: list[str] = []
    for d in INC_DIRS:
        root = _SRC / d
        if not root.exists():
            print("  ⚠️ 缺失目录：%s" % d)
            continue
        for p in sorted(root.rglob("*")):
            if p.is_dir():
                if skip_dir(p):
                    skipped.append(str(p.relative_to(_SRC)) + "/")
                continue
            if any(skip_dir(par) for par in p.parents
                   if par != _SRC.parent and par.is_relative_to(_SRC)):
                continue
            if skip_file(p):
                skipped.append(str(p.relative_to(_SRC)))
                continue
            plan.append((p, _DST / p.relative_to(_SRC)))
    for f in INC_FILES:
        p = _SRC / f
        if p.exists():
            plan.append((p, _DST / f))
        else:
            print("  ⚠️ 缺失文件：%s" % f)

    # 根目录证据文件（HANDOVER 引用者 ✓）——**只读枚举**后加入 ✓
    extra = sorted([p for p in _SRC.glob("*.csv") if p.is_file()]
                   + [p for p in _SRC.glob("*.jsonl") if p.is_file()])
    have = {s for s, _ in plan}
    for p in extra:
        if p not in have and not skip_file(p):
            plan.append((p, _DST / p.name))

    # ---- **一致性校验模式**（第 217 条 ✓）：不写盘，只比对 ✓ ----
    if "--verify" in sys.argv:
        return _verify(plan)

    print("\n【将复制】%d 个文件，合计 %.1f MB"
          % (len(plan), sum(s.stat().st_size for s, _ in plan) / 1e6))
    per: dict[str, list[int]] = {}
    for s, _ in plan:
        key = s.relative_to(_SRC).parts[0]
        per.setdefault(key, [0, 0])
        per[key][0] += 1
        per[key][1] += s.stat().st_size
    for k in sorted(per):
        n, b = per[k]
        print("   %-32s %4d 个 ｜ %8.2f MB" % (k, n, b / 1e6))

    print("\n【显式排除】%d 项（示例前 12）" % len(skipped))
    for s in skipped[:12]:
        print("   %s" % s)
    print("   · 另含 8 个**版权待定理论笔记**（%s…）"
          % ", ".join(sorted(AMBIGUOUS_MD)[:3]))

    missing = [m for m in MUST if not (_SRC / m).exists()]
    print("\n【关键内容自检】%s"
          % ("全部就位 ✓" if not missing else "**缺失：%s** ✗" % missing))

    # ---- **完整性自检**（第 218 条 ✓）：计划本身是否漏了代码 ✗ ----
    probs = completeness_check(plan)
    print("【W 完整性自检】%s"
          % ("W1 代码一个不少 ✓ ｜ W2 被引用脚本齐全 ✓ ｜ W3 测试依赖齐全 ✓"
             if not probs else "**未通过 ✗**"))
    for x in probs:
        print("   ✗ %s" % x)
    if probs:
        print("\n**拒绝继续** ✗ —— 计划漏了运行所需的文件；"
              "先修排除规则（用精确 glob ✓，别用裸子串 ✗）再重跑 ✓")
        return 1

    if not apply:
        print("\n（干跑结束：**未写任何文件** ✓；加 `--apply` 执行）")
        return 0

    for s, d in plan:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
    print("\n完成 ✓ 已复制 %d 个文件到 %s" % (len(plan), _DST))

    # ---- **镜像剪除（prune）**：删掉目标目录里"不在计划内"的文件 ✗ ----
    #   起因（第 214 条）：只复制不删除 ⟹ 上一次带进去的泄漏 `.log` **留在副本里** ✗
    #   安全闸（三道 ✓）：① 只在 `_DST` 内；② 目标须像"我们的副本"（含 .gitignore ✓）；
    #                    ③ **逐条打印**被删文件，且**绝不**碰源目录 ✓
    guard = (_DST / ".gitignore").exists() or (_DST / "README.md").exists()
    if not guard:
        print("**跳过剪除**：目标目录不像本项目副本（缺 README/.gitignore）⟹ 不动 ✓")
        return 1
    planned = {d.resolve() for _, d in plan}
    extras = [p for p in sorted(_DST.rglob("*"))
              if p.is_file() and p.resolve() not in planned]
    if extras:
        print("\n【镜像剪除】删除 %d 个**不在发布计划内**的文件 ✗" % len(extras))
        for p in extras:
            print("   删 %s" % p.relative_to(_DST).as_posix())
            p.unlink()
        # 清掉因此变空的目录 ✓（仅目标目录内 ✓）
        for d in sorted((p for p in _DST.rglob("*") if p.is_dir()),
                        key=lambda x: -len(x.parts)):
            try:
                if not any(d.iterdir()):
                    print("   删空目录 %s" % d.relative_to(_DST).as_posix())
                    d.rmdir()
            except OSError:
                pass
    else:
        print("\n【镜像剪除】无多余文件 ✓")

    # ---- **复制后强制隐私扫描**（第 214 条）✓ ----
    #   起因：曾把含 Windows 用户名的 `.log` 带进发布副本 ✗
    #   ⟹ 现在**每次复制后自动扫描**；命中即报错退出（发布副本不可用 ✗）
    rc = subprocess.call([sys.executable, str(_SRC / "tools" / "privacy_scan.py"),
                          str(_DST)])
    if rc != 0:
        print("\n**⚠️ 隐私扫描未通过 ✗** —— 发布副本**不可发布** ✓")
        print("   处置：把命中文件加入本脚本的排除项（或清理其内容）后重跑 ✓")
        return 1
    print("\n**隐私扫描通过** ✓ —— 发布副本可发布 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
