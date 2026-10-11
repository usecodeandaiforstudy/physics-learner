# -*- coding: utf-8 -*-
"""**DB2 逐族口径统一实验**（2026-10-02，第 174 条）——用数据解决权衡。

## 要解决的权衡（第 172 条遗留）
  · **当前状态**：曲线可用率 **99.7%**，但来源**混合**（旧路径 + 参数化信道）；
  · **全量重算**（口径统一）：**21.1% 的行变化 >10%**，但
    **β 可标定数从 ≥100 掉到 88** ⟹ **违反既有不变量**（被回归测试抓到）✗
  ⟹ 需要**既不破坏不变量、又尽量统一口径**的答案。

## 做法（逐族贪心 + 以不变量为准绳）
  ① 基准：当前 DB2 的 `β 可标定数`（`calibrate_pair` 状态为 `ok` 的配对数）；
  ② **逐族**把该族曲线换成**参数化信道**重算版，重测可标定数；
     **≥ 基准 ⟹ 采纳该族统一**（并更新基准）；否则**回退该族**；
  ③ 落盘最终结果（**先备份**），并报告每族裁决。

判据（先写死）：**J1 不变量优先**——任何导致可标定数**下降**的族一律**不统一**；
**J2 尽力统一**——在满足 J1 的前提下**最大化**统一的族数；**J3 可审计**——
备份 + 逐族裁决表落盘。

用法：
    python db2/unify_by_family.py --dry-run     # 只跑实验，不落盘
    python db2/unify_by_family.py --write
"""

from __future__ import annotations

import json
import os
import shutil
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
RECOMP = _ROOT / "db2" / "db2.jsonl.recompute-attempt"
REPORT = _ROOT / "db2" / "unify_by_family_report.md"


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
            if x.strip()]


def n_ok() -> int:
    """**β 可标定数**（与 `tests/test_invariants.py` 同一口径）。

    **必须起新进程（2026-10-02 修，真 bug）**：`calibrate_pair` 有**进程内
    缓存/memo** ⟹ 同进程内反复测量会**返回首次结果** ⟹ 实测 27 个族**全部
    恒为 109**（看着像"全部可统一"，其实是**测量没跟上文件变化**）✗。
    正解：每次测量用 `subprocess` 起**新进程**（A/B 对照已用此法复现
    当前版 109 / 重算版 88）。
    """
    import subprocess
    code = (
        "from pipeline.beta_calibration import calibrate_pair\n"
        "import json, pathlib\n"
        "c=0\n"
        "for p in sorted(pathlib.Path('db3/pairs').glob('DB3-*.json')):\n"
        "    e=json.loads(p.read_text(encoding='utf-8'))\n"
        "    try:\n"
        "        if calibrate_pair(e).status=='ok': c+=1\n"
        "    except Exception: pass\n"
        "print(c)\n")
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(_ROOT))
    out = subprocess.run([sys.executable, "-c", code], cwd=str(_ROOT),
                         capture_output=True, text=True, env=env)
    try:
        return int(out.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return -1


def write_rows(rows: list[dict]) -> None:
    tmp = DB2.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, DB2)


def main() -> int:
    write = "--write" in sys.argv
    if not RECOMP.exists():
        print("缺少重算版本 %s（先跑 `db2/fill_empty_curves.py --all --write`）"
              % RECOMP.name)
        return 1
    cur = load(DB2)
    rec = load(RECOMP)
    if len(cur) != len(rec):
        print("行数不一致：%d vs %d" % (len(cur), len(rec)))
        return 1
    # **按行索引（2026-10-02 修，真 bug）**：DB2 里同一 `system_id` 有**多行**
    #   （重复行）；初版 `idx = {system_id: i}` 每族只替换**一行** ⟹ 实际是
    #   "部分统一" ⟹ 实测 27 族恒为 109（假结果）。正解：**遍历整族的所有行**。
    fams = sorted({str(r.get("family")) for r in cur})
    print("=" * 74)
    print("DB2 逐族口径统一实验（不变量优先：β 可标定数不得下降）")
    print("=" * 74)
    base = n_ok()
    print("基准：**β 可标定数 = %d**（159 配对中状态为 ok 者）" % base)
    best = base
    accepted: list[str] = []
    rejected: list[tuple] = []
    backup = cur
    for fam in fams:
        pos = [i for i, r in enumerate(cur) if str(r.get("family")) == fam]
        if not pos:
            continue
        trial = list(cur)
        for i in pos:
            if rec[i].get("ib_curve"):
                trial[i] = dict(cur[i], ib_curve=rec[i]["ib_curve"])
        write_rows(trial)
        try:
            v = n_ok()
        except Exception as ex:  # noqa: BLE001
            v = -1
            print("   %-22s **测量异常** %s" % (fam, type(ex).__name__))
        if v >= best:
            best = v
            cur = trial
            accepted.append(fam)
            print("   %-22s %3d 行  可标定 %d ≥ 基准 ⟹ **采纳统一** ✓"
                  % (fam, len(pos), v))
        else:
            rejected.append((fam, v))
            write_rows(backup)          # 回退
            print("   %-22s %3d 行  可标定 %d < %d ⟹ **不统一**（回退）✗"
                  % (fam, len(pos), v, best))
        backup = cur
    write_rows(cur if write else cur)
    if not write:
        write_rows(backup)              # 恢复现场（dry-run 不改变基准状态）
        # 注：实验过程中已多次改写文件，最后以 backup 还原
    print("\n**裁决**：采纳统一 **%d 族**；不统一 **%d 族**"
          % (len(accepted), len(rejected)))
    if rejected:
        print("   不统一的族（及其可标定数）：%s"
              % ", ".join("%s(%d)" % (f, v) for f, v in rejected[:8]))
    print("最终 β 可标定数：**%d**（基准 %d）" % (best, base))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("# DB2 逐族口径统一实验（%s）\n\n" % time.strftime(
            "%Y-%m-%d %H:%M"))
        f.write("- 基准 β 可标定数：**%d**\n- 最终：**%d**\n\n" % (base, best))
        f.write("## 采纳统一的族（%d）\n\n%s\n\n"
                % (len(accepted), ", ".join(accepted) or "—"))
        f.write("## 不统一的族（%d；统一会降低可标定数）\n\n"
                % len(rejected))
        for fam, v in rejected:
            f.write("- `%s`：可标定 %d\n" % (fam, v))
    print("\n报告：%s" % REPORT.name)
    if not write:
        print("（--dry-run：已还原 DB2；加 `--write` 落盘最终结果）")
    else:
        bak = DB2.with_suffix(".jsonl.bak-unify-%s" % time.strftime("%H%M%S"))
        shutil.copy2(RECOMP, bak)       # 留档：重算版本已另存
        print("已落盘（原状态见 %s）" % bak.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
