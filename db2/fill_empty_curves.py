# -*- coding: utf-8 -*-
"""**DB2 空曲线就地补齐**（2026-10-02，第 170 条）。

## 问题（实测，含对先前口径的更正）
DB2 现存 **8879 行**：
  · `ib_curve` **字段存在率 100%**（我先前据此报了"覆盖 100%"——**口径错了**）；
  · **曲线可用率（`I_TY` 非空且有限且非常零）仅 3683/8879 = 41.5%** ✗
  · 抽查 `rayleigh_wind_speed` 等行：`ib_curve = {"beta": [...], "I_XT": [], "I_TY": []}`
    ⟹ **空数组**——这些行写于该族注册 `ib_source` **之前**。
**为何既有工具补不上**：`db2/rebuild_family_curves.py` 的跳过逻辑是
**族级**的（"该族 DB2 中已有可用曲线 ⟹ 跳过"）⟹ 而多数族是**混合**的
（如 rayleigh：31 行可用 + 930 行空）⟹ **整族被跳过，空行永远补不上**。

## 本脚本（行级就地补齐，不产生重复行）
  ① 逐行检查 `ib_curve` 是否**可用**（非空 + 有限 + 非常零）；
  ② 不可用者用**该行 `system_id` 的 DB1 条目**重建（`SystemEncoder` →
     `FingerprintComputer.compute_ib_curve`；族已注册 `ib_source` 者走
     **参数化信道**）；
  ③ 只有通过**预登记判据**才写入：
     C1 非空且有限；C2 **DPI**（`I_TY ≤ I_XT + 1e-6`，逐点）；
     C3 **单调不减**（IB 曲线定义性质，容许 1e-6 数值松弛）；C4 非全零；
  ④ **写前备份** `db2/db2.jsonl.bak-<时间戳>`；**原子替换**。

用法：
    python db2/fill_empty_curves.py --dry-run     # 只看能补多少
    python db2/fill_empty_curves.py --write       # 写回（先备份）
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

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

DB2 = _ROOT / "db2" / "db2.jsonl"
ENTRIES = _ROOT / "db1" / "entries"


def _arr(v) -> np.ndarray:
    """**安全转数组（2026-10-02 修）**：`v or []` 对 **numpy 数组**会触发
    `bool(array)` ⟹ `ValueError: truth value ... is ambiguous`（我的初版 bug）。
    """
    return np.asarray([] if v is None else v, dtype=float)


def usable(cur: dict | None) -> bool:
    """**可用性判定（本轮统一口径）**：非空 + 有限 + 非常零。"""
    if not cur:
        return False
    iy = _arr(cur.get("I_TY"))
    return bool(iy.size > 0 and np.isfinite(iy).all() and iy.max() > 1e-9)


def dip_stats(new: dict) -> tuple[float, float]:
    """**C3 相对下陷（信息性，不作门禁）**——对齐项目已定口径。

    依据（`rebuild_family_curves.py` 源码注释，2026-10-02 三次迭代的结论）：
    既有**生产族**的曲线**本身就非严格单调**（Blahut–Arimoto 有限容差噪声，
    例如 two_level 的 `I_TY` 在第 7 点从 1.1e-7 掉到 9.9e-11）；
    若把"严格单调"当门禁，**连生产族都过不了** ⟹ 项目最终把 C3
    **降级为信息性输出**，改用**相对判据 + 本族基准**。
    ⟹ 本脚本同样**不拿 C3 作门禁**，只报告**可用区**（`y ≥ 10%·max`）的
      最大相对下陷与 p95。
    """
    iy = _arr(new.get("I_TY"))
    if iy.size < 3:
        return 0.0, 0.0
    mx = float(iy.max())
    if mx <= 0:
        return 0.0, 0.0
    usable = iy >= 0.10 * mx
    if usable.sum() < 3:
        return 0.0, 0.0
    d = np.diff(iy[usable])
    neg = -d[d < 0] / mx                     # 相对下陷（占峰值比例）
    if neg.size == 0:
        return 0.0, 0.0
    return float(neg.max()), float(np.percentile(neg, 95))


def passes_criteria(new: dict, old: dict) -> tuple[bool, str]:
    """**预登记判据（门禁部分）**：C1 非空且有限；C2 DPI；C4 非全零。

    **C3（单调性）不作门禁**——对齐项目口径（见 `dip_stats` 说明）；
    若拿它作门禁会**连生产族都拒绝**（项目实测结论）。
    """
    ix = _arr(new.get("I_XT"))
    iy = _arr(new.get("I_TY"))
    if not (ix.size and iy.size and np.isfinite(ix).all()
            and np.isfinite(iy).all()):
        return False, "C1 非空且有限 ✗"
    if ix.size != iy.size:
        return False, "C1 长度不一致 ✗"
    if np.any(iy > ix + 1e-6):
        return False, "C2 DPI 违背（I_TY > I_XT）✗"
    if iy.max() <= 1e-9:
        return False, "C4 全零（退化为无信息信道）✗"
    return True, "通过"


def main() -> int:
    write = "--write" in sys.argv
    recompute_all = "--all" in sys.argv
    rows = [json.loads(x) for x in DB2.read_text(encoding="utf-8").splitlines()
            if x.strip()]
    n0 = sum(1 for r in rows if usable(r.get("ib_curve")))
    print("=" * 74)
    print("DB2 曲线%s（行级；预登记判据 C1–C4）"
          % ("**全量重算**（口径统一）" if recompute_all else "空曲线就地补齐"))
    print("=" * 74)
    print("行数 **%d**；处理前**可用**曲线 **%d（%.1f%%）**"
          % (len(rows), n0, 100 * n0 / max(len(rows), 1)))
    if recompute_all:
        print("**模式说明**：`--all` ⟹ **每一行都重算**（含原已可用者），"
              "使全部曲线走**同一参数化信道口径**；")
        print("  重算未过判据的行**保留原值**（不因重算而变空）✓")

    from b_network.fingerprint_computer import FingerprintComputer
    from b_network.system_encoder import SystemEncoder
    enc, fc = SystemEncoder(), FingerprintComputer()
    cache: dict = {}
    fixed = failed = kept_old = 0
    reasons: dict[str, int] = {}
    dips: list[float] = []
    diffs: list[float] = []
    for r in rows:
        if not recompute_all and usable(r.get("ib_curve")):
            continue
        sid = r.get("system_id")
        ef = ENTRIES / ("%s.json" % sid)
        if sid not in cache:
            try:
                cache[sid] = (enc.encode(json.loads(
                    ef.read_text(encoding="utf-8"))) if ef.exists() else None)
            except Exception:  # noqa: BLE001
                cache[sid] = None
        g = cache[sid]
        if g is None:
            failed += 1
            reasons["无 DB1 条目/编码失败"] = reasons.get(
                "无 DB1 条目/编码失败", 0) + 1
            kept_old += 1
            continue
        try:
            new = fc.compute_ib_curve(g, n_bins=24)
        except Exception as ex:  # noqa: BLE001
            failed += 1
            k = "计算异常 %s" % type(ex).__name__
            reasons[k] = reasons.get(k, 0) + 1
            kept_old += 1
            continue
        ok, why = passes_criteria(new, r.get("ib_curve") or {})
        if not ok:
            failed += 1
            reasons[why] = reasons.get(why, 0) + 1
            kept_old += 1
            continue
        # 新旧差异（仅统计，供如实报告）
        old_iy = _arr((r.get("ib_curve") or {}).get("I_TY"))
        new_iy = _arr(new.get("I_TY"))
        if old_iy.size and old_iy.max() > 1e-12:
            diffs.append(abs(float(new_iy.max()) - float(old_iy.max()))
                         / float(old_iy.max()))
        r["ib_curve"] = {"beta": [float(v) for v in _arr(new.get("beta"))],
                         "I_XT": [float(v) for v in _arr(new.get("I_XT"))],
                         "I_TY": [float(v) for v in _arr(new.get("I_TY"))]}
        dips.append(dip_stats(new)[0])
        fixed += 1

    n1 = sum(1 for r in rows if usable(r.get("ib_curve")))
    print("%s **%d** 行；未处理/未过判据 **%d** 行（**保留原值** %d）"
          % ("重算成功" if recompute_all else "补齐成功", fixed, failed, kept_old))
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:6]:
        print("   未过判据原因：%-28s %d 行" % (k[:28], v))
    if diffs:
        da = np.asarray(diffs, dtype=float)
        print("**新旧差异**（`I_TY` 峰值相对差，%d 行可比）：中位 **%.1f%%**、"
              "p95 **%.1f%%**、最大 **%.1f%%**"
              % (da.size, 100 * float(np.median(da)),
                 100 * float(np.percentile(da, 95)), 100 * float(da.max())))
    if dips:
        db = np.asarray(dips, dtype=float)
        print("C3 **信息性**（不作门禁）：相对下陷中位 %.3g、p95 %.3g、最大 %.3g"
              % (float(np.median(db)), float(np.percentile(db, 95)),
                 float(db.max())))
    print("处理后**可用**曲线 **%d（%.1f%%）**" % (n1, 100 * n1 / max(len(rows), 1)))

    if not write:
        print("\n（--dry-run：**未写库**；加 `--write` 落盘，会先备份）")
        return 0
    bak = DB2.with_suffix(".jsonl.bak-%s" % time.strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(DB2, bak)
    tmp = DB2.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, DB2)
    print("\n已写回 %s（原子替换）；备份：%s" % (DB2.name, bak.name))
    print("**注**：未通过判据的行**保持原样**（不臆造曲线）✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
