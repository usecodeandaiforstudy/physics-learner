# -*- coding: utf-8 -*-
"""任务 3：保持（势垒）曲线与工况索引曲线（2026-10-02，第 110 条）。

**为什么这两类需要"自己的曲线"而不是"另一个公式"（关键纠正）**：
  - `E_req_nats`（势垒，如「自由能垒 ≥ 40 k_B T」）**量的是寿命/能量**，
    **不是**「保持了多少信息」。若按双态链的保持互信息算，
    `I_ret ≤ ln 2 = 0.693 nats` —— **永远够不到 3–80 nats 的需求**。
  - `n_environments_count`（适应 M 种工况）量的是**可区分的工况数**。
  ⟹ 正确做法：把**需求侧的量本身**作为曲线横轴，与**表示代价 I_XT** 配对：
    | 曲线 | 横轴（能力） | 纵轴（代价） | 需求侧目标 |
    |---|---|---|---|
    | **保持曲线** | `ΔF`（势垒，nats） | `I_XT` | `ΔF_target = E_req`（**直接可比**） |
    | **工况索引曲线** | `ln M_cap`（可区分工况数） | `I_XT` | `ln M_req`（**直接可比**） |
  `β = 1/(d能力/dI_XT)` —— 即「能力的边际信息代价」，正是 Lagrange 乘子的含义。

**口径（已采纳，可追溯）**：
  - `I_XT` 取 **DB2 各扫描点在 β=4 处的 I_XT**（现有可审计数据；**依赖 β=4
    这个占位值**——本任务的目的正是给出非占位 β，此依赖已在结论中标注）；
  - `ΔF` 取**族解析式**（landau 双阱：`ΔF = a²/4`，a<0；a≥0 无垒记 0）；
  - `ln M_cap` 取 **`I_TY_max`（曲线饱和值）作下界代理**——`I_TY ≤ H(Y)`，
    故 `exp(I_TY_max)` 是有效输出态数的**下界**（**标注为代理，非 H(Y)**）。

**预登记判据**：R1 曲线单调（ΔF/lnM 对 I_XT 单调升）；R2 解锁率；
  R3 与 param 类冲突数（**报告而不择优**）；R4 每点来源可审计。

用法（项目根目录）：
  python pipeline/extra_curves.py            # 扫 barrier/env 两类并报告
  python pipeline/extra_curves.py --show landau   # 打印某族曲线点
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

DB2 = _ROOT / "db2" / "db2.jsonl"
PAIRS = _ROOT / "db3" / "pairs"
BETA_OP = 4.0

# 族 → 势垒解析式（k_B T = 1 单位）。返回 ΔF（无垒记 0）。
FAMILY_BARRIER = {
    "landau": lambda a: max(0.0, float(a) ** 2 / 4.0),
}
BARRIER_NOTE = {
    "landau": "双阱 V=aφ²/2+φ⁴/4：ΔF = a²/4（a<0；a≥0 无垒）",
}

__all__ = ["retention_curve", "env_curve", "beta_at", "calibrate_pair"]


# 族 → 设计参数名（用于从 DB2 记录取「设计点」值）
FAMILY_DESIGN_PARAM = {"landau": "a", "elastic": "k", "ou": "theta",
                       "diffusion": "kappa"}


def _rows(family: str, limit: int | None = None) -> list[dict]:
    """取该族 DB2 行：设计值、I_XT@β_op、I_TY_max。

    **设计值的取法（2026-10-02 修，第 113 条）**：优先 `scan.scan_value`；
    无 scan 元数据时（如 `rebuild_family_curves.py` 按**条目**落库的行）回退到
    **`metadata.physical_params[设计参数]`**——否则这些行会被当作"无设计值"
    跳过 ⟹ 保持曲线点数不足（初版即如此：19 条新行全被跳过，保持曲线 0 点）。
    """
    out: list[dict] = []
    if not DB2.exists():
        return out
    pname = FAMILY_DESIGN_PARAM.get(family)
    with open(DB2, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("family") != family:
                continue
            c = rec.get("ib_curve")
            if not c or not c.get("beta"):
                continue
            b = np.asarray(c["beta"], dtype=float)
            x = np.asarray(c["I_XT"], dtype=float)
            y = np.asarray(c["I_TY"], dtype=float)
            scan = rec.get("scan") or {}
            sv = scan.get("scan_value")
            src = "scan.scan_value"
            if sv is None and pname:
                pp = (rec.get("metadata") or {}).get("physical_params") or {}
                if pname in pp:
                    sv = pp[pname]
                    src = "metadata.physical_params.%s" % pname
            try:
                sv = None if sv is None else float(sv)
            except (TypeError, ValueError):
                sv = None
                src = "无法解析"
            out.append({"system_id": rec.get("system_id"),
                        "scan_value": sv, "design_src": src,
                        "I_XT_at_op": float(np.interp(BETA_OP, b, x)),
                        "I_TY_max": float(y.max())})
            if limit and len(out) >= limit:
                break
    return out


def retention_curve(family: str) -> tuple[np.ndarray, np.ndarray, str]:
    """保持曲线：ΔF（能力） vs I_XT（代价）。返回 (dF, cost, note)。"""
    fn = FAMILY_BARRIER.get(family)
    if fn is None:
        return np.asarray([]), np.asarray([]), "族 %s 无解析势垒式（未登记）" % family
    pts = []
    for r in _rows(family):
        if r["scan_value"] is None:
            continue
        pts.append((fn(r["scan_value"]), r["I_XT_at_op"]))
    if not pts:
        return np.asarray([]), np.asarray([]), "无可用扫描点"
    pts.sort()
    dF = np.asarray([p[0] for p in pts])
    cost = np.asarray([p[1] for p in pts])
    return dF, cost, BARRIER_NOTE.get(family, "解析式已登记")


def env_curve(family: str) -> tuple[np.ndarray, np.ndarray, str]:
    """工况索引曲线：ln M_cap（能力，下界代理） vs I_XT（代价）。"""
    pts = [(r["I_TY_max"], r["I_XT_at_op"]) for r in _rows(family)]
    if not pts:
        return np.asarray([]), np.asarray([]), "无可用扫描点"
    pts.sort()
    lnM = np.asarray([p[0] for p in pts])
    cost = np.asarray([p[1] for p in pts])
    return lnM, cost, ("ln M_cap 取 I_TY_max **下界代理**（I_TY ≤ H(Y)；"
                       "非 H(Y) 本身）")


def beta_at(x: np.ndarray, cost: np.ndarray, target: float
            ) -> tuple[float | None, float | None, str]:
    """在（能力 x, 代价 cost）曲线上取 target 处的切线：β = 1/(dx/dcost)。"""
    if len(x) < 4:
        return None, None, "曲线点太少（%d）" % len(x)
    if target > float(x.max()) or target < float(x.min()):
        return None, float(x.max()), ("out_of_range（目标 %.4g 不在能力 ∈ "
                                      "[%.4g, %.4g]）"
                                      % (target, x.min(), x.max()))
    j = int(np.argmin(np.abs(x - target)))
    lo, hi = max(0, j - 1), min(len(x) - 1, j + 1)
    dx = float(x[hi] - x[lo])
    dc = float(cost[hi] - cost[lo])
    if abs(dx) < 1e-12 or abs(dc) < 1e-12:
        return None, float(x[j]), "切线退化（dx=%.3g, dcost=%.3g）" % (dx, dc)
    slope = dx / dc
    if abs(slope) < 1e-9:
        return None, float(x[j]), "切线退化（d能力/dI_XT=%.3g）" % slope
    beta = dc / dx
    # **有效性判据（2026-10-02 加，第 113/114 条）**：
    #   ① **β ≤ 0**：切线方向与物理相反（能力随代价下降）⟹
    #      `R = β·I_TY − I_XT` 会**奖励少拿信息**，属荒谬值；
    #   ② **|β| < 1e-3**：奖励退化为纯代价最小化；
    #   ③ **β > 1e3**：奖励退化为纯信息最大化（代价项可忽略）。
    # 三者均判 `invalid`/`degenerate`——**不得计为解锁**（初版曾把
    # β=−0.046 与 β=0.000 的 12 条计为 ok）。
    if beta <= 0:
        return None, float(x[j]), ("**无效**：β=%.3g ≤ 0（切线 d能力/dI_XT=%.3g "
                                   "为负或零）⟹ 奖励方向反了，不可用"
                                   % (beta, slope))
    if abs(beta) < 1e-3:
        return None, float(x[j]), ("**退化**：β=%.3g 过小（切线 d能力/dI_XT=%.3g）"
                                   "⟹ 奖励退化为纯代价最小化，不可用"
                                   % (beta, slope))
    if beta > 1e3:
        return None, float(x[j]), ("**退化**：β=%.3g 过大 ⟹ 奖励退化为纯信息"
                                   "最大化（代价项失效）" % beta)
    return beta, float(x[j]), ("切线 d能力/dI_XT=%.4g ⟹ β=%.4g" % (slope, beta))


def _family_hint(pair: dict) -> str | None:
    from pipeline.beta_calibration import _family_hint as fh
    return fh(pair)


def family_inventory() -> int:
    """DB2 族盘点：行数 / 含可用曲线数 / I_TY_max 中位 —— 用于确认族名。"""
    stats: dict[str, list[int, int, list]] = {}
    with open(DB2, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fam = rec.get("family") or "?"
            e = stats.setdefault(fam, [0, 0, []])
            e[0] += 1
            c = rec.get("ib_curve")
            if c and c.get("beta"):
                e[1] += 1
                y = np.asarray(c["I_TY"], dtype=float)
                if len(y):
                    e[2].append(float(y.max()))
    print("%-18s %6s %8s %10s" % ("family", "行数", "有曲线", "I_TY_max 中位"))
    for fam, (n, k, ys) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
        print("%-18s %6d %8d %10s"
              % (fam, n, k, ("%.4f" % float(np.median(ys))) if ys else "—"))
    print("\n提示：`_family_hint` 把 R-STABLE/R-TOUGH 映射到 `landau`——"
          "若上表无 landau，须改为该族在 DB2 中的**实际族名**。")
    return 0


def _barrier_target(tv: dict) -> tuple[float | None, str]:
    """势垒目标：优先 `E_req_nats`；否则由 `delta_dimensionless` 换算 −ln δ
    （与 beta_calibration 的同名约定一致）。"""
    if "E_req_nats" in tv and tv["E_req_nats"] is not None:
        return float(tv["E_req_nats"]), "E_req_nats 直给"
    d = tv.get("delta_dimensionless")
    if d is not None:
        d = float(d)
        if 0 < d < 1:
            return -math.log(d), "−ln(δ)，δ=%g" % d
    return None, "既无 E_req_nats 也无可用 δ"


def calibrate_pair(pair: dict) -> dict:
    """对 barrier / env_index 两类需求做标定（用各自的曲线）。"""
    prop = pair["requirement_structured"].get("property_id", "")
    tv = pair["requirement_structured"].get("target_value", {}) or {}
    fam = _family_hint(pair) or "landau"
    res = {"pair_id": pair.get("pair_id"), "property": prop, "family": fam,
           "kind": None, "target": None, "beta": None, "status": "unmapped",
           "detail": "", "curve_note": ""}
    from pipeline.beta_calibration import _info_class
    icls = _info_class(pair)[0]
    # **类名已更新（第 114 条）**：`_info_class` 现返回
    # `constraint_barrier` / `constraint_env`（不再是 `barrier`/`env_index`），
    # 且这两类**已判定不适用 β** ⟹ 本函数保留 β 计算**仅作否证证据**，
    # 状态统一标 `not_applicable`（不再像 `unmapped` 那样看起来像失败）。
    if icls == "constraint_barrier":
        res["kind"] = "barrier"
        res["target"], how_t = _barrier_target(tv)
        res["detail"] = how_t
        if res["target"] is None:
            res["status"] = "not_applicable"
            return res
        x, cost, note = retention_curve(fam)
        res["curve_note"] = note
        if len(x) == 0:
            res["status"] = "not_applicable"
            res["detail"] = note
            return res
        b, ach, how = beta_at(x, cost, res["target"])
        res["detail"] = "%s；%s" % (how_t, how)
        res["beta"] = b
        res["status"] = "not_applicable"      # 不适用 β（β 仅作否证证据）
        return res
    if icls == "constraint_env":
        res["kind"] = "env_index"
        m = tv.get("n_environments_count")
        res["target"] = math.log(float(m)) if m else None
        x, cost, note = env_curve(fam)
        res["curve_note"] = note
        if res["target"] is None or len(x) == 0:
            res["status"] = "not_applicable"
            res["detail"] = note
            return res
        b, ach, how = beta_at(x, cost, res["target"])
        res["detail"] = how
        res["beta"] = b
        res["status"] = "not_applicable"
        return res
    res["detail"] = "非约束型（不属本脚本范围）"
    return res


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--families" in sys.argv:
        return family_inventory()
    if "--show" in sys.argv:
        fam = args[0] if args else "landau"
        for name, fn in (("保持曲线", retention_curve), ("工况曲线", env_curve)):
            x, c, note = fn(fam)
            print("== %s（族 %s，%d 点）==\n   %s" % (name, fam, len(x), note))
            for i in range(len(x)):
                print("   x=%10.4g  I_XT=%10.4g" % (x[i], c[i]))
        return 0

    print("=" * 72)
    print("任务 3：保持曲线 / 工况索引曲线")
    print("=" * 72)
    print("**⚠️ 状态（第 114 条）：本类需求已判定「不适用 β」**——势垒/工况数"
          "是**约束**，应走\n   达标型奖励（`a_network/req_env.py`）。"
          "下面的 β 计算**仅作为第 113 条否证实验的复现**保留：\n"
          "   实测得 β≈0 或 β<0（假解锁），已据此否证「逐设计代价」的初版定义。")
    print("口径：I_XT@β=4（DB2 扫描点）；ΔF 取族解析式；ln M_cap 取 "
          "I_TY_max 下界代理")
    pairs = [json.loads(p.read_text(encoding="utf-8"))
             for p in sorted(PAIRS.glob("DB3-*.json"))]
    from pipeline.beta_calibration import _info_class
    tgt = [(p, _info_class(p)[0]) for p in pairs]
    tgt = [(p, k) for p, k in tgt
           if k in ("constraint_barrier", "constraint_env")]
    print("\n本类需求共 %d 条（势垒型 %d / 工况型 %d）"
          % (len(tgt), sum(1 for _p, k in tgt if k == "constraint_barrier"),
             sum(1 for _p, k in tgt if k == "constraint_env")))

    # R1 曲线单调性
    print("\nR1 曲线单调性（能力对 I_XT 应单调升）：")
    for fam in sorted({calibrate_pair(p)["family"] for p, _k in tgt}):
        for name, fn in (("保持", retention_curve), ("工况", env_curve)):
            x, c, _n = fn(fam)
            if len(x) < 3:
                continue
            inc = int(np.sum(np.diff(c) >= -1e-12))
            print("   %-6s %-8s %3d 点，I_XT 随能力单调升 %d/%d"
                  % (fam, name, len(x), inc, len(x) - 1))

    rows = [calibrate_pair(p) for p, _k in tgt]
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print("\nR2 解锁率：%d/%d（%.0f%%）" % (n_ok, len(rows),
                                          100.0 * n_ok / max(len(rows), 1)))
    for r in rows:
        print("   %s [%s/%s] 目标 %.4g → %s%s"
              % (r["pair_id"], r["kind"], r["family"],
                 (r["target"] if r["target"] is not None else float("nan")),
                 r["status"],
                 ("" if r["beta"] is None else "，β=%.3f" % r["beta"])))
    print("\nR4 来源标注：每族曲线口径见 `--show <族>`；ΔF 与 ln M_cap 的定义"
          "已写入本模块 docstring。")
    print("R3 冲突检查：barrier/env 两类**不参与** param 类换算，故无冲突；"
          "若某 pair 的属性被误分类，会在上表以 kind 显示，便于核对。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
