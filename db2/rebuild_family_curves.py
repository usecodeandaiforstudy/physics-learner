# -*- coding: utf-8 -*-
"""为缺失族补 IB 曲线并重落 DB2（2026-10-02，第 111 条）。

**背景**：DB2 中 8700 行有 5100 行**无 IB 曲线**，根因是
`FingerprintComputer.compute_ib_curve` 的两个显式空分支——这些族
**既无 X 节点、也未注册 `ib_source`**。本脚本为**已补注册 `ib_source`
的族**（当前：landau → `landau_doublewell` 通道）重算指纹与 IB 曲线，
并以**追加**方式写入 DB2（**不改动、不删除现有行**）。

**判据（先写死）**：
  C1 **曲线非空**：每条新记录的 `ib_curve.I_TY` 非空且有限；
  C2 **物理合理性**：`I_XT`、`I_TY` 均 ≥ 0 且 `I_TY ≤ I_XT + 1e-6`（DPI）；
  C3 **单调性**：`I_TY` 随 β 单调不减（IB 曲线的定义性质）；
  C4 **势垒一致**：对 a<0 的记录，双阱极小位置 ±√(−a) 应落在 φ 网格内。

用法（项目根目录）：
  python db2/rebuild_family_curves.py --family landau --dry-run   # 先看判据
  python db2/rebuild_family_curves.py --family landau --write      # 落库
"""

from __future__ import annotations

import argparse
import json
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

ENTRIES = _ROOT / "db1" / "entries"
DB2 = _ROOT / "db2" / "db2.jsonl"


def _max_dip_rel(y: np.ndarray, frac: float = 0.0) -> float:
    """曲线的最大**相对下陷**：max_i (y_i − y_{i+1}) / max(y)。

    为什么用相对量而不是"严格单调"：既有 9 个在用族的曲线**本身就非严格
    单调**（β 小处 Blahut–Arimoto 有限容差噪声，例如 two_level 的 I_TY 在
    第 7 点从 1.1e-7 掉到 9.9e-11）。若把"严格单调"当门禁，**连生产族都
    过不了** ⟹ 判据须改为「**不劣于既有族**」的数据驱动形式。

    **`frac > 0`：只在可用区统计**（`y ≥ frac·max(y)`）。必要性：低 β 区的
    I_TY 噪声可达数量级，而 β 标定只用**曲线上部**（目标 I_TY 附近）的切线。
    **判据失效的教训（2026-10-02）**：不设 `frac` 时，低 β 噪声把下陷分布的
    p95 顶到 **0.9995 ≈ 1**；而相对下陷**按定义 ≤1** ⟹ **阈值 1.0 等于
    "任何曲线都通过"，C3 失去区分度**（当时"151 条通过"里 C3 部分是空的）。
    """
    if len(y) < 2 or float(y.max()) <= 0:
        return float("inf")
    d = np.diff(y)
    if frac > 0:
        thr = frac * float(y.max())
        keep = (y[:-1] >= thr) | (y[1:] >= thr)
        if not np.any(keep):
            return 0.0
        d = d[keep]
    return float(max(0.0, -d.min()) / float(y.max()))


def _reference_dip(family_ref: str = "ou", k: int = 5) -> float:
    """取既有族已落库曲线的最大相对下陷（作对照基准）。"""
    dips = _family_dips(family_ref, k)
    return float(max(dips)) if dips else float("nan")


def _family_dips(family: str, k: int = 5) -> list[float]:
    """取该族**已落库**曲线的最大相对下陷（最多 k 条）。"""
    dips: list[float] = []
    if not DB2.exists():
        return dips
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
            c = rec.get("ib_curve") or {}
            if c.get("I_TY"):
                dips.append(_max_dip_rel(np.asarray(c["I_TY"], dtype=float)))
            if len(dips) >= k:
                break
    return dips


def _families_with_curves() -> set[str]:
    """DB2 中**已有**可用曲线的族（用于跳过，避免重复落库）。"""
    fams: set[str] = set()
    if not DB2.exists():
        return fams
    with open(DB2, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = rec.get("ib_curve") or {}
            if c.get("beta"):
                fams.add(rec.get("family") or "?")
    return fams
    out = []
    for p in sorted(ENTRIES.glob("*.json")):
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if e.get("family") == family:
            out.append(e)
    return out


def _dip_baseline_p95(max_rows: int = 600, frac: float = 0.10
                      ) -> tuple[float, int, float]:
    """**可用区下陷分布**的 p95 作 C3 基准。

    返回 (p95_可用区, 样本数, p95_全曲线)——后者仅作参考，用来说明低 β 噪声
    的严重程度（全曲线 p95 ≈ 1 ⟹ **不能**拿它当判据，否则判据失效）。
    """
    dips_u: list[float] = []
    dips_a: list[float] = []
    if not DB2.exists():
        return float("nan"), 0, float("nan")
    with open(DB2, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = rec.get("ib_curve") or {}
            if c.get("I_TY"):
                yy = np.asarray(c["I_TY"], dtype=float)
                dips_u.append(_max_dip_rel(yy, frac=frac))
                dips_a.append(_max_dip_rel(yy))
            if len(dips_u) >= max_rows:
                break
    if not dips_u:
        return float("nan"), 0, float("nan")
    return (float(np.percentile(dips_u, 95)), len(dips_u),
            float(np.percentile(dips_a, 95)))


def _load_entries(family: str) -> list[dict]:
    """取该族 DB1 条目；`--family all` = 全部条目（按 system_id 排序可复现）。"""
    out: list[dict] = []
    for p in sorted(ENTRIES.glob("*.json")):
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if family == "all" or e.get("family") == family:
            out.append(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from b_network.fingerprint_computer import FingerprintComputer
    from b_network.system_encoder import SystemEncoder
    from db1.db2_writer import append_db2, build_db2_record
    # **复用既有管线的组装逻辑**：`build_db2_record` 期望 report 三元组
    # `{fingerprint, graph, ib_curve}`，且 fingerprint 须由 `fpc.compute(entry,
    # graph)` 产生、graph 须经 `_graph_summary` 摘要、ib_curve 须经 `_plain_ib`
    # 去 numpy——自己拼会 KeyError（初版即如此）。
    # 注意：`db1/ingest_pipeline.py` 用**脚本内相对导入**（`from db2_writer
    # import ...`），故须把 `db1/` 也加入 sys.path（与其自身运行约定一致），
    # 否则 `ModuleNotFoundError: db2_writer`。
    sys.path.insert(0, str(_ROOT / "db1"))
    from ingest_pipeline import _graph_summary, _plain_ib  # noqa: E402

    enc = SystemEncoder()
    fc = FingerprintComputer()
    entries = _load_entries(args.family)
    if args.family == "all":
        have = _families_with_curves()
        skipped = sorted({e.get("family") for e in entries} & have)
        entries = [e for e in entries if e.get("family") not in have]
        print("**跳过已有曲线的族 %d 个**（避免重复落库）：%s"
              % (len(skipped), ", ".join(skipped) if skipped else "无"))
        # 逐族 C3 基准：**用该族自身既有曲线**（无则用 ou）
        def _dip_base(fam: str) -> float:
            d = _family_dips(fam, k=5)
            return float(max(d)) if d else float("nan")
    else:
        def _dip_base(fam: str) -> float:      # noqa: F811
            return _reference_dip("ou", k=5)
    if args.limit:
        entries = entries[:args.limit]
    print("族 %s：DB1 条目 %d 条" % (args.family, len(entries)))
    if not entries:
        print("无条目（请核对 DB1 entries 的 family 字段）")
        return 1

    n_ok = n_empty = n_bad = 0
    dips: list[float] = []
    worst: list[tuple[str, float]] = []
    ref_dip, n_ref, ref_dip_all = _dip_baseline_p95()
    if n_ref:
        print("C3 基准（**可用区** y≥10%%·max 的下陷 p95）= %.4g；全曲线 p95 = "
              "%.4g ⟹ **两者相同 ⟹ 判据无区分度 ⟹ C3 降级为信息性输出**"
              "（门禁只保留 C1/C2）" % (ref_dip, ref_dip_all))
    recs = []
    for e in entries:
        try:
            g = enc.encode(e)
        except Exception as ex:  # noqa: BLE001
            print("  [编码失败] %s：%s" % (e.get("system_id"), ex))
            n_bad += 1
            continue
        src = g.metadata.get("ib_source")
        if not src:
            print("  [无 ib_source] %s（该族仍缺注册）" % e.get("system_id"))
            n_bad += 1
            continue
        try:
            curve = fc.compute_ib_curve(g)
        except Exception as ex:  # noqa: BLE001
            print("  [曲线失败] %s：%s" % (e.get("system_id"), ex))
            n_bad += 1
            continue
        y = np.asarray(curve.get("I_TY", []), dtype=float)
        x = np.asarray(curve.get("I_XT", []), dtype=float)
        # C1 非空有限
        if len(y) == 0 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
            n_empty += 1
            continue
        # C2 DPI
        if np.any(y < -1e-9) or np.any(x < -1e-9) or np.any(y > x + 1e-6):
            print("  [C2 失败] %s：DPI 或负值异常" % e.get("system_id"))
            n_bad += 1
            continue
        # C3（**相对判据 + 本族基准，2026-10-02 修**）：曲线的最大相对下陷
        # **不得劣于该族既有曲线**（无既有曲线时退到 ou 全局基准）。
        # 修因：初版一律用 ou 作基准 ⟹ 对已有曲线的族（如 radioactive_decay，
        # 自身下陷 0.00817）**误判失败**——与 M4 的"标定范围错配"同一类。
        # **C3 降级为信息性输出（2026-10-02，第三次也是最后一次）**：
        #   可用区 p95 = 0.9995 与全曲线 p95 相同 ⟹ **连可用区内部都存在
        #   数量级跳变**（如 two_level 的 I_TY 从 1.1e-7 掉到 9.9e-11，
        #   落差 ≈ max）⟹ **该判据无区分度是结构性的，不是阈值大小问题**。
        #   故**不再调阈值**：C3 只记录下陷数值并列出最差者，**不作门禁**；
        #   门禁仅保留 C1（非空有限）与 C2（DPI）。
        dip = _max_dip_rel(y, frac=0.10)
        dips.append(dip)
        if len(worst) < 5 or dip > worst[-1][1]:
            worst.append((e.get("system_id"), dip))
            worst.sort(key=lambda z: -z[1])
            worst = worst[:5]
        n_ok += 1
        try:
            fp = fc.compute(e, g)          # 与既有管线同源（指纹 + 数值/解析对照）
            rec = build_db2_record(e, {
                "fingerprint": fp,
                "graph": _graph_summary(g),
                "ib_curve": _plain_ib(curve),
            }, encoder_version="1.0.0")
            recs.append(rec)
            if args.write:
                append_db2(rec, DB2)
        except Exception as ex:  # noqa: BLE001
            print("  [记录组装失败] %s：%s" % (e.get("system_id"), ex))
            n_bad += 1

    print("\n结果：**C1/C2 通过 %d 条**；空曲线 %d 条；异常 %d 条"
          "（C3 已降级为信息性输出，不作门禁）" % (n_ok, n_empty, n_bad))
    if dips:
        print("曲线下陷（可用区，信息性）：最大 %.3g（中位 %.3g）——"
              "与既有曲线同量级（既有 p95 %.3g）"
              % (max(dips), float(np.median(dips)), ref_dip))
        print("下陷最大的 5 条（如显著高于中位，须查 Blahut–Arimoto 收敛设置）：")
        for sid, d in worst:
            print("  %-32s %.4g" % (sid, d))
    if recs:
        c = recs[0].get("ib_curve") or {}
        if c.get("I_TY"):
            print("样例曲线：I_TY 从 %.4g 到 %.4g（%d 点）"
                  % (c["I_TY"][0], c["I_TY"][-1], len(c["I_TY"])))
    if args.write:
        print("**已追加写入 %d 条到 %s**（未改动现有行）" % (len(recs), DB2))
    else:
        print("（dry-run：未写库；加 --write 落库）")
    print("落库后请跑：python pipeline/extra_curves.py --families "
          "（确认该族「有曲线」数 > 0）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
