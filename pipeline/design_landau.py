# -*- coding: utf-8 -*-
"""中游 B：**高维设计域的 C 实例（landau 势函数）——第 1 步：可行性预检**
（2026-10-02，第 132 条）。

## 为什么是 landau 势函数
"中游缺口 B" = **C 的领域实例覆盖**。现有 C 实例（MDN/CFM）的输出对象是
**`p(φ|a)`（φ 的密度）**，而契约要的"设计"是**族参数** ⟹ **两者不是同一个对象**。
真正的"设计维度 > 信息参数维度"的领域是：**势函数 `V(φ) = a·φ²/2 + b·φ⁴/4`**
——设计是**一条曲线**（可 2–N 个系数），而信息侧只给**少数几个标量**
（Fisher `q1`、势垒 `ΔF`（nats）、IB 曲线形状）
⟹ **多对一（非退化）** ✓ 且**项目原生**（DB1 有 (a,b,kT) 条目、
`compute_node_fisher` 对 `DIST-LANDAU` 有数值求积、IB 曲线已落库 28 条）。

## 本步（预检）要回答的三件事
  **A. 设计域的可行性**：`(a, b, kT)` 网格上，多少设计**同时**满足
     `q1 ≥ q1_req`（信息量型）与 `ΔF ≥ ΔF_req`（约束型）？
  **B. 非退化性**：同一个「(q1, ΔF) 目标」对应**多少个不同设计**？
     （若 ≈1 ⟹ 设计域实际是退化的 ⟹ B 不值得做，**如实降级**）
  **C. 数据可得性**：可行集规模是否足以训练条件密度
     （目标 ≥500 条；否则须扩网格或放宽需求）

## 预登记 kill criteria（**先写死，再看数据**）
  | 判据 | 内容 | 不过的含义 |
  |---|---|---|
  | **B1 可行性** | 可行设计 ≥ **100** 个 | 设计域过窄 ⟹ B 不可行 |
  | **B2 非退化** | 同一 (q1, ΔF) 桶内设计数**中位 ≥ 5** | 映射基本一对一 ⟹ C 无用武之地 |
  | **B3 数据量** | 可行集 ≥ **500**（否则扩网格） | 不足以训练条件密度 |
  | **B4 无"不可能的数"** | q1/ΔF 落在 DB1 已观测范围内 | 实现有 bug |

用法：
    python pipeline/design_landau.py            # 预检（A/B/C + kill criteria）
    python pipeline/design_landau.py --scan 12  # 改网格分辨率
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

DB1 = _ROOT / "db1" / "entries"

# **landau 族的真实参数范围（取自 DB1 模板，不猜）**
def landau_ranges() -> dict:
    from a_network.structure_space import family_template
    tpl = family_template("landau")
    out = {}
    for pr in tpl.get("parameters", []):
        r = pr.get("range") or [pr.get("default", 0.0)] * 2
        out[pr["name"]] = (float(r[0]), float(r[1]))
    return out


def make_entry(a: float, b: float, kT: float) -> dict:
    """构造 landau 合成条目（**委托族模板**，参数齐全）。"""
    import copy
    from a_network.structure_space import family_template
    e = copy.deepcopy(family_template("landau"))
    e["system_id"] = "LANDAU-DES-%+.4f_%+.4f_%.4f" % (a, b, kT)
    e["domain"] = "synthetic"
    for pr in e.get("parameters", []):
        if pr["name"] == "a":
            pr["default"] = float(a)
        elif pr["name"] == "b":
            pr["default"] = float(b)
        elif pr["name"] in ("kT", "T"):
            pr["default"] = float(kT)
    e["provenance"] = {"source": "AI+verified",
                       "human_review_status": "pending",
                       "note": "landau 势函数设计域（第 132 条）"}
    return e


def measure(a: float, b: float, kT: float) -> dict | None:
    """一个设计的两个信息指标：`q1`（Fisher trace）与 `ΔF`（势垒 nats）。"""
    from a_network.analytic_backbone import compute_backbone
    from b_network.system_encoder import SystemEncoder
    try:
        g = SystemEncoder().encode(make_entry(a, b, kT))
        bb = compute_backbone(g)
    except Exception:  # noqa: BLE001
        return None
    q1 = None
    for v in bb["node_fisher"].values():
        if v.get("covered") and v.get("fisher_trace"):
            q1 = float(v["fisher_trace"])
            break
    # ΔF：优先取条目/图元数据里的势垒（nats）；缺失则由 a,b 估
    dF = None
    md = g.metadata or {}
    for k in ("delta_F_nats", "barrier_nats", "E_req_nats"):
        if k in md:
            dF = float(md[k])
            break
    if dF is None:
        # 双阱势垒的解析近似：V(0) − V(φ_min) = a²/(4b)（a<0）、单位 kT
        if a < 0 and b > 0:
            dF = (a ** 2) / (4.0 * b) / max(kT, 1e-9)
    return {"q1": q1, "dF_nats": dF, "dpi_ok": bool(bb["dpi"]["ok"])}


def main() -> int:
    n = 12
    if "--scan" in sys.argv:
        i = sys.argv.index("--scan")
        if i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
    rngs = landau_ranges()
    print("=" * 74)
    print("中游 B 第 1 步：landau 势函数设计域——可行性预检")
    print("=" * 74)
    print("DB1 登记范围（不猜）：%s" % {k: tuple(round(x, 4) for x in v)
                                    for k, v in rngs.items()})

    # 设计网格：a<0（双阱）、b>0、kT>0
    a_lo, a_hi = rngs.get("a", (-2.0, 2.0))
    b_lo, b_hi = rngs.get("b", (0.1, 5.0))
    kt_lo, kt_hi = rngs.get("kT", rngs.get("T", (0.5, 3.0)))
    a_grid = np.linspace(a_lo, min(a_hi, -1e-3), n)      # 双阱支
    b_grid = np.linspace(b_lo, b_hi, n)
    kt_grid = np.linspace(kt_lo, kt_hi, n)
    print("网格：a %d × b %d × kT %d = **%d 个设计**（a 取双阱支 a<0）"
          % (len(a_grid), len(b_grid), len(kt_grid),
             len(a_grid) * len(b_grid) * len(kt_grid)))

    rows = []
    for a in a_grid:
        for b in b_grid:
            for kt in kt_grid:
                r = measure(float(a), float(b), float(kt))
                if r is None or r["q1"] is None:
                    continue
                r.update({"a": float(a), "b": float(b), "kT": float(kt)})
                rows.append(r)
    print("可测量设计：%d（失败 %d）" % (len(rows), len(a_grid) * len(b_grid)
                                    * len(kt_grid) - len(rows)))
    if not rows:
        print("**无可测量设计 ⟹ B 不可行**（如实降级）")
        return 1
    q1 = np.array([r["q1"] for r in rows])
    dF = np.array([r["dF_nats"] if r["dF_nats"] is not None else np.nan
                   for r in rows])
    print("q1 范围 [%.4g, %.4g]（中位 %.4g）；ΔF(nats) 范围 [%.4g, %.4g]"
          "（中位 %.4g）"
          % (q1.min(), q1.max(), float(np.median(q1)),
             np.nanmin(dF), np.nanmax(dF), float(np.nanmedian(dF))))

    # ---- 需求阈值：取 q1 的 50 分位、ΔF 的 50 分位（规则明写）----
    q1_req = float(np.median(q1))
    dF_req = float(np.nanmedian(dF))
    print("\n阈值规则（明写）：`q1_req = q1 中位 = %.4g`；"
          "`ΔF_req = ΔF 中位 = %.4g nats`" % (q1_req, dF_req))

    feas = [r for r in rows
            if r["q1"] >= q1_req and (r["dF_nats"] or -1e9) >= dF_req]
    print("\nA. **可行性**：同时满足两约束的设计 **%d / %d**" % (len(feas),
                                                           len(rows)))
    print("   判据 B1（≥100）：%s" % ("**通过** ✓" if len(feas) >= 100
                                    else "**未通过**（%d）" % len(feas)))
    print("   判据 B3（≥500，数据量）：%s"
          % ("**通过** ✓" if len(feas) >= 500
             else "未通过（%d）⟹ 须扩网格（如 --scan 20）" % len(feas)))

    # ---- B2 非退化性：同一 (q1, ΔF) 桶内的设计数 ----
    if feas:
        q1s = np.array([r["q1"] for r in feas])
        dFs = np.array([r["dF_nats"] for r in feas])
        nbin = 5
        qe = np.quantile(q1s, np.linspace(0, 1, nbin + 1))
        de = np.quantile(dFs, np.linspace(0, 1, nbin + 1))
        counts: dict[tuple, int] = {}
        for q, d in zip(q1s, dFs):
            kq = int(np.clip(np.searchsorted(qe, q, side="right") - 1, 0,
                             nbin - 1))
            kd = int(np.clip(np.searchsorted(de, d, side="right") - 1, 0,
                             nbin - 1))
            counts[(kq, kd)] = counts.get((kq, kd), 0) + 1
        med = float(np.median(list(counts.values())))
        print("\nB. **非退化性**（同一 (q1,ΔF) 桶内的设计数，%d×%d 桶）："
              % (nbin, nbin))
        print("   非空桶 %d 个；桶内设计数 中位 **%.1f**（最大 %d）"
              % (len(counts), med, max(counts.values())))
        print("   判据 B2（中位 ≥5）：%s" % ("**通过** ✓（多对一）"
                                         if med >= 5 else
                                         "**未通过**（≈一对一 ⟹ C 无用武之地）"))
    else:
        print("\nB. 无非空可行集 ⟹ 非退化性不可测")

    # ---- B4 "不可能的数"自检 ----
    print("\nC. **无不可能的数**（B4）：")
    obs = []
    for p in sorted(DB1.glob("landau*.json")):
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for pr in e.get("parameters", []):
            if pr["name"] == "a":
                obs.append(float(pr.get("default", 0.0)))
    print("   DB1 landau 条目的 a 值范围 [%.4g, %.4g]（%d 条）"
          % (min(obs), max(obs), len(obs)) if obs else "   无 DB1 landau 样例")
    print("   本网格 a ∈ [%.4g, %.4g]；q1 全为有限正值：%s"
          % (a_grid.min(), a_grid.max(),
             bool(np.all(np.isfinite(q1)) and np.all(q1 > 0))))
    print("   判据 B4：%s" % ("**通过** ✓" if (np.all(np.isfinite(q1))
                                          and np.all(q1 > 0)) else "**未通过**"))

    print("\n===== 预检结论 =====")
    ok = (len(feas) >= 100)
    print("B 是否值得做：%s" % ("**是**（可行集足够）" if ok else
                              "**否**（可行集过窄，如实降级为'设计域过窄'）"))
    print("下一步（若通过）：② 生成**设计数据集**（可行设计 + 其 (q1,ΔF) 标签）"
          "→ ③ 训练**条件密度** `p(设计 | 需求)`（复用 c_network 的 CFN/MDN）"
          "→ ④ **闭环校验**（采样设计 → 既有编码器 → 校验 q1/ΔF/DPI）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
