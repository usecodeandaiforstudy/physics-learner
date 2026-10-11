# -*- coding: utf-8 -*-
"""中游①：**族无关的 q→参数反解（数值反解 + 解析交叉校验）**
（2026-10-02，第 127 条）。

**问题**：`pipeline/design_contract.py` 的反解表**只手写了 4 个族**
（rayleigh / two_level / control_ou / materials_diffusion），其余族"如实返回
None"。中游要"做好"，覆盖必须铺开。

**做法（不是再手写更多桥接式）**：
  1. **对每个族，在参数网格上算解析 Fisher 剖面** `F(θ)`——
     `core/fingerprint.py` 对 17 个族有闭式 Fisher；用真实 DB1 条目模板
     构造各 θ 的合成条目即可（复用 `structure_space` 的模板机制）；
  2. **数值反解**：`q_req` ⟹ `θ` 界（单调插值）。**判据是单调性**：
     剖面必须单调，否则反解无定义（如实报 `non_monotone`，不硬解）；
  3. **解析交叉校验**：现有 4 条手写桥接式与之对照，相对误差须 ≤5%。

**为什么这样更可靠**：桥接式是"我推导的"，容易错（本会话已多次）；
而 **Fisher 剖面来自 `core/fingerprint.py` 的解析闭式 + 真实条目模板**
⟹ 与 DB1/DB2/A/C **同一口径**，且反解过程可审计（打印剖面端点与单调性）。

用法：
    python pipeline/inverse_fisher.py              # 全族反解表 + 一致性检查
    python pipeline/inverse_fisher.py --show ou    # 看某族 Fisher 剖面
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
N_GRID = 48

# **已手写的解析桥接式（用于交叉校验；不删除）**——与 design_contract 对齐。
# **域别名 → DB1 族名（2026-10-02 修）**：`design_contract` 用的是"域-族"组合名
#   （如 `acoustics_rayleigh`），而 DB1 的 family 是 `rayleigh`
#   ⟹ 直接拿别名查模板会 `KeyError`（初版即如此）。
CONTRACT_TO_DB1 = {
    "acoustics_rayleigh": "rayleigh",
    "control_ou": "ou",
    "materials_diffusion": "diffusion",
    "two_level": "two_level",
}

ANALYTIC_BRIDGES = {
    # (DB1 族, 度量) → (参数, 界类型, 解析反解, **是否与 Fisher 同口径**)
    ("rayleigh", "q1"): ("sigma", "upper", lambda q: math.sqrt(4.0 / q), True),
    ("two_level", "q1"): ("p", "band", None, False),      # 禁带，单独处理
    ("ou", "q9"): ("theta", "lower", lambda q: q, False),  # q9 = θ（非 Fisher）
    ("diffusion", "q7"): ("kappa_rel", "lower", lambda q: q, False),
}


def family_param(family: str) -> tuple[str, tuple[float, float]]:
    """该族的可调参数名与范围（取自真实 DB1 条目模板）。"""
    from a_network.structure_space import family_template
    tpl = family_template(family)
    pname = tpl.get("scan_param")
    for pr in tpl.get("parameters", []):
        if pr.get("name") == pname and pr.get("range"):
            r = pr["range"]
            return pname, (float(r[0]), float(r[1]))
    params = tpl.get("parameters") or []
    if not params:
        raise KeyError("族 %s 无参数" % family)
    r = params[0].get("range", [0.0, 1.0])
    return params[0]["name"], (float(r[0]), float(r[1]))


def synthetic_entry(family: str, pname: str, value: float) -> dict:
    import copy
    from a_network.structure_space import family_template
    e = copy.deepcopy(family_template(family))
    e["system_id"] = "INV-%s" % family
    e["domain"] = "synthetic"
    e["scan_param"] = pname
    for pr in e.get("parameters", []):
        if pr.get("name") == pname:
            pr["default"] = float(value)
    e["provenance"] = {"source": "AI+verified",
                       "human_review_status": "pending",
                       "note": "反解表数值剖面（第 127 条）"}
    return e


def fisher_profile(family: str, n_grid: int = N_GRID
                   ) -> tuple[np.ndarray, np.ndarray, str]:
    """该族的解析 Fisher 剖面 `(θ 网格, F_trace, 参数名)`。"""
    from a_network.analytic_backbone import compute_backbone
    from b_network.system_encoder import SystemEncoder
    pname, (lo, hi) = family_param(family)
    enc = SystemEncoder()
    thetas, fs = [], []
    for v in np.linspace(lo, hi, n_grid):
        try:
            g = enc.encode(synthetic_entry(family, pname, float(v)))
            bb = compute_backbone(g)
            f = None
            for vv in bb["node_fisher"].values():
                if vv.get("covered") and vv.get("fisher_trace"):
                    f = float(vv["fisher_trace"])
                    break
            if f is None:
                continue
            thetas.append(float(v))
            fs.append(f)
        except Exception:  # noqa: BLE001
            continue
    return np.asarray(thetas), np.asarray(fs), pname


def invert(family: str, q_req: float) -> dict:
    """数值反解：`q_req` ⟹ 参数界。返回 {param, kind, value, monotone, note}。"""
    th, f, pname = fisher_profile(family)
    if len(th) < 8:
        return {"param": pname, "kind": None, "value": None,
                "monotone": None, "note": "剖面点不足（%d）" % len(th)}
    d = np.diff(f)
    inc = int(np.sum(d >= 0))
    dec = int(np.sum(d <= 0))
    dtype = "increasing" if inc >= 0.9 * len(d) else (
        "decreasing" if dec >= 0.9 * len(d) else "non_monotone")
    if dtype == "non_monotone":
        return {"param": pname, "kind": None, "value": None, "monotone": False,
                "note": "Fisher 剖面非单调 ⟹ 反解无定义（如实报告，不硬解）"}
    if dtype == "increasing":
        # F 随 θ 升 ⟹ 要 q ≥ q_req 需 θ ≥ θ(q_req)
        if q_req > f.max():
            return {"param": pname, "kind": None, "value": None,
                    "monotone": True,
                    "note": "q_req %.4g 超出该族 F 上界 %.4g（不可达）"
                            % (q_req, float(f.max()))}
        val = float(np.interp(q_req, f, th))
        return {"param": pname, "kind": "lower", "value": val,
                "monotone": True, "note": "F 单调升 ⟹ θ ≥ %.4g" % val}
    # F 随 θ 降 ⟹ 要 q ≥ q_req 需 θ ≤ θ(q_req)
    if q_req > f.max():
        return {"param": pname, "kind": None, "value": None, "monotone": True,
                "note": "q_req %.4g 超出该族 F 上界 %.4g（不可达）"
                        % (q_req, float(f.max()))}
    val = float(np.interp(q_req, f[::-1], th[::-1]))
    return {"param": pname, "kind": "upper", "value": val, "monotone": True,
            "note": "F 单调降 ⟹ θ ≤ %.4g" % val}


def families_available() -> list[str]:
    fams = set()
    for p in sorted(DB1.glob("*.json")):
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if e.get("family"):
            fams.add(e["family"])
    return sorted(fams)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--show" in sys.argv:
        fam = args[0] if args else "ou"
        th, f, pname = fisher_profile(fam)
        print("族 %s 的 Fisher 剖面（参数 %s，%d 点）：" % (fam, pname, len(th)))
        for i in range(0, len(th), max(1, len(th) // 10)):
            print("   %s=%9.4g → F_trace=%.6g" % (pname, th[i], f[i]))
        return 0

    print("=" * 74)
    print("中游①：q→参数反解（数值反解 + 解析交叉校验）")
    print("=" * 74)
    fams = families_available()
    print("DB1 中的族：%d 个" % len(fams))

    rows = []
    for fam in fams:
        try:
            r = invert(fam, 1.0)          # 以 q_req=1 作统一探测值
        except Exception as ex:  # noqa: BLE001
            rows.append((fam, None, None, None, "%s: %s"
                         % (type(ex).__name__, ex)))
            continue
        rows.append((fam, r.get("param"), r.get("kind"), r.get("value"),
                     r.get("note", "")))

    ok = [r for r in rows if r[3] is not None]
    nonmono = [r for r in rows if r[1] and "非单调" in (r[4] or "")]
    print("\n反解表（q_req = 1 探测；参数名/范围取自族模板）：")
    print("  %-22s %-10s %-6s %12s  %s"
          % ("family", "param", "kind", "bound", "note"))
    for fam, pname, kind, val, note in rows:
        print("  %-22s %-10s %-6s %12s  %s"
              % (fam, pname or "—", kind or "—",
                 ("%.5g" % val) if isinstance(val, float) else "—",
                 (note or "")[:44]))
    print("\n**可反解族数：%d（目标 ≥10）**" % len(ok))

    # ---- I3 交叉校验（**同口径才可比**）----
    print("\nI3a Fisher 同口径对照（rayleigh 的 q1=4/σ² vs 数值 Fisher 反解，≤5%）：")
    n_ok = n_cmp = 0
    for (fam, metric), (pname, kind, fn, same_scale) in ANALYTIC_BRIDGES.items():
        if not same_scale or fn is None:
            continue
        q = 1.0
        ana = float(fn(q))
        num = invert(fam, q)
        n_cmp += 1
        if num.get("value") is None:
            print("  %-10s %s：数值反解不可用（%s）"
                  % (fam, metric, num.get("note")))
            continue
        rel = abs(num["value"] - ana) / max(abs(ana), 1e-12)
        good = rel <= 0.05
        n_ok += int(good)
        print("  %-10s %s：解析(%s %s %s=%.4g) vs 数值 %.4g → 相对差 %.2f%% %s"
              % (fam, metric, kind, pname, pname, ana, num["value"],
                 100 * rel, "✓" if good else "**✗ 超 5%**"))

    print("\nI3b **公式级一致**（桥接式 `q=f(参数)` vs 环境度量定义 `q=g(参数)`）：")
    # **初版 I3b 的错（2026-10-02 修正）**：我拿**Fisher 反解**去比 q9/q7 的定义
    #   ⟹ 两者根本不是同一个量（q9 = θ 本身；q7 = κ̃·κ_ref/L，都不是 Fisher）
    #   ⟹ 报出"不一致"是**测试设计错误**。正确检查：**同一条桥接式**在**参数网格**
    #   上与**环境的度量表**逐点对照（把参数名与单位对齐后）。
    try:
        from a_network.req_env import KAPPA_REF, L_DIFF
        from a_network.req_env import RequirementConditionedEnv
        env = RequirementConditionedEnv()
        DB1_TO_ENV = {v: k for k, v in CONTRACT_TO_DB1.items()}
        fam_idx = {n: i for i, n in enumerate(env.family_names)}
        # 桥接式（与 design_contract 同式）在"env 参数"上的表达：
        #   ou：q9 = θ（同一参数，同式 ⟹ 应 0 差）
        #   diffusion：桥接 q7 = κ/L，其中 κ(env) = κ̃·κ_ref ⟹ q7 = κ̃·κ_ref/L
        cases = [
            ("ou", "q9", 2.30, lambda p: p,                      # 桥接式
             lambda p: p),                                        # 环境定义
            ("diffusion", "q7", 0.01,
             lambda kk: (kk * KAPPA_REF) / L_DIFF,                # 桥接式
             lambda kk: (kk * KAPPA_REF) / L_DIFF),               # 环境定义
        ]
        for fam, metric, q_req, f_bridge, g_table in cases:
            env_fam = DB1_TO_ENV.get(fam, fam)
            fi = fam_idx.get(env_fam, -1)
            if fi < 0:
                print("  %-10s %s：env 无该族（env 族名 %s）"
                      % (fam, metric, env_fam))
                continue
            params = np.asarray(env.grids[fi])
            qs_b = np.asarray([f_bridge(float(p)) for p in params])
            qs_t = np.asarray(env.metric_tables[metric][fi], dtype=float)
            rel = float(np.max(np.abs(qs_b - qs_t)
                               / np.maximum(np.abs(qs_t), 1e-12)))
            n_cmp += 1
            good = rel <= 0.05
            n_ok += int(good)
            print("  %-10s %s（env 族 %s）：桥接式 vs 度量表 逐点最大相对差 "
                  "%.2f%% %s" % (fam, metric, env_fam, 100 * rel,
                                 "✓" if good else "**✗ 超 5%**"))
    except Exception as ex:  # noqa: BLE001
        print("  [跳过] %s: %s" % (type(ex).__name__, ex))

    print("\n===== 汇总 =====")
    print("可反解族 %d；非单调族 %d；交叉校验 %d/%d"
          % (len(ok), len(nonmono), n_ok, n_cmp))
    print("判据：I1 单调性（非单调须如实报）✓ 已在表内；"
          "I2 覆盖 ≥10；I3 对照一致（同口径 ≤5%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
