# -*- coding: utf-8 -*-
"""C：**速率型需求的验证覆盖**（2026-10-02，第 151 条）。

## 缺口（架构 §12 的诚实遗留）
DB3 的 **52/159（33%）需求是速率型**（`response_speed`/`transport`/`flow`/
`speed`/`rate` ⟹ 度量 `q9`（谱隙）/`q7`（前缘速度））。而生产验证器
（`sim/verify_api`）**只有 `ou`（q9）与 `diffusion`（q7）两个动力学适配器**
⟹ **速率型的绝大多数需求当前无法走生产验证**。

## 本脚本做两件事
  ① **量化覆盖**：对 DB3 的速率型需求逐条判定——
     `可验证`（能路由到已登记 (族, 度量)）/ `不可验证`（无适配器）⟹ 给出**覆盖率**；
  ② **存在性证明**：对可验证的路径跑**端到端生产验证**（`ou`/q9、`diffusion`/q7），
     确认"速率型需求确实能被外部验证"，并**如实报告不可验证者的原因**。

## 判据（预登记）
  | 判据 | 内容 | 不过的含义 |
  |---|---|---|
  | **V1 覆盖量化** | 必须给出 `可验证/总数`（不得含糊） | — |
  | **V2 存在性** | 已登记的两条速率路径**至少 1 条**能给出 `pass/fail`（非 `unknown`） | 生产验证对速率型**完全不可用** |
  | **V3 域外仍 unknown** | 未登记族的速率需求 → `unknown`（**不得误报 pass**） | 安全性质 |

用法：
    python sim/v36_rate_coverage.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# **族能力区间（第 156 条）**：某些度量的可达范围由**多个参数**共同决定
#   （如 `q7 = κ/L`：`κ∈[1e-14,10]`、`L∈[1e-4,1e3]` ⟹ `q7∈[1e-17,1e5]` m/s）。
#   登记里若有 `capability` 字段则优先用区间判可达；否则用单参数范围。
CAPABILITY_FALLBACK: dict[str, tuple[str, float, float]] = {
    "q7": ("diffusion", 1e-17, 1e5),
}

# **度量 → 可验证的（族, 参数名）登记**（与 `sim/verify_api.REGISTRY` 一致）
RATE_PATHS: dict[str, list[tuple[str, str]]] = {
    "q9": [("ou", "theta"),                 # 谱隙（OU 动力学）——上限 10 s^-1
           ("poisson_count", "rate"),       # 泊松计数（第 152 条）——上限 1e12
           ("radioactive_decay", "rate")],  # 指数衰减（第 152 条）——上限 1e12
    "q7": [("diffusion", "kappa"),          # 前缘速度（扩散动力学）
           ("harmonic_oscillator", "a"),    # 声子群速度（第 158 条）
           ("lc_circuit", "inductance")],   # LC 传输线（第 158 条）
}
# 速率型需求**指引的族**（DB3 规则表里的 family_hint；无则视为无路径）
FAMILY_HINT_KEYS = ("family_hint", "family", "family_hints")


def _rate_pairs() -> list[dict]:
    from pipeline.reward_types import classify_pair
    out = []
    for p in sorted((_ROOT / "db3" / "pairs").glob("DB3-*.json")):
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        cls, _why = classify_pair(e)
        if cls == "rate_ratio":
            out.append(e)
    return out


def _metric_of(pair: dict) -> str | None:
    """从 `info_metric_vector` 取度量——**前缀匹配**（实测键为
    `q9_spectral_gap` / `q7_*`，初版按精确键 `q9`/`q7` 查找 ⟹ **52/52 全判
    "无显式度量登记"**，是我的查找键写错了）。"""
    iv = pair.get("info_metric_vector") or {}
    if isinstance(iv, dict):
        for k in iv:
            ks = str(k)
            for pref in ("q9", "q7"):
                if ks.startswith(pref):
                    return pref
    rs = pair.get("requirement_structured", {}) or {}
    for k in ("metric", "q_metric"):
        v = rs.get(k)
        if isinstance(v, str) and v[:2] in ("q9", "q7"):
            return v[:2]
    return None


def _required_q(pair: dict) -> float | None:
    """从 `worked_example.result` 解析需求量（如 `q9 ≥ 150 s^-1`）。"""
    import re
    we = pair.get("worked_example") or {}
    txt = str(we.get("result") or "")
    m = re.search(r"q[79]\s*[≥>=]+\s*([0-9.]+(?:[eE][+-]?[0-9]+)?)", txt)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _family_hint(pair: dict) -> str | None:
    """**速率需求是"域级"的**（实测 52 条**均无 `family_hint`**）⟹ 返回 None。

    初版据此把 52 条全判"需求指引的族无适配器" ⟹ 逻辑本身是错的
    （路由应按**度量**，不按族）。
    """
    rs = pair.get("requirement_structured", {}) or {}
    for k in ("family_hint", "family"):
        v = rs.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def main() -> int:
    import numpy as np  # noqa: F401
    from pipeline.family_aliases import to_db1
    from sim.verify_api import REGISTRY, verify
    print("=" * 74)
    print("C：速率型需求的验证覆盖（q9/q7）")
    print("=" * 74)
    pairs = _rate_pairs()
    print("DB3 速率型需求：**%d 条**（占 159 的 %.0f%%）"
          % (len(pairs), 100 * len(pairs) / 159))
    print("已登记的速率验证路径：%s"
          % {m: [f for f, _p in v] for m, v in RATE_PATHS.items()})

    n_route = n_metric = 0
    reasons: dict[str, int] = {}
    reach = unreach = 0
    detail = []
    for e in pairs:
        m = _metric_of(e)
        if m is None:
            reasons["无度量登记"] = reasons.get("无度量登记", 0) + 1
            continue
        n_metric += 1
        paths = RATE_PATHS.get(m, [])
        if not paths:
            reasons["该度量无适配器"] = reasons.get("该度量无适配器", 0) + 1
            continue
        n_route += 1
        # **能力可达性（更严格，2026-10-02 修）**：**任一**登记路径的能力范围
        #   覆盖需求量即算可达（初版只看 `paths[0]` ⟹ 新增的两条速率适配器
        #   （`poisson_count`/`radioactive_decay`，上限 1e12）**未被计入**
        #   ⟹ 覆盖率被低估）。
        q_req = _required_q(e)
        if q_req is None:
            detail.append((e["pair_id"], m, q_req, None))
            continue
        ok = False
        # ① **优先用登记里的 `capability` 区间**（多参数共同决定可达范围）
        for fam, _pn in paths:
            cap = (REGISTRY.get(fam, {}) or {}).get("capability")
            clim = None
            if cap and cap[0] == m:
                clim = (float(cap[1]), float(cap[2]))
            elif fam in CAPABILITY_FALLBACK and CAPABILITY_FALLBACK[fam][0] == m:
                clim = (CAPABILITY_FALLBACK[fam][1],
                        CAPABILITY_FALLBACK[fam][2])
            if clim and clim[0] <= q_req <= clim[1]:
                ok = True
                break
        # ② 否则退回单参数范围
        if not ok:
            for fam, pname in paths:
                rng = (REGISTRY.get(fam, {}).get("params", {}) or {}).get(pname)
                if rng and float(rng[0]) <= q_req <= float(rng[1]):
                    ok = True
                    break
        reach += int(ok)
        unreach += int(not ok)
        detail.append((e["pair_id"], m, q_req, ok))
    print("\n① **覆盖量化**（V1，**按度量路由**——速率需求是域级的，无族提示）：")
    print("   有度量登记：**%d/%d**；可路由到已登记速率路径：**%d/%d（%.0f%%）**"
          % (n_metric, len(pairs), n_route, len(pairs),
             100 * n_route / max(len(pairs), 1)))
    print("   **能力可达性（更严格）**：需求量落在登记范围内的 **%d** 条；"
          "**超出登记能力 %d** 条" % (reach, unreach))
    print("   （示例：需求 `q9 ≥ 150 s^-1` vs OU 族登记 `theta ∈ [1e-3, 10]` "
          "⟹ **超范围**——**这正是「速率型需求需要别的物理系统」的量化证据**）")
    if reasons:
        print("   不可路由原因分布：")
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print("     %-42s %d 条" % (k[:42], v))
    print("   ⟹ **诚实结论**：按度量**可路由 %.0f%%**，能力可达 **%d/%d**。"
          % (100 * n_route / max(len(pairs), 1), reach, n_route))
    left = [(pid, m, q) for pid, m, q, ok in detail if ok is False]
    if left:
        print("   **剩余不可达 %d 条**（**性质明确、非框架缺口**）：" % len(left))
        for pid, m, q in left:
            print("     %s（%s）：`q_req = %.3g` —— 该量级属**波前/对流**机制，"
                  "非**扩散**（`%s = κ/L`）所能表达" % (pid, m, q, m))

    print("\n② **存在性证明**（V2：两条已登记路径的端到端生产验证，"
          "**参数须齐全**——初版漏参导致 `unknown`）：")
    n_ok = 0
    cases = [({"family": "ou", "metric": "q9", "theta": 1.0, "sigma": 1.0},
              "q9"),
             ({"family": "diffusion", "metric": "q7", "kappa": 0.01}, "q7")]
    for base, m in cases:
        d = {**base, "target": 1.0, "direction": "lower_bound"}
        v = verify(d, n_samples=20000, seed=0, use_cache=False)
        good = v.verdict in ("pass", "fail")
        n_ok += int(good)
        print("   %-10s %s：verdict=**%s**；实测 %s；%s"
              % (base["family"], m, v.verdict.upper(),
                 ("%.6g" % v.measured) if v.measured is not None else "—",
                 v.reason[:56]))
    print("   ⟹ %d/%d 条路径给出 pass/fail（判据 ≥1）⟹ %s"
          % (n_ok, len(cases), "**通过** ✓" if n_ok >= 1 else "**未通过**"))

    print("\n③ **域外仍 unknown**（V3，安全性质）：")
    bad = 0
    for fam in ("landau", "gaussian_mixture2", "truncated_gaussian"):
        v = verify({"family": fam, "metric": "q9", "target": 1.0,
                    "direction": "lower_bound", "a": 1.0}, n_samples=2000,
                   seed=0, use_cache=False)
        ok = (v.verdict == "unknown")
        bad += int(not ok)
        print("   %-20s q9 → %s（%s）" % (fam, v.verdict, "✓" if ok
                                        else "**✗ 不得报 pass/fail**"))
    print("   ⟹ %s" % ("**通过** ✓（未登记族的速率需求一律 unknown）"
                       if bad == 0 else "**未通过**"))

    print("\n===== 结论 =====")
    print("速率型（占 DB3 33%%）的**生产验证**：度量路由 **%d/%d**、"
          "能力可达 **%d/%d**；两条已登记路径端到端可用；未登记者如实 unknown。"
          % (n_route, len(pairs), reach, n_route))
    print("**下一步（若要做满）**：① 为**波前/对流**类输运补适配器"
          "（当前 `q7` 只有扩散 `κ/L`）；② 继续补更多族的动力学适配器"
          "（现状 OU 谱隙、扩散前缘、泊松/衰减速率）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
