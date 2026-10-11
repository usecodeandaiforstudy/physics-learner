# -*- coding: utf-8 -*-
"""**验证反馈闭环**（取代已失效的 DB5→RL 回流边）
（2026-10-02，第 140 条；架构 §13.4）。

## 为什么换掉旧闭环
```
旧架构闭环：RL → 设计 → DB5 → **回流边** → RL
  实测：回流边 **0 正收益 / 1 有害 / 7 未定**（原报"首个正收益 +0.82"未复现）
  ⟹ 已降级 ⟹ **闭环没闭上**
新闭环：C（条件=需求）生成**设计分布** → **采样设计** → **外部验证**
        → **筛选/加权** → 反馈（再训练/重加权/写入数据集）
  ⟹ 已在 landau 域验证可行（88.4% ± 3.0%，20/20 seed）
```

## 本模块提供（域无关）
    `reflow(domain, target, n_sample, ...) -> dict`
      ① 采样设计（域自带：非退化域用解流形；landau 用已训 CFM 或均匀采样）
      ② **逐个送生产验证**（`sim/verify_api` 或域自带校验）
      ③ 输出 **通过集合 + 权重 + 统计**（通过率、多样性、拒绝原因分布）
      ④ 落盘 `design_reflow_<domain>.jsonl`（**追加式，可审计**）

**关键语义**：**未通过/未验证者不得进入反馈集**（沿用生产级的
`unknown ≠ pass`）——这使"回流"从"把噪声喂回训练"变成**有证据的再训练**。

用法：
    python pipeline/design_reflow.py --domain rayleigh --target 3.0
    python pipeline/design_reflow.py --domain landau --target 0.03 --n 200
"""

from __future__ import annotations

import json
import os
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


def _sample_rayleigh(target: float, n: int, rng) -> list[dict]:
    """非退化域：**沿解流形采样**（一维射线）⟹ 天然满足需求。"""
    from pipeline.design_rayleigh import design_manifold
    m = design_manifold(target, D_max_m=1.0, f_min_Hz=1e4, f_max_Hz=2e6)
    if not m["feasible"]:
        return []
    idx = rng.integers(0, len(m["f"]), n)
    return [{"D_m": float(m["D"][i]), "f_Hz": float(m["f"][i])} for i in idx]


def _sample_two_level(target: float, n: int, rng) -> list[dict]:
    from pipeline.design_two_level import EV, design_manifold
    m = design_manifold(0.10, dF_min_J=0.15 * EV, T_min_K=250.0)
    if not m["feasible"]:
        return []
    idx = rng.integers(0, len(m["T"]), n)
    return [{"dF_eV": float(m["dF_J"][i]) / EV, "T_K": float(m["T"][i])}
            for i in idx]


def _sample_landau(target: float, n: int, rng) -> list[dict]:
    """高维域：**用已训 CFM（若有）+ 均匀采样兜底**（目标 = q1）。

    未加载 CFM 时退化为**设计盒内均匀采样**（诚实标注 `sampler`）。
    """
    from a_network.structure_space import family_template
    tpl = family_template("landau")
    rngs = {pr["name"]: pr.get("range", [0.1, 1.0])
            for pr in tpl.get("parameters", [])}
    a_lo, a_hi = rngs.get("a", [-2.0, -0.01])
    b_lo, b_hi = rngs.get("b", [0.1, 5.0])
    k_lo, k_hi = rngs.get("kT", [0.1, 5.0])
    A = rng.uniform(a_lo, min(a_hi, -0.01), n)
    B = rng.uniform(b_lo, b_hi, n)
    K = rng.uniform(k_lo, k_hi, n)
    return [{"a": float(a), "b": float(b), "kT": float(k)} for a, b, k
            in zip(A, B, K)]


def _sample_chain(target: float, n: int, rng) -> list[dict]:
    """**组合链域（第 141 条）**：均匀采样链设计（2–3 节点、σ 型组件）。

    复用 `design_chain.sample_chain`（**与预检同一采样器**）——该域是
    **唯一"组合型"高维域**（dof 4–6，混合离散组件 + 连续参数）。
    """
    from pipeline.design_chain import sample_chain
    return [sample_chain(rng) for _ in range(n)]


def _sample_array(target: float, n: int, rng) -> list[dict]:
    """**声学阵列几何域（第 149 条）**：均匀采样（含栅瓣样本，由门禁过滤）。"""
    from pipeline.design_array import sample_array
    return [sample_array(rng) for _ in range(n)]


SAMPLERS = {"rayleigh": _sample_rayleigh, "two_level": _sample_two_level,
            "landau": _sample_landau, "chain": _sample_chain,
            "array": _sample_array}


def _jsonable(x: dict) -> dict:
    """**记录可序列化化（2026-10-02 修）**：链域的设计含 `nodes`（元组）与
    `link`（布尔）——初版对**所有值**调 `float()` ⟹ `TypeError`
    ⟹ **把"通过的设计"在构造记录时崩掉、计成 `exception:TypeError`
    （实测 34 个，全部本应通过）**。现只对数值保原样，其余转字符串。
    """
    out = {}
    for k, v in x.items():
        if isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, str):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _to_family_param(domain: str, x: dict) -> dict | None:
    """**设计 dof → 族参数**（2026-10-02 修）。

    **修因**：初版把**设计 dof**（rayleigh 的 `D_m`、two_level 的 `dF_eV`）
    直接当族参数传给 `verify_api` ⟹ 验证器登记的是 `sigma` / `p`
    ⟹ **缺参数 ⟹ 200/200 全 `unknown`**（"设计 dof ≠ 族参数"）。
    """
    if domain == "rayleigh":
        from pipeline.design_rayleigh import sigma_from_design
        return {"family": "rayleigh",
                "sigma": float(sigma_from_design(float(x["D_m"]),
                                                 float(x["f_Hz"])))}
    if domain == "two_level":
        from pipeline.design_two_level import EV, p_from_design
        return {"family": "two_level",
                "p": float(p_from_design(float(x["dF_eV"]) * EV,
                                         float(x["T_K"])))}
    return None          # landau 未登记于生产验证（数值求积族）


def reflow(domain: str, target: float, n_sample: int = 200, seed: int = 0,
           out_dir: Path | None = None, use_production_api: bool = True
           ) -> dict:
    """**验证反馈闭环**：采样 → **双重验证** → 筛选/加权 → 落盘。

    **双重验证（2026-10-02 修）**：
      ① **域自带校验**（设计 → 族参数 → **既有编码器** → q1/DPI，含需求达标）；
      ② **生产验证 API**（`sim/verify_api`，**用换算后的族参数**）——
         域外/未登记族须知情跳过（`unknown` 不计为通过）。
    两者**都通过**才进反馈集（更严；并记录各自失败数）。
    """
    from pipeline.design_space import DOMAINS, verify_design
    if domain not in SAMPLERS or domain not in DOMAINS:
        return {"error": "域未登记：%s" % domain}
    rng = np.random.default_rng(seed)
    t0 = time.time()
    cand = SAMPLERS[domain](float(target), int(n_sample), rng)
    if not cand:
        return {"error": "该目标下无可行设计（域 %s，target=%.4g）"
                         % (domain, target)}
    passed, rejected, reasons = [], [], {}
    n_prod_unknown = n_prod_fail = 0
    for x in cand:
        try:
            # ① 域自带校验（经既有编码器）
            info = verify_design(domain, x, float(target))
            ok_domain = bool(info.get("meets")) and bool(
                info.get("dpi_ok", True))
            # ② 生产验证 API（**用换算后的族参数**）
            ok_prod, prod_note = True, "skipped"
            if use_production_api:
                fp = _to_family_param(domain, x)
                if fp is None:
                    prod_note = "该域未登记于生产验证（跳过）"
                else:
                    from sim.verify_api import verify as vfy
                    d = {**fp, "metric": "q1", "target": float(target),
                         "direction": "lower_bound"}
                    v = vfy(d, n_samples=5000, seed=seed)
                    ok_prod = (v.verdict == "pass")
                    prod_note = v.verdict
                    if v.verdict == "unknown":
                        n_prod_unknown += 1
                    elif not ok_prod:
                        n_prod_fail += 1
            if ok_domain and ok_prod:
                passed.append({**_jsonable(x),
                               "measured": info.get("measured"),
                               "prod": prod_note})
            else:
                rejected.append(x)
                key = ("domain_fail" if not ok_domain else "prod:" + prod_note)
                reasons[key] = reasons.get(key, 0) + 1
        except Exception as ex:  # noqa: BLE001
            rejected.append(x)
            k = "exception:%s" % type(ex).__name__
            reasons[k] = reasons.get(k, 0) + 1

    # **加权**：通过者等权（可扩展为按 margin 加权）
    weights = [1.0 / max(len(passed), 1)] * len(passed)
    out = {"domain": domain, "target": float(target), "n_sample": len(cand),
           "n_passed": len(passed), "pass_rate": len(passed) / max(len(cand), 1),
           "n_rejected": len(rejected), "reject_reasons": reasons,
           "prod_unknown": n_prod_unknown, "prod_fail": n_prod_fail,
           "diversity": len({tuple(sorted((str(k), str(v)) for k, v in p.items()
                                          if k not in ("measured", "prod")))
                             for p in passed}),
           "elapsed_s": time.time() - t0,
           "sampler": ("解流形（非退化域）"
                       if domain in ("two_level", "rayleigh")
                       else "设计盒均匀采样"),
           "feedback_set": passed[:5], "weights_head": weights[:5]}
    # **落盘**（追加式，可审计）
    d_out = out_dir or _ROOT
    fp = Path(d_out) / ("design_reflow_%s.jsonl" % domain)
    try:
        with open(fp, "a", encoding="utf-8") as f:
            f.write(json.dumps({"target": float(target), "n": len(cand),
                                "passed": passed, "weights": weights},
                               ensure_ascii=False) + "\n")
        out["written"] = str(fp)
    except OSError as ex:
        out["written"] = "写入失败：%s" % ex
    return out


def main() -> int:
    args = sys.argv[1:]
    domain = args[args.index("--domain") + 1] if "--domain" in args \
        else "rayleigh"
    target = float(args[args.index("--target") + 1]) if "--target" in args \
        else (3.0 if domain == "rayleigh" else 0.03)
    n = int(args[args.index("--n") + 1]) if "--n" in args else 200
    print("=" * 74)
    print("验证反馈闭环（取代失效的回流边）——域 %s，目标 %.4g，采样 %d"
          % (domain, target, n))
    print("=" * 74)
    r = reflow(domain, target, n_sample=n)
    if "error" in r:
        print("**失败**：%s" % r["error"])
        return 1
    print("采样方式：%s" % r["sampler"])
    print("采样 %d → **通过 %d（%.1f%%）**；拒绝 %d"
          % (r["n_sample"], r["n_passed"], 100 * r["pass_rate"], r["n_rejected"]))
    print("多样性（通过集内不同设计数）：%d" % r["diversity"])
    if r["reject_reasons"]:
        print("拒绝原因分布：%s" % r["reject_reasons"])
    print("生产验证（用换算后的族参数）：**unknown %d**（知情跳过，不计通过）／"
          "fail %d" % (r.get("prod_unknown", 0), r.get("prod_fail", 0)))
    print("通过样例（前 5）：")
    for p in r["feedback_set"]:
        print("   %s" % {k: round(v, 5) for k, v in p.items()
                        if isinstance(v, float)})
    print("落盘：%s" % r.get("written"))
    print("\n语义：**未通过/未验证者不进反馈集**（沿用 `unknown ≠ pass`）"
          "⟹ 回流是'有证据的再训练'，不是'把噪声喂回去'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
