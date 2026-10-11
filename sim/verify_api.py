# -*- coding: utf-8 -*-
"""缺口① 生产化：**外部验证 API**（2026-10-02，第 135 条）。

## 从"演示级"到"生产级"缺什么
现有 `sim/simulator_adapter.py` + `sim_check.py` + `verify_designs.py` 已能验证
（一致率 50/52、真值自检 7/7），但**演示级**：
  · 未支持的族**抛 `KeyError`** ⟹ 调用方可能把异常当失败，或直接崩；
  · **没有"域外"概念**：参数超出适配器有效范围时仍会给出数字；
  · **容差写死**（5%/10%），未与估计量噪声（∝1/√n）挂钩；
  · 无**统一判定对象**（verdict/证据/可复现性）、无**审计日志**、无缓存。

## 本模块提供（生产级契约）
    `verify(design) -> Verdict`
    `Verdict.verdict ∈ {"pass", "fail", "unknown"}`——**`unknown` 是一等公民**：
      族未登记、度量不支持、参数超出有效域、样本不足 ⟹ **一律 `unknown`**，
      **绝不把"没验证"报成"通过"**（这是生产级与演示级的**根本区别**）。
    `Verdict.evidence`：实测值、目标值、容差、方法、seed、n_samples、相对误差、
      有效域判定、以及**容差来源**（由 n_samples 推出的估计量噪声）
    `verify`：**按设计哈希缓存** + **追加式审计日志**（JSONL）

## 有效域（登记，不猜）
每个适配器登记 `validity`：参数范围 + 度量集合。**域外 → `unknown`**。

## 容差策略（与估计量噪声挂钩）
score 估计器的相对误差 ~ C/√n。容差 `tol = max(tol_floor, k/√n)`，
`k` 由**真值校准**（`calibrate()` 在已知真值上测实际误差分位）给出，
而非拍脑袋。动力学类另有方法学下限（有限步长）。

用法：
    from sim.verify_api import verify
    v = verify({"family": "rayleigh", "sigma": 0.5, "metric": "q1",
                "target": 3.0, "direction": "lower_bound"})
    print(v.verdict, v.evidence)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

AUDIT_LOG = _ROOT / "sim" / "verify_audit.jsonl"
CACHE: dict[str, "Verdict"] = {}

# **有效域登记（2026-10-02）**：族/动力学 → (可用度量, 参数范围)
#   范围取自 `FAMILY_SPECS` 的族语义与 `sim_check` 已验证的取值区间；
#   **域外一律 unknown**（不猜、不外推）。
REGISTRY: dict[str, dict] = {
    "rayleigh": {"metrics": ["q1"], "params": {"sigma": (1e-3, 1e3)}},
    "weibull": {"metrics": ["q1"],
                "params": {"shape": (0.2, 20.0), "scale": (1e-3, 1e3)}},
    "lognormal": {"metrics": ["q1"],
                  "params": {"mu": (-10.0, 10.0), "sigma": (1e-2, 10.0)}},
    "frechet": {"metrics": ["q1"],
                "params": {"shape": (0.2, 20.0), "scale": (1e-3, 1e3)}},
    "ou": {"metrics": ["q9"], "kind": "dynamics",
           "params": {"theta": (1e-3, 10.0), "sigma": (1e-3, 1e3)}},
    # **扩散/输运（第 156 条扩展）**：`q7 = κ/L` 是**速度**（m/s）。
    #   DB3 的 14 条 transport 需求跨 **14 个数量级**（1.9e-9 … 1.67e5 m/s）；
    #   初版只登记 `κ ∈ [1e-4, 10]` 且 `L` 固定 0.01 ⟹ 可达仅 [0.01, 1000]。
    #   **按物理扩展**：`κ ∈ [1e-14, 10] m²/s`（热扩散 1e-6、金属中氢 1e-8、
    #   混凝土中水分 ~1e-12 ⟹ 下界 1e-14 有余量）、`L ∈ [1e-4, 1e3] m`（几何）
    #   ⟹ 可达 `q7 ∈ [1e-17, 1e5] m/s` ✓ 覆盖 13/14 条。
    "diffusion": {"metrics": ["q7"], "kind": "dynamics",
                  "params": {"kappa": (1e-14, 10.0), "L": (1e-4, 1e3)},
                  "optional": ["L"],          # 缺省注入 0.01 m（登记约定）
                  "capability": ("q7", 1e-17, 1e5)},
    # **速率型适配器（第 152 条）**：谱隙有解析式的族 ⟹ 补 DB3 速率型需求
    #   （第 151 条实测：能力可达仅 27/51，因 `ou` 的 `theta` 上限仅 10）。
    #   **有效域取物理范围**（`[1e-3, 1e12] s^-1`；DB1 的 [0.1,10] 是扫描范围）。
    "poisson_count": {"metrics": ["q9"], "kind": "rate",
                      "params": {"rate": (1e-3, 1e12)}},
    "radioactive_decay": {"metrics": ["q9"], "kind": "rate",
                          "params": {"rate": (1e-3, 1e12)}},
    # **波前/输运速度（第 158 条）**：补 `q7` 的**非扩散**机制
    #   （第 156 条后仍剩 1 条 `1.67e5 m/s`，扩散表达不了）。
    #   `v = ω·a`（声子群速度）/ `v = 1/√(L·C)`（LC 传输线）；
    #   几何参数可选（`a` 默认 5e-10 m、`inductance` 默认 1e-9 H）。
    "harmonic_oscillator": {"metrics": ["q7"], "kind": "front",
                            "params": {"omega": (0.1, 5.0),
                                       "a": (1e-12, 1e-2)},
                            "optional": ["a"],
                            "capability": ("q7", 1e-13, 5e-2)},
    "lc_circuit": {"metrics": ["q7"], "kind": "front",
                   "params": {"capacitance": (0.1, 10.0),
                              "inductance": (1e-15, 1e3)},
                   "optional": ["inductance"],
                   "capability": ("q7", 1e-2, 1e7)},
    # **注**：`two_level` **不登记速率**——`p` 是平衡量、谱隙需动力学参数
    #   （真值自检曾差 65%/96%）；它的 **q1 路径**由 `registry.json` 提供 ✓
}

# **分域容差（2026-10-02，第 165 条）**：动力学类的**全局** `K=25`（n=20000
#   ⟹ 17.7%）是**被 OU 小 θ 档拖宽的**（实测：`θ=0.2 → 21%`、`0.5 → 7.5%`、
#   `2 → 1%`；扩散 `κ̂` 与速率/波前类误差 ≲1%）。故按**参数区间**给不同容差：
#   `θ ≥ 0.5` 用**紧容差**（k=6 ⟹ 4.2%），小 θ 档保留宽容差并**在证据里标弱**。
#   ⟹ **验证力在良态区提升 ~4×，且不掩盖坏区**（如实分段，不搞"一刀切收紧"）。
TOL_BY_REGION: dict[str, list[dict]] = {
    "ou": [{"param": "theta", "min": 0.5, "k": 6.0,
            "note": "θ≥0.5：实测估计误差 ≤7.5% ⟹ 紧容差"}],
}

# **数据驱动登记表（2026-10-02，第 136 条）**：`sim/expand_registry.py`
#   在**校准通过**（p99 相对误差 ≤10%）后写入 `sim/registry.json`；
#   本模块**优先读它**并合并（新增族 → 走 `RegistryAdapter` 测量路径）。
#   **未过校准者不登记** ⟹ 其验证**如实返回 unknown**（生产级语义）。
REGISTRY_JSON = _ROOT / "sim" / "registry.json"


def load_registry() -> int:
    """合并 `registry.json`（返回新增族数）。"""
    if not REGISTRY_JSON.exists():
        return 0
    try:
        data = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    n_new = 0
    for fam, spec in data.items():
        if fam in REGISTRY:
            continue
        REGISTRY[fam] = {"metrics": spec.get("metrics", ["q1"]),
                         "kind": "registry",
                         "params": {k: tuple(v) for k, v in
                                    (spec.get("params") or {}).items()},
                         "source": spec.get("source", "registry.json")}
        n_new += 1
    return n_new


N_REGISTRY_LOADED = load_registry()

# 真值校准得到的噪声常数（`tol = max(tol_floor, k/√n)`）；由 `calibrate()` 更新
TOL_K = {"analytic": 12.0, "dynamics": 25.0}
TOL_FLOOR = {"analytic": 0.02, "dynamics": 0.05}


@dataclass
class Verdict:
    verdict: str                       # "pass" | "fail" | "unknown"
    reason: str = ""
    measured: float | None = None
    target: float | None = None
    direction: str | None = None
    tolerance: float | None = None
    rel_err: float | None = None
    adapter: str | None = None
    n_samples: int | None = None
    seed: int | None = None
    design_hash: str = ""
    evidence: dict = field(default_factory=dict)
    elapsed_s: float | None = None
    # ---- **验证证书 + 不确定度（2026-10-02，第 205 条）** ----
    #   **纪律**：`verdict` 的三态语义**一字未改** ✓（改语义会翻既有阶段结论 ✗）
    #   ⟹ 只**增加**下面这些字段，把 `pass` 从"结论"变成"**可被外人判断的证据**" ✓
    u: float | None = None             # 不确定度（取容差：它由真值校准、随 k/√n 缩放 ✓）
    margin_u: float | None = None      # 判定边界到实测值的距离／u（>0 ⟹ 达标 ✓）
    robustness: str = ""               # "robust"（|margin_u|≥1）｜"marginal"｜"n/a"
    certificate: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# **已达成第三方外部印证的族（第 198／204 条）**——证书里如实标注 ✓
EXTERNAL_CORROBORATED = {"diffusion", "two_level", "rayleigh", "weibull",
                         "landau"}
# **已知偏差（如实登记；无记录者写"未记录"，不得臆造 ✓）**
KNOWN_BIAS = {
    "diffusion": "探针落在网格节点（比 L 近 0.251%）⟹ t_half/κ̂ 偏 0.512%"
                 "（已知，非失败 ✓，见第 198 条）",
}


def _certificate(design: dict, adapter_kind: str, tol: float,
                 ev: dict) -> dict:
    """**验证证书**：让外人**不必读代码**就能判断该 `pass` 是否可采信 ✓"""
    fam = design.get("family") or design.get("kind") or "?"
    reg = REGISTRY.get(fam) or {}
    return {
        "family": fam,
        "metric": design.get("metric"),
        "model": ("解析 Fisher（族=%s，度量=%s）" % (fam, design.get("metric"))
                  if adapter_kind == "analytic" else
                  "动力学模拟（kind=%s）" % design.get("kind")),
        "method": {"adapter": adapter_kind,
                   "n_samples": design.get("_n_samples"),
                   "seed": design.get("_seed")},
        "tolerance": tol,
        "tolerance_source": ev.get("tol_source") or "真值校准（k/√n）",
        "validity_domain": reg.get("params") or {},
        "supported_metrics": reg.get("metrics") or [],
        "registry_source": reg.get("source"),
        "known_bias": KNOWN_BIAS.get(fam, "未记录 ✓"),
        "external_corroboration": (
            "**有**（三路外部方法互洽 ✓，见 sim/external_check_*.csv）"
            if fam in EXTERNAL_CORROBORATED else
            "无（设计环外·本项目自建）⚠️"),
        "verdict_rule": (
            "measured ≥ target·(1−tol)" if design.get("direction")
            == "lower_bound" else
            "measured ≤ target·(1+tol)" if design.get("direction")
            == "upper_bound" else "|measured−target|/target ≤ tol"),
        "design": {k: v for k, v in design.items()
                   if not k.startswith("_")},
    }


def _margin(measured: float, target: float, direction: str | None,
            tol: float) -> tuple[float, float]:
    """返回 `(u, margin_u)`：边界到实测值的**相对距离**（以 u=tol 归一 ✓）。

    · `lower_bound`：边界 = `target·(1−tol)` ⟹ `m = (measured−b)/b / tol` ✓
    · `upper_bound`：边界 = `target·(1+tol)` ⟹ `m = (b−measured)/b / tol` ✓
    · 其余（双侧）：`m = (tol − |measured−target|/target)/tol` ✓
    ⟹ `m > 0` ⟺ **达标** ✓（与判定规则**同式**，不引入新口径 ✓）
    """
    u = float(tol)
    if direction == "lower_bound":
        b = target * (1.0 - tol)
        return u, ((measured - b) / max(abs(b), 1e-12)) / max(tol, 1e-12)
    if direction == "upper_bound":
        b = target * (1.0 + tol)
        return u, ((b - measured) / max(abs(b), 1e-12)) / max(tol, 1e-12)
    rel = abs(measured - target) / max(abs(target), 1e-12)
    return u, (tol - rel) / max(tol, 1e-12)


def design_hash(design: dict, n_samples: int, seed: int) -> str:
    key = json.dumps({"d": {k: design[k] for k in sorted(design)},
                      "n": n_samples, "s": seed}, sort_keys=True,
                     default=str)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def in_domain(design: dict) -> tuple[bool, str]:
    """**有效域判定**（不通过则不得给出 verdict）。

    **可选参数（第 156 条）**：族可声明 `optional` 参数（如 `diffusion` 的
    几何长度 `L`，缺省由 `verify()` 注入登记约定值）——它们**不参与必填检查**，
    但仍受范围约束（若给了就必须在域内）。
    """
    fam = design.get("family") or design.get("kind")
    if fam not in REGISTRY:
        return False, "族/动力学未登记于有效域（%s）" % fam
    reg = REGISTRY[fam]
    metric = design.get("metric")
    if metric not in reg["metrics"]:
        return False, "度量 %s 未被 %s 适配器支持（支持 %s）" % (
            metric, fam, reg["metrics"])
    opt = set(reg.get("optional") or [])
    for pname, (lo, hi) in reg["params"].items():
        if pname not in design:
            if pname in opt:
                continue                 # 可选且未给 ⟹ 由 verify() 注入
            return False, "缺参数 %s" % pname
        v = float(design[pname])
        if not (lo <= v <= hi):
            return False, "参数 %s=%.4g 超出有效域 [%.4g, %.4g]" % (
                pname, v, lo, hi)
    return True, ""


def _analytic_true(design: dict) -> float | None:
    """解析真值（用于自检与容差校准；**不作为验证结果**）。"""
    from sim.simulator_adapter import FAMILY_SPECS
    fam = design["family"]
    meta = FAMILY_SPECS.get(fam)
    if not meta:
        return None
    p = {k: float(design[k]) for k in meta["params"]}
    try:
        return float(meta["fisher_trace"](p))
    except Exception:  # noqa: BLE001
        return None


def tolerance_for(adapter: str, n_samples: int) -> float:
    k = TOL_K.get(adapter, 15.0)
    fl = TOL_FLOOR.get(adapter, 0.03)
    return max(fl, k / math.sqrt(max(n_samples, 1)))


def verify(design: dict, n_samples: int = 20000, seed: int = 0,
           use_cache: bool = True, log: bool = True) -> Verdict:
    """**生产级验证**：域外 → `unknown`（**绝不报 pass**）。"""
    t0 = time.time()
    # **可选参数的默认注入（第 156 条）**：必须在 `in_domain` **之前**做，
    #   否则可选参数会被当成必填（初版注入写在适配器分支里 ⟹ 域判定先失败）。
    design = dict(design)
    _fam0 = design.get("family") or design.get("kind")
    _reg0 = REGISTRY.get(_fam0) or {}
    _inj: dict = {}
    for _p in (_reg0.get("optional") or []):
        if _p not in design:
            design[_p] = 0.01 if _p == "L" else None
            _inj[_p] = "默认 %.3g（登记约定）" % design[_p]
    dh = design_hash(design, n_samples, seed)
    if use_cache and dh in CACHE:
        v = CACHE[dh]
        v.evidence = dict(v.evidence, cached=True)
        return v

    ok, why = in_domain(design)
    if not ok:
        v = Verdict(verdict="unknown", reason=why, design_hash=dh,
                    target=design.get("target"), direction=design.get(
                        "direction"), seed=seed, n_samples=n_samples,
                    elapsed_s=time.time() - t0,
                    robustness="n/a",
                    # **unknown 也要证书**：说明"为什么 unknown"（域外？度量未支持？）
                    #   以及该族的**有效域** ⟹ 外人可判断这个 unknown 是否合理 ✓
                    certificate=_certificate(
                        dict(design, _n_samples=n_samples, _seed=seed),
                        str(design.get("family") or design.get("kind")),
                        0.0, {"tol_source": "未进入容差阶段（域外/度量未支持）"}))
        if log:
            _append(v)
        CACHE[dh] = v
        return v

    fam = design.get("family") or design.get("kind")
    reg = REGISTRY[fam]
    adapter_kind = reg.get("kind", "analytic")
    metric = design["metric"]
    ev_extra: dict = {}          # **证据补充**（如注入的默认几何参数）
    try:
        if adapter_kind == "analytic":
            from sim.simulator_adapter import AnalyticAdapter
            out = AnalyticAdapter().run(design, n_samples=n_samples, seed=seed)
            measured = float(out.get("q1_fisher_directional"))
        elif adapter_kind == "registry":
            # **校准通过的新族（第 136 条）**：由编码器节点 → scipy 分布 →
            #   全参数 Fisher 迹（与真值同口径，已校准 ≤2.2%）
            from sim.expand_registry import measure_for_verify
            pname = next(iter(REGISTRY[fam]["params"]))
            measured = measure_for_verify(fam, float(design[pname]),
                                          n_samples=n_samples, seed=seed)
            measured = float("nan") if measured is None else float(measured)
        elif adapter_kind == "rate":
            # **速率型适配器（第 152 条）**：数值估计 λ̂（真值与解析式对照）
            from sim.rate_adapter import RATE_MODELS, RateAdapter
            meta = RATE_MODELS.get(fam) or {}
            missing = [k for k in (meta.get("params") or [])
                       if k not in design]
            if missing:
                measured = float("nan")
                ev_extra["rate_missing"] = "缺参数 %s" % missing
            else:
                out = RateAdapter().run(design, n_samples=n_samples, seed=seed)
                measured = float(out.get("q9_spectral_gap", float("nan")))
                ev_extra["lam_true"] = float(out["lam_true"])
                ev_extra["rate_label"] = out.get("label")
        elif adapter_kind == "front":
            # **波前/输运速度（第 158 条）**：数值估计 `v̂`（真值 = 解析波速）
            from sim.front_adapter import FrontAdapter
            out = FrontAdapter().run(design, n_samples=n_samples, seed=seed)
            measured = float(out.get("q7_info_propagation_speed",
                                     float("nan")))
            ev_extra["v_true"] = float(out["v_true"])
        else:
            from sim.simulator_adapter import DynamicsAdapter
            # **`kind` 注入（2026-10-02 修，真 bug）**：`DynamicsAdapter.run`
            #   按 `spec["kind"]` 分派，缺省为 `"ou"`；而验证器给的是
            #   `family` ⟹ **`diffusion` 的 q7 路径一直走成 `_ou`**（实测
            #   `KeyError: 'theta'`）——**该路径从未被端到端跑过**。
            d2 = dict(design)
            d2.setdefault("kind", fam)
            # **几何默认（登记约定）**：diffusion 需要特征长度 `L`；缺省取项目
            #   登记的 `L_DIFF = 0.01 m`，并**写入证据**（不静默）。
            if fam == "diffusion" and "L" not in d2:
                d2["L"] = 0.01
                ev_extra["L"] = "默认 0.01 m（登记约定 L_DIFF）"
            out = DynamicsAdapter().run(d2, n_samples=n_samples, seed=seed)
            # **返回键前缀匹配（2026-10-02 修，真 bug）**：适配器返回
            #   `q9_spectral_gap` / `q7_info_propagation_speed`，而初版只查
            #   `out[metric]`（精确名）⟹ **实测"适配器返回非有限值"**。
            measured = float("nan")
            for k, val in out.items():
                if str(k).startswith(str(metric)) and isinstance(
                        val, (int, float)):
                    measured = float(val)
                    break
    except Exception as ex:  # noqa: BLE001
        v = Verdict(verdict="unknown",
                    reason="适配器执行失败（%s: %s）" % (type(ex).__name__, ex),
                    design_hash=dh, seed=seed, n_samples=n_samples,
                    elapsed_s=time.time() - t0)
        if log:
            _append(v)
        CACHE[dh] = v
        return v

    if not np.isfinite(measured):
        v = Verdict(verdict="unknown", reason="适配器返回非有限值",
                    design_hash=dh, seed=seed, n_samples=n_samples,
                    elapsed_s=time.time() - t0)
        if log:
            _append(v)
        CACHE[dh] = v
        return v

    tol = tolerance_for(adapter_kind, n_samples)
    # **分域容差（第 165 条）**：`TOL_BY_REGION` 命中的族/参数区间用**更紧的 k**
    #   （良态区提升验证力），并把所用口径**写入证据**。
    tol_note = ""
    for _rg in TOL_BY_REGION.get(fam, []):
        _pv = design.get(_rg["param"])
        if isinstance(_pv, (int, float)) and float(_pv) >= float(_rg["min"]):
            _k = float(_rg["k"])
            tol = max(TOL_FLOOR.get(adapter_kind, 0.02),
                      _k / math.sqrt(max(n_samples, 1)))
            tol_note = "%s（k=%.1f，分域紧容差）" % (_rg["note"], _k)
            break
    target = design.get("target")
    direction = design.get("direction", "lower_bound")
    # **OU 小 θ 提示（第 153 条）**：实测 OU 估计器误差随 θ 减小而增大
    #   （θ=0.2 → ~21%、θ=0.5 → 7.5%、θ=2 → 1%）⟹ 小 θ 下判定较弱，
    #   **写入证据**（不静默）。
    if fam == "ou" and isinstance(design.get("theta"), (int, float)) \
            and float(design["theta"]) < 0.5:
        ev_extra["weak_verdict"] = ("θ<0.5：OU 估计器误差可达 ~21%（实测），"
                                    "本次判定较弱")
    ev = {"adapter": adapter_kind, "family": fam, "metric": metric,
          "tol_source": tol_note or ("k=%.1f/√n（真值校准）"
                                     % TOL_K.get(adapter_kind, 15.0)),
          "truth": (_analytic_true(design) if adapter_kind == "analytic"
                    else None)}
    ev.update(ev_extra)          # 注入的默认参数等（不静默）
    ev.update({("injected_" + k): v for k, v in _inj.items()})
    if target is None:
        v = Verdict(verdict="unknown", reason="未给目标值（target）",
                    measured=measured, adapter=adapter_kind,
                    tolerance=tol, design_hash=dh, seed=seed,
                    n_samples=n_samples, evidence=ev,
                    elapsed_s=time.time() - t0, robustness="n/a",
                    certificate=_certificate(
                        dict(design, _n_samples=n_samples, _seed=seed),
                        adapter_kind, tol, ev))
    else:
        target = float(target)
        rel = abs(measured - target) / max(abs(target), 1e-12)
        # **判定**：以容差为界；下界型要求 measured ≥ target·(1−tol)
        if direction == "lower_bound":
            passed = measured >= target * (1.0 - tol)
        elif direction == "upper_bound":
            passed = measured <= target * (1.0 + tol)
        else:
            passed = rel <= tol
        _u, _m = _margin(float(measured), float(target), direction, tol)
        v = Verdict(verdict=("pass" if passed else "fail"),
                    reason=("%s：实测 %.6g，%s %.6g（容差 %.1f%%，"
                            "**容差为保守值**；u=%.1f%%、margin=%.2f u ⟹ %s）"
                            % ("达标" if passed else "**未达标**", measured,
                               "要求 ≥" if direction == "lower_bound"
                               else ("要求 ≤" if direction == "upper_bound"
                                     else "目标"), target, 100 * tol,
                               100 * _u, _m,
                               "**robust**" if abs(_m) >= 1.0
                               else "**marginal**（距边界不足 1u）")),
                    measured=measured, target=target, direction=direction,
                    tolerance=tol, rel_err=rel, adapter=adapter_kind,
                    design_hash=dh, seed=seed, n_samples=n_samples,
                    evidence=ev, elapsed_s=time.time() - t0,
                    u=_u, margin_u=_m,
                    robustness=("robust" if abs(_m) >= 1.0 else "marginal"),
                    certificate=_certificate(
                        dict(design, _n_samples=n_samples, _seed=seed),
                        adapter_kind, tol, ev))
    if log:
        _append(v)
    CACHE[dh] = v
    return v


def _append(v: Verdict) -> None:
    """**追加式审计日志**（JSONL；每次验证一行，含证据）。"""
    try:
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(v.as_dict(), ensure_ascii=False,
                               default=str) + "\n")
    except OSError:
        pass          # 日志失败不得影响验证本身


def calibrate(n_samples: int = 20000, seeds: int = 5) -> dict:
    """**容差校准**：在已知真值上测实际相对误差，据此设 `TOL_K`。

    **两个来源（2026-10-02 扩展，第 147 条）**：
      ① **解析适配器 4 族**（`FAMILY_SPECS`：score 估计器 vs 解析 Fisher）；
      ② **登记表 23 族**（`registry.json`：`measure_for_verify` vs
         `compute_node_fisher` 真值）——**这才是当前实际使用的验证路径**，
         其误差分位才是容差的正确依据。
    **规则（明写）**：`k = p99 · √n · 2`（**2 倍安全系数**），夹在
      `[2.0, 12.0]`（下限防过紧误判、上限防过松失力）。
    **校准用真值、验证不用**（两者严格分离）。
    """
    cases = [{"family": "rayleigh", "sigma": 0.5},
             {"family": "weibull", "shape": 2.0, "scale": 1.0},
             {"family": "lognormal", "mu": 0.0, "sigma": 0.5},
             {"family": "frechet", "shape": 3.0, "scale": 1.0}]
    from sim.simulator_adapter import AnalyticAdapter
    errs = []
    for c in cases:
        truth = _analytic_true(dict(c, metric="q1"))
        if truth is None:
            continue
        for s in range(seeds):
            out = AnalyticAdapter().run(c, n_samples=n_samples, seed=s)
            m = float(out["q1_fisher_directional"])
            errs.append(abs(m - truth) / max(abs(truth), 1e-12))
    # **② 登记表族**（**实际验证路径**；初版只校准 4 个解析族 ⟹ 依据不完整）
    n_reg = 0
    try:
        from core.fingerprint import compute_node_fisher
        from sim.expand_registry import measure_for_verify, node_of
        data = (json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
                if REGISTRY_JSON.exists() else {})
        for fam, spec in data.items():
            try:
                pname = next(iter(spec["params"]))
                lo, hi = spec["params"][pname]
                mid = 0.5 * (float(lo) + float(hi))
                node = node_of(fam, mid)
                if node is None:
                    continue
                truth = (compute_node_fisher(node) or {}).get("fisher_trace")
                if not truth:
                    continue
                for s in range(2):
                    m = measure_for_verify(fam, mid, n_samples=n_samples,
                                           seed=s)
                    if m and m > 0:
                        errs.append(abs(m - float(truth)) / abs(float(truth)))
                        n_reg += 1
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    if not errs:
        return {"n": 0}
    e = np.array(errs)
    p99 = float(np.percentile(e, 99))
    k_new = float(np.clip(p99 * math.sqrt(n_samples) * 2.0, 2.0, 12.0))
    before = dict(TOL_K)
    TOL_K["analytic"] = k_new
    # **动力学/速率类的真值校准（第 153 条）**：此前 `TOL_K["dynamics"]=25`
    #   是**拍脑袋的保守值**（n=20000 ⟹ 17.7%），且从未校准。
    #   现用**解析真值**测：OU 的 `θ̂ vs θ`、扩散的 `κ̂ vs κ`、速率族的 `λ̂ vs λ`
    #   ⟹ 取 p99 × 2 倍安全，夹在 `[2, 25]`。
    dyn_errs: list[float] = []
    dyn_note: dict = {}
    try:
        for theta in (0.2, 0.5, 2.0):
            for s in range(3):
                from sim.simulator_adapter import DynamicsAdapter
                out = DynamicsAdapter().run({"kind": "ou", "theta": theta,
                                             "sigma": 1.0}, n_samples=20000,
                                            seed=s)
                e = abs(float(out["theta_hat"]) - theta) / theta
                dyn_errs.append(e)
                dyn_note["ou_%.1f" % theta] = round(e, 3)
        # **OU 估计器的能力边界（实测）**：误差随 θ 变小而增大
        #   （θ=0.2 → ~31%、θ=0.5 → ~17%、θ=2 → ~4%）——慢混合 + 有限样本偏差，
        #   **这是容差不能收紧到 <17% 的真实原因**（数据说话，非拍脑袋）。
        for kap in (1e-3, 1e-2):
            for s in range(3):
                from sim.simulator_adapter import DynamicsAdapter
                out = DynamicsAdapter().run({"kind": "diffusion", "kappa": kap,
                                             "L": 0.01}, n_samples=20000,
                                            seed=s)
                # **用适配器自报的 `kappa_hat`（2026-10-02 修）**：初版在此
                #   用 `L²/(4·t_half·erfcinv²)` **自行复算** ⟹ 与适配器内部
                #   口径不一致 ⟹ 恒定 **88%** 的假误差（**是我造出来的**；
                #   适配器自报误差实测 **0.16%**）。
                k_hat = out.get("kappa_hat")
                if k_hat:
                    dyn_errs.append(abs(float(k_hat) - kap) / kap)
        from sim.rate_adapter import RateAdapter
        for spec in ({"family": "poisson_count", "rate": 1e3},
                     {"family": "radioactive_decay", "rate": 1e2},
                     {"family": "two_level", "p": 0.5}):
            for s in range(3):
                out = RateAdapter().run(spec, n_samples=20000, seed=s)
                lam = float(out["lam_true"])
                dyn_errs.append(abs(float(out["lambda_hat"]) - lam)
                                / max(lam, 1e-12))
    except Exception:  # noqa: BLE001
        pass
    if dyn_errs:
        de = np.array(dyn_errs)
        p99_d = float(np.percentile(de, 99))
        k_dyn = float(np.clip(p99_d * math.sqrt(n_samples) * 2.0, 2.0, 25.0))
        TOL_K["dynamics"] = k_dyn
    return {"n": len(errs), "n_registry_points": n_reg,
            "err_med": float(np.median(e)), "err_p99": p99,
            "k_before": before, "k_after": dict(TOL_K),
            "n_dynamics": len(dyn_errs),
            "dyn_err_med": (float(np.median(dyn_errs)) if dyn_errs
                            else float("nan")),
            "dyn_err_p99": (float(np.percentile(dyn_errs, 99)) if dyn_errs
                            else float("nan")),
            "dyn_note": dyn_note,
            "tol_at_20k_before": 12.0 / math.sqrt(20000.0),
            "tol_at_20k_after": max(TOL_FLOOR["analytic"],
                                    k_new / math.sqrt(20000.0)),
            "dyn_tol_at_20k_before": 25.0 / math.sqrt(20000.0),
            "dyn_tol_at_20k_after": max(TOL_FLOOR["dynamics"],
                                        TOL_K["dynamics"] / math.sqrt(20000.0))}


def main() -> int:
    print("=" * 74)
    print("缺口① 生产化：外部验证 API 自检")
    print("=" * 74)
    print("有效域登记：%d 个适配器" % len(REGISTRY))
    for k, r in REGISTRY.items():
        print("  %-12s 度量 %-6s 参数 %s"
              % (k, ",".join(r["metrics"]),
                 {p: tuple(v) for p, v in r["params"].items()}))

    print("\n一、正常验证（域内）")
    for d in [{"family": "rayleigh", "sigma": 0.5, "metric": "q1",
               "target": 3.0, "direction": "lower_bound"},
              {"family": "rayleigh", "sigma": 2.0, "metric": "q1",
               "target": 3.0, "direction": "lower_bound"}]:
        v = verify(d, n_samples=20000, seed=0)
        print("  %s → **%s**：%s" % (d, v.verdict, v.reason))

    print("\n二、**域外必须 unknown**（生产级的关键安全性质）")
    out_of_domain = [
        ({"family": "nonexistent", "sigma": 1.0, "metric": "q1",
          "target": 1.0}, "未登记族"),
        ({"family": "rayleigh", "sigma": 1.0, "metric": "q9",
          "target": 1.0}, "度量不支持"),
        ({"family": "rayleigh", "sigma": 1e6, "metric": "q1",
          "target": 1.0}, "参数超有效域"),
        ({"family": "rayleigh", "sigma": 0.5, "metric": "q1"}, "缺目标值"),
    ]
    n_unknown = 0
    for d, why in out_of_domain:
        v = verify(d, n_samples=2000, seed=0)
        ok = v.verdict == "unknown"
        n_unknown += int(ok)
        print("  [%s] %s → %s（%s）" % ("✓" if ok else "**✗**", why,
                                       v.verdict, v.reason[:44]))
    print("  ⟹ %d/%d 正确判为 unknown（**判据：100%%**）"
          % (n_unknown, len(out_of_domain)))

    print("\n三、容差校准（用真值；与验证分离）")
    cal = calibrate(n_samples=20000, seeds=3)
    e_p99 = cal.get("err_p99", float("nan"))
    print("  真值对照：**%d 点**（其中登记表族 %d 点：**实际验证路径**）"
          % (cal.get("n", 0), cal.get("n_registry_points", 0)))
    print("  误差中位 **%.4f**；p99 **%.4f**" % (cal.get("err_med", float("nan")),
                                            e_p99))
    kb, ka = cal.get("k_before", {}), cal.get("k_after", {})
    print("  规则：`k = p99·√n·2`（2 倍安全），夹在 [2, 12]")
    print("  **容差收紧**（n=20000）：%.1f%% → **%.1f%%**（k %.2f → %.2f）"
          % (100 * cal.get("tol_at_20k_before", float("nan")),
             100 * cal.get("tol_at_20k_after", float("nan")),
             kb.get("analytic", float("nan")), ka.get("analytic", float("nan"))))
    if e_p99 > 0:
        print("  ⟹ 保守度由约 **%.1f×** 降为约 **%.1f×**（实测 p99 与容差之比）"
              % (cal.get("tol_at_20k_before", 0) / e_p99,
                 cal.get("tol_at_20k_after", 0) / e_p99))
    if cal.get("n_dynamics"):
        print("  **动力学/速率类真值校准**（第 153 条；此前 `K=25` 是拍脑袋值）："
              "%d 点，误差中位 **%.3f**、p99 **%.3f**"
              % (cal["n_dynamics"], cal.get("dyn_err_med", float("nan")),
                 cal.get("dyn_err_p99", float("nan"))))
        print("    动力学容差（n=20000）：%.1f%% → **%.1f%%**（k 25 → %.2f；"
              "依据 OU θ̂ / 扩散 κ̂ / 速率 λ̂ 的**解析真值**）"
              % (100 * cal.get("dyn_tol_at_20k_before", float("nan")),
                 100 * cal.get("dyn_tol_at_20k_after", float("nan")),
                 cal.get("k_after", {}).get("dynamics", float("nan"))))

    print("\n四、可复现性（同设计同 seed ⟹ 同结果）")
    d = {"family": "weibull", "shape": 2.0, "scale": 1.0, "metric": "q1",
         "target": 2.0, "direction": "lower_bound"}
    a = verify(d, n_samples=5000, seed=7)
    CACHE.clear()
    b = verify(d, n_samples=5000, seed=7)
    same = (a.verdict == b.verdict and a.measured == b.measured)
    print("  两次（清缓存后）：%s vs %s ⟹ %s"
          % (a.measured, b.measured, "**一致 ✓**" if same else "**不一致 ✗**"))

    print("\n五、审计日志：%s" % AUDIT_LOG.name)
    n_lines = 0
    if AUDIT_LOG.exists():
        with open(AUDIT_LOG, "r", encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        print("  已累积 %d 行（追加式 JSONL，含证据）" % n_lines)
    ok = (n_unknown == len(out_of_domain)) and same and n_lines > 0
    print("\n结论：%s" % ("**生产化自检通过**" if ok else "**未通过**"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
