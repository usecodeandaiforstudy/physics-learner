# -*- coding: utf-8 -*-
"""需求 → β 标定（2026-10-02，第 107 条）——把占位的 β=4/1 换成可推导值。

**问题**：奖励 `R = β·I_TY − I_XT` 里的 β 一直是占位常量（v0「定理」档 β=4、
其他处 β=1）。架构里 B 的职责是「需求 → 信息指标向量（**奖励参数**）」，
**β 就应由需求推导**——这是 §4.2 缺口①「B 映射严格性（生死环节）」的
工程落点，也是断点③的正面攻坚。

**标定口径（本模块确立，可审计）**：
  1. **需求 → 目标信息量 `I_TY_req`**（按已登记的换算规则，逐类可查）：
       - `E_req_nats` 直接给出        ⟹ `I_TY_req = E_req_nats`；
       - `n_environments_count = M`（适应类；规则表已用 ln M）
                                      ⟹ `I_TY_req = ln M`；
       - `delta_dimensionless = δ`（稳定裕度类）⟹ `I_TY_req = −ln δ`；
       - **精度类** `q1_req`（Fisher 下界）⟹ `I_TY_req = ½·ln(1 + q1_req)`
         （高斯信道约定，与 §2.3 桥接式同族）；
       - 其余（输运/快/韧性等**未给信息量**者）⟹ **不标定，如实报告**
         `unmapped`（不臆造换算）。
  2. **β 由 IB 曲线的局部切线给出**（Lagrange 乘子的定义）：
       在 `I_TY = I_TY_req` 处取曲线切线，`β_req = 1 / (dI_TY/dI_XT)`；
     IB 曲线取自 DB2 该族/系统的 `ib_curve`（已落库，31 点 logspace(−2,3)）。
     **目标超出曲线 I_TY 值域时如实报 `out_of_range`，不外推**。
  3. **严格性档位因子**：`strictness` = 定理/推导/类比/猜想 →
     `{定理:1.0, 推导:0.8, 类比:0.5, 猜想:0.3}`（v0 约定的形式化，
     **登记为约定而非普适定律**）。

**本模块不产生新数据**；输出为可审计的标定记录（含推导链）。

用法（项目根目录）：
  python pipeline/beta_calibration.py            # 扫全部 DB3 锚点并报告
  python pipeline/beta_calibration.py DB3-0040   # 单条详细推导链
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
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

TIER_FACTOR = {"定理": 1.0, "推导": 0.8, "类比": 0.5, "猜想": 0.3}

__all__ = ["calibrate_pair", "BetaCalibration", "load_ib_curve"]


@dataclass
class BetaCalibration:
    pair_id: str
    status: str                    # ok | unmapped | out_of_range | no_curve
    i_ty_req: float | None = None
    beta_raw: float | None = None
    beta: float | None = None
    tier: str = ""
    tier_factor: float = 1.0
    conversion: str = ""
    curve_source: str = ""
    achieved_i_ty: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"pair_id": self.pair_id, "status": self.status,
                "i_ty_req": self.i_ty_req, "beta_raw": self.beta_raw,
                "beta": self.beta, "tier": self.tier,
                "tier_factor": self.tier_factor,
                "conversion": self.conversion,
                "curve_source": self.curve_source,
                "achieved_i_ty": self.achieved_i_ty, "notes": self.notes}


def load_ib_curve(family: str | None = None,
                  system_id: str | None = None,
                  fallback: bool = True) -> tuple[dict | None, str]:
    """从 DB2 取一条 IB 曲线（按 system_id 优先，其次按 family 首条）。

    `fallback=True` 时，family/system 均无匹配则退到任意可用曲线，
    并在返回的 source 里**标明「回退：非目标族」**。
    """
    if not DB2.exists():
        return None, "缺 DB2"
    # **2026-10-02 加回退**：family 指定但无匹配时，退到任意可用曲线，并在
    # source 里**标明这是回退**（避免把回退当匹配）。
    any_curve: tuple[dict, str] | None = None
    with open(DB2, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = rec.get("ib_curve")
            if not c or not c.get("beta"):
                continue
            curve = {"beta": np.asarray(c["beta"], dtype=float),
                     "I_XT": np.asarray(c["I_XT"], dtype=float),
                     "I_TY": np.asarray(c["I_TY"], dtype=float)}
            src = "DB2 %s（family=%s）" % (rec.get("system_id"),
                                          rec.get("family"))
            if system_id and rec.get("system_id") == system_id:
                return curve, src
            if family and rec.get("family") == family:
                return curve, src
            if any_curve is None and float(curve["I_TY"].max()) > 1e-9:
                any_curve = (curve, src + "［回退：非目标族］")
    if fallback and any_curve:
        return any_curve
    return None, "DB2 中未找到匹配 IB 曲线"


def diag() -> int:
    """曲线盘点 + 饱和诊断：判断 out_of_range 是「网格到头」还是「曲线饱和」。"""
    print("=" * 72)
    print("IB 曲线盘点与饱和诊断")
    print("=" * 72)
    per_fam: dict[str, list[tuple[float, float, float]]] = {}
    n_rows = n_usable = 0
    with open(DB2, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_rows += 1
            c = rec.get("ib_curve")
            if not c or not c.get("beta"):
                continue
            n_usable += 1
            y = np.asarray(c["I_TY"], dtype=float)
            b = np.asarray(c["beta"], dtype=float)
            per_fam.setdefault(rec.get("family") or "?", []).append(
                (float(b.max()), float(y[-1]), float(y.max())))
    print("DB2 行 %d；含可用 IB 曲线 %d" % (n_rows, n_usable))
    print("\n%-16s %5s %12s %12s %10s %s"
          % ("family", "行数", "β_max", "I_TY(β_max)", "I_TY_max",
             "末段仍在升？"))
    for fam, rows in sorted(per_fam.items()):
        bmax = float(np.mean([r[0] for r in rows]))
        y_end = float(np.mean([r[1] for r in rows]))
        y_max = float(np.mean([r[2] for r in rows]))
        rising = y_end >= 0.98 * max(y_max, 1e-12)
        print("%-16s %5d %12.3g %12.4f %10.4f %s"
              % (fam, len(rows), bmax, y_end, y_max,
                 "**是（扩 β 网格即可）**" if rising else "否（已饱和）"))
    print("\n判读：多数族「末段仍在升」→ out_of_range 属**采样网格问题**，"
          "扩 β 网格可解；")
    print("      「已饱和」→ 上限来自 **DB2 的 I(X;T)/I(T;Y) 计算标度**"
          "（n_bins 影响 I_XT 标度）,")
    print("      β 标定须等 DB2 重算——**不要靠外推硬凑**。")
    return 0


def _info_class(pair: dict) -> tuple[str, str]:
    """判定需求里「信息」的**种类**（2026-10-02，第 107 条）。

    **本会话第 138 条起：判据的单一真源迁至 `pipeline/reward_types.py`**
    （`PROP_TO_CLASS` + `classify_property`），本函数保留**更细的理由文本**，
    但**类名与判据以 reward_types 为准**（避免两处漂移）。

    三类"都叫信息/nats 但对象不同"的量：
      - `param`：关于**被估计参数**的信息（精度类）→ **可**与 IB 曲线桥接；
      - `barrier`：**势垒/保持**量（`E_req_nats` 作能量单位）→ 不是信息量；
      - `env_index`：关于**工况索引**的信息（适应类 ln M）→ 目标变量不同。
    另有 `rate_ratio`（速率类，按参考尺度换算，**约定**）。
    **不可桥接者一律如实标出，不硬凑**。
    """
    prop = (pair.get("requirement_structured", {}) or {}).get("property_id", "")
    tv = (pair.get("requirement_structured", {}) or {}).get("target_value",
                                                            {}) or {}
    # **单一真源核对**：类名由 reward_types 给出（两者不一致即为缺陷）
    try:
        from pipeline.reward_types import classify_property
        cls_ref, _why_ref = classify_property(prop, tv)
    except Exception:  # noqa: BLE001
        cls_ref = None
    if cls_ref == "constraint_barrier":
        return "constraint_barrier", (
            "**约束型（势垒/保持）**：`E_req_nats` 量的是寿命/能量，"
            "**不是关于参数的信息量** ⟹ **不适用 β 标定**（β 加权项属 IB 型"
            "目标，此处是约束）；应走**已有的达标型奖励**"
            "（`a_network/req_env.py`：`R=1 if q≥q_req else exp(−((q_req−q)/σ)²)`）。"
            "第 113 条实测：强行用保持曲线标定会得到 β≈0 或 β<0 的**假解锁**")
    if cls_ref == "constraint_env":
        return "constraint_env", (
            "**约束型（工况索引）**：要求可区分 M 种工况是**约束**；"
            "且 landau 信道容量 ≤0.22 nats（最多区分 e^0.22≈1.2 种）"
            "⟹ 需求 4–16 种超出该族能力（第 113 条）。**不适用 β**，"
            "应走达标型奖励")
    if cls_ref == "rate_ratio":
        # **速率类（第 109 条）**：q9/q7 是**速率**量（谱隙、前缘速度），本身不是
        # 信息量；但 DB3 为其**文档化了参考尺度**（如「机电执行器 1e1–1e3 s^-1」）。
        # 沿用与 q1 **同一约定**：需求严苛度 = q_req/q_ref，隐含信息
        # ½·ln(1+q_req/q_ref)。**标为「约定」而非定理**——量纲上"速率比 → 信息"
        # 需理论侧补推导（已列入待办），此处不假装它是定理。
        return "rate_ratio", ("**速率类**（%s）：按「除以文档化参考尺度」的"
                              "**同一约定**换算（约定待理论侧推导）"
                              % (prop or "未标"))
    if cls_ref == "param":
        return "param", ""
    if cls_ref is None:
        # 回退：旧判据（保底，防 reward_types 导入失败时行为改变）
        if prop in {"stability", "memory", "lifetime", "retention"} or \
                "E_req_nats" in tv:
            return "constraint_barrier", "约束型（势垒/保持，回退判据）"
        if "n_environments_count" in tv or prop == "adaptability":
            return "constraint_env", "约束型（工况索引，回退判据）"
        if prop in {"response_speed", "transport", "flow", "speed", "rate"}:
            return "rate_ratio", "速率类（回退判据）"
        return "param", ""
    # reward_types 判定为 unknown（性质未登记）⟹ 交回旧默认（保持既有行为）
    return "param", ""


def _metric_numeric(pair: dict, prefix: str) -> tuple[float | None, str]:
    """取某个 q 分量（`q1`/`q7`/`q9`…）的**数值**。

    **顺序很重要（2026-10-02 修）**：先 `worked_example.result`（已算好的
    `q9 ≥ 150 s^-1`），再 `computation`，**最后**才是 `expr`——因为 expr 常是
    **带常数的公式**（如 `3/t_settle`），直接取数会拿到公式常数（3）而非
    求得值（150）。
    """
    import re
    we = pair.get("worked_example", {}) or {}
    for key in ("result", "computation"):
        s = str(we.get(key, "") or "")
        m = re.search(r"\b%s\w*\s*[≥>=]+\s*([0-9]+(?:\.[0-9]+)?"
                      r"(?:[eE][-+]?[0-9]+)?)" % prefix, s)
        if m:
            return float(m.group(1)), "worked_example.%s（%s≥N）" % (key, prefix)
    for k, v in (pair.get("info_metric_vector", {}) or {}).items():
        if k.startswith(prefix) and isinstance(v, dict) and v.get("expr"):
            q = _q1_from_expr(v["expr"])
            if q is not None:
                return q, "expr(%s)［无 result 可依，可能为公式常数］" % k
    for key in ("result", "computation"):
        s = str(we.get(key, "") or "")
        nums = [float(x) for x in re.findall(
            r"([0-9]+(?:\.[0-9]+)?[eE][-+]?[0-9]+)", s)]
        if nums:
            return max(nums, key=abs), "worked_example.%s（量级最大数）" % key
    return None, ""


def _q1_numeric(pair: dict) -> tuple[float | None, str]:
    """取 q1 的数值：expr → `worked_example.result`（**优先 `q1 ≥ N` 形态**）
    → `computation`（第 108 条修）。

    修因：`_q1_from_expr` 取"最后一个数字"，在 `"q1 ≥ 2.4e22 m^-2"` 中会把
    **单位指数 −2 误当 q1=2** ⟹ 必须先匹配 `q1 ≥/= <数字>` 形态。
    """
    import re
    for v in (pair.get("info_metric_vector", {}) or {}).values():
        if isinstance(v, dict) and v.get("expr"):
            q1 = _q1_from_expr(v["expr"])
            if q1 is not None:
                return q1, "expr"
    we = pair.get("worked_example", {}) or {}
    # ① 显式 `q1 ≥ N` / `q1 = N` 形态（result 优先，含算好的值）
    for key in ("result", "computation"):
        s = str(we.get(key, "") or "")
        m = re.search(r"q1\s*[≥>=]+\s*([0-9]+(?:\.[0-9]+)?"
                      r"(?:[eE][-+]?[0-9]+)?)", s)
        if m:
            return float(m.group(1)), "worked_example.%s（q1≥N 形态）" % key
    # ② 退化：取量级最大的数字（跳过单位指数）
    for key in ("result", "computation"):
        s = str(we.get(key, "") or "")
        nums = [float(x) for x in re.findall(
            r"([0-9]+(?:\.[0-9]+)?[eE][-+]?[0-9]+)", s)]
        if nums:
            return max(nums, key=abs), "worked_example.%s（量级最大数）" % key
    return None, ""


def _q1_dimensionful(pair: dict) -> bool:
    """**精确**判定 q1 是否带量纲（第 108 条修）。

    初版用"结果串里出现任何单位词"判定 → 误报（把无量纲 q1 也归一了）。
    改为两条硬判据：① q1 表达式含 SI 常数（`k_B` 等）；
    ② `q1 ≥ <数字>` **紧跟单位**（如 `m^-2`、`s^-1`、`N/m`）。
    """
    import re
    for v in (pair.get("info_metric_vector", {}) or {}).values():
        if isinstance(v, dict) and re.search(r"\b(k_B|hbar|h_planck)\b",
                                             str(v.get("expr", ""))):
            return True
    we = pair.get("worked_example", {}) or {}
    s = str(we.get("result", "") or "")
    return bool(re.search(r"q1\s*[≥>=]+\s*[0-9.]+(?:e[-+]?[0-9]+)?\s*"
                          r"(m\^?-?\d|s\^?-?\d|kg|J\b|N/m)", s))


def _reference_scale(pair: dict) -> tuple[float | None, str]:
    """取 q 的参考尺度（第 108/109 条）。

    **为什么必须归一**：q 常是**有量纲的定向 Fisher / 速率**（如 R-STIFF-001 的
    `q1 ≥ K/(k_B T) = 2.4e22 m^-2`、`q9 ≥ 150 s^-1`）。只有除以**文档化的
    参考尺度**（取该段量程的**几何中点**）才得到无量纲比。

    **多段尺度（2026-10-02 修）**：参考尺度串常按域分段（如 DB3-0009：
    「机电执行器 1e1–1e3 s^-1；光电探测 1e6–1e10 s^-1；热系统 1e-2–1 s^-1」）。
    初版把首尾两个数当一段（√(10·0.01)）⟹ 取错。现按**段切分**并依
    「域/需求文本」关键词选段；选不中则取**第 1 段**并在说明里标注歧义。
    """
    import re
    we = pair.get("worked_example", {}) or {}
    s = str(we.get("reference_scale", "") or "")
    if not s:
        return None, ""
    segs = [x for x in re.split(r"[；;]|(?<=[0-9])\s*[，,]\s*(?=[^\d])", s)
            if x.strip()]
    parsed: list[tuple[str, float, float]] = []
    for seg in segs:
        nums = re.findall(r"([0-9]+(?:\.[0-9]+)?[eE][-+]?[0-9]+)", seg)
        if not nums:
            nums = re.findall(r"(?<![\d.eE-])(\d+(?:\.\d+)?)(?![\d.eE])", seg)
        vals = [float(x) for x in nums if float(x) > 0]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        parsed.append((seg.strip()[:24], lo, hi))
    if not parsed:
        return None, ""
    dom = str(pair.get("domain", "")) + str(pair.get("requirement_text", ""))
    KEY = {"control": ["机电", "执行器", "actuator"],
           "mechanics": ["机电", "执行器", "机械"],
           "optics": ["光电", "探测"], "thermal": ["热"],
           "energy": ["热", "电池"], "acoustics": ["声"]}
    keys = KEY.get(str(pair.get("domain", "")), [])
    chosen, amb = parsed[0], len(parsed) > 1
    for seg, lo, hi in parsed:
        if any(k in seg or k in dom for k in keys):
            chosen, amb = (seg, lo, hi), False
            break
    seg, lo, hi = chosen
    how = "几何中点 √(%g·%g)［段：%s］" % (lo, hi, seg)
    if amb:
        how += "**多段尺度未按域匹配到，取第 1 段（存在歧义）**"
    return math.sqrt(lo * hi), how


def _looks_dimensionful(pair: dict) -> bool:
    """兼容旧名（内部转调精确判定 `_q1_dimensionful`）。"""
    return _q1_dimensionful(pair)


def i_ty_required(pair: dict) -> tuple[float | None, str, list[str]]:
    """需求 → 目标信息量（按登记换算）；未登记则返回 (None, ...)。

    **⚠️ 口径对齐（2026-10-02 修，第 107 条）**：DB2 的 IB 曲线是**族级信道、
    单次观测**的 I_TY；而需求里的 q1 形如 `1/(n·σ²)`（**已含样本数 n**），
    直接换算得到的是 **n 次观测的合计信息量**。二者相差 n 倍——必须把需求
    信息量**按 n 折算到单次观测**再与曲线比较（与第 105 条「标定范围错配」
    同一类：**口径/scope 错配**）。无 n 的场合按单次处理并**如实标注假设**。
    """
    notes: list[str] = []
    tv = pair["requirement_structured"].get("target_value", {}) or {}
    qv = pair.get("info_metric_vector", {}) or {}
    # **先判信息种类（第 107 条）**：barrier / env_index 两类**不可**与
    # 参数推断型 IB 曲线桥接——如实标出，不硬凑。
    icls, ireason = _info_class(pair)
    if icls == "rate_ratio":
        # 速率类：取 q9 / q7 的数值，除以文档化参考尺度（与 q1 同约定）
        ref, ref_how = _reference_scale(pair)
        if not ref or ref <= 0:
            notes.append("[rate_ratio] %s；但 pair **未文档化参考尺度** ⟹ "
                         "无法归一（不臆造）" % ireason)
            return None, "needs_reference_scale", notes
        for pref in ("q9", "q7", "q5"):
            qv_, src_ = _metric_numeric(pair, pref)
            if qv_ is not None:
                snr = qv_ / ref
                total = 0.5 * math.log(1.0 + snr)
                notes.append("[rate_ratio·**约定**] %s：%s=%.4g（取自 %s），"
                             "q_ref=%.4g（%s）⟹ SNR %.4g ⟹ %.4f nats"
                             % (ireason, pref, qv_, src_, ref, ref_how, snr,
                                total))
                notes.append("**该换算是约定而非定理**（速率比 → 信息量），"
                             "理论侧推导已列入待办；结论中须标注")
                n_s = _extract_n(tv, pair)
                if n_s and n_s > 1:
                    notes.append("并按 n=%g 折算单次" % n_s)
                    return (total / n_s,
                            "[约定] ½·ln(1+%s/q_ref)/n，%s≈%.4g，n=%g"
                            % (pref, pref, qv_, n_s), notes)
                return total, ("[约定] ½·ln(1+%s/q_ref)，%s≈%.4g"
                               % (pref, pref, qv_)), notes
        notes.append("[rate_ratio] 未找到 q9/q7/q5 数值")
        return None, "unmapped", notes
    if icls != "param":
        # **约束型需求不计入「标定失败」**（第 114 条）：它们**本就不该用 β**
        # ——β 加权项是 IB 型目标的参数，而势垒/工况数/速率是**约束**，
        # 对应项目已实现的**达标型奖励**（`req_env`）。故单独成一类
        # `not_applicable:*`，与「失败」区分开。
        if icls.startswith("constraint_"):
            notes.append("[%s] %s" % (icls, ireason))
            return None, "not_applicable:%s" % icls, notes
        notes.append("[%s] %s" % (icls, ireason))
        return None, "wrong_quantity:%s" % icls, notes
    n_samples = _extract_n(tv, pair)
    # 1) 直接给出的信息量
    if "E_req_nats" in tv:
        total = float(tv["E_req_nats"])
        if n_samples and n_samples > 1:
            notes.append("E_req_nats=%g 视为**合计**，按 n=%g 折算到单次观测"
                         % (total, n_samples))
            return (total / n_samples,
                    "E_req_nats/n（合计 %g nats ÷ n=%g）" % (total, n_samples),
                    notes)
        notes.append("E_req_nats=%g（按单次观测处理，需求未给 n）" % total)
        return total, "E_req_nats 直接给出（单次）", notes
    # 2) 适应类：工况数 M
    if "n_environments_count" in tv:
        m = float(tv["n_environments_count"])
        notes.append("适应类按 ln(M)=%.3f nats 处理（规则表既有口径）"
                     % math.log(m))
        return math.log(m), "ln(M)，M=%g" % m, notes
    # 3) 稳定裕度类：δ
    if "delta_dimensionless" in tv:
        d = float(tv["delta_dimensionless"])
        if d <= 0 or d >= 1:
            notes.append("δ=%g 越界，未换算" % d)
            return None, "", notes
        return -math.log(d), "−ln(δ)，δ=%g" % d, notes
    # 4) 精度类：q1_req（Fisher 下界，**已含 n**）→ 折算单次
    for k, v in qv.items():
        if k.startswith("q1") and isinstance(v, dict) and v.get("expr"):
            q1, src = _q1_numeric(pair)
            if q1 is None:
                notes.append("q1 数值无法取得（expr 与 worked_example 均无数字）："
                             "%s" % v["expr"])
                return None, "", notes
            # **参考尺度归一（第 108 条）——判据是"是否文档化了 ref"，而不是
            # "结果串里有没有单位"**：`reference_scale` 字段的存在本身就意味着
            # 「此 q1 需参考尺度才成为无量纲 SNR」（如 DB3-0003 写"无量纲，
            # 1/应变²"却仍记有 1e19–1e22 的参考尺度）。无 ref 记录的 q1
            # （如 DB3-0040 的 1/(nσ_rel²)=33.33）本身就是 SNR。
            ref, ref_how = _reference_scale(pair)
            dimful = _q1_dimensionful(pair)
            if ref and ref > 0:
                snr = q1 / ref
                total = 0.5 * math.log(1.0 + snr)
                notes.append("q1=%.4g（取自 %s）；文档化 reference_scale "
                             "q1_ref=%.4g（%s）⟹ SNR %.4g%s"
                             % (q1, src, ref, ref_how, snr,
                                "［有量纲］" if dimful else "［无量纲但需尺度］"))
            elif dimful:
                notes.append("q1=%.4g 有量纲（取自 %s）但 pair 未给 "
                             "reference_scale ⟹ 无法归一（不臆造尺度）"
                             % (q1, src))
                return None, "needs_reference_scale", notes
            else:
                total = 0.5 * math.log(1.0 + q1)
                notes.append("q1=%.4g（取自 %s，无参考尺度记录 ⟹ 视为 SNR）"
                             "⟹ ½ln(1+q1)=%.4f nats" % (q1, src, total))
            if n_samples and n_samples > 1:
                notes.append("q1_req 已含 n=%g ⟹ 合计 %.4f nats，折算单次 "
                             "%.6f nats" % (n_samples, total,
                                            total / n_samples))
                return (total / n_samples,
                        "½·ln(1+q1_req%s)/n，q1_req≈%.4g，n=%g"
                        % ("/q1_ref" if (dimful and ref) else "", q1,
                           n_samples), notes)
            return total, ("½·ln(1+q1_req%s)（单次），q1_req≈%.4g"
                           % ("/q1_ref" if (dimful and ref) else "", q1)), notes
    notes.append("该类需求未给出信息量（输运/快/韧性等）→ 不标定")
    return None, "", notes


def _extract_n(tv: dict, pair: dict) -> float | None:
    """从 target_value / given 里取样本数或等效计数 n。"""
    for k, v in tv.items():
        if "n_samples" in k or k.endswith("_count"):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    given = pair.get("requirement_structured", {}).get("given", {}) or {}
    for k, v in given.items():
        if "n_samples" in k or "n_obs" in k:
            val = v.get("value") if isinstance(v, dict) else v
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    # 从 q1 的 expr 里找 'n=300' / '400·' 之类的样本数（保守：仅识别 n=<数>）
    import re
    for v in (pair.get("info_metric_vector", {}) or {}).values():
        if isinstance(v, dict) and v.get("expr"):
            m = re.search(r"1/\(\s*([0-9]+(?:\.[0-9]+)?)\s*[·*]", v["expr"])
            if m:
                return float(m.group(1))
    return None


def _q1_from_expr(expr: str) -> float | None:
    """从 q1 的 expr 里取出数值（形如 '… = 6.25' 或 '1/(400·4e-4) = 6.25'）。"""
    import re
    m = re.findall(r"=\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)", expr)
    if m:
        return float(m[-1])
    m = re.findall(r"([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)", expr)
    return float(m[-1]) if m else None


def beta_from_curve(curve: dict, i_ty_req: float
                    ) -> tuple[float | None, float | None, str]:
    """在 IB 曲线上求 β：取 I_TY = i_ty_req 处的切线 β = 1/(dI_TY/dI_XT)。"""
    x, y = curve["I_XT"], curve["I_TY"]
    if len(x) < 4:
        return None, None, "曲线点太少"
    if i_ty_req > float(y.max()) or i_ty_req < float(y.min()):
        return None, float(y.max()), "out_of_range（目标 %.3f 不在 I_TY ∈ [%.3f, %.3f]）" \
            % (i_ty_req, y.min(), y.max())
    j = int(np.argmin(np.abs(y - i_ty_req)))
    lo, hi = max(0, j - 1), min(len(x) - 1, j + 1)
    dx = float(x[hi] - x[lo])
    dy = float(y[hi] - y[lo])
    if abs(dx) < 1e-12 or abs(dy) < 1e-12:
        return None, float(y[j]), "切线退化（dx=%g, dy=%g）" % (dx, dy)
    slope = dy / dx                      # dI_TY/dI_XT
    return 1.0 / slope, float(y[j]), "切线斜率 %.4g ⟹ β_raw=%.4g" % (slope,
                                                                    1.0 / slope)


def calibrate_pair(pair: dict) -> BetaCalibration:
    pid = pair.get("pair_id", "?")
    tier = pair.get("strictness", "") or ""
    # **未登记档位不得静默取默认（2026-10-02 修）**：`TIER_FACTOR.get(tier, 0.5)`
    # 会把「定理（马尔可夫系统）」这类**复合档名**静默当成 0.5——须显式报告，
    # 并由 `tier_exact` 字段标明是否精确命中登记表。
    tier_exact = tier in TIER_FACTOR
    factor = TIER_FACTOR.get(tier, 0.5)
    cal = BetaCalibration(pair_id=pid, status="unmapped", tier=tier,
                          tier_factor=factor)
    if tier and not tier_exact:
        cal.notes.append("档位「%s」**未在登记表中**（TIER_FACTOR 键：%s）"
                         "⟹ 暂用默认 0.5，**须补登记或归一档名**"
                         % (tier, "/".join(TIER_FACTOR)))
    v, conv, notes = i_ty_required(pair)
    cal.notes.extend(notes)
    if v is None:
        # 状态透传：`wrong_quantity:*`（信息不同源）与
        # `not_applicable:*`（**约束型，本就不该用 β**）都要保留前缀，
        # 否则报告里会被压成 `unmapped`（初版即如此）。
        if conv.startswith(("wrong_quantity", "not_applicable")):
            cal.status = conv
        else:
            cal.status = "unmapped"
        return cal
    cal.i_ty_req, cal.conversion = v, conv
    fam = _family_hint(pair)
    curve, src = load_ib_curve(family=fam)
    cal.curve_source = src
    if curve is None:
        cal.status = "no_curve"
        return cal
    beta_raw, achieved, how = beta_from_curve(curve, v)
    cal.notes.append(how)
    cal.achieved_i_ty = achieved
    if beta_raw is None:
        cal.status = "out_of_range" if "out_of_range" in how else "no_curve"
        return cal
    cal.beta_raw = beta_raw
    cal.beta = beta_raw * factor
    cal.status = "ok"
    return cal


def _family_hint(pair: dict) -> str | None:
    """由规则/域粗判该需求对应的族（用于挑 IB 曲线；取不到则用任意首条）。"""
    rule = (pair.get("rule_ids") or [""])[0]
    return {"R-PRECISE-001": "ou", "R-FAST-001": "ou",
            "R-COND-001": "diffusion", "R-STABLE-001": "landau",
            "R-STIFF-001": "elastic", "R-TOUGH-001": "landau",
            "R-ROBUST-001": "ou", "R-ADAPT-001": "landau"}.get(rule)


def main() -> int:
    if "--diag" in sys.argv:
        return diag()
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    pairs = []
    for p in sorted(PAIRS.glob("DB3-*.json")):
        pairs.append(json.loads(p.read_text(encoding="utf-8")))
    if args:
        pairs = [p for p in pairs if p["pair_id"] == args[0]]
        if not pairs:
            print("未找到 %s" % args[0])
            return 1
        for p in pairs:
            cal = calibrate_pair(p)
            print("=" * 66)
            print("%s（%s，域 %s，tier=%s）" % (p["pair_id"],
                                              p["requirement_structured"]
                                              ["property_id"],
                                              p["domain"], cal.tier))
            print("需求：%s" % p["requirement_text"])
            print(json.dumps(cal.as_dict(), ensure_ascii=False, indent=2))
        return 0

    print("=" * 72)
    print("需求 → β 标定（扫 %d 条 DB3 锚点）" % len(pairs))
    print("=" * 72)
    cals = [calibrate_pair(p) for p in pairs]
    by_status: dict[str, int] = {}
    for c in cals:
        by_status[c.status] = by_status.get(c.status, 0) + 1
    print("状态分布：%s" % by_status)
    ok = [c for c in cals if c.status == "ok"]
    print("可标定 %d/%d；β 范围 %.3f–%.3f（中位 %.3f）"
          % (len(ok), len(cals),
             min([c.beta for c in ok], default=float("nan")),
             max([c.beta for c in ok], default=float("nan")),
             float(np.median([c.beta for c in ok])) if ok else float("nan")))
    # **回退曲线记账（2026-10-02 加，第 110 条）**：DB2 中 59% 的行**没有**
    # IB 曲线（`compute_ib_curve` 的两个显式空分支）⟹ 多数标定实际用的是
    # **回退曲线（非目标族）**。这必须显式计数——否则"111 条可标定"会被
    # 误读为"111 条都有本族曲线支撑"。
    n_fb = sum(1 for c in ok if "回退" in c.curve_source)
    print("  ⚠️ 其中 **%d/%d 条用的是回退曲线（非目标族）**——因 DB2 中该族"
          "无 IB 曲线（见 `pipeline/extra_curves.py --families`）；"
          "结论强度须按此折减" % (n_fb, len(ok)))
    # 单调性检查（**同换算类型 + 同曲线**才有可比性：β 是曲线在 I_TY 处切线
    # 的函数，跨族混用不同曲线时"同 I_TY 不同 β"是正常的）
    print("\n单调性检查（按「换算类型 × 曲线」分组——同组才可比）：")
    groups: dict[tuple[str, str], list] = {}
    for c, p in zip(cals, pairs):
        if c.status == "ok":
            fam = c.curve_source.split("family=")[-1].split("）")[0]
            key = (c.conversion.split("，")[0], fam)
            groups.setdefault(key, []).append((c.i_ty_req, c.beta))
    n_viol_tot = n_cmp_tot = 0
    for (conv, fam), rows in sorted(groups.items()):
        if len(rows) < 3:
            continue
        rows.sort()
        inc = sum(1 for i in range(len(rows) - 1)
                  if rows[i + 1][1] >= rows[i][1] - 1e-9)
        n_viol = len(rows) - 1 - inc
        n_viol_tot += n_viol
        n_cmp_tot += len(rows) - 1
        print("  %-30s [%s] %3d 点，单调升 %d/%d%s"
              % (conv, fam, len(rows), inc, len(rows) - 1,
                 "" if n_viol == 0 else "  ← %d 处违规" % n_viol))
    if n_cmp_tot:
        print("  合计：%d/%d 单调（%.1f%%）"
              % (n_cmp_tot - n_viol_tot, n_cmp_tot,
                 100.0 * (n_cmp_tot - n_viol_tot) / n_cmp_tot))
    print("  说明：跨族混用曲线时「同 I_TY 不同 β」属正常；残留违规若仍在"
          "同族同换算内，则须查**曲线切线随 I_TY 单调性**或刻度问题。")
    print("\n未标定原因分布（按**状态 × 属性**汇总，供登记工作排期）：")
    by_prop: dict[tuple[str, str], int] = {}
    for c, p in zip(cals, pairs):
        if c.status == "ok":
            continue
        prop = p["requirement_structured"].get("property_id", "?")
        by_prop[(c.status, prop)] = by_prop.get((c.status, prop), 0) + 1
    for (st, prop), n in sorted(by_prop.items(), key=lambda kv: -kv[1]):
        print("  %-26s %-14s %3d" % (st, prop, n))
    print("\n归类说明：")
    print("  not_applicable:constraint_barrier **约束型（势垒/保持）**——"
          "`E_req_nats` 是寿命/能量而非信息量 ⟹ **本就不该用 β**；")
    print("      应走已有的**达标型奖励**（`a_network/req_env.py`）。"
          "第 113 条实测：强行标定得 β≈0 或 β<0（假解锁）。")
    print("  not_applicable:constraint_env     **约束型（工况索引）**——"
          "同理不用 β；且 landau 信道容量 ≤0.22 nats（第 113 条）。")
    print("  wrong_quantity:rate               **速率类**（q9/q7）——现按"
          "「除以文档化参考尺度」的**约定**换算（约定待理论侧推导）。")
    print("  unmapped                          该属性未登记换算 / 数据本身 "
          "UNMAPPED（无 q 向量）")
    print("  out_of_range                      换算后确实超出该族信道 I_TY "
          "上限 → **不外推**；须核需求尺度或换族")
    print("\n**β 标定的适用范围（第 114 条结论）**：仅「关于参数的**信息量**」类"
          "（precision/robustness/stiffness 等）适用 β；")
    print("  阈值/约束型（势垒、工况数）**不用 β**，走达标型奖励；"
          "速率类按约定换算（标注「约定」）。")
    print("\n完整明细：")
    for c, p in zip(cals, pairs):
        if c.status != "ok":
            prop = p["requirement_structured"].get("property_id", "?")
            print("  %s：%s（%s）" % (c.pair_id, c.status, prop))
    return 0


if __name__ == "__main__":
    sys.exit(main())
