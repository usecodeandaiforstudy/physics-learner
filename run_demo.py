# -*- coding: utf-8 -*-
"""端到端演示：需求文本 → B v0 三模块 → q 规格 → DB2 指纹最近邻 → top-k 系统。

这是开源最小闭环的入口脚本：
  RequirementParser → RuleMatcher → Composer（b_network/）
  → DB2Index 方向性分位匹配（db2/db2.jsonl，1530 条指纹）

匹配原理（诚实声明，写在输出里）：
- DB2 v0 指纹在 k_B T=1 自然单位下计算（玩具尺度，fisher 值 O(0.1–100)），
  而真实需求的 q 经桥接常数换算后是天文数字——绝对数值不可直接比。
- 因此匹配采用【方向性分位得分】：每个 q 分量对应一个 DB2 代理量，
  lower_bound/increase → 高分位加分，upper_bound → 低分位加分；
  得分 = 各分量代理分位（或其补）的均值。
- 代理量映射（DB2 v0 无谱隙/锥结构字段，代理已在表中注明）：
    q1_fisher_directional → fisher_trace（T 节点）
    q1_noise_direction / q2_posterior_sharpness → fisher_det
    q4_dpi_margin → I(T;Y)（mi_profile[1]）
    q5_channel_capacity → max I(X;T)（IB 曲线）
    q7_info_propagation_speed → fisher_trace（代理）
    q9_spectral_gap → params_used 中的弛豫率（θ/λ/rate，ou/decay 族直接可得）
  无代理的分量（q6/q8/q10）跳过并注明。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b_network.composer import Composer
from b_network.requirement_parser import RequirementParser
from b_network.rule_matcher import RuleMatcher

sys.stdout.reconfigure(encoding="utf-8")

DB2_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db2", "db2.jsonl")

# q 分量 → (代理量名, 需要的方向)：'high' = 高分位加分；'low' = 低分位加分
Q_PROXY = {
    "q1_fisher_directional":     ("fisher_trace", "high"),
    "q1_noise_direction":        ("fisher_det",   "low"),
    "q2_posterior_sharpness":    ("fisher_det",   "low"),
    "q4_dpi_margin":             ("i_ty",         "high"),
    "q5_channel_capacity":       ("i_xt_max",     "high"),
    "q7_info_propagation_speed": ("fisher_trace", "high"),   # 代理（无锥结构字段）
    "q9_spectral_gap":           ("rate",         "high"),    # 仅 ou/decay 族有弛豫率真值
}
# 弛豫率（谱隙 λ2）语义按族精确取参：ou→theta，radioactive_decay→rate。
# poisson/laplace 的 rate 是到达率、lc 无阻尼、ho 的 omega 是频率——均非弛豫，不取。
RATE_BY_FAMILY = {"ou": "theta", "radioactive_decay": "rate"}


class DB2Index:
    """DB2 指纹池：抽取代理量并预计算分位。"""

    def __init__(self, path: str = DB2_PATH):
        self.records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))
        self.features = [self._features(r) for r in self.records]
        quantities = ["fisher_trace", "fisher_det", "i_ty", "i_xt_max", "rate"]
        self.pct = {}
        for qname in quantities:
            vals = np.array([f[qname] if f[qname] is not None else np.nan
                             for f in self.features])
            finite = vals[np.isfinite(vals)]
            self.pct[qname] = {
                "sorted": np.sort(finite),
                "have": np.isfinite(vals),
            }

    # ------------------------------------------------------------------
    @staticmethod
    def _features(rec: dict) -> dict:
        fps = rec.get("fingerprints", {})
        tkey = next((k for k in fps if "__T__" in k), next(iter(fps), None))
        fp = fps.get(tkey, {}) if tkey else {}
        trace = fp.get("fisher_trace")
        det = fp.get("fisher_det")
        mi = rec.get("mi_profile") or [None, None]
        i_ty = mi[1] if len(mi) > 1 and mi[1] is not None else None
        ib = rec.get("ib_curve") or {}
        i_xt_max = max(ib.get("I_XT", [np.nan])) if ib.get("I_XT") else None
        params = ((rec.get("scan") or {}).get("params_used")) or {}
        rate_key = RATE_BY_FAMILY.get(rec.get("family", ""))
        rate = params.get(rate_key) if rate_key else None
        return {"fisher_trace": trace, "fisher_det": det, "i_ty": i_ty,
                "i_xt_max": i_xt_max, "rate": rate}

    # ------------------------------------------------------------------
    def _percentile(self, qname: str, idx: int) -> float | None:
        meta = self.pct[qname]
        if not meta["have"][idx]:
            return None
        v = self.features[idx][qname]
        sorted_vals = meta["sorted"]
        return float(np.searchsorted(sorted_vals, v, side="right")) / len(sorted_vals)

    # ------------------------------------------------------------------
    def match(self, spec, top_k: int = 5):
        """方向性分位得分 → (系统级聚合) top-k。返回 [(system_id, score, evidence), ...]。"""
        comp_scores = []   # [(分量, 代理量, 方向)]
        for comp, entry in spec.q.items():
            if comp in Q_PROXY:
                qname, want = Q_PROXY[comp]
                comp_scores.append((comp, qname, want))
        if not comp_scores:
            return []

        n = len(self.records)
        rec_score = np.full(n, np.nan)
        rec_ev = [dict() for _ in range(n)]
        for idx in range(n):
            parts = []
            for comp, qname, want in comp_scores:
                p = self._percentile(qname, idx)
                if p is None:
                    continue
                parts.append(p if want == "high" else 1.0 - p)
                rec_ev[idx][comp] = "%s=%s" % (
                    qname, _fmt(self.features[idx][qname]))
            if parts:
                rec_score[idx] = float(np.mean(parts))

        # 系统级聚合（避免同一系统 30 个扫描点刷屏）：取该系统最好参数点
        by_sys: dict[str, list[int]] = {}
        for i, r in enumerate(self.records):
            by_sys.setdefault(r["system_id"], []).append(i)
        sys_list = []
        for sid, idxs in by_sys.items():
            vals = rec_score[idxs]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                best = int(np.nanargmax(rec_score[idxs])) if np.isfinite(rec_score[idxs]).any() else None
                b = idxs[int(np.argmax([rec_score[i] if np.isfinite(rec_score[i]) else -1
                                       for i in idxs]))]
                sys_list.append((sid, float(np.nanmax(vals)), rec_ev[b],
                                 self.records[b]))
        sys_list.sort(key=lambda x: -x[1])
        return sys_list[:top_k]


def _fmt(v) -> str:
    if v is None:
        return "—"
    return "%.4g" % v


def run_requirement(pipeline, index, text: str, top_k: int = 5) -> None:
    parser, matcher, composer = pipeline
    print("=" * 78)
    print("需求：%s" % text)
    req = parser.parse(text)
    if req.unmapped:
        spec = composer.compose(req)
        print("解析：无属性命中 → %s" % spec.summary())
        return
    spec = composer.compose(req)
    print("解析：property=%s direction=%s target=%s" % (
        "+".join(req.property_ids), req.direction, req.target_value or "（无数值，方向性）"))
    print("规则：%s strictness=%s" % (spec.rule_ids, spec.strictness))
    for comp, entry in spec.q.items():
        num = "（数值 ≈ %.4g）" % entry["numeric"] if "numeric" in entry else ""
        rd = " [%s]" % entry["reading"] if "reading" in entry else ""
        print("  q.%s %s %s%s%s" % (comp, entry["direction"], entry["expr"], num, rd))
    for flag in spec.ambiguity_flags:
        print("  ⚑ %s" % flag)
    if spec.unmapped:
        print("  → %s" % spec.summary())
        return
    hits = index.match(spec, top_k=top_k)
    if not hits:
        print("匹配：无可评分分量（该需求涉及的分量为 DB2 v0 未覆盖字段）")
        return
    print("DB2 top-%d 系统（方向性分位得分，0–1）：" % top_k)
    print("  %-4s %-28s %-18s %-10s %s" % ("排名", "system_id", "family", "得分", "参数点"))
    for rank, (sid, score, ev, rec) in enumerate(hits, 1):
        pv = (rec.get("scan") or {}).get("scan_value")
        print("  %-4d %-28s %-18s %-10.3f %s=%s" % (
            rank, sid, rec["family"], score,
            (rec.get("scan") or {}).get("scan_param", "p"), _fmt(pv)))


def main() -> int:
    print("加载 DB2 指纹池 ...")
    index = DB2Index()
    print("共 %d 条记录 / %d 个系统" % (len(index.records),
                                          len({r["system_id"] for r in index.records})))
    parser = RequirementParser()
    matcher = RuleMatcher()
    composer = Composer(matcher)
    pipeline = (parser, matcher, composer)

    demos = [
        "弹簧要足够硬：刚度不低于 100 N/m",
        "温度传感器要精密：经 16 次独立采样平均后，温度估计标准差不超过 0.05 K",
        "执行器响应要快：响应时间不超过 20 ms",
        "散热片要把 5 mm 厚铜层快速导通：热锋在 5 s 内传过",
        "隔热层要保住 1 mm 尺度的温度图样：1 s 观测窗内热模糊不超过该尺度",
        "快速伺服平台又要硬又要快：刚度不低于 200 N/m，响应不超过 10 ms",
        "产品外壳材料要有高级质感、手感温润",
    ]
    for text in demos:
        run_requirement(pipeline, index, text)

    print("=" * 78)
    print("说明：")
    print("1. 绝对数值满足性（物理单位 q vs 自然单位指纹）须 DB5 验证层介入，")
    print("   本演示的匹配为方向性分位——这是 DB2 v0 玩具单位下的诚实做法。")
    print("2. 已知局限：q1 的代理是全方向 fisher_trace（DB2 v0 未存方向分解 g_dd），")
    print("   刚度类需求下高 trace 的非弹性族也会排名靠前；DB2 v1 增加 Fisher 方向谱后可精确化。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
