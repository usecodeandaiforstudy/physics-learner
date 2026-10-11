# -*- coding: utf-8 -*-
"""DB5 回流模块（第 63 条，2026-10-02）：验证结果 → 高权重经验。

design doc §2.4：「RL 运行后，DB5 验证通过的（G*，解码设计，验证结果）
以高权重加入下一轮微调」——本模块把 DB5 的 pass 记录映射回目标环境的状态，
供 SAC 以高权重注入经验缓冲（v0：注入权重 = 1 + overlap，密度重叠覆盖率
越高的验证结构权重越大）。

v0 边界（文档化）：只回流 DB5 的 landau 记录（当前唯一领域实例）；
映射 = 参数值 → 目标环境网格最近箱。
"""

from __future__ import annotations

import numpy as np

__all__ = ["load_validated_states", "describe_reflow", "DOMAIN_TO_FAMILY",
           "load_requirement_demos", "REQ_FAMILY"]

# 域 → 需求条件化环境的族索引（req_env，2026-10-02 第 92 条）
REQ_FAMILY: dict[str, int] = {
    "landau": 0, "control": 3, "acoustics": 4, "rayleigh": 4,
    "materials": 5,
}
# 度量键 → 需求度量名（DB5 的 sim.q_required 口径）
_OBJ_TO_METRIC = {
    "q1_fisher_directional": "q1",
    "q9_spectral_gap": "q9",
    "q7_info_propagation_speed": "q7",
}


def load_requirement_demos(env, min_reward: float = 0.999
                           ) -> list[tuple[int, int, float]]:
    """DB5 记录 → [(需求索引, 需求条件化状态, 权重)]（第 92 条回流形态）。

    **语义**：DB5 的验证记录是「**该设计满足其自身需求**」的证据——因此
    回流时按**记录自己的需求**（`sim.q_required` + objective）归属到对应
    需求任务，并把设计状态作为该任务的 demo。**与任务无关的域不再被剔除**
    （那是旧单任务环境的伪影，第 91 条）。

    需求匹配：按 (metric, target) 在 `env.requirements` 中查找（容差 1e-9）；
    找不到则该记录**报告为「需求未登记」**而非静默丢弃。
    """
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from db5.db5_writer import DB5Writer

    out: list[tuple[int, int, float]] = []
    best: dict[int, float] = {}          # state → 最大权重（**去重**）
    req_of: dict[int, int] = {}          # state → 需求索引
    skipped_req: set[tuple[str, float]] = set()
    for r in DB5Writer().load():
        dom = r.get("domain", "")
        fam = REQ_FAMILY.get(dom)
        if fam is None:
            continue
        sim = r.get("sim") or {}
        # **2026-10-02 修正**：objective 记在 `validation.objective`（由
        # verify_designs 写入），初版误读 `sim.objective` → 6 条模拟器证据
        # 全被按 legacy 需求处理（Q1 = 0 新 demo 的直接原因）
        obj = r.get("validation", {}).get("objective") \
            or sim.get("objective")
        metric = _OBJ_TO_METRIC.get(obj)
        target = sim.get("q_required")
        if metric is None or target is None:
            # 旧记录（无 sim 元数据）：按 legacy 需求（q1 ≥ 0.4）处理
            metric, target = "q1", 0.40
        req_idx = None
        for i, req in enumerate(env.requirements):
            if req.metric == metric and abs(req.target - float(target)) < 1e-9:
                req_idx = i
                break
        if req_idx is None:
            skipped_req.add((metric, float(target)))
            continue
        params = r["decoded_design"].get("params", {})
        key = {"landau": "a", "control": "theta", "acoustics": "sigma",
               "rayleigh": "sigma", "materials": "kappa"}.get(dom)
        if key is None or key not in params:
            continue
        val = float(params[key])
        if dom == "materials":
            val = val / KAPPA_REF_LOCAL
        grid = np.asarray(env.grids[fam])
        b = int(np.argmin(np.abs(grid - val)))
        state = env._encode(req_idx, fam, b)
        if env.reward(state) < min_reward:
            continue                     # 设计非该需求的解 → 不作 demo
        overlap = float(r["validation"].get("overlap", 0.0))
        w = 1.0 + overlap
        # **去重（第 93 条）**：多条设计可能落入同一网格箱 → 同状态只留最大
        # 权重（否则 5× 重复注入会放大同一状态的影响）
        if state not in best or w > best[state]:
            best[state] = w
            req_of[state] = req_idx
    if skipped_req:
        print("    [回流] %d 组 (metric,target) 未在 env.requirements 登记：%s"
              % (len(skipped_req), sorted(skipped_req)))
    return [(req_of[st], st, w) for st, w in sorted(best.items())]


KAPPA_REF_LOCAL = 1e-4

# 域 → 目标环境族（fam 索引）映射表（2026-10-02，第 90 条扩展）。
# **缺口**：MultiFamilyEnv 只有 3 个设计族（0=landau 网格）；缺口① 新增的
# 模拟器证据（rayleigh/OU/diffusion 域）**没有对应的族**——即 **RL 状态
# 空间窄于 DB5 证据空间**。本表把可映射域显式登记，不可映射域在
# `describe_reflow` 中**报告而非静默丢弃**。
DOMAIN_TO_FAMILY: dict[str, int] = {
    "landau": 0,
    # 缺口① 回流域（2026-10-02，第 91 条）：rl_env 已增加对应族
    #   4 = control_ou（θ，I_TY = θ/2）
    #   5 = acoustics_rayleigh（σ，I_TY = 4/σ²）
    #   6 = materials_diffusion（κ/κ_ref，I_TY = κ̃ 代理）
    "control": 4,
    "acoustics": 5,
    "rayleigh": 5,          # 历史记录的域标（与 acoustics 同族）
    "materials": 6,
}


def describe_reflow(env=None) -> dict:
    """回流边现状报告（按 source × domain × verdict 统计映射能力）。

    返回：{total, by_source, by_domain, mappable, unmapped_domains}
    —— 用于暴露「证据空间 vs 状态空间」的宽度差（第 90 条）。
    """
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from db5.db5_writer import DB5Writer

    recs = DB5Writer().load()
    by_source: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    mappable = 0
    for r in recs:
        src = r.get("provenance", {}).get("source", "?")
        by_source[src] = by_source.get(src, 0) + 1
        dom = r.get("domain", "?")
        by_domain[dom] = by_domain.get(dom, 0) + 1
        if dom in DOMAIN_TO_FAMILY and \
                r.get("validation", {}).get("verdict") == "pass":
            mappable += 1
    unmapped = sorted(d for d in by_domain if d not in DOMAIN_TO_FAMILY)
    return {"total": len(recs), "by_source": by_source,
            "by_domain": by_domain, "mappable": mappable,
            "unmapped_domains": unmapped}


def load_validated_states(env, min_logr: float = -1.0,
                          include_simulated: bool = True
                          ) -> list[tuple[int, float]]:
    """DB5 pass 记录 → [(目标环境状态, 注入权重)]（仅任务相关者）。

    §2.4 语义：高权重回流的是「RL 最优结构 G* 的解码验证通过」——任务
    相关性是隐含前提。v11 首跑教训：不过滤会把其他任务的低奖励验证记录
    （如双峰任务的 a=±1.5，在需求任务里 reward≈1e-11）高权重灌入缓冲，
    刷屏低奖励经验导致回退——故注入前按目标任务奖励过滤（logR ≥
    min_logr，即 reward ≥ e^min_logr）。

    **2026-10-02 扩展（第 90 条）**：
      - 走 `DOMAIN_TO_FAMILY` 映射表（不再硬编码 landau）；
      - `include_simulated=True` 时纳入 `provenance.source="AI+simulated"`
        的外部验证记录（缺口① 的新证据类型）；
      - 不可映射的域由 `describe_reflow` 报告（**不静默丢弃**）。

    env 须为 MultiFamilyEnv；权重 = 1 + overlap（模拟器证据用 overlap 或
    余量折算的置信度）。
    """
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from db5.db5_writer import DB5Writer

    recs = DB5Writer().load()
    out: list[tuple[int, float]] = []
    for r in recs:
        dom = r.get("domain", "")
        if dom not in DOMAIN_TO_FAMILY:
            continue
        src = r.get("provenance", {}).get("source")
        if src == "AI+simulated" and not include_simulated:
            continue
        if r["validation"].get("verdict") != "pass":
            continue
        params = r["decoded_design"].get("params", {})
        fam_idx = DOMAIN_TO_FAMILY[dom]
        grid = np.asarray(env.grids[fam_idx])
        # 参数 → 网格最近箱：按族取参数键（landau 用 a；新族用其自然参数）
        key = {"landau": "a", "control": "theta", "acoustics": "sigma",
               "rayleigh": "sigma", "materials": "kappa"}.get(dom)
        if key is None or key not in params:
            key = next((k for k in params
                        if isinstance(params[k], (int, float))), None)
        if key is None:
            continue
        val = float(params[key])
        if dom == "materials":
            val = val / 1e-4          # κ → κ̃ = κ/κ_ref（族网格口径）
        b = int(np.argmin(np.abs(grid - val)))
        state = env._encode(fam_idx, b)
        if np.log(max(env.reward(state), 1e-6)) < min_logr:
            continue   # 任务无关（其他任务的验证记录）→ 不回流
        overlap = float(r["validation"]["overlap"])
        out.append((state, 1.0 + overlap))
    return out
