# -*- coding: utf-8 -*-
"""缺口① 外部验证：解析自洽判定 vs 模拟器判定（2026-10-02）。

**核心问题**：DB5 此前的验证是「解析自洽」——按桥接定理用**声明的设计参数**
算出 q，与需求比。它信任声明、且假定**最优估计器**。模拟器验证则**真的跑**：
从模拟观测里测 q、并检查**实际可达精度**。

**两条判定臂**：
  - **analytic**：q_declared（由声明参数按桥接式算）≥ q_req ?
  - **simulator**：跑适配器 → 测 q_measured → 判定；并额外做
    **「实用估计器臂」**：用 n_req 个样本 + 直方图/矩估计（**非最优**）估计
    参数，看可达 σ 是否满足 σ_req。

**预期分歧来源（最有价值的部分）**：Cramér–Rao 界是**最优估计器**的界；
实用系统用次优估计器时达不到 → **解析判 pass、模拟判 fail**。

用法（项目根目录）：python sim/verify_designs.py [--write-db5]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from db5.db5_writer import DB5Writer, build_record  # noqa: E402
from sim.simulator_adapter import FAMILY_SPECS, get_adapter  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


# 设计表：(id, 域, 适配器, 规格, 需求)
#   需求含 q_key/q_req（阈值）与 n_req（样本预算，供实用估计器臂）
# **2026-10-02 扩容（第 93 条）**：原 6 条保留为基准；新增参数扫描
#   （acoustics σ / control θ / materials κ），用于「扩大证据量 → 重测回流」。
_LEGACY = [
    ("DES-RAY-01", "acoustics", "analytic",
     {"family": "rayleigh", "sigma": 1.10},
     {"q_key": "q1_fisher_directional", "q_req": 3.0, "n_req": 300,
      "sigma_req": 0.058}),
    ("DES-RAY-02", "acoustics", "analytic",
     {"family": "rayleigh", "sigma": 2.50},
     {"q_key": "q1_fisher_directional", "q_req": 3.0, "n_req": 300,
      "sigma_req": 0.058}),
    ("DES-OU-01", "control", "dynamics",
     {"kind": "ou", "theta": 2.30, "sigma": 1.0},
     {"q_key": "q9_spectral_gap", "q_req": 2.3, "n_req": 400,
      "sigma_req": 0.05}),
    ("DES-OU-02", "control", "dynamics",
     {"kind": "ou", "theta": 0.50, "sigma": 1.0},
     {"q_key": "q9_spectral_gap", "q_req": 2.3, "n_req": 400,
      "sigma_req": 0.05}),
    ("DES-DIF-01", "materials", "dynamics",
     {"kind": "diffusion", "kappa": 1e-4, "L": 0.01},
     {"q_key": "q7_info_propagation_speed", "q_req": 0.01, "n_req": 100,
      "sigma_req": 0.05}),
    ("DES-DIF-02", "materials", "dynamics",
     {"kind": "diffusion", "kappa": 1e-6, "L": 0.01},
     {"q_key": "q7_info_propagation_speed", "q_req": 0.01, "n_req": 100,
      "sigma_req": 0.05}),
]

# 参数扫描（通过 + 不通过混合）：acoustics q1=4/σ²（σ≤1.155 达标）
_RAY_SIGMA = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10,
              1.30, 1.60, 2.00, 3.00, 4.00]
# control q9=θ（θ≥2.3 达标）——**加密（第 94 条）**：达标侧细分，提升映射箱数
_OU_THETA = [0.50, 0.80, 1.20, 1.60, 2.00,
             2.30, 2.45, 2.60, 2.75, 3.00, 3.20, 3.40, 3.60, 3.80,
             4.00, 4.20, 4.40, 4.60, 4.80]
# materials q7=κ/L（κ≥1e-4 达标）——**加密（第 94 条）**：达标侧细分到 1e-3
# （κ̃=κ/1e-4 覆盖 1…10，与扩展后的环境网格一致）
_DIF_KAPPA = [1e-6, 5e-6, 2e-5, 5e-5, 8e-5,
              1e-4, 1.2e-4, 1.5e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 8e-4,
              1e-3]


def _build_designs() -> list[tuple]:
    """设计表：6 条基准 + 参数扫描。

    **σ_req 一致性（2026-10-02 修）**：实用估计器臂的 σ_req 必须与 q1_req
    自洽：`σ_req = 1/√(n·q1_req)` = 1/√(300·3) = **0.0333**（初版误写 0.058，
    那对应 q1=1 → 导致 3 个「实用臂」伪分歧：设计未达 q1 但实用 σ 仍达标）。
    """
    sig_req = 1.0 / np.sqrt(300 * 3.0)          # = 0.03333…
    out = []
    for d in _LEGACY:
        did, dom, kind, spec, req = d
        if req["q_key"] == "q1_fisher_directional":
            req = dict(req, sigma_req=sig_req)
        out.append((did, dom, kind, spec, req))
    for i, s in enumerate(_RAY_SIGMA):
        out.append(("DES-RAY-S%02d" % i, "acoustics", "analytic",
                    {"family": "rayleigh", "sigma": s},
                    {"q_key": "q1_fisher_directional", "q_req": 3.0,
                     "n_req": 300, "sigma_req": sig_req}))
    for i, th in enumerate(_OU_THETA):
        out.append(("DES-OU-T%02d" % i, "control", "dynamics",
                    {"kind": "ou", "theta": th, "sigma": 1.0},
                    {"q_key": "q9_spectral_gap", "q_req": 2.3,
                     "n_req": 400, "sigma_req": 0.05}))
    for i, k in enumerate(_DIF_KAPPA):
        out.append(("DES-DIF-K%02d" % i, "materials", "dynamics",
                    {"kind": "diffusion", "kappa": k, "L": 0.01},
                    {"q_key": "q7_info_propagation_speed", "q_req": 0.01,
                     "n_req": 100, "sigma_req": 0.05}))
    return out


DESIGNS = _build_designs()


def analytic_q(spec: dict, q_key: str) -> float:
    """解析自洽：按桥接式用**声明参数**算 q。"""
    if "family" in spec:
        meta = FAMILY_SPECS[spec["family"]]
        return float(meta["fisher_trace"](
            {k: float(spec[k]) for k in meta["params"]}))
    if spec.get("kind") == "ou":
        return float(spec["theta"])
    if spec.get("kind") == "diffusion":
        return float(spec["kappa"]) / float(spec["L"])
    raise KeyError(spec)


# ---- 定义口径登记册（db3/conventions/q_definitions.json，2026-10-02 第 88 条）----
_CONV = json.loads((_ROOT / "db3" / "conventions" /
                    "q_definitions.json").read_text(encoding="utf-8"))


def operational_factor(q_key: str) -> float:
    """该 q 的「操作定义 / 理论定义」因子（用于跨口径比较）。"""
    return float(_CONV["definitions"].get(q_key, {})
                 .get("operational_factor", 1.0))


def margin_class(margin: float) -> str:
    """余量分类（HANDOVER 第 88 条余量政策）。"""
    if margin >= 0.10:
        return "robust_pass"
    if margin >= -0.05:
        return "borderline"
    return "robust_fail"


def practical_sigma_rayleigh(sigma: float, n: int, seed: int) -> float:
    """实用（非最优）估计器：直方图峰位法估 σ → 可达 σ 的实测标准差。

    与 Cramér–Rao 界的差距 = **估计器效率缺口**（本脚本要暴露的东西）。
    """
    rng = np.random.default_rng(seed)
    est = []
    for _ in range(200):                     # 200 次重复实验估标准差
        x = rng.rayleigh(sigma, n)
        # 矩估计：σ̂ = sqrt(mean(x²)/2)（对 rayleigh 是**充分统计量**的变换，
        # 但有限样本有偏；直方图法更差——这里用矩估计代表「可实现的」）
        est.append(np.sqrt(np.mean(x ** 2) / 2.0))
    return float(np.std(est, ddof=1))


def main() -> int:
    write = "--write-db5" in sys.argv[1:]
    print("========== 缺口① 外部验证：解析自洽 vs 模拟器判定 ==========")
    rows, agree, diverged = [], 0, []
    for did, domain, kind, spec, req in DESIGNS:
        ad = get_adapter(kind)
        res = ad.run(spec, n_samples=200000, seed=0)
        q_decl = analytic_q(spec, req["q_key"])
        q_meas = float(res[req["q_key"]])
        # **操作定义修正**（第 88 条）：比较前把解析侧换算到操作口径
        f = operational_factor(req["q_key"])
        q_decl_op = q_decl * f
        v_a = "pass" if q_decl >= req["q_req"] else "fail"
        v_a_op = "pass" if q_decl_op >= req["q_req"] else "fail"
        v_s = "pass" if q_meas >= req["q_req"] else "fail"
        # **余量政策**：报 margin 而非二元判定
        m_s = q_meas / req["q_req"] - 1.0
        m_a = q_decl / req["q_req"] - 1.0
        cls_s, cls_a = margin_class(m_s), margin_class(m_a)
        agree += (v_s == v_a_op)          # 与**同口径**解析判定比
        if v_s != v_a_op:
            diverged.append((did, v_a_op, v_s))
        # 实用估计器臂（仅 rayleigh：有样本预算语义）
        extra = ""
        if "family" in spec and spec["family"] == "rayleigh":
            s_ach = practical_sigma_rayleigh(float(spec["sigma"]),
                                             int(req["n_req"]), seed=1)
            v_p = "pass" if s_ach <= req["sigma_req"] else "fail"
            extra = ("  实用估计器：σ 可达 %.4f vs 需 ≤ %.4f → **%s**"
                     % (s_ach, req["sigma_req"], v_p))
            if v_p != v_a:
                diverged.append((did + "(实用臂)", v_a, v_p))
        rows.append((did, q_decl, q_meas, v_a, v_s))
        print("  %s  q_声明 %.4g（操作口径 %.4g）| q_模拟 %.4g（需求 ≥ %.4g）"
              % (did, q_decl, q_decl_op, q_meas, req["q_req"]))
        print("      解析 %s（margin %+.1f%%，%s） / 模拟 %s"
              "（margin %+.1f%%，**%s**）%s"
              % (v_a_op, 100 * m_a, cls_a, v_s, 100 * m_s, cls_s, extra))
        if write:
            rec = build_record(
                {"structure_id": did, "domain": domain},
                {"family": spec.get("family", spec.get("kind")),
                 "params": {k: v for k, v in spec.items()
                            if k not in ("family", "kind")}},
                {"overlap": 1.0 if cls_s == "robust_pass" else 0.0,
                 "verdict": v_s, "objective": req["q_key"]})
            rec["validation_method"] = "simulator"
            rec["validator"] = "sim/verify_designs.py（外部模拟器，机器可复核）"
            rec["sim"] = {"adapter": kind, "seed": 0, "n_samples": 200000,
                          "tolerance": 0.05, "q_required": req["q_req"],
                          "q_measured": q_meas, "margin": m_s,
                          "margin_class": cls_s,
                          "operational_factor": f}
            rec["provenance"] = {"source": "AI+simulated",
                                 "anchor_eligible": False,
                                 "human_review_status": "pending"}
            out = DB5Writer().append(rec)
            if not out.get("ok"):
                print("    [DB5 写入失败]", out.get("errors"))

    n = len(DESIGNS)
    cls_counts: dict[str, int] = {}
    for did, domain, kind, spec, req in DESIGNS:
        q_meas = None
        for r in rows:
            if r[0] == did:
                q_meas = r[2]
        if q_meas is None:
            continue
        c = margin_class(q_meas / req["q_req"] - 1.0)
        cls_counts[c] = cls_counts.get(c, 0) + 1
    print("\n  解析（**操作口径修正后**）vs 模拟判定一致率：%d/%d（%.0f%%）"
          % (agree, n, 100 * agree / n))
    print("  余量分类（模拟侧，政策：robust_pass ≥+10%、borderline ±5%、"
          "robust_fail <−5%）：")
    for c in ("robust_pass", "borderline", "robust_fail"):
        print("    %-12s %d 个设计" % (c, cls_counts.get(c, 0)))
    if diverged:
        print("  **分歧项**：")
        for did, v_a, v_s in diverged:
            print("    %s：解析判 %s，模拟判 %s" % (did, v_a, v_s))
    else:
        print("  **无分歧**：操作口径修正后，解析自洽判定与模拟器判定一致")
    if write:
        print("  已写入 DB5（source=AI+simulated，anchor_eligible=false）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
