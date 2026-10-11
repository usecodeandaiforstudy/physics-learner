"""批量生成 DB1 条目并跑通入库管线。

生成 30 个教科书标准物理系统条目（覆盖 3 个 family：ou / diffusion / gibbs），
每个条目经完整管线（机器门 → 编码 → 指纹对照 → DB2 写入）入库。

运行：python db1/batch_generate_ingest.py

产出：
    db1/entries/*.json      条目文件
    db2/db2.jsonl           DB2 信息指纹库记录
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ingest_pipeline import ingest  # noqa: E402
from db2_writer import build_db2_record  # noqa: E402

ENTRIES_DIR = _HERE / "entries"
DB2_PATH = _PROJECT_ROOT / "db2" / "db2.jsonl"

CONVENTION_VERSION = "1.0.0"

# 每系统参数扫描点数（文档 §3.2 要求 10–50 点/系统）
SCAN_POINTS = 30


def _scan_grid(entry: dict, n_points: int = SCAN_POINTS) -> list:
    """按条目的 scan_param 生成扫描网格。

    - 该参数 range 跨越 ≥1 个数量级 → logspace（如 θ: 0.1–10）
    - 否则 → linspace
    - scan_param 为 None → 返回 [None]（仅 default 单点）
    """
    sp = entry.get("scan_param")
    if not sp:
        return [None]
    for p in entry["parameters"]:
        if p["name"] == sp:
            lo, hi = float(p["range"][0]), float(p["range"][1])
            if lo > 0 and hi / lo >= 10.0:
                return np.logspace(np.log10(lo), np.log10(hi), n_points).tolist()
            return np.linspace(lo, hi, n_points).tolist()
    return [None]


# ---------------------------------------------------------------------------
# 自证表（统一模板）
# ---------------------------------------------------------------------------
def make_attestation(bridge_constants: list[str] | None = None) -> dict:
    return {"convention_attestation": {
        "convention_version": CONVENTION_VERSION,
        "log_base": "natural",
        "log_base_unit": "nats",
        "sigma_parameterization": "sigma_as_std",
        "sde_calculus": "ito",
        "thermal_bath_kT": 1.0,
        "nondimensionalized": True,
        "bridge_constants": bridge_constants or [],
        "mi_estimator_used": "analytic",
        "ib_beta_grid": "logspace(-2, 3, 32)",
        "anchor_formulas": [],
        "uncertainties": [],
    }}


def _base_entry(system_id, domain, name, family, channel_template,
                params, fingerprint_expectation, strictness="定理",
                equations=None, source_ref="", bridge_constants=None,
                scan_param=None):
    """构造一个 DB1 条目 dict（补齐 schema 必填字段）。

    scan_param：该条目的参数扫描维度（须是 params 中某个 name）；
                None 表示不扫描（仅 default 点）。
    """
    if equations is None:
        equations = {"latex": "", "sympy": ""}
    # 桥接常数标记
    if bridge_constants:
        for p in params:
            if p["name"] in bridge_constants:
                p["is_bridge_constant"] = True
                p["unit"] = "energy"
    return {
        "system_id": system_id,
        "domain": domain,
        "name": name,
        "family": family,
        "equations": equations,
        "parameters": params,
        "channel_template": channel_template,
        "fingerprint_expectation": fingerprint_expectation,
        "strictness_level": strictness,
        "source_ref": source_ref,
        "scan_param": scan_param,
        "provenance": {
            # 诚实标注：source 为 AI+verified（程序化模板生成），
            # verified_by 只声明实际做过的验证（机器门 + 解析对照），
            # 人工复核状态单独由 db1/reviews/ 的复核记录跟踪（见 review_workflow.md）。
            "source": "AI+verified",
            "generator_model": "batch-generator-v1 (programmatic template fill)",
            "verified_by": "analytic_compare",
            "human_review_status": "pending",
            "review_date": None,
        },
        "convention_version": CONVENTION_VERSION,
    }


# ---------------------------------------------------------------------------
# 生成器：各 family
# ---------------------------------------------------------------------------
def gen_ou_entries() -> list[dict]:
    """OU 过程家族（CT-DIFF-001）：跨领域语义，12 个。

    每个变体对应一个物理实现（过阻尼振子 / RC 电路 / 布朗粒子等），
    数学形式均为 dX = -θ(X-μ)dt + σdW，指纹 fisher_trace = 6θ/σ²。
    """
    # (system_id 后缀, domain, name, source_ref, theta, sigma, mu, dt, sigma_obs)
    specs = [
        ("overdamped_osc_fast", "mechanics", "过阻尼振子（快弛豫）",
         "Risken, The Fokker-Planck Equation, Ch.5", 5.0, 0.5, 0.0, 1.0, 0.3),
        ("overdamped_osc_slow", "mechanics", "过阻尼振子（慢弛豫）",
         "Risken, Ch.5", 0.5, 1.0, 0.0, 1.0, 0.3),
        ("rc_lowpass", "electromagnetism", "RC 低通滤波（电压噪声）",
         "Gardiner 2004, Ch.3", 2.0, 1.0, 0.0, 1.0, 0.5),
        ("brownian_harmonic_strong", "stochastic_process", "布朗粒子（强简谐约束）",
         "Gardiner 2004, Ch.6", 3.0, 0.3, 0.0, 1.0, 0.4),
        ("brownian_harmonic_weak", "stochastic_process", "布朗粒子（弱简谐约束）",
         "Gardiner 2004, Ch.6", 0.3, 1.5, 0.0, 1.0, 0.4),
        ("velocity_fluctuation", "mechanics", "速度涨落弛豫",
         "Risken, Ch.5", 1.0, 2.0, 0.0, 1.0, 0.6),
        ("laser_intensity", "quantum", "激光强度涨落（OU 近似）",
         "Gardiner 2004, Ch.7", 1.5, 0.8, 0.0, 1.0, 0.5),
        ("population_dynamics", "network", "种群均值回归",
         "Turelli 1977", 0.8, 1.2, 0.0, 1.0, 0.5),
        ("thermostat_control", "thermodynamics", "恒温器温度调节",
         "Risken, Ch.5", 2.5, 0.6, 0.0, 1.0, 0.4),
        ("protein_folding_simple", "network", "蛋白质折叠（简化 OU）",
         "Zwanzig 2001", 1.2, 1.0, 0.0, 1.0, 0.5),
        ("damped_pendulum_linear", "mechanics", "线性阻尼摆（小角近似）",
         "Risken, Ch.5", 4.0, 0.4, 0.0, 1.0, 0.3),
        ("interest_rate_ou", "network", "利率均值回归（Vasicek）",
         "Vasicek 1977", 0.6, 0.9, 0.0, 1.0, 0.5),
    ]
    entries = []
    for suffix, domain, name, ref, theta, sigma, mu, dt, sobs in specs:
        params = [
            {"name": "theta", "symbol": "\\theta", "range": [0.1, 10.0], "default": theta},
            {"name": "sigma", "symbol": "\\sigma", "range": [0.1, 5.0], "default": sigma},
            {"name": "mu", "symbol": "\\mu", "range": [-5.0, 5.0], "default": mu},
            {"name": "dt", "symbol": "\\Delta t", "range": [0.01, 10.0], "default": dt},
            {"name": "sigma_obs", "symbol": "\\sigma_{obs}", "range": [0.0, 5.0], "default": sobs},
        ]
        fpe = [
            {"quantity": "fisher_trace", "analytic": "6*theta/sigma^2", "tol": 1.0e-6},
            {"quantity": "fisher_det", "analytic": "8*theta^2/sigma^4", "tol": 1.0e-6},
        ]
        eq = {"latex": "dX = -\\theta(X-\\mu)dt + \\sigma dW",
              "sympy": "Eq(diff(X,t), -theta*(X-mu)) + sigma*dW"}
        entries.append(_base_entry(
            f"ou_{suffix}", domain, name, "ou", "CT-DIFF-001",
            params, fpe, equations=eq, source_ref=ref,
            scan_param="theta",
        ))
    return entries


def gen_diffusion_entries() -> list[dict]:
    """纯扩散家族（CT-DIFF-002）：9 个。

    物理：∂u/∂t = D ∂²u/∂x²，指纹 fisher_trace = 3/(σ_s² + 2Dt)。
    """
    specs = [
        ("heat_rod_copper", "diffusion", "铜棒热传导",
         "Crank, The Mathematics of Diffusion, Ch.2", 1.0, 1.0, 1.0),
        ("heat_rod_insulator", "diffusion", "绝热材料热传导",
         "Crank, Ch.2", 0.2, 2.0, 1.0),
        ("particle_diffusion_fast", "stochastic_process", "快扩散粒子",
         "Crank, Ch.1", 2.0, 1.0, 0.8),
        ("particle_diffusion_slow", "stochastic_process", "慢扩散粒子",
         "Crank, Ch.1", 0.5, 3.0, 0.8),
        ("chemical_diffusion", "diffusion", "化学扩散",
         "Crank, Ch.2", 1.5, 1.0, 1.2),
        ("thermal_short_time", "diffusion", "热传导（短时观测）",
         "Crank, Ch.2", 1.0, 0.5, 1.0),
        ("thermal_long_time", "diffusion", "热传导（长时观测）",
         "Crank, Ch.2", 1.0, 2.5, 1.0),
        ("membrane_diffusion", "network", "膜扩散",
         "Crank, Ch.3", 0.8, 1.5, 0.9),
        ("tracer_diffusion", "stochastic_process", "示踪粒子扩散",
         "Crank, Ch.1", 1.2, 1.8, 0.7),
    ]
    entries = []
    for suffix, domain, name, ref, D, t, sigma_s in specs:
        params = [
            {"name": "D", "symbol": "D", "range": [0.1, 5.0], "default": D},
            {"name": "t", "symbol": "t", "range": [0.1, 5.0], "default": t},
            {"name": "sigma_signal", "symbol": "\\sigma_s", "range": [0.1, 5.0], "default": sigma_s},
            {"name": "mu", "symbol": "\\mu", "range": [-5.0, 5.0], "default": 0.0},
        ]
        fpe = [
            {"quantity": "fisher_trace", "analytic": "3/(sigma_signal^2 + 2*D*t)", "tol": 1.0e-6},
        ]
        eq = {"latex": "\\partial u/\\partial t = D \\partial^2 u/\\partial x^2",
              "sympy": "Eq(diff(u,t), D*diff(u,x,x))"}
        entries.append(_base_entry(
            f"diff_{suffix}", domain, name, "diffusion", "CT-DIFF-002",
            params, fpe, equations=eq, source_ref=ref,
            scan_param="D",
        ))
    return entries


def gen_gibbs_entries() -> list[dict]:
    """平衡态家族（CT-EQ-001）：9 个（谐振子 Gibbs 态）。

    物理：H = p²/(2m) + ½mω²x²，x 边缘 N(0, kT/(mω²))，
    指纹 fisher_trace = 3mω²/kT，fisher_det = 2m²ω⁴/kT²。
    """
    specs = [
        ("stiff", "thermodynamics", "刚性谐振子（低温）",
         "Pathria & Beale, Statistical Mechanics, Ch.3", 1.0, 3.0, 0.5),
        ("soft", "thermodynamics", "柔性谐振子（高温）",
         "Pathria & Beale, Ch.3", 1.0, 0.5, 2.0),
        ("heavy", "mechanics", "重振子",
         "Pathria & Beale, Ch.3", 3.0, 1.0, 1.0),
        ("light", "mechanics", "轻振子",
         "Pathria & Beale, Ch.3", 0.5, 1.0, 1.0),
        ("room_temp", "thermodynamics", "室温谐振子",
         "Pathria & Beale, Ch.3", 1.0, 1.0, 1.0),
        ("crystal_mode", "mechanics", "晶格振动模（Einstein）",
         "Pathria & Beale, Ch.3", 2.0, 2.0, 0.8),
        ("trap_ion", "quantum", "离子阱（经典极限）",
         "Pathria & Beale, Ch.3", 1.5, 1.5, 0.6),
        ("membrane", "mechanics", "薄膜振动模",
         "Pathria & Beale, Ch.3", 0.8, 0.8, 1.5),
        ("nanomech", "mechanics", "纳米机械振子",
         "Pathria & Beale, Ch.3", 2.5, 0.6, 1.2),
    ]
    entries = []
    for suffix, domain, name, ref, m, omega, kT in specs:
        params = [
            {"name": "m", "symbol": "m", "range": [0.1, 5.0], "default": m},
            {"name": "omega", "symbol": "\\omega", "range": [0.1, 5.0], "default": omega},
            {"name": "kT", "symbol": "k_B T", "range": [0.1, 5.0], "default": kT},
        ]
        fpe = [
            {"quantity": "fisher_trace", "analytic": "3*m*omega^2/kT", "tol": 1.0e-6},
            {"quantity": "fisher_det", "analytic": "2*m^2*omega^4/kT^2", "tol": 1.0e-6},
        ]
        eq = {"latex": "H = p^2/(2m) + \\tfrac12 m\\omega^2 x^2",
              "sympy": "H = p**2/(2*m) + m*omega**2*x**2/2"}
        entries.append(_base_entry(
            f"ho_{suffix}", domain, name, "harmonic_oscillator", "CT-EQ-001",
            params, fpe, equations=eq, source_ref=ref, bridge_constants=["kT"],
            scan_param="omega",
        ))
    return entries


def gen_elastic_entries() -> list[dict]:
    """线性弹性家族（CT-MECH-001）：3 个。

    物理：U = ½kx²，热平衡应变涨落 N(0, kT/k)（涨落-响应），
    指纹 fisher_trace = 3k/kT，fisher_det = 2k²/kT²。
    """
    specs = [
        ("stiff_spring", "elasticity", "硬弹簧（低热涨落）",
         "Landau & Lifshitz, Statistical Physics, Ch.12", 10.0, 1.0),
        ("soft_spring", "elasticity", "软弹簧（高热涨落）",
         "Landau & Lifshitz, Ch.12", 0.5, 1.0),
        ("polymer_chain", "elasticity", "高分子链段弹性",
         "Landau & Lifshitz, Ch.12", 1.0, 1.0),
        # ---- 模式② 扩族批次 4（2026-09-29）----
        ("cantilever", "mechanics", "微悬臂尖端刚度（AFM 针）",
         "Landau & Lifshitz, Ch.12", 8.0, 1.0),
        ("tendon", "elasticity", "肌腱弹性回复（生物弹性）",
         "Landau & Lifshitz, Ch.12", 2.0, 0.5),
        ("crystal_bond", "materials", "晶体键合刚度（低温）",
         "Landau & Lifshitz, Ch.12", 0.3, 2.0),
        ("microbeam", "mechanics", "微梁刚度（MEMS）",
         "Landau & Lifshitz, Ch.12", 5.0, 1.5),
        ("rubber_fiber", "materials", "橡胶纤维低模量",
         "Landau & Lifshitz, Ch.12", 0.8, 2.0),
    ]
    entries = []
    for suffix, domain, name, ref, k, kT in specs:
        params = [
            {"name": "k", "symbol": "k", "range": [0.1, 10.0], "default": k},
            {"name": "kT", "symbol": "k_B T", "range": [0.1, 5.0], "default": kT,
             "unit": "energy", "is_bridge_constant": True},
        ]
        fpe = [
            {"quantity": "fisher_trace", "analytic": "3*k/kT", "tol": 1.0e-6},
            {"quantity": "fisher_det", "analytic": "2*k^2/kT^2", "tol": 1.0e-6},
        ]
        eq = {"latex": "U = \\tfrac12 k x^2",
              "sympy": "U = k*x**2/2"}
        entries.append(_base_entry(
            f"elastic_{suffix}", domain, name, "elastic", "CT-MECH-001",
            params, fpe, equations=eq, source_ref=ref, bridge_constants=["kT"],
            scan_param="k",
        ))
    return entries


def gen_decay_entries() -> list[dict]:
    """放射性衰变家族（CT-DECAY-001）：3 个。

    物理：寿命分布 Exp(λ)，指纹 fisher_trace = fisher_det = 1/λ²。
    """
    specs = [
        ("fast", "quantum", "短半衰期核素（快衰变）",
         "标准核物理（放射性衰变律）", 5.0),
        ("slow", "quantum", "长半衰期核素（慢衰变）",
         "标准核物理（放射性衰变律）", 0.2),
        ("medium", "quantum", "中等半衰期核素",
         "标准核物理（放射性衰变律）", 1.0),
    ]
    entries = []
    for suffix, domain, name, ref, rate in specs:
        params = [
            {"name": "rate", "symbol": "\\lambda", "range": [0.1, 10.0], "default": rate},
        ]
        fpe = [
            {"quantity": "fisher_trace", "analytic": "1/rate^2", "tol": 1.0e-6},
            {"quantity": "fisher_det", "analytic": "1/rate^2", "tol": 1.0e-6},
        ]
        eq = {"latex": "N(t) = N_0 e^{-\\lambda t}",
              "sympy": "N = N0*exp(-rate*t)"}
        entries.append(_base_entry(
            f"decay_{suffix}", domain, name, "radioactive_decay", "CT-DECAY-001",
            params, fpe, equations=eq, source_ref=ref,
            scan_param="rate",
        ))
    return entries


def gen_count_entries() -> list[dict]:
    """泊松计数家族（CT-COUNT-001）：3 个。

    物理：计数分布 Poisson(λ)，指纹 fisher_trace = fisher_det = 1/λ。
    """
    specs = [
        ("low_count", "quantum", "弱光子流（低计数率）",
         "标准统计（泊松计数过程）", 0.5),
        ("high_count", "quantum", "强光子流（高计数率）",
         "标准统计（泊松计数过程）", 8.0),
        ("medium_count", "quantum", "中等计数率",
         "标准统计（泊松计数过程）", 2.0),
        # ---- 模式② 扩族批次 4（2026-09-29）----
        ("dark_rate", "quantum", "暗计数本底（极低计数率）",
         "标准统计（泊松计数过程）", 0.2),
        ("cosmic_ray", "signal", "宇宙线簇射计数（高峰值）",
         "标准统计（泊松计数过程）", 15.0),
        ("neutron_flux", "signal", "中子通量监测计数",
         "标准统计（泊松计数过程）", 5.0),
        ("photon_burst", "quantum", "光子爆发计数（峰值）",
         "标准统计（泊松计数过程）", 18.0),
        ("defect_line", "materials", "产线缺陷计数（抽样）",
         "标准统计（泊松计数过程）", 0.5),
    ]
    entries = []
    for suffix, domain, name, ref, rate in specs:
        params = [
            {"name": "rate", "symbol": "\\lambda", "range": [0.1, 20.0], "default": rate},
        ]
        fpe = [
            {"quantity": "fisher_trace", "analytic": "1/rate", "tol": 1.0e-6},
            {"quantity": "fisher_det", "analytic": "1/rate", "tol": 1.0e-6},
        ]
        eq = {"latex": "P(n) = \\frac{\\lambda^n e^{-\\lambda}}{n!}",
              "sympy": "P = rate**n*exp(-rate)/factorial(n)"}
        entries.append(_base_entry(
            f"count_{suffix}", domain, name, "poisson_count", "CT-COUNT-001",
            params, fpe, equations=eq, source_ref=ref,
            scan_param="rate",
        ))
    return entries


def gen_two_level_entries() -> list[dict]:
    """二能级家族（CT-TWOLVL-001）：3 个。

    物理：占据概率伯努利 Bern(p)，指纹 fisher_trace = fisher_det = 1/(p(1-p))。
    """
    specs = [
        ("ground_dominant", "quantum", "低温二能级（近基态占据）",
         "标准统计力学（二能级占据）", 0.1),
        ("degenerate", "quantum", "高温二能级（近简并）",
         "标准统计力学（二能级占据）", 0.5),
        ("excited_dominant", "quantum", "反转占据（激发态为主）",
         "标准统计力学（二能级占据）", 0.9),
        # ---- 模式② 扩族批次 4（2026-09-29）----
        ("spin_echo", "quantum", "自旋回波布居（弱极化）",
         "标准统计力学（二能级占据）", 0.3),
        ("single_qubit", "quantum", "单量子比特读取",
         "标准统计力学（二能级占据）", 0.7),
        ("bistable_switch", "quantum", "双稳态开关占空比",
         "标准统计力学（二能级占据）", 0.8),
        ("molecular_switch", "materials", "分子开关构象占据",
         "标准统计力学（二能级占据）", 0.25),
        ("dimer_binding", "materials", "二聚体结合占据",
         "标准统计力学（二能级占据）", 0.6),
    ]
    entries = []
    for suffix, domain, name, ref, prob in specs:
        params = [
            {"name": "p", "symbol": "p", "range": [0.1, 0.9], "default": prob},
        ]
        fpe = [
            {"quantity": "fisher_trace", "analytic": "1/(p*(1-p))", "tol": 1.0e-6},
            {"quantity": "fisher_det", "analytic": "1/(p*(1-p))", "tol": 1.0e-6},
        ]
        eq = {"latex": "p = \\frac{1}{1+e^{\\Delta E/kT}}",
              "sympy": "p = 1/(1+exp(delta_E/kT))"}
        entries.append(_base_entry(
            f"2lvl_{suffix}", domain, name, "two_level", "CT-TWOLVL-001",
            params, fpe, equations=eq, source_ref=ref,
            scan_param="p",
        ))
    return entries


def gen_laplace_entries() -> list[dict]:
    """拉普拉斯家族（CT-LAPLACE-001）：3 个。

    物理：双独立同速率指数过程到达时间差 Z=X1-X2 ~ Laplace(0, 1/λ)，
    指纹 fisher_trace = 2/b² = 2λ²，fisher_det = 1/b⁴ = λ⁴。
    """
    specs = [
        ("fast_arrival_diff", "stochastic_process", "快到达（窄分布）",
         "标准概率论（双指数差 = 拉普拉斯分布）", 5.0),
        ("slow_arrival_diff", "stochastic_process", "慢到达（宽分布）",
         "标准概率论（双指数差 = 拉普拉斯分布）", 0.5),
        ("medium_arrival_diff", "stochastic_process", "中等到达率",
         "标准概率论（双指数差 = 拉普拉斯分布）", 1.0),
    ]
    entries = []
    for suffix, domain, name, ref, rate in specs:
        params = [
            {"name": "rate", "symbol": "\\lambda", "range": [0.1, 10.0], "default": rate},
        ]
        fpe = [
            {"quantity": "fisher_trace", "analytic": "2*rate^2", "tol": 1.0e-6},
            {"quantity": "fisher_det", "analytic": "rate^4", "tol": 1.0e-6},
        ]
        eq = {"latex": "Z = X_1 - X_2,\\quad X_i \\sim \\mathrm{Exp}(\\lambda)",
              "sympy": "Z = X1 - X2"}
        entries.append(_base_entry(
            f"laplace_{suffix}", domain, name, "laplace", "CT-LAPLACE-001",
            params, fpe, equations=eq, source_ref=ref,
            scan_param="rate",
        ))
    return entries


def gen_resonator_entries() -> list[dict]:
    """谐振腔 LC 家族（CT-LC-001）：3 个。

    物理：电容电压热涨落 V ~ N(0, kT/C)（Johnson 噪声），
    指纹 fisher_trace = 3C/kT，fisher_det = 2C²/kT²。
    """
    specs = [
        ("large_cap", "electromagnetism", "大电容（低电压涨落）",
         "标准电路热噪声（能量均分）", 10.0, 1.0),
        ("small_cap", "electromagnetism", "小电容（高电压涨落）",
         "标准电路热噪声（能量均分）", 0.5, 1.0),
        ("room_temp", "electromagnetism", "室温 LC 谐振腔",
         "标准电路热噪声（能量均分）", 1.0, 1.0),
    ]
    entries = []
    for suffix, domain, name, ref, C, kT in specs:
        params = [
            {"name": "capacitance", "symbol": "C", "range": [0.1, 10.0], "default": C},
            {"name": "kT", "symbol": "k_B T", "range": [0.1, 5.0], "default": kT,
             "unit": "energy", "is_bridge_constant": True},
        ]
        fpe = [
            {"quantity": "fisher_trace", "analytic": "3*capacitance/kT", "tol": 1.0e-6},
            {"quantity": "fisher_det", "analytic": "2*capacitance^2/kT^2", "tol": 1.0e-6},
        ]
        eq = {"latex": "\\langle \\tfrac12 C V^2 \\rangle = \\tfrac12 kT",
              "sympy": "0.5*C*V**2 = 0.5*kT"}
        entries.append(_base_entry(
            f"lc_{suffix}", domain, name, "lc_circuit", "CT-LC-001",
            params, fpe, equations=eq, source_ref=ref, bridge_constants=["kT"],
            scan_param="capacitance",
        ))
    return entries


def gen_landau_entries() -> list[dict]:
    """朗道自由能家族（CT-LANDAU-001）：3 个。

    物理：序参量分布 p(φ) ∝ exp(-(aφ²/2 + bφ⁴/4)/kT)，Fisher 关于 a 需数值积分
    （无解析式），故 fingerprint_expectation 留空（机器门 3 仅警告，不阻断）。
    """
    specs = [
        ("high_temp", "thermodynamics", "高温相（单峰序参量）",
         "Landau 相变理论（序参量涨落）", 2.0, 1.0, 1.0),
        ("near_critical", "thermodynamics", "近临界（a≈0）",
         "Landau 相变理论（序参量涨落）", 0.1, 1.0, 1.0),
        ("low_temp", "thermodynamics", "低温相（双峰序参量，对称破缺）",
         "Landau 相变理论（序参量涨落）", -1.0, 1.0, 1.0),
        # ---- 模式② 扩族批次 4（2026-09-29）：b/kT 多样性（诊断 F1/R1 前置）----
        ("b05_kt1", "materials", "弱非线性低温相（b=0.5）",
         "Landau 相变理论（序参量涨落）", -1.0, 0.5, 1.0),
        ("b2_kt1", "thermodynamics", "强非线性近临界（b=2）",
         "Landau 相变理论（序参量涨落）", -0.5, 2.0, 1.0),
        ("b1_kt05", "thermodynamics", "低温浴高温相（kT=0.5）",
         "Landau 相变理论（序参量涨落）", 1.0, 1.0, 0.5),
        ("b1_kt2", "materials", "高温浴低温相（kT=2）",
         "Landau 相变理论（序参量涨落）", -1.5, 1.0, 2.0),
        ("b2_kt2", "thermodynamics", "强非线性高温浴（b=kT=2）",
         "Landau 相变理论（序参量涨落）", 0.5, 2.0, 2.0),
        ("b05_kt2", "materials", "弱非线性高温浴（b=0.5, kT=2）",
         "Landau 相变理论（序参量涨落）", -2.0, 0.5, 2.0),
    ]
    entries = []
    for suffix, domain, name, ref, a, b, kT in specs:
        params = [
            {"name": "a", "symbol": "a", "range": [-2.0, 2.0], "default": a},
            {"name": "b", "symbol": "b", "range": [0.1, 5.0], "default": b},
            {"name": "kT", "symbol": "k_B T", "range": [0.1, 5.0], "default": kT,
             "unit": "energy", "is_bridge_constant": True},
        ]
        fpe: list[dict] = []
        eq = {"latex": "F = \\tfrac12 a \\phi^2 + \\tfrac14 b \\phi^4",
              "sympy": "F = a*phi**2/2 + b*phi**4/4"}
        entries.append(_base_entry(
            f"landau_{suffix}", domain, name, "landau", "CT-LANDAU-001",
            params, fpe, equations=eq, source_ref=ref, bridge_constants=["kT"],
            scan_param="a",
        ))
    return entries


# ---------------------------------------------------------------------------
# 扩族批次 1（2026-09-26）：gamma / beta / weibull / vonmises
# ---------------------------------------------------------------------------
def gen_gamma_entries() -> list[dict]:
    """伽马家族（CT-GAMMA-001）：3 个（非对角 Fisher 的首个族）。

    物理：等待时间/能量分布 Gamma(k, θ)，
    指纹 fisher_trace = ψ₁(k) + k/θ²，fisher_det = (k·ψ₁(k) − 1)/θ²。
    （polygamma(1, ·) 即 trigamma ψ₁。）
    """
    specs = [
        ("erlang_queue", "network", "泊松流第 k 事件等待（Erlang 排队）",
         "标准排队论（Erlang 分布）", 3.0, 0.5),
        ("chi2_energy", "signal", "噪声平方能量（chi2 噪声）",
         "标准信号处理（chi2 = Gamma(nu/2, 2)）", 2.0, 2.0),
        ("rainfall", "diffusion", "站点时段降雨量（水文）",
         "标准水文统计（伽马雨量拟合）", 0.9, 2.0),
    ]
    entries = []
    for suffix, domain, name, ref, k, theta in specs:
        params = [
            {"name": "k", "symbol": "k", "range": [0.5, 10.0], "default": k},
            {"name": "theta", "symbol": "\\theta", "range": [0.1, 5.0], "default": theta},
        ]
        fpe = [
            {"quantity": "fisher_trace",
             "analytic": "polygamma(1, k) + k/theta^2", "tol": 1.0e-6},
            {"quantity": "fisher_det",
             "analytic": "(k*polygamma(1, k) - 1)/theta^2", "tol": 1.0e-6},
        ]
        eq = {"latex": "p(x) = x^{k-1} e^{-x/\\theta} / (\\Gamma(k)\\theta^k)",
              "sympy": "p = x**(k-1)*exp(-x/theta)/(gamma(k)*theta**k)"}
        entries.append(_base_entry(
            f"gamma_{suffix}", domain, name, "gamma", "CT-GAMMA-001",
            params, fpe, equations=eq, source_ref=ref,
            scan_param="k",
        ))
    return entries


def gen_beta_entries() -> list[dict]:
    """贝塔家族（CT-BETA-001）：3 个（首个有界支撑 [0,1] 族）。

    物理：占据分数/纯度/混合权重 Beta(α, β)，
    指纹 fisher_trace = ψ₁(α)+ψ₁(β)−2ψ₁(α+β)，
    fisher_det = ψ₁(α)ψ₁(β) − ψ₁(α+β)[ψ₁(α)+ψ₁(β)]。
    """
    specs = [
        ("occupancy_fraction", "materials", "固溶体 B 位占据分数涨落",
         "标准合金统计（占据涨落）", 2.0, 3.0),
        ("purity_fluctuation", "quantum", "混态谱权重涨落（有界观测量）",
         "标准量子统计（混态谱）", 5.0, 1.5),
        ("mixture_weight", "thermodynamics", "两相混合体积分数（杠杆区涨落）",
         "标准相变统计（两相涨落）", 1.0, 1.0),
        # ---- 模式② 扩族批次 4（2026-09-29）：形态多样性 ----
        ("j_shaped", "materials", "J 形分布（α<1<β）",
         "标准贝塔分布统计", 0.7, 4.0),
        ("u_shaped", "materials", "U 形分布（两端聚集）",
         "标准贝塔分布统计", 0.6, 0.6),
        ("unimodal_sym", "thermodynamics", "对称单峰（α=β）",
         "标准贝塔分布统计", 3.0, 3.0),
        ("skewed", "materials", "右偏单峰（α<β）",
         "标准贝塔分布统计", 2.0, 5.0),
    ]
    entries = []
    for suffix, domain, name, ref, alpha, beta in specs:
        params = [
            {"name": "alpha", "symbol": "\\alpha", "range": [0.5, 5.0], "default": alpha},
            {"name": "beta", "symbol": "\\beta", "range": [0.5, 5.0], "default": beta},
        ]
        fpe = [
            {"quantity": "fisher_trace",
             "analytic": "polygamma(1, alpha) + polygamma(1, beta) - 2*polygamma(1, alpha + beta)",
             "tol": 1.0e-6},
            {"quantity": "fisher_det",
             "analytic": "polygamma(1, alpha)*polygamma(1, beta) - polygamma(1, alpha + beta)*(polygamma(1, alpha) + polygamma(1, beta))",
             "tol": 1.0e-6},
        ]
        eq = {"latex": "p(x) = x^{\\alpha-1}(1-x)^{\\beta-1}/B(\\alpha,\\beta)",
              "sympy": "p = x**(alpha-1)*(1-x)**(beta-1)/beta_func(alpha, beta)"}
        entries.append(_base_entry(
            f"beta_{suffix}", domain, name, "beta", "CT-BETA-001",
            params, fpe, equations=eq, source_ref=ref,
            scan_param="alpha",
        ))
    return entries


def gen_weibull_entries() -> list[dict]:
    """威布尔家族（CT-WEIBULL-001）：3 个（失效统计）。

    物理：最弱链接失效寿命 Weibull(shape, scale)，
    指纹 fisher_trace = k²/λ² + [(1−γ_E)²+π²/6]/k²，fisher_det = π²/(6λ²)。
    """
    specs = [
        ("fracture_life", "materials", "脆性断裂寿命（最弱链接）",
         "Weibull 1951（脆断统计）", 1.5, 2.0),
        ("fatigue_life", "mechanics", "疲劳裂纹萌生寿命",
         "Weibull 1951（疲劳统计）", 2.5, 1.0),
        ("early_failure", "materials", "早期失效（婴幼期失效型）",
         "Weibull 1951（k<1 递减失效率）", 0.8, 3.0),
    ]
    entries = []
    for suffix, domain, name, ref, shape, scale in specs:
        params = [
            {"name": "shape", "symbol": "k", "range": [0.5, 5.0], "default": shape},
            {"name": "scale", "symbol": "\\lambda", "range": [0.1, 5.0], "default": scale},
        ]
        fpe = [
            {"quantity": "fisher_trace",
             "analytic": "shape^2/scale^2 + ((1 - EulerGamma)^2 + pi^2/6)/shape^2",
             "tol": 1.0e-6},
            {"quantity": "fisher_det",
             "analytic": "pi^2/(6*scale^2)", "tol": 1.0e-6},
        ]
        eq = {"latex": "p(x) = \\frac{k}{\\lambda}\\left(\\frac{x}{\\lambda}\\right)^{k-1} e^{-(x/\\lambda)^k}",
              "sympy": "p = (k/lam)*(x/lam)**(k-1)*exp(-(x/lam)**k)"}
        entries.append(_base_entry(
            f"weibull_{suffix}", domain, name, "weibull", "CT-WEIBULL-001",
            params, fpe, equations=eq, source_ref=ref,
            scan_param="shape",
        ))
    return entries


def gen_vonmises_entries() -> list[dict]:
    """冯·米塞斯家族（CT-VONMISES-001）：3 个（首个圆支撑 S¹ 族）。

    物理：角度/相位分布 VM(μ, κ)，I_μμ = κ·I₁(κ)/I₀(κ)（解析），
    I_κκ = ½(1+I₂/I₀)−(I₁/I₀)²（Bessel 闭式，由 E[cos nθ]=Iₙ/I₀ 展开）。
    fingerprint_expectation 留空（同朗道模式），对照由
    FingerprintComputer 的 vonmises 特判承担（sympy/scipy 双库对照，
    I_μμ 与 I_κκ 各一条；MC score 对照见 test_new_families.py）。
    """
    specs = [
        ("pll_phase", "electromagnetism", "锁相环相位误差（锁相深度）",
         "Gardner 2005, Phaselock Techniques", 4.0),
        ("molecular_orientation", "materials", "液晶分子取向涨落",
         "de Gennes & Prost, Physics of Liquid Crystals", 1.5),
        ("rotor_angle", "mechanics", "转子角误差分布",
         "Mardia & Jupp, Directional Statistics, Ch.3", 8.0),
        # ---- 模式② 扩族批次 4（2026-09-29）：深锁区（治 v0 盲区）----
        ("deep_lock_pll", "electromagnetism", "深锁相环（κ=20）",
         "Gardner 2005, Phaselock Techniques", 20.0),
        ("nematic_strong", "materials", "向列相强取向（κ=12）",
         "de Gennes & Prost, Physics of Liquid Crystals", 12.0),
        ("antenna_phase", "electromagnetism", "相控阵相位误差（κ=6）",
         "Mardia & Jupp, Directional Statistics, Ch.3", 6.0),
        ("molecular_rotor", "mechanics", "分子转子角锁定（κ=15）",
         "Mardia & Jupp, Directional Statistics, Ch.3", 15.0),
    ]
    entries = []
    for suffix, domain, name, ref, kappa in specs:
        params = [
            {"name": "mu", "symbol": "\\mu", "range": [-3.14, 3.14], "default": 0.0},
            {"name": "kappa", "symbol": "\\kappa", "range": [0.1, 10.0], "default": kappa},
        ]
        fpe: list[dict] = []
        eq = {"latex": "p(\\theta) = e^{\\kappa\\cos(\\theta-\\mu)}/(2\\pi I_0(\\kappa))",
              "sympy": "p = exp(kappa*cos(theta-mu))/(2*pi*besseli(0, kappa))"}
        entries.append(_base_entry(
            f"vonmises_{suffix}", domain, name, "vonmises", "CT-VONMISES-001",
            params, fpe, equations=eq, source_ref=ref,
            scan_param="kappa",
        ))
    return entries


# ---------------------------------------------------------------------------
# 扩族批次 2（2026-09-26）：geometric / cauchy / rayleigh / logistic
# ---------------------------------------------------------------------------
def gen_geometric_entries() -> list[dict]:
    """几何分布族（CT-GEOM-001）：3 个（离散非对称非幂律 Fisher）。

    物理：首次成功等待 Geo(p)，Fisher = 1/(p²(1-p))。
    """
    specs = [
        ("first_success", "quantum", "首次成功等待（重复试验）",
         "标准概率论（几何分布）", 0.1),
        ("photon_detection", "quantum", "光子探测效率（首光子等待）",
         "标准概率论（几何分布）", 0.05),
        ("defect_search", "materials", "缺陷搜索（首缺陷检查数）",
         "标准概率论（几何分布）", 0.2),
        # ---- 模式② 扩族批次 4（2026-09-29）：p 区间多样性 ----
        ("long_tail_lowp", "quantum", "低概率长尾（p=0.06）",
         "标准概率论（几何分布）", 0.06),
        ("mid_p_025", "signal", "中概率（p=0.25）",
         "标准概率论（几何分布）", 0.25),
        ("mid_p_04", "materials", "中概率（p=0.4）",
         "标准概率论（几何分布）", 0.4),
        ("rare_defect", "materials", "稀有缺陷搜索（p=0.08）",
         "标准概率论（几何分布）", 0.08),
        # ---- R4b（2026-09-29）：右腿 + 跨膝盖系统（补 U 形上升段，治 negbin）----
        ("right_leg_p07", "signal", "跨膝盖右腿（p=0.7，过最小值 p=2/3）",
         "标准概率论（几何分布）", 0.7, [0.5, 0.95]),
        ("right_leg_p085", "quantum", "右腿高概率（p=0.85）",
         "标准概率论（几何分布）", 0.85, [0.5, 0.95]),
        ("right_leg_p095", "signal", "右腿极高概率（p=0.95）",
         "标准概率论（几何分布）", 0.95, [0.5, 0.95]),
    ]
    entries = []
    for spec in specs:
        suffix, domain, name, ref, prob = spec[:5]
        rng = spec[5] if len(spec) > 5 else [0.05, 0.5]
        params = [
            {"name": "p", "symbol": "p", "range": rng, "default": prob},
        ]
        fpe = [
            {"quantity": "fisher_trace", "analytic": "1/(p**2*(1-p))", "tol": 1.0e-6},
            {"quantity": "fisher_det", "analytic": "1/(p**2*(1-p))", "tol": 1.0e-6},
        ]
        eq = {"latex": "P(k) = p(1-p)^{k-1}",
              "sympy": "P = p*(1-p)**(k-1)"}
        entries.append(_base_entry(
            f"geometric_{suffix}", domain, name, "geometric", "CT-GEOM-001",
            params, fpe, equations=eq, source_ref=ref,
            scan_param="p",
        ))
    return entries


def gen_cauchy_entries() -> list[dict]:
    """柯西分布族（CT-CAUCHY-001）：3 个（重尾，有限 Fisher 无限方差）。

    物理：洛伦兹线形 Cauchy(0, γ)，Fisher = 1/(2γ²)。
    """
    specs = [
        ("resonance_line", "electromagnetism", "共振线形（洛伦兹）",
         "标准统计（柯西/洛伦兹分布）", 1.0),
        ("spectral_width", "quantum", "谱线宽度（半高半宽）",
         "标准统计（柯西/洛伦兹分布）", 0.5),
        ("noise", "signal", "柯西噪声（重尾脉冲）",
         "标准统计（柯西/洛伦兹分布）", 2.0),
    ]
    entries = []
    for suffix, domain, name, ref, gamma in specs:
        params = [
            {"name": "gamma", "symbol": "\\gamma", "range": [0.1, 5.0], "default": gamma},
        ]
        fpe = [
            {"quantity": "fisher_trace", "analytic": "1/(2*gamma**2)", "tol": 1.0e-6},
            {"quantity": "fisher_det", "analytic": "1/(2*gamma**2)", "tol": 1.0e-6},
        ]
        eq = {"latex": "p(x) = \\frac{1}{\\pi\\gamma[1+(x/\\gamma)^2]}",
              "sympy": "p = 1/(pi*gamma*(1+(x/gamma)**2))"}
        entries.append(_base_entry(
            f"cauchy_{suffix}", domain, name, "cauchy", "CT-CAUCHY-001",
            params, fpe, equations=eq, source_ref=ref,
            scan_param="gamma",
        ))
    return entries


def gen_rayleigh_entries() -> list[dict]:
    """瑞利分布族（CT-RAYLEIGH-001）：3 个（二维高斯幅值，Fisher=4/σ²）。

    物理：幅值/包络分布 Rayleigh(σ)。
    """
    specs = [
        ("wind_speed", "mechanics", "风速（二维矢量幅值）",
         "标准统计（瑞利分布）", 1.0),
        ("scattering", "electromagnetism", "散射强度（包络幅值）",
         "标准统计（瑞利分布）", 0.8),
        ("turbulence", "mechanics", "湍流脉动（速度幅值）",
         "标准统计（瑞利分布）", 1.5),
    ]
    entries = []
    for suffix, domain, name, ref, sigma in specs:
        params = [
            {"name": "sigma", "symbol": "\\sigma", "range": [0.1, 5.0], "default": sigma},
        ]
        fpe = [
            {"quantity": "fisher_trace", "analytic": "4/sigma**2", "tol": 1.0e-6},
            {"quantity": "fisher_det", "analytic": "4/sigma**2", "tol": 1.0e-6},
        ]
        eq = {"latex": "p(x) = \\frac{x}{\\sigma^2} e^{-x^2/(2\\sigma^2)}",
              "sympy": "p = (x/sigma**2)*exp(-x**2/(2*sigma**2))"}
        entries.append(_base_entry(
            f"rayleigh_{suffix}", domain, name, "rayleigh", "CT-RAYLEIGH-001",
            params, fpe, equations=eq, source_ref=ref,
            scan_param="sigma",
        ))
    return entries


def gen_logistic_entries() -> list[dict]:
    """逻辑斯蒂分布族（CT-LOGISTIC-001）：3 个（S 形，Fisher=(π²+3)/(9s²)）。

    物理：S 形生长/阈值分布 Logistic(0, s)。
    """
    specs = [
        ("growth", "thermodynamics", "S 形生长（速率阈值）",
         "标准统计（逻辑斯蒂分布）", 1.0),
        ("activation", "network", "激活阈值（神经元）",
         "标准统计（逻辑斯蒂分布）", 0.5),
        ("phase_boundary", "thermodynamics", "相界序参量（软模涨落）",
         "标准统计（逻辑斯蒂分布）", 2.0),
    ]
    entries = []
    for suffix, domain, name, ref, scale in specs:
        params = [
            {"name": "scale", "symbol": "s", "range": [0.1, 5.0], "default": scale},
        ]
        fpe = [
            {"quantity": "fisher_trace",
             "analytic": "(pi**2 + 3)/(9*scale**2)", "tol": 1.0e-6},
            {"quantity": "fisher_det",
             "analytic": "(pi**2 + 3)/(9*scale**2)", "tol": 1.0e-6},
        ]
        eq = {"latex": "p(x) = \\frac{e^{-x/s}}{s(1+e^{-x/s})^2}",
              "sympy": "p = exp(-x/scale)/(scale*(1+exp(-x/scale))**2)"}
        entries.append(_base_entry(
            f"logistic_{suffix}", domain, name, "logistic", "CT-LOGISTIC-001",
            params, fpe, equations=eq, source_ref=ref,
            scan_param="scale",
        ))
    return entries


# ---------------------------------------------------------------------------
# 扩族批次 3（2026-09-27）：卡方（ψ₁ 族）/ 负二项（计数族）/ 对数正态（乘性噪声）
# 动机：A 网络族级 OOD 逐族分解显示 gamma（ψ₁(k)+k/θ²）为主要失败源、
# poisson（1/λ）为次失败源——训练库无纯 ψ₁ 曲线族与超散计数族可借。
# ---------------------------------------------------------------------------
def gen_chisq_entries() -> list[dict]:
    """卡方族（CT-CHISQ-001）：3 个（纯 ψ₁ trigamma Fisher，ψ₁ 覆盖缺口）。

    物理：标准二次型 χ²(k)（χ²(k)=Gamma(k/2,2)），
    指纹 fisher_trace = fisher_det = ψ₁(k/2)/4 = polygamma(1, k/2)/4。
    """
    specs = [
        ("whitened_residual", "signal", "白化残差能量（k 维最小二乘）",
         "标准统计（卡方分布）", 4.0),
        ("mahalanobis_distance", "materials", "马氏距离平方（k 维特征离群检测）",
         "标准统计（卡方分布）", 8.0),
        ("goodness_of_fit", "quantum", "拟合优度统计量（k bin 谱拟合检验）",
         "标准统计（皮尔逊卡方）", 2.0),
        # ---- 模式② 扩族批次 4（2026-09-29）：自由度多样性 ----
        ("k1_gauss_sq", "signal", "k=1 单自由度（高斯平方）",
         "标准统计（卡方分布）", 1.0),
        ("k3_whitened", "materials", "k=3 三维白化残差",
         "标准统计（卡方分布）", 3.0),
        ("k12_spectral", "signal", "k=12 月度谱估计",
         "标准统计（卡方分布）", 12.0),
        ("k20_gof", "quantum", "k=20 大自由度拟合检验",
         "标准统计（皮尔逊卡方）", 20.0),
    ]
    entries = []
    for suffix, domain, name, ref, k in specs:
        params = [
            {"name": "k", "symbol": "k", "range": [1.0, 20.0], "default": k},
        ]
        fpe = [
            {"quantity": "fisher_trace",
             "analytic": "polygamma(1, k/2)/4", "tol": 1.0e-6},
            {"quantity": "fisher_det",
             "analytic": "polygamma(1, k/2)/4", "tol": 1.0e-6},
        ]
        eq = {"latex": "p(x) = x^{k/2-1} e^{-x/2} / (2^{k/2} \\Gamma(k/2))",
              "sympy": "p = x**(k/2-1)*exp(-x/2)/(2**(k/2)*gamma(k/2))"}
        entries.append(_base_entry(
            f"chisq_{suffix}", domain, name, "chisq", "CT-CHISQ-001",
            params, fpe, equations=eq, source_ref=ref,
            scan_param="k",
        ))
    return entries


def gen_negbin_entries() -> list[dict]:
    """负二项族（CT-NEGBIN-001）：3 个（超散计数，Fisher(关于 p)=k/(p²(1-p))）。

    物理：泊松-伽马混合计数（第 k 次成功等待参数化，k=1 严格退化几何）。
    k 为结构常数（伽马混合形状），Fisher 摘要为关于 p 的条件信息。
    """
    specs = [
        ("thermal_photon", "quantum", "热光光子计数（k 模散斑）",
         "标准量子光学（散斑热光计数）", 5.0, 0.1),
        ("scintillation", "signal", "闪烁计数（伽马强度调制）",
         "标准信号处理（闪烁统计）", 3.0, 0.2),
        ("dark_count_cluster", "quantum", "暗计数成簇（雪崩倍增）",
         "标准探测统计（暗计数簇）", 2.0, 0.05),
        # ---- 模式② 扩族批次 4（2026-09-29）：结构常数多样性 ----
        ("k1_geometric_limit", "quantum", "k=1 几何退化对照",
         "标准统计（负二项分布）", 1.0, 0.3),
        ("k3_moderate", "signal", "k=3 中度超散",
         "标准统计（负二项分布）", 3.0, 0.15),
        ("k10_heavy", "quantum", "k=10 强聚集",
         "标准统计（负二项分布）", 10.0, 0.4),
        ("k8_dark", "quantum", "k=8 暗计数簇（低 p）",
         "标准统计（负二项分布）", 8.0, 0.05),
    ]
    entries = []
    for suffix, domain, name, ref, k, prob in specs:
        params = [
            {"name": "k", "symbol": "k", "range": [1.0, 10.0], "default": k},
            {"name": "p", "symbol": "p", "range": [0.05, 0.5], "default": prob},
        ]
        fpe = [
            {"quantity": "fisher_trace",
             "analytic": "k/(p**2*(1-p))", "tol": 1.0e-6},
            {"quantity": "fisher_det",
             "analytic": "k/(p**2*(1-p))", "tol": 1.0e-6},
        ]
        eq = {"latex": "P(n) = \\binom{n-1}{k-1} p^k (1-p)^{n-k}",
              "sympy": "P = binomial(n-1, k-1)*p**k*(1-p)**(n-k)"}
        entries.append(_base_entry(
            f"negbin_{suffix}", domain, name, "negbin", "CT-NEGBIN-001",
            params, fpe, equations=eq, source_ref=ref,
            scan_param="p",
        ))
    return entries


def gen_lognormal_entries() -> list[dict]:
    """对数正态族（CT-LOGNORM-001）：3 个（乘性噪声，Fisher 与高斯同形）。

    物理：X = exp(Y)，Y ~ N(mu, sigma)（几何均值回复/凝并粒径/阴影衰落），
    指纹 fisher_trace = 3/σ²，fisher_det = 2/σ⁴（双射不变性，与高斯同形）。
    本族是本批对照组：加族不加新函数形状，检验纯属性新颖性对族级 OOD 的作用。
    """
    specs = [
        ("growth_stationary", "diffusion", "对数均值回复增长（乘性噪声平稳态）",
         "标准随机过程（几何均值回复）", 0.0, 1.0),
        ("particle_size", "materials", "凝并破碎粒径（定态近似）",
         "标准气溶胶统计（对数正态粒径）", 0.5, 0.8),
        ("shadow_fading", "signal", "阴影衰落（大尺度路径损耗波动）",
         "标准信道模型（对数正态阴影）", 0.0, 0.5),
    ]
    entries = []
    for suffix, domain, name, ref, mu, sigma in specs:
        params = [
            {"name": "mu", "symbol": "\\mu", "range": [-2.0, 2.0], "default": mu},
            {"name": "sigma", "symbol": "\\sigma", "range": [0.1, 5.0], "default": sigma},
        ]
        fpe = [
            {"quantity": "fisher_trace",
             "analytic": "3/sigma**2", "tol": 1.0e-6},
            {"quantity": "fisher_det",
             "analytic": "2/sigma**4", "tol": 1.0e-6},
        ]
        eq = {"latex":
              "p(x) = \\frac{e^{-(\\ln x - \\mu)^2/(2\\sigma^2)}}{x\\sigma\\sqrt{2\\pi}}",
              "sympy":
              "p = exp(-(log(x)-mu)**2/(2*sigma**2))/(x*sigma*sqrt(2*pi))"}
        entries.append(_base_entry(
            f"lognormal_{suffix}", domain, name, "lognormal", "CT-LOGNORM-001",
            params, fpe, equations=eq, source_ref=ref,
            scan_param="sigma",
        ))
    return entries


# ---------------------------------------------------------------------------
# 扩族批次 4（2026-09-29，R4 平膝形状族）
# ---------------------------------------------------------------------------
def gen_mix2_entries() -> list[dict]:
    """双高斯混合族（CT-MIX2-001）：3 个（库中首个非单调 Fisher 曲线）。

    物理：对称双峰测量 p(x) = ½N(x|−d/2,σ) + ½N(x|+d/2,σ)。
    Fisher 关于分离度 d 数值积分：0 → 峰 → 1/(4σ²) 平台（凸起曲线）。
    fingerprint_expectation 留空（同朗道模式，门 3 仅警告）；
    数值 Fisher 与 MC 对照见 test_new_encoders_3.py。
    """
    specs = [
        ("bimodal_sensor", "signal", "双模态传感器输出（对称双峰）",
         "标准混合模型（信息几何）", 1.0),
        ("bistable_measure", "thermodynamics", "双稳势阱观测（对称双峰）",
         "标准混合模型（信息几何）", 1.0),
        ("heterodyne_beat", "electromagnetism", "外差拍频信号（双频分量）",
         "标准混合模型（信息几何）", 1.0),
    ]
    entries = []
    for suffix, domain, name, ref, sigma in specs:
        params = [
            {"name": "d", "symbol": "d", "range": [0.05, 6.0], "default": 2.0},
            {"name": "w", "symbol": "w", "range": [0.1, 0.9], "default": 0.5},
            {"name": "sigma", "symbol": "\\sigma", "range": [0.5, 2.0],
             "default": sigma},
        ]
        fpe: list[dict] = []
        eq = {"latex": "p(x) = \\tfrac12 N(x|{-}d/2,\\sigma) "
                       "+ \\tfrac12 N(x|{+}d/2,\\sigma)",
              "sympy": "p = N(x, -d/2, sigma)/2 + N(x, +d/2, sigma)/2"}
        entries.append(_base_entry(
            f"mix2_{suffix}", domain, name, "gaussian_mixture2", "CT-MIX-001",
            params, fpe, equations=eq, source_ref=ref, scan_param="d",
        ))
    return entries


def gen_truncnorm_entries() -> list[dict]:
    """截断高斯族（CT-TRUNCNORM-001）：3 个（「平→陡」上升膝，解析 Fisher）。

    物理：单边右截断观测 p(x) = N(x|μ,σ)/(1−Φ(z))，x > τ，z=(τ−μ)/σ。
    Fisher 关于截断点 τ：I(τ) = φ(z)²/(σ²(1−Φ(z))²)——
    z→−∞ 为 0（平台膝），z 近均值陡升（与 landau 的下降膝互补）。
    """
    specs = [
        ("threshold_sensor", "signal", "阈值化传感器输出（低于阈值不记录）",
         "标准统计（截断高斯分布）", -1.0),
        ("censored_lifetime", "materials", "右截断寿命观测（截尾）",
         "标准统计（截断高斯分布）", -0.5),
        ("gate_trigger", "electromagnetism", "门限触发幅度分布",
         "标准统计（截断高斯分布）", -2.0),
    ]
    entries = []
    for suffix, domain, name, ref, tau0 in specs:
        params = [
            {"name": "tau", "symbol": "\\tau", "range": [-3.0, 1.0],
             "default": tau0},
            {"name": "mu", "symbol": "\\mu", "range": [-2.0, 2.0],
             "default": 0.0},
            {"name": "sigma", "symbol": "\\sigma", "range": [0.5, 2.0],
             "default": 1.0},
        ]
        fpe = [
            {"quantity": "fisher_trace",
             "analytic": "2*exp(-((tau-mu)/sigma)**2)/"
                         "(pi*sigma**2*erfc((tau-mu)/(sigma*sqrt(2)))**2)",
             "tol": 1.0e-6},
            {"quantity": "fisher_det",
             "analytic": "2*exp(-((tau-mu)/sigma)**2)/"
                         "(pi*sigma**2*erfc((tau-mu)/(sigma*sqrt(2)))**2)",
             "tol": 1.0e-6},
        ]
        eq = {"latex": "p(x) = \\frac{N(x|\\mu,\\sigma)}"
                       "{1-\\Phi((\\tau-\\mu)/\\sigma)},\\; x>\\tau",
              "sympy": "p = N(x, mu, sigma)/(1 - Phi((tau-mu)/sigma))"}
        entries.append(_base_entry(
            f"truncnorm_{suffix}", domain, name, "truncated_gaussian",
            "CT-TRUNCNORM-001", params, fpe, equations=eq, source_ref=ref,
            scan_param="tau",
        ))
    return entries


def gen_truncgeom_entries() -> list[dict]:
    """截断几何族（CT-TRUNCGEOM-001）：3 个（R4b：negbin 输入模式同构训练族）。

    物理：有限支撑几何 p(n) = p·q^(n−1)/(1−q^k)，n=1..k。
    Fisher 关于 p 为 k 项有限和（fisher_truncgeom_p）；k→∞ 退化 Geo(p)。
    k 取 5/10/20 三个不同支撑上界——给模型 k 列的**非 ψ₁ 语义变化**
    （negbin 的 k 列在训练集无同构样本是其墙的根因假设）。
    """
    specs = [
        ("buffer5", "network", "有限缓冲排队（容量 5 的首事件等待）",
         "标准概率论（截断几何分布）", 5.0, 0.3),
        ("window10", "signal", "窗口化缺陷计数（窗口 10 的首缺陷位置）",
         "标准概率论（截断几何分布）", 10.0, 0.1),
        ("censor20", "materials", "截尾寿命级数（支撑 20 的离散寿命）",
         "标准概率论（截断几何分布）", 20.0, 0.45),
    ]
    entries = []
    for suffix, domain, name, ref, k, prob in specs:
        params = [
            {"name": "k", "symbol": "k", "range": [1.0, 50.0], "default": k},
            {"name": "p", "symbol": "p", "range": [0.05, 0.5], "default": prob},
        ]
        fpe: list[dict] = []
        eq = {"latex": "P(n) = \\frac{p(1-p)^{n-1}}{1-(1-p)^k},\\; n=1..k",
              "sympy": "P = p*(1-p)**(n-1)/(1-(1-p)**k)"}
        entries.append(_base_entry(
            f"truncgeom_{suffix}", domain, name, "truncated_geometric",
            "CT-TRUNCGEOM-001", params, fpe, equations=eq, source_ref=ref,
            scan_param="p",
        ))
    return entries


def gen_invgamma_entries() -> list[dict]:
    """逆伽马族（CT-INVGAMMA-001）：3 个（R4c：gamma 输入模式同构训练族）。

    物理：精度/尺度倒数分布 IG(k, θ)（传导率、散粒噪声方差尺度、谱尺度）。
    Fisher 与伽马完全同形（x→1/x 微分同胚）：trace=ψ₁(k)+k/θ²、
    det=(k·ψ₁(k)−1)/θ²。k/theta 双列填充 + ψ₁ 语义——gamma 的 k 列在
    训练集无同构样本是其墙的根因假设（同 TRUNCGEOM 治 negbin 的逻辑）。
    """
    specs = [
        ("conductivity", "materials", "材料电导率分布（电阻率倒数）",
         "标准统计（逆伽马分布）", 3.0, 0.5),
        ("shotnoise_scale", "signal", "散粒噪声方差尺度（逆方差先验）",
         "标准统计（逆伽马分布）", 2.0, 2.0),
        ("spectral_scale", "quantum", "功率谱尺度参数（谱估计）",
         "标准统计（逆伽马分布）", 0.9, 1.5),
    ]
    entries = []
    for suffix, domain, name, ref, k, theta in specs:
        params = [
            {"name": "k", "symbol": "k", "range": [0.5, 10.0], "default": k},
            {"name": "theta", "symbol": "\\theta", "range": [0.1, 5.0],
             "default": theta},
        ]
        fpe = [
            {"quantity": "fisher_trace",
             "analytic": "polygamma(1, k) + k/theta^2", "tol": 1.0e-6},
            {"quantity": "fisher_det",
             "analytic": "(k*polygamma(1, k) - 1)/theta^2", "tol": 1.0e-6},
        ]
        eq = {"latex": "p(x) = \\frac{\\theta^k x^{-k-1} e^{-\\theta/x}}"
                       "{\\Gamma(k)}",
              "sympy": "p = theta**k * x**(-k-1) * exp(-theta/x) / gamma(k)"}
        entries.append(_base_entry(
            f"invgamma_{suffix}", domain, name, "inverse_gamma",
            "CT-INVGAMMA-001", params, fpe, equations=eq, source_ref=ref,
            scan_param="k",
        ))
    return entries


def gen_invchisq_entries() -> list[dict]:
    """逆卡方族（CT-INVCHISQ-001）：3 个（R4d：chisq 输入模式同构训练族）。

    物理：方差先验/精度二次型 χ⁻²(k)（θ=2 固定结构常数，不占参数列）。
    Fisher 关于 k = ψ₁(k/2)/4——与 chisq 逐点相同。
    k 单列 + 纯 ψ₁ 目标——chisq 的 k 单列模式在训练集无同构样本是其
    盲测墙的根因（第 49 条），同 TRUNCGEOM/INVGAMMA 的逻辑。
    """
    specs = [
        ("var_prior4", "signal", "白噪声方差先验（k=4 自由度）",
         "标准统计（逆卡方分布）", 4.0),
        ("var_prior8", "materials", "测量方差先验（k=8 自由度）",
         "标准统计（逆卡方分布）", 8.0),
        ("var_prior2", "quantum", "量子噪声方差先验（k=2 自由度）",
         "标准统计（逆卡方分布）", 2.0),
    ]
    entries = []
    for suffix, domain, name, ref, k in specs:
        params = [
            {"name": "k", "symbol": "k", "range": [1.0, 20.0], "default": k},
        ]
        fpe = [
            {"quantity": "fisher_trace",
             "analytic": "polygamma(1, k/2)/4", "tol": 1.0e-6},
            {"quantity": "fisher_det",
             "analytic": "polygamma(1, k/2)/4", "tol": 1.0e-6},
        ]
        eq = {"latex": "p(x) = \\frac{x^{-k/2-1} e^{-1/(2x)}}"
                       "{2^{k/2}\\Gamma(k/2)}",
              "sympy": "p = x**(-k/2-1)*exp(-1/(2*x))/(2**(k/2)*gamma(k/2))"}
        entries.append(_base_entry(
            f"invchisq_{suffix}", domain, name, "invchisq",
            "CT-INVCHISQ-001", params, fpe, equations=eq, source_ref=ref,
            scan_param="k",
        ))
    return entries


def gen_frechet_entries() -> list[dict]:
    """弗雷歇族（CT-FRECHET-001）：3 个（R4d：weibull 输入模式同构训练族）。

    物理：上限极值 Fréchet(α, s)（风速极值/洪水峰值/上限强度）。
    Fisher：trace = C/α² + α²/s²（U 形，C=(1−γ)²+π²/6 与 weibull 同常数）、
    det = π²/(6s²)。shape+scale 双列 + U 形目标——weibull 同构
    （盲测第 49 条：weibull 的 shape+scale 模式无同构样本是其墙的根因）。
    """
    specs = [
        ("wind_max", "diffusion", "年最大风速（上限极值）",
         "标准极值理论（Fréchet 分布）", 1.5, 2.0),
        ("flood_peak", "diffusion", "流域年最大流量峰值",
         "标准极值理论（Fréchet 分布）", 2.5, 1.0),
        ("strength_upper", "materials", "材料上限强度（最弱链接对偶）",
         "标准极值理论（Fréchet 分布）", 0.8, 3.0),
    ]
    entries = []
    for suffix, domain, name, ref, shape, scale in specs:
        params = [
            {"name": "shape", "symbol": "\\alpha", "range": [0.5, 5.0],
             "default": shape},
            {"name": "scale", "symbol": "s", "range": [0.1, 5.0],
             "default": scale},
        ]
        fpe = [
            {"quantity": "fisher_trace",
             "analytic": "((1 - EulerGamma)^2 + pi^2/6)/shape^2 "
                         "+ shape^2/scale^2", "tol": 1.0e-6},
            {"quantity": "fisher_det",
             "analytic": "pi^2/(6*scale^2)", "tol": 1.0e-6},
        ]
        eq = {"latex": "p(x) = \\alpha s^\\alpha x^{-\\alpha-1} "
                       "e^{-(s/x)^\\alpha}",
              "sympy": "p = a*s**a*x**(-a-1)*exp(-(s/x)**a)"}
        entries.append(_base_entry(
            f"frechet_{suffix}", domain, name, "frechet",
            "CT-FRECHET-001", params, fpe, equations=eq, source_ref=ref,
            scan_param="shape",
        ))
    return entries


def gen_rician_entries() -> list[dict]:
    """莱斯族（CT-RICIAN-001）：3 个（R4d 第三刀：vonmises 输入模式同构族）。

    物理：Rician 衰落 K 因子（强主径 + 散射，κ=ν²/2σ²，σ=1 固定）。
    Fisher 关于 κ 数值积分：κ→0（Rayleigh 极限）I→1；单调降；
    κ→∞ 尾 I→1/(2κ)——与 vonmises 同向（½→0）同尾（1/(4κ)）。
    kappa 单列 + 深集中衰减目标——vonmises 的 kappa 单列模式在训练集
    无同构样本是其盲测墙的根因（第 49/50 条）。
    """
    specs = [
        ("los_channel4", "signal", "视距信道幅度（K=4 强主径）",
         "标准衰落统计（Rician 分布）", 4.0),
        ("los_channel8", "signal", "强视距信道（K=8）",
         "标准衰落统计（Rician 分布）", 8.0),
        ("weak_los", "electromagnetism", "弱主径信道（K=1.5）",
         "标准衰落统计（Rician 分布）", 1.5),
    ]
    entries = []
    for suffix, domain, name, ref, kappa in specs:
        params = [
            {"name": "kappa", "symbol": "\\kappa", "range": [0.1, 20.0],
             "default": kappa},
        ]
        fpe: list[dict] = []
        eq = {"latex": "p(x) = x e^{-(x^2+\\nu^2)/2} I_0(x\\nu),"
                       "\\; \\nu=\\sqrt{2\\kappa}",
              "sympy": "p = x*exp(-(x**2+nu**2)/2)*besseli(0, x*nu)"}
        entries.append(_base_entry(
            f"rician_{suffix}", domain, name, "rician",
            "CT-RICIAN-001", params, fpe, equations=eq, source_ref=ref,
            scan_param="kappa",
        ))
    return entries


def gen_wrapnorm_entries() -> list[dict]:
    """卷绕高斯族（CT-WRAPNORM-001）：3 个（R4d 第三刀 v2：vonmises 三层同构）。

    物理：圆上相位扩散 WN(0, κ)（环形激光相位/天线阵相位/取向松弛）。
    Fisher 关于 κ 数值积分：κ→∞（均匀极限）→0，尾 ~2κ²e^(−κ²)。
    与 vonmises 三层同构：kappa 列 + 圆支撑 attr + 衰减到 0 方向
    （第 51 条修正后选型——rician 两层同构已反噬回滚）。
    """
    specs = [
        ("ring_laser", "electromagnetism", "环形激光相位扩散（κ=1）",
         "标准方向统计（卷绕高斯分布）", 1.0),
        ("antenna_array", "signal", "天线阵相位误差（κ=0.5）",
         "标准方向统计（卷绕高斯分布）", 0.5),
        ("orient_relax", "materials", "分子取向松弛（κ=2）",
         "标准方向统计（卷绕高斯分布）", 2.0),
    ]
    entries = []
    for suffix, domain, name, ref, kappa in specs:
        params = [
            {"name": "kappa", "symbol": "\\kappa", "range": [0.3, 3.0],
             "default": kappa},
        ]
        fpe: list[dict] = []
        eq = {"latex": "p(\\theta) = \\frac{1}{2\\pi}"
                       "(1+2\\sum_{k\\ge1}\\rho^{k^2}\\cos k\\theta),"
                       "\\; \\rho=e^{-\\kappa^2/2}",
              "sympy": "p = (1 + 2*sum(rho**(k**2)*cos(k*theta), "
                       "(k, 1, K)))/(2*pi)"}
        entries.append(_base_entry(
            f"wrapnorm_{suffix}", domain, name, "wrapped_normal",
            "CT-WRAPNORM-001", params, fpe, equations=eq, source_ref=ref,
            scan_param="kappa",
        ))
    return entries


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def gen_plasma_entries() -> list[dict]:
    """等离子体领域实例（2026-10-02）：4 个诊断系统。

    结构图示例域「等离子体」的落地——复用既有分布族与通道模板（族=物理
    机制、模板=信道类型、domain=领域标签，三者正交）：
      1. 汤姆逊散射光子计数 → 泊松计数（CT-COUNT-001）；
      2. 朗缪尔探针离子饱和流计数 → 泊松计数（CT-COUNT-001）；
      3. 湍流密度脉动幅值 → 瑞利幅值分布（CT-RAYLEIGH-001）；
      4. ECE 辐射计增益涨落 → 对数正态乘性噪声（CT-LOGNORM-001）。
    红线：AI+verified、human_review_status=pending、anchor 恒 false
    （人工复核后方可入锚点）。
    """
    entries = []

    # 1–2：计数型诊断（泊松）
    count_specs = [
        ("thomson_photon_count", "汤姆逊散射每通道光子计数",
         "标准统计（泊松计数过程）；等离子体诊断（汤姆逊散射）", 12.0, 0.1, 20.0),
        ("langmuir_ion_flux", "朗缪尔探针离子饱和流计数",
         "标准统计（泊松计数过程）；等离子体诊断（静电探针）", 4.0, 0.5, 20.0),
    ]
    for suffix, name, ref, rate, lo, hi in count_specs:
        params = [{"name": "rate", "symbol": "\\lambda",
                   "range": [lo, hi], "default": rate}]
        fpe = [
            {"quantity": "fisher_trace", "analytic": "1/rate", "tol": 1.0e-6},
            {"quantity": "fisher_det", "analytic": "1/rate", "tol": 1.0e-6},
        ]
        eq = {"latex": "P(n) = \\frac{\\lambda^n e^{-\\lambda}}{n!}",
              "sympy": "P = rate**n*exp(-rate)/factorial(n)"}
        entries.append(_base_entry(
            f"plasma_{suffix}", "plasma", name, "poisson_count",
            "CT-COUNT-001", params, fpe, equations=eq, source_ref=ref,
            scan_param="rate"))

    # 3：湍流脉动幅值（瑞利）
    entries.append(_base_entry(
        "plasma_turbulence_amplitude", "plasma", "湍流密度脉动幅值（包络）",
        "rayleigh", "CT-RAYLEIGH-001",
        [{"name": "sigma", "symbol": "\\sigma", "range": [0.1, 5.0],
          "default": 1.2}],
        [{"quantity": "fisher_trace", "analytic": "4/sigma**2", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "4/sigma**2", "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{x}{\\sigma^2} e^{-x^2/(2\\sigma^2)}",
                   "sympy": "p = (x/sigma**2)*exp(-x**2/(2*sigma**2))"},
        source_ref="标准统计（瑞利幅值分布）；等离子体边缘湍流诊断",
        scan_param="sigma"))

    # 4：ECE 辐射计增益涨落（对数正态乘性噪声）
    entries.append(_base_entry(
        "plasma_ece_gain_fluctuation", "plasma",
        "ECE 辐射计增益涨落（乘性噪声）", "lognormal", "CT-LOGNORM-001",
        [{"name": "mu", "symbol": "\\mu", "range": [-2.0, 2.0], "default": 0.0},
         {"name": "sigma", "symbol": "\\sigma", "range": [0.1, 5.0],
          "default": 0.6}],
        [{"quantity": "fisher_trace", "analytic": "3/sigma**2", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "2/sigma**4", "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{1}{x\\sigma\\sqrt{2\\pi}}"
                            " e^{-(\\ln x-\\mu)^2/(2\\sigma^2)}",
                   "sympy": "p = exp(-(log(x)-mu)**2/(2*sigma**2))"
                            "/(x*sigma*sqrt(2*pi))"},
        source_ref="标准信道模型（对数正态乘性噪声）；电子回旋辐射计增益漂移",
        scan_param="sigma"))
    return entries


def gen_plasma_entries_more() -> list[dict]:
    """等离子体域扩批（2026-10-02，来源：程序化生成）：+6 个诊断系统。

    覆盖 6 个既有族的新物理场景（族=机制、模板=信道、domain=领域标签）：
      negbin（逃逸电子雪崩成簇计数）、weibull（绝缘体沿面闪络寿命）、
      chisq（谱拟合卡方统计量）、mix2（MHD 模双稳观测）、
      truncnorm（阈值化诊断窗）、frechet（偏滤器极值热流）。
    """
    entries = []

    entries.append(_base_entry(
        "plasma_runaway_electron_burst", "plasma", "逃逸电子雪崩成簇计数",
        "negbin", "CT-NEGBIN-001",
        [{"name": "k", "symbol": "k", "range": [1.0, 10.0], "default": 4.0},
         {"name": "p", "symbol": "p", "range": [0.05, 0.5], "default": 0.12}],
        [{"quantity": "fisher_trace", "analytic": "k/(p**2*(1-p))",
          "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "k/(p**2*(1-p))",
          "tol": 1.0e-6}],
        equations={"latex": "P(n) = \\binom{n+k-1}{n} p^k (1-p)^n",
                   "sympy": "P = binomial(n+k-1,n)*p**k*(1-p)**n"},
        source_ref="标准统计（负二项成簇计数）；等离子体逃逸电子雪崩诊断",
        scan_param="p"))

    entries.append(_base_entry(
        "plasma_insulator_flashover", "plasma", "绝缘体沿面闪络寿命",
        "weibull", "CT-WEIBULL-001",
        [{"name": "shape", "symbol": "k", "range": [0.5, 5.0],
          "default": 1.8},
         {"name": "scale", "symbol": "\\lambda", "range": [0.1, 5.0],
          "default": 2.5}],
        [{"quantity": "fisher_trace",
          "analytic": "shape**2/scale**2 + ((1 - EulerGamma)**2 + pi**2/6)"
                      "/shape**2", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "pi**2/(6*scale**2)",
          "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{k}{\\lambda}"
                            "(x/\\lambda)^{k-1} e^{-(x/\\lambda)^k}",
                   "sympy": "p = (shape/scale)*(x/scale)**(shape-1)"
                            "*exp(-(x/scale)**shape)"},
        source_ref="Weibull 1951（最弱链接失效）；真空沿面闪络统计",
        scan_param="shape"))

    entries.append(_base_entry(
        "plasma_spectral_fit_chisq", "plasma", "谱拟合卡方统计量（k=10）",
        "chisq", "CT-CHISQ-001",
        [{"name": "k", "symbol": "k", "range": [1.0, 20.0], "default": 10.0}],
        [{"quantity": "fisher_trace", "analytic": "polygamma(1, k/2)/4",
          "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "polygamma(1, k/2)/4",
          "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{x^{k/2-1}e^{-x/2}}"
                            "{2^{k/2}\\Gamma(k/2)}",
                   "sympy": "p = x**(k/2-1)*exp(-x/2)/(2**(k/2)*gamma(k/2))"},
        source_ref="标准统计（卡方分布）；汤姆逊散射谱拟合优度",
        scan_param="k"))

    entries.append(_base_entry(
        "plasma_mhd_mode_bimodal", "plasma", "MHD 模双稳观测（对称双峰）",
        "gaussian_mixture2", "CT-MIX-001",
        [{"name": "d", "symbol": "d", "range": [0.05, 6.0], "default": 2.0},
         {"name": "w", "symbol": "w", "range": [0.1, 0.9], "default": 0.5},
         {"name": "sigma", "symbol": "\\sigma", "range": [0.5, 2.0],
          "default": 1.0}],
        [],
        equations={"latex": "p(x) = \\tfrac12 N(x|{-}d/2,\\sigma) "
                            "+ \\tfrac12 N(x|{+}d/2,\\sigma)",
                   "sympy": "p = 0.5*exp(-(x+d/2)**2/(2*sigma**2))"
                            "/(sigma*sqrt(2*pi)) + 0.5*exp(-(x-d/2)**2"
                            "/(2*sigma**2))/(sigma*sqrt(2*pi))"},
        source_ref="标准混合模型（信息几何）；磁流体不稳定性双稳观测",
        scan_param="d"))

    entries.append(_base_entry(
        "plasma_threshold_window", "plasma", "阈值化诊断窗（低于阈值不记录）",
        "truncated_gaussian", "CT-TRUNCNORM-001",
        [{"name": "tau", "symbol": "\\tau", "range": [-3.0, 1.0],
          "default": -1.5},
         {"name": "mu", "symbol": "\\mu", "range": [-2.0, 2.0],
          "default": 0.0},
         {"name": "sigma", "symbol": "\\sigma", "range": [0.5, 2.0],
          "default": 1.0}],
        [{"quantity": "fisher_trace",
          "analytic": "2*exp(-((tau-mu)/sigma)**2)/"
                      "(pi*sigma**2*erfc((tau-mu)/(sigma*sqrt(2)))**2)",
          "tol": 1.0e-6},
         {"quantity": "fisher_det",
          "analytic": "2*exp(-((tau-mu)/sigma)**2)/"
                      "(pi*sigma**2*erfc((tau-mu)/(sigma*sqrt(2)))**2)",
          "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{N(x|\\mu,\\sigma)}{1-\\Phi(z)},"
                            "\\; x>\\tau",
                   "sympy": "p = exp(-(x-mu)**2/(2*sigma**2))"
                            "/(sigma*sqrt(2*pi)*(1-Phi((tau-mu)/sigma)))"},
        source_ref="标准统计（截断高斯）；诊断窗阈值化",
        scan_param="tau"))

    entries.append(_base_entry(
        "plasma_extreme_heat_flux", "plasma", "偏滤器极值热流（上限极值）",
        "frechet", "CT-FRECHET-001",
        [{"name": "shape", "symbol": "\\alpha", "range": [0.5, 5.0],
          "default": 2.0},
         {"name": "scale", "symbol": "s", "range": [0.1, 5.0],
          "default": 1.5}],
        [{"quantity": "fisher_trace",
          "analytic": "((1-EulerGamma)**2+pi**2/6)/shape**2"
                      " + shape**2/scale**2", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "pi**2/(6*scale**2)",
          "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{\\alpha}{s}"
                            "(x/s)^{-1-\\alpha} e^{-(x/s)^{-\\alpha}}",
                   "sympy": "p = (shape/scale)*(x/scale)**(-1-shape)"
                            "*exp(-(x/scale)**(-shape))"},
        source_ref="标准极值理论（Fréchet 分布）；偏滤器热流极值统计",
        scan_param="shape"))
    return entries


def gen_geophysics_entries() -> list[dict]:
    """地球物理领域实例（2026-10-02，程序化生成）：4 个观测系统。"""
    entries = []
    entries.append(_base_entry(
        "geo_radar_roughness", "geophysics", "雷达高度计地表粗糙度幅值",
        "rayleigh", "CT-RAYLEIGH-001",
        [{"name": "sigma", "symbol": "\\sigma", "range": [0.1, 5.0],
          "default": 1.0}],
        [{"quantity": "fisher_trace", "analytic": "4/sigma**2", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "4/sigma**2", "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{x}{\\sigma^2} e^{-x^2/(2\\sigma^2)}",
                   "sympy": "p = (x/sigma**2)*exp(-x**2/(2*sigma**2))"},
        source_ref="标准统计（瑞利幅值分布）；卫星雷达高度计测高",
        scan_param="sigma"))

    entries.append(_base_entry(
        "geo_soil_conductivity_hetero", "geophysics", "土壤电导率空间异质（乘性）",
        "lognormal", "CT-LOGNORM-001",
        [{"name": "mu", "symbol": "\\mu", "range": [-2.0, 2.0], "default": 0.0},
         {"name": "sigma", "symbol": "\\sigma", "range": [0.1, 5.0],
          "default": 0.8}],
        [{"quantity": "fisher_trace", "analytic": "3/sigma**2", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "2/sigma**4", "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{1}{x\\sigma\\sqrt{2\\pi}}"
                            " e^{-(\\ln x-\\mu)^2/(2\\sigma^2)}",
                   "sympy": "p = exp(-(log(x)-mu)**2/(2*sigma**2))"
                            "/(x*sigma*sqrt(2*pi))"},
        source_ref="标准地质统计（对数正态空间异质）；电磁法勘探",
        scan_param="sigma"))

    entries.append(_base_entry(
        "geo_earthquake_recurrence", "geophysics", "地震复发间隔（最弱链接型）",
        "weibull", "CT-WEIBULL-001",
        [{"name": "shape", "symbol": "k", "range": [0.5, 5.0], "default": 2.0},
         {"name": "scale", "symbol": "\\lambda", "range": [0.1, 5.0],
          "default": 2.0}],
        [{"quantity": "fisher_trace",
          "analytic": "shape**2/scale**2 + ((1 - EulerGamma)**2 + pi**2/6)"
                      "/shape**2", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "pi**2/(6*scale**2)",
          "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{k}{\\lambda}"
                            "(x/\\lambda)^{k-1} e^{-(x/\\lambda)^k}",
                   "sympy": "p = (shape/scale)*(x/scale)**(shape-1)"
                            "*exp(-(x/scale)**shape)"},
        source_ref="Weibull 1951；地震复发间隔统计（时间可预测模型）",
        scan_param="shape"))

    entries.append(_base_entry(
        "geo_seismic_count_rate", "geophysics", "区域地震事件计数率",
        "poisson_count", "CT-COUNT-001",
        [{"name": "rate", "symbol": "\\lambda", "range": [0.1, 20.0],
          "default": 6.0}],
        [{"quantity": "fisher_trace", "analytic": "1/rate", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "1/rate", "tol": 1.0e-6}],
        equations={"latex": "P(n) = \\frac{\\lambda^n e^{-\\lambda}}{n!}",
                   "sympy": "P = rate**n*exp(-rate)/factorial(n)"},
        source_ref="标准统计（泊松计数过程）；区域地震目录统计",
        scan_param="rate"))
    return entries


def gen_biomedical_entries() -> list[dict]:
    """生物医学领域实例（2026-10-02，程序化生成）：4 个测量系统。"""
    entries = []
    entries.append(_base_entry(
        "bio_pet_photon_count", "biomedical", "PET 符合光子计数（每体素）",
        "poisson_count", "CT-COUNT-001",
        [{"name": "rate", "symbol": "\\lambda", "range": [0.1, 20.0],
          "default": 8.0}],
        [{"quantity": "fisher_trace", "analytic": "1/rate", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "1/rate", "tol": 1.0e-6}],
        equations={"latex": "P(n) = \\frac{\\lambda^n e^{-\\lambda}}{n!}",
                   "sympy": "P = rate**n*exp(-rate)/factorial(n)"},
        source_ref="标准统计（泊松计数）；正电子发射断层成像",
        scan_param="rate"))

    entries.append(_base_entry(
        "bio_cell_size_distribution", "biomedical", "细胞尺寸分布（乘性生长）",
        "lognormal", "CT-LOGNORM-001",
        [{"name": "mu", "symbol": "\\mu", "range": [-2.0, 2.0], "default": 0.5},
         {"name": "sigma", "symbol": "\\sigma", "range": [0.1, 5.0],
          "default": 0.6}],
        [{"quantity": "fisher_trace", "analytic": "3/sigma**2", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "2/sigma**4", "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{1}{x\\sigma\\sqrt{2\\pi}}"
                            " e^{-(\\ln x-\\mu)^2/(2\\sigma^2)}",
                   "sympy": "p = exp(-(log(x)-mu)**2/(2*sigma**2))"
                            "/(x*sigma*sqrt(2*pi))"},
        source_ref="标准细胞群体统计（对数正态尺寸）；流式细胞术",
        scan_param="sigma"))

    entries.append(_base_entry(
        "bio_survival_time", "biomedical", "生存时间（风险函数型）",
        "weibull", "CT-WEIBULL-001",
        [{"name": "shape", "symbol": "k", "range": [0.5, 5.0], "default": 1.4},
         {"name": "scale", "symbol": "\\lambda", "range": [0.1, 5.0],
          "default": 3.0}],
        [{"quantity": "fisher_trace",
          "analytic": "shape**2/scale**2 + ((1 - EulerGamma)**2 + pi**2/6)"
                      "/shape**2", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "pi**2/(6*scale**2)",
          "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{k}{\\lambda}"
                            "(x/\\lambda)^{k-1} e^{-(x/\\lambda)^k}",
                   "sympy": "p = (shape/scale)*(x/scale)**(shape-1)"
                            "*exp(-(x/scale)**shape)"},
        source_ref="Weibull 1951；生存分析（参数化风险模型）",
        scan_param="shape"))

    entries.append(_base_entry(
        "bio_mri_signal_amplitude", "biomedical", "MRI 幅值信号（包络分布）",
        "rayleigh", "CT-RAYLEIGH-001",
        [{"name": "sigma", "symbol": "\\sigma", "range": [0.1, 5.0],
          "default": 0.9}],
        [{"quantity": "fisher_trace", "analytic": "4/sigma**2", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "4/sigma**2", "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{x}{\\sigma^2} e^{-x^2/(2\\sigma^2)}",
                   "sympy": "p = (x/sigma**2)*exp(-x**2/(2*sigma**2))"},
        source_ref="标准统计（瑞利包络）；MRI 幅值图像噪声模型",
        scan_param="sigma"))
    return entries


def gen_energy_entries() -> list[dict]:
    """能源领域实例（2026-10-02，程序化生成）：4 个系统。"""
    entries = []
    entries.append(_base_entry(
        "energy_battery_soc_noise", "energy", "电池 SOC 估计噪声（有色噪声）",
        "ou", "CT-DIFF-001",
        [{"name": "theta", "symbol": "\\theta", "range": [0.01, 5.0],
          "default": 0.5},
         {"name": "sigma", "symbol": "\\sigma", "range": [0.1, 3.0],
          "default": 1.0}],
        [], equations={"latex": "dX = -\\theta X\\,dt + \\sigma\\,dW",
                       "sympy": "dX = -theta*X*dt + sigma*dW"},
        source_ref="标准随机过程（OU 有色噪声）；电池 SOC 滤波残差",
        scan_param="theta"))
    entries.append(_base_entry(
        "energy_pv_irradiance_fluct", "energy", "光伏辐照度波动（乘性）",
        "lognormal", "CT-LOGNORM-001",
        [{"name": "mu", "symbol": "\\mu", "range": [-2.0, 2.0], "default": 0.0},
         {"name": "sigma", "symbol": "\\sigma", "range": [0.1, 5.0],
          "default": 0.7}],
        [{"quantity": "fisher_trace", "analytic": "3/sigma**2", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "2/sigma**4", "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{1}{x\\sigma\\sqrt{2\\pi}}"
                            " e^{-(\\ln x-\\mu)^2/(2\\sigma^2)}",
                   "sympy": "p = exp(-(log(x)-mu)**2/(2*sigma**2))"
                            "/(x*sigma*sqrt(2*pi))"},
        source_ref="标准统计（对数正态乘性波动）；光伏出力波动建模",
        scan_param="sigma"))
    entries.append(_base_entry(
        "energy_wind_turbine_load_extreme", "energy", "风机载荷极值（上限极值）",
        "frechet", "CT-FRECHET-001",
        [{"name": "shape", "symbol": "\\alpha", "range": [0.5, 5.0],
          "default": 2.2},
         {"name": "scale", "symbol": "s", "range": [0.1, 5.0],
          "default": 1.2}],
        [{"quantity": "fisher_trace",
          "analytic": "((1-EulerGamma)**2+pi**2/6)/shape**2"
                      " + shape**2/scale**2", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "pi**2/(6*scale**2)",
          "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{\\alpha}{s}"
                            "(x/s)^{-1-\\alpha} e^{-(x/s)^{-\\alpha}}",
                   "sympy": "p = (shape/scale)*(x/scale)**(-1-shape)"
                            "*exp(-(x/scale)**(-shape))"},
        source_ref="标准极值理论（Fréchet）；风机极限载荷统计",
        scan_param="shape"))
    entries.append(_base_entry(
        "energy_grid_freq_events", "energy", "电网频率越限事件计数",
        "poisson_count", "CT-COUNT-001",
        [{"name": "rate", "symbol": "\\lambda", "range": [0.1, 20.0],
          "default": 3.0}],
        [{"quantity": "fisher_trace", "analytic": "1/rate", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "1/rate", "tol": 1.0e-6}],
        equations={"latex": "P(n) = \\frac{\\lambda^n e^{-\\lambda}}{n!}",
                   "sympy": "P = rate**n*exp(-rate)/factorial(n)"},
        source_ref="标准统计（泊松计数过程）；电网频率越限统计",
        scan_param="rate"))
    return entries


def gen_agriculture_entries() -> list[dict]:
    """农业领域实例（2026-10-02，程序化生成）：4 个系统。"""
    entries = []
    entries.append(_base_entry(
        "agri_yield_count", "agriculture", "单位面积产量计数（抽样）",
        "poisson_count", "CT-COUNT-001",
        [{"name": "rate", "symbol": "\\lambda", "range": [0.1, 20.0],
          "default": 9.0}],
        [{"quantity": "fisher_trace", "analytic": "1/rate", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "1/rate", "tol": 1.0e-6}],
        equations={"latex": "P(n) = \\frac{\\lambda^n e^{-\\lambda}}{n!}",
                   "sympy": "P = rate**n*exp(-rate)/factorial(n)"},
        source_ref="标准统计（泊松计数）；产量抽样调查",
        scan_param="rate"))
    entries.append(_base_entry(
        "agri_soil_moisture_sat", "agriculture", "土壤湿度饱和响应（有界占据）",
        "beta", "CT-BETA-001",
        [{"name": "alpha", "symbol": "\\alpha", "range": [0.5, 10.0],
          "default": 2.5},
         {"name": "beta", "symbol": "\\beta", "range": [0.5, 10.0],
          "default": 2.0}],
        [{"quantity": "fisher_trace",
          "analytic": "polygamma(1, alpha) + polygamma(1, beta)"
                      " - 2*polygamma(1, alpha + beta)", "tol": 1.0e-6},
         {"quantity": "fisher_det",
          "analytic": "polygamma(1, alpha)*polygamma(1, beta)"
                      " - polygamma(1, alpha + beta)*(polygamma(1, alpha)"
                      " + polygamma(1, beta))", "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}}"
                            "{B(\\alpha,\\beta)}",
                   "sympy": "p = x**(alpha-1)*(1-x)**(beta-1)"
                            "/(gamma(alpha)*gamma(beta)/gamma(alpha+beta))"},
        source_ref="标准统计（贝塔分布）；土壤湿度归一化响应",
        scan_param="alpha"))
    entries.append(_base_entry(
        "agri_rainfall_extreme", "agriculture", "农田极端降雨（上限极值）",
        "frechet", "CT-FRECHET-001",
        [{"name": "shape", "symbol": "\\alpha", "range": [0.5, 5.0],
          "default": 1.8},
         {"name": "scale", "symbol": "s", "range": [0.1, 5.0],
          "default": 1.6}],
        [{"quantity": "fisher_trace",
          "analytic": "((1-EulerGamma)**2+pi**2/6)/shape**2"
                      " + shape**2/scale**2", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "pi**2/(6*scale**2)",
          "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{\\alpha}{s}"
                            "(x/s)^{-1-\\alpha} e^{-(x/s)^{-\\alpha}}",
                   "sympy": "p = (shape/scale)*(x/scale)**(-1-shape)"
                            "*exp(-(x/scale)**(-shape))"},
        source_ref="标准极值理论（Fréchet）；农业气象极端降雨",
        scan_param="shape"))
    entries.append(_base_entry(
        "agri_pest_outbreak_cluster", "agriculture", "虫害爆发成簇计数",
        "negbin", "CT-NEGBIN-001",
        [{"name": "k", "symbol": "k", "range": [1.0, 10.0], "default": 3.0},
         {"name": "p", "symbol": "p", "range": [0.05, 0.5], "default": 0.2}],
        [{"quantity": "fisher_trace", "analytic": "k/(p**2*(1-p))",
          "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "k/(p**2*(1-p))",
          "tol": 1.0e-6}],
        equations={"latex": "P(n) = \\binom{n+k-1}{n} p^k (1-p)^n",
                   "sympy": "P = binomial(n+k-1,n)*p**k*(1-p)**n"},
        source_ref="标准统计（负二项成簇计数）；虫害时空聚集",
        scan_param="p"))
    return entries


def gen_acoustics_entries() -> list[dict]:
    """声学领域实例（2026-10-02，程序化生成）：4 个系统。"""
    entries = []
    entries.append(_base_entry(
        "acoustics_sonar_echo_envelope", "acoustics", "声呐回波包络幅值",
        "rayleigh", "CT-RAYLEIGH-001",
        [{"name": "sigma", "symbol": "\\sigma", "range": [0.1, 5.0],
          "default": 1.1}],
        [{"quantity": "fisher_trace", "analytic": "4/sigma**2", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "4/sigma**2", "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{x}{\\sigma^2} e^{-x^2/(2\\sigma^2)}",
                   "sympy": "p = (x/sigma**2)*exp(-x**2/(2*sigma**2))"},
        source_ref="标准统计（瑞利包络）；主动声呐回波幅度",
        scan_param="sigma"))
    entries.append(_base_entry(
        "acoustics_reverb_decay_life", "acoustics", "混响衰减时间（风险型寿命）",
        "weibull", "CT-WEIBULL-001",
        [{"name": "shape", "symbol": "k", "range": [0.5, 5.0], "default": 1.6},
         {"name": "scale", "symbol": "\\lambda", "range": [0.1, 5.0],
          "default": 2.2}],
        [{"quantity": "fisher_trace",
          "analytic": "shape**2/scale**2 + ((1 - EulerGamma)**2 + pi**2/6)"
                      "/shape**2", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "pi**2/(6*scale**2)",
          "tol": 1.0e-6}],
        equations={"latex": "p(x) = \\frac{k}{\\lambda}"
                            "(x/\\lambda)^{k-1} e^{-(x/\\lambda)^k}",
                   "sympy": "p = (shape/scale)*(x/scale)**(shape-1)"
                            "*exp(-(x/scale)**shape)"},
        source_ref="Weibull 1951；房间混响衰减统计",
        scan_param="shape"))
    entries.append(_base_entry(
        "acoustics_array_phase", "acoustics", "麦克风阵列相位误差（圆上分布）",
        "vonmises", "CT-VONMISES-001",
        [{"name": "kappa", "symbol": "\\kappa", "range": [0.1, 20.0],
          "default": 5.0}],
        [], equations={"latex": "p(\\theta) = \\frac{e^{\\kappa\\cos\\theta}}"
                                "{2\\pi I_0(\\kappa)}",
                       "sympy": "p = exp(kappa*cos(theta))/(2*pi*besseli(0,kappa))"},
        source_ref="标准统计（von Mises 圆分布）；阵列相位误差",
        scan_param="kappa"))
    entries.append(_base_entry(
        "acoustics_emission_events", "acoustics", "声发射事件计数",
        "poisson_count", "CT-COUNT-001",
        [{"name": "rate", "symbol": "\\lambda", "range": [0.1, 20.0],
          "default": 5.0}],
        [{"quantity": "fisher_trace", "analytic": "1/rate", "tol": 1.0e-6},
         {"quantity": "fisher_det", "analytic": "1/rate", "tol": 1.0e-6}],
        equations={"latex": "P(n) = \\frac{\\lambda^n e^{-\\lambda}}{n!}",
                   "sympy": "P = rate**n*exp(-rate)/factorial(n)"},
        source_ref="标准统计（泊松计数）；材料声发射监测",
        scan_param="rate"))
    return entries


def gen_multi_domain_entries() -> list[dict]:
    """大规模扩域（2026-10-02，表驱动）：12 领域 × 3 系统 = 36 系统。

    族模板复用本会话已数值验证的 9 个族——fpe 公式与既有条目逐字一致，
    避免「新写法引入新错误」（本轮之前已因写法笔误修过 2 处）。
    """
    fam_tmpl = {
        "poisson_count": (
            [{"name": "rate", "symbol": "\\lambda", "range": [0.1, 20.0],
              "default": 5.0}],
            [{"quantity": "fisher_trace", "analytic": "1/rate", "tol": 1e-6},
             {"quantity": "fisher_det", "analytic": "1/rate", "tol": 1e-6}],
            {"latex": "P(n) = \\frac{\\lambda^n e^{-\\lambda}}{n!}",
             "sympy": "P = rate**n*exp(-rate)/factorial(n)"},
            "CT-COUNT-001", "rate"),
        "rayleigh": (
            [{"name": "sigma", "symbol": "\\sigma", "range": [0.1, 5.0],
              "default": 1.0}],
            [{"quantity": "fisher_trace", "analytic": "4/sigma**2",
              "tol": 1e-6},
             {"quantity": "fisher_det", "analytic": "4/sigma**2", "tol": 1e-6}],
            {"latex": "p(x) = \\frac{x}{\\sigma^2} e^{-x^2/(2\\sigma^2)}",
             "sympy": "p = (x/sigma**2)*exp(-x**2/(2*sigma**2))"},
            "CT-RAYLEIGH-001", "sigma"),
        "weibull": (
            [{"name": "shape", "symbol": "k", "range": [0.5, 5.0],
              "default": 1.8},
             {"name": "scale", "symbol": "\\lambda", "range": [0.1, 5.0],
              "default": 2.0}],
            [{"quantity": "fisher_trace",
              "analytic": "shape**2/scale**2 + ((1 - EulerGamma)**2 + pi**2/6)"
                          "/shape**2", "tol": 1e-6},
             {"quantity": "fisher_det", "analytic": "pi**2/(6*scale**2)",
              "tol": 1e-6}],
            {"latex": "p(x) = \\frac{k}{\\lambda}(x/\\lambda)^{k-1}"
                      " e^{-(x/\\lambda)^k}",
             "sympy": "p = (shape/scale)*(x/scale)**(shape-1)"
                      "*exp(-(x/scale)**shape)"}, "CT-WEIBULL-001", "shape"),
        "frechet": (
            [{"name": "shape", "symbol": "\\alpha", "range": [0.5, 5.0],
              "default": 2.0},
             {"name": "scale", "symbol": "s", "range": [0.1, 5.0],
              "default": 1.5}],
            [{"quantity": "fisher_trace",
              "analytic": "((1-EulerGamma)**2+pi**2/6)/shape**2"
                          " + shape**2/scale**2", "tol": 1e-6},
             {"quantity": "fisher_det", "analytic": "pi**2/(6*scale**2)",
              "tol": 1e-6}],
            {"latex": "p(x) = \\frac{\\alpha}{s}(x/s)^{-1-\\alpha}"
                      " e^{-(x/s)^{-\\alpha}}",
             "sympy": "p = (shape/scale)*(x/scale)**(-1-shape)"
                      "*exp(-(x/scale)**(-shape))"}, "CT-FRECHET-001", "shape"),
        "lognormal": (
            [{"name": "mu", "symbol": "\\mu", "range": [-2.0, 2.0],
              "default": 0.0},
             {"name": "sigma", "symbol": "\\sigma", "range": [0.1, 5.0],
              "default": 0.7}],
            [{"quantity": "fisher_trace", "analytic": "3/sigma**2",
              "tol": 1e-6},
             {"quantity": "fisher_det", "analytic": "2/sigma**4", "tol": 1e-6}],
            {"latex": "p(x) = \\frac{1}{x\\sigma\\sqrt{2\\pi}}"
                      " e^{-(\\ln x-\\mu)^2/(2\\sigma^2)}",
             "sympy": "p = exp(-(log(x)-mu)**2/(2*sigma**2))"
                      "/(x*sigma*sqrt(2*pi))"}, "CT-LOGNORM-001", "sigma"),
        "negbin": (
            [{"name": "k", "symbol": "k", "range": [1.0, 10.0], "default": 3.0},
             {"name": "p", "symbol": "p", "range": [0.05, 0.5],
              "default": 0.2}],
            [{"quantity": "fisher_trace", "analytic": "k/(p**2*(1-p))",
              "tol": 1e-6},
             {"quantity": "fisher_det", "analytic": "k/(p**2*(1-p))",
              "tol": 1e-6}],
            {"latex": "P(n) = \\binom{n+k-1}{n} p^k (1-p)^n",
             "sympy": "P = binomial(n+k-1,n)*p**k*(1-p)**n"},
            "CT-NEGBIN-001", "p"),
        "beta": (
            [{"name": "alpha", "symbol": "\\alpha", "range": [0.5, 10.0],
              "default": 2.5},
             {"name": "beta", "symbol": "\\beta", "range": [0.5, 10.0],
              "default": 2.0}],
            [{"quantity": "fisher_trace",
              "analytic": "polygamma(1, alpha) + polygamma(1, beta)"
                          " - 2*polygamma(1, alpha + beta)", "tol": 1e-6},
             {"quantity": "fisher_det",
              "analytic": "polygamma(1, alpha)*polygamma(1, beta)"
                          " - polygamma(1, alpha + beta)*(polygamma(1, alpha)"
                          " + polygamma(1, beta))", "tol": 1e-6}],
            {"latex": "p(x) = \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}}"
                      "{B(\\alpha,\\beta)}",
             "sympy": "p = x**(alpha-1)*(1-x)**(beta-1)"
                      "/(gamma(alpha)*gamma(beta)/gamma(alpha+beta))"},
            "CT-BETA-001", "alpha"),
        "chisq": (
            [{"name": "k", "symbol": "k", "range": [1.0, 20.0],
              "default": 8.0}],
            [{"quantity": "fisher_trace", "analytic": "polygamma(1, k/2)/4",
              "tol": 1e-6},
             {"quantity": "fisher_det", "analytic": "polygamma(1, k/2)/4",
              "tol": 1e-6}],
            {"latex": "p(x) = \\frac{x^{k/2-1}e^{-x/2}}"
                      "{2^{k/2}\\Gamma(k/2)}",
             "sympy": "p = x**(k/2-1)*exp(-x/2)/(2**(k/2)*gamma(k/2))"},
            "CT-CHISQ-001", "k"),
        "truncated_gaussian": (
            [{"name": "tau", "symbol": "\\tau", "range": [-3.0, 1.0],
              "default": -1.0},
             {"name": "mu", "symbol": "\\mu", "range": [-2.0, 2.0],
              "default": 0.0},
             {"name": "sigma", "symbol": "\\sigma", "range": [0.5, 2.0],
              "default": 1.0}],
            [{"quantity": "fisher_trace",
              "analytic": "2*exp(-((tau-mu)/sigma)**2)/"
                          "(pi*sigma**2*erfc((tau-mu)/(sigma*sqrt(2)))**2)",
              "tol": 1e-6},
             {"quantity": "fisher_det",
              "analytic": "2*exp(-((tau-mu)/sigma)**2)/"
                          "(pi*sigma**2*erfc((tau-mu)/(sigma*sqrt(2)))**2)",
              "tol": 1e-6}],
            {"latex": "p(x) = \\frac{N(x|\\mu,\\sigma)}{1-\\Phi(z)},\\; x>\\tau",
             "sympy": "p = exp(-(x-mu)**2/(2*sigma**2))"
                      "/(sigma*sqrt(2*pi)*(1-Phi((tau-mu)/sigma)))"},
            "CT-TRUNCNORM-001", "tau"),
    }

    specs = [
        ("aerospace", "panel_amplitude", "舱板振动幅值", "rayleigh",
         "航天结构统计", {"sigma": 1.2}),
        ("aerospace", "reentry_heat_extreme", "再入热流极值", "frechet",
         "极值理论", {"shape": 2.4, "scale": 1.3}),
        ("aerospace", "attitude_event_count", "姿态扰动事件计数",
         "poisson_count", "泊松计数", {"rate": 4.0}),
        ("automotive", "suspension_amplitude", "悬架振动幅值", "rayleigh",
         "车辆振动统计", {"sigma": 1.4}),
        ("automotive", "brake_fade_life", "制动热衰退寿命", "weibull",
         "Weibull 失效统计", {"shape": 2.2, "scale": 1.7}),
        ("automotive", "crash_energy_absorb", "碰撞吸能乘性波动", "lognormal",
         "碰撞试验统计", {"sigma": 0.5}),
        ("civil", "deck_roughness", "桥面粗糙度幅值", "rayleigh",
         "路面谱统计", {"sigma": 1.0}),
        ("civil", "curing_heat_life", "混凝土养护放热寿命", "weibull",
         "水化放热统计", {"shape": 1.6, "scale": 2.4}),
        ("civil", "seismic_peak_load", "地震峰值荷载（上限极值）", "frechet",
         "极值理论", {"shape": 2.6, "scale": 1.1}),
        ("chemical", "reactor_temp_fluct", "反应器温度乘性波动", "lognormal",
         "过程统计", {"sigma": 0.4}),
        ("chemical", "catalyst_deactivation", "催化剂失活寿命", "weibull",
         "Weibull 寿命", {"shape": 1.3, "scale": 3.0}),
        ("chemical", "mixing_fraction", "混合均匀度（有界占据）", "beta",
         "混合效率统计", {"alpha": 2.0, "beta": 3.0}),
        ("food", "cold_chain_fluct", "冷链温度乘性波动", "lognormal",
         "冷链统计", {"sigma": 0.3}),
        ("food", "sterilization_fraction", "杀菌致死率（有界）", "beta",
         "杀菌动力学", {"alpha": 3.0, "beta": 1.5}),
        ("food", "shelf_life_failure", "货架期失效寿命", "weibull",
         "Weibull 失效", {"shape": 2.0, "scale": 2.8}),
        ("textile", "fiber_strength_extreme", "纤维断裂强度（极值）", "frechet",
         "纤维强度统计", {"shape": 2.8, "scale": 1.4}),
        ("textile", "dye_diffusion_window", "染液扩散窗口（阈值化）",
         "truncated_gaussian", "扩散统计", {"tau": -1.2}),
        ("textile", "defect_cluster", "织物疵点成簇计数", "negbin",
         "疵点聚集统计", {"k": 2.0, "p": 0.15}),
        ("optics", "mirror_slope_amp", "镜面斜率误差幅值", "rayleigh",
         "光学面形统计", {"sigma": 0.8}),
        ("optics", "coating_damage_life", "镀膜激光损伤寿命", "weibull",
         "损伤阈值统计", {"shape": 2.5, "scale": 1.9}),
        ("optics", "wavefront_chi2", "波前拟合卡方统计量", "chisq",
         "泽尼克拟合检验", {"k": 12.0}),
        ("robotics", "joint_backlash_amp", "关节回差幅值", "rayleigh",
         "传动统计", {"sigma": 0.6}),
        ("robotics", "slip_event_count", "抓取滑移事件计数", "poisson_count",
         "泊松计数", {"rate": 2.5}),
        ("robotics", "tracking_error_window", "轨迹偏差窗口（阈值化）",
         "truncated_gaussian", "跟踪误差统计", {"tau": -0.8}),
        ("marine", "mooring_amplitude", "系泊缆张力幅值", "rayleigh",
         "波浪载荷统计", {"sigma": 1.6}),
        ("marine", "corrosion_life", "船体腐蚀寿命", "weibull",
         "腐蚀寿命统计", {"shape": 1.5, "scale": 2.6}),
        ("marine", "wave_peak_load", "波浪峰值载荷（上限极值）", "frechet",
         "极值理论", {"shape": 2.3, "scale": 1.7}),
        ("mining", "rock_vibration_amp", "岩体振动幅值", "rayleigh",
         "爆破振动统计", {"sigma": 1.3}),
        ("mining", "ventilation_life", "通风热锋寿命", "weibull",
         "矿井热环境统计", {"shape": 1.7, "scale": 2.1}),
        ("mining", "equipment_failure_life", "设备失效寿命", "weibull",
         "可靠性统计", {"shape": 1.2, "scale": 3.2}),
        ("semiconductor", "litho_overlay_amp", "光刻套刻误差幅值", "rayleigh",
         "套刻统计", {"sigma": 0.5}),
        ("semiconductor", "anneal_window", "退火扩散窗口（阈值化）",
         "truncated_gaussian", "扩散统计", {"tau": -1.4}),
        ("semiconductor", "defect_cluster_count", "缺陷成簇计数", "negbin",
         "缺陷聚集统计", {"k": 4.0, "p": 0.1}),
        ("sports", "impact_amplitude", "器材冲击幅值", "rayleigh",
         "冲击响应统计", {"sigma": 1.2}),
        ("sports", "cushion_fatigue_life", "缓震材料疲劳寿命", "weibull",
         "疲劳寿命统计", {"shape": 2.1, "scale": 2.3}),
        ("sports", "helmet_absorption", "头盔吸能分布（有界）", "beta",
         "吸能统计", {"alpha": 2.5, "beta": 2.5}),
    ]

    entries = []
    for domain, suffix, name, family, ref, over in specs:
        params, fpe, eq, ct, scan = fam_tmpl[family]
        params = [dict(p) for p in params]
        for p in params:
            if p["name"] in over:
                p["default"] = over[p["name"]]
        entries.append(_base_entry(
            f"{domain}_{suffix}", domain, name, family, ct, params,
            [dict(f) for f in fpe], equations=dict(eq),
            source_ref=f"{ref}；{domain} 域实例（程序化生成，表驱动）",
            scan_param=scan))
    return entries


def gen_multi_domain_entries_2() -> list[dict]:
    """批次 2 扩域（2026-10-02）：13 领域 × 3 = 39 系统（复用批次 1 族模板）。"""
    specs = [
        ("nuclear", "gamma_count_rate", "γ 谱计数率", "poisson_count",
         "核测量统计", {"rate": 6.0}),
        ("nuclear", "detector_degradation_life", "探测器退化寿命", "weibull",
         "辐照损伤寿命统计", {"shape": 1.9, "scale": 2.2}),
        ("nuclear", "neutron_scatter_amplitude", "中子散射幅值", "rayleigh",
         "散射统计", {"sigma": 1.1}),
        ("meteorology", "wind_gust_amplitude", "阵风幅值", "rayleigh",
         "风场统计", {"sigma": 1.7}),
        ("meteorology", "pressure_extreme", "极端气压（上限极值）", "frechet",
         "极值理论", {"shape": 2.1, "scale": 1.4}),
        ("meteorology", "cloud_condensation_count", "云凝结核计数",
         "poisson_count", "气溶胶计数", {"rate": 7.0}),
        ("hydrology", "stage_fluctuation", "水位波动（乘性）", "lognormal",
         "水文统计", {"sigma": 0.55}),
        ("hydrology", "flood_peak_extreme", "洪峰流量（上限极值）", "frechet",
         "极值理论", {"shape": 2.5, "scale": 1.2}),
        ("hydrology", "sediment_transport_life", "输沙过程寿命", "weibull",
         "输沙统计", {"shape": 1.4, "scale": 2.9}),
        ("forestry", "canopy_height_amp", "冠层高度起伏幅值", "rayleigh",
         "林分结构统计", {"sigma": 1.5}),
        ("forestry", "fire_spread_extreme", "林火蔓延极值", "frechet",
         "极值理论", {"shape": 2.7, "scale": 1.0}),
        ("forestry", "pest_defect_cluster", "林害成簇计数", "negbin",
         "林害聚集统计", {"k": 3.0, "p": 0.18}),
        ("aquaculture", "turbidity_fluct", "浊度乘性波动", "lognormal",
         "水质统计", {"sigma": 0.45}),
        ("aquaculture", "oxygen_saturation_fraction", "溶氧饱和度（有界）",
         "beta", "溶氧统计", {"alpha": 2.2, "beta": 3.2}),
        ("aquaculture", "biofouling_life", "生物污损寿命", "weibull",
         "污损寿命统计", {"shape": 1.6, "scale": 2.7}),
        ("pharmaceutical", "particle_size_fluct", "粒径乘性波动", "lognormal",
         "粒径统计", {"sigma": 0.5}),
        ("pharmaceutical", "dissolution_fraction", "溶出度（有界占据）",
         "beta", "溶出统计", {"alpha": 3.5, "beta": 1.8}),
        ("pharmaceutical", "stability_shelf_life", "稳定性货架寿命", "weibull",
         "稳定性统计", {"shape": 2.3, "scale": 2.5}),
        ("packaging", "seal_strength_amp", "封口强度幅值", "rayleigh",
         "封口统计", {"sigma": 1.3}),
        ("packaging", "drop_impact_extreme", "跌落冲击极值", "frechet",
         "极值理论", {"shape": 2.4, "scale": 1.3}),
        ("packaging", "leak_count_cluster", "泄漏点成簇计数", "negbin",
         "泄漏聚集统计", {"k": 2.5, "p": 0.12}),
        ("coating", "film_thickness_amp", "膜厚起伏幅值", "rayleigh",
         "膜厚统计", {"sigma": 0.9}),
        ("coating", "adhesion_failure_life", "附着失效寿命", "weibull",
         "附着寿命统计", {"shape": 1.8, "scale": 2.4}),
        ("coating", "curing_window", "固化窗口（阈值化）",
         "truncated_gaussian", "固化统计", {"tau": -1.1}),
        ("railway", "track_geometry_fluct", "轨道几何乘性波动", "lognormal",
         "轨道几何统计", {"sigma": 0.6}),
        ("railway", "rail_fatigue_life", "钢轨疲劳寿命", "weibull",
         "疲劳寿命统计", {"shape": 2.6, "scale": 2.0}),
        ("railway", "wheel_impact_extreme", "轮轨冲击极值", "frechet",
         "极值理论", {"shape": 2.2, "scale": 1.6}),
        ("traffic", "speed_estimate_amp", "车速估计幅值", "rayleigh",
         "交通感知统计", {"sigma": 1.0}),
        ("traffic", "queue_count", "排队车辆计数", "poisson_count",
         "泊松计数", {"rate": 12.0}),
        ("traffic", "incident_cluster", "事故成簇计数", "negbin",
         "事故聚集统计", {"k": 3.5, "p": 0.1}),
        ("water_utility", "flow_fluct", "流量乘性波动", "lognormal",
         "流量统计", {"sigma": 0.4}),
        ("water_utility", "pipe_failure_life", "管道失效寿命", "weibull",
         "管网寿命统计", {"shape": 1.3, "scale": 3.4}),
        ("water_utility", "pressure_extreme_utility", "管网压力极值", "frechet",
         "极值理论", {"shape": 2.5, "scale": 1.2}),
        ("geotechnical", "settlement_amp", "沉降幅值", "rayleigh",
         "沉降统计", {"sigma": 0.7}),
        ("geotechnical", "seepage_window", "渗流窗口（阈值化）",
         "truncated_gaussian", "渗流统计", {"tau": -1.3}),
        ("geotechnical", "slope_failure_life", "边坡失稳寿命", "weibull",
         "失稳寿命统计", {"shape": 1.5, "scale": 3.0}),
        ("metrology", "probe_noise_amp", "测头噪声幅值", "rayleigh",
         "计量统计", {"sigma": 0.4}),
        ("metrology", "gauge_block_chi2", "量块比对卡方统计量", "chisq",
         "比对检验", {"k": 15.0}),
        ("metrology", "calibration_interval_life", "校准周期寿命", "weibull",
         "校准寿命统计", {"shape": 2.4, "scale": 1.9}),
    ]
    # 复用批次 1 的族模板（避免重复定义）
    base = gen_multi_domain_entries()
    tmpl = {}
    for e in base:
        fam = e["family"]
        if fam not in tmpl:
            tmpl[fam] = (e["parameters"], e["fingerprint_expectation"],
                         e["equations"], e["channel_template"],
                         e["scan_param"])
    entries = []
    for domain, suffix, name, family, ref, over in specs:
        params, fpe, eq, ct, scan = tmpl[family]
        params = [dict(p) for p in params]
        for p in params:
            if p["name"] in over:
                p["default"] = over[p["name"]]
        entries.append(_base_entry(
            f"{domain}_{suffix}", domain, name, family, ct, params,
            [dict(f) for f in fpe], equations=dict(eq),
            source_ref=f"{ref}；{domain} 域实例（程序化生成，表驱动批次 2）",
            scan_param=scan))
    return entries


def gen_multi_domain_entries_3() -> list[dict]:
    """批次 3 扩域（2026-10-02）：13 领域 × 3 = 39 系统。"""
    specs = [
        ("astro", "star_flux_fluct", "恒星光度乘性波动", "lognormal",
         "测光统计", {"sigma": 0.35}),
        ("astro", "flare_extreme", "耀发强度（上限极值）", "frechet",
         "极值理论", {"shape": 2.0, "scale": 1.5}),
        ("astro", "photon_count_rate", "光子计数率", "poisson_count",
         "光子计数统计", {"rate": 15.0}),
        ("particle", "track_residual_amp", "径迹残差幅值", "rayleigh",
         "径迹统计", {"sigma": 0.3}),
        ("particle", "event_count_rate", "事件计数率", "poisson_count",
         "事例计数", {"rate": 9.0}),
        ("particle", "detector_chi2", "探测器拟合卡方", "chisq",
         "拟合检验", {"k": 10.0}),
        ("cryogenics", "helium_temp_fluct", "液氦温度乘性波动", "lognormal",
         "低温统计", {"sigma": 0.25}),
        ("cryogenics", "quench_life", "失超寿命", "weibull",
         "失超统计", {"shape": 2.2, "scale": 2.0}),
        ("cryogenics", "cryo_leak_count", "低温泄漏计数", "poisson_count",
         "泄漏计数", {"rate": 1.5}),
        ("vacuum", "pressure_amp", "真空压力幅值", "rayleigh",
         "真空统计", {"sigma": 0.8}),
        ("vacuum", "outgassing_life", "放气寿命", "weibull",
         "放气统计", {"shape": 1.4, "scale": 3.1}),
        ("vacuum", "particle_hit_window", "残余气体碰撞窗口（阈值化）",
         "truncated_gaussian", "碰撞统计", {"tau": -1.5}),
        ("printing", "register_error_amp", "套准误差幅值", "rayleigh",
         "套准统计", {"sigma": 0.4}),
        ("printing", "ink_transfer_fraction", "油墨转移率（有界）", "beta",
         "转移统计", {"alpha": 3.0, "beta": 2.0}),
        ("printing", "dot_defect_cluster", "网点缺陷成簇计数", "negbin",
         "缺陷聚集", {"k": 2.2, "p": 0.14}),
        ("glass", "thickness_fluct", "厚度乘性波动", "lognormal",
         "玻璃测厚统计", {"sigma": 0.3}),
        ("glass", "temper_strength_extreme", "钢化强度极值", "frechet",
         "极值理论", {"shape": 2.9, "scale": 1.2}),
        ("glass", "bubble_defect_count", "气泡缺陷计数", "poisson_count",
         "缺陷计数", {"rate": 3.5}),
        ("cement", "composition_amp", "组分幅值波动", "rayleigh",
         "组分统计", {"sigma": 0.9}),
        ("cement", "hydration_life", "水化过程寿命", "weibull",
         "水化统计", {"shape": 1.7, "scale": 2.6}),
        ("cement", "kiln_window", "窑工况窗口（阈值化）",
         "truncated_gaussian", "窑况统计", {"tau": -1.2}),
        ("pulp_paper", "basis_weight_fluct", "定量乘性波动", "lognormal",
         "定量统计", {"sigma": 0.4}),
        ("pulp_paper", "fiber_network_fraction", "纤维交织度（有界）", "beta",
         "交织统计", {"alpha": 2.8, "beta": 1.9}),
        ("pulp_paper", "web_break_life", "断纸间隔寿命", "weibull",
         "断纸统计", {"shape": 1.5, "scale": 2.9}),
        ("hvac", "supply_temp_fluct", "送风温度乘性波动", "lognormal",
         "暖通统计", {"sigma": 0.2}),
        ("hvac", "damper_window", "风阀窗口（阈值化）",
         "truncated_gaussian", "风阀统计", {"tau": -1.0}),
        ("hvac", "filter_load_life", "滤网堵塞寿命", "weibull",
         "滤网寿命", {"shape": 1.8, "scale": 2.5}),
        ("elevator", "leveling_amp", "平层偏差幅值", "rayleigh",
         "平层统计", {"sigma": 0.5}),
        ("elevator", "door_cycle_life", "门机循环寿命", "weibull",
         "门机寿命", {"shape": 2.4, "scale": 2.2}),
        ("elevator", "call_count_rate", "呼梯计数率", "poisson_count",
         "呼梯计数", {"rate": 8.0}),
        ("fire_safety", "smoke_density_amp", "烟密度幅值", "rayleigh",
         "烟气统计", {"sigma": 1.1}),
        ("fire_safety", "sprinkler_activation_life", "喷淋启动寿命", "weibull",
         "启动统计", {"shape": 2.7, "scale": 1.8}),
        ("fire_safety", "alarm_cluster", "报警成簇计数", "negbin",
         "报警聚集", {"k": 2.8, "p": 0.11}),
        ("desalination", "salinity_fluct", "盐度乘性波动", "lognormal",
         "盐度统计", {"sigma": 0.3}),
        ("desalination", "membrane_fouling_life", "膜污染寿命", "weibull",
         "膜寿命统计", {"shape": 1.6, "scale": 2.8}),
        ("desalination", "permeate_fraction", "渗透液比例（有界）", "beta",
         "渗透统计", {"alpha": 3.2, "beta": 1.7}),
        ("smart_grid", "phase_angle_amp", "相角误差幅值", "rayleigh",
         "相角统计", {"sigma": 0.6}),
        ("smart_grid", "fault_count_rate", "故障计数率", "poisson_count",
         "故障计数", {"rate": 2.0}),
        ("smart_grid", "transformer_life", "变压器寿命", "weibull",
         "变压器寿命统计", {"shape": 1.9, "scale": 3.0}),
    ]
    base = gen_multi_domain_entries()
    tmpl = {}
    for e in base:
        fam = e["family"]
        if fam not in tmpl:
            tmpl[fam] = (e["parameters"], e["fingerprint_expectation"],
                         e["equations"], e["channel_template"],
                         e["scan_param"])
    entries = []
    for domain, suffix, name, family, ref, over in specs:
        params, fpe, eq, ct, scan = tmpl[family]
        params = [dict(p) for p in params]
        for p in params:
            if p["name"] in over:
                p["default"] = over[p["name"]]
        entries.append(_base_entry(
            f"{domain}_{suffix}", domain, name, family, ct, params,
            [dict(f) for f in fpe], equations=dict(eq),
            source_ref=f"{ref}；{domain} 域实例（程序化生成，表驱动批次 3）",
            scan_param=scan))
    return entries


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="批量生成 DB1 条目并跑参数扫描管线")
    ap.add_argument("--family", default=None,
                    help="只跑指定 family（ou / diffusion / harmonic_oscillator / "
                         "elastic / radioactive_decay / poisson_count / two_level / "
                         "laplace / lc_circuit / landau / gamma / beta / weibull / "
                         "vonmises / geometric / cauchy / rayleigh / logistic / "
                         "chisq / negbin / lognormal）；"
                         "用于把长任务拆成多次可完成的分块运行")
    ap.add_argument("--reset", action="store_true",
                    help="先清空 DB2（全量首跑时用）；不带则追加")
    ap.add_argument("--skip-existing", action="store_true",
                    help="只处理 DB2 中尚无记录的系统（扩族追加模式，防重复落盘）")
    ap.add_argument("--points", type=int, default=SCAN_POINTS,
                    help=f"每系统扫描点数（默认 {SCAN_POINTS}）")
    ap.add_argument("--domain", default=None,
                    help="只处理指定 domain 的条目（领域实例新增模式，如 plasma）")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    entries = (gen_ou_entries() + gen_diffusion_entries() + gen_gibbs_entries()
               + gen_elastic_entries() + gen_decay_entries() + gen_count_entries()
               + gen_two_level_entries() + gen_laplace_entries()
               + gen_resonator_entries() + gen_landau_entries()
               + gen_gamma_entries() + gen_beta_entries()
               + gen_weibull_entries() + gen_vonmises_entries()
               + gen_geometric_entries() + gen_cauchy_entries()
               + gen_rayleigh_entries() + gen_logistic_entries()
               + gen_chisq_entries() + gen_negbin_entries()
               + gen_lognormal_entries()
               + gen_mix2_entries() + gen_truncnorm_entries()
               + gen_truncgeom_entries() + gen_invgamma_entries()
               + gen_invchisq_entries() + gen_frechet_entries()
               + gen_rician_entries() + gen_wrapnorm_entries()
               + gen_plasma_entries() + gen_plasma_entries_more()
               + gen_geophysics_entries() + gen_biomedical_entries()
               + gen_energy_entries() + gen_agriculture_entries()
               + gen_acoustics_entries() + gen_multi_domain_entries()
               + gen_multi_domain_entries_2()
               + gen_multi_domain_entries_3())
    if args.family:
        entries = [e for e in entries if e["family"] == args.family]
        if not entries:
            print(f"没有 family={args.family} 的条目")
            return 1
    if getattr(args, "domain", None):
        entries = [e for e in entries if e["domain"] == args.domain]
        if not entries:
            print(f"没有 domain={args.domain} 的条目")
            return 1
    if args.skip_existing:
        from db2_writer import load_db2
        existing = ({r["system_id"] for r in load_db2(DB2_PATH)}
                    if DB2_PATH.exists() else set())
        n_before = len(entries)
        entries = [e for e in entries if e["system_id"] not in existing]
        print(f"--skip-existing：{n_before} → {len(entries)} 个系统待处理")

    # 清空 DB2（仅 --reset 时）
    DB2_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.reset and DB2_PATH.exists():
        DB2_PATH.unlink()

    # 步骤 1：先把全部条目定义文件写出（定义与扫描分离，便于并行做复核队列）
    ENTRIES_DIR.mkdir(parents=True, exist_ok=True)
    for e in entries:
        entry_file = ENTRIES_DIR / f"{e['system_id']}.json"
        entry_file.write_text(json.dumps(e, ensure_ascii=False, indent=2), encoding="utf-8")

    # 步骤 2：逐系统 × 参数扫描网格跑管线
    # DB2 写入策略：每系统结束把该系统的全部扫描点一次性追加落盘
    # （实测单条 open/close 追加 15.9ms/条，批量 0.07ms/条，差 238x；
    #   按系统分块既省 I/O，又保证中途中断时已完成系统的数据不丢）
    results = []
    for e in entries:
        # 自证表按条目的桥接常数动态生成（gibbs 有 kT，其余无）
        bridge = [p["name"] for p in e["parameters"] if p.get("is_bridge_constant")]
        attestation = make_attestation(bridge_constants=bridge)
        t_sys = time.time()

        grid = _scan_grid(e, args.points)
        sys_rec = {"system_id": e["system_id"], "family": e["family"],
                   "scan_param": e.get("scan_param"),
                   "n_points": len(grid), "n_passed": 0,
                   "stage": "fingerprint", "errors": []}
        sys_records: list[dict] = []

        for idx, val in enumerate(grid):
            e_pt = copy.deepcopy(e)
            scan_meta = None
            if val is not None:
                sp = e["scan_param"]
                for p in e_pt["parameters"]:
                    if p["name"] == sp:
                        p["default"] = float(val)
                scan_meta = {
                    "scan_param": sp,
                    "scan_index": idx,
                    "scan_value": float(val),
                    "params_used": {p["name"]: p["default"] for p in e_pt["parameters"]},
                }

            # 单扫描点异常隔离，避免一个失败中断整批
            # 注意：不逐条写 DB2（实测单条追加 15.9ms/条 vs 批量 0.07ms/条，差 238x），
            # 改为收集到内存，循环结束后一次性写出。
            try:
                r = ingest(e_pt, attestation, db2_path=None, scan=scan_meta)
                if r["passed"]:
                    sys_rec["n_passed"] += 1
                    sys_records.append(build_db2_record(e_pt, {
                        "fingerprint": r["fingerprint"],
                        "graph": r["graph"],
                        "ib_curve": r["ib_curve"],
                    }, scan=scan_meta))
                else:
                    sys_rec["stage"] = r["stage"]
                    errs = r.get("validation", {}).get("errors", [])
                    if errs:
                        sys_rec["errors"].append(f"[point {idx}] {errs}")
            except Exception as ex:  # noqa: BLE001
                sys_rec["stage"] = "exception"
                sys_rec["errors"].append(f"[point {idx}] {type(ex).__name__}: {ex}")

        results.append(sys_rec)
        # 该系统全部扫描点一次性追加落盘（单次写 ≈0.07ms/条）
        if sys_records:
            with open(DB2_PATH, "a", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(r, ensure_ascii=False)
                                  for r in sys_records) + "\n")
        n_ok = sys_rec["n_passed"]
        elapsed = time.time() - t_sys
        print(f"  {sys_rec['system_id']}: {n_ok}/{sys_rec['n_points']} 点通过 "
              f"({elapsed:.1f}s)", flush=True)

    # 汇总
    n_sys = len(results)
    n_sys_full = sum(1 for r in results if r["n_passed"] == r["n_points"])
    n_points_total = sum(r["n_points"] for r in results)
    n_points_pass = sum(r["n_passed"] for r in results)
    print(f"\n共 {n_sys} 个系统（{n_sys_full} 个全部扫描点通过）："
          f"扫描点 {n_points_pass}/{n_points_total} 通过")

    from collections import Counter
    fam_pass = Counter(r["family"] for r in results if r["n_passed"] == r["n_points"])
    print("各 family 全通过的系统数：", dict(fam_pass))

    failed = [r for r in results if r["n_passed"] != r["n_points"]]
    if failed:
        print("\n未全通过的条目：")
        for r in failed:
            print(f"  ✗ {r['system_id']} ({r['n_passed']}/{r['n_points']}, stage={r['stage']})")
            for er in r["errors"][:3]:
                print(f"      {er}")

    # DB2 已按系统分块落盘，这里只统计累计记录数与系统分布
    from db2_writer import load_db2
    all_db2 = load_db2(DB2_PATH)
    n_records = len(all_db2)
    db2_systems = len({r.get("system_id") for r in all_db2})
    db2_families = Counter(r.get("family") for r in all_db2)
    print(f"\nDB2 累计记录数：{n_records}（来自 {db2_systems} 个系统）")
    print("DB2 各 family 记录数：", dict(db2_families))

    # 写汇总到文件（供后续读取）
    summary = {
        "run": {
            "family_filter": args.family,
            "points_per_system": args.points,
            "systems": n_sys,
            "systems_fully_passed": n_sys_full,
            "scan_points": n_points_total,
            "scan_points_passed": n_points_pass,
            "by_family": dict(fam_pass),
        },
        "db2_cumulative": {
            "records": n_records,
            "systems": db2_systems,
            "by_family": dict(db2_families),
        },
    }
    (_PROJECT_ROOT / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0 if n_sys_full == n_sys else 1


if __name__ == "__main__":
    sys.exit(main())
