# -*- coding: utf-8 -*-
"""
DB3 种子对生成器 v1（34 对黄金种子，Phase 0.5）
------------------------------------------------
一文件一对：db3/pairs/DB3-XXXX.json
依据：cross domain rl architecture plan.md §2.3.2/§2.3.3/§2.3.6/§3.4/§3.9
红线：初始 source="AI+drafted"、human_review_status="pending"、anchor_eligible=false；
      UNMAPPED 对禁止携带 q 向量；R-COND-001 对必带歧义标志；
      R-TOUGH-001 对须标 requires_db5_calibration；桥接常数带单位。
用法：python generate_pairs.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairs")
RULES_VERSION = "0.1.0"
CONVENTION_VERSION = "1.0.0"

# 严格性等级（越大越强；多规则命中取最弱）
STRICTNESS_ORDER = {
    "定理": 4, "定理（马尔可夫系统）": 3, "推导": 2, "推导/类比": 1, "类比": 0.5, "猜想": 0,
}
RULE_STRICTNESS = {
    "R-STIFF-001": "定理", "R-PRECISE-001": "定理",
    "R-FAST-001": "定理（马尔可夫系统）", "R-COND-001": "推导",
    "R-ROBUST-001": "推导", "R-STABLE-001": "推导/类比",
    "R-ADAPT-001": "类比", "R-TOUGH-001": "类比", "R-NOMAP-000": None,
}
RATIONALE_BY_RULE = {
    "R-STIFF-001": [
        "cross domain rl architecture plan.md §2.3.3 R-STIFF-001 [定理]、§2.3.5 例 2（线性弹性谐振近似）",
        "md/information mechanics derivations.md §29 涨落-耗散定理与信息理想气体（约 L1120–1147）：能量涨落-响应对应",
        "md/information mechanics derivations.md §1 定理 1.3（Cramér–Rao）：度规 = 可测量性的界限",
        "勘误记录（2 处）：①arch §2.3.3 桥接式 C_eff=(V/k_B T)·g_λλ 与同节下界 q1≥(k_B T/V)·C_req 互为倒数，按涨落-响应定理标准形式 g_λλ=C·V/(k_B T)（即 C=(k_B T/V)·g）取值，与 DB1 elastic/harmonic 族解析指纹 g=k/(k_B T) 一致；②rules.yaml R-STIFF-001 q_update.expr 初版误沿 arch 错误方向 k_B T·C_req/V（量纲 J²/m⁶ 非无量纲），经 A 角色复核发现，已修正为 C_req·V/(k_B T)（量纲自洽）。详见 db3/rules/rules.yaml R-STIFF-001 bridge_law.note",
    ],
    "R-PRECISE-001": [
        "cross domain rl architecture plan.md §2.3.3 R-PRECISE-001 [定理]",
        "md/information mechanics derivations.md §1 定理 1.3 Cramér–Rao 下界（约 L73–97）：Cov[θ̂] ⪰ g^{-1}",
    ],
    "R-FAST-001": [
        "cross domain rl architecture plan.md §2.3.3 R-FAST-001 [定理，马尔可夫系统]、§2.3.5 例 3（OU 过程：λ2=θ=MI 衰减速率）",
        "md/information mechanics derivations.md B.7 稳定性分析（弛豫与谱）",
    ],
    "R-COND-001": [
        "cross domain rl architecture plan.md §2.3.3 R-COND-001 [推导]（二义性规则）、§2.3.5 例 1（一维热传导：逐模式 MI 与 DPI 裕度）",
        "md/information mechanics derivations.md §9 锥结构与类光方程（q7=c_info 的 Fisher 锥依据）",
    ],
    "R-ROBUST-001": [
        "cross domain rl architecture plan.md §2.3.3 R-ROBUST-001 [推导]（方向性分离）",
        "md/information mechanics derivations.md A.7 信息论四件套（DPI）、§27 第二定律=DPI（q4 裕度依据）",
        "md/physical to ai.md 对应四：FHN 拓扑保护↔对抗鲁棒性（q10 受保护信息位依据）",
    ],
    "R-STABLE-001": [
        "cross domain rl architecture plan.md §2.3.3 R-STABLE-001 [推导/类比]",
        "md/physical to ai.md 对应二：FHN 临界现象↔信息传播相变（临界点距离，ξ∝|I−I_c|^{−ν}）",
        "md/information mechanics derivations.md B.7 稳定性分析、§29（Landau 双阱与热力学对应）",
    ],
    "R-ADAPT-001": [
        "cross domain rl architecture plan.md §2.3.3 R-ADAPT-001 [类比]",
        "md/ai to physical.md 对应二：信道容量的双下降定理（容量-自由度关系）",
        "md/Cover_Thomas_入门总结.md 第五章 信道容量（分辨 M 态需 ln M）",
    ],
    "R-TOUGH-001": [
        "cross domain rl architecture plan.md §2.3.3 R-TOUGH-001 [类比]：须 DB5 校准后才可入 B v1 训练集",
        "md/physical to ai.md 总论表第 7 条：锐利/平坦极小值↔自旋玻璃自由能景观",
        "md/information mechanics derivations.md B.7 稳定性分析（能垒与多稳态）",
    ],
    "R-NOMAP-000": [
        "cross domain rl architecture plan.md §2.3.3 R-NOMAP-000：规则库未命中 → UNMAPPED+最近邻规则建议，禁止硬映射，转入人工队列",
        "cross domain rl architecture plan.md §2.3.8 第 5 条（规则覆盖度）：v0 只保证热/扩散、弹性、平衡态统计三类种子领域",
    ],
}

PAIRS = [
    # ============ R-STIFF-001 ×4 ============
    dict(id=1, domain="mechanics", lang="zh",
         text="弹簧要足够硬：施加 1 N 载荷时变形不超过 10 mm。",
         prop="stiffness", dir="bound_below", target={"stiffness_N_per_m": 100},
         rv_hint="microstate_gibbs", constraints=["T_K = 300"], flags=[],
         rules=["R-STIFF-001"],
         q={"q1_fisher_directional": dict(
             direction="lower_bound", expr="K_req/(k_B T)",
             note="λ=位移方向；K_req=C·V 为总刚度（V 吸收进 C·V）")},
         we=dict(
             setup="室温弹簧，刚度 K=100 N/m（1 N / 10 mm）",
             given={"K_req": {"value": 100, "unit": "N/m", "is_bridge_constant": True},
                    "k_B_T": {"value": 4.14e-21, "unit": "J", "is_bridge_constant": True},
                    "T": {"value": 300, "unit": "K", "is_bridge_constant": False}},
             computation="q1 ≥ K_req/(k_B T) = 100/4.14e-21",
             result="q1 ≥ 2.4e22 m^-2（位移方向 Fisher 下界）",
             reference_scale="位移方向 q1 参考尺度：室温 N/m 级刚度约 1e20–1e24 m^-2")),
    dict(id=2, domain="mechanics", lang="en",
         text="The cantilever beam must have an effective tip stiffness of at least 5 kN/m at room temperature.",
         prop="stiffness", dir="bound_below", target={"stiffness_N_per_m": 5000},
         rv_hint="microstate_gibbs", constraints=["T_K = 300"], flags=[],
         rules=["R-STIFF-001"],
         q={"q1_fisher_directional": dict(
             direction="lower_bound", expr="K_req/(k_B T)",
             note="λ=tip 位移方向；K_req=C·V 总刚度")},
         we=dict(
             setup="悬臂梁，尖端等效刚度 5 kN/m，T=300 K",
             given={"K_req": {"value": 5000, "unit": "N/m", "is_bridge_constant": True},
                    "k_B_T": {"value": 4.14e-21, "unit": "J", "is_bridge_constant": True}},
             computation="q1 ≥ 5000/4.14e-21",
             result="q1 ≥ 1.2e24 m^-2",
             reference_scale="位移方向 q1 参考尺度：室温 N/m 级刚度约 1e20–1e24 m^-2")),
    dict(id=3, domain="materials", lang="zh",
         text="结构件用钢的弹性模量不得低于 200 GPa（室温，考察体积取 1 mm³）。",
         prop="stiffness", dir="bound_below", target={"modulus_Pa": 2.0e11},
         rv_hint="microstate_gibbs", constraints=["T_K = 300", "V_m3 = 1e-9"], flags=[],
         rules=["R-STIFF-001"],
         q={"q1_fisher_directional": dict(
             direction="lower_bound", expr="C_req · V / (k_B T)",
             note="λ=应变方向（无量纲），q1 无量纲；V 进分子（自平均）")},
         we=dict(
             setup="钢试样，C=200 GPa，V=1 mm³，T=300 K",
             given={"C_req": {"value": 2.0e11, "unit": "Pa", "is_bridge_constant": True},
                    "V": {"value": 1e-9, "unit": "m^3", "is_bridge_constant": True},
                    "k_B_T": {"value": 4.14e-21, "unit": "J", "is_bridge_constant": True}},
             computation="q1 ≥ C·V/(k_B T) = 2e11 × 1e-9 / 4.14e-21 = 4.8e22",
             result="q1 ≥ 4.8e22（无量纲，1/应变²）",
             reference_scale="应变方向 q1 参考尺度：室温下 GPa 级模量、mm³ 级体积约 1e19–1e22；软物质（kPa–MPa）约 1e13–1e17；体积越小 q1 越小（自平均减弱）")),
    dict(id=4, domain="materials", lang="zh",
         text="类金刚石薄膜硬度要高：压入方向的等效弹性模量不低于 400 GPa（厚度 1 μm，作用体积取 1 μm³）。",
         prop="stiffness", dir="bound_below", target={"equivalent_modulus_Pa": 4.0e11},
         rv_hint="microstate_gibbs", constraints=["T_K = 300", "V_m3 = 1e-18"], flags=[],
         rules=["R-STIFF-001"],
         q={"q1_fisher_directional": dict(
             direction="lower_bound", expr="C_req · V / (k_B T)")},
         we=dict(
             setup="DLC 薄膜，C=400 GPa，V=1 μm³，T=300 K",
             given={"C_req": {"value": 4.0e11, "unit": "Pa", "is_bridge_constant": True},
                    "V": {"value": 1e-18, "unit": "m^3", "is_bridge_constant": True},
                    "k_B_T": {"value": 4.14e-21, "unit": "J", "is_bridge_constant": True}},
             computation="q1 ≥ C·V/(k_B T) = 4e11 × 1e-18 / 4.14e-21 = 9.7e13",
             result="q1 ≥ 9.7e13（无量纲，1/应变²）",
             reference_scale="微米级作用体积下硬质膜 q1 约 1e12–1e14；体积越小 q1 越小（V 进分子，热涨落自平均减弱）")),
    # ============ R-PRECISE-001 ×4 ============
    dict(id=5, domain="measurement", lang="zh",
         text="温度传感器经 16 次独立采样平均后，温度估计标准差不超过 0.05 K。",
         prop="precision", dir="bound_below", target={"sigma_K": 0.05, "n_samples_count": 16},
         rv_hint="iid_samples", constraints=[], flags=[],
         rules=["R-PRECISE-001"],
         q={"q1_fisher_directional": dict(
             direction="lower_bound", expr="1/(n · sigma_req^2)",
             note="θ=温度方向，n=独立采样数")},
         we=dict(
             setup="电阻温度计，σ_req=0.05 K，n=16",
             given={"sigma_req": {"value": 0.05, "unit": "K", "is_bridge_constant": True},
                    "n": 16},
             computation="q1 ≥ 1/(16 × 0.05²) = 1/0.04",
             result="q1 ≥ 25 K^-2",
             reference_scale="温度方向 q1 参考尺度：实验室级 1e0–1e2 K^-2；计量级可到 1e4")),
    dict(id=6, domain="control", lang="zh",
         text="恒温槽控制精度：稳态温度波动的标准差不超过 0.01 K（以单次读数判定）。",
         prop="precision", dir="bound_below", target={"sigma_K": 0.01, "n_samples_count": 1},
         rv_hint="iid_samples", constraints=[], flags=[],
         rules=["R-PRECISE-001"],
         q={"q1_fisher_directional": dict(
             direction="lower_bound", expr="1/(n · sigma_req^2)")},
         we=dict(
             setup="恒温槽，σ_req=0.01 K，n=1（连续控制回路折算单次判定）",
             given={"sigma_req": {"value": 0.01, "unit": "K", "is_bridge_constant": True},
                    "n": 1},
             computation="q1 ≥ 1/(1 × 1e-4)",
             result="q1 ≥ 1e4 K^-2",
             reference_scale="控制方向 q1 参考尺度：精密恒温 1e3–1e5 K^-2")),
    dict(id=7, domain="measurement", lang="en",
         text="A strain gauge must resolve a 10-microstrain change with a single reading (sigma_req = 1e-5).",
         prop="precision", dir="bound_below", target={"sigma_dimensionless_fraction": 1e-5, "n_samples_count": 1},
         rv_hint="iid_samples", constraints=[], flags=[],
         rules=["R-PRECISE-001"],
         q={"q1_fisher_directional": dict(
             direction="lower_bound", expr="1/(n · sigma_req^2)",
             note="θ=应变（无量纲）方向")},
         we=dict(
             setup="应变片单次读数，σ_req=10 με=1e-5",
             given={"sigma_req": {"value": 1e-5, "unit": "dimensionless_fraction", "is_bridge_constant": True},
                    "n": 1},
             computation="q1 ≥ 1/(1 × 1e-10)",
             result="q1 ≥ 1e10（1/应变²）",
             reference_scale="应变方向 q1 参考尺度：精密应变测量 1e8–1e12")),
    dict(id=8, domain="measurement", lang="zh",
         text="测厚仪单次测量的厚度标准差不超过 1 μm。",
         prop="precision", dir="bound_below", target={"sigma_m": 1e-6, "n_samples_count": 1},
         rv_hint="iid_samples", constraints=[], flags=[],
         rules=["R-PRECISE-001"],
         q={"q1_fisher_directional": dict(
             direction="lower_bound", expr="1/(n · sigma_req^2)",
             note="θ=厚度方向")},
         we=dict(
             setup="超声测厚，σ_req=1e-6 m，n=1",
             given={"sigma_req": {"value": 1e-6, "unit": "m", "is_bridge_constant": True},
                    "n": 1},
             computation="q1 ≥ 1/(1 × 1e-12)",
             result="q1 ≥ 1e12 m^-2",
             reference_scale="长度方向 q1 参考尺度：微米级精密测量 1e10–1e14 m^-2")),
    # ============ R-FAST-001 ×4 ============
    dict(id=9, domain="control", lang="zh",
         text="执行器从阶跃指令到 95% 行程的响应时间不超过 20 ms。",
         prop="response_speed", dir="bound_below", target={"settling_time_s": 0.02, "settling_band": "95_percent"},
         rv_hint="markov_chain", constraints=[], flags=[],
         rules=["R-FAST-001"],
         q={"q9_spectral_gap": dict(
             direction="lower_bound", expr="3/t_settle（95% 建立 ≈ 3τ，一阶系统）",
             note="τ=1/λ2，Fokker–Planck 生成元第二特征值（Itô）；建立时间须经带宽修正换算为 τ，不得直接当 τ 用")},
         we=dict(
             setup="机电执行器，95% 行程时间 20 ms",
             given={"t_settle": {"value": 0.02, "unit": "s", "is_bridge_constant": True}},
             computation="τ ≈ t_settle/3 = 6.7e-3 s；q9 ≥ 3/t_settle = 3/0.02",
             result="q9 ≥ 150 s^-1",
             reference_scale="q9 参考尺度：机电执行器 1e1–1e3 s^-1；光电探测 1e6–1e10 s^-1；热系统 1e-2–1 s^-1")),
    dict(id=10, domain="measurement", lang="en",
         text="The photodetector response time must be under 10 ns.",
         prop="response_speed", dir="bound_below", target={"tau_s": 1e-8},
         rv_hint="markov_chain", constraints=[], flags=[],
         rules=["R-FAST-001"],
         q={"q9_spectral_gap": dict(direction="lower_bound", expr="1/tau_req")},
         we=dict(
             setup="光电二极管探测链，τ_req=10 ns",
             given={"tau_req": {"value": 1e-8, "unit": "s", "is_bridge_constant": True}},
             computation="q9 ≥ 1/1e-8",
             result="q9 ≥ 1e8 s^-1",
             reference_scale="q9 参考尺度：光电探测 1e6–1e10 s^-1")),
    dict(id=11, domain="thermal", lang="zh",
         text="炉温受扰动后须在 30 s 内回到设定值的 ±1% 带内。",
         prop="response_speed", dir="bound_below", target={"settling_time_s": 30, "settling_band": "pm_1_percent"},
         rv_hint="markov_chain", constraints=["disturbance_amplitude = 5 percent"], flags=[],
         rules=["R-FAST-001"],
         q={"q9_spectral_gap": dict(
             direction="lower_bound", expr="ln(100)/t_settle ≈ 4.6/t_settle（±1% 带内 ≈ 4.6τ）")},
         we=dict(
             setup="电炉热系统，±1% 回带时间 30 s",
             given={"t_settle": {"value": 30, "unit": "s", "is_bridge_constant": True}},
             computation="τ ≈ t_settle/ln(100) = 30/4.6 = 6.5 s；q9 ≥ 4.6/30",
             result="q9 ≥ 0.154 s^-1",
             reference_scale="q9 参考尺度：热系统 1e-2–1 s^-1")),
    dict(id=12, domain="electronics", lang="zh",
         text="RC 低通滤波器到 2% 建立时间不超过 1 ms。",
         prop="response_speed", dir="bound_below", target={"settling_time_s": 1e-3},
         rv_hint="markov_chain", constraints=["settling_threshold = 0.02"], flags=[],
         rules=["R-FAST-001"],
         q={"q9_spectral_gap": dict(
             direction="lower_bound", expr="(-ln(0.02))/t_settle ≈ 3.9/t_settle",
             note="2% 建立时间 t_settle ≈ 3.9·τ")},
         we=dict(
             setup="RC 一阶链，2% 建立时间 1 ms",
             given={"t_settle": {"value": 1e-3, "unit": "s", "is_bridge_constant": True}},
             computation="τ ≈ t_settle/3.9 = 2.56e-4 s；q9 ≥ 1/τ ≈ 3.9/1e-3",
             result="q9 ≥ 3.9e3 s^-1",
             reference_scale="q9 参考尺度：电路瞬态 1e3–1e8 s^-1")),
    # ============ R-COND-001 ×3（全部带歧义标志） ============
    dict(id=13, domain="thermal", lang="zh",
         text="散热片要把 5 mm 厚铜层快速导通：热锋在 5 s 内传过。",
         prop="transport", dir="bound_below", target={"kappa_m2_per_s": 5e-6, "length_scale_m": 5e-3, "front_time_s": 5},
         rv_hint="field_time_slices", constraints=["material = copper"], 
         flags=["R-COND-001 二义性：文本未显式区分读向；缺省取读向(a) 热锋传播快（κ↑，q7↑），记录标志，禁止静默选择",
                "若本意为细结构保真（读向 b），则 q6 上界方向相反，须人工消解后改判"],
         rules=["R-COND-001"],
         q={"q7_info_propagation_speed": dict(
             direction="lower_bound", expr="kappa_req/L", reading="a_heat_front_fast",
             note="扩散锋表观速度 v≈κ/L；τ≈L²/κ")},
         we=dict(
             setup="铜散热层 L=5 mm，热锋 5 s 传过（读向 a，缺省）",
             given={"L": {"value": 5e-3, "unit": "m", "is_bridge_constant": True},
                    "kappa": {"value": 5e-6, "unit": "m^2/s", "is_bridge_constant": True},
                    "front_time": {"value": 5, "unit": "s", "is_bridge_constant": True}},
             computation="κ_req = L²/τ = (5e-3)²/5 = 5e-6 m²/s；q7 ≥ κ/L = 5e-6/5e-3",
             result="q7 ≥ 1e-3 m/s",
             reference_scale="q7 参考尺度：宏观热锋（良导体）1e-3–1e-1 m/s")),
    dict(id=14, domain="thermal", lang="zh",
         text="隔热层要保住 1 mm 尺度的温度图样：1 s 观测窗内热模糊不超过该尺度。",
         prop="transport", dir="bound_above", target={"ell_req_m": 1e-3, "t_obs_s": 1},
         rv_hint="field_time_slices", constraints=["material = thermal_insulator"],
         flags=["R-COND-001 二义性：文本'保住温度图样'显式指向读向(b) 细结构保真（κ↓，q6 上界）；与读向(a) 方向相反，已记录，未静默选择"],
         rules=["R-COND-001"],
         q={"q6_correlation_length": dict(
             direction="upper_bound", expr="ell_req", reading="b_fine_structure_fidelity",
             note="热模糊关联长度 ξ(t)=√(2κt) ≤ ℓ_req，⟺ κ ≤ ℓ_req²/(2·t_obs)；严格半衰期式 κ ≤ ln2/(2k²·t_obs)，k=2π/ℓ_req")},
         we=dict(
             setup="隔热层，ℓ_req=1 mm，t_obs=1 s（读向 b）",
             given={"ell_req": {"value": 1e-3, "unit": "m", "is_bridge_constant": True},
                    "t_obs": {"value": 1, "unit": "s", "is_bridge_constant": True},
                    "kappa": {"value": 5e-7, "unit": "m^2/s", "is_bridge_constant": True}},
             computation="κ ≤ ℓ²/(2t) = (1e-3)²/2 = 5e-7 m²/s；ξ(1s)=√(2×5e-7×1)=1e-3 m ✓",
             result="q6 ≤ 1e-3 m（κ ≤ 5e-7 m²/s）",
             reference_scale="q6 参考尺度：隔热/热像应用 1e-4–1e-2 m")),
    dict(id=15, domain="electronics", lang="zh",
         text="功率芯片要把局部热点在 1 ms 内拉平到 100 μm 尺度以下。",
         prop="transport", dir="bound_below", target={"kappa_m2_per_s": 1e-5, "length_scale_m": 1e-4, "front_time_s": 1e-3},
         rv_hint="field_time_slices", constraints=["material = silicon"],
         flags=["R-COND-001 二义性：文本'把局部热点拉平'显式指向读向(a) 热锋快（κ↑，q7↑）；显式消解，仍记录标志备查"],
         rules=["R-COND-001"],
         q={"q7_info_propagation_speed": dict(
             direction="lower_bound", expr="kappa_req/L", reading="a_heat_front_fast")},
         we=dict(
             setup="硅芯片热点拉平，L=100 μm，τ=1 ms（读向 a，显式）",
             given={"L": {"value": 1e-4, "unit": "m", "is_bridge_constant": True},
                    "kappa": {"value": 1e-5, "unit": "m^2/s", "is_bridge_constant": True},
                    "front_time": {"value": 1e-3, "unit": "s", "is_bridge_constant": True}},
             computation="κ_req = L²/τ = (1e-4)²/1e-3 = 1e-5 m²/s（硅 κ≈0.9e-4 m²/s，余量约 9 倍，可行）；q7 ≥ κ/L = 1e-5/1e-4",
             result="q7 ≥ 0.1 m/s",
             reference_scale="q7 参考尺度：芯片尺度热输运 1e-2–1 m/s")),
    # ============ R-ROBUST-001 ×4 ============
    dict(id=16, domain="electronics", lang="zh",
         text="电路在 0–50 °C 温漂与电磁干扰下，增益估计标准差仍不超过 1%。",
         prop="robustness", dir="bound_below", target={"sigma_dimensionless_fraction": 0.01},
         rv_hint="joint_signal_noise_family", constraints=["T_range_K = [273, 323]"], flags=[],
         rules=["R-ROBUST-001"],
         q={"q1_fisher_directional": dict(direction="lower_bound", expr="1/(n·sigma_req^2)", note="信号 θ=增益方向"),
            "q4_dpi_margin": dict(direction="lower_bound", expr="delta_dpi = 0.5 nats"),
            "q10_redundancy": dict(direction="increase", expr=">= ln(2)（至少 1 bit 受保护信息）"),
            "q1_noise_direction": dict(direction="upper_bound", expr="lambda_max(g_eta_eta) <= epsilon = 0.01 K^-2", note="干扰 η=温漂方向 Fisher 上界")},
         we=dict(
             setup="放大电路：信号=增益（无量纲），干扰=温度漂移",
             given={"sigma_req": {"value": 0.01, "unit": "dimensionless_fraction", "is_bridge_constant": True},
                    "epsilon": {"value": 0.01, "unit": "1/K^2", "is_bridge_constant": True},
                    "delta_dpi": {"value": 0.5, "unit": "nats", "is_bridge_constant": True},
                    "n": 1},
             computation="信号方向 q1 ≥ 1/(1×0.01²)=1e4；干扰方向 λ_max(g_ηη) ≤ 0.01 K^-2；q4 ≥ 0.5 nats；q10 ≥ ln2 ≈ 0.693 nats",
             result="q1 ≥ 1e4；λ_max ≤ 0.01 K^-2；q4 ≥ 0.5 nats；q10 ≥ 0.693 nats",
             reference_scale="方向性分离参考：干扰方向 ε 典型 1e-3–1e-1（参数单位平方倒数）；q4 裕度典型 0.1–1 nats")),
    dict(id=17, domain="communication", lang="en",
         text="The communication link must keep the coded-symbol estimate at sigma_req = 1e-3 despite channel fading.",
         prop="robustness", dir="bound_below", target={"sigma_dimensionless_fraction": 1e-3},
         rv_hint="joint_signal_noise_family", constraints=["fading_model = rayleigh"], flags=[],
         rules=["R-ROBUST-001"],
         q={"q1_fisher_directional": dict(direction="lower_bound", expr="1/(n·sigma_req^2)", note="信号 θ=符号方向"),
            "q4_dpi_margin": dict(direction="lower_bound", expr="delta_dpi = 1 nat"),
            "q10_redundancy": dict(direction="increase", expr=">= ln(2)", note="拓扑/对称性保护编码"),
            "q1_noise_direction": dict(direction="upper_bound", expr="lambda_max(g_eta_eta) <= epsilon", note="干扰 η=信道衰落方向")},
         we=dict(
             setup="编码通信链路：信号=编码符号，干扰=瑞利衰落",
             given={"sigma_req": {"value": 1e-3, "unit": "dimensionless_fraction", "is_bridge_constant": True},
                    "epsilon": {"value": 1e-2, "unit": "1/dimensionless^2", "is_bridge_constant": True},
                    "delta_dpi": {"value": 1, "unit": "nats", "is_bridge_constant": True},
                    "n": 1},
             computation="q1 ≥ 1/(1e-3)² = 1e6；λ_max(g_ηη) ≤ 1e-2；q4 ≥ 1 nat；q10 ≥ ln2 nats",
             result="q1 ≥ 1e6；λ_max ≤ 1e-2；q4 ≥ 1 nat；q10 ≥ 0.693 nats",
             reference_scale="通信方向 q1 参考尺度：高可靠链路 1e4–1e8")),
    dict(id=18, domain="measurement", lang="zh",
         text="现场流量传感器在振动与温漂双干扰下，流量估计标准差不超过 2%。",
         prop="robustness", dir="bound_below", target={"sigma_dimensionless_fraction": 0.02},
         rv_hint="joint_signal_noise_family", constraints=["vibration = 1_g_rms"], flags=[],
         rules=["R-ROBUST-001"],
         q={"q1_fisher_directional": dict(direction="lower_bound", expr="1/(n·sigma_req^2)", note="信号 θ=流量方向"),
            "q4_dpi_margin": dict(direction="lower_bound", expr="delta_dpi = 0.5 nats"),
            "q10_redundancy": dict(direction="increase", expr=">= ln(2)"),
            "q1_noise_direction": dict(direction="upper_bound", expr="lambda_max(g_eta_eta) <= epsilon", note="干扰 η=振动+温漂方向")},
         we=dict(
             setup="工业流量计：信号=流量，干扰=振动+温漂",
             given={"sigma_req": {"value": 0.02, "unit": "dimensionless_fraction", "is_bridge_constant": True},
                    "epsilon": {"value": 0.05, "unit": "1/K^2", "is_bridge_constant": True},
                    "delta_dpi": {"value": 0.5, "unit": "nats", "is_bridge_constant": True},
                    "n": 1},
             computation="q1 ≥ 1/0.02² = 2500；λ_max ≤ 0.05 K^-2；q4 ≥ 0.5 nats；q10 ≥ ln2 nats",
             result="q1 ≥ 2500；λ_max ≤ 0.05 K^-2；q4 ≥ 0.5 nats；q10 ≥ 0.693 nats",
             reference_scale="工业现场 q1 参考尺度：1e3–1e5")),
    dict(id=19, domain="mechanics", lang="en",
         text="The optical bench must keep mirror alignment within sigma_req = 1e-6 rad under floor vibration.",
         prop="robustness", dir="bound_below", target={"sigma_rad": 1e-6},
         rv_hint="joint_signal_noise_family", constraints=["floor_vibration = 1e-4_m_rms"], flags=[],
         rules=["R-ROBUST-001"],
         q={"q1_fisher_directional": dict(direction="lower_bound", expr="1/(n·sigma_req^2)", note="信号 θ=镜面角度方向"),
            "q4_dpi_margin": dict(direction="lower_bound", expr="delta_dpi = 0.5 nats"),
            "q10_redundancy": dict(direction="increase", expr=">= ln(2)"),
            "q1_noise_direction": dict(direction="upper_bound", expr="lambda_max(g_eta_eta) <= epsilon", note="干扰 η=地基振动方向")},
         we=dict(
             setup="光学平台：信号=镜面角度，干扰=地基振动",
             given={"sigma_req": {"value": 1e-6, "unit": "rad", "is_bridge_constant": True},
                    "epsilon": {"value": 1e-2, "unit": "1/m^2", "is_bridge_constant": True},
                    "delta_dpi": {"value": 0.5, "unit": "nats", "is_bridge_constant": True},
                    "n": 1},
             computation="q1 ≥ 1/(1e-6)² = 1e12 rad^-2；λ_max(g_ηη) ≤ 1e-2 m^-2；q4 ≥ 0.5 nats；q10 ≥ ln2 nats",
             result="q1 ≥ 1e12 rad^-2；λ_max ≤ 1e-2 m^-2；q4 ≥ 0.5 nats；q10 ≥ 0.693 nats",
             reference_scale="精密光机 q1 参考尺度：1e10–1e14 rad^-2")),
    # ============ R-STABLE-001 ×3 ============
    dict(id=20, domain="materials", lang="zh",
         text="高温合金部件工作点须距 γ' 溶解线有足够裕度：约化温度距离不低于 0.15。",
         prop="stability", dir="bound_below", target={"delta_dimensionless": 0.15, "T_c_K": 1273, "T_work_K": 1080},
         rv_hint="landau_order_parameter", constraints=["T_c_K = 1273"], flags=[],
         rules=["R-STABLE-001"],
         q={"q8_stability_margin": dict(
             direction="lower_bound", expr="(T_c - T_work)/T_c = (beta_c - beta)/beta_c",
             note="距相变点约化距离；或等价地自由能垒判据")},
         we=dict(
             setup="镍基高温合金，T_c=1273 K（γ' 溶解），工作 T=1080 K",
             given={"T_c": {"value": 1273, "unit": "K", "is_bridge_constant": True},
                    "T_work": {"value": 1080, "unit": "K", "is_bridge_constant": True}},
             computation="q8 = (1273−1080)/1273 = 193/1273",
             result="q8 = 0.152 ≥ 0.15 ✓",
             reference_scale="q8 参考尺度：工程相稳定裕度 0.1–0.5；临界器件（存储）能垒判据 20–60 k_B T")),
    dict(id=21, domain="materials", lang="en",
         text="The shape-memory alloy actuator must operate with reduced distance at least delta = 0.2 from the order-disorder transition, equivalently a free-energy barrier of at least 3 nats.",
         prop="stability", dir="bound_below", target={"delta_dimensionless": 0.2, "E_req_nats": 3},
         rv_hint="landau_order_parameter", constraints=["kT = 1（信息单位）"], flags=[],
         rules=["R-STABLE-001"],
         q={"q8_stability_margin": dict(
             direction="lower_bound", expr="delta 或 E_req（k_B T=1 单位，nats）",
             note="Landau 双阱：ΔF‡ = a²/(4·b·k_B T)")},
         we=dict(
             setup="记忆合金（Landau 参数 a=4, b=1, k_B T=1）",
             given={"a": {"value": 4, "unit": "dimensionless", "is_bridge_constant": False},
                    "b": {"value": 1, "unit": "dimensionless", "is_bridge_constant": False},
                    "k_B_T": {"value": 1, "unit": "nats", "is_bridge_constant": True},
                    "E_req": {"value": 3, "unit": "nats", "is_bridge_constant": True}},
             computation="ΔF‡ = a²/(4·b·k_B T) = 16/4 = 4 nats",
             result="ΔF‡ = 4 nats ≥ 3 nats ✓（q8 判据通过）",
             reference_scale="能垒判据参考尺度：工程稳定 1–5 nats（k_B T=1 单位）；长寿命存储 40+ nats")),
    dict(id=22, domain="materials", lang="zh",
         text="磁存储位在工作温度下 10 年内不得热退磁：保持自由能垒不低于 40 k_B T。",
         prop="stability", dir="bound_below", target={"E_req_nats": 40},
         rv_hint="landau_order_parameter", constraints=["lifetime_years = 10"], flags=[],
         rules=["R-STABLE-001"],
         q={"q8_stability_margin": dict(
             direction="lower_bound", expr="E_req = 40 nats（k_B T=1 单位）",
             note="ΔF‡/k_B T ≥ 40 为磁存储行业惯例下限")},
         we=dict(
             setup="磁记录位（Landau a=20, b=1, k_B T=1）",
             given={"a": {"value": 20, "unit": "dimensionless", "is_bridge_constant": False},
                    "b": {"value": 1, "unit": "dimensionless", "is_bridge_constant": False},
                    "k_B_T": {"value": 1, "unit": "nats", "is_bridge_constant": True},
                    "E_req": {"value": 40, "unit": "nats", "is_bridge_constant": True}},
             computation="ΔF‡ = a²/(4·b·k_B T) = 400/4 = 100 nats",
             result="ΔF‡ = 100 nats ≥ 40 nats ✓",
             reference_scale="磁存储 q8 参考尺度：40–60 nats（k_B T=1 单位）为行业稳定线")),
    # ============ R-ADAPT-001 ×3 ============
    dict(id=23, domain="measurement", lang="zh",
         text="环境传感器要在 4 种工况（冬夏室内、室外、舱内）间自适应切换且不误判。",
         prop="adaptability", dir="bound_below", target={"n_environments_count": 4},
         rv_hint="environment_channel", constraints=[], flags=[],
         rules=["R-ADAPT-001"],
         q={"q5_channel_capacity": dict(direction="lower_bound", expr="ln(M)", note="M=工况数"),
            "q2_posterior_sharpness": dict(direction="upper_bound", expr="0（det g ≤ 1，平坦极小）")},
         we=dict(
             setup="多工况传感器，M=4",
             given={"M": 4, "C_req": {"value": 1.386, "unit": "nats", "is_bridge_constant": True}},
             computation="C_req = ln 4 = 1.386 nats；q5 ≥ 1.386；q2 ≤ 0",
             result="q5 ≥ 1.386 nats；q2 ≤ 0（后验宽容度大）",
             reference_scale="q5 参考尺度：多工况分类 1–10 nats")),
    dict(id=24, domain="control", lang="en",
         text="The legged-robot controller must distinguish and adapt across at least 8 terrain classes while keeping a flat minimum over gait parameters.",
         prop="adaptability", dir="bound_below", target={"n_environments_count": 8},
         rv_hint="environment_channel", constraints=[], flags=[],
         rules=["R-ADAPT-001"],
         q={"q5_channel_capacity": dict(direction="lower_bound", expr="ln(M)"),
            "q2_posterior_sharpness": dict(direction="upper_bound", expr="0（det g ≤ 1）")},
         we=dict(
             setup="足式机器人地形适应，M=8",
             given={"M": 8, "C_req": {"value": 2.08, "unit": "nats", "is_bridge_constant": True}},
             computation="C_req = ln 8 = 2.08 nats",
             result="q5 ≥ 2.08 nats；q2 ≤ 0",
             reference_scale="q5 参考尺度：多地形适应 2–5 nats")),
    dict(id=25, domain="electronics", lang="zh",
         text="自适应均衡器要覆盖 16 种信道状态，且参数后验保持平坦（宽容度大）。",
         prop="adaptability", dir="bound_below", target={"n_environments_count": 16},
         rv_hint="environment_channel", constraints=[], flags=[],
         rules=["R-ADAPT-001"],
         q={"q5_channel_capacity": dict(direction="lower_bound", expr="ln(M)"),
            "q2_posterior_sharpness": dict(direction="upper_bound", expr="0（det g ≤ 1）")},
         we=dict(
             setup="通信自适应均衡，M=16",
             given={"M": 16, "C_req": {"value": 2.77, "unit": "nats", "is_bridge_constant": True}},
             computation="C_req = ln 16 = 2.77 nats",
             result="q5 ≥ 2.77 nats；q2 ≤ 0",
             reference_scale="q5 参考尺度：信道自适应 2–6 nats")),
    # ============ R-TOUGH-001 ×2（须 DB5 校准） ============
    dict(id=26, domain="materials", lang="zh",
         text="复合材料层合板断裂韧性要高：裂纹扩展前单位路径自由能垒不低于 5 nats，且有至少 3 个亚稳能量谷（多谷增韧）。",
         prop="toughness", dir="bound_below", target={"E_req_nats": 5, "n_valleys_count": 3},
         rv_hint="fracture_path_landscape", constraints=["requires_db5_calibration = true"], flags=[],
         rules=["R-TOUGH-001"], tough=True,
         q={"q8_stability_margin": dict(direction="lower_bound", expr="E_req"),
            "q10_redundancy": dict(direction="increase", expr="亚稳谷数目 ≥ 3（多谷结构）")},
         we=dict(
             setup="碳纤维层合板（桥联+微裂纹机制：主谷+2 个亚稳谷）",
             given={"E_req": {"value": 5, "unit": "nats", "is_bridge_constant": True},
                    "n_valleys": {"value": 3, "unit": "count", "is_bridge_constant": False},
                    "Delta_F_ddagger": {"value": 6, "unit": "nats", "is_bridge_constant": False}},
             computation="ΔF‡ = 6 nats ≥ 5 ✓；谷数 3 ≥ 3 ✓",
             result="q8 ≥ 5 nats；多谷 3 个",
             reference_scale="类比级参考：韧性材料能垒 3–10 nats（k_B T=1 单位）；绝对值须 DB5 校准")),
    dict(id=27, domain="mechanics", lang="zh",
         text="头盔缓冲层要耐冲击：吸能路径至少有 3 个亚稳塌缩模式，能垒不低于 4 nats。",
         prop="toughness", dir="bound_below", target={"E_req_nats": 4, "n_valleys_count": 3},
         rv_hint="fracture_path_landscape", constraints=["requires_db5_calibration = true"], flags=[],
         rules=["R-TOUGH-001"], tough=True,
         q={"q8_stability_margin": dict(direction="lower_bound", expr="E_req"),
            "q10_redundancy": dict(direction="increase", expr=">= 3 metastable basins")},
         we=dict(
             setup="头盔缓冲层（逐步压溃：3 个塌缩模式）",
             given={"E_req": {"value": 4, "unit": "nats", "is_bridge_constant": True},
                    "n_valleys": {"value": 3, "unit": "count", "is_bridge_constant": False},
                    "Delta_F_ddagger": {"value": 5, "unit": "nats", "is_bridge_constant": False}},
             computation="ΔF‡ = 5 nats ≥ 4 ✓；谷数 3 ≥ 3 ✓",
             result="q8 ≥ 4 nats；多谷 3 个",
             reference_scale="类比级参考：冲击吸能结构能垒 3–8 nats；绝对值须 DB5 校准")),
    # ============ 多规则组合 ×3 ============
    dict(id=28, domain="mechanics", lang="zh",
         text="快速伺服平台又要硬又要快：刚度不低于 200 N/m，95% 响应不超过 10 ms。",
         prop=None, dir=None, target={"stiffness_N_per_m": 200, "settling_time_s": 0.01, "settling_band": "95_percent"},
         rv_hint="microstate_gibbs+markov_chain", constraints=["T_K = 300"], flags=[],
         rules=["R-STIFF-001", "R-FAST-001"],
         q={"q1_fisher_directional": dict(direction="lower_bound", expr="K_req/(k_B T)", note="位移方向"),
            "q9_spectral_gap": dict(direction="lower_bound", expr="3/t_settle（95% 建立 ≈ 3τ，一阶系统）")},
         we=dict(
             setup="伺服平台，K=200 N/m，95% 建立时间 10 ms，T=300 K",
             given={"K_req": {"value": 200, "unit": "N/m", "is_bridge_constant": True},
                    "k_B_T": {"value": 4.14e-21, "unit": "J", "is_bridge_constant": True},
                    "t_settle": {"value": 0.01, "unit": "s", "is_bridge_constant": True}},
             computation="q1 ≥ 200/4.14e-21 = 4.8e22 m^-2；q9 ≥ 3/0.01 = 300 s^-1",
             result="q1 ≥ 4.8e22 m^-2；q9 ≥ 300 s^-1",
             reference_scale="组合下界各分量参考其单规则种子对（DB3-0001/DB3-0009）")),
    dict(id=29, domain="measurement", lang="en",
         text="The inline pH sensor must achieve sigma_req = 0.02 pH in a single reading while tolerating temperature drift over 15-35 C (interference Fisher <= 1e-2 K^-2).",
         prop=None, dir=None, target={"sigma_pH": 0.02},
         rv_hint="joint_signal_noise_family", constraints=["T_range_K = [288, 308]"], flags=[],
         rules=["R-PRECISE-001", "R-ROBUST-001"],
         q={"q1_fisher_directional": dict(direction="lower_bound", expr="1/(n·sigma_req^2)", note="信号 θ=pH 方向"),
            "q4_dpi_margin": dict(direction="lower_bound", expr="delta_dpi = 0.5 nats"),
            "q10_redundancy": dict(direction="increase", expr=">= ln(2)"),
            "q1_noise_direction": dict(direction="upper_bound", expr="lambda_max(g_eta_eta) <= 1e-2 K^-2", note="干扰 η=温漂方向")},
         we=dict(
             setup="在线 pH 传感器：信号=pH，干扰=温漂",
             given={"sigma_req": {"value": 0.02, "unit": "pH", "is_bridge_constant": True},
                    "epsilon": {"value": 1e-2, "unit": "1/K^2", "is_bridge_constant": True},
                    "delta_dpi": {"value": 0.5, "unit": "nats", "is_bridge_constant": True},
                    "n": 1},
             computation="q1 ≥ 1/0.02² = 2500 pH^-2；λ_max ≤ 1e-2 K^-2；q4 ≥ 0.5 nats；q10 ≥ ln2 nats",
             result="q1 ≥ 2500 pH^-2；λ_max ≤ 1e-2 K^-2；q4 ≥ 0.5；q10 ≥ 0.693 nats",
             reference_scale="pH 方向 q1 参考尺度：过程控制 1e3–1e4 pH^-2")),
    dict(id=30, domain="thermal", lang="zh",
         text="功率模块散热要快且稳：2 mm 铜层热锋 0.5 s 内传过，且结壳工作点距焊料熔化点的约化裕度不低于 0.2。",
         prop=None, dir=None, target={"kappa_m2_per_s": 8e-6, "length_scale_m": 2e-3, "front_time_s": 0.5, "delta_dimensionless": 0.2},
         rv_hint="field_time_slices+landau_order_parameter", constraints=["material = copper"],
         flags=["R-COND-001 二义性：文本'热锋传过'显式指向读向(a)（κ↑，q7↑）；显式消解，仍记录标志备查"],
         rules=["R-COND-001", "R-STABLE-001"],
         q={"q7_info_propagation_speed": dict(direction="lower_bound", expr="kappa_req/L", reading="a_heat_front_fast"),
            "q8_stability_margin": dict(direction="lower_bound", expr="delta = (T_c - T_work)/T_c")},
         we=dict(
             setup="功率模块：铜层 L=2 mm，τ=0.5 s；焊料 T_melt=456 K，结壳 T_work=350 K",
             given={"L": {"value": 2e-3, "unit": "m", "is_bridge_constant": True},
                    "kappa": {"value": 8e-6, "unit": "m^2/s", "is_bridge_constant": True},
                    "front_time": {"value": 0.5, "unit": "s", "is_bridge_constant": True},
                    "T_c": {"value": 456, "unit": "K", "is_bridge_constant": True},
                    "T_work": {"value": 350, "unit": "K", "is_bridge_constant": True}},
             computation="κ_req = L²/τ = 4e-6/0.5 = 8e-6 m²/s；q7 ≥ 8e-6/2e-3 = 4e-3 m/s；q8 = (456−350)/456 = 0.23",
             result="q7 ≥ 4e-3 m/s；q8 = 0.23 ≥ 0.2 ✓",
             reference_scale="q7 参考尺度：功率器件热输运 1e-3–1e-1 m/s；q8 工程裕度 0.1–0.5")),
    # ============ UNMAPPED ×4（禁止携带 q 向量） ============
    dict(id=31, domain="materials", lang="zh",
         text="产品外壳材料要有高级质感、手感温润。",
         unmapped_suggestions=["R-ROBUST-001（仅当'质感'实际指耐刮擦/耐磨时才可映射——须先人工确认谓词本意）"]),
    dict(id=32, domain="general", lang="en",
         text="The whole assembly should cost under $5 in volume production.",
         unmapped_suggestions=[]),
    dict(id=33, domain="electronics", lang="zh",
         text="设备要支持无线充电功能。",
         unmapped_suggestions=[]),
    dict(id=34, domain="general", lang="zh",
         text="整套方案要环保、可回收、可持续。",
         unmapped_suggestions=[]),
]


def _normalize_given(we):
    """given 值结构统一：裸数值升级为 {value,unit,is_bridge_constant}。
    计数类（n/M 等）为无量纲规格，非桥接常数——is_bridge_constant=false。"""
    for k, v in (we.get("given") or {}).items():
        if not isinstance(v, dict):
            we["given"][k] = {
                "value": v,
                "unit": "dimensionless_count",
                "is_bridge_constant": False,
            }
    return we


def build_pair(p):
    """把紧凑数据展开为完整种子对记录。"""
    pair_id = "DB3-%04d" % p["id"]
    if "unmapped_suggestions" in p:
        return {
            "pair_id": pair_id,
            "domain": p["domain"],
            "language": p["lang"],
            "requirement_text": p["text"],
            "requirement_structured": {
                "property_id": "unmapped",
                "direction": None,
                "target_value": {},
                "rv_hint": None,
                "constraints": [],
                "ambiguity_flags": [],
            },
            "rule_ids": ["R-NOMAP-000"],
            "info_metric_vector": None,
            "worked_example": None,
            "metric_rationale": list(RATIONALE_BY_RULE["R-NOMAP-000"]),
            "strictness": None,
            "requires_db5_calibration": False,
            "expected_unmapped_output": {
                "flag": "UNMAPPED",
                "nearest_rule_suggestions": p["unmapped_suggestions"],
                "human_queue": True,
            },
            "provenance": {
                "source": "AI+drafted",
                "human_review_status": "pending",
                "anchor_eligible": False,
                "review_date": None,
                "reviewer": None,
            },
            "rules_version": RULES_VERSION,
            "convention_version": CONVENTION_VERSION,
        }
    strictness = min(
        (RULE_STRICTNESS[r] for r in p["rules"]),
        key=lambda s: STRICTNESS_ORDER[s],
    )
    rationale = []
    for r in p["rules"]:
        for item in RATIONALE_BY_RULE[r]:
            if item not in rationale:
                rationale.append(item)
    rec = {
        "pair_id": pair_id,
        "domain": p["domain"],
        "language": p["lang"],
        "requirement_text": p["text"],
        "requirement_structured": {
            "property_id": p["prop"],
            "direction": p["dir"],
            "target_value": p["target"],
            "rv_hint": p["rv_hint"],
            "constraints": p["constraints"],
            "ambiguity_flags": p["flags"],
        },
        "rule_ids": p["rules"],
        "info_metric_vector": p["q"],
        "worked_example": p["we"],
        "metric_rationale": rationale,
        "strictness": strictness,
        "requires_db5_calibration": bool(p.get("tough")),
        "provenance": {
            "source": "AI+drafted",
            "human_review_status": "pending",
            "anchor_eligible": False,
            "review_date": None,
            "reviewer": None,
        },
        "rules_version": RULES_VERSION,
        "convention_version": CONVENTION_VERSION,
    }
    return rec


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n = 0
    for p in PAIRS:
        rec = build_pair(p)
        if rec.get("worked_example"):
            rec["worked_example"] = _normalize_given(rec["worked_example"])
        path = os.path.join(OUT_DIR, rec["pair_id"] + ".json")
        # provenance 保留：已复核条目的状态不因重生成而丢失（复核是唯一升格通道）
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
            if old.get("provenance", {}).get("human_review_status") != "pending":
                rec["provenance"] = old["provenance"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
            f.write("\n")
        n += 1
    print("generated %d pairs -> %s" % (n, OUT_DIR))


if __name__ == "__main__":
    main()
