"""图张量化适配器：DB1 条目 → EncodedGraph → 稠密张量（供 GINE 消费）。

这是架构文档 §2.2.1 的"输入表示"落地，也是 A 网络训练管线的第一步：

    DB1 条目 ──SystemEncoder.encode()──▶ EncodedGraph ──compute_graph_fingerprint──▶
    {节点特征 H, 邻接 A, 边特征 E, mask} + {节点 Fisher 真值, family 标签}

关键设计：
- 图 ≤ 10 节点，用稠密邻接矩阵（[N,N]）+ 稠密边特征（[N,N,d_e]），消息聚合用
  `A @ M` 矩阵乘完成——不依赖 PyG/DGL/torch_scatter。
- 节点特征 = 分布族编码（连续属性向量，ATTR_VERSION 选 v1/v2；或 one-hot）
  + 参数类型码（仅 v2）+ 分布参数值 + 结构编码（维度/入度/出度/链位置）。
  **故意不含 Fisher**：Fisher 是回归头的预测目标，若喂进输入就成了恒等映射。
- 边特征起步只喂互信息（1 维），容量/DPI 裕度二期再补（改 EDGE_FEAT_DIM 与
  编码处即可）。
- 监督标签：每节点 fisher_trace/det 的 log（回归），family（分类）。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from b_network.system_encoder import SystemEncoder  # noqa: E402
from core.fingerprint import compute_graph_fingerprint  # noqa: E402

__all__ = [
    "DIST_FAMILIES",
    "DIST_ATTRS",
    "ATTR_VERSION",
    "PARAM_TYPES",
    "PARAM_NAMES",
    "SIGN_PARAMS",
    "RAW_SIGN_PARAMS",
    "MAX_NODES",
    "NODE_FEAT_DIM",
    "EDGE_FEAT_DIM",
    "GraphDataset",
]

# 分布族枚举（与 core.fingerprint 的 _DIST_* 对齐，顺序固定）
# 扩族只允许在末尾追加（保持既有索引不变，旧 checkpoint 可加载）。
DIST_FAMILIES: list[str] = [
    "DIST-GAUSS", "DIST-GIBBS", "DIST-EXPONENTIAL", "DIST-POISSON",
    "DIST-BERNOULLI", "DIST-LAPLACE", "DIST-LANDAU", "DIST-GAMMA",
    "DIST-BETA", "DIST-WEIBULL", "DIST-VONMISES",
    "DIST-GEOMETRIC", "DIST-CAUCHY", "DIST-RAYLEIGH", "DIST-LOGISTIC",
    "DIST-CHISQ", "DIST-NEGBIN", "DIST-LOGNORM",
    "DIST-MIX2", "DIST-TRUNCNORM",
    "DIST-TRUNCGEOM", "DIST-INVGAMMA",
    "DIST-INVCHISQ", "DIST-FRECHET",
    "DIST-RICIAN", "DIST-WRAPNORM",
]
_DIST_TO_IDX = {f: i for i, f in enumerate(DIST_FAMILIES)}

# ---------------------------------------------------------------------------
# 分布族连续属性向量（替代 one-hot 的探索，§2.2.8「改表示」方向）
# ---------------------------------------------------------------------------
# 属性版本开关（A/B 复现用；attr_version 记入 metrics JSON）：
#   v1：6 维结构属性。300 epoch 族级 OOD：test trace -0.205 / det +0.082（基线）。
#   v2：11 维 = 5 结构属性 + 6 维主参数类型 one-hot，外加每参数列类型码。
#       300 epoch 证伪：test trace -1.97（gamma -19.75）——语义近亲借错函数
#       形状：ψ₁ 类型让 gamma 借 beta 的「递减→平坦」曲线，真实为 U 形上升；
#       v1 的 gamma↔weibull「语义错误但形状巧合」（同为 U 形）反而更好。
#       教训（HANDOVER 第 24 条）：族级 OOD 迁移的是目标曲线的函数形状
#       相似性，不是参数语义类型相似性。v2 实现保留，待扩族（族数>100，
#       §2.6.4 触发条件）后复评——届时每个语义类型有多个训练族可绑定形式。
ATTR_VERSION: str = "v1"   # "v1" | "v2"

# v1：[is_discrete, support_lower_finite, support_upper_finite,
#      param_is_scale, is_symmetric, tail_infinite_var]
_DIST_ATTRS_V1: dict[str, list[float]] = {
    "DIST-GAUSS":        [0, 0, 0, 1, 1, 0],  # 连续 实轴 尺度 对称 有限方差
    "DIST-GIBBS":        [0, 0, 0, 1, 1, 0],  # 同高斯（平衡态边缘）
    "DIST-EXPONENTIAL":  [0, 1, 0, 1, 0, 0],  # 连续 正半轴 尺度(逆) 偏斜 有限方差
    "DIST-POISSON":      [1, 1, 0, 1, 0, 0],  # 离散 正半轴 速率(幂律-1) 偏斜
    "DIST-BERNOULLI":    [1, 1, 1, 0, 0, 0],  # 离散 有界 概率(非幂律) 偏斜
    "DIST-LAPLACE":      [0, 0, 0, 1, 1, 0],  # 连续 实轴 尺度 对称 有限方差
    "DIST-LANDAU":       [0, 0, 0, 0, 1, 0],  # 连续 实轴 形状 对称 有限方差
    "DIST-GAMMA":        [0, 1, 0, 0, 0, 0],  # 连续 正半轴 形状 偏斜 有限方差
    "DIST-BETA":         [0, 1, 1, 0, 0, 0],  # 连续 有界 形状 偏斜 有限方差
    "DIST-WEIBULL":      [0, 1, 0, 0, 0, 0],  # 连续 正半轴 形状 偏斜 有限方差
    "DIST-VONMISES":     [0, 1, 1, 1, 1, 0],  # 连续 圆(有界) 尺度 对称 有限方差
    "DIST-GEOMETRIC":    [1, 1, 0, 0, 0, 0],  # 离散 正半轴 概率(非幂律) 偏斜
    "DIST-CAUCHY":       [0, 0, 0, 1, 1, 1],  # 连续 实轴 尺度 对称 无限方差(重尾)
    "DIST-RAYLEIGH":     [0, 1, 0, 1, 0, 0],  # 连续 正半轴 尺度 偏斜 有限方差
    "DIST-LOGISTIC":     [0, 0, 0, 1, 1, 0],  # 连续 实轴 尺度 对称 有限方差
    # ---- 扩族批次 3：chi² 补 ψ₁ 覆盖 / negbin 补计数覆盖 / lognormal 对照 ----
    "DIST-CHISQ":        [0, 1, 0, 0, 0, 0],  # 连续 正半轴 形状(k) 偏斜 有限方差（与 gamma 同组）
    "DIST-NEGBIN":       [1, 1, 0, 0, 0, 0],  # 离散 正半轴 概率(p) 偏斜 有限方差（与 geometric 同组）
    "DIST-LOGNORM":      [0, 1, 0, 1, 0, 0],  # 连续 正半轴 尺度(σ) 偏斜 有限方差（与 exponential 同组）
    # ---- 扩族批次 4（R4 平膝族）：MIX2 与 LANDAU 同属性组（形状外推可借）----
    "DIST-MIX2":         [0, 0, 0, 0, 1, 0],  # 连续 实轴 形状(d=分离度) 对称 有限方差
    "DIST-TRUNCNORM":    [0, 1, 0, 0, 0, 0],  # 连续 下界有界(τ) 形状(截断点) 偏斜 有限方差
    "DIST-TRUNCGEOM":    [1, 1, 1, 0, 0, 0],  # 离散 上下有界(k) 形状(k,p) 偏斜 有限方差（R4b：negbin 同构）
    "DIST-INVGAMMA":     [0, 1, 0, 0, 0, 0],  # 连续 正半轴 形状(k,θ) 偏斜 有限方差（R4c：与 gamma 同组）
    "DIST-INVCHISQ":     [0, 1, 0, 0, 0, 0],  # 连续 正半轴 形状(k) 偏斜 有限方差（R4d：与 chisq 同组）
    "DIST-FRECHET":      [0, 1, 0, 0, 0, 0],  # 连续 正半轴 形状(α)×尺度(s) 偏斜 有限方差（R4d：与 weibull 同组）
    "DIST-RICIAN":       [0, 1, 0, 0, 0, 0],  # 连续 正半轴 集中度(κ) 偏斜 有限方差（R4d 第三刀：与 vonmises 同列不同支撑）
    "DIST-WRAPNORM":     [0, 1, 1, 1, 1, 0],  # 连续 圆(有界) 尺度 对称 有限方差（与 vonmises 属性完全相同）
}

# v2：[disc, lo, hi, sym, tail | pt_scale, pt_count_rate, pt_prob,
#      pt_shape_psi, pt_shape_pow, pt_transc]（主参数类型，见上版本说明）
_DIST_ATTRS_V2: dict[str, list[float]] = {
    "DIST-GAUSS":        [0, 0, 0, 1, 0,  1, 0, 0, 0, 0, 0],  # 扫 σ：3/σ²
    "DIST-GIBBS":        [0, 0, 0, 1, 0,  1, 0, 0, 0, 0, 0],  # 同高斯（平衡态边缘）
    "DIST-EXPONENTIAL":  [0, 1, 0, 0, 0,  1, 0, 0, 0, 0, 0],  # 扫 rate(逆尺度)：1/λ²
    "DIST-POISSON":      [1, 1, 0, 0, 0,  0, 1, 0, 0, 0, 0],  # 扫 rate(计数均值)：1/λ
    "DIST-BERNOULLI":    [1, 1, 1, 0, 0,  0, 0, 1, 0, 0, 0],  # 扫 p：1/(p(1-p))
    "DIST-LAPLACE":      [0, 0, 0, 1, 0,  1, 0, 0, 0, 0, 0],  # 扫 b：2/b²
    "DIST-LANDAU":       [0, 0, 0, 1, 0,  0, 0, 0, 0, 0, 1],  # 扫 a：Var(φ²)/(4k²T²)
    "DIST-GAMMA":        [0, 1, 0, 0, 0,  0, 0, 0, 1, 0, 0],  # 扫 k：ψ₁(k)+k/θ²
    "DIST-BETA":         [0, 1, 1, 0, 0,  0, 0, 0, 1, 0, 0],  # 扫 α：ψ₁ 组合
    "DIST-WEIBULL":      [0, 1, 0, 0, 0,  0, 0, 0, 0, 1, 0],  # 扫 shape：k²/λ²+C/k²
    "DIST-VONMISES":     [0, 1, 1, 1, 0,  0, 0, 0, 0, 0, 1],  # 扫 κ：κA(κ)+Bessel 项
    "DIST-GEOMETRIC":    [1, 1, 0, 0, 0,  0, 0, 1, 0, 0, 0],  # 扫 p：1/(p²(1-p))
    "DIST-CAUCHY":       [0, 0, 0, 1, 1,  1, 0, 0, 0, 0, 0],  # 扫 γ：1/(2γ²)
    "DIST-RAYLEIGH":     [0, 1, 0, 0, 0,  1, 0, 0, 0, 0, 0],  # 扫 σ：4/σ²
    "DIST-LOGISTIC":     [0, 0, 0, 1, 0,  1, 0, 0, 0, 0, 0],  # 扫 s：(π²+3)/(9s²)
    # ---- 扩族批次 3（v2 保留待复评，见文件头版本说明）----
    "DIST-CHISQ":        [0, 1, 0, 0, 0,  0, 0, 0, 1, 0, 0],  # 扫 k：ψ₁(k/2)/4（纯 ψ₁ 曲线）
    "DIST-NEGBIN":       [1, 1, 0, 0, 0,  0, 0, 1, 0, 0, 0],  # 扫 p：k/(p²(1-p))
    "DIST-LOGNORM":      [0, 1, 0, 0, 0,  1, 0, 0, 0, 0, 0],  # 扫 σ：3/σ²（高斯同形）
    # ---- 扩族批次 4（R4 平膝族，v2 保留待复评）----
    "DIST-MIX2":         [0, 0, 0, 1, 0,  0, 0, 0, 0, 0, 1],  # 扫 d：0→平台饱和曲线（超越型）
    "DIST-TRUNCNORM":    [0, 1, 0, 0, 0,  0, 0, 0, 0, 0, 1],  # 扫 τ：Mills 比平方（超越型）
    "DIST-TRUNCGEOM":    [1, 1, 1, 0, 0,  0, 0, 1, 0, 0, 0],  # 扫 p：U 形左腿（概率型）
    "DIST-INVGAMMA":     [0, 1, 0, 0, 0,  0, 0, 0, 1, 0, 0],  # 扫 k：ψ₁(k)+k/θ²（与 gamma 同形）
    "DIST-INVCHISQ":     [0, 1, 0, 0, 0,  0, 0, 0, 1, 0, 0],  # 扫 k：ψ₁(k/2)/4（与 chisq 同形）
    "DIST-FRECHET":      [0, 1, 0, 0, 0,  0, 0, 0, 0, 1, 0],  # 扫 shape：C/α²+α²/s²（与 weibull 同形）
    "DIST-RICIAN":       [0, 1, 0, 0, 0,  0, 0, 0, 0, 0, 1],  # 扫 κ：1→1/(2κ) 衰减（与 vonmises 同向同尾）
    "DIST-WRAPNORM":     [0, 1, 1, 1, 0,  0, 0, 0, 0, 0, 1],  # 扫 κ：→0 衰减（与 vonmises 同形）
}

DIST_ATTRS: dict[str, list[float]] = (
    _DIST_ATTRS_V1 if ATTR_VERSION == "v1" else _DIST_ATTRS_V2
)
ATTR_DIM: int = 6 if ATTR_VERSION == "v1" else 11

# 编码开关：True 用连续属性向量，False 用 one-hot（供 A/B 对比）
USE_ATTRS: bool = True
_ENC_DIM: int = ATTR_DIM if USE_ATTRS else len(DIST_FAMILIES)

# 节点分布参数名全集（从 SystemEncoder 各编码器的 node.params 收集，顺序固定）
PARAM_NAMES: list[str] = [
    "mu", "sigma", "rate", "p", "b", "a", "kT", "k",
    "theta", "alpha", "beta", "shape", "scale", "kappa", "gamma",
    "d", "tau", "w",
]

# ---------------------------------------------------------------------------
# 参数类型码：每族每个参数的语义角色（Fisher 函数形式的决定因素）
# ---------------------------------------------------------------------------
#   0 无该参数 / 1 位置 / 2 尺度(含逆尺度) / 3 计数强度 / 4 概率 /
#   5 ψ₁ 形状 / 6 幂律形状 / 7 超越型
# 关键机制：「参数列 × 语义」可组合外推——
#   - 同列不同义：rate 列在指数族=逆尺度(Fisher -2)、泊松族=计数均值(-1)，
#     类型码随族标注，模型不再被参数名混淆；
#   - 交叉项结构：weibull 的 (shape=6, scale=2) 让模型学会「形状×尺度⁻²」
#     组合，gamma 的 (k=5, theta=2) 借用同一结构（k/θ² 项）。
PARAM_TYPES: dict[str, dict[str, int]] = {
    "DIST-GAUSS":       {"mu": 1, "sigma": 2},
    "DIST-GIBBS":       {"sigma": 2},            # 指纹层与高斯同构（sigma 参数化）
    "DIST-EXPONENTIAL": {"rate": 2},             # 速率=逆尺度，Fisher 1/λ² 与尺度同类
    "DIST-POISSON":     {"rate": 3},             # 计数均值，Fisher 1/λ（与指数族同列不同义）
    "DIST-BERNOULLI":   {"p": 4},
    "DIST-LAPLACE":     {"mu": 1, "b": 2},
    "DIST-LANDAU":      {"a": 7, "b": 7, "kT": 2},   # 序参量超越型；kT 显式 1/kT²
    "DIST-GAMMA":       {"k": 5, "theta": 2},     # ψ₁(k) 形状 × 尺度
    "DIST-BETA":        {"alpha": 5, "beta": 5},
    "DIST-WEIBULL":     {"shape": 6, "scale": 2},    # 幂律形状 × 尺度
    "DIST-VONMISES":    {"mu": 1, "kappa": 7},    # 集中度：Bessel 比 A(κ)
    "DIST-GEOMETRIC":   {"p": 4},
    "DIST-CAUCHY":      {"gamma": 2},
    "DIST-RAYLEIGH":    {"sigma": 2},
    "DIST-LOGISTIC":    {"scale": 2},
    # ---- 扩族批次 3 ----
    "DIST-CHISQ":       {"k": 5},               # 自由度：ψ₁(k/2)/4
    "DIST-NEGBIN":      {"k": 5, "p": 4},       # k=成功次数（伽马形状，结构常数）；p 概率
    "DIST-LOGNORM":     {"mu": 1, "sigma": 2},  # 对数空间 (μ,σ)，Fisher 与高斯同形
    # ---- 扩族批次 4（R4 平膝族）----
    "DIST-MIX2":        {"d": 7, "w": 4, "sigma": 2},   # 分离度（超越型凸起）；w 概率；σ 尺度
    "DIST-TRUNCNORM":   {"tau": 7, "mu": 1, "sigma": 2},  # 截断点（Mills 比平方，超越型）
    "DIST-TRUNCGEOM":   {"k": 6, "p": 4},   # k 支撑上界（结构常数，非 ψ₁）；p 概率（U 形）
    "DIST-INVGAMMA":    {"k": 5, "theta": 2},  # k ψ₁ 形状 × theta 尺度（与 gamma 同结构）
    "DIST-INVCHISQ":    {"k": 5},              # 自由度：ψ₁(k/2)/4（与 chisq 同结构）
    "DIST-FRECHET":     {"shape": 6, "scale": 2},  # 幂律形状 × 尺度（与 weibull 同结构）
    "DIST-RICIAN":      {"kappa": 7},              # K 因子：Bessel 比（与 vonmises 同列同类型）
    "DIST-WRAPNORM":    {"kappa": 7},              # 集中度：级数比（与 vonmises 同列同类型）
}
# 每参数列一个类型码（0-7）。v2 专用：v1 无类型码（语义类型编码已被证伪，
# 见文件头版本说明），布局与产生 -0.205/+0.082 基线的原始 v1 运行完全一致。
_TYPE_DIM: int = len(PARAM_NAMES) if ATTR_VERSION == "v2" else 0

# 节点特征维度 = 编码(属性或 one-hot) + 参数类型码 + 参数 + 结构编码(4)
NODE_FEAT_DIM: int = _ENC_DIM + _TYPE_DIM + len(PARAM_NAMES) + 4
_ONHOT_DIM: int = _ENC_DIM + _TYPE_DIM   # 类别前缀（编码+类型码，标准化时跳过）
_CONT_DIM: int = NODE_FEAT_DIM - _ONHOT_DIM

# 边特征维度：Phase 1 仅互信息；二期扩展容量/DPI 裕度（改此常量 + 编码处）
EDGE_FEAT_DIM: int = 1

# 图节点数上限（架构文档 §2.2.1：≤ 10）
MAX_NODES: int = 10

# 约束 A 的边表槽位数（当前最大为 ou 的 2 条边；-1 填充空槽）
MAX_EDGES: int = 4

# 每系统参数扫描点数（对齐 batch_generate_ingest.SCAN_POINTS）
_SCAN_POINTS: int = 30

# R2b（第 30/31 条）：符号跨越参数的编码策略开关。
# 混合编码（v>0 取 log、v≤0 取原值）对可负参数非单射：a<0 支特征 ∈[-2,0)、
# a>0 支 = ln a ∈[-2.67,0.69)，两支重叠 → 同一特征对应两个目标，landau 在
# 训练集内 R² 仅 0.103（背不下来）。开关打开后 SIGN_PARAMS 中的参数用原值
# 编码（连续、单射）；正参数族保持 log（第 21 条幂律线性化成果不动）。
# 第 31 条教训：SIGN_PARAMS 只放「被扫描且目标依赖」的可负参数（当前仅 a）——
# mu 虽可负但目标不依赖它，改它的编码只会经共享列的标准化污染全库
# （实证：mu+a 全改时 geometric train 0.83→0.49、fixed-holdout 其他族全变）。
# 默认关 = 旧行为（基线可复现）。
SIGN_PARAMS: set[str] = {"a"}
RAW_SIGN_PARAMS: bool = False

_ROLE_INDEX = {"X": 0, "T": 1, "Y": 2}


def _role_index(node_id: str) -> float:
    """从 node_id（{family}__{role}__{index}）解析链位置序号。"""
    parts = node_id.split("__")
    role = parts[1] if len(parts) >= 2 else "T"
    return float(_ROLE_INDEX.get(role, 1))


def _scan_grid(entry: dict, n_points: int = _SCAN_POINTS) -> list:
    """按 scan_param 生成参数扫描网格（与 batch_generate_ingest._scan_grid 对齐）。"""
    sp = entry.get("scan_param")
    if not sp:
        return [None]
    for p in entry.get("parameters", []):
        if p["name"] == sp:
            lo, hi = float(p["range"][0]), float(p["range"][1])
            if lo > 0 and hi / lo >= 10.0:
                return np.logspace(np.log10(lo), np.log10(hi), n_points).tolist()
            return np.linspace(lo, hi, n_points).tolist()
    return [None]


def graph_to_tensors(graph, fp_result: dict) -> dict:
    """EncodedGraph + 指纹 → 稠密张量 + 监督标签。

    参数
    ----
    graph : EncodedGraph
        信息结构图（节点已带分布参数，边已带互信息）。
    fp_result : dict
        compute_graph_fingerprint 的返回（含 nodes 的 fisher_trace/det 与 mi_profile）。

    返回
    ----
    dict
        {"H", "A", "E", "mask", "fisher_target", "node_ids"}
    """
    nodes = sorted(graph.nodes, key=lambda n: n.node_id)
    id_to_idx = {n.node_id: i for i, n in enumerate(nodes)}
    N = len(nodes)
    if N > MAX_NODES:
        raise ValueError(f"节点数 {N} 超过 MAX_NODES={MAX_NODES}")

    # 度统计
    in_deg = [0] * N
    out_deg = [0] * N
    for e in graph.edges:
        if e.src in id_to_idx and e.dst in id_to_idx:
            s, d = id_to_idx[e.src], id_to_idx[e.dst]
            out_deg[s] += 1
            in_deg[d] += 1

    # 节点特征
    H = np.zeros((MAX_NODES, NODE_FEAT_DIM), dtype=np.float32)
    for i, n in enumerate(nodes):
        if USE_ATTRS:
            # 连续属性向量替代 one-hot（探索：让模型从属性外推 Fisher 函数形式）
            attr = DIST_ATTRS.get(n.distribution_family, [0.0] * ATTR_DIM)
            H[i, :ATTR_DIM] = attr
            off = ATTR_DIM
        else:
            fi = _DIST_TO_IDX.get(n.distribution_family, -1)
            if fi >= 0:
                H[i, fi] = 1.0
            off = len(DIST_FAMILIES)
        # 每参数列的语义类型码（0-7）：仅 v2 启用（v2 证伪记录见文件头）
        if ATTR_VERSION == "v2":
            ptypes = PARAM_TYPES.get(n.distribution_family, {})
            for j, pname in enumerate(PARAM_NAMES):
                H[i, off + j] = float(ptypes.get(pname, 0))
        off += _TYPE_DIM
        for j, pname in enumerate(PARAM_NAMES):
            v = n.params.get(pname)
            if v is not None:
                v = float(v)
                # 正参数取 log：Fisher 对参数是幂律依赖（如 1/σ²），log 后变线性，
                # 跨族泛化（外推 Fisher 函数形式）才可能。
                # R2b（第 30 条）：符号跨越参数（mu/a）在开关打开时用原值编码，
                # 避免混合编码的非单射歧义（见 SIGN_PARAMS 定义处注释）。
                if RAW_SIGN_PARAMS and pname in SIGN_PARAMS:
                    H[i, off + j] = v
                else:
                    H[i, off + j] = math.log(v) if v > 0 else v
        off += len(PARAM_NAMES)
        H[i, off] = float(getattr(n, "dimension", 1))
        H[i, off + 1] = float(in_deg[i])
        H[i, off + 2] = float(out_deg[i])
        H[i, off + 3] = _role_index(n.node_id)

    # 邻接（有向，A[dst, src]=1）与边特征（互信息）
    A = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
    E = np.zeros((MAX_NODES, MAX_NODES, EDGE_FEAT_DIM), dtype=np.float32)
    for e, mi in zip(graph.edges, fp_result.get("mi_profile", [])):
        if e.src in id_to_idx and e.dst in id_to_idx:
            s, d = id_to_idx[e.src], id_to_idx[e.dst]
            A[d, s] = 1.0
            if mi is not None:
                E[d, s, 0] = float(mi)

    # 有效节点 mask
    mask = np.zeros((MAX_NODES,), dtype=np.float32)
    mask[:N] = 1.0

    # 回归目标：每节点 fisher_trace/det 的 log（Fisher 正定，log 安全）
    fisher_target = np.zeros((MAX_NODES, 2), dtype=np.float32)
    for i, nid in enumerate([n.node_id for n in nodes]):
        fp = fp_result.get("nodes", {}).get(nid, {})
        ft = fp.get("fisher_trace")
        fd = fp.get("fisher_det")
        if ft is not None and ft > 0:
            fisher_target[i, 0] = math.log(float(ft))
        if fd is not None and fd > 0:
            fisher_target[i, 1] = math.log(float(fd))

    # 约束 A 的边表：[src_idx, dst_idx, 输入 MI, 是否加性噪声边]，-1 填充空槽。
    # 加性噪声边（v = u + 独立噪声）满足 MI = -½ln(1 - F_dst/F_src)（离线验证机器精度）。
    edge_tbl = np.full((MAX_EDGES, 4), -1.0, dtype=np.float32)
    k_edge = 0
    for e, mi in zip(graph.edges, fp_result.get("mi_profile", [])):
        if k_edge >= MAX_EDGES:
            break
        if e.src in id_to_idx and e.dst in id_to_idx and mi is not None:
            cov = getattr(e.channel, "cov", None)
            is_noise = 0.0
            if cov is not None:
                cov = np.asarray(cov, dtype=float)
                if abs(cov[0, 1] - cov[0, 0]) <= 1e-9 * max(1.0, abs(cov[0, 0])):
                    is_noise = 1.0
            edge_tbl[k_edge] = [id_to_idx[e.src], id_to_idx[e.dst],
                                float(mi), is_noise]
            k_edge += 1

    return {
        "H": H, "A": A, "E": E, "mask": mask,
        "fisher_target": fisher_target,
        "edge_tbl": edge_tbl,
        "node_ids": [n.node_id for n in nodes],
    }


class GraphDataset(Dataset):
    """把 DB1 条目（含参数扫描）预处理为图张量样本。

    每个样本 = 一个 (system_id, 扫描点)，对应一张信息结构图。
    全部在 __init__ 里预处理（encode + 指纹 + 张量化），避免每 epoch 重算。
    """

    def __init__(self, entries_dir: str | Path, n_points: int = _SCAN_POINTS,
                 include: set[str] | None = None):
        entries_dir = Path(entries_dir)
        encoder = SystemEncoder()

        self.samples: list[dict] = []
        self.families: list[str] = []
        self.system_ids: list[str] = []
        self.sample_system_ids: list[str] = []
        # 约束 B 的路径对：[i, j, Δlog F_analytic]（i,j 为同系统相邻扫描点的全局索引）
        self.path_pairs: list[list[float]] = []

        for ef in sorted(entries_dir.glob("*.json")):
            import json
            entry = json.loads(ef.read_text(encoding="utf-8"))
            sid = entry["system_id"]
            if include is not None and sid not in include:
                continue
            fam = entry["family"]
            if fam not in self.families:
                self.families.append(fam)
            fam_idx = self.families.index(fam)

            first = len(self.samples)
            for val in _scan_grid(entry, n_points):
                e_pt = entry if val is None else _with_scan(entry, val)
                graph = encoder.encode(e_pt)
                fp = compute_graph_fingerprint(graph)
                tensors = graph_to_tensors(graph, fp)
                tensors["family"] = fam_idx
                tensors["system_id"] = sid
                tensors["scan_value"] = val   # 物理扫描值（诊断取证用；None=无扫描）
                self.samples.append(tensors)
                self.sample_system_ids.append(sid)
            self.system_ids.append(sid)

            # 约束 B：同系统相邻扫描点的解析 log Fisher 差分（斜率/形状约束）。
            # 仅单节点图——多节点图（ou/diffusion）的扫描参数 Fisher 无单值意义。
            if (len(self.samples) > first
                    and float(self.samples[first]["mask"].sum()) == 1.0):
                for a, b in zip(range(first, len(self.samples) - 1),
                                range(first + 1, len(self.samples))):
                    lt_a = float(self.samples[a]["fisher_target"][0, 0])
                    lt_b = float(self.samples[b]["fisher_target"][0, 0])
                    self.path_pairs.append([float(a), float(b), lt_b - lt_a])

        self.system_ids = sorted(set(self.system_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "H": torch.from_numpy(s["H"]),
            "A": torch.from_numpy(s["A"]),
            "E": torch.from_numpy(s["E"]),
            "mask": torch.from_numpy(s["mask"]),
            "fisher_target": torch.from_numpy(s["fisher_target"]),
            "edge_tbl": torch.from_numpy(s["edge_tbl"]),
            "family": s["family"],
            "system_id": s["system_id"],
        }


def _with_scan(entry: dict, val: float) -> dict:
    """按扫描值构造一个条目副本（改 scan_param 的 default）。"""
    import copy
    e = copy.deepcopy(entry)
    sp = e.get("scan_param")
    if sp:
        for p in e["parameters"]:
            if p["name"] == sp:
                p["default"] = float(val)
    return e
