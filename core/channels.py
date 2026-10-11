"""信息结构图编码模块（B 网络 SystemEncoder 的接口预留）。

定义信息结构图 G=(V,E,Θ) 的轻量数据结构，作为指纹计算的输入。
遵循 `db1/conventions.yaml` 第 5 组约定：
- 节点 ID 命名 {family}__{role}__{index}
- 边方向 = 信息流方向 X -> T -> Y
- 节点特征 = 分布族类型、维数、参数化、噪声类型
- 边特征 = 互信息、容量、KL、DPI 裕度

注意：完整的"物理系统 -> 信息结构图"编码逻辑属于 B 网络（SystemEncoder），
本模块只提供数据结构与通道模板 ID 枚举，供 fingerprint 模块消费。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "ChannelTemplate",
    "Node",
    "Edge",
    "GaussianChannel",
    "EncodedGraph",
]


# ---------------------------------------------------------------------------
# 通道模板 ID 枚举（对齐 conventions.yaml 与架构文档 §2.3.4）
# ---------------------------------------------------------------------------
class ChannelTemplate:
    """通道模板 ID 常量。

    当前只注册了约定中明确出现过的模板；新模板须经评审后加入，
    并同步更新 db1.schema.json 的 channel_template pattern 匹配范围。
    """

    DIFFUSION = "CT-DIFF-001"      # 扩散型（SDE）：OU 过程 X -> T -> Y
    DIFFUSION_PDE = "CT-DIFF-002"  # 扩散型（PDE）：纯扩散方程（加性高斯扩散信道）
    EQUILIBRIUM = "CT-EQ-001"      # 平衡态：Gibbs 态测量链（谐振子）
    MECHANICAL = "CT-MECH-001"     # 力学弹性平衡：涨落-响应（线性弹性）
    MEASUREMENT = "CT-MEAS-001"    # 测量链：观测噪声信道
    DECAY = "CT-DECAY-001"         # 衰减型：指数寿命分布（放射性衰变）
    COUNTING = "CT-COUNT-001"      # 计数型：泊松计数分布（光子/粒子探测）
    TWO_LEVEL = "CT-TWOLVL-001"    # 二能级型：伯努利占据分布（自旋/能级）
    LAPLACE = "CT-LAPLACE-001"     # 拉普拉斯型：双指数差分布（到达时间差）
    RESONATOR = "CT-LC-001"        # 谐振腔型：LC 电路热涨落（高斯）
    LANDAU = "CT-LANDAU-001"       # 朗道型：序参量分布（相变）
    GAMMA = "CT-GAMMA-001"         # 伽马型：形状-尺度等待时间（Erlang/chi2/水文）
    BETA = "CT-BETA-001"           # 贝塔型：有界支撑占据分数（占据/纯度/混合权重）
    WEIBULL = "CT-WEIBULL-001"     # 威布尔型：失效寿命分布（断裂/疲劳）
    VONMISES = "CT-VONMISES-001"   # 冯·米塞斯型：圆上角度分布（相位/取向噪声）
    GEOMETRIC = "CT-GEOM-001"      # 几何型：首次成功等待（离散，探测效率/缺陷搜索）
    CAUCHY = "CT-CAUCHY-001"       # 柯西型：重尾共振线形（谱线宽度/柯西噪声）
    RAYLEIGH = "CT-RAYLEIGH-001"   # 瑞利型：幅值分布（风速/散射/湍流脉动）
    LOGISTIC = "CT-LOGISTIC-001"   # 逻辑斯蒂型：S 形生长（激活阈值/相界序参量）
    CHISQ = "CT-CHISQ-001"         # 卡方型：标准二次型自由度（白化残差能量/GOF 统计量）
    NEGBIN = "CT-NEGBIN-001"       # 负二项型：成簇计数（热光光子/闪烁/暗计数成簇）
    LOGNORM = "CT-LOGNORM-001"     # 对数正态型：乘性噪声（对数均值回复增长/粒径/阴影衰落）
    MIX2 = "CT-MIX-001"           # 混合型：双高斯混合测量（双模态/双稳/外差，扩族批次 4）
    TRUNCNORM = "CT-TRUNCNORM-001"  # 截断型：单边截断观测（窗口化/阈值化，扩族批次 4）
    TRUNCGEOM = "CT-TRUNCGEOM-001"  # 截断几何型：有限支撑等待（有限缓冲/窗口计数，R4b）
    INVGAMMA = "CT-INVGAMMA-001"    # 逆伽马型：精度/尺度倒数分布（传导率/方差标度，R4c）
    INVCHISQ = "CT-INVCHISQ-001"    # 逆卡方型：方差先验/精度二次型（k 单列，R4d）
    FRECHET = "CT-FRECHET-001"      # 弗雷歇型：上限极值（风速极值/洪水峰值，R4d）
    RICIAN = "CT-RICIAN-001"        # 莱斯型：K 因子衰落（主径+散射，R4d vonmises 同构）
    WRAPNORM = "CT-WRAPNORM-001"    # 卷绕高斯型：圆上相位扩散（R4d vonmises 三层同构）


# ---------------------------------------------------------------------------
# 图结构数据类
# ---------------------------------------------------------------------------
@dataclass
class Node:
    """信息结构图节点 = 一个随机变量 / 表示层。"""

    node_id: str                      # {family}__{role}__{index}
    distribution_family: str          # 引用 conventions.yaml distribution_families 的 id
    params: dict[str, Any] = field(default_factory=dict)
    noise_type: str | None = None     # 噪声类型（如 additive_gaussian）
    dimension: int = 1

    # 可计算的信息指纹（由 fingerprint 模块填充）
    fingerprint: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """信息结构图边 = 信息处理通道，方向为信息流方向。"""

    src: str
    dst: str
    channel: Any = None               # 条件分布 p(dst | src) 的定义
    capacity: float | None = None     # 通道容量
    kl_divergence: float | None = None
    dpi_margin: float | None = None   # DPI 裕度

    # 边的互信息（由 fingerprint 模块填充）
    mutual_information: float | None = None


@dataclass
class GaussianChannel:
    """高斯信道 p(dst | src)，以联合协方差定义。

    供 fingerprint 模块计算边的互信息（mi_gaussian_analytic）。
    """

    cov: np.ndarray                 # 联合协方差 [src, dst]，形状 (d_src+d_dst, d_src+d_dst)
    block_shape: tuple[int, int] = (1, 1)  # (d_src, d_dst)


@dataclass
class EncodedGraph:
    """信息结构图 G=(V,E,Θ)，带全局约束。"""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    beta: float = 1.0                 # IB 权衡参数
    total_capacity: float | None = None
    max_nodes: int = 10               # 节点数上限（控制搜索空间）
    metadata: dict[str, Any] = field(default_factory=dict)  # 编码元数据（审计用）

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def node_by_id(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        raise KeyError(f"节点 {node_id} 不存在")
