"""核心信息计算库。

提供信息力学框架下的基础数学计算：
- Fisher 信息度量（fisher）
- 互信息估计（mutual_info）
- 信息瓶颈 IB（ib）
- 信息作用量（action）
- 信息指纹聚合（fingerprint）
- 信息结构图编码（channels）

所有信息量一律自然对数（nats），遵循 db1/conventions.yaml。
"""

from __future__ import annotations

from . import action, channels, fingerprint, fisher, ib, mutual_info
from .action import info_action, ib_potential
from .channels import ChannelTemplate, Edge, EncodedGraph, Node
from .fisher import (
    fisher_det_gaussian,
    fisher_gaussian_matrix,
    fisher_numerical,
    fisher_trace_gaussian,
    fisher_exponential,
    fisher_poisson,
    fisher_bernoulli,
    fisher_laplace_matrix,
    fisher_trace_laplace,
    fisher_det_laplace,
    fisher_landau,
    fisher_landau_monte_carlo,
    fisher_landau_analytic,
    fisher_chisq,
    fisher_negbin,
    fisher_lognormal_trace,
    fisher_lognormal_det,
    fisher_mix2_separation,
    fisher_mix2_monte_carlo,
    fisher_truncnorm_tau,
    fisher_truncgeom_p,
    fisher_truncgeom_monte_carlo,
    fisher_geometric,
    fisher_invgamma_trace,
    fisher_invgamma_det,
    fisher_trace_gamma,
    fisher_det_gamma,
    fisher_invchisq_k,
    fisher_frechet_trace,
    fisher_frechet_det,
    fisher_frechet_trace_mc,
    fisher_det_weibull,
    fisher_rician_kappa,
    fisher_rician_kappa_mc,
    fisher_trace_vonmises,
    fisher_det_vonmises,
    fisher_vonmises_kappa,
    fisher_wrapnorm_kappa,
    fisher_wrapnorm_kappa_mc,
)
from .fingerprint import compute_graph_fingerprint, compute_node_fisher
from .ib import ib_blahut_arimoto, ib_curve
from .mutual_info import mi_binning, mi_gaussian_analytic, mi_ksg

__version__ = "0.1.0"

__all__ = [
    # 模块
    "action",
    "channels",
    "fingerprint",
    "fisher",
    "ib",
    "mutual_info",
    # Fisher
    "fisher_gaussian_matrix",
    "fisher_trace_gaussian",
    "fisher_det_gaussian",
    "fisher_exponential",
    "fisher_poisson",
    "fisher_bernoulli",
    "fisher_laplace_matrix",
    "fisher_trace_laplace",
    "fisher_det_laplace",
    "fisher_landau",
    "fisher_landau_monte_carlo",
    "fisher_landau_analytic",
    "fisher_chisq",
    "fisher_negbin",
    "fisher_lognormal_trace",
    "fisher_lognormal_det",
    "fisher_numerical",
    # 互信息
    "mi_gaussian_analytic",
    "mi_ksg",
    "mi_binning",
    # 信息瓶颈
    "ib_blahut_arimoto",
    "ib_curve",
    # 信息作用量
    "info_action",
    "ib_potential",
    # 图编码
    "ChannelTemplate",
    "Node",
    "Edge",
    "EncodedGraph",
    # 指纹
    "compute_node_fisher",
    "compute_graph_fingerprint",
]
