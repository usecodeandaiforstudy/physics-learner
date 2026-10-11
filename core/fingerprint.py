"""信息指纹聚合模块。

在信息结构图 G=(V,E,Θ) 上计算各节点的 Fisher 摘要与各边的互信息，
产出对齐 `db1.schema.json` 中 `fingerprint_expectation` 枚举的指纹。

对齐的 quantity 枚举：
    fisher_trace, fisher_det, mi_profile, ib_curve, action_s
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .channels import EncodedGraph, Node
from .fisher import (
    fisher_trace_gaussian,
    fisher_det_gaussian,
    fisher_exponential,
    fisher_poisson,
    fisher_bernoulli,
    fisher_trace_laplace,
    fisher_det_laplace,
    fisher_landau,
    fisher_trace_gamma,
    fisher_det_gamma,
    fisher_trace_beta,
    fisher_det_beta,
    fisher_trace_weibull,
    fisher_det_weibull,
    fisher_trace_vonmises,
    fisher_det_vonmises,
    fisher_geometric,
    fisher_cauchy,
    fisher_rayleigh,
    fisher_logistic,
    fisher_chisq,
    fisher_negbin,
    fisher_lognormal_trace,
    fisher_lognormal_det,
    fisher_mix2_separation,
    fisher_truncnorm_tau,
    fisher_truncgeom_p,
    fisher_invgamma_trace,
    fisher_invgamma_det,
    fisher_invchisq_k,
    fisher_frechet_trace,
    fisher_frechet_det,
    fisher_rician_kappa,
    fisher_wrapnorm_kappa,
)
from .ib import ib_curve
from .mutual_info import mi_gaussian_analytic

__all__ = [
    "compute_node_fisher",
    "compute_graph_fingerprint",
]


# 分布族 ID 常量（与 conventions.yaml 对齐）
_DIST_GAUSS = "DIST-GAUSS"
_DIST_GIBBS = "DIST-GIBBS"
_DIST_EXPONENTIAL = "DIST-EXPONENTIAL"  # 指数分布（单参数），区别于指数族 DIST-EXP
_DIST_POISSON = "DIST-POISSON"
_DIST_BERNOULLI = "DIST-BERNOULLI"
_DIST_LAPLACE = "DIST-LAPLACE"
_DIST_LANDAU = "DIST-LANDAU"
_DIST_GAMMA = "DIST-GAMMA"
_DIST_BETA = "DIST-BETA"
_DIST_WEIBULL = "DIST-WEIBULL"
_DIST_VONMISES = "DIST-VONMISES"
_DIST_GEOMETRIC = "DIST-GEOMETRIC"
_DIST_CAUCHY = "DIST-CAUCHY"
_DIST_RAYLEIGH = "DIST-RAYLEIGH"
_DIST_LOGISTIC = "DIST-LOGISTIC"
_DIST_CHISQ = "DIST-CHISQ"
_DIST_NEGBIN = "DIST-NEGBIN"
_DIST_LOGNORM = "DIST-LOGNORM"
_DIST_MIX2 = "DIST-MIX2"
_DIST_TRUNCNORM = "DIST-TRUNCNORM"
_DIST_TRUNCGEOM = "DIST-TRUNCGEOM"
_DIST_INVGAMMA = "DIST-INVGAMMA"
_DIST_INVCHISQ = "DIST-INVCHISQ"
_DIST_FRECHET = "DIST-FRECHET"
_DIST_RICIAN = "DIST-RICIAN"
_DIST_WRAPNORM = "DIST-WRAPNORM"


def compute_node_fisher(node: Node) -> dict[str, float]:
    """计算单节点的 Fisher 摘要（fisher_trace / fisher_det）。

    支持的分布族（解析式）：
        - DIST-GAUSS       : N(mu, sigma)（trace=3/σ², det=2/σ⁴）
        - DIST-EXPONENTIAL : Exp(λ)（trace=det=1/λ²）
        - DIST-POISSON     : Poisson(λ)（trace=det=1/λ）
        - DIST-BERNOULLI   : Bern(p)（trace=det=1/(p(1-p))）
        - DIST-LAPLACE     : Laplace(μ,b)（trace=2/b², det=1/b⁴）
        - DIST-GIBBS       : 归入高斯（平衡态边缘为高斯）
    其他分布族抛 NotImplementedError，提示走 fisher.fisher_numerical 数值兜底。

    参数
    ----
    node : Node
        信息结构图节点，须含 distribution_family 与 params。

    返回
    ----
    dict[str, float]
        {"fisher_trace": ..., "fisher_det": ...}
    """
    family = node.distribution_family

    if family in (_DIST_GAUSS, _DIST_GIBBS):
        sigma = float(node.params["sigma"])
        return {
            "fisher_trace": fisher_trace_gaussian(sigma),
            "fisher_det": fisher_det_gaussian(sigma),
        }

    if family == _DIST_EXPONENTIAL:
        rate = float(node.params["rate"])
        i = fisher_exponential(rate)
        return {"fisher_trace": i, "fisher_det": i}

    if family == _DIST_POISSON:
        rate = float(node.params["rate"])
        i = fisher_poisson(rate)
        return {"fisher_trace": i, "fisher_det": i}

    if family == _DIST_BERNOULLI:
        p = float(node.params["p"])
        i = fisher_bernoulli(p)
        return {"fisher_trace": i, "fisher_det": i}

    if family == _DIST_LAPLACE:
        b = float(node.params["b"])
        return {
            "fisher_trace": fisher_trace_laplace(b),
            "fisher_det": fisher_det_laplace(b),
        }

    if family == _DIST_LANDAU:
        a = float(node.params["a"])
        b = float(node.params["b"])
        kT = float(node.params.get("kT", 1.0))
        i = fisher_landau(a, b, kT)
        return {"fisher_trace": i, "fisher_det": i}

    if family == _DIST_GAMMA:
        k = float(node.params["k"])
        theta = float(node.params["theta"])
        return {
            "fisher_trace": fisher_trace_gamma(k, theta),
            "fisher_det": fisher_det_gamma(k, theta),
        }

    if family == _DIST_BETA:
        alpha = float(node.params["alpha"])
        beta = float(node.params["beta"])
        return {
            "fisher_trace": fisher_trace_beta(alpha, beta),
            "fisher_det": fisher_det_beta(alpha, beta),
        }

    if family == _DIST_WEIBULL:
        shape = float(node.params["shape"])
        scale = float(node.params["scale"])
        return {
            "fisher_trace": fisher_trace_weibull(shape, scale),
            "fisher_det": fisher_det_weibull(shape, scale),
        }

    if family == _DIST_VONMISES:
        kappa = float(node.params["kappa"])
        return {
            "fisher_trace": fisher_trace_vonmises(kappa),
            "fisher_det": fisher_det_vonmises(kappa),
        }

    if family == _DIST_GEOMETRIC:
        p = float(node.params["p"])
        i = fisher_geometric(p)
        return {"fisher_trace": i, "fisher_det": i}

    if family == _DIST_CAUCHY:
        gamma = float(node.params["gamma"])
        i = fisher_cauchy(gamma)
        return {"fisher_trace": i, "fisher_det": i}

    if family == _DIST_RAYLEIGH:
        sigma = float(node.params["sigma"])
        i = fisher_rayleigh(sigma)
        return {"fisher_trace": i, "fisher_det": i}

    if family == _DIST_LOGISTIC:
        scale = float(node.params["scale"])
        i = fisher_logistic(scale)
        return {"fisher_trace": i, "fisher_det": i}

    if family == _DIST_CHISQ:
        k = float(node.params["k"])
        i = fisher_chisq(k)
        return {"fisher_trace": i, "fisher_det": i}

    if family == _DIST_NEGBIN:
        # k 为结构常数（成功次数），Fisher 摘要为关于 p 的条件信息
        k = float(node.params["k"])
        p = float(node.params["p"])
        i = fisher_negbin(k, p)
        return {"fisher_trace": i, "fisher_det": i}

    if family == _DIST_LOGNORM:
        sigma = float(node.params["sigma"])
        return {
            "fisher_trace": fisher_lognormal_trace(sigma),
            "fisher_det": fisher_lognormal_det(sigma),
        }

    if family == _DIST_MIX2:
        d = float(node.params["d"])
        sigma = float(node.params.get("sigma", 1.0))
        i = fisher_mix2_separation(d, sigma)
        return {"fisher_trace": i, "fisher_det": i}

    if family == _DIST_TRUNCNORM:
        tau = float(node.params["tau"])
        mu = float(node.params.get("mu", 0.0))
        sigma = float(node.params.get("sigma", 1.0))
        i = fisher_truncnorm_tau(tau, mu, sigma)
        return {"fisher_trace": i, "fisher_det": i}

    if family == _DIST_TRUNCGEOM:
        k = float(node.params["k"])
        p = float(node.params["p"])
        i = fisher_truncgeom_p(k, p)
        return {"fisher_trace": i, "fisher_det": i}

    if family == _DIST_INVGAMMA:
        k = float(node.params["k"])
        theta = float(node.params["theta"])
        return {
            "fisher_trace": fisher_invgamma_trace(k, theta),
            "fisher_det": fisher_invgamma_det(k, theta),
        }

    if family == _DIST_INVCHISQ:
        k = float(node.params["k"])
        i = fisher_invchisq_k(k)
        return {"fisher_trace": i, "fisher_det": i}

    if family == _DIST_FRECHET:
        shape = float(node.params["shape"])
        scale = float(node.params["scale"])
        return {
            "fisher_trace": fisher_frechet_trace(shape, scale),
            "fisher_det": fisher_frechet_det(shape, scale),
        }

    if family == _DIST_RICIAN:
        kappa = float(node.params["kappa"])
        i = fisher_rician_kappa(kappa)
        return {"fisher_trace": i, "fisher_det": i}

    if family == _DIST_WRAPNORM:
        kappa = float(node.params["kappa"])
        i = fisher_wrapnorm_kappa(kappa)
        return {"fisher_trace": i, "fisher_det": i}

    raise NotImplementedError(
        f"分布族 {family} 尚无解析 Fisher；请用 fisher.fisher_numerical 兜底，"
        "或先补充分布族的解析式（见 conventions.yaml 第 2 组）"
    )


def compute_graph_fingerprint(graph: EncodedGraph) -> dict[str, Any]:
    """在信息结构图上聚合计算全部指纹。

    对每个节点计算 Fisher 摘要；对每条边若提供了高斯联合协方差，
    计算互信息；最后汇总 mi_profile（各边互信息列表）。

    参数
    ----
    graph : EncodedGraph
        信息结构图。

    返回
    ----
    dict
        {
          "nodes": {node_id: {"fisher_trace", "fisher_det"}, ...},
          "mi_profile": [边的互信息列表，按 edges 顺序],
          "ib_curve": None（需显式信道时另行调用 ib.ib_curve）,
          "action_s": None（需参数变化率时另行调用 action.info_action）,
        }
    """
    result: dict[str, Any] = {
        "nodes": {},
        "mi_profile": [],
        "ib_curve": None,
        "action_s": None,
    }

    # 1) 节点 Fisher 摘要
    for node in graph.nodes:
        node.fingerprint = compute_node_fisher(node)
        result["nodes"][node.node_id] = node.fingerprint

    # 2) 边互信息（若边携带高斯联合协方差）
    for edge in graph.edges:
        cov = getattr(edge.channel, "cov", None) if edge.channel is not None else None
        if cov is not None:
            block_shape = getattr(edge.channel, "block_shape", (1, 1))
            edge.mutual_information = mi_gaussian_analytic(
                np.asarray(cov), block_shape
            )
        result["mi_profile"].append(edge.mutual_information)

    return result
