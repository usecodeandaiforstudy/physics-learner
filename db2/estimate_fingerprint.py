# -*- coding: utf-8 -*-
"""数据指纹估计器（缺口①，2026-10-02）：从模拟/实验样本估计 Fisher。

两条路线（v0 都实现，互为交叉验证）：
  1. **非参数**：KDE 密度 + 参数中心差分 → F = ∫ (∂_θ p)² / p dφ
     （不做分布族假设，纯数据驱动）；
  2. **参数化**：MLE 拟合分布参数 → 解析 Fisher 公式
     （生产实用路线：先拟合定律，再算指纹）。

v0 边界（文档化）：θ = landau 的 a（b、kT 为已知结构常数）；真实模拟/实验
数据的接入（文件格式、来源核验、红线）见 `db1/ingest_sim_data.py`。
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["simulate_landau", "estimate_fisher_nonparametric",
           "estimate_fisher_parametric", "kde_density"]


def simulate_landau(a: float, b: float, kT: float, n: int,
                    phi_lo: float = -3.5, phi_hi: float = 3.5,
                    n_grid: int = 4000, seed: int = 0) -> np.ndarray:
    """landau 序参量样本（网格 CDF 逆变换；v0 的「模拟」= 定律驱动采样）。"""
    rng = np.random.default_rng(seed)
    phi = np.linspace(phi_lo, phi_hi, n_grid)
    logp = -(a * phi ** 2 / 2.0 + b * phi ** 4 / 4.0) / kT
    logp -= logp.max()
    p = np.exp(logp)
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    return np.interp(rng.uniform(0.0, 1.0, n), cdf, phi)


def kde_density(samples: np.ndarray, grid: np.ndarray,
                bw_method: float | str = "silverman") -> np.ndarray:
    """高斯核密度估计（scipy gaussian_kde 封装）。"""
    return stats.gaussian_kde(samples, bw_method=bw_method)(grid)


def estimate_fisher_nonparametric(a: float, b: float, kT: float, n: int,
                                  h: float = 0.05, seed: int = 0,
                                  grid: np.ndarray | None = None,
                                  tail_frac: float = 0.01) -> float:
    """非参数 Fisher 估计：F = ∫ (∂_a p)² / p dφ（KDE + 中心差分）。

    两个方差/偏差控制（v13 首跑 43% 误差的修正）：
      1. **共同随机数（CRN）**：a−h / a / a+h 三点用同一组分位数采样
         （同一 seed → 同一均匀抽样），差分只反映参数移动，不含抽样噪声；
      2. **积分域限制**：仅在高密度区（p_mid > tail_frac × max）积分——
         Fisher 被积函数的质量本就在密度所在处，尾部 (dp)²/p 是纯噪声
         且系统性高估（首跑三工况一致高估 17–74%）。
    """
    if grid is None:
        grid = np.linspace(-3.2, 3.2, 400)
    # CRN：同 seed → 同分位数
    s_lo = simulate_landau(a - h, b, kT, n, seed=seed)
    s_mid = simulate_landau(a, b, kT, n, seed=seed)
    s_hi = simulate_landau(a + h, b, kT, n, seed=seed)
    p_lo = kde_density(s_lo, grid)
    p_mid = kde_density(s_mid, grid)
    p_hi = kde_density(s_hi, grid)
    dp = (p_hi - p_lo) / (2.0 * h)
    integrand = dp ** 2 / np.maximum(p_mid, 1e-12)
    mask = p_mid > tail_frac * float(p_mid.max())
    return float(np.trapezoid(np.where(mask, integrand, 0.0), grid))


def estimate_fisher_parametric(samples: np.ndarray, b: float, kT: float,
                               a_bounds: tuple[float, float] = (-4.0, 4.0)
                               ) -> tuple[float, float]:
    """参数化路线：MLE 拟合 a → 解析 Fisher 公式。

    返回 (a_hat, F_analytic)。似然由定律直接给出（无初等归一化常数亦可，
    因归一化与 a 无关的假设不成立——故用网格数值归一化，见实现）。
    """
    from core.fisher import fisher_landau_analytic
    phi = np.linspace(-3.5, 3.5, 4000)

    def neg_loglik(a: float) -> float:
        F = a * phi ** 2 / 2.0 + phi ** 4 / 4.0
        logp = -F / kT
        logp -= logp.max()
        p = np.exp(logp)
        p /= (p.sum() * (phi[1] - phi[0]))
        return -float(np.mean(np.log(np.interp(samples, phi, p) + 1e-300)))

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(neg_loglik, bounds=a_bounds, method="bounded")
    a_hat = float(res.x)
    return a_hat, float(fisher_landau_analytic(a_hat, b, kT))
