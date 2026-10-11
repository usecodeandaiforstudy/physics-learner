"""互信息估计模块。

遵循 `db1/conventions.yaml` 第 1 组约定：
- 互信息定义：I(X;Y) = KL[p(x,y) || p(x)p(y)]，单位 nats
- 估计器按分布族指定：
    - 高斯 → 解析（analytic）
    - 连续非高斯 → KSG（k 近邻）
    - 离散 → binning（直方图）
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma, gammaln

__all__ = [
    "mi_gaussian_analytic",
    "mi_ksg",
    "mi_binning",
]


# ---------------------------------------------------------------------------
# 高斯解析 MI
# ---------------------------------------------------------------------------
def mi_gaussian_analytic(
    cov: np.ndarray,
    block_shape: tuple[int, int] = (1, 1),
) -> float:
    """联合高斯 (X, Y) 的互信息，解析计算。

    对于联合高斯，若 X、Y 的边际协方差为 Σ_XX、Σ_YY，联合协方差为 Σ：
        I(X;Y) = -1/2 * ln( det(Σ) / (det(Σ_XX) * det(Σ_YY)) )
    单位 nats（自然对数，符合约定）。

    参数
    ----
    cov : np.ndarray
        联合协方差矩阵，形状 (d, d)，d = d_X + d_Y。
    block_shape : tuple[int, int]
        (d_X, d_Y)，X 与 Y 的维数；前 d_X 维归 X，后 d_Y 维归 Y。

    返回
    ----
    float
        互信息（nats）。
    """
    cov = np.asarray(cov, dtype=float)
    d_x, d_y = block_shape
    d = d_x + d_y
    if cov.shape != (d, d):
        raise ValueError(f"cov 形状 {cov.shape} 与 block_shape 合计 {d} 不符")

    det_joint = np.linalg.det(cov)
    det_x = np.linalg.det(cov[:d_x, :d_x])
    det_y = np.linalg.det(cov[d_x:, d_x:])

    # 数值防护：行列式须为正（正定协方差）
    if det_joint <= 0 or det_x <= 0 or det_y <= 0:
        raise ValueError("协方差矩阵非正定，无法计算高斯互信息")

    return -0.5 * np.log(det_joint / (det_x * det_y))


# ---------------------------------------------------------------------------
# KSG 估计器（连续非高斯）
# ---------------------------------------------------------------------------
def mi_ksg(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    rng: np.random.Generator | None = None,
) -> float:
    """KSG 互信息估计器（基于 Kozachenko–Leonenko 熵估计）。

    实现 KSG（Kraskov–Stögbauer–Grassberger 2004）的 KL 熵估计形式：
        I(X;Y) = H(X) + H(Y) - H(X,Y)
    其中每个熵用 k 近邻的 Kozachenko–Leonenko 估计：
        H(X) ≈ -ψ(k) + ψ(N) + log c_d + (d/N) Σ_i log ε_x(i)
        c_d 为 d 维单位球体积，ε_x(i) 为 X 子空间第 k 近邻距离。

    该形式是 KSG 估计的直接且透明的实现，适用于连续非高斯分布族
    （对齐 conventions.yaml 第 1 组的估计器约定）。

    参数
    ----
    x, y : np.ndarray
        样本，形状 (n, d_x) 与 (n, d_y)。
    k : int
        近邻数。
    rng : np.random.Generator, 可选
        保留参数（当前实现确定性，未使用）。

    返回
    ----
    float
        互信息估计（nats）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("x 与 y 样本数不一致")
    if k >= n:
        raise ValueError(f"k={k} 须小于样本数 n={n}")

    d_x = x.shape[1]
    d_y = y.shape[1]

    # 各空间第 k 近邻距离（排序后索引 k 为第 k 近邻，索引 0 为自身）
    def _knn_dist(data: np.ndarray, k: int) -> np.ndarray:
        diff = data[:, None, :] - data[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        return np.sort(dist, axis=1)[:, k]

    eps_x = _knn_dist(x, k)
    eps_y = _knn_dist(y, k)
    eps_xy = _knn_dist(np.hstack([x, y]), k)

    # d 维单位球体积的对数：log c_d = (d/2) log π - log Γ(d/2 + 1)
    def _log_unit_ball(d: int) -> float:
        return 0.5 * d * np.log(np.pi) - gammaln(0.5 * d + 1.0)

    log_c_x = _log_unit_ball(d_x)
    log_c_y = _log_unit_ball(d_y)
    log_c_xy = _log_unit_ball(d_x + d_y)

    # 三个 Kozachenko–Leonenko 熵估计
    h_x = -digamma(k) + digamma(n) + log_c_x + (d_x / n) * np.sum(np.log(eps_x))
    h_y = -digamma(k) + digamma(n) + log_c_y + (d_y / n) * np.sum(np.log(eps_y))
    h_xy = (
        -digamma(k)
        + digamma(n)
        + log_c_xy
        + ((d_x + d_y) / n) * np.sum(np.log(eps_xy))
    )

    mi = h_x + h_y - h_xy
    # 数值防护：互信息非负
    return float(max(mi, 0.0))


# ---------------------------------------------------------------------------
# binning 估计器（离散）
# ---------------------------------------------------------------------------
def mi_binning(
    x: np.ndarray,
    y: np.ndarray,
    bins: int | tuple[int, int] = 10,
) -> float:
    """基于直方图的互信息估计，适用于离散或可离散化的变量。

    参数
    ----
    x, y : np.ndarray
        样本，形状 (n,) 或 (n, 1)。
    bins : int 或 tuple[int, int]
        直方图分箱数。

    返回
    ----
    float
        互信息估计（nats）。
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError("x 与 y 样本数不一致")

    # 二维联合直方图
    hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = hist_xy / hist_xy.sum()
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)

    # I = sum p_xy log(p_xy / (p_x p_y))，仅对 p_xy > 0 的项
    mask = p_xy > 0
    p_xy_nonzero = p_xy[mask]
    p_x_p_y = (p_x @ p_y)[mask]
    mi = float(np.sum(p_xy_nonzero * np.log(p_xy_nonzero / p_x_p_y)))
    return mi
