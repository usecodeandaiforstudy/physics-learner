"""信息瓶颈（IB）计算模块。

遵循 `db1/conventions.yaml` 第 1 组约定：
- IB 目标：L_IB = I(X;T) - beta * I(T;Y)（beta > 0 为权衡参数）
- 曲线量：I(X;T) vs I(T;Y)
- beta 网格：logspace(-2, 3, 32)

本模块提供通用 Blahut–Arimoto（BA）不动点算法，适用于离散信道 p(y|x)。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ib_blahut_arimoto",
    "ib_curve",
]


def ib_blahut_arimoto(
    px: np.ndarray,
    p_y_given_x: np.ndarray,
    beta: float,
    n_t: int | None = None,
    n_iter: int = 500,
    tol: float = 1e-12,
    seed: int = 0,
) -> dict:
    """单点 beta 的信息瓶颈 Blahut–Arimoto 迭代。

    优化 q(t|x) 最小化 I(X;T) - beta * I(T;Y)。

    参数
    ----
    px : np.ndarray
        输入分布 p(x)，形状 (nx,)。
    p_y_given_x : np.ndarray
        信道 p(y|x)，形状 (nx, ny)。
    beta : float
        权衡参数，> 0。
    n_t : int, 可选
        压缩表示 T 的簇数，默认等于 nx。
    n_iter : int
        最大迭代次数。
    tol : float
        收敛容差（I(X;T) 的变化）。
    seed : int
        随机种子（初始化 q(t|x) 用）。

    返回
    ----
    dict
        {"I_XT": float, "I_TY": float, "beta": float, "converged": bool}
    """
    px = np.asarray(px, dtype=float)
    p_y_given_x = np.asarray(p_y_given_x, dtype=float)
    nx = px.shape[0]
    ny = p_y_given_x.shape[1]

    if abs(px.sum() - 1.0) > 1e-6:
        raise ValueError("px 须归一化")
    if not np.allclose(p_y_given_x.sum(axis=1), 1.0):
        raise ValueError("p(y|x) 每一行须归一化")

    if n_t is None:
        n_t = nx

    rng = np.random.default_rng(seed)

    eps = 1e-12  # 数值防护下限（避免 log(0) 与 0*log(0)=nan）

    # 安全的 p(y|x)：clip 后重新归一化
    pyx = np.clip(p_y_given_x, eps, 1.0)
    pyx /= pyx.sum(axis=1, keepdims=True)
    log_pyx = np.log(pyx)
    # H(Y|x) = -sum_y p(y|x) log p(y|x)，逐行（用于向量化 KL）
    H_yx = -np.sum(pyx * log_pyx, axis=1)  # (nx,)

    # 初始化 q(t|x)：随机 + 归一化
    q_t_given_x = rng.random((nx, n_t))
    q_t_given_x /= q_t_given_x.sum(axis=1, keepdims=True)

    prev_I_XT = np.inf

    for _ in range(n_iter):
        # p(t) = sum_x p(x) q(t|x)
        p_t = px @ q_t_given_x  # (nt,)
        p_t = np.clip(p_t, eps, None)

        # q(x|t) = p(x) q(t|x) / p(t)，形状 (nx, nt)
        q_x_given_t = px[:, None] * q_t_given_x / p_t[None, :]

        # p(y|t) = sum_x p(y|x) q(x|t)，clip 后重新归一化
        p_y_given_t = pyx.T @ q_x_given_t  # (ny, nt)
        p_y_given_t = np.clip(p_y_given_t, eps, None)
        p_y_given_t /= p_y_given_t.sum(axis=0, keepdims=True)
        log_pyt = np.log(p_y_given_t)

        # 向量化 KL[x,t] = -H(Y|x) - sum_y p(y|x) log p(y|t)
        kl = -H_yx[:, None] - (pyx @ log_pyt)  # (nx, nt)
        kl = np.clip(kl, 0.0, None)  # KL 非负（消除数值负值）

        # q(t|x) = p(t) exp(-beta KL) / Z(x)，用 log-sum-exp 稳定
        log_q = np.log(p_t)[None, :] - beta * kl
        log_q -= log_q.max(axis=1, keepdims=True)
        q_t_given_x = np.exp(log_q)
        q_t_given_x /= q_t_given_x.sum(axis=1, keepdims=True)

        # I(X;T) = sum_x p(x) sum_t q(t|x) ln(q(t|x)/p(t))
        log_ratio = np.log(np.clip(q_t_given_x, eps, None) / p_t[None, :])
        I_XT = float(np.sum(px[:, None] * q_t_given_x * log_ratio))

        # I(T;Y) = sum_t p(t) sum_y p(y|t) ln(p(y|t)/p(y))
        # p_y_given_t 形状 (ny, nt)，[y,t] = p(y|t)；p_y[y] = 边缘 p(y)
        p_y = p_y_given_t @ p_t  # (ny,)
        p_y = np.clip(p_y, eps, None)
        log_ratio_ty = np.log(p_y_given_t / p_y[:, None])  # [y,t] = log(p(y|t)/p(y))
        I_TY = float(np.sum(p_t[None, :] * p_y_given_t * log_ratio_ty))

        if abs(I_XT - prev_I_XT) < tol:
            return {"I_XT": I_XT, "I_TY": I_TY, "beta": float(beta), "converged": True}
        prev_I_XT = I_XT

    return {"I_XT": I_XT, "I_TY": I_TY, "beta": float(beta), "converged": False}


def ib_curve(
    px: np.ndarray,
    p_y_given_x: np.ndarray,
    beta_grid: np.ndarray | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 IB 曲线（I(X;T) vs I(T;Y) 的采样点）。

    参数
    ----
    px, p_y_given_x : 同 ib_blahut_arimoto。
    beta_grid : np.ndarray, 可选
        beta 网格；默认 logspace(-2, 3, 32)（对齐 conventions.yaml）。
    **kwargs : 传给 ib_blahut_arimoto。

    返回
    ----
    tuple[np.ndarray, np.ndarray]
        (I_XT 数组, I_TY 数组)，长度与 beta_grid 相同。
    """
    if beta_grid is None:
        beta_grid = np.logspace(-2, 3, 32)

    I_XT_list = []
    I_TY_list = []
    for beta in beta_grid:
        res = ib_blahut_arimoto(px, p_y_given_x, float(beta), **kwargs)
        I_XT_list.append(res["I_XT"])
        I_TY_list.append(res["I_TY"])
    return np.asarray(I_XT_list), np.asarray(I_TY_list)
