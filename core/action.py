"""信息作用量计算模块。

遵循 `db1/conventions.yaml` 第 1 组约定（action 符号约定）：
    L_info = (1/2) g_ij * qdot_i * qdot_j  -  [I(X;T) - beta * I(T;Y)]

其中：
- 动能项 = (1/2) * theta_dot^T g theta_dot（Fisher 度量二次型）
- 势能项 = IB 泛函 I(X;T) - beta * I(T;Y)
- L = 动能项 - 势能项
"""

from __future__ import annotations

import numpy as np

__all__ = ["info_action", "ib_potential"]


def ib_potential(I_XT: float, I_TY: float, beta: float) -> float:
    """IB 泛函（势能项）：I(X;T) - beta * I(T;Y)。"""
    return float(I_XT) - float(beta) * float(I_TY)


def info_action(
    fisher_g: np.ndarray,
    theta_dot: np.ndarray,
    I_XT: float,
    I_TY: float,
    beta: float,
) -> float:
    """信息拉格朗日量（信息作用量密度）。

    参数
    ----
    fisher_g : np.ndarray
        Fisher 度量矩阵，形状 (d, d)。
    theta_dot : np.ndarray
        参数变化率向量，形状 (d,)。
    I_XT, I_TY : float
        压缩量与任务相关量（nats）。
    beta : float
        IB 权衡参数。

    返回
    ----
    float
        L_info 值。
    """
    fisher_g = np.asarray(fisher_g, dtype=float)
    theta_dot = np.asarray(theta_dot, dtype=float)
    kinetic = 0.5 * float(theta_dot @ fisher_g @ theta_dot)
    potential = ib_potential(I_XT, I_TY, beta)
    return kinetic - potential
