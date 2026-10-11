# -*- coding: utf-8 -*-
"""信息拉氏量的**变分层**（2026-10-02，第 116 条）——第 66 条缺口⑥。

**此前状态**：`core/action.py` 定义了
    `L_info = ½ g_ij θ̇^i θ̇^j − [I(X;T) − β·I(T;Y)]`
但它只被**求值**（`a_network/action_reward.py` 轨迹级包装、`db4_writer` 逐经验
记账、合法性诊断）——**没有作为"生成轨迹的原理"**（无 Euler–Lagrange /
测地线求解）。本模块补上这一层。

**变分问题**（固定端点，作用量极值）：
    min_θ  S[θ] = ∫_0^T [ ½ g_ij(θ) θ̇^i θ̇^j − V(θ) ] dt
            s.t. θ(0)=a, θ(T)=b,   V(θ) := I_XT(θ) − β·I_TY(θ)
即「Fisher 度规下的动能 − IB 泛函势能」，**与 conventions.yaml 第 1 组同式**。

**边界条件的两类**：
  - `potential="ib"`：V 由 IB 泛函给出（需 I_XT/I_TY 的解析或查表形式）；
  - `potential="harmonic"`：V = ½k(θ−θ0)²（**有解析解，用于 L1c 校验**）。

**输出**：
  - `theta`：极值轨迹（等时间节点）；
  - `action`：极值作用量 S*；
  - `geodesic_length`：Fisher 度规下的测地线长度 ∫√(g) |dθ|（V=0 时主导）。

**数值方法**：把轨迹离散为 N 个节点（端点固定），用 torch 自动微分对内部节点
做梯度下降（Adam）。**零新依赖**（torch/numpy）。

**可解析校验的三种情形（L1）**：
  ① 平坦度规 g=const、V=0：极值 = 直线，`S* = ½ g Δθ²/T`，长度 = √g·|Δθ|；
  ② 尺度族度规 `g = c/θ²`、V=0：测地线长度 = `√c·|ln(θ_b/θ_a)|`，
     轨迹 `θ(t) = θ_a (θ_b/θ_a)^{t/T}`，`S* = L²/(2T)`；
  ③ 平坦度规 + 谐振势 `V = ½kθ²`：`θ(t) = A cos(ωt) + B sin(ωt)`，`ω=√(k/g)`。

用法：
    from a_network.variational import extremal_trajectory
    r = extremal_trajectory("flat", {}, 0.0, 1.0, T=1.0, g_const=2.0)
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

__all__ = ["extremal_trajectory", "geodesic_length_1d", "euler_lagrange_residual"]


def _g_theta(kind: str, theta: torch.Tensor, params: dict) -> torch.Tensor:
    """Fisher 度规 g(θ)（1-D 情形）。"""
    if kind == "flat":
        return torch.full_like(theta, float(params.get("g", 1.0)))
    if kind == "scale":                     # 尺度族：g = c/θ²
        c = float(params.get("c", 1.0))
        return c / torch.clamp(theta ** 2, min=1e-12)
    if kind == "callable":
        fn = params["g_fn"]
        return torch.tensor([float(fn(float(t.detach()))) for t in theta],
                            dtype=theta.dtype)
    raise ValueError("未知度规类型 %s" % kind)


def _V_theta(kind: str, theta: torch.Tensor, params: dict) -> torch.Tensor:
    """势能 V(θ) = I_XT − β·I_TY 的 1-D 形式。"""
    if kind == "zero":
        return torch.zeros_like(theta)
    if kind == "harmonic":                  # V = ½k(θ−θ0)²
        k = float(params.get("k", 1.0))
        t0 = float(params.get("theta0", 0.0))
        return 0.5 * k * (theta - t0) ** 2
    if kind == "ib_table":                  # 查表（θ 网格 + V 值，线性插值）
        grid = torch.tensor(params["grid"], dtype=theta.dtype)
        vals = torch.tensor(params["V"], dtype=theta.dtype)
        idx = torch.searchsorted(grid, theta.clamp(grid[0], grid[-1]))
        idx = idx.clamp(1, len(grid) - 1)
        x0, x1 = grid[idx - 1], grid[idx]
        y0, y1 = vals[idx - 1], vals[idx]
        w = (theta - x0) / torch.clamp(x1 - x0, min=1e-12)
        return y0 + w * (y1 - y0)
    if kind == "callable":
        fn = params["V_fn"]
        return torch.tensor([float(fn(float(t.detach()))) for t in theta],
                            dtype=theta.dtype)
    raise ValueError("未知势能类型 %s" % kind)


def extremal_trajectory(metric: str = "flat", potential: str = "zero",
                        theta_a: float = 0.0, theta_b: float = 1.0,
                        T: float = 1.0, n_nodes: int = 61,
                        iters: int = 1500, lr: float = 0.02,
                        m_params: dict | None = None,
                        v_params: dict | None = None,
                        seed: int = 0) -> dict:
    """求作用量泛函 `S[θ] = ∫[½g θ̇² − V]dt` 在固定端点下的极值轨迹。

    返回 {"theta", "t", "action", "geodesic_length", "residual"}。
    """
    mp = dict(m_params or {})
    vp = dict(v_params or {})
    torch.manual_seed(seed)
    dt = T / (n_nodes - 1)
    # 初始化：直线（端点固定）
    t = torch.linspace(0.0, T, n_nodes)
    theta = torch.linspace(theta_a, theta_b, n_nodes)
    interior = theta[1:-1].clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([interior], lr=lr)
    for _ in range(iters):
        th = torch.cat([theta[:1], interior, theta[-1:]])
        mid = 0.5 * (th[1:] + th[:-1])
        dth = (th[1:] - th[:-1]) / dt
        g_mid = _g_theta(metric, mid, mp)
        V_mid = _V_theta(potential, mid, vp)
        S = torch.sum((0.5 * g_mid * dth ** 2 - V_mid) * dt)
        opt.zero_grad()
        S.backward()
        opt.step()

    with torch.no_grad():
        th = torch.cat([theta[:1], interior, theta[-1:]]).detach()
        mid = 0.5 * (th[1:] + th[:-1])
        dth = (th[1:] - th[:-1]) / dt
        g_mid = _g_theta(metric, mid, mp)
        V_mid = _V_theta(potential, mid, vp)
        S_star = float(torch.sum((0.5 * g_mid * dth ** 2 - V_mid) * dt))
        geo = float(torch.sum(torch.sqrt(g_mid) * torch.abs(th[1:] - th[:-1])))
    return {"theta": th.numpy(), "t": t.numpy(), "action": S_star,
            "geodesic_length": geo,
            # **推荐用 energy_drift**（一阶量、噪声不敏感）；EL 残差留作参考：
            # 它是二阶量，被 1/dt² 放大，直线解上也能到 O(1)。
            "energy_drift": energy_drift(metric, potential, th, dt, mp, vp),
            "residual": euler_lagrange_residual(metric, potential, th, dt,
                                                mp, vp)}


def geodesic_length_1d(metric: str, theta_a: float, theta_b: float,
                       n: int = 2001, params: dict | None = None) -> float:
    """测地线长度（1-D 解析/数值）：`∫√(g) dθ`。"""
    xs = np.linspace(theta_a, theta_b, n)
    x = torch.tensor(xs, dtype=torch.float32)
    g = _g_theta(metric, x, dict(params or {})).numpy()
    return float(np.trapezoid(np.sqrt(g), xs))


def hamiltonian(metric: str, potential: str, theta: torch.Tensor, dt: float,
                mp: dict, vp: dict) -> torch.Tensor:
    """哈密顿量 H = ½ g θ̇² + V（L 不显含 t ⟹ 极值解上 H 应为常数）。"""
    th = theta.detach()
    dth = torch.zeros_like(th)
    dth[1:-1] = (th[2:] - th[:-2]) / (2 * dt)
    dth[0] = (th[1] - th[0]) / dt
    dth[-1] = (th[-1] - th[-2]) / dt
    g = _g_theta(metric, th, mp)
    V = _V_theta(potential, th, vp)
    return 0.5 * g * dth ** 2 + V


def energy_drift(metric: str, potential: str, theta: torch.Tensor, dt: float,
                 mp: dict, vp: dict) -> float:
    """**能量漂移（推荐的收敛诊断，2026-10-02 加）**：

        drift = max|H − mean(H)| / (|mean(H)| + scale)

    L 不显含 t ⟹ H 守恒。用 H 而不用 EL 残差的原因：EL 残差是**二阶**量
    （含 θ̈），被离散噪声放大 1/dt² 倍——实测 L1a 的直线解上 EL 残差仍达
    1.83，而其轨迹偏差仅 7.6e-4（≈ 7.6e-4·60² ≈ 2.7 的放大）⟹
    EL 残差**不适合作收敛判据**；H 是一阶量，噪声不敏感。
    """
    H = hamiltonian(metric, potential, theta, dt, mp, vp)
    scale = float(H.abs().mean()) + 1e-9
    return float((H - H.mean()).abs().max() / scale)


def euler_lagrange_residual(metric: str, potential: str, theta: torch.Tensor,
                            dt: float, mp: dict, vp: dict) -> float:
    """Euler–Lagrange 残差（1-D）：

        d/dt( g θ̇ ) − ½ g' θ̇² + V'(θ) = 0

    返回轨迹内部点上的**最大绝对残差**（应随迭代下降；极值解上应 ≈0）。
    """
    th = theta.clone().detach().requires_grad_(True)
    t_idx = torch.arange(len(th), dtype=th.dtype) * dt
    # 数值微分（中心差分）
    dth = torch.zeros_like(th)
    dth[1:-1] = (th[2:] - th[:-2]) / (2 * dt)
    dth[0] = (th[1] - th[0]) / dt
    dth[-1] = (th[-1] - th[-2]) / dt
    g = _g_theta(metric, th, mp)
    # g' 与 V' 用有限差分（避免依赖可调用导数的解析式）
    eps = 1e-4
    gp = (_g_theta(metric, th + eps, mp) - _g_theta(metric, th - eps, mp)) / (2 * eps)
    Vp = (_V_theta(potential, th + eps, vp) - _V_theta(potential, th - eps, vp)) / (2 * eps)
    p = g * dth                                  # 共轭动量
    dp = torch.zeros_like(p)
    dp[1:-1] = (p[2:] - p[:-2]) / (2 * dt)
    dp[0] = (p[1] - p[0]) / dt
    dp[-1] = (p[-1] - p[-2]) / dt
    res = dp - 0.5 * gp * dth ** 2 + Vp
    return float(res[1:-1].abs().max())


if __name__ == "__main__":
    r = extremal_trajectory("flat", "zero", 0.0, 1.0, T=1.0,
                            m_params={"g": 2.0})
    print("快速自检：平坦 g=2、V=0、θ:0→1、T=1")
    print("  S* = %.6f（解析 ½gΔθ²/T = %.6f）"
          % (r["action"], 0.5 * 2.0 * 1.0 ** 2 / 1.0))
    print("  测地线长度 = %.6f（解析 √g|Δθ| = %.6f）"
          % (r["geodesic_length"], math.sqrt(2.0)))
    print("  能量漂移 = %.3g（应 ≈0；EL 残差 %.3g 为二阶量、受离散噪声放大）"
          % (r["energy_drift"], r["residual"]))
