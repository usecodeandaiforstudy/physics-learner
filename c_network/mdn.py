# -*- coding: utf-8 -*-
"""C v0（M8，2026-10-02）：领域专用 MDN 解码器（landau 实例）。

C 网络定位（design doc §2.4 + 总览表）：**领域解码器 ×N**——每个领域一个
专用实例，随用随练（需要解码某领域时现训该领域的解码器），**不是通用
解码器**。C v0 的 MDN 仅允许用于管线联调（§2.6.2：覆盖率审计 <90% 即
淘汰，生产 = CFM 条件流匹配）。

本模块 = **landau 领域**的解码器实例：输入 = 该领域的参数 a（标量），
输出 = K 分量高斯混合，训练数据 = 该领域解析分布采样（网格 CDF 逆变换）。
其他领域各自独立实例化（同一代码、各自数据）。

零新依赖（torch/numpy/scipy）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["MixtureDensityNetwork", "sample_landau"]


def sample_landau(a: float, b: float, kT: float, n: int,
                  phi_lo: float = -3.5, phi_hi: float = 3.5,
                  n_grid: int = 4000, seed: int = 0) -> np.ndarray:
    """landau 序参量采样（网格 CDF 逆变换）。

    p(φ) ∝ exp(−(aφ²/2 + bφ⁴/4)/kT)。φ 支撑取 [−3.5, 3.5]（a∈[−2,2],
    b=kT=1 时质量足够）。"""
    rng = np.random.default_rng(seed)
    phi = np.linspace(phi_lo, phi_hi, n_grid)
    logp = -(a * phi ** 2 / 2.0 + b * phi ** 4 / 4.0) / kT
    logp = logp - logp.max()
    p = np.exp(logp)
    cdf = np.cumsum(p)
    cdf = cdf / cdf[-1]
    u = rng.uniform(0.0, 1.0, n)
    return np.interp(u, cdf, phi)


class MixtureDensityNetwork(nn.Module):
    """K 分量高斯混合密度网络（输入：标量参数特征）。"""

    def __init__(self, in_dim: int, n_components: int = 4, hidden: int = 64,
                 seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_components = n_components
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi_head = nn.Linear(hidden, n_components)
        self.mu_head = nn.Linear(hidden, n_components)
        self.logs_head = nn.Linear(hidden, n_components)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        pi = torch.softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h)
        logs = torch.clamp(self.logs_head(h), -3.0, 2.0)
        return pi, mu, logs

    def nll(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pi, mu, logs = self.forward(x)
        comp = torch.distributions.Normal(mu, logs.exp() + 1e-4)
        logp = comp.log_prob(y)   # y (B,1) × batch (B,K) → (B,K)
        return -torch.logsumexp(torch.log(pi + 1e-8) + logp, dim=-1).mean()

    @torch.no_grad()
    def density(self, x: float, grid: np.ndarray) -> np.ndarray:
        """评估预测密度 p(y|x) 于 grid 上（验证用）。"""
        xt = torch.tensor([[x]], dtype=torch.float32)
        pi, mu, logs = self.forward(xt)
        yg = torch.tensor(grid[:, None], dtype=torch.float32)
        comp = torch.distributions.Normal(mu, logs.exp() + 1e-4)
        logp = comp.log_prob(yg)   # yg (G,1) × batch (1,K) → (G,K)
        return torch.exp(torch.logsumexp(
            torch.log(pi + 1e-8) + logp, dim=-1)).numpy().ravel()

    @torch.no_grad()
    def modes(self, x: float, grid: np.ndarray) -> list[float]:
        """预测密度的局部峰值位置（验证用）。"""
        d = self.density(x, grid)
        peaks = []
        for i in range(1, len(d) - 1):
            if d[i] >= d[i - 1] and d[i] > d[i + 1]:
                peaks.append(float(grid[i]))
        return peaks
