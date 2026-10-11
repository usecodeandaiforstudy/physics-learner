# -*- coding: utf-8 -*-
"""C v1（2026-10-02）：条件流匹配解码器（CFM / rectified flow）。

**定位**（design doc §2.4/§2.6.2）：C = 领域解码器 ×N，**生产形态 = CFM**；
MDN 仅允许管线联调，**覆盖率审计 <90% 即淘汰**。本模块 = landau 领域的
CFM 实例（与 `mdn.py` 同任务、同数据，便于同口径对照）。

**相对 MDN 的两个结构性优势**：
  1. **无分量数上限**：MDN 用 K 个高斯近似任意密度（多模/尖峰需 K 大，
     且易模式塌缩）；CFM 学的是**速度场**，密度形态由 ODE 决定，无此上限；
  2. **精确对数密度**：用瞬时变分公式
     `d/dt[φ_t, log p_t] = [v(φ_t,t), −∇·v(φ_t,t)]`
     从 `φ_0~N(0,1)`、`log p_0 = log N(φ_0)` 积分到 t=1 →
     **`log p_1(φ)` 是精确值**（1-D 下散度 = dv/dφ，用 autograd 精确算），
     于是可与 MDN 的解析密度做**同口径的重叠系数比较**。

零新依赖（torch/numpy）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["ConditionalFlowMatching"]


class ConditionalFlowMatching(nn.Module):
    """条件流匹配：v_θ(φ, t; a) 学线性插值路径的速度场。

    路径：`φ_t = (1−t)·φ_0 + t·φ_1`，`φ_0~N(0,1)`、`φ_1~p_data(·|a)`
    目标速度：`φ_1 − φ_0`（rectified flow 的标准条件速度）。
    """

    def __init__(self, in_dim: int = 1, hidden: int = 96, seed: int = 0,
                 time_features: int = 8, cond_dim: int = 1):
        """**多维推广（2026-10-02，第 132 条，向后兼容）**：
        `in_dim` = 数据维度（设计向量），`cond_dim` = 条件维度。
        标量调用（`in_dim=1, cond_dim=1`）行为**完全不变** ⟹ 既有
        `v8_mdn_check` / `v22_cfm_check` 不受影响。
        """
        super().__init__()
        torch.manual_seed(seed)
        self.tf = time_features
        self.in_dim = int(in_dim)
        self.cond_dim = int(cond_dim)
        # 输入 = φ（in_dim）+ 条件 a（cond_dim）+ 时间特征（2·tf）
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim + 2 * time_features, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, in_dim),          # **输出维度 = 数据维度**
        )

    # ---- 时间特征（傅里叶）----
    def _tfeat(self, t: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(1, self.tf + 1, dtype=t.dtype,
                             device=t.device) * np.pi
        return torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=-1)

    def velocity(self, phi: torch.Tensor, t: torch.Tensor,
                 a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([phi, a, self._tfeat(t)], dim=-1)
        return self.net(x)

    def loss(self, a: torch.Tensor, phi1: torch.Tensor,
             phi0: torch.Tensor | None = None) -> torch.Tensor:
        """CFM 损失：E‖v(φ_t,t) − (φ_1−φ_0)‖²。"""
        b = phi1.shape[0]
        if phi0 is None:
            phi0 = torch.randn_like(phi1)
        t = torch.rand(b, 1, dtype=phi1.dtype)
        phi_t = (1 - t) * phi0 + t * phi1
        target = phi1 - phi0
        return ((self.velocity(phi_t, t, a) - target) ** 2).mean()

    # ---- 采样（Euler 积分 t: 0 → 1）----
    @torch.no_grad()
    def sample(self, a: float, n: int, steps: int = 60, seed: int = 0
               ) -> np.ndarray:
        g = torch.Generator().manual_seed(seed)
        phi = torch.randn(n, 1, generator=g)
        at = torch.full((n, 1), float(a))
        for k in range(steps):
            t = torch.full((n, 1), k / steps)
            phi = phi + self.velocity(phi, t, at) / steps
        return phi.numpy().ravel()

    @torch.no_grad()
    def sample_cond(self, cond, n: int, steps: int = 60, seed: int = 0
                    ) -> np.ndarray:
        """**向量条件采样（2026-10-02 加）**：`cond` 为长度 `cond_dim` 的向量。

        返回形状 `(n, in_dim)`。标量情形请继续用 `sample()`（行为不变）。
        """
        c = torch.as_tensor(np.asarray(cond, dtype=np.float32)).reshape(1, -1)
        if c.shape[1] != self.cond_dim:
            raise ValueError("cond 维度 %d != cond_dim %d"
                             % (c.shape[1], self.cond_dim))
        g = torch.Generator().manual_seed(seed)
        phi = torch.randn(n, self.in_dim, generator=g)
        at = c.expand(n, self.cond_dim)
        for k in range(steps):
            t = torch.full((n, 1), k / steps)
            phi = phi + self.velocity(phi, t, at) / steps
        return phi.numpy()

    # ---- 精确对数密度（瞬时变分）----
    def log_density(self, a: float, grid: np.ndarray,
                    steps: int = 60) -> np.ndarray:
        """返回 log p_1(φ)（在 grid 上），由 ODE 变分公式精确积分。

        从 t=0（先验 N(0,1)）走到 t=1（数据分布）；每步同时积分
        `d log p/dt = −dv/dφ`（1-D 散度，autograd 精确）。
        """
        phi = torch.tensor(grid[:, None], dtype=torch.float32,
                           requires_grad=True)
        # **反向积分（2026-10-02 修正，由 v23 自检抓出）**：要评估**给定点**
        # φ 的数据密度，须从 t=1 的该点退回先验 t=0：
        #     dφ/ds = −v(φ, 1−s)，  d log p/ds = +∇·v
        # 终点加先验对数密度 log N(φ_0; 0,1)。初版误用**正向**积分（把 grid
        # 点当 t=0 起点）→ 算的是「grid 点经流映射后的像」的密度，不是 grid
        # 点的密度 ⟹ 系统性扭曲（自检 A2 的 ∫p̂dφ = 2.51 即其实锤）。
        acc = torch.zeros(len(grid), dtype=torch.float32)
        at = torch.full_like(phi, float(a))
        for k in range(steps, 0, -1):
            t = torch.full_like(phi, k / steps)
            v = self.velocity(phi, t, at)
            div = torch.autograd.grad(v.sum(), phi, retain_graph=False)[0]
            div = div.reshape(-1)      # 防 (N,1)⊗(N,) 广播
            phi = (phi - v / steps).detach().requires_grad_(True)
            acc = (acc + (div / steps).detach()).detach()
        # **符号修正（2026-10-02，第二轮）**：线性流解析校验 v=cφ：
        #   精确解 p₁=N(0,e^{2c}) ⟹ log p₁(x) = log p₀(x e^{−c}) − c
        #   ⟹ `log p₁ = log p₀(φ₀) − ∫₀¹ tr(∂v/∂φ)dt`（**减**）
        # 反向路径上积累的是 +∫tr dτ，故此处取**负**。初版取正 ⟹ 常数偏移
        # 翻倍（自检 A2 的 ∫p̂=2.61 即其实锤）。
        logp = (-0.5 * phi.reshape(-1) ** 2
                - 0.5 * np.log(2 * np.pi)) - acc
        # **防御性形状检查**：广播曾把 (N,) 与 (N,1) 静默变成 (N,N)
        assert logp.shape == (len(grid),), \
            "logp 形状异常 %s（期望 (%d,)）——检查广播" % (
                tuple(logp.shape), len(grid))
        return logp.detach().numpy().ravel()
