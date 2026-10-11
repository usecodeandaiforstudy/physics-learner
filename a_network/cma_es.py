"""零依赖紧凑版 CMA-ES（rank-μ 更新，Hansen 2006 标准公式）。

用途：Phase 2 冒烟——验证「结构编辑 → 环境 → 奖励 → 零阶优化」回路连通。
不是生产求解器（生产用 GFlowNet / 熵正则 SAC，架构文档 §2.6.5），只要求：
- 正确收敛到连续奖励函数的局部最优；
- 边界处理 = 采样裁剪（越界样本由环境拒绝 + 惩罚，合法性机制的一部分）。
"""

from __future__ import annotations

import numpy as np

__all__ = ["CMAES"]


class CMAES:
    """紧凑 rank-μ CMA-ES（最小化）。"""

    def __init__(self, x0, sigma0: float, bounds=None, popsize=None, seed: int = 0):
        self.n = int(len(x0))
        self.bounds = bounds
        self.m = np.asarray(x0, dtype=float).copy()
        self.sigma = float(sigma0)
        self.C = np.eye(self.n)
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        lam = int(popsize) if popsize else (4 + int(3 * np.log(max(self.n, 2))))
        self.lam = lam
        mu = max(1, lam // 2)
        self.mu = mu
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        self.w = w
        self.mu_eff = 1.0 / float((w ** 2).sum())
        self.cc = (4.0 + self.mu_eff / self.n) / (self.n + 4.0 + 2.0 * self.mu_eff / self.n)
        self.cs = (self.mu_eff + 2.0) / (self.n + self.mu_eff + 5.0)
        self.c1 = 2.0 / ((self.n + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1.0 - self.c1,
                       2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff)
                       / ((self.n + 2.0) ** 2 + self.mu_eff))
        self.damps = (1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0)
                                                    / (self.n + 1.0)) - 1.0)
                      + self.cs)
        self.chi_n = np.sqrt(self.n) * (1.0 - 1.0 / (4.0 * self.n)
                                        + 1.0 / (21.0 * self.n ** 2))
        self.rng = np.random.default_rng(seed)
        self.iteration = 0
        self.mean_history: list[np.ndarray] = [self.m.copy()]

    def ask(self) -> np.ndarray:
        """采样 popsize 个候选（裁剪到 bounds）。"""
        d, B = np.linalg.eigh(self.C)
        d = np.clip(d, 1e-30, None)
        z = self.rng.standard_normal((self.lam, self.n))
        y = (z @ (B * np.sqrt(d)).T)
        xs = self.m + self.sigma * y
        if self.bounds is not None:
            lo, hi = self.bounds
            xs = np.clip(xs, lo, hi)
        return xs

    def tell(self, xs: np.ndarray, fs: np.ndarray) -> None:
        """用 (xs, fs) 更新均值/协方差/步长（fs 为损失，越小越好）。"""
        order = np.argsort(fs)
        xs_sorted = xs[order][: self.mu]
        m_old = self.m.copy()
        self.m = self.w @ xs_sorted
        y_w = ((xs_sorted - m_old) / self.sigma).T @ self.w
        self.ps = ((1.0 - self.cs) * self.ps
                   + np.sqrt(self.cs * (2.0 - self.cs) * self.mu_eff) * y_w)
        hsig = float(
            np.linalg.norm(self.ps)
            / np.sqrt(1.0 - (1.0 - self.cs) ** (2.0 * (self.iteration + 1.0)))
            < (1.4 + 2.0 / (self.n + 1.0)) * self.chi_n)
        self.pc = ((1.0 - self.cc) * self.pc
                   + hsig * np.sqrt(self.cc * (2.0 - self.cc) * self.mu_eff) * y_w)
        ys = (xs_sorted - m_old) / self.sigma
        cmu = sum(w_i * np.outer(ys[i], ys[i])
                  for i, w_i in enumerate(self.w))
        self.C = ((1.0 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc)
                               + (1.0 - hsig) * self.cc * (2.0 - self.cc) * self.C)
                  + self.cmu * cmu)
        self.sigma *= float(np.exp(min(
            1.0, (self.cs / self.damps)
            * (np.linalg.norm(self.ps) / self.chi_n - 1.0))))
        self.iteration += 1
        self.mean_history.append(self.m.copy())
