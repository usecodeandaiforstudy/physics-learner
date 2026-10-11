# -*- coding: utf-8 -*-
"""CFM 仪器自检（2026-10-02，第 102 条）——**先证明尺子准，再判优劣**。

**动机**：`v22` 的 20-seed 对照里，CFM 的 NLL（0.973）**低于真值分布的 NLL
（1.135）**——这在理论上不可能（真值 NLL 最优）。这指向**密度估计的系统性
偏差**（瞬时变分积分的欠积分/误差累积），而不是 CFM 更好。**在判定
「MDN vs CFM」之前必须先把这把尺子校准**。

**两项自检**：
  **A. 密度估计真值对照**：在**已知密度**的任务上训练 CFM，比较
     `log p̂(φ)` 与解析 `log p(φ)`：
       A1 目标 = N(0,1)（条件固定）→ 解析 log 密度已知；
       A2 **网格归一化校验**：`∫p̂dφ` 是否 ≈ 1（偏差即积分误差）；
       A3 均值/方差对照（矩检验）。
  **B. 训练充分性扫描**：steps × lr 网格 → 看重叠是否随训练量上升
     （若是，则 v22 的对照对 CFM **不公平**，结论须撤回重测）。

用法（项目根目录）：python c_network/v23_cfm_selfcheck.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from c_network.cfm import ConditionalFlowMatching
from c_network.mdn import sample_landau

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

GRID = np.linspace(-6.0, 6.0, 401)


def analytic_logpdf(x: np.ndarray) -> np.ndarray:
    return -0.5 * x ** 2 - 0.5 * np.log(2 * np.pi)


def train_gaussian_cfm(steps: int = 6000, lr: float = 2e-3,
                       seed: int = 0, a_fixed: float = 0.0
                       ) -> ConditionalFlowMatching:
    """目标 = N(0,1)（与先验同族 → 理想情况下速度场 ≈ 0，密度应保持 N(0,1)）。"""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(20000).astype(np.float32)[:, None]
    a = np.full_like(y, a_fixed)
    torch.manual_seed(seed)
    cfm = ConditionalFlowMatching(1, seed=seed)
    opt = torch.optim.Adam(cfm.parameters(), lr=lr)
    Yt, At = torch.tensor(y), torch.tensor(a)
    for _ in range(steps):
        idx = rng.integers(0, len(Yt), 512)
        loss = cfm.loss(At[idx], Yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return cfm


def main() -> int:
    print("========== CFM 仪器自检（先校尺子）==========")
    grid = GRID
    p_true = np.exp(analytic_logpdf(grid))
    p_true = p_true / (p_true.sum() * (grid[1] - grid[0]))

    cfm = train_gaussian_cfm(seed=0)
    print("  A4 **ODE 步长扫描**（判断 A2 的 2.6× 是离散误差还是实现 bug）：")
    for st in (60, 200, 800):
        lp_s = cfm.log_density(0.0, grid, steps=st)
        p_s = np.exp(lp_s - lp_s.max())
        Z_s = float(p_s.sum() * (grid[1] - grid[0]))
        d_s = float(np.mean(np.abs(lp_s - analytic_logpdf(grid))))
        print("     steps %4d：∫p̂dφ = %.4f（理想 1.0）| 平均 |Δlog p| = %.4f"
              % (st, Z_s, d_s))
    print("     ⟹ 若 Z 随步数收敛到 1 → **离散误差**（提高步数即可）；"
          "若仍远离 1 → **实现 bug**（须继续查）")
    lp = cfm.log_density(0.0, grid, steps=60)
    p_hat = np.exp(lp - lp.max())
    Z = float(p_hat.sum() * (grid[1] - grid[0]))
    p_hat_n = p_hat / Z
    dlog = float(np.mean(np.abs(lp - analytic_logpdf(grid))))
    ovl = float(np.sum(np.minimum(p_true, p_hat_n)) * (grid[1] - grid[0]))
    # 矩检验（用采样）
    s = cfm.sample(0.0, 20000, steps=60, seed=1)
    print("  A1 目标 N(0,1) 的 log 密度对照：平均 |Δlog p| = %.4f nats" % dlog)
    print("  A2 **归一化校验**：∫p̂dφ = %.4f（理想 1.0；偏差即积分误差）" % Z)
    print("  A3 矩检验：采样均值 %+.4f（真值 0）、std %.4f（真值 1）"
          % (s.mean(), s.std()))
    print("     重叠（N(0,1) vs p̂）= %.4f" % ovl)
    ok_a = (abs(Z - 1.0) < 0.05 and abs(s.mean()) < 0.05
            and abs(s.std() - 1.0) < 0.05 and ovl > 0.95)
    print("     → 密度估计%s" % ("**可信**" if ok_a else
                              "**不可信**（须先修 ODE/训练，再做优劣判定）"))

    print("\n  B. 训练充分性扫描（landau a=1.0，真值密度对照）")
    a_test = 1.0
    phi_g = np.linspace(-3.2, 3.2, 241)
    logp = -(a_test * phi_g ** 2 / 2 + phi_g ** 4 / 4)
    p_t = np.exp(logp - logp.max())
    p_t = p_t / (p_t.sum() * (phi_g[1] - phi_g[0]))
    rows = []
    for steps in (2000, 6000, 20000):
        for lr in (2e-3, 1e-3):
            rng = np.random.default_rng(0)
            phi = sample_landau(a_test, 1.0, 1.0, 20000, seed=0)
            A = np.full_like(phi, a_test, dtype=np.float32)[:, None]
            P = phi.astype(np.float32)[:, None]
            torch.manual_seed(0)
            m = ConditionalFlowMatching(1, seed=0)
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            At, Pt = torch.tensor(A), torch.tensor(P)
            for _ in range(steps):
                idx = rng.integers(0, len(At), 512)
                loss = m.loss(At[idx], Pt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            lp = m.log_density(a_test, phi_g)
            ph = np.exp(lp - lp.max())
            ph = ph / (ph.sum() * (phi_g[1] - phi_g[0]))
            ov = float(np.sum(np.minimum(p_t, ph)) * (phi_g[1] - phi_g[0]))
            rows.append((steps, lr, ov))
            print("    steps %5d lr %.0e → 重叠 %.4f" % (steps, lr, ov))
    best = max(r[2] for r in rows)
    print("    最佳重叠 %.4f（MDN 在 v22 的 6 点平均为 **0.9663**）→ %s"
          % (best, "**训练充分性不是主因**（重叠不随训练量显著上升）"
             if best < 0.93 else
             "**v22 的对照对 CFM 不公平**：加大训练即可接近/超过 MDN"))
    print("\n  说明：本自检只回答「尺子准不准 / CFM 是否训练不足」；"
          "不构成 MDN 与 CFM 的最终优劣判定。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
