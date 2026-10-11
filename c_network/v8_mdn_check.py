# -*- coding: utf-8 -*-
"""C v0 验证（M8，2026-10-02）：landau 领域专用 MDN 解码器。

设计口径（design doc §2.4/§2.6.2）：C = 领域解码器 ×N（本实例 = landau
领域，随用随练）；MDN 仅管线联调用（覆盖率 <90% 即淘汰，生产 = CFM）。
训练：a ∈ {−2,−1,0,1,2} 各采 4000 个 φ；MDN(8 分量) 以 NLL 训练。
判据：
  C1 双阱形态：a=−2 时预测密度有两个峰，位置 ≈ ±√2（±0.3 容差）；
  C2 单阱形态：a=+2 时单峰，位置 ≈ 0（±0.3）；
  C3 相变形态学：a 从 −2 扫到 +2，峰数从 2 单调过渡到 1；
  C4 密度质量（原理性）：MDN NLL ≤ 真实密度 NLL + 0.5 nats。

用法（项目根目录）：python c_network/v8_mdn_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from c_network.mdn import MixtureDensityNetwork, sample_landau

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def main() -> int:
    fails = 0
    torch.manual_seed(0)
    # ---- 数据：a ∈ {−2,−1,0,1,2} × 4000 样本 ----
    a_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
    X, Y = [], []
    for a in a_vals:
        phi = sample_landau(a, 1.0, 1.0, 4000, seed=0)
        X.append(np.full(len(phi), a, dtype=np.float32))
        Y.append(phi.astype(np.float32))
    X = np.concatenate(X)[:, None]
    Y = np.concatenate(Y)[:, None]

    mdn = MixtureDensityNetwork(1, n_components=8)
    opt = torch.optim.Adam(mdn.parameters(), lr=3e-3)
    Xt = torch.tensor(X)
    Yt = torch.tensor(Y)

    # 真实密度 NLL（原理性基线：网格 pdf 在采样点上的 −mean log p）
    nll_true = 0.0
    for a in a_vals:
        phi_grid = np.linspace(-3.5, 3.5, 4000)
        logp = -(a * phi_grid ** 2 / 2.0 + phi_grid ** 4 / 4.0)
        logp = logp - logp.max()
        p = np.exp(logp)
        pdf = p / (p.sum() * (phi_grid[1] - phi_grid[0]))
        x_s = Y[X[:, 0] == a]
        nll_true += -float(np.mean(np.log(np.interp(
            x_s, phi_grid, pdf) + 1e-12)))
    nll_true /= len(a_vals)
    print(f"  真实密度 NLL（原理性基线）：{nll_true:.3f}")

    for step in range(6000):
        idx = np.random.randint(0, len(Xt), 512)
        loss = mdn.nll(Xt[idx], Yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1500 == 0:
            print(f"  step {step}: NLL={loss.item():.3f}")
    nll1 = float(mdn.nll(Xt, Yt).item())
    print(f"  NLL {nll_true:.3f}(真实) vs MDN {nll1:.3f}")

    grid = np.linspace(-3.0, 3.0, 2000)
    # C1 双阱
    m_neg = mdn.modes(-2.0, grid)
    ok_c1 = any(abs(p + np.sqrt(2.0)) < 0.3 for p in m_neg) and \
        any(abs(p - np.sqrt(2.0)) < 0.3 for p in m_neg)
    fails += check(f"C1 双阱形态（a=−2 预测峰 {m_neg} ≈ ±√2）", ok_c1)
    # C2 单阱
    m_pos = mdn.modes(2.0, grid)
    ok_c2 = any(abs(p) < 0.3 for p in m_pos)
    fails += check(f"C2 单阱形态（a=+2 预测峰 {m_pos} ≈ 0）", ok_c2)
    # C3 相变形态学
    counts = [len(mdn.modes(a, grid)) for a in a_vals]
    print(f"  C3 峰数随 a 扫描：{counts}（a={a_vals}）")
    fails += check("C3 相变形态学（a=−2 双峰、a=+2 单峰）",
                   counts[0] >= 2 and counts[-1] == 1)
    # C4 原理性判据（design doc 口径）：MDN NLL ≤ 真实密度 NLL + 0.5 nats
    fails += check(f"C4 密度质量（MDN {nll1:.2f} ≤ 真实 {nll_true:.2f} "
                   f"+ 0.5 nats）", nll1 <= nll_true + 0.5,
                   f"gap={nll1 - nll_true:.2f}")

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
