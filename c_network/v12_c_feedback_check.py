# -*- coding: utf-8 -*-
"""回流边正确实现（第 63 条修正，2026-10-02）：DB5 → C 的下一轮微调。

**架构修正（重要）**：§2.4 原文「RL 运行后，DB5 验证通过的（G*，解码设计，
验证结果）以高权重加入下一轮微调」位于**网络 C 的训练小节**——回流目标是
**C 解码器的下一轮训练**，不是 RL 智能体的经验注入。此前 7 版实验把回流
接到 SAC 缓冲/预训练上（v11 系列），属架构误读；结论（RL 经验注入在玩具
上无益、7 版记录）仍归档，但边的正确契约是「验证过的设计点 → 解码器
下一轮的加权监督样本」。

v12 实现：DB5 pass 记录的参数值 = RL 实际探索到的高价值设计点 →
在这些点上采样真值密度、以多重性（高权重）加入 C 的微调训练集。

判据（预先登记）：
  F1 机制：DB5 提取到 ≥3 个去重设计点并高权重注入；
  F2 解码增益：微调后验证点上的密度重叠 overlap 均值提升 ≥ +0.005；
  F3 无遗忘：基础网格（a∈{−2,−1,0,1,2}）上的 overlap 不退化 > 0.005。

用法（项目根目录）：python c_network/v12_c_feedback_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from c_network.mdn import MixtureDensityNetwork, sample_landau
from db5.db5_writer import DB5Writer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

GRID = np.linspace(-3.0, 3.0, 2000)
BASE_A = [-2.0, -1.0, 0.0, 1.0, 2.0]


def true_density(a: float, grid: np.ndarray) -> np.ndarray:
    logp = -(a * grid ** 2 / 2.0 + grid ** 4 / 4.0)
    logp = logp - logp.max()
    p = np.exp(logp)
    return p / (p.sum() * (grid[1] - grid[0]))


def overlap(mdn, a: float) -> float:
    p_pred = mdn.density(a, GRID)
    p_true = true_density(a, GRID)
    return float(np.trapezoid(np.minimum(p_pred, p_true), GRID))


def make_data(a_vals: list[float], n_each: int, mult: dict | None = None):
    X, Y = [], []
    for a in a_vals:
        m = 1 if mult is None else mult.get(a, 1)
        phi = sample_landau(a, 1.0, 1.0, n_each, seed=0)
        for _ in range(m):
            X.append(np.full(len(phi), a, dtype=np.float32))
            Y.append(phi.astype(np.float32))
    return np.concatenate(X)[:, None], np.concatenate(Y)[:, None]


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def main() -> int:
    fails = 0
    torch.manual_seed(0)

    # ---- DB5 提取高价值设计点（RL 实际探索到的参数值）----
    recs = DB5Writer().load()
    pts = sorted({round(float(r["decoded_design"]["params"].get("a")), 3)
                  for r in recs
                  if r["domain"] == "landau"
                  and r["validation"].get("verdict") == "pass"})
    print(f"  DB5 设计点（去重）：{pts}")
    fails += check(f"F1 设计点提取并高权重注入（{len(pts)} 个 ≥ 3）",
                   len(pts) >= 3, f"n={len(pts)}")

    # ---- 第一轮：基础网格训练 ----
    Xb, Yb = make_data(BASE_A, 4000)
    mdn = MixtureDensityNetwork(1, n_components=8)
    Xt, Yt = torch.tensor(Xb), torch.tensor(Yb)
    opt = torch.optim.Adam(mdn.parameters(), lr=3e-3)
    for step in range(6000):
        idx = np.random.randint(0, len(Xt), 512)
        loss = mdn.nll(Xt[idx], Yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1500 == 0:
            print(f"  第一轮 step {step}: NLL={loss.item():.3f}")

    ov_val_before = {a: overlap(mdn, a) for a in pts}
    ov_base_before = {a: overlap(mdn, a) for a in BASE_A}
    print("  微调前：验证点 overlap 均值 %.4f；基础网格 %.4f"
          % (float(np.mean(list(ov_val_before.values()))),
             float(np.mean(list(ov_base_before.values())))))

    # ---- 第二轮：DB5 设计点高权重微调（多重性 3）----
    mult = {a: 3 for a in pts}
    Xv, Yv = make_data(pts, 4000, mult=mult)
    Xf = np.concatenate([Xb, Xv])
    Yf = np.concatenate([Yb, Yv])
    Xf_t, Yf_t = torch.tensor(Xf), torch.tensor(Yf)
    opt2 = torch.optim.Adam(mdn.parameters(), lr=1e-3)
    for step in range(4000):
        idx = np.random.randint(0, len(Xf_t), 512)
        loss = mdn.nll(Xf_t[idx], Yf_t[idx])
        opt2.zero_grad()
        loss.backward()
        opt2.step()

    ov_val_after = {a: overlap(mdn, a) for a in pts}
    ov_base_after = {a: overlap(mdn, a) for a in BASE_A}
    v_b = float(np.mean(list(ov_val_before.values())))
    v_a = float(np.mean(list(ov_val_after.values())))
    b_b = float(np.mean(list(ov_base_before.values())))
    b_a = float(np.mean(list(ov_base_after.values())))
    print("  微调后：验证点 overlap 均值 %.4f（Δ%+.4f）；基础网格 %.4f（Δ%+.4f）"
          % (v_a, v_a - v_b, b_a, b_a - b_b))
    for a in pts:
        print("    a=%+6.3f: %.4f → %.4f" % (a, ov_val_before[a],
                                             ov_val_after[a]))
    fails += check(f"F2 解码增益（验证点 Δ{v_a - v_b:+.4f} ≥ +0.005）",
                   v_a - v_b >= 0.005, f"Δ={v_a - v_b:+.4f}")
    fails += check(f"F3 无遗忘（基础网格 Δ{b_a - b_b:+.4f} ≥ −0.005）",
                   b_a - b_b >= -0.005, f"Δ={b_a - b_b:+.4f}")

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
