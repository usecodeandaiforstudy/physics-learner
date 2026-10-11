# -*- coding: utf-8 -*-
"""R4d 第三刀 v2 编码器自测：DIST-WRAPNORM（vonmises 三层同构候选）。

验证：
  1) 数值积分 vs MC（κ ∈ {0.3, 0.5, 1, 2, 3}）；
  2) 级数截断收敛（K=10 vs K=20 相对差 <1e-6）；
  3) 单调衰减 + 衰减到 0（I(3) < 0.05，均匀极限）；
  4) 曲线对照表（wrapnorm vs vonmises，同向衰减证据）；
  5) 编码器冒烟。

用法（项目根目录）：python b_network/test_new_encoders_9.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from core import (
    fisher_vonmises_kappa,
    fisher_wrapnorm_kappa,
    fisher_wrapnorm_kappa_mc,
)
from core.fisher import _wrapnorm_density
from core.fingerprint import compute_node_fisher
from b_network.system_encoder import SystemEncoder


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def main() -> int:
    fails = 0

    # ---- 1. 积分 vs MC ----
    vals = {}
    for k in (0.3, 0.5, 1.0, 2.0, 3.0):
        q = fisher_wrapnorm_kappa(k)
        mc = fisher_wrapnorm_kappa_mc(k, n_samples=200_000)
        rel = abs(q - mc) / max(q, 1e-9)
        vals[k] = q
        fails += check(f"wrapnorm 积分vsMC κ={k}（quad={q:.4f} mc={mc:.4f}）",
                       rel < 0.03, "rel=%.3f" % rel)

    # ---- 2. 级数截断收敛 ----
    import numpy as np

    theta = np.linspace(0.0, 2.0 * np.pi, 4000)
    p10, _ = _wrapnorm_density(theta, 1.0, k_max=10)
    p20, _ = _wrapnorm_density(theta, 1.0, k_max=20)
    fails += check("wrapnorm 级数截断 K=10 vs K=20 收敛",
                   float(np.max(np.abs(p10 - p20))) < 1e-6)

    # ---- 3. 单调衰减 + 衰减到 0 ----
    fails += check(f"wrapnorm 单调衰减 I(0.3)={vals[0.3]:.3f} > "
                   f"I(1)={vals[1.0]:.3f} > I(2)={vals[2.0]:.3f} > "
                   f"I(3)={vals[3.0]:.4f}",
                   vals[0.3] > vals[1.0] > vals[2.0] > vals[3.0])
    fails += check(f"wrapnorm 均匀极限 I(3)={vals[3.0]:.4f} < 0.05",
                   vals[3.0] < 0.05)

    # ---- 4. 曲线对照表 ----
    print("\n-- 曲线对照（vonmises I_κκ vs wrapnorm I(κ)）--")
    print("  %-6s %-12s %-12s" % ("κ", "vonmises Iκκ", "wrapnorm I(κ)"))
    for k in (0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 8.0, 20.0):
        v = fisher_vonmises_kappa(k)
        w = fisher_wrapnorm_kappa(k) if k <= 3.0 else float("nan")
        print("  %-6g %-12.4f %-12s" % (k, v,
                                       "%.4f" % w if w == w else "—"))

    # ---- 5. 编码器冒烟 ----
    enc = SystemEncoder()
    e = {"family": "wrapped_normal", "channel_template": "CT-WRAPNORM-001",
         "beta": 1.0, "parameters": [{"name": "kappa", "default": 1.0}]}
    fp = compute_node_fisher(enc.encode(e).nodes[0])
    fails += check(f"wrapnorm 编码器冒烟（trace={fp['fisher_trace']:.4f}）",
                   abs(fp["fisher_trace"] - fisher_wrapnorm_kappa(1.0)) < 1e-12)

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
