# -*- coding: utf-8 -*-
"""R4b 编码器自测：DIST-TRUNCGEOM（截断几何，negbin 输入模式同构族）。

验证：
  1) 有限和 Fisher vs MC（k ∈ {5, 10, 20} × p ∈ {0.1, 0.3, 0.45}）；
  2) 极限对照：k=1000 时 I(p) → 几何的 1/(p²(1-p))（rel 容差 1e-2）；
  3) U 形左腿形状：I(p) 在 p∈[0.05,0.5] 单调下降（同几何/negbin 左腿）；
  4) 编码器 → compute_node_fisher 冒烟。

用法（项目根目录）：python b_network/test_new_encoders_5.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from core import (
    fisher_geometric,
    fisher_truncgeom_monte_carlo,
    fisher_truncgeom_p,
)
from core.fingerprint import compute_node_fisher
from b_network.system_encoder import SystemEncoder


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def main() -> int:
    fails = 0

    # ---- 1. 有限和 vs MC ----
    for k in (5, 10, 20):
        for p in (0.1, 0.3, 0.45):
            ex = fisher_truncgeom_p(k, p)
            mc = fisher_truncgeom_monte_carlo(k, p, n_samples=200_000)
            rel = abs(ex - mc) / max(ex, 1e-9)
            fails += check(f"truncgeom 有限和vsMC k={k} p={p}"
                           f"（ex={ex:.4f} mc={mc:.4f}）",
                           rel < 0.03, "rel=%.3f" % rel)

    # ---- 2. 极限对照：k→∞ → 几何 ----
    for p in (0.1, 0.3, 0.45):
        big = fisher_truncgeom_p(1000, p)
        geo = fisher_geometric(p)
        rel = abs(big - geo) / geo
        fails += check(f"truncgeom k=1000 → 几何极限 p={p}"
                       f"（{big:.4f} vs {geo:.4f}）",
                       rel < 1e-2, "rel=%.4f" % rel)

    # ---- 3. U 形左腿：p↑ → I↓ ----
    i_low = fisher_truncgeom_p(10, 0.05)
    i_mid = fisher_truncgeom_p(10, 0.3)
    i_hi = fisher_truncgeom_p(10, 0.5)
    fails += check(f"truncgeom 左腿单调降 I(0.05)={i_low:.2f} "
                   f"> I(0.3)={i_mid:.2f} > I(0.5)={i_hi:.2f}",
                   i_low > i_mid > i_hi)

    # ---- 4. 编码器冒烟 ----
    enc = SystemEncoder()
    e = {"family": "truncated_geometric", "channel_template": "CT-TRUNCGEOM-001",
         "beta": 1.0,
         "parameters": [{"name": "k", "default": 10.0},
                        {"name": "p", "default": 0.3}]}
    g = enc.encode(e)
    fp = compute_node_fisher(g.nodes[0])
    fails += check(f"truncgeom 编码器冒烟（trace={fp['fisher_trace']:.4f}）",
                   abs(fp["fisher_trace"] - fisher_truncgeom_p(10, 0.3)) < 1e-12)

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
