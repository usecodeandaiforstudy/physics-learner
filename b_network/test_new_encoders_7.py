# -*- coding: utf-8 -*-
"""R4d 编码器自测：DIST-INVCHISQ（chisq 同构）/ DIST-FRECHET（weibull 同构）。

验证：
  1) invchisq 与 chisq 逐点相同（ψ₁(k/2)/4 恒等式）；
  2) frechet trace/det 解析 vs MC（y~Exp(1) 变换采样）；
  3) frechet U 形断言：trace 有最小值（先降后升）；
  4) frechet det 与 weibull 同形（π²/(6s²)）；
  5) 编码器冒烟 ×2。

用法（项目根目录）：python b_network/test_new_encoders_7.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from math import pi

from core import (
    fisher_chisq,
    fisher_det_weibull,
    fisher_frechet_det,
    fisher_frechet_trace,
    fisher_frechet_trace_mc,
    fisher_invchisq_k,
)
from core.fingerprint import compute_node_fisher
from b_network.system_encoder import SystemEncoder


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def main() -> int:
    fails = 0

    # ---- 1. invchisq == chisq 逐点 ----
    for k in (2.0, 4.0, 8.0, 20.0):
        a = fisher_invchisq_k(k)
        b = fisher_chisq(k)
        fails += check(f"invchisq=chisq k={k}（{a:.6f} vs {b:.6f}）",
                       abs(a - b) < 1e-12)

    # ---- 2. frechet 解析 vs MC ----
    for shape, scale in ((1.5, 2.0), (2.5, 1.0), (0.8, 3.0), (4.0, 0.5)):
        an = fisher_frechet_trace(shape, scale)
        mc = fisher_frechet_trace_mc(shape, scale, n_samples=200_000)
        rel = abs(an - mc) / an
        fails += check(f"frechet trace MC α={shape} s={scale}"
                       f"（an={an:.4f} mc={mc:.4f}）",
                       rel < 0.03, "rel=%.3f" % rel)

    # ---- 3. U 形断言（s=2：最小值在 α≈(C·s²)^(1/4)≈1.64）----
    t_low = fisher_frechet_trace(0.7, 2.0)
    t_min = fisher_frechet_trace(1.6, 2.0)
    t_hi = fisher_frechet_trace(4.0, 2.0)
    fails += check(f"frechet U 形 trace(0.7)={t_low:.3f} > trace(1.6)="
                   f"{t_min:.3f} < trace(4)={t_hi:.3f}",
                   t_low > t_min and t_hi > t_min)

    # ---- 4. det 与 weibull 同形 ----
    for s in (1.0, 2.0, 3.0):
        d1 = fisher_frechet_det(2.0, s)
        d2 = pi ** 2 / (6.0 * s ** 2)
        w = fisher_det_weibull(2.0, s)
        fails += check(f"frechet det s={s}（{d1:.6f} = π²/6s² = {d2:.6f}，"
                       f"weibull det 同形 {w:.6f}）",
                       abs(d1 - d2) < 1e-12 and abs(d1 - w) < 1e-12)

    # ---- 5. 编码器冒烟 ----
    enc = SystemEncoder()
    e1 = {"family": "invchisq", "channel_template": "CT-INVCHISQ-001",
          "beta": 1.0, "parameters": [{"name": "k", "default": 4.0}]}
    fp1 = compute_node_fisher(enc.encode(e1).nodes[0])
    fails += check(f"invchisq 编码器冒烟（trace={fp1['fisher_trace']:.4f}）",
                   abs(fp1["fisher_trace"] - fisher_invchisq_k(4.0)) < 1e-12)

    e2 = {"family": "frechet", "channel_template": "CT-FRECHET-001",
          "beta": 1.0,
          "parameters": [{"name": "shape", "default": 1.5},
                         {"name": "scale", "default": 2.0}]}
    fp2 = compute_node_fisher(enc.encode(e2).nodes[0])
    fails += check(f"frechet 编码器冒烟（trace={fp2['fisher_trace']:.4f}）",
                   abs(fp2["fisher_trace"] - fisher_frechet_trace(1.5, 2.0))
                   < 1e-12)

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
