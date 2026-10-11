# -*- coding: utf-8 -*-
"""R4 扩族批次 4 编码器自测：DIST-MIX2 / DIST-TRUNCNORM。

验证（对照 test_new_encoders*.py 的同族模式）：
  1) mix2 数值积分 vs MC（d ∈ {0.5, 1, 2, 4, 6}）；
  2) mix2 饱和形状断言：0 → 单调上升 → 1/(4σ²) 平台（自测证伪过冲峰猜测）；
  3) truncnorm 解析式 vs 有限差分 score 平方期望（独立推导检查）；
  4) truncnorm 解析式 vs sympy 字符串求值一致；
  5) 编码器 → compute_node_fisher 冒烟。

用法（项目根目录）：python b_network/test_new_encoders_4.py
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, ".")

import numpy as np
from scipy import stats

from core import (
    fisher_mix2_monte_carlo,
    fisher_mix2_separation,
    fisher_truncnorm_tau,
)
from core.fingerprint import compute_node_fisher
from b_network.system_encoder import SystemEncoder


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def main() -> int:
    fails = 0

    # ---- 1. mix2：数值积分 vs MC ----
    for d in (0.5, 1.0, 2.0, 4.0, 6.0):
        quad = fisher_mix2_separation(d, sigma=1.0)
        mc = fisher_mix2_monte_carlo(d, sigma=1.0, n_samples=200_000)
        rel = abs(quad - mc) / max(quad, 1e-9)
        fails += check(f"mix2 积分vsMC d={d}（quad={quad:.4f} mc={mc:.4f}）",
                       rel < 0.03, "rel=%.3f" % rel)

    # ---- 2. mix2：饱和形状断言（0 → 单调上升 → 1/4 平台）----
    i_low = fisher_mix2_separation(0.05)
    i_1 = fisher_mix2_separation(1.0)
    i_mid = fisher_mix2_separation(2.0)
    i_4 = fisher_mix2_separation(4.0)
    i_high = fisher_mix2_separation(6.0)
    fails += check(f"mix2 单调上升 I(0.05)={i_low:.5f} < I(1)={i_1:.4f} "
                   f"< I(2)={i_mid:.4f} < I(4)={i_4:.4f}",
                   i_low < i_1 < i_mid < i_4)
    fails += check(f"mix2 平台段 I(4)={i_4:.4f} ≈ I(6)={i_high:.4f}",
                   abs(i_4 - i_high) < 0.01)
    fails += check(f"mix2 大 d 平台 I(6)≈1/4（{i_high:.4f}）",
                   abs(i_high - 0.25) < 0.01, "off=%.4f" % abs(i_high - 0.25))

    # ---- 3. truncnorm：解析 vs 有限差分 score ----
    tau, mu, sigma, h = -0.5, 0.0, 1.0, 1e-3
    rng = np.random.default_rng(0)
    n = 200_000
    z0 = (tau - mu) / sigma
    u = rng.uniform(stats.norm.cdf(z0), 1.0, n)
    x = mu + sigma * stats.norm.ppf(u)

    def logp(xx, tt):
        zz = (tt - mu) / sigma
        return stats.norm.logpdf(xx, mu, sigma) \
            - math.log(max(1.0 - stats.norm.cdf(zz), 1e-300))

    s_fd = (logp(x, tau + h) - logp(x, tau - h)) / (2 * h)
    i_fd = float(np.mean(s_fd ** 2))
    i_an = fisher_truncnorm_tau(tau, mu, sigma)
    rel = abs(i_fd - i_an) / max(i_an, 1e-9)
    fails += check(f"truncnorm FD检查 τ=−0.5（fd={i_fd:.5f} an={i_an:.5f}）",
                   rel < 0.05, "rel=%.3f" % rel)

    # ---- 4. truncnorm：解析 vs sympy 字符串（与 DB1 fpe 一致） ----
    import sympy
    t, m, s = sympy.symbols("tau mu sigma")
    expr = sympy.sympify("2*exp(-((tau-mu)/sigma)**2)/"
                         "(pi*sigma**2*erfc((tau-mu)/(sigma*sqrt(2)))**2)")
    sym_val = float(expr.subs({t: tau, m: mu, s: sigma}))
    fails += check(f"truncnorm sympy 字符串一致（{sym_val:.6f} vs {i_an:.6f}）",
                   abs(sym_val - i_an) < 1e-9)

    # ---- 5. 编码器冒烟 ----
    enc = SystemEncoder()
    e_mix = {"family": "gaussian_mixture2", "channel_template": "CT-MIX-001",
             "beta": 1.0,
             "parameters": [{"name": "d", "default": 2.0},
                            {"name": "w", "default": 0.5},
                            {"name": "sigma", "default": 1.0}]}
    g_mix = enc.encode(e_mix)
    fp_mix = compute_node_fisher(g_mix.nodes[0])
    fails += check(f"mix2 编码器冒烟（trace={fp_mix['fisher_trace']:.4f}）",
                   abs(fp_mix["fisher_trace"] - fisher_mix2_separation(2.0))
                   < 1e-9)

    e_tr = {"family": "truncated_gaussian",
            "channel_template": "CT-TRUNCNORM-001", "beta": 1.0,
            "parameters": [{"name": "tau", "default": -1.0},
                           {"name": "mu", "default": 0.0},
                           {"name": "sigma", "default": 1.0}]}
    g_tr = enc.encode(e_tr)
    fp_tr = compute_node_fisher(g_tr.nodes[0])
    fails += check(f"truncnorm 编码器冒烟（trace={fp_tr['fisher_trace']:.4f}）",
                   abs(fp_tr["fisher_trace"] - fisher_truncnorm_tau(-1.0))
                   < 1e-9)

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
