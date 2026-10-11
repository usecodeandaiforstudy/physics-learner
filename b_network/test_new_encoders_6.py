# -*- coding: utf-8 -*-
"""R4c 编码器自测：DIST-INVGAMMA（逆伽马，gamma 输入模式同构族）。

验证：
  1) Fisher 公式与 gamma 完全同形（微分同胚等价性，逐点相等）；
  2) Fisher vs MC score 采样（IG 采样 + 解析 score）；
  3) 编码器 → compute_node_fisher 冒烟。

用法（项目根目录）：python b_network/test_new_encoders_6.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np

from core import (
    fisher_det_gamma,
    fisher_invgamma_det,
    fisher_invgamma_trace,
    fisher_trace_gamma,
)
from core.fingerprint import compute_node_fisher
from b_network.system_encoder import SystemEncoder


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def main() -> int:
    fails = 0

    # ---- 1. 与 gamma 完全同形 ----
    for k, theta in ((0.9, 1.5), (2.0, 2.0), (3.0, 0.5), (8.0, 1.0)):
        t1 = fisher_invgamma_trace(k, theta)
        d1 = fisher_invgamma_det(k, theta)
        t2 = fisher_trace_gamma(k, theta)
        d2 = fisher_det_gamma(k, theta)
        fails += check(f"invgamma=gamma 同形 k={k} θ={theta}"
                       f"（trace {t1:.6f} vs {t2:.6f}）",
                       abs(t1 - t2) < 1e-12 and abs(d1 - d2) < 1e-12)

    # ---- 2. MC 对照：IG(k,θ) 采样 + score ----
    for k, theta in ((2.0, 1.0), (3.0, 0.5)):
        rng = np.random.default_rng(0)
        x = 1.0 / rng.gamma(shape=k, scale=1.0 / theta, size=200_000)
        # score w.r.t. k：ln p = k ln θ − ln Γ(k) − (k+1) ln x − θ/x
        # ∂/∂k = ln θ − ψ(k) − ln x
        from scipy.special import digamma, polygamma
        sk = np.log(theta) - digamma(k) - np.log(x)
        # score w.r.t. theta：∂/∂θ = k/θ − 1/x
        st = k / theta - 1.0 / x
        # trace = E[sk²] + E[st²]
        i_trace_mc = float(np.mean(sk ** 2) + np.mean(st ** 2))
        rel = abs(i_trace_mc - fisher_invgamma_trace(k, theta)) \
            / fisher_invgamma_trace(k, theta)
        fails += check(f"invgamma MC trace k={k} θ={theta}"
                       f"（mc={i_trace_mc:.4f} an={fisher_invgamma_trace(k, theta):.4f}）",
                       rel < 0.03, "rel=%.3f" % rel)

    # ---- 3. 编码器冒烟 ----
    enc = SystemEncoder()
    e = {"family": "inverse_gamma", "channel_template": "CT-INVGAMMA-001",
         "beta": 1.0,
         "parameters": [{"name": "k", "default": 3.0},
                        {"name": "theta", "default": 0.5}]}
    g = enc.encode(e)
    fp = compute_node_fisher(g.nodes[0])
    fails += check(f"invgamma 编码器冒烟（trace={fp['fisher_trace']:.4f}）",
                   abs(fp["fisher_trace"] - fisher_invgamma_trace(3.0, 0.5))
                   < 1e-12)

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
