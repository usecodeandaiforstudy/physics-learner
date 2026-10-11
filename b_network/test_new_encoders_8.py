# -*- coding: utf-8 -*-
"""R4d 第三刀编码器自测：DIST-RICIAN（vonmises 同构候选）。

验证：
  1) 数值积分 vs MC（κ ∈ {0.1, 1, 4, 8, 20}）；
  2) 单调衰减形状（I(0.1) > I(1) > I(4) > I(20)）；
  3) 尾渐近：I(20) ≈ 1/(2·20)=0.025（±40%）；
  4) 曲线对照表（vonmises I_κκ vs rician I(κ)——同向同尾的证据）；
  5) 编码器冒烟。

用法（项目根目录）：python b_network/test_new_encoders_8.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from core import (
    fisher_rician_kappa,
    fisher_rician_kappa_mc,
    fisher_vonmises_kappa,
)
from core.fingerprint import compute_node_fisher
from b_network.system_encoder import SystemEncoder


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def main() -> int:
    fails = 0

    # ---- 1. 积分 vs MC ----
    for k in (0.1, 1.0, 4.0, 8.0, 20.0):
        q = fisher_rician_kappa(k)
        mc = fisher_rician_kappa_mc(k, n_samples=200_000)
        rel = abs(q - mc) / max(q, 1e-9)
        fails += check(f"rician 积分vsMC κ={k}（quad={q:.4f} mc={mc:.4f}）",
                       rel < 0.03, "rel=%.3f" % rel)

    # ---- 2. 单调衰减 ----
    i01, i1, i4, i20 = (fisher_rician_kappa(k) for k in (0.1, 1.0, 4.0, 20.0))
    fails += check(f"rician 单调衰减 I(0.1)={i01:.3f} > I(1)={i1:.3f} "
                   f"> I(4)={i4:.3f} > I(20)={i20:.4f}",
                   i01 > i1 > i4 > i20)

    # ---- 3. 尾渐近 I(κ)→1/(2κ) ----
    tail = 1.0 / (2.0 * 20.0)
    fails += check(f"rician 尾渐近 I(20)={i20:.4f} ≈ 1/(2·20)={tail:.4f}",
                   abs(i20 - tail) / tail < 0.40, "off=%.2f" % (i20 / tail))

    # ---- 4. 曲线对照表（vonmises I_κκ vs rician I(κ)）----
    print("\n-- 曲线对照（vonmises I_κκ vs rician I(κ)，κ 扫描）--")
    print("  %-6s %-12s %-12s" % ("κ", "vonmises Iκκ", "rician I(κ)"))
    for k in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0):
        vk = fisher_vonmises_kappa(k)
        r = fisher_rician_kappa(k)
        print("  %-6g %-12.4f %-12.4f" % (k, vk, r))

    # ---- 5. 编码器冒烟 ----
    enc = SystemEncoder()
    e = {"family": "rician", "channel_template": "CT-RICIAN-001", "beta": 1.0,
         "parameters": [{"name": "kappa", "default": 4.0}]}
    fp = compute_node_fisher(enc.encode(e).nodes[0])
    fails += check(f"rician 编码器冒烟（trace={fp['fisher_trace']:.4f}）",
                   abs(fp["fisher_trace"] - fisher_rician_kappa(4.0)) < 1e-12)

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
