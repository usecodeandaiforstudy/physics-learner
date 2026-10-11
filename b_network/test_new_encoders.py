"""新编码器（弹性/放射性/泊松/二能级）自测。

验证：
1. core.fisher 的新分布族 Fisher 解析值与标准结果一致。
2. SystemEncoder 把物理参数正确推导为分布参数。
3. FingerprintComputer 对非高斯单节点也能算 Fisher 摘要。
4. compute_ib_curve 对非高斯源诚实返回空。

运行：python b_network/test_new_encoders.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.fisher import (  # noqa: E402
    fisher_bernoulli,
    fisher_exponential,
    fisher_poisson,
    fisher_trace_laplace,
    fisher_det_laplace,
)
from b_network.system_encoder import SystemEncoder  # noqa: E402
from b_network.fingerprint_computer import FingerprintComputer  # noqa: E402


def main():
    lines = []
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    # --- 1. Fisher 解析值 ---
    lines.append(("PASS" if abs(fisher_exponential(2.0) - 0.25) < 1e-12 else "FAIL")
                 + f"  指数分布 I(λ)=1/λ²：{fisher_exponential(2.0)} vs 0.25")
    lines.append(("PASS" if abs(fisher_poisson(4.0) - 0.25) < 1e-12 else "FAIL")
                 + f"  泊松分布 I(λ)=1/λ：{fisher_poisson(4.0)} vs 0.25")
    lines.append(("PASS" if abs(fisher_bernoulli(0.5) - 4.0) < 1e-12 else "FAIL")
                 + f"  伯努利 I(p)=1/(p(1-p))：{fisher_bernoulli(0.5)} vs 4.0")
    lines.append(("PASS" if abs(fisher_trace_laplace(2.0) - 0.5) < 1e-12 else "FAIL")
                 + f"  拉普拉斯 trace=2/b²：{fisher_trace_laplace(2.0)} vs 0.5")
    lines.append(("PASS" if abs(fisher_det_laplace(2.0) - 1.0 / 16.0) < 1e-12 else "FAIL")
                 + f"  拉普拉斯 det=1/b⁴：{fisher_det_laplace(2.0)} vs 0.0625")

    enc = SystemEncoder()
    fpc = FingerprintComputer()

    # --- 2. 弹性材料：σ_x = sqrt(kT/k) ---
    e_elastic = {
        "family": "elastic", "channel_template": "CT-MECH-001",
        "parameters": [{"name": "k", "default": 4.0}, {"name": "kT", "default": 1.0}],
    }
    g = enc.encode(e_elastic)
    sigma_x = g.nodes[0].params["sigma"]
    lines.append(("PASS" if abs(sigma_x - 0.5) < 1e-12 else "FAIL")
                 + f"  弹性 σ_x = sqrt(kT/k)：{sigma_x} vs 0.5")

    # --- 3. 放射性衰变：Exp(λ) ---
    e_decay = {
        "family": "radioactive_decay", "channel_template": "CT-DECAY-001",
        "parameters": [{"name": "rate", "default": 3.0}],
    }
    g2 = enc.encode(e_decay)
    lines.append(("PASS" if g2.nodes[0].distribution_family == "DIST-EXPONENTIAL" else "FAIL")
                 + f"  放射性衰变 → DIST-EXPONENTIAL")
    fp2 = fpc.compute(e_decay, g2)
    ft = fp2["fingerprints"]["decay__T__0"]["fisher_trace"]
    lines.append(("PASS" if abs(ft - 1.0 / 9.0) < 1e-12 else "FAIL")
                 + f"  衰变 Fisher trace = 1/λ²：{ft} vs {1.0/9.0}")

    # --- 4. 泊松计数：Poisson(λ) ---
    e_count = {
        "family": "poisson_count", "channel_template": "CT-COUNT-001",
        "parameters": [{"name": "rate", "default": 5.0}],
    }
    g3 = enc.encode(e_count)
    fp3 = fpc.compute(e_count, g3)
    ft3 = fp3["fingerprints"]["count__T__0"]["fisher_trace"]
    lines.append(("PASS" if abs(ft3 - 0.2) < 1e-12 else "FAIL")
                 + f"  泊松 Fisher trace = 1/λ：{ft3} vs 0.2")

    # --- 5. 二能级（显式 p）：Bern(p) ---
    e_2lvl = {
        "family": "two_level", "channel_template": "CT-TWOLVL-001",
        "parameters": [{"name": "p", "default": 0.25}],
    }
    g4 = enc.encode(e_2lvl)
    fp4 = fpc.compute(e_2lvl, g4)
    ft4 = fp4["fingerprints"]["2lvl__T__0"]["fisher_trace"]
    expected = 1.0 / (0.25 * 0.75)
    lines.append(("PASS" if abs(ft4 - expected) < 1e-12 else "FAIL")
                 + f"  二能级 Fisher = 1/(p(1-p))：{ft4} vs {expected}")

    # --- 6. 二能级（由 ΔE/kT 推导占据概率）---
    e_2lvl2 = {
        "family": "two_level", "channel_template": "CT-TWOLVL-001",
        "parameters": [{"name": "delta_E", "default": 1.0}, {"name": "kT", "default": 1.0}],
    }
    g5 = enc.encode(e_2lvl2)
    p_derived = g5.nodes[0].params["p"]
    p_expected = 1.0 / (1.0 + math.exp(1.0))
    lines.append(("PASS" if abs(p_derived - p_expected) < 1e-12 else "FAIL")
                 + "  二能级玻尔兹曼占据 p=1/(1+e^(ΔE/kT))："
                 + f"{p_derived:.6f} vs {p_expected:.6f}")

    # --- 7. 非高斯源现在返回非空 IB 曲线（参数化信道路径）---
    ib = fpc.compute_ib_curve(g2)  # 放射性衰变（非高斯）
    lines.append(("PASS" if len(ib["I_XT"]) > 0 else "FAIL")
                 + f"  非高斯源 ib_curve 返回非空（参数化信道，{len(ib['I_XT'])} 点）")

    n_pass = sum(1 for ln in lines if ln.startswith("PASS"))
    n_total = sum(1 for ln in lines if ln.startswith(("PASS", "FAIL")))
    lines.append(f"\n{n_pass}/{n_total} 通过")

    out = Path(r"d:\physics learn\new_encoders_test.txt")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
