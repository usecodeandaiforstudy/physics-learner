"""新编码器（拉普拉斯/谐振腔 LC/朗道自由能）自测。

验证：
1. 拉普拉斯 Fisher 解析值（trace=2/b², det=1/b⁴）与双指数差编码。
2. 谐振腔 LC 电路电压热涨落 σ_v = sqrt(kT/C) 与 Fisher 高斯解析值。
3. 朗道序参量分布 Fisher 数值积分在高斯极限下对照解析值 1/(2a²)，
   以及 a<0 双峰相 Fisher 大于 a>0 单峰相（相变信息论特征）。

运行：python b_network/test_new_encoders_2.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.fisher import (  # noqa: E402
    fisher_det_laplace,
    fisher_trace_laplace,
    fisher_landau,
)
from b_network.system_encoder import SystemEncoder  # noqa: E402
from b_network.fingerprint_computer import FingerprintComputer  # noqa: E402


def main():
    lines = []
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    enc = SystemEncoder()
    fpc = FingerprintComputer()

    # --- 1. 拉普拉斯 Fisher 解析值 ---
    b = 0.5  # rate=2.0 -> b=1/rate=0.5
    exp_trace = 2.0 / b**2  # 8.0
    exp_det = 1.0 / b**4    # 16.0
    lines.append(("PASS" if abs(fisher_trace_laplace(b) - exp_trace) < 1e-12 else "FAIL")
                 + f"  拉普拉斯 trace=2/b²：{fisher_trace_laplace(b)} vs {exp_trace}")
    lines.append(("PASS" if abs(fisher_det_laplace(b) - exp_det) < 1e-12 else "FAIL")
                 + f"  拉普拉斯 det=1/b⁴：{fisher_det_laplace(b)} vs {exp_det}")

    # --- 2. 拉普拉斯编码：rate=2.0 -> b=0.5, DIST-LAPLACE ---
    e_lap = {
        "family": "laplace", "channel_template": "CT-LAPLACE-001",
        "parameters": [{"name": "rate", "default": 2.0}],
    }
    g_lap = enc.encode(e_lap)
    node_lap = g_lap.nodes[0]
    lines.append(("PASS" if node_lap.distribution_family == "DIST-LAPLACE" else "FAIL")
                 + "  拉普拉斯编码 → DIST-LAPLACE")
    b_enc = node_lap.params["b"]
    lines.append(("PASS" if abs(b_enc - 0.5) < 1e-12 else "FAIL")
                 + f"  拉普拉斯 b = 1/rate：{b_enc} vs 0.5")
    fp_lap = fpc.compute(e_lap, g_lap)
    ft_lap = fp_lap["fingerprints"]["laplace__T__0"]["fisher_trace"]
    fd_lap = fp_lap["fingerprints"]["laplace__T__0"]["fisher_det"]
    lines.append(("PASS" if abs(ft_lap - exp_trace) < 1e-12 else "FAIL")
                 + f"  拉普拉斯指纹 trace：{ft_lap} vs {exp_trace}")
    lines.append(("PASS" if abs(fd_lap - exp_det) < 1e-12 else "FAIL")
                 + f"  拉普拉斯指纹 det：{fd_lap} vs {exp_det}")

    # --- 3. 谐振腔 LC：C=4.0, kT=1.0 -> σ_v=0.5 ---
    e_lc = {
        "family": "lc_circuit", "channel_template": "CT-LC-001",
        "parameters": [{"name": "C", "default": 4.0}, {"name": "kT", "default": 1.0}],
    }
    g_lc = enc.encode(e_lc)
    node_lc = g_lc.nodes[0]
    sigma_v = node_lc.params["sigma"]
    lines.append(("PASS" if node_lc.distribution_family == "DIST-GAUSS" else "FAIL")
                 + "  谐振腔 LC 编码 → DIST-GAUSS")
    lines.append(("PASS" if abs(sigma_v - 0.5) < 1e-12 else "FAIL")
                 + f"  谐振腔 σ_v = sqrt(kT/C)：{sigma_v} vs 0.5")
    fp_lc = fpc.compute(e_lc, g_lc)
    ft_lc = fp_lc["fingerprints"]["lc__T__0"]["fisher_trace"]
    exp_ft_lc = 3.0 / sigma_v**2  # 12.0
    lines.append(("PASS" if abs(ft_lc - exp_ft_lc) < 1e-12 else "FAIL")
                 + f"  谐振腔 Fisher trace = 3C/kT：{ft_lc} vs {exp_ft_lc}")

    # --- 4. 朗道编码：DIST-LANDAU, Fisher 数值积分 ---
    e_landau = {
        "family": "landau", "channel_template": "CT-LANDAU-001",
        "parameters": [
            {"name": "a", "default": 1.0},
            {"name": "b", "default": 1.0},
            {"name": "kT", "default": 1.0},
        ],
    }
    g_ld = enc.encode(e_landau)
    node_ld = g_ld.nodes[0]
    lines.append(("PASS" if node_ld.distribution_family == "DIST-LANDAU" else "FAIL")
                 + "  朗道编码 → DIST-LANDAU")
    fp_ld = fpc.compute(e_landau, g_ld)
    ft_ld = fp_ld["fingerprints"]["landau__T__0"]["fisher_trace"]
    lines.append(("PASS" if ft_ld > 0 else "FAIL")
                 + f"  朗道 Fisher trace 为正有限值：{ft_ld}")

    # --- 5. 朗道高斯极限对照：a 大时 φ⁴ 项可忽略，I(a) -> 1/(2a²) ---
    a_big = 100.0
    i_landau = fisher_landau(a_big, 1.0, 1.0)
    exp_lim = 1.0 / (2.0 * a_big**2)  # 5e-5
    rel = abs(i_landau - exp_lim) / exp_lim
    lines.append(("PASS" if rel < 1e-2 else "FAIL")
                 + f"  朗道高斯极限 I(a)=1/(2a²)：{i_landau:.6f} vs {exp_lim:.6f} (rel={rel:.2e})")

    # --- 6. 相变特征：a<0 双峰相 Fisher > a>0 单峰相（同 |a|）---
    i_low = fisher_landau(-1.0, 1.0, 1.0)  # 低温双峰
    i_high = fisher_landau(1.0, 1.0, 1.0)  # 高温单峰
    lines.append(("PASS" if i_low > i_high else "FAIL")
                 + f"  朗道相变：低温双峰 Fisher({i_low:.6f}) > 高温单峰({i_high:.6f})")

    n_pass = sum(1 for ln in lines if ln.startswith("PASS"))
    n_total = sum(1 for ln in lines if ln.startswith(("PASS", "FAIL")))
    lines.append(f"\n{n_pass}/{n_total} 通过")

    out = Path(r"d:\physics learn\laplace_resonator_landau_test.txt")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
