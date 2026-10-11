"""SystemEncoder（X→T→Y）+ FingerprintComputer 自测。

验证：
1. OU 编码为 3 节点 2 边的完整信道结构。
2. mi_profile = [I(X;T), I(T;Y)] 与解析值一致。
3. Fisher 摘要与解析对照通过。
4. 谐振子（单节点）仍能通过。

运行：python b_network/test_encoder.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from b_network.fingerprint_computer import FingerprintComputer  # noqa: E402
from b_network.system_encoder import SystemEncoder  # noqa: E402


def make_ou_entry():
    return {
        "system_id": "ou_process_1d",
        "family": "ou",
        "channel_template": "CT-DIFF-001",
        "parameters": [
            {"name": "theta", "default": 1.0},
            {"name": "sigma", "default": 1.0},
            {"name": "mu", "default": 0.0},
            {"name": "dt", "default": 1.0},
            {"name": "sigma_obs", "default": 0.5},
        ],
        "fingerprint_expectation": [
            {"quantity": "fisher_trace", "analytic": "6*theta/sigma^2", "tol": 1.0e-9},
            {"quantity": "fisher_det", "analytic": "8*theta^2/sigma^4", "tol": 1.0e-9},
        ],
    }


def make_ho_entry():
    return {
        "system_id": "harmonic_oscillator__thermal_equilibrium",
        "family": "harmonic_oscillator",
        "channel_template": "CT-EQ-001",
        "parameters": [
            {"name": "m", "default": 1.0},
            {"name": "omega", "default": 1.0},
            {"name": "kT", "default": 1.0, "is_bridge_constant": True, "unit": "energy"},
        ],
        "fingerprint_expectation": [
            {"quantity": "fisher_trace", "analytic": "3*m*omega^2/kT", "tol": 1.0e-9},
        ],
    }


def main():
    lines = []
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    enc = SystemEncoder()
    fpc = FingerprintComputer()

    # --- 1. OU 编码为 3 节点 2 边 ---
    ou = make_ou_entry()
    g = enc.encode(ou)
    n_nodes = len(g.nodes)
    n_edges = len(g.edges)
    lines.append(
        ("PASS" if (n_nodes == 3 and n_edges == 2) else "FAIL")
        + f"  OU 编码为 X→T→Y（{n_nodes} 节点 {n_edges} 边）"
    )

    # --- 2. 表示层 T 节点的 σ_stat = σ/sqrt(2θ) ---
    t_node = g.node_by_id("ou__T__0")
    sigma_stat = t_node.params["sigma"]
    expected_sigma_stat = 1.0 / math.sqrt(2.0)
    lines.append(
        ("PASS" if abs(sigma_stat - expected_sigma_stat) < 1e-12 else "FAIL")
        + f"  T 节点 σ_stat = σ/sqrt(2θ)  ({sigma_stat:.6f} vs {expected_sigma_stat:.6f})"
    )

    # --- 3. mi_profile 与解析值一致 ---
    # I(X;T) = -1/2 log(1 - rho^2)，rho = exp(-theta*dt)
    rho = math.exp(-1.0)
    I_XT_true = -0.5 * math.log(1 - rho**2)
    # I(T;Y) = 1/2 log((sigma_stat^2 + sigma_obs^2) / sigma_obs^2)
    I_TY_true = 0.5 * math.log((sigma_stat**2 + 0.5**2) / 0.5**2)

    fp = fpc.compute(ou, g)
    mi = fp["mi_profile"]
    lines.append(
        ("PASS" if abs(mi[0] - I_XT_true) < 1e-9 else "FAIL")
        + f"  I(X;T) = -1/2 log(1-ρ²)  ({mi[0]:.6f} vs {I_XT_true:.6f})"
    )
    lines.append(
        ("PASS" if abs(mi[1] - I_TY_true) < 1e-9 else "FAIL")
        + f"  I(T;Y) = 1/2 log(VarY/VarY|T)  ({mi[1]:.6f} vs {I_TY_true:.6f})"
    )

    # --- 4. Fisher 解析对照 ---
    lines.append(
        ("PASS" if fp["passed"] else "FAIL")
        + f"  OU fisher 解析对照 ({fp['checks']})"
    )

    # --- 5. DPI 链：I(X;T) >= I(T;Y) 不一定，但 X→Y 的 MI 应 ≤ I(X;T) ---
    # 这里只验证 mi_profile 非负且有限
    lines.append(
        ("PASS" if all(m is not None and m >= 0 for m in mi) else "FAIL")
        + f"  mi_profile 非负且有限 ({mi})"
    )

    # --- 6. 谐振子仍能通过（单节点编码，兼容） ---
    ho = make_ho_entry()
    report2 = fpc.compute(ho, enc.encode(ho))
    lines.append(
        ("PASS" if report2["passed"] else "FAIL")
        + f"  谐振子 fisher 解析对照 ({report2['checks']})"
    )

    n_pass = sum(1 for ln in lines if ln.startswith("PASS"))
    lines.append(f"\n{n_pass}/{len(lines)} 通过")

    out = Path(r"d:\physics learn\encoder_test_result.txt")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if n_pass == len(lines) - 1 else 1


if __name__ == "__main__":
    sys.exit(main())
