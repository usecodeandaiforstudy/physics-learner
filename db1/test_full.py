"""纯扩散编码 + ib_curve + DB2 写入 自测。

运行：python db1/test_full.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # db1/

from ingest_pipeline import ingest  # noqa: E402
from db2_writer import load_db2  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根

from b_network.fingerprint_computer import FingerprintComputer  # noqa: E402
from b_network.system_encoder import SystemEncoder  # noqa: E402


def make_ou_entry():
    return {
        "system_id": "ou_process_1d",
        "domain": "stochastic_process",
        "name": "OU",
        "family": "ou",
        "equations": {"latex": "dX", "sympy": "dX"},
        "parameters": [
            {"name": "theta", "symbol": "\\theta", "default": 1.0, "range": [0.1, 10.0]},
            {"name": "sigma", "symbol": "\\sigma", "default": 1.0, "range": [0.1, 5.0]},
            {"name": "mu", "symbol": "\\mu", "default": 0.0, "range": [-5.0, 5.0]},
            {"name": "dt", "symbol": "\\Delta t", "default": 1.0, "range": [0.01, 10.0]},
            {"name": "sigma_obs", "symbol": "\\sigma_{obs}", "default": 0.5, "range": [0.0, 5.0]},
        ],
        "channel_template": "CT-DIFF-001",
        "fingerprint_expectation": [
            {"quantity": "fisher_trace", "analytic": "6*theta/sigma^2", "tol": 1e-3},
        ],
        "strictness_level": "定理",
        "provenance": {"source": "AI+verified", "generator_model": "test",
                       "verified_by": "pending", "review_date": "2026-09-25"},
        "convention_version": "1.0.0",
    }


def make_attestation():
    return {"convention_attestation": {
        "convention_version": "1.0.0", "log_base": "natural",
        "anchor_formulas": [], "bridge_constants": [],
    }}


def make_diffusion_entry():
    return {
        "system_id": "diffusion_1d_rod",
        "domain": "diffusion",
        "name": "一维纯扩散",
        "family": "diffusion",
        "equations": {"latex": "du/dt = D d2u/dx2", "sympy": "Eq(diff(u,t), D*diff(u,x,x))"},
        "parameters": [
            {"name": "D", "symbol": "D", "default": 1.0, "range": [0.1, 10.0]},
            {"name": "t", "symbol": "t", "default": 1.0, "range": [0.1, 10.0]},
            {"name": "sigma_signal", "symbol": "\\sigma_s", "default": 1.0, "range": [0.1, 5.0]},
            {"name": "mu", "symbol": "\\mu", "default": 0.0, "range": [-5.0, 5.0]},
        ],
        "channel_template": "CT-DIFF-002",
        "strictness_level": "定理",
        "provenance": {"source": "AI+verified", "generator_model": "test",
                       "verified_by": "pending", "review_date": "2026-09-25"},
        "convention_version": "1.0.0",
    }


def main():
    lines = []
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    enc = SystemEncoder()
    fpc = FingerprintComputer()

    # --- 1. 纯扩散编码：σ_t = sqrt(σ_s² + 2Dt) ---
    diff = make_diffusion_entry()
    g_diff = enc.encode(diff)
    t_node = g_diff.node_by_id("diff__T__0")
    sigma_t = t_node.params["sigma"]
    expected = math.sqrt(1.0**2 + 2.0 * 1.0 * 1.0)  # sqrt(3)
    lines.append(
        ("PASS" if abs(sigma_t - expected) < 1e-12 else "FAIL")
        + f"  纯扩散 σ_t = sqrt(σ_s²+2Dt)  ({sigma_t:.6f} vs {expected:.6f})"
    )
    lines.append(
        ("PASS" if (len(g_diff.nodes) == 2 and len(g_diff.edges) == 1) else "FAIL")
        + f"  纯扩散 X→T 结构（{len(g_diff.nodes)} 节点 {len(g_diff.edges)} 边）"
    )

    # --- 2. ib_curve：OU 链，大 β 时 I_TY ≈ 端到端 I(X;Y) ---
    ou = make_ou_entry()
    g_ou = enc.encode(ou)
    ib = fpc.compute_ib_curve(g_ou, n_bins=120)
    i_xt = ib["I_XT"]
    i_ty = ib["I_TY"]
    # 端到端 I(X;Y)：Y = ρX + n，Var(Y) = ρ²σ_stat² + var_cond
    rho = math.exp(-1.0)
    sigma_stat = 1.0 / math.sqrt(2.0)
    var_cond = sigma_stat**2 * (1 - rho**2) + 0.5**2
    # I(X;Y) = 1/2 log(Var(Y)/Var(Y|X))，Var(Y) = ρ²σ_stat² + var_cond
    var_y = rho**2 * sigma_stat**2 + var_cond
    I_XY_true = 0.5 * math.log(var_y / var_cond)
    I_TY_max = i_ty[-1]  # 最大 β 处
    lines.append(
        ("PASS" if abs(I_TY_max - I_XY_true) < 0.05 else "FAIL")
        + f"  ib_curve 大β时 I_TY ≈ 端到端 I(X;Y)  ({I_TY_max:.4f} vs {I_XY_true:.4f})"
    )
    lines.append(
        ("PASS" if (len(i_xt) == 32 and np_all_monotonic(i_xt, i_ty)) else "FAIL")
        + f"  ib_curve 单调性（{len(i_xt)} 点）"
    )

    # --- 3. DB2 写入 ---
    with tempfile.TemporaryDirectory() as tmp:
        db2_path = Path(tmp) / "db2.jsonl"
        r = ingest(make_ou_entry(), make_attestation(), db2_path=str(db2_path))
        records = load_db2(db2_path)
        ok_write = r["passed"] and r["db2_written"] and len(records) == 1
        lines.append(
            ("PASS" if ok_write else "FAIL")
            + f"  DB2 写入（{len(records)} 条记录）"
        )
        if records:
            rec = records[0]
            has_fields = all(k in rec for k in
                             ["system_id", "fingerprints", "mi_profile", "ib_curve", "encoder_version"])
            lines.append(
                ("PASS" if has_fields else "FAIL")
                + f"  DB2 记录字段完整"
            )

    n_pass = sum(1 for ln in lines if ln.startswith("PASS"))
    lines.append(f"\n{n_pass}/{len(lines)} 通过")

    out = Path(r"d:\physics learn\full_test_result.txt")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if n_pass == len(lines) - 1 else 1


def np_all_monotonic(i_xt, i_ty) -> bool:
    import numpy as np
    return bool(np.all(np.diff(i_xt) >= -1e-6) and np.all(np.diff(i_ty) >= -1e-6))


if __name__ == "__main__":
    sys.exit(main())
