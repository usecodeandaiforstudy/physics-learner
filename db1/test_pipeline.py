"""ingest_pipeline 自测：完整入库链路（机器门 → 编码 → 指纹对照）。

运行：python db1/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # db1/（找 ingest_pipeline）

from ingest_pipeline import ingest  # noqa: E402


def make_ou_entry():
    return {
        "system_id": "ou_process_1d",
        "domain": "stochastic_process",
        "name": "一维 Ornstein–Uhlenbeck 过程",
        "family": "ou",
        "equations": {"latex": "dX = -\\theta (X-\\mu)dt + \\sigma dW",
                      "sympy": "Eq(diff(X,t), -theta*(X-mu))"},
        "parameters": [
            {"name": "theta", "symbol": "\\theta", "range": [0.1, 10.0], "default": 1.0},
            {"name": "sigma", "symbol": "\\sigma", "range": [0.1, 5.0], "default": 1.0},
            {"name": "mu", "symbol": "\\mu", "range": [-5.0, 5.0], "default": 0.0},
            {"name": "dt", "symbol": "\\Delta t", "range": [0.01, 10.0], "default": 1.0},
            {"name": "sigma_obs", "symbol": "\\sigma_{obs}", "range": [0.0, 5.0], "default": 0.5},
        ],
        "channel_template": "CT-DIFF-001",
        "fingerprint_expectation": [
            {"quantity": "fisher_trace", "analytic": "6*theta/sigma^2", "tol": 1.0e-3},
            {"quantity": "fisher_det", "analytic": "8*theta^2/sigma^4", "tol": 1.0e-3},
        ],
        "strictness_level": "定理",
        "provenance": {"source": "AI+verified", "generator_model": "test",
                       "verified_by": "pending", "review_date": "2026-09-25"},
        "convention_version": "1.0.0",
    }


def make_attestation():
    return {
        "convention_attestation": {
            "convention_version": "1.0.0",
            "log_base": "natural",
            "log_base_unit": "nats",
            "sigma_parameterization": "sigma_as_std",
            "sde_calculus": "ito",
            "thermal_bath_kT": 1.0,
            "bridge_constants": [],
            "mi_estimator_used": "analytic",
            "anchor_formulas": [],
            "uncertainties": [],
        }
    }


def main():
    lines = []
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    # 1. 完整链路通过
    r = ingest(make_ou_entry(), make_attestation())
    lines.append(
        ("PASS" if (r["passed"] and r["stage"] == "fingerprint") else "FAIL")
        + f"  完整链路通过（stage={r['stage']}）"
    )

    # 2. 图结构摘要（3 节点 2 边）
    g = r.get("graph", {})
    lines.append(
        ("PASS" if (len(g.get("nodes", [])) == 3 and len(g.get("edges", [])) == 2) else "FAIL")
        + f"  图结构 X→T→Y（{len(g.get('nodes', []))} 节点 {len(g.get('edges', []))} 边）"
    )

    # 3. 机器门拦截（缺必填字段 → 停在 validation 阶段）
    bad = make_ou_entry()
    del bad["family"]
    r2 = ingest(bad, make_attestation())
    lines.append(
        ("PASS" if (not r2["passed"] and r2["stage"] == "validation") else "FAIL")
        + f"  机器门拦截（stage={r2['stage']}）"
    )

    # 4. 解析对照失败（错误 analytic → 停在 fingerprint 阶段）
    bad2 = make_ou_entry()
    bad2["fingerprint_expectation"] = [
        {"quantity": "fisher_trace", "analytic": "theta/sigma^2", "tol": 1e-9}
    ]
    r3 = ingest(bad2, make_attestation())
    lines.append(
        ("PASS" if (not r3["passed"] and r3["stage"] == "fingerprint") else "FAIL")
        + f"  解析对照失败拦截（stage={r3['stage']}）"
    )

    # 5. 不支持的 family → 停在 encoding 阶段
    bad3 = make_ou_entry()
    bad3["family"] = "unknown_family"
    bad3["channel_template"] = "CT-XXX-999"
    r4 = ingest(bad3, make_attestation())
    lines.append(
        ("PASS" if (not r4["passed"] and r4["stage"] == "encoding") else "FAIL")
        + f"  不支持 family 拦截（stage={r4['stage']}）"
    )

    n_pass = sum(1 for ln in lines if ln.startswith("PASS"))
    lines.append(f"\n{n_pass}/{len(lines)} 通过")

    out = Path(r"d:\physics learn\pipeline_test_result.txt")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if n_pass == len(lines) - 1 else 1


if __name__ == "__main__":
    sys.exit(main())
