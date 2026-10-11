"""validate_entry.py 自测。

验证三道机器门对合法条目放行、对各类违规正确拦截。
运行：python db1/test_validate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_entry import validate_entry  # noqa: E402


# 合法的 OU 条目（基准）
def make_entry():
    return {
        "system_id": "ou_process_1d",
        "domain": "stochastic_process",
        "name": "一维 Ornstein–Uhlenbeck 过程",
        "family": "ou",
        "equations": {
            "latex": "dX = -\\theta (X - \\mu) dt + \\sigma dW",
            "sympy": "Eq(diff(X, t), -theta*(X - mu)) + sigma*dW",
            "sde": "dX = -theta*(X-mu)*dt + sigma*dW",
        },
        "parameters": [
            {"name": "theta", "symbol": "\\theta", "range": [0.1, 10.0], "default": 1.0},
            {"name": "sigma", "symbol": "\\sigma", "range": [0.1, 5.0], "default": 1.0},
            {"name": "mu", "symbol": "\\mu", "range": [-5.0, 5.0], "default": 0.0},
        ],
        "boundary_conditions": ["X(0) 由平稳分布采样"],
        "channel_template": "CT-DIFF-001",
        "analytic_invariants": [
            {"name": "stationary_dist", "expression": "N(mu, sigma^2/(2*theta))",
             "citation": "Gardiner 2004, Ch.3"}
        ],
        "fingerprint_expectation": [
            {"quantity": "fisher_trace", "analytic": "theta/sigma^2", "tol": 1.0e-3}
        ],
        "simulator_ref": "Euler–Maruyama 离散化求解器",
        "source_ref": "Gardiner, Handbook of Stochastic Methods, Ch.3",
        "strictness_level": "定理",
        "provenance": {
            "source": "AI+verified",
            "generator_model": "test-model",
            # 人工复核状态由 human_review_status 单独跟踪（默认 pending），
            # verified_by 只声称机器验证，否则触发"反虚假声明"检查
            "verified_by": "analytic_compare",
            "review_date": "2026-09-25",
        },
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
            "sde_calculus_note": "sigma^2 为方差速率",
            "thermal_bath_kT": 1.0,
            "nondimensionalized": True,
            "bridge_constants": [],
            "mi_estimator_used": "analytic",
            "ib_beta_grid": "logspace(-2, 3, 32)",
            "dpi_chain_direction": "X->T->Y",
            "edge_direction": "information_flow",
            "strictness_level": "定理",
            "anchor_formulas": [],
            "uncertainties": [],
        }
    }


def check(name, entry, attestation, expect_pass):
    report = validate_entry(entry, attestation)
    ok = report["passed"] == expect_pass
    status = "PASS" if ok else "FAIL"
    detail = "" if ok else f"  -> passed={report['passed']}, errors={report['errors']}"
    return f"{status}  {name}{detail}"


def main():
    lines = []

    # 1. 合法条目应通过
    lines.append(check("合法 OU 条目应通过", make_entry(), make_attestation(), True))

    # 2. 缺必填字段
    e = make_entry(); del e["family"]
    lines.append(check("缺必填字段应拦截", e, make_attestation(), False))

    # 3. system_id 命名错误
    e = make_entry(); e["system_id"] = "Bad-Name"
    lines.append(check("system_id 命名错误应拦截", e, make_attestation(), False))

    # 4. 额外字段（additionalProperties false）
    e = make_entry(); e["extra_field"] = 1
    lines.append(check("额外字段应拦截", e, make_attestation(), False))

    # 5. domain 枚举非法
    e = make_entry(); e["domain"] = "astrology"
    lines.append(check("domain 枚举非法应拦截", e, make_attestation(), False))

    # 6. strictness 非法
    e = make_entry(); e["strictness_level"] = "确定"
    lines.append(check("strictness 非法应拦截", e, make_attestation(), False))

    # 7. AI 生成物 anchor_formulas 非空
    e = make_entry()
    a = make_attestation(); a["convention_attestation"]["anchor_formulas"] = ["g_mumu=1/sigma^2"]
    lines.append(check("AI 生成物 anchor_formulas 非空应拦截", e, a, False))

    # 8. 自证表缺（AI 生成物）
    lines.append(check("AI 生成物缺自证表应拦截", make_entry(), None, False))

    # 9. bridge_constants 不一致
    e = make_entry()
    e["parameters"][0]["is_bridge_constant"] = True
    e["parameters"][0]["unit"] = "1/time"
    a = make_attestation()  # 自证表 bridge_constants 仍为空
    lines.append(check("bridge_constants 不一致应拦截", e, a, False))

    # 10. analytic 引用未声明符号
    e = make_entry()
    e["fingerprint_expectation"] = [{"quantity": "fisher_trace", "analytic": "alpha/beta"}]
    lines.append(check("analytic 引用未声明符号应拦截", e, make_attestation(), False))

    # 11. analytic 语法错误（残缺表达式）
    e = make_entry()
    e["fingerprint_expectation"] = [{"quantity": "fisher_trace", "analytic": "theta +"}]
    lines.append(check("analytic 语法错误应拦截", e, make_attestation(), False))

    # 12. 方差当参数（sigma_squared）
    e = make_entry()
    e["parameters"].append({"name": "sigma_squared", "symbol": "\\sigma^2",
                            "range": [0.0, 1.0], "default": 1.0})
    lines.append(check("方差当参数应拦截", e, make_attestation(), False))

    # 13. 桥接常数缺 unit
    e = make_entry()
    e["parameters"][0]["is_bridge_constant"] = True
    a = make_attestation(); a["convention_attestation"]["bridge_constants"] = ["theta"]
    lines.append(check("桥接常数缺 unit 应拦截", e, a, False))

    n_pass = sum(1 for ln in lines if ln.startswith("PASS"))
    lines.append(f"\n{n_pass}/{len(lines)} 通过")

    out = Path(r"d:\physics learn\validate_test_result.txt")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if n_pass == len(lines) - 1 else 1


if __name__ == "__main__":
    sys.exit(main())
