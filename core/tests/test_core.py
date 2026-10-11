"""核心信息计算库自测。

用标准解析结果验证各模块数学正确性。
运行方式：
    python -m pytest core/tests/test_core.py
或直接：
    python core/tests/test_core.py
"""

from __future__ import annotations

import os
import sys

# **直接运行时的导入引导（2026-10-02 修）**：文首声明「或直接 python
# core/tests/test_core.py」，但直接运行时不把仓库根加入 sys.path →
# `ModuleNotFoundError: No module named 'core'`。此处在导入前补上。
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from core.action import info_action, ib_potential
from core.fisher import (
    fisher_det_gaussian,
    fisher_gaussian_matrix,
    fisher_numerical,
    fisher_trace_gaussian,
    fisher_truncgeom_p,      # 第 162 条回归（符号 bug）
)
from core.ib import ib_blahut_arimoto, ib_curve
from core.mutual_info import mi_binning, mi_gaussian_analytic, mi_ksg


def _assert_close(a, b, rtol=1e-6, atol=1e-12, msg=""):
    assert np.allclose(a, b, rtol=rtol, atol=atol), (
        f"{msg} 期望 {b}, 实际 {a}"
    )


# ---------------------------------------------------------------------------
# Fisher 高斯解析
# ---------------------------------------------------------------------------
def test_fisher_gaussian_matrix():
    g = fisher_gaussian_matrix(1.0)
    _assert_close(g, np.array([[1.0, 0.0], [0.0, 2.0]]), msg="Fisher 矩阵 (sigma=1)")


def test_fisher_gaussian_trace_det():
    _assert_close(fisher_trace_gaussian(1.0), 3.0, msg="Fisher 迹")
    _assert_close(fisher_det_gaussian(1.0), 2.0, msg="Fisher 行列式")


def test_fisher_gaussian_scaling():
    # sigma=2 时：trace = 3/4, det = 2/16 = 1/8
    _assert_close(fisher_trace_gaussian(2.0), 3.0 / 4.0, msg="Fisher 迹 (sigma=2)")
    _assert_close(fisher_det_gaussian(2.0), 2.0 / 16.0, msg="Fisher 行列式 (sigma=2)")


# ---------------------------------------------------------------------------
# Fisher 数值 vs 解析（高斯单参数 mu）
# ---------------------------------------------------------------------------
def test_fisher_numerical_gaussian():
    # 高斯 N(mu, 1)，单参数 theta=[mu]，Fisher = 1/sigma^2 = 1
    def log_pdf(x, theta):
        mu = theta[0]
        return -0.5 * (x - mu) ** 2 - 0.5 * np.log(2 * np.pi)

    def sampler(n, theta, rng):
        return rng.normal(loc=theta[0], scale=1.0, size=(n,))

    g = fisher_numerical(log_pdf, np.array([0.0]), sampler, n_samples=50000, eps=1e-3)
    _assert_close(g, np.array([[1.0]]), rtol=0.05, msg="数值 Fisher 逼近解析")


# ---------------------------------------------------------------------------
# 高斯互信息解析
# ---------------------------------------------------------------------------
def test_mi_gaussian_independent():
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    _assert_close(mi_gaussian_analytic(cov, (1, 1)), 0.0, msg="独立高斯 MI=0")


def test_mi_gaussian_correlated():
    rho = 0.8
    cov = np.array([[1.0, rho], [rho, 1.0]])
    expected = -0.5 * np.log(1 - rho**2)
    _assert_close(mi_gaussian_analytic(cov, (1, 1)), expected, msg="相关高斯 MI")


# ---------------------------------------------------------------------------
# binning / KSG 互信息（量级与符号验证）
# ---------------------------------------------------------------------------
def test_mi_binning_independent():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 50000)
    y = rng.uniform(0, 1, 50000)
    mi = mi_binning(x, y, bins=20)
    # 独立变量 MI 应接近 0（有限样本有正偏差）
    assert mi < 0.05, f"独立变量 binning MI 应接近 0，实际 {mi}"


def test_mi_binning_dependent():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 50000)
    y = x + rng.normal(0, 0.01, 50000)  # 强相关
    mi = mi_binning(x, y, bins=20)
    assert mi > 1.0, f"强相关变量 binning MI 应较大，实际 {mi}"


def test_mi_ksg_independent():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (5000, 1))
    y = rng.normal(0, 1, (5000, 1))
    mi = mi_ksg(x, y, k=5)
    # 独立变量 KSG MI 应接近 0（KSG 有已知的有限样本负偏差）
    assert abs(mi) < 0.3, f"独立变量 KSG MI 应接近 0，实际 {mi}"


def test_mi_ksg_correlated_matches_analytic():
    # 用相关系数 rho 的二元高斯对比 KSG 估计与解析值
    rng = np.random.default_rng(2)
    rho = 0.5
    n = 8000
    x = rng.normal(0, 1, (n, 1))
    # y = rho*x + sqrt(1-rho^2)*noise，使 (x,y) 相关 rho 且边际 N(0,1)
    y = rho * x + np.sqrt(1 - rho**2) * rng.normal(0, 1, (n, 1))
    mi_est = mi_ksg(x, y, k=10)
    mi_true = -0.5 * np.log(1 - rho**2)  # ≈ 0.1438 nats
    # KSG 估计容差放宽到 25%
    assert abs(mi_est - mi_true) / abs(mi_true) < 0.25, (
        f"KSG 估计 {mi_est:.4f} 偏离解析值 {mi_true:.4f} 超过 25%"
    )


# ---------------------------------------------------------------------------
# 信息瓶颈 BA 算法
# ---------------------------------------------------------------------------
def test_ib_deterministic_channel():
    # 确定性信道：Y 完美保留 X 信息
    px = np.array([0.5, 0.5])
    p_y_given_x = np.array([[1.0, 0.0], [0.0, 1.0]])
    H = np.log(2.0)  # H(X) = H(Y) = ln2

    # beta 很大时，最优解 I(X;T) = I(T;Y) = H
    res = ib_blahut_arimoto(px, p_y_given_x, beta=100.0, n_iter=1000, seed=0)
    assert res["converged"], "BA 应收敛"
    _assert_close(res["I_XT"], H, rtol=0.01, msg="大 beta 时 I(X;T)=H(X)")
    _assert_close(res["I_TY"], H, rtol=0.01, msg="大 beta 时 I(T;Y)=H(Y)")


def test_ib_zero_beta_compresses():
    # beta -> 0 时，I(X;T) 应接近 0（完全压缩）
    px = np.array([0.5, 0.5])
    p_y_given_x = np.array([[1.0, 0.0], [0.0, 1.0]])
    res = ib_blahut_arimoto(px, p_y_given_x, beta=1e-4, n_iter=1000, seed=0)
    assert res["I_XT"] < 1e-3, f"beta->0 时 I(X;T) 应接近 0，实际 {res['I_XT']}"


def test_ib_curve_monotonic():
    px = np.array([0.5, 0.5])
    p_y_given_x = np.array([[0.9, 0.1], [0.1, 0.9]])
    beta_grid = np.logspace(-2, 2, 16)
    i_xt, i_ty = ib_curve(px, p_y_given_x, beta_grid=beta_grid, n_iter=500, seed=0)
    # I(X;T) 与 I(T;Y) 都应随 beta 非降
    assert np.all(np.diff(i_xt) >= -1e-6), "I(X;T) 应非降"
    assert np.all(np.diff(i_ty) >= -1e-6), "I(T;Y) 应非降"


# ---------------------------------------------------------------------------
# 信息作用量
# ---------------------------------------------------------------------------
def test_info_action_sign():
    g = np.array([[1.0]])
    theta_dot = np.array([1.0])
    # kinetic = 0.5; potential = I_XT - beta*I_TY = 2 - 0.5*1 = 1.5
    # L = 0.5 - 1.5 = -1.0
    L = info_action(g, theta_dot, I_XT=2.0, I_TY=1.0, beta=0.5)
    _assert_close(L, -1.0, msg="信息作用量符号约定")


def test_ib_potential():
    _assert_close(ib_potential(2.0, 1.0, 0.5), 1.5, msg="IB 势能项")


def test_truncgeom_fisher_k1_must_be_zero():
    """**k=1 时截断几何的 pmf 恒为 1（与 p 无关）⟹ I(p) 必须恰为 0**。

    钉住的缺陷（2026-10-02，第 162 条）：`fisher_truncgeom_p` 的 score 末项
    符号原为 **`+`**，与自身 pmf 不一致 ⟹ `k=1` 时给 `(1/p+1/q)²`
    （p=0.5 时 **16**）——按项目标准即"**不可能的数**"。
    """
    for p in (0.3, 0.5, 0.8):
        i = fisher_truncgeom_p(1, p)
        _assert_close(i, 0.0, rtol=0.0, atol=1e-12,
                      msg=f"k=1 p={p}：pmf 与 p 无关，I 必须为 0（实测 {i}）")


def test_truncgeom_fisher_matches_definition():
    """**与定义式（对 pmf 直接数值微分）逐点一致**（第 162 条回归）。

    定义：`I(p) = E[(∂_p log P(n))²]`，用中心差分对**自身 pmf**求导得到
    ⟹ 这是权威口径（解析式与 MC 曾**一致地错**，故必须对**pmf**核对）。
    """
    import numpy as np

    def brute(k: int, p: float, eps: float = 1e-6) -> float:
        n = np.arange(1, k + 1, dtype=float)

        def pmf(pp: float):
            qq = 1.0 - pp
            v = pp * qq ** (n - 1) / (1.0 - qq ** k)
            return v / v.sum()

        s = (np.log(pmf(p + eps)) - np.log(pmf(p - eps))) / (2 * eps)
        return float(np.sum(s ** 2 * pmf(p)))

    for k, p in ((1, 0.5), (5, 0.275), (5, 0.4), (20, 0.3)):
        a = fisher_truncgeom_p(k, p)
        b = brute(k, p)
        _assert_close(a, b, rtol=1e-3, atol=1e-9,
                      msg=f"k={k} p={p}：解析 {a:.6f} vs 定义式 {b:.6f}")


# ---------------------------------------------------------------------------
# 独立运行入口
# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 通过")
    return passed == len(tests)


if __name__ == "__main__":
    import sys

    sys.exit(0 if _run_all() else 1)
