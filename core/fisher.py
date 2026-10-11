"""Fisher 信息度量计算模块。

遵循 `db1/conventions.yaml` 第 1 组约定：
- 信息量一律自然对数（nats）
- Fisher 度量定义：g_ij(theta) = ∫ p(x;theta) * ∂_i ln p * ∂_j ln p dx
  = E[(∂_i ln p)(∂_j ln p)]
- 高斯分布族用解析式（conventions.yaml 第 2 组：参数化 (mu, sigma)，sigma 为标准差）
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as _gamma, iv, kv

__all__ = [
    "fisher_gaussian_matrix",
    "fisher_trace_gaussian",
    "fisher_det_gaussian",
    "fisher_exponential",
    "fisher_poisson",
    "fisher_bernoulli",
    "fisher_laplace_matrix",
    "fisher_trace_laplace",
    "fisher_det_laplace",
    "fisher_landau",
    "fisher_landau_monte_carlo",
    "fisher_landau_analytic",
    "fisher_gamma_matrix",
    "fisher_trace_gamma",
    "fisher_det_gamma",
    "fisher_beta_matrix",
    "fisher_trace_beta",
    "fisher_det_beta",
    "fisher_weibull_matrix",
    "fisher_trace_weibull",
    "fisher_det_weibull",
    "fisher_vonmises_mu",
    "fisher_vonmises_kappa",
    "fisher_trace_vonmises",
    "fisher_det_vonmises",
    "fisher_geometric",
    "fisher_cauchy",
    "fisher_rayleigh",
    "fisher_logistic",
    "fisher_chisq",
    "fisher_negbin",
    "fisher_lognormal_trace",
    "fisher_lognormal_det",
    "fisher_numerical",
]


# ---------------------------------------------------------------------------
# 高斯分布族的解析 Fisher
# ---------------------------------------------------------------------------
def fisher_gaussian_matrix(sigma: float) -> np.ndarray:
    """单变量高斯 N(mu, sigma) 关于参数 (mu, sigma) 的 Fisher 信息矩阵。

    标准结果：
        g_mumu   = 1 / sigma^2
        g_sigmasigma = 2 / sigma^2
        g_mu,sigma   = 0（交叉项）

    注意约定：sigma 是标准差（不是方差），方差为 sigma^2。

    参数
    ----
    sigma : float
        标准差，须 > 0。

    返回
    ----
    np.ndarray
        2x2 对称正定矩阵 [[g_mumu, 0], [0, g_sigmasigma]]。
    """
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma 必须 > 0，收到 {sigma}")
    return np.array([[1.0 / sigma**2, 0.0], [0.0, 2.0 / sigma**2]])


def fisher_trace_gaussian(sigma: float) -> float:
    """单变量高斯的 Fisher 矩阵迹 = 3 / sigma^2。"""
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma 必须 > 0，收到 {sigma}")
    return 3.0 / sigma**2


def fisher_det_gaussian(sigma: float) -> float:
    """单变量高斯的 Fisher 矩阵行列式 = 2 / sigma^4。"""
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma 必须 > 0，收到 {sigma}")
    return 2.0 / sigma**4


# ---------------------------------------------------------------------------
# 指数分布族 Exp(lambda) 的解析 Fisher
# ---------------------------------------------------------------------------
def fisher_exponential(rate: float) -> float:
    """指数分布 Exp(λ)（p(x)=λ e^{-λx}, x≥0）关于速率参数 λ 的 Fisher 信息。

    标准结果：I(λ) = 1 / λ²。

    参数
    ----
    rate : float
        速率参数 λ，须 > 0。

    返回
    ----
    float
        Fisher 信息（单参数，trace = det = I(λ)）。
    """
    rate = float(rate)
    if rate <= 0:
        raise ValueError(f"rate 必须 > 0，收到 {rate}")
    return 1.0 / rate**2


# ---------------------------------------------------------------------------
# 泊松分布族 Poisson(lambda) 的解析 Fisher
# ---------------------------------------------------------------------------
def fisher_poisson(rate: float) -> float:
    """泊松分布 Poisson(λ)（P(k)=λ^k e^{-λ}/k!）关于 λ 的 Fisher 信息。

    标准结果：I(λ) = 1 / λ。

    参数
    ----
    rate : float
        速率参数 λ，须 > 0。

    返回
    ----
    float
        Fisher 信息（单参数）。
    """
    rate = float(rate)
    if rate <= 0:
        raise ValueError(f"rate 必须 > 0，收到 {rate}")
    return 1.0 / rate


# ---------------------------------------------------------------------------
# 伯努利分布族 Bern(p) 的解析 Fisher
# ---------------------------------------------------------------------------
def fisher_bernoulli(p: float) -> float:
    """伯努利分布 Bern(p)（P(1)=p, P(0)=1-p）关于 p 的 Fisher 信息。

    标准结果：I(p) = 1 / (p (1-p))。

    参数
    ----
    p : float
        成功概率，须在 (0, 1) 开区间内（端点 Fisher 发散）。

    返回
    ----
    float
        Fisher 信息（单参数）。
    """
    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError(f"p 须在 (0,1) 内，收到 {p}")
    return 1.0 / (p * (1.0 - p))


# ---------------------------------------------------------------------------
# 拉普拉斯分布族 Laplace(mu, b) 的解析 Fisher
# ---------------------------------------------------------------------------
def fisher_laplace_matrix(b: float) -> np.ndarray:
    """拉普拉斯分布 Laplace(μ,b)（p(x)=1/(2b) e^{-|x-μ|/b}）关于 (μ,b) 的 Fisher 矩阵。

    标准结果（μ 与 b 的 Fisher 交叉项为 0）：
        g_mumu = 1 / b²
        g_bb   = 1 / b²
    注意 μ 处不可导的奇异性不影响该结果（在 Fisher 信息意义下正则）。

    参数
    ----
    b : float
        尺度参数，须 > 0。

    返回
    ----
    np.ndarray
        2x2 对角矩阵 diag(1/b², 1/b²)。
    """
    b = float(b)
    if b <= 0:
        raise ValueError(f"b 必须 > 0，收到 {b}")
    return np.diag([1.0 / b**2, 1.0 / b**2])


def fisher_trace_laplace(b: float) -> float:
    """拉普拉斯 Fisher 矩阵迹 = 2 / b²。"""
    b = float(b)
    if b <= 0:
        raise ValueError(f"b 必须 > 0，收到 {b}")
    return 2.0 / b**2


def fisher_det_laplace(b: float) -> float:
    """拉普拉斯 Fisher 矩阵行列式 = 1 / b⁴。"""
    b = float(b)
    if b <= 0:
        raise ValueError(f"b 必须 > 0，收到 {b}")
    return 1.0 / b**4


# ---------------------------------------------------------------------------
# 朗道序参量分布的 Fisher（数值积分）
# ---------------------------------------------------------------------------
def fisher_landau(a: float, b: float, kT: float, n_grid: int = 2000) -> float:
    """朗道自由能 F(φ) = ½aφ² + ¼bφ⁴ 的序参量分布 Fisher（关于约化温度系数 a）。

    序参量平衡分布 p(φ) ∝ exp(-F(φ)/kT)。关于 a 的 Fisher：
        I(a) = Var(φ²) / (4 k² T²)，Var(φ²) = ⟨φ⁴⟩ - ⟨φ²⟩²
    （因 ∂_a ln p = -(φ² - ⟨φ²⟩)/(2kT)，用梯形法数值积分 ⟨φ²⟩、⟨φ⁴⟩。）

    单参数（关于 a），故 trace = det = I(a)。

    参数
    ----
    a : float
        约化温度系数（a>0 高温相单峰，a<0 低温相双峰）。
    b : float
        φ⁴ 项系数，须 > 0（保证可归一化）。
    kT : float
        温度（能量单位），须 > 0。
    n_grid : int
        数值积分网格点数。

    返回
    ----
    float
        Fisher 信息（关于 a）。
    """
    a = float(a)
    b = float(b)
    kT = float(kT)
    if b <= 0 or kT <= 0:
        raise ValueError(f"b={b}, kT={kT} 均须 > 0")

    # φ 范围：φ⁴ 项给出特征宽度 (kT/b)^{1/4}；低温相双峰在 ±√(-a/b)
    phi_width = (kT / b) ** 0.25
    phi_max = 10.0 * phi_width
    if a < 0:
        phi_max = max(phi_max, 5.0 * (abs(a) / b) ** 0.5)

    phi = np.linspace(-phi_max, phi_max, n_grid)
    F = 0.5 * a * phi**2 + 0.25 * b * phi**4
    logp = -F / kT
    logp -= logp.max()
    p = np.exp(logp)
    Z = np.trapezoid(p, phi)
    p /= Z

    m2 = float(np.trapezoid(phi**2 * p, phi))
    m4 = float(np.trapezoid(phi**4 * p, phi))
    var_phi2 = m4 - m2**2
    return var_phi2 / (4.0 * kT**2)


def fisher_landau_monte_carlo(
    a: float,
    b: float,
    kT: float,
    n_samples: int = 10000,
    n_burnin: int = 2000,
    seed: int = 0,
) -> float:
    """朗道序参量分布 Fisher 的蒙特卡洛估计（与 fisher_landau 数值积分对照）。

    用 Metropolis-Hastings 从 p(φ) ∝ exp(-F(φ)/kT) 采样，样本矩估计
    Var(φ²) = ⟨φ⁴⟩ - ⟨φ²⟩²，I(a) = Var(φ²)/(4k²T²)。

    与 fisher_landau 用同一公式、不同估计手段（采样 vs 梯形积分），两者一致
    说明数值积分未算错。低温双峰相（a<0）存在势垒，proposal 步长会扩展以覆盖
    峰间距，但仍可能有采样误差——故对照容差需放宽（见 fingerprint_computer）。

    参数
    ----
    a, b, kT : 同 fisher_landau。
    n_samples : 有效样本数。
    n_burnin : 预热（burn-in）样本数。
    seed : 随机种子（固定保证可复现）。

    返回
    ----
    float
        Fisher 信息（关于 a）的蒙特卡洛估计。
    """
    a = float(a)
    b = float(b)
    kT = float(kT)
    if b <= 0 or kT <= 0:
        raise ValueError(f"b={b}, kT={kT} 均须 > 0")

    def logp(phi: float) -> float:
        F = 0.5 * a * phi**2 + 0.25 * b * phi**4
        return -F / kT

    # proposal 步长：特征宽度 (kT/b)^{1/4}；低温双峰相扩展覆盖峰间距
    phi_width = (kT / b) ** 0.25
    if a < 0:
        phi_width = max(phi_width, 0.5 * (abs(a) / b) ** 0.5)

    rng = np.random.default_rng(seed)
    phi = 0.0
    samples = np.empty(n_samples)
    idx = 0
    for i in range(n_burnin + n_samples):
        phi_new = phi + rng.normal(0.0, phi_width)
        log_ratio = logp(phi_new) - logp(phi)
        if log_ratio >= 0.0 or rng.random() < np.exp(log_ratio):
            phi = phi_new
        if i >= n_burnin:
            samples[idx] = phi
            idx += 1

    m2 = float(np.mean(samples**2))
    m4 = float(np.mean(samples**4))
    var_phi2 = m4 - m2**2
    return var_phi2 / (4.0 * kT**2)


def fisher_landau_analytic(a: float, b: float, kT: float) -> float:
    """朗道序参量分布 Fisher 的解析解（用修正贝塞尔函数闭式）。

    朗道分布 p(φ) ∝ exp(-F/kT)，F = ½aφ² + ¼bφ⁴，无初等解析 Fisher，但有
    特殊函数闭式（关于 a 的单参数 Fisher）：

        I(a) = Var(φ²)/(4k²T²)，Var(φ²) = ⟨φ⁴⟩ - ⟨φ²⟩²

    令 α = a/(2kT)，β = b/(4kT)，归一化 Z = ∫ exp(-αφ² - βφ⁴) dφ，则
    ⟨φ²⟩ = -∂lnZ/∂α、⟨φ⁴⟩ = -∂lnZ/∂β。Z 分三种相：

    - a > 0（高温单峰相）：Z 用 K_{1/4}（第二类修正贝塞尔函数），
      ⟨φ²⟩、⟨φ⁴⟩ 是 K_{3/4}、K_{5/4} 的有理组合。
    - a < 0（低温双峰相）：Z 用 I_{±1/4}（第一类修正贝塞尔函数），
      ⟨φ²⟩、⟨φ⁴⟩ 是 I_{±3/4}、I_{±5/4} 的有理组合。
    - a = 0（纯 φ⁴ 势）：矩是 Γ 函数之比，I(0) = [¼ - (Γ(¾)/Γ(¼))²]/(bkT)。

    与 fisher_landau（梯形积分）对照，机器精度一致即验证数值积分正确。

    参数
    ----
    a, b, kT : 同 fisher_landau（b > 0, kT > 0）。

    返回
    ----
    float
        Fisher 信息（关于 a）的解析值。
    """
    a = float(a)
    b = float(b)
    kT = float(kT)
    if b <= 0 or kT <= 0:
        raise ValueError(f"b={b}, kT={kT} 均须 > 0")

    if abs(a) < 1e-12:
        # 纯 φ⁴ 势：⟨φ²⟩、⟨φ⁴⟩ 为 Γ 函数之比
        g = _gamma(0.75) / _gamma(0.25)
        return (0.25 - g**2) / (b * kT)

    alpha = a / (2.0 * kT)
    beta = b / (4.0 * kT)

    if a > 0:
        # 高温单峰相：K_{1/4}，t = α²/(8β)
        t = alpha**2 / (8.0 * beta)
        k1 = kv(0.25, t)
        k3 = kv(0.75, t)
        k5 = kv(1.25, t)
        m2 = alpha * (k3 + k5) / (8.0 * beta * k1) - alpha / (4.0 * beta) - 1.0 / (2.0 * alpha)
        m4 = (alpha**2 * (-k3 + 2.0 * k1 - k5) + 8.0 * beta * k1) / (16.0 * beta**2 * k1)
    else:
        # 低温双峰相：I_{±1/4}，γ = -α/2 > 0，t = γ²/(2β)
        gamma = -alpha / 2.0
        t = gamma**2 / (2.0 * beta)
        im14 = iv(-0.25, t)
        ip14 = iv(0.25, t)
        im34 = iv(-0.75, t)
        ip34 = iv(0.75, t)
        im54 = iv(-1.25, t)
        ip54 = iv(1.25, t)
        s1 = im14 + ip14
        s2 = im54 + im34 + 2.0 * im14 + 2.0 * ip14 + ip34 + ip54
        m2 = (beta * s1 + gamma**2 * s2) / (4.0 * beta * gamma * s1)
        m4 = (2.0 * beta * s1 + gamma**2 * s2) / (4.0 * beta**2 * s1)

    var_phi2 = m4 - m2**2
    return var_phi2 / (4.0 * kT**2)


# ---------------------------------------------------------------------------
# 伽马分布族 Gamma(k, theta) 的解析 Fisher
# ---------------------------------------------------------------------------
def fisher_gamma_matrix(k: float, theta: float) -> np.ndarray:
    """伽马分布 Gamma(k, θ)（形状 k、尺度 θ）关于 (k, θ) 的 Fisher 矩阵。

    标准结果（k-θ 参数化，非对角元非零）：
        g_kk     = ψ₁(k)（trigamma 函数）
        g_θθ     = k / θ²
        g_kθ     = 1 / θ
    本族是库中首个 Fisher 矩阵带非对角元的分布族——对 A 网络的
    图结构学习是新的可区分特征（参数耦合方向）。

    参数
    ----
    k : float
        形状参数，须 > 0（k=1 退化为指数分布，k=ν/2 为 chi2）。
    theta : float
        尺度参数，须 > 0。

    返回
    ----
    np.ndarray
        2x2 对称矩阵 [[ψ₁(k), 1/θ], [1/θ, k/θ²]]。
    """
    from scipy.special import polygamma

    k = float(k)
    theta = float(theta)
    if k <= 0 or theta <= 0:
        raise ValueError(f"k={k}, theta={theta} 均须 > 0")
    g_kk = float(polygamma(1, k))
    return np.array([[g_kk, 1.0 / theta], [1.0 / theta, k / theta**2]])


def fisher_trace_gamma(k: float, theta: float) -> float:
    """伽马 Fisher 矩阵迹 = ψ₁(k) + k/θ²。"""
    from scipy.special import polygamma

    k = float(k)
    theta = float(theta)
    if k <= 0 or theta <= 0:
        raise ValueError(f"k={k}, theta={theta} 均须 > 0")
    return float(polygamma(1, k)) + k / theta**2


def fisher_det_gamma(k: float, theta: float) -> float:
    """伽马 Fisher 矩阵行列式 = (k·ψ₁(k) − 1) / θ²。"""
    from scipy.special import polygamma

    k = float(k)
    theta = float(theta)
    if k <= 0 or theta <= 0:
        raise ValueError(f"k={k}, theta={theta} 均须 > 0")
    return (k * float(polygamma(1, k)) - 1.0) / theta**2


# ---------------------------------------------------------------------------
# R4c（2026-09-29）：逆伽马（gamma 输入模式同构训练族）
# ---------------------------------------------------------------------------
def fisher_invgamma_trace(k: float, theta: float) -> float:
    """逆伽马 IG(k, θ) Fisher 迹 = ψ₁(k) + k/θ²——与伽马完全同形。

    依据：样本空间微分同胚 x → 1/x 保持 Fisher 不变
    （IG(k,θ) = {1/X : X ~ Gamma(k,θ)}）。R4c 目标：与 gamma 同构的
    输入模式（k 列 + theta 列同时填充）+ ψ₁(k)+k/θ² 目标。
    """
    from scipy.special import polygamma

    k = float(k)
    theta = float(theta)
    if k <= 0 or theta <= 0:
        raise ValueError(f"k={k}, theta={theta} 均须 > 0")
    return float(polygamma(1, k)) + k / theta**2


def fisher_invgamma_det(k: float, theta: float) -> float:
    """逆伽马 IG(k, θ) Fisher 行列式 = (k·ψ₁(k) − 1)/θ²（与伽马同形）。"""
    from scipy.special import polygamma

    k = float(k)
    theta = float(theta)
    if k <= 0 or theta <= 0:
        raise ValueError(f"k={k}, theta={theta} 均须 > 0")
    return (k * float(polygamma(1, k)) - 1.0) / theta**2


# ---------------------------------------------------------------------------
# 贝塔分布族 Beta(alpha, beta) 的解析 Fisher
# ---------------------------------------------------------------------------
def fisher_beta_matrix(alpha: float, beta: float) -> np.ndarray:
    """贝塔分布 Beta(α, β) 关于 (α, β) 的 Fisher 矩阵。

    标准结果（非对角元非零）：
        g_αα = ψ₁(α) − ψ₁(α+β)
        g_ββ = ψ₁(β) − ψ₁(α+β)
        g_αβ = −ψ₁(α+β)
    本族是库中首个有界支撑 [0,1] 的连续分布族（占据分数/混合权重类物理）。

    参数
    ----
    alpha, beta : float
        两个形状参数，均须 > 0。

    返回
    ----
    np.ndarray
        2x2 对称矩阵。
    """
    from scipy.special import polygamma

    alpha = float(alpha)
    beta = float(beta)
    if alpha <= 0 or beta <= 0:
        raise ValueError(f"alpha={alpha}, beta={beta} 均须 > 0")
    g_aa = float(polygamma(1, alpha)) - float(polygamma(1, alpha + beta))
    g_bb = float(polygamma(1, beta)) - float(polygamma(1, alpha + beta))
    g_ab = -float(polygamma(1, alpha + beta))
    return np.array([[g_aa, g_ab], [g_ab, g_bb]])


def fisher_trace_beta(alpha: float, beta: float) -> float:
    """贝塔 Fisher 矩阵迹 = ψ₁(α) + ψ₁(β) − 2ψ₁(α+β)。"""
    from scipy.special import polygamma

    alpha = float(alpha)
    beta = float(beta)
    if alpha <= 0 or beta <= 0:
        raise ValueError(f"alpha={alpha}, beta={beta} 均须 > 0")
    return (float(polygamma(1, alpha)) + float(polygamma(1, beta))
            - 2.0 * float(polygamma(1, alpha + beta)))


def fisher_det_beta(alpha: float, beta: float) -> float:
    """贝塔 Fisher 矩阵行列式 = ψ₁(α)ψ₁(β) − ψ₁(α+β)·[ψ₁(α)+ψ₁(β)]。"""
    from scipy.special import polygamma

    alpha = float(alpha)
    beta = float(beta)
    if alpha <= 0 or beta <= 0:
        raise ValueError(f"alpha={alpha}, beta={beta} 均须 > 0")
    g_a = float(polygamma(1, alpha))
    g_b = float(polygamma(1, beta))
    g_s = float(polygamma(1, alpha + beta))
    return g_a * g_b - g_s * (g_a + g_b)


# ---------------------------------------------------------------------------
# 威布尔分布族 Weibull(shape, scale) 的解析 Fisher
# ---------------------------------------------------------------------------
def fisher_weibull_matrix(shape: float, scale: float) -> np.ndarray:
    """威布尔分布（形状 k、尺度 λ）关于 (k, λ) 的 Fisher 矩阵。

    标准结果（记 γ_E 为欧拉常数）：
        g_λλ = k² / λ²            （因 (x/λ)^k ~ Exp(1)，Var=1）
        g_kk = [(1−γ_E)² + π²/6] / k²
        g_kλ = (γ_E − 1) / λ       （负交叉项）
    行列式与形状参数无关：det = π² / (6λ²)——可区分尺度方向的整体信息量
    只由 λ 控制，形状方向的全部信息进入 trace 而不进入 det。

    参数
    ----
    shape : float
        形状参数 k，须 > 0（k=1 退化为指数，k<1 早期失效，k>1 磨损型）。
    scale : float
        尺度参数 λ，须 > 0。

    返回
    ----
    np.ndarray
        2x2 对称矩阵。
    """
    shape = float(shape)
    scale = float(scale)
    if shape <= 0 or scale <= 0:
        raise ValueError(f"shape={shape}, scale={scale} 均须 > 0")
    gamma_e = 0.5772156649015329
    g_kk = ((1.0 - gamma_e) ** 2 + np.pi**2 / 6.0) / shape**2
    return np.array([
        [g_kk, (gamma_e - 1.0) / scale],
        [(gamma_e - 1.0) / scale, shape**2 / scale**2],
    ])


def fisher_trace_weibull(shape: float, scale: float) -> float:
    """威布尔 Fisher 矩阵迹 = k²/λ² + [(1−γ_E)²+π²/6]/k²。"""
    m = fisher_weibull_matrix(shape, scale)
    return float(np.trace(m))


def fisher_det_weibull(shape: float, scale: float) -> float:
    """威布尔 Fisher 矩阵行列式 = π² / (6λ²)（与形状无关）。"""
    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"scale={scale} 须 > 0")
    return float(np.pi**2 / (6.0 * scale**2))


# ---------------------------------------------------------------------------
# 冯·米塞斯分布族 VonMises(mu, kappa) 的解析 Fisher
# ---------------------------------------------------------------------------
def fisher_vonmises_mu(kappa: float) -> float:
    """冯·米塞斯分布 VM(μ, κ) 关于 μ 的 Fisher 信息（解析）。

    标准结果：I_μμ = κ·A(κ)，A(κ) = I₁(κ)/I₀(κ)（修正贝塞尔函数比，
    即圆上的"合成的平均合成向量长度"）。κ→∞ 时 A→1（高斯极限
    N(μ, 1/κ)，与高斯族 I_μμ=1/σ² 一致），κ→0 时 A→κ/2（均匀极限）。

    参数
    ----
    kappa : float
        集中度参数，须 > 0。

    返回
    ----
    float
        I(μ) = κ·I₁(κ)/I₀(κ)。
    """
    kappa = float(kappa)
    if kappa <= 0:
        raise ValueError(f"kappa={kappa} 须 > 0")
    return kappa * float(iv(1, kappa) / iv(0, kappa))


def fisher_vonmises_kappa(kappa: float) -> float:
    """冯·米塞斯分布关于 κ 的 Fisher 信息（Bessel 闭式）。

    I_κκ = d²/dκ² ln I₀(κ) = ½·[1 + I₂(κ)/I₀(κ)] − [I₁(κ)/I₀(κ)]²。

    推导：∂_κ ln p = cos(θ−μ) − A(κ)，A(κ) = I₁/I₀ = E[cos(θ−μ)]，故
    I_κκ = Var[cos(θ−μ)] = E[cos²] − A²；由 E[cos nθ] = Iₙ(κ)/I₀(κ) 得
    E[cos²θ] = ½(1 + I₂/I₀)。与 d²/dκ² ln I₀ 一致（其一阶导为 A，Bessel
    递推 I₀'=I₁、I₁'=½(I₀+I₂) 给出同一表达式）。
    与 I_μκ = 0（圆平移对称性，位置与集中度参数正交）。

    历史注记：此前用中心差分近似 d²/dκ² ln I₀，但 h=1e-6 时二阶差分
    分子（~I_κκ·h² ~ 1e-14）已被 ln I₀ 的浮点舍入噪声（~1e-15）淹没，
    h=1e-5 与 h=1e-6 的系统偏差 ~5e-4 超过步长稳定性容差（1e-4），
    故改用闭式——更强：可与 sympy besseli 做双库对照（fingerprint_computer
    的 5bis.2 检查）。

    参数
    ----
    kappa : float
        集中度参数，须 > 0。

    返回
    ----
    float
        I(κ) = ½(1 + I₂/I₀) − (I₁/I₀)²。
    """
    kappa = float(kappa)
    if kappa <= 0:
        raise ValueError(f"kappa={kappa} 须 > 0")
    i0 = float(iv(0, kappa))
    i1 = float(iv(1, kappa))
    i2 = float(iv(2, kappa))
    r1 = i1 / i0
    return 0.5 * (1.0 + i2 / i0) - r1 * r1


def fisher_trace_vonmises(kappa: float) -> float:
    """冯·米塞斯 Fisher 矩阵迹 = κ·A(κ) + ½(1+I₂/I₀) − (I₁/I₀)²。"""
    return fisher_vonmises_mu(kappa) + fisher_vonmises_kappa(kappa)


def fisher_det_vonmises(kappa: float) -> float:
    """冯·米塞斯 Fisher 矩阵行列式 = κ·A(κ) · [½(1+I₂/I₀) − (I₁/I₀)²]。"""
    return fisher_vonmises_mu(kappa) * fisher_vonmises_kappa(kappa)


# ---------------------------------------------------------------------------
# 数值 Fisher（兜底，用于非高斯/无解析式的分布族）
# ---------------------------------------------------------------------------
def fisher_numerical(
    log_pdf: "callable",
    theta: np.ndarray,
    sampler: "callable",
    n_samples: int = 20000,
    eps: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """用 score 函数 + 有限差分估计 Fisher 信息矩阵。

    原理：g_ij = E[(∂_i ln p)(∂_j ln p)]，其中 score s_i = ∂_i ln p。
    用中心差分估计 score：
        s_i(x) ≈ [ln p(x; theta + e_i eps) - ln p(x; theta - e_i eps)] / (2 eps)
    再用蒙特卡洛平均 s_i * s_j。

    注意：这是数值近似，仅用于无解析式时的兜底。指纹入库时应记录
    该方法带来的不确定度（对齐 conventions.yaml 第 1 组估计器约定）。

    参数
    ----
    log_pdf : callable(x, theta) -> float
        对数概率密度（自然对数）。
    theta : np.ndarray
        参数向量，形状 (d,)。
    sampler : callable(n_samples, theta, rng) -> np.ndarray
        从 p(x; theta) 采样，返回形状 (n_samples, ...) 的样本。
    n_samples : int
        蒙特卡洛样本数。
    eps : float
        有限差分步长。
    rng : np.random.Generator, 可选
        随机数发生器。

    返回
    ----
    np.ndarray
        形状 (d, d) 的 Fisher 矩阵估计。
    """
    if rng is None:
        rng = np.random.default_rng(0)

    theta = np.asarray(theta, dtype=float)
    d = theta.shape[0]

    # 采样一个固定批次（score 在同一批样本上评估，减少方差）
    xs = sampler(n_samples, theta, rng)

    # 计算 score 矩阵：形状 (d, n_samples)
    score = np.zeros((d, n_samples))
    for i in range(d):
        th_p = theta.copy()
        th_m = theta.copy()
        th_p[i] += eps
        th_m[i] -= eps
        lp_p = np.asarray([log_pdf(x, th_p) for x in xs])
        lp_m = np.asarray([log_pdf(x, th_m) for x in xs])
        score[i] = (lp_p - lp_m) / (2.0 * eps)

    # g_ij = (1/n) sum_k s_i(x_k) s_j(x_k)
    g = score @ score.T / n_samples
    return g


# ---------------------------------------------------------------------------
# 扩族批次 2（2026-09-26）：几何 / 柯西 / 瑞利 / 逻辑斯蒂
# ---------------------------------------------------------------------------
def fisher_geometric(p: float) -> float:
    """几何分布 Geo(p)（P(X=k)=p(1-p)^(k-1), k>=1）关于成功概率 p 的 Fisher。

    标准结果：I(p) = 1 / (p²(1-p))（首次成功等待参数化）。
    本族是离散分布族谱系（泊松 1/λ、伯努利 1/(p(1-p))）的第三个成员，
    Fisher 结构为非对称非幂律（p²(1-p) 分母），且 p→0 时发散（罕见事件信息极大）。

    参数
    ----
    p : float
        成功概率，须在 (0, 1) 开区间。

    返回
    ----
    float
        Fisher 信息（单参数，trace = det = I(p)）。
    """
    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError(f"p 须在 (0,1) 内，收到 {p}")
    return 1.0 / (p**2 * (1.0 - p))


def fisher_cauchy(gamma: float) -> float:
    """柯西分布 Cauchy(0, γ) 关于尺度参数 γ 的 Fisher（重尾）。

    标准结果（位置 x0=0 固定，仅关于尺度 γ）：
        I(γ) = 1 / (2γ²)
    与高斯关于 σ 的 Fisher（2/σ²）同为幂律 -2，但系数 1/2 不同；
    重尾导致信息更少（Fisher 更小）——本族是库中首个"有限 Fisher 但无限方差"
    的分布族（与高斯有限方差对照）。

    参数
    ----
    gamma : float
        尺度参数，须 > 0（半高半宽）。

    返回
    ----
    float
        Fisher 信息（单参数）。
    """
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError(f"gamma 须 > 0，收到 {gamma}")
    return 1.0 / (2.0 * gamma**2)


def fisher_rayleigh(sigma: float) -> float:
    """瑞利分布 Rayleigh(σ) 关于尺度参数 σ 的 Fisher（正支撑）。

    标准结果：I(σ) = 4 / σ²（Rayleigh 四阶矩 E[X⁴]=8σ⁴ 给出系数 4）。
    物理：二维各向同性高斯分量的幅值（风速、散射强度、湍流脉动）。

    参数
    ----
    sigma : float
        尺度参数，须 > 0。

    返回
    ----
    float
        Fisher 信息（单参数）。
    """
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma 须 > 0，收到 {sigma}")
    return 4.0 / sigma**2


def fisher_logistic(scale: float) -> float:
    """逻辑斯蒂分布 Logistic(0, s) 关于尺度参数 s 的 Fisher（S 形）。

    标准结果（位置 μ=0 固定，仅关于尺度 s）：
        I(s) = (π² + 3) / (9s²)
    系数 (π²+3)/9 ≈ 1.430 独特（介于高斯 2 与柯西 1/2 之间），
    且 π² 项是本族 Fisher 的代数标志。物理：S 形生长、激活阈值、相界序参量。

    参数
    ----
    scale : float
        尺度参数 s，须 > 0。

    返回
    ----
    float
        Fisher 信息（单参数）。
    """
    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"scale 须 > 0，收到 {scale}")
    return (np.pi**2 + 3.0) / (9.0 * scale**2)


# ---------------------------------------------------------------------------
# 扩族批次 3（2026-09-27）：卡方（ψ₁ 族）/ 负二项（计数族）/ 对数正态（乘性噪声）
# ---------------------------------------------------------------------------
def fisher_chisq(k: float) -> float:
    """卡方分布 χ²(k) 关于自由度 k 的 Fisher（纯 trigamma 曲线）。

    标准结果（χ²(k) = Gamma(k/2, 2)，链式换元 dα/dk = 1/2）：
        I(k) = ψ₁(k/2) / 4 = polygamma(1, k/2) / 4

    本族是库中首个**纯 ψ₁（trigamma）Fisher 曲线**的分布族。动机：A 网络
    族级 OOD 逐族分解显示 gamma（trace = ψ₁(k) + k/θ²）为主要失败源——训练库
    中无任何纯 ψ₁ 曲线族，gamma 的 ψ₁(k) 项不可借（HANDOVER 第 24 条）。

    参数
    ----
    k : float
        自由度，须 > 0（k=2 退化为指数 Exp(1/2)）。

    返回
    ----
    float
        Fisher 信息（单参数，trace = det = I(k)）。
    """
    from scipy.special import polygamma

    k = float(k)
    if k <= 0:
        raise ValueError(f"k 须 > 0，收到 {k}")
    return float(polygamma(1, k / 2.0)) / 4.0


def fisher_negbin(k: float, p: float) -> float:
    """负二项分布 NB(k, p) 关于成功概率 p 的 Fisher（k 为结构常数）。

    参数化与本库几何严格一致（第 k 次成功等待）：
        P(X=n) = C(n-1, k-1) p^k (1-p)^(n-k), n >= k
    标准结果（k 固定，关于 p 的条件 Fisher；k=1 严格退化为几何）：
        I(p) = k / (p²(1-p))

    物理为泊松-伽马混合（超散计数：热光光子/闪烁/暗计数成簇），
    补离散计数族覆盖（泊松/伯努利/几何之后的第四成员）。
    注意：I(k) 无初等闭式（含 E[ψ₁(X+k)] 无穷级数），故 Fisher 摘要
    仅声明关于 p 的条件信息（见 conventions.yaml negative_binomial 条目）。

    参数
    ----
    k : float
        成功次数（= Gamma-Poisson 混合的伽马形状），须 > 0。
    p : float
        成功概率，须在 (0, 1) 开区间。

    返回
    ----
    float
        Fisher 信息（单参数，trace = det = I(p)）。
    """
    k = float(k)
    p = float(p)
    if k <= 0:
        raise ValueError(f"k 须 > 0，收到 {k}")
    if not (0.0 < p < 1.0):
        raise ValueError(f"p 须在 (0,1) 内，收到 {p}")
    return k / (p**2 * (1.0 - p))


def fisher_lognormal_trace(sigma: float) -> float:
    """对数正态 LN(μ, σ) 的 Fisher 矩阵迹 = 3/σ²（与高斯同形）。

    推导：X = exp(Y)，Y ~ N(μ, σ)；单调双射保持 Fisher 不变，
    故 (μ, σ) 的 Fisher 矩阵 = diag(1/σ², 2/σ²)，trace = 3/σ²。
    """
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma 须 > 0，收到 {sigma}")
    return 3.0 / sigma**2


def fisher_lognormal_det(sigma: float) -> float:
    """对数正态 LN(μ, σ) 的 Fisher 行列式 = 2/σ⁴（与高斯同形）。"""
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma 须 > 0，收到 {sigma}")
    return 2.0 / sigma**4


# ---------------------------------------------------------------------------
# 扩族批次 4（2026-09-29，R4 平膝形状族）：双高斯混合 / 截断高斯
# ---------------------------------------------------------------------------
def fisher_mix2_separation(d: float, sigma: float = 1.0,
                           n_grid: int = 2000) -> float:
    """对称双高斯混合关于分离度 d 的 Fisher（数值积分）。

    p(x) = ½N(x|−d/2,σ) + ½N(x|+d/2,σ)，w=½ 对称；
    score s(x) = (z2·p2 − z1·p1) / (4σ²·p)，z1,2 = (x±d/2)/σ，p1,2 为分量密度。

    形状（R4 目标：库中首个带饱和平台的曲线；实测自测确认无过冲）：
      d=0   → 0（对称性使 score 恒零）；
      单调上升 → 1/(4σ²) 平台（d≳4σ 双峰可分辨，每峰只携带局部信息）；
      注：初版注释猜测 d≈2σ 处有过冲峰，数值/MC 一致否定（I(2)=0.184 <
      I(6)=0.250）——实为 S 形饱和曲线（斜率双膝：上升膝 + 平台膝）。

    数值方案：等距网格 ±(half_width+d/2)，half_width=12σ；与 MC 对照
    （fisher_mix2_monte_carlo）在测试脚本中验证。
    """
    d = abs(float(d))
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma 须 > 0，收到 {sigma}")
    half = 12.0 * sigma + d / 2.0
    x = np.linspace(-half, half, n_grid)
    dx = x[1] - x[0]
    z1 = (x + d / 2.0) / sigma
    z2 = (x - d / 2.0) / sigma
    p1 = np.exp(-z1 ** 2 / 2.0) / (np.sqrt(2.0 * np.pi) * sigma)
    p2 = np.exp(-z2 ** 2 / 2.0) / (np.sqrt(2.0 * np.pi) * sigma)
    p = 0.5 * (p1 + p2)
    s = (z2 * p2 - z1 * p1) / (4.0 * sigma ** 2 * p)
    return float(np.sum(s ** 2 * p) * dx)


def fisher_mix2_monte_carlo(d: float, sigma: float = 1.0,
                            n_samples: int = 200_000, seed: int = 0) -> float:
    """fisher_mix2_separation 的 MC 对照（自测用）。

    从混合分布采样，score 按解析式在采样点求值，I(d) ≈ mean(s²)。
    """
    d = abs(float(d))
    sigma = float(sigma)
    rng = np.random.default_rng(seed)
    side = rng.random(n_samples) < 0.5
    x = rng.normal(-d / 2.0, sigma, n_samples)
    x[side] = rng.normal(d / 2.0, sigma, int(side.sum()))
    z1 = (x + d / 2.0) / sigma
    z2 = (x - d / 2.0) / sigma
    p1 = np.exp(-z1 ** 2 / 2.0) / (np.sqrt(2.0 * np.pi) * sigma)
    p2 = np.exp(-z2 ** 2 / 2.0) / (np.sqrt(2.0 * np.pi) * sigma)
    p = 0.5 * (p1 + p2)
    s = (z2 * p2 - z1 * p1) / (4.0 * sigma ** 2 * p)
    return float(np.mean(s ** 2))


def fisher_truncnorm_tau(tau: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """右截断高斯关于截断点 tau 的 Fisher（解析）。

    p(x) = φ(z')/(σ·(1−Φ(z)))，x > τ；z = (τ−μ)/σ，z' = (x−μ)/σ。
    score w.r.t. τ 在支撑内为常数 φ(z)/(σ(1−Φ(z)))，
    I(τ) = φ(z)² / (σ²·(1−Φ(z))²)。

    形状（R4 目标：「平→陡」上升膝）：
      z→−∞  → 0（截断无信息）；
      z≈均值 → 陡升（φ(z)≈0.4，1−Φ(z)≈0.5 → I≈0.64/σ²）；
      z→+∞  → ∞（尾巴截断，信息发散）。
    """
    from scipy.stats import norm
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma 须 > 0，收到 {sigma}")
    z = (float(tau) - float(mu)) / sigma
    phi = norm.pdf(z)
    phic = norm.cdf(z)
    return float(phi ** 2 / (sigma ** 2 * (1.0 - phic) ** 2))


# ---------------------------------------------------------------------------
# R4b（2026-09-29）：有限支撑截断几何（negbin 输入模式同构训练族）
# ---------------------------------------------------------------------------
def fisher_truncgeom_p(k: float, p: float) -> float:
    """有限支撑截断几何关于 p 的 Fisher（k 项有限和，无积分误差）。

    P(n) = p·q^(n−1)/(1−q^k)，n=1..k，q=1−p；
    score(n) = 1/p − (n−1)/q **−** k·q^(k−1)/(1−q^k)。

    **2026-10-02 修正（符号 bug，第 162 条）**：末项原为 **`+`**，与自身 pmf
    不一致。**决定性检验**：`k=1` 时 `P(1)=p/(1−q)=1` 恒为 1、**与 p 无关**
    ⟹ `I(p)` 必须 **恰为 0**；原式给 `(1/p+1/q)²`（p=0.5 时 **16**）——按项目
    自己的标准即"**不可能的数**"。
    另用**定义式暴力法**（对 pmf 直接数值微分求 `E[s²]`）核对：
      (k=1,p=0.5) 0.0000✓ ｜ (5,0.275) 3.3416 ｜ (5,0.4) 4.0677 ｜ (20,0.3) 15.2206
    修正后逐点与暴力法一致；`k→∞` 时末项 `→0`，故**文档声称的极限**
    `1/(p²(1−p))` 不受影响（两种符号在该极限下相同 ⟹ 该检验**无法**区分）。
    **同时修正** `fisher_truncgeom_monte_carlo`（它与解析式**共用同一 score**，
    故此前是"一致地错"）。
    R4b 目标：与 negbin 同构的输入模式（k 列 + p 列同时填充）+ U 形左腿目标。
    """
    k = int(round(float(k)))
    if k < 1:
        raise ValueError(f"k 须 ≥ 1，收到 {k}")
    if not (0.0 < float(p) < 1.0):
        raise ValueError(f"p 须在 (0,1) 内，收到 {p}")
    p = float(p)
    q = 1.0 - p
    n = np.arange(1, k + 1, dtype=np.float64)
    pk = p * q ** (n - 1.0) / (1.0 - q ** k)
    s = 1.0 / p - (n - 1.0) / q - k * q ** (k - 1.0) / (1.0 - q ** k)
    return float(np.sum(s ** 2 * pk))


def fisher_truncgeom_monte_carlo(k: float, p: float,
                                 n_samples: int = 200_000,
                                 seed: int = 0) -> float:
    """fisher_truncgeom_p 的 MC 对照（inverse-CDF 采样，自测用）。"""
    k = int(round(float(k)))
    p = float(p)
    q = 1.0 - p
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, n_samples) * (1.0 - q ** k)
    n = np.floor(np.log(1.0 - u) / np.log(q)).astype(np.int64) + 1
    n = np.clip(n, 1, k)
    # **2026-10-02 修正**：末项符号（见 `fisher_truncgeom_p` 的说明）——
    #   此前与解析式**共用同一错误 score**，故两者"一致地错"。
    s = 1.0 / p - (n - 1.0) / q - k * q ** (k - 1.0) / (1.0 - q ** k)
    return float(np.mean(s ** 2))


# ---------------------------------------------------------------------------
# R4d（2026-09-29）：逆卡方（chisq 同构）/ 弗雷歇（weibull 同构）
# ---------------------------------------------------------------------------
def fisher_invchisq_k(k: float) -> float:
    """逆卡方 χ⁻²(k) 关于自由度 k 的 Fisher = ψ₁(k/2)/4——与 chisq 逐点相同。

    依据：χ⁻²(k) = 逆伽马(k/2, θ=2)，θ 固定为结构常数不占参数列。
    R4d 目标：chisq 同构（k 单列 + 纯 ψ₁ 目标）。
    """
    from scipy.special import polygamma

    k = float(k)
    if k <= 0:
        raise ValueError(f"k 须 > 0，收到 {k}")
    return float(polygamma(1, k / 2.0)) / 4.0


def _frechet_const() -> float:
    """弗雷歇 Fisher 常数 C = (1−γ)² + π²/6（与 weibull 的 U 形常数相同）。"""
    from math import pi

    gamma = 0.5772156649015329
    return (1.0 - gamma) ** 2 + pi ** 2 / 6.0


def fisher_frechet_trace(shape: float, scale: float) -> float:
    """弗雷歇 Fréchet(α, s) 的 Fisher 迹 = C/α² + α²/s²（U 形）。

    推导：y=(s/x)^α ~ Exp(1)；s_α=[1+(1−y)ln y]/α，s_s=α(1−y)/s；
    E[s_α²]=C/α²（C=(1−γ)²+π²/6，纯解析期望），E[s_s²]=α²/s²。
    R4d 目标：weibull 同构（shape+scale 双列 + 同常数 U 形目标）。
    """
    shape = float(shape)
    scale = float(scale)
    if shape <= 0 or scale <= 0:
        raise ValueError(f"shape={shape}, scale={scale} 均须 > 0")
    return _frechet_const() / shape ** 2 + shape ** 2 / scale ** 2


def fisher_frechet_det(shape: float, scale: float) -> float:
    """弗雷歇 Fréchet(α, s) 的 Fisher 行列式 = π²/(6s²)（与 weibull 同形）。"""
    from math import pi

    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"scale 须 > 0，收到 {scale}")
    return pi ** 2 / (6.0 * scale ** 2)


def fisher_frechet_trace_mc(shape: float, scale: float,
                            n_samples: int = 200_000,
                            seed: int = 0) -> float:
    """fisher_frechet_trace 的 MC 对照（y~Exp(1) 变换采样，自测用）。"""
    shape = float(shape)
    scale = float(scale)
    rng = np.random.default_rng(seed)
    y = rng.exponential(1.0, n_samples)
    s_a = (1.0 + (1.0 - y) * np.log(y)) / shape
    s_s = shape * (1.0 - y) / scale
    return float(np.mean(s_a ** 2) + np.mean(s_s ** 2))


# ---------------------------------------------------------------------------
# R4d 第三刀（2026-09-29）：莱斯 K 因子（vonmises 同构）
# ---------------------------------------------------------------------------
def fisher_rician_kappa(kappa: float, sigma: float = 1.0,
                        n_grid: int = 4000) -> float:
    """莱斯衰落关于 K 因子 κ 的 Fisher（数值积分）。

    p(x) = x e^(−(x²+ν²)/2) I₀(xν)（σ=1，ν=√(2κ)）；
    score s_κ = ∂ln p/∂κ = −1 + (x/ν)·I₁(xν)/I₀(xν)。

    形状（R4d 第三刀：与 vonmises 同向同尾）：
      κ→0（Rayleigh 极限）→ 1；单调下降；κ→∞ 尾 → 1/(2κ)
      （对照 vonmises 的 I_κκ：½→0、尾 1/(4κ)——同向衰减、同 1/κ 尾）。

    数值方案：等距网格 [0, ν+12σ]，ν=σ√(2κ)；与 MC 对照在自测中验证。
    """
    from scipy.special import i0, i1

    kappa = float(kappa)
    sigma = float(sigma)
    if kappa < 0 or sigma <= 0:
        raise ValueError(f"kappa={kappa} 须 ≥ 0，sigma={sigma} 须 > 0")
    nu = sigma * np.sqrt(2.0 * kappa)
    upper = nu + 12.0 * sigma
    x = np.linspace(1e-8, upper, n_grid)
    dx = x[1] - x[0]
    z = x * nu / sigma ** 2
    p = x * np.exp(-(x ** 2 + nu ** 2) / (2.0 * sigma ** 2)) * i0(z) / sigma ** 2
    if nu > 0:
        s = -1.0 + (x / nu) * i1(z) / i0(z)
    else:  # Rayleigh 极限：s = −1 + x²/(2σ²)
        s = -1.0 + x ** 2 / (2.0 * sigma ** 2)
    return float(np.sum(s ** 2 * p) * dx)


def fisher_rician_kappa_mc(kappa: float, sigma: float = 1.0,
                           n_samples: int = 200_000,
                           seed: int = 0) -> float:
    """fisher_rician_kappa 的 MC 对照（自测用）。

    莱斯采样：x = σ·√((Z₁+ν/σ)² + Z₂²)，Z₁,₂ ~ N(0,1)。
    """
    kappa = float(kappa)
    sigma = float(sigma)
    nu = sigma * np.sqrt(2.0 * kappa)
    rng = np.random.default_rng(seed)
    z1 = rng.normal(0.0, 1.0, n_samples) + nu / sigma
    z2 = rng.normal(0.0, 1.0, n_samples)
    x = sigma * np.sqrt(z1 ** 2 + z2 ** 2)
    from scipy.special import i0, i1

    z = x * nu / sigma ** 2
    if nu > 0:
        s = -1.0 + (x / nu) * i1(z) / i0(z)
    else:
        s = -1.0 + x ** 2 / (2.0 * sigma ** 2)
    return float(np.mean(s ** 2))


# ---------------------------------------------------------------------------
# R4d 第三刀 v2（2026-09-29）：卷绕高斯（vonmises 三层同构）
# ---------------------------------------------------------------------------
def _wrapnorm_density(theta: np.ndarray, kappa: float, k_max: int = 20):
    """卷绕高斯密度与 ∂p/∂κ（级数截断）。

    p(θ) = (1/2π)(1 + 2Σ_{k≥1} ρ^{k²} cos kθ)，ρ = e^{−κ²/2}；
    ∂p/∂κ = (1/π) Σ_{k≥1} (−κ k² ρ^{k²}) cos kθ。
    """
    rho = np.exp(-kappa ** 2 / 2.0)
    ks = np.arange(1, k_max + 1, dtype=np.float64)
    cos_terms = np.cos(np.outer(theta, ks))
    weights = rho ** (ks ** 2)
    p = (1.0 + 2.0 * np.sum(weights * cos_terms, axis=1)) / (2.0 * np.pi)
    dp = (-kappa / np.pi) * np.sum((ks ** 2) * weights * cos_terms, axis=1)
    return p, dp


def fisher_wrapnorm_kappa(kappa: float, n_grid: int = 4000,
                          k_max: int = 20) -> float:
    """卷绕高斯关于 kappa（底层 σ）的 Fisher（数值积分）。

    p(θ) = (1/2π)(1 + 2Σ ρ^{k²} cos kθ)，ρ=e^{−κ²/2}；
    I(κ) = ∫ (∂ln p/∂κ)² p dθ。

    形状（R4d 第三刀 v2）：κ→0（点质量极限）I 发散；κ→∞（均匀极限）
    I→0（尾 ~2κ²e^{−κ²}）——与 vonmises 三层同构：kappa 列 + 圆支撑 +
    衰减到 0 方向（第 51 条修正后选型）。
    """
    kappa = float(kappa)
    if kappa <= 0:
        raise ValueError(f"kappa 须 > 0，收到 {kappa}")
    if kappa < 0.5:
        # κ 小时峰值极窄：κ 自适应网格 t=θ/κ ∈ [−π/κ, π/κ]（峰值分辨率 ∝ κ）
        t = np.linspace(-np.pi / kappa, np.pi / kappa, n_grid)
        theta = kappa * t
    else:
        theta = np.linspace(0.0, 2.0 * np.pi, n_grid)
    p, dp = _wrapnorm_density(theta, kappa, k_max)
    s = dp / np.clip(p, 1e-12, None)
    s = np.clip(s, -1e6, 1e6)   # 数值保险（远离峰值的尾部）
    dtheta = theta[1] - theta[0]
    return float(np.sum(s ** 2 * p) * dtheta)


def fisher_wrapnorm_kappa_mc(kappa: float, n_samples: int = 200_000,
                             seed: int = 0, k_max: int = 20) -> float:
    """fisher_wrapnorm_kappa 的 MC 对照（自测用）。

    卷绕高斯采样：θ = (N(0, κ) mod 2π)。
    """
    kappa = float(kappa)
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, kappa, n_samples) % (2.0 * np.pi)
    p, dp = _wrapnorm_density(theta, kappa, k_max)
    s = dp / np.clip(p, 1e-300, None)
    return float(np.mean(s ** 2))
