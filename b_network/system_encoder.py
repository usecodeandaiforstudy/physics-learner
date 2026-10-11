"""B 网络 SystemEncoder：物理系统 → 信息结构图 EncodedGraph。

职责（架构文档 §2.3.9(a) 的 ChannelTemplateLibrary 线）：
把 DB1 物理系统条目（equations/parameters/family）编码为信息结构图 G=(V,E,Θ)，
其中关键一步是【物理参数 → 分布参数】的映射（理论推导产物，如平稳分布）。

Phase 0 边界：当前实现"平稳态表示层"的最小编码——把物理系统的平稳分布
（或平衡态边缘分布）编码为单节点，作为指纹（Fisher 摘要）计算对象。
完整的 X→T→Y 多级结构待后续按 §2.3.4 的通道模板扩展。

已实现的编码器：
- CT-DIFF-001 / family=ou        ：OU 过程 → 平稳分布 N(mu, sigma^2/(2 theta))
- CT-EQ-001   / family=gibbs     ：热平衡谐振子 → x 边缘分布 N(0, kT/(m omega^2))

物理 → 分布参数的推导均来自标准文献（见各函数 docstring），可机器对照。
"""

from __future__ import annotations

import math

import numpy as np

from core.channels import ChannelTemplate, Edge, EncodedGraph, GaussianChannel, Node

__all__ = ["SystemEncoder"]


class SystemEncoder:
    """把 DB1 条目编码为 EncodedGraph。"""

    # **`ib_source` 集中注册表（2026-10-02，第 118 条）**
    #   键 = family；值 = 参数化信道规格：
    #     param          源参数名（该族物理参数，作为 IB 信道 θ）
    #     range_default  (lo, hi) 兜底范围（条目未声明参数范围时用）
    #     channel        "scipy" = 通用 scipy.stats 信道（见 fingerprint_computer）
    #     dist           scipy.stats 分布名；arg = θ 驱动的参数名；fixed = 其余固定参数
    #   **只登记能用单一 scipy.stats 分布表达者**；表达不了的族（混合分布、
    #   包裹正态）**留空** ⟹ 如实返回空曲线，不臆造信道。
    IB_SOURCE_REGISTRY: dict[str, dict] = {
        # **第 167 条的 5 族注册已回滚（第 176 条，2026-10-02）**：
        #   回滚原因：注册后（曲线改走**参数化信道**）实测
        #   **`RL·环境集成冒烟（M4）` 由 PASS 变 FAIL**——
        #   代理 vs 慢路径一致性 **Spearman 0.907 → 0.714**、max|Δ| 0.1772 → 0.1883
        #   ⟹ **监控门回归**（按架构 §10.2 第 7 条不得当作结果）。
        #   **处置**：**回滚注册**（恢复绿色），**保留** `fingerprint_computer`
        #   的 `power_gauss`/`special` 两条分支与本节注释——
        #   待单独排查"为何参数化信道会改变代理一致性"后再启用。
        #   注：DB2 的曲线**不受影响**（已按第 175 条逐族统一落盘；
        #   回滚注册不改变已写数据）。
        "rayleigh": {"param": "sigma", "range_default": (0.2, 5.0),
                     "channel": "scipy", "dist": "rayleigh", "arg": "scale",
                     "fixed": {}},
        "lognormal": {"param": "sigma", "range_default": (0.1, 2.0),
                      "channel": "scipy", "dist": "lognorm", "arg": "scale",
                      "fixed": {"s": 0.5}},
        "frechet": {"param": "sigma", "range_default": (0.2, 5.0),
                    "channel": "scipy", "dist": "invweibull", "arg": "scale",
                    "fixed": {"c": 2.0}},
        "cauchy": {"param": "gamma", "range_default": (0.2, 5.0),
                   "channel": "scipy", "dist": "cauchy", "arg": "scale",
                   "fixed": {}},
        "logistic": {"param": "sigma", "range_default": (0.2, 5.0),
                     "channel": "scipy", "dist": "logistic", "arg": "scale",
                     "fixed": {}},
        "laplace": {"param": "b", "range_default": (0.2, 5.0),
                    "channel": "scipy", "dist": "laplace", "arg": "scale",
                    "fixed": {}},
        "chisq": {"param": "k", "range_default": (1.0, 20.0),
                  "channel": "scipy", "dist": "chi2", "arg": "df",
                  "fixed": {}},
        "geometric": {"param": "p", "range_default": (0.05, 0.95),
                      "channel": "scipy", "dist": "geom", "arg": "p",
                      "fixed": {}},
        "negbin": {"param": "p", "range_default": (0.05, 0.95),
                   "channel": "scipy", "dist": "nbinom", "arg": "p",
                   "fixed": {"n": 5}},
        "inverse_gamma": {"param": "a", "range_default": (1.5, 10.0),
                          "channel": "scipy", "dist": "invgamma", "arg": "a",
                          "fixed": {}},
        "invchisq": {"param": "k", "range_default": (2.5, 20.0),
                     "channel": "scipy", "dist": "invgamma", "arg": "a",
                     "fixed": {"scale": 1.0}},
        "truncated_gaussian": {"param": "sigma", "range_default": (0.2, 3.0),
                               "channel": "scipy", "dist": "truncnorm",
                               "arg": "scale", "fixed": {"a": -2.0, "b": 2.0}},
        "rician": {"param": "sigma", "range_default": (0.2, 3.0),
                   "channel": "scipy", "dist": "rice", "arg": "scale",
                   "fixed": {"b": 0.5}},
        "elastic": {"param": "k", "range_default": (0.1, 10.0),
                    "channel": "scipy", "dist": "norm", "arg": "scale",
                    "fixed": {}},
        "harmonic_oscillator": {"param": "omega", "range_default": (0.2, 5.0),
                                "channel": "scipy", "dist": "norm",
                                "arg": "scale", "fixed": {}},
        # 明确不登记（无法用单一 scipy 分布表达 → 保持空曲线并如实报告）：
        #   gaussian_mixture2（混合）、wrapped_normal（周期包裹）、lc_circuit
    }

    def encode(self, entry: dict) -> EncodedGraph:
        """按 channel_template / family 分派编码器。

        参数
        ----
        entry : dict
            DB1 物理系统条目（对齐 db1.schema.json）。

        返回
        ----
        EncodedGraph
            信息结构图，节点已带分布参数（可被 core.fingerprint 消费）。
        """
        ct = entry.get("channel_template")
        family = str(entry.get("family", ""))

        graph = self._encode_dispatch(ct, family, entry)
        # **集中注册 `ib_source`（2026-10-02，第 118 条）**：DB2 中 59% 的行
        # 没有 IB 曲线，根因是多数族**未注册 `ib_source`**（`compute_ib_curve`
        # 的两个显式空分支）。逐个改 12 个 `_encode_*` 方法易漏、难审——
        # 改为**一处注册表**：编码完成后按族补 `ib_source`（已有者不覆盖）。
        if "ib_source" not in (graph.metadata or {}):
            spec = self.IB_SOURCE_REGISTRY.get(family)   # **须 self.（初版漏写 ⟹ NameError）**
            if spec:
                src = dict(spec)
                param = src.pop("param")
                lo_d, hi_d = src.pop("range_default")
                src["range"] = self._param_range(entry, param, lo_d, hi_d)
                graph.metadata = dict(graph.metadata or {})
                graph.metadata["ib_source"] = src
        return graph

    def _encode_dispatch(self, ct, family: str, entry: dict) -> EncodedGraph:
        """按 channel_template / family 分发到具体编码器。"""
        if ct == ChannelTemplate.DIFFUSION or family == "ou":
            return self._encode_ou(entry)
        if ct == ChannelTemplate.DIFFUSION_PDE or family in ("diffusion", "heat_diffusion"):
            return self._encode_diffusion_pde(entry)
        if ct == ChannelTemplate.EQUILIBRIUM or family in ("gibbs", "harmonic_oscillator"):
            return self._encode_equilibrium(entry)
        if ct == ChannelTemplate.MECHANICAL or family in ("elastic", "linear_elastic"):
            return self._encode_elastic(entry)
        if ct == ChannelTemplate.DECAY or family in ("radioactive_decay", "exponential_decay"):
            return self._encode_radioactive_decay(entry)
        if ct == ChannelTemplate.COUNTING or family in ("poisson_count", "photon_count"):
            return self._encode_poisson_count(entry)
        if ct == ChannelTemplate.TWO_LEVEL or family in ("two_level", "spin_1_2"):
            return self._encode_two_level(entry)
        if ct == ChannelTemplate.LAPLACE or family in ("laplace", "double_exponential"):
            return self._encode_laplace(entry)
        if ct == ChannelTemplate.RESONATOR or family in ("lc_circuit", "resonator"):
            return self._encode_resonator(entry)
        if ct == ChannelTemplate.LANDAU or family in ("landau", "phase_transition"):
            return self._encode_landau(entry)
        if ct == ChannelTemplate.GAMMA or family in ("gamma", "gamma_wait"):
            return self._encode_gamma(entry)
        if ct == ChannelTemplate.BETA or family in ("beta", "bounded_fraction"):
            return self._encode_beta(entry)
        if ct == ChannelTemplate.WEIBULL or family in ("weibull", "failure_life"):
            return self._encode_weibull(entry)
        if ct == ChannelTemplate.VONMISES or family in ("vonmises", "circular_phase"):
            return self._encode_vonmises(entry)
        if ct == ChannelTemplate.GEOMETRIC or family in ("geometric", "first_success"):
            return self._encode_geometric(entry)
        if ct == ChannelTemplate.CAUCHY or family in ("cauchy", "lorentzian"):
            return self._encode_cauchy(entry)
        if ct == ChannelTemplate.RAYLEIGH or family in ("rayleigh", "envelope"):
            return self._encode_rayleigh(entry)
        if ct == ChannelTemplate.LOGISTIC or family in ("logistic", "sigmoid_growth"):
            return self._encode_logistic(entry)
        if ct == ChannelTemplate.CHISQ or family in ("chisq", "chi_square"):
            return self._encode_chisq(entry)
        if ct == ChannelTemplate.NEGBIN or family in ("negbin", "negative_binomial"):
            return self._encode_negbin(entry)
        if ct == ChannelTemplate.LOGNORM or family in ("lognormal", "log_gaussian"):
            return self._encode_lognormal(entry)
        if ct == ChannelTemplate.MIX2 or family in ("gaussian_mixture2", "mix2"):
            return self._encode_mix2(entry)
        if ct == ChannelTemplate.TRUNCNORM or family in ("truncated_gaussian", "truncnorm"):
            return self._encode_truncnorm(entry)
        if ct == ChannelTemplate.TRUNCGEOM or family in ("truncated_geometric", "truncgeom"):
            return self._encode_truncgeom(entry)
        if ct == ChannelTemplate.INVGAMMA or family in ("inverse_gamma", "invgamma"):
            return self._encode_invgamma(entry)
        if ct == ChannelTemplate.INVCHISQ or family in ("invchisq", "inverse_chi_square"):
            return self._encode_invchisq(entry)
        if ct == ChannelTemplate.FRECHET or family in ("frechet",):
            return self._encode_frechet(entry)
        if ct == ChannelTemplate.RICIAN or family in ("rician",):
            return self._encode_rician(entry)
        if ct == ChannelTemplate.WRAPNORM or family in ("wrapped_normal", "wrapnorm"):
            return self._encode_wrapnorm(entry)

        raise NotImplementedError(
            f"未实现 family={family!r} / channel_template={ct!r} 的编码器；"
            "当前支持 ou、diffusion、gibbs/harmonic_oscillator、elastic、"
            "radioactive_decay、poisson_count、two_level、laplace、lc_circuit、landau、"
            "gamma、beta、weibull、vonmises、geometric、cauchy、rayleigh、logistic、"
            "chisq、negbin、lognormal"
        )

    # ------------------------------------------------------------------
    # 参数提取
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_params(entry: dict) -> dict[str, float]:
        """把 parameters 列表转为 {name: 数值}，default 缺失时用 range 中点。"""
        params: dict[str, float] = {}
        for p in entry.get("parameters", []):
            name = p.get("name")
            if name is None:
                continue
            default = p.get("default")
            if default is None:
                rng = p.get("range")
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    default = (float(rng[0]) + float(rng[1])) / 2.0
                else:
                    raise ValueError(f"参数 {name} 缺 default 且无合法 range")
            params[str(name)] = float(default)
        return params

    @staticmethod
    def _param_range(entry: dict, name: str,
                     fallback_lo: float, fallback_hi: float) -> list[float]:
        """提取参数的 range（[lo, hi]），缺失时用 fallback。"""
        for p in entry.get("parameters", []):
            if p.get("name") == name:
                rng = p.get("range")
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    return [float(rng[0]), float(rng[1])]
        return [fallback_lo, fallback_hi]

    # ------------------------------------------------------------------
    # OU 过程（扩散型，CT-DIFF-001）
    # ------------------------------------------------------------------
    def _encode_ou(self, entry: dict) -> EncodedGraph:
        """OU 过程 dX = -theta (X - mu) dt + sigma dW → X→T→Y 信道结构。

        平稳分布（Gardiner 2004, Handbook of Stochastic Methods, Ch.3）：
            N(mu, sigma_stat^2)，sigma_stat^2 = sigma^2 / (2 theta)

        信道结构（Phase 0 扩展）：
            X（源，平稳分布）--OU 转移核--> T（中间，仍平稳）--观测噪声--> Y（观测）
            X→T 转移核：T | X=x ~ N(mu + (x-mu) rho, sigma_stat^2 (1-rho^2))
                          rho = exp(-theta * dt)
            T→Y 观测：  Y = T + eps，eps ~ N(0, sigma_obs^2)
        """
        p = self._extract_params(entry)
        theta = p["theta"]
        sigma = p["sigma"]
        mu = p.get("mu", 0.0)
        dt = p.get("dt", 1.0)
        sigma_obs = p.get("sigma_obs", 0.5)

        if theta <= 0:
            raise ValueError(f"theta 须 > 0（保证平稳性），收到 {theta}")
        if sigma_obs < 0:
            raise ValueError(f"sigma_obs 须 >= 0，收到 {sigma_obs}")

        sigma_stat = sigma / math.sqrt(2.0 * theta)
        rho = math.exp(-theta * dt)  # X→T 相关系数

        # 节点
        node_x = Node("ou__X__0", "DIST-GAUSS", {"mu": mu, "sigma": sigma_stat},
                      noise_type="additive_gaussian", dimension=1)
        node_t = Node("ou__T__0", "DIST-GAUSS", {"mu": mu, "sigma": sigma_stat},
                      noise_type="additive_gaussian", dimension=1)
        sigma_y = math.sqrt(sigma_stat**2 + sigma_obs**2)
        node_y = Node("ou__Y__0", "DIST-GAUSS", {"mu": mu, "sigma": sigma_y},
                      noise_type="additive_gaussian", dimension=1)

        # 边 X→T：OU 转移核（联合协方差）
        var = sigma_stat**2
        cov_xt = np.array([[var, rho * var], [rho * var, var]])
        edge_xt = Edge("ou__X__0", "ou__T__0",
                       channel=GaussianChannel(cov=cov_xt, block_shape=(1, 1)))

        # 边 T→Y：观测噪声（联合协方差）
        cov_ty = np.array([[var, var], [var, var + sigma_obs**2]])
        edge_ty = Edge("ou__T__0", "ou__Y__0",
                       channel=GaussianChannel(cov=cov_ty, block_shape=(1, 1)))

        graph = EncodedGraph(
            nodes=[node_x, node_t, node_y],
            edges=[edge_xt, edge_ty],
            beta=entry.get("beta", 1.0),
        )
        graph.metadata = {
            "physical_params": p,
            "derivation": "stationary_dist: N(mu, sigma^2/(2*theta)); "
                          "X->T: rho=exp(-theta*dt); T->Y: obs noise sigma_obs",
            "citation": "Gardiner 2004, Ch.3",
        }
        return graph

    # ------------------------------------------------------------------
    # 纯扩散方程（PDE，CT-DIFF-002）
    # ------------------------------------------------------------------
    def _encode_diffusion_pde(self, entry: dict) -> EncodedGraph:
        """纯扩散方程 ∂u/∂t = D ∂²u/∂x² → 加性高斯扩散信道。

        一维扩散的格林函数是高斯核，方差 σ_diff² = 2 D t（均方位移）。
        作为信息信道：初始信号 X ~ N(mu, sigma_signal²)，扩散时间 t 后
        T = X + n，n ~ N(0, 2 D t)。

        信道结构（Phase 0）：
            X（初始信号）--扩散核（高斯，方差 2Dt）--> T（扩散后信号）

        注意：纯扩散无平稳分布（方差随时间增长），故 sigma_signal 是输入参数
        （初始信号的标准差），由条目显式给出。
        """
        p = self._extract_params(entry)
        D = p.get("D", p.get("diffusivity", 1.0))
        t = p.get("t", p.get("time", 1.0))
        sigma_signal = p.get("sigma_signal", 1.0)
        mu = p.get("mu", 0.0)

        if D < 0 or t < 0:
            raise ValueError(f"D={D}, t={t} 须 >= 0")
        if sigma_signal <= 0:
            raise ValueError(f"sigma_signal 须 > 0，收到 {sigma_signal}")

        var_diff = 2.0 * D * t  # 扩散方差（一维均方位移）

        node_x = Node("diff__X__0", "DIST-GAUSS", {"mu": mu, "sigma": sigma_signal},
                      noise_type="additive_gaussian", dimension=1)
        sigma_t = math.sqrt(sigma_signal**2 + var_diff)
        node_t = Node("diff__T__0", "DIST-GAUSS", {"mu": mu, "sigma": sigma_t},
                      noise_type="additive_gaussian", dimension=1)

        # 边 X→T：加性高斯扩散信道
        var_s = sigma_signal**2
        cov = np.array([[var_s, var_s], [var_s, var_s + var_diff]])
        edge = Edge("diff__X__0", "diff__T__0",
                    channel=GaussianChannel(cov=cov, block_shape=(1, 1)))

        graph = EncodedGraph(nodes=[node_x, node_t], edges=[edge],
                             beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "diffusion kernel: N(0, 2*D*t); T = X + n, n ~ N(0, 2Dt)",
            "citation": "Crank, The Mathematics of Diffusion, Ch.2",
        }
        return graph

    # ------------------------------------------------------------------
    # 热平衡谐振子（平衡态，CT-EQ-001）
    # ------------------------------------------------------------------
    def _encode_equilibrium(self, entry: dict) -> EncodedGraph:
        """热平衡谐振子 H = p^2/(2m) + (1/2) m omega^2 x^2，Gibbs 态。

        x 边缘分布（Pathria & Beale, Statistical Mechanics, Ch.3）：
            N(0, sigma_x^2)，sigma_x^2 = kT / (m omega^2)
        """
        p = self._extract_params(entry)
        m = p.get("m", 1.0)
        omega = p.get("omega", 1.0)
        kT = p.get("kT", p.get("k_B_T", 1.0))

        if m <= 0 or omega <= 0 or kT <= 0:
            raise ValueError(f"m={m}, omega={omega}, kT={kT} 均须 > 0")

        sigma_x = math.sqrt(kT / (m * omega**2))

        node = Node(
            node_id=f"ho__T__0",
            distribution_family="DIST-GAUSS",
            params={"mu": 0.0, "sigma": sigma_x},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "x_marginal: N(0, kT/(m*omega^2))",
            "citation": "Pathria & Beale, Statistical Mechanics, Ch.3",
        }
        return graph

    # ------------------------------------------------------------------
    # 线性弹性材料（涨落-响应，CT-MECH-001）
    # ------------------------------------------------------------------
    def _encode_elastic(self, entry: dict) -> EncodedGraph:
        """线性弹性材料 → 热平衡应变涨落（高斯）。

        势能 U(x) = (1/2) k x^2（k 为刚度/弹性模量），热平衡 Gibbs 态下
        应变 x 的边缘分布（涨落-响应定理，Landau & Lifshitz, Statistical
        Physics, Ch.12）：
            N(0, sigma_x^2)，sigma_x^2 = kT / k

        注意：分布族仍是高斯（与谐振子同族），Fisher trace = 3k/kT。
        其价值在于物理对应——弹性模量 k 越大（越硬），应变涨落越小，
        Fisher 信息越大（与理论文档涨落-响应对应一致），而非引入新分布族。
        """
        p = self._extract_params(entry)
        k = p.get("k", p.get("stiffness", 1.0))
        kT = p.get("kT", p.get("k_B_T", 1.0))

        if k <= 0 or kT <= 0:
            raise ValueError(f"k={k}, kT={kT} 均须 > 0")

        sigma_x = math.sqrt(kT / k)

        node = Node(
            node_id="elastic__T__0",
            distribution_family="DIST-GAUSS",
            params={"mu": 0.0, "sigma": sigma_x},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "strain_fluctuation: N(0, kT/k) (fluctuation-response)",
            "citation": "Landau & Lifshitz, Statistical Physics, Ch.12",
        }
        return graph

    # ------------------------------------------------------------------
    # 放射性衰变（指数寿命分布，CT-DECAY-001）
    # ------------------------------------------------------------------
    def _encode_radioactive_decay(self, entry: dict) -> EncodedGraph:
        """放射性衰变 → 寿命分布 Exp(λ)。

        寿命 t 服从指数分布 p(t) = λ e^{-λt}，λ 为衰变率（λ = ln2 / 半衰期）。
        Fisher 信息关于 λ：I(λ) = 1/λ²（指数分布的标准结果）。

        单节点编码（非高斯，无 X→T→Y 信道结构）。
        """
        p = self._extract_params(entry)
        rate = p.get("rate", p.get("lambda", p.get("decay_rate", 1.0)))

        if rate <= 0:
            raise ValueError(f"rate 须 > 0，收到 {rate}")

        node = Node(
            node_id="decay__T__0",
            distribution_family="DIST-EXPONENTIAL",
            params={"rate": rate},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "lifetime_dist: Exp(rate), Fisher=1/rate^2",
            "citation": "标准核物理（放射性衰变律）",
            "ib_source": {
                "param": "rate",
                "range": self._param_range(entry, "rate", 0.1, 10.0),
                "channel": "exponential",
            },
        }
        return graph

    # ------------------------------------------------------------------
    # 泊松计数（光子/粒子探测，CT-COUNT-001）
    # ------------------------------------------------------------------
    def _encode_poisson_count(self, entry: dict) -> EncodedGraph:
        """泊松计数 → 计数分布 Poisson(λ)。

        单位时间内的探测计数 N ~ Poisson(λ)，λ 为平均计数率。
        Fisher 信息关于 λ：I(λ) = 1/λ（泊松分布的标准结果）。

        单节点编码（离散分布，无 X→T→Y 信道结构）。
        """
        p = self._extract_params(entry)
        rate = p.get("rate", p.get("lambda", p.get("mean_count", 1.0)))

        if rate <= 0:
            raise ValueError(f"rate 须 > 0，收到 {rate}")

        node = Node(
            node_id="count__T__0",
            distribution_family="DIST-POISSON",
            params={"rate": rate},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "count_dist: Poisson(rate), Fisher=1/rate",
            "citation": "标准统计（泊松计数过程）",
            "ib_source": {
                "param": "rate",
                "range": self._param_range(entry, "rate", 0.1, 20.0),
                "channel": "poisson",
            },
        }
        return graph

    # ------------------------------------------------------------------
    # 二能级系统（自旋/能级占据，CT-2LVL-001）
    # ------------------------------------------------------------------
    def _encode_two_level(self, entry: dict) -> EncodedGraph:
        """二能级系统 → 占据概率伯努利分布 Bern(p)。

        高能级占据概率 p（由玻尔兹曼因子 p = 1/(1+e^{ΔE/kT}) 给出，或直接给 p），
        单次观测为伯努利 Bern(p)。Fisher 信息关于 p：I(p) = 1/(p(1-p))。

        单节点编码（离散分布，无 X→T→Y 信道结构）。
        """
        p = self._extract_params(entry)
        # 优先用显式 p；否则由能隙 ΔE 与 kT 经玻尔兹曼占据推导
        if "p" in p:
            prob = p["p"]
        else:
            delta_E = p.get("delta_E", p.get("energy_gap", 1.0))
            kT = p.get("kT", p.get("k_B_T", 1.0))
            if delta_E <= 0 or kT <= 0:
                raise ValueError(f"delta_E={delta_E}, kT={kT} 须 > 0")
            prob = 1.0 / (1.0 + math.exp(delta_E / kT))

        if not (0.0 < prob < 1.0):
            raise ValueError(f"占据概率 p 须在 (0,1) 内，收到 {prob}")

        node = Node(
            node_id="2lvl__T__0",
            distribution_family="DIST-BERNOULLI",
            params={"p": prob},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "occupation_prob: Bern(p), Fisher=1/(p(1-p))",
            "citation": "标准统计力学（二能级占据）",
            "ib_source": {
                "param": "p",
                "range": self._param_range(entry, "p", 0.01, 0.99),
                "channel": "bernoulli",
            },
        }
        return graph

    # ------------------------------------------------------------------
    # 拉普拉斯（双指数差，CT-LAPLACE-001）
    # ------------------------------------------------------------------
    def _encode_laplace(self, entry: dict) -> EncodedGraph:
        """两个独立同速率指数过程的到达时间差 → Laplace(0, b=1/λ)。

        设 X₁, X₂ ~ Exp(λ) 独立，则 Z = X₁ - X₂ 服从拉普拉斯分布
        Laplace(0, 1/λ)（尺度参数 b = 1/λ）。Fisher 关于 (μ, b)：
        trace = 2/b²，det = 1/b⁴。

        单节点编码（μ 固定为 0，重尾分布，无 X→T→Y 高斯信道）。
        """
        p = self._extract_params(entry)
        rate = p.get("rate", p.get("lambda", 1.0))

        if rate <= 0:
            raise ValueError(f"rate 须 > 0，收到 {rate}")

        b = 1.0 / rate  # 拉普拉斯尺度参数

        node = Node(
            node_id="laplace__T__0",
            distribution_family="DIST-LAPLACE",
            params={"mu": 0.0, "b": b},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "arrival_diff: Laplace(0, 1/rate), Fisher trace=2/b^2",
            "citation": "标准概率论（双指数差 = 拉普拉斯分布）",
        }
        return graph

    # ------------------------------------------------------------------
    # 谐振腔 LC 电路（热涨落，CT-LC-001）
    # ------------------------------------------------------------------
    def _encode_resonator(self, entry: dict) -> EncodedGraph:
        """LC 谐振电路 → 电容电压热涨落（高斯）。

        热平衡下（Johnson 噪声），电容电压 V 的能量均分：
            ⟨½ C V²⟩ = ½ kT  ⟹  V ~ N(0, kT/C)
        电容 C 越大，电压涨落越小，Fisher 越大（trace = 3C/kT）。

        分布族仍为高斯（与谐振子/弹性同族），价值在于电学对应。
        """
        p = self._extract_params(entry)
        C = p.get("C", p.get("capacitance", 1.0))
        kT = p.get("kT", p.get("k_B_T", 1.0))

        if C <= 0 or kT <= 0:
            raise ValueError(f"C={C}, kT={kT} 均须 > 0")

        sigma_v = math.sqrt(kT / C)

        node = Node(
            node_id="lc__T__0",
            distribution_family="DIST-GAUSS",
            params={"mu": 0.0, "sigma": sigma_v},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "voltage_fluctuation: N(0, kT/C) (Johnson noise)",
            "citation": "标准电路热噪声（能量均分）",
        }
        return graph

    # ------------------------------------------------------------------
    # 朗道自由能（相变，CT-LANDAU-001）
    # ------------------------------------------------------------------
    def _encode_landau(self, entry: dict) -> EncodedGraph:
        """朗道自由能 F(φ) = ½aφ² + ¼bφ⁴ → 序参量分布（相变）。

        序参量平衡分布 p(φ) ∝ exp(-F(φ)/kT)。关于约化温度系数 a 的 Fisher
        需数值积分（fisher_landau）：I(a) = Var(φ²)/(4k²T²)。

        a>0 高温相（单峰），a<0 低温相（双峰，对称破缺）——Fisher 在相变点
        附近的行为是相变的信息论特征。

        单节点编码（非标准分布族，无 X→T→Y 高斯信道）。
        """
        p = self._extract_params(entry)
        a = p.get("a", p.get("reduced_temp", 1.0))
        b = p.get("b", p.get("phi4_coeff", 1.0))
        kT = p.get("kT", p.get("k_B_T", 1.0))

        if b <= 0 or kT <= 0:
            raise ValueError(f"b={b}, kT={kT} 均须 > 0")

        node = Node(
            node_id="landau__T__0",
            distribution_family="DIST-LANDAU",
            params={"a": a, "b": b, "kT": kT},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "order_parameter: p(phi) ~ exp(-F/kT), Fisher=Var(phi^2)/(4k^2T^2)",
            "citation": "Landau 相变理论（序参量涨落）",
            # **ib_source 注册（2026-10-02，第 111 条）**：landau 族此前**无
            # ib_source** ⟹ `compute_ib_curve` 走「src is None 且无
            # ib_source」分支返回**空曲线**（DB2 中 landau 270 行全部无曲线）。
            # 补注册后走参数化信道路径：源参数 = a（均匀先验），观测 = φ。
            "ib_source": {
                "param": "a",
                "range": self._param_range(entry, "a", -2.0, 2.0),
                "channel": "landau_doublewell",
                "b_fixed": b,
                "kT_fixed": kT,
            },
        }
        return graph

    # ------------------------------------------------------------------
    # 伽马分布族（CT-GAMMA-001）
    # ------------------------------------------------------------------
    def _encode_gamma(self, entry: dict) -> EncodedGraph:
        """伽马族物理 → 等待时间/能量分布 Gamma(k, θ)。

        物理场景（三者同族不同源）：
        - Erlang 排队：泊松流第 k 个事件到达的等待时间 ~ Gamma(k, θ)
          （θ 为单事件平均间隔），k 为整数形状（Erlang 是伽马特例）。
        - chi2 噪声能量：ν 个独立标准高斯的平方和 ~ chi2(ν) = Gamma(ν/2, 2)
          （桥接：k = ν/2, θ = 2）。
        - 水文降雨：站点时段雨量经伽马分布拟合（形状-尺度参数化）。

        Fisher 关于 (k, θ)：trace = ψ₁(k) + k/θ²，det = (k·ψ₁(k) − 1)/θ²
        （非对角元 g_kθ = 1/θ，本族是库中首个参数耦合的非对角 Fisher）。

        单节点编码（连续非高斯正支撑，无 X→T→Y 高斯信道）。
        """
        p = self._extract_params(entry)
        k = p.get("k", p.get("shape", 1.0))
        theta = p.get("theta", p.get("scale", 1.0))

        if k <= 0 or theta <= 0:
            raise ValueError(f"k={k}, theta={theta} 均须 > 0")

        node = Node(
            node_id="gamma__T__0",
            distribution_family="DIST-GAMMA",
            params={"k": k, "theta": theta},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "wait_time/energy: Gamma(k, theta), "
                          "Fisher trace=psi1(k)+k/theta^2, g_k_theta=1/theta",
            "citation": "标准概率论（Erlang/chi2 与伽马分布）",
            "ib_source": {
                "param": "k",
                "range": self._param_range(entry, "k", 0.5, 10.0),
                "channel": "gamma",
                "theta_fixed": theta,
            },
        }
        return graph

    # ------------------------------------------------------------------
    # 贝塔分布族（CT-BETA-001）
    # ------------------------------------------------------------------
    def _encode_beta(self, entry: dict) -> EncodedGraph:
        """有界支撑物理量 → 占据分数分布 Beta(α, β)。

        物理场景：
        - 晶格占据：A/B 二元固溶体中 B 原子占据分数 x ∈ [0,1]，
          平衡涨落可由 Beta(α, β) 描述（α+β 越大方差越小）。
        - 量子态纯度：混态的谱权重/布洛赫 z 分量涨落（有界观测量）。
        - 两相混合：两相区域中某相的体积分数（杠杆规则附近的涨落）。

        Fisher 关于 (α, β)：trace = ψ₁(α)+ψ₁(β)−2ψ₁(α+β)，
        det = ψ₁(α)ψ₁(β) − ψ₁(α+β)[ψ₁(α)+ψ₁(β)]（非对角元 g_αβ = −ψ₁(α+β)）。
        本族是库中首个有界支撑 [0,1] 的连续分布族。

        单节点编码（有界连续，无 X→T→Y 高斯信道）。
        """
        p = self._extract_params(entry)
        alpha = p.get("alpha", 1.0)
        beta = p.get("beta", 1.0)

        if alpha <= 0 or beta <= 0:
            raise ValueError(f"alpha={alpha}, beta={beta} 均须 > 0")

        node = Node(
            node_id="beta__T__0",
            distribution_family="DIST-BETA",
            params={"alpha": alpha, "beta": beta},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "bounded_fraction: Beta(alpha, beta), "
                          "Fisher trace=psi1(a)+psi1(b)-2psi1(a+b)",
            "citation": "标准统计（贝塔分布与共轭描述）",
            "ib_source": {
                "param": "alpha",
                "range": self._param_range(entry, "alpha", 0.5, 5.0),
                "channel": "beta",
                "beta_fixed": beta,
            },
        }
        return graph

    # ------------------------------------------------------------------
    # 威布尔分布族（CT-WEIBULL-001）
    # ------------------------------------------------------------------
    def _encode_weibull(self, entry: dict) -> EncodedGraph:
        """失效寿命物理 → 威布尔分布 Weibull(shape, scale)。

        物理场景（Weibull 统计断裂力学的标准模型）：
        - 脆性断裂：最弱链接模型，形状 k 对应缺陷尺寸分布（k≈1 指数型，
          k>1 磨损型老化，k<1 早期失效/婴幼期死亡率）。
        - 疲劳寿命：循环载荷下裂纹萌生寿命（k 通常 2–4）。
        - 尺度 scale 是特征寿命（63.2% 分位）。

        Fisher 关于 (shape, scale)：
        trace = k²/λ² + [(1−γ_E)²+π²/6]/k²，det = π²/(6λ²)（与形状无关），
        交叉项 g_kλ = (γ_E−1)/λ。k=1 退化为指数分布（I_λλ=1/λ² ✓ 一致）。

        单节点编码（连续非高斯正支撑，无 X→T→Y 高斯信道）。
        """
        p = self._extract_params(entry)
        shape = p.get("shape", p.get("k", 1.0))
        scale = p.get("scale", p.get("lam", 1.0))

        if shape <= 0 or scale <= 0:
            raise ValueError(f"shape={shape}, scale={scale} 均须 > 0")

        node = Node(
            node_id="weibull__T__0",
            distribution_family="DIST-WEIBULL",
            params={"shape": shape, "scale": scale},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "failure_life: Weibull(shape, scale), "
                          "Fisher trace=k^2/l^2+[(1-gE)^2+pi^2/6]/k^2, det=pi^2/(6l^2)",
            "citation": "Weibull 1951（最弱链接失效统计）",
            "ib_source": {
                "param": "shape",
                "range": self._param_range(entry, "shape", 0.5, 5.0),
                "channel": "weibull",
                "scale_fixed": scale,
            },
        }
        return graph

    # ------------------------------------------------------------------
    # 冯·米塞斯分布族（CT-VONMISES-001）
    # ------------------------------------------------------------------
    def _encode_vonmises(self, entry: dict) -> EncodedGraph:
        """圆上角度变量 → 冯·米塞斯分布 VM(μ, κ)。

        物理场景：
        - 锁相环相位噪声：VCO 相位误差在环带宽内近似 VM(μ=0, κ)
          （κ 随环路增益增大——相位集中度即"锁相深度"）。
        - 液晶分子取向：指向矢相对平均取向的方位涨落（κ ∝ 弹性/温度比）。
        - 转子角涨落：轴承摩擦约束下的角度误差分布。

        Fisher 关于 (μ, κ)：I_μμ = κ·I₁(κ)/I₀(κ)（解析，Bessel 比），
        I_κκ = ½(1+I₂/I₀)−(I₁/I₀)²（Bessel 闭式，= d²/dκ² ln I₀），
        I_μκ = 0（圆平移对称正交）。
        本族是库中首个圆支撑 S¹ 分布族（新拓扑：周期边界）。
        κ→∞ 高斯极限 N(μ, 1/κ)（I_μμ→κ ✓ 与高斯族 1/σ² 一致）。

        单节点编码（圆上连续，无 X→T→Y 高斯信道）。
        """
        p = self._extract_params(entry)
        kappa = p.get("kappa", 1.0)
        mu = p.get("mu", 0.0)

        if kappa <= 0:
            raise ValueError(f"kappa={kappa} 须 > 0")

        node = Node(
            node_id="vonmises__T__0",
            distribution_family="DIST-VONMISES",
            params={"mu": mu, "kappa": kappa},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "circular_phase: VM(mu, kappa), "
                          "I_mu=kappa*I1/I0, I_kappa=(1+I2/I0)/2-(I1/I0)^2, "
                          "I_mukappa=0",
            "citation": "Mardia & Jupp, Directional Statistics, Ch.3",
            "ib_source": {
                "param": "kappa",
                "range": self._param_range(entry, "kappa", 0.1, 10.0),
                "channel": "vonmises",
            },
        }
        return graph

    # ------------------------------------------------------------------
    # 几何分布族（CT-GEOM-001）
    # ------------------------------------------------------------------
    def _encode_geometric(self, entry: dict) -> EncodedGraph:
        """几何分布 Geo(p)（P(X=k)=p(1-p)^(k-1), k>=1）→ 首次成功等待。

        物理场景（离散，Fisher=1/(p²(1-p)) 非对称非幂律）：
        - 首次成功等待：重复试验直到首次成功的次数
        - 光子探测效率：探测到首个光子前的曝光次数
        - 缺陷搜索：发现首个缺陷前检查的样品数

        单节点编码（离散分布，无 X→T→Y 信道结构）。
        """
        p = self._extract_params(entry)
        prob = p.get("p", 0.1)

        if not (0.0 < prob < 1.0):
            raise ValueError(f"p 须在 (0,1) 内，收到 {prob}")

        node = Node(
            node_id="geometric__T__0",
            distribution_family="DIST-GEOMETRIC",
            params={"p": prob},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "first_success_wait: Geo(p), Fisher=1/(p^2*(1-p))",
            "citation": "标准概率论（几何分布）",
        }
        return graph

    # ------------------------------------------------------------------
    # 柯西分布族（CT-CAUCHY-001）
    # ------------------------------------------------------------------
    def _encode_cauchy(self, entry: dict) -> EncodedGraph:
        """柯西分布 Cauchy(0, γ) → 重尾共振线形。

        物理场景（重尾，Fisher=1/(2γ²)，有限 Fisher 但无限方差）：
        - 共振线形：洛伦兹线形（谱线自然宽度）
        - 谱线宽度：受激辐射谱线的半高半宽
        - 柯西噪声：重尾噪声（脉冲干扰）

        单节点编码（位置 x0 固定 0，仅关于尺度 γ）。
        """
        p = self._extract_params(entry)
        gamma = p.get("gamma", p.get("scale", 1.0))

        if gamma <= 0:
            raise ValueError(f"gamma={gamma} 须 > 0")

        node = Node(
            node_id="cauchy__T__0",
            distribution_family="DIST-CAUCHY",
            params={"gamma": gamma},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "lorentzian: Cauchy(0, gamma), Fisher=1/(2*gamma^2)",
            "citation": "标准统计（柯西/洛伦兹分布）",
        }
        return graph

    # ------------------------------------------------------------------
    # 瑞利分布族（CT-RAYLEIGH-001）
    # ------------------------------------------------------------------
    def _encode_rayleigh(self, entry: dict) -> EncodedGraph:
        """瑞利分布 Rayleigh(σ) → 二维高斯幅值（正支撑）。

        物理场景（Fisher=4/σ²，系数 4 由四阶矩给出）：
        - 风速：二维风速矢量的幅值
        - 散射强度：随机散射场的包络幅值
        - 湍流脉动：湍流速度脉动的幅值

        单节点编码（正支撑连续，无 X→T→Y 信道结构）。
        """
        p = self._extract_params(entry)
        sigma = p.get("sigma", p.get("scale", 1.0))

        if sigma <= 0:
            raise ValueError(f"sigma={sigma} 须 > 0")

        node = Node(
            node_id="rayleigh__T__0",
            distribution_family="DIST-RAYLEIGH",
            params={"sigma": sigma},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "envelope: Rayleigh(sigma), Fisher=4/sigma^2",
            "citation": "标准统计（瑞利分布，二维高斯幅值）",
        }
        return graph

    # ------------------------------------------------------------------
    # 逻辑斯蒂分布族（CT-LOGISTIC-001）
    # ------------------------------------------------------------------
    def _encode_logistic(self, entry: dict) -> EncodedGraph:
        """逻辑斯蒂分布 Logistic(0, s) → S 形生长阈值。

        物理场景（Fisher=(π²+3)/(9s²)，π² 项是代数标志）：
        - S 形生长：生长过程速率的阈值涨落
        - 激活阈值：神经元激活阈值的统计分布
        - 相界序参量：相变序参量的软模涨落

        单节点编码（位置 μ 固定 0，仅关于尺度 s）。
        """
        p = self._extract_params(entry)
        scale = p.get("scale", p.get("s", 1.0))

        if scale <= 0:
            raise ValueError(f"scale={scale} 须 > 0")

        node = Node(
            node_id="logistic__T__0",
            distribution_family="DIST-LOGISTIC",
            params={"scale": scale},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "sigmoid: Logistic(0, scale), Fisher=(pi^2+3)/(9*scale^2)",
            "citation": "标准统计（逻辑斯蒂分布）",
        }
        return graph

    # ------------------------------------------------------------------
    # 卡方分布族（扩族批次 3，CT-CHISQ-001）
    # ------------------------------------------------------------------
    def _encode_chisq(self, entry: dict) -> EncodedGraph:
        """卡方分布 χ²(k) → 标准二次型自由度（纯 ψ₁ 曲线）。

        物理场景（Fisher=ψ₁(k/2)/4，库中首个纯 trigamma 曲线族）：
        - 白化残差能量：k 维白化残差平方和（最小二乘）
        - 马氏距离平方：k 维特征离群检测 v^T Σ⁻¹ v
        - 拟合优度统计量：k 个 bin 标准化偏差平方和（皮尔逊卡方）

        单节点编码（χ²(k)=Gamma(k/2, 2)，自由度 k 为形状参数）。
        """
        p = self._extract_params(entry)
        k = p.get("k", 4.0)

        if k <= 0:
            raise ValueError(f"k={k} 须 > 0")

        node = Node(
            node_id="chisq__T__0",
            distribution_family="DIST-CHISQ",
            params={"k": k},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "chi_square: chi2(k)=Gamma(k/2,2), "
                          "Fisher=polygamma(1,k/2)/4",
            "citation": "标准统计（卡方分布）",
        }
        return graph

    # ------------------------------------------------------------------
    # 负二项分布族（扩族批次 3，CT-NEGBIN-001）
    # ------------------------------------------------------------------
    def _encode_negbin(self, entry: dict) -> EncodedGraph:
        """负二项分布 NB(k, p) → 成簇计数（第 k 次成功等待）。

        物理场景（超散计数，Fisher(关于 p)=k/(p²(1-p))，k=1 退化几何）：
        - 热光光子计数：r 个时空模式散斑热光（泊松-伽马混合）
        - 闪烁计数：伽马强度调制的光子计数
        - 暗计数成簇：雪崩倍增探测器暗计数的簇结构

        单节点编码。k 为结构常数（成功次数/伽马混合形状），
        Fisher 摘要为关于 p 的条件信息（I(k) 无初等闭式）。
        """
        p = self._extract_params(entry)
        kk = p.get("k", 2.0)
        prob = p.get("p", 0.1)

        if kk <= 0:
            raise ValueError(f"k={kk} 须 > 0")
        if not (0.0 < prob < 1.0):
            raise ValueError(f"p 须在 (0,1) 内，收到 {prob}")

        node = Node(
            node_id="negbin__T__0",
            distribution_family="DIST-NEGBIN",
            params={"k": kk, "p": prob},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "negative_binomial: P(n)=C(n-1,k-1)p^k(1-p)^(n-k), "
                          "Fisher_p=k/(p^2*(1-p))",
            "citation": "标准统计（负二项分布）",
        }
        return graph

    # ------------------------------------------------------------------
    # 对数正态分布族（扩族批次 3，CT-LOGNORM-001）
    # ------------------------------------------------------------------
    def _encode_lognormal(self, entry: dict) -> EncodedGraph:
        """对数正态分布 LN(μ, σ) → 乘性噪声（对数空间加性）。

        物理场景（Fisher 矩阵 diag(1/σ², 2/σ²) 与高斯同形——双射不变性）：
        - 对数均值回复增长：dX = [κ(μ−ln X)+σ²/2]X dt + σX dW（κ=1/2）
        - 凝并破碎粒径：气溶胶定态粒径分布
        - 阴影衰落：无线大尺度路径损耗绕均值波动

        单节点编码（乘性噪声 paradigm：X=exp(Y), Y~N(μ,σ)）。
        """
        p = self._extract_params(entry)
        mu = p.get("mu", 0.0)
        sigma = p.get("sigma", 1.0)

        if sigma <= 0:
            raise ValueError(f"sigma={sigma} 须 > 0")

        node = Node(
            node_id="lognormal__T__0",
            distribution_family="DIST-LOGNORM",
            params={"mu": mu, "sigma": sigma},
            noise_type="multiplicative",
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "lognormal: X=exp(Y), Y~N(mu,sigma); "
                          "Fisher trace=3/sigma^2, det=2/sigma^4",
            "citation": "标准统计（对数正态分布）",
        }
        return graph

    # ------------------------------------------------------------------
    # 双高斯混合族（扩族批次 4，CT-MIX2-001）
    # ------------------------------------------------------------------
    def _encode_mix2(self, entry: dict) -> EncodedGraph:
        """对称双高斯混合 p(x) = ½N(x|−d/2,σ) + ½N(x|+d/2,σ)。

        物理场景（双模态/双稳/外差，同族不同源）：
        - 双模态传感器：两模态输出叠加（对称双峰）
        - 双稳测量：对称双稳势阱观测
        - 外差拍频：两频率分量的测量分布

        Fisher 关于分离度 d 数值积分（fisher_mix2_separation）：
        0 → 单调上升 → 1/(4σ²) 平台（S 形饱和，库中首个饱和顶曲线，R4 平膝族）。

        单节点编码（非标准分布族，无 X→T→Y 高斯信道）。
        """
        p = self._extract_params(entry)
        d = p.get("d", 2.0)
        w = p.get("w", 0.5)
        sigma = p.get("sigma", 1.0)

        if not (0.0 < w < 1.0) or sigma <= 0:
            raise ValueError(f"w={w} 须在 (0,1)，sigma={sigma} 须 > 0")

        node = Node(
            node_id="mix2__T__0",
            distribution_family="DIST-MIX2",
            params={"d": d, "w": w, "sigma": sigma},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "mixture: p=w*N(-d/2,s)+(1-w)*N(+d/2,s); "
                          "Fisher(d) 数值积分，0→单调上升→1/(4s^2) 平台",
            "citation": "标准混合模型（信息几何）",
        }
        return graph

    # ------------------------------------------------------------------
    # 截断高斯族（扩族批次 4，CT-TRUNCNORM-001）
    # ------------------------------------------------------------------
    def _encode_truncnorm(self, entry: dict) -> EncodedGraph:
        """单边右截断高斯 p(x) = N(x|μ,σ)/(1−Φ(z))，x > τ，z=(τ−μ)/σ。

        物理场景（阈值化/截尾观测）：
        - 阈值化传感器：低于阈值不记录（单边观测）
        - 截尾寿命：右截断寿命观测
        - 门限触发：触发阈值以上的幅度分布

        Fisher 关于截断点 τ 解析：I(τ) = φ(z)²/(σ²(1−Φ(z))²)
        ——「平→陡」上升膝（R4 平膝族，与 landau 的下降膝互补）。

        单节点编码。
        """
        p = self._extract_params(entry)
        tau = p.get("tau", -1.0)
        mu = p.get("mu", 0.0)
        sigma = p.get("sigma", 1.0)

        if sigma <= 0:
            raise ValueError(f"sigma={sigma} 须 > 0")

        node = Node(
            node_id="truncnorm__T__0",
            distribution_family="DIST-TRUNCNORM",
            params={"tau": tau, "mu": mu, "sigma": sigma},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "truncated: p=N/(1-Phi(z)), x>tau; "
                          "I(tau)=phi(z)^2/(s^2*(1-Phi(z))^2)",
            "citation": "标准统计（截断高斯分布）",
        }
        return graph

    # ------------------------------------------------------------------
    # 截断几何族（R4b，CT-TRUNCGEOM-001）
    # ------------------------------------------------------------------
    def _encode_truncgeom(self, entry: dict) -> EncodedGraph:
        """有限支撑截断几何 p(n) = p·q^(n−1)/(1−q^k)，n=1..k，q=1−p。

        物理场景（有限缓冲/窗口计数）：
        - 有限缓冲排队：容量 k 的队列中首事件等待
        - 窗口化缺陷计数：检测窗口 k 内的首缺陷位置
        - 截尾寿命级数：离散截尾寿命观测

        Fisher 关于 p 为 k 项有限和（fisher_truncgeom_p）；
        k→∞ 退化 Geo(p)。R4b 目标：与 negbin 同构的输入模式
        （k 列 + p 列同时填充，且 k 非 ψ₁ 语义）+ U 形左腿目标。

        单节点编码。
        """
        p = self._extract_params(entry)
        k = p.get("k", 10.0)
        prob = p.get("p", 0.3)

        if float(k) < 1 or not (0.0 < float(prob) < 1.0):
            raise ValueError(f"k={k} 须 ≥ 1，p={prob} 须在 (0,1)")

        node = Node(
            node_id="truncgeom__T__0",
            distribution_family="DIST-TRUNCGEOM",
            params={"k": float(k), "p": float(prob)},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "truncgeom: P(n)=p*q^(n-1)/(1-q^k), n=1..k; "
                          "I(p)=finite sum of score^2; k->inf: 1/(p^2*(1-p))",
            "citation": "标准概率论（截断几何分布）",
        }
        return graph

    # ------------------------------------------------------------------
    # 逆伽马族（R4c，CT-INVGAMMA-001）
    # ------------------------------------------------------------------
    def _encode_invgamma(self, entry: dict) -> EncodedGraph:
        """逆伽马 IG(k, θ)：p(x) = θ^k x^(−k−1) e^(−θ/x) / Γ(k)，x > 0。

        物理场景（精度/尺度倒数）：
        - 传导率/电阻率分布（材料参数倒数）
        - 散粒噪声方差尺度（Bayes 逆方差先验）
        - 风速功率谱尺度

        Fisher 与伽马完全同形（样本空间微分同胚 x→1/x 不变性）：
        trace = ψ₁(k) + k/θ²，det = (k·ψ₁(k) − 1)/θ²。
        R4c 目标：与 gamma 同构的输入模式（k + theta 双列，ψ₁ 语义）。

        单节点编码。
        """
        p = self._extract_params(entry)
        k = p.get("k", 2.0)
        theta = p.get("theta", 1.0)

        if float(k) <= 0 or float(theta) <= 0:
            raise ValueError(f"k={k}, theta={theta} 均须 > 0")

        node = Node(
            node_id="invgamma__T__0",
            distribution_family="DIST-INVGAMMA",
            params={"k": float(k), "theta": float(theta)},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "invgamma: IG(k,theta); Fisher=Gamma(k,theta) 同形 "
                          "(x->1/x 微分同胚)；trace=psi1(k)+k/theta^2",
            "citation": "标准统计（逆伽马分布）",
        }
        return graph

    # ------------------------------------------------------------------
    # 逆卡方族（R4d，CT-INVCHISQ-001）
    # ------------------------------------------------------------------
    def _encode_invchisq(self, entry: dict) -> EncodedGraph:
        """逆卡方 χ⁻²(k)：p(x) ∝ x^(−k/2−1) e^(−1/(2x))（θ=2 固定结构常数）。

        物理场景（方差先验/精度二次型）：
        - 未知方差尺度（scaled-inverse-chi² 固定尺度变体）
        - 白噪声方差先验

        Fisher 关于 k = ψ₁(k/2)/4——与 chisq 逐点相同。
        R4d 目标：chisq 同构（k 单列 + 纯 ψ₁ 目标；θ 不占参数列）。

        单节点编码（k 单列）。
        """
        p = self._extract_params(entry)
        k = p.get("k", 4.0)

        if float(k) <= 0:
            raise ValueError(f"k={k} 须 > 0")

        node = Node(
            node_id="invchisq__T__0",
            distribution_family="DIST-INVCHISQ",
            params={"k": float(k)},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "invchisq: chi^-2(k)=IG(k/2, theta=2 固定); "
                          "I(k)=psi1(k/2)/4（与 chisq 逐点相同）",
            "citation": "标准统计（逆卡方分布）",
        }
        return graph

    # ------------------------------------------------------------------
    # 弗雷歇族（R4d，CT-FRECHET-001）
    # ------------------------------------------------------------------
    def _encode_frechet(self, entry: dict) -> EncodedGraph:
        """弗雷歇 Fréchet(α, s)：p(x) = α s^α x^(−α−1) e^(−(s/x)^α)，x>0。

        物理场景（上限极值）：
        - 风速极值（年最大风速）
        - 洪水峰值（流域年最大流量）
        - 材料上限强度（最弱链接的上限对偶）

        Fisher：I_αα = C/α²、I_ss = α²/s²、I_αs = (1−γ)/s，
        trace = C/α² + α²/s²（U 形，C=(1−γ)²+π²/6 与 weibull 同常数）、
        det = π²/(6s²)。
        R4d 目标：weibull 同构（shape+scale 双列 + 同常数 U 形目标）。

        单节点编码。
        """
        p = self._extract_params(entry)
        shape = p.get("shape", p.get("alpha", 2.0))
        scale = p.get("scale", p.get("s", 1.0))

        if float(shape) <= 0 or float(scale) <= 0:
            raise ValueError(f"shape={shape}, scale={scale} 均须 > 0")

        node = Node(
            node_id="frechet__T__0",
            distribution_family="DIST-FRECHET",
            params={"shape": float(shape), "scale": float(scale)},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "frechet: max-extreme; y=(s/x)^a~Exp(1); "
                          "trace=C/a^2+a^2/s^2（U，C 与 weibull 同）、det=pi^2/(6s^2)",
            "citation": "标准极值理论（Fréchet 分布）",
        }
        return graph

    # ------------------------------------------------------------------
    # 莱斯族（R4d 第三刀，CT-RICIAN-001）
    # ------------------------------------------------------------------
    def _encode_rician(self, entry: dict) -> EncodedGraph:
        """莱斯衰落 Rice(ν, σ)：p(x) = x e^(−(x²+ν²)/2σ²) I₀(xν/σ²)/σ²。

        物理场景（K 因子衰落）：
        - 无线信道幅度（强主径 + 散射，K=ν²/2σ²）
        - 声呐/雷达幅度检测

        σ=1 固定为结构常数（不占参数列）；κ=K 因子为扫描参数。
        Fisher 关于 κ 数值积分（fisher_rician_kappa）：
        κ→0（Rayleigh 极限）I→1；单调降；κ→∞ 尾 I→1/(2κ)。
        R4d 第三刀目标：vonmises 同构（kappa 单列 + 深集中 Fisher→0 衰减）。

        单节点编码（kappa 单列）。
        """
        p = self._extract_params(entry)
        kappa = p.get("kappa", p.get("k_factor", 4.0))

        if float(kappa) < 0:
            raise ValueError(f"kappa={kappa} 须 ≥ 0")

        node = Node(
            node_id="rician__T__0",
            distribution_family="DIST-RICIAN",
            params={"kappa": float(kappa)},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "rician: K=nu^2/(2s^2), s=1 固定; "
                          "I(k)=∫(−1+(x/nu)I1/I0)²p dx；0→1、∞→1/(2k)",
            "citation": "标准衰落统计（Rician 分布）",
        }
        return graph

    # ------------------------------------------------------------------
    # 卷绕高斯族（R4d 第三刀 v2，CT-WRAPNORM-001）
    # ------------------------------------------------------------------
    def _encode_wrapnorm(self, entry: dict) -> EncodedGraph:
        """卷绕高斯 WN(0, κ)：p(θ) = (1/2π)(1+2Σρ^{k²}cos kθ)，ρ=e^{−κ²/2}。

        物理场景（圆上相位扩散）：
        - 环形激光相位扩散
        - 天线阵相位误差
        - 分子取向松弛

        κ=底层高斯 σ（桥接命名「kappa」占列）；κ→∞（均匀极限）Fisher→0，
        尾 ~2κ²e^{−κ²}。与 vonmises 三层同构：kappa 列 + 圆支撑 attr +
        衰减到 0 方向（第 51 条修正后的选型）。

        单节点编码（kappa 单列）。
        """
        p = self._extract_params(entry)
        kappa = p.get("kappa", p.get("sigma", 1.0))

        if float(kappa) <= 0:
            raise ValueError(f"kappa={kappa} 须 > 0")

        node = Node(
            node_id="wrapnorm__T__0",
            distribution_family="DIST-WRAPNORM",
            params={"kappa": float(kappa)},
            noise_type=None,
            dimension=1,
        )
        graph = EncodedGraph(nodes=[node], beta=entry.get("beta", 1.0))
        graph.metadata = {
            "physical_params": p,
            "derivation": "wrapnorm: p=(1/2pi)(1+2 sum rho^{k^2}cos kθ), "
                          "rho=e^{-k^2/2}; I(k)→0（均匀极限）",
            "citation": "标准方向统计（卷绕高斯分布）",
        }
        return graph
