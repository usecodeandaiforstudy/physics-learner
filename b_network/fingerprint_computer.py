"""B 网络 FingerprintComputer：EncodedGraph → 指纹 + 数值 vs 解析对照。

职责（架构文档 §2.3.9(a) 的 ChannelTemplateLibrary 线）：
在 SystemEncoder 编码出的信息结构图上计算指纹（Fisher 摘要 + 边互信息），
并对照 DB1 条目的 fingerprint_expectation（解析预期）做数值 vs 解析对照。

这是"机器门 3（解析对照）"的完整落地：SystemEncoder 提供分布参数（数值侧），
fingerprint_expectation 提供解析表达式（解析侧），本模块把两者对齐比较。

Phase 0 边界：
- 解析对照支持 fisher_trace / fisher_det（对表示层节点 T 的 Fisher 摘要）。
- mi_profile（边互信息）作为指纹输出，解析对照留待约定表达式后扩展。
- ib_curve / action_s 需离散化或参数变化率，标记为跳过。
"""

from __future__ import annotations

import math

import numpy as np
import sympy

from core.fingerprint import compute_graph_fingerprint
from core.fisher import fisher_landau, fisher_landau_analytic, fisher_landau_monte_carlo
from core.ib import ib_curve

__all__ = ["FingerprintComputer"]

# Phase 0 支持解析对照的 quantity（其余标记跳过）
_SUPPORTED_QUANTITY = {"fisher_trace", "fisher_det"}

# 朗道族：数值积分 vs 蒙特卡洛采样的相对容差（采样有随机误差，须放宽）
_MC_TOL = 0.10
# 朗道族：数值积分 vs 解析解（Bessel 闭式）的相对容差（应机器精度一致）
_ANALYTIC_TOL = 1e-6
_DIST_LANDAU = "DIST-LANDAU"


class FingerprintComputer:
    """在 EncodedGraph 上计算指纹并做解析对照。"""

    # 参数化信道的 IB 曲线只依赖 (channel, lo, hi)（来自 metadata.ib_source 的
    # 固定 range），与扫描点的具体参数值无关，故做类级缓存，避免同一系统
    # 30 个扫描点重复计算同一条 IB 曲线（每次 BA 约 0.3s，缓存后只算一次）。
    _IB_CACHE: dict = {}

    def compute(self, entry: dict, graph) -> dict:
        """计算指纹 + 数值 vs 解析对照。

        参数
        ----
        entry : dict
            DB1 条目（提供 fingerprint_expectation 与物理参数）。
        graph : EncodedGraph
            SystemEncoder 编码出的图（节点带分布参数、边带信道）。

        返回
        ----
        dict
            {
              "fingerprints": {node_id: {"fisher_trace", "fisher_det"}, ...},
              "mi_profile": [边互信息列表],
              "checks": [{"quantity", "analytic_expr", "analytic_val",
                          "numeric_val", "rel_err", "passed", "skipped"}, ...],
              "passed": bool,
            }
        """
        # 1) 全图指纹：节点 Fisher 摘要 + 边互信息（mi_profile）
        full = compute_graph_fingerprint(graph)

        # 2) 表示层节点（T）的 Fisher 作为 fisher 对照的数值侧
        numeric = self._rep_node_fisher(graph, full)

        # 3) 物理参数符号表（求值解析表达式）
        param_symbols: dict[str, float] = {}
        for p in entry.get("parameters", []):
            name = p.get("name")
            default = p.get("default")
            if name is None or default is None:
                continue
            param_symbols[str(name)] = float(default)

        # 4) 解析对照
        checks: list[dict] = []
        all_pass = True
        for item in entry.get("fingerprint_expectation", []):
            qty = item.get("quantity")
            analytic = item.get("analytic")
            tol = float(item.get("tol", 1.0e-3))

            if qty not in _SUPPORTED_QUANTITY:
                checks.append({
                    "quantity": qty, "passed": True, "skipped": True,
                    "note": f"Phase 0 未实现 {qty} 的解析对照",
                })
                continue

            analytic_val = self._eval(analytic, param_symbols)
            numeric_val = numeric.get(qty)

            if analytic_val is None:
                checks.append({
                    "quantity": qty, "passed": False, "skipped": False,
                    "note": f"解析表达式 {analytic!r} 无法求值",
                })
                all_pass = False
                continue

            if numeric_val is None:
                checks.append({
                    "quantity": qty, "passed": False, "skipped": False,
                    "note": f"数值侧未提供 {qty}",
                })
                all_pass = False
                continue

            rel_err = abs(analytic_val - numeric_val) / (abs(analytic_val) + 1e-300)
            passed = rel_err < tol
            all_pass = all_pass and passed
            checks.append({
                "quantity": qty,
                "analytic_expr": analytic,
                "analytic_val": analytic_val,
                "numeric_val": numeric_val,
                "rel_err": rel_err,
                "passed": passed,
                "skipped": False,
            })

        # 5) 朗道族：三重验证（数值积分 = 正式值，解析解 + 蒙特卡洛 = 独立对照）
        #    三者用同一公式 I(a)=Var(φ²)/(4k²T²)，但 ⟨φ²⟩、⟨φ⁴⟩ 分别由
        #    梯形积分 / Bessel 闭式 / Metropolis 采样估计，独立手段一致则数值积分未算错。
        for node in graph.nodes:
            if node.distribution_family != _DIST_LANDAU:
                continue
            a = float(node.params["a"])
            b = float(node.params["b"])
            kT = float(node.params.get("kT", 1.0))
            num_val = full["nodes"].get(node.node_id, {}).get("fisher_trace")
            if num_val is None:
                continue

            # 5.1 解析解（修正贝塞尔函数闭式，机器精度）
            an_val = fisher_landau_analytic(a, b, kT)
            rel_err = abs(num_val - an_val) / (abs(num_val) + 1e-300)
            passed = rel_err < _ANALYTIC_TOL
            all_pass = all_pass and passed
            checks.append({
                "quantity": "fisher_trace",
                "analytic_expr": "bessel_analytic",
                "analytic_val": an_val,
                "numeric_val": num_val,
                "rel_err": rel_err,
                "passed": passed,
                "skipped": False,
            })

            # 5.2 蒙特卡洛采样（Metropolis，采样误差 ~1-6%）
            mc_val = fisher_landau_monte_carlo(a, b, kT)
            if mc_val > 0:
                rel_err = abs(num_val - mc_val) / (abs(num_val) + 1e-300)
                passed = rel_err < _MC_TOL
                all_pass = all_pass and passed
                checks.append({
                    "quantity": "fisher_trace",
                    "analytic_expr": "monte_carlo",
                    "analytic_val": mc_val,
                    "numeric_val": num_val,
                    "rel_err": rel_err,
                    "passed": passed,
                    "skipped": False,
                })

            # 5.3 a=0 临界点解析解（Γ 函数分支；扫描网格 linspace 通常不含 0，
            #     须显式验证，确保 fisher_landau_analytic 的 a=0 分支正确）
            num_val_0 = fisher_landau(0.0, b, kT)
            an_val_0 = fisher_landau_analytic(0.0, b, kT)
            rel_err_0 = abs(num_val_0 - an_val_0) / (abs(num_val_0) + 1e-300)
            passed_0 = rel_err_0 < _ANALYTIC_TOL
            all_pass = all_pass and passed_0
            checks.append({
                "quantity": "fisher_trace",
                "analytic_expr": "bessel_analytic_a0",
                "analytic_val": an_val_0,
                "numeric_val": num_val_0,
                "rel_err": rel_err_0,
                "passed": passed_0,
                "skipped": False,
            })

        # 5bis 冯·米塞斯族：双库对照（sympy besseli vs scipy iv）
        #    I_μμ = κ·I₁(κ)/I₀(κ) 与 I_κκ = ½(1+I₂/I₀)−(I₁/I₀)² 均有 Bessel
        #    闭式，但两侧用不同库实现（sympy 求值 vs scipy iv），可捕获库
        #    实现错误。（I_κκ 此前用数值微分 + 步长稳定性自检，但 h=1e-6 时
        #    二阶差分已被浮点舍入淹没，改闭式后升级为同级的双库对照。）
        #    MC score 对照（真正的独立手段）在 b_network/test_new_families.py。
        for node in graph.nodes:
            if node.distribution_family != "DIST-VONMISES":
                continue
            kappa = float(node.params["kappa"])
            from core.fisher import fisher_vonmises_mu, fisher_vonmises_kappa

            # 5bis.1 I_μμ：sympy vs scipy 双库对照
            expr = sympy.sympify("kappa*besseli(1, kappa)/besseli(0, kappa)")
            an_val = float(expr.subs({"kappa": kappa}))
            sp_val = fisher_vonmises_mu(kappa)
            rel_err = abs(an_val - sp_val) / (abs(an_val) + 1e-300)
            passed = rel_err < _ANALYTIC_TOL
            all_pass = all_pass and passed
            checks.append({
                "quantity": "fisher_trace",
                "analytic_expr": "sympy_besseli_I_mu",
                "analytic_val": an_val,
                "numeric_val": sp_val,
                "rel_err": rel_err,
                "passed": passed,
                "skipped": False,
            })

            # 5bis.2 I_κκ：闭式 sympy vs scipy 双库对照
            expr = sympy.sympify(
                "(1 + besseli(2, kappa)/besseli(0, kappa))/2 "
                "- (besseli(1, kappa)/besseli(0, kappa))**2"
            )
            an_val = float(expr.subs({"kappa": kappa}))
            sp_val = fisher_vonmises_kappa(kappa)
            rel_err = abs(an_val - sp_val) / (abs(an_val) + 1e-300)
            passed = rel_err < _ANALYTIC_TOL
            all_pass = all_pass and passed
            checks.append({
                "quantity": "fisher_trace",
                "analytic_expr": "sympy_besseli_I_kappa",
                "analytic_val": an_val,
                "numeric_val": sp_val,
                "rel_err": rel_err,
                "passed": passed,
                "skipped": False,
            })

        return {
            "fingerprints": full["nodes"],
            "mi_profile": full["mi_profile"],
            "checks": checks,
            "passed": all_pass,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _rep_node_fisher(graph, full: dict) -> dict:
        """选择表示层节点（node_id 含 __T__，否则首个节点）的 Fisher 摘要。"""
        for node in graph.nodes:
            if "__T__" in node.node_id:
                return full["nodes"].get(node.node_id, {})
        # 无 T 节点（单节点编码）时用首个节点
        if graph.nodes:
            return full["nodes"].get(graph.nodes[0].node_id, {})
        return {}

    # ------------------------------------------------------------------
    @staticmethod
    def _eval(expr_str: str, param_symbols: dict[str, float]) -> float | None:
        """用 sympy 求值解析表达式，代入物理参数 default 值。"""
        try:
            # locals 强制把参数名解析为 Symbol：否则 'beta' 会撞 sympy 的不完全
            # beta 函数、'shape' 撞命名空间中的同名函数，subs 失败或表达式
            # 被解析成函数运算（如 shape**2 → function**Integer）。
            expr = sympy.sympify(
                expr_str, locals={k: sympy.Symbol(k) for k in param_symbols}
            )
            val = float(expr.subs({sympy.Symbol(k): v for k, v in param_symbols.items()}))
        except Exception:  # noqa: BLE001
            return None
        if val != val or val in (float("inf"), float("-inf")):  # NaN / inf
            return None
        return val

    # ------------------------------------------------------------------
    # IB 曲线（X→T→Y 链的离散化 + BA 算法）
    # ------------------------------------------------------------------
    def compute_ib_curve(
        self,
        graph,
        n_bins: int = 60,
        beta_grid: np.ndarray | None = None,
    ) -> dict:
        """对信息结构图的端到端高斯信道做 IB 曲线。

        从图中提取源分布 p(x)（源节点 X）与端到端信道 p(y|x)（复合所有边），
        离散化后调用 core.ib.ib_curve（Blahut–Arimoto）。

        参数
        ----
        graph : EncodedGraph
            含 X 源节点与至少一条高斯信道边。
        n_bins : int
            离散化格点数。
        beta_grid : np.ndarray, 可选
            beta 网格；默认 logspace(-2, 3, 32)。

        返回
        ----
        dict
            {"beta": 数组, "I_XT": 数组, "I_TY": 数组}
        """
        src = self._node_by_role(graph, "X")
        # 非高斯单节点编码（指数/泊松/伯努利）：走参数化信道 IB 路径
        if src is None and "ib_source" in graph.metadata:
            return self._compute_ib_curve_parametric(graph, n_bins, beta_grid)

        if src is None:
            return {"beta": np.asarray([]), "I_XT": np.asarray([]), "I_TY": np.asarray([])}

        # 非高斯源（如指数/泊松/伯努利单节点编码）无 X→T→Y 高斯信道，
        # 无法走当前的高斯 IB 离散化路径，返回空曲线（诚实跳过而非报错）。
        if src.distribution_family != "DIST-GAUSS":
            return {"beta": np.asarray([]), "I_XT": np.asarray([]), "I_TY": np.asarray([])}

        mu = float(src.params.get("mu", 0.0))
        sigma_src = float(src.params["sigma"])

        # 端到端信道（复合所有边）：Y = mu + (X-mu)*slope + noise
        slope, var_cond = self._end_to_end_channel(graph)
        sigma_noise = math.sqrt(max(var_cond, 0.0))
        # 端到端输出标准差：Var(Y) = slope^2 * Var(X) + Var(Y|X)
        sigma_y = math.sqrt(slope**2 * sigma_src**2 + var_cond)

        # 离散化网格：须同时覆盖源分布与输出分布，否则输出质量落在网格外，
        # 导致 p(y|x) 行归一化失真、I(T;Y) 被严重低估（扩散族高 D 时尤为明显）。
        halfwidth = 4.0 * max(sigma_src, sigma_y)
        xs = np.linspace(mu - halfwidth, mu + halfwidth, n_bins)
        px = self._gaussian_pdf(xs, mu, sigma_src)
        px /= px.sum()

        p_y_given_x = np.zeros((n_bins, n_bins))
        for i, x in enumerate(xs):
            row = self._gaussian_pdf(xs, mu + (x - mu) * slope, sigma_noise)
            p_y_given_x[i] = row / row.sum()

        # tol=1e-6：放宽收敛判据，比 1e-12 快约 1.6 倍，对 I_XT/I_TY 的影响 <1e-4
        # （n_bins 与 beta_grid 不可为提速而改：前者影响 I(X;T) 标度，后者是约定）
        i_xt, i_ty = ib_curve(px, p_y_given_x, beta_grid=beta_grid,
                              n_iter=200, tol=1e-6)
        return {"beta": beta_grid if beta_grid is not None
                else np.logspace(-2, 3, 32), "I_XT": i_xt, "I_TY": i_ty}

    # ------------------------------------------------------------------
    # 非高斯参数化信道的 IB（源参数 → 观测）
    # ------------------------------------------------------------------
    def _compute_ib_curve_parametric(
        self,
        graph,
        n_bins: int = 60,
        beta_grid: np.ndarray | None = None,
    ) -> dict:
        """对非高斯分布族的"源参数 → 观测"信道做 IB 曲线。

        源 = 物理参数 θ（在 ib_source.range 上取均匀先验），
        观测 = 服从该参数分布族的变量（指数寿命 / 泊松计数 / 伯努利占据）。
        信道 p(y|θ) 由分布族定义，离散化后走 BA 算法。

        参数
        ----
        graph : EncodedGraph
            单节点非高斯编码，metadata 含 ib_source。
        n_bins : int
            源参数的离散化格点数（指数信道也用它离散化观测寿命）。
        beta_grid : np.ndarray, 可选
            beta 网格。

        返回
        ----
        dict
            {"beta", "I_XT", "I_TY"}
        """
        ib_src = graph.metadata.get("ib_source", {})
        channel = ib_src.get("channel")
        lo, hi = ib_src.get("range", [0.1, 10.0])

        cache_key = (channel, round(float(lo), 6), round(float(hi), 6), n_bins)
        if cache_key in self._IB_CACHE:
            return self._IB_CACHE[cache_key]

        # 源参数离散化（均匀先验）
        theta = np.linspace(lo, hi, n_bins)
        px = np.full(n_bins, 1.0 / n_bins)

        if channel == "exponential":
            # 观测 = 寿命 y ∈ [0, y_max]，y_max = 5 / lo（lo 越小寿命越长）
            y_max = 5.0 / max(lo, 1e-6)
            ys = np.linspace(0.0, y_max, n_bins)
            p_y_given_x = np.zeros((n_bins, n_bins))
            for i, lam in enumerate(theta):
                # p(y|λ) ∝ λ e^{-λ y}，用 PDF 在网格上求值（网格点值 ≈ 密度）
                row = lam * np.exp(-lam * ys)
                row /= row.sum()
                p_y_given_x[i] = row

        elif channel == "landau_doublewell":
            # **双阱（朗道）信道（2026-10-02 加，第 111 条）**：
            #   源参数 a（约化温度系数，均匀先验，可为负=双阱），
            #   观测 = 序参量 φ，p(φ|a) ∝ exp(−(a φ²/2 + φ⁴/4)/kT)。
            # φ 网格须覆盖双阱极小 ±√(−a)：取 max(3.5, 2√max|a| + 1)。
            b_fixed = float(ib_src.get("b_fixed", 1.0))
            kT_fixed = float(ib_src.get("kT_fixed", 1.0))
            phi_max = max(3.5, 2.0 * math.sqrt(max(abs(float(lo)),
                                                 abs(float(hi)))) + 1.0)
            ys = np.linspace(-phi_max, phi_max, n_bins)
            p_y_given_x = np.zeros((n_bins, n_bins))
            for i, a_i in enumerate(theta):
                logp = -(a_i * ys ** 2 / 2.0 + b_fixed * ys ** 4 / 4.0) / kT_fixed
                logp = logp - logp.max()
                row = np.exp(logp)
                p_y_given_x[i] = row / row.sum()

        elif channel == "scipy":
            # **通用 scipy.stats 信道（2026-10-02 加，第 118 条）**：
            #   一处实现覆盖多个族（rayleigh/lognorm/invweibull/cauchy/
            #   logistic/laplace/chi2/geom/nbinom/invgamma/truncnorm/rice/norm…）
            #   规格来自条目 `ib_source`：`dist` = scipy 分布名、
            #   `arg` = 由 θ 驱动的参数名、`fixed` = 其余固定参数。
            #   y 网格取 θ∈[lo,hi] 上 ppf(0.001)…ppf(0.999) 的**并集极值**，
            #   以覆盖各 θ 下的有效支撑。
            from scipy import stats as _st
            dist_name = ib_src.get("dist")
            arg_name = ib_src.get("arg", "scale")
            fixed = dict(ib_src.get("fixed") or {})
            p_y_given_x = np.zeros((n_bins, n_bins))
            if not dist_name or not hasattr(_st, dist_name):
                ys = np.linspace(0.0, 1.0, n_bins)
                p_y_given_x[:] = 1.0 / n_bins
            else:
                dist = getattr(_st, dist_name)
                # **离散分布须用 pmf + 整数网格（2026-10-02 修）**：
                #   scipy 的 nbinom/geom/poisson 等**只有 `pmf`、没有 `pdf`**
                #   （初版一律调 `.pdf` ⟹ `AttributeError` 全族失败）；
                #   且 y 网格必须落在整数点上，否则 pmf 全 0、行归一化退化。
                is_discrete = hasattr(dist, "pmf") and not hasattr(
                    dist, "pdf")
                _f = dist.pmf if is_discrete else dist.pdf
                los, his = [], []
                for a_i in (float(lo), float(hi)):
                    kw = dict(fixed)
                    kw[arg_name] = a_i
                    try:
                        los.append(float(dist.ppf(0.001, **kw)))
                        his.append(float(dist.ppf(0.999, **kw)))
                    except Exception:  # noqa: BLE001
                        continue
                y_lo = min(los) if los else 0.0
                y_hi = max(his) if his else 1.0
                if not np.isfinite(y_lo):
                    y_lo = 0.0
                if not np.isfinite(y_hi) or y_hi <= y_lo:
                    y_hi = y_lo + 1.0
                if is_discrete:
                    k_lo = int(np.floor(max(y_lo, 0.0)))
                    k_hi = int(np.ceil(y_hi))
                    if k_hi - k_lo > 400:      # 上限保护（避免超长支撑）
                        k_hi = k_lo + 400
                    ys = np.arange(k_lo, k_hi + 1, dtype=float)
                else:
                    ys = np.linspace(y_lo, y_hi, n_bins)
                n_y = len(ys)
                p_y_given_x = np.zeros((n_bins, n_y))
                for i, a_i in enumerate(theta):
                    kw = dict(fixed)
                    kw[arg_name] = float(a_i)
                    row = np.nan_to_num(np.asarray(_f(ys, **kw),
                                                   dtype=float),
                                        nan=0.0, posinf=0.0, neginf=0.0)
                    s = row.sum()
                    p_y_given_x[i] = (row / s) if s > 0 else np.full(
                        n_y, 1.0 / n_y)

        elif channel == "power_gauss":
            # **幂律高斯信道（2026-10-02，第 167 条）**：`scale = c · θ^p`
            #   覆盖三个此前**缺 `ib_source`** 的族的**物理**稳态/核宽度：
            #     · `ou`         ：稳态 sd = √(kT/(2θ))  ⟹ p = −1/2（θ 越大越窄）
            #     · `lc_circuit` ：热平衡 sd = √(kT/C)   ⟹ p = −1/2
            #     · `diffusion`  ：热核   sd = √(2κt)    ⟹ p = +1/2
            #   （此前它们**如实留空**，因为单一 scipy 分布名 + 线性绑定
            #     表达不了这种**非线性**映射 ⟹ 现加**变换**能力）
            c_val = float(ib_src.get("c", 1.0))
            p_val = float(ib_src.get("p", -0.5))
            scales = np.array([c_val * max(float(a), 1e-12) ** p_val
                               for a in theta], dtype=float)
            # y 覆盖 ±6σ（取最大尺度）
            span = 6.0 * (float(scales.max()) if len(scales) else 1.0)
            ys = np.linspace(-max(span, 1e-9), max(span, 1e-9), n_bins)
            p_y_given_x = np.zeros((n_bins, n_bins))
            from scipy import stats as _st2
            for i, sc in enumerate(scales):
                row = np.asarray(_st2.norm.pdf(ys, 0.0, max(sc, 1e-12)),
                                 dtype=float)
                s = row.sum()
                p_y_given_x[i] = (row / s) if s > 0 else np.full(
                    n_bins, 1.0 / n_bins)

        elif channel == "special":
            # **自建分布信道（第 167 条）**：复用 `sim/expand_registry` 的
            #   `SPECIAL_DISTS`（`mix2` 高斯混合 / `truncgeom` 截断几何 /
            #   `phi4` φ⁴），它们**不在 scipy.stats 里** ⟹ 此前无法登记。
            from sim.expand_registry import SPECIAL_DISTS, SPECIAL_NEEDS
            sname = ib_src.get("special")
            arg_name = ib_src.get("arg", "p")
            fixed = dict(ib_src.get("fixed") or {})
            gen_fn = SPECIAL_DISTS.get(sname) if sname else None
            need = list(SPECIAL_NEEDS.get(sname) or [])
            if gen_fn is None or arg_name not in need:
                ys = np.linspace(0.0, 1.0, n_bins)
                p_y_given_x = np.full((n_bins, n_bins), 1.0 / n_bins)
            else:
                # **两步：先实例化生成器，再按参数冻结（2026-10-02 修，真 bug）**
                #   `SPECIAL_DISTS[name]` 是**工厂函数**（如 `_make_mix2`），
                #   初版只调用一次（`gen = SPECIAL_DISTS[name]`）⟹ 拿到的是
                #   **未冻结**分布 ⟹ `d.pdf(...)` 报
                #   `_parse_args() missing 3 required positional arguments`
                #   ⟹ 被 `except` 吞掉 ⟹ **整族曲线静默变全零**（实测 I_TY ≈ 1e-16）。
                gen = gen_fn()
                try:
                    probe = gen(**{k: fixed.get(k, 1.0) for k in need})
                    is_disc = hasattr(probe, "pmf") and not hasattr(
                        probe, "pdf")
                except Exception:  # noqa: BLE001
                    is_disc = (sname == "truncgeom")
                _ff = (lambda d, y: d.pmf(y)) if is_disc else (
                    lambda d, y: d.pdf(y))
                if is_disc:
                    hi_k = int(round(float(fixed.get(
                        "kk", fixed.get("k", 5.0)))))
                    ys = np.arange(1, max(hi_k, 1) + 1, dtype=float)
                else:
                    ys = np.linspace(-12.0, 12.0, n_bins)
                n_y = len(ys)
                p_y_given_x = np.zeros((n_bins, n_y))
                for i, a_i in enumerate(theta):
                    kw = {k: float(fixed.get(k, 1.0)) for k in need}
                    kw[arg_name] = float(a_i)
                    try:
                        d = gen(**kw)
                        row = np.nan_to_num(
                            np.asarray(_ff(d, ys), dtype=float),
                            nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception:  # noqa: BLE001
                        row = np.zeros(n_y)
                    s = row.sum()
                    p_y_given_x[i] = (row / s) if s > 0 else np.full(
                        n_y, 1.0 / n_y)

        elif channel == "poisson":
            # 观测 = 计数 n ∈ {0, 1, ..., n_max}，n_max 覆盖到 hi 的约 3 倍
            n_max = int(math.ceil(3.0 * hi)) + 5
            ys = np.arange(n_max + 1)
            n_obs = len(ys)
            p_y_given_x = np.zeros((n_bins, n_obs))
            for i, lam in enumerate(theta):
                # p(n|λ) = λ^n e^{-λ} / n!
                row = np.exp(-lam) * lam ** ys / np.array(
                    [math.factorial(k) for k in ys])
                p_y_given_x[i] = row / row.sum()

        elif channel == "bernoulli":
            # 观测 = 占据 y ∈ {0, 1}
            p_y_given_x = np.zeros((n_bins, 2))
            for i, p_ in enumerate(theta):
                p_y_given_x[i, 0] = 1.0 - p_  # P(y=0)
                p_y_given_x[i, 1] = p_        # P(y=1)

        elif channel == "gamma":
            # 观测 = 等待时间 y ∈ (0, y_max]（覆盖 k=hi 处 6σ）。
            # 中点网格避开 y=0：k<1 时 pdf 在 0 处发散（pdf(0)=inf），
            # 一行 inf 会毒化整条 IB 曲线（行归一化失败），故同 beta 模式取中点。
            from scipy.stats import gamma as _gamma_dist

            theta_fix = float(ib_src.get("theta_fixed", 1.0))
            y_max = theta_fix * (hi + 6.0 * math.sqrt(hi))
            ys = (np.arange(n_bins) + 0.5) / n_bins * y_max
            p_y_given_x = np.zeros((n_bins, n_bins))
            for i, k in enumerate(theta):
                row = _gamma_dist.pdf(ys, a=k, scale=theta_fix)
                p_y_given_x[i] = row / row.sum()

        elif channel == "beta":
            # 观测 = 有界分数 y ∈ (0,1)，中点网格避开端点奇异（α<1 发散）
            from scipy.stats import beta as _beta_dist

            beta_fix = float(ib_src.get("beta_fixed", 1.0))
            ys = (np.arange(n_bins) + 0.5) / n_bins
            p_y_given_x = np.zeros((n_bins, n_bins))
            for i, a in enumerate(theta):
                row = _beta_dist.pdf(ys, a=a, b=beta_fix)
                p_y_given_x[i] = row / row.sum()

        elif channel == "weibull":
            # 观测 = 失效寿命 y ∈ (0, y_max]，y_max 覆盖 k=lo 的重尾
            from scipy.stats import weibull_min as _wbl

            scale_fix = float(ib_src.get("scale_fixed", 1.0))
            y_max = scale_fix * math.log(1000.0) ** (1.0 / lo)
            ys = np.linspace(y_max / n_bins, y_max, n_bins)
            p_y_given_x = np.zeros((n_bins, n_bins))
            for i, k in enumerate(theta):
                row = _wbl.pdf(ys, c=k, scale=scale_fix)
                p_y_given_x[i] = row / row.sum()

        elif channel == "vonmises":
            # 观测 = 角度 y ∈ [-pi, pi]（圆支撑，μ=0）
            from scipy.special import i0 as _i0

            ys = np.linspace(-math.pi, math.pi, n_bins)
            p_y_given_x = np.zeros((n_bins, n_bins))
            for i, kap in enumerate(theta):
                row = np.exp(kap * np.cos(ys)) / (2.0 * math.pi * _i0(kap))
                p_y_given_x[i] = row / row.sum()

        else:
            return {"beta": np.asarray([]), "I_XT": np.asarray([]), "I_TY": np.asarray([])}

        i_xt, i_ty = ib_curve(px, p_y_given_x, beta_grid=beta_grid,
                              n_iter=200, tol=1e-6)
        result = {"beta": beta_grid if beta_grid is not None
                  else np.logspace(-2, 3, 32), "I_XT": i_xt, "I_TY": i_ty}
        self._IB_CACHE[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # 辅助：节点角色与端到端信道
    # ------------------------------------------------------------------
    @staticmethod
    def _node_by_role(graph, role: str):
        for node in graph.nodes:
            if f"__{role}__" in node.node_id:
                return node
        return None

    @staticmethod
    def _edge_slope_var(edge) -> tuple[float, float]:
        """从边的高斯联合协方差提取回归斜率与条件方差。

        slope = Cov(src,dst)/Var(src) = cov[0,1]/cov[0,0]
        var_cond = Var(dst) - Cov^2/Var(src) = cov[1,1] - cov[0,1]^2/cov[0,0]
        """
        cov = np.asarray(edge.channel.cov, dtype=float)
        slope = cov[0, 1] / cov[0, 0]
        var_cond = cov[1, 1] - cov[0, 1] ** 2 / cov[0, 0]
        return float(slope), float(var_cond)

    @classmethod
    def _end_to_end_channel(cls, graph) -> tuple[float, float]:
        """复合所有边，得到端到端高斯信道的 (slope, var_cond)。

        线性高斯链复合规则（按边顺序 X→…→Y）：
            slope = slope1 * slope2 * ...
            var_cond = slope_{2..n}^2 * var1 + slope_{3..n}^2 * var2 + ... + var_n
        """
        if not graph.edges:
            return 1.0, 0.0

        slopes = []
        vars_ = []
        for e in graph.edges:
            s, v = cls._edge_slope_var(e)
            slopes.append(s)
            vars_.append(v)

        # 从前往后复合
        total_slope = 1.0
        for s in slopes:
            total_slope *= s

        total_var = 0.0
        for i, v in enumerate(vars_):
            # 该边之后所有 slope 的乘积
            downstream = 1.0
            for s in slopes[i + 1:]:
                downstream *= s
            total_var += downstream**2 * v

        return total_slope, total_var

    @staticmethod
    def _gaussian_pdf(xs: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """高斯密度（未归一化也可，离散化后统一归一）。"""
        if sigma <= 0:
            # 退化：delta 分布（仅当 x==mu 处为 1）
            return (np.abs(xs - mu) < 1e-12).astype(float)
        return np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
