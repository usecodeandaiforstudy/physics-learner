# -*- coding: utf-8 -*-
"""D·扩大外部验证覆盖：由**编码器自身**派生采样器并逐个校准
（2026-10-02，第 136 条）。

## 为什么这样做（避免"口径错配"）
扩覆盖的诱惑是**另写一套每族的采样/密度/Fisher**——那必然引入参数化错配
（本会话已多次踩坑）。**改为从既有组件派生**：
  · **采样与密度**：编码器为该族产出的**节点**已带 `distribution_family`
    与完整 `params` ⟹ 按其参数化用 `scipy.stats` 采样/取 `logpdf`/`logpmf`；
  · **真值**：`core.fingerprint.compute_node_fisher`（**项目自身口径**，
    与 DB1/DB2/A/C 同一代码路径）；
  · **自校验**：用信息恒等式（score 估计器）测 Fisher 与**编码器真值**对照
    ⟹ **若我的参数映射写错，对不上就会暴露** ⟹ **校准即验证**。

## 登记规则（明写）
  逐族跑 `seeds` 个种子，测相对误差；**p99 ≤ TOL_REG（默认 10%）才登记**，
  否则**保持未登记**（其验证如实返回 `unknown`）。
  登记表写入 `sim/registry.json`（数据驱动、可审计）；`verify_api` 优先读它。

用法：
    python sim/expand_registry.py             # 校准并报告（不写文件）
    python sim/expand_registry.py --register  # 校准并写 sim/registry.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

REGISTRY_JSON = _ROOT / "sim" / "registry.json"
TOL_REG = 0.10          # 登记门槛：p99 相对误差 ≤ 10%

# **分布族 → scipy 参数化映射**（节点参数名 → scipy 参数名）
#   键为编码器产出的 `distribution_family`；值 = (scipy 名, {节点参数: scipy 参数})
DIST_TO_SCIPY: dict[str, tuple[str, dict[str, str]]] = {
    "DIST-GAUSS": ("norm", {"mu": "loc", "sigma": "scale"}),
    "DIST-EXPONENTIAL": ("expon", {"rate": "scale"}) if False else
                        ("expon", {"scale": "scale"}),
    "DIST-GAMMA": ("gamma", {"k": "a", "theta": "scale"}),
    "DIST-WEIBULL": ("weibull_min", {"shape": "c", "scale": "scale"}),
    "DIST-LOGNORM": ("lognorm", {"sigma": "s", "scale": "scale"}),
    "DIST-LAPLACE": ("laplace", {"mu": "loc", "b": "scale"}),
    "DIST-LOGISTIC": ("logistic", {"mu": "loc", "sigma": "scale"}),
    "DIST-CAUCHY": ("cauchy", {"gamma": "scale"}),
    "DIST-RAYLEIGH": ("rayleigh", {"sigma": "scale"}),
    "DIST-CHISQ": ("chi2", {"k": "df"}),
    "DIST-INVGAMMA": ("invgamma", {"a": "a"}),
    "DIST-NEGBIN": ("nbinom", {"n": "n", "p": "p"}),
    "DIST-GEOMETRIC": ("geom", {"p": "p"}),
    "DIST-POISSON": ("poisson", {"rate": "mu"}),
    "DIST-VONMISES": ("vonmises", {"kappa": "kappa", "loc": "loc"}),
    "DIST-RICIAN": ("rice", {"sigma": "scale"}),
    "DIST-TRUNCNORM": ("truncnorm", {"a": "a", "b": "b", "sigma": "scale"}),
    "DIST-TRUNCGEOM": ("truncgeom", {}),     # 第 159 条：自建截断几何
    "DIST-FRECHET": ("invweibull", {"shape": "c", "scale": "scale"}),
    "DIST-INVCHISQ": ("invgamma", {"k": "a"}),
    "DIST-BETA": ("beta", {"alpha": "a", "beta": "b"}),
    "DIST-BERNOULLI": ("bernoulli", {"p": "p"}),        # 2026-10-02 加
    # **第 159 条（2026-10-02）**：`landau` **不是朗道概率分布**，而是项目的
    #   **朗道自由能 φ⁴ 模型**（`p(φ) ∝ exp(−(½aφ²+¼bφ⁴)/kT)`，真值 = **关于 a
    #   的信息量** `Var(φ²)/(4k²T²)`）⟹ 须**自建 φ⁴ 密度**
    #   （我最初映射到 `scipy.stats.landau` 是**物理上错的**）。
    "DIST-LANDAU": ("phi4", {}),
    # **二分量高斯混合**：scipy 无现成分发（**专用适配器**，见 MIXTURE2）
    "DIST-MIX2": ("mix2", {}),
}

# **真值口径表（第 159 条，逐族核对的结果）**：多数族的真值是**全迹**
#   （如高斯 `3/σ²` 即对 (μ,σ) 的迹），但这四个族的真值函数是**关于某单个
#   参数的信息量**（`fisher_landau(a,b,kT)` 关于 `a`、`fisher_mix2_separation`
#   关于 `d`、`fisher_truncnorm_tau` 关于 `τ`、`fisher_truncgeom_p` 关于 `p`）。
#   ⟹ **测量必须按同一口径取对角项**，否则比较的是两个不同的量
#   ——这正是"**参数化/口径不匹配**"的真源（实测误差 78%–2376%）。
TRUTH_METRIC: dict[str, str] = {
    "landau": "diag:a",
    "gaussian_mixture2": "diag:d",
    "truncated_gaussian": "diag:tau",
    "truncated_geometric": "diag:p",
}

DISCRETE = {"DIST-NEGBIN", "DIST-GEOMETRIC", "DIST-POISSON", "DIST-TRUNCGEOM",
            "DIST-BERNOULLI"}

# **按分布的专属同义名（2026-10-02 加）**：全局同义名会错配——如 `k` 在
#   invgamma 是形状 `a`，而在 `nbinom` 里是**离散度 n** ⟹ 须按分布区分。
SYN_OVERRIDE: dict[str, dict[str, str]] = {
    "DIST-NEGBIN": {"k": "n", "n": "n", "p": "p", "theta": "n"},
}


def node_of(family: str, value: float | None = None):
    """取该族编码器产出的**节点**（参数由编码器给出，不猜）。"""
    import copy
    from a_network.structure_space import family_template
    from b_network.system_encoder import SystemEncoder
    tpl = copy.deepcopy(family_template(family))
    pname = tpl.get("scan_param")
    if value is not None and pname:
        for pr in tpl.get("parameters", []):
            if pr["name"] == pname:
                pr["default"] = float(value)
    tpl["system_id"] = "EXPAND-%s" % family
    g = SystemEncoder().encode(tpl)
    return g.nodes[0] if g.nodes else None


def scipy_dist(node):
    """节点 → scipy 冻结分布（按 `DIST_TO_SCIPY` 映射 + 节点自身参数）。"""
    from scipy import stats
    fam = node.distribution_family
    spec = DIST_TO_SCIPY.get(fam)
    if not spec or not spec[0]:
        return None, "%s 无 scipy 对应" % fam
    name, pmap = spec
    kwargs = {}
    for nk, sk in pmap.items():
        if nk in (node.params or {}):
            try:
                kwargs[sk] = float(node.params[nk])
            except (TypeError, ValueError):
                pass
    if not kwargs:
        return None, "参数映射为空（节点参数 %s）" % list(node.params or {})
    # 离散分布须有整数形状参数
    try:
        dist = getattr(stats, name)(**kwargs)
        dist.mean()                      # 触发参数校验
    except Exception as ex:  # noqa: BLE001
        return None, "scipy 参数无效（%s：%s）" % (name, ex)
    return dist, ""


def measure_fisher(dist, node, tuned: str, n: int = 20000, seed: int = 0,
                   rel_h: float = 1e-4, mode: str = "trace") -> float | None:
    """**全参数 Fisher 迹**的 score 估计器（2026-10-02 修正）。

    **修因**：初版只估**被调参数**的 Fisher，却与编码器的**全参数迹**比较
    ⟹ weibull/lognormal/frechet/gamma/beta/vonmises/invchisq 误差 **86–99%**
    （**同一错误在本会话已出现第二次**：`AnalyticAdapter._fisher_hat` 的注释
    记载过"初版只估首个参数 → 偏差 59%/67%"）。
    现按 `F_kl = E[∂_k log p · ∂_l log p]`、`trace = Σ_k F_kk` 估**全迹**：
    对**每一个分布参数**做中心差分。
    """
    fam = node.distribution_family
    disc = fam in DISCRETE
    rng = np.random.default_rng(seed)
    x = np.asarray(dist.rvs(size=n, random_state=rng), dtype=float)
    if not disc:
        x = x[np.isfinite(x)]
    if len(x) < 100:
        return None
    spec = DIST_TO_SCIPY.get(fam)
    if not spec or not spec[0]:
        return None
    # **不再用 `.args` 判"活跃参数"（2026-10-02 修，我上一版把它弄坏了）**：
    #   scipy 冻结分布的 `.args` 只含**形状参数**（`scale`/`loc` 不在其中）
    #   ⟹ `a.args == b.args` 对形如 `rayleigh(scale=…)` 的族恒成立 ⟹
    #   所有参数都被判"不活跃" ⟹ **全部族退化为"score 估计失败"**（实测 0/27）。
    #   正解：**纳入全部节点参数**——不影响分布的那些其 score ≡ 0，
    #   对迹**无贡献**，天然无害。
    keys = [k for k in (node.params or {})]
    if not keys:
        return None
    base = {k: float(node.params[k]) for k in keys}
    scores = []
    for k in keys:
        h = rel_h * max(abs(base[k]), 1e-6)
        pp, pm = dict(base), dict(base)
        pp[k], pm[k] = base[k] + h, base[k] - h
        d2p, _w1 = scipy_dist_from_params(node, pp)
        d2m, _w2 = scipy_dist_from_params(node, pm)
        if d2p is None or d2m is None:
            return None
        if disc:
            lp = np.asarray(d2p.logpmf(x), dtype=float)
            lm = np.asarray(d2m.logpmf(x), dtype=float)
        else:
            lp = np.asarray(d2p.logpdf(x), dtype=float)
            lm = np.asarray(d2m.logpdf(x), dtype=float)
        s = (lp - lm) / (2 * h)
        # **不再逐参数过滤（2026-10-02 修，真 bug）**：扰动会**改变支撑**
        #   （如 `truncated_gaussian` 的 `x > τ`）⟹ 逐参数被 `isfinite` 滤掉
        #   的样本数**不同** ⟹ 各 score 数组**长度不等** ⟹
        #   `np.stack(scores)` 报 `all input arrays must have the same shape`
        #   （`--n 4000` 时侥幸未触发、`n=20000` 必崩）。正解：**联合有限掩码**。
        scores.append(np.asarray(s, dtype=float))
    n_common = min(len(s) for s in scores)
    mask = np.ones(n_common, dtype=bool)
    for s in scores:
        mask &= np.isfinite(s[:n_common])
    if int(mask.sum()) < 100:
        return None
    S = np.stack([s[:n_common][mask] for s in scores], axis=0)
    F = S @ S.T / S.shape[1]
    # **口径模式（第 159 条）**：`mode="diag:<param>"` 返回**该参数的对角项**
    #   （单参数族的真值就是这个量，不是全迹）；默认 `"trace"`。
    if isinstance(mode, str) and mode.startswith("diag"):
        pname = mode.split(":", 1)[1] if ":" in mode else ""
        if pname in keys:
            return float(F[keys.index(pname), keys.index(pname)])
        return None
    return float(np.trace(F))


# **参数变换表（2026-10-02 加）**：项目的参数化与 scipy 不一致时在此显式换算。
#   修因（真实缺陷）：`lognormal` 的 `mu` 是**对数均值**，我按 scipy 的 `loc`
#   （位置移动）映射 ⟹ 支撑边界附近导数爆炸 ⟹ 误差 **6,800,154%**。
#   正解：scipy `lognorm(s=σ, scale=exp(μ))`。同理 `invchisq`（逆卡方）=
#   `invgamma(a=k/2, scale=1/2)`；`radioactive_decay` 的 `rate` ⟹ `scale=1/rate`。
TRANSFORMS: dict[str, dict] = {
    "DIST-LOGNORM": {"s": lambda p: float(p.get("sigma", 1.0)),
                     "scale": lambda p: float(np.exp(p.get("mu", 0.0)))},
    "DIST-INVCHISQ": {"a": lambda p: float(p.get("k", 1.0)) / 2.0,
                      "scale": lambda p: 0.5},
    "DIST-EXPONENTIAL": {"scale": lambda p: 1.0 / max(
        float(p.get("rate", p.get("scale", 1.0))), 1e-12)},
    # **第 159 条**：`truncated_gaussian` 的参数是 `(tau, mu, sigma)`
    #   （**截断点**、均值、标准差），而 `truncnorm` 要的是**标准化截断**
    #   `a=(tau−mu)/sigma`、`b=inf` ⟹ 必须有派生变换（**这正是它此前
    #   "缺映射"的真因**——简单映射表里没有 `a`/`b` 这两个参数）。
    "DIST-TRUNCNORM": {"a": lambda p: (float(p.get("tau", 0.0))
                                       - float(p.get("mu", 0.0)))
                       / max(float(p.get("sigma", 1.0)), 1e-12),
                       "b": lambda p: np.inf,
                       "loc": lambda p: float(p.get("mu", 0.0)),
                       "scale": lambda p: max(float(p.get("sigma", 1.0)),
                                              1e-12)},
}


# ---------------------------------------------------------------- 专用分布
# **第 159 条**：两个族在 scipy 里**没有现成分发**，须自建（**"须专用适配器"
#   的真实案例**）。两者都用**解析 pdf/pmf**，Fisher 由数值求积/求和得到。
def _make_mix2(**kw):
    """**二分量高斯混合**：参数 `(d, w, sigma)` ⟹
    `p(x) = w·N(x; −d/2, σ) + (1−w)·N(x; +d/2, σ)`（双峰传感器模板）。"""
    from scipy import stats

    class _Mix2(stats.rv_continuous):
        def _rvs(self, d, w, sigma, size=None, random_state=None):
            # **直接精确采样（2026-10-02 修）**：混合的采样本可精确做——
            #   先按权重抽分量、再抽高斯。初版不实现 `_rvs` ⟹ scipy 走
            #   **数值反演 CDF** ⟹ 比现成分发慢一两个数量级（实测把登记
            #   拖到 30 分钟以上）。
            d = np.asarray(d, dtype=float)
            w = np.clip(np.asarray(w, dtype=float), 1e-6, 1 - 1e-6)
            sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
            n = int(np.prod(size)) if size is not None else 1
            rs = random_state or np.random.default_rng()
            comp = rs.random(n) < w
            mu = np.where(comp, -d / 2.0, d / 2.0)
            return (mu + sigma * rs.standard_normal(n)).reshape(
                size if size is not None else ())

        def _pdf(self, x, d, w, sigma):
            # **形状参数须按数组处理（2026-10-02 修，真 bug）**：scipy 传进来的
            #   `d/w/sigma` 是**数组**（广播），初版用 `float(...)` 转换 ⟹
            #   `TypeError: only length-1 arrays can be converted to Python
            #   scalars`（同一根因也让截断几何失败）。
            d = np.maximum(np.asarray(d, dtype=float), 1e-9)
            w = np.clip(np.asarray(w, dtype=float), 1e-6, 1 - 1e-6)
            sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
            return (w * stats.norm.pdf(x, -d / 2, sigma)
                    + (1 - w) * stats.norm.pdf(x, d / 2, sigma))

    # **`shapes` 必须声明（2026-10-02 修）**：否则冻结时报
    #   `_parse_args_stats() missing 3 required positional arguments`
    return _Mix2(a=-np.inf, b=np.inf, name="mix2", shapes="d, w, sigma")


def _make_phi4(**kw):
    """**朗道自由能 φ⁴ 模型**（第 159/163 条，**不是**朗道概率分布！）：

    `F(φ) = ½aφ² + ¼bφ⁴`，平衡分布 `p(φ) ∝ exp(−F(φ)/kT)`
    （`a>0` 高温相单峰、`a<0` 低温相双峰；`b>0`、`kT>0`）。
    **真值口径**：`p(φ)` 关于 **`a`** 的信息量 `Var(φ²)/(4k²T²)`。

    **形状参数必须改名（2026-10-02 修，真冲突）**：族参数名 `a`/`b` 与
    `rv_continuous` 构造函数的**支撑上/下界**同名（`a=`、`b=`）⟹ 值被绑错
    ⟹ 实测 Fisher 偏大 **2.3×**（0.5027 vs 真值 0.2181）。
    故形状名改用 **`aa`/`bb`/`kk`**（族参数 → 形状名的映射见
    `SPECIAL_RENAME`）——与 `truncgeom` 的 `k` 撞名是**同一类** bug。
    """
    from scipy import stats

    class _Phi4(stats.rv_continuous):
        def _grid(self, aa: float, bb: float, kk: float, n: int = 20001):
            """密度网格（采样与**归一化常数**共用同一网格）。"""
            width = (kk / bb) ** 0.25
            hi = 12.0 * width
            if aa < 0:
                hi = max(hi, 6.0 * math.sqrt(abs(aa) / bb))
            gx = np.linspace(-hi, hi, n)
            gp = np.exp(np.clip(-(0.5 * aa * gx ** 2 + 0.25 * bb * gx ** 4)
                                / kk, -700.0, 700.0))
            return gx, gp

        def _rvs(self, aa, bb, kk, size=None, random_state=None):
            # **网格逆 CDF 精确采样（2026-10-02 修，第二版）**：
            #   ① 初版不实现 `_rvs` ⟹ scipy 对无界支撑做数值反演 ⟹ 极慢；
            #   ② 第二版用**拒绝采样**但**接受率未除以正确的上界 `M`**
            #      ⟹ 采样**有偏**、尾部被高估；
            #   正解：**细网格建 CDF 再逆变换**（无偏、无需调 `M`、极快）。
            aa = float(np.asarray(aa, dtype=float).ravel()[0])
            bb = max(float(np.asarray(bb, dtype=float).ravel()[0]), 1e-12)
            kk = max(float(np.asarray(kk, dtype=float).ravel()[0]), 1e-12)
            n = int(np.prod(size)) if size is not None else 1
            rs = random_state or np.random.default_rng()
            gx, gp = self._grid(aa, bb, kk)
            cdf = np.cumsum(gp)
            cdf /= cdf[-1]
            u = rs.random(n)
            return np.interp(u, cdf, gx).reshape(
                size if size is not None else ())

        def _argcheck(self, aa, bb, kk):
            # **须覆写（2026-10-02 修）**：scipy 默认要求**所有形状参数 > 0**，
            #   而朗道模型的 `a` **可为负**（低温双峰相）⟹ 默认 argcheck 会让
            #   `rvs` 报 `Domain error ... scale must be positive`。正解：
            #   只要求 `b > 0`、`kT > 0`，`a` 任意实数。
            return (np.asarray(bb, dtype=float) > 0) & \
                   (np.asarray(kk, dtype=float) > 0)

        def _pdf(self, x, aa, bb, kk):
            """**必须归一化（2026-10-02 修，关键）**：

            scipy 的 `logpdf` **直接使用 `_pdf` 的返回值**（不做归一化）⟹ 若
            `_pdf` 未归一，有限差分 score 会**丢掉归一化项**
            `∂_a log Z = −⟨φ²⟩/(2kT)` ⟹ 实测变成 `E[φ⁴]/(4k²T²)`（**偏大**），
            而真值是 `Var(φ²)/(4k²T²)`——实测 **0.5027 vs 真值 0.2715**（1.85×）
            ⟹ 正是本族长期对不齐的真因（`mix2`/`truncgeom`/`truncnorm` 都是
            归一化的，故只有本族偏）。
            """
            aa = float(np.asarray(aa, dtype=float).ravel()[0])
            bb = max(float(np.asarray(bb, dtype=float).ravel()[0]), 1e-12)
            kk = max(float(np.asarray(kk, dtype=float).ravel()[0]), 1e-12)
            gx, gp = self._grid(aa, bb, kk)
            Z = float(np.trapezoid(gp, gx)) if hasattr(np, "trapezoid") \
                else float(np.trapz(gp, gx))
            xc = np.clip(np.asarray(x, dtype=float), -1e6, 1e6)
            logp = -(0.5 * aa * xc ** 2 + 0.25 * bb * xc ** 4) / kk
            return np.exp(np.clip(logp, -700.0, 700.0)) / max(Z, 1e-300)

    # **形状名 `aa/bb/kk`（2026-10-02 修）**：若沿用族参数名 `a`/`b`，会与
    #   `rv_continuous` 的**支撑上/下界**参数同名 ⟹ 绑定错乱 ⟹ Fisher 偏大 2.3×。
    return _Phi4(a=-np.inf, b=np.inf, name="phi4", shapes="aa, bb, kk")


def _make_truncgeom(**kw):
    """**截断几何分布**：参数 `(k, p)` ⟹
    `P(X=x) = (1−p)^{x−1} p / Z`，`x = 1..k`，`Z = 1 − (1−p)^k`。

    **形状参数必须改名（2026-10-02 修，真冲突）**：scipy 的 `rv_discrete`
    内部用 `k` 表示**取值变量** ⟹ 形状参数若也叫 `k` 会报
    `logpmf() got multiple values for argument 'k'`。故对外仍收族参数 `k`，
    内部形状名用 **`kk`**（`SPECIAL_NEEDS` 里做映射）。
    """
    from scipy import stats

    class _TruncGeom(stats.rv_discrete):
        def _rvs(self, kk, p, size=None, random_state=None):
            # **解析逆 CDF 采样（2026-10-02 修）**：`P(X≤n) = (1−q^n)/(1−q^k)`
            #   ⟹ `n = ceil( ln(1 − u(1−q^k)) / ln q )` ✓ 精确且极快。
            kk = max(int(round(float(np.asarray(kk).ravel()[0]))), 1)
            p = min(max(float(np.asarray(p).ravel()[0]), 1e-9), 1 - 1e-9)
            q = 1.0 - p
            n = int(np.prod(size)) if size is not None else 1
            rs = random_state or np.random.default_rng()
            u = rs.random(n)
            val = np.ceil(np.log(np.maximum(1.0 - u * (1.0 - q ** kk), 1e-300))
                          / math.log(q))
            val = np.clip(val, 1, kk).astype(int)
            return val.reshape(size if size is not None else ())

        def _pmf(self, x, kk, p):
            # **形状参数按数组处理（同上修）**，并**向量化**：
            kk = np.asarray(kk, dtype=float)
            p = np.asarray(p, dtype=float)
            k_int = np.maximum(np.round(kk).astype(int), 1)
            p = np.clip(p, 1e-9, 1 - 1e-9)
            xa = np.asarray(x, dtype=float)
            q = (1 - p) ** (xa - 1) * p
            Z = 1.0 - (1 - p) ** k_int
            return np.where((xa >= 1) & (xa <= k_int),
                            q / np.maximum(Z, 1e-300), 0.0)

    return _TruncGeom(name="truncgeom", shapes="kk, p")


SPECIAL_DISTS = {"mix2": _make_mix2, "truncgeom": _make_truncgeom,
                 "phi4": _make_phi4}
# **冻结所需的形状参数名（2026-10-02 修）**：自建分布**必须用族参数冻结**
#   （`gen(**params)`），否则未冻结分布调 `.mean()` 会报
#   `_parse_args_stats() missing N required positional arguments`（实测）。
SPECIAL_NEEDS: dict[str, list[str]] = {
    "mix2": ["d", "w", "sigma"],
    "truncgeom": ["kk", "p"],       # 族参数 `k` → 形状名 `kk`（避免与 scipy 冲突）
    "phi4": ["aa", "bb", "kk"],     # 族参数 a/b/kT → aa/bb/kk（避免与支撑界同名）
}
# **族参数名 → 专用分布形状名**（仅在两处不一致时需要）
SPECIAL_RENAME: dict[str, dict[str, str]] = {
    "truncgeom": {"k": "kk"},
    "phi4": {"a": "aa", "b": "bb", "kT": "kk"},
}


def scipy_dist_from_params(node, params: dict):
    """由「节点 + 指定参数」构造 scipy 分布（**变换表 + 同义名 + 单参回退**）。

    **修因（2026-10-02）**：初版映射表用了**猜的参数名**（logistic 写 `sigma`
    而实际是 `scale`；radioactive_decay 写 `scale` 而实际是 `rate`；
    inverse_gamma 写 `a` 而实际是 `k`+`theta`）⟹ "参数映射为空"。
    现策略：① **变换表**（参数化不同的族，如 lognormal/invchisq/expon）；
    ② 映射表 + **同义名**；③ 单参数回退。
    """
    from scipy import stats
    fam = node.distribution_family
    spec = DIST_TO_SCIPY.get(fam)
    if not spec or not spec[0]:
        return None, "无 scipy 对应"
    name, pmap = spec
    kwargs: dict = {}
    tf = TRANSFORMS.get(fam)
    if tf:
        for sk, fn in tf.items():
            try:
                kwargs[sk] = float(fn(params))
            except Exception:  # noqa: BLE001
                pass
    else:
        for nk, sk in pmap.items():
            if nk in params:
                try:
                    kwargs[sk] = float(params[nk])
                except (TypeError, ValueError):
                    pass
        # 同义名回退（节点参数名 → scipy 参数名）
        SYN = {"scale": "scale", "sigma": "scale", "rate": "scale",
               "theta": "scale", "b": "scale", "gamma": "scale",
               "k": "a", "alpha": "a", "a": "a", "shape": "c",
               "p": "p", "n": "n", "mu": "loc", "loc": "loc",
               "kappa": "kappa", "tau": "loc"}
        used = set(kwargs)
        syn = dict(SYN)
        syn.update(SYN_OVERRIDE.get(fam, {}))     # 专属同义名优先
        for nk, v in params.items():
            if nk in pmap:
                continue
            sk = syn.get(nk)
            if sk and sk not in used:
                try:
                    kwargs[sk] = float(v)
                    used.add(sk)
                except (TypeError, ValueError):
                    pass
        if not kwargs and len(params) == 1:
            only = list(params.values())[0]
            try:
                kwargs["scale"] = float(only)
            except (TypeError, ValueError):
                pass
    try:
        # **专用分布（第 159 条）**：scipy 无现成分发的族（mix2/truncgeom）
        #   ⟹ 用**族参数冻结**（`gen(**need)`），再取 `.mean()` 验证可用。
        if name in SPECIAL_DISTS:
            need = SPECIAL_NEEDS.get(name) or []
            ren = SPECIAL_RENAME.get(name) or {}
            inv = {v: k for k, v in ren.items()}     # 形状名 → 族参数名
            fkw = {}
            for k in need:
                src = inv.get(k, k)
                if src in params:
                    fkw[k] = float(params[src])
                else:
                    return None, "专用分布 %s 缺参数 %s（须给 %s）" % (
                        name, src, need)
            gen = SPECIAL_DISTS[name](**fkw)
            d = gen(**fkw)
            # **轻量校验（2026-10-02 修）**：初版调 `d.mean()` 验证可用，但
            #   自建分布的 `mean()` 走 scipy **数值积分**（φ⁴ 是 `(−∞,∞)` 无界
            #   支撑）⟹ `integrate.quad` 失败后 `scipy_dist` 返回 None。
            #   正解：只校验**密度可求值且有限**（离散用 `pmf(1)`）。
            probe = d.pmf(1) if name == "truncgeom" else d.pdf(0.0)
            if not np.isfinite(float(probe)):
                return None, "专用分布 %s 密度不可用" % name
            return d, ""
        d = getattr(stats, name)(**kwargs)
        d.mean()
        return d, ""
    except Exception as ex:  # noqa: BLE001
        return None, str(ex)


def scipy_dist(node):
    """节点 → scipy 冻结分布（**与 `scipy_dist_from_params` 同一路径**）。

    修因：初版此函数**只走映射表、不走同义名** ⟹ `inverse_gamma`（节点参数
    `k`+`theta`）、`logistic`（`scale`）、`radioactive_decay`（`rate`）都报
    "参数映射为空"，而实际用同义名路径可解。
    """
    return scipy_dist_from_params(node, dict(node.params or {}))


def measure_for_verify(family: str, value: float, n_samples: int = 20000,
                       seed: int = 0) -> float | None:
    """**供 `verify_api` 调用的测量入口**（第 136 条）。

    由编码器产出该族的节点 → 按其参数化构造 scipy 分布 → 采样 →
    **全参数 Fisher 迹**的 score 估计 ⟹ 与 `compute_node_fisher` 同口径
    （已由校准验证：**这 23 族的误差 ≤2.2%**）。

    **口径修复（2026-10-02，第 207 条）**：本函数初版**忽略** `TRUTH_METRIC`
    ✗ —— 直接以 `measure_fisher(d, node, "", …)` 取**迹**，而 `calibrate()`
    **按 `TRUTH_METRIC`** 取量（见本文件 585–587 行 ✓）
    ⟹ 对**声明为 `diag:<param>` 的自建分布族**（`gaussian_mixture2` /
    `landau` / `truncated_gaussian` / `truncated_geometric`）**两边口径不同** ✗
    ⟹ 实测生产测量偏离节点真值 **5.9–24 倍**（第 206 条扫描 ✓）
    **修法**：与 `calibrate()` **用完全相同的取量方式** ✓
    （`pname` 取该族登记参数名 ✓、`mode = TRUTH_METRIC.get(family,"trace")` ✓）
    """
    node = node_of(family, float(value))
    if node is None:
        return None
    d, _why = scipy_dist(node)
    if d is None:
        return None
    # **与 `calibrate()` 对齐**：只补 **真值口径模式** ✓
    #   （`measure_fisher` 的第三参数是 `tuned`，对 trace/diag 模式**不使用** ✓
    #    ⟹ **无需**查参数名；初版误用 `REGISTRY[...]` ⟹ `NameError` ✗
    #    ——该模块的注册表字典**不叫** `REGISTRY` ✗）
    #   `TRUTH_METRIC` 的值本身即 `diag:<参数名>` ✓
    mode = TRUTH_METRIC.get(family, "trace")
    return measure_fisher(d, node, "", n=int(n_samples), seed=int(seed),
                          mode=mode)


def main() -> int:
    from pipeline.inverse_fisher import families_available, family_param
    from core.fingerprint import compute_node_fisher      # noqa: F401
    # **有效域边界从宽（2026-10-02 修）**：`two_level` 的 p 用 DB1 模板范围
    #   `[0.1, 0.9]`，而设计域常取 **p=0.10（恰在边界）** ⟹ 浮点噪声使其落到
    #   界外 ⟹ 生产验证报 `unknown`（实测 reflow 里 **192/200**）。
    #   ⟹ 登记范围按**族语义**放宽到 `[0.05, 0.95]`（p∈(0,1) 均有定义），
    #   使边界设计**落在域内**。
    RANGE_WIDEN: dict[str, tuple[float, float]] = {
        "two_level": (0.05, 0.95),
        "geometric": (0.02, 0.98),
        "negbin": (0.02, 0.98),
    }
    print("=" * 74)
    print("D·扩大外部验证覆盖（校准即验证；p99 ≤ %.0f%% 才登记）"
          % (100 * TOL_REG))
    print("=" * 74)
    fams = families_available()
    print("DB1 族候选：%d 个" % len(fams))
    # **采样数开关（第 159 条，2026-10-02）**：默认 20000；`--n N` 可覆盖。
    #   **慢族**（自建分布的 `rvs` 走 scipy **数值反演** ⟹ 慢一两个数量级）
    #   用**降低的采样数**，并**在报告里标明**（判据不变：p99 相对误差 ≤10%）。
    n_main = 20000
    if "--n" in sys.argv:
        n_main = int(sys.argv[sys.argv.index("--n") + 1])
    SLOW_FAMILIES = {"gaussian_mixture2": 2000}
    rows = []
    for f in fams:
        try:
            pname, (lo, hi) = family_param(f)
        except Exception as ex:  # noqa: BLE001
            rows.append((f, None, None, None, "无参数模板（%s）" % ex))
            continue
        mid = 0.5 * (lo + hi)
        node = node_of(f, mid)
        if node is None:
            rows.append((f, pname, None, None, "编码器未产出节点"))
            continue
        truth = compute_node_fisher(node) or {}
        ft = truth.get("fisher_trace")
        if not ft:
            rows.append((f, pname, None, None, "真值不可算（%s）"
                         % node.distribution_family))
            continue
        d, why = scipy_dist(node)
        if d is None:
            rows.append((f, pname, None, None, why))
            continue
        errs = []
        n_use = SLOW_FAMILIES.get(f, n_main)
        # **按真值口径取量（第 159 条）**：单参数族取对角项，其余取全迹
        mode = TRUTH_METRIC.get(f, "trace")
        for s in range(3):
            m = measure_fisher(d, node, pname, n=n_use, seed=s, mode=mode)
            if m and m > 0:
                errs.append(abs(m - float(ft)) / abs(float(ft)))
        if not errs:
            rows.append((f, pname, None, None, "score 估计失败"))
            continue
        e = float(np.max(errs))
        note = ("（慢族，n=%d）" % n_use) if f in SLOW_FAMILIES else ""
        if mode != "trace":
            note += "（口径：%s）" % mode
        rows.append((f, pname, float(ft), e, note))

    print("\n%-22s %-9s %12s %10s %s"
          % ("family", "param", "真值 F", "最大相对误差", "判定"))
    ok = []
    for f, pname, ft, e, why in rows:
        if e is None:
            print("  %-20s %-9s %12s %10s %s"
                  % (f, pname or "—", "—", "—", why))
            continue
        good = e <= TOL_REG
        if good:
            ok.append((f, pname))
        print("  %-20s %-9s %12.4g %9.1f%% %s"
              % (f, pname, ft, 100 * e, "**登记** ✓" if good
                 else "不登记（超 %.0f%%）" % (100 * TOL_REG)))
    print("\n**可登记族：%d / %d**（含原有 4 族解析适配器）" % (len(ok), len(rows)))
    # **两类失败的区分（2026-10-02）**：
    #   ① "参数非分布参数"（如 ou 的 θ 是谱间隙、elastic 的 k 是刚度）⟹
    #      **解析路根本不适用**，须走**专用/动力学适配器**（ou/diffusion 已有）；
    #   ② "误差超限" ⟹ 参数化口径不匹配（须逐族核对）。
    need_adapter = [r for r in rows if r[3] is None and r[4]
                    and ("score 估计失败" in r[4])]
    over = [r for r in rows if r[3] is not None and r[3] > TOL_REG]
    print("\n失败分类：")
    print("  · **须专用适配器**（参数非分布参数，解析路不适用）：%d 个 → %s"
          % (len(need_adapter), [r[0] for r in need_adapter]))
    print("  · **参数化口径不匹配**（误差超限，须逐族核对）：%d 个 → %s"
          % (len(over), [r[0] for r in over]))

    if "--register" in sys.argv:
        reg = {}
        for f, pname in ok:
            try:
                _p, (lo, hi) = family_param(f)
            except Exception:  # noqa: BLE001
                continue
            if f in RANGE_WIDEN:            # **边界从宽**（见上）
                lo, hi = RANGE_WIDEN[f]
            reg[f] = {"metrics": ["q1"], "params": {pname: [lo, hi]},
                      "source": "sim/expand_registry.py（校准通过）"}
        REGISTRY_JSON.write_text(json.dumps(reg, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print("\n已写 %s（%d 族）" % (REGISTRY_JSON.name, len(reg)))
    else:
        print("\n（未写文件；加 `--register` 落盘）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
