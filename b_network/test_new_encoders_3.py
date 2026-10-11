"""新编码器（扩族批次 3：卡方/负二项/对数正态）自测。

验证：
1. 卡方 Fisher 解析 ψ₁(k/2)/4 与 score-MC 数值对照，及 Gamma(k/2,2)
   换元一致性（fisher_chisq(k) = gamma 矩阵 g_αα/4）。
2. 负二项 Fisher 解析 k/(p²(1-p)) 与 score-MC 对照，及 k=1 严格退化
   为几何 Geo(p)。
3. 对数正态 Fisher 与高斯同形（双射不变性：trace=3/σ², det=2/σ⁴）
   及 score-MC 对照。
4. 三族编码器全链路：entry → EncodedGraph → FingerprintComputer 指纹。

运行：python b_network/test_new_encoders_3.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.special import gammaln  # noqa: E402

from core.fisher import (  # noqa: E402
    fisher_chisq,
    fisher_det_gaussian,
    fisher_gamma_matrix,
    fisher_geometric,
    fisher_negbin,
    fisher_lognormal_det,
    fisher_lognormal_trace,
    fisher_numerical,
    fisher_trace_gaussian,
)
from b_network.system_encoder import SystemEncoder  # noqa: E402
from b_network.fingerprint_computer import FingerprintComputer  # noqa: E402


def _log_pdf_chisq(x, theta):
    """χ²(k) 对数密度（k = theta[0]）。"""
    k = float(theta[0])
    if x <= 0:
        return -np.inf
    return (k / 2.0 - 1.0) * math.log(x) - x / 2.0 - (k / 2.0) * math.log(2.0) \
        - gammaln(k / 2.0)


def _sampler_chisq(n, theta, rng):
    return rng.chisquare(df=float(theta[0]), size=n)


def _log_pdf_negbin(n_trials, theta):
    """NB(k, p) 对数 pmf（试验计数参数化：n = 第 k 次成功所在试验序号）。

    log C(n-1, k-1) = gammaln(n) - gammaln(k) - gammaln(n-k+1)。
    注意末项是 n-k+1：若误写 gammaln(n-k)，零失败样本（n=k）会触发
    gammaln(0)=+inf，score-MC 差分两侧同为 -inf → nan。
    """
    k, p = float(theta[0]), float(theta[1])
    if n_trials < k:
        return -np.inf
    return (gammaln(n_trials) - gammaln(k)
            - gammaln(n_trials - k + 1.0) + k * math.log(p)
            + (n_trials - k) * math.log1p(-p))


def _sampler_negbin(n, theta, rng):
    """numpy 负二项计失败数 x，移位到试验计数 n = x + k（Fisher 双射不变）。"""
    k = float(theta[0])
    x = rng.negative_binomial(k, float(theta[1]), size=n)
    return x + k


def _log_pdf_lognormal(x, theta):
    """LN(mu, sigma) 对数密度（theta = [mu, sigma]）。"""
    mu, sigma = float(theta[0]), float(theta[1])
    y = math.log(x)
    return (-y - math.log(sigma) - 0.5 * math.log(2.0 * np.pi)
            - (y - mu) ** 2 / (2.0 * sigma**2))


def _sampler_lognormal(n, theta, rng):
    return np.exp(rng.normal(float(theta[0]), float(theta[1]), size=n))


def main():
    lines = []
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    enc = SystemEncoder()
    fpc = FingerprintComputer()
    rng = np.random.default_rng(0)

    # --- 1. 卡方：解析 vs Gamma 换元 vs score-MC ---
    # 闭式：ψ₁(k/2)/4
    from scipy.special import polygamma
    for k in (2.0, 4.0, 8.0):
        exp_val = float(polygamma(1, k / 2.0)) / 4.0
        lines.append(("PASS" if abs(fisher_chisq(k) - exp_val) < 1e-12 else "FAIL")
                     + f"  卡方 I({k}) = ψ₁(k/2)/4：{fisher_chisq(k):.6f} vs {exp_val:.6f}")
    # 换元一致性：χ²(k) = Gamma(k/2, 2)，I(k) = g_αα(α=k/2)·(1/2)²
    k = 6.0
    g_aa = fisher_gamma_matrix(3.0, 2.0)[0, 0]
    lines.append(("PASS" if abs(fisher_chisq(k) - g_aa / 4.0) < 1e-12 else "FAIL")
                 + f"  卡方换元：fisher_chisq(6) = Gamma(3,2) g_αα/4："
                   f"{fisher_chisq(k):.6f} vs {g_aa / 4.0:.6f}")
    # score-MC（k=4：良态）
    g_mc = fisher_numerical(_log_pdf_chisq, np.array([4.0]),
                            _sampler_chisq, n_samples=20000, rng=rng)
    rel = abs(g_mc[0, 0] - fisher_chisq(4.0)) / fisher_chisq(4.0)
    lines.append(("PASS" if rel < 0.05 else "FAIL")
                 + f"  卡方 score-MC：{g_mc[0, 0]:.4f} vs {fisher_chisq(4.0):.4f} "
                   f"(rel={rel:.3%})")

    # --- 2. 负二项：k=1 退化几何 + score-MC ---
    for p in (0.1, 0.25, 0.4):
        lines.append(
            ("PASS" if abs(fisher_negbin(1.0, p) - fisher_geometric(p)) < 1e-12
             else "FAIL")
            + f"  负二项 k=1 退化几何：{fisher_negbin(1.0, p):.6f} vs "
              f"{fisher_geometric(p):.6f}")
    # 手推交叉点：I(2, 0.2) = 2/(0.04×0.8) = 62.5
    lines.append(("PASS" if abs(fisher_negbin(2.0, 0.2) - 62.5) < 1e-9 else "FAIL")
                 + f"  负二项闭式 k=2, p=0.2：{fisher_negbin(2.0, 0.2):.6f} vs 62.5")
    g_mc = fisher_numerical(_log_pdf_negbin, np.array([2.0, 0.2]),
                            _sampler_negbin, n_samples=20000, rng=rng)
    # theta=[k, p]，取 p 分量 [1,1]
    exp_val = fisher_negbin(2.0, 0.2)
    rel = abs(g_mc[1, 1] - exp_val) / exp_val
    lines.append(("PASS" if rel < 0.05 else "FAIL")
                 + f"  负二项 score-MC（p 分量）：{g_mc[1, 1]:.3f} vs {exp_val:.3f} "
                   f"(rel={rel:.3%})")

    # --- 3. 对数正态：与高斯同形 + score-MC ---
    for sigma in (0.5, 1.0, 2.0):
        lines.append(
            ("PASS" if abs(fisher_lognormal_trace(sigma)
                           - fisher_trace_gaussian(sigma)) < 1e-12 else "FAIL")
            + f"  对数正态 trace=3/σ² 同形（σ={sigma}）："
              f"{fisher_lognormal_trace(sigma):.6f}")
        lines.append(
            ("PASS" if abs(fisher_lognormal_det(sigma)
                           - fisher_det_gaussian(sigma)) < 1e-12 else "FAIL")
            + f"  对数正态 det=2/σ⁴ 同形（σ={sigma}）："
              f"{fisher_lognormal_det(sigma):.6f}")
    g_mc = fisher_numerical(_log_pdf_lognormal, np.array([0.5, 1.0]),
                            _sampler_lognormal, n_samples=20000, rng=rng)
    rel_mu = abs(g_mc[0, 0] - 1.0) / 1.0
    rel_sg = abs(g_mc[1, 1] - 2.0) / 2.0
    lines.append(("PASS" if rel_mu < 0.05 and rel_sg < 0.05 else "FAIL")
                 + f"  对数正态 score-MC：g_μμ={g_mc[0, 0]:.4f} vs 1.0 "
                   f"(rel={rel_mu:.3%}), g_σσ={g_mc[1, 1]:.4f} vs 2.0 "
                   f"(rel={rel_sg:.3%})")

    # --- 4. 编码器全链路 ---
    e_chi = {"family": "chisq", "channel_template": "CT-CHISQ-001",
             "parameters": [{"name": "k", "default": 4.0}]}
    g_chi = enc.encode(e_chi)
    node = g_chi.nodes[0]
    lines.append(("PASS" if node.distribution_family == "DIST-CHISQ" else "FAIL")
                 + "  卡方编码 → DIST-CHISQ")
    fp = fpc.compute(e_chi, g_chi)["fingerprints"]["chisq__T__0"]
    exp_val = fisher_chisq(4.0)
    lines.append(("PASS" if abs(fp["fisher_trace"] - exp_val) < 1e-9
                  and abs(fp["fisher_det"] - exp_val) < 1e-9 else "FAIL")
                 + f"  卡方指纹：trace=det={fp['fisher_trace']:.6f} vs {exp_val:.6f}")

    e_nb = {"family": "negbin", "channel_template": "CT-NEGBIN-001",
            "parameters": [{"name": "k", "default": 5.0},
                            {"name": "p", "default": 0.1}]}
    g_nb = enc.encode(e_nb)
    node = g_nb.nodes[0]
    lines.append(("PASS" if node.distribution_family == "DIST-NEGBIN" else "FAIL")
                 + "  负二项编码 → DIST-NEGBIN")
    lines.append(("PASS" if abs(node.params["k"] - 5.0) < 1e-12
                  and abs(node.params["p"] - 0.1) < 1e-12 else "FAIL")
                 + f"  负二项参数：k={node.params['k']}, p={node.params['p']}")
    fp = fpc.compute(e_nb, g_nb)["fingerprints"]["negbin__T__0"]
    exp_val = fisher_negbin(5.0, 0.1)
    lines.append(("PASS" if abs(fp["fisher_trace"] - exp_val) < 1e-9 else "FAIL")
                 + f"  负二项指纹：trace={fp['fisher_trace']:.4f} vs {exp_val:.4f}")

    e_ln = {"family": "lognormal", "channel_template": "CT-LOGNORM-001",
            "parameters": [{"name": "mu", "default": 0.5},
                            {"name": "sigma", "default": 0.8}]}
    g_ln = enc.encode(e_ln)
    node = g_ln.nodes[0]
    lines.append(("PASS" if node.distribution_family == "DIST-LOGNORM" else "FAIL")
                 + "  对数正态编码 → DIST-LOGNORM")
    lines.append(("PASS" if node.noise_type == "multiplicative" else "FAIL")
                 + "  对数正态乘性噪声标注")
    fp = fpc.compute(e_ln, g_ln)["fingerprints"]["lognormal__T__0"]
    exp_tr = fisher_lognormal_trace(0.8)
    exp_dt = fisher_lognormal_det(0.8)
    lines.append(("PASS" if abs(fp["fisher_trace"] - exp_tr) < 1e-9
                  and abs(fp["fisher_det"] - exp_dt) < 1e-9 else "FAIL")
                 + f"  对数正态指纹：trace={fp['fisher_trace']:.6f} vs {exp_tr:.6f}, "
                   f"det={fp['fisher_det']:.6f} vs {exp_dt:.6f}")

    # 汇总
    print("\n".join(lines))
    n_fail = sum(1 for l in lines if l.startswith("FAIL"))
    print(f"\n==== {len(lines) - n_fail}/{len(lines)} PASS ====")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
