"""扩族批次 1（gamma/beta/weibull/vonmises）的 MC score 独立验证。

三重验证的第三腿（独立于管线）：
    解析闭式（fisher.py）  ←→  编码链数值（fingerprint.py 路由）
                          ←→  score 蒙特卡洛（本文件，fisher_numerical）

原理：g_ij = E[(∂_i ln p)(∂_j ln p)]，用中心差分求 score、MC 平均。
MC 采样误差 ~2-5%（20000 样本），trace 容差 0.08、det 容差 0.15
（det 是两个矩阵元乘积，相对误差放大，容差放宽）。

运行：python b_network/test_new_families.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.fisher import (  # noqa: E402
    fisher_det_beta,
    fisher_det_gamma,
    fisher_det_vonmises,
    fisher_det_weibull,
    fisher_numerical,
    fisher_trace_beta,
    fisher_trace_gamma,
    fisher_trace_vonmises,
    fisher_trace_weibull,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# 每族 3 个参数点（对应 3 个系统的 default）
CASES = [
    ("gamma", [(3.0, 0.5), (2.0, 2.0), (0.9, 2.0)]),
    ("beta", [(2.0, 3.0), (5.0, 1.5), (1.0, 1.0)]),
    ("weibull", [(1.5, 2.0), (2.5, 1.0), (0.8, 3.0)]),
    ("vonmises", [(0.0, 4.0), (0.0, 1.5), (0.0, 8.0)]),
]


def _log_pdf_gamma(x, th):
    from scipy.special import gammaln

    k, th_scale = th
    return (k - 1.0) * np.log(x) - x / th_scale - gammaln(k) - k * np.log(th_scale)


def _sampler_gamma(n, th, rng):
    return rng.gamma(th[0], th[1], size=n)


def _log_pdf_beta(x, th):
    from scipy.special import betaln

    a, b = th
    return (a - 1.0) * np.log(x) + (b - 1.0) * np.log1p(-x) - betaln(a, b)


def _sampler_beta(n, th, rng):
    return rng.beta(th[0], th[1], size=n)


def _log_pdf_weibull(x, th):
    k, lam = th
    return np.log(k) - np.log(lam) + (k - 1.0) * np.log(x / lam) - (x / lam) ** k


def _sampler_weibull(n, th, rng):
    return rng.weibull(th[0], size=n) * th[1]


def _log_pdf_vonmises(x, th):
    from scipy.special import i0

    mu, kap = th
    return kap * np.cos(x - mu) - np.log(2.0 * np.pi * i0(kap))


def _sampler_vonmises(n, th, rng):
    return rng.vonmises(th[0], th[1], size=n)


_SETUPS = {
    "gamma": (_log_pdf_gamma, _sampler_gamma,
              lambda th: fisher_trace_gamma(*th), lambda th: fisher_det_gamma(*th)),
    "beta": (_log_pdf_beta, _sampler_beta,
             lambda th: fisher_trace_beta(*th), lambda th: fisher_det_beta(*th)),
    "weibull": (_log_pdf_weibull, _sampler_weibull,
                lambda th: fisher_trace_weibull(*th), lambda th: fisher_det_weibull(*th)),
    "vonmises": (_log_pdf_vonmises, _sampler_vonmises,
                 lambda th: fisher_trace_vonmises(th[1]), lambda th: fisher_det_vonmises(th[1])),
}

_TRACE_TOL = 0.08
_DET_TOL = 0.15


def main() -> int:
    n_fail = 0
    for family, cases in CASES:
        log_pdf, sampler, trace_fn, det_fn = _SETUPS[family]
        for params in cases:
            theta = np.asarray(params, dtype=float)
            g_mc = fisher_numerical(log_pdf, theta, sampler,
                                     n_samples=20000, rng=np.random.default_rng(7))
            tr_mc, det_mc = float(np.trace(g_mc)), float(np.linalg.det(g_mc))
            tr_an, det_an = trace_fn(theta), det_fn(theta)
            tr_err = abs(tr_mc - tr_an) / abs(tr_an)
            det_err = abs(det_mc - det_an) / abs(det_an)
            ok_tr, ok_det = tr_err < _TRACE_TOL, det_err < _DET_TOL
            mark = "OK " if (ok_tr and ok_det) else "FAIL"
            if not (ok_tr and ok_det):
                n_fail += 1
            print(f"[{mark}] {family:9s} params={tuple(params)}  "
                  f"trace: mc={tr_mc:.4f} vs an={tr_an:.4f} (rel={tr_err:.2e})  "
                  f"det: mc={det_mc:.4f} vs an={det_an:.4f} (rel={det_err:.2e})")
    print(f"\nMC score 验证：{len(CASES) * 3 - n_fail}/{len(CASES) * 3} 通过"
          f"（trace tol={_TRACE_TOL}, det tol={_DET_TOL}）")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
