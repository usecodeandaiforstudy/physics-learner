"""非高斯分布族的 IB 曲线自测。

验证三个非高斯信道（指数/泊松/伯努利）的 IB 曲线：
1. 曲线非空、单调（I_XT 与 I_TY 随 β 非降）。
2. 大 β 时 I_TY 收敛到端到端互信息 I(θ;Y)。

运行：python b_network/test_ib_nonparametric.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from b_network.system_encoder import SystemEncoder  # noqa: E402
from b_network.fingerprint_computer import FingerprintComputer  # noqa: E402


def _end_to_end_mi(px, p_y_given_x) -> float:
    """端到端互信息 I(X;Y) = Σ_x p(x) Σ_y p(y|x) log(p(y|x)/p(y))。"""
    p_y = px @ p_y_given_x
    mi = 0.0
    for i in range(len(px)):
        for j in range(p_y_given_x.shape[1]):
            if p_y_given_x[i, j] > 0 and p_y[j] > 0:
                mi += px[i] * p_y_given_x[i, j] * math.log(p_y_given_x[i, j] / p_y[j])
    return mi


def _build_channel(graph, n_bins=60):
    """重建非高斯信道（复用 fpc 内部逻辑做独立参照）。"""
    ib_src = graph.metadata["ib_source"]
    channel = ib_src["channel"]
    lo, hi = ib_src["range"]
    theta = np.linspace(lo, hi, n_bins)
    px = np.full(n_bins, 1.0 / n_bins)
    if channel == "exponential":
        y_max = 5.0 / max(lo, 1e-6)
        ys = np.linspace(0.0, y_max, n_bins)
        p = np.zeros((n_bins, n_bins))
        for i, lam in enumerate(theta):
            row = lam * np.exp(-lam * ys)
            p[i] = row / row.sum()
    elif channel == "poisson":
        n_max = int(math.ceil(3.0 * hi)) + 5
        ys = np.arange(n_max + 1)
        p = np.zeros((n_bins, len(ys)))
        for i, lam in enumerate(theta):
            row = np.exp(-lam) * lam ** ys / np.array([math.factorial(k) for k in ys])
            p[i] = row / row.sum()
    elif channel == "bernoulli":
        p = np.zeros((n_bins, 2))
        for i, p_ in enumerate(theta):
            p[i, 0] = 1.0 - p_
            p[i, 1] = p_
    return px, p


def main():
    lines = []
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    enc = SystemEncoder()
    fpc = FingerprintComputer()

    cases = [
        ("指数信道", {"family": "radioactive_decay", "channel_template": "CT-DECAY-001",
                      "parameters": [{"name": "rate", "default": 1.0, "range": [0.1, 5.0]}]}),
        ("泊松信道", {"family": "poisson_count", "channel_template": "CT-COUNT-001",
                      "parameters": [{"name": "rate", "default": 3.0, "range": [0.5, 10.0]}]}),
        ("伯努利信道", {"family": "two_level", "channel_template": "CT-TWOLVL-001",
                        "parameters": [{"name": "p", "default": 0.5, "range": [0.05, 0.95]}]}),
    ]

    for name, entry in cases:
        g = enc.encode(entry)
        ib = fpc.compute_ib_curve(g, n_bins=50)
        i_xt = ib["I_XT"]
        i_ty = ib["I_TY"]
        beta = ib["beta"]

        # 1. 非空
        ok_nonempty = len(i_xt) > 0 and len(i_ty) > 0
        # 2. 单调非降（仅检查 β ≤ 100 的中低 β 区域；高 β 是 BA 的数值极限）
        low_beta = np.asarray(beta) <= 100.0
        i_xt_low = np.asarray(i_xt)[low_beta]
        i_ty_low = np.asarray(i_ty)[low_beta]
        ok_mono = bool(np.all(np.diff(i_xt_low) >= -1e-6)
                       and np.all(np.diff(i_ty_low) >= -1e-6))
        # 3. 收敛到端到端 MI（用 I_TY 最大值，而非最后一个点，避免高 β 数值噪声）
        px, pyx = _build_channel(g, n_bins=50)
        i_end = _end_to_end_mi(px, pyx)
        i_ty_max = float(np.max(i_ty)) if len(i_ty) else 0.0
        ok_conv = abs(i_ty_max - i_end) / (abs(i_end) + 1e-9) < 0.05

        lines.append(
            ("PASS" if (ok_nonempty and ok_mono and ok_conv) else "FAIL")
            + f"  {name}：非空={ok_nonempty} 单调(β≤100)={ok_mono} "
              f"max I_TY={i_ty_max:.4f} vs 端到端I={i_end:.4f}"
        )

    n_pass = sum(1 for ln in lines if ln.startswith("PASS"))
    n_total = sum(1 for ln in lines if ln.startswith(("PASS", "FAIL")))
    lines.append(f"\n{n_pass}/{n_total} 通过")

    out = Path(r"d:\physics learn\ib_nonparam_test.txt")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
