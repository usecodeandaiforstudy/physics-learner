# -*- coding: utf-8 -*-
"""C 侧覆盖率审计 + MDN↔CFM 对照（2026-10-02，第 101 条）。

**触发条件的直接检验**（design doc §2.6.2）：MDN **覆盖率 <90% 即淘汰**，
生产 = CFM。本脚本做两件事：
  1. **覆盖率审计**：以**重叠系数** `∫min(p_true, p_model)dφ` 为覆盖口径，
     在**训练点插值（a=−1.5/−0.75/−0.25/0.5/1.5）+ 外推（a=2.5）**上评估
     MDN 与 CFM；
  2. **MDN↔CFM 对照**：同一数据、同一评估口径，**20 seed 配对**。

**预登记判据（运行前写死，机械套用）**：
  V1 **覆盖达标**：CFM 在 6 个评估点上的**平均重叠 ≥ 0.90**；
  V2 **优于 MDN**：配对 Δ=ovl_CFM−ovl_MDN 满足三态判定
     （达标率 ≥70% → 稳健优；≤30% → 稳健劣；其余 → 未定）；
  V3 **模式召回**：双阱区（a ≤ −0.5）两点上 CFM **两模都恢复**（在
     ±√(−a) 各 ±0.4 内各有一个峰）的比例 **≥ 90%**；
  V4 **密度质量**：CFM 的 NLL（ODE 精确 log 密度，在真值样本上）**≤ 真值
     NLL + 0.5 nats**（沿用 C4 口径）；
  V5 **外推稳健**：a=2.5（训练范围外）重叠 **≥ 0.80**。

用法（项目根目录）：
  python c_network/v22_cfm_check.py                 # 完整对照（默认 20 seed）
  python c_network/v22_cfm_check.py --report cov_c.csv   # 从 CSV 汇总
  环境变量 V22_SEEDS 改 seed 数；V22_CSV 逐 seed 落盘（默认 cov_c.csv）
"""

from __future__ import annotations

import collections
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from c_network.cfm import ConditionalFlowMatching
from c_network.mdn import MixtureDensityNetwork, sample_landau

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

N_SEED = int(os.environ.get("V22_SEEDS", "20"))
S0 = int(os.environ.get("V22_S0", "0"))
CSV_PATH = os.environ.get("V22_CSV", "cov_c.csv")

A_TRAIN = [-2.0, -1.0, 0.0, 1.0, 2.0]
A_EVAL = [-1.5, -0.75, -0.25, 0.5, 1.5, 2.5]      # 插值 5 + 外推 1
GRID = np.linspace(-3.2, 3.2, 241)
PHI_LO, PHI_HI = -3.5, 3.5


def true_pdf(a: float, grid: np.ndarray) -> np.ndarray:
    """真值密度 p(φ|a) ∝ exp(−(aφ²/2 + φ⁴/4))（kT=b=1），网格归一。"""
    logp = -(a * grid ** 2 / 2.0 + grid ** 4 / 4.0)
    logp = logp - logp.max()
    p = np.exp(logp)
    return p / (p.sum() * (grid[1] - grid[0]))


def norm_on_grid(logp: np.ndarray, grid: np.ndarray) -> np.ndarray:
    p = np.exp(logp - logp.max())
    return p / (p.sum() * (grid[1] - grid[0]))


def overlap(p: np.ndarray, q: np.ndarray, grid: np.ndarray) -> float:
    """重叠系数 ∫min(p,q)dφ（覆盖口径）。"""
    return float(np.sum(np.minimum(p, q)) * (grid[1] - grid[0]))


def two_mode_recall(a: float, p_model: np.ndarray) -> bool:
    """双阱区：真值在 ±√(−a) 两处有峰；模型两处各需有峰（±0.4 窗内）。"""
    if a >= -0.5:
        return True
    peaks = [GRID[i] for i in range(1, len(p_model) - 1)
             if p_model[i] >= p_model[i - 1] and p_model[i] > p_model[i + 1]]
    c = float(np.sqrt(-a))
    left = any(abs(pk + c) < 0.4 for pk in peaks)
    right = any(abs(pk - c) < 0.4 for pk in peaks)
    return left and right


def train_mdn(seed: int) -> MixtureDensityNetwork:
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for a in A_TRAIN:
        phi = sample_landau(a, 1.0, 1.0, 4000, seed=seed)
        X.append(np.full(len(phi), a, dtype=np.float32))
        Y.append(phi.astype(np.float32))
    X = np.concatenate(X)[:, None]
    Y = np.concatenate(Y)[:, None]
    torch.manual_seed(seed)
    mdn = MixtureDensityNetwork(1, n_components=8, seed=seed)
    opt = torch.optim.Adam(mdn.parameters(), lr=3e-3)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for _ in range(6000):
        idx = rng.integers(0, len(Xt), 512)
        loss = mdn.nll(Xt[idx], Yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return mdn


def train_cfm(seed: int) -> ConditionalFlowMatching:
    rng = np.random.default_rng(seed)
    A, P = [], []
    for a in A_TRAIN:
        phi = sample_landau(a, 1.0, 1.0, 4000, seed=seed)
        A.append(np.full(len(phi), a, dtype=np.float32))
        P.append(phi.astype(np.float32))
    A = np.concatenate(A)[:, None]
    P = np.concatenate(P)[:, None]
    torch.manual_seed(seed)
    cfm = ConditionalFlowMatching(1, seed=seed)
    opt = torch.optim.Adam(cfm.parameters(), lr=2e-3)
    At, Pt = torch.tensor(A), torch.tensor(P)
    for _ in range(6000):
        idx = rng.integers(0, len(At), 512)
        loss = cfm.loss(At[idx], Pt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return cfm


def nll_true_on_samples(a: float, samples: np.ndarray) -> float:
    """真值密度在样本上的 NLL（网格插值）。"""
    pdf = true_pdf(a, GRID)
    return float(-np.mean(np.log(np.interp(samples, GRID, pdf) + 1e-12)))


def one_seed(seed: int) -> dict:
    mdn = train_mdn(seed)
    cfm = train_cfm(seed)
    ovl_m, ovl_c, mode_ok, nll_c, nll_t, ovl_ext = [], [], [], [], [], None
    for a in A_EVAL:
        pt = true_pdf(a, GRID)
        pm = norm_on_grid(np.log(mdn.density(a, GRID) + 1e-12), GRID)
        pc = norm_on_grid(cfm.log_density(a, GRID), GRID)
        ovl_m.append(overlap(pt, pm, GRID))
        ovl_c.append(overlap(pt, pc, GRID))
        mode_ok.append(two_mode_recall(a, pc))
        # CFM 的 NLL：ODE 精确 log 密度在真值样本上
        s = sample_landau(a, 1.0, 1.0, 2000, seed=seed + 7)
        lp = cfm.log_density(a, s)
        nll_c.append(float(-np.mean(lp)))
        nll_t.append(nll_true_on_samples(a, s))
        if abs(a - 2.5) < 1e-9:
            ovl_ext = overlap(pt, pc, GRID)
    return {"ovl_mdn": float(np.mean(ovl_m)),
            "ovl_cfm": float(np.mean(ovl_c)),
            "ovl_cfm_ext": float(ovl_ext),
            "mode_recall": float(np.mean([m for a, m in
                                          zip(A_EVAL, mode_ok)
                                          if a <= -0.5])),
            "nll_cfm": float(np.mean(nll_c)),
            "nll_true": float(np.mean(nll_t)),
            "ovl_each_cfm": [round(x, 4) for x in ovl_c]}


def verdict(rate: float) -> str:
    if rate >= 0.70:
        return "稳健成立"
    if rate <= 0.30:
        return "稳健不成立"
    return "不稳定（依赖 seed）"


def report(path: str) -> int:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        print("（CSV 为空）")
        return 1
    n = len(rows)
    ovl_c = np.array([float(r["ovl_cfm"]) for r in rows])
    ovl_m = np.array([float(r["ovl_mdn"]) for r in rows])
    ext = np.array([float(r["ovl_cfm_ext"]) for r in rows])
    mode = np.array([float(r["mode_recall"]) for r in rows])
    nllc = np.array([float(r["nll_cfm"]) for r in rows])
    nllt = np.array([float(r["nll_true"]) for r in rows])
    print("========== C 侧覆盖率审计汇总（%s，%d seed）==========" % (path, n))
    print("  V1 CFM 平均重叠：%.4f（std %.4f，范围 %.4f…%.4f）→ %s"
          % (ovl_c.mean(), ovl_c.std(ddof=1), ovl_c.min(), ovl_c.max(),
             "**达标（≥0.90）**" if ovl_c.mean() >= 0.90 else "未达标"))
    d = ovl_c - ovl_m
    rate = float(np.mean(d > 0))
    who = ("**CFM 稳健优**" if rate >= 0.70 else
           "**MDN 稳健优**（CFM 稳健劣）" if rate <= 0.30 else
           "**未定**（互有胜负）")
    print("  V2 CFM vs MDN（配对 Δ 重叠）：均值 %+.4f（std %.4f）"
          "CFM 更优的 seed 占比 %d/%d（%.0f%%）→ **%s**"
          % (d.mean(), d.std(ddof=1), int((d > 0).sum()), n, 100 * rate,
             who))
    print("     （MDN 平均重叠 %.4f；CFM %.4f）" % (ovl_m.mean(), ovl_c.mean()))
    print("  V3 双阱模式召回：均值 %.3f（std %.3f）"
          % (mode.mean(), mode.std(ddof=1)))
    print("  V4 密度质量：CFM NLL %.3f vs 真值 %.3f（差 %+.3f；判据 ≤ +0.5）"
          % (nllc.mean(), nllt.mean(), (nllc - nllt).mean()))
    print("  V5 外推（a=2.5）重叠：均值 %.4f（std %.4f）→ %s"
          % (ext.mean(), ext.std(ddof=1),
             "**≥0.80 达标**" if ext.mean() >= 0.80 else "未达标"))
    return 0


def main() -> int:
    if "--report" in sys.argv:
        i = sys.argv.index("--report")
        return report(sys.argv[i + 1] if len(sys.argv) > i + 1 else CSV_PATH)
    print("========== C 侧覆盖率审计 + MDN↔CFM 对照（seed %d–%d / 共 %d）"
          "==========" % (S0, N_SEED - 1, N_SEED))
    print("  预登记：V1 CFM 重叠 ≥0.90；V2 配对优于 MDN（三态）；"
          "V3 双阱召回 ≥0.90；V4 NLL ≤ 真值+0.5；V5 外推 ≥0.80\n")
    new = not Path(CSV_PATH).exists()
    f = open(CSV_PATH, "a", encoding="utf-8", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["seed", "ovl_mdn", "ovl_cfm", "ovl_cfm_ext",
                    "mode_recall", "nll_cfm", "nll_true"])
    for seed in range(S0, N_SEED):
        r = one_seed(seed)
        print("    seed %2d：MDN 重叠 %.4f | **CFM %.4f**（外推 %.4f）| "
              "双阱召回 %.2f | NLL %.2f vs 真值 %.2f"
              % (seed, r["ovl_mdn"], r["ovl_cfm"], r["ovl_cfm_ext"],
                 r["mode_recall"], r["nll_cfm"], r["nll_true"]))
        w.writerow([seed, "%.5f" % r["ovl_mdn"], "%.5f" % r["ovl_cfm"],
                    "%.5f" % r["ovl_cfm_ext"], "%.3f" % r["mode_recall"],
                    "%.4f" % r["nll_cfm"], "%.4f" % r["nll_true"]])
        f.flush()
    f.close()
    print("\n  已写入逐点 CSV：%s（用 --report %s 汇总）" % (CSV_PATH, CSV_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
