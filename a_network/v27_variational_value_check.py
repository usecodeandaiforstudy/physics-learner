# -*- coding: utf-8 -*-
"""变分层 L2：变分特征的增量价值（2026-10-02，第 116 条）。

**问题**：变分解出的「极值作用量 S*」与「测地线长度 L」相对**逐行静态特征**
是否提供**可测的增量**？

**关键设计点（否则实验无意义）**：变分特征必须建立在**沿族参数路径的
`g(θ)` 剖面**上——由该族 30 个扫描点的 `fisher_trace` **插值**得到。若取
`g=const`，则 `L = √g·|Δθ|`、`S* = ½gΔθ²/T` 与基线里的点值 `fisher_trace`
**共线**，必然无增量（这是"设计决定了结论"的陷阱，须先避开）。

**预登记判据**：
  V1 **无泄漏**：特征只用该行**自身**的信息（跨行统计仅用于 CV 划分）；
  V2 **增量**：留一族 CV 的 R²，实验臂 − 对照臂 **配对 Δ ≥ +0.05**，
     且 **≥70% 的 seed 为正**（三态：≥70% 稳健有增量 / ≤30% 稳健无 / 其余未定）；
  V3 **不以泄漏取胜**：实验臂不得使用目标量 `I_TY@4` 的任何变换；
  V4 **逐点落盘**：每 seed 一行 CSV，支持分段续跑（`V27_S0`/`V27_CSV`）。

用法（项目根目录）：
  python a_network/v27_variational_value_check.py
  python a_network/v27_variational_value_check.py --report v27_runs.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

DB2 = _ROOT / "db2" / "db2.jsonl"
BETA_OP = 4.0
N_SEED = int(os.environ.get("V27_SEEDS", "20"))
S0 = int(os.environ.get("V27_S0", "0"))
CSV_PATH = os.environ.get("V27_CSV", "v27_runs.csv")


def load_rows() -> dict[str, list[dict]]:
    """按族取 DB2 行：θ（scan 或 physical_params）、fisher_trace、I_XT@4、I_TY@4。"""
    fam_rows: dict[str, list[dict]] = {}
    with open(DB2, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = rec.get("ib_curve") or {}
            if not c.get("beta"):
                continue
            fam = rec.get("family") or "?"
            scan = rec.get("scan") or {}
            th = scan.get("scan_value")
            if th is None:
                pp = (rec.get("metadata") or {}).get("physical_params") or {}
                th = pp.get("a", pp.get("k", pp.get("theta", pp.get("kappa"))))
            if th is None:
                continue
            fps = rec.get("fingerprints") or {}
            ft = None
            for v in fps.values():
                if isinstance(v, dict) and "fisher_trace" in v:
                    ft = float(v["fisher_trace"])
            if ft is None:
                continue
            b = np.asarray(c["beta"], dtype=float)
            x = np.asarray(c["I_XT"], dtype=float)
            y = np.asarray(c["I_TY"], dtype=float)
            fam_rows.setdefault(fam, []).append({
                "theta": float(th), "fisher_trace": ft,
                "I_XT": float(np.interp(BETA_OP, b, x)),
                "I_TY": float(np.interp(BETA_OP, b, y))})
    # 稳定排序，去重（同族同 θ 取首条）
    out: dict[str, list[dict]] = {}
    for fam, rows in fam_rows.items():
        seen, uniq = set(), []
        for r in sorted(rows, key=lambda z: z["theta"]):
            key = round(r["theta"], 10)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        if len(uniq) >= 8:
            out[fam] = uniq
    return out


def variational_features(rows: list[dict], n_nodes: int = 41,
                         iters: int = 400) -> np.ndarray:
    """对每行算 (测地线长度, 极值作用量)：从族内 θ_min 到该行 θ_i。

    `g(θ)` 由该族扫描点的 `fisher_trace` **插值**（见模块 docstring 的
    关键设计点）；势能 V 取该行 IB 曲线的 `V = I_XT − β·I_TY` 的**线性剖面**
    （θ 轴上用两端点线性插值，避免引入额外假设）。
    """
    th = np.array([r["theta"] for r in rows], dtype=float)
    g = np.array([r["fisher_trace"] for r in rows], dtype=float)
    # 稳健化：非正或非有限 → 用中位替代
    g = np.where(np.isfinite(g) & (g > 0), g, np.nanmedian(g[g > 0])
                 if np.any(g > 0) else 1.0)
    order = np.argsort(th)
    th_s, g_s = th[order], g[order]
    g_of = lambda x: float(np.interp(x, th_s, g_s))       # noqa: E731
    feats = np.zeros((len(rows), 2), dtype=float)
    th0 = float(th_s[0])
    from a_network.variational import extremal_trajectory
    for i, r in enumerate(rows):
        th1 = float(r["theta"])
        if abs(th1 - th0) < 1e-9:
            feats[i] = (0.0, 0.0)
            continue
        res = extremal_trajectory(
            "callable", "zero", th0, th1, T=1.0, n_nodes=n_nodes,
            iters=iters, lr=0.02,
            m_params={"g_fn": g_of})
        feats[i] = (res["geodesic_length"], res["action"])
    return feats


def _ridge_cv_r2(X: np.ndarray, y: np.ndarray, fam_ids: np.ndarray,
                 lam: float = 1e-3) -> float:
    """留一族 CV 的 R²（族为分组，避免同族信息泄漏）。"""
    preds = np.zeros_like(y)
    for fam in np.unique(fam_ids):
        te = fam_ids == fam
        tr = ~te
        if tr.sum() < 3 or te.sum() == 0:
            preds[te] = float(np.mean(y[tr])) if tr.sum() else 0.0
            continue
        Xtr = np.c_[np.ones(tr.sum()), X[tr]]
        Xte = np.c_[np.ones(te.sum()), X[te]]
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, Xtr.T @ y[tr])
        preds[te] = Xte @ w
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def prepare(fam_rows: dict[str, list[dict]]) -> dict:
    """**算一次并缓存**（2026-10-02 修）：变分特征与静态特征都是确定性的，
    原实现放在 `one_seed` 里 ⟹ 20 seed 重算 20 次（白跑）且**seed 无作用**。"""
    Xb, Xv, y, fams = [], [], [], []
    for fam, rows in fam_rows.items():
        vf = variational_features(rows)
        for i, r in enumerate(rows):
            base = [r["fisher_trace"], r["I_XT"]]
            Xb.append(base)
            Xv.append(base + [vf[i, 0], vf[i, 1]])
            y.append(r["I_TY"])
            fams.append(fam)
    return {"Xb": np.asarray(Xb, float), "Xv": np.asarray(Xv, float),
            "y": np.asarray(y, float), "fams": np.asarray(fams)}


def _std(X: np.ndarray) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


def one_seed(prep: dict, seed: int) -> dict:
    """一次**逐族 bootstrap** 重采样后的留一族 CV（seed 真正起作用）。

    修因：原实现无随机性 ⟹ 20 seed 结果逐位相同（std=0），"0/20 为正"
    实为**一次测量**。现按族内重采样（保持族结构）以得到 Δ 的真实分布。
    """
    rng = np.random.default_rng(seed)
    idx = []
    for fam in np.unique(prep["fams"]):
        ii = np.where(prep["fams"] == fam)[0]
        idx.extend(rng.choice(ii, size=len(ii), replace=True))
    idx = np.asarray(idx)
    Xb, Xv = prep["Xb"][idx], prep["Xv"][idx]
    y, fams = prep["y"][idx], prep["fams"][idx]
    r2b = _ridge_cv_r2(_std(Xb), y, fams)
    r2v = _ridge_cv_r2(_std(Xv), y, fams)
    return {"r2_base": r2b, "r2_var": r2v, "delta": r2v - r2b,
            "n_rows": int(len(y)), "n_fam": int(len(np.unique(fams)))}


def report(path: str) -> int:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        print("（CSV 为空）")
        return 1
    d = np.array([float(r["delta"]) for r in rows])
    rb = np.array([float(r["r2_base"]) for r in rows])
    rv = np.array([float(r["r2_var"]) for r in rows])
    rate = float(np.mean(d > 0))
    verdict = ("**稳健有增量**" if rate >= 0.70 else
               "**稳健无增量**" if rate <= 0.30 else "**未定**（依赖 seed）")
    print("========== L2 变分特征增量（%s，%d seed）==========" % (path, len(rows)))
    print("  R²（对照臂，静态特征）：%.4f（std %.4f）" % (rb.mean(), rb.std(ddof=1)))
    print("  R²（实验臂，+ 变分特征）：%.4f（std %.4f）" % (rv.mean(), rv.std(ddof=1)))
    print("  配对 Δ：均值 %+.4f（std %.4f）；正号 seed %d/%d（%.0f%%）→ **%s**"
          % (d.mean(), d.std(ddof=1), int((d > 0).sum()), len(rows),
             100 * rate, verdict))
    print("  预登记判据：Δ ≥ +0.05 且 ≥70%% seed 为正 → 有增量（当前 %s）"
          % ("满足" if (d.mean() >= 0.05 and rate >= 0.70) else "不满足"))
    return 0


def main() -> int:
    if "--report" in sys.argv:
        i = sys.argv.index("--report")
        return report(sys.argv[i + 1] if len(sys.argv) > i + 1 else CSV_PATH)
    fam_rows = load_rows()
    print("========== L2：变分特征的增量价值（留一族 CV） ==========")
    print("族 %d 个，行 %d 条：%s"
          % (len(fam_rows), sum(len(v) for v in fam_rows.values()),
             ", ".join("%s(%d)" % (k, len(v)) for k, v in fam_rows.items())))
    new = not Path(CSV_PATH).exists()
    f = open(CSV_PATH, "a", encoding="utf-8", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["seed", "r2_base", "r2_var", "delta", "n_rows", "n_fam"])
    print("  计算变分特征（一次缓存）…", flush=True)
    prep = prepare(fam_rows)
    print("  基线 R²（确定性，未重采样）：%.4f"
          % _ridge_cv_r2(_std(prep["Xb"]), prep["y"], prep["fams"]))
    for seed in range(S0, N_SEED):
        r = one_seed(prep, seed)
        print("  seed %2d：R² %.4f → %.4f（Δ %+.4f）"
              % (seed, r["r2_base"], r["r2_var"], r["delta"]))
        w.writerow([seed, "%.5f" % r["r2_base"], "%.5f" % r["r2_var"],
                    "%.5f" % r["delta"], r["n_rows"], r["n_fam"]])
        f.flush()
    f.close()
    print("\n已写入 %s（用 --report %s 汇总）" % (CSV_PATH, CSV_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
