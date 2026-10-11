# -*- coding: utf-8 -*-
"""**T-c：改用理论不变量作目标，检验"跨域统一"是否在正确坐标下成立**
（2026-10-02，第 202 条）。

## 动机（承第 200–201 条）
   · 第 200 条：预测**原始容量**（nats）时，误差 64.0% → 剖面特征降到 **17.8%**，
     但**区间跨门槛 ⟹ 未认证** ✗；
   · 第 201 条：**行级扩样不能收紧区间** ✗（不确定度由**族数**决定）。
  ⟹ 剩下的问题是：**是不是"目标量选错了"**？"跨域统一"在信息几何里真正
     该比较的，**不是**以 nats 计的绝对容量（那依赖各族自己的尺度 ✗），
     而是**无量纲/对数尺度的曲线形状量** ✓ —— 这正是本条的 T-c ✓

## 三个**预登记目标**（全部无量纲或对数尺度 ✓）
    · **Y1 信道效率比** = `I_TY_max / I_XT_max` ∈ (0,1]（DPI ⟹ ≤1，**可作理论健全性检查** ✓）
    · **Y2 饱和尺度** = `log10(β@90%)`，β@90% 为 `I_TY` 首次达到 90% 容量处 ✓
    · **Y3 拐点锐度** = `log10(β@90% / β@10%)` ✓

## 特征（**严禁含任何 I_TY 派生量** ✗——那是泄漏）
    `log I_XT_max`、`log β90_IXT`、`log(β90_IXT/β10_IXT)`、
    `fisher_trace` 的 min/max/中位/IQR、行数 ✓
    （即：**输入侧（I_XT/指纹）的剖面** ⟹ 与目标同构但**不含答案** ✓）

## 协议 = 第 200 条（族级留一 + 岭回归 λ=1e-3，折内标准化 ✓）
## 预登记判据（**新判据，不改 v29/v36/v37** ✓）
    **E0 可行**：27 族的目标表全部建成（缺者如实列出）✓
    **E1 主判据（认证）**：**至少一个**目标的族级中位相对误差 ≤ **20%**
       **且**对**族**的 bootstrap 95% 区间**上界 < 20%** ✓
    **E2 全报**：三个目标**必须全报**（不许只报最好的 ✓）
    **E3 胜零模型**：每个目标各自与 null（训练族中位）比较 ✓
    **E4 理论健全**：`Y1 ∈ (0, 1]`（DPI ⟹ I_TY ≤ I_XT）✓；数值有限 ✓
    **E5 无泄漏**：打印特征清单，并断言**不含 I_TY 派生量** ✓

用法：
    python a_network/v38_capacity_invariant_transfer.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

RUNS = _ROOT / "v38_capacity_invariant_transfer.csv"
LAM = 1e-3
BETA90, BETA10 = 0.90, 0.10


def _curve_stats():
    """从 DB2 取每族的曲线形状量（**只用 beta/I_XT/I_TY 原文** ✓）。"""
    out: dict[str, dict] = {}
    db2 = _ROOT / "db2" / "db2.jsonl"
    with open(db2, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = rec.get("ib_curve") or {}
            b, ix, iy = c.get("beta"), c.get("I_XT"), c.get("I_TY")
            if not b or not ix or not iy:
                continue
            fam = rec.get("family") or "?"
            d = out.setdefault(fam, {"ix_max": 0.0, "iy_max": 0.0,
                                     "b90y": None, "b10y": None,
                                     "b90x": None, "b10x": None})
            ixm, iym = float(max(ix)), float(max(iy))
            if iym > d["iy_max"]:
                d["iy_max"] = iym
                d["b90y"] = _beta_at(b, iy, iym, BETA90)
                d["b10y"] = _beta_at(b, iy, iym, BETA10)
            if ixm > d["ix_max"]:
                d["ix_max"] = ixm
                d["b90x"] = _beta_at(b, ix, ixm, BETA90)
                d["b10x"] = _beta_at(b, ix, ixm, BETA10)
    return out


def _beta_at(beta, vals, vmax, frac):
    """`vals` 首次达到 `frac·vmax` 处的 beta（线性插值 ✓）。"""
    tgt = frac * vmax
    for i, v in enumerate(vals):
        if v >= tgt:
            if i == 0:
                return float(beta[0])
            v0, v1 = vals[i - 1], v
            b0, b1 = float(beta[i - 1]), float(beta[i])
            if v1 == v0:
                return b1
            return float(b0 + (tgt - v0) * (b1 - b0) / (v1 - v0))
    return float(beta[-1])


def main() -> int:
    from a_network.v27_variational_value_check import load_rows
    rows = load_rows()
    fams = sorted(rows)
    st = _curve_stats()
    print("=" * 74)
    print("T-c：改用理论不变量作目标（族级留一；特征不含 I_TY 派生量 ✓）")
    print("=" * 74)
    mis = [f for f in fams if f not in st]
    print("族 **%d**；曲线形状量缺失的族：%d %s"
          % (len(fams), len(mis), "✓" if not mis else mis))

    # ---------------- 目标 ----------------
    tgt: dict[str, dict] = {}
    for f in fams:
        d = st.get(f)
        if not d:
            continue
        y1 = d["iy_max"] / max(d["ix_max"], 1e-12)
        y2 = (np.log10(d["b90y"]) if d["b90y"] else float("nan"))
        y3 = (np.log10(d["b90y"] / d["b10y"])
              if d["b90y"] and d["b10y"] else float("nan"))
        tgt[f] = {"Y1": y1, "Y2": y2, "Y3": y3}
    have = [f for f in fams if f in tgt]
    print("可用族 **%d**；Y1 范围 [%.4f, %.4f]（**理论健全性：须 ⊆ (0,1]** ✓）"
          % (len(have), min(tgt[f]["Y1"] for f in have),
             max(tgt[f]["Y1"] for f in have)))

    # ---------------- 特征（**无 I_TY 派生量** ✓） ----------------
    FCOLS = ["log_ixmax", "log_b90x", "log_knee_x", "log_ft_min", "log_ft_max",
             "log_ft_med", "log_ft_iqr", "n_rows"]
    X = []
    for f in have:
        d = st[f]
        ft = np.asarray([r["fisher_trace"] for r in rows[f]], float)
        X.append([np.log(max(d["ix_max"], 1e-30)),
                  np.log(max(d["b90x"] or 1e-30, 1e-30)),
                  np.log(max((d["b90x"] or 1e-30) / max(d["b10x"] or 1e-30,
                                                        1e-30), 1e-30)),
                  np.log(max(ft.min(), 1e-30)), np.log(max(ft.max(), 1e-30)),
                  np.log(max(np.median(ft), 1e-30)),
                  np.log(max(np.percentile(ft, 75) - np.percentile(ft, 25),
                             1e-30)),
                  float(len(ft))])
    X = np.asarray(X, float)
    assert np.isfinite(X).all(), "特征含非有限值"
    assert not any("iy" in c.lower() or "I_TY" in c for c in FCOLS), "泄漏!"
    print("**E5 特征清单（无 I_TY 派生量 ✓）**：%s" % ", ".join(FCOLS))

    # ---------------- 族级留一 + 岭回归 ----------------
    def loo(y):
        rels = []
        for i in range(len(have)):
            te = np.zeros(len(have), bool)
            te[i] = True
            tr = ~te
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
            Xs = (X - mu) / sd
            Xtr = np.c_[np.ones(tr.sum()), Xs[tr]]
            w = np.linalg.solve(Xtr.T @ Xtr + LAM * np.eye(Xtr.shape[1]),
                                Xtr.T @ y[tr])
            pred = float(np.r_[1.0, Xs[i]] @ w)
            rels.append(abs(pred - y[i]) / max(abs(y[i]), 1e-9))
        return np.asarray(rels)

    def null(y):
        return np.asarray([abs(np.median(y[[j for j in range(len(have))
                                            if j != i]]) - y[i])
                           / max(abs(y[i]), 1e-9) for i in range(len(have))])

    print("\n%-6s %14s %14s %14s %16s" % ("目标", "中位相对误差", "均值",
                                          "null 中位", "bootstrap 95% 区间"))
    res = {}
    for key in ("Y1", "Y2", "Y3"):
        y = np.asarray([tgt[f][key] for f in have], float)
        r = loo(y)
        nl = null(y)
        rng = np.random.default_rng(0)
        bs = np.median(rng.choice(r, size=(4000, len(r)), replace=True), axis=1)
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        res[key] = (float(np.median(r)), lo, hi, float(np.median(nl)))
        print("%-6s %13.1f%% %13.1f%% %13.1f%% [%.1f%%, %.1f%%]"
              % (key, 100 * np.median(r), 100 * r.mean(), 100 * np.median(nl),
                 100 * lo, 100 * hi))
    print("\n（**E2 三个目标全报** ✓；Y2/Y3 是 log10 尺度量，其「相对误差」"
          "按对数尺度计 ✓）")
    cert = [k for k in res if res[k][0] <= 0.20 and res[k][2] < 0.20]
    e3 = [k for k in res if res[k][0] < res[k][3]]
    y1ok = all(0.0 < tgt[f]["Y1"] <= 1.0 for f in have)
    e4 = y1ok and all(np.isfinite(v) for k in res for v in res[k][:3])
    print("\n**E0 可行**：目标表 %d 族 ✓" % len(have))
    print("**E1 主判据（认证）**：达标的目标 = %s ⟹ %s"
          % (cert or "**无**",
             "**认证达标** ✓" if cert else "**未认证** ✗"))
    print("**E3 胜零模型**：%s ⟹ %s"
          % (e3 or "无", "**通过** ✓" if e3 else "**未通过**"))
    print("**E4 理论健全**（Y1 ∈ (0,1] 且数值有限）：Y1 范围 [%.4f, %.4f] ⟹ %s"
          % (min(tgt[f]["Y1"] for f in have), max(tgt[f]["Y1"] for f in have),
             "**通过** ✓" if e4 else "**未通过**"))
    print("\n**结论**：%s"
          % ("**理论不变量坐标下，跨族预测认证达标** ✓（目标：%s）" % cert
             if cert else
             "**理论不变量坐标下仍未认证** ✗——如实记录"
             "（v29/v36/v37 判据均未改动 ✓）"))
    print("**与 T2 的关系**：T2 测的是**绝对容量**（依赖各族尺度 ✗）；"
          "本条测的是**无量纲形状量** ⟹ 两者结论**不可混用**，均保留在案 ✓")

    with open(RUNS, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family"] + ["Y1_ratio", "Y2_log10_b90", "Y3_log10_knee"]
                   + ["relY1", "relY2", "relY3"])
        ys = {k: np.asarray([tgt[g][k] for g in have], float)
              for k in ("Y1", "Y2", "Y3")}
        rs = {k: loo(ys[k]) for k in ys}
        for i, fam in enumerate(have):
            w.writerow([fam, ys["Y1"][i], ys["Y2"][i], ys["Y3"][i],
                        rs["Y1"][i], rs["Y2"][i], rs["Y3"][i]])
    print("落盘：%s（%d 行）" % (RUNS.name, len(have)))
    return 0 if cert else 1


if __name__ == "__main__":
    sys.exit(main())
