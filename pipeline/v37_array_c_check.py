# -*- coding: utf-8 -*-
"""R5-③：**`array` 域的 C 实例**（第 3 个可用生成域）（2026-10-02，第 154 条）。

## 任务形式（沿用 v34 第三轮验证过的**可行集条件化**）
    对每个需求水平 `q_req`，只保留**可行设计**（`q1 ≥ q_req` **且无栅瓣**）
    ⟹ 训练 `p(5 维设计向量 | q_req)` ⟹ 采样后经**域校验**判达标
    **判据与 v33/v34 一致**：C1（达标 seed 比例 ≥70%）**且** C2（多样性门禁）。

## 为什么它应当成功（预期）
`array` 的可行集**厚**：4 dof、信息侧 2 标量 ⟹ 可行集是 **2 维流形**（4−2）
（而 `chain` 的可行集是 7 维里的**薄流形**）⟹ 预期**不坍缩**、多样性好。

用法：
    python pipeline/v37_array_c_check.py --seeds 3
    python pipeline/v37_array_c_check.py
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

def _runs_name(kind: str) -> str:
    """**正式记录保护（第 164 条）**：`--seeds < 20` 的**试点跑**写到
    `*_runs_pilot_<n>seed.csv`，**绝不覆盖 20 seed 的正式记录**。

    修因：我曾用 `--seeds 3` 冒烟，**把 20 seed 的正式 `v37_runs.csv` 覆盖成
    3 行**（数据已重跑恢复，但这是**流程缺陷**）。
    """
    n = 20
    if "--seeds" in sys.argv:
        try:
            n = int(sys.argv[sys.argv.index("--seeds") + 1])
        except (ValueError, IndexError):
            n = 20
    return ("%s_runs.csv" % kind) if n >= 20 else \
        ("%s_runs_pilot_%dseed.csv" % (kind, n))


RUNS = _ROOT / _runs_name("v37")
DATA_CSV = _ROOT / "v37_dataset.csv"
LEVELS = (1.5, 3.0, 6.0, 12.0)       # 需求水平（q1）
PER_LEVEL = 600
TRAIN_STEPS = 4000
LR = 1.5e-3
HIDDEN = 160
AUG_SIGMA = 0.02
N_SAMPLE = 40
VEC_DIM = 5
DIV_MIN = 10
_REBUILD = False


def _norm_q(q: float, lo: float = 0.5, hi: float = 30.0) -> float:
    return float((math.log10(max(q, 1e-6)) - math.log10(lo))
                 / max(math.log10(hi) - math.log10(lo), 1e-9))


def build_dataset(seed: int = 0):
    """**可行集数据集**：每水平只留 `q1 ≥ 水平 ∧ 无栅瓣` 的设计。"""
    from pipeline.design_array import sample_array, to_vector
    from pipeline.design_space import verify_design
    if DATA_CSV.exists() and not _REBUILD:
        X, C = [], []
        with open(DATA_CSV, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    X.append([float(x) for x in row["vec"].split(",")])
                    C.append(_norm_q(float(row["q_req"])))
                except (TypeError, ValueError):
                    continue
        if X:
            return (np.asarray(X, dtype=np.float32),
                    np.asarray(C, dtype=np.float32)[:, None],
                    "**可行集数据集（缓存 %s，%d 条）**" % (DATA_CSV.name,
                                                        len(X)))
    rng = np.random.default_rng(seed)
    X, C, rows = [], [], []
    for lv in LEVELS:
        got, tries = 0, 0
        while got < PER_LEVEL and tries < PER_LEVEL * 200:
            tries += 1
            d = sample_array(rng)
            v = verify_design("array", d, target=lv)
            if not v.get("meets"):
                continue                       # **只有可行设计入库**
            vec = to_vector(d)
            X.append(vec)
            C.append(_norm_q(lv))
            rows.append((lv, ",".join("%.6f" % x for x in vec),
                         float(v.get("measured") or 0.0)))
            got += 1
    with open(DATA_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q_req", "vec", "q1"])
        w.writerows(rows)
    return (np.asarray(X, dtype=np.float32),
            np.asarray(C, dtype=np.float32)[:, None],
            "**可行集数据集（%d 条 = %d 水平 × %d；已缓存 %s）**"
            % (len(X), len(LEVELS), PER_LEVEL, DATA_CSV.name))


def train_one(seed: int, X, C):
    import torch
    from c_network.cfm import ConditionalFlowMatching
    xt, ct = torch.tensor(X), torch.tensor(C)
    g = torch.Generator().manual_seed(seed)
    xt = torch.clamp(xt + AUG_SIGMA * torch.randn(xt.shape, generator=g),
                     0.0, 1.0)
    model = ConditionalFlowMatching(in_dim=VEC_DIM, cond_dim=1, hidden=HIDDEN,
                                    seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed)
    loss = None
    for _ in range(TRAIN_STEPS):
        idx = torch.tensor(rng.integers(0, len(X), 256))
        loss = model.loss(ct[idx], xt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, float(loss.item())


def evaluate(model, targets) -> dict:
    """**闭环校验**：采样 → 解码 → **域校验**（含栅瓣门禁）→ 判达标。"""
    from pipeline.design_array import from_vector
    from pipeline.design_space import verify_design
    n_ok = n_tot = 0
    uniq = set()
    qs = []
    for q_t in targets:
        c = np.array([_norm_q(q_t)], dtype=np.float32)
        Z = model.sample_cond(c, N_SAMPLE, steps=60, seed=int(q_t * 997))
        for z in Z:
            d = from_vector(np.clip(z, 0.0, 1.0))
            uniq.add((int(d["N"]), round(d["d_over_lam"], 3),
                      round(d["taper_p"], 3), round(math.log10(d["f_Hz"]), 3)))
            v = verify_design("array", d, target=q_t)
            n_tot += 1
            if v.get("meets"):
                n_ok += 1
            if v.get("measured") is not None:
                qs.append(float(v["measured"]))
    return {"pass_rate": n_ok / max(n_tot, 1), "n_tot": n_tot,
            "diversity": len(uniq),
            "q_min": (min(qs) if qs else float("nan")),
            "q_max": (max(qs) if qs else float("nan"))}


def main() -> int:
    global _REBUILD
    seeds = 20
    if "--seeds" in sys.argv:
        seeds = int(sys.argv[sys.argv.index("--seeds") + 1])
    _REBUILD = "--rebuild" in sys.argv
    print("=" * 74)
    print("R5-③ `array` 域 C 实例（需求 → 设计分布）+ 闭环校验")
    print("=" * 74)
    t0 = time.time()
    X, C, src = build_dataset(seed=0)
    print("训练数据：%s；样本 %d；向量 %d 维（混合型）" % (src, len(X),
                                                   X.shape[1]))
    print("需求水平：%s（可行集条件化：仅 `q1 ≥ 水平 ∧ 无栅瓣` 入库）"
          % list(LEVELS))
    rec = []
    print("\n%4s %10s %10s %9s %9s %8s" % ("seed", "闭环达标率", "多样性",
                                          "q1_min", "q1_max", "loss"))
    for s in range(seeds):
        model, loss = train_one(s, X, C)
        ev = evaluate(model, list(LEVELS))
        rec.append({"seed": s, **ev, "loss": loss})
        print("%4d %9.1f%% %10d %9.4g %9.4g %8.4f"
              % (s, 100 * ev["pass_rate"], ev["diversity"], ev["q_min"],
                 ev["q_max"], loss))
    pr = np.array([r["pass_rate"] for r in rec])
    dv = np.array([r["diversity"] for r in rec])
    print("\n===== 汇总（%d seed）=====" % seeds)
    print("C1 闭环达标率：均值 **%.1f%% ± %.1f%%**；达标(≥70%%) seed：%d/%d"
          % (100 * pr.mean(), 100 * pr.std(), int(np.sum(pr >= 0.70)), seeds))
    print("C2 多样性：均值 **%.0f**（每条件 %d 采样，%d 条件）；"
          "达标(≥%d) seed：%d/%d"
          % (dv.mean(), N_SAMPLE, len(LEVELS), DIV_MIN,
             int(np.sum(dv >= DIV_MIN)), seeds))
    q_all = np.array([r["q_min"] for r in rec] + [r["q_max"] for r in rec])
    print("C3 无不可能的数：实测 q1 ∈ [%.4g, %.4g] ⟹ %s"
          % (np.nanmin(q_all), np.nanmax(q_all),
             "✓" if (np.nanmin(q_all) > 0 and np.isfinite(q_all).all())
             else "**查 bug**"))
    rate70 = float(np.mean(pr >= 0.70))
    rate_div = float(np.mean(dv >= DIV_MIN))
    print("C4 三态判定（C1 达标 seed 比例 %.0f%%）：%s"
          % (100 * rate70, "**稳健成立**" if rate70 >= 0.70 else
             ("**稳健不成立**" if rate70 <= 0.30 else "**不稳定**")))
    print("C2 **门禁**（多样性 ≥%d 的 seed 比例 %.0f%%）：%s"
          % (DIV_MIN, 100 * rate_div, "**通过** ✓" if rate_div >= 0.70
             else "**未通过**（模式坍缩）"))
    print("⟹ **综合判定**：%s"
          % ("**成立**（C1 与 C2 双过）"
             if (rate70 >= 0.70 and rate_div >= 0.70)
             else "**部分成立**（C1 %s / C2 %s）"
             % ("过" if rate70 >= 0.70 else "不过",
                "过" if rate_div >= 0.70 else "不过")))
    with open(RUNS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec[0]))
        w.writeheader()
        w.writerows(rec)
    print("\n逐 seed 落盘：%s（%d 行）；耗时 %.1fs"
          % (RUNS.name, len(rec), time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
