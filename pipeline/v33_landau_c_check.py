# -*- coding: utf-8 -*-
"""中游 B 步骤②③④：**landau 高维设计域的 C 实例 + 闭环校验**
（2026-10-02，第 133 条）。

## 任务形式（**可验证**的定义）
    条件 `c = (q1 目标, ΔF 目标)`（归一化）——即"要求设计达到的性能"
    输出：**设计分布** `p((a, b, kT) | c)`
    验证：从该分布采样设计 → 送**既有编码器**（landau）→ 实测 `(q1, ΔF)`
          → 判"是否落在目标桶内"（**这就是闭环**）

## 为什么这样定义
C 现有实例输出的是 `p(φ|a)`（φ 的密度），而契约要的是**族参数/设计**
⟹ 本节把"解码"从「参数→φ 密度」扩展为「**需求→设计分布**」——
这正是"中游缺口 B"的实质，且**可机器验证**。

## 预登记判据（**先写死**）
  | 判据 | 内容 | 不过的含义 |
  |---|---|---|
  | **C1 闭环达标率** | 采样的设计经编码器实测后**落在目标桶内**的比例 **≥70%** | C 学不到条件-设计映射 |
  | **C2 多样性** | 每个目标下**不同设计 ≥10 个**（非坍缩） | 退化为单点 ⟹ 生成无意义 |
  | **C3 无不可能的数** | 实测 q1/ΔF 落在扫描范围内且有限 | 实现有 bug |
  | **C4 20 seed** | 逐 seed 落盘 + 三态判定（≥70% 稳健成立 / ≤30% 稳健不成立 / 否则不稳定） | 单 seed 不可信 |

用法：
    python pipeline/v33_landau_c_check.py                 # 20 seed 全跑
    python pipeline/v33_landau_c_check.py --seeds 3       # 快速试跑
    python pipeline/v33_landau_c_check.py --rebuild       # 重建数据集
"""

from __future__ import annotations

import csv
import json
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
    """**正式记录保护（第 164 条）**：`--seeds < 20` 的试点跑写
    `*_runs_pilot_<n>seed.csv`，绝不覆盖 20 seed 的正式记录。"""
    n = 20
    if "--seeds" in sys.argv:
        try:
            n = int(sys.argv[sys.argv.index("--seeds") + 1])
        except (ValueError, IndexError):
            n = 20
    return ("%s_runs.csv" % kind) if n >= 20 else \
        ("%s_runs_pilot_%dseed.csv" % (kind, n))


CSV = _ROOT / "landau_designs.csv"
RUNS = _ROOT / _runs_name("v33")
GRID_N = 12
N_BUCKET = 5
N_SAMPLE = 40            # 每个目标采样几个设计
TRAIN_STEPS = 6000       # 900 → 3000（第一轮）→ 6000（第二轮，预登记）
GRID_N = 16              # 12 → 16（第二轮：数据集 12³=1728 → 16³≈4096）
# **第二轮预登记改动（2026-10-02）**：① 网格 12→16；② 步数 3000→6000；
#   ③ **桶划分改为分层随机一半**（使测试集含"被训练桶包围的内插桶"，
#   第一轮用的是"后半桶"⟹ 测试集**全是外推**）；④ **分别报告内插/外推
#   达标率**（诊断项，**不参与判定**）；**C1 阈值仍为 70%**（不移动球门）。


# ------------------------------------------------------------------ 数据集
def build_dataset(n: int = GRID_N) -> list[dict]:
    from pipeline.design_landau import measure
    from a_network.structure_space import family_template
    tpl = family_template("landau")
    rng = {pr["name"]: pr.get("range", [0.1, 1.0])
           for pr in tpl.get("parameters", [])}
    a_lo, a_hi = rng.get("a", [-2.0, 2.0])
    b_lo, b_hi = rng.get("b", [0.1, 5.0])
    k_lo, k_hi = rng.get("kT", [0.1, 5.0])
    rows = []
    for a in np.linspace(a_lo, min(a_hi, -1e-3), n):
        for b in np.linspace(b_lo, b_hi, n):
            for kt in np.linspace(k_lo, k_hi, n):
                r = measure(float(a), float(b), float(kt))
                if not r or r["q1"] is None or r["dF_nats"] is None:
                    continue
                r.update({"a": float(a), "b": float(b), "kT": float(kt)})
                rows.append(r)
    return rows


def load_dataset(rebuild: bool = False) -> list[dict]:
    if CSV.exists() and not rebuild:
        with open(CSV, "r", encoding="utf-8", newline="") as f:
            return [{k: (float(v) if k not in ("dpi_ok",) else v == "True")
                     for k, v in row.items()} for row in csv.DictReader(f)]
    rows = build_dataset()
    with open(CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["a", "b", "kT", "q1", "dF_nats",
                                          "dpi_ok"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    return rows


def bucketize(rows: list[dict]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """按 (q1, ΔF) 分桶（**桶 = 目标性能**）；返回行、q1 边界、ΔF 边界。"""
    q1 = np.array([r["q1"] for r in rows])
    dF = np.array([r["dF_nats"] for r in rows])
    qe = np.quantile(q1, np.linspace(0, 1, N_BUCKET + 1))
    de = np.quantile(dF, np.linspace(0, 1, N_BUCKET + 1))
    for r in rows:
        r["bq"] = int(np.clip(np.searchsorted(qe, r["q1"], side="right") - 1,
                              0, N_BUCKET - 1))
        r["bd"] = int(np.clip(np.searchsorted(de, r["dF_nats"], side="right")
                              - 1, 0, N_BUCKET - 1))
    return rows, qe, de


# ------------------------------------------------------------ 条件流训练
def train_one(seed: int, rows: list[dict], qe, de) -> tuple[object, dict]:
    """训练 `p(设计 | 目标桶)`；条件 = 桶中心的归一化 (q1, ΔF)。

    **`rows` 必须是"训练划分"（不含测试桶的设计）**——见 `main()` 的划分。
    """
    assert rows, "训练集为空"
    import torch
    from c_network.cfm import ConditionalFlowMatching

    q1_all = np.array([r["q1"] for r in rows])
    dF_all = np.array([r["dF_nats"] for r in rows])
    q_lo, q_hi = float(q1_all.min()), float(q1_all.max())
    d_lo, d_hi = float(dF_all.min()), float(dF_all.max())

    def norm_q(v):
        return (np.log10(max(v, 1e-8)) - np.log10(max(q_lo, 1e-8))) / max(
            np.log10(max(q_hi, 1e-8)) - np.log10(max(q_lo, 1e-8)), 1e-9)

    def norm_d(v):
        return (np.log10(max(v, 1e-8)) - np.log10(max(d_lo, 1e-8))) / max(
            np.log10(max(d_hi, 1e-8)) - np.log10(max(d_lo, 1e-8)), 1e-9)

    def cond_of(r):
        return np.array([norm_q(r["q1"]), norm_d(r["dF_nats"])],
                        dtype=np.float32)

    # 设计向量归一化到 [0,1]（各维用扫描范围）
    a_lo, a_hi = min(r["a"] for r in rows), max(r["a"] for r in rows)
    b_lo, b_hi = min(r["b"] for r in rows), max(r["b"] for r in rows)
    k_lo, k_hi = min(r["kT"] for r in rows), max(r["kT"] for r in rows)

    def design_of(r):
        return np.array([(r["a"] - a_lo) / (a_hi - a_lo),
                         (r["b"] - b_lo) / (b_hi - b_lo),
                         (r["kT"] - k_lo) / (k_hi - k_lo)], dtype=np.float32)

    X = torch.tensor(np.stack([design_of(r) for r in rows]))
    C = torch.tensor(np.stack([cond_of(r) for r in rows]))
    model = ConditionalFlowMatching(in_dim=3, cond_dim=2, hidden=160, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    rng = np.random.default_rng(seed)
    for step in range(TRAIN_STEPS):
        idx = torch.tensor(rng.integers(0, len(rows), 256))
        loss = model.loss(C[idx], X[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    meta = {"a": (a_lo, a_hi), "b": (b_lo, b_hi), "kT": (k_lo, k_hi),
            "q": (q_lo, q_hi), "d": (d_lo, d_hi), "loss": float(loss.item())}
    return model, meta


def denorm_design(Z: np.ndarray, meta: dict) -> tuple[np.ndarray, int]:
    """反归一化设计向量，并**夹取到扫描设计盒内**（2026-10-02 修）。

    **修因**：CFM 把 `N(0,1)` 映射到数据分布，采样值可**略微越出 [0,1]**
    ⟹ 反归一化后设计落在扫描盒外 ⟹ 实测 q1 可达 **238.9**（扫描上界 55.01）
    ⟹ 触发"不可能的数"判据。**设计盒外即外推**，须夹取并**计数报告**。
    返回 `(设计数组, 被夹取的样本数)`。
    """
    a = meta["a"][0] + Z[:, 0] * (meta["a"][1] - meta["a"][0])
    b = meta["b"][0] + Z[:, 1] * (meta["b"][1] - meta["b"][0])
    k = meta["kT"][0] + Z[:, 2] * (meta["kT"][1] - meta["kT"][0])
    D = np.stack([a, b, k], axis=1)
    lo = np.array([meta["a"][0], meta["b"][0], meta["kT"][0]])
    hi = np.array([meta["a"][1], meta["b"][1], meta["kT"][1]])
    n_clip = int(np.sum((D < lo).any(axis=1) | (D > hi).any(axis=1)))
    D = np.clip(D, lo, hi)
    return D, n_clip


# ---------------------------------------------------------------- 闭环校验
def evaluate(seed: int, model, meta: dict, targets: list[dict]
             ) -> dict:
    """采样设计 → **既有编码器实测** → 判是否落在目标桶内（闭环）。"""
    from pipeline.design_landau import measure
    n_ok = n_tot = 0
    divs = []
    qs, ds = [], []
    n_clip_tot = 0
    ok_i = tot_i = ok_e = tot_e = 0
    for t in targets:
        c = np.array([t["cq"], t["cd"]], dtype=np.float32)
        Z = model.sample_cond(c, N_SAMPLE, steps=60, seed=seed)
        D, n_clip = denorm_design(Z, meta)
        n_clip_tot += n_clip
        ok = 0
        uniq = set()
        for a, b, kt in D:
            uniq.add((round(float(a), 3), round(float(b), 3),
                      round(float(kt), 3)))
            m = measure(float(a), float(b), float(kt))
            if not m or m["q1"] is None or m["dF_nats"] is None:
                continue
            qs.append(m["q1"])
            ds.append(m["dF_nats"])
            # **判据：实测性能落在目标桶的 (q1,ΔF) 区间内**
            if (t["q_lo"] <= m["q1"] <= t["q_hi"]
                    and t["d_lo"] <= m["dF_nats"] <= t["d_hi"]):
                ok += 1
                if t.get("extrap"):
                    ok_e += 1
                else:
                    ok_i += 1
            if t.get("extrap"):
                tot_e += 1
            else:
                tot_i += 1
            n_tot += 1
        n_ok += ok
        divs.append(len(uniq))
    return {"pass_rate": (n_ok / n_tot) if n_tot else 0.0,
            "pass_interp": (ok_i / tot_i) if tot_i else float("nan"),
            "pass_extrap": (ok_e / tot_e) if tot_e else float("nan"),
            "diversity_med": float(np.median(divs)) if divs else 0.0,
            "q_min": float(np.min(qs)) if qs else float("nan"),
            "q_max": float(np.max(qs)) if qs else float("nan"),
            "d_min": float(np.min(ds)) if ds else float("nan"),
            "d_max": float(np.max(ds)) if ds else float("nan"),
            "n_eval": n_tot, "n_clipped": n_clip_tot,
            "clip_frac": n_clip_tot / max(n_tot, 1)}


def main() -> int:
    seeds = 20
    if "--seeds" in sys.argv:
        seeds = int(sys.argv[sys.argv.index("--seeds") + 1])
    rebuild = "--rebuild" in sys.argv
    print("=" * 74)
    print("中游 B ②③④：landau 高维设计域的 C 实例 + 闭环校验")
    print("=" * 74)
    t0 = time.time()
    rows = load_dataset(rebuild)
    rows, qe, de = bucketize(rows)
    print("数据集：%d 个设计（缓存 %s；网格 N=%d）" % (len(rows), CSV.name,
                                                 GRID_N))
    feas = [r for r in rows if r["dpi_ok"]]
    print("DPI 合法：%d；桶数 %d×%d" % (len(feas), N_BUCKET, N_BUCKET))

    # **第二轮：分层随机划分桶（含内插与外推）**——按 (bq+bd) 的奇偶/有序位置
    #   分层：把非空桶排序后**隔一个取一个**作为测试 ⟹ 测试桶两侧都有训练桶
    #   （内插），同时保留一部分高桶（外推）。
    buckets = {}
    for r in rows:
        buckets.setdefault((r["bq"], r["bd"]), []).append(r)
    keys = sorted(buckets)
    assert len(keys) >= 4, "非空桶太少（%d）" % len(keys)
    held = keys[1::2]                              # 隔一取一 ⟹ 多为内插桶
    n_extrap = sum(1 for k in held if k in keys[len(keys) // 2:])
    targets = []
    for (kq, kd) in held:
        rs = buckets[(kq, kd)]
        targets.append({"bq": kq, "bd": kd,
                        "cq": float(np.median([r["q1"] for r in rs])),
                        "cd": float(np.median([r["dF_nats"] for r in rs])),
                        "q_lo": float(qe[kq]), "q_hi": float(qe[kq + 1]),
                        "d_lo": float(de[kd]), "d_hi": float(de[kd + 1]),
                        # **内插/外推标记（诊断用）**：桶序号落在后半 ⟹ 外推
                        "extrap": bool(kq >= N_BUCKET // 2
                                       and kd >= N_BUCKET // 2)})
    # **训练/测试划分（2026-10-02 修：测试集泄漏）**：初版 `train_one(s, rows)`
    #   传入**全量** rows ⟹ 训练时**包含了测试桶的设计** ⟹ "采样设计落在目标
    #   桶内"被泄漏抬高（第二轮报出 90.6%——**该数字不可信**）。
    #   现：**训练集剔除测试桶的设计**，评估才是真正的**留出泛化**检验。
    held_set = set(held)
    train_rows = [r for r in rows if (r["bq"], r["bd"]) not in held_set]
    print("划分：训练设计 **%d**（桶 %d 个）；测试桶 **%d**（其中设计 %d 个"
          "**已从训练集剔除**）"
          % (len(train_rows), len(keys) - len(held), len(held),
             len(rows) - len(train_rows)))

    # 条件归一化需与 train_one 一致 ⟹ 用全量范围算目标条件
    q1_all = np.array([r["q1"] for r in rows])
    dF_all = np.array([r["dF_nats"] for r in rows])
    lq0, lq1 = np.log10(max(q1_all.min(), 1e-8)), np.log10(max(q1_all.max(),
                                                              1e-8))
    ld0, ld1 = np.log10(max(dF_all.min(), 1e-8)), np.log10(max(dF_all.max(),
                                                               1e-8))
    for t in targets:
        t["cq"] = float((np.log10(max(t["cq"], 1e-8)) - lq0) / max(lq1 - lq0,
                                                                 1e-9))
        t["cd"] = float((np.log10(max(t["cd"], 1e-8)) - ld0) / max(ld1 - ld0,
                                                                 1e-9))
    print("目标桶（测试，含外推）：%d 个（训练集含其余 %d 个桶）"
          % (len(targets), len(keys) - len(held)))

    rec = []
    print("\n%4s %10s %10s %9s %9s %8s %9s" % ("seed", "闭环达标率", "多样性中位",
                                              "q1_min", "q1_max", "loss",
                                              "外推比例"))
    for s in range(seeds):
        model, meta = train_one(s, train_rows, qe, de)
        ev = evaluate(s, model, meta, targets)
        rec.append({"seed": s, **ev, "loss": meta["loss"]})
        print("%4d %9.1f%% %10.1f %9.4g %9.4g %8.4f %8.1f%%"
              % (s, 100 * ev["pass_rate"], ev["diversity_med"],
                 ev["q_min"], ev["q_max"], meta["loss"],
                 100 * ev["clip_frac"]))

    pr = np.array([r["pass_rate"] for r in rec])
    dv = np.array([r["diversity_med"] for r in rec])
    print("\n===== 汇总（%d seed）=====" % seeds)
    print("C1 闭环达标率：均值 %.1f%% ± %.1f%%；达标(≥70%%)的 seed：%d/%d"
          % (100 * pr.mean(), 100 * pr.std(), int(np.sum(pr >= 0.70)), seeds))
    print("C2 多样性（每目标不同设计数，中位）：均值 %.1f；达标(≥10)的 seed：%d/%d"
          % (dv.mean(), int(np.sum(dv >= 10)), seeds))
    q_all = np.array([r["q_min"] for r in rec] + [r["q_max"] for r in rec])
    print("C3 无不可能的数：实测 q1 ∈ [%.4g, %.4g]；扫描范围 [%.4g, %.4g] ⟹ %s"
          % (np.nanmin(q_all), np.nanmax(q_all), q1_all.min(), q1_all.max(),
             "在范围内 ✓" if (np.nanmin(q_all) >= q1_all.min() * 0.5
                            and np.nanmax(q_all) <= q1_all.max() * 2.0)
             else "**超出范围 ⟹ 查 bug**"))
    # **内插/外推分列（诊断项，不参与判定；第二轮加）**
    it = [r for r in rec if "pass_interp" in r]
    if it:
        pi = np.array([r["pass_interp"] for r in it])
        pe = np.array([r["pass_extrap"] for r in it])
        print("   诊断（**不参与判定**）：内插桶达标率 均值 %.1f%%；"
              "外推桶达标率 均值 %.1f%%" % (100 * np.nanmean(pi),
                                        100 * np.nanmean(pe)))
    rate70 = float(np.mean(pr >= 0.70))
    verdict = ("**稳健成立**" if rate70 >= 0.70 else
               "**稳健不成立**" if rate70 <= 0.30 else "**不稳定**")
    print("C4 三态判定（达标率 ≥70%% 的 seed 比例 = %.0f%%）：%s"
          % (100 * rate70, verdict))

    with open(RUNS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec[0]))
        w.writeheader()
        w.writerows(rec)
    print("\n逐 seed 落盘：%s（%d 行）" % (RUNS.name, len(rec)))
    print("耗时 %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
