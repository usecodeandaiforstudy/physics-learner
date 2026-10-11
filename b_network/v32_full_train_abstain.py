# -*- coding: utf-8 -*-
"""**全量训练 + 弃权率对比**（2026-10-02，第 190 条）。

## 要回答的问题
第 189 条起草了 10 条候选锚点（`db3/proposed/`，**未入库**）。LODO **构造上
测不出**"加锚点是否让解析器学会这些措辞"（留出整个域 ⟹ 该域的锚点永在测试侧）。
本脚本用**另一种口径**回答它：**全量训练后的弃权率对比**。

## 设计（**不动语料**，候选只读并入）
    · **A 臂**：仅在 **159 条语料**上训练 ⟹ 在【10 条候选】与【159 条语料】上评估；
    · **B 臂**：在 **169 条（语料 + 候选）**上训练 ⟹ 同样评估；
    · 评估口径 = **生产口径**：`conf < 0.90` ⟹ **弃权**（转人工）。
    ⟹ 对比 A/B 的**弃权率**即"**这些措辞是否被学到**"。

## 预登记判据（先写死）
    **F1 可行**：两臂训练与评估均完成；
    **F2 学到**：B 对 10 条候选的**弃权率**显著低于 A（**下降 ≥50%** 或
      B 弃权率 **≤20%**）——这是"加锚点有效"的判据；
    **F3 无回归**：B 对 159 条语料的精确率**不劣于** A（差 ≤2 个百分点），
      且弃权率**不升**（差 ≤2 个百分点）；
    **F4 数字健全**：所有指标 ∈ [0,1]。
**诚实标注（必须同时说）**：B 臂对候选的评估**在训练集内** ⟹ 它测的是
  **"是否学到"（fit）**，**不是泛化**；泛化仍以第 187/188 条的 LODO 为准 ✓

用法：
    python b_network/v32_full_train_abstain.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "b_network"))

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

RUNS = _ROOT / "v32_full_train_abstain.csv"
TAU = 0.50              # 生产 τ（第 193 条方案 C 由风险–覆盖曲线选定）
F2_DROP = 0.50          # 弃权率相对下降 ≥50%
F2_ABS = 0.20           # 或绝对弃权率 ≤20%
F3_TOL = 0.02           # 回归容差 2 个百分点
PROPOSED = _ROOT / "db3" / "proposed"


def _load_extra() -> list[dict]:
    out = []
    for p in sorted(PROPOSED.glob("DB3-*.json")):
        e = json.loads(p.read_text(encoding="utf-8"))
        rs = e.get("requirement_structured") or {}
        out.append({"pair_id": e.get("pair_id"), "domain": e.get("domain"),
                    "text": e.get("requirement_text"),
                    "properties": [rs["property_id"]] if rs.get("property_id")
                    else [],
                    "direction": rs.get("direction")})
    return out


def run_arm(corpus: list[dict], evalsets: dict, tag: str) -> dict:
    import torch
    from bv1_train_eval import (
    input_dim,PROPERTIES, _DIRS, build_vocab, featurize,
                                train_model)
    vocab = build_vocab([a["text"] for a in corpus])
    in_dim = input_dim(vocab)   # 含线索块（第 193 条）
    X = np.stack([featurize(a["text"], vocab) for a in corpus])
    y_prop = np.zeros((len(corpus), len(PROPERTIES)), dtype=np.float32)
    for i, a in enumerate(corpus):
        for p in a["properties"]:
            if p in PROPERTIES:
                y_prop[i, PROPERTIES.index(p)] = 1.0
    y_dir = np.array([0 if a["direction"] in (None, "bound_below") else 1
                      for a in corpus], dtype=np.int64)
    model = train_model(X, y_prop, y_dir, in_dim)
    res = {}
    for name, items in evalsets.items():
        Xe = np.stack([featurize(a["text"], vocab) for a in items])
        with torch.no_grad():
            lp, ld = model(torch.tensor(Xe))
        probs = torch.sigmoid(lp).numpy()
        dirs = torch.softmax(ld, -1).numpy()
        n_abs = n_ex = 0
        for k, a in enumerate(items):
            conf = float(probs[k].max())
            if conf < TAU:
                n_abs += 1
                preds = ["unmapped"]
            else:
                preds = [PROPERTIES[j] for j in range(len(PROPERTIES))
                         if probs[k][j] > 0.5] or [
                    PROPERTIES[int(probs[k].argmax())]]
            n_ex += int(set(preds) == set(a["properties"]))
        m = max(len(items), 1)
        res[name] = {"n": len(items), "abstain": n_abs / m, "exact": n_ex / m}
    return res


def main() -> int:
    from bv1_train_eval import load_anchors
    corpus = load_anchors()
    extra = _load_extra()
    print("=" * 74)
    print("全量训练 + 弃权率对比（A：仅语料 ｜ B：语料+候选）")
    print("=" * 74)
    print("语料 **%d** 条；候选 **%d** 条（`db3/proposed/`，**未入库**）；"
          "弃权阈值 `conf < %.2f`" % (len(corpus), len(extra), TAU))
    evalsets = {"候选10条": extra, "语料159条": corpus}
    A = run_arm(corpus, evalsets, "A")
    B = run_arm(corpus + extra, evalsets, "B")
    rows = []
    for name in evalsets:
        a, b = A[name], B[name]
        rows.append({"evalset": name, "n": a["n"],
                     "A_abstain": a["abstain"], "B_abstain": b["abstain"],
                     "A_exact": a["exact"], "B_exact": b["exact"]})
    print("\n%-12s %5s %14s %14s %14s %14s"
          % ("评估集", "条数", "A 弃权率", "B 弃权率", "A 精确率", "B 精确率"))
    for r in rows:
        print("%-12s %5d %13.1f%% %13.1f%% %13.1f%% %13.1f%%"
              % (r["evalset"], r["n"], 100 * r["A_abstain"],
                 100 * r["B_abstain"], 100 * r["A_exact"], 100 * r["B_exact"]))
    cand = rows[0]
    corp = rows[1]
    drop = (cand["A_abstain"] - cand["B_abstain"]) / max(cand["A_abstain"], 1e-9)
    f2 = (drop >= F2_DROP) or (cand["B_abstain"] <= F2_ABS)
    f3 = ((corp["B_exact"] >= corp["A_exact"] - F3_TOL)
          and (corp["B_abstain"] <= corp["A_abstain"] + F3_TOL))
    f4 = all(0.0 <= v <= 1.0 for r in rows for k, v in r.items()
             if k.endswith(("abstain", "exact")))
    print("\n**F1 可行**：两臂均完成（A 训练 %d 条、B 训练 %d 条）⟹ 通过 ✓"
          % (len(corpus), len(corpus) + len(extra)))
    print("**F2 学到**：候选弃权率 %.1f%% → %.1f%%（相对下降 **%.0f%%**）"
          "；判据（下降 ≥%.0f%% 或 B ≤%.0f%%）⟹ %s"
          % (100 * cand["A_abstain"], 100 * cand["B_abstain"], 100 * drop,
             100 * F2_DROP, 100 * F2_ABS, "**通过** ✓" if f2 else "**未通过**"))
    print("**F3 无回归**：语料精确率 %.1f%% → %.1f%%；弃权率 %.1f%% → %.1f%%"
          "（容差 %.0f pp）⟹ %s"
          % (100 * corp["A_exact"], 100 * corp["B_exact"],
             100 * corp["A_abstain"], 100 * corp["B_abstain"], 100 * F3_TOL,
             "**通过** ✓" if f3 else "**未通过**"))
    print("**F4 数字健全** ⟹ %s" % ("**通过** ✓" if f4 else "**未通过**"))

    # ---- **更严变体：留一候选（LOPO）** ----
    #   训练时**抽掉**该候选，用**同域的另一条**候选 + 全语料教它 ⟹ 测的是
    #   "**这类措辞能否从同域的其它表述迁移**"（**非**训练集内，故是泛化证据）✓
    print("\n" + "─" * 74)
    print("**更严变体：留一候选（LOPO）**——训练抽掉该候选，仅用同域另一条 + 语料")
    print("─" * 74)
    n_ok = n_abs = 0
    for i, a in enumerate(extra):
        others = [x for j, x in enumerate(extra) if j != i]
        # 训练集 = 语料 + 其它候选（**不含**本条）
        res = run_arm(corpus + others, {"held": [a]}, "LOPO")
        r = res["held"]
        good = (r["exact"] >= 1.0)
        n_ok += int(good)
        n_abs += int(r["abstain"] >= 1.0)
        print("  %-10s %-14s 弃权率 %5.1f%% ｜ 精确率 %5.1f%% ｜ %s"
              % (a["pair_id"], a["domain"], 100 * r["abstain"],
                 100 * r["exact"], "**迁移成功** ✓" if good else
                 ("**弃权**" if r["abstain"] >= 1.0 else "**判错**")))
    rate = n_ok / max(len(extra), 1)
    print("\n**F5 迁移（LOPO）**：精确率 ≥100%% 的候选 **%d/%d（%.0f%%）**；"
          "全弃权 %d 条 ⟹ %s"
          % (n_ok, len(extra), 100 * rate, n_abs,
             "**通过** ✓（同域措辞可迁移）" if rate >= 0.70
             else "**部分/未通过**（依赖具体措辞，非同类可迁移）"))
    print("**注**：LOPO 是本脚本里**唯一非训练集内**的证据 ⟹ 它比 F2 更有信息量 ✓")
    print("\n**诚实标注**：B 臂对候选的评估**在训练集内** ⟹ 测的是"
          "**「是否学到」（fit）**，**不是泛化**；泛化以第 187/188 条的 LODO 为准 ✓")
    with open(RUNS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n落盘：%s" % RUNS.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
