# -*- coding: utf-8 -*-
"""**新锚点的域内迁移（留一）复测——常设阶段**（2026-10-02，第 196–197 条）。

## 它回答什么
"**给某个域补 1–2 条锚点，解析器能不能据此认出该域的同类措辞？**"
（**域内迁移**）——这是第 190 条 LOPO 判"不成立"、第 196 条复测判"成立"的
**同一问题**，本条把它**常设化**，用于**持续观察漂移** ✓

## 口径（与第 194/196 条一致）
**留一（LOO）**：训练用**其余全部锚点**（**含同域兄弟** ✓），测该条。
默认测**ID 最大的 N 条**（= 最近入库的锚点 ✓），可用 `--ids` 指定。
默认只跑**生产特征（v2）**；`--compare-v1` 才同时跑旧特征做对照 ✓

## 判定语义（**测量类**，不构成通过/失败）
跑通即 OK；**并列打印与既定基线的漂移**：
    · 第 196 条基线：v2 **5/10** 精确、top-1 **6/10**、弃权 3；
    · 第 190 条基线（**口径不同，不可混用**）：LOPO v1 1/10。
仅做**健全性断言**（指标 ∈ [0,1]；条数 > 0）✓

用法：
    python b_network/v35_new_anchor_transfer.py               # 最新 10 条
    python b_network/v35_new_anchor_transfer.py --n 5
    python b_network/v35_new_anchor_transfer.py --ids DB3-0196 DB3-0197
    python b_network/v35_new_anchor_transfer.py --compare-v1
"""

from __future__ import annotations

import csv
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

RUNS = _ROOT / "v35_new_anchor_transfer.csv"
TAU = 0.50                      # 生产 τ（第 193 条方案 C）
BASE_EXACT = 5                  # 第 196 条基线（v2，10 条，精确数）
BASE_TOP1 = 6                   # 第 196 条基线（v2，top-1 命中数）


def _feats(mode: str):
    if mode == "v2":
        from features_v2 import dim_v2, featurize_v2
        return featurize_v2, dim_v2
    from features_v2 import dim_legacy, featurize_legacy
    return featurize_legacy, dim_legacy


def main() -> int:
    import torch
    from bv1_train_eval import PROPERTIES, _DIRS, build_vocab, load_anchors, \
        train_model
    anchors = load_anchors()
    # ---- 目标锚点：默认 ID 最大的 N 条 ----
    if "--ids" in sys.argv:
        i0 = sys.argv.index("--ids") + 1
        want = [a for a in sys.argv[i0:] if a.startswith("DB3-")]
    else:
        n = 10
        if "--n" in sys.argv:
            n = int(sys.argv[sys.argv.index("--n") + 1])
        want = [a["pair_id"] for a in
                sorted(anchors, key=lambda x: x["pair_id"])[-n:]]
    modes = ("v2", "v1") if "--compare-v1" in sys.argv else ("v2",)
    print("=" * 74)
    print("新锚点域内迁移（留一 LOO；训练**含同域兄弟**；生产 τ=%.2f）" % TAU)
    print("=" * 74)
    print("语料 **%d** 条；目标 %d 条：%s" % (len(anchors), len(want),
                                          ", ".join(want)))
    if not want:
        print("**无目标锚点** ⟹ 无可测")
        return 0
    rows = []
    for mode in modes:
        featurize, dim_fn = _feats(mode)
        vocab = build_vocab([a["text"] for a in anchors])
        X = np.stack([featurize(a["text"], vocab) for a in anchors])
        y = np.zeros((len(anchors), len(PROPERTIES)), dtype=np.float32)
        for i, a in enumerate(anchors):
            for p in a["properties"]:
                if p in PROPERTIES:
                    y[i, PROPERTIES.index(p)] = 1.0
        yd = np.array([0 if a["direction"] in (None, "bound_below") else 1
                       for a in anchors], dtype=np.int64)
        n_ok = n_abs = n_top1 = 0
        for i, a in enumerate(anchors):
            if a["pair_id"] not in want:
                continue
            tr = [j for j in range(len(anchors)) if j != i]
            model = train_model(X[tr], y[tr], yd[tr], dim_fn(vocab))
            with torch.no_grad():
                lp, ld = model(torch.tensor(X[i:i + 1]))
            probs = torch.sigmoid(lp)[0].numpy()
            dirs = torch.softmax(ld, -1)[0].numpy()
            conf = float(probs.max())
            top1 = PROPERTIES[int(probs.argmax())]
            if conf < TAU:
                preds, abst = ["unmapped"], True
            else:
                preds = [PROPERTIES[k] for k in range(len(PROPERTIES))
                         if probs[k] > 0.5] or [top1]
                abst = False
            exact = set(preds) == set(a["properties"])
            n_ok += int(exact)
            n_abs += int(abst)
            n_top1 += int(top1 in a["properties"])
            rows.append({"mode": mode, "pair_id": a["pair_id"],
                         "domain": a["domain"], "conf": conf,
                         "exact": exact, "abstained": abst,
                         "top1_hit": top1 in a["properties"]})
        print("  [%s] 精确 **%d/%d** ｜ top-1 命中真值 **%d/%d** ｜ 弃权 %d"
              % (mode, n_ok, len(want), n_top1, len(want), n_abs))
        if mode == "v2" and not ("--ids" in sys.argv):
            print("       与第 196 条基线（%d 条）对照：精确 %d→%d、top-1 %d→%d"
                  " ⟹ %s"
                  % (10, BASE_EXACT, n_ok, BASE_TOP1, n_top1,
                     "**无漂移** ✓" if (n_ok == BASE_EXACT
                                       and n_top1 == BASE_TOP1)
                     else "**有漂移**（供观察，非判据）"))
    ok = all(0.0 <= r["conf"] <= 1.0 for r in rows) and len(rows) > 0
    print("健全性：指标 ∈ [0,1] 且条数 > 0 ⟹ %s"
          % ("**通过** ✓" if ok else "**未通过** ✗"))
    with open(RUNS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("落盘：%s（%d 行）" % (RUNS.name, len(rows)))
    print("**语义**：测量类——数值不构成通过/失败，仅用于观察漂移 ✓")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
