# -*- coding: utf-8 -*-
"""B v1 训练与评估（LOOCV，34 锚点）+ q 级集成验收。

诚实边界（报告必读）：训练集只有 34 个锚点——LOOCV 只能给出**锚点分布内**的
估计，不能证明锚点外的泛化；B v1 本阶段的真实产出 = 管线 + abstain 机制 +
评估协议，数据扩展靠模式③（LLM API 批量生成，待建）。

输出：
  - LOOCV：属性集合精确率 / 逐属性 F1 / 方向准确率 / unmapped 召回；
  - abstain 扫描：阈值 × 覆盖率 × 精确率（选择部署阈值）；
  - 集成验收：Bv1（tau=0 与 tau=0.7）经 matcher→composer 的规则级 roundtrip
    vs v0 基线（v0 = 33/34 完全 + 1 部分）；
  - 保存 `b_network/bv1_model.pt`（全量 34 锚点训练，部署用）。

用法（项目根目录）：python b_network/bv1_train_eval.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

sys.stdout.reconfigure(encoding="utf-8")

from b_network.bv1_dataset import PROPERTIES, load_anchors  # noqa: E402
from b_network.bv1_parse import (  # noqa: E402
    Bv1Model,
    Bv1Parser,
    build_vocab,
    featurize,
    input_dim,
    save_parser,
)
from b_network.composer import Composer  # noqa: E402
from b_network.requirement_parser import RequirementParser  # noqa: E402
from b_network.rule_matcher import RuleMatcher  # noqa: E402

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "bv1_model.pt")
_DIRS = ["bound_below", "bound_above"]


def train_model(Xs, y_prop, y_dir, in_dim, steps=600, lr=1e-2, seed=0) -> Bv1Model:
    torch.manual_seed(seed)
    model = Bv1Model(in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(Xs)
    YP = torch.tensor(y_prop, dtype=torch.float32)
    YD = torch.tensor(y_dir, dtype=torch.long)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        lp, ld = model(X)
        loss = bce(lp, YP) + ce(ld, YD)
        loss.backward()
        opt.step()
    model.eval()
    return model


def _roundtrip(parser, matcher, composer, anchors) -> dict:
    exact = partial = miss = 0
    for a in anchors:
        expect = set(a["rule_ids"]) - {"R-NOMAP-000"}
        req = parser.parse(a["text"])
        got = set(matcher.match(req))
        spec = composer.compose(req)
        if not expect:
            if spec.unmapped:
                exact += 1
            else:
                miss += 1
            continue
        if got == expect:
            exact += 1
        elif got and got <= expect:
            partial += 1
        else:
            miss += 1
    return {"exact": exact, "partial": partial, "miss": miss,
            "total": len(anchors)}


def main() -> int:
    anchors = load_anchors()
    n = len(anchors)
    vocab = build_vocab([a["text"] for a in anchors])
    in_dim = input_dim(vocab)   # **含线索块（第 193 条）**，不是 len(vocab) ✗
    Xs = np.stack([featurize(a["text"], vocab) for a in anchors])
    y_prop = np.zeros((n, len(PROPERTIES)), dtype=np.float32)
    for i, a in enumerate(anchors):
        for p in a["properties"]:
            y_prop[i, PROPERTIES.index(p)] = 1.0
    y_dir = np.array([0 if a["direction"] in (None, "bound_below") else 1
                      for a in anchors], dtype=np.int64)

    # ---- LOOCV ----
    fold = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        model = train_model(Xs[tr], y_prop[tr], y_dir[tr], in_dim)
        with torch.no_grad():
            lp, ld = model(torch.tensor(Xs[i:i + 1]))
        probs = torch.sigmoid(lp)[0].numpy()
        preds = [PROPERTIES[k] for k in range(len(PROPERTIES)) if probs[k] > 0.5]
        if not preds:
            preds = [PROPERTIES[int(probs.argmax())]]
        conf = float(probs.max())
        pred_dir = _DIRS[int(torch.softmax(ld, -1)[0].argmax())]
        true = anchors[i]
        fold.append({
            "pair_id": true["pair_id"],
            "exact": set(preds) == set(true["properties"]),
            "conf": conf,
            "unmapped_ok": ("unmapped" in true["properties"])
                           == ("unmapped" in preds),
            "dir_ok": None if "unmapped" in true["properties"]
                      else (pred_dir == true["direction"]),
        })

    set_exact = sum(f["exact"] for f in fold) / n
    unmapped_ok_rate = sum(f["unmapped_ok"] for f in fold) / n
    dir_vals = [f["dir_ok"] for f in fold if f["dir_ok"] is not None]
    dir_acc = sum(dir_vals) / len(dir_vals) if dir_vals else 0.0

    print("================ B v1 训练评估（LOOCV，%d 锚点） ================" % n)
    from collections import Counter
    dom = Counter(a["domain"] for a in anchors)
    dirdist = Counter(a["direction"] for a in anchors)
    print("锚点域分布（%d 域）：%s" % (len(dom), dict(dom)))
    print("方向标签分布（物理向，口径修正后）：%s" % dict(dirdist))
    print("特征维度：%d（关键词先验 + 单位/数字 + 字符 3-gram/英文词，传导式）" % in_dim)
    print("属性集合精确率：%d/%d（%.1f%%）" % (sum(f["exact"] for f in fold), n,
                                               100 * set_exact))
    print("unmapped 判定一致率：%.1f%%" % (100 * unmapped_ok_rate))
    print("方向准确率（非 unmapped）：%d/%d（%.1f%%）" % (sum(dir_vals),
                                                       len(dir_vals),
                                                       100 * dir_acc))
    print("-- 逐对（仅列错判）--")
    for f in fold:
        if not f["exact"] or f["dir_ok"] is False:
            print("  %s: exact=%s dir_ok=%s conf=%.2f"
                  % (f["pair_id"], f["exact"], f["dir_ok"], f["conf"]))

    # ---- abstain 扫描 ----
    print("\n-- abstain 扫描（τ × 覆盖率 × 精确率）--")
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        cov = [f for f in fold if f["conf"] >= tau]
        acc = (sum(f["exact"] for f in cov) / len(cov)) if cov else 0.0
        print("  τ=%.2f  覆盖率 %2d/%-2d（%.0f%%）  精确率 %.1f%%"
              % (tau, len(cov), n, 100 * len(cov) / n, 100 * acc))

    # ---- 最终模型（全量训练，部署用） ----
    final = train_model(Xs, y_prop, y_dir, in_dim, steps=800, seed=0)
    save_parser(final, vocab, 0.7, _MODEL_PATH)
    print("\n最终模型已保存：%s（tau=0.7）" % _MODEL_PATH)

    # ---- q 级集成验收：v0 vs v1（tau=0 全模型 / tau=0.7 带回退） ----
    matcher = RuleMatcher()
    composer = Composer(matcher)
    v0 = _roundtrip(RequirementParser(), matcher, composer, anchors)
    v1_full = _roundtrip(Bv1Parser(final, vocab, tau_abstain=0.0), matcher,
                         composer, anchors)
    v1_gate = _roundtrip(Bv1Parser(final, vocab, tau_abstain=0.7), matcher,
                         composer, anchors)
    print("\n================ q 级集成验收（规则级 roundtrip） ================")
    for name, r in (("v0 基线", v0), ("v1（tau=0，全模型）", v1_full),
                    ("v1（tau=0.7，带回退）", v1_gate)):
        print("  %-22s 完全 %2d  部分 %d  错判/未命中 %d"
              % (name, r["exact"], r["partial"], r["miss"]))

    report = {
        "note": "B v1 LOOCV + 集成验收。诚实边界：34 锚点只估计分布内表现，"
                "分布外泛化待模式③扩数据。",
        "n_anchors": n, "feature_dim": in_dim,
        "loocv": {"set_exact": set_exact, "unmapped_ok_rate": unmapped_ok_rate,
                  "direction_acc": dir_acc},
        "roundtrip": {"v0": v0, "v1_tau0": v1_full, "v1_tau07": v1_gate},
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "bv1_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n报告已写入 b_network/bv1_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
