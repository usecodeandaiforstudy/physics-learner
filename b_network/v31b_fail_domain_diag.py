# -*- coding: utf-8 -*-
"""**LODO 全错域失败模式定位**（2026-10-02，第 188 条）。

## 背景
严格 LODO（留一域，52 域）中 **4 个域全错**：
`biomedical` / `communication` / `general` / `geophysics`（第 187 条）。

## 本脚本对每个全错域给出**可判定的证据**
① 锚点原文、**真实**属性/方向、**预测**属性/方向与置信度；
② **词表覆盖率**（归纳式词表由训练域构建）——该域文本的 n-gram/词有多少
   落在训练词表内 ⟹ **区分"词表外术语"与"模型/标注问题"**；
③ **传导式 vs 归纳式**对照——若传导式对、归纳式错 ⟹ **词表问题**；
   两者都错 ⟹ **模型/标注口径问题**（非词表）；
④ 判定标签（预登记）：
   · **V1 词表外**：归纳式错且传导式对，或归纳覆盖率显著低于训练域中位；
   · **V2 标注口径**：两口径都错且覆盖率不低 ⟹ 属性组合/方向约定与训练分布不同；
   · **V3 样本过少**：该域留出锚点数 = 1（无法区分 0%/100%）。

用法：
    python b_network/v31b_fail_domain_diag.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "b_network"))

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

BAD = ("biomedical", "communication", "general", "geophysics")


def main() -> int:
    import torch
    from bv1_train_eval import (
    input_dim,PROPERTIES, _DIRS, build_vocab, featurize,
                                load_anchors, train_model)
    anchors = load_anchors()
    print("=" * 74)
    print("LODO 全错域失败模式定位：%s" % ", ".join(BAD))
    print("=" * 74)

    # 训练域（除 4 个全错域之外）用于估计"正常词表覆盖率"的量级
    train_pool = [a for a in anchors if a["domain"] not in BAD]
    vocab_tr = build_vocab([a["text"] for a in train_pool])
    vocab_all = build_vocab([a["text"] for a in anchors])

    def coverage(text: str, vocab) -> float:
        """该文本的特征有多少落在词表内（按非零特征比例）。"""
        x = np.asarray(featurize(text, vocab), dtype=float)
        nz = int((x > 0).sum())
        full = np.asarray(featurize(text, vocab_all), dtype=float)
        nz_all = int((full > 0).sum())
        return (nz / nz_all) if nz_all else float("nan")

    cov_pool = [coverage(a["text"], vocab_tr) for a in train_pool]
    print("对照：**训练域**锚点的归纳式词表覆盖率 中位 **%.2f**"
          % float(np.median(cov_pool)))

    for dom in BAD:
        te = [a for a in anchors if a["domain"] == dom]
        tr = [a for a in anchors if a["domain"] != dom]
        y_prop = np.zeros((len(tr), len(PROPERTIES)), dtype=np.float32)
        for i, a in enumerate(tr):
            for p in a["properties"]:
                y_prop[i, PROPERTIES.index(p)] = 1.0
        y_dir = np.array([0 if a["direction"] in (None, "bound_below") else 1
                          for a in tr], dtype=np.int64)
        print("\n" + "─" * 74)
        print("域 **%s**（留出锚点 %d 个）" % (dom, len(te)))
        for mode in ("inductive", "transductive"):
            vocab = vocab_tr if mode == "inductive" else vocab_all
            Xtr = np.stack([featurize(a["text"], vocab) for a in tr])
            Xte = np.stack([featurize(a["text"], vocab) for a in te])
            model = train_model(Xtr, y_prop, y_dir, input_dim(vocab))
            with torch.no_grad():
                lp, ld = model(torch.tensor(Xte))
            probs = torch.sigmoid(lp).numpy()
            dirs = torch.softmax(ld, -1).numpy()
            tag = "归纳式" if mode == "inductive" else "传导式"
            for k, a in enumerate(te):
                preds = [PROPERTIES[j] for j in range(len(PROPERTIES))
                         if probs[k][j] > 0.5] or [
                    PROPERTIES[int(probs[k].argmax())]]
                pd = _DIRS[int(dirs[k].argmax())]
                ok = set(preds) == set(a["properties"])
                cov = coverage(a["text"], vocab)
                print("  [%s] %s" % (tag, a["pair_id"]))
                print("      文本：%s" % a["text"][:78])
                print("      真值：%s ｜ 方向 %s"
                      % (sorted(a["properties"]), a["direction"]))
                print("      预测：%s ｜ 方向 %s ｜ conf %.2f ｜ %s"
                      % (sorted(preds), pd, float(probs[k].max()),
                         "**对**" if ok else "**错**"))
                print("      归纳词表覆盖率：**%.2f**" % cov)
        # 预登记判定
        covs = [coverage(a["text"], vocab_tr) for a in te]
        med = float(np.median(covs))
        if len(te) == 1:
            verdict = "**V3 样本过少**（仅 1 个留出锚点，0%%/100%% 不可区分）"
        elif med < 0.6 * float(np.median(cov_pool)):
            verdict = "**V1 词表外**（归纳覆盖率 %.2f 显著低于训练域中位 %.2f）" % (
                med, float(np.median(cov_pool)))
        else:
            verdict = ("**V2 标注/模型口径**（覆盖率 %.2f 不低 ⟹ 非词表问题；"
                       "属性组合或方向约定与训练分布不同）" % med)
        props = Counter(p for a in te for p in a["properties"])
        print("  **判定**：%s" % verdict)
        print("  该域属性分布：%s ｜ 锚点数 %d" % (dict(props), len(te)))
    print("\n" + "=" * 74)
    print("**说明**：V1 可用「扩词表/加同义」缓解；V2 需**核对标注口径**（人判）；"
          "V3 需**加锚点**——三者都**不改变**第 187 条的 LODO 结论（42/52 域仍全对）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
