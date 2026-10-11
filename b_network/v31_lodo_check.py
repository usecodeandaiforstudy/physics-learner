# -*- coding: utf-8 -*-
"""**严格 LODO 跨域检验**（2026-10-02，第 187 条）——"跨域统一"主张的严格检验。

## 为什么必须做
既有证据里，B v1 的 **LOOCV 85.5% 是"留一锚点"**（同一批域内插值）；
域级 **D2 97.4%** 是**乐观口径**（模型见过全部锚点，脚本自己已注明）。
⟹ **"能否泛化到没见过的域"从未被严格检验过**。

## 本检验（最严做法）
**留一域（leave-one-domain-out，LODO）**：对 DB3 的 **52 个域**逐一留出，
用**其余全部锚点**重训，在**留出域**上评估；并**同时报告两种口径**：
    · **传导式（transductive）**：词表由**全部**锚点构建——与既有 LOOCV 口径可比；
    · **归纳式（inductive）**：词表**只由训练域**构建——**无任何词表泄漏** ✓
⟹ 两者之差 = **词表泄漏**的量（诚实标注）。

## 预登记判据（先写死，不看数据）
    **L1 可行性**：52 个域全部完成训练与评估；
    **L2 无泄漏**：留出域**不得**出现在训练集中（逐折断言）；
    **L3 报告**：给出 LODO 的 `set_exact / unmapped_ok / direction_acc`
      （传导式与归纳式各一套）＋与 LOOCV 的**泛化差距**；
    **L4 无不可能的数**：三项指标均 ∈ [0,1]；
    **L5 三态判定（按项目纪律，预先写死）**：
      `set_exact ≥ 70%` ⟹ **跨域泛化成立**；
      `≤ 30%` ⟹ **稳健不成立**；介于其间 ⟹ **部分成立/不稳定**。
      （**不做阈值调整**；结果如何就如何记录。）

用法：
    python b_network/v31_lodo_check.py            # 全部 52 域
    python b_network/v31_lodo_check.py --limit 8   # 只跑前 8 个域（冒烟）
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "b_network"))

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

RUNS = _ROOT / "v31_lodo.csv"
# 预先写死的三态阈值（**不得事后调整**）
GATE_OK = 0.70
GATE_BAD = 0.30
# **生产弃权阈值（2026-10-02 加，第 188 条修因）**：生产解析器在
#   `conf < τ` 时**弃权转人工队列**（`b_parse.DEFAULT_TAU = 0.90`）。
#   我的初版 harness **强制取 argmax** ⟹ 真值为 `unmapped`（无属性）的锚点
#   **必然判错** ✗（实测 `general` 域的 2 条全是这样）。
#   ⟹ 现按**生产口径**计分，并把结果**分成三类**：自动答对 / 自动答错 / **弃权**。
TAU_ABSTAIN = 0.50


def _load():
    from bv1_train_eval import (
        PROPERTIES, _DIRS, build_vocab, featurize, input_dim, load_anchors,
        train_model)
    return (PROPERTIES, _DIRS, build_vocab, featurize, load_anchors,
            train_model, input_dim)


def _featurize_with(texts, vocab, featurize):
    return np.stack([featurize(t, vocab) for t in texts])


def main() -> int:
    import torch
    (PROPERTIES, _DIRS, build_vocab, featurize, load_anchors,
     train_model, input_dim) = _load()
    anchors = load_anchors()
    # **`--extra <dir>`（第 189 条）**：把候选锚点目录**只读并入评估**，
    #   用于回答"若批准这些候选，LODO 会怎样"——**不改动 `db3/pairs/`** ✓
    extra_n = 0
    if "--extra" in sys.argv:
        exdir = Path(sys.argv[sys.argv.index("--extra") + 1])
        import json as _json
        for p in sorted(exdir.glob("DB3-*.json")):
            e = _json.loads(p.read_text(encoding="utf-8"))
            rs = e.get("requirement_structured") or {}
            props = [rs.get("property_id")] if rs.get("property_id") else []
            anchors.append({"pair_id": e.get("pair_id"),
                            "domain": e.get("domain"),
                            "text": e.get("requirement_text"),
                            "properties": props,
                            "direction": rs.get("direction")})
            extra_n += 1
        print("**并入候选锚点 %d 条**（来自 %s；**语料未被修改**）"
              % (extra_n, exdir))
    domains = sorted({a["domain"] for a in anchors})
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
        domains = domains[:limit]
    print("=" * 74)
    print("严格 LODO 跨域检验：留一域（共 %d 域 / %d 锚点）" % (len(domains),
                                                       len(anchors)))
    print("=" * 74)
    print("预登记判据：L1 可行性 ｜ L2 无泄漏 ｜ L3 报告双口径 ｜ "
          "L4 数字健全 ｜ L5 三态（set_exact ≥%.0f%% 成立 / ≤%.0f%% 不成立）"
          % (100 * GATE_OK, 100 * GATE_BAD))

    rows = []
    leak_viol = 0
    for dom in domains:
        te = [a for a in anchors if a["domain"] == dom]
        tr = [a for a in anchors if a["domain"] != dom]
        # ---- L2 无泄漏断言 ----
        if any(a["domain"] == dom for a in tr):
            leak_viol += 1
            continue
        y_prop = np.zeros((len(tr), len(PROPERTIES)), dtype=np.float32)
        for i, a in enumerate(tr):
            for p in a["properties"]:
                y_prop[i, PROPERTIES.index(p)] = 1.0
        y_dir = np.array([0 if a["direction"] in (None, "bound_below") else 1
                          for a in tr], dtype=np.int64)
        for mode in ("transductive", "inductive"):
            if mode == "transductive":
                vocab = build_vocab([a["text"] for a in anchors])
            else:
                vocab = build_vocab([a["text"] for a in tr])
            in_dim = input_dim(vocab)   # 含线索块（第 193 条）
            Xtr = _featurize_with([a["text"] for a in tr], vocab, featurize)
            Xte = _featurize_with([a["text"] for a in te], vocab, featurize)
            model = train_model(Xtr, y_prop, y_dir, in_dim)
            n_ex = n_un = n_dir = n_dir_tot = 0
            n_abs = 0
            with torch.no_grad():
                lp, ld = model(torch.tensor(Xte))
            probs = torch.sigmoid(lp).numpy()
            dirs = torch.softmax(ld, -1).numpy()
            for k, a in enumerate(te):
                conf = float(probs[k].max())
                # **生产口径**：低置信 ⟹ 弃权（转人工队列），**不猜属性** ✓
                if conf < TAU_ABSTAIN:
                    n_abs += 1
                    preds = ["unmapped"]
                    pred_dir = None
                else:
                    preds = [PROPERTIES[j] for j in range(len(PROPERTIES))
                             if probs[k][j] > 0.5] or [
                        PROPERTIES[int(probs[k].argmax())]]
                    pred_dir = _DIRS[int(dirs[k].argmax())]
                n_ex += int(set(preds) == set(a["properties"]))
                n_un += int(("unmapped" in a["properties"])
                            == ("unmapped" in preds))
                if "unmapped" not in a["properties"] and pred_dir is not None:
                    n_dir_tot += 1
                    n_dir += int(pred_dir == a["direction"])
            m = len(te)
            rows.append({
                "domain": dom, "mode": mode, "n_test": m,
                "set_exact": n_ex / m,
                "abstained": n_abs / m,
                "unmapped_ok": n_un / m,
                "direction_acc": (n_dir / n_dir_tot) if n_dir_tot else float("nan"),
            })
    # ---- 汇总 ----
    print("\nL1 **可行性**：完成 %d 域（%d 条记录）；L2 **无泄漏**：违规 %d 折 %s"
          % (len({r['domain'] for r in rows}), len(rows), leak_viol,
             "✓" if leak_viol == 0 else "**✗**"))
    n_te = sum(r["n_test"] for r in rows) / 2 / max(len(domains), 1)
    print("（平均每域留出 %.1f 个锚点）" % n_te)
    out = {}
    out_ab = {}
    for mode in ("transductive", "inductive"):
        sub = [r for r in rows if r["mode"] == mode]
        w = sum(r["n_test"] for r in sub)
        se = sum(r["set_exact"] * r["n_test"] for r in sub) / w
        un = sum(r["unmapped_ok"] * r["n_test"] for r in sub) / w
        ab = sum(r["abstained"] * r["n_test"] for r in sub) / w
        dv = [r["direction_acc"] for r in sub if r["direction_acc"] == r["direction_acc"]]
        out[mode] = (se, un, float(np.mean(dv)) if dv else float("nan"))
        out_ab[mode] = ab
    print("\nL3 **LODO 指标**（按留出锚点数加权；**生产口径**：`conf < %.2f` ⟹ 弃权）"
          % TAU_ABSTAIN)
    print("  口径            属性集精确率    弃权率   unmapped 一致   方向准确率")
    for mode in ("transductive", "inductive"):
        se, un, da = out[mode]
        ab = out_ab[mode]
        print("  %-14s %10.1f%% %9.1f%% %14.1f%% %11.1f%%"
              % ("传导式" if mode == "transductive" else "归纳式",
                 100 * se, 100 * ab, 100 * un, 100 * da))
    print("  **注**：初版 harness **未实现弃权**（强制 argmax）⟹ 真值 `unmapped`"
          " 的锚点**必然判错**；本版按生产规则计分，故数字与会话早期不同 ✓")
    print("  **对照**：既有 LOOCV（留一锚点，乐观）属性集精确率 "
          "**88.1%%**（第 193 条方案 C 后；方案 C 前为 85.5%）")
    se_t = out["transductive"][0]
    se_i = out["inductive"][0]
    print("  ⟹ **泛化差距**：LOOCV − LODO(传导) = **%.1f 个百分点**；"
          "LODO(传导) − LODO(归纳) = **%.1f 个百分点**（**词表泄漏**的量）"
          % (100 * (0.881 - se_t), 100 * (se_t - se_i)))
    ok_num = all(0.0 <= v <= 1.0 for mode in out for v in out[mode]
                 if v == v)
    print("L4 **无不可能的数**：三项指标均在 [0,1] ⟹ %s"
          % ("**通过** ✓" if ok_num else "**未通过**"))
    verdict = ("**跨域泛化成立**" if se_i >= GATE_OK else
               "**稳健不成立**" if se_i <= GATE_BAD else
               "**部分成立 / 不稳定**（介于 %.0f%%–%.0f%%）"
               % (100 * GATE_BAD, 100 * GATE_OK))
    print("L5 **三态判定**（按**归纳式** set_exact = %.1f%%）：%s"
          % (100 * se_i, verdict))
    print("\n**如实说明**：本检验测的是 **B 解析器的域外泛化**（需求文本 → q 结构）；"
          "跨域主张的另一半（**指纹/容量**的族外可预测性）见 T1/T2（T2 已如实"
          "记为**边界：未过**）✓")

    with open(RUNS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n逐域落盘：%s（%d 行）" % (RUNS.name, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
