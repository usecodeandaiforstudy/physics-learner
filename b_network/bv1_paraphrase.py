# -*- coding: utf-8 -*-
"""B-parse 语言层数据扩充（模式③）：锚点文本改写 + 分组 LOOCV 复评。

定位与红线：改写文本的**槽位标签直接取自已双人复核的锚点**（q 不变、标签不变），
仅用于 B-parse 语言可变性训练——**改写文本本身永不成为 DB3 锚点**，provenance
记为 AI+paraphrase-of-anchor。

流程：
  1) 对 34 个锚点各生成 K 条同语义改写（一次调用 K 条，数值/单位/术语不变）；
  2) 存 b_network/bv1_paraphrase.jsonl（{pair_id, text, properties, direction}）；
  3) **分组 LOOCV 复评**：折按 pair_id 划分（同一锚点的全部改写随锚点同折，
     防标签泄漏）；训练 = 其他对的锚点+改写，测试 = 本对的原锚点文本；
  4) 对比基线 47.1%（第 38 条）与 60% 切换门槛。

用法（项目根目录）：
    python b_network/bv1_paraphrase.py --dry-run
    python b_network/bv1_paraphrase.py --k 8 --limit 3          # 先小试 3 对
    python b_network/bv1_paraphrase.py --skip-generate --eval-only   # 复评已有改写
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

sys.stdout.reconfigure(encoding="utf-8")

from b_network.bv1_dataset import PROPERTIES, load_anchors  # noqa: E402
from b_network.bv1_parse import (
    input_dim,  # noqa: E402
    Bv1Model,
    build_vocab,
    featurize,
)
from db1.mode3_llm import chat, load_config  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARA_PATH = os.path.join(_HERE, "bv1_paraphrase.jsonl")
_CONFIG_DEFAULT = os.path.join(_PROJECT_ROOT, "db1", "mode3_config.json")
_DIRS = ["bound_below", "bound_above"]

PARA_SYSTEM = (
    "你是工程需求文本改写器。把给定的需求文本改写为语义完全相同的不同措辞。"
    "铁律：所有数值、单位、比较方向（不低于/不超过/至少/以内）、技术术语"
    "（刚度/精度/响应/热锋/鲁棒/稳定/适应/韧性等）必须原样保留，只换句式、"
    "语序与同义表达；中英文可互译改写。输出 K 条改写，每条一行，格式 "
    "“数字. 改写文本”，不要任何额外说明。"
)


def generate_paraphrases(cfg: dict, anchors: list[dict], k: int,
                         limit: int, dry_run: bool) -> list[dict]:
    rows: list[dict] = []
    for a in anchors[:limit] if limit else anchors:
        user = "【需求文本】%s\nK = %d" % (a["text"], k)
        if dry_run:
            print("[dry-run] %s：%s" % (a["pair_id"], a["text"][:50]))
            continue
        raw = chat(cfg, [{"role": "system", "content": PARA_SYSTEM},
                         {"role": "user", "content": user}])
        lines = []
        for line in raw.splitlines():
            line = re.sub(r"^\s*\d+[\.、\)]\s*", "", line).strip()
            if line and line != a["text"]:
                lines.append(line)
        n_ok = 0
        for t in lines[:k]:
            rows.append({"pair_id": a["pair_id"], "text": t,
                         "properties": a["properties"],
                         "direction": a["direction"]})
            n_ok += 1
        print("[PARA] %s：%d/%d 条" % (a["pair_id"], n_ok, k))
    return rows


def grouped_loocv(anchors: list[dict], paras: list[dict]) -> dict:
    """分组 LOOCV：折按 pair_id；训练含其他对的改写；测试 = 本对原锚点文本。"""
    by_pair = {a["pair_id"]: a for a in anchors}
    vocab = build_vocab([a["text"] for a in anchors] + [p["text"] for p in paras])
    in_dim = input_dim(vocab)   # 含线索块（第 193 条）

    def encode(rows):
        xs = np.stack([featurize(r["text"], vocab) for r in rows])
        yp = np.zeros((len(rows), len(PROPERTIES)), dtype=np.float32)
        yd = np.zeros(len(rows), dtype=np.int64)
        for i, r in enumerate(rows):
            for p in r["properties"]:
                yp[i, PROPERTIES.index(p)] = 1.0
            yd[i] = 0 if r["direction"] in (None, "bound_below") else 1
        return xs, yp, yd

    all_rows = [{"pair_id": a["pair_id"], "text": a["text"],
                 "properties": a["properties"], "direction": a["direction"]}
                for a in anchors] + paras

    fold = []
    for pid in sorted(by_pair):
        tr = [r for r in all_rows if r["pair_id"] != pid]
        te = by_pair[pid]
        xs, yp, yd = encode(tr)
        torch.manual_seed(0)
        model = Bv1Model(in_dim)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
        X = torch.tensor(xs)
        for _ in range(600):
            opt.zero_grad()
            lp, ld = model(X)
            loss = bce(lp, torch.tensor(yp)) + ce(ld, torch.tensor(yd))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            lp, ld = model(torch.tensor(featurize(te["text"], vocab)[None, :]))
        probs = torch.sigmoid(lp)[0].numpy()
        preds = [PROPERTIES[i] for i in range(len(PROPERTIES))
                 if probs[i] > 0.5] or [PROPERTIES[int(probs.argmax())]]
        pred_dir = _DIRS[int(torch.softmax(ld, -1)[0].argmax())]
        fold.append({"pair_id": pid,
                     "exact": set(preds) == set(te["properties"]),
                     "dir_ok": None if "unmapped" in te["properties"]
                     else (pred_dir == te["direction"]),
                     "conf": float(probs.max())})
    n = len(fold)
    exact = sum(f["exact"] for f in fold) / n
    dirs = [f["dir_ok"] for f in fold if f["dir_ok"] is not None]
    return {"n_folds": n, "set_exact": exact,
            "direction_acc": sum(dirs) / len(dirs),
            "baseline_47_1": 0.471, "threshold_60": 0.60,
            "passed_threshold": exact >= 0.60}


def main() -> int:
    ap = argparse.ArgumentParser(description="B-parse 语言层数据扩充")
    ap.add_argument("--config", default=_CONFIG_DEFAULT)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 个锚点（0=全部 34）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-generate", action="store_true",
                    help="复用已有 bv1_paraphrase.jsonl，只做复评")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    anchors = load_anchors()
    if not args.skip_generate:
        cfg = load_config(args.config)
        paras = generate_paraphrases(cfg, anchors, args.k, args.limit,
                                     args.dry_run)
        if args.dry_run:
            return 0
        with open(_PARA_PATH, "a", encoding="utf-8") as f:
            for r in paras:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("改写已追加 %d 条 → %s" % (len(paras), _PARA_PATH))
    else:
        paras = []
        if os.path.exists(_PARA_PATH):
            paras = [json.loads(line) for line in
                     open(_PARA_PATH, encoding="utf-8").read().splitlines()
                     if line.strip()]
        print("复用已有改写 %d 条" % len(paras))

    if args.eval_only or args.skip_generate or paras:
        report = grouped_loocv(anchors, paras)
        print("\n================ B-parse 分组 LOOCV 复评（含改写扩充） ================")
        print("锚点 %d + 改写 %d；属性集合精确率 %.1f%%（基线 47.1%%，门槛 60%%）"
              % (len(anchors), len(paras), 100 * report["set_exact"]))
        print("方向准确率 %.1f%%" % (100 * report["direction_acc"]))
        print("判定：%s" % ("✅ 达切换门槛" if report["passed_threshold"]
                        else "❌ 未达门槛（继续扩数据或保持 v0）"))
        with open(os.path.join(_HERE, "bv1_paraphrase_report.json"), "w",
                  encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
