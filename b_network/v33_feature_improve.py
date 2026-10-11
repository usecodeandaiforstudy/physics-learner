# -*- coding: utf-8 -*-
"""**特征改进 A/B：v1 vs v2（规则谓词线索块）**（2026-10-02，第 191 条）。

## 三个口径一起测（都在**生产弃权语义**下：`conf < 0.90` ⟹ 弃权）
  1. **严格 LODO**（52 域留一，归纳式词表）——**主判据**（跨域泛化）；
  2. **LOPO**（10 条候选留一，同域兄弟教它）——**域内迁移**（第 190 条的硬口径）；
  3. **语料 fit**（159 条训练集内）——**回归检查**（不得变差）。

## 预登记判据（先写死，未看数据）
    **I1 主判据**：LODO **归纳式** `set_exact` **提升 ≥ 5 个百分点**
       （基线 75.5% ⟹ 需 ≥ 80.5%）；
    **I2 域内迁移**：LOPO 迁移成功数 **≥ 5 / 10**（基线 1/10）；
    **I3 无回归**：语料 fit 精确率 **仍为 100%**，且 LODO 方向准确率
       **不降超过 2 个百分点**；
    **I4 数字健全**：所有指标 ∈ [0,1]。
    ⟹ **I1 且 I3 通过**才把 v2 接入生产（`bv1_train_eval` 的 featurize）；
       否则**如实记为负结果**，不接入 ✓

用法：
    python b_network/v33_feature_improve.py
"""

from __future__ import annotations

import csv
import json
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

RUNS = _ROOT / "v33_feature_improve.csv"
TAU = 0.50              # 生产 τ（第 193 条方案 C；旧值 0.90 的结果见 HANDOVER）
BASE_LODO_IND = 0.755     # 第 188 条实测基线（归纳式 set_exact）
BASE_LOPO = 1             # 第 190 条实测基线（1/10）
I1_GAIN = 0.05
I2_NEED = 5
I3_DIR_TOL = 0.02
PROPOSED = _ROOT / "db3" / "proposed"


def _feats(mode: str):
    """返回 (featurize, dim_fn)。

    **v1 臂必须用 legacy 实现**：方案 C（第 193 条）已把线索块并入生产
    `bv1_parse.featurize` ⟹ 若此处仍用生产 featurize，v1 会**静默变成 v2** ✗
    """
    if mode == "v2":
        from features_v2 import dim_v2, featurize_v2
        return featurize_v2, dim_v2
    from features_v2 import dim_legacy, featurize_legacy
    return featurize_legacy, dim_legacy


def _train_predict(corpus, test, mode):
    import torch
    from bv1_train_eval import PROPERTIES, build_vocab, train_model
    featurize, dim_fn = _feats(mode)
    vocab = build_vocab([a["text"] for a in corpus])
    Xtr = np.stack([featurize(a["text"], vocab) for a in corpus])
    y_prop = np.zeros((len(corpus), len(PROPERTIES)), dtype=np.float32)
    for i, a in enumerate(corpus):
        for p in a["properties"]:
            if p in PROPERTIES:
                y_prop[i, PROPERTIES.index(p)] = 1.0
    y_dir = np.array([0 if a["direction"] in (None, "bound_below") else 1
                      for a in corpus], dtype=np.int64)
    model = train_model(Xtr, y_prop, y_dir, dim_fn(vocab))
    Xte = np.stack([featurize(a["text"], vocab) for a in test])
    with torch.no_grad():
        lp, ld = model(torch.tensor(Xte))
    return torch.sigmoid(lp).numpy(), torch.softmax(ld, -1).numpy()


def score(probs, dirs, items, PROPERTIES, _DIRS):
    n_abs = n_ex = n_dir = n_dir_tot = 0
    for k, a in enumerate(items):
        conf = float(probs[k].max())
        if conf < TAU:
            n_abs += 1
            preds, pd = ["unmapped"], None
        else:
            preds = [PROPERTIES[j] for j in range(len(PROPERTIES))
                     if probs[k][j] > 0.5] or [
                PROPERTIES[int(probs[k].argmax())]]
            pd = _DIRS[int(dirs[k].argmax())]
        n_ex += int(set(preds) == set(a["properties"]))
        if "unmapped" not in a["properties"] and pd is not None:
            n_dir_tot += 1
            n_dir += int(pd == a["direction"])
    m = max(len(items), 1)
    return {"n": len(items), "exact": n_ex / m, "abstain": n_abs / m,
            "dir": (n_dir / n_dir_tot) if n_dir_tot else float("nan")}


def main() -> int:
    from bv1_train_eval import PROPERTIES, _DIRS, load_anchors
    corpus = load_anchors()
    extra = []
    for p in sorted(PROPOSED.glob("DB3-*.json")):
        e = json.loads(p.read_text(encoding="utf-8"))
        rs = e.get("requirement_structured") or {}
        extra.append({"pair_id": e.get("pair_id"), "domain": e.get("domain"),
                      "text": e.get("requirement_text"),
                      "properties": [rs["property_id"]] if rs.get("property_id")
                      else [], "direction": rs.get("direction")})
    domains = sorted({a["domain"] for a in corpus})
    print("=" * 74)
    print("特征改进 A/B：v1（原）vs v2（**规则谓词线索块**）")
    print("=" * 74)
    print("语料 %d 条 / %d 域；候选 %d 条；弃权阈值 %.2f"
          % (len(corpus), len(domains), len(extra), TAU))
    rows = []
    lopo_sig: dict[str, int] = {}
    for mode in ("v1", "v2"):
        # ---- 1) 严格 LODO（归纳式：词表只用训练域）----
        n_te = n_ex = n_abs = n_d = n_dt = 0
        for dom in domains:
            tr = [a for a in corpus if a["domain"] != dom]
            te = [a for a in corpus if a["domain"] == dom]
            probs, dirs = _train_predict(tr, te, mode)
            s = score(probs, dirs, te, PROPERTIES, _DIRS)
            n_te += s["n"]
            n_ex += s["exact"] * s["n"]
            n_abs += s["abstain"] * s["n"]
            if s["dir"] == s["dir"]:
                n_d += s["dir"] * s["n"]
                n_dt += s["n"]
        lodo = {"exact": n_ex / n_te, "abstain": n_abs / n_te,
                "dir": (n_d / n_dt) if n_dt else float("nan")}
        # ---- 2) LOPO（域内迁移）----
        ok = 0
        lopo_detail = []
        for i, a in enumerate(extra):
            tr = corpus + [x for j, x in enumerate(extra) if j != i]
            probs, dirs = _train_predict(tr, [a], mode)
            s = score(probs, dirs, [a], PROPERTIES, _DIRS)
            ok += int(s["exact"] >= 1.0)
            # **诊断（第 191 条）**：记录**最大概率**与**真值是否在 top-1**
            top1 = PROPERTIES[int(probs[0].argmax())]
            lopo_detail.append({
                "pair_id": a["pair_id"], "domain": a["domain"],
                "max_prob": float(probs[0].max()),
                "top1": top1,
                "top1_in_truth": top1 in a["properties"],
            })
        # ---- 3) 语料 fit（回归检查）----
        probs, dirs = _train_predict(corpus, corpus, mode)
        fit = score(probs, dirs, corpus, PROPERTIES, _DIRS)
        rows.append({"mode": mode, "lodo_exact": lodo["exact"],
                     "lodo_abstain": lodo["abstain"], "lodo_dir": lodo["dir"],
                     "lopo_ok": ok, "lopo_n": len(extra),
                     "fit_exact": fit["exact"], "fit_abstain": fit["abstain"]})
        print("  [%s] LODO 精确率 %.1f%%（弃权 %.1f%%、方向 %.1f%%）｜ "
              "LOPO %d/%d ｜ fit %.1f%%"
              % (mode, 100 * lodo["exact"], 100 * lodo["abstain"],
                 100 * lodo["dir"], ok, len(extra), 100 * fit["exact"]))
        # **LOPO 性质诊断**：区分「**没信号**」与「**只是不够自信**」
        n_sig = sum(1 for d in lopo_detail if d["top1_in_truth"])
        confs = [d["max_prob"] for d in lopo_detail]
        print("       LOPO 性质：**top-1 命中真值 %d/%d**；"
              "最大概率 中位 %.2f、最高 %.2f"
              % (n_sig, len(lopo_detail),
                 float(np.median(confs)), max(confs)))
        for d in lopo_detail:
            print("         %-10s %-14s maxp=%.2f top1=%-14s %s"
                  % (d["pair_id"], d["domain"], d["max_prob"], d["top1"],
                     "命中真值" if d["top1_in_truth"] else "**偏了**"))
        lopo_sig[mode] = n_sig
    a, b = rows[0], rows[1]
    gain = b["lodo_exact"] - a["lodo_exact"]
    i1 = gain >= I1_GAIN
    i2 = b["lopo_ok"] >= I2_NEED
    i3 = (b["fit_exact"] >= 0.999
          and (b["lodo_dir"] >= a["lodo_dir"] - I3_DIR_TOL))
    i4 = all(0.0 <= v <= 1.0 for r in rows for k, v in r.items()
             if k.startswith(("lodo", "fit")))
    print("\n**I1 主判据（LODO 归纳式提升 ≥%.0f pp）**：%.1f%% → %.1f%%"
          "（**%+.1f pp**）⟹ %s"
          % (100 * I1_GAIN, 100 * a["lodo_exact"], 100 * b["lodo_exact"],
             100 * gain, "**通过** ✓" if i1 else "**未通过**"))
    print("**I2 域内迁移（LOPO ≥%d/%d）**：%d/%d → **%d/%d** ⟹ %s"
          % (I2_NEED, len(extra), a["lopo_ok"], len(extra), b["lopo_ok"],
             len(extra), "**通过** ✓" if i2 else "**未通过**"))
    print("**I3 无回归**：fit 精确率 %.1f%% → %.1f%%；LODO 方向 %.1f%% → %.1f%%"
          " ⟹ %s" % (100 * a["fit_exact"], 100 * b["fit_exact"],
                      100 * a["lodo_dir"], 100 * b["lodo_dir"],
                      "**通过** ✓" if i3 else "**未通过**"))
    print("**I4 数字健全** ⟹ %s" % ("**通过** ✓" if i4 else "**未通过**"))
    verdict = ("**采纳 v2 并接入生产**" if (i1 and i3) else
               "**不接入生产**（如实记为负结果）")
    print("\n⟹ **结论**：%s（判据：I1 且 I3）" % verdict)
    with open(RUNS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("落盘：%s" % RUNS.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
