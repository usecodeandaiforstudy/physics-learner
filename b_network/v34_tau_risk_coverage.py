# -*- coding: utf-8 -*-
"""**弃权阈值 τ 的风险–覆盖分析**（2026-10-02，第 192 条）。

## 动因（第 191 条的诊断）
LOPO 诊断显示：模型的 **top-1 常常是对的**（v1 3/10、v2 4/10），但
**最大概率中位仅 0.21**；多条锚点"方向已对、只是概率 <0.90" ⟹
**生产弃权阈值 `τ=0.90` 把"对的但不够自信"判成了失败** ✓
⟹ 需要的是 **τ 的风险–覆盖曲线**（selective prediction），
而不是继续加特征。

## 本脚本做什么（**只测量，不改生产**）
在**严格 LODO 协议**（52 域留一、**归纳式**词表）下，对 **v1 与 v2** 各扫一遍
`τ ∈ {0.30, 0.40, …, 0.90}`，报告：
    · **覆盖率** = 1 − 弃权率（自动处理的比例）；
    · **选择性精确率** = 在**自动处理**的样本上的属性集精确率
      （这才是"敢回答时有多准"）；
    · **总体精确率** = 全部样本上的精确率（弃权算错）；
    · **方向准确率**（自动处理者）。
**注意**：τ 是**生产策略**（弃权 ⟹ 转人工，成本不同）⟹ 本脚本**只给曲线**，
**由人选择**工作点 ✓（**不预设**哪个 τ 最好，也**不据此改**既有判据）

用法：
    python b_network/v34_tau_risk_coverage.py
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

RUNS = _ROOT / "v34_tau_risk_coverage.csv"
TAUS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def _feats(mode: str):
    """**v1 臂用 legacy 实现**（方案 C 后生产 featurize 已含线索块，
    仍用它会让 v1 静默变成 v2 ✗）。"""
    if mode == "v2":
        from features_v2 import dim_v2 as d, featurize_v2 as f
        return f, d
    from features_v2 import dim_legacy as d, featurize_legacy as f
    return f, d


def _probs_for(tr, te, mode):
    """一次训练，返回测试集的 (probs, dirs)（供 τ 扫描复用）。"""
    import torch
    from bv1_train_eval import PROPERTIES, build_vocab, train_model
    featurize, dim_fn = _feats(mode)
    vocab = build_vocab([a["text"] for a in tr])
    Xtr = np.stack([featurize(a["text"], vocab) for a in tr])
    y = np.zeros((len(tr), len(PROPERTIES)), dtype=np.float32)
    for i, a in enumerate(tr):
        for p in a["properties"]:
            if p in PROPERTIES:
                y[i, PROPERTIES.index(p)] = 1.0
    yd = np.array([0 if a["direction"] in (None, "bound_below") else 1
                   for a in tr], dtype=np.int64)
    model = train_model(Xtr, y, yd, dim_fn(vocab))
    Xte = np.stack([featurize(a["text"], vocab) for a in te])
    with torch.no_grad():
        lp, ld = model(torch.tensor(Xte))
    return torch.sigmoid(lp).numpy(), torch.softmax(ld, -1).numpy()


def main() -> int:
    from bv1_train_eval import PROPERTIES, _DIRS, load_anchors
    corpus = load_anchors()
    domains = sorted({a["domain"] for a in corpus})
    print("=" * 74)
    print("弃权阈值 τ 的风险–覆盖分析（严格 LODO，52 域，归纳式词表）")
    print("=" * 74)
    print("τ 是**生产策略**（弃权 ⟹ 转人工）；下表**只给曲线**，工作点由人选择 ✓")
    rows = []
    for mode in ("v1", "v2"):
        # 收集每折的 (probs, dirs, items)
        fold = []
        for dom in domains:
            tr = [a for a in corpus if a["domain"] != dom]
            te = [a for a in corpus if a["domain"] == dom]
            p, d = _probs_for(tr, te, mode)
            fold.append((p, d, te))
        print("\n[%s]" % mode)
        print("  %-6s %10s %14s %14s %12s" %
              ("τ", "覆盖率", "选择性精确率", "总体精确率", "方向准确率"))
        for tau in TAUS:
            n_tot = n_auto = n_auto_ok = n_all_ok = n_d = n_dt = 0
            for probs, dirs, items in fold:
                for k, a in enumerate(items):
                    n_tot += 1
                    conf = float(probs[k].max())
                    if conf < tau:
                        continue                     # 弃权 ⟹ 转人工
                    n_auto += 1
                    preds = [PROPERTIES[j] for j in range(len(PROPERTIES))
                             if probs[k][j] > 0.5] or [
                        PROPERTIES[int(probs[k].argmax())]]
                    ok = set(preds) == set(a["properties"])
                    n_auto_ok += int(ok)
                    n_all_ok += int(ok)
                    if "unmapped" not in a["properties"]:
                        n_dt += 1
                        n_d += int(_DIRS[int(dirs[k].argmax())]
                                   == a["direction"])
            rows.append({
                "mode": mode, "tau": tau,
                "coverage": n_auto / n_tot,
                "selective_exact": (n_auto_ok / n_auto) if n_auto else
                float("nan"),
                "overall_exact": n_all_ok / n_tot,
                "dir": (n_d / n_dt) if n_dt else float("nan"),
            })
            print("  %-6.2f %9.1f%% %13.1f%% %13.1f%% %11.1f%%"
                  % (tau, 100 * n_auto / n_tot,
                     100 * ((n_auto_ok / n_auto) if n_auto else float("nan")),
                     100 * n_all_ok / n_tot,
                     100 * ((n_d / n_dt) if n_dt else float("nan"))))
    # ---- 拐点提示（供人决策，**不是**自动选择）----
    print("\n**可读结论（供你选工作点）**：")
    for mode in ("v1", "v2"):
        sub = [r for r in rows if r["mode"] == mode]
        best = max(sub, key=lambda r: r["overall_exact"])
        print("  [%s] 总体精确率最高在 τ=%.2f（覆盖 %.1f%%、选择性精确率 %.1f%%、"
              "总体 %.1f%%）" % (mode, best["tau"],
                              100 * best["coverage"],
                              100 * best["selective_exact"],
                              100 * best["overall_exact"]))
        hi = [r for r in sub if r["coverage"] >= 0.5]
        if hi:
            b2 = max(hi, key=lambda r: r["selective_exact"])
            print("       覆盖率 ≥50%% 时选择性精确率最高：τ=%.2f（覆盖 %.1f%%、"
                  "选择性 %.1f%%）" % (b2["tau"], 100 * b2["coverage"],
                                     100 * b2["selective_exact"]))
    with open(RUNS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n落盘：%s（%d 行）" % (RUNS.name, len(rows)))
    print("**未修改任何生产参数**（`bv1_train_eval` 的 τ 与特征均保持原样）✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
