# -*- coding: utf-8 -*-
"""A 网络 v0 冒烟训练：DB2 指纹向量 → 族分类（10 类）。

诚实定位：这不是架构文档 §2.2 的 A 网络（GINE + 虚拟节点，吃信息结构图）——
本机未装 PyG，故用【指纹向量 MLP 探针】证明数据通路 DB2 → 训练 → 评测 跑通。
A 真正的职责（三元组评分/对比测试）在图编码侧，待 PyG 环境就绪后按 §2.2 实现。

实验设计：
- 特征：T 节点 fisher_trace、log fisher_det、I(X;T)、I(T;Y)、max I(X;T)（IB 曲线）
- 标签：family（10 类）
- 划分：按 system_id 分组（38 训练 / 13 测试），测试系统整组保留 = 真 OOD
- 基线：多数类
输出：训练/测试准确率对比基线；指标写 a_network/smoke_mlp_metrics.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

DB2_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "db2", "db2.jsonl")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_mlp_metrics.json")
SEED = 0
HOLDOUT_FRAC = 0.25


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (X, y, system_idx)；特征含 log 变换稳定量纲。"""
    feats, labels, sys_ids = [], [], []
    families: list[str] = []
    with open(DB2_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            fps = rec.get("fingerprints", {})
            tkey = next((k for k in fps if "__T__" in k), next(iter(fps), None))
            fp = fps.get(tkey, {}) if tkey else {}
            trace = fp.get("fisher_trace") or 0.0
            det = fp.get("fisher_det") or 0.0
            mi = rec.get("mi_profile") or [0.0, 0.0]
            ib = rec.get("ib_curve") or {}
            i_xt_max = max(ib.get("I_XT", [0.0])) if ib.get("I_XT") else 0.0
            feats.append([np.log1p(max(trace, 0.0)), np.log1p(max(det, 0.0)),
                          mi[0] if mi[0] is not None else 0.0,
                          mi[1] if len(mi) > 1 and mi[1] is not None else 0.0,
                          np.log1p(max(i_xt_max, 0.0))])
            fam = rec["family"]
            if fam not in families:
                families.append(fam)
            labels.append(families.index(fam))
            sys_ids.append(rec["system_id"])
    return np.array(feats, dtype=np.float32), np.array(labels), np.array(sys_ids)


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X, y, sys_ids = load_dataset()
    systems = sorted(set(sys_ids))
    rng = np.random.default_rng(SEED)
    n_hold = max(1, int(round(HOLDOUT_FRAC * len(systems))))
    holdout = set(rng.choice(systems, size=n_hold, replace=False))
    tr_mask = np.array([s not in holdout for s in sys_ids])
    te_mask = ~tr_mask

    # 标准化（用训练集统计量）
    mu, sd = X[tr_mask].mean(0), X[tr_mask].std(0) + 1e-8
    Xn = (X - mu) / sd
    n_cls = int(y.max()) + 1

    model = nn.Sequential(nn.Linear(X.shape[1], 32), nn.ReLU(),
                          nn.Linear(32, n_cls))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    Xtr = torch.from_numpy(Xn[tr_mask])
    ytr = torch.from_numpy(y[tr_mask])
    Xte = torch.from_numpy(Xn[te_mask])

    model.train()
    for epoch in range(200):
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_tr = model(Xtr).argmax(1).numpy()
        pred_te = model(Xte).argmax(1).numpy()
    acc_tr = float((pred_tr == y[tr_mask]).mean())
    acc_te = float((pred_te == y[te_mask]).mean())
    majority = float(np.bincount(y[te_mask]).max() / te_mask.sum())

    print("===== A v0 冒烟：指纹向量 MLP 族分类 =====")
    print("数据：%d 条记录 / %d 个系统 / %d 族；划分：训练 %d 系统 / 测试 %d 系统（整组 OOD）"
          % (len(X), len(systems), n_cls, (~tr_mask).sum() and len(set(sys_ids[tr_mask])), n_hold))
    print("训练准确率：%.3f" % acc_tr)
    print("测试准确率：%.3f（多数类基线 %.3f，提升 %+.3f）"
          % (acc_te, majority, acc_te - majority))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model": "MLP-fingerprint-probe（非 GINE，PyG 未装）",
            "n_records": int(len(X)), "n_systems": len(systems), "n_families": n_cls,
            "holdout_systems": sorted(holdout),
            "train_acc": acc_tr, "test_acc": acc_te, "majority_baseline": majority,
            "seed": SEED,
        }, f, ensure_ascii=False, indent=2)
    print("指标已写入 %s" % OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
