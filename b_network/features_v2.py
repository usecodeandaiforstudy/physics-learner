# -*- coding: utf-8 -*-
"""**特征对照工具**（2026-10-02，第 191–193 条）。

第 193 条（方案 C）已把**线索块并入生产特征**（`b_network/bv1_parse.py` 的
`featurize` / `input_dim` / `CUES`）。本模块因此改为：

  · **再导出**生产 v2 的实现（`CUES` / `CUE_DIM` / `cue_block` /
    `featurize_v2` / `dim_v2`）——**单一真源**，避免两处定义漂移 ✗；
  · 并提供 **`featurize_legacy` / `dim_legacy`**：**方案 C 之前**的 v1 特征
    （无线索块），供 `v33_feature_improve.py` / `v34_tau_risk_coverage.py`
    继续做 **v1 vs v2** 对照 ✓（否则"v1 臂"会静默变成 v2 ✗）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "b_network"))

import numpy as np  # noqa: E402

from b_network.bv1_parse import (  # noqa: E402
    CUE_DIM,
    CUE_PROPS,
    CUES,
    build_vocab,
    cue_block,
    featurize,
    input_dim,
)

__all__ = ["CUES", "CUE_PROPS", "CUE_DIM", "cue_block", "featurize_v2",
           "dim_v2", "featurize_legacy", "dim_legacy", "build_vocab"]

# ---------------- 生产 v2（再导出，单一真源） ----------------
featurize_v2 = featurize
dim_v2 = input_dim


def _char3(text: str) -> list[str]:
    t = text.lower()
    return [t[i:i + 3] for i in range(len(t) - 2)
            if not any(c in t[i:i + 3] for c in " \t\n")]


def _en_words(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text.lower())


def featurize_legacy(text: str, vocab: list[str]) -> np.ndarray:
    """**方案 C 之前**的 v1 特征（无线索块）——仅供对照实验使用。"""
    from b_network.bv1_parse import (DIR_ABOVE, DIR_BELOW, PROPERTY_KEYWORDS,
                                     SIGMA_CONTEXT_KW, UNIT_MAP)
    v = {d: 0.0 for d in vocab}
    v["n_numbers"] = float(len(re.findall(r"\d+(?:\.\d+)?", text)))
    v["dir_below"] = float(any(k in text for k in DIR_BELOW))
    v["dir_above"] = float(any(k in text for k in DIR_ABOVE))
    v["kw_soft_hard"] = float("硬" in text)
    v["kw_sigma"] = float(any(k in text for k in SIGMA_CONTEXT_KW))
    for p, kws in PROPERTY_KEYWORDS.items():
        v["kw_%s" % p] = float(any(re.search(k, text) for k in kws))
    for u in UNIT_MAP:
        v["unit_%s" % u] = float(u in text)
    for g in _char3(text):
        key = "g_%s" % g
        if key in v:
            v[key] = 1.0
    for w in _en_words(text):
        key = "g_%s" % w
        if key in v:
            v[key] = 1.0
    return np.asarray([v[d] for d in vocab], dtype=np.float32)


def dim_legacy(vocab: list[str]) -> int:
    """v1 的输入维度（无线索块）。"""
    return max(len(vocab), 1)
