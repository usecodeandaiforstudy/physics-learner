# -*- coding: utf-8 -*-
"""A 表示接线（文档 §2.6.2 第 2 条，2026-10-02）：智能体状态用 A 的「同一双眼睛」。

文档原话：**智能体状态直接用 A 的冻结骨干嵌入**——智能体与裁判共用同一双
眼睛，感知鸿沟（「智能体看到的优势裁判检查不到」）整个失效模式被消除。

v0 落地层次（文档化）：
  - **本模块 = A 的输入表示**（同一编码管线：SystemEncoder → EncodedGraph →
    compute_graph_fingerprint → graph_to_tensors 的节点特征 H），每槽组件
    （族, 参数）的节点特征池化后拼接 —— 与 A 共用**同一编码与指纹管线**；
  - **升级路径**：A 的**冻结骨干嵌入**（GINE 输出）需落盘检查点
    （`train_gine.py --save-checkpoint`，已加），有检查点后本模块自动切换
    为骨干嵌入模式。

用法：
    feats = state_features(env)          # (n_states, D) 数组，可直接当 feat_fn
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

__all__ = ["component_feature", "state_features", "CKPT_PATH"]

CKPT_PATH = _ROOT / "a_network" / "checkpoints" / "gine_best.pt"


def _entry_for(env, slot_type: int, bin_idx: int) -> dict:
    """结构空间 (类型, 箱) → DB1 风格条目（复用既有族与通道模板）。"""
    fam = {1: "landau", 2: "elastic"}[slot_type]
    val = float(env.grid_for(slot_type)[bin_idx])
    if fam == "landau":
        params = [{"name": "a", "default": val},
                  {"name": "b", "default": 1.0},
                  {"name": "kT", "default": 1.0, "is_bridge_constant": True,
                   "unit": "energy"}]
        ct = "CT-LANDAU-001"
    else:
        params = [{"name": "k", "default": val},
                  {"name": "kT", "default": 1.0, "is_bridge_constant": True,
                   "unit": "energy"}]
        ct = "CT-ELASTIC-001"
    return {"family": fam, "channel_template": ct, "beta": 1.0,
            "parameters": params}


def component_feature(env, slot_type: int, bin_idx: int) -> np.ndarray:
    """单组件（族, 参数）的 A 输入表示：节点特征池化向量。"""
    from a_network.graph_dataset import graph_to_tensors
    from b_network.system_encoder import SystemEncoder
    from core.fingerprint import compute_graph_fingerprint

    enc = SystemEncoder()
    graph = enc.encode(_entry_for(env, slot_type, bin_idx))
    fp = compute_graph_fingerprint(graph)
    tens = graph_to_tensors(graph, fp)
    H = np.asarray(tens["H"], dtype=np.float32)
    mask = np.asarray(tens["mask"]).astype(bool)
    valid = H[mask] if mask.any() else H
    return valid.mean(axis=0)          # 节点特征池化（v0）


def _assemble(env, comp: dict) -> np.ndarray:
    """组件特征 → 状态特征矩阵（每槽：组件特征 + 活跃标志；吸收态全零）。"""
    dim = next(iter(comp.values())).shape[0] + 1
    F = np.zeros((env.n_states, env.n_slots * dim), dtype=np.float32)
    for s in range(env.absorbing):
        vec = []
        for t, b in env.decode(s):
            if t == 0:
                vec.append(np.zeros(dim, dtype=np.float32))
            else:
                vec.append(np.concatenate(
                    [comp[(t, b)], np.ones(1, dtype=np.float32)]))
        F[s] = np.concatenate(vec)
    return F


def state_features(env, use_backbone: bool = True) -> np.ndarray:
    """结构空间全部状态的 A 表示特征 (n_states, D)。

    每槽 = 组件特征（空槽为零）+ 活跃标志；吸收态全零。
    """
    if use_backbone and CKPT_PATH.exists():
        return _backbone_features(env)     # 升级路径：A 冻结骨干嵌入
    comp = {}
    for t in (1, 2):
        for b in range(env.n_bins):
            comp[(t, b)] = component_feature(env, t, b)
    return _assemble(env, comp)


def _backbone_features(env) -> np.ndarray:
    """A 冻结骨干嵌入模式（需 train_gine.py --save-checkpoint 产出的检查点）。

    检查点约定：{"model": GINE 权重, "d_in"/"d_e"/"d_h"/"n_layers": 结构超参}。
    """
    import torch
    from a_network.gine import GINE
    from a_network.graph_dataset import graph_to_tensors
    from b_network.system_encoder import SystemEncoder
    from core.fingerprint import compute_graph_fingerprint

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = GINE(d_in=int(ckpt["d_in"]), d_e=int(ckpt["d_e"]),
                 d_h=int(ckpt["d_h"]), n_layers=int(ckpt.get("n_layers", 6)))
    model.load_state_dict(ckpt["model"])
    model.eval()
    enc = SystemEncoder()
    comp = {}
    for t in (1, 2):
        for b in range(env.n_bins):
            graph = enc.encode(_entry_for(env, t, b))
            fp = compute_graph_fingerprint(graph)
            tens = graph_to_tensors(graph, fp)
            with torch.no_grad():
                _hv, h_g = model(torch.tensor(tens["H"])[None],
                                 torch.tensor(tens["A"])[None],
                                 torch.tensor(tens["E"])[None],
                                 torch.tensor(tens["mask"])[None])
            comp[(t, b)] = h_g.numpy().reshape(-1)
    return _assemble(env, comp)
