"""编辑效应头的慢路径管线（架构文档 §2.2.9）。

编辑效应头（世界模型）：(G, edit) → Δf̂，预测"结构扰动如何改变指纹"。
Phase 1 简化：编辑 = 改条目的 scan_param 值（合法编辑，仍在 range 内），
慢路径 = SystemEncoder.encode + compute_graph_fingerprint 重算指纹。

训练数据构造（§2.2.9「对 DB2 每图枚举编辑，慢路径算真值 Δf」）：
    编辑前 G_before（default 参数）─┐
                                    ├─ 慢路径算指纹 f_before / f_after
    编辑后 G_after（改 scan_param）─┘    Δf = log f_after − log f_before
    编辑编码 edit_enc = 新参数值 − 旧参数值（1 维）

接口对齐 heads.MultiHead.edit_head（Linear(d_h + d_edit, 2)）：
    d_edit=1（参数变化量），输出 2 维（Δ log trace, Δ log det）。

Phase 1 边界：只做合法编辑（range 内改参数）；非法编辑（改出 range / 断裂链 /
容量倒挂）的负样本留 Phase 2。1-step 预测 + 每步重新锚定（§2.2.9，禁长程开环）。
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from a_network.graph_dataset import _scan_grid, graph_to_tensors  # noqa: E402
from b_network.system_encoder import SystemEncoder  # noqa: E402
from core.fingerprint import compute_graph_fingerprint  # noqa: E402

__all__ = ["EditDataset"]


def _t_log_fisher(tensors: dict) -> np.ndarray:
    """提取表示层（T）节点的 log fisher_trace/det（[2]，log 空间）。"""
    node_ids = tensors["node_ids"]
    fisher = tensors["fisher_target"]  # [MAX_NODES, 2]
    for i, nid in enumerate(node_ids):
        if "__T__" in nid:
            return fisher[i].copy()
    if node_ids:
        return fisher[0].copy()
    return np.zeros(2, dtype=np.float32)


class EditDataset(Dataset):
    """编辑对数据集：((G_before 张量), edit_enc) → Δf_true。"""

    def __init__(self, entries_dir: str | Path, n_edits: int = 5):
        entries_dir = Path(entries_dir)
        encoder = SystemEncoder()
        self.samples: list[dict] = []

        for ef in sorted(entries_dir.glob("*.json")):
            entry = json.loads(ef.read_text(encoding="utf-8"))
            sid = entry["system_id"]
            sp = entry.get("scan_param")
            if not sp:
                continue  # 无扫描参数的条目暂不构造编辑（Phase 1 边界）
            old_val = _default_of(entry, sp)
            if old_val is None:
                continue

            # 编辑前（default 参数）
            graph_before = encoder.encode(entry)
            fp_before = compute_graph_fingerprint(graph_before)
            tensors_before = graph_to_tensors(graph_before, fp_before)
            f_before = _t_log_fisher(tensors_before)

            # 编辑后（改 scan_param 到 range 内 n_edits 个点）
            for new_val in _scan_grid(entry, n_edits):
                if new_val is None or abs(float(new_val) - old_val) < 1e-9:
                    continue
                edited = copy.deepcopy(entry)
                for p in edited["parameters"]:
                    if p["name"] == sp:
                        p["default"] = float(new_val)
                graph_after = encoder.encode(edited)
                fp_after = compute_graph_fingerprint(graph_after)
                tensors_after = graph_to_tensors(graph_after, fp_after)
                f_after = _t_log_fisher(tensors_after)

                sample = dict(tensors_before)  # 浅拷贝（numpy 数组只读共享，安全）
                sample["system_id"] = sid
                sample["edit_enc"] = np.array([float(new_val) - old_val],
                                              dtype=np.float32)
                sample["delta_f"] = (f_after - f_before).astype(np.float32)
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "H": torch.from_numpy(s["H"]),
            "A": torch.from_numpy(s["A"]),
            "E": torch.from_numpy(s["E"]),
            "mask": torch.from_numpy(s["mask"]),
            "edit_enc": torch.from_numpy(s["edit_enc"]),
            "delta_f": torch.from_numpy(s["delta_f"]),
        }


def _default_of(entry: dict, param_name: str) -> float | None:
    for p in entry.get("parameters", []):
        if p["name"] == param_name:
            d = p.get("default")
            return None if d is None else float(d)
    return None
