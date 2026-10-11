"""数据增广：约束负样本 + 排序 pairwise 标签（编辑对留 Phase 2）。

对应架构文档 §2.2.5「负样本必须显式构造」与 §2.2.3 排序头 / 编辑效应头。
Phase 1 实现两件可在 batch 内动态构造的增广：
- 约束负样本：篡改边互信息使信息链不再单调（DPI 违反），标签 0
- 排序 pairwise：按 Fisher 信息量大小构造优劣对（代理标签）

编辑效应头（(G,edit)→Δf）需要慢路径重算指纹的数据管线，留 Phase 2
（§2.2.9：对 DB2 每图枚举合法/非法编辑，慢路径算真值 Δf）。
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["constraint_negative", "rank_pairs"]


def constraint_negative(H: torch.Tensor, A: torch.Tensor, E: torch.Tensor,
                        mask: torch.Tensor, n_neg: int) -> tuple[torch.Tensor, torch.Tensor]:
    """从合法图批量构造约束负样本：篡改边互信息破坏 DPI 单调性。

    参数
    ----
    H, A, E, mask : 同 GINE 输入（E 最后一维是边互信息）。
    n_neg : 要生成的负样本数（≤ batch 大小）。

    返回
    ----
    (E_neg, y_neg) : 篡改后的边特征张量 [n_neg,N,N,d_e]，标签全 0（违反约束）。
    """
    B = H.shape[0]
    n_neg = min(n_neg, B)
    idx = np.random.choice(B, size=n_neg, replace=False)
    E_neg = E[idx].clone()
    # 对每个负样本，随机选一条有向边，把它的互信息放大 2 倍（破坏"信息沿链递减"）
    for k in range(n_neg):
        nnz = (A[idx[k]] > 0).nonzero(as_tuple=False)  # [E,2] (dst, src)
        if len(nnz) == 0:
            continue
        r = int(torch.randint(0, len(nnz), (1,)).item())
        d, s = nnz[r][0].item(), nnz[r][1].item()
        E_neg[k, d, s, 0] *= 2.0
    return E_neg, torch.zeros(n_neg, 1, device=H.device)


def rank_pairs(fisher_trace: torch.Tensor, n_pairs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """按 Fisher 信息量构造 pairwise 排序对（代理标签）。

    Fisher trace 大 = 信息灵敏度高 = 排序"更优"（Phase 1 代理，正式优劣
    标签应来自 DB1 物理性能记录，见 §2.2.3 排序头注）。

    参数
    ----
    fisher_trace : [B] 每图的 T 节点 fisher_trace（或全局摘要）。
    n_pairs : 生成的 pairwise 对数。

    返回
    ----
    (pair_i, pair_j) : 两个 [n_pairs] 索引张量，fisher_trace[i] > fisher_trace[j]。
    """
    B = fisher_trace.shape[0]
    if B < 2:
        return (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
    n_pairs = min(n_pairs, B * (B - 1) // 2)
    i = torch.randint(0, B, (n_pairs * 2,), device=fisher_trace.device)
    j = torch.randint(0, B, (n_pairs * 2,), device=fisher_trace.device)
    valid = (i != j) & (fisher_trace[i] > fisher_trace[j])
    if valid.sum() < n_pairs:
        # 不足则按 Fisher 排序取极值对补足
        order = torch.argsort(fisher_trace, descending=True)
        i = order[:n_pairs].repeat_interleave(1)
        j = order[-n_pairs:]
    else:
        i, j = i[valid][:n_pairs], j[valid][:n_pairs]
    return i, j
