"""手搓 GINE（Graph Isomorphism Network with Edge features）+ 虚拟节点。

纯 PyTorch 实现，不依赖 PyG/DGL/torch_scatter。依据架构文档 §2.2.2 的
Phase 1 起步骨干（GINE 层 ×6 + 虚拟节点）。

GINE 更新式（Xu et al. 2019，边特征版）：
    h_i' = MLP( (1+ε)·h_i + Σ_{j∈N(i)} ReLU(h_j + e_{ji}) )

用稠密张量实现（图 ≤10 节点，邻接 A ∈ [B,N,N]）：
    - 边特征投影 Eh = Linear(E)                       ∈ [B,N,N,dh]
    - 消息 M = ReLU(H.unsqueeze(1) + Eh)              ∈ [B,N,N,dh]，M[i,j]=h_j+e_{ji}
    - 聚合 agg = einsum('bij,bijd->bid', A, M)        ∈ [B,N,dh]（矩阵乘做邻居求和）
    - 更新 H = MLP((1+ε)·H + agg)
    - 虚拟节点：全局池化 g → 回灌 H += virtual(g)

接口只依赖节点嵌入 H_v、边嵌入（内部）、全局嵌入 H_g，与文档 §2.2.3 六个
读出头解耦。
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["GINELayer", "GINE"]


class GINELayer(nn.Module):
    """单层 GINE（含虚拟节点）。"""

    def __init__(self, d_h: int, d_e: int):
        super().__init__()
        self.edge_proj = nn.Linear(d_e, d_h)
        self.mlp = nn.Sequential(
            nn.Linear(d_h, d_h), nn.ReLU(), nn.Linear(d_h, d_h),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(d_h)
        self.virtual = nn.Linear(d_h, d_h)

    def forward(self, H: torch.Tensor, A: torch.Tensor,
                E: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """前向。

        参数
        ----
        H : [B, N, d_h] 节点特征（已投影）。
        A : [B, N, N]   邻接，A[i, j]=1 表示有边 j→i。
        E : [B, N, N, d_e] 边特征，E[i, j] 为边 j→i 的特征。
        mask : [B, N]   有效节点掩码。

        返回
        ----
        H' : [B, N, d_h]
        """
        Eh = self.edge_proj(E)                       # [B,N,N,dh]
        M = torch.relu(H.unsqueeze(1) + Eh)          # [B,N,N,dh]，M[i,j]=h_j+e_{ji}
        agg = torch.einsum("bij,bijd->bid", A, M)    # [B,N,dh] 邻居求和
        H = self.mlp((1.0 + self.eps) * H + agg)
        H = self.norm(H)
        # 虚拟节点：全局均值池化后回灌各节点
        denom = mask.sum(1, keepdim=True).clamp(min=1.0)
        g = (H * mask.unsqueeze(-1)).sum(1) / denom   # [B,dh]
        H = H + self.virtual(g).unsqueeze(1)
        return H


class GINE(nn.Module):
    """GINE 骨干：节点特征投影 + L 层 GINELayer，输出节点嵌入 H_v 与全局嵌入 H_g。"""

    def __init__(self, d_in: int, d_e: int, d_h: int, n_layers: int = 6):
        super().__init__()
        self.node_enc = nn.Linear(d_in, d_h)
        self.layers = nn.ModuleList(
            [GINELayer(d_h, d_e) for _ in range(n_layers)]
        )
        self.n_layers = n_layers

    def forward(self, H: torch.Tensor, A: torch.Tensor,
                E: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向，返回 (节点嵌入 H_v, 全局嵌入 H_g)。"""
        H = self.node_enc(H)
        for layer in self.layers:
            H = layer(H, A, E, mask)
        denom = mask.sum(1, keepdim=True).clamp(min=1.0)
        H_g = (H * mask.unsqueeze(-1)).sum(1) / denom   # [B,dh]
        return H, H_g
