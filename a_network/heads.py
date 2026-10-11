"""六读出头（架构文档 §2.2.3）。

共享 GINE 骨干输出（节点嵌入 H_v、全局嵌入 H_g），六个头各自独立，
接口只依赖 H_v / H_g，换骨干不动损失函数（§2.2.2 接口不变性）。

六头与 Phase 1 实现策略：
1. 指纹回归头 f̂(G)：节点级 log fisher_trace/det（MSE）—— 核心知识任务
2. 约束满足头 c(G)：DPI 链满足概率（BCE）—— RL 硬约束，需负样本
3. 行为分类头：family（CE）
4. 编辑效应头 (G,edit)→Δf̂：编辑前后指纹差（MSE）—— world model
5. 排序头：结构-任务匹配优势分（margin ranking）—— RL 奖励整形
6. 不确定性头 U(G)：每输出认知不确定度（异方差 NLL）—— 防 reward hacking
   （外加掩码重建头，属 L_mask 自监督而非 §2.2.3 六头，一并放这里便于共享骨干）
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["MultiHead"]


class MultiHead(nn.Module):
    """六读出头 + 掩码重建头，共享 GINE 骨干输出。"""

    def __init__(self, d_h: int, n_classes: int, d_mask: int, d_edit: int = 1):
        super().__init__()
        # 1. 指纹回归头（均值）
        self.fisher_head = nn.Linear(d_h, 2)            # [log trace, log det]
        # 6. 不确定性头（异方差：log var，NLL 训练）
        self.uncertainty_head = nn.Linear(d_h, 2)       # [log var_trace, log var_det]
        # 2. 约束满足头
        self.constraint_head = nn.Sequential(
            nn.Linear(d_h, d_h), nn.ReLU(), nn.Linear(d_h, 1),
        )
        # 3. 行为分类头
        self.class_head = nn.Sequential(
            nn.Linear(d_h, d_h), nn.ReLU(), nn.Linear(d_h, n_classes),
        )
        # 4. 编辑效应头（全局嵌入 + 编辑编码 → Δf 增量）
        self.edit_head = nn.Sequential(
            nn.Linear(d_h + d_edit, d_h), nn.ReLU(), nn.Linear(d_h, 2),
        )
        # 5. 排序头
        self.rank_head = nn.Sequential(
            nn.Linear(d_h, d_h), nn.ReLU(), nn.Linear(d_h, 1),
        )
        # L_mask 掩码重建头（自监督）
        self.mask_head = nn.Linear(d_h, d_mask)

    def forward(self, H_v: torch.Tensor, H_g: torch.Tensor,
                edit_enc: torch.Tensor | None = None) -> dict:
        fisher_mu = self.fisher_head(H_v)               # [B,N,2]
        fisher_var = self.uncertainty_head(H_v)         # [B,N,2]
        constraint_logit = self.constraint_head(H_g)    # [B,1]
        logits = self.class_head(H_g)                   # [B,n_classes]
        param_recon = self.mask_head(H_v)               # [B,N,d_mask]
        rank_score = self.rank_head(H_g)                # [B,1]
        delta_f = None
        if edit_enc is not None:
            delta_f = self.edit_head(torch.cat([H_g, edit_enc], dim=-1))  # [B,2]
        return {
            "fisher_mu": fisher_mu,
            "fisher_var": fisher_var,
            "constraint_logit": constraint_logit,
            "logits": logits,
            "param_recon": param_recon,
            "rank_score": rank_score,
            "delta_f": delta_f,
        }
