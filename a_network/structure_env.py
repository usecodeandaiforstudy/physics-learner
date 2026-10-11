# -*- coding: utf-8 -*-
"""结构空间环境（缺口②，2026-10-02）：参→结构 的搜索空间升级。

设计（design doc 结构图「RL 智能体：信息结构空间搜索」的 v0 落地）：
  - **状态 = 结构**：3 个槽位，每槽 = (类型, 参数箱)，类型 ∈ {空, landau,
    elastic}、参数箱 5 档 → 每槽 15 态，全空间 15³ = 3375 态 + 吸收终止态；
  - **动作 = 结构编辑 + 参数位移 + 终止**（10 个）：
      0–2  槽 i 类型循环（空→landau→elastic→空）——**结构级动作**；
      3–8  槽 i 参数箱 ±1；
      9    终止；
  - **需求（异构链）**：I_total = Σ 各槽 I_TY 落在 [target−3σ, target+3σ] 且
    **至少 2 个活跃槽且类型不全同**——纯调参解不了（必须选出类型组合）。
    reward = exp(−((I_total − target)/σ)²) × (1.0 异构 / 0.15 同构)。
零新依赖（numpy + core.fisher 解析公式）。
"""

from __future__ import annotations

import numpy as np

from core.fisher import fisher_landau_analytic

__all__ = ["StructureChainEnv"]

_TYPES = ["none", "landau", "elastic"]
_BINS = 5
_S_PER_SLOT = len(_TYPES) * _BINS      # 15
_A_GRID = np.linspace(-2.0, 2.0, _BINS)
_K_GRID = np.logspace(np.log10(0.1), np.log10(10.0), _BINS)


class StructureChainEnv:
    n_types = len(_TYPES)
    n_bins = _BINS

    def __init__(self, n_slots: int = 2, target_i: float = 0.35,
                 sigma_i: float = 0.25):
        """n_slots：结构链槽位数（v0 = 2 槽 / 226 态）。

        **容量边界记录**：3 槽（3376 态）下四臂命中仅 5.2–6.4%（含结构化
        特征与演示），远低于可达基准 → v0.1 GFN 容量不覆盖 3000+ 态组合
        空间（与 M6 组合动作边界同族）；v0 结构空间取 2 槽。
        σ=0.25（首跑 σ=0.08 使 reward 近似 δ 函数、梯度处处为零）。"""
        self.n_slots = n_slots
        self.per_slot = _S_PER_SLOT
        self.absorbing = self.per_slot ** self.n_slots
        self.n_states = self.absorbing + 1
        # 动作 = 每槽 4 个（类型+1 / 类型−1 / 箱−1 / 箱+1）+ 1 个终止。
        # **可逆性要求**（v14 第三跑诊断）：早先的「类型循环 t→(t+1)%3」
        # 不可逆（逆操作需循环两次，动作集里没有）→ P_B 反向映射错误、
        # 流平衡失真；改为对称 ±1 类型动作后每个动作都有精确逆元。
        self.term_action = 4 * self.n_slots
        self.n_actions = self.term_action + 1
        self.target_i = target_i
        self.sigma_i = sigma_i
        self.type_names = _TYPES
        # 每 (类型, 箱) 的 I_TY
        self.i_table = np.zeros((self.n_types, self.n_bins))
        for j, a in enumerate(_A_GRID):
            self.i_table[1, j] = float(fisher_landau_analytic(a, 1.0, 1.0))
        for j, k in enumerate(_K_GRID):
            self.i_table[2, j] = 3.0 * float(k)      # 弹性：I = 3k/kT（kT=1）

    # ---------- 状态编解码 ----------
    def decode(self, state: int) -> list[tuple[int, int]]:
        out = []
        s = int(state)
        for _ in range(self.n_slots):
            out.append(divmod(s % self.per_slot, self.n_bins))
            s //= self.per_slot
        return out[::-1]          # 槽 0 在最高位

    def encode(self, slots: list[tuple[int, int]]) -> int:
        s = 0
        for t, b in slots:
            s = s * self.per_slot + t * self.n_bins + b
        return s

    # ---------- 物理量 ----------
    def grid_for(self, slot_type: int) -> np.ndarray:
        """槽类型的参数网格（1=landau 的 a，2=elastic 的 k）。"""
        return _A_GRID if slot_type == 1 else _K_GRID

    def i_total(self, state: int) -> float:
        return float(sum(self.i_table[t, b] for t, b in self.decode(state)))

    def is_hetero(self, state: int) -> bool:
        active = [t for t, _ in self.decode(state) if t != 0]
        return len(active) >= 2 and len(set(active)) >= 2

    def reward(self, state: int) -> float:
        if state == self.absorbing:
            return 0.0
        pen = ((self.i_total(state) - self.target_i) / self.sigma_i) ** 2
        factor = 1.0 if self.is_hetero(state) else 0.15
        return float(max(np.exp(-pen) * factor, 1e-6))

    # ---------- 动力学 ----------
    def step(self, state: int, action: int) -> tuple[int, float]:
        if state == self.absorbing:
            return self.absorbing, 0.0
        a = action % self.n_actions
        if a == self.term_action:                   # 终止
            return self.absorbing, self.reward(state)
        slots = self.decode(state)
        i, k = divmod(a, 4)     # k: 0=类型+1 1=类型−1 2=箱−1 3=箱+1
        t, b = slots[i]
        if k == 0:
            t = (t + 1) % self.n_types
        elif k == 1:
            t = (t - 1) % self.n_types
        elif k == 2:
            b = int(np.clip(b - 1, 0, self.n_bins - 1))
        else:
            b = int(np.clip(b + 1, 0, self.n_bins - 1))
        slots[i] = (t, b)
        nxt = self.encode(slots)
        return nxt, self.reward(nxt)

    def is_terminal_action(self, action: int) -> bool:
        return (action % self.n_actions) == self.term_action

    def reverse_action(self, action: int) -> int:
        """精确逆元（类型 ±1 互换、箱 ∓1 互换、终止自逆）。

        边界裁剪（箱在端点时 ±1 落在同态）仍是近似——与 M6 同约定。
        """
        a = action % self.n_actions
        if a == self.term_action:
            return self.term_action
        i, k = divmod(a, 4)
        return i * 4 + {0: 1, 1: 0, 2: 3, 3: 2}[k]

    # ---------- 结构化状态特征（缺口②：one-hot 在结构空间无泛化）----------
    @property
    def feat_dim(self) -> int:
        return self.n_slots * (self.n_types + 1)   # 每槽：类型 one-hot + 箱

    def feat(self, state: int) -> np.ndarray:
        """结构化状态特征 = design doc §2.6.2「A 冻结骨干嵌入」的 v0 替身。"""
        v = np.zeros((self.n_slots, self.n_types + 1), dtype=np.float32)
        if state == self.absorbing:
            return v.reshape(-1)
        for i, (t, b) in enumerate(self.decode(state)):
            v[i, t] = 1.0
            v[i, self.n_types] = b / (self.n_bins - 1)
        return v.reshape(-1)

    # ---------- 诊断 ----------
    def solution_states(self, thresh: float = 0.8) -> list[int]:
        return [s for s in range(self.absorbing)
                if self.reward(s) >= thresh]
