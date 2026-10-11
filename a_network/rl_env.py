# -*- coding: utf-8 -*-
"""RL 生产环境（M5，2026-09-29）：DesignTaskEnv。

把 A v2 环境（背板 + 信息作用量奖励）封装成 GFlowNet 的离散设计任务：

  - 状态 = 一族单扫描参数 θ 的网格分箱（v0：33 箱）；
  - 动作 = 名义位移 {stay, ±1, ±3 箱}（越界裁剪）；
  - 奖励 = **信息作用量**：R(s) = exp(β·I_TY(s) − I_XT(s))（任务 A：找相变点
    信息峰值；任务 B：双峰需求 → 双模覆盖验证）。每状态的表在初始化时用
    analytic_backbone 预计算（RL 步内零物理计算）。

诚实边界（文档化）：v0 是「单参数族 × 一维网格」的管线演示——验证
GFlowNet 的多样解采样与信息作用量奖励的连通性；全图结构编辑（多参数/
拓扑动作）与 β 正式标定属 M6+/B v1。
"""

from __future__ import annotations

import numpy as np

from a_network.analytic_backbone import compute_backbone
from b_network.system_encoder import SystemEncoder

__all__ = ["DesignTaskEnv"]


class DesignTaskEnv:
    """离散设计任务环境（landau 族，a ∈ [−2, 2]，33 箱）。"""

    def __init__(self, n_bins: int = 33, a_lo: float = -2.0,
                 a_hi: float = 2.0, beta: float = 4.0,
                 task: str = "critical_point", mode_a: float = 1.5,
                 sigma_r: float = 0.25, b: float = 1.0, kT: float = 1.0):
        self.n_states = n_bins
        self.n_actions = 5
        self.deltas = [0, -1, 1, -3, 3]
        self.beta = beta
        self.a_grid = np.linspace(a_lo, a_hi, n_bins)
        self.task = task
        self.mode_a = mode_a
        self.sigma_r = sigma_r

        enc = SystemEncoder()
        self._table: list[dict] = []
        for a in self.a_grid:
            entry = {"family": "landau", "channel_template": "CT-LANDAU-001",
                     "beta": 1.0,
                     "parameters": [
                         {"name": "a", "default": float(a)},
                         {"name": "b", "default": float(b)},
                         {"name": "kT", "default": float(kT),
                          "is_bridge_constant": True, "unit": "energy"}]}
            graph = enc.encode(entry)
            bb = compute_backbone(graph)
            # I_TY = 任务相关信息 = 序参量关于 a 的可测信息（Fisher trace）；
            # I_XT = 压缩代价（v0：以 Fisher 的逆对数尺度表示的状态熵成本）
            i_ty = float(bb["node_fisher"]["landau__T__0"]["fisher_trace"])
            i_xt = float(np.log1p(i_ty))     # v0 占位：压缩量随信息增长
            self._table.append({"a": float(a), "I_TY": i_ty, "I_XT": i_xt,
                                "dpi_ok": bb["dpi"]["ok"]})

    def step(self, state: int, action: int) -> tuple[int, float]:
        """(next_state, reward)；reward 在终止步由调用方用末状态查询。"""
        nxt = int(np.clip(state + self.deltas[action % self.n_actions],
                          0, self.n_states - 1))
        return nxt, self.reward(nxt)

    def reward(self, state: int) -> float:
        """信息作用量终端奖励（恒正，供 TB 损失 log）。

        任务 critical_point：R = exp(β·I_TY − I_XT)——landau 的 I_TY(a) 曲线
        实测近平坦（v5 首跑），「集中到峰值」在该奖励下不可学（P_T∝R 本身
        接近均匀）——保留为文档化反例。
        任务 info_target：R = exp(β·I_TY − I_XT − ((a−a_t)/σ)²)——信息作用量
        × 需求邻近窗（物理叙事：需求指向深双阱信息峰值附近的最大可测性
        区域），景观尖锐、可学。
        任务 bimodal：双高斯需求（±mode_a 两模）。
        """
        t = self._table[state]
        if not t["dpi_ok"]:
            return 1e-6
        base = self.beta * t["I_TY"] - t["I_XT"]
        if self.task == "critical_point":
            return float(np.exp(np.clip(base, -20.0, 20.0)))
        if self.task == "info_target":
            pen = ((t["a"] - self.mode_a) / self.sigma_r) ** 2
            return float(np.exp(np.clip(base - pen, -30.0, 30.0)))
        # bimodal
        r = (np.exp(-((t["a"] - self.mode_a) / self.sigma_r) ** 2)
             + np.exp(-((t["a"] + self.mode_a) / self.sigma_r) ** 2))
        return float(max(r, 1e-6))

    def table(self) -> dict:
        return {"a": self.a_grid,
                "I_TY": np.array([t["I_TY"] for t in self._table]),
                "reward": np.array([self.reward(i)
                                    for i in range(self.n_states)])}


class MultiFamilyEnv:
    """跨族结构编辑环境（M6，2026-09-29；M6.2 显式终止；缺口① 回流域 2026-10-02）。

    状态 = (族, 参数箱) + 吸收终止态：**6 族 × 33 箱 + 1 = 199 状态**
    （`extra_families=False` 时为原 3 族 × 33 + 1 = 100）；
      1. landau：a ∈ [−2, 2]，I_TY = fisher_trace(a)（深双阱边界 ~0.389）；
      2. elastic：k ∈ [0.1, 10]，I_TY = 3k/kT（kT=1，单调升）；
      3. two_level：p ∈ [0.1, 0.9]，I_TY = 1/(p(1−p))（U 形，≥4）；
      **4. control_ou（新增）**：θ ∈ [0.2, 5]，I_TY = θ/2（**已推导桥接式**
         q1 = q9/2，见 DB5 解析验证）；
      **5. acoustics_rayleigh（新增）**：σ ∈ [0.5, 5]，I_TY = 4/σ²（解析
         Fisher 迹）；
      **6. materials_diffusion（新增）**：κ̃ = κ/κ_ref ∈ [0.01, 1]
         （κ_ref = 1e-4），I_TY = κ̃（**v0 代理**：q7 = κ/L ∝ κ；口径与
         q1 族不同，已登记为 q_definitions 待办）。
    新增族的用途：让 DB5 的**模拟器验证证据**（control/acoustics/materials
    域）能被 `db5_feedback.DOMAIN_TO_FAMILY` 映射回流（第 90 条诊断的缺口）。
    动作 = 8 个：{stay, ±1 箱, ±3 箱}（参数位移）+ {族+1, 族−1}（结构编辑）
    + **终止**（M6.2：任意状态停下取奖励——固定 horizon 游走漂移的解药，
    第 55 条 M6 失败模式的对策）。
    任务（requirement_m6）：需求 I_TY ≈ 0.4 nats（精密测量工况）：
      R = exp(−((I_TY−0.4)/0.05)²)——landau 边界区与 elastic 软弹簧区两个
      解区；two_level 无解（I_TY≥4）；新增三族各有一个解区（θ=0.8、
      σ≈3.16、κ̃=0.4）——跨族多解发现的最小化问题。
    """

    def __init__(self, n_bins: int = 33, target_i: float = 0.4,
                 sigma_i: float = 0.05, switch_bonus: float = 0.4,
                 extra_families: bool = False):
        """`extra_families`（2026-10-02，第 91 条）：**默认 False（opt-in）**
        ——保持既有实验（v6c/v7 等按 3 族 100 状态标定的协议）不受影响；
        只有回流域（`v17_reflow_check.py`）显式开启 6 族 199 状态。"""
        self.n_bins = n_bins
        self.family_names = ["landau", "elastic", "two_level"]
        if extra_families:
            # 缺口① 回流域（2026-10-02，第 91 条）：使 DB5 的**模拟器验证
            # 证据**（control/acoustics/materials 域）可映射到 RL 状态空间。
            self.family_names += ["control_ou", "acoustics_rayleigh",
                                  "materials_diffusion"]
        self.n_families = len(self.family_names)
        self.absorbing = self.n_families * n_bins   # 终止态索引
        self.n_states = self.n_families * n_bins + 1
        self.n_actions = 8
        self.target_i = target_i
        self.sigma_i = sigma_i
        # 结构编辑的固有信息价值（v0 标定 0.4 nats/次）：族切换 = 探索新族 =
        # 获取结构信息——形状项与任务奖励共同定义生产目标（正式标定属 B v1）
        self.switch_bonus = switch_bonus

        enc = SystemEncoder()
        grids = []
        tables: list[list[float]] = []
        # landau
        a_grid = np.linspace(-2.0, 2.0, n_bins)
        i_landau = []
        for a in a_grid:
            entry = {"family": "landau", "channel_template": "CT-LANDAU-001",
                     "beta": 1.0,
                     "parameters": [
                         {"name": "a", "default": float(a)},
                         {"name": "b", "default": 1.0},
                         {"name": "kT", "default": 1.0,
                          "is_bridge_constant": True, "unit": "energy"}]}
            bb = compute_backbone(enc.encode(entry))
            i_landau.append(float(
                bb["node_fisher"]["landau__T__0"]["fisher_trace"]))
        # elastic：I_TY = 3k/kT（kT=1）；对数网格保证软弹簧解区 k≈0.133
        # 落在网格上（线性网格时 [0.117,0.15] 落在箱间、无解点——v6 设计修正）
        k_grid = np.logspace(np.log10(0.1), np.log10(10.0), n_bins)
        i_elastic = [3.0 * k for k in k_grid]
        # two_level：I_TY = 1/(p(1−p))
        p_grid = np.linspace(0.1, 0.9, n_bins)
        i_two = [1.0 / (p * (1.0 - p)) for p in p_grid]
        grids = [a_grid, k_grid, p_grid]
        tables = [i_landau, i_elastic, i_two]

        if extra_families:
            # control_ou：θ ∈ [0.2, 5]，**已推导的桥接式** I_TY = q1 = θ/2
            # （DB5 解析验证：q1 = q9/2 = θ/2）→ 值域 [0.1, 2.5]
            th_grid = np.linspace(0.2, 5.0, n_bins)
            grids.append(th_grid)
            tables.append([0.5 * t for t in th_grid])
            # acoustics_rayleigh：σ ∈ [0.5, 5]，**解析式** I_TY = 4/σ²
            # （Fisher 迹，见 DB2 族表 / q_definitions）→ 值域 [0.16, 16]
            sg_grid = np.linspace(0.5, 5.0, n_bins)
            grids.append(sg_grid)
            tables.append([4.0 / (s * s) for s in sg_grid])
            # materials_diffusion：κ̃ = κ/κ_ref ∈ [0.01, 1]（κ_ref = 1e-4），
            # **v0 代理** I_TY = κ̃（因 q7 = κ/L ∝ κ）——口径与 q1 族不同，
            # 属近似（已登记为 q_definitions 待办项，第 91 条）
            kt_grid = np.logspace(np.log10(0.01), np.log10(1.0), n_bins)
            grids.append(kt_grid)
            tables.append([float(k) for k in kt_grid])

        self.grids = grids
        self.i_tables = tables
        self.param_names = (["a", "k", "p"]
                            + (["theta", "sigma", "kappa_rel"]
                               if extra_families else []))

    def _decode(self, state: int) -> tuple[int, int]:
        return state // self.n_bins, state % self.n_bins

    def _encode(self, fam: int, bin_: int) -> int:
        return fam * self.n_bins + bin_

    def step(self, state: int, action: int) -> tuple[int, float]:
        if state == self.absorbing:
            return self.absorbing, 0.0
        a = action % self.n_actions
        if a == 7:   # 终止（M6.2）：停在此状态取奖励
            return self.absorbing, self.reward(state)
        fam, b = self._decode(state)
        if a <= 4:   # 参数位移
            delta = [0, -1, 1, -3, 3][a]
            b = int(np.clip(b + delta, 0, self.n_bins - 1))
        else:        # 族切换（保持箱位）——结构编辑
            fam = (fam + (1 if a == 5 else -1)) % self.n_families
        nxt = self._encode(fam, b)
        return nxt, self.reward(nxt)

    def is_terminal_action(self, action: int) -> bool:
        return (action % self.n_actions) == 7

    def action_bonus(self, state: int, action: int) -> float:
        """逐步形状奖励：族切换动作的固有信息价值（gflownet 训练时计入
        logR；与任务奖励共同定义生产目标，v6 第四轮诊断后引入）。"""
        return self.switch_bonus if (action % self.n_actions) in (5, 6) \
            else 0.0

    def reverse_action(self, action: int) -> int:
        """反向动作映射（可学习 P_B 的边会计）：stay↔stay、±1↔∓1、
        ±3↔∓3、族+1↔族−1、终止↔终止。边界裁剪边的近似（v0.1 文档化）。"""
        return [0, 2, 1, 4, 3, 6, 5, 7][action % self.n_actions]

    def reward(self, state: int) -> float:
        if state == self.absorbing:
            return 0.0
        fam, b = self._decode(state)
        i_ty = self.i_tables[fam][b]
        return float(max(np.exp(-((i_ty - self.target_i)
                                  / self.sigma_i) ** 2), 1e-6))

    def table(self) -> dict:
        return {
            "families": self.family_names,
            "grids": self.grids,
            "i_tables": [np.array(t) for t in self.i_tables],
            "reward": np.array([self.reward(s)
                                for s in range(self.n_states)]),
        }
