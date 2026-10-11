# -*- coding: utf-8 -*-
"""需求条件化 RL 环境（2026-10-02，第 92 条）。

**动机**（第 91 条诊断）：回流边的瓶颈是**任务对齐**——DB5 的设计按**各自
需求**验证（分辨力 σ / 弛豫率 θ / 扩散 κ），而旧 `MultiFamilyEnv` 只有
**单一硬编码任务**（I_TY≈0.4 双侧高斯）⟹ 证据与任务不同源。本环境把 RL
恢复成架构文档的本职：**给定需求求设计**。

**两个设计决定（都由前一节的发现驱动）**：
  1. **需求进入状态**（goal-conditioned）：`s = (需求, 族, 参数箱)`——
     同一策略可对不同需求给出不同设计；
  2. **达标语义（satisficing）而非相等语义**：
     `R = 1` if `q ≥ q_req` else `exp(−((q_req − q)/σ)²)`（下界型需求）。
     旧环境用双侧高斯 ⟹ **有余量的设计反而不算解**（DES-RAY-01 q1=3.3
     vs 需求 3.0 就被判为非解）——这与「需求 = 达标」的物理语义不符。

**度量支持表**：每族声明它支持哪些 q 度量（跨族不可比，故按需求筛选族）。

用法：
    env = RequirementConditionedEnv()
    s0 = env.start_state(req_idx=1)
    s, r = env.step(s0, action)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__all__ = ["RequirementConditionedEnv", "Requirement", "REQUIREMENTS"]

KAPPA_REF = 1e-4        # materials 族参考扩散率（与 rl_env 扩展族一致）
L_DIFF = 0.01           # 特征长度（DES-DIF-01 口径）


@dataclass(frozen=True)
class Requirement:
    """一条需求：度量 + 目标值 + 容差（下界型：q ≥ target）。"""
    metric: str          # "q1" | "q7" | "q9"
    target: float
    sigma: float         # 软惩罚宽度（默认 5%·target，与余量政策一致）
    label: str = ""


# 4 条需求：R0 = 旧单任务（保持可比），R1–R3 = DB5 模拟器证据各自的需求
REQUIREMENTS: list[Requirement] = [
    Requirement("q1", 0.40, 0.02, "legacy 精密测量（旧 env 任务）"),
    Requirement("q1", 3.00, 0.15, "acoustics 分辨力（DES-RAY-01）"),
    Requirement("q9", 2.30, 0.115, "control 弛豫率（DES-OU-01）"),
    Requirement("q7", 0.01, 0.0005, "materials 输运（DES-DIF-01）"),
]

# **约束型需求（2026-10-02 加，第 115 条）**：阈值/约束型需求**不用 β**，
# 走本环境既有的**达标型奖励**（`R=1 if q≥target else exp(−((target−q)/σ)²)`）。
# 与任务 3 的结论一致（第 114 条）：势垒是**约束**，不是可用 β 权衡的目标。
CONSTRAINT_REQUIREMENTS: list[Requirement] = [
    Requirement("barrier", 0.60, 0.03,
                "landau 势垒（约束型：ΔF ≥ 0.6 nats）"),
]

# **族级容量表（工况索引约束用）**：族的信道容量（I_TY 上限，nats）——
# 取自 DB2 各行 `ib_curve.I_TY` 的**最大值**（最好情形；逐行差异很大，
# 故标注为"族上限"）。用于判断「适应 M 种工况」这类**族级约束**。
FAMILY_CAPACITY_HINT = {
    "landau": 0.1812, "elastic": None, "two_level": 0.1190,
    "control_ou": 0.7170, "acoustics_rayleigh": None,
    "materials_diffusion": 0.1748,
}


def family_capacity_from_db2(family: str) -> float | None:
    """从 DB2 取该族的容量上限（`I_TY` 最大值）；无曲线则返回 None。"""
    import json as _json
    from pathlib import Path as _Path
    p = _Path(__file__).resolve().parent.parent / "db2" / "db2.jsonl"
    if not p.exists():
        return None
    best = None
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if rec.get("family") != family:
                continue
            c = rec.get("ib_curve") or {}
            y = c.get("I_TY") or []
            if y:
                m = float(max(y))
                best = m if best is None else max(best, m)
    return best


def families_supporting_env_count(n_env: int) -> list[str]:
    """返回信道容量足以区分 `n_env` 种工况的族（容量 ≥ ln M）。

    **这是族级约束**（不是逐状态奖励）：工况索引信息量取决于**族**的信道
    容量，与设计参数取值无关 ⟹ 它该在**族选择**阶段筛，而不是进 RL 状态空间。
    """
    import math as _math
    need = _math.log(max(int(n_env), 1))
    out = []
    for fam, hint in FAMILY_CAPACITY_HINT.items():
        cap = family_capacity_from_db2(fam)
        if cap is None:
            cap = hint
        if cap is not None and cap >= need:
            out.append(fam)
    return out


class RequirementConditionedEnv:
    """需求条件化的跨族设计环境（6 族 × 33 箱 × 4 需求 + 吸收态）。"""

    def __init__(self, n_bins: int = 33,
                 requirements: list[Requirement] | None = None,
                 extra_constraints: bool = False):
        """`extra_constraints=True` 时追加**约束型需求**（势垒）——**opt-in**，
        默认 False 以保持既有状态空间（793）与脚本兼容。"""
        self.n_bins = n_bins
        reqs = list(requirements or REQUIREMENTS)
        if extra_constraints and requirements is None:
            reqs = reqs + list(CONSTRAINT_REQUIREMENTS)
        self.requirements = reqs
        self.n_req = len(self.requirements)
        self.family_names = ["landau", "elastic", "two_level", "control_ou",
                             "acoustics_rayleigh", "materials_diffusion"]
        self.n_families = len(self.family_names)
        self.absorbing = self.n_req * self.n_families * n_bins
        self.n_states = self.absorbing + 1
        self.n_actions = 8

        nb = n_bins
        a_grid = np.linspace(-2.0, 2.0, nb)
        # landau 的 q1 表：解析近似（双阱 Fisher，边界处 ~0.389）——与
        # rl_env 的编码器结果同量级；此处用解析式避免重复编码开销
        i_landau = 0.4 * np.exp(-((np.abs(a_grid) - 0.35) / 0.6) ** 2) + 0.02
        k_grid = np.logspace(np.log10(0.1), np.log10(10.0), nb)
        i_elastic = 3.0 * k_grid
        p_grid = np.linspace(0.1, 0.9, nb)
        i_two = 1.0 / (p_grid * (1.0 - p_grid))
        th_grid = np.linspace(0.2, 5.0, nb)
        sg_grid = np.linspace(0.5, 5.0, nb)
        # **materials 网格扩展（2026-10-02 第 94 条）**：κ̃ ∈ [0.01, 10]
        # （初版 [0.01, 1]）——需求 q7 ≥ 0.01 ⟺ κ̃ ≥ 1，初版网格下**只有
        # 顶端 1 个状态满足**（R3 解状态 =1 是**设计空间覆盖不足**的伪影，
        # 而非任务固有难度）；DB5 的 κ 扫描已到 3e-4（κ̃=3）⟹ 设计空间
        # 本应覆盖到 κ̃=10。
        kt_grid = np.logspace(np.log10(0.01), np.log10(10.0), nb)

        self.grids = [a_grid, k_grid, p_grid, th_grid, sg_grid, kt_grid]
        self.param_names = ["a", "k", "p", "theta", "sigma", "kappa_rel"]
        # metric → {family: 数值表}
        self.metric_tables: dict[str, dict[int, np.ndarray]] = {
            "q1": {0: i_landau, 1: i_elastic, 2: i_two,
                   3: 0.5 * th_grid, 4: 4.0 / sg_grid ** 2},
            "q9": {3: th_grid},
            "q7": {5: kt_grid * KAPPA_REF / L_DIFF},
            # **约束型度量（第 115 条）**：landau 双阱势垒 ΔF = a²/4（a<0；
            # a≥0 单阱 ⟹ 无垒记 0）。达标型奖励据此判定（**不用 β**）。
            "barrier": {0: np.where(a_grid < 0, a_grid ** 2 / 4.0, 0.0)},
        }

    # ---- 状态编解码 ----
    def _encode(self, req: int, fam: int, bin_: int) -> int:
        return (req * self.n_families + fam) * self.n_bins + bin_

    def _decode(self, state: int) -> tuple[int, int, int]:
        req, rest = divmod(state, self.n_families * self.n_bins)
        fam, b = divmod(rest, self.n_bins)
        return req, fam, b

    def start_state(self, req_idx: int, rng: np.random.Generator | None = None
                    ) -> int:
        rng = rng or np.random.default_rng()
        fam = int(rng.integers(0, self.n_families))
        b = int(rng.integers(0, self.n_bins))
        return self._encode(req_idx, fam, b)

    # ---- 动力学 ----
    def step(self, state: int, action: int) -> tuple[int, float]:
        if state == self.absorbing:
            return self.absorbing, 0.0
        a = action % self.n_actions
        if a == 7:                      # 终止：停在此状态取奖励（M6.2 语义）
            return self.absorbing, self.reward(state)
        req, fam, b = self._decode(state)
        if a <= 4:
            delta = [0, -1, 1, -3, 3][a]
            b = int(np.clip(b + delta, 0, self.n_bins - 1))
        else:
            fam = (fam + (1 if a == 5 else -1)) % self.n_families
        nxt = self._encode(req, fam, b)
        return nxt, self.reward(nxt)

    def is_terminal_action(self, action: int) -> bool:
        return (action % self.n_actions) == 7

    def reverse_action(self, action: int) -> int:
        return [0, 2, 1, 4, 3, 6, 5, 7][action % self.n_actions]

    # ---- 奖励（达标语义）----
    def metric_value(self, state: int) -> float | None:
        """该状态在其需求度量下的数值；族不支持该度量时返回 None。"""
        if state == self.absorbing:
            return None
        req, fam, b = self._decode(state)
        table = self.metric_tables.get(self.requirements[req].metric, {})
        if fam not in table:
            return None
        return float(table[fam][b])

    def reward(self, state: int) -> float:
        if state == self.absorbing:
            return 0.0
        v = self.metric_value(state)
        if v is None:
            return 0.0                   # 族不支持该需求度量 → 非解
        r = self.requirements[self._decode(state)[0]]
        if v >= r.target:
            return 1.0                   # **达标即满分**（satisficing）
        return float(max(np.exp(-((r.target - v) / r.sigma) ** 2), 1e-6))

    def solution_states(self, req_idx: int) -> list[int]:
        return [self._encode(req_idx, fam, b)
                for fam in range(self.n_families)
                for b in range(self.n_bins)
                if self.reward(self._encode(req_idx, fam, b)) >= 0.999]

    def table(self) -> dict:
        return {"families": self.family_names, "grids": self.grids,
                "requirements": [r.__dict__ for r in self.requirements],
                "reward": np.array([self.reward(s)
                                    for s in range(self.n_states)])}
