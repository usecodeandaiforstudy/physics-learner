# -*- coding: utf-8 -*-
"""**设计空间 RL 接口（可选层）**（2026-10-02，第 169 条；架构 §13.2）。

## 定位（为什么需要它）
§13.2：**C 主导生成**（需求 → 设计分布），RL **仅在目标非信息量时**才需要
——例如「**在满足信息量需求的前提下把成本压到最低**」。信息量目标本身由
解析/生成路径解决，**不该用 RL 硬学**。

## 本模块做什么（与不做什么）
    · **做**：把 5 个已登记设计域接成**设计向量空间**的 RL 接口——
      状态 = `encode_design` 的定长向量；动作 = 有界扰动 δ；奖励 =
      「是否达标（生产验证器判定）」+ **成本项**（非信息量目标）；
      非法/未达标设计**奖励为负**（与 `unknown ≠ pass` 同语义）；
    · **不做**：**不声明任何 RL 性能**。本轮只验证**接口与门禁**：
      R1 接口可用、R2 约束门禁、R3 基线对比（如实报告）、R4 无不可能的数。
      （项目纪律：RL 侧结论需 20 seed + 预登记；此处**只给基线**。）

用法：
    python pipeline/rl_design_env.py            # 20 seed 接口门禁检查
    python pipeline/rl_design_env.py --env      # 只跑单域 smoke
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

RUNS = _ROOT / "v38_rl_interface.csv"
DOMAINS = ("two_level", "rayleigh", "landau", "chain", "array")
COST_LAMBDA = 0.5          # 成本项权重（非信息量目标）
N_EVAL = 200               # 每次运行的评估预算
TARGET = 3.0               # 信息量需求（q1 ≥ 3）


class DesignEnv:
    """**设计向量空间环境**（统一接口，覆盖 5 个已登记域）。"""

    def __init__(self, domain: str, target: float = TARGET):
        from pipeline.design_space import DOMAINS as DOM, vector_dim
        if domain not in DOM:
            raise KeyError("未登记域：%s" % domain)
        self.domain = domain
        self.target = float(target)
        self.dim = int(vector_dim(domain))
        self.rng = np.random.default_rng(0)

    # ---------------------------------------------------------------- 核心
    def decode(self, z) -> dict:
        from pipeline.design_space import decode_design
        return decode_design(self.domain, np.clip(np.asarray(z, float),
                                                  0.0, 1.0))

    def reward(self, z) -> tuple[float, dict]:
        """奖励 = 达标(+1/−1) − λ·成本代理；**未达标一律为负**（门禁语义）。"""
        from pipeline.design_space import verify_design
        x = self.decode(z)
        v = verify_design(self.domain, x, target=self.target)
        meets = bool(v.get("meets"))
        cost = self._cost(x)
        r = (1.0 if meets else -1.0) - COST_LAMBDA * cost
        return float(r), {"meets": meets, "cost": float(cost),
                          "measured": v.get("measured")}

    def _cost(self, x: dict) -> float:
        """**成本代理（非信息量目标）**：按域取一个"越小越省"的量并归一。"""
        try:
            if self.domain == "rayleigh":
                return float(np.clip(float(x["D_m"]) / 3.0, 0, 1))
            if self.domain == "array":
                return float(np.clip(float(x["N"]) / 4096.0, 0, 1))
            if self.domain == "two_level":
                return float(np.clip(float(x["T_K"]) / 2000.0, 0, 1))
            if self.domain == "landau":
                return float(np.clip(abs(float(x["a"])) / 2.0, 0, 1))
            if self.domain == "chain":
                nodes = x.get("nodes") or ()
                return float(np.clip(len(nodes) / 3.0, 0, 1))
        except Exception:  # noqa: BLE001
            pass
        return 0.0

    def random_z(self) -> np.ndarray:
        return self.rng.random(self.dim)

    def step(self, z, delta) -> tuple[np.ndarray, float, dict]:
        z2 = np.clip(np.asarray(z, float) + np.asarray(delta, float), 0.0, 1.0)
        r, info = self.reward(z2)
        return z2, r, info


def run_random(env: DesignEnv, n: int = N_EVAL, seed: int = 0) -> dict:
    """**基线 1：均匀随机搜索**。"""
    rng = np.random.default_rng(seed)
    best, best_info = -np.inf, {}
    for _ in range(n):
        r, info = env.reward(rng.random(env.dim))
        if r > best:
            best, best_info = r, info
    return {"best_reward": float(best), "best_meets": bool(
        best_info.get("meets")), "n": n}


def run_es(env: DesignEnv, n: int = N_EVAL, seed: int = 0,
           sigma: float = 0.15) -> dict:
    """**基线 2：(1+1) 演化策略**（黑箱局部搜索；仅验证接口，非 RL 性能声明）。"""
    rng = np.random.default_rng(seed)
    z = rng.random(env.dim)
    r, info = env.reward(z)
    best, best_info = r, info
    for _ in range(n - 1):
        zc, rc, ic = env.step(z, rng.normal(0.0, sigma, env.dim))
        if rc >= r:
            z, r, info = zc, rc, ic
        if rc > best:
            best, best_info = rc, ic
    return {"best_reward": float(best), "best_meets": bool(
        best_info.get("meets")), "n": n}


def main() -> int:
    only_smoke = "--env" in sys.argv
    seeds = 20
    if "--seeds" in sys.argv:
        seeds = int(sys.argv[sys.argv.index("--seeds") + 1])
    t0 = time.time()
    print("=" * 74)
    print("设计空间 RL 接口（可选层）：接口与门禁验证（**不做性能声明**）")
    print("=" * 74)
    print("域：%s；状态 = 设计向量（`encode_design`）；动作 = 有界扰动；"
          "奖励 = 达标±1 − %.1f·成本" % (list(DOMAINS), COST_LAMBDA))
    print("需求：`q1 ≥ %.1f`；**未达标者奖励为负**（门禁语义）\n" % TARGET)

    # ---- R1 接口可用 + R4 无不可能的数（单域 smoke）----
    r1 = True
    for dom in DOMAINS:
        try:
            env = DesignEnv(dom)
            z = env.random_z()
            _z2, r, info = env.step(z, np.zeros(env.dim))
            ok = math.isfinite(r) and isinstance(info, dict)
            r1 = r1 and ok
            print("  [%s] %-18s 维度=%d 奖励=%.4f 达标=%s 成本=%.3f"
                  % ("✓" if ok else "✗", dom, env.dim, r, info["meets"],
                     info["cost"]))
        except Exception as ex:  # noqa: BLE001
            r1 = False
            print("  [✗] %-18s 异常 %s: %s" % (dom, type(ex).__name__, ex))
    print("\nR1 **接口可用**（5 域状态/动作/奖励形状一致）⟹ %s"
          % ("**通过** ✓" if r1 else "**未通过**"))
    if only_smoke:
        return 0 if r1 else 1

    # ---- R2 门禁 + R3 基线对比（20 seed）----
    rows = []
    print("\n%4s %-10s %14s %14s %8s" % ("seed", "域", "随机基线", "(1+1)-ES",
                                         "门禁"))
    for dom in DOMAINS:
        env = DesignEnv(dom)
        for s in range(seeds):
            a = run_random(env, seed=s)
            b = run_es(env, seed=s)
            gate_ok = True
            # **R2 门禁**：未达标的设计奖励必须 ≤ −1（不得被当成成功）
            r_neg, info_neg = env.reward(np.zeros(env.dim))
            if info_neg["meets"] is False and r_neg > -1.0 + 1e-9:
                gate_ok = False
            rows.append({"domain": dom, "seed": s,
                         "random_best": a["best_reward"],
                         "es_best": b["best_reward"],
                         "random_meets": a["best_meets"],
                         "es_meets": b["best_meets"],
                         "gate_ok": gate_ok})
            if s < 2:
                print("%4d %-10s %14.4f %14.4f %8s"
                      % (s, dom, a["best_reward"], b["best_reward"],
                         "✓" if gate_ok else "**✗**"))
    arr = {k: np.array([r[k] for r in rows], dtype=float)
           for k in ("random_best", "es_best")}
    gate_all = all(r["gate_ok"] for r in rows)
    print("\nR2 **约束门禁**（未达标 ⟹ 奖励 ≤ −1）：%s"
          % ("**通过** ✓" if gate_all else "**未通过**"))
    diff = arr["es_best"] - arr["random_best"]
    print("R3 **基线对比**（**(1+1)-ES** vs 均匀随机，%d seed × %d 域 = %d 次）"
          % (seeds, len(DOMAINS), len(rows)))
    print("   随机基线均值 **%.4f**；ES 均值 **%.4f**；差 **%+.4f**；"
          "ES 更好的比例 **%.0f%%**"
          % (arr["random_best"].mean(), arr["es_best"].mean(), diff.mean(),
             100 * float(np.mean(diff > 0))))
    print("   ⟹ **如实报告**：%s"
          % ("ES 优于随机（**接口能驱动改进**）" if diff.mean() > 0
             else "ES **未**优于随机（**不作性能声明**，仅接口可用）"))
    fin = bool(np.isfinite(arr["random_best"]).all()
               and np.isfinite(arr["es_best"]).all())
    print("R4 **无不可能的数**（奖励全有限）⟹ %s" % ("**通过** ✓" if fin
                                                else "**未通过**"))
    with open(RUNS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n逐 seed 落盘：%s（%d 行）；耗时 %.1fs"
          % (RUNS.name, len(rows), time.time() - t0))
    print("**注**：本检查**只验证接口与门禁**；RL 性能结论须另行预登记"
          "（项目纪律：≥20 seed + 预登记）——**此处不作性能声明**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
