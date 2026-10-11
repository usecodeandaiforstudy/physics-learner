# -*- coding: utf-8 -*-
"""模拟器接口自检（缺口①，2026-10-02）——真值对照。

**判据**：模拟器测得的 q 与**解析真值**的相对误差（容差 5%，动力学类
放宽到 10%，因为轨迹估计依赖有限步长）。

  S1 rayleigh 族 q1：score 估计 vs 4/σ²
  S2 weibull 族 q1：score 估计 vs 解析式
  S3 lognormal 族 q1：score 估计 vs 3/σ²
  S4 frechet 族 q1：score 估计 vs 解析式
  S5 OU 动力学：轨迹回归 θ_hat vs 真 θ（谱间隙 q9）
  S6 diffusion 动力学：数值半值穿透时间 → κ_hat vs 真 κ（q7 关系）

用法（项目根目录）：python sim/sim_check.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sim.simulator_adapter import FAMILY_SPECS, get_adapter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def main() -> int:
    fails = 0
    print("================ 模拟器接口自检（真值对照） ================")
    ad = get_adapter("analytic")
    cases = [
        ("S1", "rayleigh", {"sigma": 1.1}, 0.05),
        ("S2", "weibull", {"shape": 1.8, "scale": 2.0}, 0.05),
        ("S3", "lognormal", {"mu": 0.0, "sigma": 0.7}, 0.05),
        ("S4", "frechet", {"shape": 2.0, "scale": 1.5}, 0.05),
    ]
    for sid, fam, params, tol in cases:
        spec = {"family": fam, **params}
        res = ad.run(spec, n_samples=200000, seed=0)
        q_hat = res["q1_fisher_directional"]
        q_true = FAMILY_SPECS[fam]["fisher_trace"](res["params"])
        err = rel_err(q_hat, q_true)
        ok = err < tol
        fails += 0 if ok else 1
        print("  [%s] %s q1：估计 %.4g vs 解析 %.4g  相对误差 %.2f%%  %s"
              % ("PASS" if ok else "FAIL", sid + " " + fam, q_hat, q_true,
                 100 * err, "" if ok else "<-- 超容差"))

    dyn = get_adapter("dynamics")
    res = dyn.run({"kind": "ou", "theta": 0.5, "sigma": 1.0},
                  n_samples=200000, seed=0)
    err = rel_err(res["theta_hat"], res["theta_true"])
    ok = err < 0.10
    fails += 0 if ok else 1
    print("  [%s] S5 ou 谱间隙：θ_hat %.4f vs 真值 %.4f  相对误差 %.2f%%"
          % ("PASS" if ok else "FAIL", res["theta_hat"], res["theta_true"],
             100 * err))

    for kappa in (1e-6, 1e-4):
        res = dyn.run({"kind": "diffusion", "kappa": kappa, "L": 0.01},
                      seed=0)
        err = rel_err(res["kappa_hat"], res["kappa_true"])
        ok = err < 0.05
        fails += 0 if ok else 1
        print("  [%s] S6 diffusion κ=%.0e：κ_hat %.4g vs 真值 %.4g  "
              "相对误差 %.2f%%  （q7 数值 %.4g vs 解析 %.4g）"
              % ("PASS" if ok else "FAIL", kappa, res["kappa_hat"],
                 res["kappa_true"], 100 * err,
                 res["q7_info_propagation_speed"], res["q7_analytic"]))

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
