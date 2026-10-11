# -*- coding: utf-8 -*-
"""**速率型适配器（RateAdapter）**（2026-10-02，第 152 条）。

## 缺口（第 151 条量化）
DB3 的速率型需求（52 条 = 33%）里，**按度量可路由 98%**，但**能力可达仅
27/51**——例如 `q9 ≥ 150 s⁻¹`（机电执行器 20 ms 建立）甚至 `q9 ≥ 1e8 s⁻¹`
（光电探测 10 ns），而唯一登记的速率适配器 `ou` 的 `theta ∈ [1e-3, 10]`
**远达不到**。

## 本模块（"解析公式 + 数值估计"双轨）
对**谱隙有解析式**的族各给一个速率模型：
    | 族 | 谱隙 `λ₂`（解析） | 物理 |
    |---|---|---|
    | `poisson_count` | `rate` | 泊松计数过程的生成元第二特征值 = 率 |
    | `radioactive_decay` | `rate` | 指数衰减：`λ = 1/τ`（`τ` 为平均寿命） |
    | `two_level` | `1/(2·p·(1−p))`·（归一化） | 二能级马尔可夫链的弛豫率 |
**数值估计**（真值自检用，与解析对照）：
    · 泊松：模拟计数 ⟹ `λ̂ = 平均计数/Δt`；
    · 衰减：模拟寿命 ⟹ `λ̂ = 1/mean(t)`；
    · 二能级：模拟两态链 ⟹ 由 lag-1 自相关 `a` 得 `λ̂ = −ln(a)/Δt`。
⟹ **验证用数值估计，真值自检比解析式**（与既有适配器同构）。

## 有效域（登记，物理范围而非 DB1 扫描范围）
DB1 模板的 `rate ∈ [0.1, 10]` 是**扫描范围**；速率在物理上**无上界**
（光电探测 1e8 s⁻¹ 是常规值）⟹ 本适配器登记 **`[1e-3, 1e12] s⁻¹`**，
并**在文档与证据里标明这是"验证器的有效域"**（与 DB1 扫描范围不同口径）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

__all__ = ["RateAdapter", "RATE_MODELS", "RATE_RANGE"]

RATE_RANGE = (1e-3, 1e12)          # **验证器有效域**（s^-1；物理范围，非扫描范围）


def _lam_poisson(p: dict) -> float:
    return float(p["rate"])


def _lam_decay(p: dict) -> float:
    return float(p["rate"])


def _lam_two_level(p: dict) -> float:
    """**已弃用（保留供审计）**：`λ = 1/(2p(1−p))` 是**我编的归一化**，不是物理。

    **为何移除**：`p` 是**平衡占据**（热力学量），而谱隙是**动力学量**——
    要由二能级系统给出 `λ₂` 必须有**动力学参数**（尝试频率 `ν`/势垒 `ΔF`），
    而 DB1 的 `two_level` 条目**只有 `p`** ⟹ **该族无法提供解析谱隙**。
    实测：真值自检 `λ̂/λ` 差 **65%（p=0.5）与 96%（p=0.1）** ⟹ 如实移除。
    """
    pp = float(p["p"])
    return 1.0 / max(2.0 * pp * (1.0 - pp), 1e-12)


RATE_MODELS: dict[str, dict] = {
    "poisson_count": {"params": ["rate"], "lam": _lam_poisson,
                      "label": "泊松计数（λ₂ = 率）"},
    "radioactive_decay": {"params": ["rate"], "lam": _lam_decay,
                          "label": "指数衰减（λ₂ = 1/τ）"},
    # **`two_level` 不在此表**（见 `_lam_two_level` 的说明：缺动力学参数）
}


class RateAdapter:
    """速率型适配器：**数值估计** `λ̂`（真值与解析式对照）。"""

    name: str = "rate"

    def run(self, spec: dict, n_samples: int = 20000, seed: int = 0,
            **kw) -> dict:
        fam = spec.get("family") or spec.get("kind")
        meta = RATE_MODELS.get(fam)
        if meta is None:
            raise KeyError("未支持的速率族：%s" % fam)
        p = {k: float(spec[k]) for k in meta["params"]}
        lam_true = float(meta["lam"](p))
        rng = np.random.default_rng(seed)
        if fam == "poisson_count":
            dt = float(spec.get("dt", 1e-3))
            n = int(max(n_samples, 100))
            counts = rng.poisson(lam_true * dt, size=n)
            lam_hat = float(counts.mean() / dt)
        elif fam == "radioactive_decay":
            n = int(max(n_samples, 100))
            t = rng.exponential(1.0 / max(lam_true, 1e-12), size=n)
            lam_hat = float(1.0 / max(t.mean(), 1e-12))
        else:                                   # two_level：两态链自相关
            n = int(max(n_samples, 100))
            q = 1.0 / (2.0 * lam_true)          # 单步翻转概率（归一化时间）
            x = np.empty(n, dtype=float)
            x[0] = 1.0 if rng.random() < p["p"] else 0.0
            flip = rng.random(n) < q
            for i in range(1, n):
                x[i] = (1.0 - x[i - 1]) if flip[i] else x[i - 1]
            xc = x - x.mean()
            denom = float(np.dot(xc[:-1], xc[:-1]))
            a = float(np.dot(xc[:-1], xc[1:]) / denom) if denom > 0 else 0.0
            lam_hat = float(-np.log(max(a, 1e-12)))
        return {"obs": None, "family": fam, "lam_true": lam_true,
                "lambda_hat": lam_hat, "label": meta["label"],
                # **前缀匹配用**：验证器以 `q9` 前缀取值
                "q9_spectral_gap": lam_hat}


def main() -> int:
    """**真值自检**：数值估计 `λ̂` vs 解析 `λ`（判据 ≤10%）。"""
    print("=" * 74)
    print("速率型适配器（RateAdapter）真值自检")
    print("=" * 74)
    print("有效域：`[%.0e, %.0e] s^-1`（**验证器有效域**，物理范围；"
          "DB1 扫描范围是 [0.1, 10]）" % RATE_RANGE)
    cases = [({"family": "poisson_count", "rate": 1e3}, "poisson_count"),
             ({"family": "poisson_count", "rate": 1e8}, "poisson_count"),
             ({"family": "radioactive_decay", "rate": 1e2}, "decay"),
             ({"family": "radioactive_decay", "rate": 1e7}, "decay")]
    print("\n（`two_level` **不在表内**：`p` 是平衡量，谱隙需动力学参数——"
          "真值自检曾差 65%/96%，如实移除）")
    fails = 0
    print("\n%-18s %12s %12s %8s %s" % ("族/参数", "解析 λ", "数值 λ̂",
                                        "相对差", "判定"))
    for spec, tag in cases:
        est = []
        for s in range(3):
            out = RateAdapter().run(spec, n_samples=200000, seed=s)
            est.append(out["lambda_hat"])
        lam = RateAdapter().run(spec, n_samples=1000, seed=0)["lam_true"]
        e = float(np.median(est))
        rel = abs(e - lam) / max(abs(lam), 1e-12)
        ok = rel <= 0.10
        fails += int(not ok)
        print("%-18s %12.5g %12.5g %7.1f%% %s"
              % (tag + str({k: v for k, v in spec.items() if k != "family"}),
                 lam, e, 100 * rel, "✓" if ok else "**✗ 超 10%**"))
    print("\n%s" % ("===== 全部通过 =====" if fails == 0
                    else "===== 有 %d 项失败 =====" % fails))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
