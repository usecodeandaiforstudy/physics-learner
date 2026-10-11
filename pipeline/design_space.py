# -*- coding: utf-8 -*-
"""**设计空间注册表：状态 = 设计向量**（2026-10-02，第 139 条；架构 §13.2）。

## 架构修正的实质
旧架构把 **A 网络的嵌入**当成 RL 的"状态"，但那是**表征**、不是**设计**：
```
旧：state = 793–991 个抽象离散态          （表征层）
新：state = **设计向量/场**（连续，各域登记） （设计层）
     A 的嵌入退为**条件化输入**（conditioning），不再当状态
```
**依据**：中游建成的非退化域表明，真实设计是连续高维的——
`two_level: (ΔF, T)`、`rayleigh: (D, f)`、`landau: (a, b, kT)`。

## 本模块提供
    `DOMAINS`：域 → {dof 名称, 取值范围, 需求类型, 验证函数, 可行域构造}
    `design_vector(domain, x)` / `from_vector(domain, x)`：向量的规范化/反规范化
    `verify_design(domain, x, target)`：**经既有编码器/生产验证 API** 校验
    `feasible_set(domain, target)`：解流形（非退化域给**一维射线**）

用法：
    from pipeline.design_space import DOMAINS, design_vector, verify_design
    print(DOMAINS["rayleigh"]["dof"])          # ['D_m', 'f_Hz']
    ok = verify_design("rayleigh", {"D_m": 0.8, "f_Hz": 1.6e6}, target=3.0)
"""

from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402


def _verify_two_level(x: dict, target: float | None = None) -> dict:
    """**注意口径（2026-10-02 修）**：需求是 `q1 ≥ target`（**下界**），
    不是 `q1 ≈ target`。初版把 `target` 传给 `q1_expected`（**等值判据**）
    ⟹ 流形上的设计（p=0.10 ⟹ q1=11.1）被判"不达标"（相对差 270%）
    ⟹ reflow 里 **200/200 domain_fail**。现改为**下界判据**。
    """
    from pipeline.design_two_level import EV, verify_design
    v = verify_design(float(x["dF_eV"]) * EV, float(x["T_K"]))
    q1 = v.get("q1_from_encoder")
    meets = None if (q1 is None or target is None) else bool(
        q1 >= float(target) * (1.0 - 1e-6))
    return {"metric": "q1", "measured": q1, "dpi_ok": v.get("dpi_ok"),
            "meets": meets}


def _verify_rayleigh(x: dict, target: float | None = None) -> dict:
    from pipeline.design_rayleigh import verify_design
    v = verify_design(float(x["D_m"]), float(x["f_Hz"]))
    q1 = v.get("q1_from_encoder")
    # **下界判据**（与 two_level 同口径；需求是 q1 ≥ target）
    meets = None if (q1 is None or target is None) else bool(
        q1 >= float(target) * (1.0 - 1e-6))
    return {"metric": "q1", "measured": q1, "dpi_ok": v.get("dpi_ok"),
            "meets": meets}


def _verify_landau(x: dict, target: float | None = None) -> dict:
    from pipeline.design_landau import measure
    m = measure(float(x["a"]), float(x["b"]), float(x["kT"]))
    if not m:
        return {"metric": "q1", "measured": None, "dpi_ok": None,
                "meets": None}
    meets = None if target is None else bool(m["q1"] >= float(target))
    return {"metric": "q1", "measured": m["q1"], "dpi_ok": m["dpi_ok"],
            "meets": meets, "dF_nats": m["dF_nats"]}


def _verify_chain(x: dict, target: float | None = None) -> dict:
    """**组合链域**（第 141 条）：设计 = 链（组件 + 连续参数 + 是否连边）。

    信息量 `q1` = X 节点 Fisher；**要求链 DPI 合法**（否则不是有效设计）。
    """
    from pipeline.design_chain import measure_chain
    m = measure_chain(x)
    if not m:
        return {"metric": "q1", "measured": None, "dpi_ok": None,
                "meets": None}
    meets = None if target is None else bool(
        m["q1"] >= float(target) and m["dpi_ok"])
    return {"metric": "q1", "measured": m["q1"], "dpi_ok": m["dpi_ok"],
            "meets": meets, "mi": m["mi"], "n_edges": m["n_edges"]}


def _chain_to_vector(x: dict) -> "np.ndarray":
    """链域 → **定长 7 维向量**（委托 `design_chain.to_vector`，单一真源）。"""
    from pipeline.design_chain import to_vector
    return to_vector(x)


def _chain_from_vector(z) -> dict:
    """**定长 7 维向量 → 链域设计**（委托 `design_chain.from_vector`）。"""
    from pipeline.design_chain import from_vector
    return from_vector(z)


def _verify_array(x: dict, target: float | None = None) -> dict:
    """**声学阵列几何域**（第 149 条）：要求 `q1 ≥ 目标` **且无栅瓣**。

    `σ_eff` 由阵因子首零点给出，**与 rayleigh 域交叉校验 0.00%** ✓
    """
    from pipeline.design_array import measure_array, q1_from_sigma
    m = measure_array(x)
    if not m:
        return {"metric": "q1", "measured": None, "dpi_ok": None,
                "meets": None}
    q1 = q1_from_sigma(m["sigma_eff"])
    # **有效设计** = 无栅瓣（与"DPI 违规=无效"同型的安全性质）
    meets = None if target is None else bool(
        q1 >= float(target) and not m["grating"])
    return {"metric": "q1", "measured": q1, "dpi_ok": not m["grating"],
            "meets": meets, "sigma_eff": m["sigma_eff"],
            "SLL_dB": m["SLL_dB"], "grating": m["grating"]}


def _array_to_vector(x: dict):
    from pipeline.design_array import to_vector
    return to_vector(x)


def _array_from_vector(z) -> dict:
    from pipeline.design_array import from_vector
    return from_vector(z)


# **域注册表（单一真源）**：dof / 范围 / 需求类型 / 验证函数 / 计数
DOMAINS: dict[str, dict] = {
    "two_level": {
        "dof": ["dF_eV", "T_K"],
        "range": {"dF_eV": (0.01, 2.0), "T_K": (200.0, 1500.0)},
        "family": "two_level",
        "req_types": ["param", "constraint_barrier"],
        "constraint": "p = 定住 ΔF/k_BT（比值）；绝对势垒/工况定标度",
        "verify": _verify_two_level,
    },
    "rayleigh": {
        "dof": ["D_m", "f_Hz"],
        "range": {"D_m": (0.05, 3.0), "f_Hz": (1e4, 2e6)},
        "family": "rayleigh",
        "req_types": ["param"],
        "constraint": "σ = 定住 λ/D（比值）；尺寸/频段定标度",
        "verify": _verify_rayleigh,
    },
    "landau": {
        "dof": ["a", "b", "kT"],
        "range": {"a": (-20.0, -0.01), "b": (0.1, 5.0), "kT": (0.1, 5.0)},
        "family": "landau",
        "req_types": ["param", "constraint_barrier"],
        "constraint": "势函数 V(φ)=aφ²/2+bφ⁴/4（**高维设计域**，"
                      "设计 3 dof → 信息标量 (q1, ΔF)）",
        "verify": _verify_landau,
    },
    # **组合链域（第 141 条，R5-③）**：设计 = 一条链（组件 + 连续参数 + 是否连边）
    #   ⟹ **4–6 dof**（混合：离散组件 + 连续参数）；信息侧 2 标量 (q1, ΣMI)。
    #   预检实测：可评估 400/400、**DPI 合法 41%**、非退化桶内中位 **20.0**、
    #   单边 MI ≤1.52（≤1.83 上界）✓ ⟹ **唯一的"组合型"高维设计域**。
    "chain": {
        "dof": ["comp0", "val0", "comp1", "val1", "comp2", "val2"],
        "mixed": True,          # **混合型**：偶数位为组件索引（离散）
        "range": {"val0": (0.0, 1.0), "val1": (0.0, 1.0),
                  "val2": (0.0, 1.0)},
        "family": None,         # 无单一族（链由多个族组成）
        "req_types": ["param"],
        "constraint": "链式结构：σ 必须沿链递增（**DPI 合法**）——"
                      "实测合法率 **41%**（真实边界）",
        "verify": _verify_chain,
        # **统一向量接口（第 144 条）**：混合型域自带编解码
        "to_vector": _chain_to_vector,
        "from_vector": _chain_from_vector,
        "vec_dim": 7,
    },
    # **声学阵列几何域（第 149 条，R5-①）**：设计 = 阵元数/间距/加权/频率
    #   ⟹ **4 dof（混合型：N 为整数）**；信息侧 =（σ_eff, SLL_dB）。
    #   **预检 A1–A6 全过**（含与 `rayleigh` 域**交叉校验 0.00%** 与
    #   均匀加权 SLL 解析基准 −13.26 dB）；栅瓣门禁：**broadside 下 d ≥ λ**。
    "array": {
        "dof": ["N", "d_over_lam", "taper_p", "f_Hz"],
        "mixed": True,          # N 为整数
        "range": {"d_over_lam": (0.25, 1.5), "taper_p": (0.0, 3.0),
                  "f_Hz": (1e4, 2e6)},
        "family": "rayleigh",   # **与 rayleigh 族同信息侧口径**（σ=λ/D）
        "req_types": ["param"],
        "constraint": "栅瓣门禁（broadside：`d ≥ λ` 即无效设计）；"
                      "`σ_eff` 与 rayleigh 域交叉校验 0.00%",
        "verify": _verify_array,
        "to_vector": _array_to_vector,
        "from_vector": _array_from_vector,
        "vec_dim": 5,
    },
}

__all__ = ["DOMAINS", "design_vector", "from_vector", "verify_design",
           "dof_of", "feasible_set"]


def dof_of(domain: str) -> list[str]:
    return list(DOMAINS[domain]["dof"])


def design_vector(domain: str, x: dict) -> np.ndarray:
    """设计 → **规范化向量 [0,1]^d**（用登记范围；越界不静默外推）。"""
    d = DOMAINS[domain]
    out = []
    for k in d["dof"]:
        lo, hi = d["range"][k]
        v = (float(x[k]) - lo) / (hi - lo)
        out.append(float(np.clip(v, 0.0, 1.0)))
    return np.asarray(out, dtype=float)


def from_vector(domain: str, z) -> dict:
    """规范化向量 → 设计字典。"""
    d = DOMAINS[domain]
    z = np.asarray(z, dtype=float).ravel()
    if len(z) != len(d["dof"]):
        raise ValueError("维度不符：%d vs %d" % (len(z), len(d["dof"])))
    return {k: float(lo + z[i] * (hi - lo))
            for i, (k, (lo, hi)) in enumerate(
                (k, d["range"][k]) for k in d["dof"])}


def verify_design(domain: str, x: dict, target: float | None = None) -> dict:
    """**经既有编码器/生产验证**校验一个设计（各域自带验证函数）。"""
    if domain not in DOMAINS:
        return {"metric": None, "measured": None, "dpi_ok": None,
                "meets": None, "reason": "域未登记：" + str(domain)}
    return DOMAINS[domain]["verify"](x, target)


# ---------------------------------------------------------------- 统一向量接口
# **第 144 条（任务 4）**：生成模型需要**定长实向量**；混合型域（`chain`：
# 离散组件 + 连续参数）不能走 `design_vector`（它假定纯连续且用登记范围），
# 故引入**统一分派**：域可自带 `to_vector/from_vector`，否则用标准规范化。
def vector_dim(domain: str) -> int:
    d = DOMAINS[domain]
    if d.get("to_vector") is not None:
        return int(d["vec_dim"])
    if d.get("mixed"):
        raise ValueError("域 %s 标为混合型但未提供 to_vector/from_vector"
                         % domain)
    return len(d["dof"])


def encode_design(domain: str, x: dict) -> np.ndarray:
    """设计 → **定长实向量**（统一入口；供生成模型用作数据）。"""
    d = DOMAINS[domain]
    if d.get("to_vector") is not None:
        return np.asarray(d["to_vector"](x), dtype=float)
    return design_vector(domain, x)


def decode_design(domain: str, z) -> dict:
    """**定长实向量 → 设计**（`encode_design` 的逆）。"""
    d = DOMAINS[domain]
    if d.get("from_vector") is not None:
        return d["from_vector"](z)
    return from_vector(domain, z)


def feasible_set(domain: str, target: float, **kw) -> dict:
    """解流形（**非退化域给一维射线**；landau 给可行集摘要）。"""
    if domain == "two_level":
        from pipeline.design_two_level import EV, design_manifold
        return design_manifold(float(target), dF_min_J=kw.get(
            "dF_min_eV", 0.15) * EV, T_min_K=kw.get("T_min_K", 250.0))
    if domain == "rayleigh":
        from pipeline.design_rayleigh import design_manifold
        return design_manifold(float(target), D_max_m=kw.get("D_max_m", 1.0),
                               f_min_Hz=kw.get("f_min_Hz", 1e4),
                               f_max_Hz=kw.get("f_max_Hz", 2e6))
    return {"note": "域 %s 无解析解流形（须采样/搜索）" % domain}
