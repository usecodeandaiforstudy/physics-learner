# -*- coding: utf-8 -*-
"""规范键注册表（B v1 口径统一，2026-10-02）——单一事实来源。

背景（HANDOVER 第 70 条立项②）：人工锚点的 `target_value` 用规则语义键
（T_c_K / E_req_nats / kappa_m2_per_s / n_environments_count / settling_band…），
而 B v0 解析器的 UNIT_MAP 只产出一小撮自造键（tau_s / length_scale_m /
n_valleys_count…）——**27 项键差距**即此。本模块把锚点实际使用的键固化为
**规范键表**（按属性分组），并提供解析器键 → 规范键的别名映射。

使用方：
  - `b_network/requirement_parser.py`：按（单位, 属性）映射直出规范键；
  - `db3/check_parser_consistency.py`：以别名表比较解析输出与锚点声明。

说明：规范键来自 39 条已复核锚点的实际取值（2026-10-02 提取），
新增规则键时须同步此处（作为 B v1 的口径契约）。
"""

from __future__ import annotations

__all__ = ["CANONICAL_KEYS", "PARSER_ALIAS", "canon", "is_canonical"]

# 规范键（按属性分组）—— 取自 39 条已复核锚点的 target_value 实际键
CANONICAL_KEYS: dict[str, list[str]] = {
    "stiffness": ["stiffness_N_per_m", "modulus_Pa", "equivalent_modulus_Pa"],
    "precision": ["sigma_K", "sigma_m", "sigma_rad", "sigma_pH",
                  "sigma_dimensionless_fraction", "n_samples_count"],
    "response_speed": ["settling_time_s", "settling_band", "tau_s"],
    "transport": ["kappa_m2_per_s", "length_scale_m", "ell_req_m",
                  "front_time_s", "t_obs_s"],
    "robustness": ["sigma_K", "sigma_pH", "epsilon"],
    "stability": ["delta_dimensionless", "T_c_K", "T_work_K", "E_req_nats"],
    "adaptability": ["n_environments_count", "n_valleys_count"],
    "toughness": ["E_req_nats"],
    "unmapped": [],
}

# 解析器键 → 规范键（等价量不同命名；历史兼容）
# 注：n_valleys_count 与 n_environments_count 均为规范键（能量谷 vs 工况/环境），
# 不互为别名（2026-10-02 修正）。
PARSER_ALIAS: dict[str, str] = {
    "tau_s": "settling_time_s",          # 一阶系统时间常数 = 建立时间口径
    "sigma_dimensionless": "sigma_dimensionless_fraction",
    "delta_dimensionless_fraction": "delta_dimensionless",
    "epsilon_K_inv2": "epsilon",
    # 锚点内同名量的两种写法（0003 modulus_Pa / 0004 equivalent_modulus_Pa）
    "equivalent_modulus_Pa": "modulus_Pa",
}


def canon(key: str) -> str:
    """解析器键 → 规范键（无别名则原样返回）。"""
    return PARSER_ALIAS.get(key, key)


def is_canonical(key: str) -> bool:
    """该键是否出现在任一属性的规范键表中。"""
    return any(key in keys for keys in CANONICAL_KEYS.values())
