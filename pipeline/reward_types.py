# -*- coding: utf-8 -*-
"""**奖励类型分类：一等公民**（2026-10-02，第 138 条；架构 §13.3）。

## 为什么需要（架构级的缺口）
架构原先把奖励默认成 **IB 型**（`R = β·I_TY − I_XT`）。但项目自己的结论
（第 114 条）是：**β 只适用于「信息量型」需求（109/159 = 70%）**：
  · **约束型**（势垒/保持、工况索引）⟹ 必须走**达标型奖励**（`R=1 if q≥q_req`）
    ——第 113 条实测：强行用保持曲线标定会得到 **β≈0 或 β<0 的假解锁**；
  · **速率型**（q9/q7）⟹ 按**文档化参考尺度**换算，且**标为「约定」而非定理**。

这套分类原先**散落在各脚本**（`beta_calibration._info_class`、`req_env` 的
`CONSTRAINT_REQUIREMENTS`、`extra_curves` 的类名…）⟹ 易漂移。
**本模块把它集中为单一真源**，并给出**每种类型对应的奖励形式**。

用法：
    from pipeline.reward_types import classify_pair, reward_kind
    cls, why = classify_pair(pair)          # → ("param" | "constraint_barrier" | …)
    kind = reward_kind(cls)                 # → "ib" | "satisficing" | "convention"
"""

from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# 四类（与 `beta_calibration._info_class` 的返回值一致，**不新造名字**）
REWARD_KIND: dict[str, str] = {
    "param": "ib",                 # 信息量型 → β 适用（IB 奖励）
    "constraint_barrier": "satisficing",   # 约束型（势垒/保持）→ 达标型
    "constraint_env": "satisficing",       # 约束型（工况索引）→ 达标型
    "rate_ratio": "convention",    # 速率型 → 参考尺度换算（**约定**）
}
# 未登记/无法判定 ⟹ **既不套 β 也不假装达标**，如实返回 unknown
UNKNOWN = "unknown"

CLASS_LABEL: dict[str, str] = {
    "param": "**信息量型**（关于被估参数的信息）→ β 适用（IB 奖励）",
    "constraint_barrier": "**约束型（势垒/保持）**→ 达标型奖励（β 不适用）",
    "constraint_env": "**约束型（工况索引）**→ 达标型奖励（β 不适用）",
    "rate_ratio": "**速率型**（q9/q7）→ 参考尺度换算（**约定**，非定理）",
    UNKNOWN: "**未登记** → 不套 β、不假装达标（如实 unknown）",
}

# 性质 → 类（与 `beta_calibration._info_class` 的判据一致；**改这里就够**）
PROP_TO_CLASS: dict[str, str] = {
    "stability": "constraint_barrier",
    "memory": "constraint_barrier",
    "lifetime": "constraint_barrier",
    "retention": "constraint_barrier",
    "toughness": "constraint_barrier",
    "adaptability": "constraint_env",
    "response_speed": "rate_ratio",
    "transport": "rate_ratio",
    "flow": "rate_ratio",
    "speed": "rate_ratio",
    "rate": "rate_ratio",
    "precision": "param",
    "robustness": "param",
    "stiffness": "param",
}


def reward_kind(cls: str) -> str:
    """类 → 奖励形式：`ib`（β 适用）/ `satisficing`（达标型）/ `convention`。"""
    return REWARD_KIND.get(cls, UNKNOWN)


def classify_property(prop: str, target_value: dict | None = None
                      ) -> tuple[str, str]:
    """由**性质名 + 目标量**判定类（不依赖 DB3 条目结构）。

    **口径单一真源**：`beta_calibration._info_class` 现委托本函数（见其实现），
    避免两处判据漂移。
    """
    tv = target_value or {}
    if "E_req_nats" in tv:
        return "constraint_barrier", "目标量含 `E_req_nats`（势垒/寿命单位）"
    if "n_environments_count" in tv:
        return "constraint_env", "目标量含 `n_environments_count`（工况数）"
    cls = PROP_TO_CLASS.get(prop or "")
    if cls is None:
        return UNKNOWN, "性质 `%s` 未登记于奖励类型表" % (prop or "空")
    return cls, "性质 `%s` → %s" % (prop, CLASS_LABEL[cls].split("→")[0].strip())


def classify_pair(pair: dict) -> tuple[str, str]:
    """由 DB3 条目判定类（与既有 `_info_class` **同源**）。

    先尝试复用 `beta_calibration._info_class`（它含更细的理由文本）；
    失败则回退到 `classify_property`。
    """
    try:
        from pipeline.beta_calibration import _info_class
        return _info_class(pair)
    except Exception:  # noqa: BLE001
        rs = pair.get("requirement_structured", {}) or {}
        return classify_property(rs.get("property_id", ""),
                                 rs.get("target_value") or {})


def main() -> int:
    import json
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print("=" * 74)
    print("奖励类型分类（一等公民，架构 §13.3）")
    print("=" * 74)
    print("\n类 → 奖励形式：")
    for cls, kind in REWARD_KIND.items():
        print("  %-20s → %-12s（%s）" % (cls, kind, CLASS_LABEL[cls]))

    pairs = sorted((_ROOT / "db3" / "pairs").glob("DB3-*.json"))
    print("\nDB3 全量分类（%d 条）：" % len(pairs))
    from collections import Counter
    cnt: Counter = Counter()
    kinds: Counter = Counter()
    for p in pairs:
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        cls, _why = classify_pair(e)
        cnt[cls] += 1
        kinds[reward_kind(cls)] += 1
    for cls, n in cnt.most_common():
        print("  %-22s %4d（%.0f%%）→ 奖励形式 **%s**"
              % (cls, n, 100 * n / max(sum(cnt.values()), 1), reward_kind(cls)))
    print("\n奖励形式汇总：%s" % dict(kinds))
    n_ib = kinds.get("ib", 0)
    print("⟹ **信息量型（β 由 IB 语义）比例 %.0f%%**（%d/%d）"
          % (100 * n_ib / max(sum(kinds.values()), 1), n_ib,
             sum(kinds.values())))
    print("   **口径修正（第 138 条）**：此前「β 适用 109/159 = 70%」"
          "**混同了两类**——109 = **信息量型 57** + **速率型约定 52**；")
    print("   真正的「信息量型」只占 **%.0f%%**，另 **%.0f%%** 是**约定**"
          "（非定理）、**%.0f%%** 是约束型（不用 β）。"
          % (100 * kinds.get("ib", 0) / max(sum(kinds.values()), 1),
             100 * kinds.get("convention", 0) / max(sum(kinds.values()), 1),
             100 * kinds.get("satisficing", 0) / max(sum(kinds.values()), 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
