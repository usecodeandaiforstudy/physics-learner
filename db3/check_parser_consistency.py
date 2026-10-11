# -*- coding: utf-8 -*-
"""需求对 ↔ 解析器一致性检查（2026-10-02）。

动机：DB3-0035 初稿把「σ ≤ 5 K」误标为 bound_below，靠人工偶然发现——
本工具把这类错误**自动抓出来**，并分两层报告：
  - **错误（errors）**：属性未命中、quantity_bound 与解析不符、direction 与 q 向量
    内部矛盾
    ——这些是 B v0 或需求对自身的问题，需修；
  - **差距（gaps）**：数值目标的键/值未被解析器复现——**不是 bug**：人工锚点
    用规则语义键（T_c_K/E_req_nats/kappa_m2_per_s/n_valleys_count…），B v0
    解析器的 UNIT_MAP 键空间远窄于此。这是 **B v1 要学的集成缺口**，量化记录
    （口径统一属 B v1 工作；量纲等价键经别名表归一后比较）。
**语义分歧（semantics）**：`direction` 字段在已复核锚点里指**信息度量的界方向**
    （q 有下界 → bound_below），而解析器输出的是**物理量的界方向**
    （「不超过」→ bound_above）——新旧对未标 `quantity_bound` 时的差异属口径
    分歧，非数据错误（详见 HANDOVER 第 70 条：B v1 需先统一此 schema 歧义）。

用法（项目根目录）：python db3/check_parser_consistency.py
退出码：0 = 无错误（差距与警告不影响退出码）；1 = 存在错误
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent

# 键名别名表：**改用规范键注册表**（单一事实来源，2026-10-02 口径统一）
try:
    from b_network.key_registry import canon as _canon, is_canonical
except Exception:  # noqa: BLE001
    def _canon(k: str) -> str:      # 兜底：无注册表时原样返回
        return k

    def is_canonical(k: str) -> bool:
        return True


def _has_both_dirs(text: str) -> bool:
    """文本同时含上下界关键词（多约束需求）——direction 单值口径的固有歧义。"""
    above = ["不超过", "以内", "<=", "≤", "至多", "压回", "回稳到",
             "within", "at most", "up to", "no more than"]
    below = ["不低于", "不小于", "至少", ">=", "≥",
             "at least", "no less than", "more than"]
    return (any(k in text for k in above) and any(k in text for k in below))


def main() -> int:
    from b_network.requirement_parser import RequirementParser

    parser = RequirementParser()
    n_pairs = 0
    n_combo = 0
    errors, gaps, warnings, semantics = [], [], [], []
    for p in sorted((ROOT / "db3" / "pairs").glob("DB3-*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        n_pairs += 1
        pid = rec["pair_id"]
        declared = rec["requirement_structured"]
        parsed = parser.parse(rec["requirement_text"])
        text = rec["requirement_text"]

        # 0) 故意未映射对（R-NOMAP-000）：属性应为空且 unmapped=True
        if declared["property_id"] == "unmapped":
            if not parsed.unmapped:
                errors.append(f"{pid}: 声明 unmapped 但解析器给出属性 "
                              f"{parsed.property_ids}")
            continue

        # 1) 属性命中（组合对 property=None → 跳过）
        if declared["property_id"] is not None and \
                declared["property_id"] not in parsed.property_ids:
            errors.append(
                f"{pid}: property {declared['property_id']!r} 未被解析器命中"
                f"（命中 {parsed.property_ids}；文本={text[:40]}…）")
        # 2) 方向：**两种语义分开校验**（2026-10-02 语义碰撞发现）
        #    - pair.direction = **信息度量 q 的界方向**（34 条已复核锚点的约定：
        #      q 有下界 → bound_below；与 q 向量的 lower_bound 对应）
        #    - parser.direction = **物理量的界方向**（「不超过」→ bound_above）
        #    二者名字相同、语义不同：历史对的差异属口径分歧而非数据错误。
        q_dir = next(iter(rec.get("info_metric_vector", {}).values()),
                     {}).get("direction")
        want_metric = {"lower_bound": "bound_below",
                       "upper_bound": "bound_above"}.get(q_dir)
        if declared["direction"] is None:
            n_combo += 1          # 组合对：无单一方向（故意留空）→ 跳过
        elif want_metric and declared["direction"] != want_metric:
            errors.append(f"{pid}: direction({declared['direction']}) 与 q 向量"
                          f" {q_dir} 不一致（度量方向内部矛盾）")

        qb = declared.get("quantity_bound")
        if qb is not None:
            want_phys = {"upper": "bound_above",
                         "lower": "bound_below"}[qb]
            if parsed.direction != want_phys:
                # 区分「模板反向」（文本界落在派生量上，目标量是另一量——
                # 如 DB3-0001 变形≤ ⟹ 刚度≥）与解析器能力不足
                tk = set(declared.get("target_value") or {})
                lower_class = {"stiffness_N_per_m", "modulus_Pa",
                               "equivalent_modulus_Pa", "kappa_m2_per_s",
                               "E_req_nats", "delta_dimensionless",
                               "n_environments_count", "n_valleys_count"}
                if qb == "lower" and (tk & lower_class) \
                        and parsed.direction == "bound_above":
                    semantics.append(
                        f"{pid}: 文本上界⟹目标量下界（模板反向，锚点正确；"
                        f"解析器读到文本表面）")
                else:
                    errors.append(f"{pid}: quantity_bound={qb} 但解析器给 "
                                  f"{parsed.direction}")
        elif parsed.direction != declared["direction"]:
            # 旧对未标 quantity_bound → 语义分歧，记录不计错误
            semantics.append(f"{pid}: pair.direction={declared['direction']}"
                             f"（度量向）vs 解析 {parsed.direction}（物理向）")
        # 3) 数值/字符串目标复现情况（未复现或不等 → 差距，不是错误）
        parsed_canon = {_canon(k): v for k, v in parsed.target_value.items()}
        for key, val in declared["target_value"].items():
            got = parsed_canon.get(_canon(key))
            if got is None:
                gaps.append(f"{pid}: {key}={val}（解析得 "
                            f"{dict(parsed.target_value)}）")
            elif isinstance(val, str) or isinstance(got, str):
                if str(got) != str(val):      # 字符串键（如 settling_band）精确比对
                    gaps.append(f"{pid}: {key} 声明 {val!r} vs 解析 {got!r}")
            elif abs(float(got) - float(val)) > 1e-9:
                errors.append(f"{pid}: target {key} 声明 {val} vs 解析 {got}")

    print(f"检查 {n_pairs} 对需求（其中组合对 {n_combo}）：错误 {len(errors)} 项、"
          f"B v0↔锚点键差距 {len(gaps)} 项、"
          f"direction 语义分歧 {len(semantics)} 项、口径警告 {len(warnings)} 项")
    for it in errors:
        print("  [错误]", it)
    for w in warnings:
        print("  [警告]", w)
    if semantics:
        print(f"  ---- direction 语义分歧（锚点=度量方向 / 解析器=物理方向；"
              f"旧对未标 quantity_bound，共 {len(semantics)} 项）----")
        for s in semantics[:5]:
            print("  [语义]", s)
        if len(semantics) > 5:
            print(f"  … 其余 {len(semantics) - 5} 条同类")
    print("  ---- 键差距（B v1 待学的集成缺口；前 12 条）----")
    for g in gaps[:12]:
        print("  [差距]", g)
    if len(gaps) > 12:
        print(f"  … 其余 {len(gaps) - 12} 条同类")
    print(f"  ---- 汇总：错误 {len(errors)}（须为 0）／键差距 {len(gaps)}"
          f"（口径统一后残差，B v1 待学）／语义 {len(semantics)} ----")
    print("===== 一致性检查通过（无错误）=====" if not errors
          else "===== 存在错误（见上）=====")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
