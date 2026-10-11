"""DB1 入库校验脚本：三道机器门。

对 AI 生成的 DB1 条目（JSON 区块 1 + 自证表 YAML 区块 2）执行入库前的机器自动门，
见 `db1/review_workflow.md` 第二节：

    机器门 1：JSON Schema 校验（手写，对齐 db1.schema.json）
    机器门 2：约定一致性检查（对照 conventions.yaml + 自证表交叉比对）
    机器门 3：解析表达式可计算性（sympy 解析 + 符号检查 + 求值）

用法：
    python validate_entry.py <entry.json> [attestation.yaml]

依赖：numpy、scipy、sympy、pyyaml（无 jsonschema 依赖，schema 校验手写）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import sympy
import yaml

# 本文件所在目录（db1/），用于定位 conventions.yaml
_HERE = Path(__file__).resolve().parent
_CONVENTIONS_PATH = _HERE / "conventions.yaml"


# ---------------------------------------------------------------------------
# 常量：与 db1.schema.json / conventions.yaml 对齐
# ---------------------------------------------------------------------------
_REQUIRED_FIELDS = [
    "system_id", "domain", "name", "family", "equations",
    "parameters", "provenance", "strictness_level", "convention_version",
]

_ALLOWED_TOP_LEVEL = set(_REQUIRED_FIELDS) | {
    "lagrangian", "hamiltonian", "boundary_conditions", "channel_template",
    "analytic_invariants", "fingerprint_expectation", "simulator_ref",
    "source_ref", "license", "scan_param",
}

_ALLOWED_PROVENANCE_KEYS = {
    "source", "generator_model", "verified_by", "review_date",
    "human_review_status", "reviewers",
}

_HUMAN_REVIEW_STATUS = {"pending", "spot_checked", "fully_reviewed"}

_DOMAIN_ENUM = {
    "stochastic_process", "diffusion", "elasticity", "electromagnetism",
    "plasma", "thermodynamics", "quantum", "mechanics", "network",
    # 扩族批次 1（2026-09-26）新增：weibull/beta/vonmises 的失效统计与
    # 材料占据场景属材料科学，gamma chi2 属信号处理（schema 标注可扩展）
    "materials", "signal",
    # 领域实例扩域（2026-10-02）：地球物理、生物医学、能源、农业、声学（×N 新增流程）
    "geophysics", "biomedical", "energy", "agriculture", "acoustics",
    # 大规模表驱动扩域（2026-10-02）：12 领域
    "aerospace", "automotive", "civil", "chemical", "food", "textile",
    "optics", "robotics", "marine", "mining", "semiconductor", "sports",
    # 批次 2（2026-10-02）：13 领域
    "nuclear", "meteorology", "hydrology", "forestry", "aquaculture",
    "pharmaceutical", "packaging", "coating", "railway", "traffic",
    "water_utility", "geotechnical", "metrology",
    # 批次 3（2026-10-02）：13 领域
    "astro", "particle", "cryogenics", "vacuum", "printing", "glass",
    "cement", "pulp_paper", "hvac", "elevator", "fire_safety",
    "desalination", "smart_grid",
}

_STRICTNESS_ENUM = {"定理", "推导", "类比", "猜想"}
_SOURCE_ENUM = {"textbook", "literature", "AI+verified"}
_QUANTITY_ENUM = {"fisher_trace", "fisher_det", "mi_profile", "ib_curve", "action_s"}

# 命名规则 {family}__{variant}：family 与 variant 均可含单下划线，
# 段间以单/多下划线分隔；仅允许小写字母、数字、下划线。
_SYSTEM_ID_PATTERN = re.compile(r"^[a-z0-9]+(_+[a-z0-9]+)+$")
_CHANNEL_TEMPLATE_PATTERN = re.compile(r"^CT-[A-Z]+-\d{3}$")

_PARAM_KEYS = {"name", "symbol", "range"}
_PARAM_ALLOWED = _PARAM_KEYS | {"default", "unit", "is_bridge_constant"}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Issues:
    """收集校验问题（按严重度分组）。"""

    def __init__(self) -> None:
        self.errors: list[str] = []     # 致命（门不通过）
        self.warnings: list[str] = []   # 警告（不阻断）

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def ok(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------------------
# 机器门 1：schema 校验（手写）
# ---------------------------------------------------------------------------
def gate1_schema(entry: dict, issues: Issues) -> None:
    """手写 JSON Schema 校验（对齐 db1.schema.json，additionalProperties=false）。"""
    if not isinstance(entry, dict):
        issues.error("条目必须是 JSON 对象")
        return

    # 1. 必填字段
    for f in _REQUIRED_FIELDS:
        if f not in entry:
            issues.error(f"缺少必填字段: {f}")

    # 2. 禁止额外字段（additionalProperties: false）
    for key in entry:
        if key not in _ALLOWED_TOP_LEVEL:
            issues.error(f"不允许的字段: {key}")

    # 3. system_id 命名规则
    sid = entry.get("system_id", "")
    if not isinstance(sid, str) or not _SYSTEM_ID_PATTERN.match(sid):
        issues.error(f"system_id 须匹配 {{family}}__{{variant}}（小写+数字+双下划线），收到: {sid!r}")

    # 4. domain 枚举
    if entry.get("domain") not in _DOMAIN_ENUM:
        issues.error(f"domain 非法枚举值: {entry.get('domain')!r}")

    # 5. equations 结构
    eq = entry.get("equations")
    if isinstance(eq, dict):
        if "latex" not in eq or "sympy" not in eq:
            issues.error("equations 须含 latex 与 sympy 两字段")
    else:
        issues.error("equations 须为对象")

    # 6. parameters 结构
    params = entry.get("parameters")
    if isinstance(params, list) and params:
        for i, p in enumerate(params):
            if not isinstance(p, dict):
                issues.error(f"parameters[{i}] 须为对象")
                continue
            for k in _PARAM_KEYS:
                if k not in p:
                    issues.error(f"parameters[{i}] 缺少字段: {k}")
            for k in p:
                if k not in _PARAM_ALLOWED:
                    issues.error(f"parameters[{i}] 不允许的字段: {k}")
            rng = p.get("range")
            if not (isinstance(rng, list) and len(rng) == 2
                    and all(isinstance(v, (int, float)) for v in rng)):
                issues.error(f"parameters[{i}].range 须为 [min, max] 数值区间")
    elif not isinstance(params, list) or not params:
        issues.error("parameters 须为非空数组")

    # 7. strictness_level 枚举
    if entry.get("strictness_level") not in _STRICTNESS_ENUM:
        issues.error(f"strictness_level 非法枚举值: {entry.get('strictness_level')!r}")

    # 8. provenance 结构
    prov = entry.get("provenance")
    if isinstance(prov, dict):
        if "source" not in prov or "verified_by" not in prov or "review_date" not in prov:
            issues.error("provenance 须含 source / verified_by / review_date")
        if prov.get("source") not in _SOURCE_ENUM:
            issues.error(f"provenance.source 非法枚举值: {prov.get('source')!r}")
        # 8.1 禁止额外字段
        for k in prov:
            if k not in _ALLOWED_PROVENANCE_KEYS:
                issues.error(f"provenance 不允许的字段: {k}")
        # 8.2 review_date 允许 null（未复核），非 null 须为日期串
        rd = prov.get("review_date")
        if rd is not None and not (isinstance(rd, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", rd)):
            issues.error(f"provenance.review_date 须为 YYYY-MM-DD 或 null，收到: {rd!r}")
        # 8.3 human_review_status 枚举
        hrs = prov.get("human_review_status")
        if hrs is not None and hrs not in _HUMAN_REVIEW_STATUS:
            issues.error(f"provenance.human_review_status 非法枚举值: {hrs!r}")
        # 8.4 反虚假声明：verified_by 不得声称做过人工复核
        #     （人工复核状态由 human_review_status 单独跟踪）
        vb = str(prov.get("verified_by", ""))
        if "human_review" in vb.lower() and prov.get("human_review_status") in (None, "pending"):
            issues.error(
                "provenance.verified_by 声称 human_review，但 human_review_status 仍为 pending"
                "（禁止声明未做过的人工复核）"
            )
    else:
        issues.error("provenance 须为对象")

    # 9. channel_template pattern（若提供）
    ct = entry.get("channel_template")
    if ct is not None and (not isinstance(ct, str) or not _CHANNEL_TEMPLATE_PATTERN.match(ct)):
        issues.error(f"channel_template 须匹配 CT-XXX-NNN，收到: {ct!r}")

    # 10. scan_param 须是已声明的参数名（或 null）
    sp = entry.get("scan_param", None)
    if sp is not None:
        if not isinstance(sp, str):
            issues.error(f"scan_param 须为 string 或 null，收到: {sp!r}")
        else:
            param_names = {p.get("name") for p in entry.get("parameters", [])}
            if sp not in param_names:
                issues.error(f"scan_param={sp!r} 不在 parameters 的 name 集合中: {sorted(param_names)}")


# ---------------------------------------------------------------------------
# 机器门 2：约定一致性检查
# ---------------------------------------------------------------------------
def gate2_conventions(entry: dict, attestation: dict | None, issues: Issues) -> None:
    """约定一致性：对照 conventions.yaml + 自证表交叉比对。"""
    prov = entry.get("provenance", {})
    source = prov.get("source")

    # 2.1 AI 生成物不可作 DB3 锚点（自证表 anchor_formulas 必须为空）
    if source == "AI+verified":
        if attestation is None:
            issues.error("source=AI+verified 时须提供自证表（区块 2）")
        else:
            att = attestation.get("convention_attestation", attestation)
            anchors = att.get("anchor_formulas")
            if anchors not in (None, [], ""):
                issues.error(
                    f"AI 生成物 anchor_formulas 必须为空，收到: {anchors!r}（不可作 DB3 锚点）"
                )

    # 2.2 convention_version 一致性（自证表 vs 条目）
    if attestation is not None:
        att = attestation.get("convention_attestation", attestation)
        att_ver = att.get("convention_version")
        entry_ver = entry.get("convention_version")
        if att_ver and entry_ver and att_ver != entry_ver:
            issues.error(
                f"自证表 convention_version({att_ver}) 与条目({entry_ver}) 不一致"
            )

    # 2.3 桥接常数：自证表 bridge_constants 与 parameters.is_bridge_constant 对齐
    params = entry.get("parameters", [])
    bridge_in_params = [p.get("name") for p in params if p.get("is_bridge_constant")]
    if attestation is not None:
        att = attestation.get("convention_attestation", attestation)
        att_bridge = att.get("bridge_constants") or []
        if set(att_bridge) != set(bridge_in_params):
            issues.error(
                f"自证表 bridge_constants({att_bridge}) 与 parameters 中 is_bridge_constant 项"
                f"({bridge_in_params}) 不一致"
            )

    # 2.4 桥接常数必须有 unit
    for p in params:
        if p.get("is_bridge_constant") and not p.get("unit"):
            issues.error(f"桥接常数 {p.get('name')} 必须填 unit")

    # 2.5 kT 约定：热浴默认 k_B T = 1，偏离须显式作参数
    if attestation is not None:
        att = attestation.get("convention_attestation", attestation)
        kt = att.get("thermal_bath_kT")
        if kt is not None and abs(float(kt) - 1.0) > 1e-9:
            has_kt_param = any(
                p.get("name") in ("kT", "k_B_T", "kT_bridge") for p in params
            )
            if not has_kt_param:
                issues.error(
                    f"自证表 thermal_bath_kT={kt} 偏离默认 1.0，但 parameters 中未显式声明 kT"
                )

    # 2.6 sigma 参数化：禁止 sigma_squared 当参数（最高频 AI 幻觉点）
    for p in params:
        name = str(p.get("name", "")).lower()
        if "sigma_squared" in name or "variance" in name:
            issues.error(
                f"参数 {p.get('name')} 疑似用方差当参数；高斯参数化须用 sigma（标准差），"
                f"方差写作 sigma^2"
            )

    # 2.7 provenance 完整性（AI 生成须有 generator_model）
    if source == "AI+verified" and not prov.get("generator_model"):
        issues.warn("AI 生成物建议填 provenance.generator_model（模型+版本）")


# ---------------------------------------------------------------------------
# 机器门 3：解析表达式可计算性
# ---------------------------------------------------------------------------
def gate3_analytic(entry: dict, issues: Issues) -> None:
    """解析对照的可计算性检查：analytic 表达式可 sympy 解析 + 符号可解析 + 可求值。

    注意：真正的"数值 vs 解析"对比需要 B 网络 SystemEncoder 先把物理参数
    编码为分布参数，此处只做表达式层面的可计算性检查（Phase 0 边界）。
    """
    fpe = entry.get("fingerprint_expectation")
    if not fpe:
        issues.warn("未提供 fingerprint_expectation，跳过解析对照")
        return

    # 参数符号表：name -> default 值
    param_symbols: dict[str, float] = {}
    for p in entry.get("parameters", []):
        name = p.get("name")
        default = p.get("default")
        if name and default is not None:
            param_symbols[str(name)] = float(default)

    if not isinstance(fpe, list):
        issues.error("fingerprint_expectation 须为数组")
        return

    for i, item in enumerate(fpe):
        if not isinstance(item, dict):
            issues.error(f"fingerprint_expectation[{i}] 须为对象")
            continue
        qty = item.get("quantity")
        if qty not in _QUANTITY_ENUM:
            issues.error(f"fingerprint_expectation[{i}].quantity 非法枚举值: {qty!r}")
            continue

        analytic = item.get("analytic")
        if not isinstance(analytic, str) or not analytic.strip():
            issues.error(f"fingerprint_expectation[{i}].analytic 须为非空字符串")
            continue

        # sympy 解析（locals 强制参数名为 Symbol：'beta' 会撞 sympy 的不完全
        # beta 函数、'shape' 撞同名函数，导致解析失败或 subs 报错）
        try:
            expr = sympy.sympify(
                analytic, locals={k: sympy.Symbol(k) for k in param_symbols}
            )
        except Exception as e:  # noqa: BLE001
            issues.error(f"fingerprint_expectation[{i}].analytic 无法解析: {analytic!r} ({e})")
            continue

        # 自由符号须都能在 parameters 里解析
        free_syms = expr.free_symbols
        unknown = {str(s) for s in free_syms} - set(param_symbols)
        if unknown:
            issues.error(
                f"fingerprint_expectation[{i}].analytic 引用了未声明的符号: {sorted(unknown)}"
            )
            continue

        # 代入 default 求值，须得到有限实数
        try:
            val = float(expr.subs({sympy.Symbol(k): v for k, v in param_symbols.items()}))
        except Exception as e:  # noqa: BLE001
            issues.error(f"fingerprint_expectation[{i}].analytic 代入 default 求值失败: {e}")
            continue
        if not (val == val and val not in (float("inf"), float("-inf"))):  # NaN / inf
            issues.error(f"fingerprint_expectation[{i}].analytic 求值得到非有限值: {val}")


# ---------------------------------------------------------------------------
# 汇总入口
# ---------------------------------------------------------------------------
def validate_entry(entry: dict, attestation: dict | None = None) -> dict:
    """执行三道机器门，返回结构化报告。

    参数
    ----
    entry : dict
        DB1 条目（区块 1 JSON）。
    attestation : dict, 可选
        约定自证表（区块 2 YAML）。

    返回
    ----
    dict
        {"passed": bool, "errors": [...], "warnings": [...]}
    """
    issues = Issues()
    gate1_schema(entry, issues)
    gate2_conventions(entry, attestation, issues)
    gate3_analytic(entry, issues)
    return {
        "passed": issues.ok(),
        "errors": issues.errors,
        "warnings": issues.warnings,
    }


def _print_report(report: dict) -> None:
    if report["passed"]:
        print("✓ 全部机器门通过")
    else:
        print("✗ 机器门未通过，错误如下：")
    for e in report["errors"]:
        print(f"  [ERROR]   {e}")
    for w in report["warnings"]:
        print(f"  [WARN]    {w}")
    if report["passed"] and report["warnings"]:
        print(f"  （{len(report['warnings'])} 条警告，不阻断）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DB1 入库校验（三道机器门）")
    parser.add_argument("entry", help="条目 JSON 文件路径")
    parser.add_argument("attestation", nargs="?", default=None, help="自证表 YAML 文件路径（可选）")
    args = parser.parse_args(argv)

    entry = _load_json(Path(args.entry))
    attestation = _load_yaml(Path(args.attestation)) if args.attestation else None
    report = validate_entry(entry, attestation)
    _print_report(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
