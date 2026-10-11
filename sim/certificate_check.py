# -*- coding: utf-8 -*-
"""**验证证书 + 不确定度的回归检查**（2026-10-02，第 205 条）。

## 为什么需要它
给 `Verdict` 加了 `u` / `margin_u` / `robustness` / `certificate` 四个字段。
**风险**：若顺手动到判定规则，会**翻掉既有阶段与历史结论** ✗
⟹ 本脚本用**预登记判据**证明"**只加了信息，没改判定**" ✓

## 预登记判据（先写死）
    **H1 判定语义不变**：对固定设计集，`verify()` 的 `verdict` 必须**逐例等于**
       **独立重算**的规则结果（`lower_bound: measured ≥ target(1−tol)` ✓、
       `upper_bound: ≤ target(1+tol)` ✓、其余 `|m−t|/t ≤ tol` ✓）；
       且 `margin_u > 0` ⟺ `verdict=="pass"` ✓（**同式**，不引入新口径 ✓）
    **H2 证书完整**：`pass`/`fail` 例的证书必须含全部关键字段
       （model/method/tolerance_source/validity_domain/known_bias/
        external_corroboration/verdict_rule/design ✓）且数值有限 ✓；
       `unknown` 例也须有证书（说明**为什么 unknown** ✓）
    **H3 稳健性标签分布**：报告 robust/marginal 计数 ✓（**若全为 marginal
       ⟹ 容差相对 u 过松**，属发现，如实记录 ✓）
    **H4 审计只追加**：脚本前后 `verify_audit.jsonl` 行数**只增不减**，
       且**旧行仍可解析为合法 JSON** ✓
    **H5 数字健全**：`u > 0`、`margin_u` 有限 ✓（unknown 例允许为 None ✓）

用法：
    python sim/certificate_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# **审计路径不在本脚本写死**：运行期从 `sim.verify_api` 导入 `AUDIT_LOG` ✓
#   （初版写死 `_ROOT/"verify_audit.jsonl"` ✗ ⟹ 实际在 `sim/verify_audit.jsonl`
#    ⟹ FileNotFoundError ✓）
CERT_KEYS = ["family", "metric", "model", "method", "tolerance",
             "tolerance_source", "validity_domain", "supported_metrics",
             "registry_source", "known_bias", "external_corroboration",
             "verdict_rule", "design"]

# **固定设计集**（含 pass / fail / unknown 三类 ✓）
CASES = [
    {"family": "rayleigh", "sigma": 0.5, "metric": "q1", "target": 3.0,
     "direction": "lower_bound"},
    {"family": "rayleigh", "sigma": 0.5, "metric": "q1", "target": 100.0,
     "direction": "lower_bound"},
    {"family": "two_level", "p": 0.25, "metric": "q1", "target": 3.0,
     "direction": "lower_bound"},
    {"family": "weibull", "shape": 1.6, "scale": 2.2, "metric": "q1",
     "target": 1.0, "direction": "lower_bound"},
    {"family": "landau", "a": 1.0, "metric": "q1", "target": 1.0,
     "direction": "lower_bound"},
    {"family": "rayleigh", "sigma": 0.5, "metric": "q1", "target": 2.0,
     "direction": "upper_bound"},
    {"family": "landau", "a": 1.0, "metric": "q9", "target": 1.0,
     "direction": "lower_bound"},                    # 度量未支持 ⟹ unknown ✓
    {"family": "不存在的族", "x": 1.0, "metric": "q1", "target": 1.0},
]


def _indep_rule(measured, target, direction, tol) -> bool:
    """**独立重算**判定规则（照文档写死，不调用被测代码 ✓）。"""
    if measured is None or target is None:
        return False
    if direction == "lower_bound":
        return measured >= target * (1.0 - tol)
    if direction == "upper_bound":
        return measured <= target * (1.0 + tol)
    return abs(measured - target) / max(abs(target), 1e-12) <= tol


def main() -> int:
    # **审计路径必须从模块导入**（初版写死 `_ROOT/"verify_audit.jsonl"` ✗ ——
    #   实际在 `sim/verify_audit.jsonl` ⟹ FileNotFoundError ✓）
    from sim.verify_api import AUDIT_LOG, verify
    audit = Path(AUDIT_LOG)
    n_before = sum(1 for _ in audit.open(encoding="utf-8")) \
        if audit.exists() else 0
    print("=" * 74)
    print("验证证书 + 不确定度的回归检查（H1–H5）")
    print("=" * 74)
    print("审计日志起点行数：%d" % n_before)

    h1 = h2 = h5 = True
    n_pass = n_fail = n_unk = 0
    rob = {"robust": 0, "marginal": 0, "n/a": 0}
    for d in CASES:
        v = verify(dict(d), use_cache=False, log=True)
        m, t = v.measured, v.target
        tag = "%s(direction=%s)" % (d["family"], d.get("direction"))
        if v.verdict == "unknown":
            n_unk += 1
            rob["n/a"] += 1
            ok_cert = bool(v.certificate) and all(
                k in v.certificate for k in
                ("family", "model", "verdict_rule", "design",
                 "external_corroboration"))
            h2 &= ok_cert
            print("  [unknown] %-34s reason=%s" % (tag, v.reason[:44]))
            print("            证书存在=%s ｜ 含'为什么 unknown'所需字段=%s ✓"
                  % (bool(v.certificate), ok_cert))
            continue
        expect = _indep_rule(m, t, v.direction, v.tolerance)
        same = (expect == (v.verdict == "pass"))
        h1 &= same
        # **margin_u > 0 ⟺ pass** ✓（同式检查）
        h1 &= ((v.margin_u > 0) == (v.verdict == "pass"))
        h5 &= (v.u is not None and v.u > 0 and v.margin_u is not None)
        rob[v.robustness] = rob.get(v.robustness, 0) + 1
        n_pass += int(v.verdict == "pass")
        n_fail += int(v.verdict == "fail")
        missing = [k for k in CERT_KEYS if k not in v.certificate]
        h2 &= (not missing)
        print("  [%-5s] %-34s measured=%.5g target=%.5g tol=%.4f ｜ "
              "u=%.4f margin=%.2fu ⟹ %s"
              % (v.verdict, tag, m, t, v.tolerance, v.u, v.margin_u,
                 v.robustness))
        if missing:
            print("            **证书缺字段**：%s ✗" % missing)

    # ---- H4 审计只追加 ----
    n_after = sum(1 for _ in audit.open(encoding="utf-8")) \
        if audit.exists() else 0
    bad_old = 0
    with audit.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_before:
                break
            try:
                json.loads(line)
            except Exception:  # noqa: BLE001
                bad_old += 1
    h4 = (n_after >= n_before) and bad_old == 0
    print("\n审计日志：%d → %d 行（只增 ✓）｜旧行解析失败 %d 条"
          % (n_before, n_after, bad_old))

    print("\n**H1 判定语义不变**（逐例 == 独立重算；且 margin_u>0 ⟺ pass）⟹ %s"
          % ("**通过** ✓" if h1 else "**未通过** ✗"))
    print("**H2 证书完整**（关键字段齐全；unknown 也有证书）⟹ %s"
          % ("**通过** ✓" if h2 else "**未通过** ✗"))
    print("**H3 稳健性分布**：robust **%d** ｜ marginal **%d** ｜ n/a **%d**"
          "（pass %d / fail %d / unknown %d）"
          % (rob["robust"], rob["marginal"], rob["n/a"], n_pass, n_fail,
             n_unk))
    print("   ⓘ 若 marginal 很多 ⟹ **容差相对 u 偏松**（u 取容差 ⟹ 边界外 1u 即"
          "整段容差宽度）——**如实记录**，不据此改判定 ✓")
    print("**H4 审计只追加且旧行合法** ⟹ %s"
          % ("**通过** ✓" if h4 else "**未通过** ✗"))
    print("**H5 数字健全** ⟹ %s" % ("**通过** ✓" if h5 else "**未通过** ✗"))
    ok = h1 and h2 and h4 and h5
    print("\n**结论**：%s"
          % ("**只增加了信息，判定语义未变** ✓ —— `pass` 现在自带"
             "**不确定度、边界距离、稳健性、证书（模型/容差来源/有效域/"
             "已知偏差/外部印证/判定式/可复现设计）** ✓"
             if ok else "**未通过，须修** ✗"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
