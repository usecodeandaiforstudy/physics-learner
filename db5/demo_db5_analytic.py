# -*- coding: utf-8 -*-
"""DB5 解析级验证示例：解码设计 → 解析判据 → pass/fail → 验证记录。

演示闭环最后一段（arch §3.6 / §3.9 DB5）：
  需求（自然单位）→ B v0 解析/合成 q → 从 DB2 检索候选系统 →
  解码设计参数 → 解析验证（桥接定理直接算）→ 写 db5/validations.jsonl

场景：弹簧刚度需求（k_B T = 1 自然单位，故桥接退化为 q1 = K_req/(k_B T)）。
弹性族解析指纹（DB1 elastic 条目 fingerprint_expectation）：
  fisher_trace = 3k/kT（三参数 (λ, x0, kT) 全 trace）；应变方向 g_λλ = k/kT。
验证判据（±10%，arch §3.9 DB5 材料判据）：q1_actual 相对 q1_req 的裕度 ≥ −10%。
回流规则（arch §3.6）：pass 的条目才有资格（经人工确认）以 rl_validated 身份回写 DB3；
本演示只写验证记录，不改 DB3 pending 状态。
"""

from __future__ import annotations

import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from b_network.composer import Composer
from b_network.requirement_parser import RequirementParser
from b_network.rule_matcher import RuleMatcher

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB2_PATH = os.path.join(BASE, "db2", "db2.jsonl")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validations.jsonl")
TOLERANCE = -0.10  # 裕度下限：实际值允许比需求低 10%


def find_elastic_record(system_id: str) -> dict:
    with open(DB2_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["system_id"] == system_id and rec["family"] == "elastic":
                return rec
    raise KeyError(system_id)


def validate_design(system_id: str, k: float, kT: float, q1_req: float) -> dict:
    """解析验证：应变方向 g_λλ = k/kT（涨落-响应定理，DB1 elastic 解析指纹）。"""
    rec = find_elastic_record(system_id)
    q1_actual = k / kT
    margin = (q1_actual - q1_req) / q1_req
    passed = margin >= TOLERANCE
    return {
        "validation_id": "VAL-DB5-%s-%04g" % (system_id.split("_")[-1], k * 10),
        "structure_id": system_id,                     # FK → DB2 记录（演示级溯源）
        "domain": "elasticity",
        "requirement_natural_units": "q1 = K_req/(k_B T) >= %.4g（k_B T = %g 自然单位）" % (q1_req, kT),
        "decoded_design": {"family": "elastic", "params": {"k": k, "kT": kT},
                           "scan_param_of_record": rec["scan"]["scan_param"]},
        "validation_method": "analytic",
        "metrics": {
            "q1_required": q1_req,
            "q1_actual_g_lambda_lambda": q1_actual,
            "bridge": "g_λλ = k/kT（涨落-响应定理；DB1 elastic 解析指纹）",
            "margin": round(margin, 4),
            "tolerance": TOLERANCE,
        },
        "pass_fail": "pass" if passed else "fail",
        "date": datetime.date.today().isoformat(),
        "validator": "demo_db5_analytic.py（解析级，机器可复核）",
        "reflow_note": "pass 条目经人工确认后可以 rl_validated 身份回写 DB3（本演示不改 DB3 状态）",
    }


def main() -> int:
    parser = RequirementParser()
    matcher = RuleMatcher()
    composer = Composer(matcher)

    text = "弹簧要足够硬：刚度不低于 5 N/m（k_B T = 1 自然单位演示）"
    req = parser.parse(text)
    spec = composer.compose(req)
    print("需求：%s" % text)
    print("解析→规则：%s｜q1 数值 ≈ %s" % (
        spec.rule_ids, spec.q.get("q1_fisher_directional", {}).get("numeric", "符号")))

    q1_req = spec.q["q1_fisher_directional"].get("numeric")
    if q1_req is None:  # 解析器未抽到数值时的兜底（本例文本给了 5 N/m）
        q1_req = float(req.target_value["stiffness_N_per_m"])

    # 两个候选设计（DB2 elastic 扫描网格 k ∈ [0.1, 10] 上的点）
    results = [
        validate_design("elastic_stiff_spring", k=8.0, kT=1.0, q1_req=q1_req),
        validate_design("elastic_soft_spring", k=0.3, kT=1.0, q1_req=q1_req),
    ]
    print("\n===== 解析验证（±10% 判据）=====")
    for r in results:
        print("  %s：k=%-4g kT=1 → q1=%-5.3g（需求 ≥%.3g，裕度 %+.1f%%）→ %s"
              % (r["structure_id"], r["decoded_design"]["params"]["k"],
                 r["metrics"]["q1_actual_g_lambda_lambda"], q1_req,
                 100 * r["metrics"]["margin"], r["pass_fail"].upper()))

    mode = "a" if os.path.exists(OUT_PATH) else "w"
    with open(OUT_PATH, mode, encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n验证记录已追加到 %s（pass=%d / fail=%d）"
          % (OUT_PATH, sum(r["pass_fail"] == "pass" for r in results),
             sum(r["pass_fail"] == "fail" for r in results)))
    print("回流规则：只有 pass 条目经人工确认后才以 rl_validated 身份回写 DB3。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
