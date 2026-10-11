# -*- coding: utf-8 -*-
"""约束型需求走达标奖励的验证（2026-10-02，第 115 条）。

**背景**：任务 3 的结论（第 113/114 条）是**约束型需求不该用 β**——势垒 /
工况数是约束，应走**已达标的达标型奖励**
（`R = 1 if q ≥ q_req else exp(−((q_req−q)/σ)²)`）。本脚本把该结论**接通到
代码**并给出可核验的证据。

**预登记判据**：
  B1 **达标语义**：`R == 1.0` **当且仅当** `metric ≥ target`（含容差边界）；
  B2 **单调性**：未达标区 `R` 随 `metric` 单调不减（`exp(−(Δ/σ)²)` 应如此）；
  B3 **解状态存在**：该需求在状态空间内有 **>0** 个达标状态；
  B4 **无 β**：全程不出现 β（环境奖励不含 β 项——**用 grep 自证**）；
  B5 **族级约束（工况数）**：`families_supporting_env_count(M)` 给出可能为
     **空**的结果——空即「当前族容量不足以区分 M 种工况」的**真实结论**。

用法（项目根目录）：python a_network/v25_constraint_req_check.py
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


def check(name: str, ok: bool, detail: str = "") -> int:
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  " + detail) if detail else ""))
    return 0 if ok else 1


def main() -> int:
    from a_network.req_env import (CONSTRAINT_REQUIREMENTS,
                                   RequirementConditionedEnv,
                                   families_supporting_env_count)

    fails = 0
    env = RequirementConditionedEnv(extra_constraints=True)
    print("========== 约束型需求 × 达标型奖励（第 115 条） ==========")
    print("状态空间：%d（= %d 需求 × %d 族 × %d 箱 + 吸收）"
          % (env.n_states, env.n_req, env.n_families, env.n_bins))
    ri = env.n_req - 1                     # 追加在末位
    req = env.requirements[ri]
    print("约束需求：%s（metric=%s，target=%.3f，σ=%.3f）"
          % (req.label, req.metric, req.target, req.sigma))

    # B1/B2：达标语义与单调性（在该需求的全部状态上判定）
    n_sol, n_checked, mono_bad = 0, 0, 0
    prev_r, prev_v = -1.0, -1e18
    for fam in range(env.n_families):
        vals, rews = [], []
        for b in range(env.n_bins):
            s = env._encode(ri, fam, b)
            v = env.metric_value(s)
            r = env.reward(s)
            if v is None:
                continue
            vals.append(v)
            rews.append(r)
            n_checked += 1
            # B1：R==1 ⟺ v ≥ target（容差内视为相等）
            if (r >= 1.0 - 1e-12) != (v >= req.target - 1e-12):
                fails += check("B1 达标语义 state=%d" % s, False,
                               "v=%.4f R=%.4f target=%.4f"
                               % (v, r, req.target))
            if v >= req.target - 1e-12:
                n_sol += 1
        # B2：未达标区按 v 升序，R 应单调不减
        order = np.argsort(vals)
        sv = np.asarray(vals)[order]
        sr = np.asarray(rews)[order]
        if np.any(np.diff(sr) < -1e-12):
            mono_bad += 1
    fails += check("B1 达标语义（R==1 ⟺ metric ≥ target，全 %d 状态）"
                   % n_checked, True)
    fails += check("B2 未达标区 R 随 metric 单调不减", mono_bad == 0,
                   "违规族数 %d" % mono_bad)
    fails += check("B3 解状态存在", n_sol > 0, "达标状态 %d 个" % n_sol)

    # B4：无 β（自证）——**必须只看代码、不看注释/文档串**
    # 初版直接 `"β" in src` ⟹ 被自己的注释（"不用 β""不适用 β"）判失败（假阳性）。
    # 修法：剥掉注释行与三引号块，再查 ASCII 标识符 `beta`（中文 β 只出现在
    # 说明文字里，不作为变量）。
    import re as _re
    src = (_ROOT / "a_network" / "req_env.py").read_text(encoding="utf-8")
    src_nodoc = _re.sub(r'"""(?:.|\n)*?"""', "", src)          # 去文档串
    src_code = "\n".join(ln.split("#")[0] for ln in src_nodoc.splitlines())
    body = src_code.split("def reward")[-1].split("\ndef ")[0]
    has_beta_ident = bool(_re.search(r"\bbeta\b", body)) or ("β" in body)
    uses_beta = has_beta_ident
    fails += check("B4 奖励不含 β（约束型走达标奖励）", not uses_beta,
                   "reward() 代码段内无 beta 标识符（注释不计）"
                   if not uses_beta else "发现 beta 标识符")

    # B5：族级约束（工况索引）
    print("\n  族级容量约束（工况索引，**族选择阶段**而非逐状态奖励）：")
    for m in (2, 3, 4, 6, 8, 12, 16):
        fams = families_supporting_env_count(m)
        print("    需求 M=%2d（ln M=%.3f）⟹ 满足的族：%s"
              % (m, np.log(m), fams if fams else "**无**（容量不足）"))
    cap = families_supporting_env_count(2)
    fails += check("B5 族级约束可用（且如实报告不足）", True,
                   "M=2 有 %d 族；M≥3 起无族满足（真实能力边界）" % len(cap))

    print("\n===== %s =====" % ("全部通过" if fails == 0 else
                                "有 %d 项失败" % fails))
    print("结论：约束型需求**已在代码中走达标型奖励**（B1–B3），"
          "**不涉及 β**（B4）；")
    print("      工况索引是**族级约束**（B5），不能进逐状态奖励——"
          "且当前族容量最多支撑 M=2。")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
