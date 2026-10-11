# -*- coding: utf-8 -*-
"""系统端到端运行器（2026-10-02）——**一条命令验证整条链是否跑通**。

把散落在 100+ 实验脚本里的检查，编排成**有阶段、有分层、有判定**的统一入口：
  账本层 → 理论层 → B 线 → A/RL 层 → C 线 → 外部验证层 → 可信度层
每阶段独立子进程执行、单独超时、单独判定；输出汇总表与退出码。

**三档运行深度**（默认 --fast，几十秒到几分钟）：
  --fast   账本/理论/B/C/模拟器 的关键检查（日常改代码后跑）
  --full   追加重型验证（B v1 LOOCV、多 seed 对照、可信度复检套件）
  --list   只列阶段表

**判定语义**：
  PASS        退出码 0
  FAIL        退出码非 0 且**不在已知边界清单**里 → 计入失败
  边界(期望失败) 该脚本是**已归档的边界/负结果**（如 M6.2 4/5 失败）→ 不计失败
  SKIP        按深度档跳过
  TIMEOUT     超时（单独标注，不计入 FAIL 但会让退出码非 0）

用法（项目根目录）：
  python run_system.py              # 快档
  python run_system.py --full       # 全档
  python run_system.py --only "B 生产冒烟"
  python run_system.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# ---------------------------------------------------------------------------
# 阶段表：(名称, 命令, 档位, 是否已知边界/期望失败, 超时秒, 模式)
#   tier: 1 = fast 档执行；2 = 仅 full 档执行
#   mode: "gate" = 退出码即判定（门禁）；"info" = 测量类（跑通即 OK，
#         其数值不构成通过/失败——**避免把测量数字误读为合格**）
#   stale: True = **已知待修**（记录不可复现/受上游变更影响）——单独统计、
#         不使整轮失败，但**必须随时间下降且每项在 HANDOVER 有条目**
# ---------------------------------------------------------------------------
STAGES: list[tuple[str, list[str], int, bool, int, str, bool]] = [
    # ---- 账本层：数据完整性与口径 ----
    ("账本·DB3 校验", ["db3/validate_pair.py"], 1, False, 300, "gate", False),
    ("账本·DB3↔解析器一致性", ["db3/check_parser_consistency.py"], 1, False, 300, "gate", False),
    # ---- 理论层：核心不变量 ----
    ("理论·core 数学自测", ["core/tests/test_core.py"], 1, False, 300, "gate", False),
    ("理论·DPI/Fisher 约束", ["b_network/test_theory_constraints.py"], 1, False, 900, "gate", False),
    # ---- B 线：生产解析路径 ----
    ("B·生产入口冒烟", ["b_network/smoke_b_parse.py"], 1, False, 600, "gate", False),
    ("B·新表述测试集（测量）", ["b_network/eval_b_testset.py"], 1, False, 900, "info", False),
    # ---- 功能闭环 ----
    ("闭环·需求→设计→验证（含断点报告）", ["pipeline/run_loop.py"], 1, False, 1200, "gate", False),
    # ---- 理论侧：口径与标定（第 107–117 条新增）----
    ("理论·β 标定（159 锚点，测量）", ["pipeline/beta_calibration.py"], 1, False, 600, "info", False),
    ("理论·约束型需求达标奖励（B1–B5）", ["a_network/v25_constraint_req_check.py"], 1, False, 600, "gate", False),
    ("理论·变分层 L1 解析对照（3 情形）", ["a_network/v26_variational_check.py"], 1, False, 1200, "gate", False),
    # **已知边界（2026-10-02 修）**：T1（指纹 LOFO gap 4.2%）与 T3（反解方向 3/3）
    #   通过，**T2「族级容量不可外推」是第 120 条已归档的负结果**（误差 95.9%）
    #   ⟹ 本阶段整体标为**边界(期望失败)**；初版误设为 False，导致它被当作
    #   "非预期失败"（运行器元数据错误，非新退化）。
    ("理论·跨域迁移·族级（T1–T3）", ["a_network/v29_cross_domain_transfer.py"], 1, True, 900, "gate", False),
    ("理论·**跨族不变量迁移**（T-c，测量）", ["a_network/v38_capacity_invariant_transfer.py"], 1, False, 900, "info", False),
    ("理论·跨域迁移·域级（D1–D2）", ["a_network/v30_domain_transfer_check.py"], 1, False, 900, "gate", False),
    ("理论·**严格 LODO 跨域检验**（留一域，52 域）", ["b_network/v31_lodo_check.py"], 2, False, 3600, "info", False),
    ("B·**新锚点域内迁移**（留一 LOO，测量）", ["b_network/v35_new_anchor_transfer.py"], 1, False, 900, "info", False),
    ("理论·核心不变量（pytest 兼容）", ["tests/test_invariants.py"], 1, False, 600, "gate", False),
    # ---- 中游：反解覆盖 + 非退化设计域（第 127–130 条）----
    ("中游·族名别名自检（单一真源）", ["pipeline/family_aliases.py"], 1, False, 300, "gate", False),
    ("中游·q→参数反解（21 族，含三重校验）", ["pipeline/inverse_fisher.py"], 1, False, 1800, "gate", False),
    ("中游·非退化设计域（two_level，闭环）", ["pipeline/design_two_level.py"], 1, False, 900, "gate", False),
    ("中游·rayleigh 非退化域（第 2 族）", ["pipeline/design_rayleigh.py"], 1, False, 900, "gate", False),
    # ---- 架构 §13：奖励分类 / 设计空间 / 验证反馈闭环 ----
    ("架构·奖励类型分类（一等公民）", ["pipeline/reward_types.py"], 1, False, 300, "gate", False),
    ("架构·验证反馈闭环（rayleigh）", ["pipeline/design_reflow.py", "--domain", "rayleigh", "--target", "3.0", "--n", "120"], 1, False, 900, "gate", False),
    ("架构·验证反馈闭环（two_level）", ["pipeline/design_reflow.py", "--domain", "two_level", "--target", "3.0", "--n", "120"], 1, False, 900, "gate", False),
    ("架构·组合链高维域预检（dof 4–6）", ["pipeline/design_chain.py"], 1, False, 900, "gate", False),
    ("架构·验证反馈闭环（组合链）", ["pipeline/design_reflow.py", "--domain", "chain", "--target", "5.0", "--n", "120"], 1, False, 900, "gate", False),
    ("架构·闭环消费（DB5 + 加权数据集）", ["pipeline/design_reflow_consume.py", "--all"], 1, False, 900, "gate", False),
    ("架构·声学阵列域预检（A1–A6，交叉校验）", ["pipeline/design_array.py"], 1, False, 1800, "gate", False),
    ("架构·验证反馈闭环（声学阵列）", ["pipeline/design_reflow.py", "--domain", "array", "--target", "3.0", "--n", "120"], 1, False, 1800, "gate", False),
    ("架构·设计空间 RL 接口（门禁，5 域）", ["pipeline/rl_design_env.py", "--env"], 1, False, 900, "gate", False),
    # ---- RL 层：环境与接线 ----
    ("RL·环境集成冒烟（M4）", ["a_network/v4_env_smoke.py"], 1, False, 600, "gate", False),
    ("RL·M6.2 结构编辑边界", ["a_network/v6c_m6_2_check.py"], 1, True, 900, "gate", False),
    # ---- C 线：解码器 ----
    ("C·MDN 解码器判据", ["c_network/v8_mdn_check.py"], 1, False, 900, "gate", False),
    # ---- 外部验证层（缺口①）----
    ("验证·模拟器真值自检", ["sim/sim_check.py"], 1, False, 900, "gate", False),
    ("验证·**验证证书 + 不确定度**（H1–H5，测量）", ["sim/certificate_check.py"], 2, False, 1800, "info", False),
    ("验证·**外部印证**（diffusion/q7，三外部方法）", ["sim/external_check_diffusion.py"], 2, False, 1800, "info", False),
    ("验证·**外部印证扩到 5 族**（闭式/数值/MC）", ["sim/external_check_five_families.py"], 2, False, 1800, "info", False),
    ("验证·生产级验证 API（域外=unknown）", ["sim/verify_api.py"], 1, False, 900, "gate", False),
    # **第 206 条发现 → 第 207 条修复**：3 族（`gaussian_mixture2`/`landau`/
    #   `truncated_gaussian`）的「注册表测量 vs 节点真值」曾超限
    #   （1951.86% / 467.55% / 2404.91% ✗）——根因是 `measure_for_verify`
    #   **忽略 `TRUTH_METRIC` 的真值口径**（与 `calibrate()` 不一致 ✗）。
    #   修后实测 **0/27 超限** ✓ ⟹ **本阶段由「已知待修」转为常规门禁** ✓
    ("验证·**注册表测量 vs 真值扫描**（27 族，门禁）", ["sim/registry_truth_scan.py"], 1, False, 1800, "gate", False),
    ("验证·速率适配器真值自检", ["sim/rate_adapter.py"], 1, False, 900, "gate", False),
    ("验证·速率型需求覆盖（q9/q7）", ["sim/v36_rate_coverage.py"], 1, False, 900, "gate", False),
    ("验证·解析自洽 vs 模拟器（测量）", ["sim/verify_designs.py"], 1, False, 1800, "info", False),
    ("验证·覆盖校准（27 族，测量）", ["sim/expand_registry.py"], 1, False, 1800, "info", False),
    # ---- 重型（--full）：训练类与多 seed 复检 ----
    ("重型·接线检查（DB4离策略+A嵌入，含训练）", ["a_network/v15_wiring_check.py"], 2, False, 5400, "gate", False),
    ("重型·变分层 L2 增量（测量，20 seed）", ["a_network/v27_variational_value_check.py"], 2, False, 3600, "info", False),
    ("重型·中游 B·C 实例需求→设计（20 seed，测量）", ["pipeline/v33_landau_c_check.py"], 2, False, 3600, "info", False),
    ("重型·R5②·rayleigh C 实例（20 seed，测量）", ["pipeline/v34_rayleigh_c_check.py"], 2, False, 3600, "info", False),
    ("重型·R5①·chain C 实例（20 seed，测量）", ["pipeline/v35_chain_c_check.py"], 2, False, 3600, "info", False),
    ("重型·R5③·array C 实例（20 seed，测量）", ["pipeline/v37_array_c_check.py"], 2, False, 3600, "info", False),
    ("重型·设计空间 RL 接口（20 seed，测量）", ["pipeline/rl_design_env.py"], 2, False, 3600, "info", False),
    ("重型·迁移缺口 vs 几何距离（测量，20 seed）", ["a_network/v28_transfer_gap_check.py"], 2, False, 3600, "info", False),
    ("重型·B 标定刷新（Platt，5 折 CV）", ["b_network/bv1_calibrate.py"], 2, False, 3600, "gate", False),
    ("重型·B v1 训练评估（LOOCV 159 锚点）", ["b_network/bv1_train_eval.py"], 2, False, 7200, "gate", False),
    ("重型·M6.2+M7 复检（20 seed，测量）", ["a_network/v19_rl_credibility_check.py"], 2, False, 7200, "info", False),
    ("重型·M5 双模复检（20 seed，测量）", ["a_network/v21_m5_credibility_check.py"], 2, False, 3600, "info", False),
    ("重型·MDN↔CFM 对照（20 seed，测量）", ["c_network/v22_cfm_check.py"], 2, False, 7200, "info", False),
    ("重型·SAC 单峰寻优（3 seed，seed 敏感）", ["a_network/v7_sac_check.py"], 2, True, 3600, "gate", False),
]

TAIL_LINES = 4


def run_stage(name: str, cmd: list[str], timeout: int) -> tuple[str, float, str]:
    """执行单个阶段，返回 (状态, 秒数, 输出尾部)。"""
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, str(ROOT / cmd[0])] + cmd[1:],
                           cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8",
                           errors="replace")
        dur = time.time() - t0
        tail = "\n".join((p.stdout or "").strip().splitlines()[-TAIL_LINES:])
        if p.returncode == 0:
            return "PASS", dur, tail
        err = "\n".join((p.stderr or "").strip().splitlines()[-TAIL_LINES:])
        return "FAIL", dur, (tail + "\n" + err).strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - t0, "超时 %ds" % timeout
    except Exception as ex:  # noqa: BLE001
        return "ERROR", time.time() - t0, "%s: %s" % (type(ex).__name__, ex)


def main() -> int:
    ap = argparse.ArgumentParser(description="系统端到端运行器")
    ap.add_argument("--full", action="store_true", help="含重型验证阶段")
    ap.add_argument("--list", action="store_true", help="只列阶段表")
    ap.add_argument("--only", default=None, help="只跑名称包含该子串的阶段")
    ap.add_argument("--quiet", action="store_true", help="不打印阶段尾部输出")
    args = ap.parse_args()

    depth = 2 if args.full else 1
    stages = [s for s in STAGES
              if s[2] <= depth and (args.only is None or args.only in s[0])]
    if args.list or not stages:
        print("阶段表（档位 1=fast / 2=full；* = 已知边界，期望失败；"
              "[i] = 测量类，数值不构成通过/失败；[S] = 已知待修）：")
        for n, c, t, known, to, mode, stale in STAGES:
            print("  [%s] %-40s %s%s%s%s"
                  % ("fast" if t == 1 else "full", n, " ".join(c),
                     "  *" if known else "",
                     "  [i]" if mode == "info" else "",
                     "  [S]" if stale else ""))
        return 0

    print("=" * 74)
    print("系统端到端运行：%s 档，%d 个阶段" % ("full" if args.full else "fast",
                                              len(stages)))
    print("=" * 74)
    results = []
    for name, cmd, _tier, known, timeout, mode, stale in stages:
        print("\n>>> %s（%s）" % (name, " ".join(cmd)), flush=True)
        status, dur, tail = run_stage(name, cmd, timeout)
        shown = status
        if known and status != "PASS":
            shown = "边界(期望失败)"
        elif stale and status != "PASS":
            shown = "已知待修"
        elif mode == "info" and status == "PASS":
            shown = "OK(测量类)"          # 明确语义：跑通，但数值非判定
        results.append((name, status, shown, dur, tail, mode, stale))
        print("    -> %s（%.1fs）" % (shown, dur), flush=True)
        if not args.quiet and tail:
            for line in tail.splitlines():
                print("       | " + line)

    print("\n" + "=" * 74)
    print("%-44s %-14s %8s" % ("阶段", "判定", "秒"))
    print("-" * 74)
    for name, status, shown, dur, _tail, _m, _s in results:
        print("%-44s %-14s %8.1f" % (name, shown, dur))
    n_pass = sum(1 for r in results if r[1] == "PASS")
    n_bound = sum(1 for r in results if r[2].startswith("边界"))
    n_stale = sum(1 for r in results if r[2] == "已知待修")
    # **计数修正（2026-10-02）**：`n_fail` 须同时排除 **已知边界** 与 **已知待修**
    #   ——初版只排除 stale ⟹ **已知边界项被"边界"与"FAIL"各计一次**（实测
    #   PASS 21 + 边界 1 + FAIL 2 = 24 > 23）。
    n_fail = sum(1 for r in results
                 if r[1] in ("FAIL", "ERROR") and not r[6]
                 and not r[2].startswith("边界"))
    n_to = sum(1 for r in results
               if r[1] == "TIMEOUT" and not r[6]
               and not r[2].startswith("边界"))
    print("-" * 74)
    print("PASS %d / 边界 %d / **已知待修 %d** / FAIL %d / TIMEOUT %d（共 %d）"
          % (n_pass, n_bound, n_stale, n_fail, n_to, len(results)))
    print("总耗时 %.1fs" % sum(r[3] for r in results))
    if n_stale:
        print("已知待修项（须随时间下降，每项在 HANDOVER 有条目）：")
        for name, status, shown, _d, _t, _m, s in results:
            if s and status != "PASS":
                print("  - %s" % name)
    if n_fail or n_to:
        print("**存在非预期失败/超时——按架构文档 §10.2 第 7 条：不得当作结果，"
              "须先查实现**")
        return 1
    print("系统跑通：无未解释的失败；已知边界与已知待修项见上。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
