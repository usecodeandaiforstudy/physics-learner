# -*- coding: utf-8 -*-
"""缺口③ 任务①：**多约束满足任务（G3 重定义）+ 可行性预检**
（2026-10-02，第 126 条）。

**为什么必须重定义**（第 125 条的发现）：DPI ⟹ 链越长信息越少
⟹ 用 `R = β·I_TY − I_XT` 时**最优策略永远是停在单节点**
⟹ **组合编辑没有回报信号**（实测最佳 2 节点链 +0.0138 vs 1 节点 +2.656）。
⟹ 组合的价值只能落在**约束满足**上：

    G3 任务 = 找结构 s 使得
        `节点数 ≥ k`  ∧  `I_TY ≥ a`  ∧  `I_XT ≤ b`
    奖励 = **达标型**（`R=1` 若全部满足；否则按违反量给 0–1 的软惩罚），
    与第 114 条确立的口径一致（阈值/约束型需求**不用 β**）。

**本脚本做两件事**：
  **A. 前沿测量**：按结构规模（1/2/3 节点）抽样**DPI 合法**结构，
     统计 `(I_TY, I_XT, R_ib)` 的分布 ⟹ 得到"多节点能到什么水平"的**事实**。
  **B. 约束定标 + 可行率**：阈值**由数据按明写规则**取（3 节点的 I_TY 中位、
     I_XT 中位、k=3），再统计**满足率**（目标 10–70%：太易则平凡、太难则无信号）。
     并列出满足约束的**样例结构**（存在性证据）。

用法（项目根目录）：python a_network/v32_g3_task_check.py
"""

from __future__ import annotations

import json
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

N_SAMPLE = 60          # 每个规模的抽样数（合法结构内）
N_NODES_LIST = (1, 2, 3, 4)


def sample_legal(env, rng, n_nodes: int, n_try: int = 4000) -> list:
    """在**DPI 合法**的多节点链里抽样（随机组件 + 随机参数）。"""
    from a_network.graph_env import COMPONENTS
    from a_network.structure_space import StructSpec, chain_is_dpi_legal
    comps = list(COMPONENTS)
    out = []
    tries = 0
    while len(out) < n_nodes and tries < n_try:
        tries += 1
        nodes = []
        for _ in range(n_nodes):
            c = comps[int(rng.integers(0, len(comps)))]
            lo, hi = COMPONENTS[c][2]
            nodes.append((c, float(rng.uniform(lo, hi))))
        sp = StructSpec(tuple(nodes), tuple((i, i + 1)
                                            for i in range(n_nodes - 1)))
        if chain_is_dpi_legal(sp):
            out.append(sp)
    return out


def sample_legal_batch(env, rng, n_nodes: int, want: int) -> list:
    from a_network.graph_env import COMPONENTS
    from a_network.structure_space import StructSpec, chain_is_dpi_legal
    comps = list(COMPONENTS)
    out, tries = [], 0
    while len(out) < want and tries < 20000:
        tries += 1
        nodes = []
        for _ in range(n_nodes):
            c = comps[int(rng.integers(0, len(comps)))]
            lo, hi = COMPONENTS[c][2]
            nodes.append((c, float(rng.uniform(lo, hi))))
        sp = StructSpec(tuple(nodes), tuple((i, i + 1)
                                            for i in range(n_nodes - 1)))
        if chain_is_dpi_legal(sp):
            out.append(sp)
    return out


def main() -> int:
    from a_network.graph_env import GraphEditEnv
    env = GraphEditEnv()
    rng = np.random.default_rng(0)
    print("=" * 74)
    print("缺口③ 任务①：多约束满足任务（G3 重定义）——可行性预检")
    # **版本自证标记（2026-10-02 加）**：连续两次运行输出与旧版逐字节相同，
    #   但文件已确认更新 ⟹ 加标记以判定"实际执行的是哪份代码"。
    #   若此行不出现，说明跑的是**旧文件**（路径/缓存/复制副本问题）。
    print("[[v32 build-2：含 (a) 可评估率一等读数 (b) 阈值退化守卫 "
          "]]")
    print("[[本文件：%s；修改时间 %s]]"
          % (Path(__file__).resolve(),
             __import__("datetime").datetime.fromtimestamp(
                 Path(__file__).stat().st_mtime).strftime(
                     "%Y-%m-%d %H:%M:%S")))
    print("=" * 74)

    # ---------- A. 前沿测量 ----------
    print("\nA. 前沿测量（**DPI 合法**结构，每规模抽 %d 个）：" % N_SAMPLE)
    frontier: dict[int, dict] = {}
    for n in N_NODES_LIST:
        sps = sample_legal_batch(env, rng, n, N_SAMPLE)
        if not sps:
            print("  %d 节点：**抽到 0 个合法结构**（合法区为空）" % n)
            continue
        ity, ixt, rib = [], [], []
        for sp in sps:
            r, info = env._reward_of(sp)
            if "I_TY" in info:
                ity.append(float(info["I_TY"]))
                ixt.append(float(info["I_XT"]))
                rib.append(r)
        if not ity:
            print("  %d 节点：%d 个合法结构，但**全部无可评估曲线**"
                  % (n, len(sps)))
            continue
        frontier[n] = {"ity": np.array(ity), "ixt": np.array(ixt),
                       "rib": np.array(rib), "n_sampled": len(sps),
                       "n_eval": len(ity)}
        print("  %d 节点：合法抽样 %d；**可评估 %d（%.0f%%）**；"
              "I_TY 中位 %.4f（最大 %.4f）；I_XT 中位 %.4f；R_ib 中位 %+.4f"
              % (n, len(sps), len(ity), 100.0 * len(ity) / max(len(sps), 1),
                 float(np.median(ity)), float(np.max(ity)),
                 float(np.median(ixt)), float(np.median(rib))))
    print("  ⟹ **可评估率随节点数下降**（这是本题最重要的结构事实："
          "多节点链大多算不出 IB 曲线或信息趋 0）")

    # ---------- B. 约束定标（规则明写 + **退化守卫**） ----------
    print("\nB. 约束定标（**阈值由数据按明写规则取**）：")
    if 3 not in frontier:
        print("  **3 节点无合法可评估结构 ⟹ G3 任务不可行**（须改设计）")
        return 1
    f3 = frontier[3]
    # **退化守卫（2026-10-02 加，关键）**：初版规则"取中位数"在 3 节点
    #   I_TY 中位 = 0.0000 时**退化为 `a=0, b=0`** ⟹ `I_TY ≥ 0` 恒真、
    #   `I_XT ≤ 0` 只有**零信息结构**满足 ⟹ "满足率 14%" 是**假结论**
    #   （满足者是垃圾结构）。而初版判读**只看了比率、没检查阈值是否有意义**
    #   —— 这是本会话第三次"判据自检缺失"（前两次：C3 阈值饱和、
    #   保序性质选错）。现规则：**中位数若退化为 0，则改用 90 分位**，
    #   并**强制 `a > 0`**（否则判任务不可行）。
    a_med = float(np.median(f3["ity"]))
    b_med = float(np.median(f3["ixt"]))
    k = 3
    rule = "3 节点 I_TY 中位"
    a, b = a_med, b_med
    # **退化阈值下限（1e-3）而非"恰好 0"（2026-10-02 修）**：实测中位数是
    #   **极小的正数**（`%.4f` 打成 0.0000 但 `> 0`）⟹ 初版 `a <= 0.0` 守卫
    #   **不触发**，于是沿用了一个数值上无意义的阈值。判据须用**有意义的下限**。
    DEGEN = 1e-3
    if a < DEGEN:
        a = float(np.percentile(f3["ity"], 90))
        b = float(np.percentile(f3["ixt"], 90))
        rule = ("中位 %.2e < 1e-3（数值上无意义）⟹ 改用 **90 分位**"
                "（'进入 3 节点链能做的前 10%%'）" % a_med)
    print("  规则：k = %d；a = **%.4f**；b = **%.4f**（%s）" % (k, a, b, rule))
    if a < DEGEN:
        print("  **即使 90 分位 I_TY = %.2e 仍 < 1e-3 ⟹ 3 节点链几乎不产生信息**"
              % a)
        print("  ⟹ **G3（以信息量为目标）在当前组件库下不可行**——"
              "这与 DPI 一致（链越长信息越少），是有价值的负结论")
        return 1
    sat = [(f3["ity"][i] >= a and f3["ixt"][i] <= b)
           for i in range(len(f3["ity"]))]
    rate = float(np.mean(sat)) if sat else 0.0
    print("  在 3 节点**合法可评估**结构中的满足率：**%.0f%%**（%d/%d）"
          % (100 * rate, int(np.sum(sat)), len(sat)))
    # 跨规模：约束是否被单节点"平凡满足"？（若 1/2 节点也能满足，任务无组合性）
    for n in (1, 2):
        if n in frontier:
            fn = frontier[n]
            r_n = float(np.mean([(fn["ity"][i] >= a and fn["ixt"][i] <= b)
                                 for i in range(len(fn["ity"]))]))
            print("  %d 节点结构的满足率：%.0f%%（**>0 表示约束未强制组合**）"
                  % (n, 100 * r_n))
    # 可行率判读
    verdict = ("**任务可行且非平凡**（10–70%）" if 0.10 <= rate <= 0.70 else
               "**过易**（>70%，组合不必要）" if rate > 0.70 else
               "**过难**（<10%，信号稀疏）")
    print("  判读：%s" % verdict)

    # ---------- C. 样例（存在性证据） ----------
    print("\nC. 满足约束的样例结构（存在性证据；**要求非退化** I_TY ≥ 1e-3）：")
    shown = 0
    for i, sp in enumerate(sample_legal_batch(env, rng, 3, 200)):
        r, info = env._reward_of(sp)
        if "I_TY" not in info:
            continue
        if info["I_TY"] >= max(a, 1e-3) and info["I_XT"] <= b:
            print("   %s → I_TY %.4f / I_XT %.4f（R_ib %+.4f）"
                  % ([c for c, _v in sp.nodes],
                     info["I_TY"], info["I_XT"], r))
            shown += 1
            if shown >= 5:
                break
    if shown == 0:
        print("   **未找到非退化样例** ⟹ 3 节点链的'满足约束'多由**零信息"
              "结构**造成（判据须要求 I_TY ≥ 1e-3，否则会奖励垃圾结构）")
    print("\n结论：G3 任务定义 = `节点数 ≥ %d ∧ I_TY ≥ %.4f ∧ I_XT ≤ %.4f`"
          "（**达标型奖励**，不用 β）" % (k, a, b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
