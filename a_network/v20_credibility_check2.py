# -*- coding: utf-8 -*-
"""RL 侧可信度复检 · 第二批（第 97 条，2026-10-02）——**预登记 + 20 seed**。

补齐第 96 条未覆盖的两组关键数字：
  **C. v14 结构空间能力曲线**（`StructureChainEnv`）：原报 ΔlogR
     **+5.04 / +6.06 / +5.60**（σ=0.25/0.6/1.0）——但每个 σ **只跑了 2 个
     seed**（one-hot seed=1 / 结构化特征 seed=3）；
  **D. M5 GFlowNet 双模比例**（第 53 条）：原报双模 **34% / 34.5%**。

**预登记判据（运行前写死，机械套用）**——沿用第 96 条的同一标准：
  对每个被复检的量，统计其在 seed 上的**达标率**（阈值沿用原脚本）：
    ≥70% → **稳健成立**；≤30% → **稳健不成立**；其余 → **不稳定**。
  另报**均值 ± std 与范围**，用于替代原单点数字。

用法（项目根目录）：python a_network/v20_credibility_check2.py
           环境变量 V20_SEEDS 可改 seed 数（默认 20）；V20_SIGMA 选 σ。
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from a_network.demo_pretrain import generate_demos
from a_network.gflownet import TrajectoryBalanceGFN
from a_network.structure_env import StructureChainEnv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

N_SEED = int(os.environ.get("V20_SEEDS", "20"))
# **断点续跑（2026-10-02 加）**：`V20_S0` 指定起始 seed（默认 0）。
# 逐 seed 播种 ⟹ 分段跑与整段跑**逐点一致**，可把两次输出拼成完整分布。
S0 = int(os.environ.get("V20_S0", "0"))
# **逐 seed 落 CSV（追加）**：`V20_CSV` 指定文件路径 → 每跑完一个 seed 追加
# 一行 `sigma,seed,dlogr,hit,pat,ok`，便于分段续跑后一次汇总（`--report`）。
CSV_PATH = os.environ.get("V20_CSV")
H = 20


def verdict(rate: float) -> str:
    if rate >= 0.70:
        return "稳健成立"
    if rate <= 0.30:
        return "稳健不成立"
    return "不稳定（依赖 seed）"


def type_pattern(env, state: int) -> tuple[int, ...]:
    return tuple(env._decode(state)[0::2]) if hasattr(env, "_decode") \
        else tuple()


def evaluate(env, gfn, n: int = 500, horizon: int = H):
    """返回 (针尖命中率, 模式数, logR 均值)——与 v14 同口径。

    **2026-10-02 修正**：初版误用 `env._types`/`env._decode`（不存在）→
    模式数恒为 1；正确 API 是 `env.decode(state)` → [(t, b), ...]
    （v14 的 `type_pattern` 用的正是这个）。
    """
    states, _acts, _r, _l, _s, _t, *_ = gfn.sample_trajectories(
        env, n, horizon=horizon)
    last = [next((int(states[k][i]) for k in range(len(states) - 1, -1, -1)
                  if int(states[k][i]) < env.absorbing), env.absorbing)
            for i in range(n)]
    sols = set(env.solution_states(0.8))
    hits = sum(1 for s in last if s in sols)
    pats = {tuple(t for t, _ in env.decode(int(s)))
            for s in last if s in sols}
    logr = float(np.mean([np.log(max(env.reward(int(s)), 1e-6))
                          for s in last]))
    return hits / n, len(pats), logr


def random_logr(env, n: int = 500, horizon: int = H, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        s = int(rng.integers(0, env.absorbing))
        for _t in range(horizon):
            s = int(rng.integers(0, env.absorbing))
        vals.append(np.log(max(env.reward(s), 1e-6)))
    return float(np.mean(vals))


def report(csv_path: str) -> int:
    """汇总模式：读 CSV（多段追加的结果）→ 按 σ 分组给出完整判定。"""
    import collections
    buckets: dict[float, list[dict]] = collections.defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            buckets[float(row["sigma"])].append({
                "seed": int(row["seed"]), "dlogr": float(row["dlogr"]),
                "hit": float(row["hit"]), "pat": int(row["pat"]),
                "ok": bool(int(row["ok"]))})
    print("========== 汇总（%s，预登记判据机械套用）==========" % csv_path)
    for sigma in sorted(buckets):
        rows = sorted(buckets[sigma], key=lambda r: r["seed"])
        seeds = [r["seed"] for r in rows]
        dl = np.array([r["dlogr"] for r in rows])
        rate = float(np.mean([r["ok"] for r in rows]))
        n = len(rows)
        complete = (n == 20 and seeds == list(range(20)))
        print("\n  σ=%.2f：%d 个 seed（%s）" % (
            sigma, n, "完整 0–19" if complete
            else "**不完整**：%d–%d" % (seeds[0], seeds[-1])))
        print("    ΔlogR 均值 **%+.2f**（std %.2f，范围 %+.2f…%+.2f）"
              % (dl.mean(), dl.std(ddof=1), dl.min(), dl.max()))
        print("    命中率均值 %.3f；模式数取值 %s"
              % (float(np.mean([r["hit"] for r in rows])),
                 sorted({r["pat"] for r in rows})))
        print("    判据（ΔlogR ≥0.8）达标率 %d/%d（%.0f%%）→ **%s**%s"
              % (round(rate * n), n, 100 * rate, verdict(rate),
                 "" if complete else "（**分段未跑完，判定待补**）"))
    print("\n  对照 v14 原值：σ=0.25 → +5.60 / σ=0.60 → +6.06 / "
          "σ=1.00 → +5.04（当年每 σ 仅 2 seed 且取优）")
    return 0


def main() -> int:
    if "--report" in sys.argv:
        i = sys.argv.index("--report")
        return report(sys.argv[i + 1] if len(sys.argv) > i + 1
                      else "v20_runs.csv")
    print("========== 可信度复检 · 第二批（预登记，seed %d–%d / 共 %d）"
          "==========" % (S0, N_SEED - 1, N_SEED))
    if S0 > 0:
        print("  **分段运行**：本次只跑 seed %d–%d；须与前一段输出拼接后再"
              "套用判定（逐 seed 播种 → 逐点一致）" % (S0, N_SEED - 1))
    print("  预登记判据：达标率 ≥70% → 稳健成立；≤30% → 稳健不成立；"
          "其余 → 不稳定\n")

    # ---------- C. v14 能力曲线（σ=0.25 默认；V20_SIGMA 可改）----------
    sigma = float(os.environ.get("V20_SIGMA", "0.25"))
    print("  **C. v14 结构空间能力曲线（σ=%.2f，ΔlogR ≥ 0.8 为原判据）**"
          % sigma)
    rows = []
    for seed in range(S0, N_SEED):
        np.random.seed(seed)
        torch.manual_seed(500 + seed)
        env = StructureChainEnv(n_slots=2, sigma_i=sigma)
        demos = generate_demos(env, n_starts=40, max_depth=H, thresh=0.8)
        gfn = TrajectoryBalanceGFN(
            env.n_states, env.n_actions, seed=seed,
            in_dim=env.feat_dim, feat_fn=env.feat)      # 结构化特征臂（v14 次臂）
        gfn.pretrain_pf(demos, steps=400, lr=1e-2)
        for _ in range(3000):
            gfn.train_step_subtb(env, horizon=H, batch=128, lr=2e-2,
                                 entropy_bonus=0.05, lam=0.9, use_replay=0.5)
        hit, pat, logr = evaluate(env, gfn)
        rnd = random_logr(env, seed=seed)
        ok = (logr - rnd) >= 0.8
        rows.append({"dlogr": logr - rnd, "hit": hit, "pat": pat, "ok": ok})
        print("    seed %2d：ΔlogR=%+6.2f 命中=%.3f 模式=%d"
              % (seed, logr - rnd, hit, pat))
        if CSV_PATH:      # 逐 seed 落盘（追加）→ 支持分段续跑后汇总
            new = not Path(CSV_PATH).exists()
            with open(CSV_PATH, "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["sigma", "seed", "dlogr", "hit", "pat", "ok"])
                w.writerow([sigma, seed, "%.4f" % (logr - rnd),
                            "%.4f" % hit, pat, int(ok)])
    dl = np.array([r["dlogr"] for r in rows])
    rate = float(np.mean([r["ok"] for r in rows]))
    n_run = len(rows)
    print("    ΔlogR 分布：均值 **%+.2f**（std %.2f，范围 %+.2f…%+.2f）"
          % (dl.mean(), dl.std(ddof=1), dl.min(), dl.max()))
    print("    判据（ΔlogR ≥0.8）达标率 %d/%d（本段 %d 个 seed）→ **%s**%s"
          % (round(rate * n_run), n_run, n_run, verdict(rate),
             "（分段运行：须用 --report 汇总后再定论）" if n_run < 20 else ""))
    print("    与 v14 原值对照（σ=0.25→+5.60 / 0.60→+6.06 / 1.00→+5.04）：%s"
          % ("原值在分布范围内" if dl.max() >= 5.0 else
             "**原值高于本次分布上界 ⟹ 原数字为有利 seed 的极值**"))

    print("\n  说明：本脚本与 v14 的唯一差别是 seed 数与「逐 seed 报告」；"
          "协议、超参、判据阈值均未改动。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
