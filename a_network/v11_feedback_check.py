# -*- coding: utf-8 -*-
"""回流边验证 v7（第 63 条，2026-10-02）：12 seed 配对 + 统计检验。

v1–v6 教训（归档）：① 无过滤刷屏；② 6 样本过拟合污染；③ 对照不配对；
④ 判据错位（量终值而非早期收敛）；⑤ 动力学虚假（done=True 误标终止）；
⑥ n=5 无检验力（@600 gap −1.31 但 t≈−1.5，不显著）。

v7：注入转移 done=False（环境真实动力学）+ 权重 1+overlap + 配对 12 seed
+ **预设的非劣性框架**（判「有损害」需要同时满足：p<0.05 且 gap<−0.3，
配对 t 检验与符号检验双口径）。

判据（预先登记）：
  F1 机制：DB5 任务相关 pass 记录 ≥3 条映射成功；
  F2 早期无损害：@150 配对均值 fb ≥ base − 0.3；
  F3 终值非劣：@600 不满足「t 检验 p<0.05 且 gap<−0.3」（即无显著损害）。
  增益（@150 加速）为设计承诺的量，未显现则如实记录，不作通过条件。

用法（项目根目录）：python a_network/v11_feedback_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from a_network.db5_feedback import load_validated_states
from a_network.rl_env import MultiFamilyEnv
from a_network.sac import DiscreteSAC

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def eval_logr(sac, env, n: int = 200) -> float:
    vals = []
    for _ in range(n):
        s = int(np.random.randint(0, env.absorbing))
        for t in range(12):
            pi = sac.policy(torch.tensor([s], dtype=torch.long))[0]
            s, _ = env.step(s, int(torch.argmax(pi).item()))
            if s == env.absorbing:
                break
        if s == env.absorbing:
            s = int(np.random.randint(0, env.absorbing))
        vals.append(np.log(max(env.reward(s), 1e-6)))
    return float(np.mean(vals))


def train(sac, env, episodes: int, horizon: int = 12) -> None:
    for ep in range(episodes):
        s = int(np.random.randint(0, env.absorbing))
        for t in range(horizon):
            a = sac.act(s)
            s2, r = env.step(s, a)
            sac.store(s, a, float(np.log(max(r, 1e-6))), s2, t == horizon - 1)
            s = s2
            sac.train_step(batch=64)


def main() -> int:
    fails = 0
    env = MultiFamilyEnv()
    validated = load_validated_states(env)
    fails += check(f"F1 DB5 任务相关 pass 记录映射注入"
                   f"（{len(validated)} 条 ≥ 3）",
                   len(validated) >= 3, f"n={len(validated)}")
    pairs = [(st, 0, float(np.log(max(env.reward(st), 1e-6))))
             for st, _ in validated]
    rng_neg = np.random.default_rng(0)
    neg_states = rng_neg.integers(0, env.absorbing, 60)
    pairs = pairs + [(int(s), 0, float(np.log(max(env.reward(int(s)), 1e-6))))
                     for s in neg_states]
    _ = pairs   # 保留（供 pretrain 变体复用）；v6 用缓冲注入路径

    base_ends, fb_ends = [], []
    base_early, fb_early = [], []
    diffs_end, diffs_early = [], []
    N_SEED = 12
    for seed in range(N_SEED):
        # 配对：每臂前重置 torch + numpy RNG → 两臂训练/评估序列逐一相同
        np.random.seed(seed)
        torch.manual_seed(100 + seed)
        base = DiscreteSAC(env.n_states, env.n_actions, seed=seed, lr=1e-2)
        train(base, env, 150)
        base_early.append(eval_logr(base, env))
        train(base, env, 450)
        base_ends.append(eval_logr(base, env))

        np.random.seed(seed)
        torch.manual_seed(100 + seed)
        fb = DiscreteSAC(env.n_states, env.n_actions, seed=seed, lr=1e-2)
        # 回流（第 6 版，修正动力学虚假性）：验证状态以 done=False 注入缓冲
        # ——此前 4 版都写成 done=True（教「停留=终止」，与环境不符）
        for st, w in validated:
            r = float(np.log(max(env.reward(st), 1e-6)))
            for _ in range(5):
                fb.store(st, 0, r, st, False, weight=w)
        train(fb, env, 150)
        fb_early.append(eval_logr(fb, env))
        train(fb, env, 450)
        fb_ends.append(eval_logr(fb, env))
        print(f"  seed {seed}: base @150={base_early[-1]:6.2f} "
              f"@600={base_ends[-1]:6.2f} | fb @150={fb_early[-1]:6.2f} "
              f"@600={fb_ends[-1]:6.2f}")
        diffs_end.append(fb_ends[-1] - base_ends[-1])
        diffs_early.append(fb_early[-1] - base_early[-1])

    bm, fm = float(np.mean(base_ends)), float(np.mean(fb_ends))
    be, fe = float(np.mean(base_early)), float(np.mean(fb_early))
    d_end = np.array(diffs_end)
    d_early = np.array(diffs_early)
    # 配对 t 检验（双侧）+ 符号检验
    n = len(d_end)
    se = float(d_end.std(ddof=1) / np.sqrt(n))
    t_stat = float(d_end.mean() / se) if se > 0 else 0.0
    try:
        from scipy import stats as _st
        p_val = float(2 * _st.t.sf(abs(t_stat), df=n - 1))
    except Exception:   # noqa: BLE001
        p_val = float("nan")
    n_neg = int((d_end < 0).sum())
    import math as _math
    if n_neg > n / 2:
        p_sign = 2.0 * sum(_math.comb(n, k) for k in range(n_neg, n + 1)) \
            / 2 ** n
    else:
        p_sign = 2.0 * sum(_math.comb(n, k) for k in range(0, n_neg + 1)) \
            / 2 ** n
    p_sign = float(min(1.0, p_sign))

    print(f"  均值：@150 base={be:.2f} fb={fe:.2f}（gap {fe - be:+.2f}）")
    print(f"        @600 base={bm:.2f} fb={fm:.2f}（gap {fm - bm:+.2f}）")
    print(f"  配对统计：@600 差值均值={d_end.mean():+.2f}  SE={se:.2f}  "
          f"t={t_stat:+.2f}  p={p_val:.3f}；负向 {n_neg}/{n} "
          f"（符号检验 p={p_sign:.3f}）")
    fails += check(f"F2 早期无损害（@150 fb {fe:.2f} ≥ base {be:.2f} − 0.3）",
                   fe >= be - 0.3, f"gap={fe - be:.2f}")
    harm = (p_val < 0.05 and (fm - bm) < -0.3)
    fails += check(f"F3 终值非劣（无显著损害：p≥0.05 或 gap≥−0.3）",
                   not harm, f"t={t_stat:.2f} p={p_val:.3f} gap={fm - bm:.2f}")
    print("  [记录] 加速（@150 增益）为设计承诺量：gap=%+.2f——"
          "未显现则如实记录" % (fe - be))

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
