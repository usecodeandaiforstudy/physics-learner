# -*- coding: utf-8 -*-
"""DB5 建库（v0，2026-10-02）：DB4 landau 结构 → C 解码 → 解析真值验证。

闭环（design doc §3.6）：DB4 ─▶ C（landau MDN）─▶ DB5 验证 ─▶ 回流。
验证口径（§2.6.2 覆盖率审计 v0 代理）：
  - mode_match：预测密度峰 vs 解析峰（a<0：±√(−a/b)；a≥0：0）；
  - overlap：∫ min(p_pred, p_true) dφ（网格）≥ 0.85 → pass；
  - 汇总 pass 率 < 90% → 触发 MDN 淘汰条款（文档级，上报）。

注意：DB4 里 landau 记录的 a（±1.5、−2.0 等）多为 MDN 训练网格
{−2,−1,0,1,2} 之外的值——解码即插值外推的真实验证。

用法（项目根目录）：python db5/v10_db5_build.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from c_network.mdn import MixtureDensityNetwork, sample_landau
from db4.db4_writer import DB4Writer
from db5.db5_writer import DB5Writer, build_record

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

_GRID = np.linspace(-3.0, 3.0, 2000)


def true_density(a: float, grid: np.ndarray) -> np.ndarray:
    logp = -(a * grid ** 2 / 2.0 + grid ** 4 / 4.0)
    logp = logp - logp.max()
    p = np.exp(logp)
    return p / (p.sum() * (grid[1] - grid[0]))


def true_modes(a: float) -> list[float]:
    if a < 0:
        m = float(np.sqrt(-a / 1.0))
        return [-m, m]
    return [0.0]


def main() -> int:
    torch.manual_seed(0)
    # ---- 训练 landau 领域解码器（与 M8 同款）----
    a_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
    X, Y = [], []
    for a in a_vals:
        phi = sample_landau(a, 1.0, 1.0, 4000, seed=0)
        X.append(np.full(len(phi), a, dtype=np.float32))
        Y.append(phi.astype(np.float32))
    X = np.concatenate(X)[:, None]
    Y = np.concatenate(Y)[:, None]
    mdn = MixtureDensityNetwork(1, n_components=8)
    opt = torch.optim.Adam(mdn.parameters(), lr=3e-3)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for step in range(6000):
        idx = np.random.randint(0, len(Xt), 512)
        loss = mdn.nll(Xt[idx], Yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    # ---- DB4 → C 解码 → 验证 ----
    db4 = DB4Writer().load()
    landau_recs = [r for r in db4 if r["domain"] == "landau"]
    print(f"DB4 landau 记录：{len(landau_recs)} 条")
    writer = DB5Writer()
    n_new = n_dup = n_err = n_pass = 0
    for r in landau_recs:
        a = float(r["params"]["a"])
        p_pred = mdn.density(a, _GRID)
        p_true = true_density(a, _GRID)
        overlap = float(np.trapezoid(np.minimum(p_pred, p_true), _GRID))
        pm = mdn.modes(a, _GRID)
        tm = true_modes(a)
        mode_match = (len(pm) == len(tm)) and all(
            any(abs(p - t) < 0.3 for t in tm) for p in pm)
        verdict = "pass" if overlap >= 0.85 else "fail"
        rec = build_record(
            r,
            {"params": r["params"], "n_modes": len(pm),
             "mode_positions": [round(float(p), 3) for p in pm],
             "nll": float(r["log_reward"])},
            {"true_modes": [round(float(t), 3) for t in tm],
             "overlap": overlap, "mode_match": bool(mode_match),
             "verdict": verdict})
        st = writer.append(rec)
        n_new += st.get("ok", False) and not st.get("duplicate")
        n_dup += st.get("duplicate", False)
        n_err += not st.get("ok", False)
        n_pass += verdict == "pass"

    recs = writer.load()
    pass_rate = n_pass / len(recs) if recs else 0.0
    print("DB5 建库汇总：新增 %d、去重 %d、校验失败 %d（总 %d 条）"
          % (n_new, n_dup, n_err, len(recs)))
    print("通过率（覆盖率 ≥0.85）：%d/%d = %.1f%%"
          % (n_pass, len(recs), 100 * pass_rate))
    # §2.6.2 覆盖率审计（文档级淘汰条款）
    if pass_rate < 0.90:
        print("[审计] ⚠️ 通过率 < 90%——触发 MDN 淘汰条款（§2.6.2 第 1 条："
              "升级 CFM）")
    else:
        print("[审计] ✅ 通过率 ≥ 90%——MDN 管线联调阶段有效，暂不淘汰")
    fails = [r for r in recs if r["validation"]["verdict"] == "fail"]
    if fails:
        print("  fail 明细（前 5）：")
        for r in fails[:5]:
            print("    %s a=%s overlap=%.3f"
                  % (r["structure_id"],
                     r["decoded_design"]["params"].get("a"),
                     r["validation"]["overlap"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
