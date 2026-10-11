"""R5 · 不确定度拒绝门控实验（离线标定，零训练）。

输入：a_network/landau_pred_dump.jsonl（D3 取证 run 的测试集逐节点 dump：
族/系统/参数/真值/预测/pred_var；fixed-holdout {landau,negbin,ou,gamma,laplace}，
base seed 0，1440 节点）。
问题：用不确定度头的 pred_var 做「低置信拒绝」门控，能否把灾难性错误挡在
奖励之外（A v2 环境里 RL 只应消费高置信分量）？三个子问题：
  R5a 排序质量：pred_var 与实际误差 |pred−true| 的 Spearman 相关；
  R5b 门控收益：按 var 分位拒绝后，接受集 R² 的提升曲线 + 误差质量捕获率；
  R5c 分族表现：landau 是否被优先拒绝（Q3c 的定量化）。
判据（提前说好）：
  - Spearman ρ ≥ 0.3 且「拒绝 top-10% var 节点」移除 ≥30% 总误差质量
    → 门控有信息量，值得进 A v2；
  - 否则 → 不确定度头需在 OOD 数据上重训/重标定（记录结论，不硬上）。
注意：pred_var 是 log 方差（NLL 头输出，越「不负责」越高）。标定结果只对本
模型版本（base seed 0）有效；A v2 换模型后需重标定（脚本可复用）。

输出：console + a_network/r5_rejection_report.json。
用法：python a_network/r5_rejection_gate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

_A_DIR = Path(__file__).resolve().parent
DUMP = _A_DIR / "landau_pred_dump.jsonl"
OUT_JSON = _A_DIR / "r5_rejection_report.json"

QUANTILES = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相关（不用 scipy，避免依赖；秩用 argsort 二次实现）。"""
    def rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(a) + 1, dtype=float)
        return ranks

    rx, ry = rank(x), rank(y)
    n = len(x)
    # 均值秩差积相关
    mx, my = rx.mean(), ry.mean()
    cov = float(((rx - mx) * (ry - my)).sum())
    vx = float(((rx - mx) ** 2).sum())
    vy = float(((ry - my) ** 2).sum())
    return cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0


def main() -> int:
    if not DUMP.exists():
        print(f"[FAIL] 未找到 {DUMP.name}——先跑 "
              f"`python a_network/diagnose_landau_forensics.py` 生成 dump")
        return 1

    rows = [json.loads(line) for line in
            DUMP.read_text(encoding="utf-8").splitlines() if line.strip()]
    fam = np.array([r["family"] for r in rows])
    true = np.array([r["true_ln_trace"] for r in rows])
    pred = np.array([r["pred_ln_trace"] for r in rows])
    var = np.array([r["pred_var"] for r in rows])
    err = np.abs(pred - true)

    # R5a：排序质量
    rho = _spearman(var, err)

    # R5b：门控收益（全局阈值扫分位）
    base_r2 = _r2(pred, true)
    sweep = []
    for q in QUANTILES:
        th = float(np.quantile(var, q))
        keep = var <= th
        if keep.sum() < 10:
            continue
        rej = ~keep
        sweep.append({
            "quantile": q,
            "var_threshold": th,
            "rejection_rate": float(rej.mean()),
            "accepted_r2_trace": _r2(pred[keep], true[keep]),
            "error_mass_removed": float(err[rej].sum() / err.sum()),
        })

    # R5c：分族（用全局 90 分位阈值，展示「谁被优先拒绝」）
    th90 = float(np.quantile(var, 0.90))
    per_family = {}
    rej_all = var > th90
    for f in sorted(set(fam)):
        m = fam == f
        keep = m & ~rej_all
        per_family[f] = {
            "n_nodes": int(m.sum()),
            "mean_var": float(var[m].mean()),
            "rejection_rate_at_p90": float(rej_all[m].mean()),
            "r2_all": _r2(pred[m], true[m]),
            "r2_accepted": _r2(pred[keep], true[keep])
            if keep.sum() >= 10 else None,
        }

    p90 = sweep[-3] if len(sweep) >= 8 else sweep[-1]
    report = {
        "note": "R5 门控标定：pred_var（log 方差）作为低置信拒绝信号。",
        "n_nodes": int(len(rows)),
        "base_r2_trace": base_r2,
        "r5a_spearman_var_err": rho,
        "r5b_sweep": sweep,
        "r5c_per_family_at_p90": per_family,
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("================ R5 不确定度拒绝门控标定 ================")
    print(f"节点数 {len(rows)}，基线 R²(trace) = {base_r2:+.3f}")
    print(f"\n[R5a 排序质量] Spearman(pred_var, |err|) = {rho:+.3f}"
          f"（≥0.3 及格）")
    print(f"\n[R5b 门控收益] 按 var 分位拒绝：")
    for s in sweep:
        print(f"    拒绝 top {int(round((1 - s['quantile']) * 100)):>2}%："
              f"接受集 R² {s['accepted_r2_trace']:+8.3f}  "
              f"误差质量移除 {s['error_mass_removed'] * 100:5.1f}%")
    print(f"\n[R5c 分族（全局 p90 阈值 var>{th90:.2f}）]")
    for f, v in per_family.items():
        acc = f"{v['r2_accepted']:+.2f}" if v["r2_accepted"] is not None else "  —"
        print(f"    {f:10s} n={v['n_nodes']:4d}  mean_var={v['mean_var']:+.2f}  "
              f"拒绝率={v['rejection_rate_at_p90'] * 100:5.1f}%  "
              f"R² {v['r2_all']:+7.2f} → 接受集 {acc}")

    ok_a = rho >= 0.3
    ok_b = (p90["error_mass_removed"] >= 0.30)
    if ok_a and ok_b:
        print("\n判定：✅ 门控有信息量，值得进 A v2（拒绝 top-10% 移除 "
              f"{p90['error_mass_removed'] * 100:.0f}% 误差质量，ρ={rho:+.2f}）")
    elif ok_a or ok_b:
        print("\n判定：⚠️ 部分通过（ρ=" + f"{rho:+.2f}" + "，移除率=" +
              f"{p90['error_mass_removed'] * 100:.0f}%），门控可用但需更精细标定")
    else:
        print("\n判定：❌ 不确定度头未提供有效排序，需在 OOD 数据上重训/重标定")
    print(f"\n报告已写入 {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
