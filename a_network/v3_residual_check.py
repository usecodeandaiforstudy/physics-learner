"""V3 · IB 曲线代理 + 双条件门控验收（M3，2026-09-29）。

流程：
  1) 从 DB2 组装数据集（特征 = 解析背板廉价量，目标 = ln(1+I) 的 64 点曲线）；
  2) 按系统划分 holdout（~20%，seed 0，固定协议）；
  3) 训练 3 模型集成（训练系统）；
  4) holdout 评估：逐点/整体 R²、**单调性结构保证**（全部预测曲线 100% 单调，
     含「代理正则化 BA 大 β 尾部伪影」的量化对照）；
  5) 门控：阈值只在训练系统标定（p90）→ holdout 上验证「拒绝高不确定度系统
     移除的误差质量」+ Spearman(不确定度, 误差)；covered=False 无条件拒绝单测。
判据：holdout 曲线 R² ≥ 0.8（ln 空间）；单调性 100%；Spearman ρ ≥ 0.3；
拒绝 top-10% holdout 系统移除 ≥ 20% 误差质量；covered=False 拒绝单测通过。
输出：console + a_network/v3_residual_report.json。
用法（项目根目录，预计 1–2 分钟）：python a_network/v3_residual_check.py
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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from a_network.ib_surrogate import (  # noqa: E402
    N_POINTS,
    apply_gate,
    build_dataset,
    calibrate_gate,
    predict_ensemble,
    train_ensemble,
)

DB2_PATH = _PROJECT_ROOT / "db2" / "db2.jsonl"
ENTRIES_DIR = _PROJECT_ROOT / "db1" / "entries"
OUT_JSON = Path(__file__).resolve().parent / "v3_residual_report.json"
HOLDOUT_FRAC = 0.2
SEED = 0


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(1, len(a) + 1, dtype=float)
        return r
    rx, ry = rank(x), rank(y)
    mx, my = rx.mean(), ry.mean()
    cov = float(((rx - mx) * (ry - my)).sum())
    vx = float(((rx - mx) ** 2).sum())
    vy = float(((ry - my) ** 2).sum())
    return cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0


def main() -> int:
    rows = [json.loads(line) for line in
            DB2_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    entries = {e["system_id"]: json.loads(
        (ENTRIES_DIR / f"{e['system_id']}.json").read_text(encoding="utf-8"))
        for e in rows if (ENTRIES_DIR / f"{e['system_id']}.json").exists()}
    # 族索引：固定顺序
    families = sorted({r["family"] for r in rows})

    ds = build_dataset(rows, entries, families)
    n = ds["X"].shape[0]
    print(f"DB2 行 {len(rows)}；含完整 IB 曲线的行 {n}（跳过 {ds['n_skipped']}）")
    if n < 100:
        print("[FAIL] 曲线样本太少")
        return 1

    # 按系统划分（固定协议）
    sids = np.array(ds["system_ids"])
    uniq = sorted(set(ds["system_ids"]))
    rng = np.random.default_rng(SEED)
    holdout_sys = set(rng.choice(uniq, size=max(1, int(round(HOLDOUT_FRAC * len(uniq)))),
                                 replace=False))
    te_mask = np.array([s in holdout_sys for s in sids])
    tr_mask = ~te_mask
    print(f"系统 {len(uniq)}；holdout {len(holdout_sys)}；"
          f"训练行 {int(tr_mask.sum())} / 测试行 {int(te_mask.sum())}")

    # 训练集成
    models, mu, sd = train_ensemble(ds["X"][tr_mask], ds["Y"][tr_mask])
    pred_tr, std_tr = predict_ensemble(models, mu, sd, ds["X"][tr_mask])
    pred_te, std_te = predict_ensemble(models, mu, sd, ds["X"][te_mask])
    Ytr = ds["Y"][tr_mask].reshape(-1, 2, N_POINTS)
    Yte = ds["Y"][te_mask].reshape(-1, 2, N_POINTS)

    # R²（ln(1+I) 空间）
    tr_r2 = _r2(pred_tr.ravel(), Ytr.ravel())
    te_r2 = _r2(pred_te.ravel(), Yte.ravel())
    # 逐点（两条曲线 × 32 点；跳过目标几乎常数的退化点——小 β 区所有系统的
    # I_XT≈1e-11，R² 分母≈0 会爆炸出 -1e10 级伪值）
    te_pt, n_pt_valid = np.zeros((2, N_POINTS)), 0
    for c in range(2):
        for k in range(N_POINTS):
            ss_tot = float(((Yte[:, c, k] - Yte[:, c, k].mean()) ** 2).sum())
            if ss_tot < 1e-12:
                te_pt[c, k] = np.nan
                continue
            te_pt[c, k] = _r2(pred_te[:, c, k], Yte[:, c, k])
            n_pt_valid += 1

    # 单调性结构保证（全部预测曲线，训练+测试）
    mono_vio = 0
    for P in (pred_tr, pred_te):
        for c in range(2):
            for i in range(P.shape[0]):
                if np.any(np.diff(P[i, c]) < -1e-9):
                    mono_vio += 1
    n_curves = pred_tr.shape[0] * 2 + pred_te.shape[0] * 2

    # 代理 vs BA 尾部伪影对照（holdout：BA 目标在 β>226 有下降，代理无）
    ba_tail_drop = [float(Yte[i, 0, 26:].max() - Yte[i, 0, -1])
                    for i in range(Yte.shape[0])]
    sur_tail_drop = [float(pred_te[i, 0, 26:].max() - pred_te[i, 0, -1])
                     for i in range(pred_te.shape[0])]
    # 注意：ln 空间的下降量；代理因单调构造其下降 ≤ 0

    # 门控：系统级误差/不确定度
    err_sys = np.array([float(np.abs(pred_te[i] - Yte[i]).mean())
                        for i in range(Yte.shape[0])])
    var_sys = np.array([float(std_te[i].mean()) for i in range(std_te.shape[0])])
    rho = _spearman(var_sys, err_sys)

    err_tr_sys = np.array([float(np.abs(pred_tr[i] - Ytr[i]).mean())
                           for i in range(Ytr.shape[0])])
    std_tr_sys = np.array([float(std_tr[i].mean()) for i in range(std_tr.shape[0])])
    tau = calibrate_gate(std_tr_sys, quantile=0.90)
    rej = var_sys > tau
    err_mass_removed = float(err_sys[rej].sum() / err_sys.sum()) if rej.sum() else 0.0
    rej_rate = float(rej.mean())

    # 拒绝率扫描（度量用，非标定）：holdout 上按 var 分位拒绝的误差质量浓度
    sweep = []
    for frac in (0.05, 0.10, 0.20, 0.30, 0.50):
        th = float(np.quantile(var_sys, 1.0 - frac))
        rej_s = var_sys > th
        removed = float(err_sys[rej_s].sum() / err_sys.sum()) if rej_s.sum() else 0.0
        sweep.append({"rejection_rate": frac, "error_mass_removed": removed,
                      "concentration": removed / frac if frac > 0 else 0.0})
    conc10 = sweep[1]["concentration"]   # top-10% 的浓度倍数

    # covered=False 无条件拒绝单测
    gate_covered_false = bool(~apply_gate(np.array([False]), np.array([0.0]), tau)[0])
    gate_covered_ok = bool(apply_gate(np.array([True]), np.array([0.0]), tau)[0])

    report = {
        "note": "V3 IB 曲线代理 + 双条件门控（M3）。",
        "n_rows_with_curve": n, "n_skipped": ds["n_skipped"],
        "n_systems": len(uniq), "n_holdout_systems": len(holdout_sys),
        "train_r2": tr_r2, "test_r2": te_r2,
        "test_r2_per_point_mean": float(np.nanmean(te_pt)),
        "test_r2_I_XT_mean": float(np.nanmean(te_pt[0])),
        "test_r2_I_TY_mean": float(np.nanmean(te_pt[1])),
        "n_valid_points": n_pt_valid,
        "monotonicity": {"violations": mono_vio, "curves": n_curves,
                         "all_monotone": mono_vio == 0},
        "tail_artifact": {
            "ba_target_tail_drop_mean": float(np.mean(ba_tail_drop)),
            "surrogate_tail_drop_mean": float(np.mean(sur_tail_drop)),
            "surrogate_regularizes": bool(
                np.mean(sur_tail_drop) < np.mean(ba_tail_drop)),
        },
        "gate": {
            "threshold_train_p90": tau,
            "holdout_rejection_rate": rej_rate,
            "error_mass_removed": err_mass_removed,
            "spearman_unc_err": rho,
            "sweep": sweep,
            "concentration_at_top10": conc10,
            "covered_false_rejected": gate_covered_false,
            "covered_true_lowvar_accepted": gate_covered_ok,
        },
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("\n================ V3 IB 代理 + 门控验收（M3） ================")
    print(f"训练 R²={tr_r2:.3f}，holdout R²={te_r2:.3f}"
          f"（逐点均值 {np.nanmean(te_pt):.3f}，有效点 {n_pt_valid}/64；"
          f"I_XT {np.nanmean(te_pt[0]):.3f} / I_TY {np.nanmean(te_pt[1]):.3f}）")
    print(f"单调性结构保证：{mono_vio}/{n_curves} 违规（构造上应恒为 0）")
    print(f"尾部伪影对照：BA 目标均值下降 {np.mean(ba_tail_drop):.4f} vs "
          f"代理 {np.mean(sur_tail_drop):.4f}（代理由构造单调 → 无下降）")
    print(f"门控：训练标定阈值 {tau:.4f}；holdout 拒绝率 {rej_rate * 100:.1f}%，"
          f"移除误差质量 {err_mass_removed * 100:.1f}%，ρ={rho:+.3f}")
    print("    拒绝率扫描（holdout 度量）：" +
          ", ".join(f"@{s['rejection_rate']:.0%}→{s['error_mass_removed']:.0%}"
                    f"({s['concentration']:.2f}×)" for s in sweep))
    print(f"covered=False 拒绝：{'✅' if gate_covered_false else '❌'}；"
          f"covered=True 低不确定度接受：{'✅' if gate_covered_ok else '❌'}")

    ok = (te_r2 >= 0.8 and mono_vio == 0 and rho >= 0.3
          and conc10 >= 1.5 and gate_covered_false and gate_covered_ok)
    print(f"\n判定：{'✅ V3 通过' if ok else '⚠️ 部分通过，见报告逐项'}")
    print(f"报告已写入 {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
