"""D3 · landau 预测取证（诊断方案 D3，1 次训练 ~5 分钟 + 秒级分析）。

做法：base 300ep seed 0、固定 holdout {landau,negbin,ou,gamma,laplace}、
`--dump_preds` 把测试集逐节点 (族, 系统, 参数, 真值, 预测, 不确定度) 写为 JSONL；
随后对 landau 的 90 个节点做取证分析，回答：
- Q3a 借道检验：预测的 log-log 斜率是否恒定（≈ 借了 -2/-1 直线）？真值是
  平膝 + 弯曲尾（斜率 -1.4 → ~0，见 D1.2）；
- Q3b 分支检验：a<0 / a>0 两侧预测是否被当同一函数（模型是否把编码跳变抹平）？
- Q3c 不确定度：landau 的 pred_var 是否显著高于其他 holdout 族（R5 拒绝兜底
  的可行性证据）。

复现自检：本 run 的 landau test R² 应逐位等于第 29 条 base seed 0 的
（trace -30.413 / det -2.243）——同 seed、同 holdout、同代码路径。

输出：console + `a_network/landau_forensics_report.json`；
指标归档 `gine_metrics_family_base_landau_dump_run.json`。

用法（项目根目录）：
    python a_network/diagnose_landau_forensics.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_A_DIR = Path(__file__).resolve().parent
TRAIN_PY = _A_DIR / "train_gine.py"
OUT_JSON = _A_DIR / "landau_forensics_report.json"
ARCHIVE = _A_DIR / "gine_metrics_family_base_landau_dump_run.json"
DUMP = _A_DIR / "landau_pred_dump.jsonl"

HOLDOUT = ["landau", "negbin", "ou", "gamma", "laplace"]   # 第 29 条固定 holdout


def _slopes(a_pos: np.ndarray, y: np.ndarray) -> dict:
    """正侧 (物理 a>0) 的 log-log 斜率序列：d y / d ln a。"""
    order = np.argsort(a_pos)
    a, yv = a_pos[order], y[order]
    ln_a = np.log(a)
    s = np.gradient(yv, ln_a)
    return {"slope_min": float(s.min()), "slope_max": float(s.max()),
            "slope_std": float(s.std()), "slopes": [float(x) for x in s]}


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def analyze(dump_path: Path) -> dict:
    rows = [json.loads(line) for line in dump_path.read_text(encoding="utf-8")
            .splitlines() if line.strip()]
    land = [r for r in rows if r["family"] == "landau"]
    if not land:
        raise RuntimeError("dump 中没有 landau 样本")

    # 用物理扫描值分正负支（dump 的 params 是编码后特征，a>0 侧已被取 log，不能直接分）
    # 3 个系统的扫描网格相同 → 同一 scan_value 有 3 条重复记录；按 scan_value
    # 取均值去重后再算斜率（否则 np.gradient 遇 dx=0 产生 NaN）。
    by_a: dict[float, list[dict]] = {}
    for r in land:
        by_a.setdefault(r["scan_value"], []).append(r)
    a = np.array(sorted(by_a))
    true = np.array([np.mean([r["true_ln_trace"] for r in by_a[x]]) for x in a])
    pred = np.array([np.mean([r["pred_ln_trace"] for r in by_a[x]]) for x in a])
    var = np.array([np.mean([r["pred_var"] for r in by_a[x]]) for x in a])

    pos = a > 0
    neg = ~pos

    report: dict = {
        "n_landau_nodes": int(len(land)),
        "r2_landau_trace": _r2(pred, true),      # 复现自检：应 ≈ -30.413（第 29 条 seed 0）
        "true_pos_side": _slopes(a[pos], true[pos]),
        "pred_pos_side": _slopes(a[pos], pred[pos]),
        "neg_side": {
            # 负侧特征编码是原始 a，直接看 d pred / da 与 d true / da
            "true_slope_da": float(np.gradient(true[neg], a[neg]).mean()),
            "pred_slope_da": float(np.gradient(pred[neg], a[neg]).mean()),
        },
        "branch_symmetry": {
            # 模型是否把两侧当同一函数：比较两段预测的整体水平差
            "true_mean_gap": float(true[pos].mean() - true[neg].mean()),
            "pred_mean_gap": float(pred[pos].mean() - pred[neg].mean()),
        },
        "uncertainty": {
            "landau_pred_var_mean": float(var.mean()),
            "other_families_pred_var_mean": float(np.mean(
                [r["pred_var"] for r in rows if r["family"] != "landau"])),
        },
    }
    # 判定
    tp = report["true_pos_side"]; pp = report["pred_pos_side"]
    report["q3a_borrowed_line"] = pp["slope_std"] < tp["slope_std"] * 0.5
    report["q3b_branches_collapsed"] = (
        abs(report["branch_symmetry"]["pred_mean_gap"]
            - report["branch_symmetry"]["true_mean_gap"]) > 0.5)
    report["q3c_var_elevated"] = (
        report["uncertainty"]["landau_pred_var_mean"]
        > report["uncertainty"]["other_families_pred_var_mean"] * 1.5)
    return report


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="landau 预测取证（D3）")
    ap.add_argument("--analyze-only", action="store_true",
                    help="跳过训练，直接分析已有的 landau_pred_dump.jsonl")
    args = ap.parse_args()

    wall: float | None = None
    if args.analyze_only:
        if not DUMP.exists():
            print(f"[FAIL] {DUMP.name} 不存在，先跑一次完整流程")
            return 1
        print(f"[analyze] 直接分析 {DUMP.name}", flush=True)
    else:
        cmd = [sys.executable, str(TRAIN_PY), "--split", "family", "--epochs", "300",
               "--seed", "0", "--holdout", ",".join(HOLDOUT),
               "--dump_preds", str(DUMP)]
        print(f"[run] {' '.join(cmd)}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=_PROJECT_ROOT)   # 输出透传
        wall = time.time() - t0
        if proc.returncode != 0 or not DUMP.exists():
            print(f"[FAIL] rc={proc.returncode}，dump 缺失")
            return 1
        # 归档无后缀指标（第 28 条教训）
        src = _A_DIR / "gine_metrics_family_base.json"
        if src.exists():
            ARCHIVE.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    src = _A_DIR / "gine_metrics_family_base.json"
    m = json.loads(src.read_text(encoding="utf-8"))
    te_landau = m["test"]["per_family"].get("landau", {})

    report = analyze(DUMP)
    report["wall_seconds"] = round(wall, 1) if wall is not None else None
    report["expected_seed0_landau"] = {"r2_trace": -30.413, "r2_det": -2.243,
                                       "note": "第 29 条 base seed 0 记录（复现自检）"}
    report["actual_metrics_landau"] = te_landau
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("\n================ landau 预测取证（D3） ================")
    print(f"复现自检：本次 landau test R² trace={report['r2_landau_trace']:.3f}"
          f"（第 29 条 seed 0 = {te_landau.get('r2_trace', float('nan')):.3f}，"
          f"应逐位一致）")
    tp, pp = report["true_pos_side"], report["pred_pos_side"]
    print(f"\n[Q3a 借道检验] 正侧 log-log 斜率（真值 vs 预测）")
    print(f"    真值：斜率范围 [{tp['slope_min']:+.2f}, {tp['slope_max']:+.2f}]，"
          f"std={tp['slope_std']:.3f}（平膝+弯曲尾，强变化）")
    print(f"    预测：斜率范围 [{pp['slope_min']:+.2f}, {pp['slope_max']:+.2f}]，"
          f"std={pp['slope_std']:.3f}")
    print(f"    → {'预测斜率接近恒定 = 借了直线（形状缺口坐实）'
              if report['q3a_borrowed_line'] else '预测斜率也变化（借道假设不成立，需再查）'}")
    nb = report["neg_side"]
    print(f"\n[Q3b 分支检验] 负侧 d(ln f)/da：真值 {nb['true_slope_da']:+.3f} / "
          f"预测 {nb['pred_slope_da']:+.3f}")
    bs = report["branch_symmetry"]
    print(f"    两侧水平差：真值 {bs['true_mean_gap']:+.3f} / "
          f"预测 {bs['pred_mean_gap']:+.3f}")
    print(f"    → {'模型把两侧当同一函数（编码跳变被抹平）'
              if report['q3b_branches_collapsed'] else '两侧预测差异与真值接近'}")
    uc = report["uncertainty"]
    print(f"\n[Q3c 不确定度] landau pred_var 均值 {uc['landau_pred_var_mean']:.3f} vs "
          f"其他族 {uc['other_families_pred_var_mean']:.3f}")
    print(f"    → {'landau 不确定度显著偏高：R5 拒绝兜底可行'
              if report['q3c_var_elevated'] else '不确定度未区分 landau：R5 需重新设计'}")
    print(f"\n报告已写入 {OUT_JSON.name}；指标归档 {ARCHIVE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
