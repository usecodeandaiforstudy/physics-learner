"""D2 · landau 容量对照（诊断方案 D2，1 次训练 ~4–5 分钟）。

问题：landau 在训练集内时模型能拟合吗？（区分「容量不足」vs「外推失败」）
做法：固定 holdout 选**不含 landau** 的 5 族，让 landau 进入训练集；
跑 base 300ep seed 0；读 train 逐族 R²（train_gine 的 metrics 里 train 也带
per_family）。顺带看 geometric 的 train R² 与 holdout 各族是否被拖累。

判读（提前说好的标准）：
- landau train R² ≈ 0.9+ → 容量足够，失败纯在外推（形状缺口/平膝曲率是主犯）；
- landau train R² 明显低于其他族 → 混合编码（a=0 特征跳变）在样本内就咬人，
  R2 编码修复升为第一优先级。

输出：console + `a_network/landau_capacity_report.json`；train_gine 的无后缀
指标文件会被立即归档为 `gine_metrics_family_base_capacity_run.json`
（第 28 条教训：临时测速必须换 tag 或备份）。

用法（项目根目录）：
    python a_network/diagnose_landau_capacity.py                    # 基线（旧混合编码）
    python a_network/diagnose_landau_capacity.py --raw-sign-params  # R2b 编码修复验证
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_A_DIR = Path(__file__).resolve().parent
TRAIN_PY = _A_DIR / "train_gine.py"
OUT_JSON = _A_DIR / "landau_capacity_report.json"
ARCHIVE = _A_DIR / "gine_metrics_family_base_capacity_run.json"

# 5 族 holdout 不含 landau（与第 29 条 holdout 同规模、同链族构成）
HOLDOUT_NO_LANDAU = ["beta", "gamma", "laplace", "negbin", "ou"]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="landau 容量对照（D2）")
    ap.add_argument("--raw-sign-params", action="store_true",
                    help="R2b：透传 train_gine 的 --raw_sign_params（编码修复验证）")
    args = ap.parse_args()

    cmd = [sys.executable, str(TRAIN_PY), "--split", "family", "--epochs", "300",
           "--seed", "0", "--holdout", ",".join(HOLDOUT_NO_LANDAU)]
    if args.raw_sign_params:
        cmd.append("--raw_sign_params")
    print(f"[run] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=_PROJECT_ROOT)   # 输出直接透传（含 UTF-8 reconfigure）
    wall = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL] rc={proc.returncode}")
        return 1

    src = _A_DIR / "gine_metrics_family_base.json"
    if not src.exists():
        print(f"[FAIL] 未找到 {src.name}")
        return 1
    m = json.loads(src.read_text(encoding="utf-8"))
    ARCHIVE.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    tr_pf = m["train"]["per_family"]
    te_pf = m["test"]["per_family"]
    landau_tr = tr_pf.get("landau")
    geo_tr = tr_pf.get("geometric")
    others = [v["r2_trace"] for k, v in tr_pf.items() if k != "landau"]
    other_mean = statistics.mean(others) if others else None

    report = {
        "note": "landau 容量对照（D2）：holdout 不含 landau，landau 进训练集。",
        "holdout": m["holdout"],
        "raw_sign_params": args.raw_sign_params,
        "wall_seconds": round(wall, 1),
        "metrics_archive": ARCHIVE.name,
        "landau_train": landau_tr,
        "geometric_train": geo_tr,
        "train_others_trace_mean": other_mean,
        "test_overall": {"r2_trace": m["test"]["r2_trace"],
                         "r2_det": m["test"]["r2_det"]},
        "test_per_family": te_pf,
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("\n================ landau 容量对照（D2） ================")
    if landau_tr:
        lt, ld = landau_tr["r2_trace"], landau_tr["r2_det"]
        print(f"landau 训练内 R²：trace={lt:.3f}  det={ld:.3f}")
        if lt >= 0.85:
            verdict = ("✅ 容量足够 → landau 失败纯在外推（训练集无平膝形状可借，"
                       "D1.2 的「形状缺口」是主犯）")
        elif lt >= 0.5:
            verdict = ("⚠️ 部分拟合 → 容量 + 外推混合病因；混合编码（D1.3）在样本内"
                       "已造成明显困难，R2 编码修复优先级上升")
        else:
            verdict = ("❌ 训练内都拟合不好 → 混合编码（a=0 特征跳变）/目标形状是"
                       "样本内问题，不只是 OOD；R2 编码修复升为第一优先级")
        print(f"    判定：{verdict}")
    else:
        print("[FAIL] train per_family 里没有 landau——检查 holdout 是否写对")
    if geo_tr:
        print(f"geometric 训练内 R²：trace={geo_tr['r2_trace']:.3f}  "
              f"det={geo_tr['r2_det']:.3f}（对照组，单调左支）")
    if other_mean is not None:
        print(f"其他训练族 trace 均值：{other_mean:.3f}（landau 是否低于此值 0.1+ 是判据）")
    print(f"holdout 各族（观察是否被拖累）：" +
          ", ".join(f"{k}={v['r2_trace']:+.2f}" for k, v in sorted(te_pf.items())))
    print(f"\n报告已写入 {OUT_JSON.name}；指标归档 {ARCHIVE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
