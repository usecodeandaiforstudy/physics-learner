"""D2.5 · landau 编码可学性探针（隔离实验，秒级，无训练管线）。

背景：容量对照（D2）的结果被两个混杂因素污染——
  ① 共享列的标准化尺度（原值 a 标准化后跨度 ±7.7，优化变难）；
  ② 多任务训练的轨迹方差（同一 setup 的 per-family train R² 摆动巨大，
     geometric 三次轨迹 0.95/0.49/0.22）。
本探针把问题缩小到纯数学断言：**编码非单射 → 不可学；单射 → 可学**。
不做 GINE、不做共享列、不做多任务——只用 30 个唯一点训小 MLP。

做法：a ∈ linspace(-2,2,30)（与 DB1 landau 扫描网格一致），目标
y = ln f(a)（f = 4kTb·I，见 diagnose_landau_shape.py；b=kT=1）。
两种编码对比：
  - mixed（现役）：x = ln(a) if a>0 else a —— a<0 支 ∈[-2,0)、a>0 支 ∈[-2.67,0.69)，
    两支在 [-2,0) 重叠 → 同一 x 对应两个目标（如 x=-0.5 ↔ a=-0.5 或 a=e^-0.5）；
  - raw（R2b）：x = a —— 连续单射。
各 3 seeds × 3000 步 Adam(lr=1e-2) 全批量，报 train R²。

预期：mixed 显著低于 raw（重叠区约 1/3 点歧义）；raw ≈ 0.99+。
若成立 → 「编码非单射是 landau 样本内不可学的充分原因」成立，
R2b 的修复方向正确（D2 容量实验的低分是混杂，不是修复无效）。

输出：console + a_network/landau_probe_report.json。
用法：python a_network/diagnose_landau_probe.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.fisher import fisher_landau  # noqa: E402

_A_DIR = Path(__file__).resolve().parent
OUT_JSON = _A_DIR / "landau_probe_report.json"

A_GRID = np.linspace(-2.0, 2.0, 30)
SEEDS = [0, 1, 2]
STEPS = 3000
LR = 1e-2


def enc_mixed(a: np.ndarray) -> np.ndarray:
    return np.array([math.log(v) if v > 0 else v for v in a])


def enc_raw(a: np.ndarray) -> np.ndarray:
    return a.copy()


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def train_probe(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = nn.Sequential(
        nn.Linear(1, 16), nn.Tanh(),
        nn.Linear(16, 16), nn.Tanh(),
        nn.Linear(16, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    xt = torch.tensor(x[:, None], dtype=torch.float32)
    yt = torch.tensor(y[:, None], dtype=torch.float32)
    for _ in range(STEPS):
        opt.zero_grad()
        loss = loss_fn(net(xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(xt).numpy().ravel()
    return _r2(pred, y)


def main() -> int:
    y = np.log(4.0 * np.array([fisher_landau(float(a), 1.0, 1.0) for a in A_GRID]))

    # 重叠区统计：mixed 编码下落在 [-2,0) 的点中，有多少既可能来自 a<0 支也可能来自 a>0 支
    xm = enc_mixed(A_GRID)
    pos_log = np.array([math.log(a) for a in A_GRID if a > 0])
    overlap_zone = (pos_log > -2.0) & (pos_log < 0.0)
    n_pos_in_overlap = int(overlap_zone.sum())
    n_neg_in_overlap = int(np.sum((A_GRID < 0) & (A_GRID > -2.0)))

    results: dict = {"encodings": {}, "grid_n": int(len(A_GRID)),
                     "steps": STEPS, "lr": LR, "seeds": SEEDS,
                     "overlap": {
                         "n_neg_branch_in_zone": n_neg_in_overlap,
                         "n_pos_branch_in_zone": n_pos_in_overlap,
                         "note": "mixed 编码下这两支的特征值落在同一区间 [-2,0)："
                                 "同一 x 对应两个不同目标（输入歧义）",
                     }}
    for name, enc in [("mixed", enc_mixed), ("raw", enc_raw)]:
        x = enc(A_GRID)
        r2s = [train_probe(x, y, s) for s in SEEDS]
        results["encodings"][name] = {
            "r2_per_seed": r2s,
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
        }

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("================ landau 编码可学性探针（D2.5） ================")
    ov = results["overlap"]
    print(f"重叠区统计：负支 {ov['n_neg_branch_in_zone']} 点、正支 "
          f"{ov['n_pos_branch_in_zone']} 点的特征值落在同一区间（非单射）")
    for name in ("mixed", "raw"):
        r = results["encodings"][name]
        print(f"{name:6s} train R² = {r['r2_mean']:.3f} ± {r['r2_std']:.3f}  "
              f"(seeds={[round(v, 3) for v in r['r2_per_seed']]})")
    mm = results["encodings"]["mixed"]["r2_mean"]
    rr = results["encodings"]["raw"]["r2_mean"]
    if rr >= 0.95 and mm < rr - 0.2:
        print("判定：✅ 编码非单射是 landau 样本内不可学的充分原因；R2b 方向正确")
    elif rr - mm > 0.05:
        print("判定：⚠️ raw 更优但差距未达预期——需查是否还有其他混杂")
    else:
        print("判定：❌ 编码假设不成立，需重新诊断")
    print(f"\n报告已写入 {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
