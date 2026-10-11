# -*- coding: utf-8 -*-
"""模拟/实验数据入库（缺口①，2026-10-02）。

把「定律·模拟·实验」三支中的**模拟/实验**支落成可运行管线：
  1. 生成/接收样本数据（v0：定律驱动采样 = 模拟数据；真实模拟器或实验
     数据的接入同格式：`db1/data/<system_id>.npz`，字段 samples/params）；
  2. 写数据条目 `db1/entries_data/<system_id>.json`（来源类型、样本量、
     参数、数据路径、红线 provenance）；
  3. 指纹由 `db2/estimate_fingerprint.py` 从**数据**估计（非解析），
     与解析真值对照的验证见 `db2/v13_data_ingest_check.py`。

红线：模拟/实验数据条目 = AI+verified、anchor_eligible 恒 false；
真实实验数据须人工复核后方可入库（与 DB1 条目同规则）。

用法（项目根目录）：
  python db1/ingest_sim_data.py            # 默认 3 个 landau 工况
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = Path(__file__).resolve().parent / "data"
ENTRIES_DIR = Path(__file__).resolve().parent / "entries_data"

# v0 工况：landau 三个参数点（b=kT=1 为结构常数）
CASES = [
    {"system_id": "sim_landau_a-2.0", "family": "landau",
     "params": {"a": -2.0, "b": 1.0, "kT": 1.0}, "n_samples": 30000},
    {"system_id": "sim_landau_a-0.5", "family": "landau",
     "params": {"a": -0.5, "b": 1.0, "kT": 1.0}, "n_samples": 30000},
    {"system_id": "sim_landau_a+0.5", "family": "landau",
     "params": {"a": 0.5, "b": 1.0, "kT": 1.0}, "n_samples": 30000},
]


def main() -> int:
    from db2.estimate_fingerprint import simulate_landau

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ENTRIES_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for case in CASES:
        sid = case["system_id"]
        pr = case["params"]
        samples = simulate_landau(pr["a"], pr["b"], pr["kT"],
                                  case["n_samples"], seed=0)
        np.savez_compressed(DATA_DIR / f"{sid}.npz", samples=samples,
                            params=np.array([pr["a"], pr["b"], pr["kT"]]))
        rec = {
            "system_id": sid,
            "family": case["family"],
            "source_type": "simulation",
            "simulator": "landau-grid-cdf-v0（定律驱动采样；真实模拟器/实验"
                         "数据同格式接入）",
            "params": pr,
            "n_samples": int(case["n_samples"]),
            "data_path": f"db1/data/{sid}.npz",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provenance": {"source": "AI+verified",
                           "anchor_eligible": False},
        }
        (ENTRIES_DIR / f"{sid}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1
        print(f"[ingest] {sid}: {case['n_samples']} 样本 → "
              f"db1/data/{sid}.npz + 条目")
    print(f"入库完成：{written} 条模拟数据条目（红线：AI+verified、"
          f"anchor 恒 false）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
