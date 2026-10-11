# -*- coding: utf-8 -*-
"""等离子体领域实例验证（第 68 条，2026-10-02）——×N 新增流程走通。

结构图示例域「等离子体」的落地检查（DB1 → DB2 → B → C → DB4/DB5 全链）：
  P1 DB1：4 个等离子体诊断系统条目（红线 provenance）；
  P2 DB2：这些系统的信息指纹已入库；
  P3 B 编码：等离子体需求（DB3-0035）被 B v0 解析器正确结构化；
  P4 C 解码器实例：为等离子体湍流系统（rayleigh 族）训练领域解码器，
     与解析真值验证 overlap ≥ 0.85 → 写 DB4 + DB5（domain=plasma）。

用法（项目根目录，先跑入库）：
  python db1/batch_generate_ingest.py --domain plasma --skip-existing
  python c_network/v16_plasma_check.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from c_network.mdn import MixtureDensityNetwork

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
GRID = np.linspace(0.0, 8.0, 2000)
SIGMA = 1.2


def check(name: str, ok: bool, detail: str = "") -> int:
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name,
                             "" if ok else detail))
    return 0 if ok else 1


def rayleigh_pdf(x: np.ndarray, sigma: float) -> np.ndarray:
    return (x / sigma ** 2) * np.exp(-x ** 2 / (2 * sigma ** 2))


def sample_rayleigh(sigma: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return sigma * np.sqrt(-2.0 * np.log(rng.uniform(1e-12, 1.0, n)))


def main() -> int:
    fails = 0
    torch.manual_seed(0)

    # ---- P1：DB1 等离子体条目 ----
    entries = []
    for p in sorted((ROOT / "db1" / "entries").glob("plasma_*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        # DB1 条目的红线字段 = source + human_review_status（anchor_eligible
        # 是 DB3/DB4/DB5 的规则，DB1 条目本身不参与锚点）
        prov = rec["provenance"]
        ok_prov = (prov.get("source") == "AI+verified"
                   and prov.get("human_review_status") is not None)
        entries.append((rec, ok_prov))
    print(f"  P1 DB1 等离子体系统：{[e[0]['system_id'] for e in entries]}")
    fails += check(f"P1 DB1 领域实例条目（{len(entries)} 个 ≥ 4，红线合规）",
                   len(entries) >= 4 and all(o for _, o in entries))

    # ---- P2：DB2 指纹 ----
    ids = {e[0]["system_id"] for e in entries}
    n_rows = 0
    db2 = ROOT / "db2" / "db2.jsonl"
    if db2.exists():
        for line in db2.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("system_id") in ids:
                    n_rows += 1
    print(f"  P2 DB2 指纹记录：{n_rows} 条（{len(ids)} 个系统）")
    fails += check(f"P2 DB2 领域指纹入库（{n_rows} 条 ≥ {len(ids) * 10}）",
                   n_rows >= len(ids) * 10, f"rows={n_rows}")

    # ---- P3：B 编码等离子体需求 ----
    from b_network.requirement_parser import RequirementParser
    pair = json.loads((ROOT / "db3" / "pairs" / "DB3-0035.json")
                      .read_text(encoding="utf-8"))
    req = RequirementParser().parse(pair["requirement_text"])
    print(f"  P3 B 解析：property={req.property_ids}、direction={req.direction}、"
          f"target={dict(req.target_value)}、unmapped={req.unmapped}")
    fails += check("P3 B 编码等离子体需求（precision 命中且 σ_T=5 K）",
                   "precision" in req.property_ids
                   and any(abs(v - 5.0) < 1e-6
                           for v in req.target_value.values()))

    # ---- P4：C 领域解码器实例（等离子体湍流：rayleigh）----
    x = sample_rayleigh(SIGMA, 30000, seed=0)
    mdn = MixtureDensityNetwork(1, n_components=8)
    opt = torch.optim.Adam(mdn.parameters(), lr=3e-3)
    X = np.full_like(x, SIGMA, dtype=np.float32)[:, None]
    Y = x.astype(np.float32)[:, None]
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for step in range(6000):
        idx = np.random.randint(0, len(Xt), 512)
        loss = mdn.nll(Xt[idx], Yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    p_pred = mdn.density(SIGMA, GRID)
    p_true = rayleigh_pdf(GRID, SIGMA)
    ov = float(np.trapezoid(np.minimum(p_pred, p_true), GRID))
    modes = mdn.modes(SIGMA, GRID)
    mode_match = any(abs(m - SIGMA) < 0.3 for m in modes)
    verdict = "pass" if ov >= 0.85 else "fail"
    print(f"  P4 解码器实例（plasma/rayleigh）：overlap={ov:.4f}、"
          f"预测峰={[round(m, 3) for m in modes]}、真值峰={SIGMA}")
    fails += check(f"P4 C 解码验证（overlap {ov:.4f} ≥ 0.85 且模式匹配）",
                   verdict == "pass" and mode_match)

    # ---- 写 DB4 + DB5（domain=plasma）----
    from db4.db4_writer import DB4Writer, build_record
    from db5.db5_writer import DB5Writer, build_record as build5
    r4 = build_record("rayleigh", {"sigma": SIGMA}, 4.0 / SIGMA ** 2, 1.0,
                      0.0, "domain-instance-plasma", "plasma_turbulence")
    s4 = DB4Writer().append(r4)
    r5 = build5(r4,
                {"params": {"sigma": SIGMA},
                 "n_modes": len(modes),
                 "mode_positions": [round(float(m), 3) for m in modes],
                 "nll": float(ov)},
                {"true_modes": [SIGMA], "overlap": ov,
                 "mode_match": bool(mode_match), "verdict": verdict})
    s5 = DB5Writer().append(r5)
    print(f"  DB4 写入：{s4}；DB5 写入：{s5}")
    fails += check("P4b DB4/DB5 领域记录落库（domain=plasma/rayleigh）",
                   s4.get("ok") and s5.get("ok"))

    print("\n%s（%d 项失败）" % ("===== 全部通过 =====" if not fails
                              else "===== 有失败 =====", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
