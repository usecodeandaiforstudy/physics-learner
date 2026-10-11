# -*- coding: utf-8 -*-
"""DB4 写入器（v0，2026-10-02）：追加 / 校验 / 幂等去重 / 读取。

红线：全部记录 source=AI+verified、anchor_eligible=false；校验规则见
db4.schema.md。structure_id = db4- + sha1(family∥sorted(params)∥agent) 前 8 位。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["make_structure_id", "validate_record", "DB4Writer"]

_DB4_PATH = Path(__file__).resolve().parent / "db4.jsonl"
KNOWN_DOMAINS = {
    "landau", "elastic", "two_level", "gaussian_mixture2",
    "truncated_geometric", "inverse_gamma", "invchisq", "frechet",
    "ou", "geometric", "negbin", "chisq", "gamma", "beta", "weibull",
    "vonmises", "cauchy", "rayleigh", "logistic", "lognormal",
    "poisson_count", "radioactive_decay", "harmonic_oscillator",
    "laplace", "lc_circuit", "diffusion", "truncated_gaussian",
    "wrapped_normal", "rician",
}


def make_structure_id(domain: str, params: dict, agent: str) -> str:
    key = domain + "|" + json.dumps(params, sort_keys=True) + "|" + agent
    return "db4-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def validate_record(rec: dict) -> list[str]:
    errs: list[str] = []
    for k in ("structure_id", "domain", "params", "q", "reward",
              "log_reward", "info_action", "agent", "policy_tag",
              "timestamp", "provenance"):
        if k not in rec:
            errs.append(f"缺字段 {k}")
    if rec.get("domain") not in KNOWN_DOMAINS:
        errs.append(f"domain 未知：{rec.get('domain')!r}")
    if not isinstance(rec.get("params"), dict) or not rec.get("params"):
        errs.append("params 须为非空 dict")
    q = rec.get("q", {})
    if not isinstance(q.get("I_TY"), (int, float)):
        errs.append("q.I_TY 须为数值")
    r = rec.get("reward")
    if not isinstance(r, (int, float)) or not (1e-6 <= r <= 1.0):
        errs.append(f"reward 须在 [1e-6, 1]，收到 {r!r}")
    prov = rec.get("provenance", {})
    if prov.get("source") != "AI+verified":
        errs.append("provenance.source 须为 AI+verified")
    if prov.get("anchor_eligible") is not False:
        errs.append("anchor_eligible 须恒为 false（红线）")
    expect = make_structure_id(rec.get("domain", ""), rec.get("params", {}),
                               rec.get("agent", ""))
    if rec.get("structure_id") != expect:
        errs.append(f"structure_id 与哈希不一致（期望 {expect}）")
    return errs


class DB4Writer:
    def __init__(self, path: Path = _DB4_PATH):
        self.path = path

    def append(self, rec: dict) -> dict:
        """校验 + 幂等追加（按 structure_id 去重）。返回状态。"""
        errs = validate_record(rec)
        if errs:
            return {"ok": False, "errors": errs}
        existing = {r["structure_id"] for r in self.load()}
        if rec["structure_id"] in existing:
            return {"ok": True, "duplicate": True}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"ok": True, "duplicate": False}

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        recs = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return recs


def build_record(domain: str, params: dict, i_ty: float, reward: float,
                 info_action: float, agent: str, policy_tag: str) -> dict:
    import math
    r = max(float(reward), 1e-6)
    return {
        "structure_id": make_structure_id(domain, params, agent),
        "domain": domain,
        "params": params,
        "q": {"I_TY": float(i_ty),
              "fisher_trace": float(i_ty)},   # 单参数族 trace=I(θ) 精确（v0）
        "reward": float(reward),
        "log_reward": math.log(r),
        "info_action": float(info_action),
        "agent": agent,
        "policy_tag": policy_tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provenance": {"source": "AI+verified", "anchor_eligible": False},
    }
