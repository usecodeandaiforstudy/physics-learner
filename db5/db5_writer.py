# -*- coding: utf-8 -*-
"""DB5 写入器（v0，2026-10-02）：校验 / 幂等追加 / 读取。

validation_id = db5- + sha1(structure_id) 前 8 位（一个结构一条验证记录）。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["DB5Writer"]

_DB5_PATH = Path(__file__).resolve().parent / "db5.jsonl"


def make_validation_id(structure_id: str) -> str:
    return "db5-" + hashlib.sha1(structure_id.encode("utf-8")).hexdigest()[:8]


def validate_record(rec: dict) -> list[str]:
    errs: list[str] = []
    for k in ("validation_id", "structure_id", "domain", "decoded_design",
              "validation", "timestamp", "provenance"):
        if k not in rec:
            errs.append(f"缺字段 {k}")
    if rec.get("validation_id") != make_validation_id(
            rec.get("structure_id", "")):
        errs.append("validation_id 与 structure_id 哈希不一致")
    v = rec.get("validation", {})
    if not isinstance(v.get("overlap"), (int, float)) or not (
            0.0 <= v.get("overlap", -1) <= 1.0):
        errs.append("validation.overlap 须在 [0,1]")
    if v.get("verdict") not in ("pass", "fail"):
        errs.append("validation.verdict 须为 pass/fail")
    prov = rec.get("provenance", {})
    # 证据来源（2026-10-02，缺口①）：解析自洽（AI+verified）或**外部模拟器**
    # （AI+simulated，须带 sim 元数据）。红线不变：anchor_eligible 恒 false。
    if prov.get("source") not in ("AI+verified", "AI+simulated"):
        errs.append("provenance.source 须为 AI+verified 或 AI+simulated")
    if prov.get("source") == "AI+simulated":
        sim = rec.get("sim", {})
        for k in ("adapter", "seed", "n_samples", "tolerance", "q_required",
                  "q_measured"):
            if k not in sim:
                errs.append(f"模拟器证据缺 sim.{k}")
    if prov.get("anchor_eligible") is not False:
        errs.append("anchor_eligible 须恒为 false（红线）")
    return errs


class DB5Writer:
    def __init__(self, path: Path = _DB5_PATH):
        self.path = path

    def append(self, rec: dict) -> dict:
        errs = validate_record(rec)
        if errs:
            return {"ok": False, "errors": errs}
        existing = {r["validation_id"] for r in self.load()}
        if rec["validation_id"] in existing:
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


def build_record(rec4: dict, decoded: dict, validation: dict) -> dict:
    return {
        "validation_id": make_validation_id(rec4["structure_id"]),
        "structure_id": rec4["structure_id"],
        "domain": rec4["domain"],
        "decoded_design": decoded,
        "validation": validation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provenance": {"source": "AI+verified", "anchor_eligible": False},
    }
