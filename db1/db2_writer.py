"""DB2 信息指纹库写入。

把通过入库管线的指纹（Fisher 摘要 + mi_profile + ib_curve）落库为 JSONL 记录。
DB2 是 A 网络的唯一训练源（架构文档 §3.3），每条记录须带 encoder_version 与
convention_version 以支持 B 升级时的全库重编码（分布漂移管理）。

存储格式：每行一个 JSON 对象（JSONL），便于追加与增量读取。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["build_db2_record", "append_db2", "load_db2"]


def build_db2_record(
    entry: dict,
    report: dict,
    encoder_version: str = "1.0.0",
    scan: dict | None = None,
) -> dict:
    """从 pipeline report 构建一条 DB2 记录。

    参数
    ----
    entry : dict
        DB1 条目。
    report : dict
        ingest() 返回的报告（须 stage == fingerprint 且 passed）。
    encoder_version : str
        B 网络编码器版本。
    scan : dict, 可选
        参数扫描元数据 {"scan_param", "scan_index", "scan_value", "params_used"}。
        同一条目的扫描点靠 scan_index 区分。

    返回
    ----
    dict
        DB2 记录。
    """
    fp = report["fingerprint"]
    graph = report.get("graph", {})

    # 把 numpy 标量转为原生类型（保证可 JSON 序列化）
    def _plain(v):
        if v is None:
            return None
        if hasattr(v, "item"):  # numpy 标量
            return v.item()
        return v

    mi_profile = [_plain(m) for m in fp.get("mi_profile", [])]

    record = {
        "system_id": entry.get("system_id"),
        "domain": entry.get("domain"),
        "family": entry.get("family"),
        "encoder_version": encoder_version,
        "convention_version": entry.get("convention_version"),
        "fingerprints": fp.get("fingerprints", {}),
        "mi_profile": mi_profile,
        "ib_curve": report.get("ib_curve"),
        "metadata": graph.get("metadata", {}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # 扫描元数据：标识该记录来自哪个扫描点
    if scan is not None:
        record["scan"] = {
            "scan_param": scan.get("scan_param"),
            "scan_index": scan.get("scan_index"),
            "scan_value": _plain(scan.get("scan_value")),
            "params_used": {k: _plain(v) for k, v in (scan.get("params_used") or {}).items()},
        }
    return record


def append_db2(record: dict, db2_path: str | Path) -> None:
    """追加一条记录到 DB2 JSONL 文件（不存在则创建）。"""
    db2_path = Path(db2_path)
    db2_path.parent.mkdir(parents=True, exist_ok=True)
    with open(db2_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_db2(db2_path: str | Path) -> list[dict]:
    """读取 DB2 JSONL 全部记录。"""
    db2_path = Path(db2_path)
    if not db2_path.exists():
        return []
    records = []
    with open(db2_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
