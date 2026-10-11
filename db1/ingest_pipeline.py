"""DB1 入库管线：机器门 → SystemEncoder → FingerprintComputer。

串联完整入库流程：
    1. validate_entry（三道机器门：schema / 约定一致性 / 解析可计算性）
    2. SystemEncoder.encode（物理系统 → 信息结构图 EncodedGraph）
    3. FingerprintComputer.compute（指纹 + 数值 vs 解析对照）

任一阶段失败即终止，返回结构化报告。用法：
    python ingest_pipeline.py <entry.json> [attestation.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 把项目根与 b_network 加入搜索路径（本文件位于 db1/ 下）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from b_network.fingerprint_computer import FingerprintComputer  # noqa: E402
from b_network.system_encoder import SystemEncoder  # noqa: E402
from db2_writer import append_db2, build_db2_record  # noqa: E402
from validate_entry import validate_entry  # noqa: E402


def ingest(entry: dict, attestation: dict | None = None, db2_path: str | None = None,
           scan: dict | None = None) -> dict:
    """执行完整入库流程，返回结构化报告。

    参数
    ----
    entry : dict
        DB1 条目。
    attestation : dict, 可选
        约定自证表。
    db2_path : str, 可选
        DB2 JSONL 路径；提供且全部通过时，把指纹记录追加写入。
    scan : dict, 可选
        参数扫描元数据，透传给 DB2 记录（见 db2_writer.build_db2_record）。

    返回
    ----
    dict
        {"passed": bool, "stage": str, "validation": ..., "graph": ...,
         "fingerprint": ..., "ib_curve": ..., "db2_written": bool}
    """
    # 阶段 1：三道机器门
    validation = validate_entry(entry, attestation)
    if not validation["passed"]:
        return {
            "passed": False,
            "stage": "validation",
            "validation": validation,
        }

    # 阶段 2：编码为信息结构图
    try:
        graph = SystemEncoder().encode(entry)
    except Exception as e:  # noqa: BLE001
        return {
            "passed": False,
            "stage": "encoding",
            "validation": validation,
            "error": str(e),
        }

    # 阶段 3：指纹 + 数值 vs 解析对照
    fpc = FingerprintComputer()
    fingerprint = fpc.compute(entry, graph)

    passed = fingerprint["passed"]

    # 阶段 4：IB 曲线（若图含 X 源节点与信道边）
    ib = fpc.compute_ib_curve(graph)

    # 阶段 5：DB2 写入（全部通过且指定了 db2_path）
    db2_written = False
    if passed and db2_path:
        record = build_db2_record(entry, {
            "fingerprint": fingerprint,
            "graph": _graph_summary(graph),
            "ib_curve": _plain_ib(ib),
        }, scan=scan)
        append_db2(record, db2_path)
        db2_written = True

    return {
        "passed": passed,
        "stage": "fingerprint",
        "validation": validation,
        "graph": _graph_summary(graph),
        "fingerprint": fingerprint,
        "ib_curve": _plain_ib(ib),
        "db2_written": db2_written,
    }


def _plain_ib(ib: dict) -> dict:
    """把 ib_curve 结果里的 numpy 数组转为 list（可 JSON 序列化）。"""
    return {
        "beta": [float(b) for b in ib.get("beta", [])],
        "I_XT": [float(v) for v in ib.get("I_XT", [])],
        "I_TY": [float(v) for v in ib.get("I_TY", [])],
    }


def _graph_summary(graph) -> dict:
    """把 EncodedGraph 摘要为可序列化的 dict。"""
    return {
        "nodes": [
            {
                "node_id": n.node_id,
                "distribution_family": n.distribution_family,
                "params": n.params,
            }
            for n in graph.nodes
        ],
        "edges": [{"src": e.src, "dst": e.dst} for e in graph.edges],
        "beta": graph.beta,
        "metadata": graph.metadata,
    }


def _print_report(report: dict) -> None:
    stage = report["stage"]
    if report["passed"]:
        print(f"✓ 入库通过（stage={stage}）")
    else:
        print(f"✗ 入库失败（stage={stage}）")

    val = report.get("validation")
    if val:
        for e in val.get("errors", []):
            print(f"  [门] {e}")

    if "error" in report:
        print(f"  [编码错误] {report['error']}")

    if "fingerprint" in report:
        fp = report["fingerprint"]
        print(f"  mi_profile = {fp['mi_profile']}")
        for c in fp["checks"]:
            if c.get("skipped"):
                print(f"  [跳过] {c['quantity']}: {c.get('note')}")
            else:
                mark = "✓" if c["passed"] else "✗"
                print(f"  {mark} {c['quantity']}: 解析 {c['analytic_val']:.6f} "
                      f"vs 数值 {c['numeric_val']:.6f} (rel_err={c['rel_err']:.2e})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DB1 入库管线")
    parser.add_argument("entry", help="条目 JSON 文件路径")
    parser.add_argument("attestation", nargs="?", default=None, help="自证表 YAML（可选）")
    args = parser.parse_args(argv)

    with open(args.entry, "r", encoding="utf-8") as f:
        entry = json.load(f)

    attestation = None
    if args.attestation:
        import yaml
        with open(args.attestation, "r", encoding="utf-8") as f:
            attestation = yaml.safe_load(f) or {}

    report = ingest(entry, attestation)
    _print_report(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
