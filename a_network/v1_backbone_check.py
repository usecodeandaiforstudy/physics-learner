"""V1 · 解析背板 vs DB2 逐位对照（M1 验收，2026-09-29）。

验收标准（design doc §5 V1）：背板输出的节点 Fisher 与边 MI 与 DB2 全量
2520 条**逐位一致**（DB2 本身就是解析管线产物）；同时统计 DPI 硬判据
（A1 单调 / A2 加性噪声闭式）的检查数与违规数（预期 0 违规）。

匹配方式：DB2 行不含扫描值，故对每个系统预计算 30 个扫描点的背板输出，
用「指纹签名」（节点 Fisher 的精确浮点元组）反查每行对应的扫描点——
与行序无关，历史批量追加顺序不影响校验。

输出：console + a_network/v1_backbone_report.json。
用法（项目根目录，预计 1–3 分钟）：
    python a_network/v1_backbone_check.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from a_network.analytic_backbone import compute_backbone  # noqa: E402
from a_network.graph_dataset import _scan_grid, _with_scan  # noqa: E402
from b_network.system_encoder import SystemEncoder  # noqa: E402

DB2_PATH = _PROJECT_ROOT / "db2" / "db2.jsonl"
ENTRIES_DIR = _PROJECT_ROOT / "db1" / "entries"
OUT_JSON = Path(__file__).resolve().parent / "v1_backbone_report.json"
SCAN_POINTS = 30


def _fp_signature(node_fisher: dict) -> tuple:
    """节点 Fisher 精确签名（浮点按位比较——同一解析路径应逐位一致）。"""
    return tuple(sorted(
        (k, v["fisher_trace"], v["fisher_det"])
        for k, v in node_fisher.items()
        if v["covered"]
    ))


def _mi_tuple(edge_mi: list[dict]) -> tuple:
    return tuple(float(e["mi"]) if e["mi"] is not None else float("nan")
                 for e in edge_mi)


def main() -> int:
    rows = [json.loads(line) for line in
            DB2_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"DB2 行数：{len(rows)}")
    systems = sorted({r["system_id"] for r in rows})
    print(f"涉及系统：{len(systems)}")

    entries: dict[str, dict] = {}
    for ef in sorted(ENTRIES_DIR.glob("*.json")):
        e = json.loads(ef.read_text(encoding="utf-8"))
        entries[e["system_id"]] = e

    encoder = SystemEncoder()
    t0 = time.time()
    # 每系统预计算 30 扫描点：签名 → (scan_val, backbone)
    by_system: dict[str, dict] = {}
    for sid in systems:
        entry = entries.get(sid)
        if entry is None:
            print(f"[FAIL] 找不到条目 {sid}")
            return 1
        sig_map: dict = {}
        for val in _scan_grid(entry, SCAN_POINTS):
            e_pt = entry if val is None else _with_scan(entry, val)
            graph = encoder.encode(e_pt)
            bb = compute_backbone(graph)
            sig_map[_fp_signature(bb["node_fisher"])] = {
                "scan": val, "bb": bb,
                "mi": _mi_tuple(bb["edge_mi"]),
            }
        by_system[sid] = sig_map
    build_wall = time.time() - t0

    # 逐行比对
    n_matched = n_mi_mismatch = n_unmatched = 0
    n_dpi_ok = n_dpi_vio = 0
    a1_total = a2_total = 0
    n_nodes_covered = n_nodes_total = 0
    n_edges_covered = n_edges_total = 0
    unmatched_examples: list[dict] = []

    for row in rows:
        sid = row["system_id"]
        sig_map = by_system[sid]
        target_fp = row["fingerprints"]
        target_sig = tuple(sorted(
            (k, float(v["fisher_trace"]), float(v["fisher_det"]))
            for k, v in target_fp.items()))
        hit = sig_map.get(target_sig)
        if hit is None:
            n_unmatched += 1
            if len(unmatched_examples) < 5:
                unmatched_examples.append({"system_id": sid,
                                           "fp_keys": list(target_fp)})
            continue
        bb = hit["bb"]
        if hit["mi"] == tuple(float(x) for x in row["mi_profile"]):
            n_matched += 1
        else:
            n_mi_mismatch += 1
        d = bb["dpi"]
        a1_total += d["n_a1_checks"]
        a2_total += d["n_a2_checks"]
        if d["ok"]:
            n_dpi_ok += 1
        else:
            n_dpi_vio += 1
        for v in bb["node_fisher"].values():
            n_nodes_total += 1
            n_nodes_covered += int(v["covered"])
        for e in bb["edge_mi"]:
            n_edges_total += 1
            n_edges_covered += int(e["covered"])

    report = {
        "note": "V1 解析背板验收：节点 Fisher/边 MI vs DB2 逐位对照 + DPI 硬判据统计。",
        "db2_rows": len(rows),
        "systems": len(systems),
        "build_wall_seconds": round(build_wall, 1),
        "matched_bit_exact": n_matched,
        "mi_mismatch": n_mi_mismatch,
        "unmatched": n_unmatched,
        "unmatched_examples": unmatched_examples,
        "dpi": {"ok_graphs": n_dpi_ok, "violating_graphs": n_dpi_vio,
                "a1_checks_total": a1_total, "a2_checks_total": a2_total},
        "coverage": {"nodes_covered": n_nodes_covered,
                     "nodes_total": n_nodes_total,
                     "edges_covered": n_edges_covered,
                     "edges_total": n_edges_total},
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("\n================ V1 解析背板验收 ================")
    print(f"建索引（{len(systems)} 系统 × {SCAN_POINTS} 点）：{build_wall:.1f}s")
    print(f"逐位匹配（指纹 + MI）：{n_matched}/{len(rows)}  "
          f"MI 不一致 {n_mi_mismatch}  未匹配 {n_unmatched}")
    print(f"DPI 硬判据：检查 A1={a1_total} / A2={a2_total}；"
          f"违规图 {n_dpi_vio} / 通过 {n_dpi_ok}")
    print(f"覆盖：节点 {n_nodes_covered}/{n_nodes_total}，"
          f"边 {n_edges_covered}/{n_edges_total}")
    if n_matched == len(rows) and n_dpi_vio == 0:
        print("判定：✅ V1 通过——背板与 DB2 逐位一致，DPI 零违规")
    elif n_matched == len(rows) and n_dpi_vio > 0:
        print(f"判定：⚠️ 指纹逐位一致但 DPI 有 {n_dpi_vio} 处违规——查判据容差/数据")
    else:
        print("判定：❌ 未达标——见 unmatched_examples / 报告细节")
    print(f"\n报告已写入 {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
