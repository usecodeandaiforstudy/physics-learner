# -*- coding: utf-8 -*-
"""模式③：LLM API 批量生成 DB1 条目 + 约定漂移前置拦截 + DB2 落盘。

流程（每条目）：
  1) 提示词 = conventions.yaml（唯一权威）+ 非协商项 + 本族范例条目 + 多样性要求；
  2) LLM 返回两段式 {entry, attestation}（区块 1 进 DB1、区块 2 进复核比对）；
  3) **约定漂移前置拦截**：三道机器门（validate_entry）+ 追加检查
     （family/channel_template 与范例一致、scan_param 符合 spec、system_id
     唯一、provenance 强制模板覆盖）；
  4) 失败 → 把错误清单喂回 LLM 重试（max_retries），仍失败 → 拒绝落盘并记漂移日志；
  5) 通过 → 写 db1/entries/<sid>.json + 参数扫描管线（ingest 数值 vs 解析对照）
     → DB2 按系统分块追加（镜像 batch_generate_ingest 的写入策略）。

红线遵守：provenance.source=AI+verified、anchor_eligible=false、
human_review_status=pending、generator_model=mode3-llm-v1-<model>、
verified_by=analytic_compare（ingest 实际执行该对照）。生成物不升格锚点，
后续走 review_queue 人工复核。

用法（项目根目录）：
    python db1/mode3_generate.py --dry-run                    # 只看提示词不调 API
    python db1/mode3_generate.py --limit 2 --points 30        # 首批小试
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DB1 = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_DB1) not in sys.path:
    sys.path.insert(0, str(_DB1))

from mode3_llm import chat, extract_json, load_config  # noqa: E402
from validate_entry import validate_entry  # noqa: E402
from ingest_pipeline import ingest  # noqa: E402
from db2_writer import build_db2_record  # noqa: E402
from batch_generate_ingest import DB2_PATH, _scan_grid  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ENTRIES_DIR = _DB1 / "entries"
CONVENTIONS = _DB1 / "conventions.yaml"
DRIFT_LOG = _DB1 / "mode3_drift_log.jsonl"

PROVENANCE_TEMPLATE = {
    "source": "AI+verified",
    "generator_model": "mode3-llm-v1",
    "verified_by": "analytic_compare",
    "human_review_status": "pending",
    "review_date": None,
}

NON_NEGOTIABLES = """非协商项（违反即拒收）：
- 高斯参数化用 sigma（标准差），方差写成 sigma^2；禁止 sigma_squared/variance 作参数名
- 信息量一律 nats（自然对数），禁止 bits
- SDE 默认 Itô，sigma^2 是方差速率
- 热浴默认 kBT=1；偏离时把 kT 显式写进 parameters 且 is_bridge_constant=true
- 桥接常数须带 unit 且标 is_bridge_constant=true
- 边方向 = 信息流方向 X→T→Y
- strictness_level 只用 定理/推导/类比/猜想，按依据真实分级
- fingerprint_expectation 的 analytic 必须是文献可查、sympy 可解析的解析式，不许写估算值
- system_id 匹配 ^[a-z0-9]+(_+[a-z0-9]+)+$；channel_template 匹配 ^CT-[A-Z]+-\\d{3}$
- provenance.source=AI+verified、anchor_eligible=false、human_review_status=pending
- 自证表：anchor_formulas=[]（AI 生成物不可作锚点）
"""


def build_system_prompt(conventions_text: str, example: dict) -> str:
    return (
        "你是信息力学项目的 DB1 物理系统条目生成器。严格按约定生成，输出两段式 JSON：\n"
        '{"entry": {...}, "attestation": {"convention_attestation": {...}}}\n\n'
        "【约定（唯一权威）】\n" + conventions_text + "\n\n"
        "【非协商项】\n" + NON_NEGOTIABLES + "\n"
        "【attestation 必须含】convention_version、log_base=natural、"
        "log_base_unit=nats、sigma_parameterization=sigma_as_std、"
        "sde_calculus=ito、thermal_bath_kT=1.0、"
        "bridge_constants=[条目中实际标 is_bridge_constant 的参数名]、anchor_formulas=[]\n\n"
        "【本族范例（对齐其 family/channel_template/参数化/单位风格）】\n"
        + json.dumps(example, ensure_ascii=False, indent=2)
    )


def _existing_ids() -> set[str]:
    return {p.stem for p in ENTRIES_DIR.glob("*.json")}


def enforce_provenance(entry: dict, model: str) -> dict:
    prov = entry.setdefault("provenance", {})
    prov.update(PROVENANCE_TEMPLATE)
    prov["generator_model"] = "mode3-llm-v1-%s" % model
    entry["provenance"] = prov
    entry["convention_version"] = "1.0.0"
    return entry


def drift_checks(entry: dict, spec: dict, example: dict,
                 used_ids: set[str]) -> list[str]:
    errs: list[str] = []
    if entry.get("family") != spec["family"]:
        errs.append("family=%r 应为 %r" % (entry.get("family"), spec["family"]))
    if entry.get("channel_template") != example.get("channel_template"):
        errs.append("channel_template=%r 应为 %r"
                    % (entry.get("channel_template"),
                       example.get("channel_template")))
    if spec.get("scan_param") and entry.get("scan_param") != spec["scan_param"]:
        errs.append("scan_param=%r 应为 %r"
                    % (entry.get("scan_param"), spec["scan_param"]))
    if entry.get("system_id") in used_ids:
        errs.append("system_id=%r 重复" % entry.get("system_id"))
    return errs


def ingest_to_db2(entry: dict, attestation: dict, points: int) -> dict:
    """参数扫描 + ingest（数值 vs 解析对照）→ DB2 按系统分块追加。"""
    grid = _scan_grid(entry, points)
    sys_records: list[dict] = []
    n_passed = 0
    errors: list[str] = []
    for idx, val in enumerate(grid):
        e_pt = json.loads(json.dumps(entry))
        scan_meta = None
        if val is not None:
            sp = entry["scan_param"]
            for p in e_pt["parameters"]:
                if p["name"] == sp:
                    p["default"] = float(val)
            scan_meta = {"scan_param": sp, "scan_index": idx,
                         "scan_value": float(val),
                         "params_used": {p["name"]: p["default"]
                                         for p in e_pt["parameters"]}}
        try:
            r = ingest(e_pt, attestation, db2_path=None, scan=scan_meta)
            if r["passed"]:
                n_passed += 1
                sys_records.append(build_db2_record(e_pt, {
                    "fingerprint": r["fingerprint"],
                    "graph": r["graph"],
                    "ib_curve": r["ib_curve"],
                }, scan=scan_meta))
            else:
                errors.append("[point %d] %s" % (idx, r.get("stage")))
        except Exception as ex:  # noqa: BLE001
            errors.append("[point %d] %s: %s" % (idx, type(ex).__name__, ex))
    if sys_records:
        with open(DB2_PATH, "a", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r, ensure_ascii=False)
                              for r in sys_records) + "\n")
    return {"n_points": len(grid), "n_passed": n_passed, "errors": errors}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="模式③ DB1 批量生成")
    ap.add_argument("--config", default=str(_DB1 / "mode3_config.json"))
    ap.add_argument("--specs", default=str(_DB1 / "mode3_batch_specs.json"))
    ap.add_argument("--points", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑前 N 条（0=全部）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印提示词，不调 API")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    specs = json.loads(Path(args.specs).read_text(encoding="utf-8"))["specs"]
    conventions_text = CONVENTIONS.read_text(encoding="utf-8")
    used_ids = _existing_ids()
    total_done = total_rejected = total_retried = 0

    for spec in specs:
        family = spec["family"]
        examples = sorted(ENTRIES_DIR.glob("*.json"))
        example = None
        for ef in examples:
            e = json.loads(ef.read_text(encoding="utf-8"))
            if e.get("family") == family:
                example = e
                break
        if example is None:
            print("[SKIP] family=%s 无范例条目" % family)
            continue
        system_prompt = build_system_prompt(conventions_text, example)
        for k in range(int(spec["n"])):
            if args.limit and total_done + total_rejected >= args.limit:
                break
            user_prompt = ("请生成 1 个新的 %s 族物理系统条目。要求：%s"
                           % (family, spec["diversity_note"]))
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}]
            if args.dry_run:
                print("\n===== DRY-RUN 提示词（%s #%d） =====" % (family, k))
                print(system_prompt[:400] + " ...")
                print("[user]", user_prompt)
                total_done += 1
                continue

            ok = False
            for attempt in range(int(cfg.get("max_retries", 3)) + 1):
                try:
                    raw = chat(cfg, messages)
                    data = extract_json(raw)
                    entry, attestation = data["entry"], data["attestation"]
                    enforce_provenance(entry, cfg["model"])
                    report = validate_entry(entry, attestation)
                    errs = list(report["errors"]) + drift_checks(
                        entry, spec, example, used_ids)
                except Exception as ex:  # noqa: BLE001
                    errs = [type(ex).__name__ + ": " + str(ex)]
                if not errs:
                    ok = True
                    break
                total_retried += 1
                messages.append({"role": "user",
                                 "content": "上一份被机器门拒收，错误：%s"
                                            % errs})
            if not ok:
                total_rejected += 1
                with open(DRIFT_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"family": family, "attempts": attempt + 1,
                                        "errors": errs}, ensure_ascii=False)
                            + "\n")
                print("[REJECT] %s #%d：%s" % (family, k, errs[:2]))
                continue

            sid = entry["system_id"]
            used_ids.add(sid)
            out = ENTRIES_DIR / ("%s.json" % sid)
            out.write_text(json.dumps(entry, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            t0 = time.time()
            res = ingest_to_db2(entry, attestation, args.points)
            total_done += 1
            print("[OK] %-28s %s/%s 点通过（%.1fs）%s"
                  % (sid, res["n_passed"], res["n_points"],
                     time.time() - t0,
                     res["errors"][:1] if res["errors"] else ""))
        if args.limit and total_done + total_rejected >= args.limit:
            break

    print("\n模式③ 汇总：落盘 %d 条；拒收 %d 条；重试 %d 次（漂移日志 %s）"
          % (total_done, total_rejected, total_retried, DRIFT_LOG.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
