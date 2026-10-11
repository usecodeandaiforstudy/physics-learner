"""multi-seed 复核 +A+B np32 vs base（HANDOVER 第 26 条「下一步①」）。

目标：把「理论约束效应」与「随机波动」分开。第 26 条结论只来自单 seed
（seed 0），需 3–5 seed 复核。做法：用 subprocess 逐 (配置, seed) 调用
train_gine.py（同一套训练代码，保证与第 26 条数字可比），每 seed 跑两个配置：

  - base        无理论约束（w_consist=0, w_shape=0）
  - wA1_wB1_np32 +A+B（w_consist=1, w_shape=1, n_shape_pairs=32）

配对设计：split=family 的 holdout 由 seed 决定（rng.choice），同 seed 两配置
holdout 完全相同 → 可做配对差 Δ(plus−base)；跨 seed holdout 不同 → 检验
结论对 holdout 选择的稳健性。

汇总口径对齐第 26 条的教训：**overall（数据加权）、族等权、无链族等权三个
口径必须同时报告**（overall 会被 ou 的大节点数主导）。

输出（全部在 a_network/ 下）：
  - gine_metrics_family_{tag}_seed{n}.json   每 (配置, seed) 的完整指标（拷贝自
    train_gine.py 输出，原文件会被下一次运行覆盖，故逐 seed 归档）
  - multi_seed_verify.jsonl                  每跑完一轮追加一行（中断安全、可续跑）
  - multi_seed_verify_summary.json           聚合：mean±std、配对 Δ、逐族汇总
  - multi_seed_verify.log                    完整运行日志（含每轮耗时）

用法（在项目根目录你自己的终端里）：
  # 第一步：先只跑 seed 0，复现第 26 条单 seed 数字 + 计时一轮
  python a_network/multi_seed_verify.py --seeds 0 --epochs 300

  # 第二步：补全 3–5 seed（seed 0 已完成会自动跳过）
  python a_network/multi_seed_verify.py --seeds 0 1 2 --epochs 300
  python a_network/multi_seed_verify.py --seeds 0 1 2 3 4 --epochs 300

  # 固定 holdout 协议（第 27 条教训 (c)：同 holdout 跨 seed，只变训练随机数，
  # 把「约束效应」与「随机波动」真正分开；与随机 holdout 的记录按签名分组存储）
  python a_network/multi_seed_verify.py --seeds 0 1 2 3 4 --epochs 300 --holdout landau,negbin,ou,gamma,laplace
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_A_DIR = Path(__file__).resolve().parent
TRAIN_PY = _A_DIR / "train_gine.py"

# 配置名 → train_gine.py 的额外参数（--split/--epochs/--seed 由 runner 统一加）
CONFIGS: dict[str, list[str]] = {
    "base": ["--w_consist", "0", "--w_shape", "0"],
    "base_best": ["--w_consist", "0", "--w_shape", "0", "--best-checkpoint"],
    "wA1_wB1_np32": ["--w_consist", "1", "--w_shape", "1",
                     "--n_shape_pairs", "32"],
}

# 全库仅有链结构的两个族（HANDOVER 第 26 条：ou(12)/diffusion(9) 共 21 系统有边），
# 「无链族等权」口径 = holdout 中排除这两族后逐族等权。
CHAIN_FAMILIES = {"ou", "diffusion"}

RESULTS_JSONL = _A_DIR / "multi_seed_verify.jsonl"
SUMMARY_JSON = _A_DIR / "multi_seed_verify_summary.json"
LOG_FILE = _A_DIR / "multi_seed_verify.log"

# 需要统计 / 配对比较的指标列（rec 的键名）
METRICS = [
    "test_trace", "test_det", "fam_eq_trace", "fam_eq_det",
    "chainless_eq_trace", "chainless_eq_det", "train_trace", "train_det",
]


def _tag_of(config_name: str) -> str:
    # 与 train_gine.py 的输出文件名 tag 一致（base / wA1_wB1_np32 / base_best）
    return config_name


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _std(values: list[float]) -> float | None:
    return float(statistics.stdev(values)) if len(values) >= 2 else None


def _derived(m: dict) -> dict:
    """从 train_gine.py 的 metrics JSON 计算族等权 / 无链族等权。"""
    pf = m["test"].get("per_family", {})
    fam_tr = [v["r2_trace"] for v in pf.values()]
    fam_det = [v["r2_det"] for v in pf.values()]
    chainless = [(k, v["r2_trace"], v["r2_det"]) for k, v in pf.items()
                 if k not in CHAIN_FAMILIES]
    return {
        "fam_eq_trace": _mean(fam_tr),
        "fam_eq_det": _mean(fam_det),
        "chainless_eq_trace": _mean([c[1] for c in chainless]),
        "chainless_eq_det": _mean([c[2] for c in chainless]),
        "n_holdout_families": len(pf),
        "n_chainless_families": len(chainless),
    }


def load_all_records() -> list[dict]:
    """解析 jsonl，按 (key) 去重（保留最后一次出现的记录）。"""
    by_key: dict[str, dict] = {}
    if RESULTS_JSONL.exists():
        for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("key"):
                by_key[rec["key"]] = rec
    return list(by_key.values())


def run_one(config_name: str, seed: int, epochs: int, overwrite: bool,
            done: set[str], log, holdout: list[str] | None,
            n_db1_systems: int) -> dict | None:
    tag = _tag_of(config_name)
    if holdout:
        hl = sorted(set(holdout))
        slug = "h-" + "-".join(hl)
        harg = ["--holdout", ",".join(hl)]
    else:
        slug = "rand"
        harg = []
    # 数据版本标记：DB1 系统数不同 = 数据集不同，旧记录不可跳过/混算（第 39 条）
    slug = f"{slug}-d{n_db1_systems}"
    key = f"{tag}_seed{seed}_e{epochs}_{slug}"
    seed_out = _A_DIR / f"gine_metrics_family_{tag}_seed{seed}_{slug}.json"
    if not overwrite and key in done and seed_out.exists():
        print(f"[skip] {key} 已完成（用 --overwrite 重跑）", flush=True)
        return None

    cmd = [sys.executable, str(TRAIN_PY), "--split", "family",
           "--epochs", str(epochs), "--seed", str(seed)] + CONFIGS[config_name] + harg
    print(f"[run ] {key}  开始（epochs={epochs}）", flush=True)
    log.write(f"\n===== {key} @ {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
    log.write("cmd: " + " ".join(cmd) + "\n")
    log.flush()

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd, cwd=_PROJECT_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        rc = proc.wait()
    except Exception as exc:  # noqa: BLE001
        wall = time.time() - t0
        print(f"[FAIL] {key} 启动异常：{exc!r}  wall={wall:.1f}s", flush=True)
        log.write(f"[FAIL] {key} 启动异常：{exc!r}\n")
        log.flush()
        return None

    wall = time.time() - t0
    if rc != 0:
        print(f"[FAIL] {key} rc={rc}  wall={wall:.1f}s（见 {LOG_FILE.name}）",
              flush=True)
        log.write(f"[FAIL] {key} rc={rc} wall={wall:.1f}s\n")
        log.flush()
        return None

    src = _A_DIR / f"gine_metrics_family_{tag}.json"
    if not src.exists():
        print(f"[FAIL] {key} 训练结束但未找到 {src.name}", flush=True)
        return None

    m = json.loads(src.read_text(encoding="utf-8"))
    shutil.copyfile(src, seed_out)
    d = _derived(m)
    rec = {
        "key": key,
        "config": config_name,
        "tag": tag,
        "seed": seed,
        "epochs": epochs,
        "wall_seconds": round(wall, 1),
        "attr_version": m.get("attr_version"),
        "holdout": m.get("holdout"),
        "holdout_mode": "fixed" if holdout else "random",
        "holdout_sig": slug,
        "n_db1_systems": n_db1_systems,
        "train_trace": m["train"]["r2_trace"],
        "train_det": m["train"]["r2_det"],
        "test_trace": m["test"]["r2_trace"],
        "test_det": m["test"]["r2_det"],
        "test_acc": m["test"]["acc"],
        "per_family": m["test"]["per_family"],
        "metrics_file": seed_out.name,
        **d,
    }
    with RESULTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done] {key}  wall={wall:.1f}s  "
          f"train={rec['train_trace']:.3f}/{rec['train_det']:.3f}  "
          f"test overall={rec['test_trace']:+.3f}/{rec['test_det']:+.3f}  "
          f"族等权={rec['fam_eq_trace']:+.3f}/{rec['fam_eq_det']:+.3f}  "
          f"无链族等权={rec['chainless_eq_trace']:+.3f}/{rec['chainless_eq_det']:+.3f}",
          flush=True)
    return rec


def _fmt(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "      —"
    if std is None:
        return f"{mean:+.3f}"
    return f"{mean:+.3f}±{std:.3f}"


def summarize(records: list[dict]) -> dict:
    by_cfg: dict[str, list[dict]] = {}
    for r in records:
        by_cfg.setdefault(r["config"], []).append(r)

    summary: dict = {"by_config": {}, "paired_delta": {}, "per_family": {}}
    for cfg, recs in by_cfg.items():
        cfg_sum: dict[str, dict] = {}
        for metric in METRICS:
            vals = [r[metric] for r in recs if r.get(metric) is not None]
            cfg_sum[metric] = {
                "values": vals,
                "mean": _mean(vals),
                "std": _std(vals),
            }
        summary["by_config"][cfg] = cfg_sum

        # 逐族汇总（仅统计「该 seed 里位于 holdout」的族）
        fam_by_name: dict[str, dict[str, list[float]]] = {}
        for r in recs:
            for fam, v in (r.get("per_family") or {}).items():
                slot = fam_by_name.setdefault(fam, {"trace": [], "det": []})
                slot["trace"].append(v["r2_trace"])
                slot["det"].append(v["r2_det"])
        summary["per_family"][cfg] = {
            fam: {
                "trace": {"values": s["trace"], "mean": _mean(s["trace"]),
                          "std": _std(s["trace"]), "n_seeds": len(s["trace"])},
                "det": {"values": s["det"], "mean": _mean(s["det"]),
                        "std": _std(s["det"]), "n_seeds": len(s["det"])},
            }
            for fam, s in sorted(fam_by_name.items())
        }

    # 配对差 Δ(plus − base)：同 seed 同 holdout，可直接相减
    base = {r["seed"]: r for r in by_cfg.get("base", [])}
    plus = {r["seed"]: r for r in by_cfg.get("wA1_wB1_np32", [])}
    for seed in sorted(set(base) & set(plus)):
        for metric in METRICS:
            if base[seed].get(metric) is None or plus[seed].get(metric) is None:
                continue
            d = summary["paired_delta"].setdefault(metric, [])
            d.append(plus[seed][metric] - base[seed][metric])
    summary["paired_delta"] = {
        k: {"values": v, "mean": _mean(v), "std": _std(v)}
        for k, v in summary["paired_delta"].items()
    }
    return summary


def print_summary(records: list[dict], summary: dict, sig: str = "") -> None:
    print()
    print("================ multi-seed 复核汇总 ================")
    print(f"holdout 签名：{sig}")
    print(f"记录数：{len(records)}（seeds="
          f"{sorted({r['seed'] for r in records})}）")
    b = summary["by_config"].get("base", {})
    p = summary["by_config"].get("wA1_wB1_np32", {})

    def row(name: str, metric: str) -> str:
        bm = b.get(metric, {})
        pm = p.get(metric, {})
        return (f"{name:14s} base={_fmt(bm.get('mean'), bm.get('std')):14s}  "
                f"+A+B np32={_fmt(pm.get('mean'), pm.get('std')):14s}")

    print()
    print("-- 跨 seed mean±std --")
    for name, metric in [
        ("test trace", "test_trace"), ("test det", "test_det"),
        ("族等权 trace", "fam_eq_trace"), ("族等权 det", "fam_eq_det"),
        ("无链族 trace", "chainless_eq_trace"),
        ("无链族 det", "chainless_eq_det"),
        ("train trace", "train_trace"), ("train det", "train_det"),
    ]:
        print(row(name, metric))

    print()
    print("-- 配对差 Δ(+A+B np32 − base)，同 seed 同 holdout --")
    for name, metric in [
        ("test trace", "test_trace"), ("test det", "test_det"),
        ("族等权 trace", "fam_eq_trace"), ("族等权 det", "fam_eq_det"),
        ("无链族 trace", "chainless_eq_trace"),
        ("无链族 det", "chainless_eq_det"),
    ]:
        d = summary["paired_delta"].get(metric, {})
        vals = d.get("values", [])
        print(f"{name:14s} Δ={_fmt(d.get('mean'), d.get('std')):14s}"
              f"  values={[round(v, 3) for v in vals]}")

    print()
    print("-- 逐族（仅出现在 ≥1 个 holdout 的族；trace/det mean±std）--")
    pf_b = summary["per_family"].get("base", {})
    pf_p = summary["per_family"].get("wA1_wB1_np32", {})
    for fam in sorted(set(pf_b) | set(pf_p)):
        bm = pf_b.get(fam, {})
        pm = pf_p.get(fam, {})
        bt, bd = bm.get("trace", {}), bm.get("det", {})
        pt, pd = pm.get("trace", {}), pm.get("det", {})
        print(f"{fam:20s} base {_fmt(bt.get('mean'), bt.get('std')):14s}"
              f"/{_fmt(bd.get('mean'), bd.get('std')):14s}   "
              f"+AB {_fmt(pt.get('mean'), pt.get('std')):14s}"
              f"/{_fmt(pd.get('mean'), pd.get('std')):14s}"
              f"   (n={bt.get('n_seeds', 0)})")
    print()
    print(f"汇总已写入 {SUMMARY_JSON.name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="multi-seed 复核 +A+B np32 vs base")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="seed 列表（默认 0 1 2；3–5 个建议）")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS),
                    default=["base", "wA1_wB1_np32"])
    ap.add_argument("--overwrite", action="store_true",
                    help="重跑已完成的 (config, seed)（默认跳过）")
    ap.add_argument("--holdout", default=None,
                    help="固定 holdout（逗号分隔族名，透传给 train_gine.py；"
                         "None=按 seed 随机抽）")
    args = ap.parse_args(argv)

    holdout_list = ([h.strip() for h in args.holdout.split(",") if h.strip()]
                    if args.holdout else None)
    n_db1_systems = len(list((_PROJECT_ROOT / "db1" / "entries").glob("*.json")))
    cur_sig = (("h-" + "-".join(sorted(set(holdout_list)))
                if holdout_list else "rand") + "-d%d" % n_db1_systems)

    n_runs = len(args.configs) * len(args.seeds)
    mode = (f"固定 holdout={','.join(sorted(set(holdout_list)))}"
            if holdout_list else "随机 holdout（按 seed）")
    print(f"计划运行 {n_runs} 轮（{len(args.configs)} 配置 × {len(args.seeds)} seed，"
          f"epochs={args.epochs}，{mode}）；已完成自动跳过。", flush=True)
    if not TRAIN_PY.exists():
        print(f"[FAIL] 找不到 {TRAIN_PY}")
        return 2

    with LOG_FILE.open("a", encoding="utf-8") as log:
        log.write(f"\n\n########## runner 启动 @ {datetime.now():%Y-%m-%d %H:%M:%S}"
                  f"（seeds={args.seeds} epochs={args.epochs} "
                  f"holdout={holdout_list}）##########\n")
        log.flush()
        done = {r["key"] for r in load_all_records()}
        for config_name in args.configs:
            for seed in args.seeds:
                run_one(config_name, seed, args.epochs, args.overwrite, done, log,
                        holdout_list, n_db1_systems)

    records = load_all_records()
    if not records:
        print("没有可用记录。")
        return 1
    # 若同一 (配置, seed, holdout) 存在不同 epochs 的记录（如先跑 e100 试速再跑
    # e300），只取最大 epochs 的批次，避免混入未收敛数字。
    max_epochs = max(r["epochs"] for r in records)
    records = [r for r in records if r["epochs"] == max_epochs]

    # 按 holdout 签名分组汇总（固定 holdout 与随机 holdout 的记录不可混算）
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r.get("holdout_sig", "rand-legacy"), []).append(r)
    per_group: dict[str, dict] = {}
    for sig, recs in sorted(groups.items()):
        per_group[sig] = {"per_run": recs, **summarize(recs)}

    out = {
        "note": "multi-seed 复核 +A+B np32 vs base。口径：overall=数据加权、"
                "族等权=每个 holdout 族一票、无链族等权=holdout 中排除 "
                "ou/diffusion 后逐族等权。分组键=holdout 签名（h-*=固定 holdout，"
                "rand=按 seed 随机，rand-legacy=第 27 条旧记录）。",
        "epochs": args.epochs,
        "groups": per_group,
    }
    SUMMARY_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    cur = groups.get(cur_sig)
    if cur:
        print_summary(cur, per_group[cur_sig], cur_sig)
    else:
        print(f"[note] 当前 holdout 签名 {cur_sig} 无记录。")
    others = [s for s in groups if s != cur_sig]
    if others:
        print(f"[note] 另存 {len(others)} 组其他 holdout 记录：{others}（见 "
              f"{SUMMARY_JSON.name}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
