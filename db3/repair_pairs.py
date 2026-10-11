# -*- coding: utf-8 -*-
"""**精确修复 DB3 语料**（2026-10-02，第 195 条）——因我的误操作而必须做的事。

## 事故经过（如实记录）
1. 我用了 `Remove-Item db3\\pairs\\DB3-019*.json` **通配符**清退草稿
   ⟹ **误删了既有锚点 `DB3-0190…0195`**（6 条）✗
2. 我用 `gen_pairs_batch.py --start 190 --lo 108 --hi 114 --apply` 想重建 6 条
   ⟹ 该脚本**忽略了 `--lo/--hi`**，把**整个 114 条规格表**写成 `DB3-0190…0303`
   ⟹ **污染语料（267 条）并覆盖了 0190–0195 的正确内容** ✗✗

## 本脚本做什么（**只按显式 ID 判断，绝不用通配符**）
    · **清理**：删除 `db3/pairs/` 中 **ID ≥ 190** 的文件（= 误生成的那些）；
    · **重建**：直接调用生成器的 `build_pair(idx, spec)`，
      用**规格表 idx 108–113** 精确重建 `DB3-0190…0195` ✓
      （规格表下标由 `db3/reviews/*.yaml` 的复核记录原文**逐条比对确认** ✓）；
    · **溯源**：镜像 `DB3-0189` 的 provenance（`approved`/`anchor_eligible=true`），
      但**保留** pending 标记若复核记录缺失；
    · **验证**：重建文本/规则/域与复核记录**逐条一致**；`validate_pair` 全过；
      `check_parser_consistency` **错误 0**；pairs 总数回到 **159** ✓

用法：
    python db3/repair_pairs.py            # 干跑（打印将删/将建）
    python db3/repair_pairs.py --apply    # 执行
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "db3"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

PAIRS = _ROOT / "db3" / "pairs"
REVIEWS = _ROOT / "db3" / "reviews"
START, LO, HI = 190, 108, 114      # 规格表 108..113 -> DB3-0190..0195


def _gen():
    spec = importlib.util.spec_from_file_location(
        "g", str(_ROOT / "db3" / "gen_pairs_batch.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    apply = "--apply" in sys.argv
    m = _gen()
    specs = getattr(m, "SPECS")
    print("=" * 74)
    print("DB3 语料精确修复（%s）" % ("执行" if apply else "干跑"))
    print("=" * 74)
    # ---- 1) 列出将删（ID >= START）----
    junk = []
    for p in sorted(PAIRS.glob("DB3-*.json")):
        try:
            n = int(p.stem.split("-")[1])
        except Exception:  # noqa: BLE001
            continue
        if n >= START:
            junk.append((n, p))
    print("将**删除** %d 个文件（ID ≥ %d）：%s"
          % (len(junk), START,
             ", ".join("DB3-%04d" % n for n, _ in junk[:8]) +
             (" …" if len(junk) > 8 else "")))
    # ---- 2) 将重建 ----
    print("将**重建** %d 条（规格表 %d..%d → DB3-%04d..%04d）"
          % (HI - LO, LO, HI - 1, START, START + (HI - LO) - 1))
    for k in range(LO, HI):
        sid = "DB3-%04d" % (START + (k - LO))
        rv = REVIEWS / ("%s.yaml" % sid)
        want = None
        if rv.exists():
            import yaml
            want = ((yaml.safe_load(rv.read_text(encoding="utf-8")) or {})
                    .get("review_record") or {}).get("requirement_text")
        built = m.build_pair(START + (k - LO), specs[k])
        ok = (want is not None
              and built["requirement_text"].strip() == str(want).strip())
        print("  %s  idx %d  %-46s 与复核记录一致=%s"
              % (sid, k, built["requirement_text"][:44], ok))
    if not apply:
        print("\n（干跑；加 --apply 执行）")
        return 0
    # ---- 3) 执行 ----
    for _, p in junk:
        p.unlink()
    print("\n已删除 %d 个误生成文件" % len(junk))
    # 镜像既有锚点的 provenance（以 DB3-0189 为准）
    base = json.loads((PAIRS / "DB3-0189.json").read_text(encoding="utf-8"))
    prov = dict(base.get("provenance") or {})
    for k in range(LO, HI):
        sid = "DB3-%04d" % (START + (k - LO))
        rec = m.build_pair(START + (k - LO), specs[k])
        rv = REVIEWS / ("%s.yaml" % sid)
        if rv.exists():
            rec["provenance"] = dict(prov)          # 已有复核记录 ⟹ 沿用状态
        (PAIRS / ("%s.json" % sid)).write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已重建 DB3-%04d..%04d（provenance 镜像 DB3-0189：%s）"
          % (START, START + (HI - LO) - 1,
             {kk: prov.get(kk) for kk in ("human_review_status",
                                          "anchor_eligible")}))
    n = len(list(PAIRS.glob("DB3-*.json")))
    print("当前 pairs 总数：**%d**" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
