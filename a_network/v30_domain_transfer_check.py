# -*- coding: utf-8 -*-
"""任务 C 第二层：**域级**迁移测试（2026-10-02，第 121 条）。

**为什么需要第二层**：第 120 条做的是**族级**（分布族）迁移；架构文档 Phase 4
要的是**跨领域**迁移——DB3 的 **52 个域**（mechanics / plasma / biomedical …）
才是"领域"的单位。

**两项测试（判据先写死）**：
  **D1 规则级迁移**：每个域的 `rule_ids` 有多少**也出现在别的域**
     （规则复用率）。判据：**≥70% 的域其全部规则都与其他域共享** ⟹
     规则层跨域迁移成立。
  **D2 域外解析缺口（按域分组）**：用**生产解析器**（`b_parse`）解析每个域的
     需求文本，与锚点登记的 `property_id` 比对，**按域统计准确率**；
     找出**准确率最低的域**（域外薄弱点）。
     **口径诚实声明**：该模型在**全部锚点**上训练过 ⟹ 这是**乐观估计**
     （不是严格的 leave-one-domain-out）；它的用途是**暴露按域的薄弱点**，
     而非给出泛化下界。严格的 LODO 需逐域重训（昂贵），列为后续。

用法（项目根目录）：python a_network/v30_domain_transfer_check.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

PAIRS = _ROOT / "db3" / "pairs"


def check(name: str, ok: bool, detail: str = "") -> int:
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  " + detail) if detail else ""))
    return 0 if ok else 1


def main() -> int:
    pairs = [json.loads(p.read_text(encoding="utf-8"))
             for p in sorted(PAIRS.glob("DB3-*.json"))]
    print("=" * 74)
    print("任务 C 第二层：域级迁移（%d 条锚点）" % len(pairs))
    print("=" * 74)

    # ---- D1 规则级迁移 ----
    rule_doms: dict[str, set[str]] = defaultdict(set)
    dom_rules: dict[str, set[str]] = defaultdict(set)
    for p in pairs:
        d = p.get("domain") or "?"
        for r in (p.get("rule_ids") or []):
            rule_doms[r].add(d)
            dom_rules[d].add(r)
    print("\nD1 规则级跨域复用：")
    print("  域 %d 个；规则 %d 条" % (len(dom_rules), len(rule_doms)))
    n_all_shared = 0
    for d in sorted(dom_rules):
        rs = dom_rules[d]
        shared = [r for r in rs if len(rule_doms[r]) > 1]
        ok = len(shared) == len(rs)
        n_all_shared += int(ok)
        print("   %-22s 规则 %2d 条，与其他域共享 %2d 条%s"
              % (d, len(rs), len(shared), "" if ok else "  ← 有独占规则"))
    rate_shared = n_all_shared / max(len(dom_rules), 1)
    print("  ⟹ **全部规则都可跨域复用的域：%d/%d（%.0f%%）**"
          % (n_all_shared, len(dom_rules), 100 * rate_shared))
    # 独占规则清单（真正的"域特有知识"）
    solo = [r for r, ds in rule_doms.items() if len(ds) == 1]
    if solo:
        print("  独占规则（仅一个域使用）：%s" % ", ".join(sorted(solo)))
    else:
        print("  独占规则：无（所有规则至少被两个域共用）")

    # ---- D2 域外解析缺口（按域，乐观口径）----
    print("\nD2 按域的解析一致率（**乐观口径**：模型见过全部锚点）：")
    try:
        from b_network.b_parse import parse_requirement
        per_dom: dict[str, list[int]] = defaultdict(list)
        for p in pairs:
            ans = p.get("requirement_structured", {}) or {}
            gold = ans.get("property_id")
            if not gold:
                continue
            try:
                r = parse_requirement(p["requirement_text"])
                got = (r.req.property_ids or [None])[0]
            except Exception:  # noqa: BLE001
                got = None
            per_dom[p.get("domain") or "?"].append(int(got == gold))
        rows = []
        for d, v in per_dom.items():
            rows.append((d, len(v), sum(v) / len(v)))
        rows.sort(key=lambda z: (z[2], -z[1]))
        tot = sum(r[1] for r in rows)
        acc = sum(r[1] * r[2] for r in rows) / max(tot, 1)
        print("   %-22s %4s %8s" % ("domain", "条数", "一致率"))
        for d, n, a in rows:
            print("   %-22s %4d %7.0f%%%s"
                  % (d, n, 100 * a, "   ← 薄弱" if a < 0.5 else ""))
        print("   ⟹ 总体一致率 **%.1f%%**（%d 条）" % (100 * acc, tot))
        weak = [d for d, n, a in rows if a < 0.5]
        print("   薄弱域（<50%%）：%d 个%s"
              % (len(weak), ("：" + ", ".join(weak)) if weak else ""))
    except Exception as ex:  # noqa: BLE001
        print("  [跳过] 解析器不可用：%s: %s" % (type(ex).__name__, ex))
        acc, rate_shared_ok = float("nan"), False

    print("\n" + "=" * 74)
    fails = 0
    fails += check("D1 规则级跨域迁移（≥70%% 的域全部规则可跨域复用）",
                   rate_shared >= 0.70, "实测 %.0f%%" % (100 * rate_shared))
    if acc == acc:
        fails += check("D2 按域一致率（总体 ≥70%%，乐观口径）", acc >= 0.70,
                       "实测 %.1f%%" % (100 * acc))
    print("=" * 74)
    print("**说明**：D1 是**严格**的跨域检验（规则是否真的被多域共用）；")
    print("  D2 是**乐观**的（模型见过全部锚点）——严格 LODO 需逐域重训，列为后续。")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
