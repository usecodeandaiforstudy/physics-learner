# -*- coding: utf-8 -*-
"""DB3 复核后修正脚本（2026-09-29，第 37 条配套）。

三件事：
  1) 全部 34 个种子对 rules_version 0.1.0 → 0.2.0（与修订后的注册表一致，
     gate3 强制版本一致，见 validate_pair.py L228）；
  2) A 类注释级修正（不改任何 q 向量数值与 worked_example 值）：
     - DB3-0020：expr 括号内 β 等价式修正（原式符号与分母均错）；
     - DB3-0016：「1 bit」→「ln 2 nats」表述；
     - DB3-0021/0022：a 符号约定注明（双阱要求 a<0，示例按 |a| 理解）。
  3) 对全部对重跑三道机器门（复用 validate_pair 的 gate1/2/3），确认 34/34 通过。

用法（项目根目录）：
    python db3/post_review_fixes.py
"""

import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
PAIRS_DIR = os.path.join(BASE, "pairs")
NEW_VERSION = "0.2.0"

# A 类文本修正（旧串 → 新串，精确匹配）
TEXT_FIXES = {
    "DB3-0020": [
        ("\"expr\": \"(T_c - T_work)/T_c = (beta_c - beta)/beta_c\"",
         "\"expr\": \"(T_c - T_work)/T_c（β 空间距离为 (β−β_c)/β_c = "
         "(T_c−T)/T_work，两者勿混；本对取温度约化距离）\""),
    ],
    "DB3-0016": [
        ("\"expr\": \">= ln(2)（至少 1 bit 受保护信息）\"",
         "\"expr\": \">= ln(2)（≥ ln 2 nats 受保护信息）\""),
    ],
    "DB3-0021": [
        ("\"note\": \"Landau 双阱：ΔF‡ = a²/(4·b·k_B T)\"",
         "\"note\": \"Landau 双阱（a<0）：ΔF‡ = a²/(4·b·k_B T)；"
         "示例 a 按 |a| 理解\""),
    ],
    "DB3-0022": [
        ("\"note\": \"ΔF‡/k_B T ≥ 40 为磁存储行业惯例下限\"",
         "\"note\": \"ΔF‡/k_B T ≥ 40 为磁存储行业惯例下限；a 按 |a| 理解\""),
    ],
}


def main() -> int:
    paths = sorted(glob.glob(os.path.join(PAIRS_DIR, "DB3-*.json")))
    n_bump = n_fix = 0
    for path in paths:
        raw = open(path, encoding="utf-8").read()
        rec = json.loads(raw)
        pid = rec["pair_id"]

        if rec.get("rules_version") != NEW_VERSION:
            raw = raw.replace('"rules_version": "%s"' % rec.get("rules_version"),
                              '"rules_version": "%s"' % NEW_VERSION)
            n_bump += 1

        for old, new in TEXT_FIXES.get(pid, []):
            if old in raw:
                raw = raw.replace(old, new)
                n_fix += 1
            else:
                print("[WARN] %s: 未找到待修正文本 %s" % (pid, old[:40]))

        if '"rules_version": "%s"' % NEW_VERSION not in raw:
            print("[FAIL] %s: 版本未更新" % pid)
            return 1
        open(path, "w", encoding="utf-8").write(raw)

    print("版本升级：%d/%d；文本修正：%d 处" % (n_bump, len(paths), n_fix))

    # 三道门复验
    sys.path.insert(0, BASE)
    from validate_pair import gate1_schema, gate2_rules, gate3_provenance, load_registry

    rules, rv = load_registry()
    errs = []
    for path in paths:
        rec = json.load(open(path, encoding="utf-8"))
        gate1_schema(rec, errs)
        gate2_rules(rec, rules, errs)
        gate3_provenance(rec, rv, errs)
    if errs:
        print("===== 复验失败（%d 项） =====" % len(errs))
        for e in errs[:20]:
            print(" ", e)
        return 1
    print("===== 复验通过：%d/%d（注册表 %s） =====" % (len(paths), len(paths), rv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
