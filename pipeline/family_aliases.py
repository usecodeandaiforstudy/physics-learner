# -*- coding: utf-8 -*-
"""族名别名映射（**单一真源**，2026-10-02，第 129 条）。

**为什么需要**：项目里同一族有两套名字——
  · `a_network/req_env.py` 与 `pipeline/design_contract.py` 用**"域-族"组合名**
    （`control_ou` / `acoustics_rayleigh` / `materials_diffusion`），
    因为它们指的是"**某域下的**某族实例"；
  · `db1` / `db2` 用**纯族名**（`ou` / `rayleigh` / `diffusion`）。
⟹ 任何跨模块按族名查询都会失败。**本会话因此踩坑两次**
  （`design_contract` 的 `param_bound_for` 查不到、`inverse_fisher` 的 I3b 崩），
  而且各脚本会各自维护一份映射——**那正是反复出错的模式**。
⟹ 本模块把映射**集中一处**，并提供：
     `to_db1(name)` / `to_env(name)` / `check_aliases()`（自检：别名必须都能解析）

**本模块是叶子模块（不 import 项目内任何东西）**⟹ 可被 `pipeline/` 与
`a_network/` 双向安全导入，无循环依赖。
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# 域-族组合名（env/契约用） ↔ DB1/DB2 纯族名
DOMAIN_TO_DB1: dict[str, str] = {
    "acoustics_rayleigh": "rayleigh",
    "control_ou": "ou",
    "materials_diffusion": "diffusion",
    # 名字相同的族（显式登记，便于自检时统一遍历）
    "two_level": "two_level",
    "landau": "landau",
    "elastic": "elastic",
    "weibull": "weibull",
    "gamma": "gamma",
    "negbin": "negbin",
    "geometric": "geometric",
}

_DB1_TO_DOMAIN = {v: k for k, v in DOMAIN_TO_DB1.items()}


def to_db1(name: str) -> str:
    """任意一套名字 → **DB1/DB2 纯族名**（未知名字原样返回，不猜）。"""
    if name in DOMAIN_TO_DB1:
        return DOMAIN_TO_DB1[name]
    return name


def to_env(name: str) -> str:
    """任意一套名字 → **env/契约的域-族组合名**（未知名字原样返回）。"""
    if name in _DB1_TO_DOMAIN:
        return _DB1_TO_DOMAIN[name]
    return name


def db1_families() -> set[str]:
    """DB1 条目里实际出现的 family 集合（用于自检）。"""
    out: set[str] = set()
    d = _ROOT / "db1" / "entries"
    if not d.exists():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if e.get("family"):
            out.add(e["family"])
    return out


def check_aliases() -> dict:
    """**自检**：每个别名必须指向一个真实存在的 DB1 族（防映射腐烂）。"""
    real = db1_families()
    bad = {k: v for k, v in DOMAIN_TO_DB1.items() if v not in real}
    dup = {}
    for k, v in DOMAIN_TO_DB1.items():
        dup.setdefault(v, []).append(k)
    multi = {v: ks for v, ks in dup.items() if len(ks) > 1}
    return {"n_alias": len(DOMAIN_TO_DB1), "n_db1_family": len(real),
            "unresolved": bad, "multi_source": multi,
            "ok": not bad}


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    r = check_aliases()
    print("族名别名自检：别名 %d 条；DB1 族 %d 个" % (r["n_alias"],
                                                r["n_db1_family"]))
    print("  无法解析（别名指向不存在的 DB1 族）：%s"
          % (r["unresolved"] or "无 ✓"))
    print("  多别名指向同一族：%s" % (r["multi_source"] or "无 ✓"))
    print("  env 名 → DB1 名：" + "，".join(
        "%s→%s" % (k, v) for k, v in sorted(DOMAIN_TO_DB1.items())))
    print("结论：%s" % ("**通过**（别名全部可解析）" if r["ok"]
                        else "**未通过**（见上）"))
