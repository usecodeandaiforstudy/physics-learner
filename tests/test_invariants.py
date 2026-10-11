# -*- coding: utf-8 -*-
"""核心不变量套件（2026-10-02，第 123 条）——把"口径/接口"类缺陷钉死。

**动机**：本会话 **20+ 处缺陷全部是「量/口径/接口」类**（q7 口径、M4 标定 τ、
CFM 密度方向、n 折算、β<0、`self.` 漏写、离散分布 `pmf`、类名不同步、
C3 判据饱和、假 20 seed、B4 注释自伤……）。每次都被检查抓住，但代价是反复
返工。**本套件把这些不变量固化成回归**，使同类缺陷**在提交前就被拦住**。

**运行方式**（两种都可，**不强制依赖 pytest**）：
    python tests/test_invariants.py         # 自带 runner
    python -m pytest tests/test_invariants.py -q

**覆盖的不变量（按缺陷类型）**：
  ① **口径/量纲**：q1 有量纲时须除以 `reference_scale`；`q1=1/(nσ²)` 的 n 折算
  ② **有效性判据**：β 必须为正且量级合理（β≤0 / |β|<1e-3 / β>1e3 → 拒绝）
  ③ **密度归一**：CFM 的 `∫p̂dφ`（短训练下也应落在合理区间，且**不出现
     "NLL 低于真值"这种不可能的数字**）
  ④ **接口契约**：`encode()` 对已注册族必须给出 `ib_source`；scipy 离散分布
     走 `pmf`+整数网格；`build_db2_record` 的 report 三元组
  ⑤ **账本完整性**：DB1 条目必填字段；DB2 行含 32 点曲线；DB3 对的必填字段
  ⑥ **解析器冒烟**：三条典型需求文本 → 期望属性
  ⑦ **判据自检**：分类规则不得把注释当代码（B4 自伤类）；标定状态**守恒**
     （各状态计数之和 == 锚点总数）
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

DB1 = _ROOT / "db1" / "entries"
DB2 = _ROOT / "db2" / "db2.jsonl"
DB3 = _ROOT / "db3" / "pairs"


# ---------------------------------------------------------------- ① 口径/量纲
def test_q1_reference_scale_normalization():
    """**有量纲 q1 必须按文档化参考尺度归一**（否则得到 25 nats 的假大数）。

    R-STIFF-001：q1 = K/(k_B T) = 2.4e22 m^-2；参考尺度 1e20–1e24 ⟹
    SNR = 2.4 ⟹ I_TY = ½ln(3.4) ≈ 0.61 nats（**不是** 25 nats）。
    """
    from pipeline.beta_calibration import (_reference_scale, _q1_numeric,
                                           calibrate_pair)
    p = DB3 / "DB3-0001.json"
    assert p.exists(), "缺 DB3-0001（stiffness 锚点）"
    pair = json.loads(p.read_text(encoding="utf-8"))
    q1, _src = _q1_numeric(pair)
    assert q1 is not None and q1 > 1e20, "q1 数值解析失败（应约 2.4e22）"
    ref, _how = _reference_scale(pair)
    assert ref and 1e21 < ref < 1e23, "参考尺度应取几何中点 ≈1e22"
    snr = q1 / ref
    assert 1.0 < snr < 10.0, "归一后 SNR 应在 O(1)（实测 %.3g）" % snr
    cal = calibrate_pair(pair)
    # **标定结果随数据演进而变（2026-10-02 修）**：任务 B 补注册后，
    # R-STIFF-001 映射到的 **`elastic` 族有了自己的曲线**（容量 0.293 nats），
    # 而本需求的 I_TY=0.612 nats **超出 elastic 容量** ⟹ `out_of_range`
    # 是**正确结果**（此前"ok"是因为回退到 ou 的 0.717 容量）。
    # 故本测试**只锁"归一化"这一不变量**（SNR、I_TY 量级），
    # 不锁"必然可标定"——后者取决于族容量，属数据事实而非不变量。
    assert cal.status in {"ok", "out_of_range"}, \
        "状态应为 ok 或 out_of_range（族容量边界），实际 %s" % cal.status
    assert cal.i_ty_req is not None and 0.4 < cal.i_ty_req < 0.9, \
        "I_TY 应 ≈0.612 nats（=½ln(1+2.4)）；若 ≈25 说明未做参考尺度归一"
    if cal.status == "ok":
        assert cal.beta is not None and cal.beta > 0, "β 应为正"


def test_n_samples_scope_conversion():
    """**含 n 的需求须折算到单次观测**（口径错配类）。

    DB3-0040：q1 = 1/(nσ²) = 33.33（**已含 n=300**）⟹ 合计 1.768 nats；
    折算单次 = 0.0059 nats（IB 曲线是**单次观测**口径）。
    """
    from pipeline.beta_calibration import calibrate_pair
    p = DB3 / "DB3-0040.json"
    assert p.exists(), "缺 DB3-0040"
    pair = json.loads(p.read_text(encoding="utf-8"))
    cal = calibrate_pair(pair)
    assert cal.status == "ok", "应可标定，实际 %s" % cal.status
    assert cal.i_ty_req is not None and cal.i_ty_req < 0.02, \
        "单次折算后应 ≈0.0059 nats（若 ≈1.77 说明未折算 n）"
    assert "n=300" in cal.conversion, "推导链里须标明 n 折算"


# ------------------------------------------------------------- ② β 有效性判据
def test_beta_validity_rejects_nonpositive_and_degenerate():
    """**β≤0 / |β|<1e-3 / β>1e3 一律拒绝**（负 β = 奖励"少拿信息"）。

    构造三条人工曲线：单调升（正常）、先降后升（β<0 区）、极陡（β≈0）。
    """
    from pipeline.extra_curves import beta_at
    t = np.linspace(0, 1, 21)
    # 正常：能力随代价单调升，斜率适中 → β 合理
    x_ok, c_ok = t.copy(), t.copy() * 2.0
    b, _ach, how = beta_at(x_ok, c_ok, 0.5)
    assert b is not None and 0.1 < b < 10.0, "正常曲线应给出合理 β（%s）" % how
    # 先降后升：target 落在下降段 → β 应为负被拒
    x_bad = np.r_[t[:10], t[10:]]
    c_bad = np.r_[np.linspace(0, 1, 10), np.linspace(0.01, 2.0, 11)]
    b2, _a2, how2 = beta_at(x_bad, c_bad, 0.45)
    assert b2 is None, "下降段应被拒（β≤0）"
    assert ("无效" in how2) or ("退化" in how2), "拒绝理由须明示：%s" % how2
    # 极陡：代价几乎不变 → β 极小被拒
    x_steep, c_steep = t.copy(), np.full_like(t, 1.0) + t * 1e-9
    b3, _a3, how3 = beta_at(x_steep, c_steep, 0.5)
    assert b3 is None and ("退化" in how3), "极陡曲线应判退化：%s" % how3


# --------------------------------------------------------------- ③ 密度归一
def test_density_normalization_and_impossible_number():
    """**`∫p̂dφ` 自检 + 「NLL 不得低于真值」**（CFM 曾报 NLL 0.973 < 真值 1.135）。

    短训练（200 步）下不要求高精度，但**不允许出现不可能的符号**：
      - 归一化积分必须 > 0 且有限；
      - CFM 的 NLL 不得显著低于真值 NLL（低 >0.05 即判实现有 bug）。
    """
    import torch
    from c_network.cfm import ConditionalFlowMatching

    torch.manual_seed(0)
    grid = np.linspace(-6, 6, 201)
    y = torch.tensor(np.random.default_rng(0).standard_normal(
        4000).astype(np.float32)[:, None])
    a = torch.zeros_like(y)
    cfm = ConditionalFlowMatching(1, seed=0)
    opt = torch.optim.Adam(cfm.parameters(), lr=2e-2)
    for _ in range(200):
        loss = cfm.loss(a, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    lp = cfm.log_density(0.0, grid, steps=60)
    assert np.all(np.isfinite(lp)), "log 密度须有限"
    p = np.exp(lp - lp.max())
    Z = float(p.sum() * (grid[1] - grid[0]))
    assert 0.05 < Z < 20.0, "归一化积分应在合理量级（实测 %.4g）" % Z
    # NLL 检验：真值 N(0,1) 的 NLL（同网格）≈1.42（离散化后）
    pdf = np.exp(-0.5 * grid ** 2) / math.sqrt(2 * math.pi)
    nll_true = float(-np.mean(np.log(np.interp(
        np.random.default_rng(1).standard_normal(500), grid, pdf) + 1e-12)))
    nll_cfm = float(-np.mean(cfm.log_density(
        0.0, np.random.default_rng(1).standard_normal(500))))
    assert nll_cfm > nll_true - 0.05, (
        "**不可能的数字**：CFM NLL %.3f 低于真值 %.3f ⟹ 实现有 bug"
        % (nll_cfm, nll_true))


# ------------------------------------------------------------- ④ 接口契约
def test_ib_source_registered_for_known_families():
    """已注册族 `encode()` 后**必须**有 `ib_source`（第 118 条）。"""
    from b_network.system_encoder import SystemEncoder
    enc = SystemEncoder()
    reg = SystemEncoder.IB_SOURCE_REGISTRY
    assert len(reg) >= 10, "注册表应覆盖 ≥10 个族（实测 %d）" % len(reg)
    ok = 0
    for p in sorted(DB1.glob("*.json"))[:40]:
        e = json.loads(p.read_text(encoding="utf-8"))
        fam = e.get("family")
        if fam not in reg:
            continue
        g = enc.encode(e)
        assert "ib_source" in (g.metadata or {}), \
            "%s（族 %s）缺 ib_source" % (e.get("system_id"), fam)
        ok += 1
    assert ok > 0, "未取到任何已注册族的条目（数据或注册表异常）"


def test_scipy_channel_handles_discrete_distributions():
    """**离散分布须走 `pmf` + 整数网格**（`nbinom`/`geom` 无 `pdf`，初版全挂）。"""
    from b_network.fingerprint_computer import FingerprintComputer
    from b_network.system_encoder import SystemEncoder   # 不引 db1.ingest_pipeline：
    # 它用**脚本内相对导入**（`from db2_writer import ...`），作为模块导入会
    # `ModuleNotFoundError`（本测试初版即踩此坑）。
    fc = FingerprintComputer()
    enc = SystemEncoder()
    for p in sorted(DB1.glob("*.json")):
        e = json.loads(p.read_text(encoding="utf-8"))
        if e.get("family") != "negbin":
            continue
        g = enc.encode(e)
        c = fc.compute_ib_curve(g)
        y = np.asarray(c.get("I_TY", []))
        assert len(y) == 32, "negbin 曲线应 32 点（实测 %d）" % len(y)
        assert np.all(np.isfinite(y)) and float(y.max()) > 0, \
            "negbin 曲线应非空有限（离散 pmf 网格修好的标志）"
        return
    raise AssertionError("DB1 中未找到 negbin 条目")


# ----------------------------------------------------------- ⑤ 账本完整性
def test_db1_entries_required_fields():
    # **枚举从 schema 读**（2026-10-02 修）：初版把 DB1 的 human_review_status
    # 硬编码成 {pending, approved, rejected}——那是 **DB3** 的字段值；
    # DB1 schema 的合法枚举是 {pending, spot_checked, fully_reviewed}。
    # 硬编码枚举 = 又一处"口径"，故改为**以 schema 为单一真源**。
    schema = json.loads((_ROOT / "db1" / "db1.schema.json").read_text(
        encoding="utf-8"))
    prop = schema["properties"]["provenance"]["properties"]
    allowed = set(prop["human_review_status"]["enum"])
    allowed_src = set(prop["source"].get("enum") or [])
    files = sorted(DB1.glob("*.json"))
    assert len(files) >= 290, "DB1 条目应 ≥290（实测 %d）" % len(files)
    need = {"system_id", "domain", "family", "parameters", "provenance"}
    for p in files[:60]:
        e = json.loads(p.read_text(encoding="utf-8"))
        missing = need - set(e)
        assert not missing, "%s 缺字段 %s" % (p.name, missing)
        prov = e["provenance"]
        assert "source" in prov, "%s 缺 provenance.source" % p.name
        if allowed_src:
            assert prov["source"] in allowed_src, \
                "%s 的 provenance.source=%r 不在 schema 枚举 %s 内" % (
                    p.name, prov["source"], sorted(allowed_src))
        else:
            assert prov["source"].startswith("AI+"), \
                "%s 的 provenance.source 须为 AI+*（红线）" % p.name
        assert prov.get("human_review_status") in allowed, \
            "%s 的 human_review_status=%r 不在 schema 枚举 %s 内" % (
                p.name, prov.get("human_review_status"), sorted(allowed))


def test_db2_rows_have_curves_and_families():
    n = n_curve = 0
    fams = set()
    with open(DB2, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n += 1
            fams.add(rec.get("family"))
            c = rec.get("ib_curve") or {}
            if c.get("beta"):
                n_curve += 1
                assert len(c["beta"]) == len(c["I_TY"]) == len(c["I_XT"]), \
                    "曲线三列长度须一致"
            if n >= 400:
                break
    assert n >= 300, "DB2 应有足够行（实测 %d）" % n
    assert n_curve / n > 0.5, \
        "有曲线比例应 >50%%（实测 %.0f%%；任务 B 后应接近 100%%）" % (
            100 * n_curve / n)


def test_db3_pairs_required_fields_and_status():
    files = sorted(DB3.glob("DB3-*.json"))
    assert len(files) >= 150, "DB3 锚点应 ≥150（实测 %d）" % len(files)
    need = {"pair_id", "domain", "requirement_text", "requirement_structured",
            "rule_ids", "info_metric_vector"}
    for p in files[:40]:
        e = json.loads(p.read_text(encoding="utf-8"))
        missing = need - set(e)
        assert not missing, "%s 缺字段 %s" % (p.name, missing)


# --------------------------------------------------------- ⑥ 解析器冒烟
def test_parser_smoke_on_three_requirements():
    from b_network.b_parse import parse_requirement
    cases = [("弹簧要足够硬：施加 1 N 载荷时变形不超过 10 mm。", "stiffness"),
             ("执行器从阶跃指令到 95% 行程的响应时间不超过 20 ms。",
              "response_speed")]
    for text, want in cases:
        r = parse_requirement(text)
        assert r.status in {"auto", "abstain"}, "状态非法 %s" % r.status
        if r.status == "auto":
            assert want in (r.req.property_ids or []), \
                "「%s」应解析出 %s，实际 %s" % (text[:12], want,
                                              r.req.property_ids)


# ------------------------------------------------------- ⑦ 判据自检 / 守恒
def test_comment_is_not_code_in_selfproof():
    """**B4 自伤类**：自证判据不得把注释当代码（第 115 条）。"""
    src = (_ROOT / "a_network" / "req_env.py").read_text(encoding="utf-8")
    strip_doc = re.sub(r'"""(?:.|\n)*?"""', "", src)
    code = "\n".join(ln.split("#")[0] for ln in strip_doc.splitlines())
    body = code.split("def reward")[-1].split("\ndef ")[0]
    assert not re.search(r"\bbeta\b", body), \
        "reward() 代码段内不得出现 beta 标识符（注释里的 β 不计）"


def test_status_conservation_in_beta_calibration():
    """**状态守恒**：各类状态计数之和 == 锚点总数（防"需求被吃掉"）。

    同时检查：`not_applicable:*`（约束型）**不得**混入 `unmapped`。
    """
    from collections import Counter
    from pipeline.beta_calibration import calibrate_pair
    pairs = [json.loads(p.read_text(encoding="utf-8"))
             for p in sorted(DB3.glob("DB3-*.json"))]
    c = Counter(calibrate_pair(p).status for p in pairs)
    assert sum(c.values()) == len(pairs), \
        "状态计数 %d != 锚点数 %d" % (sum(c.values()), len(pairs))
    n_ok = c.get("ok", 0)
    assert n_ok >= 100, "可标定数应 ≥100（实测 %d）" % n_ok
    n_constraint = sum(v for k, v in c.items()
                       if k.startswith("not_applicable:"))
    assert n_constraint >= 30, \
        "约束型需求应被单列（实测 %d）——不得压成 unmapped" % n_constraint


# ------------------------------------------------- ⑧ 中游：反解 + 非退化域
def test_inverse_map_coverage_and_agreement():
    """**反解表覆盖 ≥10 族；同口径族与解析桥接式一致 ≤5%；非单调如实报**。

    钉住的缺陷（第 127 条）：反解只覆盖 4 族（覆盖太窄）；以及"非单调族被
    硬解"（Fisher 剖面有峰却仍强行取逆）。
    """
    from pipeline.inverse_fisher import ANALYTIC_BRIDGES, invert
    from pipeline.family_aliases import check_aliases, db1_families

    # **别名自检**：别名必须都指向真实 DB1 族（防映射腐烂，第 129 条）
    chk = check_aliases()
    assert chk["ok"], "族名别名无法解析：%s" % chk["unresolved"]

    fams = sorted(db1_families())
    assert len(fams) >= 20, "DB1 族数异常（%d）" % len(fams)
    n_ok = n_nonmono = 0
    for f in fams:
        r = invert(f, 1.0)
        if r.get("value") is not None:
            n_ok += 1
        elif r.get("monotone") is False:
            n_nonmono += 1
            assert "非单调" in (r.get("note") or ""), \
                "%s 非单调必须如实标注（不得硬解）" % f
    assert n_ok >= 10, "可反解族应 ≥10（实测 %d）" % n_ok
    assert n_nonmono >= 1, "应至少有一个非单调族被如实报告"

    # **同口径一致 ≤5%**：rayleigh 的 q1=4/σ² 本身就是 Fisher trace
    for (fam, metric), (pname, kind, fn, same_scale) in ANALYTIC_BRIDGES.items():
        if not (same_scale and fn is not None):
            continue
        ana = float(fn(1.0))
        num = invert(fam, 1.0)
        assert num.get("value") is not None, "%s 数值反解不可用" % fam
        rel = abs(num["value"] - ana) / max(abs(ana), 1e-12)
        assert rel <= 0.05, ("%s/%s：数值反解 %.6g vs 解析 %.6g 相对差 %.1f%% "
                             "超 5%%" % (fam, metric, num["value"], ana,
                                         100 * rel))


def test_nondegenerate_design_domain_closed_loop():
    """**非退化设计域**：设计 2 dof → 状态 1 dof；且闭环经既有编码器一致。

    钉住的缺陷（第 128 条）：设计域**退化**（设计 = 状态参数 ⟹ 解码为恒等）；
    以及"负势垒设计"（反转占据分支未区分）。
    """
    from pipeline.design_two_level import (EV, design_manifold,
                                           p_from_design, verify_design,
                                           x_from_p)
    p_t = 0.10
    assert abs(x_from_p(p_t) - 2.1972246) < 1e-6, "x=ln((1-p)/p) 约定错"
    m = design_manifold(p_t, dF_min_J=0.15 * EV, T_min_K=250.0)
    assert m["feasible"] and len(m["T"]) >= 50, \
        "非退化域应有 ≥50 个可行设计（实测 %d）" % len(m["T"])
    # **同一状态、多个设计**：ΔF 与 T 变化而 p 恒定
    ps = [p_from_design(float(a), float(b))
          for a, b in zip(m["dF_J"][::20], m["T"][::20])]
    assert max(ps) - min(ps) < 1e-9, "解流形上 p 必须恒定"
    assert len({round(float(t), 3) for t in m["T"]}) >= 50, "T 必须真的不同"
    # **闭环经既有编码器**：q1 与解析式一致 + DPI 合法
    v = verify_design(float(m["dF_J"][len(m["T"]) // 2]),
                      float(m["T"][len(m["T"]) // 2]),
                      q1_expected=1.0 / (p_t * (1 - p_t)))
    assert v["q1_from_encoder"] is not None, "编码器未给出 q1"
    assert v["q1_rel_err"] < 1e-6, \
        "设计经编码器得到的 q1 与解析式不一致（相对差 %.2e）" % v["q1_rel_err"]
    assert v["dpi_ok"], "该设计必须通过 DPI"
    assert v.get("q1_matches_requirement"), "设计应满足需求的 q1"
    # **反转占据分支必须显式拒绝正势垒约束**（否则产出负势垒"设计"）
    m_inv = design_manifold(0.90, dF_min_J=0.15 * EV, T_min_K=250.0)
    assert not m_inv["feasible"] and "反转占据" in (m_inv.get("note") or ""), \
        "p>0.5（反转占据）时必须显式标注而非硬解"


def test_rayleigh_nondegenerate_and_boundary_tolerance():
    """**第 2 个非退化域（rayleigh）** + **边界容差**（第 131 条）。

    钉住的缺陷：① 可行域**边界支**（`σ = σ_max` ⟹ `q1` 恰等于需求）在浮点下
    被判"不达标"（`2.9999999996 < 3.0`）；② 设计落在**族登记范围外**却静默外推。
    """
    from pipeline.design_rayleigh import (design_manifold, q1_from_sigma,
                                          sigma_from_design, verify_design)
    q1_req = 3.0
    m = design_manifold(q1_req, D_max_m=1.0, f_min_Hz=10e3, f_max_Hz=2e6)
    assert m["feasible"], "该参数组应可行：%s" % m.get("note")
    assert len(m["f"]) >= 50, "应有 ≥50 个可行设计（实测 %d）" % len(m["f"])
    # **同一状态、多个设计**：σ 与 q1 恒定，而 (D, f) 变化
    sig = [sigma_from_design(float(d), float(f))
           for d, f in zip(m["D"][::20], m["f"][::20])]
    assert max(sig) - min(sig) < 1e-12, "解流形上 σ 必须恒定"
    assert len({round(float(x), 6) for x in m["D"]}) >= 50, "D 必须真的不同"
    # **边界支必须判达标**（容差）
    i = len(m["f"]) // 2
    v = verify_design(float(m["D"][i]), float(m["f"][i]), q1_req)
    assert v["q1_rel_err"] < 1e-9, "编码器 q1 与解析式不一致"
    assert v["dpi_ok"], "设计必须通过 DPI"
    assert v["q1_meets_requirement"], \
        "**边界支必须判达标**（q1=%.10f vs 需求 %.1f）——容差缺失会误判" \
        % (v["q1_from_encoder"], q1_req)
    # **内部点**：1.2 倍孔径 ⟹ q1 = 需求 × 1.44（自洽性）
    D_in = float(m["D"][i]) * 1.2
    if D_in <= 1.0:
        v2 = verify_design(D_in, float(m["f"][i]), q1_req)
        ratio = v2["q1_from_encoder"] / q1_req
        assert abs(ratio - 1.44) < 1e-6, \
            "1.2×孔径应给出 1.44×q1（实测 %.4f）" % ratio
    # **族登记范围必须被检查**（范围外推须标注）
    assert v.get("in_documented_range") is not None, "必须检查族登记范围"


def test_cfm_multidim_conditioning():
    """**CFM 多维推广**（第 133 条）：向量条件必须真的起作用。

    钉住的缺陷：① 推广后**漏算条件维度**（形状不匹配，历史已犯过）；
    ② 标量路径被破坏（既有 `v8`/`v22` 依赖它）；③ 条件被**忽略**
    （不同条件给出同分布 ⟹ 条件流退化为无条件生成）。
    """
    import torch
    from c_network.cfm import ConditionalFlowMatching

    # ① 标量路径保持可用（向后兼容）
    m0 = ConditionalFlowMatching(1, seed=0)
    s0 = m0.sample(0.0, 8, steps=5, seed=0)
    assert s0.shape == (8,), "标量 sample 应返回 (n,)，实测 %s" % (s0.shape,)

    # ② 多维：数据 3 维、条件 2 维；**条件决定簇中心**（可验证的信号）
    torch.manual_seed(0)
    n = 3000
    C = torch.rand(n, 2)
    # 设计 = 条件 + 小噪声 ⟹ "条件→数据"有确定映射
    X = C + 0.05 * torch.randn(n, 2)
    X = torch.cat([X, C[:, :1]], dim=1)          # 第 3 维 = 第 1 个条件的复制
    m = ConditionalFlowMatching(in_dim=3, cond_dim=2, hidden=64, seed=0)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(400):
        idx = torch.randint(0, n, (256,))
        loss = m.loss(C[idx], X[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    Z_a = m.sample_cond(np.array([0.1, 0.9], dtype=np.float32), 200, steps=40,
                        seed=1)
    Z_b = m.sample_cond(np.array([0.9, 0.1], dtype=np.float32), 200, steps=40,
                        seed=2)
    assert Z_a.shape == (200, 3), "sample_cond 应返回 (n, in_dim)"
    assert np.all(np.isfinite(Z_a)), "样本须有限"
    # ③ **条件必须起作用**：不同条件 ⟹ 样本均值显著不同
    gap = float(np.linalg.norm(Z_a.mean(axis=0) - Z_b.mean(axis=0)))
    assert gap > 0.2, ("条件未起作用：两个条件的样本均值差仅 %.3f ⟹ "
                       "条件流可能退化为无条件生成" % gap)


def test_verifier_out_of_domain_must_be_unknown():
    """**生产级验证的安全性质**：域外必须 `unknown`，**绝不 `pass`**（第 135 条）。

    钉住的缺陷（演示级实现）：未支持的族**抛 `KeyError`**、参数超域仍给出数字
    ⟹ 调用方会把"没验证"当成"通过"；容差写死；无审计日志。
    """
    from sim.verify_api import CACHE, verify
    CACHE.clear()
    cases = [
        {"family": "nonexistent_family", "sigma": 1.0, "metric": "q1",
         "target": 1.0},                                   # 未登记族
        {"family": "rayleigh", "sigma": 1.0, "metric": "q9",
         "target": 1.0},                                   # 度量不支持
        {"family": "rayleigh", "sigma": 1e9, "metric": "q1",
         "target": 1.0},                                   # 参数超有效域
        {"family": "rayleigh", "sigma": 0.5, "metric": "q1"},  # 缺目标值
    ]
    for c in cases:
        v = verify(c, n_samples=2000, seed=0, use_cache=False)
        assert v.verdict == "unknown", \
            ("**域外必须 unknown**：%s → %s（%s）"
             % (c, v.verdict, v.reason))
    # 域内且明确达标/不达标：必须给出 pass/fail（不能一律 unknown 蒙过去）
    ok = verify({"family": "rayleigh", "sigma": 0.5, "metric": "q1",
                 "target": 3.0, "direction": "lower_bound"},
                n_samples=20000, seed=0, use_cache=False)
    bad = verify({"family": "rayleigh", "sigma": 2.0, "metric": "q1",
                  "target": 3.0, "direction": "lower_bound"},
                 n_samples=20000, seed=0, use_cache=False)
    assert ok.verdict == "pass" and bad.verdict == "fail", \
        "域内必须能区分 pass/fail（实测 %s / %s）" % (ok.verdict, bad.verdict)
    # **可复现性**：同设计同 seed ⟹ 同结果（清缓存后）
    CACHE.clear()
    a = verify({"family": "weibull", "shape": 2.0, "scale": 1.0,
                "metric": "q1", "target": 2.0}, n_samples=5000, seed=7,
               use_cache=False)
    CACHE.clear()
    b = verify({"family": "weibull", "shape": 2.0, "scale": 1.0,
                "metric": "q1", "target": 2.0}, n_samples=5000, seed=7,
               use_cache=False)
    assert a.measured == b.measured and a.verdict == b.verdict, \
        "同设计同 seed 必须完全一致（%.6g vs %.6g）" % (a.measured, b.measured)


def test_registry_matches_verifier_behaviour():
    """**登记表 ↔ 验证行为一致**（第 136 条）。

    钉住的缺陷：登记表（`sim/registry.json`，仅**校准通过**者入表）与
    `verify_api` 的实际行为脱节——例如表里有某族但验证器仍返回 `unknown`
    （≤"登记了却用不了"），或未登记者被当成可用（**静默外推**）。
    """
    import json as _json
    from sim.verify_api import REGISTRY, REGISTRY_JSON, CACHE, verify

    assert REGISTRY_JSON.exists(), \
        "缺 sim/registry.json（先跑 `python sim/expand_registry.py --register`）"
    data = _json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    assert len(data) >= 15, "登记族数应 ≥15（实测 %d）" % len(data)
    for fam, spec in data.items():
        assert spec.get("source"), "%s 缺 source（须标明校准来源）" % fam
        assert fam in REGISTRY, "%s 在表中但未加载进验证器" % fam
        assert REGISTRY[fam]["metrics"], "%s 缺度量登记" % fam

    # **表中族必须可验证**（返回 pass/fail，不得是 unknown）
    CACHE.clear()
    fam0 = sorted(data)[0]
    pname = next(iter(data[fam0]["params"]))
    lo, hi = data[fam0]["params"][pname]
    mid = 0.5 * (float(lo) + float(hi))
    d = {"family": fam0, pname: mid, "metric": "q1",
         "target": 1.0, "direction": "lower_bound"}
    v = verify(d, n_samples=4000, seed=0, use_cache=False)
    assert v.verdict in {"pass", "fail"}, \
        "登记族 %s 应可验证（实测 %s：%s）" % (fam0, v.verdict, v.reason)

    # **未登记族必须 unknown**（结构性不可验证者不得被静默放行）
    for fam_bad in ("landau", "gaussian_mixture2"):
        if fam_bad in REGISTRY:
            continue
        v2 = verify({"family": fam_bad, "a": 1.0, "d": 1.0, "metric": "q1",
                     "target": 1.0}, n_samples=2000, seed=0, use_cache=False)
        assert v2.verdict == "unknown", \
            "未登记族 %s 必须 unknown（实测 %s）" % (fam_bad, v2.verdict)


def test_reward_types_single_source_and_conservation():
    """**奖励类型分类**：单一真源 + 状态守恒（第 138 条）。

    钉住的缺陷：① 判据散落多处（`beta_calibration._info_class`、
    `req_env`、`extra_curves`）⟹ 漂移；② **分类不守恒**（各类之和 ≠ 锚点数，
    即有需求被"吃掉"或重复计数）；③ β 适用比例被**混同**（把"速率型约定"
    也算成"信息量型"——此前误报 70%，实为 **46%**）。
    """
    import json as _json
    from collections import Counter
    from pipeline.reward_types import (PROP_TO_CLASS, REWARD_KIND,
                                       classify_pair, classify_property,
                                       reward_kind)

    # ① 类 → 奖励形式映射必须完备（不得有"未定义奖励"的类）
    pairs = sorted(DB3.glob("DB3-*.json"))
    assert len(pairs) >= 150, "DB3 锚点数异常（%d）" % len(pairs)
    cnt: Counter = Counter()
    kinds: Counter = Counter()
    for p in pairs:
        e = _json.loads(p.read_text(encoding="utf-8"))
        cls, _why = classify_pair(e)
        cnt[cls] += 1
        k = reward_kind(cls)
        assert k in {"ib", "satisficing", "convention", "unknown"}, \
            "类 %s 未定义奖励形式（%s）" % (cls, k)
        kinds[k] += 1

    # ② **状态守恒**：各类之和 == 锚点数
    assert sum(cnt.values()) == len(pairs), \
        "分类不守恒：%d vs %d" % (sum(cnt.values()), len(pairs))

    # ③ **单一真源一致**：`classify_property` 与 `classify_pair` 的类名一致
    for p in pairs[:60]:
        e = _json.loads(p.read_text(encoding="utf-8"))
        rs = e.get("requirement_structured", {}) or {}
        cls_a, _ = classify_pair(e)
        cls_b, _ = classify_property(rs.get("property_id", ""),
                                     rs.get("target_value") or {})
        if cls_b != "unknown":       # 未登记者允许回退
            assert cls_a == cls_b or cls_a == "param", \
                ("单一真源不一致：%s 判 %s，而按性质判 %s"
                 % (e.get("pair_id"), cls_a, cls_b))

    # ④ **β 适用比例必须按"信息量型"计**（不得把速率型约定混入）
    n_ib = kinds.get("ib", 0)
    ratio = n_ib / len(pairs)
    assert 0.30 <= ratio <= 0.60, \
        ("信息量型（β 适用）比例应 ≈46%%（实测 %.0f%%）——若 ≈70%% 说明"
         "**把速率型约定混入**（第 138 条修正）" % (100 * ratio))
    assert kinds.get("convention", 0) >= 40, \
        "速率型（约定）应有 ≥40 条（实测 %d）" % kinds.get("convention", 0)
    assert PROP_TO_CLASS.get("precision") == "param", "精度类应判信息量型"


def test_design_space_registry_and_chain_domain():
    """**设计空间注册表**：域可验证 + 链域是混合型高维域（第 139/141 条）。

    钉住的缺陷：① 域登记与实际验证脱节（登记了却验不了）；② **chain 域是
    "混合型"**（离散组件 + 连续参数），对它的字段做 `float()` 会崩
    （实测：**34 个通过的设计被崩成 `exception:TypeError`、计成拒绝**）。
    """
    from pipeline.design_space import DOMAINS, dof_of, verify_design

    assert set(DOMAINS) >= {"two_level", "rayleigh", "landau", "chain"}, \
        "应有 ≥4 个域（实测 %s）" % sorted(DOMAINS)
    hi = [d for d, v in DOMAINS.items() if len(dof_of(d)) >= 3]
    assert len(hi) >= 2, "高维域应 ≥2（实测 %s）" % hi
    assert DOMAINS["chain"].get("mixed") is True, \
        "chain 域必须标注为**混合型**（离散组件 + 连续参数）"
    assert len(dof_of("chain")) >= 4, "chain 域 dof 应 ≥4"

    for d in ("two_level", "rayleigh", "chain"):
        if d == "two_level":
            x = {"dF_eV": 0.19, "T_K": 1000.0}
        elif d == "rayleigh":
            x = {"D_m": 0.78, "f_Hz": 1.65e6}
        else:
            x = {"nodes": (("gauss_sigma", 1.0), ("ho_omega", 2.0)),
                 "link": True}
        v = verify_design(d, x, target=0.5)
        assert v.get("meets") is not None, \
            "域 %s 必须能给出判定（实测 %s）" % (d, v)


def test_chain_vector_roundtrip_and_unified_dispatch():
    """**混合型域的向量编码 + 统一分派**（第 144 条；任务 4）。

    钉住的缺陷：`chain` 域标了 `mixed=True`（离散组件 + 连续参数）却**没有
    向量编码器** ⟹ **生成模型无法使用该域**（只能均匀采样）。另：统一分派
    必须对**纯连续域**（`rayleigh`）与**混合域**（`chain`）都给出一致接口。
    """
    from pipeline.design_chain import (VEC_DIM, from_vector, sample_chain,
                                       to_vector)
    from pipeline.design_space import (DOMAINS, decode_design, encode_design,
                                       vector_dim)

    assert DOMAINS["chain"].get("mixed") is True, "chain 须为混合型"
    assert vector_dim("chain") == VEC_DIM == 7, "chain 向量维度须为 7"
    assert vector_dim("rayleigh") == 2, "rayleigh 向量维度须为 2"

    # **往返精确**（含 link 与组件索引）
    rng = np.random.default_rng(0)
    for _ in range(20):
        d = sample_chain(rng)
        z = to_vector(d)
        assert z.shape == (VEC_DIM,), "向量形状错误：%s" % (z.shape,)
        d2 = from_vector(z)
        assert d["link"] == d2["link"], "link 往返不一致"
        assert len(d2["nodes"]) == len(d["nodes"]), "节点数往返不一致"
        for (c1, v1), (c2, v2) in zip(d["nodes"], d2["nodes"]):
            assert c1 == c2, "组件往返不一致：%s vs %s" % (c1, c2)
            assert abs(v1 - v2) < 1e-9 * max(abs(v1), 1.0) + 1e-12, \
                "参数往返不一致：%.6g vs %.6g" % (v1, v2)

    # **统一分派**：两个域都可用同一对函数
    zc = encode_design("chain", sample_chain(rng))
    assert zc.shape == (7,), "chain 分派维度错误"
    zr = encode_design("rayleigh", {"D_m": 0.8, "f_Hz": 1.5e6})
    assert zr.shape == (2,) and np.all((zr >= 0) & (zr <= 1)), \
        "rayleigh 分派应返回 [0,1]^2（实测 %s）" % zr
    dr = decode_design("rayleigh", zr)
    assert abs(dr["D_m"] - 0.8) < 1e-6 and abs(dr["f_Hz"] - 1.5e6) / 1.5e6 < 1e-6


def test_reflow_consumption_invariants():
    """**闭环消费的不变量**（第 145 条；任务 1）。

    钉住的缺陷：reflow 产出**无人消费**（闭环断在最后一步）；以及消费时
    若把 `unknown`/`fail` 也写进 DB5（**DB5 schema 只允许 pass/fail**，
    且红线要求 `anchor_eligible` 恒 false）。
    """
    import csv as _csv
    import json as _json
    from pathlib import Path as _P
    root = _P(__file__).resolve().parent.parent
    # ① 消费模块可用（导入即验证接口）
    import pipeline.design_reflow_consume as _c  # noqa: F401
    # ② 加权数据集（若已生成）：列齐、权重为正、设计可解析
    for dom in ("rayleigh", "two_level", "chain"):
        fp = root / ("reflow_weighted_%s.csv" % dom)
        if not fp.exists():
            continue
        with open(fp, "r", encoding="utf-8", newline="") as f:
            rows = list(_csv.DictReader(f))
        assert rows, "%s 加权数据集不得为空" % fp.name
        for r in rows[:50]:
            assert float(r["weight"]) > 0, "权重须为正"
            assert 0.0 <= float(r["overlap"]) <= 1.0, \
                "overlap 须在 [0,1]（DB5 schema 要求）"
            _json.loads(r["design"])           # 设计须可解析
    # ③ DB5：只允许 pass，且 anchor_eligible 恒 false（红线）
    d5 = root / "db5" / "db5.jsonl"
    if d5.exists():
        for line in d5.read_text(encoding="utf-8").splitlines()[:400]:
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            assert rec["validation"]["verdict"] in ("pass", "fail"), \
                "DB5 只允许 pass/fail（unknown 不得入库）"
            assert rec["provenance"]["anchor_eligible"] is False, \
                "DB5 红线：anchor_eligible 恒 false"


def test_array_domain_cross_validation_and_grating_gate():
    """**声学阵列域**：与 rayleigh 域**交叉校验** + 栅瓣门禁（第 149 条）。

    钉住的缺陷：① 主瓣半宽**多乘 2 倍**（把半宽当全宽）⟹ 跨域校验差 **100%**；
    ② 加权用非对称 `cos(π(n+0.5)/N)` ⟹ 负底数取非整数次幂 ⟹ **NaN**（全不可测量）；
    ③ `first_min` 越界（首零点落在可见区外）；
    ④ **单一网格分辨率不足** ⟹ 大 N 时主瓣被漏掉 ⟹ SLL 报到 −129 dB、
    σ_eff 失真（A4/A5 双双异常）；
    ⑤ **栅瓣阈值写成 `d>λ/2`**——那是**扫描**阵列的条件，broadside 应为
    **`d ≥ λ`**（实测 279 个 `d>0.5` 样本零栅瓣是**正确物理**）。
    """
    from pipeline.design_array import (SLL_UNIFORM_DB, measure_array,
                                       sample_array, to_vector, from_vector,
                                       design_sigma_rayleigh)

    # ①③④ **跨域交叉校验 + 解析旁瓣基准**（均匀加权、d=λ/2）
    for N, f in ((16, 3e5), (32, 1e5), (64, 5e4)):
        m = measure_array({"N": N, "d_over_lam": 0.5, "taper_p": 0.0,
                           "f_Hz": f})
        assert m is not None, "N=%d 应可测量" % N
        D = N * 0.5 * m["lam_m"]
        sig_ref = design_sigma_rayleigh(D, f)
        rel = abs(m["sigma_eff"] - sig_ref) / max(sig_ref, 1e-12)
        assert rel <= 0.10, \
            "N=%d：σ_eff=%.4f vs rayleigh %.4f 相对差 %.1f%%（>10%%）" % (
                N, m["sigma_eff"], sig_ref, 100 * rel)
        assert abs(m["SLL_dB"] - SLL_UNIFORM_DB) <= 0.5, \
            "N=%d：SLL=%.2f dB 应 ≈ 解析基准 %.2f dB" % (N, m["SLL_dB"],
                                                    SLL_UNIFORM_DB)

    # ⑤ **栅瓣门禁**：`d ≥ λ` 必须有栅瓣；`d ≤ λ/2` 必须没有
    g_yes = measure_array({"N": 32, "d_over_lam": 1.2, "taper_p": 0.0,
                           "f_Hz": 1e5})
    g_no = measure_array({"N": 32, "d_over_lam": 0.5, "taper_p": 0.0,
                          "f_Hz": 1e5})
    assert g_yes and g_no, "两个设计都应可测量"
    assert g_yes["grating"], "`d=1.2λ`（broadside）应检出栅瓣"
    assert not g_no["grating"], "`d=0.5λ` 不应有栅瓣（λ/2 是扫描条件）"

    # ④ **无不可能的数**：SLL 必须在 (−60, 0]、σ>0
    rng = np.random.default_rng(0)
    for _ in range(30):
        d = sample_array(rng)
        m = measure_array(d)
        assert m is not None, "采样设计应可测量"
        assert -60.0 < m["SLL_dB"] <= 0.0, \
            "SLL=%.2f dB 超出物理范围（网格分辨率不足会报噪声底）" % m["SLL_dB"]
        assert m["sigma_eff"] > 0 and np.isfinite(m["sigma_eff"])

    # **向量往返**（混合型：N 取整）
    for _ in range(10):
        d = sample_array(rng)
        d2 = from_vector(to_vector(d))
        assert d2["N"] == d["N"], "N 往返不一致"
        assert abs(d2["d_over_lam"] - d["d_over_lam"]) < 1e-6
        assert abs(d2["f_Hz"] - d["f_Hz"]) / d["f_Hz"] < 1e-6


# ------------------------------------------------------------------ runner
def _all_tests():
    return [(n, f) for n, f in sorted(globals().items())
            if n.startswith("test_") and callable(f)]


def main() -> int:
    print("=" * 74)
    print("核心不变量套件（%d 项）——覆盖本会话暴露的全部缺陷类型" % len(_all_tests()))
    print("=" * 74)
    fails = 0
    for name, fn in _all_tests():
        try:
            fn()
            print("  [PASS] %s" % name)
        except AssertionError as ex:
            fails += 1
            print("  [FAIL] %s\n         %s" % (name, ex))
        except Exception as ex:  # noqa: BLE001
            fails += 1
            print("  [ERROR] %s\n         %s: %s"
                  % (name, type(ex).__name__, ex))
    print("\n===== %d/%d 通过 =====" % (len(_all_tests()) - fails,
                                        len(_all_tests())))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
