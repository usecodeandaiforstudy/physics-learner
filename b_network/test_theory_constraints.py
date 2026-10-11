"""方案 A/B 理论约束的离线解析验证（最小验证第一步）。

在真实 DB1 条目上验证两个理论约束**是否解析成立**。若不成立，后续接入训练无意义。

A. DPI 的 Fisher 形式（信息处理不增信息 → 分布更模糊 → 对参数更不敏感）：
   A1 沿信息流方向 u->v，Fisher trace 单调非增：F_trace(v) <= F_trace(u)
   A2 对「加性噪声边」（v = u + 独立噪声，判据 cov_12 == cov_11）：
      MI(u;v) = -1/2 * ln(1 - F_trace(v)/F_trace(u))   （精确闭式）
      推导：ρ² = σ_u²/σ_v²，I = -1/2 ln(1-ρ²)，F_trace = 3/σ²
B. 信息几何定义式（Fisher = KL 的二阶展开，即信息拉格朗日量动能项的微观形式）：
   F(θ_s) ≈ [KL(p_s||p_{s+1}) + KL(p_{s+1}||p_s)] / (θ_{s+1}-θ_s)² + O(h²)
   对「单参数族且 trace = 该参数 Fisher」的族验证精度与 O(h²) 收敛性。

运行：python b_network/test_theory_constraints.py
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from b_network.system_encoder import SystemEncoder  # noqa: E402
from core.fingerprint import compute_graph_fingerprint  # noqa: E402
from core.fisher import (  # noqa: E402
    fisher_exponential,
    fisher_geometric,
    fisher_poisson,
    fisher_rayleigh,
)
from a_network.graph_dataset import _scan_grid  # noqa: E402

ENTRIES = Path(r"d:\physics learn\db1\entries")


# ---------------------------------------------------------------------------
# 解析 KL（与各编码器参数化一致；自然对数 nats）
# ---------------------------------------------------------------------------
def kl_exponential(l1: float, l2: float) -> float:
    """Exp(λ)（p=λe^{-λx}）：KL = ln(λ1/λ2) + λ2/λ1 - 1。"""
    return math.log(l1 / l2) + l2 / l1 - 1.0


def kl_poisson(l1: float, l2: float) -> float:
    """Poisson(λ)：KL = λ1 ln(λ1/λ2) - λ1 + λ2。"""
    return l1 * math.log(l1 / l2) - l1 + l2


def kl_geometric(p: float, q: float) -> float:
    """Geo(p)（P(X=k)=p(1-p)^(k-1), k>=1）：KL = ln(p/q) + (1-p)/p·ln((1-p)/(1-q))。"""
    return math.log(p / q) + (1.0 - p) / p * math.log((1.0 - p) / (1.0 - q))


def kl_rayleigh(s1: float, s2: float) -> float:
    """Rayleigh(σ)：KL = ln(σ2²/σ1²) + σ1²/σ2² - 1。"""
    return math.log(s2**2 / s1**2) + s1**2 / s2**2 - 1.0


# 族 → (scan_param 名, KL 函数, Fisher 函数)
FAM_B = {
    "radioactive_decay": ("rate", kl_exponential, fisher_exponential),
    "poisson_count": ("rate", kl_poisson, fisher_poisson),
    "geometric": ("p", kl_geometric, fisher_geometric),
    "rayleigh": ("sigma", kl_rayleigh, fisher_rayleigh),
}


def verify_A(lines: list[str]) -> tuple[int, int]:
    """验证 A：DPI 单调性 + 加性噪声边的 Fisher–MI 闭式。"""
    enc = SystemEncoder()
    a1_ok = a1_tot = 0
    a2_ok = a2_tot = 0
    a2_maxrel = 0.0
    a2_details = []
    a1_fail = []
    n_entries = 0

    for f in sorted(glob.glob(str(ENTRIES / "*.json"))):
        entry = json.loads(Path(f).read_text(encoding="utf-8"))
        if entry.get("family") not in ("ou", "diffusion"):
            continue
        n_entries += 1
        graph = enc.encode(entry)
        fp = compute_graph_fingerprint(graph)
        for edge in graph.edges:
            Fs = fp["nodes"][edge.src]["fisher_trace"]
            Fd = fp["nodes"][edge.dst]["fisher_trace"]
            # A1 单调非增
            a1_tot += 1
            if Fd <= Fs * (1.0 + 1e-12):
                a1_ok += 1
            else:
                a1_fail.append(
                    f"      {entry['system_id']}: {edge.src}(F={Fs:.4f}) -> "
                    f"{edge.dst}(F={Fd:.4f})  链位置 "
                    f"{edge.src.split('__')[1]}->{edge.dst.split('__')[1]}")
            # A2 加性噪声边
            cov = getattr(edge.channel, "cov", None)
            if cov is None:
                continue
            cov = np.asarray(cov, dtype=float)
            if abs(cov[0, 1] - cov[0, 0]) <= 1e-9 * max(1.0, abs(cov[0, 0])):
                mi = float(edge.mutual_information)
                mi_theory = -0.5 * math.log(1.0 - Fd / Fs)
                rel = abs(mi - mi_theory) / abs(mi)
                a2_tot += 1
                a2_maxrel = max(a2_maxrel, rel)
                if rel < 1e-9:
                    a2_ok += 1
                if len(a2_details) < 3:
                    a2_details.append(
                        f"      {entry['system_id']}: {edge.src}->{edge.dst} "
                        f"MI={mi:.6f} vs -½ln(1-Fd/Fs)={mi_theory:.6f} (rel={rel:.2e})")

    lines.append(f"A. DPI 的 Fisher 形式（{n_entries} 个 ou/diffusion 系统）")
    lines.append(f"  A1 单调非增 F(v)<=F(u)：{a1_ok}/{a1_tot} 通过")
    if a1_fail:
        lines.append(f"  A1 反例（前 {min(6, len(a1_fail))} 条，共 {len(a1_fail)}）：")
        lines.extend(a1_fail[:6])
    lines.append(f"  A2 加性噪声边 MI=-½ln(1-Fd/Fs)：{a2_ok}/{a2_tot} 通过"
                 f"（最大相对误差 {a2_maxrel:.2e}）")
    lines.extend(a2_details)
    lines.append("")
    ok = (a1_ok == a1_tot and a1_tot > 0 and a2_ok == a2_tot and a2_tot > 0)
    return (1 if ok else 0), (a1_tot + a2_tot)


def verify_B(lines: list[str]) -> tuple[int, int]:
    """验证 B：KL 曲率 vs Fisher（直接式 + 对数差分式）+ O(h²) 收敛。

    对数差分式 d ln g_KL ≈ d ln F 能消去扫描步长引入的常数偏差，
    是更适合做训练约束的形式（约束的是 Fisher 沿参数路径的对数斜率，
    即函数形状/幂律指数——正是族级 OOD 的痛点）。
    """
    by_fam: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(str(ENTRIES / "*.json"))):
        e = json.loads(Path(f).read_text(encoding="utf-8"))
        by_fam.setdefault(e.get("family"), []).append(e)

    n_pass = 0
    n_tot = 0
    for fam, (pname, kl_fn, f_fn) in FAM_B.items():
        entries = by_fam.get(fam, [])
        if not entries:
            lines.append(f"B. [{fam}] 无条目，跳过")
            continue
        entry = entries[0]
        sp = entry.get("scan_param")
        lines.append(f"B. [{fam}] system={entry['system_id']!r} scan_param={sp!r}")
        if sp != pname:
            lines.append(f"  !! scan_param 不匹配（要求 {pname!r}），跳过")
            continue

        for n_pts, tag in ((30, "实际扫描网格"), (200, "细网格")):
            grid = [float(x) for x in _scan_grid(entry, n_pts)]
            gk, F = [], []
            for s in range(len(grid) - 1):
                t0, t1 = grid[s], grid[s + 1]
                h = t1 - t0
                if h == 0:
                    continue
                gk.append((kl_fn(t0, t1) + kl_fn(t1, t0)) / (h * h))
                F.append(float(f_fn(t0)))
            gk = np.array(gk)
            F = np.array(F)
            rel_direct = np.abs(gk - F) / F
            dln_kl = np.diff(np.log(gk))
            dln_F = np.diff(np.log(F))
            rel_diff = np.abs(dln_kl - dln_F) / np.abs(dln_F)
            lines.append(
                f"  {tag}（{len(grid)} 点）：直接式 max_rel={rel_direct.max():.3e} "
                f"mean={rel_direct.mean():.3e} | 对数差分式 "
                f"max_rel={rel_diff.max():.3e} mean={rel_diff.mean():.3e}")
            if tag == "细网格":
                n_tot += 1
                if rel_diff.max() < 5e-3:
                    n_pass += 1
                else:
                    lines.append("  !! 对数差分式细网格误差仍 >0.5%")
    lines.append("")
    return n_pass, n_tot


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    lines = ["===== 方案 A/B 理论约束离线解析验证 =====", ""]
    a_ok, a_tot = verify_A(lines)
    b_ok, b_tot = verify_B(lines)
    lines.append(f"汇总：A {'PASS' if a_ok else 'FAIL'}（{a_tot} 条边关系）；"
                 f"B {'PASS' if b_ok else 'FAIL'}（{b_ok}/{b_tot} 族细网格 <0.5%）")

    out = Path(r"d:\physics learn\b_network\theory_constraints_report.txt")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if (a_ok and b_ok) else 1


if __name__ == "__main__":
    import traceback
    try:
        code = main()
    except BaseException:  # noqa: BLE001
        Path(r"d:\physics learn\b_network\theory_constraints_error.txt").write_text(
            traceback.format_exc(), encoding="utf-8")
        code = 99
    sys.exit(code)
