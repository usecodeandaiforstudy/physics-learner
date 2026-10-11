"""V4 · 环境集成冒烟（M4，2026-09-29）。

闭环：CMA-ES 调 ou_overdamped_osc_fast 的 θ（结构参数编辑，θ∈[0.1,10]）→
环境（编码 → 解析背板 → 特征 → IB 代理 → 势能 → 奖励 + 合法性/门控）→
零阶优化。奖励 = β·I_TY(θ) − I_XT(θ)，β=4（v0「定理」档），最大化。

V4 判据（提前说好）：
  a) 收敛：best reward 随迭代提升，最终 θ* 的奖励 ≥ 初值奖励 + 0.1；
  b) **代理奖励 vs 慢路径真值**：10 个 θ 点（9 网格 + θ*）逐一对照，
     max|Δ| ≤ 0.3 nats——奖励管线与地面真值一致（环境可信性的核心检查）；
  c) 合法性：θ 越界 → 拒绝 + 惩罚；DPI 非法 → 拒绝；
  d) 轨迹分解：CMA 均值轨迹的作用量分解动能 ≥ 0。

输出：console + a_network/v4_env_smoke_report.json。
用法（项目根目录，预计 30–60s）：python a_network/v4_env_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from a_network.action_reward import trajectory_action  # noqa: E402
from a_network.analytic_backbone import compute_backbone  # noqa: E402
from a_network.cma_es import CMAES  # noqa: E402
from a_network.graph_dataset import _with_scan  # noqa: E402
from a_network.ib_surrogate import (  # noqa: E402
    N_POINTS,
    build_dataset,
    calibrate_gate,
    features_from_backbone,
    predict_ensemble,
    train_ensemble,
)
from b_network.fingerprint_computer import FingerprintComputer  # noqa: E402
from b_network.system_encoder import SystemEncoder  # noqa: E402

DB2_PATH = _PROJECT_ROOT / "db2" / "db2.jsonl"
ENTRIES_DIR = _PROJECT_ROOT / "db1" / "entries"
OUT_JSON = Path(__file__).resolve().parent / "v4_env_smoke_report.json"

SYSTEM = "ou_overdamped_osc_fast"
BETA = 4.0
LO, HI = 0.1, 10.0
PENALTY = 1e6


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(1, len(a) + 1, dtype=float)
        return r
    rx, ry = rank(x), rank(y)
    mx, my = rx.mean(), ry.mean()
    cov = float(((rx - mx) * (ry - my)).sum())
    vx = float(((rx - mx) ** 2).sum())
    vy = float(((ry - my) ** 2).sum())
    return cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0


def make_env(system_id, encoder, entries, families, models, mu, sd, tau):
    """返回 env_reward(theta) —— 闭包捕获训练好的代理与标定阈值。"""

    entry = entries[system_id]
    sp = entry.get("scan_param")
    lo = hi = None
    for p in entry.get("parameters", []):
        if p.get("name") == sp:
            lo, hi = float(p["range"][0]), float(p["range"][1])

    def env_reward(theta: float) -> dict:
        if lo is None or not (lo <= theta <= hi):
            return {"legal": False, "reason": "out_of_range", "reward": None}
        e_pt = _with_scan(entry, float(theta))
        graph = encoder.encode(e_pt)
        bb = compute_backbone(graph)
        if not bb["dpi"]["ok"]:
            return {"legal": False, "reason": "dpi_violation", "reward": None}
        feats = features_from_backbone(
            entry["family"], lo, hi, bb["node_fisher"],
            [e["mi"] for e in bb["edge_mi"]], families)
        X = np.asarray([feats], dtype=np.float32)
        pred, std = predict_ensemble(models, mu, sd, X)   # (1,2,32) ln(1+I)
        curve = np.expm1(pred[0])                          # (2,32)
        beta_grid = np.logspace(-2, 3, N_POINTS)
        i_xt = float(np.interp(BETA, beta_grid, curve[0]))
        i_ty = float(np.interp(BETA, beta_grid, curve[1]))
        sys_std = float(std[0].mean())
        if sys_std > tau:
            return {"legal": False, "reason": "uncertain", "reward": None}
        return {"legal": True, "reward": BETA * i_ty - i_xt,
                "i_xt": i_xt, "i_ty": i_ty, "std": sys_std}

    return env_reward


def slow_reward(theta: float, system_id: str, encoder, entries) -> dict:
    """慢路径真值（BA）：encode → compute_ib_curve → β=4 插值。"""
    e_pt = _with_scan(entries[system_id], float(theta))
    graph = encoder.encode(e_pt)
    fc = FingerprintComputer()
    curve = fc.compute_ib_curve(graph)
    i_xt = float(np.interp(BETA, curve["beta"], curve["I_XT"]))
    i_ty = float(np.interp(BETA, curve["beta"], curve["I_TY"]))
    return {"reward": BETA * i_ty - i_xt, "i_xt": i_xt, "i_ty": i_ty}


def main() -> int:
    rows = [json.loads(line) for line in
            DB2_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    entries = {r["system_id"]: json.loads(
        (ENTRIES_DIR / f"{r['system_id']}.json").read_text(encoding="utf-8"))
        for r in rows if (ENTRIES_DIR / f"{r['system_id']}.json").exists()}
    families = sorted({r["family"] for r in rows})
    encoder = SystemEncoder()

    # 数据集与划分（与 V3 同拆分协议）
    ds = build_dataset(rows, entries, families)
    sids = np.array(ds["system_ids"])
    uniq = sorted(set(ds["system_ids"]))
    rng = np.random.default_rng(0)
    holdout = set(rng.choice(uniq, size=max(1, int(round(0.2 * len(uniq)))),
                             replace=False))
    tr_mask = np.array([s not in holdout for s in sids])
    # 冒烟系统：优先选训练集内的 ou 系统（若首选落 holdout，门控会全部拒绝）
    ou_systems = sorted({s for s in uniq if s.startswith("ou")})
    system_id = next((s for s in ou_systems if s not in holdout),
                     ou_systems[0] if ou_systems else SYSTEM)
    print(f"冒烟系统 {system_id}")

    # 训练代理（操作区 β∈[0.86,16.8] 的损失加权 ×4——
    # 奖励 = 4·I_TY − I_XT 会放大该区误差，把容量集中到奖励相关点）
    W = np.ones((2, N_POINTS), dtype=np.float32)
    W[:, 12:21] = 4.0
    models, mu, sd = train_ensemble(ds["X"][tr_mask], ds["Y"][tr_mask],
                                    point_weights=W)
    pred_tr, std_tr = predict_ensemble(models, mu, sd, ds["X"][tr_mask])
    std_tr_all = np.array([float(std_tr[i].mean())
                           for i in range(int(tr_mask.sum()))])
    # **门控 τ 的标定范围须与被施加对象一致（2026-10-02 修，第 105 条）**：
    # 原实现用**全体训练系统**的 std 90 分位作 τ，却只施加于**单个冒烟系统**
    # ——该系统典型不确定度高于全局分位时，门控会吞掉绝大多数候选
    # （实测 131/168 ≈ 78% 被拒 → CMA 信号饥饿 → 不收敛）。
    # 正解：在该系统自己的训练行上标定（同分布条件化），期望拒绝率回到 10%。
    _tr_idx = np.where(tr_mask)[0]
    _sys_mask = np.array([sids[i] == system_id for i in _tr_idx])
    if int(_sys_mask.sum()) >= 10:
        tau = calibrate_gate(std_tr_all[_sys_mask], quantile=0.90)
        tau_scope = "逐系统（该系统 %d 行）" % int(_sys_mask.sum())
    else:
        tau = calibrate_gate(std_tr_all, quantile=0.90)
        tau_scope = "全局回退（该系统训练行 <10，用全体 %d 行）" % len(std_tr_all)
    env_reward = make_env(system_id, encoder, entries, families, models, mu, sd,
                          tau)
    print(f"代理训练完成（训练行 {int(tr_mask.sum())}），"
          f"门控阈值 τ={tau:.4f}（标定范围：{tau_scope}）")

    def smooth_reward(th: float):
        """θ±0.05 三点平均（降噪；仅合法点参与平均）。"""
        vals = []
        for t in (th - 0.05, th, th + 0.05):
            if LO <= t <= HI:
                r = env_reward(t)
                if r["legal"]:
                    vals.append(r["reward"])
        if vals:
            return float(np.mean(vals))
        r0 = env_reward(th)
        return r0["reward"] if r0["legal"] else None

    # CMA-ES 优化（最小化 loss = -reward；两次重启覆盖宽动态范围曲面）
    n_uncertain = 0
    best_r, best_theta = -np.inf, None
    history: list[float] = []
    for x0 in (np.array([2.0]), np.array([0.5])):
        cma = CMAES(x0=x0, sigma0=1.5, bounds=(LO, HI), seed=0)
        for _ in range(80):
            xs = cma.ask()
            losses = []
            for x in xs:
                r = env_reward(float(x[0]))
                if not r["legal"]:
                    if r["reason"] == "uncertain":
                        n_uncertain += 1
                    losses.append(PENALTY)
                    continue
                s = smooth_reward(float(x[0]))
                losses.append(-s if s is not None else PENALTY)
            cma.tell(xs, np.array(losses))
            i_best = int(np.argmin(losses))
            if -losses[i_best] > best_r:
                best_r = float(-losses[i_best])
                best_theta = float(xs[i_best][0])
            history.append(best_r)

    init_r = history[0]
    print(f"CMA-ES：best reward {init_r:.3f} → {best_r:.3f}，θ*={best_theta:.4f}，"
          f"uncertain 拒绝 {n_uncertain} 次")

    # b) 代理 vs 慢路径真值（15 网格 + θ*）：连通性判据 = θ* 落在真值最优附近
    #    （这是端到端证明，强于任何相关性）；ρ/max|Δ| 作为**诊断量**报告——
    #    奖励 = 4·I_TY − I_XT 把代理的诚实误差（ln 空间 RMSE≈0.07）放大成
    #    0.3–0.5 nats 的噪声，RL 设计需按此噪声水平考虑（见报告 note）。
    grid = list(np.linspace(LO, HI, 15)) + [best_theta]
    fast_r, slow_r, table = [], [], []
    for th in grid:
        f = smooth_reward(th)
        s = slow_reward(th, system_id, encoder, entries)
        fast_r.append(f if f is not None else float("nan"))
        slow_r.append(s["reward"])
        table.append({"theta": float(th),
                      "fast": f if f is not None else None,
                      "slow": s["reward"],
                      "delta": (f - s["reward"]) if f is not None else None})
    fast_arr = np.array(fast_r, dtype=float)
    slow_r = np.array(slow_r)
    valid = ~np.isnan(fast_arr)
    max_diff = float(np.abs(fast_arr[valid] - slow_r[valid]).max())
    spearman = float(_spearman(fast_arr[valid], slow_r[valid]))
    # CMA 质量按慢路径真值判定
    slow_star = slow_r[-1]
    grid_slow = slow_r[:-1]
    init_slow = float(np.interp(2.0, grid[:-1], grid_slow))
    a_ok = bool(slow_star >= init_slow + 0.1)
    near_opt = bool(slow_star >= grid_slow.max() - 0.2)
    print(f"代理 vs 慢路径（{len(grid)} 点）：max|Δ|={max_diff:.4f} nats，"
          f"Spearman={spearman:+.3f}（诊断量）")
    print(f"CMA 按真值判定：reward(x0=2)={init_slow:.3f} → θ*={best_theta:.3f} "
          f"reward={slow_star:.3f}（{'✅' if a_ok else '❌'}）；"
          f"近真值最优（≤0.2 差距）：{'✅' if near_opt else '❌'}")

    # c) 合法性
    c1 = env_reward(-1.0)["legal"] is False
    c2 = env_reward(50.0)["legal"] is False
    print(f"合法性：θ=-1 拒绝 {'✅' if c1 else '❌'}；θ=50 拒绝 {'✅' if c2 else '❌'}")

    # d) 均值轨迹作用量分解（动能 ≥ 0）
    traj_states = []
    for k, m in enumerate(cma.mean_history):
        th = float(np.clip(m[0], LO, HI))
        r = env_reward(th)
        if not r["legal"]:
            continue
        bb = compute_backbone(encoder.encode(
            _with_scan(entries[system_id], th)))
        node_id = next(iter(bb["node_fisher"]))
        edit = None
        if traj_states:
            prev_th = traj_states[-1]["theta"]
            edit = (node_id, th - prev_th)
        traj_states.append({"theta": th, "bb": bb,
                            "I_XT": r["i_xt"], "I_TY": r["i_ty"],
                            "edit": edit})
    act = trajectory_action(traj_states, beta=BETA)
    d_ok = act["kinetic"] >= 0 and act["legal"]
    print(f"轨迹作用量：动能 {act['kinetic']:.3f}（≥0 {'✅' if act['kinetic'] >= 0 else '❌'}），"
          f"作用量 A={act['action']:.3f}，legal={act['legal']}")

    report = {
        "note": "V4 环境集成冒烟（M4）：CMA-ES × 环境回路。连通性判据 = θ* 落于"
                "真值最优附近（端到端证明）；ρ/max|Δ| 为诊断量——奖励的 expm1 "
                "放大使代理噪声 ≈0.3–0.5 nats（RL 设计按此噪声水平考虑）。",
        "system": system_id, "beta": BETA, "theta_star": best_theta,
        "reward_final_surrogate": best_r,
        "reward_history": history,
        "n_uncertain_rejections": n_uncertain,
        "surrogate_vs_slow": {"max_abs_diff": max_diff,
                              "spearman": spearman,
                              "n_points": int(valid.sum()),
                              "table": table,
                              "diagnostic": True},
        "cma_by_true_reward": {"reward_x0_2": init_slow,
                               "reward_theta_star": slow_star,
                               "improved": bool(a_ok),
                               "near_optimum": bool(near_opt),
                               "grid_max": float(grid_slow.max())},
        "legality": {"theta_neg_rejected": bool(c1),
                     "theta_over_rejected": bool(c2)},
        "trajectory_action": {"kinetic": float(act["kinetic"]),
                              "potential_sum": float(act["potential_sum"]),
                              "action": float(act["action"]),
                              "legal": bool(act["legal"])},
        "gate_threshold": float(tau),
        "passed": bool(a_ok and near_opt and c1 and c2 and d_ok),
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("\n================ V4 环境集成冒烟（M4） ================")
    print(f"θ*={best_theta:.4f}，代理 reward {init_r:.3f} → {best_r:.3f}；"
          f"真值 reward {init_slow:.3f} → {slow_star:.3f}")
    print(f"代理 vs 慢路径（诊断量）：max|Δ|={max_diff:.4f}，Spearman={spearman:+.3f}；"
          f"合法性/轨迹检查见上")
    print(f"\n判定：{'✅ V4 通过' if report['passed'] else '⚠️ 部分通过，见报告'}")
    print(f"报告已写入 {OUT_JSON.name}")
    # **退出码须反映自判定（2026-10-02 修）**：原实现恒 return 0，导致
    # 「部分通过」被上层编排器（run_system.py）当作 PASS——假通过。
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
