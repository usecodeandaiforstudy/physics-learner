"""A 网络 Phase 1 训练：GINE 骨干 + 指纹回归头 + 行为分类头。

训练数据全部来自 DB1 条目（经 SystemEncoder 编码为信息结构图），
指纹真值与 family 标签来自 compute_graph_fingerprint —— 与 DB2 同源，
但图结构直接从 DB1 重编码（DB2 只存扁平指纹，不含图结构）。

两个读出头（架构文档 §2.2.3 的子集，Phase 1 起步）：
- 指纹回归头：节点嵌入 → 每节点 fisher_trace/det 的 log（MSE，核心知识任务）
- 行为分类头：全局嵌入 → family（CE，跨领域结构-行为对应）

OOD 划分两种：
- system：按 system_id 分组 holdout（同 smoke_mlp 协议，可对比分类准确率）
- family：按 family 分组 holdout（更严格，看指纹回归 R² 的跨族泛化）

运行：
    python a_network/train_gine.py --split system
    python a_network/train_gine.py --split family
    python a_network/train_gine.py --split family --holdout landau,negbin,ou,gamma,laplace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from a_network.gine import GINE  # noqa: E402
from a_network.heads import MultiHead  # noqa: E402
from a_network.augment import constraint_negative, rank_pairs  # noqa: E402
from a_network.edit_pipeline import EditDataset  # noqa: E402
from a_network import graph_dataset as gd  # noqa: E402  （R2b 开关：gd.RAW_SIGN_PARAMS）
from a_network.graph_dataset import (  # noqa: E402
    ATTR_VERSION,
    EDGE_FEAT_DIM,
    GraphDataset,
    NODE_FEAT_DIM,
    PARAM_NAMES,
    _ONHOT_DIM,
)

ENTRIES_DIR = _PROJECT_ROOT / "db1" / "entries"
OUT_PATH = Path(__file__).resolve().parent / "gine_metrics.json"

SEED = 0
HOLDOUT_FRAC = 0.25


def _stack(samples: list[dict]) -> dict:
    return {
        "H": torch.stack([s["H"] for s in samples]),
        "A": torch.stack([s["A"] for s in samples]),
        "E": torch.stack([s["E"] for s in samples]),
        "mask": torch.stack([s["mask"] for s in samples]),
        "fisher_target": torch.stack([s["fisher_target"] for s in samples]),
        "edge_tbl": torch.stack([s["edge_tbl"] for s in samples]),
        "family": torch.tensor([s["family"] for s in samples]),
    }


def theory_consistency_loss(log_trace, edge_tbl):
    """约束 A：DPI 的 Fisher 形式自洽（PINN 式残差，log 空间，无量纲稳定）。

    A1 单调非增：沿信息流 u→v，log F_trace(v) ≤ log F_trace(u)。离线验证 33/33。
    A2 加性噪声边闭式：由 MI = -½ln(1 - F_v/F_u) 反解
       log F(v) - log F(u) = ln(1 - exp(-2·MI))，
       右边只依赖**输入 MI**（真值特征），是常数——故本残差不需要额外标签。

    参数
    ----
    log_trace : [B,N]   网络预测的 log Fisher trace（= fisher_mu[:,:,0]）
    edge_tbl  : [B,E,4] [src_idx, dst_idx, 输入 MI, 是否加性噪声边]，-1 为空槽

    返回
    ----
    (loss_a1, loss_a2, n_a1, n_a2)
    """
    valid = edge_tbl[:, :, 0] >= 0
    if not bool(valid.any()):
        z = log_trace.sum() * 0.0
        return z, z, 0, 0

    src = edge_tbl[:, :, 0].clamp(min=0).long()
    dst = edge_tbl[:, :, 1].clamp(min=0).long()
    mi_in = edge_tbl[:, :, 2]
    is_noise = edge_tbl[:, :, 3] > 0.5

    lt_s = torch.gather(log_trace, 1, src)
    lt_d = torch.gather(log_trace, 1, dst)

    # A1：单调非增（log 空间，裕度 1e-3）
    a1 = torch.relu(lt_d - lt_s + 1e-3) * valid.float()
    loss_a1 = a1.sum() / (valid.float().sum() + 1e-8)

    # A2：加性噪声边的闭式关系（目标为常数，数值稳定、无 log 奇点）
    m2 = valid & is_noise
    n2 = int(m2.sum())
    if n2 > 0:
        # 空槽 MI=-1 会让 log1p(-exp(2)) 产生 nan，故先在无效槽填入正值再算
        mi_safe = torch.where(m2, mi_in, torch.ones_like(mi_in))
        target = torch.log1p(-torch.exp(-2.0 * mi_safe))
        a2 = ((lt_d - lt_s - target) ** 2) * m2.float()
        loss_a2 = a2.sum() / (m2.float().sum() + 1e-8)
    else:
        loss_a2 = log_trace.sum() * 0.0
    return loss_a1, loss_a2, int(valid.sum()), n2


def _stack_edit(samples: list[dict]) -> dict:
    return {
        "H": torch.stack([s["H"] for s in samples]),
        "A": torch.stack([s["A"] for s in samples]),
        "E": torch.stack([s["E"] for s in samples]),
        "mask": torch.stack([s["mask"] for s in samples]),
        "edit_enc": torch.stack([s["edit_enc"] for s in samples]),
        "delta_f": torch.stack([s["delta_f"] for s in samples]),
    }


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def evaluate(model, heads, dataset, indices, device) -> dict:
    model.eval()
    heads.eval()
    all_pred_tr, all_true_tr = [], []
    all_pred_det, all_true_det = [], []
    all_fam = []   # 每个有效节点的族索引（按样本广播），供逐族 R² 分解
    n_correct, n_total = 0, 0
    bs = 256
    with torch.no_grad():
        for i in range(0, len(indices), bs):
            batch = _stack([dataset[j] for j in indices[i:i + bs]])
            H = batch["H"].to(device)
            A = batch["A"].to(device)
            E = batch["E"].to(device)
            mask = batch["mask"].to(device)
            Hv, Hg = model(H, A, E, mask)
            out = heads(Hv, Hg)
            fisher = out["fisher_mu"]
            logits = out["logits"]

            # 分类
            pred_cls = logits.argmax(1).cpu().numpy()
            true_cls = batch["family"].numpy()
            n_correct += int((pred_cls == true_cls).sum())
            n_total += len(true_cls)

            # 回归（仅有效节点）
            m = mask.cpu().numpy().astype(bool)
            ft = batch["fisher_target"].numpy()
            fp = fisher.cpu().numpy()
            fam = batch["family"].numpy()
            all_pred_tr.append(fp[m, 0])
            all_true_tr.append(ft[m, 0])
            all_pred_det.append(fp[m, 1])
            all_true_det.append(ft[m, 1])
            for b in range(m.shape[0]):
                all_fam.extend([int(fam[b])] * int(m[b].sum()))

    pred_tr = np.concatenate(all_pred_tr)
    true_tr = np.concatenate(all_true_tr)
    pred_det = np.concatenate(all_pred_det)
    true_det = np.concatenate(all_true_det)
    fam_arr = np.array(all_fam)

    # 逐族 R² 分解：族级 OOD 下哪个族失败一目了然（诊断外推缺口的标准工具）
    per_family = {}
    for fi, fname in enumerate(dataset.families):
        sel = fam_arr == fi
        if sel.sum() == 0:
            continue
        per_family[fname] = {
            "n_nodes": int(sel.sum()),
            "r2_trace": _r2(pred_tr[sel], true_tr[sel]),
            "r2_det": _r2(pred_det[sel], true_det[sel]),
        }

    return {
        "acc": n_correct / max(1, n_total),
        "r2_trace": _r2(pred_tr, true_tr),
        "r2_det": _r2(pred_det, true_det),
        "per_family": per_family,
    }


def _dump_test_preds(dataset, model, heads, te_idx, device, mu, sd, out_path) -> None:
    """测试集逐节点预测取证（诊断用）：每行 = 一个有效节点的
    {family, system_id, params(未标准化), true_ln_trace/det, pred_ln_trace/det,
     pred_var}。params 里缺失参数的列是标准化偏移还原后的噪声值（无意义），
    只读该族真实存在的参数列（如 landau 读 "a"）。"""
    lines: list[str] = []
    bs = 256
    with torch.no_grad():
        for i in range(0, len(te_idx), bs):
            idxs = te_idx[i:i + bs]
            batch = _stack([dataset[j] for j in idxs])
            H = batch["H"].to(device)
            A = batch["A"].to(device)
            E = batch["E"].to(device)
            mask = batch["mask"].to(device)
            Hv, Hg = model(H, A, E, mask)
            out = heads(Hv, Hg)
            fp = out["fisher_mu"].cpu().numpy()
            fv = out["fisher_var"].cpu().numpy()
            ft = batch["fisher_target"].numpy()
            m = mask.cpu().numpy().astype(bool)
            fam = batch["family"].numpy()
            # 还原未标准化的连续特征（PARAM_NAMES 列在 _ONHOT_DIM 之后）
            H_raw = batch["H"].numpy()[:, :, _ONHOT_DIM:] * sd + mu
            sids = [dataset.samples[j]["system_id"] for j in idxs]
            svals = [dataset.samples[j].get("scan_value") for j in idxs]
            for b in range(m.shape[0]):
                fname = dataset.families[int(fam[b])]
                for n in range(m.shape[1]):
                    if not m[b, n]:
                        continue
                    lines.append(json.dumps({
                        "family": fname,
                        "system_id": sids[b],
                        "scan_value": None if svals[b] is None else float(svals[b]),
                        "params": {pn: float(H_raw[b, n, k])
                                   for k, pn in enumerate(PARAM_NAMES)},
                        "true_ln_trace": float(ft[b, n, 0]),
                        "true_ln_det": float(ft[b, n, 1]),
                        "pred_ln_trace": float(fp[b, n, 0]),
                        "pred_ln_det": float(fp[b, n, 1]),
                        "pred_var": float(fv[b, n, 0]),
                    }, ensure_ascii=False))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"预测取证已写入 {out_path}（{len(lines)} 节点）")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="A 网络 Phase 1 训练（GINE）")
    ap.add_argument("--split", default="system", choices=["system", "family"],
                    help="OOD 划分方式（system=按系统分组，family=按族分组）")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--mask_prob", type=float, default=0.25,
                    help="掩码增广：随机 mask 节点参数的比例")
    ap.add_argument("--mask_weight", type=float, default=1.0,
                    help="L_mask 掩码重建损失的权重")
    ap.add_argument("--w_consist", type=float, default=0.0,
                    help="约束 A（DPI 的 Fisher 形式自洽，PINN 式无标签残差）权重，0=关闭")
    ap.add_argument("--w_shape", type=float, default=0.0,
                    help="约束 B（沿参数扫描路径的 log Fisher 形状/斜率约束）权重，0=关闭")
    ap.add_argument("--n_shape_pairs", type=int, default=32,
                    help="约束 B 每步采样的路径对数（越大约束越强；默认 32）")
    ap.add_argument("--holdout", default=None,
                    help="固定 holdout（逗号分隔；family 划分=族名，system 划分=系统 id）。"
                         "None=按 seed 随机抽（默认，与历史基线一致）")
    ap.add_argument("--dump_preds", default=None,
                    help="评估后把测试集逐节点 (族, 系统, 参数, 真值, 预测, 不确定度)"
                         "写为 JSONL（诊断取证用，默认关闭）")
    ap.add_argument("--raw_sign_params", action="store_true",
                    help="R2b：被扫描的可负参数(a)原值编码（默认关=旧混合编码，基线可复现）")
    ap.add_argument("--best-checkpoint", action="store_true",
                    help="按 EMA 平滑训练损失选择最佳 checkpoint 再评估"
                         "（训练指标选择，无测试泄漏；治 e600 偶发崩溃 seed）")
    ap.add_argument("--save-checkpoint", action="store_true",
                    help="落盘 GINE 骨干权重到 a_network/checkpoints/gine_best.pt"
                         "（供 RL 智能体用 A 冻结骨干嵌入作状态，§2.6.2 第 2 条；"
                         "struct_embed.py 检测到该文件即自动切换骨干嵌入模式）")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # R2b 开关（第 30 条）：必须在构造数据集前设置（graph_to_tensors 读模块级开关）
    gd.RAW_SIGN_PARAMS = args.raw_sign_params

    # 1) 数据
    dataset = GraphDataset(ENTRIES_DIR)
    n_classes = len(dataset.families)
    n_systems = len(dataset.system_ids)
    edit_dataset = EditDataset(ENTRIES_DIR)

    # 2) 划分（按 system 或 family）。--holdout 显式指定时固定 holdout
    #    （固定 holdout multi-seed 协议，HANDOVER 第 27 条教训 (c)：跨 seed 只变
    #    训练随机数、不变 holdout——把「约束效应」与「随机波动」真正分开）。
    #    注意：holdout 的选择用独立 default_rng，不消耗全局 np.random 序列，
    #    故固定 holdout 不影响训练流（同 seed 训练过程与随机划分时逐位一致）。
    rng = np.random.default_rng(args.seed)
    if args.split == "system":
        groups = sorted(dataset.system_ids)
    else:
        groups = sorted(set(dataset.families))
    if args.holdout:
        holdout = set(h.strip() for h in args.holdout.split(",") if h.strip())
        unknown = sorted(holdout - set(groups))
        if unknown:
            print(f"[FAIL] holdout 含未知组：{unknown}")
            print(f"       合法组：{groups}")
            return 2
        holdout_mode = "fixed"
    else:
        n_hold = max(1, int(round(HOLDOUT_FRAC * len(groups))))
        holdout = set(rng.choice(groups, size=n_hold, replace=False))
        holdout_mode = "random"

    tr_idx, te_idx = [], []
    for i, s in enumerate(dataset.samples):
        key = s["system_id"] if args.split == "system" else dataset.families[s["family"]]
        (te_idx if key in holdout else tr_idx).append(i)

    # ---- 训练/测试信息隔离（协议修复，2026-09-28）----
    # 约束 B 的 path_pairs 与编辑头的 EditDataset 都覆盖全库（含 holdout 系统）。
    # 若全量使用，等于把 holdout 系统的解析斜率 / 编辑增量当作训练目标喂给模型
    # ——holdout 标签泄漏，虚增约束收益（第 26/27 条的 +A+B 数字是被泄漏抬高的
    # 上界）。此处只保留训练系统的路径对 / 编辑对，holdout 真正「不可见」。
    tr_set = set(tr_idx)
    path_pairs_tr = [p for p in dataset.path_pairs
                     if int(p[0]) in tr_set and int(p[1]) in tr_set]
    tr_systems = {dataset.samples[i]["system_id"] for i in tr_idx}
    edit_tr_idx = [j for j, s in enumerate(edit_dataset.samples)
                   if s.get("system_id") in tr_systems]

    # 3) 特征标准化（连续列，用训练集统计量；one-hot 列不动）
    Hs = np.stack([dataset.samples[i]["H"] for i in tr_idx])
    mu = Hs[:, :, _ONHOT_DIM:].mean(axis=(0, 1))
    sd = Hs[:, :, _ONHOT_DIM:].std(axis=(0, 1))
    sd[sd < 1e-6] = 1.0   # std≈0 的列（如 mu/kT log 编码后恒 0）避免除以极小值
    for s in dataset.samples:
        s["H"][:, _ONHOT_DIM:] = (s["H"][:, _ONHOT_DIM:] - mu) / sd
    for s in edit_dataset.samples:
        s["H"][:, _ONHOT_DIM:] = (s["H"][:, _ONHOT_DIM:] - mu) / sd

    # 4) 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GINE(NODE_FEAT_DIM, EDGE_FEAT_DIM, args.hidden, args.layers).to(device)
    heads = MultiHead(args.hidden, n_classes, len(PARAM_NAMES)).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(heads.parameters()), lr=args.lr)
    reg_fn = nn.MSELoss()
    cls_fn = nn.CrossEntropyLoss()
    bce_fn = nn.BCEWithLogitsLoss()
    rank_fn = nn.MarginRankingLoss(margin=0.5)

    # 5) 训练
    bs = 128
    best_epoch = -1
    best_smooth = float("inf")
    smooth = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        heads.train()
        order = np.random.permutation(tr_idx)
        total_loss = 0.0
        for i in range(0, len(order), bs):
            batch = _stack([dataset[j] for j in order[i:i + bs]])
            H = batch["H"].to(device)
            A = batch["A"].to(device)
            E = batch["E"].to(device)
            mask = batch["mask"].to(device)
            ft = batch["fisher_target"].to(device)
            fam = batch["family"].to(device)

            # 掩码增广（L_mask 自监督）：随机 mask 节点分布参数，迫使模型
            # 从分布族 one-hot + 图结构 + 未 mask 上下文重建参数，抑制记忆化
            param_start = _ONHOT_DIM
            param = H[:, :, param_start:param_start + len(PARAM_NAMES)]
            node_mask = (torch.rand(H.shape[0], H.shape[1], 1, device=device)
                         < args.mask_prob).float()
            node_mask = node_mask * mask.unsqueeze(-1)   # 只在有效节点上 mask
            masked_H = H.clone()
            masked_H[:, :, param_start:param_start + len(PARAM_NAMES)] *= (1.0 - node_mask)

            Hv, Hg = model(masked_H, A, E, mask)
            out = heads(Hv, Hg)
            fisher = out["fisher_mu"]
            fisher_var = out["fisher_var"]
            logits = out["logits"]
            param_recon = out["param_recon"]
            rank_score = out["rank_score"]

            # 回归损失（仅有效节点）
            m = mask.unsqueeze(-1)  # [B,N,1]
            reg_loss = reg_fn(fisher * m, ft * m)
            # 不确定性（异方差 NLL；mu 用 detach 防回流，让 uncertainty_head 专学方差）
            nll = 0.5 * (fisher_var + (ft - fisher.detach()) ** 2 * torch.exp(-fisher_var))
            calib_loss = (nll * m).sum() / (m.sum() + 1e-8)
            cls_loss = cls_fn(logits, fam)
            # 掩码重建损失（仅被 mask 的节点）
            mask_loss = ((param_recon - param) ** 2 * node_mask).sum() / (
                node_mask.sum() + 1e-8)
            # 约束损失（正样本 label 1；负样本=篡改边 MI 破坏 DPI 单调，label 0）
            n_neg = max(1, H.shape[0] // 4)
            E_neg, y_neg = constraint_negative(H, A, E, mask, n_neg=n_neg)
            _, Hg_neg = model(H[:n_neg], A[:n_neg], E_neg, mask[:n_neg])
            c_neg = heads.constraint_head(Hg_neg)
            c_pos = heads.constraint_head(Hg[:n_neg])
            constraint_loss = (bce_fn(c_pos, torch.ones_like(c_pos))
                               + bce_fn(c_neg, y_neg))
            # 排序损失（Fisher trace 大者排前，代理标签）
            trace_pool = (fisher[:, :, 0] * mask).sum(1) / (mask.sum(1) + 1e-8)
            pi, pj = rank_pairs(trace_pool, n_pairs=H.shape[0])
            if pi.numel() > 0:
                rank_loss = rank_fn(rank_score[pi], rank_score[pj],
                                    torch.ones_like(rank_score[pi]))
            else:
                rank_loss = 0.0 * rank_score.sum()
            # 编辑效应损失（world model：从编辑前图嵌入 + 编辑编码预测指纹增量 Δf）
            # 只用训练系统的编辑对（协议修复：防 holdout 泄漏，见划分处注释）
            if edit_tr_idx:
                e_idx = np.random.choice(edit_tr_idx,
                                         size=min(len(edit_tr_idx), H.shape[0]),
                                         replace=False)
                eb = _stack_edit([edit_dataset[j] for j in e_idx])
                _, Hg_e = model(eb["H"].to(device), eb["A"].to(device),
                                eb["E"].to(device), eb["mask"].to(device))
                edit_enc = eb["edit_enc"].to(device)
                delta_f = eb["delta_f"].to(device)
                delta_f_pred = heads.edit_head(torch.cat([Hg_e, edit_enc], dim=-1))
                edit_loss = reg_fn(delta_f_pred, delta_f)
            else:
                edit_loss = 0.0 * rank_score.sum()

            # ---- 约束 A：DPI 的 Fisher 形式自洽（PINN 式残差，不需要额外标签）----
            # A1 链上 Fisher trace 单调非增；A2 加性噪声边的 MI 闭式关系。
            # 仅作用于含边的样本（当前 ou/diffusion 共 21 系统），其余样本残差为 0。
            if args.w_consist > 0:
                edge_tbl = batch["edge_tbl"].to(device)
                a1_loss, a2_loss, n_a1, n_a2 = theory_consistency_loss(
                    fisher[:, :, 0], edge_tbl)
                consist_loss = a1_loss + a2_loss
            else:
                consist_loss = 0.0 * fisher.sum()

            # ---- 约束 B：信息几何形状约束（沿参数扫描路径的 log Fisher 斜率）----
            # 离线验证：d ln F ≈ d ln g_KL 对幂律族达机器精度（指数/泊松/瑞利 1e-14）。
            # 约束的是「函数形状」而非「数值」，正对族级 OOD 的痛点。
            if args.w_shape > 0 and len(path_pairs_tr) > 0:
                n_pairs = min(args.n_shape_pairs, len(path_pairs_tr))
                sel = np.random.choice(len(path_pairs_tr), size=n_pairs,
                                       replace=False)
                ii = [int(path_pairs_tr[k][0]) for k in sel]
                jj = [int(path_pairs_tr[k][1]) for k in sel]
                dt_t = torch.tensor([path_pairs_tr[k][2] for k in sel],
                                    dtype=torch.float32, device=device)
                bi = _stack([dataset[k] for k in ii])
                bj = _stack([dataset[k] for k in jj])
                Hvi, _ = model(bi["H"].to(device), bi["A"].to(device),
                               bi["E"].to(device), bi["mask"].to(device))
                Hvj, _ = model(bj["H"].to(device), bj["A"].to(device),
                               bj["E"].to(device), bj["mask"].to(device))
                lt_i = heads.fisher_head(Hvi)[:, 0, 0]
                lt_j = heads.fisher_head(Hvj)[:, 0, 0]
                shape_loss = ((lt_j - lt_i - dt_t) ** 2).mean()
            else:
                shape_loss = 0.0 * fisher.sum()

            loss = (reg_loss + cls_loss + args.mask_weight * mask_loss
                    + 0.1 * calib_loss + 0.1 * constraint_loss + 0.1 * rank_loss
                    + 0.1 * edit_loss
                    + args.w_consist * consist_loss
                    + args.w_shape * shape_loss)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(heads.parameters()), 1.0)
            opt.step()
            total_loss += float(loss.item()) * len(batch["family"])

        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch + 1}/{args.epochs}  loss={total_loss / len(tr_idx):.4f}")

        # best-checkpoint：EMA 平滑训练损失选点（仅训练指标，无测试泄漏）
        if args.best_checkpoint:
            ep_loss = total_loss / len(tr_idx)
            smooth = ep_loss if best_epoch < 0 else 0.9 * smooth + 0.1 * ep_loss
            if smooth < best_smooth:
                best_smooth = smooth
                best_epoch = epoch + 1
                import copy as _copy
                best_state = (_copy.deepcopy(model.state_dict()),
                              _copy.deepcopy(heads.state_dict()))

    if args.best_checkpoint and best_state is not None:
        model.load_state_dict(best_state[0])
        heads.load_state_dict(best_state[1])
        print(f"best-checkpoint：采用 epoch {best_epoch}"
              f"（EMA 平滑训练损失 {best_smooth:.4f}）")

    # 6) 评估
    tr_metrics = evaluate(model, heads, dataset, tr_idx, device)
    te_metrics = evaluate(model, heads, dataset, te_idx, device)

    # 6b) 预测取证（诊断用，默认关闭）
    if args.dump_preds:
        _dump_test_preds(dataset, model, heads, te_idx, device, mu, sd,
                         Path(args.dump_preds))

    print("===== A 网络 Phase 1：GINE + 六读出头 =====")
    print(f"数据：{len(dataset)} 样本 / {n_systems} 系统 / {n_classes} 族")
    if args.w_consist or args.w_shape:
        print(f"理论约束：A(DPI 自洽) w={args.w_consist:g}  "
              f"B(路径形状) w={args.w_shape:g}  "
              f"路径对={len(path_pairs_tr)}/{len(dataset.path_pairs)}（仅训练集）")
    print(f"划分：{args.split} 级 OOD，holdout {len(holdout)}/{len(groups)} 组"
          f"（{'固定' if holdout_mode == 'fixed' else '按 seed 随机'}）")
    print(f"训练集：{len(tr_idx)} 样本，测试集：{len(te_idx)} 样本")
    print(f"行为分类准确率   train={tr_metrics['acc']:.3f}  test={te_metrics['acc']:.3f}")
    print(f"指纹回归 R²(trace) train={tr_metrics['r2_trace']:.3f}  test={te_metrics['r2_trace']:.3f}")
    print(f"指纹回归 R²(det)   train={tr_metrics['r2_det']:.3f}  test={te_metrics['r2_det']:.3f}")
    print("-- test 逐族 R² 分解（诊断外推缺口）--")
    for fname, fm in sorted(te_metrics["per_family"].items()):
        print(f"    {fname:22s} trace={fm['r2_trace']:+.3f}  det={fm['r2_det']:+.3f}  "
              f"(n={fm['n_nodes']})")

    out = {
        "model": "GINE+virtual-node（手搓 PyTorch，非 PyG）",
        "attr_version": ATTR_VERSION,
        "raw_sign_params": args.raw_sign_params,
        "n_samples": len(dataset), "n_systems": n_systems, "n_families": n_classes,
        "split": args.split, "holdout": sorted(holdout),
        "holdout_mode": holdout_mode,
        "n_train": len(tr_idx), "n_test": len(te_idx),
        "train": tr_metrics, "test": te_metrics,
        "hidden": args.hidden, "layers": args.layers, "seed": args.seed,
        "w_consist": args.w_consist, "w_shape": args.w_shape,
        "n_shape_pairs": args.n_shape_pairs,
        "n_path_pairs": len(path_pairs_tr),
        "n_path_pairs_total": len(dataset.path_pairs),
    }
    if args.w_consist or args.w_shape:
        tag = f"wA{args.w_consist:g}_wB{args.w_shape:g}"
        if args.w_shape > 0:
            tag += f"_np{args.n_shape_pairs}"   # 区分不同 B 强度
    else:
        tag = "base"
    if args.best_checkpoint:
        tag += "_best"   # best-checkpoint 变体独立命名（第 44 条）
    out_path = Path(__file__).resolve().parent / f"gine_metrics_{args.split}_{tag}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("指标已写入", out_path)

    if args.save_checkpoint:
        ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "gine_best.pt"
        torch.save({
            "model": model.state_dict(),
            "d_in": int(NODE_FEAT_DIM),
            "d_e": int(EDGE_FEAT_DIM),
            "d_h": int(args.hidden),
            "n_layers": int(args.layers),
        }, ckpt_path)
        print("骨干已落盘", ckpt_path, "（RL 智能体状态可切 A 冻结骨干嵌入）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
