# -*- coding: utf-8 -*-
"""GFlowNet（轨迹平衡，TB）v0——RL 生产管线（M5，2026-09-29）。

设计目标（design doc §2.4/§7，Phase 2 RL 生产）：
  - 以「信息作用量」为奖励本体，让结构编辑代理**采样多样解**而非单点最优；
  - v0 边界：离散状态（设计任务环境的一维参数网格），P_B 固定均匀（文档化
    简化；回退方案 SAC 见 design doc），Z 为可学习参数；
  - 零新依赖（仅 torch + numpy）。

TB 损失：
    L(τ) = ( log Z + Σ_t log P_F(s_{t+1}|s_t) − log R(s_T) − Σ_t log P_B )²
    P_B 均匀 → 后一项为常数，损失简化为 (logZ + ΣlogP_F − logR + C)²。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["TrajectoryBalanceGFN"]


class TrajectoryBalanceGFN(nn.Module):
    """固定长度轨迹 + 终端奖励的 TB-GFlowNet。

    P_F(s, a) = softmax(logits_F(s))，P_B 均匀（v0）；Z 可学习。
    状态 = 0..n_states-1 的离散索引；动作 = n_actions 个名义位移（δ 由环境
    提供，越界裁剪由环境采样函数负责）。
    """

    def __init__(self, n_states: int, n_actions: int, hidden: int = 64,
                 seed: int = 0, in_dim: int | None = None,
                 feat_fn=None):
        super().__init__()
        torch.manual_seed(seed)
        self.n_states = n_states
        self.n_actions = n_actions
        # 状态表示：默认 one-hot；**feat_fn 提供结构化特征**（缺口②：
        # one-hot 在结构空间（3376 态）无泛化——相邻结构态嵌入互相独立，
        # v14 首跑基线 0.00% 命中即此因；design doc §2.6.2 第 2 条
        # 「智能体状态直接用 A 的冻结骨干嵌入」的 v0 落地 = 结构化特征）
        self.feat_fn = feat_fn
        d_in = in_dim if in_dim is not None else n_states
        self.d_in = d_in
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        # 可学习反向策略 P_B（TB 标准升级：v0 的均匀 P_B 无法给组合策略
        # 「先走位再切换」路由信用——v6 第四轮诊断）
        self.bnet = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        # 状态流函数 F(s)（SubTB，M6.1：信用在子轨迹粒度分配）
        self.fnet = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.logZ = nn.Parameter(torch.zeros(1))
        self._opt: torch.optim.Optimizer | None = None   # 持久优化器
        self.pb_alpha: float = 0.1   # P_B 与均匀分布混合系数（防 TB 协同塌缩）
        self._buffer: list[tuple] = []   # 回放缓冲（M6.1：off-policy 化）

    def features(self, states: torch.Tensor) -> torch.Tensor:
        """状态特征（one-hot 或 feat_fn 结构化特征）。"""
        if self.feat_fn is None:
            return torch.nn.functional.one_hot(
                states.long(), num_classes=self.n_states).float()
        arr = np.stack([self.feat_fn(int(s)) for s in states.tolist()])
        return torch.tensor(arr, dtype=torch.float32)

    def _logits(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(states))

    def forward_log_probs(self, states: torch.Tensor,
                          actions: torch.Tensor) -> torch.Tensor:
        """给定 (state, action) 批量，返回 log P_F(action|state)。"""
        logits = self._logits(states)
        return torch.log_softmax(logits, dim=-1).gather(1, actions[:, None])[:, 0]

    def sample_trajectories(self, env, n: int, horizon: int,
                            start_states: torch.Tensor | None = None,
                            eps: float = 0.0,
                            return_entropy: bool = False):
        """rollout n 条轨迹，返回 (states, actions, log_P_F_sum, rewards)。

        env 须提供：n_states / n_actions / deltas / step(state, action)
        返回 (next_state, reward)（reward 仅在终止步非零）。
        起点默认**均匀随机**（标准 GFN 实践，避免固定起点引入模式偏差）。
        eps>0：以概率 eps 取均匀随机动作（ε 探索；off-policy 偏差风险，
        v6 第三轮已实测反噬，慎用）。
        return_entropy=True：额外返回每步 P_F 熵均值（熵正则用）。
        """
        if start_states is None:
            start_states = torch.randint(0, self.n_states, (n,),
                                         dtype=torch.long)
        states = [start_states.clone()]
        acts: list[torch.Tensor] = []
        log_pf = torch.zeros(n)
        log_pb = torch.zeros(n)
        log_pf_steps: list[torch.Tensor] = []
        log_pb_steps: list[torch.Tensor] = []
        total_ent = torch.zeros(1)
        cur = start_states
        rewards = torch.zeros(n)
        shaping = torch.zeros(n)
        term_active = hasattr(env, "is_terminal_action")
        term_a = env.n_actions - 1 if term_active else -1
        absorbing = env.n_states - 1 if term_active else -1
        reg = torch.full((n,), -1.0)   # 终止锁定的奖励寄存器
        for _ in range(horizon):
            logits = self._logits(cur)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            if eps > 0.0:
                explore = torch.rand(a.shape) < eps
                if explore.any():
                    a_uni = torch.randint(0, self.n_actions, a.shape)
                    a = torch.where(explore, a_uni, a)
            if term_active:
                at_abs = (cur == absorbing)
                if at_abs.any():
                    a = torch.where(at_abs, torch.full_like(a, term_a), a)
            lp = dist.log_prob(a)
            if term_active:
                lp = torch.where(at_abs, torch.zeros_like(lp), lp)
            log_pf = log_pf + lp
            log_pf_steps.append(lp)
            total_ent = total_ent + dist.entropy().mean()
            if term_active:
                # 终止动作：锁定该轨迹在此状态的奖励（吸收态后不再变化）
                for k, (s, ac) in enumerate(zip(cur.tolist(), a.tolist())):
                    if ac == term_a and float(reg[k]) < 0:
                        reg[k] = env.reward(int(s))
            nxt = torch.tensor(
                [env.step(int(s), int(ac))[0]
                 for s, ac in zip(cur.tolist(), a.tolist())],
                dtype=torch.long)
            if hasattr(env, "action_bonus"):
                shaping = shaping + torch.tensor(
                    [env.action_bonus(int(s), int(ac))
                     for s, ac in zip(cur.tolist(), a.tolist())],
                    dtype=torch.float32)
            # P_B：在**下一状态**上评估反向动作（环境提供 reverse_action）；
            # 与均匀分布混合（α=0.1）防 TB 协同塌缩（v6 第五轮诊断）；
            # 终止边/吸收态边的 P_B=1（log 0，文档化）
            if hasattr(env, "reverse_action"):
                a_rev = torch.tensor(
                    [env.reverse_action(int(ac)) for ac in a.tolist()],
                    dtype=torch.long)
                pb = torch.log_softmax(self.bnet(self.features(nxt)),
                                       dim=-1)
                pb_mix = torch.log(torch.exp(pb) * (1.0 - self.pb_alpha)
                                   + self.pb_alpha / self.n_actions)
                pb_step = pb_mix.gather(1, a_rev[:, None])[:, 0]
                if term_active:
                    term_edge = ((a == term_a) | (nxt == absorbing))
                    pb_step = torch.where(term_edge,
                                          torch.zeros_like(pb_step), pb_step)
                log_pb = log_pb + pb_step
                log_pb_steps.append(pb_step)
            cur = nxt
            states.append(cur.clone())
            acts.append(a.clone())
        # 终止奖励（末状态；终止轨迹用锁定寄存器）
        rewards = torch.tensor(
            [env.step(int(s), 0)[1] for s in cur.tolist()],
            dtype=torch.float32)
        if term_active:
            rewards = torch.where(reg >= 0, reg, rewards)
        if return_entropy:
            return (states, acts, log_pf_steps, log_pb_steps, rewards, cur,
                    total_ent, shaping, log_pf, log_pb)
        return (states, acts, log_pf_steps, log_pb_steps, rewards, cur,
                shaping, log_pf, log_pb)

    def train_step(self, env, horizon: int, batch: int, lr: float = 1e-2,
                   opt: torch.optim.Optimizer | None = None,
                   eps: float = 0.0, entropy_bonus: float = 0.0):
        """一步 TB 训练：rollout batch 条轨迹 → TB 损失 → 更新。

        优化器持久复用（首次调用创建）——此前每步新建 Adam 导致动量清零、
        训练近乎停滞（v5 首跑诊断，2026-09-29）。
        entropy_bonus>0：TB − λ·H(P_F)（熵正则，保持罕见动作存活，破
        「早负强化塌缩」——v6 切换动作死锁的标准解法，无 off-policy 偏差）。
        """
        if opt is None:
            if self._opt is None:
                self._opt = torch.optim.Adam(self.parameters(), lr=lr)
            opt = self._opt
        out = self.sample_trajectories(env, batch, horizon, eps=eps,
                                       return_entropy=(entropy_bonus > 0.0))
        if entropy_bonus > 0.0:
            states, acts, pf_s, pb_s, rewards, cur, total_ent, shaping, \
                log_pf, log_pb = out
        else:
            states, acts, pf_s, pb_s, rewards, cur, shaping, log_pf, \
                log_pb = out
        # logR = log(任务奖励) + 形状项（结构编辑固有价值）；P_B 可学习
        # （v0.1 升级：信用路由到组合策略的「先走位再切换」路径）
        log_r = torch.log(rewards.clamp(min=1e-6)) + shaping
        loss = (self.logZ + log_pf - log_r - log_pb) ** 2
        loss = loss.mean()
        if entropy_bonus > 0.0:
            loss = loss - entropy_bonus * total_ent
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.item())

    def pretrain_pf(self, pairs: list[tuple[int, int]],
                    steps: int = 400, lr: float = 1e-2) -> float:
        """专家演示初始化 P_F（缺口②：组合/结构动作的早负强化死亡对策）。

        pairs = [(state, action)]——对演示轨迹上的状态-动作对做交叉熵模仿，
        让结构级动作在 RL 开始前就拥有正概率（M6/M6.2 诊断的处方）。
        """
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        S = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        A = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        loss_val = float("nan")
        for _ in range(steps):
            logits = self._logits(S)
            loss = torch.nn.functional.cross_entropy(logits, A)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_val = float(loss.item())
        return loss_val

    def pretrain_flow(self, pairs: list[tuple[int, float]],
                      steps: int = 300, lr: float = 1e-2) -> float:
        """离线流预训练（§2.6.2 第 (i) 条：**DB4 经验库作离策略训练源**）。

        pairs = [(state, log_reward)]——把已知状态的状态流拟合到其对数奖励
        （F(s) ← R(s)，GFN 的标准离线信号），再进入在线 SubTB 微调。
        """
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        S = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        Y = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
        loss_val = float("nan")
        for _ in range(steps):
            logF = self.fnet(self.features(S)).squeeze(-1)
            loss = ((logF - Y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_val = float(loss.item())
        return loss_val

    def _subtb_loss(self, S, PF, PB, log_r):
        """SubTB 损失核心（S: (T+1,B) 状态、PF/PB: (T,B)、log_r: (B,)）。"""
        T, batch = PF.shape[0], PF.shape[1]
        feats = self.features(S.reshape(-1)).reshape(T + 1, batch, -1)
        logF = self.fnet(feats).squeeze(-1).clone()        # (T+1, B)
        logF[T] = log_r                                    # 终止覆盖
        cumPF = torch.cat([torch.zeros(1, batch),
                           torch.cumsum(PF, dim=0)], dim=0)
        cumPB = torch.cat([torch.zeros(1, batch),
                           torch.cumsum(PB, dim=0)], dim=0)
        loss = torch.zeros(1)
        total_weight = 0.0
        for l in range(1, T + 1):
            idx_i = torch.arange(T - l + 1)
            idx_j = idx_i + l
            residual = (logF[idx_i] - logF[idx_j]
                        + (cumPF[idx_j] - cumPF[idx_i])
                        - (cumPB[idx_j] - cumPB[idx_i]))
            w = self._lam ** (l - 1)
            loss = loss + w * (residual ** 2).mean()
            total_weight += w
        return loss / max(total_weight, 1e-9)

    def train_step_subtb(self, env, horizon: int, batch: int,
                         lr: float = 1e-2,
                         opt: torch.optim.Optimizer | None = None,
                         eps: float = 0.0, entropy_bonus: float = 0.0,
                         lam: float = 0.9, use_replay: float = 0.5,
                         buffer_cap: int = 400):
        """SubTB（子轨迹平衡，M6.1）：信用在子轨迹粒度分配 + 回放缓冲。

        L = Σ_{i<j} λ^{j−i−1} ( logF(s_i) + Σ_{i..j−1}(logP_F−logP_B)
            − logF(s_j) )² / Σλ^{l−1}，终止 logF(s_T) = logR + 形状。
        use_replay：以该概率从回放缓冲取旧轨迹训练（off-policy 化——防
        「组合路径只在训练早期被采样一次、随后被遗忘」的探索侧塌缩）。
        """
        if opt is None:
            if self._opt is None:
                self._opt = torch.optim.Adam(self.parameters(), lr=lr)
            opt = self._opt
        self._lam = lam
        out = self.sample_trajectories(env, batch, horizon, eps=eps,
                                       return_entropy=True)
        states, acts, pf_s, pb_s, rewards, cur, total_ent, shaping, \
            _, _ = out
        log_r = torch.log(rewards.clamp(min=1e-6)) + shaping

        # 入缓冲（仅新鲜轨迹）+ 以 use_replay 概率回放
        import numpy as np
        if len(self._buffer) < buffer_cap:
            self._buffer.append((np.stack([s.numpy() for s in states]),
                                 np.stack([p.detach().numpy() for p in pf_s]),
                                 np.stack([p.detach().numpy() for p in pb_s]),
                                 log_r.detach().numpy()))
        elif self._buffer:
            self._buffer.pop(0)
            self._buffer.append((np.stack([s.numpy() for s in states]),
                                 np.stack([p.detach().numpy() for p in pf_s]),
                                 np.stack([p.detach().numpy() for p in pb_s]),
                                 log_r.detach().numpy()))
        if self._buffer and np.random.rand() < use_replay:
            s_arr, pf_arr, pb_arr, lr_arr = self._buffer[
                int(np.random.randint(len(self._buffer)))]
            S = torch.tensor(s_arr, dtype=torch.float32)
            PF = torch.tensor(pf_arr, dtype=torch.float32)
            PB = torch.tensor(pb_arr, dtype=torch.float32)
            log_r_use = torch.tensor(lr_arr, dtype=torch.float32)
        else:
            S = torch.stack([s.float() for s in states])
            PF = torch.stack(pf_s)
            PB = torch.stack(pb_s)
            log_r_use = log_r

        loss = self._subtb_loss(S, PF, PB, log_r_use)
        if entropy_bonus > 0.0:
            loss = loss - entropy_bonus * total_ent
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.item())

    @torch.no_grad()
    def sample_terminals(self, env, n: int, horizon: int) -> np.ndarray:
        """采样 n 个终止状态（验证用）。"""
        _, _, _, _, _, cur, *_ = self.sample_trajectories(env, n, horizon)
        return cur.numpy().astype(int)
