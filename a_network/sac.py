# -*- coding: utf-8 -*-
"""离散 SAC（Soft Actor-Critic，M7，2026-09-29）——RL 生产回退路径。

定位（design doc §7 预案 + 第 55 条）：单峰寻优场景 SAC 足够且更稳，
多样采样再上 GFN。本模块是 SAC-discrete（Christodoulou 2019）的最小实现：

  - Q 网络 q(s) → Q(s,·)（全动作值）；
  - 策略 π(a|s) = softmax(Q(s,·)/α)（策略由 Q 派生，SAC-discrete 形式）；
  - V(s) = Σ_a π(a|s)·(Q(s,a) − α·log π(a|s))；
  - 损失：J_Q = (Q − (r + γ(1−d)V_target(s')))²；
         J_π = mean_a π·(α·log π − Q_detach)；
  - 目标网络软更新 τ、回放缓冲；温度 α 固定（v0，自动调温属后续）。
零新依赖（仅 torch/numpy）。环境接口：env.step(state, action) →
(next_state, reward)；episode 长度 H 处 done（调用方负责）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["DiscreteSAC"]


class DiscreteSAC(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden: int = 64,
                 alpha: float = 0.5, gamma: float = 0.99, tau: float = 0.005,
                 lr: float = 3e-3, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.q = nn.Sequential(
            nn.Linear(n_states, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.q_target = nn.Sequential(
            nn.Linear(n_states, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad = False
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer: list[tuple] = []
        self.buffer_cap = 4000
    def _qvals(self, states: torch.Tensor) -> torch.Tensor:
        onehot = torch.nn.functional.one_hot(
            states.long(), num_classes=self.n_states).float()
        return self.q(onehot)

    @torch.no_grad()
    def policy(self, states: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self._qvals(states) / self.alpha, dim=-1)

    def act(self, state: int) -> int:
        pi = self.policy(torch.tensor([state], dtype=torch.long))[0]
        return int(torch.multinomial(pi, 1).item())

    def store(self, s: int, a: int, r: float, s2: int, done: bool,
              weight: float = 1.0) -> None:
        """存入经验；weight 为采样权重（DB5 回流的高权重经验，第 63 条）。"""
        self.buffer.append((s, a, r, s2, float(done), float(weight)))
        if len(self.buffer) > self.buffer_cap:
            self.buffer.pop(0)

    def pretrain_on(self, pairs: list[tuple[int, int, float]],
                    steps: int = 300) -> None:
        """监督预训练 Q（DB5 回流暖启动，第 63 条）。

        pairs = [(state, action, log_reward)]——把已验证结构的 Q 值先拟合
        到其对数奖励，再进入正常 RL。与「持久高权重缓冲注入」相比不挤占
        新鲜探索（v11 第二跑教训）。
        """
        for _ in range(steps):
            s = torch.tensor([p[0] for p in pairs], dtype=torch.long)
            a = torch.tensor([p[1] for p in pairs], dtype=torch.long)
            r = torch.tensor([p[2] for p in pairs], dtype=torch.float32)
            q = self._qvals(s).gather(1, a[:, None])[:, 0]
            loss = ((q - r) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            with torch.no_grad():
                for tp, p in zip(self.q_target.parameters(),
                                 self.q.parameters()):
                    tp.mul_(1.0 - self.tau).add_(self.tau * p)

    def train_step(self, batch: int = 64) -> float:
        if len(self.buffer) < batch:
            return float("nan")
        weights = np.array([t[5] for t in self.buffer], dtype=np.float64)
        probs = weights / weights.sum()
        idx = np.random.choice(len(self.buffer), batch, p=probs,
                               replace=True)
        s = torch.tensor([self.buffer[i][0] for i in idx], dtype=torch.long)
        a = torch.tensor([self.buffer[i][1] for i in idx], dtype=torch.long)
        r = torch.tensor([self.buffer[i][2] for i in idx], dtype=torch.float32)
        s2 = torch.tensor([self.buffer[i][3] for i in idx], dtype=torch.long)
        d = torch.tensor([self.buffer[i][4] for i in idx], dtype=torch.float32)

        q = self._qvals(s).gather(1, a[:, None])[:, 0]
        with torch.no_grad():
            q2 = self.q_target(
                torch.nn.functional.one_hot(
                    s2.long(), num_classes=self.n_states).float())
            pi2 = torch.softmax(q2 / self.alpha, dim=-1)
            v2 = (pi2 * (q2 - self.alpha * torch.log(pi2 + 1e-8))).sum(-1)
            y = r + self.gamma * (1.0 - d) * v2
        q_loss = ((q - y) ** 2).mean()

        with torch.no_grad():
            q_detach = self._qvals(s).detach()
        pi = torch.softmax(q_detach / self.alpha, dim=-1)
        pi_loss = (pi * (self.alpha * torch.log(pi + 1e-8)
                         - q_detach)).mean()

        self.opt.zero_grad()
        (q_loss + pi_loss).backward()
        self.opt.step()
        # 目标网络软更新
        with torch.no_grad():
            for tp, p in zip(self.q_target.parameters(),
                             self.q.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)
        return float(q_loss.item())
