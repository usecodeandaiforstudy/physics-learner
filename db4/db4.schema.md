# DB4 结构与经验库 · Schema（v0，2026-10-02）

定位（design doc §3.4/3.5）：RL 运行产生的「信息结构 + 经验」（代理生成的设计
候选 + 信息力学评分），是 GFlowNet 离策略训练的主源（§2.6.2），并作为 C
解码的输入（DB4 ─▶ C 解码 ─▶ DB5 验证）。

v0 边界（文档化）：
  - 记录粒度 = 单个设计结构（族 + 参数 + 指纹 + 评分 + 代理标签）；
  - 字段 `q` 以 backbone 量（I_TY/fisher_trace）为主；全图 G* 结构记录的
    编码（节点/边/容量）属 DB4 v1（M6.2 之后）；
  - 红线：全部记录 source=AI+verified、anchor_eligible=false（AI 生成物
    不可作锚点，与 DB1/DB3 同规则）。

## 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| structure_id | str | `db4-` + 8 位哈希（family∥sorted(params)∥agent），幂等去重 |
| domain | str | 族名（landau/elastic/two_level…） |
| params | dict | 设计参数（扫描参数 + 结构常数） |
| q | dict | 信息量摘要：`I_TY`（任务相关信息，nats）、`fisher_trace`（单参数族 trace=I(θ) 精确） |
| reward | float | 任务奖励 R（[1e-6, 1]） |
| log_reward | float | ln R（训练/排序口径） |
| info_action | float | 信息作用量 logR 值（β·I_TY − I_XT，v0 β=4） |
| agent | str | 生成代理：`gfn-tb-v0` / `sac-v0` / `random` |
| policy_tag | str | 任务标签（bimodal / requirement_m6 / …） |
| timestamp | str | ISO 时间 |
| provenance | dict | `{source: "AI+verified", anchor_eligible: false}` |

## 校验规则（db4_writer.validate）

- 必填字段齐全；`domain` 在已知族集合；`params` 为非空 dict；
- `q.I_TY`、`reward` 为数值且 reward ∈ [1e-6, 1]；
- provenance.source == "AI+verified" 且 anchor_eligible 恒 false（红线）；
- structure_id 与 (domain, params, agent) 哈希一致（防篡改/漂移）。
