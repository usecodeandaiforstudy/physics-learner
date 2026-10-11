# DB5 解码验证库 · Schema（v0，2026-10-02）

定位（design doc §3.6 闭环回流枢纽）：DB4 结构 ─▶ C 解码 ─▶ DB5 验证 ─▶
高权重回流 RL 微调（§2.4）。本库记录**解码出的物理设计 + 对解析真值的
验证结果**。

v0 边界（文档化）：
  - 验证对象 = landau 领域（C v0 解码器实例的唯一领域）；
  - 验证口径（§2.6.2 覆盖率审计的 v0 代理）：模式匹配（预测峰 vs 解析
    峰位置）+ 密度重叠覆盖率（∫min(p_pred, p_true)，≥0.85 判 pass）；
  - 红线：AI+verified、anchor 恒 false。

## 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| validation_id | str | `db5-` + sha1(structure_id) 前 8 位（幂等） |
| structure_id | str | 引用 DB4 记录 |
| domain | str | 领域 |
| decoded_design | dict | 解码设计：参数 + 密度摘要（n_modes、mode_positions、nll） |
| validation | dict | 验证结果：true_modes、overlap（重叠覆盖率）、mode_match、verdict（pass/fail） |
| timestamp / provenance | — | 同 DB4 约定 |

## 判据（§2.6.2 覆盖率审计 v0 代理）

- overlap = ∫ min(p_pred, p_true) dφ（网格数值）≥ **0.85** → pass；
- DB5 汇总的 pass 率 < **90%** → 触发文档级 MDN 淘汰条款（升级 CFM）。
