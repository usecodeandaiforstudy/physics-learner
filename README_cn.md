# 信息力学跨域设计系统

**Information Mechanics Cross-Domain Design System**

> [English](README.md) ｜ **中文**

> 一套把**需求文本**变成**可验证设计**的跨领域流水线。
> 它的统一语言不是领域术语，而是**信息量**：所有需求都被翻译成同一组
> 信息指标（q 向量），因此不同领域的"需求 → 设计 → 验证"可以复用同一套机制。
> 理论上它只依赖一条公理——**数据处理不等式（DPI）**。
>
> **背后的基本判断**：**物理系统本质上仍是信号处理系统，仍受信息论约束；
> 物理场承载的信息不能以违反信息论的方式传递，从而间接约束了物理场的演化。**
> 这条判断及其可检验推论，就是本项目的哲学与设计出发点
> （见 [§1.1](#11-基本判断本项目的哲学出发点)）。

**它是什么**：可直接运行的研究/工程原型。跑得通、数字可审计、边界写在案 ✓
**它不是什么**：不是产品级系统，也不是"用 AI 发现新物理" ✗（详见
[`CLAIMS_AND_BOUNDARIES.md`](CLAIMS_AND_BOUNDARIES.md)）

---

## 目录

1. [核心思想](#一核心思想为什么用信息量做统一语言)
2. [系统架构](#二系统架构现在的主链)
3. [五个数据库](#三五个数据库db1db5)
4. [现在能做到什么程度](#四现在能做到什么程度)
5. [目录与每个文件的作用](#五目录与每个文件的作用)
6. [快速开始](#六快速开始)
7. [数据红线（必读）](#七数据红线必读)
8. [许可、引用与延伸阅读](#八许可引用与延伸阅读)

---

## 一、核心思想：为什么用"信息量"做统一语言

### 1.1 基本判断（本项目的哲学出发点）

> **物理系统本质上仍然是一个信号处理系统，仍然受信息论的约束。**
> **一个物理场所承载的信息，不能以违反信息论的方式传递**——
> 于是信息论**间接地约束了物理场的演化**。

这句话是本项目的**基本判断**（立场），不是被证明的定理 ✓。
它的力量在于**可检验的推论**，而本项目的全部机制都是这些推论的实现：

| 判断的分解 | 推出的东西 | 项目里的落地 |
|---|---|---|
| **① 物理系统 = 信号处理系统** | 任何物理过程都可写成"环境/初始条件（X）→ 内部状态（T）→ 可观测/功能（Y）"的**信息通道** | `system_encoder` 把物理系统编码为 X→T→Y 结构图；`fingerprint_computer` 算其信息指纹 |
| **② 场承载的信息不能违反信息论** | 信息**传递**必须服从 **DPI**：`I(X;T2) ≤ I(X;T1)`，只能丢不能造 | `core/fisher.py`、`core/ib.py`；运行器中的 **DPI/Fisher 约束检查**把它当**硬门禁** |
| **③ 于是"演化"被间接约束** | 场的可用**分辨力有上限** ⟹ "精度/响应/输运"等需求指标都有**信息论的界**，不是可任意调的工程参数 | **q 向量**（q1 精度 / q7 前缘 / q9 谱隙…）＋ **q→参数反解**（21 族）＋ **β 标定** |
| **④ 推论：违反 DPI 的设计物理上不可能** | "不可能"是**先验排除**，不是仿真跑出来的经验结论 | 验证 API 的**三态**：域外/不成立一律 `unknown`，**绝不报 `pass`**；DPI 余量（q4）被打成可检查的量 |

**这个判断带来的三个直接后果**（也是本项目的三条设计原则）：

1. **统一语言**：底层约束来自同一条信息论不等式，领域之间的差异只是"**信道类型不同**"，
   而不是"**约束不同**" ⟹ 需求可被翻译成**同一组 q 量**，解析/反解/验证的机制得以
   **跨领域复用**（27 族 / 52 域共用一套）。
2. **可验证**：若物理受信息论约束，则"某设计是否可达"是**可判定**的问题——
   验证不是"跑个仿真看看"，而是**对着界做三态判定**（`pass`/`fail`/`unknown`），
   并附**不确定度与证书**。
3. **可解释的边界**：约束是**普遍**的（DPI 处处成立），但**具体容量是族特定的**
   （信道类型、支撑、噪声结构不同）⟹ 这正解释了本项目实测的边界：
   **跨族可预测的是"曲线形状尺度"（认证 3.0% ✓），而不是"绝对容量"（未过 ✗）**。

> **不声称什么** ✗：本项目**不**声称"由信息论推出了新物理"，也**不**把这条判断当作
> 已证定理。它是**指导性判断**，价值在于**可检验的推论**——这些推论在仓库里
> 被实现、被门禁守住、被外部方法印证 ✓
> （展开推导与划界见 `md/information mechanics derivations.md`、
> `md/information mechanics delimitation.md`、`md/why these limits.md`）

### 1.2 唯一公理：数据处理不等式（DPI）

信息处理的任何一步都**不会增加**关于目标的信息：

```
I(X; T2) ≤ I(X; T1)        （X 原始数据，T 中间表征，Y 目标）
```

这条不等式**不含任何自由参数**，是整套体系的唯一出发点。它给出一个硬约束：
表征只能**丢**信息，不能凭空造信息——因此"压缩多少 / 保留多少"成为一个**可计算**的权衡。

### 1.3 从公理到度量：Fisher 度规

在参数空间上，唯一与 DPI 相容、且满足不变性的自然度量是 **Fisher 信息度规**：

```
g_ij(θ) = E[ ∂log p(x|θ)/∂θ_i · ∂log p(x|θ)/∂θ_j ]
```

它把"测量精度"变成几何量：**Fisher 越大 = 越容易分辨参数**。
本项目对 **27 个分布族**实现了 Fisher（解析式或数值/蒙特卡洛），并逐一与真值校准。

### 1.4 从度量到动力学：信息拉氏量

把 Fisher 度规当作"动能项"，把信息瓶颈（IB）泛函当作"势能项"，就得到**信息拉氏量**：

```
L_info = ½ · θ̇ᵀ g(θ) θ̇  −  [ I(X;T) − β·I(T;Y) ]
         └ 动能（Fisher 度规）┘   └ 势能（IB 泛函）┘
```

（实现见 `core/action.py`，符号约定见 `db1/conventions.yaml`）
它的意义：**"最优表征"= 使作用量取极值的轨迹**，而不是靠调参试出来的。
本项目不把它当作训练目标，而当作**理论层的桥接与校验**（变分层与理论约束检查在持续跑它）。

### 1.5 从动力学到工程：q 向量

工程上需要一个**可测量、可比较、可复现**的接口。于是定义一组信息指标 **q 向量**：

| 指标 | 含义 | 典型场景 |
|---|---|---|
| `q1` Fisher | 测量/控制的**精度**（信息量） | 传感器精度、分辨力 |
| `q7` 前缘速度 | **输运/传播**快慢 | 热传导、扩散 |
| `q9` 谱隙 | **响应/回稳**速度（Fokker–Planck 第二特征值） | 控制回路、执行器 |
| `q4/q5/q8/q10` | DPI 余量 / 信道容量 / 稳定余量 / 冗余 | 鲁棒性、适应性、稳定性 |

**关键点**：一个需求（"不确定度 ≤0.5%"、"响应 ≤20 ms"、"导热要快"）
经翻译后都变成"某几个 q 量的界"。**跨域复用就发生在这一层**——
不同领域的物理不同，但 q 量的语义相同，所以判据、反解、验证器可以共用。

### 1.6 IB 曲线与"容量/代价"权衡

每个系统的信息行为由一条 **IB 曲线**描述：横轴 β（压缩强度），
纵轴 `I(X;T)`（压缩量）与 `I(T;Y)`（任务相关信息）。曲线平台值即该系统的**容量**。
本项目对每个系统计算并存储这条曲线（DB2），用于容量分析、代理建模与跨族研究。

### 1.7 这套思想在工程上换来什么

| 收益 | 具体表现 |
|---|---|
| **跨域复用** | 需求解析、反解、验证的机制与代码**不分领域**（27 族 / 52 域共用） |
| **可验证** | 每条判据都能对应到可计算的量，且**域外一律返回 `unknown`**（不猜） |
| **可审计** | 每次验证落盘，带模型、容差来源、有效域、已知偏差、判定式 |
| **诚实** | 主张与边界写在一页里；负结果不删；口径不一致会被门禁抓住 |

---

## 二、系统架构（现在的主链）

### 2.1 一张图

```
                     ┌──────────────── 五库 ────────────────┐
                     │ DB1 物理系统  DB2 信息指纹  DB3 需求映射 │
                     │ DB4 RL 经验   DB5 解码验证              │
                     └───────────────────────────────────────┘
需求文本（DB3 锚点）
   │ ①
   ▼
B 解析器 ──（弃权：转人工队列）──┐
   │ ② 属性集合 / 方向 / 数值     │
   ▼                            │
奖励类型分类（信息量 / 约定 / 约束）│   ← 一等公民：决定"走哪条路"
   │ ③                          │
   ├── 信息量型 ──► q 指标 ──► q→参数反解（21 族）
   ├── 约定型   ──► 查表 / 约定
   └── 约束型   ──► 约束路径（**不适用 β**）
   │ ④
   ▼
设计（**状态 = 设计向量**）──► C 生成器：需求 → 设计分布
   5 个设计域：two_level / rayleigh / landau / chain / array
   │ ⑤
   ▼
验证 API（27 族）──► pass / fail / **unknown**
   ＋ 不确定度 u ＋ 距边界距离 ＋ 稳健性 ＋ **证书**
   │ ⑥
   ▼
反馈闭环：reflow → DB5 ＋ 加权数据集 → 训练消费
   │ ⑦
   └─► RL（**可选层**）：设计向量空间接口，仅用于非信息量目标
```

### 2.2 逐层说明

#### ② B 解析器（需求文本 → q 结构）

| 项 | 说明 |
|---|---|
| **输入** | 一行需求文本（中/英混合） |
| **输出** | 属性集合（如 `precision`）、方向（`bound_above/below`）、置信度；**低置信则弃权** |
| **实现** | `b_network/bv1_parse.py`（模型）、`features_v2.py`（特征）、`b_network/b_parse.py`（生产入口） |
| **特征** | 关键词/单位/数字 ＋ 字符 3-gram/英文词 ＋ **规则谓词引导的双语线索块**（依据 `db3/rules/rules.yaml` 的物理谓词，**不是域标签**） |
| **弃权** | `τ = 0.50`（由**风险–覆盖曲线**选定）⟹ 弃权即**转人工**，绝不硬猜 |
| **实测** | 域内 LOOCV **87.0%**；域外（严格 LODO 52 域）传导 **85.2%** / 归纳 **82.8%**；方向 98.5%；**弃权 13–17%** |

#### ③ 奖励类型分类（一等公民）

一个需求先要判断"它是什么类型的约束"，因为**类型决定走哪条路**：

| 类型 | 占比 | 处理 |
|---|---|---|
| **信息量型** | 46% | 转成 q 目标 ⟹ 反解物理参数、标定 β |
| **约定型** | 33% | 查表/标准约定 |
| **约束型** | 21% | 走**约束路径**（如势垒/疲劳类：**β 不适用** ✓） |

实现：`pipeline/reward_types.py`；β 标定：`pipeline/beta_calibration.py`。

#### ④ 设计（状态 = 设计向量）

**关键设计取舍**：状态不是图结构，而是**设计向量**——同一状态可对应多个设计（非退化）。

| 设计域 | 维度/类型 | 说明 |
|---|---|---|
| `two_level` | 2 dof（ΔF, T） | 同一状态 **200 个设计** |
| `rayleigh` | 幅值 σ | 声学/光学幅值类 |
| `landau` | 势函数参数 | 高维域，C 实例最成熟 |
| `chain` | 7 维（**混合**） | 组合链（多物理串联） |
| `array` | 5 维（**混合**） | 声学阵列几何（FFT 阵列因子） |

实现：`pipeline/design_space.py`（注册表 + 编码/解码）、`design_contract.py`（解码契约）。
**C 生成器**（需求 → 设计分布）：`c_network/cfm.py`（条件流匹配，含精确对数密度）、
`c_network/mdn.py`（领域专用 MDN）。

#### ⑤ 验证（三态 + 证书）

| 特性 | 说明 |
|---|---|
| **三态** | `pass` / `fail` / **`unknown`** — 域外、缺目标、适配器异常**一律 `unknown`**，绝不报 `pass` |
| **有效域登记** | `sim/registry.json`：27 族，每族登记**校准通过**的度量与容差 |
| **容差** | `max(floor, k/√n)`，`k` 由**真值校准** |
| **证书 + 不确定度** | 每次验证附 **u（不确定度）**、**距判定边界的距离（x·u）**、**稳健性**（robust/marginal）与**证书**：模型 / 容差来源 / 有效域 / 支持度量 / **已知偏差** / **是否有第三方外部印证** / **判定式** / **可复现的设计输入** |
| **审计** | `sim/verify_audit.jsonl`（追加式，每行含证据与证书） |

实现：`sim/verify_api.py`（API）、`sim/expand_registry.py`（校准与登记）、
`sim/simulator_adapter.py` / `rate_adapter.py` / `front_adapter.py`（适配器）。

#### ⑥ 反馈闭环

验证结果 → **只写通过者** → DB5 ＋ **加权数据集**（`reflow_weighted_*.csv`）→ 训练消费。
实现：`pipeline/design_reflow.py`、`design_reflow_consume.py`。

#### ⑦ RL（可选层）

**定位**：只为**非信息量目标**（成本、约束）预留的接口；**状态 = 设计向量**。
**明确不声称性能** ✗（基线 ES 均值**不如随机**）。
实现：`pipeline/rl_design_env.py`；研究代码在 `a_network/`（**不在主链上** ✓）。

---

## 三、五个数据库（DB1–DB5）

| 库 | 内容 | 规模 | 谁在用 | 红线 |
|---|---|---|---|---|
| **DB1** | 物理系统条目：域/族/参数/方程 ＋ **声明式 Fisher 指纹** | **300 条** | 校准真值底座、β 标定、验证容差 | 机器门 **300/300** ✓；人工抽检**清单 60/300（20%）**、复核进度 **294 待办 / 6 已抽检**；**AI 生成不得作锚点** |
| **DB2** | 信息指纹：每系统的 IB 曲线、Fisher 迹、容量 | **8879 行**（可用曲线 **99.7%**） | 反解、代理建模、跨族研究 | 曲线与指纹须与真值口径一致（有门禁 ✓） |
| **DB3** | 需求映射锚点：文本 ↔ q 结构（属性/方向/数值） | **169 条** | B 解析器的训练与评估 | **须 A/B 双角色人工复核**；校验 169/169 ✓ |
| **DB4** | RL 经验（历史层） | 87 条 | RL 研究 | 仅作历史参照 ✓ |
| **DB5** | 解码验证结果（只写通过者） | 390 条 | 反馈闭环、加权训练集 | 与 `verdict ∈ {pass, fail}` 一致 ✓ |

---

## 四、现在能做到什么程度

### 4.1 能力表（均带实测与复现命令）

| 能力 | 实测 | 复现 |
|---|---|---|
| **需求文本 → q 结构** | LOOCV **87.0%**；严格 LODO 归纳 **82.8%**；方向 **98.5%**；**弃权 13–17% 转人工** | `python b_network/bv1_train_eval.py`、`python b_network/v31_lodo_check.py` |
| **q → 参数反解** | **21 族**（三级优先：解析式 → 数值反解 → **如实 None**） | `python pipeline/inverse_fisher.py` |
| **需求 → 设计分布（C）** | **4/4 设计域**（各 20 seed、双门禁）：landau 88.4% / rayleigh 96.8% / chain 82.8% / array 76.2% | `python pipeline/v3*_c_check.py` |
| **设计 → 验证** | **27/27 族**登记；**域外一律 `unknown`**；**测量与真值一致 ≤10%**（逐族偏差 **0.00%–1.32%**；**常设门禁** ✓） | `python sim/verify_api.py`、`python sim/registry_truth_scan.py` |
| **验证可采信度** | 每个 `pass` 自带**不确定度 + 距边界 + 稳健性 + 证书** | `python sim/certificate_check.py`、`python tools/demo_minimal.py` |
| **第三方外部印证** | **5/27 族**：`diffusion/q7`、`two_level`、`rayleigh`、`weibull`、`landau`（三路外部方法互洽，项目 vs 外部 **0.00%** ✓） | `python sim/external_check_diffusion.py`、`python sim/external_check_five_families.py` |
| **反馈闭环** | reflow → DB5（390）＋ 加权数据集 → 训练消费（**实质闭合** ✓） | `python pipeline/demo_end_to_end.py --pair DB3-0009` |
| **全链门禁** | fast 档 **38 阶段 = PASS 36 / 边界 2 / 已知待修 0 / FAIL 0**；套件 **23/23**；core **17/17** | `python run_system.py`、`python tests/test_invariants.py` |

### 4.2 明确做不到的（附原因）

| 不能用于 | 原因 |
|---|---|
| 宣称"跨域统一框架已成立" | 解析侧成立 ✓；**绝对容量**跨族不可预测 ✗（**理论不变量**可，认证 3.0% ✓）⟹ 表述必须**分层** |
| 把 `PASS` 当外部背书 | 验证器**自建**（"设计环外" ≠ 第三方）；第三方印证仅 **5/27 族** ⚠️ |
| 生产系统嵌入 | 无服务化 API；DB1 人工复核未到 100%；**真实需求文本未评估** ✗ |
| 任何 RL 性能声明 | RL 仅接口与门禁，**无性能声明** ✗（SAC 单峰 **73.6% ± 32.7、双稳态**；v14 达标率 **40–45%** ⟹ 均不可用） |
| 全自动文本 → q | 槽位预测已自动 ✓；**数值抽取/规则匹配仍有半自动环节** ✗ |
| 引用为实验事实 | 数据为 **AI 生成 + 人工抽查**，**非实验权威数据** ✗ |

### 4.3 已归档的负结果与边界（当前状态）

| 项 | 状态 |
|---|---|
| **T1 族级指纹迁移** | ✅ 过 |
| **T2 容量族外预测（绝对量）** | ❌ **未过**（现测中位 64.0%）：对数口径无效 ✗；剖面特征降到 17.8% 但**区间跨门槛未认证** ✗；扩行级样本**不能**收紧区间 ✗ |
| **T-c 理论不变量跨族预测** | ✅ **认证达标**：`log₁₀(β@90%)` 族级留一中位 **3.0%**，bootstrap 95% 区间 **[1.4%, 6.6%]**（限定：特征含输入侧尺度 ⟹ 是"输入↔输出尺度一致性"命题） |
| **M6.2 结构编辑** | ✅ **边界稳**（4.50 / 1.75 / 2.00 / 2.00，可复现） |
| **SAC 单峰寻优** | ❌ **73.6% ± 32.7，双稳态 ⟹ 不可用** |
| **v14 结构空间能力曲线** | ❌ **达标率 40–45% ⟹ 不可用** |
| **回流边** | ❌ **0 正 / 1 害 / 7 未定 ⟹ 不可行** |
| **变分层 L2 增量** | ❌ 两个目标量上均**无实用增量** |
| **diffusion 网格对齐偏差** | ⚠️ 已知 **0.512%**（探针在节点而非精确 L）——须随结论一起报 |
| **27 族口径一致性（常设门禁）** | ✅ **逐族测量与真值偏差 0.00%–1.32%**（判据 ≤10%）；**注册表校准 27/27** ✓ |

> **方法学**（本项目最独特的部分）：所有结论执行统一可信度标准 ——
> **≥20 seed ＋ 预登记判据 ＋ 逐点落盘 ＋ 三态判定 ＋ 负面结论同样复检**；
> "不可能的数字"一律按 bug 处理；判据须自检"是否还有区分度"。

---

## 五、目录与每个文件的作用

> **说明**：以下各表列出**关键文件**（入口、判据、数据契约 ✓），**不是**逐文件全量清单 ✓。
> 各包另有 `__init__.py`、`db*/**.schema.json`、`db*/rules|conventions/`、`tools/*.ps1`
> 等支撑文件；未列出**不代表不存在** ✓。

### 5.1 顶层

| 路径 | 作用 |
|---|---|
| `README.md` / `README_cn.md` | 本文（思想 / 架构 / 能力 / 文件说明；英文 / 中文 ✓） |
| `CLAIMS_AND_BOUNDARIES.md` / `_cn.md` | **一页主张与边界表**（能/不能做什么 + 证据 + 引用必带的范围声明） |
| （内部文档，不随发布） | 开发日志与结论可信度索引（**214 条**，含全部否定结果与失败记录）、架构方案、开源发布清单 |
| `run_system.py` | **端到端运行器**：38 阶段门禁（fast）/ ＋18 个重型阶段（`--full`，共 56）；分档、已知边界与已知待修分类 |
| `run_demo.py` | 早期演示脚本（保留） |
| `requirements.txt` | 依赖：`numpy / scipy / sympy / PyYAML / torch`（无其他第三方包） |
| `LICENSE` | 代码 **MIT**；数据与文档 **CC BY 4.0** |
| `*.csv` / `*.jsonl`（根目录） | **证据文件**：LODO / 迁移 / 特征 A/B / τ 扫描 / 外部印证 / 设计 reflow 等逐点结果 |

  **根目录证据文件索引** —— `vNN` = 项目开发过程中第 N 个编号验证会话（按时间顺序；
  各生成脚本的 docstring 记录其问题与所属日志条目），后缀本身语义化：

  | 文件 | 证明什么 | 来源 |
  |---|---|---|
  | `v31_lodo.csv` | 需求→q 解析器的严格留一域外推（LODO）泛化 | `b_network/v31_lodo_check.py` |
  | `v32_full_train_abstain.csv` | 全量训练精度与弃权转人工比例 | `b_network/v32_full_train_abstain.py` |
  | `v33_feature_improve.csv` | 指纹特征 A/B（改进）对照 | `b_network/v33_feature_improve.py` |
  | `v33_runs.csv` / `v34_runs.csv` / `v35_runs.csv` / `v37_runs.csv` | 多 seed 协议的逐 seed 原始运行记录（汇总数字见 CREDIBILITY） | 多 seed 复检（开发树） |
  | `v34_tau_risk_coverage.csv` | 弃权阈值 τ 扫描：风险–覆盖权衡 | `b_network/v34_tau_risk_coverage.py` |
  | `v35_new_anchor_transfer.csv` | 新增锚点域的迁移检验 | `b_network/v35_new_anchor_transfer.py` |
  | `v36_capacity_transfer.csv` | T2 负结果诊断（S1+S2）：容量族外可预测性的口径与特征 | `a_network/v36_capacity_transfer.py` |
  | `v37_row_capacity_transfer.csv` | 行级容量族迁移检验 | `a_network/v37_row_capacity_transfer.py` |
  | `v38_capacity_invariant_transfer.csv` | 容量不变量迁移检验 | `a_network/v38_capacity_invariant_transfer.py` |
  | `v38_rl_interface.csv` | RL 环境接口探针（设计即动作） | `pipeline/rl_design_env.py` |
  | `registry_truth_scan.csv` | 全 27 族三态验证 vs 真值（≤10%） | `sim/registry_truth_scan.py` |
  | `external_check_diffusion.csv` | 外部印证：q7 扩散（三条外部路线，κ̂ 差 0.00%） | `sim/external_check_diffusion.py` |
  | `external_check_five_families.csv` | 外部印证：5/27 族 | `sim/external_check_five_families.py` |
  | `reflow_weighted_chain.csv` / `_rayleigh.csv` / `_two_level.csv` | 验证→设计加权 reflow 汇总（闭环消费证据；消费方：`pipeline/demo_end_to_end.py`、`pipeline/v34_rayleigh_c_check.py`） | reflow 汇总步骤 |

### 5.2 `core/` —— 信息计算核心（定义层）

| 文件 | 作用 |
|---|---|
| `fisher.py` | **Fisher 信息度量**：27 族的解析式 + 数值/蒙特卡洛实现 + 有限差分通用入口 |
| `mutual_info.py` | 互信息估计（KSG k-NN 与 binning 两法） |
| `ib.py` | **信息瓶颈**：Blahut–Arimoto 求 IB 曲线（`I_XT`/`I_TY` vs β） |
| `action.py` | **信息拉氏量**：`ib_potential`（IB 泛函）与 `info_action`（动能−势能） |
| `fingerprint.py` | 信息指纹聚合：节点 Fisher → 迹/行列式/方向性 |
| `channels.py` | 信息结构图编码的接口预留（B 侧使用） |
| `tests/test_core.py` | 核心自测 **17 项**（含 Fisher 符号/定义回归） |

### 5.3 `b_network/` —— B 解析器与系统编码（生产入口）

| 文件 | 作用 |
|---|---|
| `b_parse.py` | **B 生产解析入口**（统一调用：模型槽位 + 数值抽取 + 规则匹配 + 弃权） |
| `bv1_parse.py` | 神经解析器本体：`Bv1Model`（多标签属性 + 方向头）、`featurize`（**v2 线索特征**）、`input_dim`、`Bv1Parser`（含 τ 弃权与转人工） |
| `features_v2.py` | 特征对照工具：生产 v2 特征的再导出 ＋ **旧版特征**（供 A/B 实验） |
| `bv1_train_eval.py` | 训练与评估（LOOCV）＋ 弃权扫描 |
| `bv1_dataset.py` | 训练数据集装配（DB3 锚点 → 文本/槽位标签） |
| `bv1_paraphrase.py` | 语言层扩充（锚点改写 + 分组 LOOCV 复评） |
| `bv1_calibrate.py` | 置信度标定（温度缩放 / Platt） |
| `eval_b_testset.py` | 新表述测试集评估（v0 / v1 / 混合三配置对比） |
| `smoke_b_parse.py` | 生产入口冒烟测试 |
| `system_encoder.py` | **系统 → 信息结构图**（`EncodedGraph`：X→T→Y 节点与边；含 IB 源注册表） |
| `fingerprint_computer.py` | **指纹计算**：结构图 → 指纹（含数值 vs 解析对照、IB 曲线） |
| `metric_spec.py` | 需求侧核心数据类（`MetricSpec`） |
| `requirement_parser.py` | v0 规则解析器：文本 → 结构化需求 |
| `rule_matcher.py` | 结构化需求 → 命中规则 |
| `composer.py` | 命中规则 → `MetricSpec`（规则线组装） |
| `key_registry.py` | 规范键注册表（**单一事实源**，防键名漂移） |
| `v31_lodo_check.py` | **严格 LODO 跨域检验**（留一域 52 域，传导/归纳双口径） |
| `v31b_fail_domain_diag.py` | 全错域的失败模式定位（词表外 / 标注 / 样本不足） |
| `v32_full_train_abstain.py` | 全量训练 + 弃权率对比（A/B 两臂 + LOPO） |
| `v33_feature_improve.py` | 特征 A/B（v1 vs v2 线索块，三口径） |
| `v34_tau_risk_coverage.py` | 弃权阈值 τ 的**风险–覆盖曲线** |
| `v35_new_anchor_transfer.py` | 新锚点**域内迁移**复测（常设阶段） |
| `test_new_encoders*.py`（9 个） | 扩族编码器自测（每批次一族/数族） |
| `test_ib_nonparametric.py` / `test_theory_constraints.py` / `test_encoder.py` / `test_new_families.py` | IB 非参数化 / 理论约束 / 编码器 / 扩族 MC 验证 |

### 5.4 `pipeline/` —— 工程层（中游主链）

| 文件 | 作用 |
|---|---|
| `reward_types.py` | **奖励类型分类（一等公民）**：信息量型 / 约定型 / 约束型 |
| `beta_calibration.py` | **需求 → β 标定**（把占位 β 换成可推导值；约束型不适用） |
| `inverse_fisher.py` | **q → 参数反解**（21 族；解析式 → 数值反解 → 如实 None） |
| `design_space.py` | **设计空间注册表**：5 个域，状态 = 设计向量，`encode/decode_design` |
| `design_contract.py` | 设计解码契约 ＋ 需求反解（打通断点） |
| `design_two_level.py` / `design_rayleigh.py` / `design_landau.py` / `design_chain.py` / `design_array.py` | 各设计域的**预检与非退化性验证**（`array` 含 FFT 阵列因子与栅瓣条件） |
| `v33_landau_c_check.py` / `v34_rayleigh_c_check.py` / `v35_chain_c_check.py` / `v37_array_c_check.py` | **C 实例**：需求 → 设计分布（各 20 seed、双门禁、逐 seed 落盘） |
| `design_reflow.py` / `design_reflow_consume.py` | **验证反馈闭环**：只写通过者 → DB5 ＋ 加权数据集 |
| `family_aliases.py` | 族名别名（**单一真源**） |
| `extra_curves.py` | 势垒曲线与工况索引曲线 |
| `rl_design_env.py` | **RL 接口（可选层）**：设计向量空间环境 ＋ 门禁 |
| `run_loop.py` | 功能级端到端闭环（含断点报告） |
| `demo_end_to_end.py` | **单条需求全链演示**（解析 → 分类 → 反解 → 设计 → 验证 → 反馈） |

### 5.5 `sim/` —— 验证（设计环外 + 第三方外部印证）

| 文件 | 作用 |
|---|---|
| `verify_api.py` | **生产级验证 API**：三态判定 ＋ **不确定度/证书**；域外一律 `unknown` |
| `expand_registry.py` | **校准与登记**：由编码器派生采样器，逐族与真值校准（p99 ≤10% 才登记） |
| `registry_truth_scan.py` | **门禁**：逐族检查「注册表测量 vs 节点真值」一致性（≤10%） |
| `certificate_check.py` | 证书与不确定度的**预登记回归检查**（H1–H5：判定语义不变、字段完整、审计只追加…） |
| `simulator_adapter.py` | 模拟器适配器：解析族 + 动力学（如热传导半值穿透时间） |
| `rate_adapter.py` / `front_adapter.py` | 速率型 / 波前型适配器（`q9`/`q7` 路径） |
| `v36_rate_coverage.py` | 速率型需求的**度量路由与能力可达性**检查 |
| `sim_check.py` / `verify_designs.py` | 模拟器真值自检 / 解析判定 vs 模拟判定 |
| `external_check_diffusion.py` | **第三方印证**：diffusion/q7 三外部方法（解析级数 / Crank–Nicolson / 蒙特卡洛） |
| `external_check_five_families.py` | **第三方印证扩到 5 族**（闭式 / 数值微分积分 / MC 分数平方） |
| `probe_landau_node.py` / `diagnose_special_families.py` / `check_landau_convention.py` / `locate_landau_gap.py` / `decompose_landau_gap.py` | **口径一致性诊断工具链**（定位"同一可观测量"的取值差异） |

### 5.6 `c_network/` —— C 生成器（主链生成侧）

| 文件 | 作用 |
|---|---|
| `cfm.py` | 条件流匹配解码器（含**精确对数密度**） |
| `mdn.py` | 领域专用 MDN 解码器（landau 实例） |
| `v22_cfm_check.py` / `v23_cfm_selfcheck.py` / `v24_mdn_symmetry_check.py` | 覆盖率审计 / 仪器自检 / 对称测量（**先证明尺子准，再判优劣**） |
| `v8_mdn_check.py` / `v12_c_feedback_check.py` / `v16_plasma_check.py` | MDN 判据 / 反馈边 / 领域实例验证 |

### 5.7 `a_network/` —— A 侧表征与 RL 研究（**不在主链上**）

| 分组 | 文件 | 作用 |
|---|---|---|
| **主链仍调用** | `analytic_backbone.py` | 解析背板（快路径解析量计算） |
| | `action_reward.py` | 作用量奖励（`β·I_TY − I_XT` 语义） |
| | `ib_surrogate.py` | IB 曲线代理（从 DB2 组装数据集、训练单调代理） |
| **表征** | `gine.py` / `heads.py` / `struct_embed.py` / `graph_dataset.py` | GINE 编码器 / 六读出头 / 结构嵌入 / 图张量化 |
| **RL 与搜索** | `gflownet.py` / `sac.py` / `cma_es.py` / `graph_env.py` / `structure_env.py` / `req_env.py` / `rl_env.py` / `edit_pipeline.py` | GFlowNet / SAC / CMA-ES / 图编辑与结构环境 / 编辑效应管线 |
| **数据与反馈** | `db5_feedback.py` / `offline_db4.py` / `augment.py` | DB5 回流 / DB4 离策略接线 / 增广 |
| **复检与诊断** | `multi_seed_verify.py` / `v*_check.py`（十余个） / `diagnose_landau_*.py` | 多 seed 复检 / 逐版本能力检查 / landau 诊断 |
| **跨族研究** | `v29_cross_domain_transfer.py` / `v36_capacity_transfer.py` / `v37_row_capacity_transfer.py` / `v38_capacity_invariant_transfer.py` | T1–T3 迁移 / T2 诊断 / 行级复测 / **T-c 理论不变量认证** |

> 这些是**研究代码**：主链不依赖它们（除上表"主链仍调用"三项）✓

### 5.8 `db1/` – `db5/` —— 五库的构建与维护

| 库 | 关键文件 | 作用 |
|---|---|---|
| **DB1** | `validate_entry.py` | **机器门**校验（schema / 约定 / 可算性 / 解析对照） |
| | `review_queue.py` | **人工复核队列**（`next` / `brief` / `worksheet` / `apply`；A/B 双角色） |
| | `independent_check.py` | **独立数值核验**（直接数值微分求 Fisher 定义式，不调用真值函数） |
| | `next_batch.py` / `batch_generate_ingest.py` / `ingest_pipeline.py` / `ingest_sim_data.py` | 批量生成与入库 / 模拟数据入库 |
| | `mode3_generate.py` / `mode3_llm.py` / `gen_landau_deep_entries.py` | 条目生成（模板/深度场景） |
| | `migrate_note_to_source_ref.py` / `remove_systems.py` | 数据迁移与清理 |
| | `db2_writer.py` | DB1 → DB2 的指纹写入 |
| **DB2** | `estimate_fingerprint.py` / `fill_empty_curves.py` / `rebuild_family_curves.py` / `unify_by_family.py` | 指纹估计 / 曲线补齐 / 重建 / 逐族统一（**均带不变量复验**） |
| | `attribute_fill_effect.py` / `m4_variance_check.py` | 数据变更的**归因实验**与判据**方差核查** |
| **DB3** | `validate_pair.py` / `review_queue.py` / `batch_review.py` | 锚点校验 / 复核队列 / 批量复核（A+B） |
| | `check_parser_consistency.py` | **锚点 ↔ 解析器一致性**（门禁：错误数须为 0） |
| | `generate_pairs.py` / `gen_pairs_batch.py` / `propose_anchors.py` | 锚点生成（表驱动）/ **候选起草** |
| | `dedupe_pairs.py` / `migrate_direction_semantics.py` / `post_review_fixes.py` / `confirm_quantity_bounds.py` / `roundtrip_check.py` / `repair_pairs.py` | 去重 / 方向语义迁移 / 复核后修复 / 量词确认 / 往返检查 / **精确修复器** |
| **DB4** | `v9_db4_build.py` / `db4_writer.py` | RL 经验库构建与写入 |
| **DB5** | `v10_db5_build.py` / `db5_writer.py` / `demo_db5_analytic.py` | 解码验证库构建 / 写入 / 演示 |

### 5.9 `tests/` 与 `tools/`

| 文件 | 作用 |
|---|---|
| `tests/test_invariants.py` | **核心不变量套件 23 项**：覆盖**全部已知缺陷类型**（量/口径不匹配、缓存、索引、符号、归一化…），pytest 兼容 |
| `tools/demo_minimal.py` | **最小验证示范**：需求 → q → 参数界 → 设计 → 三态验证 → 外部印证（约 10 秒） |
| `tools/privacy_scan.py` | **隐私/密钥扫描**：13 条预登记规则（密钥/私钥/邮箱/手机号/家目录/环境路径 ＋ 凭据类文件名），**命中即非零退出** |
| `tools/make_release_copy.py` | **生成发布副本**：显式清单 ＋ **镜像剪除**（删目标内多余文件）＋ **复制后自动隐私扫描** |
| `tools/db1_checklist.py` | 生成 DB1 人工复核清单（本地输出，不随仓库发布） |
| `tools/tidy_for_release.py` | 开源前整理（**显式清单、默认干跑、只移不删**） |

### 5.10 `md/` —— 自写理论笔记（**参考读物，非项目产物**）

**本发布版含 17 篇**：推导链与划界（`information mechanics derivations.md`、
`information mechanics delimitation.md`、`appendix derivations.md`）、理论 ↔ AI
双向映射（`physical to ai*.md`、`ai to physical.md`）、专题（信息因果、信息分析、
信息时间、信息量子引力、信息力学中的质量、粒子信息、物理信息、
`physical_information_theory_notes.md`、`theory ai.md`、`why these limits.md`）。

> 这些是**参考读物、不是项目产物**，运行任何阶段都不需要它们。
> **第三方书籍/论文笔记不在本仓库内**（版权原因 ✓）：更大的一组笔记（含教科书摘要）
> 只存在于作者的工作目录里，**有意排除**于此 ✓

### 5.11 工作语言说明

代码注释、文档字符串与**内部开发日志以中文为主**（本项目的工作语言 ✓）；
面向读者的文档在 §5.1 中列出**英文对应版本** ✓。
另需说明：`b_network/` 里的**中文字符串字面量是有意设计**（不是未翻译的残留 ✓）——
需求解析器要处理**中英混合**的需求文本 ✓。

---

## 六、快速开始

```powershell
# 0) 依赖（无其他第三方包）
pip install -r requirements.txt

# 1) 全链门禁（fast 档约 6–8 分钟）
python run_system.py            # 期望：PASS 36 / 边界 2 / 已知待修 0 / FAIL 0（共 38 阶段）
python run_system.py --full     # 追加 18 个重型阶段（训练类 / 多 seed / 严格 LODO / 外部印证 / 证书）
python run_system.py --list     # 只看阶段表

# 2) 核心不变量套件（秒级，23 项）
python tests/test_invariants.py

# 3) 单条需求走完全链（推荐先看这个）
python pipeline/demo_end_to_end.py --pair DB3-0009

# 4) 最小验证示范（含不确定度与证书）
python tools/demo_minimal.py

# 5) 直接调用验证 API
python -c "from sim.verify_api import verify; v=verify({'family':'rayleigh','sigma':0.5,'metric':'q1','target':3.0,'direction':'lower_bound'}); print(v.verdict, v.measured, v.u, v.robustness, v.certificate['verdict_rule'])"
```

**预期输出示例**（验证 `pass` 例）：

```
verdict=pass ｜ measured=15.81 ｜ u=0.0849 ｜ 距边界 +56.07 u ⟹ robust
证书：模型=解析 Fisher（族=rayleigh，度量=q1）｜ 容差来源=… ｜ 有效域={'sigma':[0.1,5.0]}
      已知偏差=未记录 ｜ 外部印证=有（三路外部方法互洽）｜ 判定式=measured ≥ target·(1−tol)
```

### 模型权重（**随仓库提供** ✓）

B 解析器需要两个权重文件，**它们随仓库发布**，因此 **clone 后无需任何训练即可获得完整能力** ✓：

| 文件 | 大小 | 作用 |
|---|---|---|
| `b_network/bv1_model.pt` | 0.33 MB | 神经槽位模型（属性集合 + 方向头） |
| `b_network/bv1_model_cal.pt` | 0.31 MB | 置信度**标定**版本（Platt/温度，用于弃权判定） |

**若你想重训 / 重标定**（例如改了特征或加了 DB3 锚点）：

```powershell
# 重训槽位模型（LOOCV 评估 + 训练全量并保存 b_network/bv1_model.pt）
python b_network/bv1_train_eval.py

# 重新标定置信度（产出 b_network/bv1_model_cal.pt）
python b_network/bv1_calibrate.py
```

**若权重缺失会发生什么**（设计如此 ✓）：`b_parse` **不会报错**，而是**优雅回退**到
v0 规则解析器，并在结果里**明确标注**状态：

```python
r = parse("执行器从阶跃指令到 95% 行程的响应时间不超过 20 ms。")
r.status      # "auto"（神经）｜ "human_queue"（弃权转人工）｜ "v0_fallback"（模型缺失）
```

> ⚠️ **口径提示**：回退到 v0 时**能力显著下降**——在同一**新表述测试集**上，
> v1 的属性/方向为 **74.1% / 80.0%**，v0 为 **59.3% / 60.0%**（同口径可直接比 ✓）；
> 而 README 前面列的 **87.0%** 是**域内 LOOCV（169 锚点）**，属**另一种协议，不可与上面两组混用** ✓
> ⟹ 看到 `v0_fallback` 就说明**权重没加载上** ✓
> 另外生产弃权阈值以**代码中的显式 τ = 0.50** 为准（权重文件内记录的 τ 仅作参考 ✓）

---

## 七、数据红线（必读）

1. **本仓库绝大多数数据为 AI 生成**（模板填充 + 机器门校验），
   人工复核状态见每条目 `provenance.human_review_status`
   （`pending` / `spot_checked` / `fully_reviewed`）。
2. **AI 生成条目永不作为锚点**（`anchor_eligible` 恒为 `false`）；
   DB3 需求锚点须经 **A/B 双角色人工复核**。
3. **不要把本仓库的数字当作实验事实引用**。每个关键结论都标注了
   seed 数 / 复检状态 / 可否引用（逐条可信度索引在内部开发日志中，不随发布）。
4. **不要删掉** `README.md` / `CLAIMS_AND_BOUNDARIES.md` / `LICENSE` 中的数据性质声明。

---

## 八、许可、引用与延伸阅读

- **代码**：[MIT](LICENSE) ｜ **数据与文档**：CC BY 4.0（见 `LICENSE` 末节）
- **引用**：使用本仓库数据或方法时请引用本仓库，并注意其中标注的**复检状态**
- **延伸阅读**：
  - [`CLAIMS_AND_BOUNDARIES.md`](CLAIMS_AND_BOUNDARIES.md) ｜ [中文](CLAIMS_AND_BOUNDARIES_cn.md) —— 能做什么 / 不能做什么 / 引用必带的范围
  - 内部开发日志（**214 条**，含全部否定结果与失败记录；**不随发布**）
