## 物理场信息传播与神经网络信息传播的统一理论框架：研究思路、数学推导与文献指南

**核心问题**：一个物理场 φ(x,t) 演化的信息学本质，和一个神经网络 f_θ(x) 前向传播的信息学本质，是不是同一个理论架构下的不同实例？如果是，那个架构是什么？

---

## Part 0: 问题定位与香农类比的启示

香农在 1948 年做的事：
- 不问 "电话线怎么传信号"
- 问的是 "任何通信信道的信息极限是什么"
- 结果：发现了一个与物理介质无关的数学框架 (信息论)

我们要做的事：
- 不问 "FHN 介质怎么传螺旋波"
- 问的是 "任何物理场传播和任何神经网络传播的共同信息学基础是什么"
- 目标：找到一个同时描述 PDE 演化与层间映射的数学框架

**类比映射**：

| 香农的通信系统 | 我们的物理-神经统一框架 |
|------|------|
| 信源 | 源点构型 R(t) / 输入数据 x |
| 编码器 | 物理场的初始条件设定 / 权重矩阵 W |
| 信道 | PDE 演化 (FHN) / 层间非线性映射 |
| 噪声 | 扩散耗散 / Dropout、随机梯度 |
| 解码器 | 观测点采样 / 分类头 |
| 信宿 | 损失函数 / 下游任务 |
| 信道容量 | 物理介质的最大可分辨构型数 |

---

## Part 1: 信息场论 (Information Field Theory) — 核心数学框架

### 1.1 基本思想 (Enßlin, 2009-2019)

香农信息论处理离散随机变量 X，其信息量为：

$$\displaystyle I(X;Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$$

信息场论将其推广到无限维场 φ(x)：

$$\displaystyle I[\varphi; \psi] = \int \mathcal{D}\varphi \, \mathcal{D}\psi \; P(\varphi, \psi) \log \frac{P(\varphi, \psi)}{P(\varphi) P(\psi)}$$

其中 $\mathcal{D}\varphi$ 表示函数空间上的泛函积分测度。

### 1.2 物理场的信息泛函

对于 FHN 介质，设 u(x,y,t) 为激发场。定义：
- φ_R(x,y) = 由源点构型 R 生成的稳态场构型
- φ_O = 在观测点集合 O 上的场采样

则物理层作为一个信息信道：

$$\displaystyle C_{\text{physical}} = \max_{P(R)} I[\varphi_R; \varphi_O]$$

这就是"物理神经层"的信道容量。

### 1.3 场的 Fisher 信息矩阵

设 p(S | R, θ) 为给定源点 R 和物理参数 θ 下观测信号 S 的概率密度。Fisher 信息矩阵：

$$\displaystyle F_{ij}(R) = \mathbb{E}\!\left[ \partial_{r_i} \log p(S|R) \cdot \partial_{r_j} \log p(S|R) \right]$$

这个矩阵的几何意义：
- 特征值 λ_k → 源点 R 沿方向 v_k 移动时，观测信号的可分辨程度
- 行列式 det(F) → 局部信息密度
- 迹 Tr(F^{-1}) → Cramér-Rao 下界：源点定位的极限精度

在 FHN 介质中：
- S_j = u(O_j; R) + ε_j（观测 = 场的采样 + 噪声）
- p(S|R) ∝ exp(-||S - u(·;R)||² / (2σ²))

则：

$$\displaystyle F_{ij} = \frac{1}{\sigma^2} \sum_k \frac{\partial u(O_k)}{\partial r_i} \cdot \frac{\partial u(O_k)}{\partial r_j}$$

即 Fisher 信息 = 观测点处场对源点参数的敏感度的外积。

### 1.4 场的信息传输：传递熵 (Transfer Entropy)

Schreiber (2000) 定义的传递熵：

$$\displaystyle T_{X\to Y} = I(Y_{t+1}; X_t \mid Y_t) = \sum P(y_{t+1}, x_t, y_t) \log \frac{P(y_{t+1}\mid x_t, y_t)}{P(y_{t+1}\mid y_t)}$$

在 FHN 介质中放置探针网格 {P_k}，计算两两之间的 T_{P_i→P_j}，得到信息流矢量场 v_info(x,y)。

预期：v_info 的旋度与螺旋波核心位置重合。这给出了"信息流 = 物理因果"的可计算对应关系。

---

## Part 2: 算子谱理论 — 物理传播与网络传播的统一语言

### 2.1 核心观察

物理场演化和神经网络传播都可以写成：

$$\displaystyle \varphi_{t+1} = \mathcal{T}[\varphi_t]$$

其中 T 是一个算子。

| | 物理场 (FHN) | 神经网络 |
|------|------|------|
| 算子 T | L = D∇² + f'(u) | W (权重矩阵) |
| 谱 | 连续谱 {λ_k, e_k(x)} | 离散谱 {σ_k, u_k, v_k} |
| 特征函数/向量 | 拉普拉斯特征函数 | 左右奇异向量 |
| 信息传播 | 高频模态被扩散衰减 | 小奇异值分量被截断 |
| 有效维度 | 场构型的本征维度 | 激活流形的本征维度 |
| 信息瓶颈 | 扩散 = 低通滤波 | 低秩 = 信息压缩 |

### 2.2 物理侧：FHN 算子的谱分解

线性化 FHN 方程在静息态附近：

$$\displaystyle \begin{pmatrix} \partial_t (\delta u) \\ \partial_t (\delta v) \end{pmatrix} = \begin{pmatrix} D\nabla^2 + f'(u^*) & -1 \\ \varepsilon & -\varepsilon\gamma \end{pmatrix} \begin{pmatrix} \delta u \\ \delta v \end{pmatrix}$$

其中 f'(u\*) = 1 - u\*² ≈ 1 - 1.075 ≈ -0.075（在 u\* = -1.037 处）。

令 A(k) 为波数 k 处的 2×2 矩阵：

$$\displaystyle A(k) = \begin{bmatrix} -Dk^2 + f'(u^*) & -1 \\ \varepsilon & -\varepsilon\gamma \end{bmatrix}$$

特征值 λ_{1,2}(k) 决定了波数 k 的分量是指数增长还是衰减。

**信息论解读**：
- 对于 |k| > k_cutoff，Re(λ) < 0 → 高频信息被扩散抹掉
- 对于 |k| < k_cutoff，可能出现 Re(λ) > 0 → 低频信息被非线性放大
- 这个 k_cutoff 就是物理层的"信息带宽"

### 2.3 网络侧：权重矩阵的 SVD

对于一层全连接：h = σ(Wx + b)

W 的 SVD 分解：W = U Σ V^T
- Σ = diag(σ_1, σ_2, ..., σ_n)
- 信息沿 V 的列方向传播，缩放因子为 σ_i

有效信息维度（参与率）：

$$\displaystyle d_{\text{eff}} = \frac{(\sum_i \sigma_i)^2}{\sum_i \sigma_i^2}$$

这是 Tishby 信息瓶颈中 I(X;T) 的自然估计量。

### 2.4 谱对应原理

**关键猜想**：存在一个对应关系——
- FHN 中波数 k 的信息衰减率 ↔ SVD 中奇异值 σ 的衰减率
- 物理场的空间频率谱 ↔ 神经网络的信息瓶颈谱

如果可以建立这个对应，就能回答："什么样的物理介质等价于什么样的网络架构？"

---

## Part 3: 信息瓶颈原理 — 物理与学习的统一变分原理

### 3.1 Tishby 的信息瓶颈 (IB)

IB 拉格朗日量：

$$\displaystyle \mathcal{L}_{\text{IB}} = I(X;T) - \beta I(T;Y)$$

其中：
- X = 输入 (源点构型 / 输入数据)
- T = 中间表示 (场构型 / 隐藏激活)
- Y = 输出目标 (观测信号 / 标签)
- β = 压缩-保真权衡参数

### 3.2 物理介质在 IB 平面上的轨迹

对 FHN 介质：
- X = 源点的位置和振幅向量 R
- T = 介质中任意时刻的场构型 u(x,y)
- Y = 观测点处的信号向量 S

控制参数 β 对应物理参数：
- 小 ε (慢恢复) → 小 β → 更多压缩 (信息在 v 场中累积)
- 大 D → 大压缩 → 向左移动
- I_ext → 外部驱动 → 影响 T 的结构

对神经网络：
- X = 输入数据
- T = 第 L 层的激活
- Y = 输出标签
- 训练 epoch → 沿 IB 曲线向右上移动

### 3.3 信息瓶颈的泛函形式 (推广到连续场)

将 IB 推广到连续场：

$$\displaystyle \mathcal{L}_{\text{IB}}[P(T|X)] = I[X; T] - \beta I[T; Y]$$

其中：

$$\displaystyle I[X; T] = \int \mathcal{D}x \, \mathcal{D}t \; P(x,t) \log \frac{P(x,t)}{P(x)P(t)}$$

$$\displaystyle I[T; Y] = \int \mathcal{D}t \, \mathcal{D}y \; P(t,y) \log \frac{P(t,y)}{P(t)P(y)}$$

对于高斯场 (线性化 FHN)：u = G * s + η（G = Green函数, η = 噪声）

则互信息有闭合形式：

$$\displaystyle I(S; U) = \frac{1}{2} \log \det\!\big(I + \Sigma_S^{-1} G \Sigma_R G^T\big)$$

其中 Σ_S, Σ_R 分别是信号和噪声的协方差。

### 3.4 物理层的信道容量计算

你的 FHN 介质的"计算能力"可以精确量化为：

$$\displaystyle C = \max_{P(R)} I(R; S)$$

对于高斯通道：

$$\displaystyle C = \frac{1}{2} \sum_i \log\!\left(1 + \frac{\lambda_i}{\sigma^2}\right)$$

其中 λ_i 是 G G^T 的特征值，G_ij = ∂S_i/∂R_j（敏感度矩阵）。

这个 G 矩阵可以直接从你的 FHNSolver 数值估计：

$$\displaystyle G_{ij} \approx \frac{u(R + \delta e_j; O_i) - u(R; O_i)}{\delta}$$

---

## Part 4: 自由能原理 — 物理"计算"的热力学解释

### 4.1 Friston 的变分自由能

$$\displaystyle F = D_{\text{KL}}[Q(s) \| P(s)] - \mathbb{E}_Q[\ln P(o|s)]$$

其中：
- s = 内部状态 (FHN 的源点 / 神经网络的权重和偏置)
- o = 观测 (FHN 的观测信号 / 网络的输出)
- Q(s) = 系统对自身状态的变分信念
- P(s) = 先验 (正则化项)
- P(o|s) = 生成模型 (物理传播 / 网络前向)

### 4.2 对应关系

| 自由能分量 | FHN 物理介质 | 神经网络训练 |
|------|------|------|
| E_Q[ln P(o\|s)]：精度 (Accuracy) | 观测信号对目标模式的匹配程度 | 交叉熵损失 |
| D_KL[Q(s)\|P(s)]：复杂度 (Complexity) | 源点构型的能量代价（接近静息态的趋势） | L2 权重衰减正则化 |
| 最小化自由能 F | 螺旋波 = 局部极小，维持稳定的耗散结构 | SGD = 沿自由能景观梯度下降 |

### 4.3 物理学习的热力学解释

关键公式 (Still et al., 2012)：

> 热力学熵产生 ≥ 预测信息
> 即：物理系统为了维持对环境的预测能力，必须以至少同等速率向外界排放熵。

在 FHN 场景中：
- 螺旋波 = 非平衡稳态 (持续耗散能量)
- 螺旋波核心的对环境响应 = 一种"物理计算"
- 耗散的能量 = 信息处理的"热力学代价"

这给出了一个定量的 trade-off：η = I(预测; 真实) / ΔS_环境，即"每单位熵产生的信息增益"——物理计算的热力学效率。

---

## Part 5: 范畴论视角 — 最深层的统一语言

### 5.1 传播 = 态射合成

在范畴论中：
- 对象 = 场的构型空间 / 激活空间
- 态射 = 算子 (PDE 解算子 / 权重矩阵)
- 态射合成 = 时间演化 / 多层前向传播

信息的一个范畴论定义：
> "信息 = 态射所不保持的结构量"
> 即：T: A → B 携带的信息 = A 中那些在 B 中仍然可区分的结构

### 5.2 弦图表示

FHN 层:

```
      R ───[FHN介质]───→ S
      │                  │
      源点              观测
```

神经网络:

```
      x ───[W_1, σ]───→ h_1 ───[W_2, σ]───→ h_2 ───...───→ y
```

在弦图语言中，两者都是同样的语法结构：对象 + 态射 + 合成。区别仅在态射的具体实现。

### 5.3 潜在的定理

**猜想**：存在一个函子

$$\displaystyle F: \mathbf{PhysicalSystem} \to \mathbf{NeuralNetwork}$$

将 PDE 解算子映射为权重矩阵，使得：
- 物理演化时间 t 对应网络深度 L
- 扩散系数 D 对应权重矩阵的谱范数
- 非线性阈值 ε 对应激活函数的陡峭度

如果这个函子可以精确构造，那么就可以"编译"任何 PDE 为等价的神经网络。

---

## Part 6: 可操作的数值实验方案

基于你现有的代码，以下实验可以直接实施：

### 实验 6.1: 敏感度矩阵 G 的数值估计

```python
def estimate_sensitivity_matrix(solver, R0, O_points, delta=0.01):
    n_sources = len(R0)
    n_obs = len(O_points)
    G = np.zeros((n_obs, n_sources))
    
    solver.set_source_positions(R0)
    S_base = solver.observe(O_points)
    
    for j in range(n_sources):
        R_perturbed = R0.copy()
        R_perturbed[j] += delta
        solver.set_source_positions(R_perturbed)
        S_perturbed = solver.observe(O_points)
        G[:, j] = (S_perturbed - S_base) / delta
    
    return G
```

### 实验 6.2: Fisher 信息矩阵和信道容量

```python
sigma_noise = 0.01
G = estimate_sensitivity_matrix(...)
F = G.T @ G / sigma_noise**2

eigenvalues = np.linalg.eigvalsh(F)
C = 0.5 * np.sum(np.log2(1 + eigenvalues))
print(f"信道容量: {C:.2f} bits")
```

### 实验 6.3: IB 曲线在 FHN 介质上的数值计算

- 对不同 D (扩散系数) 运行模拟
- 计算每个 D 下的 I(X;T) 和 I(T;Y)
- 绘制 FHN 介质在 IB 平面上的位置
- 与标准神经网络的 IB 曲线对比

### 实验 6.4: 传递熵信息流场

- 在 FHN 介质中放置 10×10 探针网格
- 记录每个探针的 u(t) 时间序列
- 计算 pair-wise transfer entropy
- 绘制信息流矢量场
- 验证：旋度中心 = 螺旋波核心位置

---

## Part 7: 核心文献及对应内容总结

### 文献 1: Cover & Thomas — Elements of Information Theory

地位：信息论圣经。

| 章节 | 内容及与你的工作的关系 |
|------|------|
| 第 2 章 | 熵、互信息、KL 散度的精确定义。I(X;Y) = H(X) - H(X\|Y)。→ 你用它量化"物理场携带了多少信息" |
| 第 7 章 | 信道容量：C = max_{P(X)} I(X;Y)。→ 你计算 PhysicalNeuralLayer 的最大信息吞吐。§7.3 带宽有限信道，§7.5 连续时间信道 |
| 第 8 章 | 微分熵。高斯信道的容量公式。C = 0.5 log(1 + SNR) 是场信道容量的基础 |
| 第 10 章 | 率失真理论。信息瓶颈的数学原型。R(D) = min I(X;X̂) 受限于 E[d(X,X̂)] ≤ D。→ 给出物理介质的"不可压缩信息"下界 |
| 第 9 章 | 高斯信道 (你的 FHN 在高斯噪声下的理论模型)。MIMO 信道：y = Hx + n。→ 你的：S = G(R) + noise (G 是 FHN Green 函数) |

### 文献 2: MacKay — Information Theory, Inference, and Learning Algorithms

地位：唯一一本同时覆盖信息论、贝叶斯推断和神经网络的书。

| 章节 | 内容及与你的工作的关系 |
|------|------|
| 第 3-6 章 | 信息论基础 + 信源编码定理。更直观的互信息解释（比 Cover 更易懂） |
| 第 22-24 章 | 变分推断 + 自由能 = 证据下界。神经网络训练 = 变分自由能最小化。→ 这是你和 Friston 自由能原理的数学连接点 |
| 第 25 章 | 高斯过程。场的概率描述 = 无限维多元正态。→ FHN 介质在弱非线性极限下可以用 GP 近似 |
| 第 27 章 | 从信息论看神经网络。I(input; hidden) 和 I(hidden; output) 的探讨。→ 这就是 IB 的前身 |
| 第 34-35 章 | 能量与信息。麦克斯韦妖的信息论分析。kT ln 2 = 抹掉 1 bit 信息所需的最小能量。→ 你可以计算物理介质传播 1 bit 的热力学代价 |

### 文献 3: Enßlin — Information Theory for Fields (2019)

地位：直接回答你的核心问题——如何把香农信息论推广到连续场。

核心内容：
1. 场的信息度量：I[φ] = ∫ Dφ P(φ) log [P(φ) / P_0(φ)]——泛函形式的 KL 散度
2. 场的 Fisher 信息：泛函 Fisher 度量 = (δ²/δφ(x)δφ(y)) log P(data|φ)
3. 场的信道容量：C_field = sup_{P(φ_in)} I[φ_in; φ_out]（φ_out 是经过物理传播后的场）
4. 关键应用例子（Enßlin 主要用于天体物理，但框架通用）：宇宙微波背景辐射场的信息重建、银河系磁场的信息场推断。这些和你的 FHN 场在数学上是同构的
5. 你可以直接类比：他的"星系磁场重建" ≈ 你的"FHN 源点位置推断"，他的"观测噪声模型" ≈ 你的"观测点采样噪声"

### 文献 4: Amari — Information Geometry and Its Applications

地位：信息几何的权威教材。给物理系统和神经网络提供共同的几何语言。

| 章节 | 内容及与你的工作的关系 |
|------|------|
| 第 2-3 章 | 统计流形 + Fisher 信息作为黎曼度量。ds² = Σ g_{ij} dθ_i dθ_j, g_{ij} = F_{ij}。→ 你的物理参数的每个构型就是流形上的一个点 |
| 第 4-5 章 | 对偶联络 (e-connection 和 m-connection)。→ 指数族 = 神经网络的 softmax 输出；混合族 = 物理系统的玻尔兹曼分布 |
| 第 6-7 章 | 自然梯度下降。θ_{t+1} = θ_t - η F^{-1} ∇L。→ 沿 Fisher 度量的最陡方向下降。→ 你的源点参数更新应该用自然梯度而非欧氏梯度 |
| 第 11 章 | EM 算法的信息几何解释。→ 交替投影 = 物理演化 + 测量 |

### 文献 5: Tishby & Zaslavsky — Deep Learning and the Information Bottleneck Principle (2015)

核心内容：
1. 神经网络训练过程在 IB 平面上画出两阶段曲线：第一阶段 (拟合) I(T;Y) 快速上升；第二阶段 (压缩) I(X;T) 缓慢下降
2. 信息 = 神经网络隐藏层对输入的"记忆"量
3. 泛化性能 = 在 IB 平面上沿特定方向移动的结果
4. 对你工作的启示：你的 FHN 物理层是否也画出类似的两阶段曲线？扩散 D 和 epsilon 的作用 ≈ 信息瓶颈中的 β？不同的耦合函数对应 IB 平面上的不同路径？

### 文献 6: Friston — The Free-Energy Principle (2010, 2019)

核心内容：
1. 任何自组织系统 (从细胞到大脑到生态系统) 都在最小化变分自由能
2. 自由能 = 惊喜上界 = -log P(观测|模型) 的上界
3. 感知 = 变分推断 (推断隐藏状态)；行动 = 主动推断 (选择使预测误差最小的行动)；学习 = 模型参数的贝叶斯更新

对 FHN 的对应：
- FHN 介质的螺旋波 = 最小化自身自由能的耗散结构
- 源点的调整 = 主动推断（action in Friston's sense）
- 神经网络训练 = 模型参数的学习（learning in Friston's sense）

### 文献 7: Still et al. — Thermodynamics of Prediction (2012, PRL)

核心结论（精确定理）：对于任何具有记忆的预测系统，总熵产生率 ≥ 预测信息率。即：系统越能预测环境，就必须越"浪费"能量。

数学形式：dS/dt ≥ I_pred，其中 I_pred = I(X_future; M_past) = 系统记忆 M 中包含的关于未来输入 X 的信息。

对你的 FHN 系统的启示：
- 螺旋波的旋转 = 系统在"预测"自身的周期性动力学
- 维持螺旋所需的能量耗散 ≥ 螺旋结构编码的信息量
- 你可以计算：dS/dt (从 v 场的耗散估算) vs I_pred (从 u 的自相关估算)
- 验证你的 FHN 系统是否在这个热力学界限上运行

### 文献 8: Baez & Stay — Physics, Topology, Logic and Computation: A Rosetta Stone (2011)

核心内容：证明物理学、拓扑学、逻辑学和计算理论之间存在精确的结构对应：

| Physics | ↔ Topology | ↔ Logic | ↔ Computation |
|------|------|------|------|
| 物理系统 | 拓扑空间 | 逻辑命题 | 计算过程 |
| cobordism | nCob | category | circuit |

具体对应表：
- 物理过程 = 流形之间的配边 (cobordism)
- 计算过程 = 范畴中的态射合成
- 量子场论 = 张量网络 (Tensor Networks)

对你的启示：
- FHN 的时间演化 = 流形演化 = 信息态射
- 神经网络的前向 = 态射合成 = 信息处理链
- 两者是同一种范畴结构的两个实例

---

## Part 8: 潜在的研究论文框架

如果把这套思路写成论文，可能的结构：

**标题**：Information Propagation in Excitable Media as Neural Computation: A Unified Framework via Information Field Theory and the Information Bottleneck

**8.1 引言**：香农信息论的物理无关性；物理系统和神经网络在传播信息时的结构相似性；本文目标：找到统一的数学描述

**8.2 背景**：FHN 方程和可激发介质；信息瓶颈原理；信息场论基础

**8.3 理论框架：物理-信息对偶**：
- 定义 1：物理场的互信息泛函 I[φ_A; φ_B]
- 定义 2：场算子 T 的谱信息容量
- 定理 1：FHN 介质的信息带宽 = f(D, ε) 的显式表达
- 定理 2：信息瓶颈的变分原理在连续场极限下的形式

**8.4 数值实验**（基于你现有的代码）：
- 实验 A：FHN 介质的信道容量数值标定
- 实验 B：FHN 在 IB 平面上的轨迹 vs 标准神经网络
- 实验 C：传递熵信息流场与螺旋波核心的对应

**8.5 讨论**：物理介质作为计算基底的极限；与量子计算的信息论比较；生物学启示：心肌螺旋波的信息编码

**8.6 结论**：物理场传播和神经网络传播共享同一套信息几何语言；信息瓶颈是两者的统一变分描述；未来：自由能原理给出热力学解释

---

## Part 9: 学习路线图

**阶段 1 (1-2 个月): 信息论基础**
1. Cover & Thomas 第 2, 7, 8, 9, 10 章
2. MacKay 第 1-6 章, 第 27 章
目标：能用 I(X;Y) 和信道容量精确地思考和表达

**阶段 2 (1-2 个月): 信息 + 物理**
3. Enßlin 信息场论论文 (2019)
4. Kardar 统计场论 (选读第 1-3 章)
5. 自己推导：FHN 线性化 → 谱 → 有效信息维度
目标：把 FHN 介质当作一个信息信道来分析

**阶段 3 (1 个月): 信息瓶颈 + 神经网络**
6. Tishby 的 IB 论文 (2015)
7. Schwartz-Ziv & Tishby (2017) — 实证 IB
8. Saxe et al. (2019) — IB 在深度学习中的批判性分析
目标：在 IB 平面上理解神经网络和物理介质

**阶段 4 (1-2 个月): 自由能与信息几何**
9. Amari 第 2-3, 6-7, 11 章 (信息几何基础 + 自然梯度)
10. Friston (2010) 自由能原理综述
11. Still et al. (2012) 预测热力学
目标：用自由能统一描述物理学习和网络训练

**阶段 5 (选读): 最深层次**
12. Baez & Stay (2011) Rosetta Stone
13. Spivak 的范畴论教材
目标：范畴论视角下的物理-信息统一

---

## 补充：香农 1948 原典的阅读要点

香农的 The Mathematical Theory of Communication 分两部分：

**Part I: 离散无噪声信道** — §1-6：熵、信源编码定理。关键概念：典型序列 (typical sequences) → 信息的"可压缩性"

**Part II: 离散有噪声信道** — §7-13：信道容量、编码定理。关键概念：香农极限 → 任何信道都存在可靠通信的上界

**Part III: 连续信道** — §14-21：微分熵、高斯信道容量。关键概念：C = W log(1 + P/N) → 带宽和信噪比的权衡

**Part IV: 连续信源** → 率失真理论

**阅读策略**：把香农原文中所有的"电报信号"替换为"FHN 场构型"，把"编码"替换为"源点初始条件设定"，把"噪声"替换为"扩散耗散"，把"解码"替换为"观测点采样"。你会发现：1948 年香农建立的框架，恰好就是你需要的"物理-计算"统一框架的离散原型。

---

## Part 10: 可立刻着手的最小可行研究

基于你现有的 fhn_solver.py + neural_layer.py，可以直接做的：

**Step 1: 数值估计 I(R; S) — 物理互信息**
对不同噪声水平 σ，用蒙特卡洛采样估计 P(S|R)，数值计算：I(R; S) ≈ log|det(G)| + constant

**Step 2: 计算信道容量作为物理参数的函数**
C(D, ε, σ) = ?（扩散、兴奋性、噪声 → 比特/通道使用），画出 C vs D, C vs ε 的曲线

**Step 3: 计算有效信息维度**
d_eff = Tr(F)² / Tr(F²)，其中 F 是 Fisher 信息矩阵。画出 d_eff vs 观测点数目的关系

**Step 4: 对比实验**
用一个同规模的 MLP 层替代 PhysicalNeuralLayer，比较两者的 I(X;T) 和 I(T;Y)，看物理层在 IB 曲线上的位置 vs 标准神经网络

这四个实验已经足够成为一篇 workshop paper 的核心实验。

---

## 附录：关键数学符号表

| 符号 | 含义 | 所在框架 |
|------|------|------|
| I(X;Y) | 互信息 | 信息论 (Shannon) |
| C | 信道容量 | 信息论 (Shannon) |
| F_ij | Fisher 信息矩阵 | 信息几何 (Amari) |
| g_ij | 黎曼度量 | 信息几何 (Amari) |
| T_{X→Y} | 传递熵 | 因果信息论 (Schreiber) |
| D_KL | KL 散度 | 信息论 |
| β | IB 压缩参数 | 信息瓶颈 (Tishby) |
| F | 变分自由能 | 自由能原理 (Friston) |
| G_ij | 敏感度矩阵 ∂S_i/∂R_j | 数值分析 |
| λ_k | 算子谱 (物理或矩阵) | 谱理论 |
| d_eff | 有效信息维度 | 信息瓶颈 |
| η | dS/dt / I_pred | 预测热力学 (Still) |
