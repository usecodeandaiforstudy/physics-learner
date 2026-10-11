---

## 本书定位：为什么信息场论是你的核心桥梁

香农在 1948 年建立的信息论处理的是*离散随机变量*：
$$
I(X;Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}
$$

但你的 FHN 可激发介质是一个*连续物理场* $u(x,y,t)$——它是一个*无限维*的随机过程。香农的离散公式在这里直接失效了。

Torsten Enßlin（马普天体物理所）发展了一套名为**信息场论 (Information Field Theory, IFT)** 的数学框架，其核心任务就是：

> **把香农信息论的所有概念——熵、互信息、信道容量、Fisher 信息——从离散随机变量推广到连续物理场。**

**和你的关系**：Enßlin 研究的是宇宙微波背景辐射场和银河系磁场的信息重建。他在做的就是你正在做的——只不过他的物理场是天体物理的，你的是 FHN 可激发介质。数学框架是通用的。

**注记（最重要的类比）：**

Enßlin 的``星系磁场重建'' $\approx$ 你的``FHN 源点位置推断''。

Enßlin 的``望远镜观测噪声模型'' $\approx$ 你的``观测点采样噪声''。

两者在数学上是同构的。你可以直接套用他的框架。

## 从香农信息论到信息场论：核心推广

### 离散 $\to$ 连续：离散变量的信息论回顾

对于离散随机变量 $X$，取值 $x_i$，概率 $p_i=P(x_i)$：

- **熵**：$H(X)=-\sum_i p_i\log p_i$ —— 不确定性的度量（单位：bit）
- **KL 散度**：$D_{\text{KL}}(P\|Q)=\sum_i p_i\log\frac{p_i}{q_i}$ —— 两个分布的距离
- **互信息**：$I(X;Y)=\sum_{x,y}P(x,y)\log\frac{P(x,y)}{P(x)P(y)}$ —— 共享的信息量
- **信道容量**：$C=\max_{P(X)}I(X;Y)$ —— 信道的最大信息吞吐

这些公式的核心结构是：$\sum_i p_i \log(\cdots)$ —— 对*有限的、离散的*可能性求和。

### 从 $\sum$ 到 $\int\mathcal{D}\phi$：泛函积分

信息场论的核心推广是用*泛函积分*（路径积分）代替普通求和：

**定义（泛函积分）：**

设 $\phi(x)$ 是一个定义在空间区域 $\Omega$ 上的连续场，$\mathcal{F}[\phi]$ 是场的某个泛函。场空间上的积分（泛函积分）记为：
$$
\int \mathcal{D}\phi \; \mathcal{F}[\phi] \equiv \lim_{N\to\infty} \int \prod_{i=1}^{N} d\phi(x_i) \; \mathcal{F}(\phi(x_1),\ldots,\phi(x_N))
$$
其中 $N$ 个格点覆盖整个空间 $\Omega$。这等价于对所有可能的场构型进行求和。

**直观理解**：你把你 FHN 介质的空间离散化为 $100\times 100$ 的网格，每个格点上的 $u$ 值是一个连续变量。场的概率分布 $P[\phi]$ 就是这 $10000$ 个变量的联合概率分布。泛函积分就是对这 $10000$ 维空间做积分。

### 场的熵泛函

**定义（场的熵）：**

设场 $\phi(x)$ 的概率密度泛函为 $P[\phi]$，场的熵定义为：
$$
H[\phi] = -\int \mathcal{D}\phi \; P[\phi] \log P[\phi]
$$

注意：和离散熵不同，场的熵*没有绝对零点*——它依赖于场的参数化（坐标选择）。就像 Cover & Thomas 中微分熵 $h(X)$ 可以为负一样。但互信息在推广中保持良好定义。

### 场的 KL 散度

**定义（场的 KL 散度）：**

两个场分布 $P[\phi]$ 和 $Q[\phi]$ 之间的 KL 散度：
$$
D_{\text{KL}}(P \| Q) = \int \mathcal{D}\phi \; P[\phi] \log \frac{P[\phi]}{Q[\phi]}
$$

这是你整个理论框架的*核心运算*：
- Friston 自由能中的复杂度项 $D_{\text{KL}}[Q(s)\|P(s)]$
- 信息瓶颈中的压缩项 $I(X;T)=D_{\text{KL}}[p(x,t)\|p(x)p(t)]$
- 源点推断中的贝叶斯后验逼近
都直接使用这个公式。

## 场的互信息——回答你问题的核心量

### 定义

**定义（场的互信息泛函）：**

设 $\phi_A$ 和 $\phi_B$ 是两个物理场（例如 FHN 介质中不同区域的场构型），其联合分布为 $P[\phi_A,\phi_B]$。它们之间的互信息为：
$$
I[\phi_A; \phi_B] = \int \mathcal{D}\phi_A \mathcal{D}\phi_B \; P[\phi_A, \phi_B] \log \frac{P[\phi_A, \phi_B]}{P[\phi_A] \, P[\phi_B]}
$$

**定理（互信息的坐标不变性）：**

$I[\phi_A;\phi_B]$ 在任意连续可微的坐标变换下保持不变。这使它成为*唯一可靠的信息度量*——不依赖于你如何参数化你的物理场。

### 三种等价形式（离散 $\to$ 场的版本）

$$\begin{aligned}
I[\phi_A;\phi_B] &= H[\phi_A] - H[\phi_A \mid \phi_B]  \\
    I[\phi_A;\phi_B] &= H[\phi_B] - H[\phi_B \mid \phi_A]  \\
    I[\phi_A;\phi_B] &= D_{\text{KL}}(P[\phi_A,\phi_B] \;\|\; P[\phi_A]P[\phi_B])
\end{aligned}$$

形式 (\ref{eq:mi_form1})：$\phi_B$ 减少了多少关于 $\phi_A$ 的不确定性。

形式 (\ref{eq:mi_form3})：联合分布和独立分布之间的距离。

### 在你的 FHN 系统中的应用

设：
- $\phi_R(x,y)$ = 由源点构型 $R$ 生成的稳态 FHN 场构型
- $\phi_O$ = 在观测点集合 $O$ 上的场采样（经过噪声污染）

物理层作为一个信息信道，其传输的信息量为：
$$
I_{\text{phys}} = I[\phi_R; \phi_O]
$$

这个量直接回答了你的核心问题：**``物理介质传播了多少信息？''**

**注记（数据处理不等式在连续场中的形式）：**

在 FHN 介质中：
\[
R \longrightarrow u(x,y) \longrightarrow S
\]
构成一个马尔可夫链（观测信号只通过物理场依赖源点）。因此：
\[
I(R; S) \leq I(R; u) \leq H(R)
\]
即：观测点能接收的信息 $\leq$ 物理场携带的信息 $\leq$ 源点本身的信息。

## 场的 Fisher 信息——信息几何的场论版本

### 离散版本的回顾

对于参数化模型 $p(x \mid \theta)$，Fisher 信息矩阵为：
$$
F_{ij}(\theta) = \mathbb{E}_{x\sim p(\cdot\mid\theta)}\left[ \frac{\partial \log p(x\mid\theta)}{\partial \theta_i} \cdot \frac{\partial \log p(x\mid\theta)}{\partial \theta_j} \right]
$$

它衡量``给定参数 $\theta$，对 $\theta$ 的每个分量做微小扰动时，观测数据 $x$ 能多好地分辨出来''。

### 场的泛函 Fisher 度量

**定义（场的 Fisher 信息）：**

设观测数据 $d$ 依赖于物理场 $\phi(x)$ 的概率模型为 $P(d \mid \phi)$。场的 Fisher 信息是一个*泛函算子*：
$$
\mathcal{F}(x, y; \phi) = \mathbb{E}_{d\sim P(\cdot\mid\phi)}\left[ \frac{\delta \log P(d\mid\phi)}{\delta\phi(x)} \cdot \frac{\delta \log P(d\mid\phi)}{\delta\phi(y)} \right]
$$
其中 $\frac{\delta}{\delta\phi(x)}$ 是*泛函导数*（将场的某个点 $x$ 处的值做无穷小扰动时，泛函的变化率）。

**为什么要用泛函导数？** 场是无限维的，$\phi(x)$ 在每一点 $x$ 都是一个``参数''。普通的偏导数 $\partial/\partial\theta_i$ 不够用了，需要泛函导数 $\delta/\delta\phi(x)$。

### 物理意义

场的 Fisher 信息 $\mathcal{F}(x,y;\phi)$ 告诉你：
- 在点 $x$ 处扰动场 $\phi$，在点 $y$ 处的观测会有多敏感的反应
- 如果 $\mathcal{F}(x,y)$ 在某两个区域之间很大 → 这两个区域的场值强耦合 → 信息容易在这两个区域之间流动
- 如果 $\mathcal{F}(x,y)$ 很小 → 场在该区域的变化几乎不影响观测 → 该区域``信息盲区''

### FHN 介质中的具体形式

在你的 FHN 系统中，观测模型为：
$$
S_j = u(O_j; R) + \varepsilon_j, \quad \varepsilon_j \sim \mathcal{N}(0, \sigma^2)
$$
其中 $O_j$ 是第 $j$ 个观测点的位置，$u(O_j;R)$ 是 FHN 方程在源点为 $R$ 时在 $O_j$ 处产生的场值。

高斯噪声下，场 Fisher 信息的简化形式：
$$
\mathcal{F}(O_i, O_j; R) = \frac{1}{\sigma^2} \cdot \frac{\partial u(O_i; R)}{\partial R} \otimes \frac{\partial u(O_j; R)}{\partial R}
$$

这和你之前总结的 Cover & Thomas 中高斯信道的 Fisher 信息矩阵完全一致——只是参数从 $\theta$ 变成了 $R$（源点构型）。

## 场的信道容量——物理介质的计算极限

### 定义

**定义（场的信道容量）：**

对于物理场信道 $\phi_{\text{in}} \to \phi_{\text{out}}$（输入场经过物理传播后变成输出场），场的信道容量定义为：
$$
C_{\text{field}} = \sup_{P[\phi_{\text{in}}]} \; I[\phi_{\text{in}}; \phi_{\text{out}}]
$$
其中 $\sup$ 在所有可能的输入场分布上取最大值。

### 物理直觉

这回答了：**``这个物理介质最多能编码多少比特的有意义信息？''**

- 无论你如何设计输入（源点构型），介质能传输的信息量有一个硬上限
- 这个上限只依赖于介质的物理参数（$D$, $\varepsilon$, 几何尺寸）和噪声水平 $\sigma^2$
- 改变物理参数就是改变 $C_{\text{field}}$——这是你``设计物理计算''的自由度

### 高斯场信道的闭式解

如果物理传播是线性的（FHN 在弱激发近似下）：
$$
\phi_{\text{out}}(x) = \int G(x,y) \, \phi_{\text{in}}(y) \, dy + \eta(x)
$$
其中 $G(x,y)$ 是 Green 函数（传播算子），$\eta(x)$ 是高斯噪声。

那么信道容量有闭式解：
$$
C_{\text{field}} = \frac{1}{2} \sum_{k} \log_2\left(1 + \frac{\lambda_k}{\sigma^2}\right)
$$
其中 $\lambda_k$ 是 Green 函数算子 $\hat{G}$ 的特征值（$\hat{G}$ 是一个积分算子，核为 $G(x,y)$）。

**注记（和 Cover & Thomas MIMO 容量的对应）：**

公式 (\ref{eq:gaussian_field_capacity}) 和 Cover & Thomas 中 MIMO 高斯信道的容量公式
\[
C = \frac{1}{2}\sum_i \log_2(1 + \lambda_i)
\]
是同构的——只是求和从有限个特征值变成了*无限个*特征值（场算子的谱）。

### 离散化近似——你可以用代码计算的形式

在数值实现中，你把 FHN 介质离散化为 $M\times M$ 网格，Green 函数 $G$ 变成一个 $M^2\times M^2$ 的矩阵。此时：
$$
C_{\text{field}} \approx \frac{1}{2}\sum_{k=1}^{K} \log_2\left(1 + \frac{\lambda_k}{\sigma^2}\right)
$$
其中 $\lambda_k$ 是矩阵 $GG^T$ 的前 $K$ 个最大特征值。

**这个公式可以直接用你的 FHN 代码计算！**——你只需要估计 $G_{ij}=\partial S_i/\partial R_j$（敏感度矩阵），然后做特征值分解。

## 信息场论的非高斯推广——Enßlin 的核心技术

### 为什么需要非高斯信息场论？

如果你的 FHN 系统只在弱激发态运行（源点扰动很小），线性近似和高斯场假设是合理的。但 FHN 的螺旋波是*高度非线性*的——它涉及：
- 激发阈值：$u$ 必须超过某个临界值才能触发再生（非连续依赖）
- 非线性振子：恢复变量 $v$ 和 $u$ 的耦合产生极限环
- 波前碰撞湮灭：两个螺旋波碰撞时发生不可逆的场重构

这些非线性效应意味着 $P[\phi]$ 不是高斯的。Enßlin 的信息场论为此提供了一套*非高斯场的信息推断*方法。

### 信息哈密顿量 (Information Hamiltonian)

Enßlin 的一个核心工具是定义场的*信息哈密顿量*：

**定义（信息哈密顿量）：**

对于场 $\phi$ 的概率分布 $P[\phi]$（或后验分布 $P[\phi \mid d]$），信息哈密顿量定义为：
$$
\mathcal{H}[\phi] = -\log P[\phi] \quad \text{（先验）}, \qquad
    \mathcal{H}[\phi \mid d] = -\log P[\phi \mid d] \quad \text{（后验）}
$$

**为什么叫``哈密顿量''？** 因为对它做变分 $\frac{\delta\mathcal{H}}{\delta\phi}$ 得到的场方程，在形式上完全类比于统计力学中的配分函数和哈密顿量。场的概率分布 = 指数的负能量：
$$
P[\phi] \propto e^{-\mathcal{H}[\phi]}
$$

这恰好等价于一个*欧几里得量子场论*（或经典统计场论）——你的 FHN 介质有一个自然的非平衡统计场论解释。

### 对 FHN 系统的直接应用

FHN 场的信息哈密顿量可以分解为：
$$
\mathcal{H}[u, v \mid S] = \underbrace{\frac{1}{2\sigma^2} \sum_j \big(S_j - u(O_j)\big)^2}_{\text{观测项}} + \underbrace{\mathcal{H}_{\text{FHN}}[u, v]}_{\text{物理动力学约束}}
$$

其中：
- **观测项**：观测信号 $S_j$ 与场值 $u(O_j)$ 的平方误差（高斯噪声假设下）
- **物理动力学约束** $\mathcal{H}_{\text{FHN}}$：惩罚那些``不满足 FHN 方程''的场构型 $$ \mathcal{H}_{\text{FHN}}[u, v] \propto \int \left|\partial_t u - (D\nabla^2 u + u - u^3/3 - v + I_{\text{ext}})\right|^2 + \left|\partial_t v - \varepsilon(u + \beta - \gamma v)\right|^2 \; dx\,dy\,dt $$

最小化 $\mathcal{H}[u,v\mid S]$ 等价于同时满足两个要求：
1. 场构型在观测点处匹配观测数据（观测项）
2. 场构型满足 FHN 动力学方程（物理约束）

**这就是你的``物理推断''问题的精确变分形式。**

### 连接 Friston 自由能原理

信息哈密顿量 $\mathcal{H}[\phi]$ 直接通向 Friston 的变分自由能：
$$
F(Q) = \mathbb{E}_{Q}[\mathcal{H}[\phi \mid d]] - H[Q]
$$
其中 $Q[\phi]$ 是后验的变分近似，$H[Q]$ 是 $Q$ 的熵。

第一项 $\mathbb{E}_Q[\mathcal{H}]$ 是``模型和数据匹配有多好''；
第二项 $-H[Q]$ 是``变分分布的熵''——倾向于更简单（更高熵）的近似。

联合最小化这两项 = 在``精确性''和``简单性''之间找最优平衡。

## 传递熵的场论推广——信息流场

### 传递熵的离散回顾

Schreiber (2000) 定义的传递熵：
$$
T_{X\to Y} = I(Y_{t+1}; X_t \mid Y_t)
$$
衡量``$X$ 的历史在已知 $Y$ 的历史之外，还能提供多少关于 $Y$ 未来的信息''。

### 连续场中的传递熵

Enßlin 的框架将传递熵推广到连续场：

**定义（场传递熵）：**

对于两个空间区域 $A$ 和 $B$ 中的场 $\phi_A(t)$ 和 $\phi_B(t)$：
$$
T_{\phi_A \to \phi_B} = I[\phi_B(t+\Delta t); \phi_A(t) \mid \phi_B(t)]
$$
其中 $I[\,\cdot\,;\,\cdot\,\mid\,\cdot\,]$ 是条件互信息的泛函形式。

**在你的 FHN 介质中**：在介质中放置 $10\times 10$ 的探针网格，记录每个探针处 $u(t)$ 的时间序列。对每对相邻探针计算 $T_{A\to B}$。得到的矢量场 $v_{\text{info}}(x,y)$ 就是*物理介质中的信息流场*。

**核心预测**：信息流场的旋度中心应当与螺旋波的核心位置重合。即：
$$
\nabla \times v_{\text{info}}(x,y) \neq 0 \quad \Longleftrightarrow \quad (x,y) \text{ 是螺旋波核心}
$$

如果这个预测在你的 FHN 模拟中成立，它就提供了``信息流 = 物理因果''的直接数值证据。

## Enßlin 框架在 FHN 介质上的操作指南

### 步骤一：从 Green 函数到场互信息

FHN 方程在弱激发近似下可以线性化为：
$$
\frac{\partial}{\partial t} \begin{pmatrix} \delta u \\ \delta v \end{pmatrix}
    = \begin{pmatrix} D\nabla^2 + f'(u^*) & -1 \\ \varepsilon & -\varepsilon\gamma \end{pmatrix}
    \begin{pmatrix} \delta u \\ \delta v \end{pmatrix}
$$
其中 $f'(u^*) = 1 - (u^*)^2$ 在静息态 $u^* \approx -1.037$ 处。

线性算子的 Green 函数 $G(x,y,t)$ 满足：
$$
\left(\frac{\partial}{\partial t} - \mathcal{L}\right) G(x,y,t) = \delta(x-y)\delta(t)
$$

在傅里叶空间中，$\mathcal{L}$ 的谱给出了场的*信息传输模式*。

### 步骤二：数值估计敏感度矩阵

基于你现有的 `FHNSolver` 代码：

```Python
def estimate_sensitivity_matrix(solver, R0, O_points, delta=0.01):
    """
    估计 FHN 介质的敏感度矩阵 G_ij = dS_i / dR_j
    
    参数:
        solver: FHNSolver 实例
        R0: 源点构型向量 [x1, y1, A1, x2, y2, A2, ...]
        O_points: 观测点列表 [(x1,y1), (x2,y2), ...]
        delta: 扰动大小
    """
    n_sources = len(R0)
    n_obs = len(O_points)
    G = np.zeros((n_obs, n_sources))
    
    # 基准信号
    solver.set_source_config(R0)
    S_base = solver.measure(O_points)
    
    # 对每个源点参数做有限差分
    for j in range(n_sources):
        R_pert = R0.copy()
        R_pert[j] += delta
        solver.set_source_config(R_pert)
        S_pert = solver.measure(O_points)
        G[:, j] = (S_pert - S_base) / delta
    
    return G

```

### 步骤三：计算场的信道容量

```Python
def field_channel_capacity(G, sigma_noise):
    """
    基于敏感度矩阵 G 和噪声水平 sigma 计算场的信道容量
    
    使用公式: C = 0.5 * sum_i log2(1 + lambda_i / sigma^2)
    """
    # 构建 Fisher 信息矩阵
    F = G.T @ G / (sigma_noise ** 2)
    eigenvalues = np.linalg.eigvalsh(F)
    
    # 数值稳定的信道容量
    snr = eigenvalues / (sigma_noise ** 2)
    C = 0.5 * np.sum(np.log2(1.0 + snr))
    
    return C, eigenvalues

```

### 步骤四：计算互信息（高斯近似下）

在线性化 FHN + 高斯噪声假设下：
$$
I(R; S) \approx \frac{1}{2} \log_2 \det\!\left(I + \frac{1}{\sigma^2} \, G\,\Sigma_R\,G^T\right)
$$
其中 $\Sigma_R$ 是源点构型的先验协方差。

这是一个闭式解——你不需要蒙特卡洛采样就能估算出信息量。

### 步骤五：计算传递熵信息流场

```Python
def compute_transfer_entropy_field(probe_signals, probe_positions, dt=1):
    """
    计算 FHN 介质中的传递熵信息流场
    
    参数:
        probe_signals: shape (n_probes, n_timesteps) 的 u(t) 数组
        probe_positions: shape (n_probes, 2) 的 (x,y) 坐标数组
    """
    n_probes = probe_signals.shape[0]
    te_matrix = np.zeros((n_probes, n_probes))
    
    for i in range(n_probes):
        for j in range(n_probes):
            if i != j:
                # TE_{j->i} = I(X_i(t+1); X_j(t) | X_i(t))
                te_matrix[i, j] = estimate_transfer_entropy(
                    source=probe_signals[j, :-dt],
                    target_past=probe_signals[i, :-dt],
                    target_future=probe_signals[i, dt:]
                )
    
    # 将 TE 矩阵转换为矢量场
    v_info = np.zeros((n_probes, 2))
    for i in range(n_probes):
        for j in range(n_probes):
            if i != j:
                direction = probe_positions[j] - probe_positions[i]
                direction /= np.linalg.norm(direction)
                v_info[i] += te_matrix[i, j] * direction
    
    return v_info, te_matrix

```

## Enßlin 框架与你之前讨论的 5 个框架的连接

信息场论是你整个研究计划中*承上启下*的枢纽：

| **信息场论的概念** | **来自哪本书** | **通向哪个框架** |
| --- | --- | --- |
| 场的互信息 $I[\phi_A;\phi_B]$ | Cover \ | Thomas 第 2 章 |
| 场的 Fisher 信息 $\mathcal{F}(x,y)$ | Cover \ | Thomas 第 2 章 |
| 场的信道容量 $C_{\text{field}}$ | Cover \ | Thomas 第 7 章 |
| 信息哈密顿量 $\mathcal{H}[\phi]$ | MacKay 第 24 章 | 自由能原理 (Friston) |
| 场传递熵 $T_{\phi_A\to\phi_B}$ | Cover \ | Thomas 第 2 章 |
| 泛函积分 $\int\mathcal{D}\phi$ | Zinn-Justin | 范畴论观点 |

**注记（一句话定位）：**

信息场论 (Enßlin) 是你的 5 个理论框架中的*数学基础层*——它把香农信息论的所有工具升级为可以处理物理场的语言。在这个基础上，你可以构建信息瓶颈（Tishby）、信息几何（Amari）和自由能原理（Friston）这些高层框架。

## 核心论文与扩展阅读

### Enßlin 的主要论文

1. **Enßlin, T. (2019).** \textit{Information theory for fields.} Annalen der Physik, 531(3), 1800127. 这是你需要的*核心论文*——信息场论的完整数学框架。
2. **Enßlin, T., Frommert, M., & Kitaura, F. S. (2009).** \textit{Information field theory for cosmological perturbation reconstruction and nonlinear signal analysis.} Physical Review D, 80(10), 105005. 信息场论在天体物理中的第一个完整应用——宇宙微波背景辐射的信息重建。和你的 FHN 场重建问题数学同构。
3. **Enßlin, T. & Weyg, C. (2021).** \textit{Information field theory and artificial intelligence.} Entropy, 23(9), 1141. **这篇对你特别重要**——Enßlin 明确讨论了信息场论和人工智能（神经网络）的关系！
4. **Selig, M., Bell, M. R., & Enßlin, T. (2015).** \textit{Denoising, deconvolving, and decomposing photon observations——derivation of the D3PO algorithm.} Astronomy & Astrophysics, 574, A74. 信息场论在非高斯场推断中的应用——如果你的 FHN 螺旋波是高度非高斯的，这篇的算法直接适用。

### 相关数学资源

- **泛函积分**：Zinn-Justin, \textit{Path Integrals in Quantum Mechanics}（第 1-3 章）
- **统计场论**：Kardar, \textit{Statistical Physics of Fields}（第 1-3 章：场的概率论语言）
- **高斯过程**：Rasmussen & Williams, \textit{Gaussian Processes for Machine Learning}

## 总结：Enßlin 框架对你研究的三层价值

1. **数学层面**：信息场论把香农的累积求和 $\sum$ 升级为泛函积分 $\int\mathcal{D}\phi$，让你可以用信息论语言精确描述 连续物理场 $u(x,y,t)$。
2. **计算层面**：场的信道容量 $C_{\text{field}} = \frac{1}{2}\sum_k \log_2(1+\lambda_k/\sigma^2)$ 可以直接用你的 FHN 代码数值计算。你只需要估计敏感度矩阵 $G_{ij} = \partial S_i/\partial R_j$。
3. **理论层面**：信息哈密顿量 $\mathcal{H}[\phi]$ 将 Friston 自由能原理和你的 FHN 介质做了精确的数学链接。FHN 螺旋波 = 最小化某个信息哈密顿量的场构型。神经网络的 SGD = 在参数空间最小化类似的哈密顿量。

> **Enßlin 回答了你的核心问题：**
> 如何把信息论从离散变量推广到连续物理场？
> 答案：用泛函积分代替普通求和，用场的概率密度泛函 $P[\phi]$ 代替离散概率 $p(x)$。\\
> 互信息 $I[\phi_A;\phi_B]$、信道容量 $C_{\text{field}}$、Fisher 信息 $\mathcal{F}(x,y)$ \\
> 都在这个框架下有严格定义——这就是你分析 FHN 介质所需的全部数学语言。
