---

# 基本原理：从最小作用量到信息变分原理

## 引言：分析力学和信息处理的统一

分析力学的核心是一个极其简洁的陈述：**物理系统的真实演化路径是使作用量泛函取极值的路径**。即 $\delta S = 0$。

我们在此证明：**任何信息处理系统——物理场、神经网络、通信信道——都遵循同一个变分原理。** 信息瓶颈原理 $\min[I(X;T)-\beta I(T;Y)]$ 是这个普遍变分原理在信息空间中的具体表达。

本文的目标是从分析力学的第一原理出发，逐层建立信息力学的完整框架——定义信息的拉格朗日量、推导信息的 Euler-Lagrange 方程、构造信息的哈密顿量和正则方程、证明信息的 Noether 定理和守恒律，最后将其应用于物理场和神经网络的具体分析。

## 公理化基础

**公理（信息变分公理）：**

任何信息处理系统——定义为将输入随机变量 $X$ 映射为输出随机变量 $Y$ 的*态射* $T$——的动力学行为完全由其*信息作用量* $\mathcal{S}[T]$ 的变分条件决定：
$$
\delta \mathcal{S}[T] = 0
$$
其中 $\mathcal{S}[T]$ 是定义在条件分布空间 $\{P(T|X)\}$ 上的泛函。

**公理（信息局部性公理）：**

信息作用量 $\mathcal{S}[T]$ 可以写为一个*信息拉格朗日量* $\mathcal{L}_{\text{info}}$ 的积分：
$$
\mathcal{S}[T] = \int \mathcal{L}_{\text{info}}[T, \dot{T}] \, d\mu(x)
$$
其中 $d\mu(x)$ 是输入空间上的测度，$\dot{T}$ 表示 $T$ 在某种``信息时间''参数下的变化率。

**公理（信息-能量对偶公理）：**

信息作用量 $\mathcal{S}[T]$ 和物理作用量 $S[q]$ 通过 Wick 转动对偶：
$$
iS_{\text{物理}}[q] \;\longleftrightarrow\; -\mathcal{S}_{\text{信息}}[P]
$$
即：量子力学的振荡权重 $e^{iS/\hbar}$ 对应于信息力学的玻尔兹曼权重 $e^{-\mathcal{S}_{\text{信息}}}$。

这一公理的直接推论是：信息力学是经典统计力学在*概率分布空间（统计流形）*上的推广。量子力学描述的是振幅的相位干涉，信息力学描述的是概率分布的 KL 散度最小化。

# 拉格朗日力学：信息作用量的构造

## 信息拉格朗日量

### 信息动能和信息势能

在分析力学中，拉格朗日量是动能和势能之差：$\mathcal{L} = K - \mathcal{V}$。在信息力学中，我们需要定义对应的信息量。

**定义（信息动能）：**

系统在信息空间中 ``运动的活跃程度'' 由 Fisher 信息度量给出。设信息系统的状态由条件分布 $P(T|X)$ 参数化，参数为 $\theta$（如源点构型 $R$ 或网络权重 $W$），定义*信息动能*为：
$$
\mathcal{K}_{\text{info}}[\dot{\theta}] = \frac{1}{2} \sum_{i,j} g_{ij}(\theta) \, \dot{\theta}_i \dot{\theta}_j
$$
其中 $g_{ij}(\theta) = \mathbb{E}_{x\sim P(\cdot|\theta)}[\partial_i\log P(x|\theta) \cdot \partial_j\log P(x|\theta)]$ 是 Fisher 信息矩阵。$\dot{\theta} = d\theta/d\tau$，$\tau$ 是 ``信息时间''——系统在统计流形上演化的序参量（训练步数、物理时间等）。

**直观**：$\mathcal{K}_{\text{info}}$ 衡量了当前参数变化有多快。Fisher 度量 $g_{ij}$ 保证了快的方向和高信息区分度的方向一致——在信息容易区分的维度上，动能更大。

**定义（信息势能）：**

信息势能由两个竞争项组成——*压缩势* $\mathcal{V}_{\text{compress}}$ 和*任务势* $\mathcal{V}_{\text{task}}$：
$$
\mathcal{V}_{\text{info}}[\theta] = \mathcal{V}_{\text{compress}}[\theta] + \mathcal{V}_{\text{task}}[\theta]
$$

其中：
$$\begin{aligned}
\mathcal{V}_{\text{compress}}[\theta] &= I(X; T_\theta)  \\
    \mathcal{V}_{\text{task}}[\theta] &= -\beta \cdot I(T_\theta; Y)
\end{aligned}$$

压缩势衡量中间表示 $T_\theta$ 保留了多少输入信息（越大越 ``高能''——系统偏离了最优压缩）。任务势衡量 $T_\theta$ 保留了多任务相关信息（负号意味着信息越多越好——``势能越低''）。

**核心公式——信息拉格朗日量**：
$$
\boxed{\mathcal{L}_{\text{info}}[\theta, \dot{\theta}] = \frac{1}{2} \sum_{i,j} g_{ij}(\theta) \, \dot{\theta}_i \dot{\theta}_j - \Big[I(X; T_\theta) - \beta \, I(T_\theta; Y)\Big]}
$$

这就是信息瓶颈原理的拉格朗日表述。第一项是信息动能（由 Fisher 度量赋权），第二项是信息势能（信息瓶颈目标函数的负值）。

### 和经典力学拉格朗日量的对照

| **概念** | **经典力学** | **信息力学** |
| --- | --- | --- |
| 广义坐标 $q$ | 参数 $\theta$（源点构型、网络权重） | 统计流形上的坐标 |
| 广义速度 $\dot{q}$ | $\dot{\theta} = d\theta/d\tau$ | 参数更新速率 |
| 动能 $K$ | $\frac{1}{2}g_{ij}\dot{\theta}_i\dot{\theta}_j$ | Fisher 度量的测地动能 |
| 势能 $\mathcal{V}$ | $I(X;T) - \beta I(T;Y)$ | 信息瓶颈目标函数 |
| 质量矩阵 $m_{ij}$ | Fisher 信息 $g_{ij}$ | 统计流形的黎曼度量 |
| 拉格朗日量 $\mathcal{L}$ | $K - \mathcal{V}$ | 信息动能 $-$ 信息势能 |

## 信息 Euler-Lagrange 方程

将变分原理 $\delta\mathcal{S} = \delta\int\mathcal{L}_{\text{info}}\,d\tau = 0$ 应用于信息拉格朗日量，得到*信息 Euler-Lagrange 方程*：

**定理（信息 Euler-Lagrange 方程）：**

信息系统的真实演化路径 $\theta(\tau)$ 满足：
$$
\boxed{\frac{d}{d\tau}\frac{\partial \mathcal{L}_{\text{info}}}{\partial \dot{\theta}_i} - \frac{\partial \mathcal{L}_{\text{info}}}{\partial \theta_i} = 0}
$$

代入 $\mathcal{L}_{\text{info}} = \frac{1}{2}g_{jk}\dot{\theta}_j\dot{\theta}_k - \mathcal{V}_{\text{info}}$，展开：
$$
\sum_j g_{ij}(\theta)\,\ddot{\theta}_j + \sum_{j,k} \Gamma_{ijk}(\theta)\,\dot{\theta}_j\dot{\theta}_k = -\frac{\partial \mathcal{V}_{\text{info}}}{\partial \theta_i}
$$

其中 $\Gamma_{ijk}$ 是 Fisher 度量 $g_{ij}$ 的 Christoffel 符号（第一类）：
$$
\Gamma_{ijk} = \frac{1}{2}\left(\frac{\partial g_{ij}}{\partial\theta_k} + \frac{\partial g_{ik}}{\partial\theta_j} - \frac{\partial g_{jk}}{\partial\theta_i}\right)
$$

### 方程各项的物理含义

1. **惯性项** $\sum_j g_{ij}\ddot{\theta}_j$：参数在信息流形上的 ``加速''，抵抗方向变化。Fisher 度量的各向异性意味着：在不同方向上 ``质量'' 不同——高 Fisher 信息的方向更 ``重''，更难以加速。
2. **Christoffel 项** $\sum_{j,k}\Gamma_{ijk}\dot{\theta}_j\dot{\theta}_k$：信息流形的 ``弯曲效应''——如果统计流形是弯曲的（非欧氏），匀速的参数运动也会产生 ``惯性力''。
3. **势能力** $-\partial\mathcal{V}_{\text{info}}/\partial\theta_i$：信息瓶颈目标函数的负梯度——驱动系统向更低的信息势能（更好的压缩-保真权衡）方向演化。

### 特殊情况：欧氏极限（平坦统计流形）

当 $g_{ij} = m\delta_{ij}$（常数 Fisher 度量——例如在独立高斯噪声下的线性 FHN），Christoffel 符号为零，信息 Euler-Lagrange 方程退化为：
$$
m\,\ddot{\theta}_i = -\frac{\partial\mathcal{V}_{\text{info}}}{\partial\theta_i}
$$

即：**信息牛顿第二定律**——信息 ``加速度'' 正比于信息 ``力''。这里的 ``质量'' $m$ 是标量 Fisher 信息。

### 信息拉格朗日量和 SGD 的关系

标准 SGD 更新 $\theta_{t+1} = \theta_t - \eta\nabla L(\theta_t)$ 可以看作信息 Euler-Lagrange 方程在强耗散极限（忽略惯性项）下的离散化：
$$
\theta_{t+1} - \theta_t = -\eta \cdot g^{-1}(\theta_t) \cdot \frac{\partial\mathcal{V}_{\text{info}}}{\partial\theta}
$$

这意味着：**SGD 是信息力学在过阻尼极限下的实现**——系统完全受势能力驱动，无信息惯性。带动量的 SGD（如 Adam）则保留了一部分惯性项。

# 哈密顿力学：信息正则方程

## 信息正则动量和信息哈密顿量

### 从拉格朗日到哈密顿：Legendre 变换

**定义（信息正则动量）：**

和广义坐标 $\theta_i$ 对偶的信息正则动量定义为：
$$
p_i = \frac{\partial \mathcal{L}_{\text{info}}}{\partial \dot{\theta}_i} = \sum_j g_{ij}(\theta)\,\dot{\theta}_j
$$

$p_i$ 的意义：沿第 $i$ 个参数方向 ``移动'' 时，信息动能的变化率。它衡量了在信息意义上移动多 ``费劲''——因为质量张量 $g_{ij}$ 的参与，$p_i$ 同时包含了速度和方向两方面的信息。

对于 FP 神经层，正则动量有直接的计算公式：
$$
p_i^{\text{FHN}} = \frac{1}{\sigma^2}\sum_{j,k} \frac{\partial u(O_k; R)}{\partial R_i}\frac{\partial u(O_k; R)}{\partial R_j} \cdot \dot{R}_j
$$

**定义（信息哈密顿量）：**

通过 Legendre 变换定义信息哈密顿量：
$$
\boxed{\mathcal{H}_{\text{info}}(\theta, p) = \sum_i p_i \dot{\theta}_i - \mathcal{L}_{\text{info}}(\theta, \dot{\theta})}
$$

代入 $\dot{\theta}_i = \sum_j g^{ij}(\theta) p_j$（$g^{ij}$ 是 Fisher 度量的逆矩阵），得到：
$$
\mathcal{H}_{\text{info}}(\theta, p) = \frac{1}{2}\sum_{i,j} g^{ij}(\theta)\,p_i p_j + \mathcal{V}_{\text{info}}(\theta)
$$

第一项是信息的 ``动能表示''（在动量空间），第二项是信息势能。信息哈密顿量是系统的*总信息能*。

### 信息哈密顿量的物理含义

对于信息瓶颈问题：
$$
\mathcal{H}_{\text{info}} = \frac{1}{2}p^T G^{-1}(\theta) p + \big[I(X;T_\theta) - \beta I(T_\theta; Y)\big]
$$

在自然梯度下降中，每一步的参数更新等价于在哈密顿系统中走一步——动量方向由 $p \propto G\dot{\theta}$ 给出，势能力由 $-\partial\mathcal{V}_{\text{info}}/\partial\theta$ 给出。Adam 优化器可以看作是对这个哈密顿系统做了*辛积分*。

## 信息正则方程

**定理（信息哈密顿正则方程）：**

信息系统的动力学由以下一阶耦合方程组完全描述：
$$
\boxed{
    \begin{aligned}
        \dot{\theta}_i &= \frac{\partial \mathcal{H}_{\text{info}}}{\partial p_i} = \sum_j g^{ij}(\theta)\,p_j 
        \dot{p}_i &= -\frac{\partial \mathcal{H}_{\text{info}}}{\partial \theta_i} = -\frac{1}{2}\sum_{j,k}\frac{\partial g^{jk}}{\partial\theta_i}p_j p_k - \frac{\partial\mathcal{V}_{\text{info}}}{\partial\theta_i}
    \end{aligned}}
$$

第一个方程是速度-动量关系（$p = G\dot{\theta}$ 的反转）。第二个方程是力的平衡——左侧是动量的变化率，右侧是统计流形的 ``几何力''（第一项）和信息势能的 ``势力''（第二项）。

### 正则方程的物理意义

- **速度方程** $\dot{\theta}_i = g^{ij}p_j$：在 Fisher 度量大的方向上，给定动量产生较小的速度——因为该方向 ``质量'' 大。
- **力方程** $\dot{p}_i = -\partial\mathcal{H}/\partial\theta_i$：动量变化率等于信息力的负值。这里的信息力包括两个贡献：(a) 流形弯曲导致的几何力；(b) 信息势能（压缩-保真）的梯度力。

### FHN 物理介质的正则方程

将信息正则方程应用于 FHN 介质（参数 $\theta = R$，源点构型）：

$$
\dot{R}_i = \frac{\sigma^2}{2}\sum_j \left(\sum_k \frac{\partial u(O_k)}{\partial R_i}\frac{\partial u(O_k)}{\partial R_j}\right)^{-1} p_j
$$
$$
\dot{p}_i = -\frac{\partial}{\partial R_i}\left[I(R;u) - \beta I(u;S)\right] - \text{几何修正项}
$$

这意味着：FHN 介质中源点的 ``最优移动方向'' 不是简单的梯度方向，而是经过了 Fisher 度量的逆变换——这正是自然梯度的动力学基础。

### 信息能量守恒

当 $\mathcal{H}_{\text{info}}$ 不显含 ``信息时间'' $\tau$ 时：

**定理（信息能量守恒）：**

如果 $\partial\mathcal{H}_{\text{info}}/\partial\tau = 0$，则信息哈密顿量是守恒量：
$$
\frac{d\mathcal{H}_{\text{info}}}{d\tau} = \sum_i\left(\frac{\partial\mathcal{H}_{\text{info}}}{\partial\theta_i}\dot{\theta}_i + \frac{\partial\mathcal{H}_{\text{info}}}{\partial p_i}\dot{p}_i\right) = 0
$$

物理意义：在没有任何 ``外部信息注入''（如 SGD 的新的 mini-batch）的情况下，系统的总信息能在参数空间中守恒——信息在 Fisher 度量的动能和信息瓶颈势能之间来回转换，但总量不变。

**和一个实际的训练的对比**：SGD 训练不满足这个条件，因为每个 mini-batch 引入了新的数据，等效于 *外部信息力* 做功，改变了系统的总信息能。

# Noether 定理：信息守恒律

## Noether 定理在信息空间中的完整推广

**定理（信息 Noether 定理）：**

如果信息作用量 $\mathcal{S}_{\text{info}} = \int \mathcal{L}_{\text{info}}\,d\tau$ 在一个连续变换 $\theta_i \to \theta_i + \varepsilon\,\xi_i(\theta)$ 下不变（$\delta\mathcal{S}_{\text{info}} = 0$），则存在一个守恒的信息量：
$$
\boxed{Q = \sum_i p_i\,\xi_i(\theta) = \sum_{i,j} g_{ij}(\theta)\,\dot{\theta}_j\,\xi_i(\theta) = \text{常数}}
$$

**证明：**

在变分 $\delta\theta_i = \varepsilon\xi_i(\theta)$ 下，拉格朗日量的变分为：
$$
\delta\mathcal{L}_{\text{info}} = \sum_i\left(\frac{\partial\mathcal{L}_{\text{info}}}{\partial\theta_i}\xi_i + \frac{\partial\mathcal{L}_{\text{info}}}{\partial\dot{\theta}_i}\dot{\xi}_i\right)\varepsilon
$$

如果 $\delta\mathcal{L}_{\text{info}} = 0$（对称性），利用 Euler-Lagrange 方程 $\frac{\partial\mathcal{L}_{\text{info}}}{\partial\theta_i} = \frac{d}{d\tau}\frac{\partial\mathcal{L}_{\text{info}}}{\partial\dot{\theta}_i}$，得：
$$
0 = \sum_i\left(\frac{d}{d\tau}\frac{\partial\mathcal{L}_{\text{info}}}{\partial\dot{\theta}_i}\xi_i + \frac{\partial\mathcal{L}_{\text{info}}}{\partial\dot{\theta}_i}\dot{\xi}_i\right) = \frac{d}{d\tau}\sum_i p_i \xi_i
$$
因此 $Q = \sum_i p_i \xi_i$ 是守恒量。

### 信息空间中的三种基本对称性和守恒律

1. **信息平移不变性** $\to$ **信息动量守恒** 如果 $\mathcal{L}_{\text{info}}$ 在参数空间中对某个方向 $\theta_k$ 的平移保持不变（$\partial\mathcal{L}_{\text{info}}/\partial\theta_k = 0$），则： $$ p_k = \sum_j g_{kj}(\theta)\,\dot{\theta}_j = \text{常数} $$ **在 FHN 中的含义**：如果源点的某个移动方向不改变信息瓶颈目标函数的值（对称方向），则该方向上的 ``信息动量'' 守恒——参数在该方向上匀速移动。这等价于 Amari 信息几何中的测地线。
2. **信息``时间''平移不变性** $\to$ **信息能量守恒** 如果 $\mathcal{L}_{\text{info}}$ 不显含 $\tau$，则 $\mathcal{H}_{\text{info}} = \text{常数}$（定理 \ref{thm:energy}）。信息时间平移对称性意味着系统的信息处理规则不随时间变化。
3. **信息规范不变性** $\to$ **数据处理不等式作为守恒律** 考虑变换 $T \to f(T)$，其中 $f$ 是 $T$ 的任意函数。如果 $X \to T \to Y$ 构成马尔可夫链，则： $$ I(X; f(T)) \leq I(X; T) $$ 数据处理不等式可以被看作一个*信息单向下界*——它不是严格守恒律，而是一个 *信息熵不减* 的 Noether 型定理。在信息空间中的 ``规范自由度''（对 $T$ 做任意可逆变换不应该改变系统的信息量）给出了 $I(X;T)$ 的某种上限。

### 和信息瓶颈守恒量的连接

信息瓶颈目标函数 $I(X;T) - \beta I(T;Y)$ 的最小值在训练中是逐渐降低的——这不是因为 Noether 定理被违反，而是因为*外部信息力*（SGD 步）在做功。如果把 SGD 的步看作一个 ``外源项''，Noether 定理的形式推广为：
$$
\frac{dQ}{d\tau} = \text{外源做功速率}
$$

# 辛几何：信息相空间的结构

## 信息相空间和信息辛形式

### 从哈密顿力学到辛几何

分析力学的现代形式是辛几何 (symplectic geometry)。信息哈密顿力学同样具有自然的辛结构。

**定义（信息辛形式）：**

在信息相空间 $\{(\theta_i, p_i)\}_{i=1}^{d}$ 上，定义*信息辛形式*：
$$
\omega_{\text{info}} = \sum_{i=1}^{d} dp_i \wedge d\theta_i
$$

这是一个封闭的（$d\omega_{\text{info}} = 0$）、非退化的二阶微分形式。它赋予了信息相空间一个辛结构。

### 信息哈密顿矢量场

**定义（信息哈密顿矢量场）：**

给定信息哈密顿量 $\mathcal{H}_{\text{info}}$，对应的*信息哈密顿矢量场* $X_{\mathcal{H}}$ 满足：
$$
\omega_{\text{info}}(X_{\mathcal{H}}, \cdot) = d\mathcal{H}_{\text{info}}
$$

在正则坐标系中，其分量为：
$$
X_{\mathcal{H}} = \sum_i \left(\frac{\partial\mathcal{H}_{\text{info}}}{\partial p_i}\frac{\partial}{\partial\theta_i} - \frac{\partial\mathcal{H}_{\text{info}}}{\partial\theta_i}\frac{\partial}{\partial p_i}\right)
$$

这和正则方程 (\ref{eq:info_hamilton_equations}) 等价——系统在相空间中的轨迹 $(\theta(\tau), p(\tau))$ 就是哈密顿矢量场 $X_{\mathcal{H}}$ 的积分曲线：
$$
\frac{d}{d\tau}\begin{pmatrix}\theta \\ p\end{pmatrix} = X_{\mathcal{H}}(\theta, p)
$$

### Fisher 度量和辛结构的统一

在信息力学中，Fisher 度量（黎曼结构）和辛形式（辛结构）通过信息哈密顿量自然地统一：
$$
\mathcal{H}_{\text{info}} = \frac{1}{2}g^{ij}p_i p_j + \mathcal{V}_{\text{info}}(\theta)
$$

其中 $g^{ij}$ 来自 Fisher 黎曼度量（逆），而辛形式 $\omega_{\text{info}} = dp_i \wedge d\theta_i$ 来自相空间的自然结构。**Amari 的信息几何提供了黎曼部分，分析力学的辛几何提供了相空间部分，两者在信息哈密顿力学中统一。**

## 信息泊松括号

**定义（信息泊松括号）：**

两个信息相空间函数 $F(\theta,p)$ 和 $G(\theta,p)$ 的*信息泊松括号*定义为：
$$
\boxed{\{F, G\}_{\text{info}} = \sum_{i=1}^{d} \left(\frac{\partial F}{\partial\theta_i}\frac{\partial G}{\partial p_i} - \frac{\partial F}{\partial p_i}\frac{\partial G}{\partial\theta_i}\right)}
$$

泊松括号满足反交换律、双线性、Jacobi 恒等式和 Leibniz 规则。

信息泊松括号的直接物理应用——任何信息量 $F(\theta,p)$ 沿着哈密顿流的时间演化为：
$$
\frac{dF}{d\tau} = \{F, \mathcal{H}_{\text{info}}\}_{\text{info}} + \frac{\partial F}{\partial\tau}
$$

**基本泊松括号**：
$$
\{\theta_i, \theta_j\}_{\text{info}} = 0, \quad \{p_i, p_j\}_{\text{info}} = 0, \quad \{\theta_i, p_j\}_{\text{info}} = \delta_{ij}
$$

### 信息测不准关系的推导

从泊松括号可以直接推导信息的 ``测不准关系''。对于两个信息可观测量 $\hat{A}, \hat{B}$（如 $I(X;T)$ 和 $I(T;Y)$），在统计流形上：

**命题（信息测不准关系）：**

$$
\Delta A \cdot \Delta B \geq \frac{1}{2} \big|\mathbb{E}[\{A, B\}_{\text{info}}]\big|
$$
其中 $\Delta A = \sqrt{\operatorname{Var}(A)}$，期望在变分后验 $Q(\theta,p)$ 上取。

**物理意义**：你不能同时最小化压缩 $I(X;T)$ 和最大化保真 $I(T;Y)$ 到任意精度——它们之间的泊松括号非零，给出一个不可逾越的折中下界。这与信息瓶颈的 $\beta$ 参数的意义一致：$\beta$ 控制着这个权衡的相对权重，但从泊松括号的角度看，这个权衡是几何的——来自相空间的正则对易关系。

# 信息 Liouville 定理和信息熵守恒

## 信息相空间中的体积守恒

**定理（信息 Liouville 定理）：**

在信息哈密顿力学中，相空间体积元在哈密顿流下保持不变：
$$
\frac{d}{d\tau}\left(\prod_i d\theta_i dp_i\right) = 0
$$

等价地，相空间密度 $\rho(\theta, p, \tau)$ 满足*信息 Liouville 方程*：
$$
\frac{\partial\rho}{\partial\tau} + \{\rho, \mathcal{H}_{\text{info}}\}_{\text{info}} = 0
$$

**证明：**

由辛形式 $\omega_{\text{info}} = \sum_i dp_i \wedge d\theta_i$ 的封闭性和哈密顿矢量场的无散度性质：
$$
\operatorname{div} X_{\mathcal{H}} = \sum_i\left(\frac{\partial}{\partial\theta_i}\frac{\partial\mathcal{H}_{\text{info}}}{\partial p_i} - \frac{\partial}{\partial p_i}\frac{\partial\mathcal{H}_{\text{info}}}{\partial\theta_i}\right) = 0
$$
因此哈密顿流是无散的，相空间体积不变。

### 对 SGD 训练的深刻含义

**SGD 带有随机噪声，不是严格的哈密顿系统。** 在 SGD 中，有效动力学为：
$$
d\theta = -G^{-1}(\theta)\nabla\mathcal{V}_{\text{info}}(\theta)\,d\tau + \sqrt{2\eta\Sigma(\theta)}\,dW_\tau
$$

噪声项破坏了 Liouville 定理——相空间体积不再守恒，而是以信息熵产生率 $\dot{S}_{\text{info}}$ 的速率收缩或膨胀。这直接连通了 Still 定理：$\dot{S}_{\text{热力学}} \geq I_{\text{pred}}$。

在 Fokker-Planck 描述下：
$$
\frac{\partial\rho}{\partial\tau} + \{\rho, \mathcal{H}_{\text{info}}\}_{\text{info}} = \eta\sum_{i,j}\frac{\partial^2}{\partial\theta_i\partial\theta_j}(\Sigma_{ij}\rho)
$$

右侧的扩散项表示噪声导致的相空间体积变化——它和 Still 定理中的 $\dot{S}_{\text{热力学}}$ 成正比。

## 信息吉布斯分布和信息温度

在平衡态（$\partial\rho/\partial\tau = 0$），Liouville 方程给出：
$$
\rho_{\text{eq}}(\theta, p) \propto \exp\!\left(-\frac{\mathcal{H}_{\text{info}}(\theta, p)}{k_B T_{\text{info}}}\right)
$$

其中 $T_{\text{info}}$ 是*信息温度*——在 SGD 中 $T_{\text{info}} \propto \eta/B$（学习率除以 batch size）。这和物理学中的 Gibbs 分布 $\rho \propto e^{-E/k_B T}$ 在形式上完全一致。

**信息温度的实际值**：在典型的神经网络训练中，$\eta \approx 10^{-3}$，$B \approx 128$，$T_{\text{info}} \approx 10^{-5}$——这是一个极低的 ``信息温度''。系统在接近绝对零度的状态下运行，这就是为什么 SGD 几乎总是找到局部极小值。

# 信息正则变换和信息作用量-角度变量

## 信息正则变换

**定义（信息正则变换）：**

一个从 $(\theta, p)$ 到 $(\Theta, P)$ 的变换是信息正则变换，如果它保持信息辛形式不变：
$$
\sum_i dp_i \wedge d\theta_i = \sum_i dP_i \wedge d\Theta_i
$$
即雅可比矩阵 $M = \frac{\partial(\Theta, P)}{\partial(\theta, p)}$ 满足 $M^T J M = J$，其中 $J = \begin{pmatrix}0 & I \\ -I & 0\end{pmatrix}$。

**信息正则变换的物理意义**：在神经网络中，正则变换对应*重新参数化*网络，使得在新坐标下的动力学更简单。例如：
- 批量归一化 (BatchNorm) 可以看作一个特殊的信息正则变换——它将激活分布映射为各向同性的高斯，从而简化 Fisher 度量。
- 权重正交化初始化可以看作一个初始的信息正则变换——让初始坐标接近 ``笛卡尔'' 坐标。

## 信息 Hamilton-Jacobi 理论

**定理（信息 Hamilton-Jacobi 方程）：**

信息系统的动力学可以等价地由一个偏微分方程——*信息 Hamilton-Jacobi 方程*——来描述。设 $\mathcal{W}(\theta, \tau)$ 是信息作用量函数（Hamilton 主函数），它满足：
$$
\boxed{\frac{\partial\mathcal{W}}{\partial\tau} + \mathcal{H}_{\text{info}}\!\left(\theta, \frac{\partial\mathcal{W}}{\partial\theta}\right) = 0}
$$

其中 $\mathcal{H}_{\text{info}}(\theta, p)$ 是信息哈密顿量，$p_i = \partial\mathcal{W}/\partial\theta_i$。

**信息作用量函数的含义**：
$$
\mathcal{W}(\theta, \tau) = \int_{\theta_0}^{\theta} \mathcal{L}_{\text{info}}(\theta', \dot{\theta}')\,d\tau'
$$

$\mathcal{W}$ 是从初始状态到当前状态的真实信息路径上的 ``累计拉格朗日量''。它类似于光学中的程函 (eikonal)，也类似于量子力学中的相位。

### 和信息瓶颈的深层连接

在信息瓶颈的稳态（$\partial\mathcal{W}/\partial\tau = 0$，即 $\mathcal{H}_{\text{info}} = 0$——能量为零的临界解），Hamilton-Jacobi 方程退化为：
$$
\frac{1}{2}\sum_{i,j}g^{ij}(\theta)\frac{\partial\mathcal{W}}{\partial\theta_i}\frac{\partial\mathcal{W}}{\partial\theta_j} + \big[I(X;T_\theta) - \beta I(T_\theta; Y)\big] = 0
$$

将动量 $p_i = \partial\mathcal{W}/\partial\theta_i$ 换回速度 $\dot{\theta}_i$，这正是信息瓶颈的最优解条件——Fisher 动能和信息势能的平衡。

### 可分离性——当信息自由度可以分解

如果信息哈密顿量是可分离的：$\mathcal{H}_{\text{info}}(\theta, p) = \sum_i h_i(\theta_i, p_i)$，则 Hamilton-Jacobi 方程可分离，$\mathcal{W}(\theta) = \sum_i \mathcal{W}_i(\theta_i)$。

在神经网络中，这对应于*各层的信息处理可以被解耦*——即信息瓶颈在对每一层独立优化。深度网络的逐层预训练正是这一性质在离散时间中的实现。

## 信息作用量-角度变量

**定义（信息作用量-角度变量）：**

如果信息相空间的运动是周期的（如 FHN 螺旋波的旋转、神经网络的振荡模式），可以定义信息作用量 $J_i$ 和信息角度 $\phi_i$：
$$
J_i = \frac{1}{2\pi}\oint p_i\,d\theta_i \quad \text{（作用量：沿一个周期的积分）}
$$
$$
\dot{\phi}_i = \frac{\partial\mathcal{H}_{\text{info}}}{\partial J_i} = \omega_i(J) \quad \text{（角度以恒定频率演化）}
$$

**在 FHN 螺旋波中**：
- 信息作用量 $J = \frac{1}{2\pi}\oint p\,dR$ 沿螺旋波核心的一个完整旋转积分，等于螺旋波编码的信息量 $I(R; u)$ 乘以周期 $T$。
- 信息频率 $\omega = \partial\mathcal{H}_{\text{info}}/\partial J$ 是螺旋波的旋转频率（对应 FHN 的 $\sim 1/\tau_v$）。

**在神经网络中**：
- 信息作用量是每一层的 Fisher 信息容量（谱总和）。
- 信息频率是每层的信息转速——即信息在层间衰减/增长的速率（由 Jacobian 的谱决定）。

# 信息-物理统一场论框架

## 信息场的拉格朗日密度

将离散的参数空间推广到连续场。信息场 $\phi(x)$（如 FHN 的 $u$ 场、神经网络的连续深度极限）的信息拉格朗日量为：

$$
\mathcal{L}_{\text{info}}[\phi, \partial_\mu\phi] = \frac{1}{2}g^{\mu\nu}\,\partial_\mu\phi\,\partial_\nu\phi - V_{\text{info}}(\phi)
$$

其中 $g^{\mu\nu}$ 是信息时空（$x$ = 空间坐标, $\tau$ = 信息时间坐标）的 Fisher 度量张量，$V_{\text{info}}(\phi)$ 是信息势能密度：
$$
V_{\text{info}}(\phi) = \frac{D}{2}|\nabla\phi|^2 + U(\phi) - I_{\text{ext}}\cdot\phi
$$

**第一项** $D|\nabla\phi|^2$ 是信息扩散势（空间上的 Fisher 信息），**第二项** $U(\phi)$ 是局部信息处理势（如 FHN 的 $u - u^3/3$），**第三项** $I_{\text{ext}}\cdot\phi$ 是外部信息驱动。

## 信息场的 Euler-Lagrange 方程

对信息场作用量做变分 $\delta\mathcal{S}_{\text{info}} = 0$：

**定理（信息场方程）：**

$$
\boxed{\partial_\mu\frac{\partial\mathcal{L}_{\text{info}}}{\partial(\partial_\mu\phi)} - \frac{\partial\mathcal{L}_{\text{info}}}{\partial\phi} = 0}
$$

展开得到信息场的 ``运动方程''：
$$
g^{\mu\nu}\partial_\mu\partial_\nu\phi + \partial_\mu g^{\mu\nu}\partial_\nu\phi + \frac{\partial V_{\text{info}}}{\partial\phi} = 0
$$

第二项是信息时空 ``弯曲''（$g^{\mu\nu}$ 依赖位置）导致的 Christoffel 力——等价于统计流形上的曲率效应。如果 Fisher 度量是常数（欧氏极限），这一项为零。

**从信息场到场信道模型**：FHN PDE 就是这个方程在特定 $g^{\mu\nu}$ 和 $V_{\text{info}}$ 下的实例——$g^{00} = 1$, $g^{ij} = -D\delta^{ij}$ (空间部分), $V_{\text{info}} = -\frac{1}{2}u^2 + \frac{1}{12}u^4 + uv - I_{\text{ext}}u$。

## 信息场的 Noether 定理和守恒流

**定理（信息场的 Noether 定理）：**

如果信息场拉格朗日密度 $\mathcal{L}_{\text{info}}$ 在变换 $\phi \to \phi + \varepsilon\delta\phi$ 下不变，则存在一个守恒的*信息 Noether 流* $j^\mu$：
$$
j^\mu = \frac{\partial\mathcal{L}_{\text{info}}}{\partial(\partial_\mu\phi)}\delta\phi, \quad \partial_\mu j^\mu = 0
$$

守恒的信息荷为 $Q = \int j^0\,dx$。

**信息时空平移不变性** 直接给出信息场版本的能量-动量张量：
$$
T^{\mu\nu}_{\text{info}} = \frac{\partial\mathcal{L}_{\text{info}}}{\partial(\partial_\mu\phi)}\partial^\nu\phi - g^{\mu\nu}\mathcal{L}_{\text{info}}
$$

$T^{00}_{\text{info}}$ 是*信息能量密度*（局部信息哈密顿量密度），$T^{0i}_{\text{info}}$ 是*信息动量流密度*（信息在空间中流动的速率）。

## 信息场的传播子——从拉格朗日到场互信息

在信息场论中，场的传播子（Green 函数）$G(x,y)$ ——在 $y$ 点注入单位信息后在 $x$ 点产生的响应——直接从信息场方程导出。在弱场极限（高斯近似）：
$$
G(x,y) = \langle\phi(x)\phi(y)\rangle = \left(\frac{\delta^2\mathcal{S}_{\text{info}}}{\delta\phi(x)\delta\phi(y)}\bigg|_{\phi=\phi_0}\right)^{-1}
$$

而信道容量公式 $C = \frac{1}{2}\sum_k\log_2(1+\lambda_k/\sigma^2)$ 中的 $\lambda_k$ 正是这个信息传播算子的特征值。**场的拉格朗日量完整决定了场的信息传输特性——从最小作用量原理直接到信道容量。**

# 应用：从信息力学到具体系统

## FHN 介质的信息拉格朗日量

将前述框架具体化到 FHN 可激发介质：

$$
\mathcal{L}_{\text{FHN}}[u, v, R] = \frac{1}{2\sigma^2}\sum_{k}\left(\frac{\partial u(O_k)}{\partial R_i}\dot{R}_i\right)^2 - \big[I(R; u) - \beta I(u; S)\big]
$$

信息 Euler-Lagrange 方程给出源点的最优移动规律：
$$
\sum_j g_{ij}^{\text{FHN}}(R)\,\ddot{R}_j + \Gamma_{ijk}^{\text{FHN}}\dot{R}_j\dot{R}_k = -\frac{\partial}{\partial R_i}\big[I(R; u) - \beta I(u; S)\big]
$$

这个方程揭示了 FHN 介质中 ``物理学习'' 的本质——源点不是简单地沿欧氏梯度下降，而是沿 Fisher 度量加权的测地线在信息流形上移动。

## 神经网络的信息哈密顿量

对于一个 $L$ 层网络，设 $\theta = \{W^{(l)}, b^{(l)}\}_{l=1}^{L}$。信息哈密顿量：
$$
\mathcal{H}_{\text{NN}}(\theta, p) = \frac{1}{2}\sum_{l,i,j} g^{ij}_{(l)}(\theta)\,p_i^{(l)}p_j^{(l)} + \big[I(X; T_\theta) - \beta I(T_\theta; Y)\big]
$$

SGD + 动量的更新可以看作对这个哈密顿系统做了离散化的辛积分（如 Störmer-Verlet 格式）。动量项 $p^{(l)}$ 就是优化器中的动量缓冲——它在统计流形上保持了 ``惯性''。

## 最优传输的哈密顿形式

最优传输 (Optimal Transport) 可以看作信息哈密顿力学的一个特例。将分布 $P(X)$ 传输到分布 $P(Y)$ 的 ``最小信息作用量'' 由 Wasserstein 距离给出：
$$
\mathcal{W}_2^2(P, Q) = \min \int_0^1 \int \|\dot{\phi}_t(x)\|^2 \, d\rho_t(x)\,dt
$$

这在信息力学中对应：$g_{ij} = \delta_{ij}$（欧氏度量）、$V_{\text{info}} = 0$（无信息势能）下的最短路径。

# 结论

## 完整的信息力学层级结构

| **分析力学概念** | **信息力学对应** | **具体形式** |
| --- | --- | --- |
| 最小作用量原理 $\delta S = 0$ | $\delta\mathcal{S}_{\text{info}} = 0$ | 信息瓶颈变分原理 |
| 拉格朗日量 $\mathcal{L} = K - V$ | $\mathcal{L}_{\text{info}} = \frac{1}{2}g_{ij}\dot{\theta}_i\dot{\theta}_j - \mathcal{V}_{\text{info}}$ | Fisher 动能 - IB 势能 |
| Euler-Lagrange 方程 | 信息运动方程 | 自然梯度 + Christoffel 力 = IB 梯度 |
| 正则动量 $p_i = \partial\mathcal{L}/\partial\dot{q}_i$ | $p_i = g_{ij}\dot{\theta}_j$ | Fisher 加权的速度 |
| 哈密顿量 $\mathcal{H} = p\dot{q} - \mathcal{L}$ | $\mathcal{H}_{\text{info}} = \frac{1}{2}g^{ij}p_i p_j + \mathcal{V}_{\text{info}}$ | Fisher 动能(动量空间) + IB |
| 正则方程 | 信息正则方程 | $\dot{\theta} = g^{-1}p$, $\dot{p} = -\partial\mathcal{H}/\partial\theta$ |
| Noether 定理 | 信息守恒律 | DPI, 平移对称性 $\to p_k$ 守恒 |
| 泊松括号 $\{F,G\}$ | $\{F,G\}_{\text{info}}$ | 信息测不准 $\Delta I(X;T)\cdot\Delta I(T;Y) \geq \cdots$ |
| Liouville 定理 | 信息相空间体积守恒 | SGD 噪声破坏它 $\to$ Still 定理 |
| Hamilton-Jacobi 方程 | 信息 H-J 方程 | $\partial\mathcal{W}/\partial\tau + \mathcal{H}(\theta, \partial\mathcal{W}/\partial\theta) = 0$ |
| 正则变换 | 信息坐标变换 | BatchNorm, 权重正交化 |
| 作用量-角度变量 $(J, \phi)$ | $(J_i, \phi_i)$ 信息振荡 | FHN 螺旋波频率, 网络振荡模式 |
| 场的拉格朗日密度 | 信息场拉格朗日量 | $g^{\mu\nu}\partial_\mu\phi\partial_\nu\phi - V_{\text{info}}(\phi)$ |
| 传播子 $G(x,y)$ | 信息 Green 函数 | 信道容量的特征值 $\lambda_k$ |

## 这个框架回答了什么

1. **信息瓶颈为什么是最小作用量原理**：信息拉格朗日量 $\mathcal{L}_{\text{info}} = \frac{1}{2}g_{ij}\dot{\theta}_i\dot{\theta}_j - [I(X;T) - \beta I(T;Y)]$ 将 IB 写成一个分析力学系统。变分 $\delta\int\mathcal{L}_{\text{info}} = 0$ 直接给出 IB 的最优条件。
2. **为什么 SGD 能用**：SGD 是信息 Euler-Lagrange 方程在过阻尼极限下的离散化。带动量的 SGD 是信息哈密顿方程的低阶辛积分子。
3. **为什么物理场能做计算**：物理场的 PDE 演化就是信息场的 Euler-Lagrange 方程——物理定律自动执行了信息变分原理的优化。
4. **Fisher 度量的物理意义**：$g_{ij}$ 是信息空间的 ``质量张量''——它决定了信息加速的方向依赖性和 Noether 守恒量的形式。
5. **信息处理的基本限制**：信息泊松括号给出压缩和保真之间的测不准关系。信息 Liouville 定理 + Still 定理给出了信息处理的不可逆代价。
6. **物理-网络的深层统一**：两者共享完全相同的信息拉格朗日量和信息哈密顿量——区别仅在于实现（PDE vs 矩阵乘法）和参数化（物理参数 vs 权重）。

> **核心陈述：**
> **信息力学 = 分析力学在统计流形上的推广。**
> 最小作用量原理 $\;\to\;$ 信息拉格朗日量 $\;\to\;$ 信息 Euler-Lagrange 方程 $\;\to\;$ 信息哈密顿量 $\;\to\;$ 
> 信息正则方程 $\;\to\;$ 信息 Noether 定理 $\;\to\;$ 信息泊松括号 $\;\to\;$ 信息 Liouville 定理 
> **这是一套*严格的、自洽的*数学体系，**
> **它统一了物理场的信息传播和神经网络的信息处理，**
> **其深度和精度与分析力学本身相当。**
