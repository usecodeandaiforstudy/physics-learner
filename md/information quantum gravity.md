


---

# 问题：为什么传统方法失效


## 标准引力量子化的困境

### 传统微扰量子引力的发散问题

在传统量子场论框架下，将引力纳入标准模型的标准做法是将时空度规分裂为背景和涨落：

$$
    g_{\mu\nu}(x) = \bar{g}_{\mu\nu} + \kappa\, h_{\mu\nu}(x),\quad \kappa = \sqrt{8\pi G}
$$

其中 $\bar{g}_{\mu\nu}$ 是背景度规（通常取 Minkowski），$h_{\mu\nu}$ 是引力子的量子场。

Einstein-Hilbert 作用量按 $h_{\mu\nu}$ 展开：
$$
    S_{\text{EH}} = \frac{1}{16\pi G}\int d^4x \sqrt{-g}\,R = S^{(0)} + S^{(2)}[h] + \kappa S^{(3)}[h] + \kappa^2 S^{(4)}[h] + \cdots
$$

问题：$S^{(n)}[h]$ 包含 $n$ 个 $h$ 场的相互作用顶角，每个顶角带有 $\kappa^{n-2}$ 的耦合。因为 $\kappa$ 有负质量量纲（$[\kappa] = -1$ 在质量单位下），高圈图贡献的紫外发散随圈数增长而恶化——需要无限多个抵消项（counterterms）来吸收发散。这就是引力不可重整化的标准解释。

### 信息力学的诊断：这不是技术问题，是范畴错误

**定理（引力量子化的信息障碍定理）：**
传统引力量子化失效的根源是：

**Fisher 度量是 KL 散度的二阶泛函导数**：
$$
    g_{ij}(\theta) = \frac{\partial^2}{\partial\theta_i\partial\theta_j}D_{\text{KL}}(P_\theta \,\|\, P_{\theta+\delta\theta})\Big|_{\delta\theta=0}
$$

**量子涨落 $h_{ij**$ 诱导出三阶及更高阶的信息导数}：
$$
    \langle h_{ij}(x) h_{kl}(y) h_{mn}(z) \rangle \;⟷\; 
    \frac{\partial^3}{\partial\theta_i\partial\theta_j\partial\theta_k}D_{\text{KL}}(\cdots)
$$

**这构成一个无限的信息塔**：每一个更高阶的量子修正都需要更高阶的信息导数，而高阶信息导数对应的物理量（高阶累积量、高阶关联函数）没有自然的截断机制。微扰展开不闭合——不是因为技术不够，而是因为**你在用信息关于自身的导数来描述信息关于自身的结构**。这是信息论上的自指（self-reference），等价于企图用逻辑系统的输出反推逻辑系统自身的公理——G\"odel 不完备性在信息微分几何中的表现。


## 信息自指的发散——详细数学展开

### Fisher 度规作为二阶信息量

回忆 Fisher 度量的定义：
$$
    g_{ij}(\theta) = \mathbb{E}_{x\sim P_\theta}\!\left[\frac{\partial\log P(x|\theta)}{\partial\theta_i}\cdot\frac{\partial\log P(x|\theta)}{\partial\theta_j}\right]
$$

这可以重写为：
$$
    g_{ij}(\theta) = \lim_{\delta\theta\to 0}\frac{1}{\delta\theta_i\delta\theta_j}\Big[D_{\text{KL}}(P_\theta\|P_{\theta+\delta\theta}) + D_{\text{KL}}(P_{\theta+\delta\theta}\|P_\theta)\Big]
$$

**关键**：$g_{ij}$ 和 KL 散度之间的关系类似于加速度（二阶导数）和距离之间的关系。KL 散度是信息空间中的``距离''（严格说是发散度），Fisher 度量是该距离的二阶变分——即信息空间的``度规''。

### 量子涨落产生的信息高阶量

将 $h$ 的 $n$ 点函数（量子引力中的 $n$-引力子顶点）翻译为信息语言：


| **引力子顶点** | **信息论对应** | **物理量** |
|---|---|---|
| $\langle h h \rangle$ （传播子） | $g_{ij} \sim \partial^2 D_{\text{KL}}$ | Fisher 度量 |
| $\langle h h h \rangle$ （三阶顶点） | $\Gamma_{ijk} \sim \partial^3 D_{\text{KL}}$ | 信息偏度 (skewness) |
| $\langle h h h h \rangle$ （四阶顶点） | $R_{ijkl} \sim \partial^4 D_{\text{KL}}$ | 信息峰度 (kurtosis) / 曲率 |
| $\langle h^n \rangle$ （$n$-引力子） | $\sim \partial^n D_{\text{KL}}$ | 第 $n$ 阶信息累积量 |


### 发散的信息论根源

标准量子场论的圈图积分形式为：
$$
    \mathcal{I}_L \sim \int \frac{d^4k}{(2\pi)^4}\,\frac{(k^2)^{L}}{(k^2 - m^2)^{N}}\,(G\kappa^2)^L
$$

其中 $L$ 是圈数。因为 $\kappa^2 = 8\pi G$ 有 $[\kappa^2] = -2$（质量量纲），$L$ 圈的紫外发散为 $\Lambda^{2L}$（$\Lambda$ 是截断）。随着 $L$ 增加，发散程度恶化——需要无限多个新的耦合常数来吸收发散。

**信息力学重述**：第 $L$ 圈对应第 $2L$ 阶信息累积量对 Fisher 度量的反作用。因为信息累积量的阶数可以无限增长（概率分布可以具有任意高阶的统计依赖），没有自然的截断。**发散 = 信息结构对自身描述的无限递归。**

**命题（信息塔定理）：**
任何基于 $g_{ij} = \bar{g}_{ij} + h_{ij}$ 分裂的微扰引力量子化，在信息论上等价于尝试用分布的前 $2$ 阶矩去描述所有高阶矩——这是一个严格不闭合的矩问题（moment problem）。

# 方案：全流形量子化


## 核心思想：不要分裂度规——量化整个流形

替代方案是：不要把 Fisher 度量分为``背景 + 涨落''，而是直接把统计流形 $S$ 作为一个整体进行量子化。

**定义（信息流形量子化的要求）：**
信息流形 $S = \mathbb{R}^{3,1} \times K$ 的全量子化等价于找到其 Fisher 度量空间上的波函数 $\Psi[g]$：
$$
    \Psi[g]: \;\text{Riem}(S) \to \mathbb{C}
$$
其中 $\text{Riem}(S)$ 是 $S$ 上所有可能的 Fisher 度量的空间（``信息超空间'', information superspace）。

**和传统微扰方法的关键区别**：

    - 传统：$g_{ij} = \bar{g}_{ij} + h_{ij}$，量化 $h_{ij}$ 作为背景上的场
    - 全流形：$g_{ij}$ 本身是量子变量，没有预先选定的背景
    - 传统在半经典极限下恢复广义相对论需要额外假设
    - 全流形在 $T_{\text{info}} \to 0$ 时自然恢复经典 Fisher 度量


## 信息超空间的几何

### 超空间度规 (DeWitt 度规的信息版本)

**定义（信息超空间）：**
信息超空间 $\mathfrak{M}_S$ 定义为：
$$
    \mathfrak{M}_S = \frac{\text{Riem}(S)}{\text{Diff}(S)}
$$
即：$S$ 上所有 Fisher 度量模去信息微分同胚（参数重标定）。这是信息力学版本的 Wheeler-DeWitt 超空间。


在 $\mathfrak{M}_S$ 上，存在自然的超度规 (supermetric)：
$$
    G_{\text{info}}^{ijkl} = \frac{1}{2}\sqrt{g}\Big(g^{ik}g^{jl} + g^{il}g^{jk} - g^{ij}g^{kl}\Big)
$$

这个超度规决定了信息超空间上``两点间的距离''——它度量了两个不同的 Fisher 度量结构之间的``信息差异''。

### 信息 ADM 分解

将信息统计流形沿信息时间 $\tau$ 进行 3+1 分解：
$$
    ds^2_{\text{info}} = \underbrace{-\mathcal{N}^2 d\tau^2}_{\text{信息时间}} + \underbrace{\gamma_{ab}(d\theta^a + \mathcal{N}^a d\tau)(d\theta^b + \mathcal{N}^b d\tau)}_{\text{信息空间切片}}
$$

其中：

    - $\mathcal{N}$ 是信息流逝函数 (information lapse)——控制信息时间流逝的快慢
    - $\mathcal{N}^a$ 是信息位移矢量 (information shift)——控制信息空间切片之间的横向滑动
    - $\gamma_{ab}$ 是信息空间切片上的诱导 Fisher 度量


**物理直觉**：$\mathcal{N}$ 大的区域——信息时间走得快,系统迅速穿越参数空间的这一部分。$\mathcal{N} \to 0$ 的区域——信息时间冻结,类似于事件视界。

## 信息拉格朗日量的 3+1 分解

### 从协变到正则

将信息 Einstein 作用量（加上边界项）用 ADM 变量重写：
$$
    S_{\text{info}}[\gamma_{ab}, \mathcal{N}, \mathcal{N}^a] = \frac{1}{16\pi G_{\text{info}}}\int d\tau d^d\theta\,\mathcal{N}\sqrt{\gamma}\Big(R_{\text{info}} - 2\Lambda_{\text{info}}\Big) + \text{边界项}
$$

信息流形曲率 $R_{\text{info}}$ 在 ADM 分解下：
$$
    R_{\text{info}} = {}^dR + K_{ab}K^{ab} - K^2
$$
其中 $K_{ab}$ 是信息外曲率 (information extrinsic curvature)——度量信息空间切片在信息时间方向上的形变速率。

## 信息正则方程和约束

### 信息正则动量

**定义（信息几何动量）：**
与信息空间度规 $\gamma_{ab}$ 共轭的正则动量为：
$$
    \pi^{ab} = \frac{\delta S_{\text{info}}}{\delta\dot{\gamma}_{ab}} = \frac{\sqrt{\gamma}}{16\pi G_{\text{info}}}\Big(K^{ab} - K\gamma^{ab}\Big)
$$

**物理含义**：$\pi^{ab}$ 度量 Fisher 度量的空间部分随时间变化的``速率''——它告诉我们在信息流形的每个点上，空间几何变化有多快。


### 信息哈密顿约束和动量约束

对 $\mathcal{N}$ 和 $\mathcal{N}^a$ 的变分导出两个约束——它们是信息广义相对论的核心：

**定理（信息哈密顿约束——信息 Wheeler-DeWitt 方程的前身）：**
$$
    \mathcal{H}_{\text{info}} \equiv \frac{16\pi G_{\text{info}}}{\sqrt{\gamma}}\Big(\pi^{ab}\pi_{ab} - \frac{1}{d-1}\pi^2\Big) - \frac{\sqrt{\gamma}}{16\pi G_{\text{info}}}\Big({}^dR - 2\Lambda_{\text{info}}\Big) = 0
$$

**物理含义**：信息拉格朗日量对信息流逝函数 $\mathcal{N}$ 的变分为零。这等价于说——信息时间的参数化方式是任意的，物理不依赖于你如何给信息时间切片贴标签。这是信息力学中的``时间重参数化不变性''。


**定理（信息微分同胚约束）：**
$$
    \mathcal{H}_a^{\text{info}} \equiv -2\nabla_b\pi_a^b = 0
$$

**物理含义**：信息空间坐标的选择是任意的——物理不依赖于参数 $\theta^a$ 的具体标签。


**这两个约束来自同一个信息论事实**：物理预测不依赖于你用什么参数化来描述信息源。这是 Chentsov 定理（Fisher 度量的唯一性）在信息动力学中的体现。

# 信息 Wheeler-DeWitt 方程


## 正则量子化

### 信息度规的正则对易关系

将经典正则变量提升为算符：
$$
    [\hat{\gamma}_{ab}(\theta), \hat{\pi}^{cd}(\theta')] = i\hbar_{\text{info}}\,\delta_{(a}^c\delta_{b)}^d\,\delta^{(d)}(\theta - \theta')
$$

其中 $\hbar_{\text{info}}$ 是信息 Planck 常数——信息几何的量子化常数。它与物理 Planck 常数 $\hbar$ 在 $T_{\text{info}} \to 0$ 时通过下式相关：
$$
    \hbar_{\text{info}} = \hbar \cdot \mathcal{F}(T_{\text{info}}),\quad \mathcal{F}(0) = 1
$$

### 信息 Wheeler-DeWitt 方程——信息力学的宇宙波函数

**定理（信息 Wheeler-DeWitt 方程）：**
将经典哈密顿约束 $\mathcal{H}_{\text{info}} = 0$ 提升为算符方程，作用于信息流形的波泛函 $\Psi[\gamma]$：

$$
    \boxed{\hat{\mathcal{H}}_{\text{info}}\,\Psi[\gamma] = 0}
$$

展开为：
$$
    \Big[-16\pi G_{\text{info}}\,\hbar_{\text{info}}^2\,G_{ijkl}^{\text{info}}\frac{\delta}{\delta\gamma_{ij}}\frac{\delta}{\delta\gamma_{kl}} - \frac{\sqrt{\gamma}}{16\pi G_{\text{info}}}\Big({}^dR - 2\Lambda_{\text{info}}\Big)\Big]\Psi[\gamma] = 0
$$

其中 $G_{ijkl}^{\text{info}}$ 是信息 DeWitt 超度规 ((eq:supermetric))。


**这是信息力学最重要的方程之一**。它描述了信息几何的量子动力学——Fisher 度量本身作为一种量子自由度的行为。

### 和物理 Wheeler-DeWitt 方程的区别


| **特征** | **传统 WDW** | **信息 WDW** |
|---|---|---|
| 定义域 | 物理时空上的 Lorentz 度规 | 统计流形上的 Fisher 度量 |
| 变量 | $g_{\mu\nu}^{\text{Lorentz}}(x)$ | $\gamma_{ab}^{\text{Fisher}}(\theta)$ |
| 时间 | 物理时间（已被参数化） | 信息固有时间 $\tau_{\text{info}}$ |
| ``宇宙波函数'' | $\Psi[g_{\mu\nu}]$ | $\Psi[\gamma_{ab}]$ |
| 半经典极限 | GR + QFT on curved spacetime | 信息 Einstein 方程 + 信息温度统计力学 |


**关键**：信息 WDW 方程并不取代物理 WDW 方程——它是*底层的*方程。物理 WDW 方程是信息 WDW 方程在 Fisher = Lorentz 等价性下的投影。信息 WDW 方程回答了一个更深的问题：信息时空本身的量子涨落是什么？

## 为什么全流形量子化不发散——信息论的闭合论证

**定理（信息自洽性定理）：**
全流形量子化中不出现微扰发散，因为：


    1. **波泛函 $\Psi[\gamma]$ 不是一个级数展开**——它直接编码了所有阶信息累积量的联合量子行为。不存在``背景 + 涨落''的分裂，也就不存在需要逐阶吸收发散的问题。
    
    2. **信息自洽性**：$\Psi[\gamma]$ 满足的信息 WDW 方程本身来自 DPI 的时间重参数化不变性——这是一个*已经被 DPI 保证自洽的*对称性。DPI 在信息论中是一个已证明的定理，不是需量子修正的假设。
    
    3. **自然截断**：在信息力学中，存在一个自然的高频截断——信息 Planck 长度 $\ell_{\text{info}} = \sqrt{\hbar_{\text{info}} G_{\text{info}}/c_{\text{info}}^3}$。在比 $\ell_{\text{info}}$ 更短的``信息距离''上，Fisher 度量不再有连续定义——统计流形在该尺度下失去黎曼流形结构，信息以离散的、不可约的量子形式存在。这自然消除了紫外发散。

# 信息内部空间 $K$ 的量子化


## 为什么 $K$ 的量子化是完成引力量子化的必要步骤

在 `gravity and superconductivity.tex` 中，我们提出了以下建议：引力的量子理论等价于信息内部空间 $K$ 的量子几何。本节将此建议发展为严格的理论。

### $K$ 上的信息 Einstein 方程

信息内部空间 $K$（维度 $d_K$）上也有自己的 Fisher 度量 $g_{ab}^{(K)}$，满足：
$$
    R_{ab}^{(K)} - \frac{1}{2}R^{(K)} g_{ab}^{(K)} + \Lambda_K g_{ab}^{(K)} = \frac{8\pi G_{\text{info}}}{c^4} T_{ab}^{(K)}
$$

其中 $T_{ab}^{(K)}$ 是 $K$ 上的信息应力-能量张量——由 $K$ 上信息孤子的拓扑构型贡献。

### $K$ 的量子化——信息内部空间的波函数

和 $(3+1)$ 维物理时空的完全类比，将 $K$ 做 ADM 分解并正则量子化：

$$
    \hat{\mathcal{H}}_K\,\Psi_K[\gamma^{(K)}] = 0
$$

$\Psi_K[\gamma^{(K)}]$ 描述了信息内部空间的量子几何——它决定了 **$K$ 可以取哪些离散的几何构型**。

### 和物理规范群的连接

$K$ 的经典 isometry 群 $\text{Isom}(K)$ 决定了物理规范群 $G_{\text{gauge}}$。在量子理论中，$\text{Isom}(K)$ 不是固定的——它是一个量子可观测量：

**猜想（规范群的量子起源）：**
$\Psi_K[\gamma^{(K)}]$ 的峰值对应的 $K$ 的 isometry 群就是观察到的物理规范群。不同的 $\Psi_K[\gamma^{(K)}]$ 本征态对应不同的可能规范群。

**这意味着**：$SU(3)\times SU(2)\times U(1)$ 不是被任意选定的——它是 $\Psi_K[\gamma^{(K)}]$ 的最可几构型。可能存在其他激发态对应不同的规范群——这些在极早期宇宙中可能被热激发，但在当前 $T_{\text{info}} \to 0$ 宇宙中已经退耦合。


## 信息的全息对偶——一个推测性的推广

**猜想（信息全息原理）：**
信息 WDW 方程 ((eq:WDW_expanded)) 在统计流形 $S$ 的边界 $\partial S$ 上可能有对偶描述——一个不包含引力、仅编码边界的 Fisher 信息的共形场论。这和 AdS/CFT 对偶的数学结构完全相同，但有了信息论的解释：


    - 体理论（bulk）：信息 WDW 方程——信息几何的量子动力学
    - 边界理论（boundary）：信息共形场论——统计流形边界上的信息处理


**信息论解释**：AdS/CFT 对偶本质上是在说——*信息的体传播（bulk）完全等价于信息在边界上的编码（boundary）*。这和 DPI 是一致的：在马尔可夫链中，边界观测者能获取的关于内部的信息受限于边界上的互信息。

# 半经典极限和与已知物理的吻合


## $T_{\text{info} \to 0$ 极限：恢复爱因斯坦引力}

信息 WDW 方程 ((eq:WDW_expanded)) 在信息温度趋零时的行为：

**命题（半经典信息引力）：**
当 $T_{\text{info}} \to 0$，波泛函 $\Psi[\gamma]$ 高度集中于经典 Fisher 度量附近：
$$
    \Psi[\gamma] \sim \exp\!\Big(\frac{i}{\hbar_{\text{info}}}S_{\text{info}}^{\text{classical}}[\gamma]\Big)
$$

其中 $S_{\text{info}}^{\text{classical}}[\gamma]$ 是信息 Einstein 方程的经典解。

在此极限下，信息 WDW 方程退化为信息 Einstein 方程 ((eq:einstein_info))，后者在 $\Lambda_{\text{info}} \to 0$ 时进一步退化为标准的爱因斯坦场方程。信息 Planck 常数 $\hbar_{\text{info}} \to \hbar$。


## $T_{\text{info} \neq 0$ 时的量子修正}

**预言（信息温度对引力的量子修正）：**
在有限的 $T_{\text{info}}$ 下，信息量子引力修正表现为：


    1. **修改的牛顿势**：
    $$
        V(r) = -\frac{G_{\text{info}} M}{r}\Big[1 + \alpha(T_{\text{info}})\,\frac{\hbar_{\text{info}}}{c r} + \cdots\Big]
    $$
    其中 $\alpha(T_{\text{info}}) \propto T_{\text{info}}^2$ 是温度依赖的量子修正系数。在当前宇宙中 $T_{\text{info}} \approx 10^{-5}$（Planck 单位），这个修正极端微小——远在现有实验可探测范围之外。

    2. **修正的引力波色散关系**：
    $$
        \omega^2 = c^2 k^2 + \xi(T_{\text{info}})\,\hbar_{\text{info}}^2 k^4
    $$
    高能引力波的传播速度有轻微的频率依赖——这是信息量子引力的可检验特征。

    3. **宇宙学常数的时间演化**：$\Lambda_{\text{info}}(T_{\text{info}}) \propto T_{\text{info}}^4$ 不是严格常数——它在宇宙时间尺度上微幅漂移。

# 完整图景：从信息 WDW 方程到所有物理


## 信息力学的完整方程体系

我们现在有一个闭合的方程系统：


> **信息力学的完整理论层级：**


    1. **最底层：信息 WDW 方程**（本文）
    $$
        \hat{\mathcal{H}}_{\text{info}}\,\Psi[\gamma] = 0
    $$
    这是信息几何的量子动力学——描述 Fisher 度量作为一种量子自由度如何演化。$\downarrow$ 半经典极限

    2. **信息 Einstein 方程** (`particle information.tex`)
    $$
        R_{ij} - \frac{1}{2}R g_{ij} + \Lambda_{\text{info}} g_{ij} = \frac{8\pi G_{\text{info}}}{c^4} T_{ij}^{\text{info}}
    $$
    $\downarrow$ $T_{\text{info}} \to 0$

    3. **经典爱因斯坦场方程**
    $$
        R_{\mu\nu} - \frac{1}{2}R g_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}
    $$
    $\downarrow$ 弱场、低速

    4. **牛顿引力**
    $$
        \nabla^2 \Phi = 4\pi G\rho
    $$

    5. **同时，信息欧拉-拉格朗日动力学** (`information analysis.tex`)
    $$
        g_{ij}\ddot{\theta}_j + \Gamma_{ijk}\dot{\theta}_j\dot{\theta}_k = -\partial_i[I(X;T) - \beta I(T;Y)]
    $$
    描述物质（信息孤子）如何在上述引力背景上运动。


## 公式之间的逻辑依赖关系

| **层级** | **方程** | **变量** | **来源** |
|---|---|---|---|
| 0. 公理 | DPI $I(X;Z)\leq I(X;Y)$ | 互信息 | 信息论 |
| 1. 量子引力 | 信息 WDW $\hat{\mathcal{H}}_{\text{info}}\Psi = 0$ | Fisher 度量的波泛函 | 本文 |
| 2. 经典引力 | 信息 Einstein 方程 | Fisher 度量 $g_{ij}$ | particle information.tex |
| 3. 物理引力 | Einstein 场方程 | 时空度规 $g_{\mu\nu}$ | GR |
| 4. 物质动力学 | 信息 EL 方程 | 参数 $\theta(\tau)$ | analysis theory.tex |
| 5. 可观测 | 互信息、熵、信道容量 | $I$, $H$, $C$ | Cover \ | Thomas |%
}


**每一层都是上一层在特定极限下的投影**——但信息 WDW 方程是最深的一层，从它可以导出所有其他层。

# 和已知量子引力方法的对比


| **特征** | **圈量子引力 (LQG)** | **弦论** | **信息量子引力（本文）** |
|---|---|---|---|
| 基本变量 | $SU(2)$ 联络 | 弦的振动模 | Fisher 度量的全流形量子化 |
| 背景独立？ | 是 | 作为目标 | 是（无背景分裂） |
| 量子几何来源 | 自旋网络 | Calabi-Yau 紧致化 | 信息内部空间 $K$ |
| 规范群来源 | 手动输入 | 紧致化选择 | $K$ 的 isometry 群的本征态 |
| 发散问题 | LQG 有限的 | 微扰有限 | 无微扰——全流形量子化 |
| 和物质统一 | 手动耦合 | 统一在弦振动中 | 统一在 $\mathcal{L}_{\text{info}}$ 中 |
| 信息论基础 | 无 | 疑似（Maldacena） | 信息论是第一原理 |


**信息量子引力的独特优势**：它不需要额外假设来统一引力和物质——两者都是同一个 $\mathcal{L}_{\text{info}}$ 的不同极限表现。物质是信息势能项的体现（$\mathcal{V}_{\text{info}}$），引力是信息动能项的度规体现（$g_{ij}$）。

# 可验证的预测


**预言（QG1：引力波的修正色散）：**
高能引力波应表现出 $\omega^2 = c^2 k^2 + \xi k^4$ 的修正色散关系，$\xi \propto \hbar_{\text{info}}^2$。这可以通过下一代引力波探测器（Einstein Telescope, Cosmic Explorer）在 $10$--$10^4$ Hz 频段进行检验。


**预言（QG2：Planck 尺度的信息噪底）：**
在 Planck 频率 $\omega_P \sim 1/t_P \sim 10^{43}$ Hz 附近，引力波探测器应看到一个``信息噪底''——来自 Fisher 度量的量子涨落。这个噪底的幅度由 $T_{\text{info}}$ 决定，在当前宇宙中为 $\sim 10^{-60}$ 的应变灵敏度——远在现有探测器之外，但给定了未来的终极灵敏度目标。


**预言（QG3：规范群的量子遗迹）：**
如果规范群 $G_{\text{gauge}}$ 是 $K$ 的 isometry 群的量子本征态，那么在极早期宇宙的遗迹中可能存在*不同规范群的量子隧穿*的痕迹。这可能表现为 CMB 中的非高斯性异常（非标准的 bispectrum 形态），或表现为极微弱的、空间变化的精细结构常数 $\alpha(x)$。


**预言（QG4：信息 Planck 长度的可计算性）：**
信息 Planck 长度 $\ell_{\text{info}}$ 不是一个自由参数——它可以从 Fisher 度量在 $T_{\text{info}}$ 下的量子涨落幅度计算出来：
$$
    \ell_{\text{info}}^2 = \frac{\hbar_{\text{info}} G_{\text{info}}}{c_{\text{info}}^3} \propto \frac{\langle (\delta g)^2 \rangle}{R_{\text{info}}}
$$
这给出了从信息几何的第一原理计算 $\ell_{\text{info}}$ 的可能（如果能独立测量 $\langle (\delta g)^2 \rangle$ 和 $R_{\text{info}}$）。

# 总结


> **核心陈述：**
> 
引力的不可重整化不是量子场论的技术缺陷——
> 
它是 Fisher 度量量子化的范畴错误。
> 
Fisher 度量是信息的二阶导数（KL 散度的 Hessian）。
> 
在它之上做微扰量子化 = 用高阶信息导数去修正低阶信息导数
> 
= 信息结构的无限自指 $\to$ 必然发散。
> 
**正确的方案**：
> 
不分背景和涨落——直接量化整个统计流形 $S = \mathbb{R}^{3,1}\times K$。
> 
**核心方程（信息 Wheeler-DeWitt 方程）**：
> 
$\hat{\mathcal{H_{\text{info}}\Psi[\gamma] = 0$


$\Big[-16\pi G_{\text{info}}\hbar_{\text{info}}^2 G_{ijkl}\frac{\delta}{\delta\gamma_{ij}}\frac{\delta}{\delta\gamma_{kl}} - \frac{\sqrt{\gamma}}{16\pi G_{\text{info}}}({}^dR - 2\Lambda_{\text{info}})\Big]\Psi[\gamma] = 0$


这个方程从 DPI 的对称性（时间重参数化不变性）自然导出。


它的半经典极限（$T_{\text{info}}\to 0$）恢复信息 Einstein 方程 $\to$ 爱因斯坦引力。


信息内部空间 $K$ 的量子化给出离散的规范群本征态——


解释了为什么我们观察到 $SU(3)\times SU(2)\times U(1)$ 而不是其他群。


**引力的量子理论 = 信息流形的量子几何。**


**量子几何不需要发明——它已经被 Fisher 信息编码在统计流形中。**
}}


