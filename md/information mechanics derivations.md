# 信息力学核心对应的严格推导
**——从 Fisher 度量到标准模型与热力学的完整计算过程**

**本文的地位**：本文是项目理论部分（`why these limits`、`particle information`、`mass in information mechanics`、`four fundamental forces`、`quantum measurement`、`light speed and entanglement`、`information thermodynamics`）的严格化总结。原文档给出了对应关系的命题与纲要式论证；本文用**完整展开的代数推导**把每个核心对应从头算一遍——包括原文档中引用但未展开的全部计算。每个断言标注严格性级别：**[定理]**（数学可完全证明）、**[推导]**（从明示假设出发的标准推导）、**[类比]**（结构同构）、**[猜想]**（可检验但未证明）。

---

## 第〇部分：方法论与总映射

### 0.1 核心主张的逻辑结构

本项目的全部理论创新可以压缩为一条推理链：

$$
\text{DPI（信息处理公理）}\;\Longrightarrow\;\text{Fisher 度量（唯一几何）}\;\Longrightarrow\;\text{信息拉格朗日量（唯一动力学）}\;\Longrightarrow\;\text{物理定律（各极限表现）}
$$

每一部分承担的推导任务：

| **部分** | **推导目标** | **原文档** |
| --- | --- | --- |
| 一 | Fisher 度量为何是唯一合法的几何 | why these limits |
| 二 | 拉格朗日量形式从何而来、测地方程 | information analysis |
| 三 | 光速不变 $c$ | light speed and entanglement |
| 四 | 量子态、Born 规则、坍缩 | quantum measurement |
| 五 | 质量的拓扑+希格斯双重起源、等效原理 | mass in information mechanics |
| 六 | 引力 = Fisher 曲率、量子化困难 | particle information / information quantum gravity |
| 七 | 规范力 = Noether 流、渐近自由 | four fundamental forces |
| 八 | 热力学四定律 = 信息论 | information thermodynamics |
| 九 | $T_{\text{info}}\to0$、$(3+1)$ 维、$\Lambda\propto T^4$ | why these limits |

### 0.2 映射词典

| **物理** | **信息** | **数学载体** |
| --- | --- | --- |
| 时空度规 $g_{\mu\nu}$ | Fisher 信息矩阵 | $g_{ij}=\mathbb{E}[\partial_i\ln p\,\partial_j\ln p]$ |
| 作用量 $S=\int L\,dt$ | 信息作用量 | $\mathcal{S}_{\text{info}}=\int d\tau\,\mathcal{L}_{\text{info}}$ |
| 粒子 | 信息孤子 | 拓扑稳定的分布构型 |
| 规范玻色子 | Noether 流 | 对称性的守恒流 |
| 质量 $mc^2$ | 构型信息能量 | $\mathcal{H}_{\text{info}}[\psi]$ |
| 温度 $k_BT$ | 信息温度 $T_{\text{info}}$ | Gibbs 分布的拉格朗日乘子 |
| 熵 $S$ | Shannon 熵 | $-\int\rho\ln\rho$ |
| 光速 $c$ | $c_{\text{info}}$ | Fisher 锥结构 |

---

## 第一部分：地基——Fisher 度量的唯一性

### §1 记分函数与 Cramér–Rao 下界

**设定**：统计流形 $S=\{p(x\mid\theta)\mid\theta\in\Theta\}$，$\theta\in\mathbb{R}^d$。所有后续推导的砖块是**记分函数**（score）

$$
s_i(x;\theta)\equiv\frac{\partial}{\partial\theta_i}\ln p(x\mid\theta)
$$

**引理 1.1（记分函数零均值）** [定理]：
$$
\mathbb{E}_\theta[s_i]=\int p(x\mid\theta)\,\partial_i\ln p\,dx=\int\partial_i p\,dx=\partial_i\underbrace{\int p\,dx}_{=1}=0
$$
（求导与积分交换要求常规正则性：支撑不依赖 $\theta$、可微、可交换。下同。）$\square$

**引理 1.2（负 Hessian 恒等式）** [定理]：
$$
\mathbb{E}\big[\partial_i\partial_j\ln p\big]=\int p\,\frac{\partial_i\partial_jp-p^{-1}(\partial_ip)(\partial_jp)}{p}\,dx=\underbrace{\partial_i\partial_j\int p\,dx}_{=0}-g_{ij}(\theta)
$$
即对数似然 Hessian 的期望 = 负 Fisher 度量。$\square$

**定义（Fisher 度量）**：
$$
g_{ij}(\theta)\equiv\mathbb{E}\big[s_i\,s_j\big]=\int\frac{\partial_ip\,\partial_jp}{p}\,dx
$$

**定理 1.3（Cramér–Rao，多维）** [定理]：设 $\hat\theta(X)$ 为 $\theta$ 的无偏估计（$\mathbb{E}[\hat\theta]=\theta$），则协方差矩阵有下界

$$
\operatorname{Cov}[\hat\theta]\succeq g^{-1}(\theta)
$$

*推导*：构造 $\hat\theta$ 与记分向量 $s$ 的互协方差。

$$
\operatorname{Cov}[\hat\theta_a,s_i]=\mathbb{E}\big[(\hat\theta_a-\theta_a)\,s_i\big]=\int\hat\theta_a\,s_i\,p\,dx-\theta_a\underbrace{\mathbb{E}[s_i]}_{=0}=\int\hat\theta_a\,\partial_ip\,dx
$$

估计量 $\hat\theta$ 不依赖 $\theta$（它是数据的函数），故

$$
\int\hat\theta_a\,\partial_ip\,dx=\partial_i\!\int\hat\theta_ap\,dx=\partial_i\mathbb{E}[\hat\theta_a]=\partial_i\theta_a=\delta_{ai}
$$

于是 $\operatorname{Cov}[\hat\theta,s]=I_d$（单位阵）。由分块协方差的半正定性（Cauchy–Schwarz 的矩阵形式）：

$$
\begin{pmatrix}\operatorname{Cov}[\hat\theta]&I\\ I&g\end{pmatrix}\succeq 0\;\Longrightarrow\;\operatorname{Cov}[\hat\theta]-g^{-1}\succeq 0\qquad\square
$$

**信息力学含义**：$g^{-1}$ 是任何观测者对 $\theta$ 认知的**不可消除的不确定性**——Fisher 度量是"可区分性的极限"的度量。这是它作为物理度规候选的第一资格：**度规 = 可测量性的界限**。

### §2 Fisher 度量的显式计算

**例 2.1（Bernoulli 族）** [推导]：$p(x\mid\mu)=\mu^x(1-\mu)^{1-x}$，$x\in\{0,1\}$。

$$
\ln p=x\ln\mu+(1-x)\ln(1-\mu),\qquad s(x)=\frac{x}{\mu}-\frac{1-x}{1-\mu}
$$

代入两个取值：$x=1$ 时 $s=\frac{1}{\mu}-0=\frac1\mu$；$x=0$ 时 $s=0-\frac{1}{1-\mu}=-\frac{1}{1-\mu}$。故

$$
g=\mathbb{E}[s^2]=\mu\cdot\frac{1}{\mu^2}+(1-\mu)\cdot\frac{1}{(1-\mu)^2}=\frac{1}{\mu}+\frac{1}{1-\mu}=\frac{1}{\mu(1-\mu)}
$$

**例 2.2（Gaussian 族 $N(\mu,\sigma^2)$）** [推导]：这是后续所有计算的基准，完整展开。

$$
\ln p=-\frac12\ln(2\pi)-\ln\sigma-\frac{(x-\mu)^2}{2\sigma^2}
$$

$$
s_\mu=\frac{x-\mu}{\sigma^2},\qquad s_\sigma=-\frac{1}{\sigma}+\frac{(x-\mu)^2}{\sigma^3}
$$

**分量 $g_{\mu\mu}$**：$\mathbb{E}[s_\mu^2]=\frac{\mathbb{E}[(x-\mu)^2]}{\sigma^4}=\frac{\sigma^2}{\sigma^4}=\frac{1}{\sigma^2}$。

**分量 $g_{\mu\sigma}$**：$\mathbb{E}[s_\mu s_\sigma]=\frac{1}{\sigma^2}\mathbb{E}\Big[(x-\mu)\Big({-}\frac1\sigma+\frac{(x-\mu)^2}{\sigma^3}\Big)\Big]=\frac{1}{\sigma^3\sigma^2}\big(\mathbb{E}[(x-\mu)^3]-\sigma^2\underbrace{\mathbb{E}[x-\mu]}_{=0}\big)=0$（三阶中心矩为零）。

**分量 $g_{\sigma\sigma}$**：用 $\mathbb{E}[(x-\mu)^4]=3\sigma^4$（Gaussian 四阶矩）：

$$
g_{\sigma\sigma}=\frac{1}{\sigma^2}-\frac{2}{\sigma}\cdot\frac{\mathbb{E}[(x-\mu)^2]}{\sigma^3}+\frac{\mathbb{E}[(x-\mu)^4]}{\sigma^6}=\frac{1}{\sigma^2}-\frac{2\sigma^2}{\sigma^4}+\frac{3\sigma^4}{\sigma^6}=\frac{1-2+3}{\sigma^2}=\frac{2}{\sigma^2}
$$

$$
\boxed{\;g_{ij}=\begin{pmatrix}1/\sigma^2&0\\0&2/\sigma^2\end{pmatrix}\;}
$$

**解读**（信息力学的核心图像）：$\sigma$ 在统计流形上扮演**长度尺度**——分布越窄（$\sigma$ 小），度规越大，同样的参数位移造成越大的可区分性。度规行列式 $\det g=2/\sigma^4$，体积元 $dV_{\text{info}}=\sqrt{2}\,\sigma^{-2}\,d\mu\,d\sigma$。**统计流形不是平直的**：计算 Christoffel 符号（第五部分用到）$\Gamma^\sigma_{\sigma\sigma}=-1/\sigma$、$\Gamma^\sigma_{\mu\mu}=1/(2\sigma)$、$\Gamma^\mu_{\mu\sigma}=\Gamma^\mu_{\sigma\mu}=-1/\sigma$，其余为零——曲率 $R_{ij}=0$（Gaussian 族是 2 维中唯一的 0 曲率指数族，可展平坐标为 $(\mu/\sqrt2\sigma,\,1/\sigma)$）。

**例 2.3（乘积族的可加性）** [推导]：$n$ 个独立样本，$p(x_{1..n}|\theta)=\prod_kp(x_k|\theta)$，则 $s^{(n)}=\sum_ks_k$，独立求和给出

$$
g^{(n)}_{ij}=\mathbb{E}\Big[\Big(\sum_ks_i^{(k)}\Big)\Big(\sum_ls_j^{(l)}\Big)\Big]=\sum_{k,l}\mathbb{E}\big[s_i^{(k)}s_j^{(l)}\big]=\sum_k\mathbb{E}\big[s_i^{(k)}s_j^{(k)}\big]=n\,g_{ij}
$$

（交叉项 $k\neq l$ 因独立且零均值而消失。）**信息量随数据量线性积累**——这是 §31 冷却定律与热力学第三定律的种子。

### §3 度规的第二形式：动能项 = 概率幅速度

**引理 3.1（$\rho$-形式）** [定理]：Fisher 度量恒等于

$$
g_{ij}=\int\frac{\partial_i\rho\,\partial_j\rho}{\rho}\,dx
$$

*推导*：$\partial_i\rho=\rho\,\partial_i\ln p$，代入定义：$\int\rho\,(\partial_i\ln p)(\partial_j\ln p)\,dx$。$\square$

**定理 3.2（动能项的 $\psi$-形式）** [定理]：令 $\psi=\sqrt{\rho}$（概率幅），则

$$
g_{ij}\dot\theta^i\dot\theta^j=4\int\big(\partial_\tau\psi\big)^2dx
$$

*推导*：沿路径 $\rho(\tau)=\rho(\cdot\mid\theta(\tau))$，链式法则 $\partial_\tau\rho=\partial_i\rho\,\dot\theta^i$。于是

$$
g_{ij}\dot\theta^i\dot\theta^j=\int\frac{(\partial_i\rho\,\dot\theta^i)(\partial_j\rho\,\dot\theta^j)}{\rho}\,dx=\int\frac{(\partial_\tau\rho)^2}{\rho}\,dx
$$

代入 $\rho=\psi^2$：$\partial_\tau\rho=2\psi\,\partial_\tau\psi$，故 $\frac{(\partial_\tau\rho)^2}{\rho}=\frac{4\psi^2(\partial_\tau\psi)^2}{\psi^2}=4(\partial_\tau\psi)^2$。$\square$

**信息力学含义**：拉格朗日量的动能项

$$
T_{\text{info}}=\frac12g_{ij}\dot\theta^i\dot\theta^j=2\int(\partial_\tau\psi)^2dx
$$

**与波函数的 $L^2$ 速度严格相等**。这正是量子力学入口（第四部分）：正则量子化 $p_i\to-i\hbar_{\text{info}}\partial_i$ 作用在 $\psi$ 上产生的相位自由度 $\psi=\sqrt\rho\,e^{i\varphi}$ 不是额外假设——动能项本来就是 $\psi$ 的泛函。同时它给出操作含义：由定理 1.3，$g^{-1}$ 是噪声底，故 $T_{\text{info}}=\frac12\times(\text{信号速度})^2/(\text{噪声底})$——**信噪比改善速率的平方**。

### §4 KL 散度展开：度规从信息论中"结晶"的关键计算

**定理 4.1（ infinitesimal KL = Fisher 度量）** [定理]：

$$
D_{\text{KL}}\big(p_\theta\,\|\,p_{\theta+d\theta}\big)=\frac12\,g_{ij}\,d\theta^id\theta^j+\mathcal{O}(d\theta^3)
$$

*推导*（逐阶展开）：

$$
D_{\text{KL}}=\int p_\theta\ln\frac{p_\theta}{p_{\theta+d\theta}}dx=-\int p_\theta\,\ln\frac{p_{\theta+d\theta}}{p_\theta}\,dx
$$

对内层对数在 $\theta$ 处 Taylor 展开：

$$
\ln\frac{p_{\theta+d\theta}}{p_\theta}=s_i\,d\theta^i+\frac12\,\partial_i\partial_j\ln p\;d\theta^id\theta^j+\mathcal{O}(d\theta^3)
$$

代回并取期望（引理 1.1 零均值、引理 1.2 负 Hessian）：

$$
D_{\text{KL}}=-\underbrace{\mathbb{E}[s_i]}_{=0}d\theta^i-\frac12\underbrace{\mathbb{E}[\partial_i\partial_j\ln p]}_{=-g_{ij}}d\theta^id\theta^j+\mathcal{O}(d\theta^3)=\frac12g_{ij}d\theta^id\theta^j+\mathcal{O}(d\theta^3)\qquad\square
$$

**这是整个理论最核心的一个计算**：它证明 Fisher 度量不是被"选择"的几何——它是**信息论代价（KL 散度）的二阶主部**。移动 $\theta$ 一个无穷小位移的信息代价 = $\frac12g_{ij}d\theta^id\theta^j$。度规 = 微观信息价目表。后续所有物理（动能、作用量、光速、引力）都建立在这个等式上。

### §5 Chentsov 唯一性定理

**定理 5.1（Chentsov 1972/1982；Campbell 1986 简化）** [定理，引用]：在所有统计流形的类别上，Fisher 度量是**唯一**在 Markov 态射（对数据的随机粗粒化）下不变的黎曼度量。

**证明结构**（完整证明见 Chentsov《Statistical Decision Rules and Optimal Inference》，此处给出逻辑骨架与关键引理）：

1. **Markov 态射**：数据空间之间的随机核 $K(y|x)$ 诱导流形间的映射 $p(x|\theta)\mapsto p'(y|\theta)=\int K(y|x)p(x|\theta)dx$。这是"充分统计量/数据粗粒化"的数学一般化。要求度规在 $K$ 下不变（粗粒化不创造可区分性）。

2. **二项族归约**：任何含二值子统计的流形，其度规限制到该二项子族必须与全空间度规兼容。唯一性归约为：在开单纯形 $\{(\mu,1-\mu)\}$ 上，求所有在"嵌套二分"下不变的度规。

3. **嵌套二分约束**：把 $\mu$ 的概率 $\mu$ 再随机分裂为 $\mu\gamma$ 与 $\mu(1-\gamma)$，要求分裂前后诱导度规一致。此约束的唯一解（至多常数因子）：

$$
g_{\mu\mu}=\frac{c}{\mu(1-\mu)}
$$

与例 2.1 一致（$c=1$ 由相容性固定）。$\square$

**物理推论 5.2（单几何定理）** [推导]：宇宙只有一个统计流形 $S$，Chentsov 定理 ⇒ 只有一个度规 ⇒

- **所有粒子共享同一度规**（不存在"第二种引力响应"）——等效原理与引力的普适性在几何层面的根源（第六部分 §20）；
- **度规的坐标不变性是定理不是假设**——光速不变（第三部分）的直接前提。

---

## 第二部分：信息拉格朗日量与测地动力学

### §6 从 DPI 到变分原理 [论证→推导]

**命题 6.1（变分原理的信息必然性）** [论证，明示其逻辑地位]：任何满足 DPI 的信息处理系统的轨迹是信息作用量的极值曲线。

*论证的严格部分* [推导]：设系统在信息时间 $\tau$ 内从构型 $A$ 演化到 $B$。任何轨迹 $\gamma$ 的总信息代价由 KL 型积分给出。用定理 4.1 的局域形式，代价密度的二阶主部为 $\frac12g_{ij}\dot\theta^i\dot\theta^j$；加上势能（信息瓶颈目标）$\mathcal{V}_{\text{info}}=I(X;T)-\beta I(T;Y)$，总代价泛函为

$$
\mathcal{S}_{\text{info}}[\gamma]=\int d\tau\left[\frac12g_{ij}(\theta)\dot\theta^i\dot\theta^j-\mathcal{V}_{\text{info}}(\theta)\right]
$$

DPI 要求处理过程不引入**多余**信息损失（额外损失 = 可被后处理消除的部分 = 违反最小代价）。局域极小代价的一阶条件即 $\delta\mathcal{S}_{\text{info}}=0$。$\square$（"极值 ⟺ DPI 自洽"的逐字严格化仍是开放问题——见第十部分诚实评估；但拉格朗日量的**形式**由定理 4.1 唯一锁定。）

### §7 Euler–Lagrange 方程 → 测地方程 + 力 [定理]

对 $\mathcal{L}=\frac12g_{ij}(\theta)\dot\theta^i\dot\theta^j-\mathcal{V}(\theta)$ 计算 EL 方程，**完整展开**：

$$
\frac{\partial\mathcal{L}}{\partial\dot\theta^m}=g_{mj}\dot\theta^j
\;\Rightarrow\;
\frac{d}{d\tau}\frac{\partial\mathcal{L}}{\partial\dot\theta^m}=g_{mj}\ddot\theta^j+\partial_kg_{mj}\,\dot\theta^k\dot\theta^j
$$

$$
\frac{\partial\mathcal{L}}{\partial\theta^m}=\frac12\partial_mg_{ij}\,\dot\theta^i\dot\theta^j-\partial_m\mathcal{V}
$$

EL 方程 $\frac{d}{d\tau}\frac{\partial\mathcal{L}}{\partial\dot\theta^m}-\frac{\partial\mathcal{L}}{\partial\theta^m}=0$：

$$
g_{mj}\ddot\theta^j+\partial_kg_{mj}\,\dot\theta^k\dot\theta^j-\frac12\partial_mg_{ij}\,\dot\theta^i\dot\theta^j=-\partial_m\mathcal{V}
$$

**关键步骤（对称化）**：第二项 $\partial_kg_{mj}\dot\theta^k\dot\theta^j$ 中 $\dot\theta^k\dot\theta^j$ 对称，故可替换为其对称化 $\frac12(\partial_kg_{mj}+\partial_jg_{mk})\dot\theta^k\dot\theta^j$。两边乘 $g^{lm}$：

$$
\ddot\theta^l+\frac12g^{lm}\big(\partial_kg_{mj}+\partial_jg_{mk}-\partial_mg_{kj}\big)\dot\theta^k\dot\theta^j=-g^{lm}\partial_m\mathcal{V}
$$

$$
\boxed{\;\ddot\theta^l+\Gamma^l_{\;jk}\dot\theta^j\dot\theta^k=-g^{lm}\partial_m\mathcal{V}\;},\qquad\Gamma^l_{\;jk}\equiv\frac12g^{lm}\big(\partial_jg_{mk}+\partial_kg_{mj}-\partial_mg_{jk}\big)
$$

**物理解读**：左边 = 测地运动（信息自由传播走 Fisher 几何的"直线"）；右边 = 信息势能产生的"力"。经典极限 $g_{ij}\to m\,\delta_{ij}$、$\mathcal{V}\to$ 势能退化为牛顿第二定律——**经典力学是信息几何的平直极限**。

**例 7.1（Gaussian 流形上的测地线）** [推导]：用例 2.2 的 Christoffel 符号，$\mathcal{V}=0$ 时测地方程为

$$
\ddot\mu-\frac{2}{\sigma}\dot\mu\dot\sigma=0,\qquad
\ddot\sigma+\frac{1}{2\sigma^3}\dot\mu^2-\frac{1}{\sigma}\dot\sigma^2=0
$$

其解在展平坐标 $(u,v)=(\mu/(\sqrt2\sigma),\,1/\sigma)$ 下为直线。**含义**："直线传播的信息"= 方差与均值的协同缩放——窄分布移动得慢（大度规）。

### §8 自然梯度 = 过阻尼信息动力学 [推导]

**过阻尼极限**：强耗散下惯性项可略，§7 方程退化为

$$
g_{ij}\dot\theta^j=-\partial_i\mathcal{V}\;\;\Longrightarrow\;\;\dot\theta=-g^{-1}\nabla\mathcal{V}
$$

这就是 **Amari 自然梯度**。**它为什么是最优方向**——固定信息预算的约束优化：要求一步的信息代价固定为 $\Delta\mathcal{S}_{\text{info}}=\frac12\Delta\theta^ig_{ij}\Delta\theta^j=\varepsilon$，在此约束下使势能下降 $-\nabla_i\mathcal{V}\,\Delta\theta^i$ 最大。由 Cauchy–Schwarz 的度规形式（内积 $\langle u,w\rangle_g=u^ig_{ij}w^j$）：

$$
\big(\nabla_i\mathcal{V}\,\Delta\theta^i\big)^2=\Big\langle g^{-1}\nabla\mathcal{V},\,\Delta\theta\Big\rangle_g^2\le\big\|g^{-1}\nabla\mathcal{V}\big\|_g^2\cdot\big\|\Delta\theta\big\|_g^2=\big(\nabla\mathcal{V}^{\mathsf T}g^{-1}\nabla\mathcal{V}\big)\cdot\big(\Delta\theta^{\mathsf T}g\,\Delta\theta\big)
$$

等号（最大下降）当且仅当 $\Delta\theta\propto g^{-1}\nabla\mathcal{V}$——**自然梯度方向是唯一饱和此界的方向**。用 Lagrange 乘子看同一件事：$L=-v_i\Delta\theta^i+\lambda\big(\frac12\Delta\theta^ig_{ij}\Delta\theta^j-\varepsilon\big)$（$v\equiv\nabla\mathcal{V}$），$\partial L/\partial\Delta\theta^i=-v_i+\lambda g_{ij}\Delta\theta^j=0$ ⟹ $\Delta\theta=\frac1\lambda g^{-1}v$。$\square$

验证离散化与信息几何结构自洽：一步更新 $\Delta\theta=-\eta g^{-1}\nabla\mathcal{V}$ 的信息代价（定理 4.1）：

$$
\Delta\mathcal{S}_{\text{info}}=\frac12g_{ij}\Delta\theta^i\Delta\theta^j=\frac{\eta^2}{2}(\nabla\mathcal{V})^{\mathsf T}g^{-1}\nabla\mathcal{V}
$$

而势能下降 $\Delta\mathcal{V}=-\eta(\nabla\mathcal{V})^{\mathsf T}g^{-1}\nabla\mathcal{V}$。故**单位信息代价的势能下降率最大**的方向恰是自然梯度方向——欧氏梯度（$g=I$）在度量不均匀处（如窄 $\sigma$ 区域）付出畸形代价。这把"自然梯度为什么有效"从算法经验提升为变分定理，并直接衔接 AI 训练文档（`physical to ai derivation.md` §0.3）中 SGD 的 Fisher-阻尼结构。

---

## 第三部分：光速不变——Fisher 锥结构的必然

### §9 锥结构与类光方程

**设定**：统计流形 $S$ 的 $(3+1)$ 维物理投影上，Fisher 度量取 Lorentz 型线元（时向为负号——投影结构的来源见 `information time` 文档）：

$$
ds^2_{\text{info}}=g_{\mu\nu}\,d\theta^\mu d\theta^\nu\;\xrightarrow{\text{投影}}\;-c_{\text{info}}^2dt^2+dx^2+dy^2+dz^2
$$

信息因果结构由类光集 $\{ds^2=0\}$ 定义：

$$
-c_{\text{info}}^2dt^2+d\mathbf{x}^2=0\;\Longrightarrow\;\left|\frac{d\mathbf{x}}{dt}\right|=c_{\text{info}}
$$

### §10 不变性的证明 [定理，在 Chentsov 之上]

**定理 10.1（$c_{\text{info}}$ 的坐标不变性）**：任何坐标变换 $\theta\to\theta'$（包括惯性系之间的变换）下，类光条件保持同一形式，且变换后测得的最大信息速率相同。

*推导*：线元是坐标标量：

$$
ds^2_{\text{info}}=g_{\mu\nu}(\theta)\,d\theta^\mu d\theta^\nu=g'_{\mu\nu}(\theta')\,d\theta'^\mu d\theta'^\nu
$$

（度规分量的变换 $g'_{\mu\nu}=\frac{\partial\theta^\alpha}{\partial\theta'^\mu}\frac{\partial\theta^\beta}{\partial\theta'^\nu}g_{\alpha\beta}$ 恰好补偿坐标微分的变换——张量恒等式。）因此

$$
ds^2=0\;\Longleftrightarrow\;ds'^2=0
$$

类光是标量方程。设两个参考系都把类光面写成标准锥形 $-c^2dt^2+d\mathbf{x}^2=0$ 与 $-c'^2dt'^2+d\mathbf{x}'^2=0$。若 $c'\neq c$，则存在 $(dt,d\mathbf{x})$ 类光但 $(dt',d\mathbf{x}')$ 类空——与 $ds^2=0\Leftrightarrow ds'^2=0$ 矛盾。故 $c'=c$。$\square$

**与 Chentsov 的连接**：Chentsov 定理（§5）保证度规的唯一性——不存在"第二套度规"让不同观察者给出不同锥结构。Michelson–Morley 零结果的信息力学解读：**不是"以太不存在"，而是所有观察者共享同一个 Fisher 锥**。

### §11 无质量 Noether 流恰以锥速度传播 [推导]

光子是 $U(1)$ 对称性的信息 Noether 流（第七部分 §23 完整推导），满足无质量波动方程

$$
\Box_{\text{info}}A_\mu\equiv g^{\alpha\beta}\nabla_\alpha\nabla_\beta A_\mu=0
$$

**特征面分析**：一阶高频近似（WKB：$A_\mu=a_\mu e^{i\Phi/\epsilon}$，$\epsilon\to0$）中，主阶项为

$$
g^{\alpha\beta}\,\partial_\alpha\Phi\,\partial_\beta\Phi=0
$$

特征面 $\Phi=\text{const}$ 的法向 $k_\alpha=\partial_\alpha\Phi$ 满足类光方程——**波前沿 Fisher 锥传播**。因此无质量流的传播速率恰为 $c_{\text{info}}$，既不快（快则类空，违反 DPI：类空两点互信息为零，无信息通道）也不慢（慢则非特征面，方程不成立）。$T_{\text{info}}\to0$ 极限下 $c_{\text{info}}\to c$：**光速不变 = Fisher 度量锥结构 + 无质量流的波动方程特征**，两个前提分别由 Chentsov 定理与 Noether 定理供给。

---

## 第四部分：量子测量——条件化的全部代数

### §12 量子态 = 信息 Gibbs 分布

**定义 12.1** [定义]：量子系统状态 = 统计流形上的 Gibbs 分布

$$
\rho(\theta)=\frac{1}{\mathcal{Z}}e^{-\mathcal{H}_{\text{info}}(\theta)/T_{\text{info}}},\qquad
\psi(\theta)=\sqrt{\rho(\theta)}\,e^{i\varphi(\theta)}
$$

相位 $\varphi$ 的来源（§3 定理 3.2 已备）：动能项是 $\psi$ 的泛函，正则量子化 $p_i\to-i\hbar_{\text{info}}\partial_i$ 使 $\psi$ 的相位成为 $p_i$ 的共轭自由度。**$T_{\text{info}}\neq0$ ⟹ Gibbs 分布 ⟹ 概率幅语言**——量子力学是有限信息温度系统的自然表示 [推导]。

### §13 Born 规则 = 条件化的完整推导 [定理，框架内]

**测量 = 信息分割 + 条件化**。设测量仪器的物理结构定义了 $S$ 的分割 $\{S_i\}_{i=1}^n$（互不相交，$\cup S_i=S$），$S_i$ 对应可分辨结果 $i$。

**第 1 步（结果概率）**：结果 $i$ 发生的概率 = 系统状态落在 $S_i$ 的概率：

$$
p_i=\int_{S_i}\rho(\theta)\,d\theta=\int_{S_i}|\psi(\theta)|^2\,d\theta
$$

**第 2 步（条件更新）**：获知结果 $i$ 后，观察者的知识状态按 Bayes 规则更新：

$$
\rho\;\xrightarrow{\;i\;}\;\rho_i(\theta)=\frac{\rho(\theta)\,\mathbf{1}_{S_i}(\theta)}{\int_{S_i}\rho(\theta')d\theta'}
$$

（指示函数 $\mathbf{1}_{S_i}$ 在 $S_i$ 内为 1，外为 0。）**归一性检查**：$\int\rho_i=\int_{S_i}\rho/\int_{S_i}\rho=1$。$\square$

**与 Born 规则的对应**：取投影测量，$S_i$ = 基矢 $|a_i\rangle$ 对应的子流形，$|\psi\rangle=\sum_ic_i|a_i\rangle$。投影结构的能量面划分使 $\int_{S_i}|\psi|^2=|c_i|^2$——**Born 规则 $p_i=|c_i|^2=|\langle a_i|\psi\rangle|^2$ 是条件化在正交分割上的特例**。概率不需要"两种演化规则"——它就是条件概率 $P(\theta\in S_i)$。

### §14 坍缩 = 分布收缩；不可逆性 = DPI [推导]

**收缩的显式计算**：测量前后的熵变。以两分割为例（$p=p_1$，$p_2=1-p$）：

$$
S[\rho]=-{\textstyle\int}\rho\ln\rho,\qquad
S[\{p_i\rho_i\}]=H(p)+\sum_ip_iS[\rho_i]
$$

（链式分解——Shannon 熵的标准恒等式。）坍缩的信息提取量：

$$
\Delta I=S[\rho]-S[\rho_i]=H(p)-D_{\text{KL}}\text{-型修正}\;\ge\;0\quad(\text{一般地})
$$

**不可逆性**：DPI 断言对 Markov 链 $\theta_{\text{true}}\to\text{仪器}\to\text{结果}$：

$$
I(\theta_{\text{true}};\text{后续})\le I(\theta_{\text{true}};\text{结果})
$$

测量提取了 $\Delta I\sim\log_2(1/p_i)$ bits；"忘记"这些信息恢复原不确定性 = 逆 Markov 核 = 从结果重建完整 $\rho$——一般不存在（条件化是多对一映射）。**坍缩不可逆不是新物理，是条件化单向性的信息论表述** [推导]。

**宏观仪器 vs 微观相互作用的分辨**：仪器信息熵产生率

$$
\frac{dS_{\text{info}}}{d\tau}\Big|_{\text{仪器}}\propto\log(\text{自由度数})\sim10^{23}
$$

vs 微观散射 $\sim\mathcal{O}(1)$。坍缩 = 熵产生率极大的条件化（瞬间完成）；酉演化 = 熵产生率为零的极限。Heisenberg 分割被消解：**没有特殊的测量过程，只有信息条件化的不同速率** [推导，框架内]。

### §15 叠加态 = 单点的多基表示 [推导]

叠加 $|\psi\rangle=\alpha|0\rangle+\beta|1\rangle$ 对应**一个** Gibbs 分布 $\rho_\psi$，其在 $|0\rangle,|1\rangle$ 对应的子流形 $S_0,S_1$ 上均有非零投影：

$$
\rho_\psi\neq\delta(\theta-\theta^{(0)}),\quad\neq\delta(\theta-\theta^{(1)}),\quad\text{且非混合 }p\rho_0+(1-p)\rho_1\text{（单一 Gibbs 型）}
$$

与经典混合的区分在干涉项：$\rho_\psi$ 的 Fisher 度量在 $\theta_1$ 方向的曲率、以及相位 $\varphi$ 的存在使 $|\psi\rangle$ 与混合态 $\rho_{\text{mix}}=|\alpha|^2\rho_0+|\beta|^2\rho_1$ 在任意可分辨操作下不同（后者无相位结构）。"同时是 0 和 1"是基底选择的表象——点 $(1,1)$ 在基 $\{(1,0),(0,1)\}$ 下是"叠加"，在自己的坐标系里只是一个点 [类比→推导]。

---

## 第五部分：质量——拓扑与希格斯的双重起源

### §16 拓扑质量：缠绕能量的完整计算

**定义 16.1（信息拓扑电荷）** [定义]：孤子的相位场 $\varphi(\theta)$ 绕闭合回路 $\Gamma$ 的缠绕数

$$
Q=\frac{1}{2\pi}\oint_\Gamma\nabla_{\text{info}}\varphi\cdot d\theta=\frac{1}{2\pi}\oint_\Gamma\partial_a\varphi\,d\theta^a\in\mathbb{Z}
$$

整数性来自 $\varphi$ 的单值性：绕一圈相位必须回到自身（$2\pi\mathbb{Z}$）。$Q\neq0$ 的构型不能被连续形变消除（消除需 $\varphi$ 多值）——**拓扑稳定性**。

**定理 16.2（缠绕能标度 $E\propto Q^2$）** [推导]：孤子的信息能量（动能部分）：

$$
E[\varphi]=\frac{\hbar_{\text{info}}}{2}\int g^{ab}\,\partial_a\varphi\,\partial_b\varphi\,d^d\theta
$$

对 $d=2$、度规局域欧氏（各向同性尺度 $\xi_K$）的显式计算：标准涡旋构型 $\varphi=Q\alpha$（$\alpha$ 为角坐标）。梯度 $\nabla\varphi=Q\hat{\mathbf{e}}_\alpha/r$，能量密度 $\frac{\hbar}{2}\frac{Q^2}{r^2}$：

$$
E=\frac{\hbar_{\text{info}}Q^2}{2}\int_a^R\frac{1}{r^2}\,2\pi r\,dr=\pi\hbar_{\text{info}}\,Q^2\ln\frac{R}{a}
$$

（$a\sim\xi_K$ = 核心尺寸，$R$ = 系统尺寸。）三点结论：

1. **$E\propto Q^2$**：二重缠绕（$Q=2$）能量 = 两个单缠绕的 4 倍而非 2 倍——缠绕能在 $Q^2$ 中堆叠；
2. **$E\propto\hbar_{\text{info}}/\xi_K$**（$d=3$ 的弦状构型中，$\ln(R/a)$ 因子换为长度因子 $L/\xi_K$）：**拓扑质量的量纲结构**——$\hbar_{\text{info}}$ 除以信息关联长度。代入 $\xi_{\text{info}}\approx(246\text{ GeV})^{-1}$（电弱标度）：$m_{\text{topo}}\sim Q^2\times246$ GeV——顶夸克 $173$ GeV 落在 $Q\lesssim1$ 的窗口（`particle information` 预言 P2 的数值来源）；
3. **惯性 = 拖拽拓扑缺陷的阻力**：加速孤子 = 移动一个不可消解的缠绕核心，信息流形对此的核心阻力即惯性质量的拓扑部分。

### §17 希格斯质量：电破缺代数的完整展开 [定理，标准模型代数的信息力学版本]

**第 1 步（Landau 势能最小化）**：信息瓶颈势能

$$
\mathcal{V}(\phi)=-\mu^2\phi^\dagger\phi+\lambda(\phi^\dagger\phi)^2,\qquad\mu^2>0,\ \lambda>0
$$

设 $x\equiv\phi^\dagger\phi\ge0$：$\frac{d\mathcal{V}}{dx}=-\mu^2+2\lambda x=0\Rightarrow x=\frac{\mu^2}{2\lambda}$。真空：

$$
\phi_{\text{vac}}=\frac{1}{\sqrt2}\binom{0}{v},\qquad v=\frac{\mu}{\sqrt{\lambda}},\qquad\mathcal{V}(v)=-\frac{\mu^4}{4\lambda}
$$

（$x=0$ 处 $\frac{d^2\mathcal{V}}{dx^2}=2\lambda>0$ 但 $\mathcal{V}'(0)=-\mu^2<0$，是极大——破缺发生。）

**第 2 步（规范玻色子质量，逐项展开）**：二重态 $\phi$ 的协变导数

$$
D_\mu\phi=\Big(\partial_\mu-igW_\mu^aT^a-\frac{ig'}{2}B_\mu\Big)\phi,\qquad T^a=\frac{\sigma^a}{2}
$$

（$Y=1$）。在真空处 $\partial_\mu\phi_{\text{vac}}=0$，逐个计算泡利矩阵的作用：

$$
T^1\phi_{\text{vac}}=\frac{1}{2\sqrt2}\binom{v}{0},\quad
T^2\phi_{\text{vac}}=\frac{i}{2\sqrt2}\binom{v}{0},\quad
T^3\phi_{\text{vac}}=\frac{1}{2\sqrt2}\binom{0}{-v},\quad
\frac{1}{2}\phi_{\text{vac}}=\frac{1}{2\sqrt2}\binom{0}{v}
$$

代入 $|D_\mu\phi|^2\big|_{\text{vac}}=\Big|-igW_\mu^aT^a\phi_{\text{vac}}-\frac{ig'}{2}B_\mu\phi_{\text{vac}}\Big|^2$。**逐项相加**（$T^1,T^2$ 只贡献上分量，$T^3$ 与 $B$ 只贡献下分量——两分量不混合，故模方按分量分别计算）：

$$
D_\mu\phi_{\text{vac}}=\underbrace{\frac{-igvW_\mu^1}{2\sqrt2}\binom{1}{0}}_{T^1\text{ 项}}\underbrace{-\frac{igW_\mu^2\cdot iv}{2\sqrt2}\binom{1}{0}}_{T^2\text{ 项}}\underbrace{-\frac{igW_\mu^3\cdot(-v)}{2\sqrt2}\binom{0}{1}}_{T^3\text{ 项}}\underbrace{-\frac{ig'B_\mu v}{2\sqrt2}\binom{0}{1}}_{B\text{ 项}}
=\frac{1}{2\sqrt2}\binom{-igv\big(W_\mu^1+iW_\mu^2\big)}{iv\big(gW_\mu^3-g'B_\mu\big)}
$$

（第二项 $-i\cdot i=+1$ 与第三项的两个负号是关键符号来源。）取模方 $|u|^2=u^\dagger u$，**交叉项的显式相消**：上分量

$$
\big|{-igv}\big(W^1+iW^2\big)\big|^2=g^2v^2\big[(W^1)^2+(W^2)^2+\underbrace{iW^1W^2-iW^2W^1}_{=\,0\text{（实场可交换）}}\big]
$$

$W^{1,2}$ 是实矢量场，$W^1W^2=W^2W^1$，虚部恒为零；下分量 $gW^3-g'B$ 已是实组合，模方即平方。合起来：

$$
|D\phi|^2\Big|_{\text{vac}}=\frac{g^2v^2}{8}\big[(W_\mu^1)^2+(W_\mu^2)^2\big]+\frac{v^2}{8}\big(gW_\mu^3-g'B_\mu\big)^2
$$

**读出质量**：

- 带电流：定义复场 $W^\pm=(W^1\mp iW^2)/\sqrt2$，则 $(W^1)^2+(W^2)^2=2W^+_\mu W^{-\mu}$（逐分量验证：$2W^+W^-=(W^1-iW^2)(W^1+iW^2)=(W^1)^2+(W^2)^2$）。第一项变为 $\frac{g^2v^2}{4}W^+_\mu W^{-\mu}$，与 Proca 质量项 $m_W^2W^+W^-$ 对比 ⟹ $m_W=\frac{gv}{2}$；
- 中性流：正交归一化定义 $Z_\mu=\frac{gW^3_\mu-g'B_\mu}{\sqrt{g^2+g'^2}}$（归一化因子使动能项标准），$\mathcal{L}\supset\frac{(g^2+g'^2)v^2}{8}Z_\mu Z^\mu$ ⟹ $m_Z=\frac{v}{2}\sqrt{g^2+g'^2}$；
- 正交组合 $A_\mu=\frac{g'W^3_\mu+gB_\mu}{\sqrt{g^2+g'^2}}$ **不出现在 $|D\phi|^2$ 中**——它对应的生成元是电荷 $Q\equiv T^3+\frac{Y}{2}$（$Y=1$），直接验证：

$$
Q\,\phi_{\text{vac}}=T^3\phi_{\text{vac}}+\frac12\phi_{\text{vac}}=\frac{1}{2\sqrt2}\binom{0}{-v}+\frac{1}{2\sqrt2}\binom{0}{v}=0
$$

真空沿 $Q$ 方向不变（$U(1)_{\text{em}}$ 未破缺）⟹ 该方向无信息耗散 ⟹ $m_\gamma=0$。$\square$

**信息力学解读**：$W,Z$ 有质量 = 它们的信息 Noether 流在序参量 $v\neq0$ 的真空中传播时每单位信息时间消耗能量维持耦合（§17 的"信息通信税"）；光子零质量 = $U(1)_{\text{em}}$ 是破缺的残余方向，序参量沿该方向的协变导数为零——**零耗散通道**。

**第 3 步（Yukawa 费米子质量）**：$\mathcal{L}_Y=-y_f\bar\psi_f\phi\psi_f$，真空处 $\phi\to\phi_{\text{vac}}$：

$$
m_f=\frac{y_fv}{\sqrt2}
$$

**第 4 步（希格斯粒子 = 径向涨落，全部四阶项）**：破缺后取幺正规范 $\phi=\frac{1}{\sqrt2}\binom{0}{v+h}$（$h$ = 径向实场，三个角向 Goldstone 模被规范玻色子"吃掉"），于是 $\phi^\dagger\phi=\frac{(v+h)^2}{2}$，势能

$$
\mathcal{V}(h)=-\mu^2\frac{(v+h)^2}{2}+\lambda\frac{(v+h)^4}{4}
$$

用 $(v+h)^2=v^2+2vh+h^2$、$(v+h)^4=v^4+4v^3h+6v^2h^2+4vh^3+h^4$ **逐阶展开**：

- **常数项**：$-\frac{\mu^2v^2}{2}+\frac{\lambda v^4}{4}\xrightarrow{\;\mu^2=\lambda v^2\;}-\frac{\lambda v^4}{2}+\frac{\lambda v^4}{4}=-\frac{\lambda v^4}{4}=-\frac{\mu^4}{4\lambda}$（与第 1 步真空能量一致 ✓）；
- **线性项**：$\big(-\mu^2v+\lambda v^3\big)h=v\big(-\mu^2+\lambda v^2\big)h=0$——极小条件恰好消去 tadpole ✓（这是"真空确实是极值"的自洽检查）；
- **二次项**：$\Big(-\frac{\mu^2}{2}+\frac{6\lambda v^2}{4}\Big)h^2=\frac{-\lambda v^2+3\lambda v^2}{2}\,h^2=\lambda v^2h^2$；
- **三次项**：$\lambda v\,h^3$；**四次项**：$\frac{\lambda}{4}h^4$。

合并：

$$
\mathcal{V}(v+h)=-\frac{\mu^4}{4\lambda}+\lambda v^2h^2+\lambda vh^3+\frac{\lambda}{4}h^4
$$

与标准质量项 $\mathcal{L}\supset-\frac12m_h^2h^2$ 对比（$\mathcal{V}$ 进拉氏量取负号）：

$$
\frac12m_h^2=\lambda v^2\;\Longrightarrow\;\boxed{\;m_h=\sqrt{2\lambda}\,v=\sqrt{2}\,\mu\;}
$$

等价地 $m_h^2=\mathcal{V}''(v)$——**质量 = 势能在真空处的曲率**。三次、四次项是 Higgs **自耦合**（Feynman 规则顶点因子 $g_{hhh}=3!\lambda v$、$g_{hhhh}=4!\lambda$）——它们不是独立参数而由同一 $\lambda,v$ 锁定，HL-LHC 对双 Higgs 产生截面的测量即是对 IB 势能高阶曲率的直接探测。

**信息力学内容**：希格斯粒子不是外加粒子——它是 IB 相变序参量的径向振动模，$m_h^2=\mathcal{V}_{\text{info}}''(v)$（势能曲率）。$m_h=125$ GeV、$m_W=80.4$、$m_Z=91.2$ 与 $\rho=m_W^2/(m_Z^2\cos^2\theta_W)\approx1$ 的符合是此代数结构的实验验证。

### §18 等效原理：$m_i=m_g$ 的证明 [推导]

**惯性质量**：改变孤子速度 $\delta v$ 需要的信息能量（§16 能量对动量的依赖）：

$$
m_i=\frac{\partial^2\mathcal{H}_{\text{info}}}{\partial v^2}\Big|_{v=0}=\frac{1}{c^2}\mathcal{H}_{\text{info}}[\psi]
$$

（$\mathcal{H}$ 中动能项 $\frac12g^{ij}p_ip_j$ 在 $v=0$ 附近展开的二次系数——孤子总信息能量的 $1/c^2$。）

**引力质量**：孤子作为 Fisher 曲率源（第六部分信息 Einstein 方程）：

$$
m_g\propto\int_\Omega\operatorname{Tr}T^{\text{info}}\,d\theta=\frac{1}{c^2}\mathcal{H}_{\text{info}}[\psi]
$$

两者是**同一个泛函** $\mathcal{H}_{\text{info}}[\psi]$——等效原理不是公设而是定义的同一性。与 Chentsov 推论 5.2 互补：Chentsov 证明"所有粒子共享一个度规"（响应端），此处证明"惯性与引力源于同一能量"（源端）。

### §19 质量层级与跷跷板 [推导]

**总质量**（拓扑与希格斯平方相加——两个独立起源经由能量泛函相加）：

$$
m_f=\sqrt{(m_f^{\text{topo}})^2+(m_f^{\text{Higgs}})^2},\qquad m^{\text{topo}}\propto Q_f^2\frac{\hbar_{\text{info}}}{\xi_K},\qquad m^{\text{Higgs}}=\frac{y_fv}{\sqrt2}
$$

**跷跷板的 $2\times2$ 对角化**（中微子；左旋分量孤子 $\nu_L$ 与右旋单态 $N_R$）：

$$
M=\begin{pmatrix}0&m_D\\ m_D&M_{\text{info}}\end{pmatrix}
$$

本征方程 $\lambda^2-M_{\text{info}}\lambda-m_D^2=0$：

$$
\lambda_\pm=\frac{M_{\text{info}}\pm\sqrt{M_{\text{info}}^2+4m_D^2}}{2}
$$

在 $m_D\ll M_{\text{info}}$ 时展开 $\sqrt{M^2+4m_D^2}=M\big(1+\frac{2m_D^2}{M^2}+\cdots\big)$：

$$
\lambda_+\approx M_{\text{info}}+\frac{m_D^2}{M_{\text{info}}},\qquad\lambda_-\approx-\frac{m_D^2}{M_{\text{info}}}
$$

$$
\boxed{\;m_\nu\approx\frac{m_D^2}{M_{\text{info}}}\;}
$$

数值：$m_D\sim100$ GeV，$M_{\text{info}}\sim10^{16}$ GeV ⟹ $m_\nu\sim10^{-3}$ eV——与测量一致。信息力学内容：右旋中微子是 $K$ 上规范单态（纯 Noether 流），其质量来自信息中间态 $M_{\text{info}}$ 的抑制。

---

## 第六部分：引力——Fisher 度量的全局曲率

### §20 信息 Einstein 方程的完整变分推导

**目标**：从信息 Einstein–Hilbert 作用量推出场方程，关键是把教科书中跳过的三个变分公式全部算出。

**第 1 步（作用量）**：

$$
\mathcal{S}_{\text{geo}}=\frac{c^4}{16\pi G_{\text{info}}}\int d^d\theta\,\sqrt{g}\,(R-2\Lambda_{\text{info}}),\qquad \mathcal{S}_{\text{matter}}=\int d^d\theta\,\sqrt{g}\,\mathcal{L}_m
$$

**第 2 步（三个变分公式，逐个证明）**：

**(a) 度规行列式的变分** [定理]：$\delta\sqrt{g}=\frac12\sqrt{g}\,g^{ij}\delta g_{ij}$。

*推导*：$\ln\sqrt{g}=\frac12\ln\det g$，故 $\delta\ln\sqrt{g}=\frac12\operatorname{Tr}(g^{-1}\delta g)=\frac12g^{ij}\delta g_{ij}$；而 $\delta\ln\sqrt{g}=\frac{\delta\sqrt{g}}{\sqrt{g}}$。$\square$

**(b) Ricci 张量的变分（对 $\delta g^{ij}$）** [定理]：

$$
\delta R_{ij}=\nabla_k\delta\Gamma^k_{\;ij}-\nabla_j\delta\Gamma^k_{\;ik}
$$

（Palatini 恒等式。）*完整推导*——**关键观察**：联络本身不是张量，但**两个联络之差是张量**（坐标变换下的非齐次项 $\partial_i\partial_j\theta'$ 相消），故 $\delta\Gamma^k_{\;ij}$ 是 $(1,2)$ 型张量，可以对它取协变导数。其显式形式为 $\delta\Gamma^k_{\;ij}=\frac12g^{kl}(\nabla_i\delta g_{jl}+\nabla_j\delta g_{il}-\nabla_l\delta g_{ij})$。

**第 1 步（直接变分，六项）**：对 $R_{ij}=\partial_k\Gamma^k_{\;ij}-\partial_j\Gamma^k_{\;ik}+\Gamma^k_{\;kl}\Gamma^l_{\;ij}-\Gamma^k_{\;jl}\Gamma^l_{\;ik}$ 变分（$\delta$ 与 $\partial$ 可交换，因为变分不改变坐标）：

$$
\delta R_{ij}=\underbrace{\partial_k\delta\Gamma^k_{\;ij}}_{(\mathrm{i})}\underbrace{-\partial_j\delta\Gamma^k_{\;ik}}_{(\mathrm{ii})}+\underbrace{\delta\Gamma^k_{\;kl}\,\Gamma^l_{\;ij}}_{(\mathrm{iii})}+\underbrace{\Gamma^k_{\;kl}\,\delta\Gamma^l_{\;ij}}_{(\mathrm{iv})}\underbrace{-\delta\Gamma^k_{\;jl}\,\Gamma^l_{\;ik}}_{(\mathrm{v})}\underbrace{-\Gamma^k_{\;jl}\,\delta\Gamma^l_{\;ik}}_{(\mathrm{vi})}
$$

**第 2 步（展开两个协变导数）**：$\delta\Gamma^k_{\;ij}$ 按 $(1,2)$ 张量处理——上指标贡献 $+\Gamma\delta\Gamma$，每个下指标贡献 $-\Gamma\delta\Gamma$：

$$
\nabla_k\delta\Gamma^k_{\;ij}=\partial_k\delta\Gamma^k_{\;ij}+\Gamma^k_{\;kl}\delta\Gamma^l_{\;ij}-\Gamma^l_{\;ki}\delta\Gamma^k_{\;lj}-\Gamma^l_{\;kj}\delta\Gamma^k_{\;il}
$$

$$
\nabla_j\delta\Gamma^k_{\;ik}=\partial_j\delta\Gamma^k_{\;ik}+\Gamma^k_{\;jl}\delta\Gamma^l_{\;ik}-\Gamma^l_{\;ji}\delta\Gamma^k_{\;lk}-\Gamma^l_{\;jk}\delta\Gamma^k_{\;il}
$$

**第 3 步（相减，六项对六项配对）**：两式相减。末项 $-\Gamma^l_{\;kj}\delta\Gamma^k_{\;il}$ 与 $+\Gamma^l_{\;jk}\delta\Gamma^k_{\;il}$ 因 **Levi-Civita 联络无挠**（$\Gamma^l_{\;kj}=\Gamma^l_{\;jk}$）精确相消。余下逐项配对：

| 协变导数差的项 | 配对操作 | $\delta R_{ij}$ 的项 |
| --- | --- | --- |
| $\partial_k\delta\Gamma^k_{\;ij}$、$-\partial_j\delta\Gamma^k_{\;ik}$ | 恒等 | (i)、(ii) |
| $+\Gamma^k_{\;kl}\delta\Gamma^l_{\;ij}$ | 恒等 | (iv) |
| $-\Gamma^k_{\;jl}\delta\Gamma^l_{\;ik}$ | 恒等 | (vi) |
| $-\Gamma^l_{\;ki}\delta\Gamma^k_{\;lj}$ | 哑标 $k\leftrightarrow l$：$-\Gamma^k_{\;li}\delta\Gamma^l_{\;kj}$；无挠 $\Gamma^k_{\;li}=\Gamma^k_{\;il}$ 且 $\delta\Gamma^l_{\;kj}=\delta\Gamma^l_{\;jk}$ ⟹ $-\Gamma^k_{\;il}\delta\Gamma^l_{\;jk}$；再 $k\leftrightarrow l$ | (v) |
| $+\Gamma^l_{\;ji}\delta\Gamma^k_{\;lk}$ | 哑标 $k\leftrightarrow l$：$+\Gamma^k_{\;ji}\delta\Gamma^l_{\;lk}=\Gamma^k_{\;ij}\delta\Gamma^l_{\;lk}$；再 $k\leftrightarrow l$ | (iii) |

六项全部对应，恒等式成立。$\square$ 注意整个推导只用了**无挠性**（下指标对称）——Levi-Civita 联络的度量相容性在第 1 步的 $\delta\Gamma$ 显式形式中进入。

**(c) 主项积分**：用 (b) 计算 $\delta(\sqrt g\,R)$：

$$
\int\sqrt g\,g^{ij}\delta R_{ij}=\int\sqrt g\,g^{ij}\big(\nabla_k\delta\Gamma^k_{\;ij}-\nabla_j\delta\Gamma^k_{\;ik}\big)=\int\sqrt g\,\nabla_k\big(\underbrace{g^{ij}\delta\Gamma^k_{\;ij}-g^{ik}\delta\Gamma^j_{\;ij}}_{\text{全散度}}\big)=0
$$

边界项因变分紧支而消失。

**第 3 步（合并）**：

$$
\delta\mathcal{S}_{\text{geo}}=\frac{c^4}{16\pi G_{\text{info}}}\int d^d\theta\,\sqrt g\,\Big(R_{ij}-\frac12Rg_{ij}+\Lambda_{\text{info}}g_{ij}\Big)\delta g^{ij}
$$

$$
\delta\mathcal{S}_{\text{matter}}=-\frac12\int d^d\theta\,\sqrt g\,T_{ij}^{\text{info}}\,\delta g^{ij},\qquad T_{ij}^{\text{info}}\equiv-\frac{2}{\sqrt g}\frac{\delta(\sqrt g\mathcal{L}_m)}{\delta g^{ij}}
$$

$\delta\mathcal{S}=0$ 对任意 $\delta g^{ij}$：

$$
\boxed{\;R_{ij}-\frac12Rg_{ij}+\Lambda_{\text{info}}g_{ij}=\frac{8\pi G_{\text{info}}}{c^4}T_{ij}^{\text{info}}\;}
$$

**收缩检查**：两边乘 $g^{ij}$（$d$ 维）：左边 $=(1-\frac d2)R+d\Lambda_{\text{info}}$。$d=4$ 时 $-R+4\Lambda_{\text{info}}=\frac{8\pi G_{\text{info}}}{c^4}T$——与广义相对论迹方程一致。

### §21 牛顿极限：与 Poisson 方程的对接 [推导]

**弱场展开**：$g_{\mu\nu}=\eta_{\mu\nu}+h_{\mu\nu}$（号差 $\eta=\mathrm{diag}(-1,+1,+1,+1)$，与 §9 的 $-c_{\text{info}}^2dt^2+d\mathbf x^2$ 一致），$|h_{\mu\nu}|\ll1$；低速 $\dot\theta^\mu\approx(c,0,0,0)$；静态 $\partial_t h_{\mu\nu}=0$。

**第 1 步（Christoffel 符号线性化）**：$\Gamma^\mu_{\;00}=\frac12g^{\mu\lambda}(\partial_0g_{\lambda0}+\partial_0g_{0\lambda}-\partial_\lambda g_{00})\approx-\frac12\eta^{\mu\lambda}\partial_\lambda h_{00}$（静态使前两项为零；$g^{\mu\lambda}\to\eta^{\mu\lambda}$ 到一阶）。$\mu=0$ 分量因 $\partial_0h_{00}=0$ 为零；仅空间分量非零——注意 $\eta^{ij}=+\delta^{ij}$，**总负号来自定义中的 $-\partial_\lambda g_{00}$**：

$$
\Gamma^i_{\;00}=-\frac12\eta^{ij}\partial_jh_{00}=-\frac12\partial_ih_{00}
$$

**第 2 步（测地方程 → 牛顿）**：低速下 $\ddot x^i=-\Gamma^i_{\;00}(\dot x^0)^2=-c^2\Gamma^i_{\;00}=+\frac{c^2}{2}\partial_ih_{00}$。与牛顿方程 $\ddot{\mathbf x}=-\nabla\Phi$ 逐项比较：

$$
\frac{c^2}{2}\partial_ih_{00}=-\partial_i\Phi\;\Longrightarrow\;\boxed{\;h_{00}=-\frac{2\Phi}{c^2}\;}
$$

（点质量 $\Phi=-GM/r<0$ ⟹ $h_{00}>0$，即 $g_{00}=-(1+2\Phi/c^2)$：引力势阱中信息时钟变慢——引力红移的萌芽。）

**第 3 步（场方程线性化）**：静态无压尘埃 $T_{\mu\nu}=\rho\,u_\mu u_\nu$（$u^\mu\approx(c,\mathbf 0)$）：$T_{00}\approx\rho c^2$，迹 $T=\eta^{\mu\nu}T_{\mu\nu}=-T_{00}+O(h)\approx-\rho c^2$。

Ricci 分量（静态、线性阶，二次 $\Gamma\Gamma$ 项弃去）：

$$
R_{00}=\partial_k\Gamma^k_{\;00}-\partial_0\Gamma^k_{\;0k}+O(h^2)=\partial_i\Gamma^i_{\;00}=-\frac12\nabla^2h_{00}=+\frac{\nabla^2\Phi}{c^2}
$$

迹方程（$d=4$ 收缩 §20 末的检查式）：$R=-\frac{8\pi G_{\text{info}}}{c^4}T$（注意负号——收缩 $R-\frac d2R=\frac{8\pi G}{c^4}T$ 得 $(1-\frac d2)R=\frac{8\pi G}{c^4}T$，$d=4$ 时 $-R=\frac{8\pi G}{c^4}T$），代入 $T\approx-\rho c^2$：

$$
R=+\frac{8\pi G_{\text{info}}\rho}{c^2}
$$

代入 $00$ 分量 $R_{00}-\frac12 Rg_{00}=\frac{8\pi G_{\text{info}}}{c^4}T_{00}$（$g_{00}\approx\eta_{00}=-1$，故 $-\frac12Rg_{00}=+\frac R2$）：

$$
\frac{\nabla^2\Phi}{c^2}+\frac12\cdot\frac{8\pi G_{\text{info}}\rho}{c^2}=\frac{8\pi G_{\text{info}}}{c^4}\cdot\rho c^2
\;\Longrightarrow\;
\frac{\nabla^2\Phi}{c^2}=\frac{8\pi G_{\text{info}}\rho}{c^2}-\frac{4\pi G_{\text{info}}\rho}{c^2}=\frac{4\pi G_{\text{info}}\rho}{c^2}
$$

$$
\boxed{\;\nabla^2\Phi=4\pi G_{\text{info}}\rho\;}
$$

**与 Newton 引力对接要求 $G_{\text{info}}=G$**——信息引力常数在 $T_{\text{info}}\to0$ 极限下的取值由此固定。整条链条：DPI → Fisher 度量 → 信息 Einstein 方程 → 弱场极限 → Poisson 方程，每一环都是显式计算。

**引力弱的原因** [推导]：单圈修正下 $\hbar_{\text{info}}=\hbar\,(1+aT_{\text{info}}^2+\cdots)$，有效耦合

$$
G_{\text{eff}}(T_{\text{info}})=G\big(1+\gamma T_{\text{info}}^2+O(T_{\text{info}}^4)\big)
$$

当前宇宙 $T_{\text{info}}\sim10^{-5}$（Planck 单位）⟹ 修正 $\sim10^{-10}$；早期宇宙 $T_{\text{info}}\sim1$ ⟹ 引力与规范力强度可比——大统一的信息力学起源（不必引入新群）。

### §22 引力量子化障碍与信息 Wheeler–DeWitt 方程

**定理 22.1（信息塔定理）** [推导]：微扰量子化 $g_{ij}=\bar g_{ij}+\kappa h_{ij}$ 必然发散，其信息论根源是**累积量塔的非闭合性**。

*推导*：Fisher 度量是 KL 散度的 Hessian（定理 4.1 的逆命题）：

$$
g_{ij}(\theta)=\frac{\partial^2}{\partial d\theta_i\,\partial d\theta_j}\,D_{\text{KL}}\big(P_\theta\,\|\,P_{\theta+d\theta}\big)\Big|_{d\theta=0}
$$

$n$ 点引力子关联 $\langle h(x_1)\cdots h(x_n)\rangle$ 对应 $D_{\text{KL}}$ 的 $2n$ 阶变分导数——即分布的 $2n$ 阶累积量 $\kappa_{2n}$。量子场论要求对 $n$ 无截断（任意高阶顶点都要进入圈图）；而概率分布的高阶累积量无自然截断（三阶 = 偏度、四阶 = 峰度……）。微扰引力 = 用前二阶矩（度规）的涨落修正去重写所有高阶矩——**矩问题的严格不闭合**。发散不是技术困难，是用"信息的二阶导数"去描述"信息的高阶结构"的范畴错误。$\square$

**定理 22.2（信息 Wheeler–DeWitt 方程）** [推导，结构平行于 ADM 正则量子化]：

**第 1 步（ADM 分解）**：$ds^2_{\text{info}}=-\mathcal{N}^2d\tau^2+\gamma_{ab}(d\theta^a+\mathcal{N}^ad\tau)(d\theta^b+\mathcal{N}^bd\tau)$。Gauss–Codazzi 方程给出

$$
R_{\text{info}}={}^{(d-1)}R+K_{ab}K^{ab}-K^2+(\text{全散度})
$$

其中 $K_{ab}=\frac{1}{2\mathcal{N}}(\dot\gamma_{ab}-\nabla_a\mathcal{N}_b-\nabla_b\mathcal{N}_a)$ 为信息外曲率。

**第 2 步（正则动量）**：$\mathcal{S}_{\text{geo}}$ 对 $\dot\gamma_{ab}$ 变分：

$$
\pi^{ab}=\frac{\delta\mathcal{S}}{\delta\dot\gamma_{ab}}=\frac{\sqrt\gamma}{16\pi G_{\text{info}}}\big(K^{ab}-K\gamma^{ab}\big)
$$

（利用 $\delta K_{ab}/\delta\dot\gamma_{cd}=\frac{1}{2\mathcal N}\delta_{(a}^c\delta_{b)}^d$ 与二次型 $K_{ab}K^{ab}-K^2$ 的微分——注意此二次型的符号结构使 $\pi^{ab}$ 好定义。）

**第 2½ 步（反解 $K[\pi]$——系数 $\frac{1}{d-2}$ 的来源）**：空间维数为 $d-1$（$\gamma^{ab}\gamma_{ab}=d-1$）。对 $\pi^{ab}=\frac{\sqrt\gamma}{16\pi G_{\text{info}}}(K^{ab}-K\gamma^{ab})$ 取迹：

$$
\pi\equiv\gamma_{ab}\pi^{ab}=\frac{\sqrt\gamma}{16\pi G_{\text{info}}}\big(K-(d-1)K\big)=-\frac{\sqrt\gamma\,(d-2)}{16\pi G_{\text{info}}}\,K
\;\Longrightarrow\;
K=-\frac{16\pi G_{\text{info}}}{(d-2)\sqrt\gamma}\,\pi
$$

回代：$K^{ab}=\frac{16\pi G_{\text{info}}}{\sqrt\gamma}\pi^{ab}+K\gamma^{ab}=\frac{16\pi G_{\text{info}}}{\sqrt\gamma}\Big(\pi^{ab}-\frac{1}{d-2}\pi\gamma^{ab}\Big)$。计算动能二次型：

$$
K_{ab}K^{ab}=\Big(\frac{16\pi G_{\text{info}}}{\sqrt\gamma}\Big)^2\Big[\pi^{ab}\pi_{ab}-\frac{2}{d-2}\pi^2+\frac{d-1}{(d-2)^2}\pi^2\Big],\qquad K^2=\Big(\frac{16\pi G_{\text{info}}}{\sqrt\gamma}\Big)^2\frac{\pi^2}{(d-2)^2}
$$

$\pi^2$ 系数合并：$\frac{-2(d-2)+(d-1)-1}{(d-2)^2}=\frac{-(d-2)}{(d-2)^2}=-\frac{1}{d-2}$。故

$$
K_{ab}K^{ab}-K^2=\Big(\frac{16\pi G_{\text{info}}}{\sqrt\gamma}\Big)^2\Big[\pi^{ab}\pi_{ab}-\frac{1}{d-2}\pi^2\Big]
$$

**系数 $\frac{1}{d-2}$（而非 $\frac{1}{d-1}$）由此唯一确定**；$d=4$ 时退化为 ADM 的 $\pi^{ab}\pi_{ab}-\frac12\pi^2$ ✓。

**第 3 步（约束）**：$\mathcal{N},\mathcal{N}^a$ 无时间导数 ⟹ 是 Lagrange 乘子。变分给出

$$
\mathcal{H}_{\text{info}}\equiv\frac{16\pi G_{\text{info}}}{\sqrt\gamma}\Big(\pi^{ab}\pi_{ab}-\frac{1}{d-2}\pi^2\Big)-\frac{\sqrt\gamma}{16\pi G_{\text{info}}}\big({}^{(d-1)}R-2\Lambda_{\text{info}}\big)=0
$$

$$
\mathcal{H}_a^{\text{info}}\equiv-2\nabla_b\pi_a^{\;b}=0
$$

**物理含义**：$\mathcal{H}_{\text{info}}=0$ = 信息时间重参数化不变性（$\tau$ 的标定不产生物理）；$\mathcal{H}_a=0$ = 信息坐标不变性（$\theta^a$ 的标签不产生物理）。两者都是 Chentsov 定理（度规唯一性）的动力学影子。

**第 4 步（量子化）**：$[\hat\gamma_{ab}(\theta),\hat\pi^{cd}(\theta')]=i\hbar_{\text{info}}\,\delta_{(a}^c\delta_{b)}^d\,\delta(\theta-\theta')$，代换 $\hat\pi^{ab}\to-i\hbar_{\text{info}}\frac{\delta}{\delta\gamma_{ab}}$：

$$
\boxed{\;\Big[-16\pi G_{\text{info}}\,\hbar_{\text{info}}^2\,G^{abcd}\frac{\delta}{\delta\gamma_{ab}}\frac{\delta}{\delta\gamma_{cd}}-\frac{\sqrt\gamma}{16\pi G_{\text{info}}}\big({}^{(d-1)}R-2\Lambda_{\text{info}}\big)\Big]\Psi[\gamma]=0\;}
$$

其中 $G^{abcd}=\frac{1}{2\sqrt\gamma}\Big(\gamma^{ac}\gamma^{bd}+\gamma^{ad}\gamma^{bc}-\frac{2}{d-2}\gamma^{ab}\gamma^{cd}\Big)$（DeWitt 超度规——系数 $\frac{2}{d-2}$ 与第 2½ 步的动能项系数 $\frac{1}{d-2}$ 一致：动能算符 $=G^{abcd}\hat\pi_{ab}\hat\pi_{cd}$；$d=4$ 时退化为标准 ADM 形式 $-\gamma^{ab}\gamma^{cd}$ 项系数 $-1$。$d>2$ 时该超度规为 Lorentz 型——超空间的"时间"方向是共形因子 $\sim\det\gamma$，这就是 WdW 方程双曲性的几何根源）。

**为什么全流形量子化无发散** [推导]：$\Psi[\gamma]$ 是度规空间上的波泛函而非级数——所有阶累积量被联合编码，无"背景+涨落"分裂，故无需逐阶抵消发散；且信息 Planck 长度 $\ell_{\text{info}}=\sqrt{\hbar_{\text{info}}G_{\text{info}}/c_{\text{info}}^3}$ 提供天然紫外截断——短于 $\ell_{\text{info}}$ 的信息距离上"分布"本身失去意义（采样数 $<1$）。

---

## 第七部分：规范力——Noether 流的完整代数

### §23 信息 Noether 定理：光子/胶子/W,Z 的统一起源 [定理]

**设定**：信息内部空间 $K$ 上，Fisher 度规的 isometry 群 $\operatorname{Isom}(K)\supseteq SU(3)\times SU(2)\times U(1)$（E. Cartan：紧致流形的 isometry 群是紧致李群——规范群的紧致性不是假设而是定理）。Killing 矢量 $\xi_\alpha^a(\theta)$ 满足 $\mathcal{L}_\xi g^{(K)}_{ab}=0$。

**第 1 步（流的一般推导）**：场 $\Phi$（信息孤子构型）的拉格朗日密度 $\mathcal{L}(\Phi,\nabla_a\Phi,g^{(K)})$ 在无穷小 isometry $\delta\Phi=i\epsilon^\alpha T_\alpha\Phi$ 下不变：

$$
0=\delta\mathcal{S}=\int d^d\theta\sqrt g\left[\frac{\partial\mathcal{L}}{\partial\Phi}\delta\Phi+\frac{\partial\mathcal{L}}{\partial(\nabla_a\Phi)}\nabla_a(\delta\Phi)\right]
$$

分部积分 + EL 方程：

$$
\nabla_a j_\alpha^a=0,\qquad j_\alpha^a\equiv\frac{\partial\mathcal{L}}{\partial(\nabla_a\Phi)}\,(iT_\alpha\Phi)
$$

**第 2 步（荷的守恒）**：$Q_\alpha\equiv\int_\Sigma\sqrt\gamma\,j_\alpha^a n_a$，$\dot Q_\alpha=\int_\Sigma\sqrt\gamma\,\nabla_aj^a=0$（边界无流）。

**第 3 步（规范场 = 局域化后的联络）**：整体对称 $\epsilon^\alpha=$const 时流守恒；**局域化**（$\epsilon^\alpha(\theta)$ 任意）要求引入补偿场 $A_a^\alpha(\theta)$（$K$ 纤维上的联络），$\nabla_a\to D_a=\nabla_a-iA_a^\alpha T_\alpha$。$A_a^\alpha$ 的传播自由度 = 规范玻色子：

$$
\mathcal{L}_{\text{YM}}=-\frac14F^{\alpha\mu\nu}F_{\alpha\mu\nu},\qquad F_{\alpha}^{\mu\nu}=\partial^\mu A_\alpha^\nu-\partial^\nu A_\alpha^\mu+f_{\alpha\beta\gamma}A_\beta^\mu A_\gamma^\nu
$$

**结构常数决定一切** [推导]：

- $U(1)$：$f=0$ ⟹ 无自相互作用（光子不带电）、表示一维（电荷 = 特征标 $q\in\mathbb{Z}$ 的重数——**电荷量子化**是 $S^1$ 纤维拓扑 $\pi_1(S^1)=\mathbb{Z}$ 的推论）、正负电荷 = 顺/逆时针缠绕方向；
- $SU(2)$：$f_{abc}=\epsilon_{abc}$ ⟹ 3 个流（$W^1,W^2,W^3$，组合成 $W^\pm,Z$）；
- $SU(3)$：$f_{abc}\neq0$（8 维）⟹ 8 个流（胶子）+ 三/四胶子顶点（§25 渐近自由的来源）。

### §24 力律 = 传播子的 Fourier 计算 [推导]

**Proca/Yang–Mills 静态 Green 函数**：有质量矢量场（$\xi$-规范下）静态方程

$$
(-\nabla^2+m^2)A_0(\mathbf{x})=q\,\delta^{(3)}(\mathbf{x})
$$

**Fourier 求解，完整计算**：

$$
A_0(\mathbf{x})=\int\frac{d^3k}{(2\pi)^3}\,\frac{q\,e^{i\mathbf{k}\cdot\mathbf{x}}}{k^2+m^2}
$$

取 $\mathbf{x}$ 沿 $z$ 轴，球坐标 $d^3k=k^2dk\,d\Omega$：

$$
A_0(r)=\frac{q}{(2\pi)^3}\int_0^\infty\frac{k^2dk}{k^2+m^2}\,\underbrace{\int_0^\pi2\pi\sin\theta\,e^{ikr\cos\theta}d\theta}_{=4\pi\frac{\sin kr}{kr}}=\frac{q}{2\pi^2r}\int_0^\infty\frac{k\sin kr}{k^2+m^2}\,dk
$$

标准围道积分（极点 $k=\pm im$，留数 $\frac12e^{-mr}$）：

$$
\int_0^\infty\frac{k\sin kr}{k^2+m^2}dk=\frac{\pi}{2}e^{-mr}
\;\Longrightarrow\;
\boxed{\;A_0(r)=\frac{q}{4\pi}\,\frac{e^{-mr}}{r}\;}
$$

**两荷之间的势与力**：

$$
V(r)=q_1A_0(r)=\frac{q_1q_2}{4\pi}\frac{e^{-mr}}{r},\qquad F(r)=-\frac{dV}{dr}=\frac{q_1q_2}{4\pi}\,\frac{e^{-mr}(1+mr)}{r^2}
$$

**两个极限**：

1. **$m=0$（光子）**：$V\propto1/r$，$F\propto1/r^2$——Coulomb 长程力。零质量的原因：$U(1)_{\text{em}}$ 是电破缺后残余对称方向，序参量沿该方向的协变导数 $D_{\text{em}}\phi_{\text{vac}}=0$（§17 第 2 步中 $A_\mu$ 不出现在 $|D\phi|^2$ 中）——**零耗散信息通道**；
2. **$m=m_W$（弱流）**：力程 $\lambda=1/m_W=\hbar c/(80.4\text{ GeV})\approx2.4\times10^{-18}$ m——短程。弱力之"弱"不是耦合小（$g\approx0.65$ 与电磁同量级）而是**质量门槛**：低能过程中 $W$ 交换被压低 $\sim(q^2/m_W^2)^2$，有效 Fermi 常数：

$$
\frac{G_F}{\sqrt2}=\frac{g^2}{8m_W^2}\;\Longrightarrow\;G_F\approx1.17\times10^{-5}\text{ GeV}^{-2}
$$

代入数值验证：$g=0.653$，$m_W=80.4$ ⟹ $G_F/\sqrt2=\frac{0.426}{8\times6464}=8.2\times10^{-6}\text{ GeV}^{-2}$——与实测 $1.166\times10^{-5}/\sqrt2$ 一致（差别来自辐射修正）。**弱相互作用的全部"弱"来自电破缺相变的质量赋予**——这是 §17 代数的直接物理后果。

### §25 强力：跑动耦合与禁闭的定量推导

**§25.1 渐近自由的单圈计算** [推导]：

**第 1 步（$\beta$ 函数）**：非 Abel 规范理论的 Callan–Symanzik $\beta$ 函数（单圈）：

$$
\beta(g)\equiv\mu\frac{dg}{d\mu}=-\frac{g^3}{16\pi^2}\Big(\frac{11}{3}C_A-\frac{4}{3}T_Fn_f\Big)
$$

其中 $C_A=3$（$SU(3)$ 伴随表示 Casimir），$T_F=1/2$（基础表示指标），$n_f$ 为夸克味数。系数来源：$-\frac{11}{3}C_A$ 来自胶子圈（反屏蔽——非 Abel 结构常数 $f^{abc}\neq0$ 使胶子自耦合）；$+\frac43T_Fn_f$ 来自夸克圈（普通屏蔽）。

**第 2 步（渐近自由条件）**：

$$
\beta(g)<0\iff n_f<\frac{11C_A}{4T_F}=\frac{11\times3}{4\times\frac12}=16.5
$$

标准模型 $n_f=6\ll16.5$——渐近自由成立。**信息力学解读**：反屏蔽 = 信息 Noether 流的自耦合在高信息能标（短 Fisher 距离）上"去屏蔽"色荷——非 Abel 流的量子涨落在短距离重新规整信息几何。

**第 3 步（积分 RG 方程）**：令 $\alpha_s=g^2/(4\pi)$：

$$
\frac{d\alpha_s}{d\ln\mu}=\frac{2g}{4\pi}\beta(g)=-\frac{\alpha_s^2}{2\pi}\Big(11-\frac{2n_f}{3}\Big)\equiv-b\,\alpha_s^2
$$

分离变量 $\frac{d\alpha_s}{\alpha_s^2}=-b\,d\ln\mu$，积分 $\frac{1}{\alpha_s(\mu)}-\frac{1}{\alpha_s(\mu_0)}=b\ln\frac{\mu}{\mu_0}$。定义 $\Lambda_{\text{QCD}}$（积分常数，$\alpha_s$ 发散处）：

$$
\boxed{\;\alpha_s(Q)=\frac{12\pi}{(33-2n_f)\ln(Q/\Lambda_{\text{QCD}})}\;}
$$

**数值**：$n_f=5$，$\Lambda_{\text{QCD}}\approx210$ MeV；$Q=m_Z=91.2$ GeV：$\ln(91200/210)\approx6.07$，$\alpha_s=\frac{12\pi}{23\times6.07}\approx0.118$——与 LEP 实测 $0.1181\pm0.0011$ 精确一致。**信息力学无需修改此计算——它提供的是解释**：跑动的"能量标度"$\mu$ = Fisher 信息密度标度；$\Lambda_{\text{QCD}}$ = 信息无序局域化标度（下节）。

**§25.2 色禁闭 = Anderson 局域化 [推导，标度计算]**

**第 1 步（色真空 = 信息无序介质）**：非 Abel 真空中胶子场强涨落 $\langle F^a_{\mu\nu}F^{a\mu\nu}\rangle\neq0$（凝涨落），构成 $K$ 上 $SU(3)$ 纤维方向的随机势——数学上等价于无序介质中的随机矩阵乘积（AI 侧对应：深网络梯度消失，`physical to ai derivation.md` §3）。

**第 2 步（局域化长度）**：Anderson 局域化给出流传播子包络衰减

$$
\langle A^a(x)A^b(0)\rangle\propto\delta^{ab}\,\frac{e^{-|x|/\xi_A}}{|x|},\qquad \xi_A\approx\Lambda_{\text{QCD}}^{-1}\approx1\text{ fm}
$$

（局域化长度由无序强度 $\sim\Lambda_{\text{QCD}}$ 设定——RG 中耦合在 $Q\to\Lambda_{\text{QCD}}$ 处发散 ⟹ 该标度以下扰动论失效、涨落强到局域化一切色流。）

**第 3 步（弦张力）**：色电通量被挤压进横截面 $\sim\xi_A^2$ 的通量管。单位长度能量：

$$
\sigma\approx\frac{\text{场强能量}}{\text{单位体积}}\times\text{截面}\sim\Lambda_{\text{QCD}}^2\,\xi_A^2/\xi_A^2\cdot\xi_A^{-1}\cdot\Lambda_{\text{QCD}}\;\Rightarrow\;\sigma\sim\Lambda_{\text{QCD}}^2\approx(0.2\text{ GeV})^2\approx0.9\text{ GeV/fm}
$$

与格点 QCD 实测 $\sigma\approx0.89$ GeV/fm（$\sqrt\sigma\approx420$ MeV）一致。

**第 4 步（弦断裂 → 无孤立夸克）**：$E(r)=\sigma r$；当 $E>2m_\pi$（实际是 $E>$ 产生 $q\bar q$ 对的阈值 $\sim2m_q+\sigma\cdot\xi_A\sim\Lambda_{\text{QCD}}$）时真空自发产生夸克对、弦断裂成两个介子。**孤立色荷携带能量 $\sigma r\to\infty$——禁闭不是禁令，是能量的指数压制。**

**第 5 步（强子质量预算）**：囚禁于 $\xi_A\sim1$ fm 内的夸克零点动能（信息不确定关系 $\Delta p\,\Delta x\sim\hbar$）：

$$
E_{\text{kin}}\sim\frac{\hbar c}{\xi_A}\sim200\text{ MeV}
$$

三个夸克 $\sim600$ MeV + 胶子场自耦合能 + 拓扑贡献 ⟹ 质子 $938$ MeV，而流夸克质量之和 $\sim9$ MeV。**质子质量的 99% 是信息动能与规范场能量——质量层级的最深层来自禁闭的信息几何。**

---

## 第八部分：热力学四定律 = 信息论的完整代数

### §26 第一定律：链式分解的全部细节 [定理]

**设定**：$\rho(\theta)=\mathcal{Z}^{-1}e^{-\mathcal{H}_{\text{info}}(\theta)/T_{\text{info}}}$，$U=\langle\mathcal{H}\rangle$，$S=-\int\rho\ln\rho$。

**Leibniz 分解**：

$$
dU=\underbrace{\int\mathcal{H}\,d\rho\,d\theta}_{\text{分布移动（热）}}+\underbrace{\int(d\mathcal{H})\,\rho\,d\theta}_{\text{哈密顿量改变（功）}}
$$

**热项的完整计算**（关键恒等式）：由 Gibbs 分布取对数：

$$
\ln\rho=-\frac{\mathcal{H}}{T_{\text{info}}}-\ln\mathcal{Z}\;\Longrightarrow\;\mathcal{H}=-T_{\text{info}}(\ln\rho+\ln\mathcal{Z})
$$

代入热项：

$$
\int\mathcal{H}\,d\rho=-T_{\text{info}}\int(\ln\rho+\ln\mathcal{Z})\,d\rho=-T_{\text{info}}\int\ln\rho\,d\rho-\;T_{\text{info}}\ln\mathcal{Z}\underbrace{\int d\rho}_{=0}
$$

（$\int d\rho=d\int\rho=d(1)=0$。）再用 $\int\rho\,d\ln\rho=\int d\rho=0$：

$$
dS=-d\int\rho\ln\rho=-\int d\rho\ln\rho-\int\rho\,d\ln\rho=-\int d\rho\ln\rho
$$

合并：

$$
\int\mathcal{H}\,d\rho=T_{\text{info}}\,dS\equiv\delta Q_{\text{info}}
$$

**功项**：$\langle d\mathcal{H}\rangle=\delta W_{\text{info}}$（外参改变：$g_{ij}$ 形变、$V_{\text{info}}$ 改变、$N_{\text{info}}$ 增减）。

$$
\boxed{\;dU_{\text{info}}=T_{\text{info}}\,dS_{\text{info}}+\delta W_{\text{info}}\;}
$$

**自由能恒等式（同一组代数）**：

$$
F=U-TS=\langle\mathcal{H}\rangle+T\int\rho\ln\rho=\langle\mathcal{H}\rangle+T\Big\langle-\frac{\mathcal{H}}{T}-\ln\mathcal{Z}\Big\rangle=\langle\mathcal{H}\rangle-\langle\mathcal{H}\rangle-T\ln\mathcal{Z}
$$

$$
\boxed{\;F_{\text{info}}=-T_{\text{info}}\ln\mathcal{Z}\;}
$$

**Boltzmann 分布 = 最大熵**（Jaynes 变分，一行复述）：$\delta\big[-\!\int\rho\ln\rho-\lambda_0(\int\rho-1)-\lambda_1(\langle\mathcal{H}\rangle-U)\big]=0$ ⟹ $-\ln\rho-1-\lambda_0-\lambda_1\mathcal{H}=0$ ⟹ Gibbs 分布；$\lambda_1=1/T_{\text{info}}$ 由 $dS/dU=\lambda_1\overset{!}{=}1/T$ 固定。**Boltzmann 分布不再是公设——它是 DPI 自洽性（最小偏见）的唯一解。**

### §27 第二定律 = DPI [定理]

**三步证明**（每一环都显式）：

**第 1 步（DPI）**：Markov 链 $X\to T\to Y$（$Y$ 由 $T$ 经条件核 $P(y|t)$ 生成）满足

$$
I(X;Y)\le I(X;T)
$$

*证明*：马尔可夫性 ⟹ $I(X;Y|T)=0$。链式法则 $I(X;Y,T)=I(X;T)+I(X;Y|T)=I(X;T)$；又 $I(X;Y,T)=I(X;Y)+I(X;T|Y)\ge I(X;Y)$（互信息非负）。合并即得。$\square$

**第 2 步（熵产生 = 条件熵差）**：

$$
I(X;T)-I(X;Y)=\big[H(X)-H(X|T)\big]-\big[H(X)-H(X|Y)\big]=H(X|Y)-H(X|T)\ge0
$$

（等号 ⟺ $Y$ 是 $T$ 关于 $X$ 的充分统计量 ⟺ 过程可逆。）

**第 3 步（热力学形式）**：孤立系统中全部演化都是 Markov 粗粒化（分子碰撞 = 随机核），故

$$
dS_{\text{info}}\big/d\tau=H(X|Y)-H(X|T)\ge0
$$

与 §26 的 $\delta Q_{\text{info}}=T_{\text{info}}dS|_{\text{rev}}$ 结合：

$$
dS_{\text{info}}\ge\frac{\delta Q_{\text{info}}}{T_{\text{info}}}\qquad(\text{Clausius})
$$

**Carnot 效率的上界**（推论，一行）：热机在 $T_H,T_C$ 间循环，$\eta=1-\frac{Q_C}{Q_H}$；由 Clausius 不等式 $\oint\frac{\delta Q}{T}\le0$ ⟹ $\frac{Q_H}{T_H}-\frac{Q_C}{T_C}\le0$ ⟹ $\eta\le1-\frac{T_C}{T_H}$。**Carnot 定理 = DPI 的工程学形式。**

### §28 第三定律与 Liouville 破坏 [推导]

**第三定律**：$T_{\text{info}}\to0$ 时 $\rho\to$ 基态上的均匀分布。若基态 $g$ 重简并：$\rho=\frac1g\sum_{k=1}^g\delta_k$，$S=\ln g$；非简并：$S=0$。Nernst 定理 $S(T\to0)\to S_0$ = **Shannon 熵在确定分布下的值**——不需要新原理。

**为何绝对零度不可达**：配分函数低温展开 $U(T)=E_0+A\,e^{-\Delta/T}$（$\Delta$ = 第一激发能隙）。冷却需要抽取的热量

$$
Q=\int_0^{T}C\,dT',\qquad C=\frac{dU}{dT}=\frac{A\Delta}{T^2}e^{-\Delta/T}
$$

随 $T\to0$ 指数消失——**每个单位降温需要指数增多的操作步数**（信息论：擦除越来越接近纯态的分布，Landauer 代价 $T\ln2$/bit 虽小但步数发散）。

**信息 Liouville 定理与破坏**：

**无噪声**（$T_{\text{info}}=0$）：哈密顿流保辛体积。弯曲流形上的正确表述用 Fisher 体积元：

$$
\frac{d}{d\tau}\big(\rho\sqrt{\det g}\big)=0
$$

*推导*：相空间速度场 $(\dot\theta,\dot p)=(g^{ij}p_j,-\partial_i\mathcal V-\frac12\partial_ig^{jk}p_jp_k)$ 的散度（含 $\sqrt{\det g}$ 权重）：

$$
\frac{1}{\sqrt{\det g}}\partial_i\big(\sqrt{\det g}\,\dot\theta^i\big)+\frac{1}{\sqrt{\det g}}\partial_{p_i}\big(\sqrt{\det g}\,\dot p_i\big)=\nabla_i^{\text{LC}}\dot\theta^i+0=\frac{1}{2}\frac{d}{d\tau}\ln\det g
$$

（Levi-Civita 散度恰好被体积元的时间变化抵消。）$\square$

**有噪声**（$T_{\text{info}}\neq0$）：Fokker–Planck 方程

$$
\partial_\tau\rho+\{\rho,\mathcal{H}\}=\eta\,\Delta_{\text{LB}}\rho,\qquad \Delta_{\text{LB}}=\frac{1}{\sqrt{\det g}}\partial_i\big(\sqrt{\det g}\,g^{ij}\partial_j\big)
$$

**熵产生率的非负性**（关键计算）：

$$
\frac{dS}{d\tau}=-\frac{d}{d\tau}\int\rho\ln\rho\,\sqrt{g}\,d\theta=-\int(1+\ln\rho)\,\partial_\tau\rho\,\sqrt g\,d\theta
$$

代入 Fokker–Planck：泊松括号部分对熵无贡献（辛流保熵——Liouville）；扩散部分分部积分两次（边界项消失，$\Delta_{\text{LB}}$ 自伴）：

$$
\frac{dS}{d\tau}=\eta\int\frac{(\partial_i\rho)(\partial_j\rho)}{\rho}\,g^{ij}\sqrt g\,d\theta=\eta\int g^{ij}\,\partial_i\ln\rho\,\partial_j\ln\rho\;\rho\sqrt g\,d\theta
$$

被积函数 $= \eta\,g^{ij}s_is_j\ge0$（$g$ 正定，二次型非负）。$\square$

$$
\boxed{\;\frac{dS_{\text{info}}}{d\tau}=\eta\,I_{\text{Fisher}}[\rho]\ge0\;}
$$

**熵产生率 = $\eta\times$（分布的 Fisher 信息）**——热力学第二定律的微观形式：扩散把分布磨平，速率由分布自身的 Fisher 信息（梯度陡度）决定。陡的地方磨得快；$I_{\text{Fisher}}\to0$（分布已平）时熵产生停止——平衡态。

### §29 涨落-耗散定理与信息理想气体 [定理+推导]

**涨落-耗散**：对 $U=\mathcal{Z}^{-1}\int\mathcal{H}e^{-\mathcal{H}/T}$ 求 $T$ 导数（分母求导产生第二项）：

$$
\frac{\partial U}{\partial T}=\frac{1}{T^2}\Big[\langle\mathcal{H}^2\rangle-2\langle\mathcal{H}\rangle^2+\langle\mathcal{H}\rangle^2\Big]=\frac{\langle(\Delta\mathcal{H})^2\rangle}{T^2}
$$

（核心步骤：$\partial_T\mathcal{Z}=\frac{U}{T^2}\mathcal{Z}$，商法则中 $-U\times\frac{U}{T^2}$ 与 $\langle\mathcal{H}^2\rangle/T^2$ 合并成方差。）**信息热容 = 能量涨落的归一方差**——AI 侧操作：从 SGD 轨迹测 $\mathrm{Var}(\mathcal{H})$ 即得 $C_{\text{info}}$，无需调温实验。

**信息理想气体**（可计算的基准实例）：$N$ 个独立孤子，平直度规 $g_{ij}=m\delta_{ij}$，$\mathcal{H}=p^2/2m$，$d$ 维：

$$
\mathcal{Z}_1=\int e^{-p^2/2mT}d^dp=(2\pi mT_{\text{info}})^{d/2}
$$

（Gaussian 积分 $\int e^{-ap^2}dp=\sqrt{\pi/a}$ 的 $d$ 次幂，$a=1/2mT$。）不可区分修正 $N!$（Stirling $\ln N!\approx N\ln N-N$）：

$$
F=-T\ln\frac{\mathcal{Z}_1^NV_{\text{info}}^N}{N!}\approx-NT\Big[\ln\frac{V_{\text{info}}}{N}+\frac d2\ln(2\pi mT)+1\Big]
$$

$$
\mathcal{P}_{\text{info}}=-\frac{\partial F}{\partial V_{\text{info}}}=\frac{NT_{\text{info}}}{V_{\text{info}}}
\;\;\Longleftrightarrow\;\;PV=Nk_BT
$$

**$k_B$ 的消解**：$k_B=1.38\times10^{-23}$ J/K 不是自然常数而是单位换算（bit ↔ Joule ↔ Kelvin）。信息力学自然单位制 $k_B=1$：物理温度 $T_{\text{phys}}$、信息温度 $T_{\text{info}}$、Gibbs 乘子 $\beta_{\text{Gibbs}}$ 是**同一个量**的三种单位。

---

## 第九部分：极限条件——为什么是这些值

### §30 信息温度的宇宙学冷却 [推导]

**冷却方程**：信息正则系统在膨胀的统计流形上（体积 $V_{\text{info}}(\tau)\propto a(\tau)^d$，$a$ = 信息尺度因子）：

**第 1 步（绝热条件）**：膨胀缓慢（每个微观信息时间内 $a$ 几乎不变）⟹ 熵不变。对理想信息气体（§29），$S\propto N\ln\big(V\,T^{d/2}\big)$ 恒定：

$$
d\ln V+\frac d2\,d\ln T=0
$$

**第 2 步（代入膨胀）**：$d\ln V=d\,d\ln a$：

$$
\frac{dT_{\text{info}}}{d\tau}=-\frac{T_{\text{info}}}{d}\cdot d\cdot\frac{\dot a}{a}=-\,T_{\text{info}}\,\frac{\dot a}{a}
\;\Longrightarrow\;
T_{\text{info}}\propto a^{-1}
$$

**一般形式**（含势能 expansion scalar $\Theta_{\text{info}}=\nabla_i\nabla^i\mathcal{V}_{\text{info}}$）：

$$
\frac{dT_{\text{info}}}{d\tau}=-\frac{2}{\dim S}\,T_{\text{info}}\,\Theta_{\text{info}}(\tau)
$$

**第 3 步（熵产生修正）**：实际宇宙有耗散（$\eta>0$，§28）：

$$
\frac{dT}{d\tau}=-T\frac{\dot a}{a}+\underbrace{\frac{\eta}{C_{\text{info}}}\,I_{\text{Fisher}}[\rho]}_{\text{耗散加热}}
$$

早期宇宙 $\rho$ 陡（信息梯度大）⟹ 耗散项显著；晚期 $\rho\to$ 平衡 ⟹ 冷却律渐近 $T\propto a^{-1}\to0$，**但永不严格为零**——残留 $T_{\text{info}}\neq0$ 表现为暗能量。热涨落能量密度按 Stefan–Boltzmann 型标度（量纲：$[\text{能量}]^4/(\hbar c)^3=$ 能量/体积）：

$$
\Lambda_{\text{info}}\propto T_{\text{info}}^4\;\Longrightarrow\;\rho_\Lambda=\frac{(k_BT_{\text{info}})^4}{(\hbar c)^3}\times(\text{数值系数})
$$

**单位换算的显式核验**（这条链是 $\Lambda\propto T^4$ 声称的定量内容）：观测 $\Omega_\Lambda\approx0.68$、临界密度 $\rho_{\rm crit}\approx8.5\times10^{-27}\,\mathrm{kg/m^3}$ 给出

$$
\rho_\Lambda c^2=\Omega_\Lambda\rho_{\rm crit}c^2\approx0.68\times8.5\times10^{-27}\times(3.0\times10^8)^2\approx5.2\times10^{-10}\ \mathrm{J/m^3}
$$

用 $\hbar c=1.055\times10^{-34}\times3.0\times10^8=3.16\times10^{-26}\,\mathrm{J\cdot m}$，$(\hbar c)^3\approx3.15\times10^{-77}\,\mathrm{J^3\,m^3}$，反解有效信息温度：

$$
k_BT_{\text{info}}=\big[\rho_\Lambda c^2\cdot(\hbar c)^3\big]^{1/4}=\big(5.2\times10^{-10}\times3.15\times10^{-77}\big)^{1/4}\,\mathrm J=\big(1.6\times10^{-86}\big)^{1/4}\,\mathrm J\approx3.6\times10^{-22}\,\mathrm J
$$

换算成 eV（$1\,\mathrm{eV}=1.602\times10^{-19}\,\mathrm J$）：$k_BT_{\text{info}}\approx2.2\times10^{-3}\,\mathrm{eV}$，即

$$
\rho_\Lambda\sim(2.3\ \mathrm{meV})^4
$$

**与 CMB 的对照**：当前 CMB 温度 $2.725\,\mathrm K\leftrightarrow k_BT_{\rm CMB}\approx0.235\,\mathrm{meV}$，其光子能量密度 $u_{\rm CMB}=\frac{\pi^2}{15}\frac{(k_BT_{\rm CMB})^4}{(\hbar c)^3}\approx4.2\times10^{-14}\,\mathrm{J/m^3}$。同一 $T^4$ 律给出 $\rho_\Lambda/u_{\rm CMB}\sim10^4$，即两者温度仅差**一个数量级**——**暗能量 = 宇宙信息温度的残余**：暗能量密度与 CMB 温度这两个看似无关的观测值，在信息力学中由同一条冷却律 $T\propto a^{-1}$ 与同一个 $T^4$ 标度联系。

### §31 为什么是 $(3+1)$ 维与不可逆世界 [推导+论证]

**维数论证（Green 函数衰减）** [推导]：$d$ 维空间中 Laplace 方程点源解 $G(r)\propto r^{2-d}$（$d\neq2$），$G(r)\propto\ln r$（$d=2$）。

- $d\le2$：$G$ 不衰减（或对数增长）——信息传播过于弥漫，无法形成局域束缚结构（稳定孤子需要衰减核）；
- $d=3$：$G\propto1/r$——**临界**：既保证长程信息可达（稳定轨道、化学键），又允许局域化（有限能量束缚态，$E\sim\int^\infty|\nabla G|^2r^{2}dr<\infty$ 需要 $d<4$）；
- $d\ge4$：$G\propto1/r^{d-2}$ 衰减过快，且束缚态能量积分发散上界——稳定轨道不存在（氢原子在 $d\ge4$ 无稳定基态：Bohr 论证 $E(r)=-Cr^{2-d}+\frac{L^2}{2mr^2}$ 在 $d\ge4$ 时无极小）。

**结论**：支持稳定信息孤子 + 长程信息通道的空间维度**恰好是 3**（时间维度 1 由因果结构/初值问题良定性固定——双曲型方程需要 1 个类时方向）。$(3+1)$ 不是偶然——是信息可局域化与可传播的同时要求。

**不可逆性论证** [论证，明示逻辑地位]：$D=0$ 且 $\beta=0$ 的信息系统是信息镜子（$Y=X$ 的恒等通道）——无压缩、无决策、无计算。任何**非平凡**信息处理（分类、记忆、推理）必须选择性丢弃信息（$\beta>0$）或扩散混合信息（$D>0$）——两者都产生熵（§28）⟹ **时间箭头 = 计算能力的代价**。可逆宇宙里没有观察者，因为观察者是计算结构。

**规范群三因子的地位** [猜想，标注]：$\operatorname{Isom}(K)=SU(3)\times SU(2)\times U(1)$ 的三种候选解释——(A) 信息作用量的变分极值（$K$ 像肥皂泡一样解 $\delta\mathcal{S}/\delta g^{(K)}=0$）；(B) 反常消除的信息版约束（纤维丛的示性类条件）；(C) 信息景观中支持生命的少数解。三者均可检验：A ⟹ 规范耦合可从第一原理计算；B ⟹ $K$ 拓扑唯一；C ⟹ 统计分布可算。当前无法裁决——诚实标注为开放问题。

### §32 量纲与物理常数的信息力学起源 [推导+定理]

**问题的提出**：前 31 节的推导中反复出现四个常数——$c$（§9–§11）、$\hbar$（§3、§12）、$k_B$（§29）、$G$（§20–§21）。传统物理把它们当作"自然界的基本常数"，只能测量、不能解释。信息力学给出更深的回答：

> **核心命题**：在信息力学自然单位制中，**一切物理量都是无量纲的**——都是 bit 的幂次。"量纲"不是自然界的内禀属性，而是同一个信息力学量在**不同投影极限**下被赋予的解读标签；$c,\hbar,k_B,G$ 是这些投影极限之间的**单位转换因子**。本节给出这个命题的完整代数内容（与 `dimensional analysis.md` 对应；量纲分析作为**工具**的用法见附录 B.5）。

#### (a) 信息力学量的自然量纲 [定义]

信息力学的唯一基本量是**信息作用量** $\mathcal{S}_{\text{info}}=\int\mathcal{L}_{\text{info}}\,dt$（bit）。所有其他量的"量纲"都是 bit 的幂次：

| **信息量** | **自然量纲** | **说明** |
| --- | --- | --- |
| $\mathcal{S}_{\text{info}}$ | bit | 唯一基本量 |
| $\mathcal{H}_{\text{info}}$ | bit | 信息能量（$=\partial\mathcal{S}/\partial t_{\text{info}}$，$t_{\text{info}}$ 以 bit 计） |
| $T_{\text{info}}$ | bit | 信息温度 |
| $\beta$ | bit$^{-1}$ | 逆温度/保真权重 |
| $\theta^i$ | bit$^{1/2}$ | Fisher 度规无量纲，$ds^2=g_{ij}d\theta^id\theta^j$ 以 bit 计 |
| $t_{\text{info}}$ | bit | 信息时间（轨迹的仿射参数） |
| $G_{\text{info}}$ | bit | 信息曲率-信息能量耦合 |
| $c_{\text{info}}$ | $1$ | 锥速度，无量纲 |
| $\hbar_{\text{info}}$ | bit | 最小信息作用量量子 |

**没有独立的长度、质量、时间、温度量纲**——它们全部来自投影（见 (b)）。

#### (b) 四个常数 = 四个投影极限的转换因子 [对应，附各自起源]

| **常数** | **信息力学起源** | **转换的方向** | **正文出处** |
| --- | --- | --- | --- |
| $c$ | 锥速度（Fisher 度量的因果结构） | 时间投影标度 $\tau$ ⟷ 空间投影标度 $\xi$：$c\sim\xi/\tau$ | §9–§11 |
| $\hbar$ | 最小信息作用量 | bit ⟷ J·s：把"信息量"兑换为"物理作用量" | §3、§12 |
| $k_B$ | 信息温度 ⟷ 物理温度的兑换率 | bit ⟷ J/K | §29 |
| $G$ | 信息曲率 ⟷ 物理时空曲率的残余耦合 | bit ⟷ m³/(kg·s²) | §20–§21 |

**关键观察**：$c,\hbar,k_B$ 在信息力学中都有**定义性的起源**（锥结构、Gibbs 温度、路径积分量子），唯独 $G$ 目前是**拟合常数**（§21 牛顿对接要求 $G_{\text{info}}=G$，其数值未被推导）——这就是为什么引力最难量子化的另一种表述：**它对应的常数是唯一没有被信息力学解释的一个**（见 §34 未解决问题清单）。

#### (c) 一对多对应：同一信息量 ⟹ 多个物理量纲 [对应表]

| **信息量** | **物理对应 1** | **物理对应 2** | **物理对应 3** |
| --- | --- | --- | --- |
| $\mathcal{H}_{\text{info}}$ | 质量 $m$ | 能量 $E$ | 内能 $U$ |
| $T_{\text{info}}$ | 温度 $T$ | SGD 噪声 | 暗能量 |
| $\mathcal{S}_{\text{info}}$ | 作用量 | 熵（差 $k_B$） | 相位（差 $\hbar$） |

**示例 1（质量的三种解读）**：$\mathcal{H}_{\text{info}}$ 在零温经典极限（§7，$T_{\text{info}}\to0$）下是**质量**；在热力学极限（§26–§29）下是**内能**；在相对论性流动（§9–§11 的锥结构投影）下是**能量**。三者在 SI 中量纲不同（kg vs J vs J），但在自然单位制中是**同一个 bit 数**。推论：$E=mc^2$ 中的 $c^2$ 不是物理内容——它是同一信息量两种投影单位之间的换算因子。

**示例 2（温度的三种解读）**：$T_{\text{info}}$ 在平衡态热力学中是**温度**；在 SGD 中是**噪声强度**（$T_{\text{info}}=\frac{\eta}{B}\mathbb{E}\|\nabla L_B-\nabla L\|^2$）；在宇宙学极限下是**暗能量密度**（$\Lambda\propto T_{\text{info}}^4$，§30）——三个"不同的物理现象"共享同一个信息源项。

#### (d) 量纲分析的线性代数本质 [定理+完整推导]

**定义（量纲向量）**：选 SI 基本量纲（长度 $L$、质量 $M$、时间 $T$、温度 $\Theta$），定义映射

$$
\Phi:\{\text{物理量}\}\to\mathbb Z^4,\qquad \Phi(Q)=(l,m,t,\theta)\ \Longleftrightarrow\ [Q]=L^lM^mT^t\Theta^\theta
$$

**同态性质**（由量纲乘法规则直接得到）：$\Phi(Q_1Q_2)=\Phi(Q_1)+\Phi(Q_2)$，$\Phi(Q^\alpha)=\alpha\Phi(Q)$——$\Phi$ 是乘法群到加法群的同态。**这就是为什么量纲分析本质是线性代数。**

**四常数的量纲向量**：

$$
\Phi(c)=(1,0,-1,0),\qquad\Phi(\hbar)=(2,1,-1,0),\qquad\Phi(k_B)=(2,1,-2,-1),\qquad\Phi(G)=(3,-1,-2,0)
$$

排成矩阵（行 = 常数）：

$$
M=\begin{pmatrix}1&0&-1&0\\2&1&-1&0\\2&1&-2&-1\\3&-1&-2&0\end{pmatrix}
$$

**行列式显式计算**（按第 4 列展开——只有第 3 行的 $-1$ 非零）：

$$
\det M=(-1)^{3+4}\cdot(-1)\cdot\det\begin{pmatrix}1&0&-1\\2&1&-1\\3&-1&-2\end{pmatrix}=1\cdot\big[1\cdot(-2-1)-0\cdot(\cdots)+(-1)\cdot(2\cdot(-1)-1\cdot3)\big]
$$

$$
=1\cdot\big[{-3}+5\big]=2\neq0
$$

**定理（量纲构造的完备性）**：$\det M\neq0$ ⟹ 四常数的量纲向量在 $\mathbb{Q}^4$ 中线性无关 ⟹ **任何物理量的量纲都可以唯一地写成 $c^\alpha\hbar^\beta k_B^\gamma G^\delta$**，指数由线性方程组 $M^{\mathsf T}(\alpha,\beta,\gamma,\delta)^{\mathsf T}=\Phi(Q)$ 唯一解出。$\square$

**算例 1（质量的无量纲化）**：$\Phi(m)=(0,1,0,0)$。方程组：

$$
\begin{cases}\text{质量：}&\beta+\gamma-\delta=1\\ \text{长度：}&\alpha+2\beta+2\gamma+3\delta=0\\ \text{时间：}&-\alpha-\beta-2\gamma-2\delta=0\\ \text{温度：}&-\gamma=0\end{cases}
$$

逐步求解：温度方程给出 $\gamma=0$；长度方程与时间方程**相加消去 $\alpha$**：$(2\beta+3\delta)-(\beta+2\delta)=\beta+\delta=0$ ⟹ $\beta=-\delta$；代入质量方程：$-2\delta=1$ ⟹ $\delta=-\frac12,\ \beta=\frac12$；回代时间方程：$\alpha=-\beta-2\delta=\frac12$。得

$$
m\sim c^{1/2}\hbar^{1/2}G^{-1/2}\times(\text{无量纲数})\;\Longrightarrow\;\text{无量纲质量}=\frac{m}{m_P},\qquad m_P=\sqrt{\frac{\hbar c}{G}}
$$

**算例 2（能量）**：$\Phi(E)=(2,1,-2,0)$。同样步骤：$\gamma=0$；质量方程给出 $\beta=1+\delta$；长度方程代入得 $\alpha=-5\delta$；时间方程 $-\alpha-\beta-2\delta=2\delta-1=-2$ ⟹ $\delta=-\frac12,\ \beta=\frac12,\ \alpha=\frac52$：

$$
E\sim c^{5/2}\hbar^{1/2}G^{-1/2}\times(\text{无量纲数})\;\Longrightarrow\;\text{无量纲能量}=\frac{E}{E_P},\qquad E_P=\sqrt{\frac{\hbar c^5}{G}}
$$

**算例 3（熵——最干净的情形）**：$\Phi(S)=(2,1,-2,-1)$。温度方程 $-\gamma=-1$ ⟹ $\gamma=1$；质量方程 $\beta+1-\delta=1$ ⟹ $\beta=\delta$；长度+时间相加：$\beta+\delta=0$；联立得 $\beta=\delta=0$，回代 $\alpha=0$：

$$
S\sim k_B^1\times(\text{无量纲数})\;\Longrightarrow\;\text{无量纲熵}=\frac{S}{k_B}
$$

——**$S/k_B$ 已经是无量纲的（以 bit 为单位）**。这是 §29"$k_B$ 的消解"的逆命题：熵与信息的同一性 $S=k_B\times(\text{bit 数})$ 在量纲层面是**恒等式而非近似**。

#### (e) Buckingham Π 定理的信息力学对应 [定理]

**Π 定理**：$n$ 个物理量、$k$ 个独立量纲 ⟹ 恰好 $n-k$ 个独立无量纲组合。

**信息力学版本**：$\Phi$ 的核 $\ker\Phi=\{$无量纲量$\}$ 由**纯信息比值**构成——**无量纲群 ⟺ 信息不变量**。实例：精细结构常数

$$
\alpha=\frac{e^2}{4\pi\epsilon_0\hbar c}
$$

不含 $k_B$、$G$——它只度量**信息流耦合的相对强度**（§11），与投影标度无关。所有基本无量纲常数（$\alpha$、质量比 $m_p/m_e\approx1836$、混合角）都是 $\ker\Phi$ 的元素：**它们是唯一"真正物理"的常数，其余常数都是单位制的人为产物**。

#### (f) 结构总结：纤维丛、$k_B$ 特殊性与量纲封闭性

**纤维丛图景**：底空间 $B=\{$投影极限：经典、量子、相对论、热力学$\}$；纤维 $F=\{$单位制选择：SI、自然单位、cgs、…$\}$；截面 $s:B\to F$ 的选取 = 常数数值的选取。**物理常数不是底空间的内禀几何——它们是截面的坐标**。

**Planck 单位制的不完整性**：$c=\hbar=G=1$ 吸收了三常数，但 $k_B$ 未被吸收——Planck 温度 $T_P=\sqrt{\hbar c^5/(Gk_B^2)}$ 的定义里仍含 $k_B$，是混合单位制。真正完备的是**信息自然单位制** $\hbar_{\text{info}}=c_{\text{info}}=G_{\text{info}}=k_B^{\text{info}}=1$：一切量皆为 bit 幂次，$k_B$ 不再独立存在（§29）。

**量纲封闭性定理** [自洽性验证]：本文出现的全部物理量——力、能量、温度、熵、作用量、质量、长度、时间、压强、耦合常数、场强、电荷、曲率——的量纲向量都属于 $\operatorname{span}\{\Phi(c),\Phi(\hbar),\Phi(k_B),\Phi(G)\}$（由 (d) 的 $\det M=2\neq0$ 自动保证）。**信息力学的量纲结构自洽封闭**。$\square$

**遗留预言**：$k_B$ 在框架中仍由**定义**（信息温度 ⟷ 物理温度的兑换率）而非**推导**给出——其微观起源（单个信息孤子的熵-能交换量子）是框架尚待完成的第一原理计算（见 §34 清单）。

---

## 第十部分：诚实评估——严格性账本与可证伪预言

### §33 严格性账本

| **结论** | **级别** | **依赖** |
| --- | --- | --- |
| Fisher 度量 = KL 二阶主部（定理 4.1） | 数学定理 | 正则性条件 |
| Chentsov 唯一性 | 数学定理（引用） | Markov 态射范畴 |
| Cramér–Rao 下界 | 数学定理 | 无偏性 |
| 测地方程 + 自然梯度（§7–8） | 标准推导 | 拉格朗日量形式 |
| 光速不变（§10–11） | 框架内定理 | Lorentz 型投影 + Chentsov |
| Born 规则 = 条件化（§13） | 框架内推导 | 分割假设 |
| $W/Z/\gamma$ 质量代数（§17） | 标准模型代数 | IB 相变 ⟹ 电破缺 |
| 缠绕能 $E\propto Q^2$（§16） | 标准推导 | 涡旋构型假设 |
| 等效原理（§18） | 定义同一性 | 信息 Einstein 方程 |
| 信息 Einstein 方程（§20） | 结构推导 | EH 型作用量选择 |
| 信息塔发散定理（§22） | 框架内推导 | 累积量对应 |
| 力律 Coulomb/Yukawa（§24） | 标准推导 | 传播子计算 |
| 渐近自由数值（§25.1） | 标准物理 | 单圈 RG |
| 禁闭弦张力（§25.2） | 量纲估算 | Anderson 对偶假设 |
| 热力学四定律（§26–28） | 定理 | Gibbs 分布 + DPI |
| $T\propto a^{-1}$、$\Lambda\propto T^4$（§30） | 推导 | 绝热 + 状态方程 |
| $(3+1)$ 维（§31） | 论证 | 束缚态稳定性 |
| $K$ 的拓扑选择（§31） | **猜想** | 开放问题 |
| 量纲封闭性与四常数投影起源（§32） | 线性代数定理 + 定义性对应 | $\det M=2\neq0$；$G$ 数值仍为拟合 |

### §34 未解决问题清单

1. **DPI ⟹ 变分原理的逐字证明**（§6）：当前是论证 + 形式锁定，"极值曲线 ⟺ 最小信息损失"的泛函分析证明缺失；
2. **Lorentz 型投影的来源**：为何 Fisher 度量的物理投影是 $(3+1)$ Lorentz 而非 Riemann——`information time` 文档给出启发式，需要严格化；
3. **$K$ 的几何确定**（§31 猜想 A/B/C）：完全开放；
4. **$\mathcal{V}_{\text{info}}=I(X;T)-\beta I(T;Y)$ 的唯一性**：为何不是别的信息泛函——只有"同时编码压缩与保真"的启发性论证；
5. **数值预言能力**：$\alpha\approx1/137$、三代费米子、质量谱——框架给出结构，未给出第一原理数值。

### §35 可证伪预言汇总（全部来自前文推导）

| **预言** | **来源** | **检验** |
| --- | --- | --- |
| 质子不衰变（拓扑电荷守恒） | §16 | 超级 Kamiokande（已限制 $\tau>10^{34}$ yr，支持） |
| 中微子质量 $\sim m_D^2/M_{\text{info}}\sim10^{-3}$ eV | §19 | KATRIN、无中微子双 $\beta$ 衰变 |
| 顶夸克质量窗口 $Q_t\lesssim1$（拓扑下界附近） | §16 | 已测量 173 GeV，一致性窗口开放 |
| $G_{\text{eff}}$ 随宇宙学时间漂移 $\propto T_{\text{info}}^2$ | §21 | CMB + BBN + 引力波标准汽笛 |
| $\alpha$ 微漂移 $\dot\alpha/\alpha<10^{-17}$/yr 边界处 | §31 | 原子钟、Quasar 吸收谱 |
| 引力波色散修正 $\omega^2=c^2k^2+\xi k^4$ | §22 | Einstein Telescope、LISA |
| TeV 区第二标量 $h'$（IB 多峰相变） | §17 | LHC Run-3/HL-LHC |
| 弦张力-局域化长度标度关系 $\sigma\xi_A^2\approx1$ | §25.2 | 格点 QCD 对比 |

---

## 结语：一句话总结

$$
\underbrace{\text{DPI}}_{\text{公理}}\;\Rightarrow\;\underbrace{g_{ij}=\partial_i\partial_jD_{\text{KL}}}_{\text{定理 4.1}}\;\Rightarrow\;\underbrace{\mathcal{L}_{\text{info}}}_{\text{唯一形式}}\;\Rightarrow\;\begin{cases}\text{光速}=\text{Fisher 锥}\\ \text{量子}=\text{有限温度 Gibbs}\\ \text{质量}=\text{缠绕}+\text{破缺}\\ \text{引力}=\text{Fisher 曲率}\\ \text{规范力}=\text{Noether 流}\\ \text{热力学}=\text{DPI 统计极限}\end{cases}
$$

物理定律不需要被"解释"为信息现象——**它们本来就是同一个信息几何在不同极限下的投影**。本文档把这条链条的每一环都算到了代数层面；链条中每一处不是推导而是假设的地方（§34），都已明确标出。

---

# 附录 A：本文使用的数学工具（教学篇）

本附录把正文推导中反复用到的数学工具逐一讲清楚，每个工具给出"定义 → 操作规则 → 与正文对应的完整算例"。读完本附录，正文中的所有计算都应能独立复现。

## A.1 高斯积分与高斯矩

**正文使用处**：例 2.2（$\mathbb{E}[(x-\mu)^4]=3\sigma^4$）、§29（配分函数的 Gaussian 积分）。

**基本公式**：

$$
\int_{-\infty}^{\infty}e^{-ax^2}dx=\sqrt{\frac{\pi}{a}},\qquad
\int_{-\infty}^{\infty}x^2e^{-ax^2}dx=\frac{1}{2a}\sqrt{\frac{\pi}{a}},\qquad
\int_{-\infty}^{\infty}x^4e^{-ax^2}dx=\frac{3}{4a^2}\sqrt{\frac{\pi}{a}}
$$

**推导技巧（参数微分法）**：对 $I(a)=\int e^{-ax^2}dx=\sqrt{\pi/a}$ 两边求 $-\partial/\partial a$：

$$
-\frac{\partial I}{\partial a}=\int x^2e^{-ax^2}dx=\frac{1}{2a}\sqrt{\frac{\pi}{a}}
$$

再求一次得 $x^4$ 矩。一般地，$-\partial_a$ 作用一次就在被积函数中"降下"一个 $x^2$。

**算例（正文的 $\mathbb{E}[(x-\mu)^4]=3\sigma^4$）**：高斯分布 $p(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/2\sigma^2}$，令 $y=x-\mu$，$a=1/(2\sigma^2)$：

$$
\mathbb{E}[y^4]=\frac{\int y^4 e^{-ay^2}dy}{\int e^{-ay^2}dy}=\frac{\frac{3}{4a^2}\sqrt{\pi/a}}{\sqrt{\pi/a}}=\frac{3}{4a^2}=\frac{3}{4}\cdot(2\sigma^2)^2=3\sigma^4
$$

奇数阶中心矩全部为零（被积函数是奇函数）——这正是例 2.2 中 $g_{\mu\sigma}=0$ 的原因。

## A.2 多元 Taylor 展开

**正文使用处**：定理 4.1（KL 散度的二阶展开）、§21（弱场展开）。

**规则**：$f(\theta+d\theta)$ 在 $\theta$ 处展开：

$$
f(\theta+d\theta)=f(\theta)+\partial_if\,d\theta^i+\frac12\partial_i\partial_jf\,d\theta^id\theta^j+\mathcal{O}(d\theta^3)
$$

注意二阶项里 $d\theta^id\theta^j$ 对 $i,j$ 对称，所以只有 Hessian 的对称部分有贡献。

**算例（正文定理 4.1 的内层展开）**：$f(d\theta)=\ln\frac{p(x|\theta+d\theta)}{p(x|\theta)}$，把 $d\theta$ 当小量：

$$
f(d\theta)=\underbrace{\partial_i\ln p}_{=\,s_i}d\theta^i+\frac12\underbrace{\partial_i\partial_j\ln p}_{\text{Hessian}}d\theta^id\theta^j+\cdots
$$

再对 $p_\theta$ 取期望：一阶项被引理 1.1（$\mathbb{E}[s_i]=0$）杀掉，二阶项被引理 1.2（$\mathbb{E}[\partial_i\partial_j\ln p]=-g_{ij}$）转成度规。**取期望逐项杀项**是信息几何计算的通用节奏。

## A.3 变分法与泛函导数

**正文使用处**：§6–7（$\delta\mathcal{S}_{\text{info}}=0$ → 测地方程）、§20（$\delta\mathcal{S}/\delta g^{ij}$ → Einstein 方程）、§26（最大熵变分）。

**核心规则**：作用量 $\mathcal{S}[q]=\int L(q,\dot q,\tau)d\tau$ 的变分

$$
\delta\mathcal{S}=\int\left[\frac{\partial L}{\partial q}\delta q+\frac{\partial L}{\partial\dot q}\delta\dot q\right]d\tau
=\int\left[\frac{\partial L}{\partial q}-\frac{d}{d\tau}\frac{\partial L}{\partial\dot q}\right]\delta q\,d\tau+\underbrace{\left[\frac{\partial L}{\partial\dot q}\delta q\right]_{\text{边界}}}_{=\,0\text{（端点固定）}}
$$

第二步用了**分部积分**（$\delta\dot q=\frac{d}{d\tau}\delta q$）。$\delta\mathcal{S}=0$ 对任意 $\delta q$ 成立 ⟹ 括号内为零，即 Euler–Lagrange 方程。

**算例（正文 §7 的动能项）**：$L=\frac12g_{ij}(\theta)\dot\theta^i\dot\theta^j$。

$$
\frac{\partial L}{\partial\dot\theta^m}=g_{mj}\dot\theta^j,\qquad
\frac{d}{d\tau}\frac{\partial L}{\partial\dot\theta^m}=g_{mj}\ddot\theta^j+\partial_kg_{mj}\dot\theta^k\dot\theta^j
$$

（第二项来自链式法则：$g_{mj}$ 依赖 $\theta(\tau)$。）

$$
\frac{\partial L}{\partial\theta^m}=\frac12\partial_mg_{ij}\dot\theta^i\dot\theta^j
$$

两者相减、乘 $g^{lm}$、利用 $\dot\theta^k\dot\theta^j$ 的对称性把 $\partial_kg_{mj}$ 对称化，即得 Christoffel 符号——正文 §7 的每一步就是这个模板。

**对度规变分（§20 用）**：把 $g_{ij}$ 当作"$q$"，泛函导数记作

$$
\frac{\delta\mathcal{S}}{\delta g^{ij}(x)}\equiv\lim_{\epsilon\to0}\frac{\mathcal{S}[g+\epsilon h]-\mathcal{S}[g]}{\epsilon}\Bigg|_{\text{沿 }h\text{ 方向的系数}},\qquad h_{ij}\text{ 紧支}
$$

要点恒等式（正文 §20 (a) 的证明模板）：$\delta\det g=\det g\cdot g^{ij}\delta g_{ij}$，来自矩阵恒等式 $\ln\det M=\operatorname{Tr}\ln M$ 的变分。

## A.4 Christoffel 符号与曲率的实操计算

**正文使用处**：例 2.2 注记、例 7.1、§20–21。

**定义与算法**：

$$
\Gamma^l_{\;jk}=\frac12g^{lm}\big(\partial_jg_{mk}+\partial_kg_{mj}-\partial_mg_{jk}\big)
$$

**算例（正文 Gaussian 流形，度规 $g=\mathrm{diag}(1/\sigma^2,\,2/\sigma^2)$，坐标 $(\mu,\sigma)$）**：

度规只依赖 $\sigma$，故只有含 $\partial_\sigma$ 的项非零。算 $\Gamma^\sigma_{\;\mu\mu}$：取 $l=\sigma,j=k=\mu$，

$$
\Gamma^\sigma_{\;\mu\mu}=\frac12g^{\sigma\sigma}\big(\partial_\mu g_{\sigma\mu}+\partial_\mu g_{\mu\sigma}-\partial_\sigma g_{\mu\mu}\big)=\frac12\cdot\frac{\sigma^2}{2}\cdot\Big(0+0-\partial_\sigma\frac{1}{\sigma^2}\Big)=\frac{\sigma^2}{4}\cdot\frac{2}{\sigma^3}=\frac{1}{2\sigma}
$$

算 $\Gamma^\mu_{\;\mu\sigma}$：取 $l=\mu,j=\mu,k=\sigma$，

$$
\Gamma^\mu_{\;\mu\sigma}=\frac12g^{\mu\mu}\big(\partial_\mu g_{\mu\sigma}+\partial_\sigma g_{\mu\mu}-\partial_\mu g_{\sigma\mu}\big)=\frac12\sigma^2\cdot\partial_\sigma\frac{1}{\sigma^2}=\frac12\sigma^2\cdot\Big(-\frac{2}{\sigma^3}\Big)=-\frac{1}{\sigma}
$$

同理 $\Gamma^\sigma_{\;\sigma\sigma}=\frac12g^{\sigma\sigma}\partial_\sigma g_{\sigma\sigma}=\frac{\sigma^2}{4}\cdot\partial_\sigma\frac{2}{\sigma^2}=-\frac{1}{\sigma}$，其余分量为零。**操作口诀**：对角度规 + 单变量依赖 ⟹ 每个 $\Gamma$ 只含一项。

**曲率**：$R^l_{\;ijk}=\partial_i\Gamma^l_{\;jk}-\partial_j\Gamma^l_{\;ik}+\Gamma^l_{\;im}\Gamma^m_{\;jk}-\Gamma^l_{\;jm}\Gamma^m_{\;ik}$；Ricci 张量 $R_{jk}=R^i_{\;jik}$；标量曲率 $R=g^{jk}R_{jk}$。二维流形只有一个独立曲率分量，代入上例可验证 $R=0$（Gaussian 族可展平——例 7.1 换坐标 $(\mu/\sqrt2\sigma,1/\sigma)$ 后度规变平直，是同一事实的两面）。

## A.5 Fourier 变换与围道积分

**正文使用处**：§24（Yukawa 势）、§25.2、§31（Green 函数）。

**三维 Fourier  convention**：

$$
\tilde f(\mathbf{k})=\int f(\mathbf{x})e^{-i\mathbf{k}\cdot\mathbf{x}}d^3x,\qquad f(\mathbf{x})=\int\frac{d^3k}{(2\pi)^3}\tilde f(\mathbf{k})e^{i\mathbf{k}\cdot\mathbf{x}}
$$

核心性质：$\partial_j\leftrightarrow ik_j$，故微分方程 $(-\nabla^2+m^2)A=\delta^{(3)}$ 变成代数方程 $(k^2+m^2)\tilde A=1$。

**算例（正文 §24 的关键积分）**：$I=\int_0^\infty\frac{k\sin kr}{k^2+m^2}dk$。

**第 1 步（偶延拓）**：被积函数是 $k$ 的偶函数 ⟹ $I=\frac12\int_{-\infty}^{\infty}\frac{k\sin kr}{k^2+m^2}dk$。

**第 2 步（指数化）**：$\sin kr=\operatorname{Im}e^{ikr}$ ⟹ $I=\frac12\operatorname{Im}\int_{-\infty}^{\infty}\frac{k\,e^{ikr}}{(k-im)(k+im)}dk$。

**第 3 步（围道闭合）**：$r>0$ 时 $e^{ikr}$ 在上半平面衰减，围道沿实轴 + 上半圆闭合。上半平面只含极点 $k=im$。

**第 4 步（留数定理）**：$\oint=2\pi i\sum\operatorname{Res}$。$k=im$ 处留数：

$$
\operatorname{Res}_{k=im}\frac{k\,e^{ikr}}{(k-im)(k+im)}=\frac{im\,e^{-mr}}{2im}=\frac12e^{-mr}
$$

**第 5 步（合并）**：

$$
I=\frac12\operatorname{Im}\Big[2\pi i\cdot\frac12e^{-mr}\Big]=\frac{\pi}{2}e^{-mr}
$$

这就是正文 Yukawa 势 $A_0(r)=\frac{q}{4\pi}\frac{e^{-mr}}{r}$ 的全部积分工作量。

## A.6 李群、李代数与结构常数

**正文使用处**：§17（电破缺）、§23（Noether 流）、§25（$\beta$ 函数）。

**李代数基础**：李群（如 $SU(2)$）在单位元附近的无穷小生成元 $T^a$ 满足对易关系

$$
[T^a,T^b]=if^{abc}T^c
$$

$f^{abc}$ 是结构常数。$SU(2)$：$T^a=\sigma^a/2$（泡利矩阵之半），$f^{abc}=\epsilon^{abc}$；$SU(3)$：$T^a=\lambda^a/2$（Gell-Mann 矩阵），8 个生成元。

**泡利矩阵**：

$$
\sigma^1=\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad
\sigma^2=\begin{pmatrix}0&-i\\i&0\end{pmatrix},\quad
\sigma^3=\begin{pmatrix}1&0\\0&-1\end{pmatrix}
$$

**算例（验证 $[\sigma^1,\sigma^2]=2i\sigma^3$，即 $f^{123}=1$）**：

$$
\sigma^1\sigma^2=\begin{pmatrix}0&1\\1&0\end{pmatrix}\begin{pmatrix}0&-i\\i&0\end{pmatrix}=\begin{pmatrix}i&0\\0&-i\end{pmatrix}=i\sigma^3,\qquad
\sigma^2\sigma^1=-i\sigma^3
$$

$$
[\sigma^1,\sigma^2]=2i\sigma^3\;\Longrightarrow\;[T^1,T^2]=\frac{1}{4}\cdot2i\sigma^3=i\epsilon^{123}T^3\;\checkmark
$$

**算例（正文 §17 的 $T^a\phi_{\text{vac}}$）**：$\phi_{\text{vac}}=\frac{1}{\sqrt2}\binom{0}{v}$，

$$
T^1\phi_{\text{vac}}=\frac{1}{2\sqrt2}\begin{pmatrix}0&1\\1&0\end{pmatrix}\binom{0}{v}=\frac{1}{2\sqrt2}\binom{v}{0},\qquad
T^3\phi_{\text{vac}}=\frac{1}{2\sqrt2}\begin{pmatrix}1&0\\0&-1\end{pmatrix}\binom{0}{v}=\frac{1}{2\sqrt2}\binom{0}{-v}
$$

正是正文第 2 步列出的四个向量的来源。

**Casimir 与指标（§25 用）**：$SU(N)$ 伴随表示 $C_A=N$，基础表示 $T_F=1/2$。$\beta$ 函数系数 $\frac{11}{3}C_A-\frac43T_Fn_f$ 中的数字就是从这两个群论因子算出来的。

## A.7 信息论四件套：熵、互信息、KL、DPI

**正文使用处**：全文（§4、§14、§27 是密集使用处）。

**定义**（离散版，连续版把求和换积分）：

$$
H(X)=-\sum_xp(x)\log p(x),\qquad D_{\text{KL}}(p\|q)=\sum_xp(x)\log\frac{p(x)}{q(x)},\qquad I(X;Y)=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)}
$$

**三个恒等式**（§27 证明的零件）：

1. $I(X;Y)=H(X)-H(X|Y)$（互信息 = 不确定性的减少）；
2. 链式法则：$I(X;Y,T)=I(X;T)+I(X;Y|T)=I(X;Y)+I(X;T|Y)$；
3. 条件化使互信息不增（Markov 链 $X\to T\to Y$ ⟹ $I(X;Y|T)=0$）。

**算例（DPI 的数值演示）**：$X$ 均匀比特，$T$ 经擦除概率 $\epsilon=0$ 的信道（$T=X$），$Y$ 再经 $\epsilon=1/2$ 的全擦除信道（$Y$ 与 $T$ 独立）。则

$$
I(X;T)=1\text{ bit},\qquad I(X;Y)=0,\qquad I(X;Y)\le I(X;T)\;\checkmark
$$

且等号只在无损失处理时成立——§27"等号 ⟺ 充分统计量"的直观版本。

**Markov 条件的形式化**：$X\to T\to Y$ 指 $p(x,t,y)=p(x)p(t|x)p(y|t)$（$Y$ 只通过 $T$ 依赖 $X$）。由此 $p(x|t,y)=p(x|t)$，于是 $I(X;Y|T)=0$——这是 §27 第 1 步证明用到的唯一输入。

## A.8 随机矩阵乘积与 Lyapunov 指数（局域化的数学）

**正文使用处**：§25.2（色禁闭）、`physical to ai derivation` §3（梯度消失）。

**基本对象**：随机矩阵乘积 $M_L=W_LW_{L-1}\cdots W_1$（$W_l$ 独立同分布随机矩阵）。Furstenberg 定理：范数的对数渐近线性增长，

$$
\lim_{L\to\infty}\frac1L\log\|M_Lv\|=\lambda_{\max}\quad(\text{最大 Lyapunov 指数，与 }v\text{ 无关，几乎必然})
$$

$\lambda_{\max}<0$ ⟹ 乘积指数衰减 ⟹ 信号（信息流/梯度）被局域化，特征长度 $\xi=1/|\lambda_{\max}|$。

**算例（一维可严格计算版）**：$W_l$ 取标量，$\log|W_l|$ 均值 $\mu<0$、方差 $s^2$。由大数定律 $\frac1L\log|M_L|=\frac1L\sum\log|W_l|\to\mu$，故 $\xi=1/|\mu|$。正文中 $\xi_{\text{loc}}\propto1/\log(\sigma_w^2\mathbb{E}[\sigma'^2])$ 就是这个结构的矩阵版——对数均值为负时深度网络上的信息传播被指数压制。

---

# 附录 B：本文使用的物理理论方法（教学篇）

## B.1 最小作用量原理

**正文使用处**：§6–7、§20。

**方法**：物理轨迹 = 作用量 $\mathcal{S}=\int L\,dt$ 的驻值曲线。操作流程固定：写 $L$ → 变分 → 分部积分 → EL 方程（A.3 已给模板）。

**算例（自由粒子 → 匀速直线）**：$L=\frac12m\dot x^2$。$\frac{\partial L}{\partial\dot x}=m\dot x$，$\frac{\partial L}{\partial x}=0$ ⟹ $m\ddot x=0$。正文 §7 的测地方程就是这个算例在弯曲流形（$m\delta_{ij}\to g_{ij}(\theta)$）上的推广——多出来的 $\Gamma$ 项完全来自度规的 $\theta$ 依赖。

**为什么物理定律有这个形式**：作用量是被积量的"总量"，驻值条件把全局约束转成局域微分方程。信息力学中，作用量 = 总信息代价，驻值 = DPI 自洽（§6）。

## B.2 对称性与 Noether 定理

**正文使用处**：§23（规范流）。

**定理陈述**：作用量在连续变换 $q\to q+\epsilon\,\delta q$ 下不变 ⟹ 存在守恒量

$$
Q=\frac{\partial L}{\partial\dot q}\,\delta q,\qquad \dot Q=0
$$

**推导（一行版）**：$\delta\mathcal{S}=\int[\text{EL 项}]\delta q\,dt+[\frac{\partial L}{\partial\dot q}\delta q]_{\partial}$；EL 项沿真实轨迹为零，对称性要求 $\delta\mathcal{S}=0$ ⟹ 边界项不随端点移动 ⟹ $Q$ 守恒。

**算例（时间平移 → 能量）**：$\delta q=\dot q\,\epsilon$（$q(t)\to q(t+\epsilon)$）。守恒量

$$
Q=\frac{\partial L}{\partial\dot q}\dot q-L\equiv H
$$

对 $L=\frac12m\dot x^2-V(x)$：$H=\frac12m\dot x^2+V$。正文 §23 把同一模板用到内部空间 $K$ 的 isometry 上：$\delta\Phi=i\epsilon^\alpha T_\alpha\Phi$ ⟹ 守恒流 $j_\alpha^a$——规范荷的守恒与时间平移 ⟹ 能量守恒是**同一个定理**的两个实例。

**口诀**：时间平移 ⟹ 能量；空间平移 ⟹ 动量；旋转 ⟹ 角动量；内部 isometry ⟹ 规范荷。

## B.3 自发对称破缺与 Higgs 机制

**正文使用处**：§17。

**方法骨架**：

1. 势能 $V(\phi)=-\mu^2|\phi|^2+\lambda|\phi|^4$（$\mu^2>0$）：原点是不稳定极大，真空在 $|\phi|=v/\sqrt2$ 的简并圆环上；
2. 系统"选择"一个真空方向 ⟹ 整体对称性破缺；
3. 把场按真空方向展开 $\phi=\phi_{\text{vac}}+(\text{涨落})$，代入动能项 $|D_\mu\phi|^2$；
4. $\phi_{\text{vac}}$ 处的协变导数给出规范场的**二次项** = 质量项；正交方向（残余对称）无此项 = 无质量玻色子。

**算例（离散玩具版：$\mathbb{Z}_2$ 对称破缺）**：$V(\phi)=-\phi^2+\phi^4$。$V'=-2\phi+4\phi^3=0$ ⟹ $\phi=0$（极大，$V''=-2<0$）或 $\phi=\pm1/\sqrt2$（极小，$V''=4>0$）。系统在两个真空中二选一，$\phi\to-\phi$ 对称性被破缺。连续对称版本的区别：破缺方向构成简并流形（圆环/球面），沿简并方向的涨落无能量代价 = Goldstone 模；有规范场时这些模被"吃掉"变成规范场质量（正文 §17 第 2 步的代数）。

## B.4 Green 函数与传播子

**正文使用处**：§24、§31。

**方法**：线性方程 $\hat L\,G(\mathbf{x})=\delta^{(3)}(\mathbf{x})$ 的解 $G$ 是"点源响应"；任意源的解 = 源与 $G$ 的卷积。

**算例 1（三维 Poisson 方程）**：$-\nabla^2G=\delta^{(3)}$。球对称下 $\nabla^2G=\frac{1}{r^2}\partial_r(r^2\partial_rG)$（$r>0$），解 $G=A/r+B$。积分定常数：对半径 $\epsilon$ 小球积分，$-\oint\nabla G\cdot d\mathbf S=1$ ⟹ $-4\pi\epsilon^2\cdot(-A/\epsilon^2)=1$ ⟹ $A=1/4\pi$。故

$$
G(\mathbf{x})=\frac{1}{4\pi r}\quad\Longrightarrow\quad V(r)=\frac{q_1q_2}{4\pi r}
$$

**算例 2（d 维衰减律，§31 用）**：球对称 Laplace 方程 $\frac{1}{r^{d-1}}\partial_r(r^{d-1}\partial_rG)=0$ ⟹ $r^{d-1}\partial_rG=C$ ⟹ $\partial_rG=Cr^{1-d}$ ⟹ $G\propto r^{2-d}$（$d\neq2$）、$G\propto\ln r$（$d=2$）。这就是 §31 维数论证的全部输入：$d=3$ 时 $G\propto1/r$，衰减与可达性的唯一平衡点。

## B.5 量纲分析

**正文使用处**：§16（$m_{\text{topo}}\sim Q^2\hbar_{\text{info}}/\xi_K$）、§25.2（$\sigma\sim\Lambda_{\text{QCD}}^2$）、§30（$\rho_\Lambda\sim T^4$）；**量纲与四常数起源的完整理论分析见正文 §32**。

**方法**：自然单位制 $\hbar=c=k_B=1$ 中只剩一个独立量纲（取能量），$[\text{长度}]=[\text{时间}]=\text{GeV}^{-1}$，$[\text{质量}]=\text{GeV}$。

**算例 1（弦张力）**：弦张力 $\sigma$ 是"单位长度的能量"：$[\sigma]=\text{GeV}/\text{GeV}^{-1}=\text{GeV}^2$。理论中唯一的标度是 $\Lambda_{\text{QCD}}\approx0.2$ GeV ⟹ $\sigma\sim\Lambda_{\text{QCD}}^2\approx0.04\text{ GeV}^2$。换算：$1\text{ GeV}^{-1}=0.197$ fm ⟹ $\sigma\approx0.04/0.197\approx0.9$ GeV/fm——正文 §25.2 的数值。

**算例 2（暗能量密度）**：能量密度 $[\rho]=\text{GeV}^4$。若唯一标度是信息温度 $T_{\text{info}}$ ⟹ $\rho_\Lambda\sim T_{\text{info}}^4$。$T_{\text{info}}\sim10^{-3}$ eV ⟹ $\rho_\Lambda\sim(10^{-3}\text{ eV})^4=(10^{-12}\text{ GeV})^4\sim10^{-48}\text{ GeV}^4$——与观测暗能量密度 $\sim10^{-47}\text{ GeV}^4$ 同量级。

**纪律**：量纲分析给出的是**标度与量级**，数值系数需要完整计算——正文对 $\sigma$ 的处理（标度推导 + 格点数值对比）是正确用法。

## B.6 微扰论与线性化

**正文使用处**：§21（弱场引力）、§22（量子化困难）。

**方法**：把场分成背景 + 小涨落 $g=\bar g+h$（$|h|\ll1$），方程按 $h$ 的阶数展开，逐阶求解。一阶方程是线性的，可用 Green 函数；高阶产生相互作用顶点。

**算例（正文 §21 第 1 步）**：$g_{\mu\nu}=\eta_{\mu\nu}+h_{\mu\nu}$，$\Gamma^\mu_{\;\nu\rho}$ 本身是 $\mathcal{O}(h)$，故线性化时

$$
\Gamma^i_{\;00}=\frac12\eta^{i\lambda}(2\partial_0h_{\lambda0}-\partial_\lambda h_{00})+\mathcal{O}(h^2)
$$

静态场 $\partial_0h=0$ ⟹ $\Gamma^i_{\;00}=-\frac12\partial_ih_{00}$。**常见错误**：把 $g^{\mu\lambda}$ 也展开会带来 $h$ 的修正项，但那些是 $\mathcal{O}(h^2)$，线性化时全部丢弃——"只保留 $h$ 的一阶"是每一步的纪律。

**失效模式**：微扰论假设高阶项越来越小。§22 的信息塔定理指出，引力的 $n$ 点顶点对应 $2n$ 阶累积量，阶数越高没有越小的结构 ⟹ 微扰展开不收敛 ⟹ 这是方法适用性问题，不是计算错误。

## B.7 稳定性分析

**正文使用处**：§31（维数与束缚态）、§16（拓扑稳定性）。

**方法**：平衡点附近把能量展开到二阶，$E=E_0+\frac12k(\delta x)^2+\cdots$；$k>0$ 稳定，$k<0$ 不稳定，$k=0$ 需看高阶。

**算例（正文 §31 的氢原子维数判据）**：$d$ 维类氢体系有效径向能量（$C>0$ 为耦合常数）

$$
E(r)=-Cr^{2-d}+\frac{L^2}{2mr^2}
$$

- $d=3$：$E=-C/r+L^2/(2mr^2)$。$E'=C/r^2-L^2/(mr^3)=0$ ⟹ $r_*=L^2/(mC)$，$E''(r_*)=-2C/r_*^3+3L^2/(mr_*^4)=C^4m/L^4>0$ ⟹ **稳定极小**（Bohr 半径）；
- $d=4$：$E=-C/r^2+L^2/(2mr^2)=(L^2/2m-C)/r^2$——单调函数，无极小：$L^2/2m>C$ 时粒子飞向无穷，反之坍缩到 $r=0$。**无稳定原子**；
- $d=2$：$E=-C\ln r+\cdots$ 对数势，束缚态性质病态。

结论：$d=3$ 是"吸引势不太弱、离心势垒能挡住坍缩"的唯一窗口。这是正文"空间三维性"论证的完整计算。

## B.8 重整化群与跑动耦合

**正文使用处**：§25.1。

**方法**：物理观测量不能依赖人为截断/重整化标度 $\mu$，这给出耦合常数随 $\mu$ 的"跑动"方程 $\mu\,dg/d\mu=\beta(g)$。

**算例（正文 $\alpha_s$ 的积分，含全部代数）**：$\beta$ 方程 $\frac{d\alpha_s}{d\ln\mu}=-b\alpha_s^2$（$b=(33-2n_f)/6\pi$ 的约定下）。分离变量：

$$
\frac{d\alpha_s}{\alpha_s^2}=-b\,d\ln\mu\;\Longrightarrow\;-\frac{1}{\alpha_s}=-b\ln\mu+C
$$

定积分常数：选 $\alpha_s$ 在 $\mu=\Lambda_{\text{QCD}}$ 处发散（$1/\alpha_s\to0$）⟹ $C=b\ln\Lambda_{\text{QCD}}$。故

$$
\frac{1}{\alpha_s(\mu)}=b\ln\frac{\mu}{\Lambda_{\text{QCD}}}\;\Longrightarrow\;\alpha_s(\mu)=\frac{1}{b\ln(\mu/\Lambda_{\text{QCD}})}=\frac{12\pi}{(33-2n_f)\ln(\mu/\Lambda_{\text{QCD}})}
$$

**物理图像**：$\beta<0$ ⟹ 高能（短距离）耦合弱（渐近自由，可用微扰论）；$\mu\to\Lambda_{\text{QCD}}$ 时耦合发散（微扰失效，禁闭区）。一个超越函数的积分常数 $\Lambda_{\text{QCD}}$ 承载了 QCD 的全部非微扰标度——这就是"量纲嬗变"（dimensional transmutation）。

## B.9 WKB 近似与特征面

**正文使用处**：§11（光速 = 特征速度）。

**方法**：波动方程 $\Box A=0$ 的高频极限。试探解 $A=a(\mathbf{x})e^{i\Phi(\mathbf{x})/\epsilon}$，$\epsilon\to0$ 形式化高频。代入后按 $\epsilon$ 的阶数收集：

$$
\Box A=\Big[\frac{1}{\epsilon^2}g^{\alpha\beta}\partial_\alpha\Phi\,\partial_\beta\Phi\cdot(-a)+\frac{i}{\epsilon}(2g^{\alpha\beta}\partial_\alpha\Phi\,\partial_\beta a+a\,\Box\Phi)+\Box a\Big]e^{i\Phi/\epsilon}=0
$$

- $\epsilon^{-2}$ 阶（eikonal 方程）：$g^{\alpha\beta}\partial_\alpha\Phi\,\partial_\beta\Phi=0$——**波前沿的特征面方程**，与度规的类光锥完全一致；
- $\epsilon^{-1}$ 阶：振幅输运方程（沿射线守恒）。

**结论的使用方式**（正文 §11）：任何无质量波动方程的高频波前都沿度规的特征面传播 ⟹ "信息传播的最大速度"不需要从波动方程的解里猜，直接从度规的代数（$g^{\alpha\beta}k_\alpha k_\beta=0$）读出。声波、光波、引力波共用这一个结构。

## B.10 正则量子化与约束系统（ADM 模板）

**正文使用处**：§22（Wheeler–DeWitt 方程）。

**方法流程**（任何约束 Hamilton 系统通用）：

1. **Legendre 变换**：$p=\partial L/\partial\dot q$，$H=p\dot q-L$；
2. **识别初级约束**：若 $L$ 不含某变量 $\mathcal{N}$ 的时间导数，则其共轭动量 $\pi_{\mathcal{N}}=\partial L/\partial\dot{\mathcal{N}}=0$ 是约束（$\mathcal{N}$ 是 Lagrange 乘子）；
3. **Lagrange 乘子的变分给出次级约束**：$\delta\mathcal{S}/\delta\mathcal{N}=0$ ⟹ $\mathcal{H}=0$；
4. **Dirac 量子化**：$\hat p=-i\hbar\,\partial/\partial q$，约束变成波泛函的算符方程 $\hat{\mathcal{H}}\Psi=0$。

**算例（参数化粒子——WDW 方程的玩具模型）**：取作用量 $\mathcal{S}=\int[\frac{1}{2\mathcal{N}}\dot q^2-\mathcal{N}V(q)]d\tau$（$\mathcal{N}$ 是"重参数化"的 Lagrange 乘子）。

1. $p_q=\dot q/\mathcal{N}$，$\pi_{\mathcal{N}}=0$（初级约束）；
2. $H=p_q\dot q-L=\mathcal{N}[\frac12p_q^2+V(q)]$——总 Hamilton 量正比于约束；
3. $\delta\mathcal{S}/\delta\mathcal{N}=0$ ⟹ $\frac12p_q^2+V=0$（次级约束 = 能量固定）；
4. 量子化：$[-\frac{\hbar^2}{2}\partial_q^2+V(q)]\Psi(q)=0$——**约束方程就是 Schrödinger 方程**。

正文 §22 对引力做的完全是这个模板：$\mathcal{N}$ = lapse、约束 $\mathcal{H}_{\text{info}}=0$、量子化后 $\hat{\mathcal{H}}_{\text{info}}\Psi[\gamma]=0$ 就是 Wheeler–DeWitt 方程。"时间消失"（$\Psi$ 不依赖外部时间）是所有重参数化不变系统的共同命运——玩具模型里同样没有独立的 $\tau$ 演化。

---

**附录使用说明**：正文每个标注 [推导] 的段落都只使用了本附录列出的工具组合。复现建议路径：A.1→A.2→A.7（信息几何地基）→ A.3→A.4（动力学）→ B.2→B.3（对称性与质量）→ A.5→B.4（力律）→ B.8（跑动耦合）→ B.6→B.10（引力与量子化）。


