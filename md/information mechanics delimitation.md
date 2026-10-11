# 信息力学框架划界调研：真新增与既有结论重述

**本文档的地位**：本文是项目理论部分的**外部划界调研**（2026-09-16），与主文档 `information mechanics derivations.md` 配套使用。**本文不改动主文档任何内容**，只回答一个问题：主文档九部分的主张中，哪些是新增，哪些是既有结论的重新发现或重述。

**调研方法**。按项目纪律"先检索后写"执行，四路并行对抗性检索：

1. "物理从 Fisher 信息涌现"总程序 + 光速/时空签名；
2. 引力与宇宙学；
3. 量子与信息几何；
4. 信息热力学与粒子谱。

检索类目刻意覆盖 cs.LG 之外的物理类目：gr-qc、hep-th、hep-ph、quant-ph、cond-mat.stat-mech、math-ph、astro-ph、physics.gen-ph、physics.hist-ph。方法要求：默认被调研主张"很可能已被做过"，以**找出占位者**为目标，而不是确认空白。

**一句话结论**：主文档纲领是"从信息导出物理定律"这一既有研究纲领的**重新发现**。最接近的整体先行方案是 Frieden《Physics from Fisher Information》(CUP 1998)，次接近是 Caticha 的熵动力学。九部分中：**3 部分属教科书重述，4 部分有直接同行纲领，2 部分存在极窄可能缝隙**（A 组）；另有 **4 项超出"新颖性"范畴的技术风险**（D 组）。

---

## A. 可能真新增（六条窄缝，均需先确认）

| 主张 | 位置 | 状态 | 说明 |
| --- | --- | --- | --- |
| IB 相变 ⇒ 电弱破缺（W/Z/γ 质量代数） | §17 | **未找到占位者** | 垂直相邻只有涌现规范对称性综述（Bass 2022）与 string-net。但 $m_W=gv/2$、$m_Z=v\sqrt{g^2+g'^2}/2$ 本身是标准模型结果，"导出"实质是重推已知代数 |
| Fisher 作用量做**含物质源的完整变分** ⇒ Einstein 方程 | §20 | 未见完整占位 | Caticha（arXiv:1909.09657）已做**真空**情形：最大熵 ⇒ 信息几何度规 ⇒ 真空 Einstein 方程 |
| "量子不可逆性 = DPI"的**字面等同** | §14 | 表述层窄缝 | 内容被 Lindblad 1975 / Uhlmann 1977 / Petz 单调性定理覆盖；通行写法是"单调性 ⇒ H 定理" |
| "DPI ⇒ Fisher ⇒ 唯一拉氏量 ⇒ 测地线"**单链组装** | §0–§8 | 表述层窄缝 | Caticha 的 Hamilton–Killing 流已实质覆盖"几何 ⇒ 唯一动力学"这一环 |
| "无质量 Noether 流恰以锥速传播" | §11 | 未见先行者 | 但它本是双曲 PDE 特征线理论的标准结论，属平凡环 |
| "信息 Wheeler–DeWitt 方程"、"信息塔发散定理"命名命题 | §22 | 未见同名者 | 内核已被占：热时间假说（Connes–Rovelli 1994）、熵时间（arXiv:2509.07745）、面积律阻塞（Qi, Nature Physics 2018） |

---

## B. 既有同行纲领（必须划界，否则会被判"重述"）

| 主文档对应 | 先行纲领 | 关键文献 |
| --- | --- | --- |
| 总程序（Fisher 信息 ⇒ 物理定律） | **Frieden EPI** | Frieden, *Physics from Fisher Information*, CUP 1998；Frieden & Soffer, PRE 52, 2274 (1995)；Reginatto, PRA 58, 1775 (1998) |
| 唯一几何 + 唯一动力学 + Born 规则 | **Caticha 熵动力学** | arXiv:1403.3822；arXiv:1908.04693（Entropy 21, 943, 2019）；arXiv:2208.02156 |
| 量子态 = Gibbs、Born = 条件化 | 量子贝叶斯 / QBism | Caves–Fuchs–Schack, PRA 65, 022305 (2002)；Leifer–Spekkens, PRA 88, 052130 (2013) |
| 引力 = 热力学（§20） | Jacobson / Padmanabhan / Sakharov | gr-qc/9504004 (1995)；hep-th/0205278 (2002)；Sakharov 1967 |
| 引力 = 熵力（§21） | Verlinde | arXiv:1001.0785 (2011)；arXiv:1611.02269 (2016) |
| 引力 = 纠缠 | Ryu–Takayanagi / Van Raamsdonk / Jacobson | arXiv:1312.7856；arXiv:1505.04753 (2016) |
| $g_{\mu\nu}$ = Fisher 信息矩阵 | Caticha / Matsueda / Miyamoto | arXiv:1909.09657；arXiv:1408.6633；arXiv:1205.3211 |
| 第二定律 = DPI（§27） | Merhav / Lindblad | arXiv:0903.1484 (2009)；Lindblad 1975 |
| $\Lambda \propto T^4$（§30） | Alicki–Barenboim–Jenkins | arXiv:2307.04800 |
| $(3+1)$ 维（§31） | Tegmark | Class. Quantum Grav. 14, L69 (1997) |
| 不可逆性 = 信息论机制 | Zurek einselection / 量子达尔文主义 | Entropy 24, 1520 (2022) |
| 从信息公理重建量子论 | Chiribella–D'Ariano–Perinotti / Hardy | arXiv:1011.6451 (2011)；quant-ph/0101012 |
| 相对论色散关系从 Fisher–Rao 几何 | Yoo-Kong | arXiv:2601.12764 (2026) |

---

## C. 属"引用既有定理"（正确引用即可，勿作创新主张）

Chentsov 唯一性（1982 原始定理）；Fisher 度量 = KL 散度二阶主部（Amari 教科书）；Cramér–Rao 下界；Amari 自然梯度（1998）；Noether 定理（1918）；Yang–Mills（1954）；单圈 RG（Gross–Wilczek / Politzer 1973）；Wilson 禁闭（1974）；标准模型电弱质量代数；最大熵原理（Jaynes 1957）；涨落–耗散定理（Callen–Welton 1951）；相对熵单调性（Lindblad 1975 / Uhlmann 1977 / Petz）。

拓扑质量（Schwinger 1962；Jackiw–Rebbi 1976）与 Skyrmion = 重子（Witten 1983）亦属既有结果。

---

## D. 四项技术风险

### D1｜Lorentz 签名的来源：此处不是推导，而是假设 + 张量恒等式

**读源后的确认（比初次检索更严重）**：§9 直接**设定**"Fisher 度量取 Lorentz 型线元（时向为负号）"，并注明"投影结构的来源见 `information time` 文档"——即 Lorentz 结构在第三部分是**被假设**的。§10 的"定理 10.1（$c_{\text{info}}$ 的坐标不变性）"其推导实质为：线元是坐标标量，故 $ds^2=0\Leftrightarrow ds'^2=0$，且"若 $c'\neq c$ 则存在类光变类空"——这两步都只是**"同一个度规张量"这一前提的直接后果**，对任何伪黎曼度规都成立（类光锥是几何对象）。它**不**从信息论导出洛伦兹不变性，而是复述"锥是几何的"。

**准确级别**：第三部分应为"[推导]（**给定** Lorentz 型投影）"，而"投影为何是 Lorentz 型"才是真正的缺口（§34 开放问题 2）。

**该缺口已有障碍性文献**：Fisher 度量恒为半正定（Euclidean 签名）；非退化度规的**签名局部恒定**，从正定到 Lorentz 必经过**退化面**；Erdmenger–Grosvenor–Jefferson（arXiv:2001.02683, JHEP 2020）明文指出从 Fisher 度量出发 Lorentz 签名只能**事后 Wick 旋转**。检索到的一条可行机制是用**复值**概率分布（Calmet & Calmet, math-ph/0403043, 2004）。

**建议**：不要把"光速不变"写成由 Fisher 锥导出的定理。改为"给定 Lorentz 型投影，类光锥的坐标不变性是张量恒等式"，并把 Lorentz 投影的来源作为醒目的一级开放问题（要么给出机制，要么明确列为**输入假设**）。

### D2｜§16：$E\propto Q^2$ 标度本身是标准结果，真正的问题在数值步骤

**先更正一条误判**：初次检索报告称 $E\propto Q^2$ 与已知拓扑缺陷标度（$E\propto|Q|$ 的 BPS 界、Faddeev–Hopf 结的 $E\gtrsim|Q|^{3/4}$）冲突。读源后确认**这是跨设定误判**——那些是 3D Skyrmion / 结的能量泛函；§16 做的是 **2D 涡旋 + 二次梯度能**，其 $E=\pi\hbar_{\text{info}}Q^2\ln(R/a)$ **正是 Kosterlitz–Thouless 涡旋能量的标准结果**（对缠绕数 $s$ 有 $E\propto s^2$）。

**因此**：$Q^2$ 标度既不与已知结果冲突，**也不是新结果**——§16 的实质是"2D 涡旋能量的标准计算 + 命名"。真正的问题有三条：

1. **被丢弃的因子恰是决定性的**。定理给出 $E=\pi\hbar_{\text{info}}Q^2\ln(R/a)$，但紧接着结论写成"$m_{\text{topo}}\sim Q^2\times246$ GeV"，**丢掉了 $\pi$ 与 $\ln(R/a)$**。若计入，$Q=1$ 时为 $\pi\ln(R/a)\times246$ GeV；要落到 173 GeV 需 $\ln(R/a)\approx0.22$，而该 $\ln$ 形式的适用前提正是 $R\gg a$（即 $\ln\gtrsim1$）。因子不能一边被当作估算丢弃、一边又靠它支撑"落在窗口"的结论。
2. **"$Q\lesssim1$ 的窗口"实质不存在**。定义 16.1 自己给出 $Q\in\mathbb{Z}$，故 $Q\lesssim1$ 只有 $Q=1$ 一个取值（$Q=0$ 即无孤子）。此时 $m_{\text{topo}}\sim246$ GeV 与实测 173 GeV 相差约 42%——这是**量级相容**，不是数值命中；且这个偏差在计入 $\pi\ln$ 因子后更大。
3. **循环性**。$\hbar_{\text{info}}/\xi_{\text{info}}\approx246$ GeV 中的 $\xi_{\text{info}}$ 取自电弱标度（实验输入）。用实验输入设定标度、再"预言"同一能区的顶夸克质量，不构成独立数值预言。

**可保留的部分**：$m\propto\hbar_{\text{info}}/\xi_{\text{info}}$（能量 = 作用量/长度）是量纲上的必然，属 [推导]；而 $\xi$ 的具体取值与 $Q$ 的取值都是输入。**建议**把 §16 的"数值上落在顶夸克窗口"改为"量纲结构相容"，并下调该节的严格性标注。

### D3｜"Born 规则 = 条件化"有直接反例

Leifer & Spekkens（PRA 88, 052130, 2013）建立量子 Bayes 定理与量子条件化，并**明确论证投影假设不是贝叶斯条件化的特例**。若 §13 以定理形式主张，会被这一条直接引用反驳。此外 Caticha（arXiv:2208.02156）已主张"Born 规则可由熵动力学导出"——即该结论本身亦有先行者。

### D4｜最接近的先行纲领已被同行判为"根本性错误"

Lavis & Streater 2002 书评判定 Frieden 1998"在总体构想与数学细节上根本性错误"；Streater《Lost Causes in Theoretical Physics》持同类立场。审稿人很可能带着这一既有印象阅读本文，因此**必须主动说明本文方案与 Frieden 的差别**（尤其：以 DPI 为唯一公理、依赖 Chentsov 唯一性、不引入 EPI 的信息博弈变分）。

---

## E. 定位建议

可以继续写，但**定位必须改**：

- **不能主张**"物理定律的信息起源"是本文的发现——那是 Frieden / Caticha / Jacobson / Verlinde / Merhav 的既有纲领；
- **可站住的定位**：以 DPI 为唯一公理 + Chentsov 唯一性，**重新组装**既有对应链，并在 A 组六条窄缝中给出具体结果；同时正面处理 D 组四项风险；
- 论文应设一节"与既有信息–物理纲领的划界"，至少在 B 组每一行给出"本文与它的差别是什么"。

---

## F. 检索局限与漏检风险（自报）

1. 四路检索均为公开网页检索，**未做** arXiv / INSPIRE / Semantic Scholar 的全库 API 枚举，也未逐条核对定理编号；
2. Frieden 1998 原书未逐页核读，其对时空度规的处理据目录与二手描述；
3. 术语 "information mechanics" 未见既有同名纲领——这可能是术语被吸收，而非确不存在；
4. 2024–2026 年存在大量低质量同名预印本与博客（如 Footballhedron、节点信息动力学等），索引不全；
5. 因此 A 组"未找到占位者"**不得当作"确认空白"**使用。定稿前应对最接近的先行者（Frieden 1998、Caticha arXiv:2208.02156、arXiv:2510.25553）各做一次**反向追引（cited-by）**核查。

---

## 参考文献

1. Frieden, B. R., *Physics from Fisher Information: A Unification*, Cambridge University Press, 1998.
2. Frieden, B. R. & Soffer, B. H., "Lagrangians of physics and the game of Fisher-information transfer", Phys. Rev. E 52, 2274 (1995).
3. Reginatto, M., "Derivation of the equations of nonrelativistic quantum mechanics using the principle of minimum Fisher information", Phys. Rev. A 58, 1775 (1998).
4. Caticha, A. 等，熵动力学系列：arXiv:1403.3822；arXiv:1908.04693；arXiv:2208.02156。
5. Chiribella, G., D'Ariano, G. M. & Perinotti, P., arXiv:1011.6451 (2011).
6. Hardy, L., quant-ph/0101012 (2001).
7. Caves, C. M., Fuchs, C. A. & Schack, R., Phys. Rev. A 65, 022305 (2002).
8. Leifer, M. S. & Spekkens, R. W., Phys. Rev. A 88, 052130 (2013)（arXiv:1107.5849）。
9. Zurek, W. H., Entropy 24, 1520 (2022).
10. Jacobson, T., gr-qc/9504004 (1995)；Jacobson, T., arXiv:1505.04753 (2016)。
11. Padmanabhan, T., hep-th/0205278 (2002)。
12. Verlinde, E., arXiv:1001.0785 (2011)；arXiv:1611.02269 (2016)。
13. Faulkner, T. 等，arXiv:1312.7856。
14. Caticha, A., arXiv:1909.09657；Matsueda, H., arXiv:1408.6633；Miyamoto, T. & Yahikozawa, S., arXiv:1205.3211。
15. Merhav, N., arXiv:0903.1484 (2009)。
16. Parrondo, J. M. R., Horowitz, J. M. & Sagawa, T., Nature Physics 11, 131 (2015)；Still, S. 等，PRL 109, 120604 (2012)。
17. Alicki, R., Barenboim, G. & Jenkins, A., arXiv:2307.04800。
18. Tegmark, M., Class. Quantum Grav. 14, L69 (1997)。
19. Erdmenger, J., Grosvenor, K. T. & Jefferson, R., arXiv:2001.02683（JHEP 2020）。
20. Calmet, X. & Calmet, J., math-ph/0403043 (2004)。
21. Faddeev–Hopf 结综述：arXiv:0804.1357。
22. Masanes, L. & Oppenheim, J., arXiv:1412.3828（第三定律不可达性）。
23. Norton, J. D., "Too Good to Be True"（对"信息论补全热力学"的 Liouville 反驳）。
24. arXiv:2510.25553（重整化群 + 量纲分析 → 学习曲线幂律）。

*文档版本：2026-09-16。*
