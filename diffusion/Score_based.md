# Unified View of Diffusion and Score-based Models via SDEs

> **核心参考文献**: 
> - [Score-Based Generative Modeling through Stochastic Differential Equations (ICLR 2021)](https://arxiv.org/abs/2011.13456)
> - [Denoising Diffusion Probabilistic Models (NeurIPS 2020)](https://arxiv.org/abs/2006.11239)
> - [Generative Modeling by Estimating Gradients of the Data Distribution (NeurIPS 2019)](https://arxiv.org/abs/1907.05600)
> - [A Connection Between Score Matching and Denoising Autoencoders (Neural Computation 2011)](https://www.researchgate.net/publication/220320057_A_Connection_Between_Score_Matching_and_Denoising_Autoencoders)
> - [Reverse-time diffusion equation models (Stochastic Processes and their Applications 1982)](https://www.sciencedirect.com/science/article/pii/0304414982900515)
> 
> 本文旨在提供一个**统一的数学框架**，解释 Diffusion Model (如 DDPM) 和 Score-based Model (如 NCSN) 如何在 **随机微分方程 (SDE)** 的视角下被视为同一事物的不同面相。

## 1. 宏观图景：SDE 作为统一框架

传统的 Diffusion Model (DDPM) 是离散时间的，Score-based Model (NCSN) 也是离散的。如果我们将时间步数推向无穷 ($T \to \infty$)，这两个过程都会收敛到连续时间的 **SDE**。

在这个统一视角下，生成模型的核心思想分为两步：
1.  **前向过程 (Forward Process)**：通过一个 SDE 将复杂的数据分布 $p_{data}$ 逐渐平滑地变成简单的噪声分布 $p_{prior}$ (通常是高斯)。
2.  **逆向过程 (Reverse Process)**：通过求解逆向 SDE，将噪声分布 $p_{prior}$ 还原回数据分布 $p_{data}$。

---

## 2. 关键理论：为什么需要 Score?

我们首先定义前向过程。假设数据 $\mathbf{x}(t)$ 随时间 $t \in [0, T]$ 演化：

$$
d\mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{w} \quad (\text{前向 SDE})
$$

*   $\mathbf{f}(\mathbf{x}, t)$: 漂移系数 (Drift)，控制确定性趋势。
*   $g(t)$: 扩散系数 (Diffusion)，控制噪声强度。
*   $\mathbf{w}$: 标准布朗运动。

### 2.1 逆向 SDE 的存在性
如何生成数据？我们需要时间倒流。Anderson (1982) 证明了一个惊人的定理：**上述前向扩散过程的时间倒流，仍然是一个扩散过程**，其方程为：

$$
d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})] dt + g(t) d\bar{\mathbf{w}} \quad (\text{逆向 SDE})
$$

*   $dt$ 在此处表示时间倒流。
*   $\bar{\mathbf{w}}$ 是逆向布朗运动。
*   $p_t(\mathbf{x})$ 是时刻 $t$ 的边缘概率密度。

### 2.2 Score Function 的必要性
观察逆向 SDE 公式，其中 $\mathbf{f}$ 和 $g$ 都是我们人为设计的（已知），**唯一的未知项**就是：
$$
\nabla_\mathbf{x} \log p_t(\mathbf{x})
$$
这就是 **Score Function**。

**结论**：要通过逆向 SDE 生成数据，我们**必须**知道每个时刻的概率密度梯度（Score）。这就是为什么这类模型被称为 "Score-based Models"。

---

## 3. 学习 Score：连接 DDPM 与 Score Matching

既然目标明确了——估计 $\nabla_\mathbf{x} \log p_t(\mathbf{x})$，我们如何训练神经网络来做到这一点？

### 3.1 训练目标：Denoising Score Matching (DSM)
直接计算 $\nabla \log p_t(\mathbf{x})$ 是不可能的（不知道 $p_t$）。但我们可以通过 Vincent (2011) 提出的 DSM 方法，利用条件分布 $q(\mathbf{x}_t | \mathbf{x}_0)$ 来训练：

$$
\mathcal{L}_{DSM} = \mathbb{E}_{\mathbf{x}_0, \mathbf{x}_t} \left[ \| \mathbf{s}_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0) \|^2 \right]
$$

### 3.2 证明：DDPM 本质上就是在做 Score Matching
这是连接两者的**最关键推导**。

对于高斯扩散过程（如 DDPM），条件分布为 $q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$。
我们可以直接算出它的 Score（详细推导见下文）：
$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0) = -\frac{\mathbf{x}_t - \mu}{\sigma^2} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}
$$
其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是加入的噪声。

将此代入 DSM 损失函数：
$$
\mathcal{L} \propto \mathbb{E} \left[ \| \mathbf{s}_\theta(\mathbf{x}_t, t) - (-\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}) \|^2 \right]
$$

如果我们**参数化**神经网络，让它不直接输出 Score，而是去预测噪声 $\boldsymbol{\epsilon}_\theta$，即令 $\mathbf{s}_\theta = -\frac{\boldsymbol{\epsilon}_\theta}{\sqrt{1 - \bar{\alpha}_t}}$，则损失函数变为：
$$
\mathcal{L} \propto \mathbb{E} \left[ \| \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \boldsymbol{\epsilon} \|^2 \right]
$$
这正是 **DDPM 的损失函数**！

**统一结论**：
*   **Score-based Models**: 显式地学习 Score Function。
*   **DDPM**: 通过预测噪声来隐式地学习 Score Function。
*   两者在数学上是**等价**的，只是参数化方式不同。预测了噪声，就等于知道了 Score。

---

## 4. 采样：从 SDE 到 ODE

一旦训练好了 Score 网络 $\mathbf{s}_\theta(\mathbf{x}, t) \approx \nabla \log p_t(\mathbf{x})$，我们就可以通过数值求解方程来生成样本。SDE 框架赋予了我们极大的灵活性，主要有三种采样方式：

### 4.1 方式一：求解逆向 SDE (Stochastic Sampling)
直接离散化逆向 SDE：
$$
\mathbf{x}_{t-\Delta t} = \mathbf{x}_t + [\mathbf{f} - g^2 \mathbf{s}_\theta] \Delta t + g \sqrt{\Delta t} \mathbf{z}
$$
*   这就是 DDPM 的 **Ancestral Sampling** 的连续形式。
*   特点：每一步都注入随机噪声 $\mathbf{z}$，具有随机性。

### 4.2 方式二：求解 Probability Flow ODE (Deterministic Sampling)
对于任意扩散 SDE，都存在一个确定性的 **ODE**，其边缘分布演化与 SDE 完全一致（推导基于 Fokker-Planck 方程）：
$$
d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})] dt
$$
*   这就是 **DDIM** 的连续形式。
*   特点：没有随机项。给定初始噪声，生成结果是确定的。
*   优势：可以使用黑盒 ODE 求解器（如 Runge-Kutta）来加速采样；可以进行 Latent Space 插值。

### 4.3 方式三：Langevin Dynamics (Corrector)
Score-based model 独有的技巧。在求解 SDE 的每一步之后，可以额外运行几步 MCMC (Langevin Dynamics)：
$$
\mathbf{x}_{i+1} = \mathbf{x}_i + \epsilon \nabla \log p_t(\mathbf{x}_i) + \sqrt{2\epsilon} \mathbf{z}
$$
*   作用：利用 Score 将当前样本“推”向概率密度更高的区域，纠正数值误差。
*   这种 "Predictor-Corrector" 采样器通常能获得更高质量的样本。

---

## 5. 附录：详细数学推导

### A. 逆向 SDE 漂移项推导

我们希望找到前向过程 $d\mathbf{x} = \mathbf{f} dt + g d\mathbf{w}$ 的逆向过程。

1.  **前向 Fokker-Planck 方程**:
    前向过程对应的概率密度 $p(\mathbf{x}, t)$ 满足：
    $$
    \frac{\partial p}{\partial t} = -\nabla \cdot (\mathbf{f} p) + \frac{1}{2} g^2 \Delta p
    $$

2.  **逆向过程假设**:
    设逆向过程（时间 $\tau = T-t$）为扩散过程：
    $$
    d\mathbf{x} = \tilde{\mathbf{f}} d\tau + g d\bar{\mathbf{w}}
    $$
    其 Fokker-Planck 方程为：
    $$
    \frac{\partial p}{\partial \tau} = -\nabla \cdot (\tilde{\mathbf{f}} p) + \frac{1}{2} g^2 \Delta p
    $$

3.  **匹配条件**:
    由于 $\frac{\partial p}{\partial \tau} = -\frac{\partial p}{\partial t}$，将前向方程代入可得：
    $$
    -\nabla \cdot (\tilde{\mathbf{f}} p) + \frac{1}{2} g^2 \Delta p = -\left( -\nabla \cdot (\mathbf{f} p) + \frac{1}{2} g^2 \Delta p \right)
    $$
    整理得：
    $$
    \nabla \cdot (\tilde{\mathbf{f}} p) = -\nabla \cdot (\mathbf{f} p) + g^2 \Delta p
    $$

4.  **利用恒等式求解**:
    利用 $\Delta p = \nabla \cdot (p \nabla \log p)$，代入上式右侧：
    $$
    \begin{aligned}
    \nabla \cdot (\tilde{\mathbf{f}} p) &= -\nabla \cdot (\mathbf{f} p) + \nabla \cdot (g^2 p \nabla \log p) \\
    &= \nabla \cdot [ (-\mathbf{f} + g^2 \nabla \log p) p ]
    \end{aligned}
    $$
    由此解得逆向漂移（关于时间 $\tau$）：
    $$
    \tilde{\mathbf{f}} = -\mathbf{f} + g^2 \nabla \log p
    $$
    换回时间 $t$（注意 $dt = -d\tau$），逆向 SDE 的漂移项为 $-\tilde{\mathbf{f}}$，即 $\mathbf{f} - g^2 \nabla \log p$。

### B. 为什么 DDPM 的噪声预测等价于 Score

1.  **计算高斯扰动核的 Score**:
    $$ q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I}) $$
    $$ \log q = -\frac{1}{2(1-\bar{\alpha}_t)} \|\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0\|^2 + C $$
    $$ \nabla_{\mathbf{x}_t} \log q = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{1-\bar{\alpha}_t} $$
    代入 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$：
    $$ \nabla_{\mathbf{x}_t} \log q = -\frac{\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}}{1-\bar{\alpha}_t} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar{\alpha}_t}} $$

2.  **DSM 目标函数变换**:
    $$ \mathcal{L}_{DSM} = \mathbb{E} [ \| \mathbf{s}_\theta - \nabla \log q \|^2 ] = \mathbb{E} \left[ \left\| \mathbf{s}_\theta - \left( -\frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar{\alpha}_t}} \right) \right\|^2 \right] $$
    
3.  **参数化**:
    令 $\mathbf{s}_\theta(\mathbf{x}, t) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}, t)}{\sqrt{1-\bar{\alpha}_t}}$，代入上式：
    $$ \mathcal{L} = \mathbb{E} \left[ \left\| -\frac{\boldsymbol{\epsilon}_\theta}{\sqrt{1-\bar{\alpha}_t}} + \frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar{\alpha}_t}} \right\|^2 \right] = \frac{1}{1-\bar{\alpha}_t} \mathbb{E} [ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta \|^2 ] $$
    这证明了预测噪声等价于预测 Score。

### C. Probability Flow ODE 推导

我们寻找一个 ODE $d\mathbf{x} = \tilde{\mathbf{f}} dt$，使其边缘分布演化与 SDE 一致。

1.  **SDE 的 Fokker-Planck 方程**:
    $$ \frac{\partial p}{\partial t} = -\nabla \cdot (\mathbf{f} p) + \frac{1}{2} g^2 \Delta p $$
    
2.  **变换扩散项**:
    利用 $\Delta p = \nabla \cdot (p \nabla \log p)$：
    $$ \frac{\partial p}{\partial t} = -\nabla \cdot (\mathbf{f} p) + \frac{1}{2} g^2 \nabla \cdot (p \nabla \log p) $$
    $$ \frac{\partial p}{\partial t} = -\nabla \cdot \left[ \left( \mathbf{f} - \frac{1}{2} g^2 \nabla \log p \right) p \right] $$

3.  **匹配 ODE 连续性方程**:
    ODE 的连续性方程为 $\frac{\partial p}{\partial t} = -\nabla \cdot (\tilde{\mathbf{f}} p)$。
    直接对比括号内的项，可得：
    $$ \tilde{\mathbf{f}} = \mathbf{f} - \frac{1}{2} g^2 \nabla \log p $$
    即得 Probability Flow ODE：
    $$ d\mathbf{x} = \left[ \mathbf{f}(\mathbf{x}, t) - \frac{1}{2} g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x}) \right] dt $$

---

## 6. 总结图谱

| 概念 | DDPM (离散) | Score-based / SDE (连续) | 关系 |
| :--- | :--- | :--- | :--- |
| **前向过程** | 马尔可夫链 $q(x_t\|x_{t-1})$ | Forward SDE $dx = f dt + g dw$ | DDPM 是 VP-SDE 的离散化 |
| **训练目标** | 预测噪声 $\epsilon_\theta$ | 预测分数 $s_\theta$ | $\epsilon_\theta \propto -s_\theta$ (线性等价) |
| **采样(随机)** | Ancestral Sampling | Reverse SDE Solver | 本质相同 |
| **采样(确定)** | DDIM | Probability Flow ODE | 本质相同 |
