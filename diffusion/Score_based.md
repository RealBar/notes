# Unified View of Diffusion and Score-based Models via SDEs

> 本文旨在提供一个**统一的数学框架**，解释 Diffusion Model (如 DDPM, DDIM)、Score-based Model (如 NCSN) 以及最新的 **Flow Matching** 如何在 **随机微分方程 (SDE)** 和 **常微分方程 (ODE)** 的视角下被视为同一事物的不同面相。

## 1. 宏观图景：SDE 与 ODE 作为统一框架

生成模型的核心任务是建立一个从简单噪声分布 $p_{prior}$ 到复杂数据分布 $p_{data}$ 的映射。

1.  **Diffusion / Score-based Models (SDE 视角)**：
    *   **前向**：通过 SDE 将数据逐渐破坏为噪声。
    *   **逆向**：通过逆向 SDE（需 Score）将噪声还原为数据。
    *   **确定性路径**：每个 SDE 都有一个对应的 Probability Flow ODE。

2.  **Flow Matching (ODE 视角)**：
    *   **核心思想**：直接定义一个 ODE（向量场），将噪声分布平滑地推运 (Push-forward) 到数据分布。
    *   **优势**：不再局限于扩散过程（高斯噪声），可以设计更直的轨迹（如 Optimal Transport 路径），从而实现更快、更稳定的采样。

在这个统一视角下，**Flow Matching 可以看作是 Probability Flow ODE 的推广**。

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
*   **与 DDIM 的关系**：这个 Probability Flow ODE 正是 **DDIM** 在连续时间极限下的形式。
    *   DDIM 的核心观察是：在 DDPM 的前向过程中，只要保证边缘分布 $q(\mathbf{x}_t|\mathbf{x}_0)$ 形式不变，中间的条件分布 $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ 可以是非马尔可夫的，且可以去除随机性。
    *   Probability Flow ODE 提供了相同的性质：它与原 SDE 共享相同的边缘分布 $p_t(\mathbf{x})$，但轨迹是确定性的。
*   **特点**：
    1.  **确定性**：给定初始噪声 $z_T$，生成的样本 $x_0$ 是唯一确定的。
    2.  **一致性**：为 Latent Space 提供了一致的语义插值（Interpolation）能力。
    3.  **加速采样**：由于是 ODE，可以使用高阶数值求解器（如 Runge-Kutta）以更大的步长进行采样，从而大幅减少采样步数（例如从 DDPM 的 1000 步减少到 DDIM 的 50 步）。

### 4.3 方式三：Langevin Dynamics (Corrector)
Score-based model 独有的技巧。在求解 SDE 的每一步之后，可以额外运行几步 MCMC (Langevin Dynamics)：
$$
\mathbf{x}_{i+1} = \mathbf{x}_i + \epsilon \nabla \log p_t(\mathbf{x}_i) + \sqrt{2\epsilon} \mathbf{z}
$$
*   作用：利用 Score 将当前样本“推”向概率密度更高的区域，纠正数值误差。
*   这种 "Predictor-Corrector" 采样器通常能获得更高质量的样本。

### 4.4 进阶：Flow Matching (ODE Generalization)

**背景**：
Diffusion Models 的 Probability Flow ODE 虽然是确定性的，但其轨迹通常是弯曲的（因为必须遵循扩散过程的边缘分布）。
能不能直接设计一条**更直、更简单**的 ODE 轨迹，从噪声 $p_0$ 映射到数据 $p_1$？

**Flow Matching 核心思想**：
1.  **定义目标概率路径 (Target Probability Path)** $p_t(\mathbf{x})$：
    我们不再受限于扩散过程，而是可以显式构造一个分布插值路径。例如 **Optimal Transport (OT) Path**：
    $$ \mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1 $$
    其中 $\mathbf{x}_0 \sim \mathcal{N}(0, I)$，$\mathbf{x}_1 \sim p_{data}$。（注意时间定义通常与 Diffusion 相反，这里 $t=0$ 是噪声，$t=1$ 是数据）。

2.  **定义向量场 (Vector Field)** $\mathbf{u}_t(\mathbf{x})$：
    我们需要寻找一个向量场 $\mathbf{u}_t$，使得由它驱动的 ODE $d\mathbf{x}/dt = \mathbf{u}_t(\mathbf{x})$ 生成的概率流恰好等于我们定义的目标路径 $p_t$。
    这等价于满足连续性方程 (Continuity Equation)：
    $$ \frac{\partial p_t}{\partial t} + \nabla \cdot (p_t \mathbf{u}_t) = 0 $$

3.  **训练目标 (Conditional Flow Matching)**：
    直接回归向量场 $\mathbf{u}_t$ 很难。Flow Matching 证明了，我们可以通过回归**条件向量场** $\mathbf{u}_t(\mathbf{x}|\mathbf{z})$ 来简化训练：
    $$ \mathcal{L}_{FM} = \mathbb{E}_{t, q(\mathbf{z}), p_t(\mathbf{x}|\mathbf{z})} \| \mathbf{v}_\theta(\mathbf{x}, t) - \mathbf{u}_t(\mathbf{x}|\mathbf{z}) \|^2 $$
    对于 OT 路径，条件向量场极其简单：$\mathbf{u}_t(\mathbf{x}|\mathbf{x}_0, \mathbf{x}_1) = \mathbf{x}_1 - \mathbf{x}_0$。这意味着模型只需要学习预测**常数速度**！

**与 Diffusion 的关系**：
*   **Diffusion** 学习的是 Score $\nabla \log p_t$，对应的向量场是 $\mathbf{u}_t \propto \nabla \log p_t$。
*   **Flow Matching** 直接学习向量场 $\mathbf{v}_\theta \approx \mathbf{u}_t$。
*   当目标路径设定为 VP-SDE 的边缘分布时，Flow Matching 等价于 Diffusion Model。
*   **优势**：Flow Matching 允许使用 **Optimal Transport** 路径，这种路径是直线的，因此在采样时数值误差更小，可以用更少的步数（Few-step Sampling）得到高质量结果。例如 **Stable Diffusion 3** 就采用了这种 Rectified Flow / Flow Matching 技术。

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

## 参考
> - [Flow Matching for Generative Modeling (ICLR 2023)](https://arxiv.org/abs/2210.02747)
> - [Score-Based Generative Modeling through Stochastic Differential Equations (ICLR 2021)](https://arxiv.org/abs/2011.13456)
> - [Denoising Diffusion Probabilistic Models (NeurIPS 2020)](https://arxiv.org/abs/2006.11239)
> - [Denoising Diffusion Implicit Models (ICLR 2021)](https://arxiv.org/abs/2010.02502)
> - [Generative Modeling by Estimating Gradients of the Data Distribution (NeurIPS 2019)](https://arxiv.org/abs/1907.05600)
> - [A Connection Between Score Matching and Denoising Autoencoders (Neural Computation 2011)](https://www.researchgate.net/publication/220320057_A_Connection_Between_Score_Matching_and_Denoising_Autoencoders)
> - [Reverse-time diffusion equation models (Stochastic Processes and their Applications 1982)](https://www.sciencedirect.com/science/article/pii/0304414982900515)