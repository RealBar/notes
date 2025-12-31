# Score-based Generative Models, SDEs, and ODEs: Mathematical Foundations

> References:
> - [Score-Based Generative Modeling through Stochastic Differential Equations (ICLR 2021)](https://arxiv.org/abs/2011.13456)
> - [Generative Modeling by Estimating Gradients of the Data Distribution (NeurIPS 2019)](https://arxiv.org/abs/1907.05600)
> - [Denoising Diffusion Probabilistic Models (NeurIPS 2020)](https://arxiv.org/abs/2006.11239)
> - [Reverse-time diffusion equation models (Stochastic Processes and their Applications 1982)](https://www.sciencedirect.com/science/article/pii/0304414982900515)

本文档旨在从严格的数学角度阐述 Diffusion 模型与 Score-based Models、SDE 以及 ODE 之间的深刻联系，重点提供核心结论的推导与证明。

## 1. 预备知识与符号定义

假设数据分布为 $p_{data}(\mathbf{x})$，$\mathbf{x} \in \mathbb{R}^d$。
**Score Function (分数函数)** 定义为对数概率密度的梯度：
$$
\nabla_\mathbf{x} \log p(\mathbf{x})
$$

## 2. 随机微分方程 (SDE) 框架

### 2.1 前向过程 (Forward SDE)

我们将扩散过程建模为一个连续时间的 SDE，时间 $t \in [0, T]$。
$$
d\mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{w}
$$
其中：
*   $\mathbf{f}(\cdot, t): \mathbb{R}^d \to \mathbb{R}^d$ 是漂移系数 (Drift coefficient)。
*   $g(t) \in \mathbb{R}$ 是扩散系数 (Diffusion coefficient)（简化起见假设为标量）。
*   $\mathbf{w}$ 是标准维纳过程 (Wiener process)。

此过程将数据分布 $p_0$ 逐渐转化为噪声分布 $p_T$（通常是高斯分布）。

### 2.2 逆向过程 (Reverse SDE)

Anderson 在其 1982 年的经典论文 **"Reverse-time diffusion equation models"** 中证明了重要定理：对于上述前向过程，存在一个逆向 SDE 可以在时间上从 $T$ 倒流回 $0$，其边缘分布轨迹与前向过程完全一致：

$$
d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})] dt + g(t) d\bar{\mathbf{w}}
$$

*   $dt$ 这里代表时间倒流的无穷小增量（实际上是 $dt < 0$，通常书写时取绝对值并在积分限上体现方向）。
*   $\bar{\mathbf{w}}$ 是逆向时间流中的标准维纳过程。
*   $p_t(\mathbf{x})$ 是前向过程在时刻 $t$ 的边缘概率密度。

**核心洞见**：要实现逆向生成，唯一的未知量是 **Score Function** $\nabla_\mathbf{x} \log p_t(\mathbf{x})$。如果我们能学习到这个分数函数，就可以数值模拟这个 SDE 来生成数据。

#### 推导证明 (Derivation)

这是一个非常深刻的问题。直觉上，扩散（熵增）是不可逆的，似乎逆向过程不应该是一个“扩散过程”。
然而，Anderson (1982) 的核心贡献正是**证明**了（而非假设）：**在一个满足特定正则性条件的扩散过程的时间反转，仍然是一个扩散过程。**

*   **数学定义的“扩散过程”**：指的是具有连续路径（Continuous paths）的强马尔可夫过程（Strong Markov Process）。
*   **物理直觉**：前向过程是“无序扩散”，逆向过程虽然数学形式上仍是 $d\mathbf{x} = \mu dt + \sigma d\mathbf{w}$，但其漂移项 $\mu$ 中包含了特殊的 **Score 项**。这个 Score 项提供了“聚拢”的力，使得粒子能够克服随机噪声，从无序回到有序。

推导步骤如下：

1.  **前向过程的 Fokker-Planck 方程**
    前向 SDE $d\mathbf{x} = \mathbf{f} dt + g d\mathbf{w}$ 对应的概率密度 $p(\mathbf{x}, t)$ 演化满足：
    $$
    \frac{\partial p}{\partial t} = -\nabla \cdot (\mathbf{f} p) + \frac{1}{2} g^2 \Delta p
    $$

2.  **构造逆向过程的形式**
    根据 Anderson 的定理，逆向过程（时间 $\tau = T - t$）也是一个扩散过程。我们设其形式为：
    $$
    d\mathbf{x} = \tilde{\mathbf{f}}(\mathbf{x}, \tau) d\tau + g d\bar{\mathbf{w}}
    $$
    我们需要找到 $\tilde{\mathbf{f}}$，使得该过程产生的边缘分布演化与原过程的时间反转一致。
    该过程关于 $\tau$ 的 Fokker-Planck 方程为：
    $$
    \frac{\partial p}{\partial \tau} = -\nabla \cdot (\tilde{\mathbf{f}} p) + \frac{1}{2} g^2 \Delta p
    $$

3.  **时间反转匹配**
    由于 $\tau = T - t$，我们有 $\frac{\partial p}{\partial \tau} = -\frac{\partial p}{\partial t}$。将前向方程代入：
    $$
    \frac{\partial p}{\partial \tau} = -\left( -\nabla \cdot (\mathbf{f} p) + \frac{1}{2} g^2 \Delta p \right) = \nabla \cdot (\mathbf{f} p) - \frac{1}{2} g^2 \Delta p
    $$
    我们希望逆向过程的演化与此一致，即联立上述两式：
    $$
    -\nabla \cdot (\tilde{\mathbf{f}} p) + \frac{1}{2} g^2 \Delta p = \nabla \cdot (\mathbf{f} p) - \frac{1}{2} g^2 \Delta p
    $$
    整理得：
    $$
    \nabla \cdot (\tilde{\mathbf{f}} p) = -\nabla \cdot (\mathbf{f} p) + g^2 \Delta p
    $$

4.  **求解漂移系数**
    利用恒等式 $\Delta p = \nabla \cdot (\nabla p) = \nabla \cdot (p \nabla \log p)$，代入上式右边：
    $$
    \begin{aligned}
    \nabla \cdot (\tilde{\mathbf{f}} p) &= -\nabla \cdot (\mathbf{f} p) + \nabla \cdot (g^2 p \nabla \log p) \\
    &= \nabla \cdot [ (-\mathbf{f} + g^2 \nabla \log p) p ]
    \end{aligned}
    $$
    去掉散度算子 $\nabla \cdot$ 和密度 $p$，得到逆向漂移（关于时间 $\tau$）：
    $$
    \tilde{\mathbf{f}} = -\mathbf{f} + g^2 \nabla \log p
    $$

5.  **回到时间 $t$**
    在逆向 SDE 中，如果我们将时间变量写回 $t$（注意 $dt = -d\tau$），则 $d\mathbf{x} = \tilde{\mathbf{f}} d\tau = \tilde{\mathbf{f}} (-dt)$。
    为了保持通常 SDE 的形式 $d\mathbf{x} = (\dots) dt + \dots$，漂移项系数为 $-\tilde{\mathbf{f}}$：
    $$
    -\tilde{\mathbf{f}} = \mathbf{f} - g^2 \nabla \log p
    $$
    这就得到了最终的逆向 SDE 公式：
    $$
    d\mathbf{x} = [\mathbf{f} - g^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})] dt + g d\bar{\mathbf{w}}
    $$

---

## 3. 证明：最优噪声预测器与 Score Function 的线性关系

这是连接 DDPM 和 Score-based Models 的核心桥梁。我们将证明：**训练一个去噪自动编码器（预测噪声）等价于训练一个 Score Estimator（预测分数）。**

### 3.1 设定：高斯扰动核 (Gaussian Perturbation Kernel)

考虑最常见的扩散过程（如 VP-SDE 或 DDPM），给定初始数据 $\mathbf{x}_0$，时刻 $t$ 的状态 $\mathbf{x}_t$ 服从高斯分布：
$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \mu_t(\mathbf{x}_0), \sigma_t^2 \mathbf{I})
$$
在 DDPM 的记号中，$\mu_t(\mathbf{x}_0) = \sqrt{\bar{\alpha}_t}\mathbf{x}_0$，$\sigma_t^2 = 1 - \bar{\alpha}_t$。
利用重参数化技巧，我们可以写成：
$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

### 3.2 扰动分布的 Score (Score of Perturbation Kernel)

我们可以直接计算条件分布 $q(\mathbf{x}_t | \mathbf{x}_0)$ 关于 $\mathbf{x}_t$ 的 Score：
$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0) &= \nabla_{\mathbf{x}_t} \log \left( \frac{1}{(2\pi \sigma_t^2)^{d/2}} \exp\left( -\frac{\|\mathbf{x}_t - \mu_t(\mathbf{x}_0)\|^2}{2\sigma_t^2} \right) \right) \\
&= \nabla_{\mathbf{x}_t} \left( -\frac{\|\mathbf{x}_t - \mu_t(\mathbf{x}_0)\|^2}{2\sigma_t^2} \right) \\
&= -\frac{\mathbf{x}_t - \mu_t(\mathbf{x}_0)}{\sigma_t^2}
\end{aligned}
$$
代入 DDPM 的参数：
$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{1 - \bar{\alpha}_t}
$$
由于 $\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0 = \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$，代入上式得：
$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0) = -\frac{\sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}}{1 - \bar{\alpha}_t} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}
$$

**结论 1**：给定 $\mathbf{x}_0$ 条件下的 Score 与该样本所加的噪声 $\boldsymbol{\epsilon}$ 成负线性关系。

### 3.3 降噪分数匹配 (Denoising Score Matching, DSM)

我们要估计的是边缘分布的 Score $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$，而不是条件 Score。Vincent (2011) 证明了最小化以下两个目标函数是等价的：

1.  **显式 Score Matching (SM)**:
    $$ \mathcal{L}_{SM} = \frac{1}{2} \mathbb{E}_{\mathbf{x}_t \sim p_t} [ \| \mathbf{s}_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \|^2 ] $$
    *(难以直接计算，因为不知道 $p_t$)*

2.  **降噪 Score Matching (DSM)**:
    $$ \mathcal{L}_{DSM} = \frac{1}{2} \mathbb{E}_{\mathbf{x}_0 \sim p_{data}, \mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_0)} [ \| \mathbf{s}_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0) \|^2 ] $$
    *(可以计算，因为 $q(\mathbf{x}_t|\mathbf{x}_0)$ 已知)*

### 3.4 建立联系

将 **结论 1** 中的 $\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0) = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}$ 代入 DSM 目标函数：

$$
\mathcal{L}_{DSM} = \frac{1}{2} \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \mathbf{s}_\theta(\mathbf{x}_t, t) - \left( -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}} \right) \right\|^2 \right]
$$

**参数化技巧 (Parameterization)**：
观察上式，我们的优化目标是让 $\mathbf{s}_\theta(\mathbf{x}_t, t)$ 去逼近 $-\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}$。
为了简化学习过程，我们不直接让神经网络输出分数 $\mathbf{s}_\theta$，而是让网络去预测噪声 $\boldsymbol{\epsilon}$。
具体来说，我们**定义**网络的分数输出形式为：
$$
\mathbf{s}_\theta(\mathbf{x}_t, t) := -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$
其中 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 是一个输出与数据维度相同的神经网络（即 U-Net）。

将这个定义代入 DSM 损失函数：
$$
\begin{aligned}
\mathcal{L}_{DSM} &= \frac{1}{2} \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}} - \left( -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}} \right) \right\|^2 \right] \\
&= \frac{1}{2(1 - \bar{\alpha}_t)} \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 \right]
\end{aligned}
$$
这正是 DDPM 中的简化损失函数（忽略了前面的加权系数 $\lambda(t)$）。
这个推导解释了为什么 DDPM 虽然是在预测噪声，但本质上是在训练一个 Score-based Model。

**最终结论**：
最优的 Score 模型 $\mathbf{s}^*(\mathbf{x}_t, t)$ 与最优的噪声预测模型 $\boldsymbol{\epsilon}^*(\mathbf{x}_t, t)$ 满足以下关系：
$$
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = \mathbf{s}^*(\mathbf{x}_t, t) = -\frac{\boldsymbol{\epsilon}^*(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$
这严格证明了预测噪声本质上就是在估计 Score（的缩放版本）。

---

## 4. Probability Flow ODE 的推导

对于任意扩散 SDE，为什么存在一个确定性的 ODE 具有相同的边缘分布？

### 4.1 Fokker-Planck 方程 (Kolmogorov Forward Equation)

对于 SDE $d\mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{w}$，其边缘概率密度 $p_t(\mathbf{x})$ 的演化遵循 Fokker-Planck 方程：
$$
\frac{\partial p_t(\mathbf{x})}{\partial t} = -\nabla_\mathbf{x} \cdot [\mathbf{f}(\mathbf{x}, t) p_t(\mathbf{x})] + \frac{1}{2} g(t)^2 \Delta_\mathbf{x} p_t(\mathbf{x})
$$
其中 $\nabla_\mathbf{x} \cdot$ 是散度算子，$\Delta_\mathbf{x}$ 是拉普拉斯算子。

### 4.2 构造 ODE

考虑如下形式的 ODE：
$$
d\mathbf{x} = \tilde{\mathbf{f}}(\mathbf{x}, t) dt
$$
其对应的连续性方程（描述密度演化）为：
$$
\frac{\partial p_t(\mathbf{x})}{\partial t} = -\nabla_\mathbf{x} \cdot [\tilde{\mathbf{f}}(\mathbf{x}, t) p_t(\mathbf{x})]
$$

我们要寻找 $\tilde{\mathbf{f}}$，使得 ODE 的密度演化与 SDE 的 Fokker-Planck 方程一致。
利用恒等式 $\Delta_\mathbf{x} p = \nabla_\mathbf{x} \cdot (\nabla_\mathbf{x} p) = \nabla_\mathbf{x} \cdot (p \nabla_\mathbf{x} \log p)$，我们可以重写 SDE 的 Fokker-Planck 方程：

$$
\begin{aligned}
\frac{\partial p_t}{\partial t} &= -\nabla \cdot (\mathbf{f} p_t) + \frac{1}{2} g(t)^2 \nabla \cdot (p_t \nabla \log p_t) \\
&= -\nabla \cdot \left[ \left( \mathbf{f} - \frac{1}{2} g(t)^2 \nabla \log p_t \right) p_t \right]
\end{aligned}
$$

对比 ODE 的连续性方程，我们可以直接看出，如果取：
$$
\tilde{\mathbf{f}}(\mathbf{x}, t) = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2} g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})
$$
那么 ODE $d\mathbf{x} = \tilde{\mathbf{f}}(\mathbf{x}, t) dt$ 的概率密度演化将完全等同于原 SDE 的概率密度演化。

**这就是 Probability Flow ODE**：
$$
d\mathbf{x} = \left[ \mathbf{f}(\mathbf{x}, t) - \frac{1}{2} g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x}) \right] dt
$$

### 4.3 物理意义

*   SDE 描述的是粒子的随机扩散运动。
*   ODE 描述的是概率流体（Probability Fluid）的平滑流动。
*   虽然单个粒子的轨迹在 SDE 和 ODE 中截然不同，但它们组成的整体分布在任何时刻 $t$ 都是完全一样的。

---

## 5. 总结：统一视角

通过上述推导，我们建立了如下严密的逻辑链条：

1.  **SDE 定义了扩散过程**：将数据映射为先验噪声。
2.  **Score Function 是逆向关键**：逆向 SDE 需要 $\nabla \log p_t$。
3.  **DSM 连接了噪声预测与 Score**：证明了 $\epsilon_\theta \approx -\sigma \nabla \log p_t$，使得我们可以用标准的神经网络（U-Net 预测噪声）来近似 Score。
4.  **Fokker-Planck 方程导出了 ODE**：证明了存在一个确定性过程，其边缘分布与随机过程一致，为确定性采样 (DDIM) 提供了理论基础。
