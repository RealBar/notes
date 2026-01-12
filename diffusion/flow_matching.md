# flow matching 学习笔记
> https://arxiv.org/pdf/2210.02747

### 条件矢量场
论文中定义条件矢量场用到了这个定义：

$$
\begin{equation}
u_t(x)=\int u_t(x|x_1)\frac {p_t(x|x_1)q(x_1)}{p_t(x)}dx_1
\end{equation}
$$

要理解这个式子，必须了解连续定方程。概率形式的连续性方程如下：

$$
\begin{equation}
\frac {\partial p_t(x)} {\partial t} +\nabla \cdot(p_t(x)u_t(x))=0
\end{equation}
$$

我们会根据(2)式来证明(1)。

**证**: 首先根据(2)整理得条件形式的连续性方程：

$$
\frac {\partial p_t(x|x_1)} {\partial t} = -\nabla \cdot (p_t(x|x_1)u_t(x|x_1))
$$

两边同时对乘以q(x_1)并对x_1积分

$$
\begin{equation}
\int q(x_1)\frac {\partial p_t(x|x_1)} {\partial t} dx_1 = -\int q(x_1)\nabla \cdot (p_t(x|x_1)u_t(x|x_1))dx_1
\end{equation}
$$

我们观察上式，左边根据莱布尼茨积分法则可将积分内的偏导提出

$$
(3)式左边=\frac {\partial}{\partial t}\int q(x_1)p_t(x|x_1)dx_1=\frac {\partial p_t(x)}{\partial t}
$$

再看右边，根据莱布尼兹积分法则可将散度符号提出

$$
(3)式右边=-\nabla \cdot \int q(x_1)p_t(x|x_1)u_t(x|x_1) dx_1
$$

综合上边的分析，有

$$
\frac {\partial p_t(x)}{\partial t}=-\nabla \cdot \int q(x_1)p_t(x|x_1)u_t(x|x_1) dx_1
$$

再根据(2)式有

$$
\frac {\partial p_t(x)} {\partial t} =-\nabla \cdot (p_t(x)u_t(x))
$$

对比上边两个式子，有

$$
-\nabla \cdot \int q(x_1)p_t(x|x_1)u_t(x|x_1) dx_1=-\nabla \cdot (p_t(x)u_t(x))
$$

同时去掉散度算符，两边同时除以 $p_t(x)$ ，得到

$$
u_t(x)=\int u_t(x|x_1)\frac {p_t(x|x_1)q(x_1)} {p_t(x)}  dx_1
$$

即为(1)式

### FM和CFM有相同的梯度
FM损失函数：

$$
\mathcal L_{FM}(\theta)=\Bbb E_{t,p_t(x)}\|v_t(x;\theta)-u_t(x)\|^2
$$

CFM损失函数：

$$
\mathcal L_{CFM}(\theta)=\Bbb E_{t,x\sim p_t(x|x_1),x_1\sim q(x_1)}\|v_t(x;\theta)-u(x|x_1)\|^2
$$

论文直接给出结论：这两个损失函数对于模型参数 $\theta$ 具有相同的梯度。下面给出证明

**证**:

要证 $\nabla_{\theta} \Bbb E_{t,p_t(x)}\|v_t(x;\theta)-u_t(x)\|^2=\nabla_{\theta} \Bbb E_{t,x\sim p_t(x|x_1),x_1\sim q(x_1)}\|v_t(x;\theta)-u(x|x_1)\|^2$

只需证 $ 2\Bbb E_{t,p_t(x)}[(v_t(x;\theta)-u_t(x))\nabla_{\theta} v_t(x;\theta)]=2\Bbb  E_{t,x\sim p_t(x|x_1),x_1\sim q(x_1)}[(v_t(x;\theta)-u_t(x|x_1))\nabla_{\theta} v_t(x;\theta)]$

观察上式，等号左右期望内分别有两项，拆开后第一项显然相等，即

$$
\Bbb E_{t,p_t(x)}[v_t \nabla_\theta v_t]=\Bbb E_{t,x\sim p_t(x|x_1),x_1\sim q(x_1)}[v_t \nabla_\theta v_t]
$$

因为根据 $p_t(x)=\int q(x_1)p_t(x|x_1)dx_1$ ，有对于任意函数h(x)的期望 $\Bbb E_{t,p_t(x)}[h(x)]=\Bbb E_t\int h(x)p_t(x)dx=\Bbb E_t\int h(x)\int q(x_1)p_t(x|x_1)dx_1dx=\Bbb E_{t,x\sim p_t(x|x_1),x_1\sim q(x_1)}[h(x)]$


所以只需证后一项相等，即只需证 

$$
\begin{equation}
\Bbb E_{t,p_t(x)}[u_t(x)\nabla_{\theta} v_t(x;\theta)]=\Bbb  E_{t,x\sim p_t(x|x_1),x_1\sim q(x_1)}[u_t(x|x_1)\nabla_{\theta} v_t(x;\theta)]
\end{equation}
$$

$$
(4)式左边=\Bbb E_t\int u_t(x)\nabla_\theta v_t(x;\theta)p_t(x)dx
$$

将(1)代入，得

$$
(4)式左边=\Bbb E_t\int (\int u_t(x|x_1)\frac {p_t(x|x_1)q(x_1)}{p_t(x)}dx_1)\nabla_\theta v_t(x;\theta)p_t(x)dx
$$

$$
=\Bbb E_t\int (\int u_t(x|x_1)p_t(x|x_1)q(x_1)dx_1)\nabla_\theta v_t(x;\theta)dx
$$

根据富比尼定理，可以将积分顺序交换，得到

$$
=\Bbb E_t\int q(x_1)(\int u_t(x|x_1)p_t(x|x_1)\nabla_\theta v_t(x;\theta)dx)dx_1
$$

$$
=\Bbb E_t\int q(x_1) \Bbb E_{x \sim p_t(x|x_1)}[u_t(x|x_1)\nabla_\theta v_t(x;\theta)]dx_1
$$

$$
=\Bbb E_{t,x \sim p_t(x|x_1),x_1 \sim q(x_1)}[u_t(x|x_1)\nabla v_t(x;\theta)]=(4)式右边
$$

原命题得证