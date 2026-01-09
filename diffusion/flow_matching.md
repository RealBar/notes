# flow matching 学习笔记
> https://arxiv.org/pdf/2210.02747
### Preliminary
#### 连续性方程
论文中定义条件矢量场用到了这个定义：

$$
\begin{equation}
u_t(x)=\int u_t(x|x_1)\frac {p_t(x|x_1)q(x_1)}{p_t(x)}dx_1
\end{equation}
$$

要理解这个式子，必须了解连续定方程：

$$
\begin{equation}
\frac {\partial p_t(x)} {\partial t} +div(p_t(x)u_t(x))=0
\end{equation}
$$

它是连续性方程通用形式的一种特殊情况。通用的连续性方程为：

$$
\frac {\partial \rho} {\partial t} +\nabla \cdot \mathbf f = s
$$

> 这里的 $\nabla \cdot $ 是求散度，等价于 $div(\cdot)$ 。 这里的s是目标物理量在每单位体积每单位时间的生成量。假若s>0则称 为“源点”；假若 s<0则称 s为“汇点”。假设 $\varphi$ 是没有产生或湮灭的守恒量（例如，电荷），则s=0，连续性方程变为更通用的守恒形式：

$$
\begin{equation}
\frac {\partial \rho} {\partial t} +\nabla \cdot \mathbf f = 0
\end{equation}
$$

这里一定要清楚 $\rho$ 和 $\mathbf f$ 的关系：
- $\rho$ 是某物理量的密度，是该物理量在每单位体积的物理量。
- $\mathbf f$ 是该物理量的流量密度，单位时间通过单位面积的物理量的矢量函数，单位例如质量对应 $kg/(m^2\cdot s)$ ，电荷对应 $A/m^2 = C/(m^2\cdot s)$ 。如果能知道该物理量的密度场和速度场v，则可以写成 $\mathbf f = \rho \mathbf v$ 。注意这里的 $\rho$ 是密度场，是个标量场； $\mathbf v$ 是速度场，是个矢量场。

举例：经典流体力学中的连续性方程为

$$
\frac {\partial \rho} {\partial t} +\nabla \cdot (\rho\mathbf v) = 0
$$

其中，$\rho$ 是密度场，可以理解为每个质点的质量； $\mathbf v$ 是速度矢量场，可以理解为处于每个位置的质点的速度矢量， $\rho \mathbf v$ 表示质量流密度。

现在回到(2)式，就是一个连续性方程，研究的对象是概率密度。我们会根据(2)式来证明(1)。首先根据(2)整理得

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
(4)式左边=\frac {\partial}{\partial t}\int q(x_1)p_t(x|x_1)dx_1=\frac {\partial p_t(x)}{\partial t}
$$

再看右边，根据散度定理（高斯定理）可将积分内的散度提出

$$
(4)式右边=-\nabla \cdot \int q(x_1)p_t(x|x_1)u_t(x|x_1) dx_1
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