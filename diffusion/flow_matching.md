# flow matching 学习笔记
> https://arxiv.org/pdf/2210.02747
### Preliminary
#### 连续性方程
论文中定义条件矢量场用到了这个定义：

$$
u_t(x)=\int u_t(x|x_1)\frac {p_t(x|x_1)q(x_1)}{p_t(x)}dx_1
$$

要理解这个式子，必须了解连续定方程：

$$
\frac {\partial p_t(x)} {\partial t} +div(p_t(x)u_t(x))=0
$$

它是连续性方程通用形式的一种特殊情况：

$$
\frac {\partial \varphi} {\partial t} +\nabla \cdot \mathbf f = s
$$

注意这里的 $\nabla \cdot \mathbf f$ 表示求向量f的散度，不是梯度。其中， $\varphi$ 是某物理量q的密度（每单位体积的物理量），f是q的流量密度（每单位面积每单位时间的物理量）的矢量函数（vector function），s是q在每单位体积每单位时间的生成量。假若s>0则称 为“源点”；假若 s<0则称 s为“汇点”。假设 $\varphi$ 是没有产生或湮灭的守恒量（例如，电荷），则s=0，连续性方程变为。
