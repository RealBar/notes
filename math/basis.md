# 一些基础
### 初等函数
初等函数是由六类基本初等函数通过有限次合法运算得到的函数。
> 六类基本初等函数：常数函数，幂函数，指数函数，对数函数，三角函数，反三角函数
> 合法运算：四则运算和复合运算。复合运算是指在一个函数中嵌套使用另一个函数例如 $sin(x^2), log(sin(x))$ 等

### 光滑和可导
函数的 $C^k$ 函数类来表示光滑性质，$k$ 表示函数的阶数。例如 $C^0$ 表示函数连续，$C^1$ 一阶导存在且连续，$C^2$ 二阶导存在且连续，以此类推到 $C^\infty$ 表示无限可微且连续。

**一般如无特殊说明，光滑都指 $C^\infty$ 函数类。** 导数存在且连续只能证明是 $C^1$ 函数类，而不能证明光滑。

下面举几个例子：

$$
f(x)=\begin{cases}
x^2\sin (\frac{1}{x}) & x\neq 0 \\
0 & x=0
\end{cases}
$$

在 $\Bbb R$ 上连续且处处可导，但导函数在 $x=0$ 处不连续。属于 $C^0$ 函数类。

$$
f(x)=\begin{cases}
x^3\sin (\frac{1}{x}) & x\neq 0 \\
0 & x=0
\end{cases}
$$

在 $\Bbb R$ 上连续且处处可导且连续，但是二阶导在 $x=0$ 不存在。属于 $C^1$ 函数类。

$$
f(x)=|x|^3
$$

一阶导存在且连续，二阶导在 $x=0$ 处不存在，属于 $C^1$ 函数类。

$$
f(x)=x^4|x|
$$

一二三阶导均存在，四阶导在 $x=0$ 处不存在，属于 $C^3$ 函数类。

### 微分同胚映射(diffeomorphic map)
从纯数学上，diffeomorphic可以完全等价于以下三个性质：
1. 光滑
2. 双射，即单射+满射。（因为双射，所以可逆）
3. 逆映射也光滑

微分同胚可以在拓扑学上有更深入的解释，后面再补充。

### $\nabla$ 算子
读作del或者nabla，也被称为哈密顿算子。可以表示三种意思：梯度（Gradient）、散度（Divergence）和旋度（Curl），分别通过数乘、点乘和叉乘作用于不同函数。例如
- 梯度： $\nabla f(x,y,z) = (\frac{\partial f(x,y,z)}{\partial x} , \frac{\partial f(x,y,z)}{\partial y} , \frac{\partial f(x,y,z)}{\partial z})$ 
- 散度： $\nabla \cdot \mathbf f = \frac{\partial \mathbf f}{\partial x} + \frac{\partial \mathbf f}{\partial y} + \frac{\partial \mathbf f}{\partial z}$
- 旋度： $\nabla \times \mathbf f = (\frac{\partial \mathbf f}{\partial y} - \frac{\partial \mathbf f}{\partial z}, \frac{\partial \mathbf f}{\partial z} - \frac{\partial \mathbf f}{\partial x}, \frac{\partial \mathbf f}{\partial x} - \frac{\partial \mathbf f}{\partial y})$

### 怎么理解散度和旋度
散度 $\nabla \cdot \mathbf f$ 一般通过向量场的通量体密度来理解：  
通量：向量场通过一个封闭曲面的总量 $\int\int\mathbf f\cdot d\mathbf S$ 。注意S的方向即为封闭曲面微元的向外法向量。  
散度：当这个封闭体的体积趋近到一点，通量与体积的比值即为散度: $\nabla \cdot \mathbf f = \lim_{V\to 0}\frac{1}{V}\int\int\mathbf f\cdot d\mathbf S$ 。注意积分内的运算符是点乘，所以结果是一个标量。  

旋度 $\nabla \times \mathbf f$ 向量场在点的旋度是环量（旋转趋势）的面密度：  
环量：闭合路径上的“旋转趋势总量”，它量化了向量场在一条（平面的）闭合曲线上驱动流体（或场）旋转的净效果，正环量表示会驱动流体逆时针旋转，负环量表示会驱动流体顺时针旋转，零环量表示不会驱动流体旋转。计算公式： $\Gamma=\oint_c \mathbf f \cdot d\mathbf l$ 。注意l的方向为曲线在该点切线方向，切线环绕的方向按照右手定则确定了法向量的方向。  
旋度：当这个封闭面的面积趋近到0，环量与面积的比值即为旋度**的模长**: $(\nabla \times \mathbf f) \cdot \mathbf n = \lim_{S\to 0}\frac{1}{S}\oint_c \mathbf f \cdot d\mathbf l$ 。可以看到，环量本身是一个标量，然后我们用封闭面的法向量定义了旋度的方向。

麦克斯韦方程组：

$$
\begin{cases}
\nabla \cdot \mathbf E = \frac{\rho_e}{\epsilon_0} \\
\nabla \times \mathbf E = -\frac{\partial \mathbf B}{\partial t} \\
\nabla \cdot \mathbf B = 0\\
\nabla \times \mathbf B = \mu_0 \mathbf J + \mu_0 \epsilon_0 \frac{\partial \mathbf E}{\partial t} \\
\end{cases}
$$

### 莱布尼茨积分法则(Leibniz integral rule)
也叫积分的微分规则，是数学中用来交换积分和微分运算顺序的方法。

固定积分区域情况描述为：对于含参积分 $\int_Zf(t,z)dz$ ，若满足以下条件：
1.  $f(t,z)$ 在t的定义域内关于t可微且连续；
2.  存在可积函数 $g(z)$ 使得 $|\frac {\partial f}{\partial t}| \le g(z)$ 对于任意t一致成立；
3.  对于任意t，积分 $\int_Zf(t,z)dz$ 收敛；

则有 

$$
\frac {\partial}{\partial t}\int_Zf(t,z)dz=\int_Z\frac {\partial}{\partial t}f(t,z)dz
$$

更一般情况下的莱布尼兹积分法则表述为：设函数 $f(x,t)$ 及偏导 $\frac {\partial f(x,t)}{\partial t}$ 在区间 $R=\{(x,t):a \le x \le b,c\le t \le d\}$ 上连续，函数 $\alpha(t), \beta(t)$ 在区间[c,d]上可导，且 $a\le \alpha(t)\le b, a\le \beta(t)\le b$ ，对于积分

$$
F(t) = \int_{\alpha(t)}^{\beta(t)}f(x,t)dx
$$

其导数为

$$
F'(t)=\int_{\alpha(t)}^{\beta(t)} \frac {\partial f(x,t)}{\partial t}dx + f(\beta (t),t)\beta '(t)-f(\alpha (t),t)\alpha '(t)
$$

### 富比尼定理（Fubini's theorem）
它的核心作用是：在满足f(x,y)在A,B上可积的条件下，允许我们将一个“重积分”转化为“累次积分”（或称迭代积分），或者可以交换积分的顺序而不改变结果。其黎曼积分表述如下：

对于 $x\in A, y\in B$ 函数 $f(x,y)$ 可积，则有

$$
\iint_{A\times B} f(x,y)dxdy = \int_A (\int_B f(x,y)dy )dx=\int_B (\int_A f(x,y)dx )dy
$$

### 连续性方程
概率形式的连续性方程如（flow matching 中，u是向量场，p是概率密度）：

$$
\frac {\partial p_t(x)} {\partial t} +div(p_t(x)u_t(x))=0
$$

它是连续性方程通用形式的一种特殊情况。通用的连续性方程为：

$$
\frac {\partial \rho} {\partial t} +\nabla \cdot \mathbf f = s
$$

> 这里的 $\nabla \cdot$ 是求散度，等价于 $div(\cdot)$ 。 这里的s是目标物理量在每单位体积每单位时间的生成量。假若s>0则称 为“源点”；假若 s<0则称 s为“汇点”。假设 $\rho$ 是没有产生或湮灭的守恒量（例如，电荷，质量，概率），则s=0，连续性方程变为更通用的守恒形式：

$$
\frac {\partial \rho} {\partial t} +\nabla \cdot \mathbf f = 0
$$

这里一定要清楚 $\rho$ 和 $\mathbf f$ 的关系：
- $\rho$ 是某物理量的密度，是该物理量在每单位体积的物理量。
- $\mathbf f$ 是该物理量的流量密度，单位时间通过单位面积的物理量的矢量函数，单位例如质量对应 $kg/(m^2\cdot s)$ ，电荷对应 $A/m^2 = C/(m^2\cdot s)$ 。如果能知道该物理量的密度场和速度场v，则可以写成 $\mathbf f = \rho \mathbf v$ 。注意这里的 $\rho$ 是密度场，是个标量场； $\mathbf v$ 是速度场，是个矢量场。

举例：经典流体力学中的连续性方程为

$$
\frac {\partial \rho} {\partial t} +\nabla \cdot (\rho\mathbf v) = 0
$$

其中， $\rho$ 是密度场，可以理解为每个质点的质量； $\mathbf v$ 是速度矢量场，可以理解为处于每个位置的质点的速度矢量， $\rho \mathbf v$ 表示质量流密度。

现在回到(2)式，就是一个连续性方程，研究的对象是概率密度。