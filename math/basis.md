# 一些基础
### 初等函数
初等函数是由六类基本初等函数通过有限次合法运算得到的函数。
> 六类基本初等函数：常数函数，幂函数，指数函数，对数函数，三角函数，反三角函数
> 合法运算：四则运算和复合运算。复合运算是指在一个函数中嵌套使用另一个函数例如 $sin(x^2), log(sin(x))$ 等

### 光滑和可导
函数的 $C^k$ 函数类来表示光滑性质，$k$ 表示函数的阶数。例如 $C^0$ 表示函数连续，$C^1$ 一阶导存在且连续，$C^2$ 二阶导存在且连续，以此类推到 $C^\infty$ 表示无限可微且连续。

**一般如无特殊说明，光滑都指 $C^\infty$ 函数类。**

导数存在且连续只能证明是 $C^1$ 函数类，而不能证明光滑。下面举几个例子：

- $f(x)=\begin{cases}
x^2\sin (\frac{1}{x}) & x\neq 0 \\
0 & x=0
\end{cases}$: 在 $\Bbb R$ 上连续且处处可导，但导函数在 $x=0$ 处不连续。属于 $C^0$ 函数类。
- $f(x)=\begin{cases}
x^3\sin (\frac{1}{x}) & x\neq 0 \\
0 & x=0
\end{cases}$: 在 $\Bbb R$ 上连续且处处可导且连续，但是二阶导在 $x=0$ 不存在。属于 $C^1$ 函数类。
- $f(x)=|x|^3$: 一阶导存在且连续，二阶导在 $x=0$ 处不存在，属于 $C^1$ 函数类。
- $f(x)=x^4|x|$: 一二三阶导均存在，四阶导在 $x=0$ 处不存在，属于 $C^3$ 函数类。

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

### 莱布尼茨积分法则
莱布尼兹积分法则(leibniz integral rule)也叫积分的微分规则，是数学中用来交换积分和微分运算顺序的方法。标准描述为：

设函数 $f(x,t)$ 在及其偏导数 $\frac {\partial f(x,t)}{\partial x}$ 区间 $R=\{(x,t):a \le x \le b,c\le t \le d\}$ 上连续，函数 $\alpha(t), \beta(t)$ 在区间[c,d]上可导，且 $a\le \alpha(t)\le b, a\le \beta(t)\le b$ ，对于积分

$$
F(t) = \int_{\alpha(t)}^{\beta(t)}f(x,t)dx
$$

其导数为

$$
F'(t)=\int_{\alpha(t)}^{\beta(t)} \frac {\partial f(x,t)}{\partial t}dx + f(\beta (t),t)\beta '(t)-f(\alpha (t),t)\alpha '(t)
$$

这是一般形式，如果积分区域与函数变量无关（固定区间或者全空间），其表达形式可简化为：

$$
\frac {d}{dt}\int f(x,t)dx=\int \frac {d f(x,t)}{dt}dx
$$

使用莱布尼兹积分法则一定要注意前提条件：函数和偏导数连续