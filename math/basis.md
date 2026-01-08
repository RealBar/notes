# 一些基础
### 初等函数
初等函数是由六类基本初等函数通过有限次合法运算得到的函数。
> 六类基本初等函数：常数函数，幂函数，指数函数，对数函数，三角函数，反三角函数
> 合法运算：四则运算和复合运算。复合运算是指在一个函数中嵌套使用另一个函数例如 $sin(x^2), log(sin(x))$ 等

### 微分同胚映射(diffeomorphic map)
是连接两个光滑流形的，“双向光滑可逆映射”

### $\nabla$ 算子
读作del或者nabla或者哈密顿算子。可以表示三种意思：梯度、散度和旋度（Gradient, Divergence & Curl），分别通过数乘、点乘和叉乘作用于不同函数。例如
- 梯度： $\nabla f(x,y,z) = (\frac{\partial f(x,y,z)}{\partial x} , \frac{\partial f(x,y,z)}{\partial y} , \frac{\partial f(x,y,z)}{\partial z})$ 
- 散度： $\nabla \cdot \mathbf f = \frac{\partial \mathbf f}{\partial x} + \frac{\partial \mathbf f}{\partial y} + \frac{\partial \mathbf f}{\partial z}$
- 旋度： $\nabla \times \mathbf f = (\frac{\partial \mathbf f}{\partial y} - \frac{\partial \mathbf f}{\partial z}, \frac{\partial \mathbf f}{\partial z} - \frac{\partial \mathbf f}{\partial x}, \frac{\partial \mathbf f}{\partial x} - \frac{\partial \mathbf f}{\partial y})$

麦克斯韦方程组：
$$
\begin{cases}
\nabla \cdot \mathbf E = \frac{\rho_e}{\epsilon_0} \\
\nabla \times \mathbf E = -\frac{\partial \mathbf B}{\partial t} \\
\nabla \cdot \mathbf B = 0\\
\nabla \times \mathbf B = \mu_0 \mathbf J + \mu_0 \epsilon_0 \frac{\partial \mathbf E}{\partial t} 
\end{cases}
$$

### 怎么理解散度和旋度
散度一般通过向量场的通量体密度来理解，散度本质上是单位体积的通量。

通量：向量场通过一个封闭曲面的总量 $\int\int\mathbf f\cdot d\mathbf S$
散度：当这个封闭体的体积趋近到一点，通量与体积的比值 $\lim_{V\to 0}\frac{1}{V} \int\int\mathbf f\cdot d\mathbf S$
表示单位体积内，向量场 $\mathbf f$ 通过一个封闭曲面的总量。

### 光滑和处处可导

