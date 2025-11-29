以下是对欧拉公式 $ e^{i\theta} = \cos \theta + i \sin \theta $（其中 $\theta \in \mathbb{R}$）的**严格证明**。证明基于复分析中幂级数的定义和性质，这是最严谨且基础的方法（避免循环论证）。证明分为四个步骤：定义复指数与三角函数、验证级数收敛性、分离实虚部、得出等式。

---

### **步骤 1：定义复指数函数与三角函数（幂级数定义）**
在复分析中，以下幂级数对**所有复数 $z$** 绝对收敛，因此可作为严格定义：
- **复指数函数**：
  $$
  e^z \overset{\text{def}}{=} \sum_{n=0}^{\infty} \frac{z^n}{n!}
  $$
- **复余弦函数**：
  $$
  \cos z \overset{\text{def}}{=} \sum_{n=0}^{\infty} \frac{(-1)^n z^{2n}}{(2n)!}
  $$
- **复正弦函数**：
  $$
  \sin z \overset{\text{def}}{=} \sum_{n=0}^{\infty} \frac{(-1)^n z^{2n+1}}{(2n+1)!}
  $$

> **注**：当 $z$ 为实数时，这些定义与实分析中的标准定义一致。复指数函数的定义满足 $e^{z_1 + z_2} = e^{z_1} e^{z_2}$ 等基本性质（可通过柯西乘积证明），但本证明无需此性质。

---

### **步骤 2：验证级数收敛性**
对 $z = i\theta$（$\theta \in \mathbb{R}$），考虑级数：
$$
e^{i\theta} = \sum_{n=0}^{\infty} \frac{(i\theta)^n}{n!}
$$
- **收敛性证明**：  
  由比值判别法（Ratio Test），
  $$
  \lim_{n \to \infty} \left| \frac{(i\theta)^{n+1} / (n+1)!}{(i\theta)^n / n!} \right| = \lim_{n \to \infty} \frac{|\theta|}{n+1} = 0 < 1.
  $$
  因此级数**绝对收敛**（对任意 $\theta \in \mathbb{R}$）。  
- **关键推论**：  
  绝对收敛性保证级数可**任意重排项**而不改变和（复级数的重排定理）。

---

### **步骤 3：分离实部与虚部**
将 $e^{i\theta}$ 按 $n$ 的奇偶性拆分为实部和虚部：
$$
e^{i\theta} = \sum_{n=0}^{\infty} \frac{(i\theta)^n}{n!} = \sum_{k=0}^{\infty} \frac{(i\theta)^{2k}}{(2k)!} + \sum_{k=0}^{\infty} \frac{(i\theta)^{2k+1}}{(2k+1)!}
$$
利用 $i^{2k} = (i^2)^k = (-1)^k$ 和 $i^{2k+1} = i \cdot i^{2k} = i (-1)^k$：
$$
e^{i\theta} = \sum_{k=0}^{\infty} \frac{(-1)^k \theta^{2k}}{(2k)!} + i \sum_{k=0}^{\infty} \frac{(-1)^k \theta^{2k+1}}{(2k+1)!}
$$

---

### **步骤 4：与三角函数的幂级数匹配**
根据步骤 1 的定义：
- $\cos \theta = \sum_{k=0}^{\infty} \frac{(-1)^k \theta^{2k}}{(2k)!}$（实部），
- $\sin \theta = \sum_{k=0}^{\infty} \frac{(-1)^k \theta^{2k+1}}{(2k+1)!}$（虚部系数）。

代入上式：
$$
e^{i\theta} = \underbrace{\sum_{k=0}^{\infty} \frac{(-1)^k \theta^{2k}}{(2k)!}}_{\cos \theta} + i \underbrace{\sum_{k=0}^{\infty} \frac{(-1)^k \theta^{2k+1}}{(2k+1)!}}_{\sin \theta} = \cos \theta + i \sin \theta.
$$

---

### **严格性说明**
1. **无循环论证**：  
   证明仅依赖幂级数的**定义**和**收敛性**，未假设欧拉公式或复指数的其他性质（如 $e^{i\theta}$ 的几何意义）。
2. **收敛性保障**：  
   绝对收敛性允许重排级数项（关键步骤），这在复分析中是严格的（由 Weierstrass M-判别法或比值判别法保证）。
3. **适用范围**：  
   证明对任意实数 $\theta$ 成立（$\theta \in \mathbb{R}$）。当 $\theta = \pi$ 时，即得欧拉恒等式 $e^{i\pi} + 1 = 0$。
4. **与实函数的一致性**：  
   当 $\theta = 0$ 时，$e^{i \cdot 0} = 1$，且 $\cos 0 + i \sin 0 = 1 + i \cdot 0 = 1$，验证成立。

---

### **附：其他证明方法的局限性**
- **微分方程法**（如设 $f(\theta) = e^{-i\theta}(\cos \theta + i \sin \theta)$ 并证 $f'(\theta)=0$）：  
  需先定义复指数函数的导数，通常仍需幂级数基础。
- **极限法**（$e^{i\theta} = \lim_{n \to \infty} (1 + \frac{i\theta}{n})^n$）：  
  需严格处理复数模与辐角的极限，过程繁琐且依赖三角恒等式。
- **泰勒展开法**：  
  本质与幂级数法相同，但需预先验证复函数的泰勒展开唯一性。

> **结论**：幂级数法是最直接、严格且基础的证明，符合现代复分析的公理化体系。

$$
\boxed{e^{i\theta} = \cos \theta + i \sin \theta}
$$