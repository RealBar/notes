# DDPM
## 论文详细解读
> https://arxiv.org/pdf/2006.11239
### Background
论文首先给出了Diffusion Model的表示
```math
p_\theta(\bm x_0)=\int p_\theta(\bm x_{0:T})d\bm x_{1:T}
```
其中$\bm x_0\sim q(\bm x_0)$表示现实数据， $\bm x_{1:T}=\bm x_1 ...\bm x_T$表示同维度的隐层状态

然后定义了前向过程(forward process)和反向过程(reverse process)。
- 反向过程是$p_\theta(\bm x_{0:T})$，可以理解成从一个高斯分布（prior distribution）到真实数据分布的还原过程
- 前向过程也叫扩散过程（diffusion process）是$q(\bm x_{1:T}|\bm x_0)$，可以理解成从真实数据到噪音的扩散过程
 
假设了q和p都是马尔科夫链,注意到p是反向的，q是正向的，贝叶斯展开后分别注意他们的条件序列时间
```math
p_{\theta}(\bm x_{0:T})=p_{\theta}(\bm x_0,\bm x_1,...\bm x_n)=p_\theta(\bm x_T)\displaystyle\prod_{t=1}^Tp_\theta（\bm x_{t-1}|\bm x_t）
\\
q(\bm x_{1:T}|\bm x_0)=\displaystyle \prod _{t=1}^Tq(\bm x_t|\bm x_{t-1})
```
这个就是马尔科夫性(Markov Property)在贝叶斯公式上的推导
> 马尔科夫性，假设后一个时刻的状态只跟前一个时刻的状态有关系，用公式表达即为$p(x_t|x_{t-1},x_{t-2},...x_{0})=p(x_t|x_{t-1})$

然后论文给出了训练过程中负对数似然上的变分下界，这里补充推导过程：
```math
\log p_{\theta}(\bm x_0)=\log \int p_\theta(\bm x_{0:T})d\bm x_{1:T}\\
=\log \int q(\bm x_{1:T}|\bm x_0)\frac {p_\theta(\bm x_{0:T})}{q(\bm x_{1:T}|\bm x_0)}d\bm x_{1:T}\\
=\log E_{q(\bm x_{1:T}|\bm x_0)}\bigg[\frac {p_\theta(\bm x_{0:T})}{q(\bm x_{1:T}|\bm x_0)}\bigg]
```
>注意这里把积分转化成期望的过程实际上用到了一个定理：随机变量的函数作期望：$E[g(x)]=\int g(x)p(x)d(x)$

根据Jensen不等式
> Jensen不等式的来源是下凸函数（就是凸函数）的性质：对于任意0<t<1和下凸函数f(x)，以及其定义域上任意两点x1，x2，有
> $tf(x_1)+(1-t)f(x_2)>=f(tx_1+(1-t)x_2)$
> 然后推广到多参数的场景就是Jenson不等式:
> $f\Big(\displaystyle \sum_{i=1}^M\lambda_ix_i\Big)\le \displaystyle \sum _{i=1}^Mf(\lambda_ix_i)$
> 其中$\displaystyle \sum _{i=1}^M\lambda_i=1,\lambda _i\ge0$，这个条件刚好和期望中的概率密度函数相符合，因为概率也是求和之后等为1，所以可以把Jensen不等式写成期望的形式：
> $f(E[x])\le E[f(x)]$，其中f(x)为下凸函数

因为对数函数是个上凸函数（凹函数），因此不等号得反过来
```math
\log p_{\theta}(\bm x_0)\ge E_{q(\bm x_{1:T}|\bm x_0)}\bigg[\log \frac {p_\theta(\bm x_{0:T})}{q(\bm x_{1:T}|\bm x_0)}\bigg]
```
两边同时加符号，同时再求一次对q的期望（不等号右边是一个期望，可以理解成一个常数，所以再求一次期望也是他本身），就得到了负对数似然的变分下界：
```math
E[-\log p_\theta(x_0)]\le E_{q(\bm x_{1:T}|\bm x_0)}\bigg[-\log \frac {p_\theta(\bm x_{0:T})}{q(\bm x_{1:T}|\bm x_0)}\bigg]
```



