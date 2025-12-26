# DDPM学习笔记
> https://arxiv.org/pdf/2006.11239
论文在background部分给了大量信息，需要仔细推导
### 基础假设
论文首先给出了Diffusion Model的表示
```math
p_\theta(\mathbf x_0)=\int p_\theta(\mathbf x_{0:T})d\mathbf x_{1:T}
```
其中 $\mathbf x_0\sim q(\mathbf x_0)$ 表示现实数据， $\mathbf x_{1:T}=\mathbf x_1 ...\mathbf x_T$ 表示同维度的隐层状态

然后定义了前向过程(forward process)和反向过程(reverse process)。
- 反向过程是 $p_\theta(\mathbf x_{0:T})$ ，可以理解成从一个高斯分布（prior distribution）到真实数据分布的还原过程
- 前向过程也叫扩散过程（diffusion process）是p的后验近似 $q(\mathbf x_{1:T}|\mathbf x_0)$ ，可以理解成从真实数据到噪音的扩散过程
 
假设了q和p都是马尔科夫链,注意到p是反向的，q是正向的，贝叶斯展开后分别注意他们的条件序列时间
```math
p_{\theta}(\mathbf x_{0:T})=p_{\theta}(\mathbf x_0,\mathbf x_1,...\mathbf x_n)=p_\theta(\mathbf x_T)\prod_{t=1}^Tp_\theta（\mathbf x_{t-1}|\mathbf x_t）
\\
q(\mathbf x_{1:T}|\mathbf x_0)=\prod _{t=1}^Tq(\mathbf x_t|\mathbf x_{t-1})
```
这个就是马尔科夫性(Markov Property)在贝叶斯公式上的推导
> 马尔科夫性，假设后一个时刻的状态只跟前一个时刻的状态有关系，用公式表达即为 $p(x_t|x_{t-1},x_{t-2},...x_{0})=p(x_t|x_{t-1})$

### 前向过程假设
此时论文给出了一项重要假设：前向过程的是一个增加标准差为$\beta_t$ 的高斯噪声的过程：

$$
\mathbf x_{t} = \sqrt{1-\beta_t}\mathbf x_{t-1}+\sqrt{\beta_t}\epsilon_t, 其中\epsilon_t\sim \mathcal N(0,\mathcal I)
$$

根据重参数化技巧，单步的前向过程可以写成一个高斯分布
$$
\mathbf x_{t} \sim \mathcal N(\sqrt{1-\beta_t}\mathbf x_{t-1},\beta_t\mathcal I)
$$

### 证据下界
然后论文给出了训练过程中负对数似然上的变分下界，这里补充推导过程：
$$
\log p_{\theta}(\mathbf x_0)=\log \int p_\theta(\mathbf x_{0:T})d\mathbf x_{1:T}\\
=\log \int q(\mathbf x_{1:T}|\mathbf x_0)\frac {p_\theta(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}d\mathbf x_{1:T}\\
=\log \Bbb E_{q(\mathbf x_{1:T}|\mathbf x_0)}\bigg[\frac {p_\theta(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}\bigg]
$$
>注意这里把积分转化成期望的过程实际上用到了一个定理：随机变量的函数作期望： $\Bbb E[g(x)]=\int g(x)p(x)d(x)$

根据Jensen不等式
> Jensen不等式的来源是下凸函数（就是凸函数）的性质：对于任意0<t<1和下凸函数f(x)，以及其定义域上任意两点x1，x2，有
> $tf(x_1)+(1-t)f(x_2)>=f(tx_1+(1-t)x_2)$
> 然后推广到多参数的场景就是Jenson不等式:
> $f\Big(\displaystyle \sum_{i=1}^M\lambda_ix_i\Big)\le \displaystyle \sum _{i=1}^Mf(\lambda_ix_i)$
> 其中 $\displaystyle \sum _{i=1}^M\lambda_i=1,\lambda _i\ge0$，这个条件刚好和期望中的概率密度函数相符合，因为概率也是求和之后等为1，所以可以把Jensen不等式写成期望的形式：
> $f(E[x])\le E[f(x)]$，其中f(x)为下凸函数

因为对数函数是个上凸函数（凹函数），因此不等号得反过来
$$
\log p_{\theta}(\mathbf x_0)\ge \Bbb E_{q(\mathbf x_{1:T}|\mathbf x_0)}\bigg[\log \frac {p_\theta(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}\bigg]
$$
两边同时加负号，同时再求一次对 $\mathbf x_0 \sim p_{data}$ 的期望，就得到了论文中负对数似然的上界的第1个式子：
$$
\Bbb E[-\log p_\theta(\mathbf x_0)]\le \Bbb E_q\bigg[-\log \frac {p_\theta(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}\bigg]
$$
> 有一个关键信息论文隐去了，这里的 $\Bbb E_q$ 究竟指什么？根据我们的推导过程，右边式子这里的期望求了两次，分别是对两个分布求期望（这个信息非常重要，在后面的推导过程中期关键作用）：
> $$\Bbb E_q = \Bbb E_{\mathbf {x_0} \sim p_{data}, \mathbf x_{1:T} \sim q(\mathbf x_{1:T}|\mathbf x_0)}$$

右边的式子继续推导，先只看内部的分式项，将前面p和q的表达式带入得到
$$
\frac {p_\theta(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}=\frac {p_\theta(\mathbf x_T)\prod_{t=1}^Tp_\theta（\mathbf x_{t-1}|\mathbf x_t)}{\prod _{t=1}^Tq(\mathbf x_t|\mathbf x_{t-1})}
$$
两边同时取负对数
$$
-\log \frac {p_\theta(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}=-\log p_\theta(\mathbf x_T)-\sum_{t=1}^T\log p_\theta(\mathbf x_{t-1}|\mathbf x_t)+\sum_{t=1}^T\log q(\mathbf x_t|\mathbf x_{t-1}) \\
=-\log p_\theta(\mathbf x_T)-\sum_{t=1}^T\log \frac {p_\theta(\mathbf x_{t-1}|\mathbf x_{t})}{q(\mathbf x_t|\mathbf x_{t-1})}
$$
两边同时对q求期望（注意这里的q指的是上边的对两个分布的复合期望），就得到了论文中负对数似然的变分上界的第2个式子：
$$
\Bbb E_q\bigg[-\log \frac {p_\theta(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}\bigg]= 
\Bbb E_q\bigg[-\log p_\theta(\mathbf x_T)-\sum_{t=1}^T\log \frac {p_\theta(\mathbf x_{t-1}|\mathbf x_{t})}{q(\mathbf x_t|\mathbf x_{t-1})}\bigg]
$$
右边的式子可以继续推导，先根据贝叶斯公式有
$$
q(\mathbf x_t|\mathbf x_{t-1}) = \frac {q(\mathbf x_{t-1}|\mathbf x_{t},x_0)q(\mathbf x_t|\mathbf x_0)}{q(\mathbf x_{t-1}|\mathbf x_0)} 
$$
带入上式，得到
$$
\Bbb E_q\bigg[-\log p_\theta(\mathbf x_T)+\sum_{t=1}^T\log \frac {q(\mathbf x_t|\mathbf x_{t-1})} {p_\theta(\mathbf x_{t-1}|\mathbf x_{t})}\bigg]
=\Bbb E_q\bigg[-\log p_\theta(\mathbf x_T)+\sum_{t=1}^T{\log \frac {q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)} {p_\theta(\mathbf x_{t-1}|\mathbf x_{t})} + \log \frac {q(\mathbf x_t|\mathbf x_0)} {q(\mathbf x_{t-1}|\mathbf x_0)}}\bigg]
$$
观察上式求和项内部的第2项，能通过迭代相消
$$
\sum_{t=1}^T\log \frac {q(\mathbf x_t|\mathbf x_0)} {q(\mathbf x_{t-1}|\mathbf x_0)}
=\log \frac {q(\mathbf x_1|\mathbf x_0)} {q(\mathbf x_0|\mathbf x_0)} + \log \frac {q(\mathbf x_2|\mathbf x_0)} {q(\mathbf x_1|\mathbf x_0)} + \cdots + \log \frac {q(\mathbf x_T|\mathbf x_{0})} {q(\mathbf x_{T-1}|\mathbf x_0)}\\
=\log \frac {q(\mathbf x_T|\mathbf x_0)} {q(\mathbf x_0|\mathbf x_0)}\\
= \log q(\mathbf x_T|\mathbf x_0)
$$
因此原式等于
$$
\Bbb E_q\bigg[-\log p_\theta(\mathbf x_T)+\log q(\mathbf x_T|\mathbf x_0)+\sum_{t=1}^T\frac {q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)} {p_\theta(\mathbf x_{t-1}|\mathbf x_{t})}\bigg]\\
=\Bbb E_q\bigg[\frac {\log q(\mathbf x_T|\mathbf x_0)}{p_\theta(\mathbf x_T)} + \sum_{t=1}^T\frac {q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)} {p_\theta(\mathbf x_{t-1}|\mathbf x_{t})} \bigg]\\
$$
此时要将 $\Bbb E_q$ 拆开，得到是两重期望 $\Bbb E_q = \Bbb E_{\mathbf x_0 \sim p_{data},\mathbf x_{1:T-1}\sim q(\mathbf x_{1:T-1}|\mathbf x_0)}\big[ \Bbb E_{\mathbf x_T\sim q(\mathbf x_T|\mathbf x_{0})}[\cdot] \big]$ 为了书写方便将外边的期望依旧记为 $\Bbb E_q$，另外还需要将t=1的情况单独写出来，可以写出最终的上界表达式
$$
\Bbb E_q\bigg[ \Bbb E_{\mathbf x_T\sim q(\mathbf x_T|\mathbf x_{0})}\Big[\frac {\log q(\mathbf x_T|\mathbf x_0)} {p_\theta(\mathbf x_T)}\Big] + \sum_{t=2}^T\Bbb E_{\mathbf x_{t-1}\sim q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)}\Big[\frac {q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)}{p_\theta(\mathbf x_{t-1}|\mathbf x_t)}\Big]+\Bbb E_{x_0 \sim p_{data}}\Big[\log \frac { q(\mathbf x_0|\mathbf x_{1},\mathbf x_0)} {p_\theta(\mathbf x_0|\mathbf x_1)}\Big]\bigg]
$$
注意到期望内的前两项都符合KL散度的形式，第三项可以化简因为 $q(\mathbf x_0|\mathbf x_{1},\mathbf x_0)=1$ ，所以 $\log \frac { q(\mathbf x_0|\mathbf x_{1},\mathbf x_0)} {p_\theta(\mathbf x_0|\mathbf x_1)} = -\log p_\theta(\mathbf x_0|\mathbf x_1)$ ，所以上式可以写成最终KL散度的形式
$$
\Bbb E_q\bigg[ D_{KL}(q(\mathbf x_T|\mathbf x_0)||p_\theta(\mathbf x_T)) + \sum_{t=2}^TD_{KL}(q(\mathbf x_{t-1}|\mathbf x_t,x_0)||p_\theta(\mathbf x_{t-1}|\mathbf x_t)) - \log p_\theta(\mathbf x_0| \mathbf x_1)\bigg]
$$

### 前向过程的x_0推导
根据前向过程的单步假设，我们可以推导出从 $\mathbf x_0$ 到 $\mathbf x_t$ 的分布计算公式。我们先推导从 $\mathbf x_{t-2}$ 到 $\mathbf x_t$:
$$
\mathbf x_{t} = \sqrt{1-\beta_t}\mathbf x_{t-1}+\sqrt{\beta_t}\epsilon_t, 其中\epsilon_t\sim \mathbf N(0,\mathcal I)
$$
令 $\alpha_t=1-\beta_t, \bar \alpha_t=\prod_{s=1}^t\alpha_s$ 则有
$$
\mathbf x_{t} = \sqrt{\alpha_t}\mathbf x_{t-1}+\sqrt{\beta_t}\epsilon_t\\
= \sqrt{\alpha_t}(\sqrt {\alpha_{t-1}} \mathbf x_{t-2}+\sqrt \beta_{t-1}\epsilon_{t-1})+\sqrt {\beta_t}\epsilon_t\\
=\sqrt{\alpha_t}\sqrt{\alpha_{t-1}} \mathbf x_{t-2}+\sqrt{\alpha_t}\sqrt{\beta_{t-1}}\epsilon_{t-1}+\sqrt{\beta_t}\epsilon_t\\
$$
因为 $\epsilon_t \sim \mathcal N(0,\mathcal I)$，则有 $a\epsilon_t\sim \mathcal N(0,a^2\mathcal I)$ 所以上式的后两项可以写成一个高斯分布，它的均值是0，方差可以表示为:
$$
\alpha_t \beta_{t-1} + \beta_t\\
= \alpha_t(1-\alpha_{t-1}) + \beta_t\\
=1-\alpha_t\alpha_{t-1}\\
$$
所以表示成分布就有
$$
\mathbf x_{t} \sim \mathcal N(\sqrt{\alpha_t\alpha_{t-1}} \mathbf x_{t-2},\sqrt{1-\alpha_t \alpha_{t-1}}\mathcal I)
$$
推广到 $\mathbf x_0$
$$
\mathbf x_{t} \sim \mathcal N(\sqrt{\bar \alpha_t}\mathbf x_0,\sqrt{1-\bar \alpha_t}\mathcal I)
$$
逆重参数化得到
$$
\mathbf x_{t} = \sqrt{\bar \alpha_t}\mathbf x_0+\sqrt{1-\bar \alpha_t}\epsilon, 其中\epsilon\sim \mathcal N(0,\mathcal I)
$$

### 前向过程的后验推导


