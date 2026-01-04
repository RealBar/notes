# DDPM学习笔记
>论文： https://arxiv.org/pdf/2006.11239  
## 背景知识
论文在background部分给了大量信息，需要仔细推导
### 基础假设
论文在background部分给出了Diffusion Model的表示
```math
p_\theta(\mathbf x_0)=\int p_\theta(\mathbf x_{0:T})d\mathbf x_{1:T}
```
其中 $\mathbf x_0\sim q(\mathbf x_0)$ 表示现实数据， $\mathbf x_{1:T}=\mathbf x_1 ...\mathbf x_T$ 表示同维度的隐藏态

然后定义了前向过程(forward process)和反向过程(reverse process)。
- 反向过程是 $p_\theta(\mathbf x_{0:T})$ ，可以理解成从一个高斯分布（prior distribution）到真实数据分布的还原过程
- 前向过程也叫扩散过程（diffusion process）是p的后验近似 $q(\mathbf x_{1:T}|\mathbf x_0)$ ，可以理解成从真实数据到噪音的扩散过程
 
假设了q和p都是马尔科夫链，注意到p是反向的，q是正向的，贝叶斯展开后分别注意他们的条件序列时间
```math
p_{\theta}(\mathbf x_{0:T})=p_{\theta}(\mathbf x_0,\mathbf x_1,...\mathbf x_n)=p_\theta(\mathbf x_T)\prod_{t=1}^Tp_\theta（\mathbf x_{t-1}|\mathbf x_t）
\\
q(\mathbf x_{1:T}|\mathbf x_0)=\prod _{t=1}^Tq(\mathbf x_t|\mathbf x_{t-1})
```
这个就是马尔科夫性(Markov Property)在贝叶斯公式上的推导
> 马尔科夫性，假设后一个时刻的状态只跟前一个时刻的状态有关系，用公式表达即为 $p(x_t|x_{t-1},x_{t-2},...x_{0})=p(x_t|x_{t-1})$

### 前向过程假设
background部分论文给出了一项重要假设：前向过程的是一个增加标准差为 $\beta_t$ 的高斯噪声的过程：

$$
\begin{equation}
\mathbf x_{t} = \sqrt{1-\beta_t}\mathbf x_{t-1}+\sqrt{\beta_t}\epsilon_t, 其中\epsilon_t\sim \mathcal N(0,\mathcal I)
\end{equation}
$$

**为什么系数必须是 $\sqrt{1-\beta_t}$ 和 $\sqrt{\beta_t}$？**
这个设计的核心目的是**保持方差守恒 (Variance Preserving)**，防止数据在加噪过程中方差爆炸或消失。
假设输入数据 $\mathbf x_{t-1}$ 已经经过归一化，即均值为0，方差为 $\mathbf I$（$\text{Var}(\mathbf x_{t-1}) = 1$）。
我们希望加噪后的 $\mathbf x_t$ 仍然保持单位方差，即 $\text{Var}(\mathbf x_t) = 1$。
根据方差性质 $\text{Var}(aX + bY) = a^2\text{Var}(X) + b^2\text{Var}(Y)$（假设 $X, Y$ 独立），对于公式 $\mathbf x_t = a \cdot \mathbf x_{t-1} + b \cdot \epsilon_t$：

$$
\text{Var}(\mathbf x_t) = a^2 \underbrace{\text{Var}(\mathbf x_{t-1})}_{1} + b^2 \underbrace{\text{Var}(\epsilon_t)}_{1} = a^2 + b^2
$$

如果我们定义注入噪声的方差为 $\beta_t$（即 $b^2 = \beta_t \Rightarrow b = \sqrt{\beta_t}$），那么为了保持 $\text{Var}(\mathbf x_t) = 1$，必须有：

$$
a^2 + \beta_t = 1 \implies a = \sqrt{1-\beta_t}
$$

这就是为什么 $\mathbf x_{t-1}$ 前面的系数必须是 $\sqrt{1-\beta_t}$ 的原因。如果不乘这个系数，随着 $t$ 增大，数据的方差会变成 $1 + \sum \beta_t$，导致数值不稳定（Variance Exploding）。
根据重参数化技巧，单步的前向过程可以写成一个高斯分布

$$
\mathbf x_{t} \sim \mathcal N(\sqrt{1-\beta_t}\mathbf x_{t-1},\beta_t\mathcal I)
$$

### 证据下界和损失函数初步
background部分论文给出了训练过程中负对数似然上的证据下界，这里补充推导过程：

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

两边同时加负号，同时再求一次对 $\mathbf x_0 \sim q(\mathbf x_0)$ 的期望，就得到了论文中负对数似然的上界的第1个式子：

$$
\Bbb E[-\log p_\theta(\mathbf x_0)]\le \Bbb E_q\bigg[-\log \frac {p_\theta(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}\bigg]
$$

> 有一个关键信息论文隐去了，这里的 $\Bbb E_q$ 究竟指什么？根据我们的推导过程，右边式子这里的期望求了两次，分别是对两个分布求期望，一次是对 $\mathbf x_{1:T}\sim p(\mathbf x_{1:T}|\mathbf x_0)$ ，一次是对 $\mathbf x_0 \sim q(\mathbf x_0)$ ：
> $$\Bbb E_q = \Bbb E_{\mathbf {x_0} \sim q(\mathbf x_0), \mathbf x_{1:T} \sim q(\mathbf x_{1:T}|\mathbf x_0)}=\Bbb E_{\mathbf x_{0:T}\sim q(\mathbf x_{0:T})}$$
> 所以 $\Bbb E_q$ 事实上是对 $x_{0:T}$ 这T+1个变量的联合分布的期望。这一点在后边的推导中至关重要。

右边的式子继续推导，先只看内部的分式项，将前面p和q的表达式带入得到

$$
\frac {p_\theta(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}=\frac {p_\theta(\mathbf x_T)\prod_{t=1}^Tp_\theta（\mathbf x_{t-1}|\mathbf x_t)}{\prod _{t=1}^Tq(\mathbf x_t|\mathbf x_{t-1})}
$$

两边同时取负对数

$$
-\log \frac {p_\theta(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}=-\log p_\theta(\mathbf x_T)-\sum_{t=1}^T\log p_\theta(\mathbf x_{t-1}|\mathbf x_t)+\sum_{t=1}^T\log q(\mathbf x_t|\mathbf x_{t-1}) \\
=-\log p_\theta(\mathbf x_T)-\sum_{t=1}^T\log \frac {p_\theta(\mathbf x_{t-1}|\mathbf x_{t})}{q(\mathbf x_t|\mathbf x_{t-1})}
$$

两边同时对q求期望（注意这个期望指的是上变说过的复合期望）

$$
\begin{equation}
\Bbb E_q\bigg[-\log \frac {p_\theta(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}\bigg]= 
\Bbb E_q\bigg[-\log p_\theta(\mathbf x_T)-\sum_{t=1}^T\log \frac {p_\theta(\mathbf x_{t-1}|\mathbf x_{t})}{q(\mathbf x_t|\mathbf x_{t-1})}\bigg]
\end{equation}
$$

(2)式即为论文中的(3)。(2)可以继续推导，先根据贝叶斯公式和前向过程的马尔科夫性有：

$$
q(\mathbf x_t|\mathbf x_{t-1}) = q(\mathbf x_t|\mathbf x_{t-1},\mathbf x_0)=\frac {q(\mathbf x_{t-1}|\mathbf x_{t},x_0)q(\mathbf x_t|\mathbf x_0)}{q(\mathbf x_{t-1}|\mathbf x_0)} 
$$

带入(2)式，得到

$$
\Bbb E_q\bigg[-\log p_\theta(\mathbf x_T)+\sum_{t=1}^T\log \frac {q(\mathbf x_t|\mathbf x_{t-1})} {p_\theta(\mathbf x_{t-1}|\mathbf x_{t})}\bigg]
=\Bbb E_q\bigg[-\log p_\theta(\mathbf x_T)+\sum_{t=1}^T{\log \frac {q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)} {p_\theta(\mathbf x_{t-1}|\mathbf x_{t})} + \log \frac {q(\mathbf x_t|\mathbf x_0)} {q(\mathbf x_{t-1}|\mathbf x_0)}}\bigg]
$$

观察上式求和项内部的第2项，能通过迭代相消

$$
\sum_{t=1}^T\log \frac {q(\mathbf x_t|\mathbf x_0)} {q(\mathbf x_{t-1}|\mathbf x_0)}
=\log \frac {q(\mathbf x_1|\mathbf x_0)} {q(\mathbf x_0|\mathbf x_0)} + \log \frac {q(\mathbf x_2|\mathbf x_0)} {q(\mathbf x_1|\mathbf x_0)} + \cdots + \log \frac {q(\mathbf x_T|\mathbf x_{0})} {q(\mathbf x_{T-1}|\mathbf x_0)}
$$

$$
=\log \frac {q(\mathbf x_T|\mathbf x_0)} {q(\mathbf x_0|\mathbf x_0)}
$$

$$
= \log q(\mathbf x_T|\mathbf x_0)
$$

因此原式等于

$$
\Bbb E_q\bigg[-\log p_\theta(\mathbf x_T)+\log q(\mathbf x_T|\mathbf x_0)+\sum_{t=1}^T\log \frac {q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)} {p_\theta(\mathbf x_{t-1}|\mathbf x_{t})}\bigg]
$$

$$
\begin{equation}
=\Bbb E_q\bigg[\log\frac { q(\mathbf x_T|\mathbf x_0)}{p_\theta(\mathbf x_T)} + \sum_{t=1}^T\log \frac {q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)} {p_\theta(\mathbf x_{t-1}|\mathbf x_{t})} \bigg]
\end{equation}
$$

还记得我们上边说过的 $\Bbb E_q$ 真实的意义对 $x_{0:T}$ 这T+1个变量的联合分布的期望 $\Bbb E_{\mathbf x_{0:T}\sim q(\mathbf x_{0:T})}$ ，所以我们可以将其中的目标随机变量分离出来，写成两重期望的形式。例如对于随机变量 $\mathbf x_t$ 

$$
\Bbb E_q=\Bbb E_{\mathbf x_{0:t-1,t+1:T}\sim q(\mathbf x_{0:t-1,t+1:T})}\big[\Bbb E_{\mathbf x_t\sim q(\mathbf x_t)}[\cdot]\big]
$$

注意到，对于分离到内层期望的随机变量，我们可以在最外层再求一次对它的期望，因为这个变量在内层已经被积分掉了，所以最外层对它再求期望并不会改变结果

$$
\Bbb E_{\mathbf x_{0:t-1,t+1:T}\sim q(\mathbf x_{0:t-1,t+1:T})}\big[\Bbb E_{\mathbf x_t\sim q(\mathbf x_t)}[\cdot]\big]
$$

$$
=\Bbb E_{\mathbf x_t\sim q(\mathbf x_t)} \Big[ \Bbb E_{\mathbf x_{0:t-1,t+1:T}\sim q(\mathbf x_{0:t-1,t+1:T})}\big[\Bbb E_{\mathbf x_t\sim q(\mathbf x_t)}[\cdot]\big]\Big]
$$

$$
=\Bbb E_q\big[\Bbb E_{\mathbf x_t\sim q(\mathbf x_t)}[\cdot]\big]
$$

将上式带入(3)并选择各项目标变量，另外还需要将t=1的情况单独写出来

$$
\Bbb E_q\bigg[ \Bbb E_{\mathbf x_T\sim q(\mathbf x_T|\mathbf x_{0})}\Big[\log \frac {q(\mathbf x_T|\mathbf x_0)} {p_\theta(\mathbf x_T)}\Big] + \sum_{t=2}^T\Bbb E_{\mathbf x_{t-1}\sim q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)}\Big[\frac {q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)}{p_\theta(\mathbf x_{t-1}|\mathbf x_t)}\Big]+\Bbb E_{\mathbf x_0 \sim q(\mathbf x_0)}\Big[\log \frac { q(\mathbf x_0|\mathbf x_{1},\mathbf x_0)} {p_\theta(\mathbf x_0|\mathbf x_1)}\Big]\bigg]
$$

注意到期望内的前两项都符合KL散度的形式，第三项可以化简因为 $q(\mathbf x_0|\mathbf x_{1},\mathbf x_0)=1$ ，所以 $\log \frac { q(\mathbf x_0|\mathbf x_{1},\mathbf x_0)} {p_\theta(\mathbf x_0|\mathbf x_1)} = -\log p_\theta(\mathbf x_0|\mathbf x_1)$ ，所以上式可以写成最终KL散度的形式

$$
\begin{equation}
\Bbb E_q\bigg[ D_{KL}(q(\mathbf x_T|\mathbf x_0)||p_\theta(\mathbf x_T)) + \sum_{t=2}^TD_{KL}(q(\mathbf x_{t-1}|\mathbf x_t,x_0)||p_\theta(\mathbf x_{t-1}|\mathbf x_t)) - \log p_\theta(\mathbf x_0| \mathbf x_1)\bigg]
\end{equation}
$$

此式即为原文中的式(5)。注意论文在background中给出此式之后，后面的过程都是围绕这个式子展开。它把这个式子的三项分别称为 $L_T,L_{t-1},L_0$ 。

### 前向过程的x_0推导
根据前向过程的单步假设，我们可以推导出从 $\mathbf x_0$ 到 $\mathbf x_t$ 的分布计算公式。我们先推导从 $\mathbf x_{t-2}$ 到 $\mathbf x_t$:
令 $\alpha_t=1-\beta_t, \bar \alpha_t=\prod_{s=1}^t\alpha_s$ ，根据式(1)有

$$
\mathbf x_{t} = \sqrt{\alpha_t}\mathbf x_{t-1}+\sqrt{\beta_t}\epsilon_t
$$

$$
= \sqrt{\alpha_t}(\sqrt {\alpha_{t-1}} \mathbf x_{t-2}+\sqrt \beta_{t-1}\epsilon_{t-1})+\sqrt {\beta_t}\epsilon_t
$$

$$
\begin{equation}
=\sqrt{\alpha_t}\sqrt{\alpha_{t-1}} \mathbf x_{t-2}+\sqrt{\alpha_t}\sqrt{\beta_{t-1}}\epsilon_{t-1}+\sqrt{\beta_t}\epsilon_t\\
\end{equation}
$$

因为 $\epsilon_t,\epsilon_{t-1} \sim \mathcal N(0,\mathcal I)$，且根据高斯分布的性质 $a\epsilon_t+b\epsilon_{t-1} \sim \mathcal N(0,(a^2+b^2)\mathcal I)$ ，所以上式的后两项可以写成一个高斯分布的噪声，它的均值是0，方差可以表示为:

$$
\alpha_t \beta_{t-1} + \beta_t\\
= \alpha_t(1-\alpha_{t-1}) + \beta_t\\
=1-\alpha_t\alpha_{t-1}\\
$$

所以式(5)可以看做 $x_t$ 从一个高斯分布采样过程重参数化的结果，其高斯分布为

$$
\mathbf x_{t} \sim \mathcal N(\sqrt{\alpha_t\alpha_{t-1}} \mathbf x_{t-2},\sqrt{1-\alpha_t \alpha_{t-1}}\mathcal I)
$$

我们将这个过程推广到 $\mathbf x_0$

$$
\mathbf x_{t} \sim \mathcal N(\sqrt{\bar \alpha_t \bar \alpha_{t-1} \cdots \bar \alpha_1}\mathbf x_0,\sqrt{1-\bar \alpha_t \bar \alpha_{t-1} \cdots \bar \alpha_1}\mathcal I)=\mathcal N(\sqrt{\bar \alpha_t}\mathbf x_0,(1-\bar \alpha_t)\mathcal I)
$$

逆重参数化得到

$$
\begin{equation}
\bf x_{t} = \sqrt{\bar \alpha_t}\bf x_0+\sqrt{1-\bar \alpha_t}\bf \epsilon, 其中\bf \epsilon\sim \mathcal N(0,\bf I)
\end{equation}
$$

### 前向过程的后验推导
我们的目标是推导前向过程的后验分布 $q(\mathbf x_{t-1}|\mathbf x_t)$ ，但是在实践上一般会用 $q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)$ 来近似，因为式(4)中的 $L_{T-1}$ 项计算的是 $q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)$ 和 $p_\theta(\mathbf x_{t-1}|\mathbf x_{t})$ 的DL散度；而且从实现上考虑，前向过程中 $\mathbf x_0$ 是已知的（一定要区分前向过程和后向过程的区别）。

$$
q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0) =\frac {q(\mathbf x_{t}|\mathbf x_{t-1},\mathbf x_0)q(\mathbf x_{t-1}|\mathbf x_0)}{q(\mathbf x_{t}|\mathbf x_0)} \\
= \frac {q(\mathbf x_{t}|\mathbf x_{t-1})q(\mathbf x_{t-1}|\mathbf x_0)}{q(\mathbf x_{t}|\mathbf x_0)}
$$

带入 $\mathbf x_t \sim \mathcal N(\sqrt{1-\beta_t}\mathbf x_{t-1},\beta_t\bf I)$ 以及我们前边推导的结论 $\mathbf x_{t} \sim \mathcal N(\sqrt{\bar \alpha_t}\mathbf x_0,(1-\bar \alpha_t)\bf I)$ 可以得到：

$$
\frac {q(\mathbf x_{t}|\mathbf x_{t-1})q(\mathbf x_{t-1}|\mathbf x_0)}{q(\mathbf x_{t}|\mathbf x_0)}= \frac {\frac {1}{\sqrt{2\pi\beta_t}}\exp\Big(-\frac {(\mathbf x_t-\sqrt{1-\beta_t}\mathbf x_{t-1})^2}{2\beta_t}\Big)\cdot \frac {1}{\sqrt{2\pi(1-\bar \alpha_{t-1})}}\exp\Big(-\frac {(\mathbf x_{t-1}-\sqrt{\bar \alpha_{t-1}}\mathbf x_0)^2}{2(1-\bar \alpha_{t-1})}\Big)}{\frac {1}{\sqrt{2\pi(1-\bar \alpha_t)}}\exp\Big(-\frac {(\mathbf x_t-\sqrt{\bar \alpha_t}\mathbf x_0)^2}{2(1-\bar \alpha_t)}\Big)}
$$

$$
=\sqrt {\frac {1-\bar \alpha_t}{2\pi(1-\bar \alpha_{t-1})\beta_t}}\exp-\frac 1 2\Big( \frac {(\mathbf x_t-\sqrt{1-\beta_t}\mathbf x_{t-1})^2}{\beta_t}+\frac {(\mathbf x_{t-1}-\sqrt{\bar \alpha_{t-1}}\mathbf x_0)^2}{1-\bar \alpha_{t-1}}-\frac {(\mathbf x_t-\sqrt{\bar \alpha_t}\mathbf x_0)^2}{1-\bar \alpha_t}\Big)
$$

注意到，我们的目的是求 $q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)$ 的分布，所以上式可以写成关于 $\mathbf x_{t-1}$ 高斯分布形式：

$$
\propto \exp\bigg[-\frac 1 2\Big[ (\frac {1-\beta_t}{\beta_t}\mathbf+\frac 1{1-\bar \alpha_{t-1}}) \mathbf x_{t-1}^2 -2(\frac {\sqrt{1-\beta_t}\mathbf x_t}{\beta_t}+\frac {\sqrt {\bar \alpha_{t-1}}\mathbf x_0}{1-\bar \alpha_{t-1}})\mathbf x_{t-1}+ C\Big]\bigg]
$$

$$
=\exp\bigg[ -\frac 1 2\Big[ \frac {\alpha_t-\alpha_t\bar \alpha_{t-1}+\beta_t}{\beta_t(1-\bar \alpha_t)}\mathbf x_{t-1}^2 -2\frac {\sqrt \alpha_t(1-\bar\alpha_{t-1})\mathbf x_t+\sqrt{\bar \alpha_{t-1}}\beta_t\mathbf x_0}{\beta_t(1-\bar\alpha_{t-1})} \mathbf x_{t-1}+ C\Big]\bigg]
$$

令平方项系数为A,线性项系数为B,常数项系数为C，则有

$$
A=\frac {\alpha_t-\alpha_t\bar \alpha_{t-1}+\beta_t}{\beta_t(1-\bar \alpha_{t-1})}
=\frac {1-\bar\alpha_{t}}{\beta_t(1-\bar\alpha_{t-1})}
$$

$$
B=-2 \frac {\sqrt \alpha_t(1-\bar\alpha_{t-1})\mathbf x_t+\sqrt{\bar \alpha_{t-1}}\beta_t\mathbf x_0}{\beta_t(1-\bar\alpha_{t-1})}
$$

上式可以写成

$$
\exp\bigg[ -\frac 1 2\Big( A\mathbf x_{t-1}^2 + B\mathbf x_{t-1}+C\Big)\bigg]\\
=\exp\bigg[ -\frac 1 2A\Big( \mathbf x_{t-1}+\frac {B}{2A}\Big)^2+C'\bigg]\\
= \exp \Big(-\frac {\Big( \mathbf x_{t-1}+\frac {B}{2A}\Big)^2}{2\frac {1}{A}}\Big).C''
$$

可以看到，具备高斯分布的性质，所以 $q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)$ 也符合高斯分布，且均值为 $\tilde\mu=-\frac {B}{2A}$，方差为 $\sigma^2=\frac {1}{A}$ ，带入A和B得

$$
\begin{equation}
\sigma_t^2 = \frac {1}{A} = \frac {\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_{t}},
\tilde\mu(\bf x_t,\bf x_0) = -\frac {B}{2A}=\frac {\sqrt \alpha_t(1-\bar\alpha_{t-1})\mathbf x_t+\sqrt{\bar \alpha_{t-1}}\beta_t\mathbf x_0}{1-\bar\alpha_{t}}
\end{equation}
$$

上边的(7)即到论文中的式(7)。 $\tilde \mu$ 是一个根据前向过程的定义式算出来的准确值，所以不带模型参数 $\theta$ ，而且前向过程中 $\bf x_0$ 是已知的，所以 $\tilde \mu$ 可以直接用 $\bf x_0$ 来计算。

注意到目前为止，所有的值都是准确的，因为前向过程是我们人为定义的，所有分布都有解析解，包括噪音 $\epsilon$ ，它服从的是标准高斯分布。


## 反向过程的设计（核心）
首先要明确，正向过程是我们定义的明确的一系列高斯分布，参数都是明确的；而反向过程是我们要用模型估计的，所以反向过程的分布带下标 $p_\theta(\mathbf x_{t-1}|\mathbf x_t)$ 。

根据式(4)，$L_T$ 项是 $q(\mathbf x_T|\mathbf x_0)$ 和 $p_\theta(\mathbf x_T)$ 的DL散度，我们已知 $\mathbf x_T$ 服从标准高斯分布，所以DL散度的距离理论上是0，应该忽略不计。下面重点看 $L_{t-1}$ 项。

这一项是 $q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)$ 和 $p_\theta(\mathbf x_{t-1}|\mathbf x_t)$ 的DL散度，根据我们上边的分析， $q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)$ 已经很明确了：

$$
q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)=\mathcal N(\tilde\mu(\mathbf x_t,\mathbf x_0),\sigma^2_t\bf I)
$$

$$
\sigma_t^2 = \frac {\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_{t}},
\tilde\mu(\bf x_t,\bf x_0) =\frac {\sqrt \alpha_t(1-\bar\alpha_{t-1})\mathbf x_t+\sqrt{\bar \alpha_{t-1}}\beta_t\mathbf x_0}{1-\bar\alpha_{t}}
$$

根据所以我们要设计一个前向过程模型 $p_\theta(\mathbf x_{t-1}|\mathbf x_t)$ ，它的目标是尽可能接近这个已知的高斯分布。那还不简单，直接把p也设成高斯分布，他的均值和方差都设成跟q一样不就好了吗？说干就干：

$$
p_\theta(\mathbf x_{t-1}|\mathbf x_t) = \mathcal N( \mathbf x_{t-1};\mu_\theta,\sigma_\theta^{2}\bf I)
$$

其中

$$
\sigma_\theta^2=\sigma^2_t
$$

$$
\mu_\theta=\tilde\mu(\mathbf x_t,\mathbf x_0)
$$

方差没啥问题，因为都是确定常数；但是均值这里问题来了： $p_\theta(\mathbf x_{t-1}|\mathbf x_t)$ 可没有一个given的 $\bf x_0$ ，所以我们不能直接用 $\tilde\mu$ 来计算 $\mu_\theta$ 。

为了解决这个问题，我们尝试在 $\tilde \mu$ 中去掉 $\bf x_0$ ：根据正向过程的推导式(6)可以反解出 $\mathbf x_0 = \frac {\mathbf x_t-\sqrt{1-\bar\alpha_t}\epsilon}{\sqrt{\bar \alpha_t}}$ ，带入(7)

$$
\tilde\mu(\mathbf x_t,\mathbf x_0) = \frac {\sqrt \alpha_t(1-\bar\alpha_{t-1})\mathbf x_t+\beta_t\frac{\mathbf x_t-\sqrt{1-\bar\alpha_t}\epsilon}{\sqrt{\alpha_t}}}{1-\bar\alpha_{t}}
$$

$$
=\frac {{\alpha_t(1-\bar\alpha_{t-1})+\beta_t}}{(1-\bar\alpha_{t})\sqrt{\alpha_t}}\mathbf x_t-\frac {\beta_t}{\sqrt{\alpha_t(1-\bar\alpha_t)}}\epsilon
$$

化简

$$
\begin{equation}
\tilde\mu(\mathbf x_t,t) = \frac {1}{\sqrt{\bar\alpha_t}}\mathbf (\mathbf x_t-\frac {\beta_t}{\sqrt{1-\bar\alpha_{t}}}\epsilon)
\end{equation}
$$

此式对应论文的式(10)的一部分。这里我们终于在前向过程均值中去掉了 $\bf x_0$ ，让它只依赖于 $\bf x_t$ 和 $\bf \epsilon$ （注意这里的 $\bf \epsilon$ 是前向过程中采样的噪音，是一个人为定义的准确值）。很完美了，直接用它来做反向过程的均值：

$$
\begin{equation}
\mu_\theta(\mathbf x_t,t) = \frac {1}{\sqrt{\bar\alpha_t}}\mathbf (\mathbf x_t-\frac {\beta_t}{\sqrt{1-\bar\alpha_{t}}}\epsilon_\theta(\mathbf x_t,t))
\end{equation}
$$

此式对应论文的式(11)。注意由于反向过程并不知道准确的噪声 $\epsilon$ ，所以只能根据模型来预测的噪声 $\epsilon_\theta$ ，而这就是我们模型的核心作用。

## 损失函数详细
有了以上的分析，我们终于可以来看看DDPM的损失函数是怎么算的了。根据(4)和之前的分析， $L_T=0$ ，我们只需要计算 $L_{t-1}, L_0$ 两项即可。

### $L_{t-1}$的计算
根据(4)有

$$
L_{t-1} = D_{KL}(q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)||p_\theta(\mathbf x_{t-1}|\mathbf x_t))
$$

首先 $q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)$ 在前边已经明确求出来了，此外在上一小节中，我们已经把 $p_\theta(\mathbf x_{t-1}|\mathbf x_t)$ 设计成了一个高斯分布，它的均值是 $\mu_\theta(\mathbf x_t,t)$ ，方差是 $\sigma^2_t$ 。所以 $L_{t-1}$ 项就可以写成：

$$
L_{t-1} = \int q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)\log \frac {q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)}{p_\theta(\mathbf x_{t-1}|\mathbf x_t)} d\mathbf x_{t-1}
$$

$$
= \int q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)\log \frac {\frac {1}{\sqrt{2\pi\sigma^2_t}}\exp\Big(-\frac {\|\mathbf x_{t-1}-\tilde\mu(\mathbf x_t,\mathbf x_0)\|^2}{2\sigma^2_t}\Big)}{\frac {1}{\sqrt{2\pi\sigma^2_t}}\exp\Big(-\frac {\|\mathbf x_{t-1}-\mu_\theta(\mathbf x_t,t)\|^2}{2\sigma^2_t}\Big)} d\mathbf x_{t-1}
$$

$$
=\int q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0) \frac 1 {2\sigma^2_t}\|\mu_\theta(\mathbf x_t,t)-\tilde\mu(\mathbf x_t,\mathbf x_0)\|^2 d\mathbf x_{t-1}
$$

因为积分是对 $\mathbf x_{t-1}$ 积分，所以可以把跟 $\mathbf x_t$ 无关的项 $\frac 1 {2\sigma^2_t}$ 提出来

$$
=\frac 1 {2\sigma^2_t}\|\mu_\theta(\mathbf x_t,t)-\tilde\mu(\mathbf x_t,\mathbf x_0)\|^2\int q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0) d\mathbf x_{t-1}
$$

积分部分等于求 $\mathbf x_{t-1}$ 的全概率，所以是1

$$
=\frac 1 {2\sigma^2_t}\|\mu_\theta(\mathbf x_t,t)-\tilde\mu(\mathbf x_t,\mathbf x_0)\|^2
$$

代入(8)，(9)得

$$
\begin{equation}
=\frac {\beta_t^2} {2\sigma^2_t\bar \alpha_t(1-\bar \alpha_t)}\|\epsilon-\epsilon_\theta(\mathbf x_t,t)\|^2
\end{equation}
$$

此式即对应论文的式(12)。说明结论：DDPM最终训练的目标是任意步骤的反向过程预测的噪声 $\epsilon_\theta(\mathbf x_t,t)$ 与真实噪声 $\epsilon$ 之间的距离尽可能小。

### $L_0$的处理
根据(4)

$$
L_0 = -\log p_\theta(\mathbf x_0|\mathbf x_1)
$$

> 为什么L0要单独拎出来？很多博客这里都没提到，要么说的是错的。真实的原因论文中有说明：从 $\mathbf x_1$ 到 $\mathbf x_0$ 的反向过程和其他的反向过程有一个根本性的区别，那就是这里有一个从连续到离散跳变：因为 $\mathbf x_0$ 是真实图片，它的每个元素是0~255的整数的像素值，是离散值；而 $\mathbf x_1$ 是从一个连续高斯分布里采样得到的，它的每个元素都是连续值。所以论文专门设计了一个独立离散解码器(independent discrete decoder)来解决这个问题。

discrete decoder算法为：首先把离散空间的x0值线性归一化到 $[-1,1]$ ，然后

$$
\begin{equation}
p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1) = \prod_{i=1}^D \int_{\delta_-(x_0^i)}^{\delta_+(x_0^i)} \mathcal{N}\left(x; \mu_\theta^i(\mathbf{x}_1, 1), \sigma_1^2\right) dx
\end{equation}
$$

$$
\delta_+(x) = 
\begin{cases} 
\infty & \text{if } x = 1 \\
x + \frac{1}{255} & \text{if } x < 1 
\end{cases}, \quad 
\delta_-(x) = 
\begin{cases} 
-\infty & \text{if } x = -1 \\
x - \frac{1}{255} & \text{if } x > -1 
\end{cases}
$$

理解一下：这个decoder实际上是把离散空间的x0值线性归一化到 $[-1,1]$ ，然后把每个点 $x_0^i$ 的概率等价为从连续概率密度 $p(\mathbf x_0^i|\mathbf x_1)$ 上的以 $x_0^i$ 为中心，宽度为 $\frac 2 {255}$ 的小区间，即 $(x_0^i-\frac 1 {255},x_0^i+\frac 1 {255})$ 的概率密度的积分（对于255/0对应的点1/-1，就是它到正无穷/负无穷的积分）。这就很好地解决了x0是离散值，但是概率密度又是个连续函数问题。

论文对于 $L_0$ 项的处理是最终将它近似到 $L_{t-1}$ 项中了。我们可以证明这个结论

$$
\begin{equation}
-\log p_\theta(\mathbf x_0|\mathbf x_1) \approx \frac {\beta_1^2} {2\sigma^2_1\bar \alpha_1(1-\bar \alpha_1)}\|\epsilon-\epsilon_\theta(\mathbf x_1,1)\|^2
\end{equation}
$$

证明：首先看公式右边，因为 $\beta_1$ 一般取很小值，所以 $\alpha_1 \approx 1$ ，且有 $\sigma_1^2=\beta_1, \bar\alpha_1=\alpha_1$

$$
式(12)右边=\frac {\beta_1^2} {2\beta_1 \alpha_1(1-\alpha_1)}\|\epsilon-\epsilon_\theta(\mathbf x_1,1)\|^2=\frac 1 2 \|\epsilon-\epsilon_\theta(\mathbf x_1,1)\|^2
$$

根据(9)有

$$
\mu_\theta(\mathbf x_1,1) = \frac {1}{\sqrt{\alpha_1}}\mathbf (\mathbf x_1-\frac {\beta_1}{\sqrt{\beta_1}}\epsilon_\theta(\mathbf x_1,1))
$$

根据(6) $\mathbf x_1=\sqrt{ \alpha_1}\mathbf x_0+\sqrt{\beta_1}\mathbf \epsilon$ 代入上式

$$
=\frac {1}{\sqrt{\alpha_1}}\mathbf (\sqrt{ \alpha_1}\mathbf x_0+\sqrt{\beta_1}\mathbf \epsilon-\sqrt{\beta_1}\epsilon_\theta(\mathbf x_1,1))
$$

$$
=\mathbf x_0+ \sqrt {\frac {\beta_1}{\alpha_1}}( \epsilon-\epsilon_\theta(\mathbf x_1,1))
$$

因为 $\alpha_1=1-\beta_1$ ，其中 $\beta_1$ 一般取很小值(按照T=1000， $\beta_1$ 取 $10^{-3}$ 左右)，所以 $\alpha_1 \approx 1$

$$
\begin{equation}
\mu_\theta(\mathbf x_1,1)\approx \mathbf x_0+\sqrt{\beta_1}( \epsilon-\epsilon_\theta(\mathbf x_1,1))
\end{equation}
$$

根据(11)中的积分项，我们可以做近似，把

$$
\log\int _{\delta_- (x_0^i)}^{\delta_+ (x_0^i)} \mathcal N(x; \mu_\theta^i(\mathbf{x}_1, 1), \sigma_1^2) dx \approx\log[\mathcal N(x_0^i;\mu_\theta^i(\mathbf{x}_1, 1), \sigma_1^2) \cdot \varDelta] = \log \mathcal N(x_0^i;\mu_\theta^i(\mathbf{x}_1, 1), \sigma_1^2) + \log \varDelta
$$

所以有

$$
\begin{equation}
式(12)左边=-\log p_\theta(\mathbf x_0|\mathbf x_1) \approx -\sum_{i=1}^D \log \mathcal N(x_0^i;\mu_\theta^i(\mathbf{x}_1, 1), \sigma_1^2) - \log \varDelta
\end{equation}
$$

先看求和内部项，将(13)以及 $\sigma_1^2=\beta_1$ 代入，得到

$$
\log \mathcal N(x_0^i;\mu_\theta^i(\mathbf{x}_1, 1), \beta_1) =-\frac {\|\mathbf x_0^i-\mu_\theta^i(\mathbf{x}_1, 1)\|^2}{2\beta_1}-\frac {1}{2}\log(2\pi\beta_1)
$$

$$
\approx-\frac {\|\epsilon^i-\epsilon_\theta^i(\mathbf x_1,1)\|^2}{2}-\frac {1}{2}\log(2\pi\beta_1)
$$

将上式代入(14)，得到

$$
式(12)左边 \approx \frac 1 2\sum_{i=1}^D \|\epsilon^i-\epsilon_\theta^i(\mathbf x_1,1)\|^2 + \sum_{i=1}^D {\frac 1 2\log(2\pi\beta_1)-\log \varDelta}
$$

由于上式第2项与 $\theta$ 无关，所以可以忽略不计。所以(12)左边可以近似为

$$
式(12)左边 \approx \frac 1 2\sum_{i=1}^D \|\epsilon^i-\epsilon_\theta^i(\mathbf x_1,1)\|^2
=\frac 1 2 \|\epsilon-\epsilon_\theta(\mathbf x_1,1)\|^2
$$

左边约等于右边，原命题得证。
### 思考和总结
论文给出的DDPM算法如下：

![DDPM](/resource/ddpm_algorithm.png)
我们提炼几个重点：
#### 训练
- 训练过程是一个随机迭代过程，对于每个输入的数据只会训练一个随机的步数，这个步数从均匀分布中采样。
- DDPM的模型核心是这个预测噪声的模型，它的输入是 $\mathbf x_t$ 和时间步 $t$，输出是噪声，这个模型就是上图中的 $\theta$（图中把 $\mathbf x_t$ 用推导式展开了）。这个预测噪声的模型理论上可以是任何模型，只要它的输入是 $\mathbf x_t$ 和时间步 $t$，能输出一个维度和 $\mathbf x_t$ 一致的噪声 $\epsilon_\theta$ 。不过在实践中一般用Unet或Transformer来做这个模型。
- 可以这样理解训练过程，先随机采样一个步数，再从标准高斯分布中采样一个噪声 $\epsilon$ ，然后根据步数算一个混合比例（ $\sqrt{1-\bar\alpha_t},\sqrt {\bar \alpha_t}$ ）将 $\epsilon$ 和原始图像 $\mathbf x_0$ 混合（得到 $\mathbf x_{t}$ ），将混合图像和步数作为参数输入模型，要求模型输出尽量趋近于混入的纯噪声 $\epsilon$ 。
    > 这个过程值得深入思考，如果一个模型能从任意一张图像按照任意比例和噪声混合的带噪信息中把纯噪声分离出来，那这个模型是不是其实也就学会了原图的分布？
#### 重构
- 重构过程是一个顺序迭代过程，步数从T一直迭代到1。注意这里其实每次迭代其实都是采样，从上一步预测的高斯分布中进行采样，但是由于运用了重参数化技巧，看起来像迭代（增加噪声）。
- 由于前向过程的后验分布 $q(\mathbf x_{t-1}|\mathbf x_t)$ 为高斯分布且 $x_T\sim\mathcal N(0,\mathcal I)$ ，那么从T步开始，每次从t步的高斯分布采样一个 $\mathbf x_t$ （重参数化）和时间步 $t$ 输入模型预测出噪声 $\epsilon_\theta(\mathbf x_t,t)$  ，计算出 $\mathbf x_{t-1}$ 的高斯均值和方差，作为下一步t-1采样的高斯分布参数。重复以上过程，直到步数变成1，此时根据 $\mathbf x_1$ 和 $\epsilon_\theta(\mathbf x_1,1)$ 计算出均值即可作为重构出的图像 $\mathbf x_0$ ，不需要计算方差了。
- 注意这里有两个噪声，一个是模型预估的噪声 $\epsilon_\theta(\mathbf x_t,t)$ ，一个每次重参数化采样加上去的噪声 $\sigma_t \mathbf z$ ，要理解这两个是完全不同的意义：
    - $\epsilon_\theta(\mathbf x_t,t)$ 是模型预测的当前步图像信息中的噪声，是需要从混合信息中（按比例）减掉的，是我们模型能重构原图的关键。
    - $\sigma_t \mathbf z$ 由于使用了重参数化采样而加上去的噪声，它是一个重构过程中从标准高斯随机中采样后按比例（标准差）加上的。理论上它的作用是增加每次迭代的随机性，避免模型过拟合。李宏毅曾试过去掉这个项，效果会变得很差。

