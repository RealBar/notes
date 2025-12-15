# 为什么Transformer要用LayerNorm？

作者：Cv大法代码酱
链接：https://www.zhihu.com/question/487766088/answer/1976682564862355377

## 第一、 为什么必须要有Norm？真的是为了解决ICS吗？
首先解决第一个问题，为什么神经网络需要Normalization？你要是去面试，十个面试官有九个会问你什么是ICS（Internal Covariate Shift，内部协变量偏移）。你要是照着Google那篇BN的论文背，说是因为每一层的输入分布都在变，导致后面层要不停适应前面层的变化，所以我们要强行把分布拉回到均值0方差1……这回答在2018年能拿满分，但在2025年，我只能给你打个及格。

这玩意儿在学术界早就被扒皮了。麻省理工的好几篇论文都证明了，哪怕你把ICS加剧，只要有Norm，模型依然收敛得很好。Norm真正的作用，是在平滑优化的地形。想象一下，你站在神农架的深山老林里（这是高维参数空间），你要下山（找Loss最低点）。如果没有Norm，这山路极其崎岖，一会儿是悬崖，一会儿是深坑，你的步子（Learning Rate）稍微大一点，直接摔死（梯度爆炸），或者卡在坑里出不来（梯度消失）。

Norm做的事情，就是把这地形给你推平了，把悬崖削平，把坑填浅。它限制了权重矩阵的Lipschitz常数，说人话就是：它限制了模型输出对输入的敏感度。你输入变一点点，输出不会剧烈波动。这就让梯度变得很乖，你可以放心大胆地开大你的Learning Rate。对于Transformer这种层数极深（现在的LLM动不动就80层、100层）的模型，没有Norm，信号传不到两层就挂了。这里推荐一篇我觉得被严重低估的论文，是2018年NeurIPS上的 《How Does Batch Normalization Help Optimization?》。这文章虽然是讲BN的，但原理通用于所有Norm。它非常直观地展示了有无Norm时损失平面的平滑程度差异。建议读一下，尤其是做大模型预训练的兄弟，这能帮你理解为什么有时候Loss飞了是Norm没设好。

## 第二、 为什么BN在Transformer里水土不服？
好，既然要Norm，那BN（Batch Normalization）可是CV（计算机视觉）界的扛把子，ResNet靠它打下了江山，为什么到了NLP，到了Transformer这里，就被LN干趴下了？这里面有几个非常硬核的原因，跟数据形态、硬件限制和序列特性都有关。
1. 变长序列的噩梦这是最直观的原因。搞CV的兄弟幸福啊，图片进来，全是224x224，整整齐齐。搞NLP的呢？这句子一会儿长一会儿短。BN是干嘛？它是在Batch维度上算均值和方差。假设你的Batch Size是32，你就要算这32个样本在同一个特征通道上的统计量。现在问题来了，句子不一样长，你得Padding（补零）吧？比如一个Batch里，最长的句子有100个词，最短的只有5个词。剩下的95个位置全是0。当你做BN的时候，这些0全被算进去了！这会导致计算出来的均值和方差被严重稀释，根本代表不了真实数据的分布。这就好比你算全班平均分，结果混进去一半的0分考生，这平均分还能看吗？虽然你可以通过Mask来某种程度上解决计算问题，但在工程实现上极其恶心，而且效果并不好。
2. Batch Size的硬伤这在2025年的今天尤为明显。现在的模型太大了。回想2017年，Transformer刚出来，大家还在用几G显存的卡，跑个Base模型。现在呢？Llama-4（假设它叫这个）几千亿参数，即便你是H800或者B200集群，单卡能塞进去的Batch Size也非常小，甚至可能是1（做Micro-batch的时候）。BN严重依赖Batch Size。如果Batch Size太小，样本的统计量就代表不了总体的统计量，噪声极大。Batch Size为1的时候，BN直接就废了（除非你用Running Statistics，但训练时也不对劲）。而LN呢？LayerNorm是独立于Batch Size的。不管你Batch是1还是1000，LN只关心当前这个样本，它是在Feature维度（Hidden Dimension）上做归一化。我是第一句话，我就只算我这句话里词向量的均值方差，我不看隔壁那句话。这对于大模型训练时的稳定性至关重要。
3. 序列数据的时序依赖性（虽然Transformer是并行的）RNN时代大家就发现BN不行，因为时间步t的统计量和t+1的不一样。Transformer虽然是一次性并行输入的，但每个Token在不同位置的语义强度是不一样的。BN强行把所有Token在同一个特征维度上拉齐，这其实破坏了Token之间的一些微妙差异。而LN是在每个Token内部，把它的各个特征维度拉齐，这保留了Token之间的相对大小关系（虽然最后也会有缩放，但它是逐Token独立操作的）。这里穿插一个资源。如果你想深入理解BN和LN在RNN/Transformer里的区别，去看看Google Brain 2016年的那篇 《Layer Normalization》 原作。Geoffrey Hinton老头子挂名的。虽然有点老，但它里面对RNN无法使用BN的分析，完美契合Transformer的语境。

## 为什么LN能赢？
那么，LayerNorm到底在干什么？我们来看Transformer里的一个Token，比如它的Embedding维度是4096（像Llama 2 70B这种级别）。LN是对这4096个数值求均值和方差，然后归一化。这意味着什么？这意味着LN把这个Token的幅度（Magnitude）给抹平了，只保留了方向（Direction）。这非常关键！在高维空间里，向量的模长（长度）往往代表了这个特征的“置信度”或者“能量”，而向量的方向代表了“语义”。但在深层网络里，模长很容易失控。有的Token模长特别大，有的特别小。如果直接拿去算Attention（点积），模长大的Token会主导整个Attention Score，导致Softmax之后变成One-hot，梯度消失，模型这就学不动了。LN强制把所有Token的能量拉到同一个水平线上。大家都别争，能量一样，这时候我们再来比方向（语义）。谁跟Query的方向一致，谁的Attention就高。这才是LN在Transformer里真正的物理意义：Gain Control（增益控制）。这让我想起我在2020年调一个大规模推荐模型的时候，当时用了Transformer做序列特征提取。刚开始我不信邪，非要试BN，结果Loss震荡得像过山车。后来老老实实换回LN，曲线顺滑得像德芙巧克力。实战经验告诉我，在NLP和序列任务里，Token内部的特征一致性（LN捕捉的东西），远比跨样本的特征一致性（BN捕捉的东西）重要。
## LN的变体与RMSNorm的称霸

时间走到今天，标准的LayerNorm其实也发生了一些变化。现在的所谓Transformer，绝大多数用的都不是原始论文里的LN，而是RMSNorm（Root Mean Square Layer Normalization）。从Llama一代开始，到现在的各种开源大模型，清一色RMSNorm。为什么？标准LN公式是：x−μσ+ϵ⋅γ+βσ+ϵx−μ​⋅γ+β要算均值 μμ，要算方差 σσ，要减均值。RMSNorm公式是：xRMS(x)⋅γRMS(x)x​⋅γ它不去中心化（不减均值），直接除以均方根。你要知道，在大模型训练里，几千张卡跑几个月，电费都是几百万几千万。哪怕一个算子能快10%，那都是真金白银。**RMSNorm少了一个减均值的操作**，计算量省了一丢丢，但在万亿次计算的放大下，这点收益很可观。而且实验证明，去不去均值对模型效果影响微乎其微，重要的是缩放（Scaling）。所以，如果你现在自己手写Transformer，千万别傻乎乎用 torch.nn.LayerNorm 了，去找个优化好的 RMSNorm kernel，或者直接用 flash_attn 库里带的那些高性能算子。这里必须推荐一个资源：FlashAttention 的 GitHub 仓库。哪怕你不写底层CUDA，你去读它的源码，看看它是怎么把Norm和Attention融合在一起优化的，这对你理解现代大模型的底层算力分配极其有帮助。现在的算法工程师，不懂算子优化，路走不远。

## Pre-Norm vs Post-Norm 是一场血的教训
讲Transformer的Norm，绝对绕不开Pre-Norm和Post-Norm之争。这也是面试必考题。原始的《Attention Is All You Need》论文，用的是Post-Norm。也就是：LayerNorm(x + Sublayer(x))。先做Attention/FFN，残差连接，然后再Norm。后来的GPT-2，GPT-3，Llama，全改成了Pre-Norm。也就是：x + Sublayer(LayerNorm(x))。先Norm，再进Sublayer，最后残差。
为什么？
Post-Norm在理论上效果可能更好（因为Norm在最后，把输出再次规整了），但是它太难训练了。在深层网络中，Post-Norm会导致梯度在反向传播时这就不仅是消失的问题，而是极度不稳定。早期的BERT训练需要非常小心地设计Warmup（热身）策略，学习率要先很小很小，慢慢爬坡，否则模型直接崩。而Pre-Norm把Norm放在了残差分支的里面。主干通路（Highway）是纯净的 x + ...。这就好比你修了一条直通的高速公路，梯度可以毫无阻碍地从最后一层直接流到第一层。
这使得Pre-Norm的模型极其稳定，哪怕你不需要复杂的Warmup，学习率给猛一点，它也能收敛。
在现在动辄百亿参数的年代，训练稳定性大于一切。模型跑崩一次的代价太大了（Checkpoint回滚几个小时就是几万块钱没了）。所以Pre-Norm（配合RMSNorm）成为了工业界的标准配置。
## 为什么不用Instance Norm、Group Norm？
有人会问，那Instance Norm（IN）和Group Norm（GN）呢？IN主要用在风格迁移（Style Transfer）里。因为它把图像的内容和风格剥离了。但在NLP里，我们需要Token的全部信息，不需要剥离什么风格。GN是BN和LN的折中。说实话，在Vision Transformer（ViT）里，GN有时候确实比LN好一点点，或者差不多。但在纯NLP的Transformer里，GN引入了额外的超参数（Group的数量），调参太麻烦。
工程界的铁律：如无必要，勿增实体。LN（特别是RMSNorm）不需要任何超参数（除了那个极小的epsilon），简单粗暴，效果好，硬件亲和力强。为什么要换？数据科学干久了你会发现，很多所谓的创新，最后都会回归到最本质的数学性质上。
Transformer之所以强，不是因为它有什么魔法，而是因为它通过Residual Connection和LayerNorm，构造了一个信息高速公路。LayerNorm在这里扮演的角色，就是一个交通管制员。它不管你这辆车（Token）是从哪来的（Batch无关），也不管你这辆车要去哪，它只管把你这辆车的超载部分卸下来（归一化），确保你在高速公路上跑的时候，不会因为太重把路压坏（梯度爆炸），也不会因为太轻飘起来（梯度消失）。
我看过太多新人，拿着大模型微调，遇到Loss不降，第一反应是去改模型结构，去加各种花哨的Attention变体。别折腾那些。先看看你的Norm层对不对。我就遇到过一个案子，有个小兄弟把Pre-Norm写成了Post-Norm，然后跑FP16（半精度）训练。结果Loss全是NaN。因为FP16的数值范围很小，Post-Norm中间很容易溢出。改成Pre-Norm或者RMSNorm，问题立马解决。这就是经验。这些经验书上不会写，论文里不会提，只有你在满屏报错的终端前熬过夜才能懂。
洋洋洒洒说了这么多，给现在的从业者几点实在的建议：认准RMSNorm + Pre-Norm组合。如果你在做Transformer相关的新模型设计，或者魔改LLM，这是目前的版本答案。别回头去试Post-Norm BN了，那是2017年的老皇历。关注数值稳定性。LayerNorm不只是为了收敛，更是为了在低精度训练（FP16, BF16，甚至现在的FP8）时不溢出。理解这一点，你调试模型时会有新视角。读懂硬件。理解为什么LN适合GPU并行，为什么BN依赖Batch Size导致显存利用率和通信开销问题。未来的算法专家，一定是懂系统的。最后，推荐一本不那么技术的书，《The Bitter Lesson》（苦涩的教训），Rich Sutton写的博客文章。虽然短，但道出了AI发展的真谛：长远来看，利用算力的通用方法（比如Transformer + Norm + 巨量数据）总是会战胜利用人类先验知识的方法。LayerNorm就是这种通用方法的基石之一。它不假设你的数据是图片还是文本，不假设你的Batch大还是小，它只是简单、鲁棒地处理好每一个向量。这就是为什么Transformer要用LayerNorm。简单，也是一种力量。