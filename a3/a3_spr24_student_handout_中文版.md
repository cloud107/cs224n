# CS224n Spring 2024 — Assignment 3(作业 3)

**截止日期:2024 年 4 月 30 日(周二)下午 4:30 PST。**

本作业分为两部分:**《基于 RNN 的神经机器翻译》** 和 **《分析 NMT 系统》**。第一部分以编码实现为主,第二部分全部是书面分析题。如果你在第一部分卡住了,可以先做第二部分,两者相互独立。注意,这个 NMT 系统比本课程之前构建的神经网络都要复杂,在 GPU 上训练大约需要 **2 小时**。因此我们强烈建议你尽早开始本作业。最后,这个 NMT 系统的符号和实现有点 tricky,如果你在过程中卡住了,欢迎来 Office Hours 让助教帮你。

---

## 第 1 题:基于 RNN 的神经机器翻译(45 分)

在机器翻译中,我们的目标是把句子从*源*语言(如中文)转换到*目标*语言(如英文)。在本作业中,我们将实现一个带 attention 的序列到序列(Seq2Seq)网络,构建一个神经机器翻译(NMT)系统。本节我们描述所提出的 NMT 系统的**训练过程**,该系统使用**双向 LSTM 编码器**和**单向 LSTM 解码器**。

> 图:带乘性注意力(Multiplicative Attention)的 Seq2Seq 模型,图中展示了解码器的第 3 步。编码器的隐状态 $h_i^{enc}$ 和细胞状态 $c_i^{enc}$ 在下一页定义。(见 `a3_spr24_student_latex/images/Assignment 4 Figure.png`)

### 模型描述(训练过程)

给定一句源语言句子,我们从**嵌入矩阵**中查找字符或词的嵌入,得到 $\mathbf{x}_1, \dots, \mathbf{x}_m$($\mathbf{x}_i \in \mathbb{R}^{e \times 1}$,其中 $m$ 是源句长度,$e$ 是嵌入维度)。然后把这些嵌入送入一个**卷积层**(如果对卷积层不熟悉,可查看 <https://cs231n.github.io/convolutional-networks> 的详细介绍),保持形状不变。卷积层输出送入**双向编码器**,得到前向($\rightarrow$)和后向($\leftarrow$)两个 LSTM 的隐状态与细胞状态。前向和后向拼接,得到隐状态 $\mathbf{h}_i^{enc}$ 和细胞状态 $\mathbf{c}_i^{enc}$:

$$
\mathbf{h}_i^{enc} = [\mathbf{h}_i^{enc,bw}; \mathbf{h}_i^{enc,fw}] \quad \text{其中 } \mathbf{h}_i^{enc} \in \mathbb{R}^{2h \times 1},\ \mathbf{h}_i^{enc,bw}, \mathbf{h}_i^{enc,fw} \in \mathbb{R}^{h \times 1} \quad 1 \le i \le m
$$

$$
\mathbf{c}_i^{enc} = [\mathbf{c}_i^{enc,bw}; \mathbf{c}_i^{enc,fw}] \quad \text{其中 } \mathbf{c}_i^{enc} \in \mathbb{R}^{2h \times 1},\ \mathbf{c}_i^{enc,bw}, \mathbf{c}_i^{enc,fw} \in \mathbb{R}^{h \times 1} \quad 1 \le i \le m
$$

然后我们用编码器**最终隐状态和最终细胞状态**的线性投影来初始化**解码器**的第一个隐状态 $\mathbf{h}_0^{dec}$ 和细胞状态 $\mathbf{c}_0^{dec}$。

> 注脚:如果不明显,想想为什么我们把 $[\mathbf{h}_1^{enc,bw}, \mathbf{h}_m^{enc,fw}]$ 视为编码器的"最终隐状态"。

$$
\mathbf{h}_0^{dec} = \mathbf{W}_h[\mathbf{h}_1^{enc,bw}; \mathbf{h}_m^{enc,fw}] \quad \text{其中 } \mathbf{h}_0^{dec} \in \mathbb{R}^{h \times 1},\ \mathbf{W}_h \in \mathbb{R}^{h \times 2h}
$$

$$
\mathbf{c}_0^{dec} = \mathbf{W}_c[\mathbf{c}_1^{enc,bw}; \mathbf{c}_m^{enc,fw}] \quad \text{其中 } \mathbf{c}_0^{dec} \in \mathbb{R}^{h \times 1},\ \mathbf{W}_c \in \mathbb{R}^{h \times 2h}
$$

解码器初始化完成后,需要喂入目标句子。在第 $t$ 步,查找第 $t$ 个子词的嵌入 $\mathbf{y}_t \in \mathbb{R}^{e \times 1}$。把它与上一步的 *combined-output 向量* $\mathbf{o}_{t-1} \in \mathbb{R}^{h \times 1}$(本页后面会解释!)拼接,得到 $\overline{\mathbf{y}_t} \in \mathbb{R}^{(e+h) \times 1}$。注意,对于第一个目标子词(即起始 token),$\mathbf{o}_0$ 是零向量。然后把 $\overline{\mathbf{y}_t}$ 作为输入送入解码器:

$$
\mathbf{h}_t^{dec}, \mathbf{c}_t^{dec} = \text{Decoder}(\overline{\mathbf{y}_t}, \mathbf{h}_{t-1}^{dec}, \mathbf{c}_{t-1}^{dec}) \quad \text{其中 } \mathbf{h}_t^{dec} \in \mathbb{R}^{h \times 1},\ \mathbf{c}_t^{dec} \in \mathbb{R}^{h \times 1}
$$

然后用 $\mathbf{h}_t^{dec}$ 对 $\mathbf{h}_1^{enc}, \dots, \mathbf{h}_m^{enc}$ 计算**乘性注意力**:

$$
\mathbf{e}_{t,i} = (\mathbf{h}_t^{dec})^T \mathbf{W}_{\text{attProj}} \mathbf{h}_i^{enc} \quad \text{其中 } \mathbf{e}_t \in \mathbb{R}^{m \times 1},\ \mathbf{W}_{\text{attProj}} \in \mathbb{R}^{h \times 2h} \quad 1 \le i \le m
$$

$$
\boldsymbol{\alpha}_t = \text{softmax}(\mathbf{e}_t) \quad \text{其中 } \boldsymbol{\alpha}_t \in \mathbb{R}^{m \times 1}
$$

$$
\mathbf{a}_t = \sum_{i=1}^{m} \alpha_{t,i} \mathbf{h}_i^{enc} \quad \text{其中 } \mathbf{a}_t \in \mathbb{R}^{2h \times 1}
$$

其中 $\mathbf{e}_{t,i}$ 是 $\mathbf{e}_t \in \mathbb{R}^{m \times 1}$ 的第 $i$ 个元素(标量),由第 $t$ 步的解码器隐状态 $\mathbf{h}_t^{dec} \in \mathbb{R}^{h \times 1}$、注意力投影 $\mathbf{W}_{\text{attProj}} \in \mathbb{R}^{h \times 2h}$ 和编码器第 $i$ 步隐状态 $\mathbf{h}_i^{enc} \in \mathbb{R}^{2h \times 1}$ 计算得到。

现在把注意力输出 $\mathbf{a}_t$ 与解码器隐状态 $\mathbf{h}_t^{dec}$ 拼接,经过一个线性层、tanh 和 dropout,得到 *combined-output* 向量 $\mathbf{o}_t$:

$$
\mathbf{u}_t = [\mathbf{a}_t; \mathbf{h}_t^{dec}] \quad \text{其中 } \mathbf{u}_t \in \mathbb{R}^{3h \times 1}
$$

$$
\mathbf{v}_t = \mathbf{W}_u \mathbf{u}_t \quad \text{其中 } \mathbf{v}_t \in \mathbb{R}^{h \times 1},\ \mathbf{W}_u \in \mathbb{R}^{h \times 3h}
$$

$$
\mathbf{o}_t = \text{dropout}(\text{tanh}(\mathbf{v}_t)) \quad \text{其中 } \mathbf{o}_t \in \mathbb{R}^{h \times 1}
$$

然后在第 $t$ 个时间步,产生目标子词上的概率分布 $\mathbf{P}_t$:

$$
\mathbf{P}_t = \text{softmax}(\mathbf{W}_{\text{vocab}} \mathbf{o}_t) \quad \text{其中 } \mathbf{P}_t \in \mathbb{R}^{V_t \times 1},\ \mathbf{W}_{\text{vocab}} \in \mathbb{R}^{V_t \times h}
$$

这里 $V_t$ 是目标词汇表的大小。最后,为了训练网络,我们计算 $\mathbf{P}_t$ 与 $\mathbf{g}_t$(第 $t$ 步目标子词的 one-hot 向量)之间的交叉熵损失:

$$
J_t(\theta) = \mathrm{CrossEntropy}(\mathbf{P}_t, \mathbf{g}_t)
$$

其中 $\theta$ 表示模型的所有参数,$J_t(\theta)$ 是解码器第 $t$ 步的损失。

现在模型已经描述完毕,让我们实现它,做中译英吧!

### 搭建本地开发环境

确保已安装 [conda](https://docs.anaconda.com/free/miniconda/index.html)。

解压 starter code,用你喜欢的 Python IDE(VSCode 是热门选择)打开解压后的目录。打开终端,`cd` 到包含 `env-cpu.yml` 和 `env-gpu.yml` 的目录。创建并激活 `cs224n-cpu` conda 环境:

```bash
conda env create --file env-cpu.yml
conda activate cs224n-cpu
```

### 搭建云端 GPU 虚拟机

按照 [CS224n GCP 指南](https://docs.google.com/document/d/1FLx0CXIn-SoExxKM1efC-E-6iBjUR4uEnpGnfemMMR0/edit?pli=1#heading=h.4tqnggp12z76)(课程网站和 Ed 上也有链接)创建你的 VM 实例。大约需要 45 分钟。虽然训练模型需要 GPU,但我们强烈建议你**先在本地开发代码并确保能跑通,再放到 VM 上训练**。GPU 时间既贵又有限,训练 NMT 系统大约需要 **1.5 到 2 小时**。我们不希望你把 GPU 时间全花在调试模型上,而不是训练和评估。最后,**不用 VM 时一定把它关掉。**

如果你的 GCP 订阅余额耗尽,VM 会被临时锁定而无法访问。如果发生这种情况,请发一个私密的 Ed 帖描述情况。

配置好云端 VM 后,在本地做编程作业期间应把它关掉。

### 实现题和书面题

**(1a)[2 分,编码]** 为了应用张量运算,我们必须保证一个 batch 内所有句子等长。因此要找出 batch 中最长的句子,把其他句子补齐到同样长度。在 `utils.py` 中实现 `pad_sents` 函数。

**(1b)[3 分,编码]** 在 `model_embeddings.py` 中实现 `__init__` 函数,初始化必要的源语言和目标语言嵌入。

**(1c)[4 分,编码]** 在 `nmt_model.py` 中实现 `__init__` 函数,初始化 NMT 系统所需的模型层(LSTM、CNN、投影层和 dropout)。

**(1d)[8 分,编码]** 在 `nmt_model.py` 中实现 `encode` 函数。该函数把填充后的源句子转换为张量 $\mathbf{X}$,生成 $\mathbf{h}_1^{enc}, \dots, \mathbf{h}_m^{enc}$,并计算解码器的初始状态 $\mathbf{h}_0^{dec}$ 和初始细胞状态 $\mathbf{c}_0^{dec}$。你可以运行以下命令做一次非全面的 sanity check:

```bash
python sanity_check.py 1d
```

**(1e)[8 分,编码]** 在 `nmt_model.py` 中实现 `decode` 函数。该函数构造 $\bar{\mathbf{y}}$ 并对输入逐时间步运行 `step` 函数。运行以下命令做 sanity check:

```bash
python sanity_check.py 1e
```

**(1f)[10 分,编码]** 在 `nmt_model.py` 中实现 `step` 函数。该函数在单个时间步内应用解码器的 LSTM cell,计算目标子词的编码 $\mathbf{h}_t^{dec}$、注意力分数 $\mathbf{e}_t$、注意力分布 $\boldsymbol{\alpha}_t$、注意力输出 $\mathbf{a}_t$,最后是 combined output $\mathbf{o}_t$。运行以下命令做 sanity check:

```bash
python sanity_check.py 1f
```

**(1g)[3 分,书面]** `nmt_model.py` 中的 `generate_sent_masks()` 函数生成一个叫 `enc_masks` 的张量。它的形状是 (batch size, 最大源句长度),输入中 `pad` token 的位置上是 1,非 pad 位置是 0。看看在 `step()` 函数的注意力计算中(第 295–296 行)是怎么用这些 mask 的。

首先解释(大约三句话)这些 mask 对整个注意力计算有什么影响。然后解释(一两句话)为什么必须以这种方式使用 mask。

---

现在该让系统跑起来了!如前所述,我们建议你在个人电脑上开发代码。确认在正确的 conda 环境中,然后执行以下命令在本地机器上训练模型:

```bash
sh run.sh train_local
(Windows) run.bat train_local
```

为了帮助监控和调试,starter code 使用 tensorboard 在训练期间记录 loss 和 perplexity。要打开 TensorBoard 监控训练过程,在**另一个**也激活了 `cs224n-cpu` 环境的终端里运行以下命令,然后在 <http://localhost:6006/> 访问:

```bash
tensorboard --logdir=runs --port 6006
```

> 图:Tensorboard 显示本地机器上的 loss 和 perplexity 值(见 `a3_spr24_student_latex/images/tensorboard_cpu.jpg`)

初始若干 iteration 你应该能看到 loss 显著下降(见上图)。一旦你的代码跑了几百个 iteration 不崩溃,就打开 GCP Console 启动你的 VM。

**在 VM 上训练 NMT 系统:** 参考 GCP How-to 附录:
- 设置 GCP VM 并获得 SSH 连接信息
- 连接 VM(带 SSH 隧道设置,以便远程查看 tensorboard 日志)
- 把代码从你的电脑复制到云端 VM
- 在云端 VM 上训练 NMT 系统
- 从云端 VM 下载 Gradescope 提交包

---

**(1h)[3 分,书面]** 模型训练完成后(**在 VM 上应该 2 小时以内**),执行以下命令测试模型:

```bash
sh run.sh test
```

请报告模型的 corpus BLEU 分数。应该大于 18。

**(1i)[4 分,书面]** 课堂上我们学过点积注意力(dot product attention)、乘性注意力(multiplicative attention)和加性注意力(additive attention)。回忆一下:点积注意力是 $\mathbf{e}_{t,i} = \mathbf{s}_t^T \mathbf{h}_i$,乘性注意力是 $\mathbf{e}_{t,i} = \mathbf{s}_t^T \mathbf{W} \mathbf{h}_i$,加性注意力是 $\mathbf{e}_{t,i} = \mathbf{v}^T \text{tanh}(\mathbf{W}_1 \mathbf{h}_i + \mathbf{W}_2 \mathbf{s}_t)$。

- **(i) [2 分]** 与乘性注意力相比,点积注意力有什么优点和缺点?各说一个。
- **(ii) [2 分]** 与乘性注意力相比,加性注意力有什么优点和缺点?各说一个。

---

## 第 2 题:分析 NMT 系统(25 分)

**(2.1)[3 分]** 看看 `src.vocab` 文件里源语言词汇表中的一些短语和词的例子。把输入的中文序列编码为词汇表中的 "pieces" 时,tokenizer 把序列映射为一系列词汇项,每个词汇项由一个或多个字符组成(得益于 `sentencepiece` tokenizer,即使原文没有空格,我们也能做这种切分)。基于这些信息,在嵌入层之后、送入双向编码器之前加一个 1D 卷积层,对我们的 NMT 系统有什么帮助?

**提示:** 每个中文字符要么是一个完整的词,要么是词中的一个语素。分别查一下 电、脑、电脑 的意思:字符"电"(electricity)和"脑"(brain)组合成"电脑"意思是 computer。

**(2.2)[8 分]** 下面是我们 NMT 模型(和你训练的相同)输出中的一系列错误。对每个参考(gold)英文翻译和 NMT(模型)英文翻译的例子,请:

1. 指出 NMT 翻译中的错误。
2. 给出模型可能犯错的原因(可能是特定的语言结构,也可能是特定的模型局限)。
3. 描述一种我们可能修改 NMT 系统来修复该错误的方法。一个错误的修复方法不止一种。比如,可以调整隐藏层大小,或者改变注意力机制。

只需分析每个句子里**加下划线**的错误。放心,你不需要懂中文也能回答这些问题,只需要懂英文!如果你想对源句子有更多了解,可以自由使用 <https://www.archchinese.com/chinese_english_dictionary.html> 之类的资源查词。也可以搜训练数据文件,更好地了解某些字符出现的频率。

- **(a) [2 分]**
  - **源句子:** 贼人其后被警方拘捕及被判处盗窃罪名成立。
  - **参考翻译:** *<u>the culprits were</u> subsequently arrested and convicted.*
  - **NMT 翻译:** *<u>the culprit was</u> subsequently arrested and sentenced to theft.*

- **(b) [2 分]**
  - **源句子:** 几乎已经没有地方容纳这些人,资源已经用尽。
  - **参考翻译:** *there is almost no space to accommodate these people, and resources have run out.*
  - **NMT 翻译:** *the resources have been exhausted and <u>resources have been exhausted</u>.*

- **(c) [2 分]**
  - **源句子:** 当局已经宣布今天是国殇日。
  - **参考翻译:** *authorities have announced <u>a national mourning today</u>.*
  - **NMT 翻译:** *the administration has announced <u>today's day</u>.*

- **(d) [2 分]**
  - **源句子**(注:这是粤语句子!本作业的数据来自 GALE Phase 3,这是从网上各种来源抓取的中文简体新闻及其翻译的汇编。更多细节见 <https://catalog.ldc.upenn.edu/LDC2017T02>):俗语有云:"唔做唔错"。
  - **参考翻译:** *<u>"act not, err not"</u>, so a saying goes.*
  - **NMT 翻译:** *as the saying goes, <u>"it's not wrong."</u>*

**(2.3)[14 分]** BLEU 分数是 NMT 系统最常用的自动评估指标。通常对整个测试集计算,但这里我们考虑单句的 BLEU。

> 注:这个单句 BLEU 的定义与 Python 包 `nltk` 中的 `sentence_bleu()` 函数一致。注意 NLTK 函数对大小写敏感。本题中所有文本都是小写,所以大小写无关紧要。<http://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.sentence_bleu>

假设有源句子 $\mathbf{s}$、一组 $k$ 个参考翻译 $\mathbf{r}_1, \dots, \mathbf{r}_k$、以及一个候选翻译 $\mathbf{c}$。计算 $\mathbf{c}$ 的 BLEU 分数时,先对 $n = 1, 2, 3, 4$ 分别计算 $\mathbf{c}$ 的 *modified $n$-gram precision* $p_n$($n$ 是 [n-gram](https://en.wikipedia.org/wiki/N-gram) 中的 $n$):

$$
p_n = \frac{\displaystyle \sum_{\text{ngram} \in \mathbf{c}} \min \left( \max_{i=1,\dots,k} \text{Count}_{\mathbf{r}_i}(\text{ngram}),\ \text{Count}_{\mathbf{c}}(\text{ngram}) \right)}{\displaystyle \sum_{\text{ngram} \in \mathbf{c}} \text{Count}_{\mathbf{c}}(\text{ngram})}
$$

对候选翻译 $\mathbf{c}$ 中出现的每个 $n$-gram,统计它在**任意一个**参考翻译中出现的最大次数,再以它在 $\mathbf{c}$ 中出现的次数为上限(取两者较小值,这就是分子)。再除以 $\mathbf{c}$ 中 $n$-gram 的总数(分母)。

接下来计算 *brevity penalty* BP。令 $len(c)$ 为 $\mathbf{c}$ 的长度,$len(r)$ 为最接近 $len(c)$ 的参考翻译的长度(如果两个参考翻译长度一样接近,选择较短的那个):

$$
BP = \begin{cases} 1 & \text{if } len(c) \ge len(r) \\ \exp\left(1 - \frac{len(r)}{len(c)}\right) & \text{otherwise} \end{cases}
$$

最后,候选翻译 $\mathbf{c}$ 关于 $\mathbf{r}_1, \dots, \mathbf{r}_k$ 的 BLEU 分数是:

$$
BLEU = BP \times \exp\left(\sum_{n=1}^{4} \lambda_n \log p_n\right)
$$

其中 $\lambda_1, \lambda_2, \lambda_3, \lambda_4$ 是权重,和为 1。这里的 $\log$ 是自然对数。

- **(a) [5 分]** 考虑这个例子:
  - 源句子 $\mathbf{s}$:**需要有充足和可预测的资源。**
  - 参考翻译 $\mathbf{r}_1$:*resources have to be sufficient and they have to be predictable*
  - 参考翻译 $\mathbf{r}_2$:*adequate and predictable resources are required*
  - NMT 翻译 $\mathbf{c}_1$:there is a need for adequate and predictable resources
  - NMT 翻译 $\mathbf{c}_2$:resources be sufficient and predictable to

  请计算 $\mathbf{c}_1$ 和 $\mathbf{c}_2$ 的 BLEU 分数。令 $\lambda_1 = \lambda_2 = 0.5$,$\lambda_3 = \lambda_4 = 0$(**这意味着忽略 3-gram 和 4-gram**,即不计算 $p_3$ 和 $p_4$)。计算 BLEU 时请写出计算过程(即写出你算出的 $p_1$、$p_2$、$len(c)$、$len(r)$ 和 $BP$)。注意 BLEU 分数可以表示为 0–1 或 0–100。代码用的是 0–100 刻度,而本题我们用 **0–1** 刻度。请把答案四舍五入到 3 位小数。

  根据 BLEU 分数,两个 NMT 翻译中哪个更好?你同意它是更好的翻译吗?

- **(b) [5 分]** 我们的硬盘坏了,丢失了参考翻译 $\mathbf{r}_1$。请只针对 $\mathbf{r}_2$ 重新计算 $\mathbf{c}_1$ 和 $\mathbf{c}_2$ 的 BLEU 分数。现在哪个 NMT 翻译获得更高的 BLEU 分数?你同意它是更好的翻译吗?

- **(c) [2 分]** 由于数据可得性的限制,NMT 系统经常只用单个参考翻译来评估。请解释(几句话)为什么这可能有问题。在解释中,讨论有多个参考翻译 vs 单个参考翻译时,BLEU 指标是如何评估 NMT 翻译质量的。

- **(d) [2 分]** 与人工评估相比,作为机器翻译的评估指标,BLEU 的两个优点和两个缺点是什么?

**(2.4)[4 分]** *Beam search* 常被用来提高机器翻译系统的质量。你训练模型时,同一个示例句在不同 iteration 的 beam search 结果也被记录在 TensorBoard 里,可以在 *TEXT* 标签页访问(见下图)。

> 图:带 beam search 结果的翻译,在 tensorboard 里记录了多个 iteration 的示例句(见 `a3_spr24_student_latex/images/example_translation_beam.jpg`)。同样的数据在你工作目录的 `outputs/beam_search_diagnostics/` 文件夹里也有。

记录的诊断信息包括 json 文档,字段如下:`example_source`(源句 token)、`example_target`(目标句 token 真值)、`hypotheses`(beam size 10 对应的 10 个假设,即搜索结果的 10 条候选翻译)。注意,预测翻译在 NMT 术语里常被称为 *hypothesis*。

- **(a) [2 分]** 随着训练 iteration 增加,翻译质量提高了吗?给出示例句在 iteration 200、3000 和最后一次 iteration 的翻译各一个例子来说明你的答案。每个 iteration 取第一个 beam search 假设为例。
- **(b) [2 分]** beam search 产生的不同假设质量上如何比较?给出最后一次 iteration 里 beam search 提出的另外三个假设为例说明。

---

## 提交说明

在 GradeScope 上分两次提交本作业 — 一次 "Assignment 3 [coding]",一次 "Assignment 3 [written]":

1. 在云端 VM 上运行 `collect_submission.sh` 脚本,生成你的 `assignment3.zip` 文件。参见 GCP How-to 附录中的 *How to Download the Gradescope Submission Package from Your Cloud VM*。
2. 把 `assignment3.zip` 上传到 GradeScope 的 "Assignment 3 [coding]"。
3. 把书面解答上传到 GradeScope 的 "Assignment 3 [written]"。提交时,请按照 Gradescope 的提交说明为每个问题的所有页面打上对应的标签。没有正确打标签会扣分。

---

## 附录:GCP How-to

### 如何获得 SSH 连接信息

按照 [CS224n GCP 指南](https://docs.google.com/document/d/1FLx0CXIn-SoExxKM1efC-E-6iBjUR4uEnpGnfemMMR0/edit?pli=1#heading=h.4tqnggp12z76)操作后,你的 Google Cloud Compute Engine 仪表盘应该如下图所示。点击 SSH 旁边的箭头按钮,选择 *View gcloud command*。你会得到类似下面的命令。记下 zone、machine name 和 project name,示例中分别是 `us-west4-b`、`nvidia-gpu-optimized-vmi-1-vm` 和 `grand-hangar-420500`。**你自己的会不一样!**

```bash
gcloud compute ssh --zone "us-west4-b" "nvidia-gpu-optimized-vmi-1-vm"  --project "grand-hangar-420500"
```

> 图:你创建的云 VM 应该基于 [nvidia-gpu-optimized-vmi](https://console.cloud.google.com/marketplace/product/nvidia-ngc-public/nvidia-gpu-optimized-vmi) 虚拟机镜像(VMI)。部署实例后,点击 SSH 旁的菜单,选择 *View gcloud command* 获得 SSH 连接信息。(见 `a3_spr24_student_latex/images/gcloud-dashboard.jpg`)

### 如何用 SSH 隧道连接云 VM

下面的 ssh 连接命令包含了 `--ssh-flag "-L 6007:localhost:6007"` 选项,它会把云机器的 6007 端口转发到本地机器的同一端口(ssh 隧道)。因此,如果 tensorboard 服务器运行在云 VM 上并监听 6007 端口,建立如下 ssh 连接后,你就可以在本地浏览器的 <http://localhost:6007/> 访问它。

运行下面的 ssh 命令连接云机器。**一定要用你自己的 zone、project name 和 machine name**:

```bash
gcloud compute ssh --zone "us-west4-b" "nvidia-gpu-optimized-vmi-1-vm" --project "grand-hangar-420500" --ssh-flag "-L 6007:localhost:6007"
```

### 如何把代码从电脑复制到云 VM

```bash
# 1. 把当前目录(应包含 env-cpu.yml 和 env-gpu.yml)打成 zip
zip -r student.zip *

# 2. 把得到的 zip 复制到云机器
gcloud compute scp student.zip  nvidia-gpu-optimized-vmi-1-vm:~/  --zone "us-west4-b" --project "grand-hangar-420500"
```

### 如何在云 VM 上训练 NMT 系统

```bash
# 如果云机器没装 zip,先安装再重试
sudo apt-get install zip

# 在云机器上解压 student 文件夹
unzip student.zip -d student
cd student

# 创建 conda GPU 环境
conda env create --file env-gpu.yml

# 激活 conda GPU 环境
conda activate cs224n-nmt-gpu

# 创建一个 tmux 会话,在其中启动翻译模型训练
tmux new -s s-nmt

# (在 tmux 会话里)开始训练
sh run.sh train

# 脱离 tmux 会话
CTRL+B D

# 创建一个 tmux 会话,在其中启动 TensorBoard
tmux new -s s-tboard

# (在 tmux 会话里)启动 tensorboard
tensorboard --logdir runs/ --port 6007

# 脱离 tmux 会话
CTRL+B D
```

- 需要时可以随时 attach/detach 这两个 tmux 会话。参见附录 Tmux 的基本用法。
- 如果你用合适的端口转发选项建立了 SSH 连接,就可以在本地浏览器的 <http://localhost:6007/> 查看 tensorboard(见下图)。
- 训练完成后,运行测试阶段:

```bash
tmux attach -t s-nmt
sh run.sh test
```

- 下一节介绍如何用云 VM 上生成的产物制作 gradescope 提交包。

> 图:Tensorboard 运行在本地机器上,显示云虚拟机的 loss 和 perplexity 值(见 `a3_spr24_student_latex/images/tensorboard_gpu.jpg`)

### 如何从云 VM 下载 Gradescope 提交包

```bash
# 1. 按上面说明连接云 VM
# 2. 在云 VM 上运行以下命令,生成 gradescope 提交包 (assignment3.zip)
sh collect_submission.sh

# 3. 断开与云 VM 的连接(运行 exit),或在本地机器上开一个终端
# 4. 把 GradeScope 提交包下载到本地
gcloud compute scp nvidia-gpu-optimized-vmi-1-vm:~/student/assignment3.zip . --zone "us-west4-b" --project "grand-hangar-420500"

# 5. 把 assignment3.zip 提交到 GradeScope
```

---

## 附录:Tmux How-to

`tmux`(terminal multiplexer)是一个工具,可以让你创建并在多个后台运行的终端之间切换,而且即使你断开 ssh 会话,它们也会继续运行。

下面假设你已经(通过 ssh)连接到了云虚拟机实例,展示 tmux 的基本用法。更多 tmux 知识见 <https://github.com/tmux/tmux/wiki>。

```bash
# 创建并进入一个新 tmux 会话(名为 s-tboard)
tmux new -s s-tboard

# 在新会话里运行程序,这里运行 tensorboard
tensorboard --logdir runs/ --port 6007

# (在 tmux 会话内)脱离会话,让程序继续在会话里运行
CTRL+b d

# (脱离 tmux 后)查看所有 tmux 会话
tmux ls

# 进入某个特定 tmux 会话(名为 s-nmt)
tmux attach -t s-nmt

# 终止一个 tmux 会话(在会话内运行)
exit
```
