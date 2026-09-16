# Part (c) 详解：你写的那段 finetune 逻辑到底在干什么

## 1. part (c) 在整体中的位置

作业的主线是一条"对照实验"：

| 阶段 | 训练数据 | 起点 | 结果 |
|---|---|---|---|
| part (c/d) | birth_places_train.tsv | 随机初始化 | dev 2.4% |
| part (f) | wiki.txt → birth_places_train.tsv | 随机初始化 → 预训练权重 | dev 15%+ |

part (c) 是三段里最简单的一段：**不预训练，直接拿任务数据训练**。
它本身不计分，但它是"对照组"，也是 part (f) 的代码基础。

---

## 2. 你写的四步，逐个讲

```python
# ① 条件加载预训练权重（part c 时 reading_params_path 是 None，跳过）
if args.reading_params_path is not None:
    model.load_state_dict(torch.load(args.reading_params_path))

# ② 构造训练数据集
dataset_text = open(args.finetune_corpus_path, encoding='utf-8').read()
finetune_dataset = dataset.NameDataset(pretrain_dataset, dataset_text)

# ③ 配置 Trainer
tconf = trainer.TrainerConfig(max_epochs=75, batch_size=256, ...)

# ④ 训练 + 保存
finetune_trainer = trainer.Trainer(model, finetune_dataset, None, tconf)
finetune_trainer.train()
torch.save(model.state_dict(), args.writing_params_path)
```

---

## 3. 核心概念：为什么"微调"和"从头训练"是同一段代码

这是 part (c)/(f) 最值得理解的一点。

**因为两个任务的数学形式完全相同：都是"预测下一个字符"。**

- 预训练（span corruption）：输入被挖空的 wiki 文本，预测被挖掉的内容
- 微调（出生地问答）：输入 "Where was X born?⁇"，预测 "Place⁇"

两者的损失函数都是 `F.cross_entropy`（models.py:125），模型结构一模一样，
Trainer 也一模一样。**唯一的区别是权重的起点**：

- part (c)：从随机初始化开始
- part (f)：从预训练权重开始（就是那个 `if` 分支）

这就是为什么 handout 说 "modify your finetune portion to handle finetuning
in the case with pretraining" —— 只需要加一个 if，而不是重写一个函数。

---

## 4. 数据格式的巧妙设计（NameDataset）

你看到的数据长这样：

```
x: Where was Khatchig Mouradian born?⁇Lebanon⁇□□□□□□□□□□...
y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Lebanon⁇□□□□□□□□□□...
```

看 `NameDataset.__getitem__`（dataset.py:136-145）：

```python
x = inp + MASK_CHAR + oup + MASK_CHAR      # 问题 ⁇ 答案 ⁇
x = x + PAD_CHAR*(block_size - len(x))     # 补 pad 到 128
y = PAD_CHAR*(len(inp)-1) + x[len(inp):]   # y 是 x 右移一位，但开头补 pad
```

三个设计点：

### (1) 为什么 y 开头是一串 pad

`y` 本该是 `x` 右移一位（标准的 next-character prediction）。
但在 "Where was ... born?" 这段位置上，**答案被 pad 覆盖了**。

原因：模型在那些位置本来要预测 "Where was..." 的下一个字符，
但模型**不需要学怎么复述问题**——那是浪费。把那些位置的标签设成 pad 之后，
它们被 `ignore_index=0` 排除在 loss 之外（models.py:125），
模型只被要求学一件事：**看到 `⁇` 之后，输出地名**。

### (2) 那个结尾的 `⁇` 是终止符

看 x 的结构：`问题 ⁇ 答案 ⁇`。第二个 `⁇` 告诉模型"答案到此结束"。
evaluate 时的代码（run.py:167-174）就是靠它切出预测的：

```python
x = x + '⁇'                                  # 给输入也补一个 ⁇
pred = completion.split('⁇')[1]              # 取两个 ⁇ 之间的内容
```

### (3) block_size 固定 128

不管句子多长，都补到 128 个字符。这也是为什么 `pad` 占了序列的大头——
不补齐的话 batch 里的样本长度不一，没法堆成张量。

---

## 5. 超参数为什么这么设

| 参数 | 值 | 含义 |
|---|---|---|
| `max_epochs` | 75 | 没预训练，要跑很多轮才学得动 |
| `batch_size` | 256 | 一次看 256 个样本 |
| `learning_rate` | `args.finetune_lr` = 6e-4 | 比预训练的 6e-3 小 10 倍 |
| `lr_decay` | True | 用余弦衰减，后期减小学习率 |
| `warmup_tokens` | 512*20 | 预热阶段的 token 数 |
| `final_tokens` | 200*len(...)*block_size | 衰减到初始 lr 10% 的位置 |

**一个小细节**：warmup 在这套配置下几乎不起作用。
`self.tokens += (y >= 0).sum()`（trainer.py:108）每步加的是 `batch × block`
= 256 × 128 = 32768，而 `warmup_tokens` = 10240 —— **第一步就超过阈值了**，
所以直接进入余弦衰减。这个值是照搬 GPT-3 paper 的默认值，在这里偏小。

**注意 `final_tokens` 用 75（无预训练）还是 10（有预训练）的差别**：
它控制学习率衰减的快慢。如果这里填错（比如用 notebook 里的 `2`），
衰减会在训练刚开始就完成，之后全程在极低学习率下爬行，准确率会明显偏低。
这就是我之前提醒你把它从 `2*` 改成 `200*` 的原因。

---

## 6. 结果解读：2.4% 这个数字意味着什么

**先记住 London baseline = 5.0%。**

也就是说：**你从头训练的模型（2.4%）比"不管问谁都回答 London"（5.0%）还差。**

这正是 handout 让你算 London baseline 的目的——给准确率一个参照系。
在一个 birthplace 分布高度集中（伦敦占比最高）的数据集上，
5% 是"完全不学习、只靠数据分布"就能拿到的分数。

而 2.4% 说明模型**连数据分布都没学好**，它在编造：
你预测里出现的 `'Goanada'`、`'Hunga'` 都不是真实地名，
说明模型只学到了一点字符级的表面模式（比如"地名开头大写"），
完全没有"世界知识"。

**为什么必然如此？** 训练集只有几千条 人名→地名 的配对，
模型从随机初始化开始，不可能在这点数据里学会"Khatchig Mouradian 是黎巴嫩人"
这种事实——它没有见过任何关于世界的文本。

**这就是 part (f) 要解决的问题**：先用 wiki.txt 预训练，
让模型在参数里存下世界知识，再用微调把那部分知识"调出来"。

---

## 7. 一句话总结

part (c) 建立了一个**基线**：一个只会字符统计、没有任何世界知识的模型，
在需要事实知识的任务上，表现比"永远回答 London"还差。

part (f) 要做的，就是让同一个模型、同一套代码，只因为换了权重起点，
就把分数从 2.4% 拉到 15% 以上。
