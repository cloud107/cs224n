# Part (f) 指南：预训练 + 加载预训练权重微调（10 分）

这是整个作业的**核心实验**：证明预训练能让同一个模型从 2.4% 提升到 15%+。

---

## 1. 要写的代码：两处

### (1) `run.py` 的 pretrain 分支（约 105 行）

注释里已经把超参数全列出来了：

```
max_epochs=650
batch_size=128
learning_rate=args.pretrain_lr        # 默认 6e-3
lr_decay=True
warmup_tokens=512*20
final_tokens=650*len(pretrain_dataset)*block_size
num_workers=4        ← 本地改成 0
writer=writer
```

结构和你已经写好的 finetune **几乎一模一样**：

1. 构造 `TrainerConfig`（用上面的参数）
2. `Trainer(model, pretrain_dataset, None, tconf)`
3. `.train()`
4. `torch.save(model.state_dict(), args.writing_params_path)`

注意 `pretrain_dataset` **已经在 run.py:51 建好了**，直接用，不用重建。

> 和你 finetune 的区别只有一个：数据集换成 `pretrain_dataset`
> （`CharCorruptionDataset`，你刚写完的那个），超参数不同。

### (2) `run.py` 的 finetune 分支：epoch 数要分岔

**加载权重的部分你已经写好了**（`if args.reading_params_path is not None` ✓）。
还差最后一件事——注释里说：

> You can use the args.reading_params_path flag to switch between the number of epochs

- 没有预训练（`reading_params_path is None`）→ `max_epochs=75`
- 有预训练 → `max_epochs=10`

因为预训练过的模型只需要很少的轮次就能"适配"到任务上，
而从头训练的模型需要反复看很多遍数据。

---

## 2. 要跑的四条命令

```bash
cd /Users/cloud/cs_learning/cs224n/cs224n-spring2024/assignments/a4/student

# ① 预训练（最久，40-60 分钟起步）
.venv/bin/python src/run.py pretrain vanilla wiki.txt \
    --writing_params_path vanilla.pretrain.params

# ② 用预训练权重微调
.venv/bin/python src/run.py finetune vanilla wiki.txt \
    --reading_params_path vanilla.pretrain.params \
    --writing_params_path vanilla.finetune.params \
    --finetune_corpus_path birth_places_train.tsv

# ③ 评估 dev（期望 ≥ 15%）
.venv/bin/python src/run.py evaluate vanilla wiki.txt \
    --reading_params_path vanilla.finetune.params \
    --eval_corpus_path birth_dev.tsv \
    --outputs_path vanilla.pretrain.dev.predictions

# ④ 评估 test
.venv/bin/python src/run.py evaluate vanilla wiki.txt \
    --reading_params_path vanilla.finetune.params \
    --eval_corpus_path birth_test_inputs.tsv \
    --outputs_path vanilla.pretrain.test.predictions
```

---

## 3. ⚠️ 时间问题（提前规划）

预训练是这个作业最耗时的部分：

- 650 epochs × 2937 篇文档 ÷ batch_size 128 ≈ **15,000 步**
- handout 说 GCP/Colab GPU 上 40-60 分钟
- **本地 Mac MPS 会慢很多**，可能是几小时

**建议先测速再决定**：把 `max_epochs` 临时改成 2，跑一遍看耗时 T，
那么完整训练大约是 `T × 325`。如果算出要跑 5 小时，就考虑：
- 上 Colab / GCP（handout 推荐的做法）
- 或者接受本机慢慢跑（挂着去写书面题）

> 注意 `num_workers` 本地必须改成 0，否则 DataLoader 多进程会崩。

---

## 4. ⚠️ 重要：RoPE 在 Mac 上跑不了 MPS（影响 part g）

看 run.py:33-34：

```python
elif torch.backends.mps.is_available() and args.variant == 'vanilla':
    device = 'mps'
```

**只有 vanilla 能用 MPS，rope 会回落到 CPU。**

trainer.py:53 也有同样的限制：

```python
elif torch.backends.mps.is_available() and not self.model.rope:
```

所以 part (g) 的 RoPE 模型在你这台 Mac 上会**跑在 CPU 上**，
速度大概是 MPS 的几十分之一。这是 part (g) 最大的现实障碍，
**提前想好是上 GPU 还是用本机硬扛**。

---

## 5. 做完后要记录的数据

part (f) 和 part (d) 的对比是整个作业的论证核心，记下来：

| 配置 | dev 准确率 |
|---|---|
| 无预训练（part d） | 2.4% |
| 有预训练（part f） | ?（期望 ≥15%） |
| London baseline | 5.0% |

这三行数字要写进书面题（Q3 末尾那部分 + Q4）。

**另外**：handout 让你"take a look at some of the correct predictions...
as well as some of the errors"——去翻一翻
`vanilla.pretrain.dev.predictions` 和 `birth_dev.tsv` 的对照，
观察模型答对的和答错的分别长什么样。Q4(b) 的两点讨论要靠这些观察。
