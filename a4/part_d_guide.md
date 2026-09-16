# Part (d) 操作指南

冒烟通过了，`train()` 也补上了 ✅。现在正式进入 part (d)。

## 第一步：改回正式超参数

`src/run.py` 里两处（改完别再动）：

- `max_epochs=2` → `75`
- `final_tokens=2*len(...)` → `200*len(...)`

> 注意：part (f) 时这里要按 `args.reading_params_path` 是否为空在 75 / 10 之间切换。现在先写死 75，到 part (f) 再补。

## 第二步：正式训练

```bash
cd /Users/cloud/cs_learning/cs224n/cs224n-spring2024/assignments/a4/student
.venv/bin/python src/run.py finetune vanilla wiki.txt \
    --writing_params_path vanilla.model.params \
    --finetune_corpus_path birth_places_train.tsv
```

MPS 上大概几分钟到十几分钟。产出 `vanilla.model.params`。
想可视化 loss 曲线可以另开一个终端跑 `tensorboard --logdir expt/`。

## 第三步：评估 dev 集（part d 要报的数）

```bash
.venv/bin/python src/run.py evaluate vanilla wiki.txt \
    --reading_params_path vanilla.model.params \
    --eval_corpus_path birth_dev.tsv \
    --outputs_path vanilla.nopretrain.dev.predictions
```

记下它打印的 `Correct: X out of 500: Y%` —— 这就是要交的 dev 准确率。
handout 说 "well below 10%" 是正常的，别慌，这正是在给 part (f) 做对照。

## 第四步：评估 test 集

```bash
.venv/bin/python src/run.py evaluate vanilla wiki.txt \
    --reading_params_path vanilla.model.params \
    --eval_corpus_path birth_test_inputs.tsv \
    --outputs_path vanilla.nopretrain.test.predictions
```

test 集没有标准答案（只有问句），所以它会打印 "no targets provided" 而不是准确率
——这是正常的，只要文件生成了就行。

## 第五步：写 `london_baseline.py`

思路（代码你自己写，注释里说 "only a few lines"）：

`utils.evaluate_places(filepath, predicted_places)` 的接口是（utils.py:56-76）：

- 传入**文件路径** + **预测列表**（Python 字符串 list）
- 返回 `(total, correct)` 两个 float
- 内部有个 `assert total == len(predicted_places)`
  —— 所以你的预测列表长度必须和 dev 集行数一致

你要做三件事：

1. 造一个**全是 `"London"` 的列表**，长度等于 dev 集的样本数。
   怎么知道多少个？数一下文件行数（`birth_dev.tsv`），或用 Python 读文件后 `len(...)`
2. 拿这个列表和 dev 集路径调 `utils.evaluate_places`
3. 把 `(total, correct)` 换算成**百分比**
   （返回值是 0-100 区间，看第 13 行注释 `Compute accuracy in the range [0.0, 100.0]`）

两个提示：

- 骨架里 `import argparse` 已经给你了，你可以加个命令行参数传 dev 集路径，
  或者直接把 `'birth_dev.tsv'` 写死 —— 后者更简单
- dev 集路径就是 `birth_dev.tsv`（相对当前目录）

写好后运行：

```bash
.venv/bin/python src/london_baseline.py
```

它会把结果写进 `london_baseline_accuracy.txt`（第 22 行已经写好了，**文件名不用你操心**，
注意这个文件是提交清单里的一项）。

> 自查方法：用 `grep -c` 数一下 dev 集里真实答案是 London 的有多少行，除以 500
> 就该等于你的结果。如果对不上，说明预测列表长度或换算有误。

---

跑训练的时候如果慢，正好可以同时写书面题 Q1/Q2。
