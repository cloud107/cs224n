# Part (e) 指南：实现 span corruption

目标文件：`src/dataset.py` 里的 `CharCorruptionDataset.__getitem__`
（`__init__` 已经写好了，不要改任何函数的签名）

---

## 1. 先理解要产出什么

给定一篇文档（wiki.txt 的一行），要把它变成这样的字符串：

```
[prefix] ⁇ [suffix] ⁇ [masked_content] □□□□□□...
```

也就是说：**把文档中间挖掉一段，把挖掉的内容挪到末尾、用 `⁇` 标记两次。**

然后：
- `x = masked_string[:-1]`（前 128 个字符）
- `y = masked_string[1:]`（后 128 个字符）

因为 `masked_string` 长度是 `block_size + 1 = 129`，所以 x 和 y 都是 128。

---

## 2. 手工拆解注释里给的样例

```
x: Khatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□...
y: hatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□...
```

把 `masked_string` 还原出来（= x 的 128 字符 + y 的最后一个字符），然后按 `⁇` 切开：

| 部分 | 内容 | 长度 |
|---|---|---|
| prefix | `Khatchig Mouradian. Khatchig Mouradian is a jour` | 48 |
| `⁇` | | 1 |
| suffix | `and tran` | 8 |
| `⁇` | | 1 |
| masked_content | `nalist, writer ` | 15 |
| pads | `□...` | 补满到 129 |

**验证一下**：prefix + masked_content + suffix =
`...is a jour` + `nalist, writer ` + `and tran`
= `...is a journalist, writer and tran`

对上了——原来挖掉的是 `nalist, writer `，它在原文里正好夹在 prefix 和 suffix 中间。

注意 `masked_content` 的 15 字符 ≈ 71（截断后文档长度）的 1/4 ✓

---

## 3. 五步实现

### 步骤 0：取出文档

`self.data` 是 `wiki.txt` 按 `\n` 切出来的 list（在 `__init__` 里已经做好了）。
用 `idx` 取一行。

### 步骤 1：随机截断

长度在 `[4, int(self.block_size*7/8)]` 之间随机。
`int(128 * 7/8)` = **112**。

**关键**：必须是**均匀随机**——4 到 112 每个长度都要有机会被选到。
autograder 会检查这一点。

Python 的 `random.randint(a, b)` 是**两端都包含**的，正好适合。

> 为什么是 7/8 而不是 128？因为后面要加 2 个 `⁇` 和 mask 内容，
> 留出空间才能保证补齐后不超过 `block_size + 1`。

### 步骤 2：切成三段

要满足 `prefix + masked_content + suffix == 截断后的文档`。

`masked_content` 的长度要**随机**，且**平均约为文档长度的 1/4**。

一个满足"平均 1/4"的简单做法：在 `[1, 文档长度 // 2]` 里均匀取长度。
因为 `Uniform(1, L/2)` 的均值 ≈ `L/4`。

然后还要随机决定 `masked_content` 的**起始位置**，由此得出 prefix 和 suffix。

⚠️ 注意边界：文档最短只有 4 个字符，要保证切片不越界。
建议让 prefix 保持非空（模型需要上下文才能猜出被挖掉的内容），
不过严格按 spec 说，空 prefix 也不算错。

### 步骤 3：拼装 + 补齐

```
masked_string = prefix + MASK_CHAR + suffix + MASK_CHAR + masked_content
```

然后用 `PAD_CHAR` 补齐，使**总长度恰好为 `self.block_size + 1`**。

补多少个？`block_size + 1 - len(masked_string)`。
可以用字符串乘法 `PAD_CHAR * n`。

### 步骤 4：切出 x 和 y

```python
x = masked_string[:-1]
y = masked_string[1:]
```

就是标准的 next-character prediction。

### 步骤 5：编码成张量

用 `self.stoi` 把每个字符转成整数索引。
`CharCorruptionDataset` 的 `__init__` 里已经建好了词表，
而且 `PAD_CHAR` 是索引 **0**、`MASK_CHAR` 是索引 **1**
（看 `__init__` 里两次 `chars.insert(0, ...)` 的顺序）。

返回 `torch.tensor(..., dtype=torch.long)`。

---

## 4. 三个容易错的地方

### (1) `randint` 的边界

`random.randint(a, b)` **包含 b**；`random.randrange(a, b)` **不包含 b**。
这里 spec 要求两端都能取到，所以用 `randint`。

### (2) pad 数量算错

`masked_string` 必须**恰好**是 129 个字符。
如果算错，`x`/`y` 就不是 128，训练时会在 `models.py:107` 的 assert 处报错，
autograder 也会判错。

### (3) 掩码长度不够随机

如果你写成固定值（比如总是 `L//4`），平均值虽然对，
但 spec 明确要求"has a chance of being more or less than 1/4"——
必须每次都不一样。用 `random.randint` 而不是直接算。

---

## 5. 验证方法

```bash
cd /Users/cloud/cs_learning/cs224n/cs224n-spring2024/assignments/a4/student
.venv/bin/python src/dataset.py charcorruption
```

会打印 4 个样例。**对照检查**：

1. 每行的 `x` 和 `y` 应该是**错位一位**的（y 就是 x 往后挪一格）
2. 每行应该有**恰好两个 `⁇`**：第一个在 prefix 后面，第二个在 suffix 后面、
   紧跟着被挖出来的内容
3. `⁇` 后面的内容和后面的 `□` 之间应该有真正的文本（就是 masked_content）
4. 每次运行输出**都不一样**（随机性生效）

**加一个自查**（可选）：写几行代码统计一下平均掩码比例，
应该接近 1/4；再统计 x 的长度，应该恒为 128。

---

## 6. 这一步在整个作业里的作用

`CharCorruptionDataset` 是 part (f) **预训练**用的数据集。

它的逻辑是：
- 输入：被挖掉一段的 wiki 文本
- 目标：让模型学会**根据上下文补全被挖掉的内容**

在做这件事的过程中，模型被迫学习语言的规律、以及文本里的事实知识
（"某人是某地出生的"这类模式）。这些知识就沉淀在参数里，
供 part (f) 的微调"调出来"。
