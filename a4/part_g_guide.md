# Part (g) 指南：RoPE（旋转位置编码），11 分

分值构成：**(i) 2 分 + (ii) 1 分（书面） + (iii) 8 分（代码 + 训练）**

代码量比 part (e) 小得多，但**概念密度最高**——这是整个作业里唯一需要你真正
理解"位置编码为什么这么做"的部分。训练也是四连命令，和 part (f) 完全对称。

---

## 0. 先明确：这一部分要产出什么

| 类型 | 内容 |
|---|---|
| 代码 | `attention.py` 4 处 + `run.py` 1 处（`[part g]` 标记） |
| 书面 | (i) 证明 Eq 4 的复数算法等价于 Eq 3 的矩阵形式 |
| 书面 | (ii) 证明 RoPE 的点积只依赖相对位置 `t1 - t2` |
| 产物 | `rope.pretrain.params` / `rope.finetune.params` / 两个 `.predictions` |
| 指标 | **test 集 ≥ 30%**（注意比 vanilla 的 15% 要求更高） |

---

## 1. RoPE 到底在做什么（一句话版）

**vanilla 把位置信息"加"进输入 embedding；RoPE 把位置信息"转"进 q 和 k。**

- vanilla：`x_input = token_emb + pos_emb`，位置是**绝对**的、可学习的一组向量。
- RoPE：input embedding **完全不加位置信息**（`models.py:111-112` 已经写好了，
  你不用改），而是在每一层的 attention 里，把 **q 和 k** 按它们在序列中的位置
  **旋转**一个角度。

旋转的美妙之处：两个向量各自转 `t1θ` 和 `t2θ` 之后，它们的**点积**只跟
`(t1 - t2)θ` 有关——也就是只跟**相对距离**有关。这正是 (ii) 要证的。

```python
# models.py:111  —— 已经给你了，不要改
if self.rope:
    x_input = token_embeddings      # 没有 position_embeddings 这一项
else:
    x_input = token_embeddings + position_embeddings
```

推论：**rope 模型的 state_dict 里没有 `pos_emb`，vanilla 的有。**
两个变体的 `.params` 文件**不能互相加载**，别搞混。

---

## 2. 数学部分：把公式落到代码上

### 角度 θ 的定义（最容易写错的地方）

handout 给的是：`θ_i = 10000^(-2(i-1)/d)`，`i ∈ {1, ..., d/2}`

⚠️ **`i` 是从 1 开始的**。换成 0 起始的下标 `j = i - 1`：

```
θ_j = 10000^(-2j/d),   j ∈ {0, 1, ..., d/2 - 1}
```

也就是 `10000 ** (torch.arange(0, d, 2) / d)` 的倒数。写成代码就是
`inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2) / d))`，
形状 `(d/2,)`。

> `d` 是**每个 head 的维度**，不是 `n_embd`。这里 `n_embd=256, n_head=8`
> → `head_dim = 32` → 缓存里最后一个维度是 **16**（= 32/2）。
> `attention.py:82` 那个 assert `(n_embd % n_head) % 2 == 0` 就是在保证
> head_dim 是偶数、能两两配对。

### 配对方式：相邻两个特征配成一对

Eq 4 里的复数向量是 `(x⁽¹⁾ + i x⁽²⁾), (x⁽³⁾ + i x⁽⁴⁾), ...`

**是"相邻两个"配对，不是"前半段和后半段"配对。**

这正是为什么注释建议你用 `torch.view_as_complex`：
`reshape(..., head_dim//2, 2)` 之后再 `view_as_complex`，得到的实部/虚部
就是 `(x1, x2), (x3, x4), ...`，**天然就是正确的配对**。

> ⚠️ 如果你用另一种常见的写法（`rotate_half`：把 head_dim 劈成前后两半，
> `[-x[d/2:], x[:d/2]]`），配对方式变成 `(x1, x9), (x2, x10), ...`。
> 数学上它也是一种合法的位置编码，但**不符合本题 Eq 4 的定义**，
> 而且 θ 的分配也对不上。别用。

### 复数乘法 = 旋转

`(a + ib) · (cos tθ + i sin tθ) = (a cos tθ - b sin tθ) + i(a sin tθ + b cos tθ)`

对照 Eq 3 的 `[[cos, -sin], [sin, cos]]`（作用在 `[a, b]` 上）——完全一致 ✓
这就是 (i) 要写的东西。

---

## 3. 五处代码

### (1) `precompute_rotary_emb(dim, max_positions)` → `(max_positions, dim//2, 2)`

三行思路：

1. `inv_freq`：形状 `(dim//2,)`，按上面的 `θ_j` 公式
2. `t`：`torch.arange(max_positions)`，形状 `(max_positions,)`
3. 用**外积**得到每个位置每个维度的角度 `angles[t, j] = t * θ_j`，
   形状 `(max_positions, dim//2)`
4. `cos` 和 `sin` 各算一次，在最后一维拼起来 → `(max_positions, dim//2, 2)`

⚠️ 拼接顺序：注释说 "the last dimension contains the cos and sin values"，
即 `[..., 0] = cos`，`[..., 1] = sin`。这个顺序直接决定了
`view_as_complex` 出来的实部/虚部对不对，**写反了整个模型就学不动**。

### (2) `CausalSelfAttention.__init__` 里建缓存

```python
# Hint: The maximum sequence length is given by config.block_size.
```
把 `dim` 传成 `config.n_embd // config.n_head`，`max_positions` 传
`config.block_size`（=128）。结果 `(128, 16, 2)`，然后按注释
`self.register_buffer("rope_cache", rope_cache)`。

> 用 `register_buffer` 的好处：`.to(device)` 时自动跟着走，
> 也会进 state_dict（所以 `.params` 文件比 vanilla 略大/略小都正常）。

### (3) `apply_rotary_emb(x, rope_cache)`

`x` 的形状在这里是 `(B, nh, T, head_dim)`（看 `forward` 里 q/k 的形状）。

步骤：

1. **截断**：`rope_cache = rope_cache[:T]` —— 注释专门提醒了这一点，
   因为 evaluate 时 prompt 可能短于 128。
2. 把 `x` 的最后两维 `(head_dim)` reshape 成 `(head_dim//2, 2)`，
   **`view_as_complex` 要求内存连续**，所以这里需要 `.contiguous()`，
   否则会报错（这是最常见的报错来源）。
3. `view_as_complex` 得到 `(B, nh, T, head_dim//2)` 的复数张量
4. 缓存同样 `view_as_complex` 得 `(T, head_dim//2)`，
   `unsqueeze` 成 `(1, 1, T, head_dim//2)` 以便对 `B, nh` 广播
5. 相乘 → `view_as_real` → reshape 回 `(B, nh, T, head_dim)`

### (4) `CausalSelfAttention.forward` 里应用

```python
if self.rope:
    # 只对 q 和 k 做，v 不动
```

⚠️ **不要动 `v`。** RoPE 的目的是让 attention 打分 `q·k` 带相对位置信息，
`v` 是"被加权的内容"，跟位置分数无关。给 v 也加会让结果变差。

### (5) `run.py` 的 rope 分支（第 72-77 行）

两行：

```python
mconf.rope = True
model = models.GPT(mconf).to(device)
```

`mconf` 是第 55 行建好的 `GPTConfig`，`rope` 是它的类属性（`models.py:33`
默认 `False`）。注意**先设 `mconf.rope = True` 再建模型**——因为
`GPT.__init__` 会读 `config.rope` 决定要不要建 `pos_emb`，attention 的
`__init__` 也会读它决定要不要建 `rope_cache`。

---

## 4. 书面题 (i)(ii) 的证明思路

### (i) Eq 4 的元素 = Eq 3 的元素（2 分）

三步说完即可，注释明确说了"不需要详细推导"：

1. Eq 3 是一个**分块对角矩阵**，每个 2×2 块作用在一对相邻特征
   `(x^(2j-1), x^(2j))` 上，做的是角度 `tθ_j` 的旋转。
2. Eq 4 把每对相邻特征打包成复数 `x^(2j-1) + i x^(2j)`，乘以
   `cos tθ_j + i sin tθ_j`。由复数乘法定义展开，实部
   `= x^(2j-1) cos tθ_j - x^(2j) sin tθ_j`，虚部
   `= x^(2j-1) sin tθ_j + x^(2j) cos tθ_j`。
3. 把复数结果按 `(实部, 虚部)` 还原成 2 维实向量，逐对拼回去，正好等于
   Eq 3 的结果。Eq 4 用逐元素乘法代替了矩阵乘法，计算更高效。

### (ii) 点积只依赖相对位置（1 分）

用 handout 给的 hint，两行就够：

设 `z1, z2` 是复数。RoPE 就是乘一个单位复数：
`RoPE(z, t) = z · e^{itθ}`（因为 `|e^{itθ}| = 1`，只旋转不缩放）。

复数表示的向量点积是 `⟨z1, z2⟩ = Re(z1 · z̄2)`，所以：

```
⟨RoPE(z1, t1), RoPE(z2, t2)⟩
  = Re( z1 e^{i t1 θ} · conj(z2 e^{i t2 θ}) )
  = Re( z1 z̄2 · e^{i t1 θ} · e^{-i t2 θ} )        ← conj(e^{it2θ}) = e^{-it2θ}
  = Re( z1 z̄2 · e^{i (t1 - t2) θ} )
  = ⟨RoPE(z1, t1 - t2), RoPE(z2, 0)⟩             ← RoPE(z2,0) = z2·e^0 = z2
```

关键一步是**共轭把 `e^{i t2 θ}` 变成 `e^{-i t2 θ}`**，两个指数才会相减成
`t1 - t2`。(multiplicative 结构 + 共轭 = 相对位置)

---

## 5. 训练：四条命令 + ⏱️ 时间预算（实测）

```bash
cd /Users/cloud/cs_learning/cs224n/cs224n-spring2024/assignments/a4/student

# ① 预训练 RoPE
.venv/bin/python src/run.py pretrain rope wiki.txt \
    --writing_params_path rope.pretrain.params

# ② 微调
.venv/bin/python src/run.py finetune rope wiki.txt \
    --reading_params_path rope.pretrain.params \
    --writing_params_path rope.finetune.params \
    --finetune_corpus_path birth_places_train.tsv

# ③ 评估 dev（对比用）
.venv/bin/python src/run.py evaluate rope wiki.txt \
    --reading_params_path rope.finetune.params \
    --eval_corpus_path birth_dev.tsv \
    --outputs_path rope.pretrain.dev.predictions

# ④ 评估 test（**≥30% 才算过**）
.venv/bin/python src/run.py evaluate rope wiki.txt \
    --reading_params_path rope.finetune.params \
    --eval_corpus_path birth_test_inputs.tsv \
    --outputs_path rope.pretrain.test.predictions
```

### 实测速度（我刚在这台 Mac 上跑的 benchmark）

预训练总步数 `650 × (2937 // 128) = 14,300` 步。

| 设备 | 每步耗时 | 预训练总时长 |
|---|---|---|
| CPU | 0.69 s | **≈ 2.7 小时** |
| MPS | 0.25 s | **≈ 1.0 小时** |

> 更正 part (f) 指南里的预估：当时说"CPU 是 MPS 的几十分之一"，
> **实测只差 2.7 倍**。这台 M 系列芯片的 CPU 比想象中能打，2.7 小时挂一夜
> 是完全可以接受的方案。

### ⚠️ 但按原样跑，rope 会被强制走 CPU

`run.py:33` 和 `trainer.py:53` 的 MPS 判断都带了 `and ... == 'vanilla'` /
`not self.model.rope`——**课程代码故意/顺手把 RoPE 排除在 MPS 之外**。

三条路，按推荐顺序：

1. **在 Linux 那台机器上跑**（如果 part f 是在那儿跑的）。`run.py:31` 的
   CUDA 分支对两个变体一视同仁，handout 说 GPU 上 40-60 分钟。
   跑完照旧 commit + 在 Mac 上 pull。
2. **本地放开 MPS 限制**：把那两处条件改成不排除 rope。我实测过
   **torch 2.14 的 MPS 支持 complex 运算**（`view_as_complex` + 复数乘法
   正常返回），所以理论上可行 → 1.0 小时。代价是改了两行课程给的设备选择
   代码（`run.py:63` 那句 note 只是说"模型要搬到 30-34 行选定的 device 上"，
   并没有禁止改设备逻辑本身）。改之前建议先跑 2 个 epoch 验证不报错。
3. **本地 CPU 硬扛**：2.7 小时，不用改任何代码，最省心。

---

## 6. 写完代码后怎么自查（趁训练还没开始）

把 `max_epochs` 临时改成 2 跑一遍太慢，**先在 python 里做几个数值检查**：

```python
# 在 student/ 目录下
import sys, torch; sys.path.insert(0, 'src')
import attention
cache = attention.precompute_rotary_emb(32, 128)
print(cache.shape)      # 期望 torch.Size([128, 16, 2])
print(cache[0, 0])      # t=0 时角度为 0 → cos=1, sin=0 → tensor([1., 0.])
print(cache[1, 0])      # t=1, θ_0 = 1 → cos(1)≈0.5403, sin(1)≈0.8415
```

四个**不依赖实现细节**的性质检查（推荐都做一遍）：

1. **t=0 是恒等变换**：位置 0 的向量旋转 0 度，`apply_rotary_emb` 作用后
   第 0 个位置应和输入完全相等。
2. **保范数**：旋转不改变模长，所以每对特征的 L2 范数不变 →
   `rotated.norm() == x.norm()`（浮点误差内）。
3. **相对位置性质**（就是 (ii) 的数值验证）：
   `⟨RoPE(q, t1), RoPE(k, t2)⟩ ≈ ⟨RoPE(q, t1-t2), RoPE(k, 0)⟩`
4. **能跑通整个模型**：`models.GPT(GPTConfig(100, 128, n_layer=4, n_head=8,
   n_embd=256, rope=True))` 前向一次，检查输出形状 `(B, T, vocab_size)`
   且 loss 不是 nan。

这四条过了，再去跑真实的 650 epochs，不会浪费 2 小时。

---

## 7. 训练完要记录的数据（写进 Q3/Q4 的对比）

| 配置 | dev 准确率 |
|---|---|
| 无预训练（part d） | 2.4% ✓ |
| 预训练 vanilla（part f） | **22.4%** ✓（你已跑出） |
| 预训练 RoPE（part g） | ? |
| London baseline | 5.0% ✓ |

RoPE 的 dev 大概是 20-30% 这个量级——**它不一定比 vanilla 高**，
这不重要：g 的评分点是 **test ≥ 30%** 和"证明相对位置编码也 work"，
不是"打败 vanilla"。如果 dev 明显低于 15%，那是实现有 bug，别急着跑完。

---

## 8. 提交物

`collect_submission.sh` 会检查这些文件（RoPE 那三行是 part g 新增的）：

- `rope.finetune.params`
- `rope.pretrain.dev.predictions`
- `rope.pretrain.test.predictions`

> ⚠️ 注意命名坑：**dev 的 predictions 文件里存的是 finetune 后的权重跑出来的
> 结果**，但按 handout 要求文件名是 `rope.pretrain.dev.predictions`
> （vanilla 同理）。别被 `pretrain` 这个词误导。

---

## 9. 一句话总结 part g 在整份作业里的位置

part (a)-(e) 你搭了一个能训练的 Transformer；part (f) 证明了**预训练有用**；
part (g) 让你亲手把**位置编码**这一个组件换成另一套设计，并且亲眼看到：
换掉之后模型照样能训练、照样能到 30%。

这就是现代 LLM 里 RoPE 成为默认选择的原因——它不需要学习参数、
天然编码相对位置、还能通过外推支持比训练时更长的上下文
（这正是 handout 开头说的 "the issue with absolute positions"）。
