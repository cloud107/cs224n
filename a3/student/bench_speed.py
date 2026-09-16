#!/usr/bin/env python3
"""
bench_speed.py — 训练速度基准测试(仅用于估算训练时间)

测出本机(Apple M5)上 a3 NMT 模型(embed=1024, hidden=768)的
真实训练吞吐量(target words/sec),并外推到完整训练时长。

注意:
- 通过 monkey-patch 把 NMT.decode / NMT.step 换成标准参考实现(Luong attention),
  纯粹是为了让测得的计算量等于"作业做完后"真实训练的计算量。
  本脚本【不会】修改 nmt_model.py。
- 用法:
    python bench_speed.py stats      # 只统计语料规模
    python bench_speed.py cpu [N]    # CPU 基准(N 为采样句对数, 默认 1500)
    python bench_speed.py mps [N]    # MPS 基准
"""
import sys
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import read_corpus, batch_iter
from vocab import Vocab
from nmt_model import NMT

EMBED = 1024
HIDDEN = 768
BATCH = 32


# ---------------- 参考实现(只用于计时) ----------------
def ref_step(self, Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks):
    dec_state = self.decoder(Ybar_t, dec_state)
    dec_hidden, dec_cell = dec_state
    e_t = torch.bmm(enc_hiddens_proj, dec_hidden.unsqueeze(2)).squeeze(2)  # (b, src_len)
    if enc_masks is not None:
        e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))
    alpha_t = F.softmax(e_t, dim=1)
    a_t = torch.bmm(alpha_t.unsqueeze(1), enc_hiddens).squeeze(1)  # (b, 2h)
    U_t = torch.cat((dec_hidden, a_t), dim=1)
    V_t = self.combined_output_projection(U_t)
    O_t = self.dropout(torch.tanh(V_t))
    return dec_state, O_t, e_t


def ref_decode(self, enc_hiddens, enc_masks, dec_init_state, target_padded):
    target_padded = target_padded[:-1]
    dec_state = dec_init_state
    batch_size = enc_hiddens.size(0)
    o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)
    combined_outputs = []
    enc_hiddens_proj = self.att_projection(enc_hiddens)
    Y = self.model_embeddings.target(target_padded)
    for Y_t in torch.split(Y, 1, dim=0):
        Y_t = Y_t.squeeze(0)
        Ybar_t = torch.cat((Y_t, o_prev), dim=1)
        dec_state, o_t, _ = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
        combined_outputs.append(o_t)
        o_prev = o_t
    return torch.stack(combined_outputs, dim=0)


NMT.step = ref_step
NMT.decode = ref_decode


def train_step(model, opt, src_sents, tgt_sents):
    opt.zero_grad()
    losses = -model(src_sents, tgt_sents)
    loss = losses.sum() / len(src_sents)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    return sum(len(s) - 1 for s in tgt_sents)


def bench(device, model, data, iters=25, warmup=5, label=""):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def next_batch(gen):
        try:
            return next(gen)
        except StopIteration:
            gen = batch_iter(data, batch_size=BATCH, shuffle=True)
            return next(gen)

    gen = batch_iter(data, batch_size=BATCH, shuffle=True)
    words = 0
    t0 = None
    for i in range(iters + warmup):
        src, tgt = next_batch(gen)
        w = train_step(model, opt, src, tgt)
        if i == warmup - 1:
            t0 = time.time()
        elif i >= warmup:
            words += w
    dt = time.time() - t0
    wps = words / dt
    print(f"[{label}] {device}: {wps:.0f} target words/sec "
          f"({words} words in {dt:.1f}s)", flush=True)
    return wps


def load_all():
    print("loading train corpus (200k pairs) ...", flush=True)
    t0 = time.time()
    train_src = read_corpus('zh_en_data/train.zh', source='src', vocab_size=21000)
    train_tgt = read_corpus('zh_en_data/train.en', source='tgt', vocab_size=8000)
    print(f"loaded {len(train_src)} pairs in {time.time()-t0:.1f}s", flush=True)
    return train_src, train_tgt


def stats():
    train_src, train_tgt = load_all()
    lens_src = np.array([len(s) for s in train_src])
    lens_tgt = np.array([len(s) for s in train_tgt])
    total_tgt_words = int((lens_tgt - 1).sum())
    iters_per_epoch = int(np.ceil(len(train_src) / BATCH))
    print(f"src len: mean {lens_src.mean():.1f}  p95 {np.percentile(lens_src,95):.0f}  max {lens_src.max()}")
    print(f"tgt len: mean {lens_tgt.mean():.1f}  p95 {np.percentile(lens_tgt,95):.0f}  max {lens_tgt.max()}")
    print(f"total target words per epoch: {total_tgt_words}")
    print(f"iters per epoch (batch {BATCH}): {iters_per_epoch}")
    with open('/tmp/a3_stats.json', 'w') as f:
        json.dump({'n_pairs': len(train_src), 'total_tgt_words': total_tgt_words,
                   'iters_per_epoch': iters_per_epoch}, f)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 1500

    if mode == 'stats':
        stats()
        return

    train_src, train_tgt = load_all()
    rng = np.random.RandomState(0)
    idx = rng.choice(len(train_src), n_sample, replace=False)
    sample = list(zip([train_src[i] for i in idx], [train_tgt[i] for i in idx]))

    vocab = Vocab.load('vocab.json')
    model = NMT(embed_size=EMBED, hidden_size=HIDDEN, dropout_rate=0.3, vocab=vocab)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"model params: {nparams/1e6:.1f}M", flush=True)

    if mode == 'cpu':
        print(f"torch default threads: {torch.get_num_threads()}", flush=True)
        torch.set_num_threads(10)
        print(f"torch threads set to 10 (4P+6E cores)", flush=True)
        wps = bench(torch.device('cpu'), model, sample, label='CPU')
    elif mode == 'cpu4':
        print(f"torch default threads: {torch.get_num_threads()} (未设置)", flush=True)
        wps = bench(torch.device('cpu'), model, sample, label='CPU-4threads')
    elif mode == 'small':
        print("改用默认模型尺寸 embed=256, hidden=256", flush=True)
        torch.set_num_threads(10)
        model = NMT(embed_size=256, hidden_size=256, dropout_rate=0.3, vocab=vocab)
        print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
        wps = bench(torch.device('cpu'), model, sample, label='CPU-small')
    elif mode == 'mps':
        if not torch.backends.mps.is_available():
            print("MPS NOT available")
            return
        torch.set_num_threads(10)
        wps = bench(torch.device('mps'), model, sample, label='MPS')
    else:
        print("unknown mode")
        return

    with open('/tmp/a3_stats.json') as f:
        st = json.load(f)
    sec_per_epoch = st['total_tgt_words'] / wps
    h30 = sec_per_epoch * 30 / 3600
    print(f"\n>>> {mode}: {wps:.0f} words/sec")
    print(f">>> {sec_per_epoch/60:.1f} min/epoch   ->   30 epochs ~ {h30:.1f} hours "
          f"(纯训练,不含验证)", flush=True)


if __name__ == '__main__':
    main()
