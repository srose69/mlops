# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Training with RoPE: a 3-layer transformer learns to sort. A sequence is 24 numbers (0..63, with repeats), a
separator, then the same numbers in ascending order; the model writes the sorted half token by token. Every output
token needs the rank of each input among the others and the position reached so far in the output, so it takes
several attention layers and positions (RoPE). Chance is 1/64 per token.

RoPE is differentiable (its backward is the inverse rotation, the same kernel) and works on the strided q / k
views of one packed qkv projection. mlops-optimizers' samples/08_transformer_training.py trains the same model with
NVFP4 weights.

    python 04_training.py          (40 s on a GTX 1080: 98 % of the tokens, two thirds of the sequences sorted)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlops.positional import RoPE

N, NUMS = 24, 64                                   # sort 24 numbers from 0..63; token 64 separates
VOCAB, DIM, HEADS, KV_HEADS, LAYERS, STEPS = NUMS + 1, 256, 8, 2, 3, 2000


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.d = DIM // HEADS
        self.qkv = nn.Linear(DIM, (HEADS + 2 * KV_HEADS) * self.d, bias=False)
        self.out = nn.Linear(DIM, DIM, bias=False)
        self.rope = RoPE(self.d, base=10000.0)

    def forward(self, x):                                       # x [B, S, DIM]
        B, S, _ = x.shape
        qkv = self.qkv(x).view(B, S, HEADS + 2 * KV_HEADS, self.d)
        q, k, v = qkv.split((HEADS, KV_HEADS, KV_HEADS), dim=2)   # strided views of one projection
        q, k = self.rope(q, k)                                  # [B, S, H, D], positions 0..S-1
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.out(o.transpose(1, 2).reshape(B, S, DIM))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.attn, self.n2 = nn.RMSNorm(DIM), Attention(), nn.RMSNorm(DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM, 4 * DIM, bias=False), nn.SiLU(), nn.Linear(4 * DIM, DIM, bias=False))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


def batch(n):
    xs = torch.randint(0, NUMS, (n, N), device="cuda")
    seq = torch.cat((xs, torch.full((n, 1), NUMS, device="cuda"), xs.sort(-1).values), 1)
    return seq[:, :-1], seq[:, 1:]                              # inputs, next tokens


def scored(t):  # the predictions of the sorted half
    return t[:, N:]


def evaluate(model):
    with torch.no_grad():
        x, y = batch(512)
        ok = scored(model(x)).argmax(-1) == scored(y)
    return ok.float().mean().item(), ok.all(-1).float().mean().item()


torch.manual_seed(0)
model = nn.Sequential(nn.Embedding(VOCAB, DIM), *[Block() for _ in range(LAYERS)], nn.RMSNorm(DIM),
                      nn.Linear(DIM, VOCAB, bias=False)).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50))
for step in range(STEPS + 1):
    x, y = batch(64)
    loss = F.cross_entropy(scored(model(x)).reshape(-1, VOCAB), scored(y).reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 250 == 0:
        tok, seq = evaluate(model)
        print(f"step {step:4d}  loss {loss.item():.3f}  sorted tokens {tok:6.1%}  fully sorted sequences {seq:6.1%}")
