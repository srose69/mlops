# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""A 3-layer transformer that learns to sort 24 numbers, trained with NVFP4 weights on the whole mlops stack
(the dense model: mlops-positional's samples/04_training.py):

* every linear layer of the blocks is an mlops.optim.QATLinear: its weight IS an NVFP4 tensor (E2M1 values, a UE5M3
  scale per 16, an FP32 global), the forward runs NVFP4EffMA's GEMM on it, the backward hands the
  optimizer FP8 (E5M2) gradients;
* AdamWQAT trains those weights in place -- no FP32 master copy, FP8 moments, 5 B of state per weight -- and
  mlops.optim.AdamW the dense rest (embedding, norms, output head);
* mlops.positional.RoPE rotates q and k.

It prints the weight + optimizer memory of the NVFP4 layers (6 B per weight) next to what FP32 weights with torch's
AdamW take (16 B), and ends within a few points of the dense model: on a GTX 1080, 98.3 % of the tokens and 64.5 % of
the sequences sorted, against 98.4 % and 67.6 %.

    python 08_transformer_training.py [e4m3]    (needs mlops-positional and nvfp4effma; 40 s on a GTX 1080)

"e4m3" trains with E4M3 block scales (8x of growth instead of UE5M3's 256x); this model never needs more, so
the run is the same bit for bit.
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlops.optim import AdamW, AdamWQAT, QATLinear, split_params
from mlops.positional import RoPE

N, NUMS = 24, 64
VOCAB, DIM, HEADS, KV_HEADS, LAYERS, STEPS = NUMS + 1, 256, 8, 2, 3, 2000
GRAD = torch.float8_e5m2                            # FP8 gradients for the NVFP4 weights
SCALES = "e4m3" if "e4m3" in sys.argv[1:] else "ue5m3"


def nvfp4_linear(i, o):
    return QATLinear(i, o, init_std=0.02, scale_dtype=SCALES, grad_dtype=GRAD)   # bias-free, in_features a multiple of 64


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.d = DIM // HEADS
        self.qkv = nvfp4_linear(DIM, (HEADS + 2 * KV_HEADS) * self.d)
        self.out = nvfp4_linear(DIM, DIM)
        self.rope = RoPE(self.d, base=10000.0)

    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.qkv(x).view(B, S, HEADS + 2 * KV_HEADS, self.d)
        q, k, v = qkv.split((HEADS, KV_HEADS, KV_HEADS), dim=2)
        q, k = self.rope(q, k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.out(o.transpose(1, 2).reshape(B, S, DIM))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.attn, self.n2 = nn.RMSNorm(DIM), Attention(), nn.RMSNorm(DIM)
        self.mlp = nn.Sequential(nvfp4_linear(DIM, 4 * DIM), nn.SiLU(), nvfp4_linear(4 * DIM, DIM))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


def batch(n):
    xs = torch.randint(0, NUMS, (n, N), device="cuda")
    seq = torch.cat((xs, torch.full((n, 1), NUMS, device="cuda"), xs.sort(-1).values), 1)
    return seq[:, :-1], seq[:, 1:]


def scored(t):
    return t[:, N:]


def evaluate(model):
    with torch.no_grad():
        x, y = batch(512)
        ok = scored(model(x)).argmax(-1) == scored(y)
    return ok.float().mean().item(), ok.all(-1).float().mean().item()


torch.manual_seed(0)
model = nn.Sequential(nn.Embedding(VOCAB, DIM), *[Block() for _ in range(LAYERS)], nn.RMSNorm(DIM),
                      nn.Linear(DIM, VOCAB, bias=False)).cuda()

dense, qat = split_params(model)                    # QATLinear weights vs everything else
layers = [m for m in model.modules() if isinstance(m, QATLinear)]
n_w = sum(m.in_features * m.out_features for m in layers)
n_bytes = sum(b.numel() for m in layers for b in (m.nvfp4_data, m.nvfp4_scales, m.nvfp4_amax, m.qat_state))
print(f"{len(layers)} NVFP4 layers, {n_w / 1e6:.2f} M weights: weight + optimizer state {n_bytes / 2**20:.2f} MiB "
      f"({n_bytes / n_w:.2f} B / weight); FP32 weights + torch AdamW: {16 * n_w / 2**20:.2f} MiB (16 B / weight)")

opt = AdamW(dense, lr=2e-3, weight_decay=0.01)
opt_qat = AdamWQAT(qat, lr=2e-3, weight_decay=0.01)
warmup = lambda s: min(1.0, (s + 1) / 50)
scheds = [torch.optim.lr_scheduler.LambdaLR(o, warmup) for o in (opt, opt_qat)]
for step in range(STEPS + 1):
    x, y = batch(64)
    loss = F.cross_entropy(scored(model(x)).reshape(-1, VOCAB), scored(y).reshape(-1))
    loss.backward()                                 # QATLinear weights get .grad_fp8 (E5M2 + scale)
    opt.step(); opt_qat.step()                      # the NVFP4 codes and scales are rewritten in place
    opt.zero_grad(); opt_qat.zero_grad()
    for s in scheds:
        s.step()
    if step % 250 == 0:
        tok, seq = evaluate(model)
        print(f"step {step:4d}  loss {loss.item():.3f}  sorted tokens {tok:6.1%}  fully sorted sequences {seq:6.1%}")

q = layers[0].nvfp4_weight()
print(f"\nthe first layer's weight is still pure NVFP4: {q}")
