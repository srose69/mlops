# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Training: RoPE is differentiable (its backward is the inverse rotation, the same kernel).

    python 04_training.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlops.positional import RoPE


class Attention(nn.Module):
    def __init__(self, dim=256, heads=8, kv_heads=2):
        super().__init__()
        self.h, self.kv, self.d = heads, kv_heads, dim // heads
        self.qkv = nn.Linear(dim, (heads + 2 * kv_heads) * self.d, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(self.d, base=10000.0)

    def forward(self, x):                                     # x [B, S, dim]
        B, S, _ = x.shape
        qkv = self.qkv(x).view(B, S, self.h + 2 * self.kv, self.d)
        q, k, v = qkv.split((self.h, self.kv, self.kv), dim=2)  # strided views of one projection: fine
        q, k = self.rope(q, k)                                # [B, S, H, D]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.out(o.transpose(1, 2).reshape(B, S, -1))


torch.manual_seed(0)
model = nn.Sequential(nn.Embedding(100, 256), Attention(), nn.Linear(256, 100)).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
data = torch.randint(0, 100, (32, 65), device="cuda")
for step in range(100):                                       # learn to copy the token 4 positions back
    x, y = data[:, :-1], torch.roll(data[:, :-1], 4, dims=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = F.cross_entropy(model(x)[:, 4:].float().reshape(-1, 100), y[:, 4:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 25 == 0 or step == 99:
        print(f"step {step:3d}  loss {loss.item():.3f}")
