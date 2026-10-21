# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Parameter groups, an LR schedule and gradient clipping: everything a real training loop has.

    python 03_groups_schedule_clip.py

Both optimizers are torch.optim.Optimizer subclasses, so groups, `param_groups[i]["lr"]` and
torch.optim.lr_scheduler work unchanged.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlops.optim import AdamW, AdamWQAT, QATLinear, split_params

torch.manual_seed(0)
vocab, dim, steps, warmup = 1000, 128, 400, 40


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        nn.init.normal_(self.emb.weight, std=0.02)  # std=1 (the default) gives huge tied-head logits
        self.norm = nn.LayerNorm(dim)
        self.up = QATLinear(dim, 4 * dim, grad_dtype=torch.float8_e5m2)
        self.down = QATLinear(4 * dim, dim, grad_dtype=torch.float8_e5m2)

    def forward(self, idx):
        h = self.emb(idx)
        h = h + self.down(F.gelu(self.up(self.norm(h))))
        return h @ self.emb.weight.t()  # tied head


model = TinyLM().cuda()
dense, qat = split_params(model)

# Dense groups: weight decay on matrices (the embedding), none on 1-D parameters (norm gains / biases).
opt = AdamW([
    {"params": [p for p in dense if p.dim() >= 2], "weight_decay": 0.1},
    {"params": [p for p in dense if p.dim() < 2], "weight_decay": 0.0},
], lr=1e-3, betas=(0.9, 0.95))

# NVFP4 groups work the same way, e.g. a smaller lr for the output projection.
# scale_z: how much evidence a 16-weight block needs before its scale moves (default 3).
opt_qat = AdamWQAT([
    {"params": [model.up.weight]},
    {"params": [model.down.weight], "lr": 5e-4},
], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, scale_z=3.0, seed=0)


# Linear warm-up then cosine decay to 10%; LambdaLR multiplies each group's own initial lr.
def warmup_cosine(step):
    if step < warmup:
        return (step + 1) / warmup
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup)))


schedulers = [torch.optim.lr_scheduler.LambdaLR(o, warmup_cosine) for o in (opt, opt_qat)]


def clip_grad_norm_all_(model, max_norm):
    """Global-norm clipping over dense and NVFP4 gradients together.

    Float32 gradients (p.grad) are scaled in place. FP8 gradients (p.grad_fp8 = (bytes, scale)) are
    clipped exactly by multiplying their decode scale: no FP8 bytes change."""
    dense, qat = split_params(model)
    sq = [p.grad.float().pow(2).sum() for p in dense + qat if p.grad is not None]
    sq += [(g8.float().pow(2).sum() * gs * gs) for g8, gs in (p.grad_fp8 for p in qat if p.grad_fp8 is not None)]
    norm = torch.stack(sq).sum().sqrt().item()
    coef = min(1.0, max_norm / (norm + 1e-6))
    if coef < 1.0:
        for p in dense + qat:
            if p.grad is not None:
                p.grad.mul_(coef)
            if getattr(p, "grad_fp8", None) is not None:
                p.grad_fp8 = (p.grad_fp8[0], p.grad_fp8[1] * coef)
    return norm


# Synthetic language: token t+1 = (a * token t + b) mod vocab, learnable by the model.
def batch(n=32, t=64):
    x = torch.randint(0, vocab, (n, 1), device="cuda")
    seq = [x]
    for _ in range(t):
        seq.append((seq[-1] * 7 + 3) % vocab)
    seq = torch.cat(seq, 1)
    return seq[:, :-1], seq[:, 1:]


for step in range(steps):
    x, y = batch()
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    loss.backward()
    gnorm = clip_grad_norm_all_(model, max_norm=1.0)
    opt.step(); opt_qat.step()
    opt.zero_grad(); opt_qat.zero_grad()
    for s in schedulers:
        s.step()
    if step % 50 == 0 or step == steps - 1:
        lrs = [g["lr"] for g in opt_qat.param_groups]
        print(f"step {step:4d}  loss {loss.item():.4f}  grad norm {gnorm:7.3f}  NVFP4 lrs {lrs[0]:.2e} / {lrs[1]:.2e}")
