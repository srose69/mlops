# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Quickstart: a small MLP whose linear layers are NVFP4, trained with AdamW + AdamWQAT.

    python 01_quickstart.py

The two optimizers split a model between them:
  * AdamW     dense parameters (norms, embeddings, biases, ordinary nn.Linear, ...)
  * AdamWQAT  the NVFP4 weights of QATLinear layers, nothing else
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlops.optim import AdamW, AdamWQAT, QATLinear, split_params

torch.manual_seed(0)
dim, hidden = 256, 1024  # QATLinear needs in_features % 64 == 0

# QATLinear: bias-free linear layer whose weight is stored as NVFP4 (4-bit codes + UE5M3 block scales).
# grad_dtype=float8_e5m2: the weight gradient reaches the optimizer as FP8 (smaller, what the kernel reads).
model = nn.Sequential(
    nn.LayerNorm(dim),
    QATLinear(dim, hidden, grad_dtype=torch.float8_e5m2),
    nn.GELU(),
    QATLinear(hidden, dim, grad_dtype=torch.float8_e5m2),
).cuda()
print(model)

# split_params: (dense parameters, NVFP4 parameters). Passing either list to the wrong optimizer raises.
dense, qat = split_params(model)
opt = AdamW(dense, lr=3e-3, weight_decay=0.0)
opt_qat = AdamWQAT(qat, lr=3e-3, weight_decay=0.1)

# Synthetic task: regress a fixed random nonlinear map.
teacher = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, dim)).cuda().requires_grad_(False)
with torch.no_grad():
    teacher[2].weight.mul_(10)  # make the targets O(1)

for step in range(1, 501):
    x = torch.randn(256, dim, device="cuda")
    loss = F.mse_loss(model(x), teacher(x))

    loss.backward()
    opt.step()
    opt_qat.step()
    opt.zero_grad()
    opt_qat.zero_grad()  # also clears the FP8 gradients (weight.grad_fp8)

    if step == 1 or step % 100 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")

# Memory per NVFP4 weight: codes 0.5 B + scales and optimizer state, all in the layer's buffers.
lin = model[1]
nbytes = sum(b.numel() * b.element_size() for b in lin.buffers())
print(f"{lin}: {nbytes / lin.weight.numel():.2f} B per weight (weight + optimizer state)")
