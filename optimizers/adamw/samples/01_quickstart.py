"""Quickstart: a small MLP whose linear layers are NVFP4, trained with AdamW + AdamWQAT.

    python 01_quickstart.py

The two optimizers split a model between them:
  * mlops_adamw.AdamW     dense parameters (norms, embeddings, biases, ordinary nn.Linear, ...)
  * mlops_adamw.AdamWQAT  the NVFP4 weights of mlops_adamw.QATLinear layers, nothing else
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import mlops_adamw

torch.manual_seed(0)
dim, hidden = 256, 1024  # QATLinear needs in_features % 64 == 0

# QATLinear: bias-free linear layer whose weight is stored as NVFP4 (4-bit codes + E4M3 block scales).
# grad_dtype=float8_e5m2: the weight gradient reaches the optimizer as FP8 (smaller, what the kernel reads).
model = nn.Sequential(
    nn.LayerNorm(dim),
    mlops_adamw.QATLinear(dim, hidden, grad_dtype=torch.float8_e5m2),
    nn.GELU(),
    mlops_adamw.QATLinear(hidden, dim, grad_dtype=torch.float8_e5m2),
).cuda()
print(model)

# split_params: (dense parameters, NVFP4 parameters). Passing either list to the wrong optimizer raises.
dense, qat = mlops_adamw.split_params(model)
opt = mlops_adamw.AdamW(dense, lr=3e-3, weight_decay=0.0)
opt_qat = mlops_adamw.AdamWQAT(qat, lr=3e-3, weight_decay=0.1)

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
