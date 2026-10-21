# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Low-level API: the kernels without torch.optim, and the bit-identical PyTorch reference.

    python 07_low_level.py

Use this when you manage the training loop / state yourself (custom layers, another framework on top),
or to read and modify the algorithm in PyTorch (mlops.optim.reference).
"""
import torch

from mlops.optim import functional, reference
import nvfp4effma

torch.manual_seed(0)
rows, cols = 256, 512

# ---- functional.adamw: one fused AdamW step, in place -------------------------------------------------
param = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
exp_avg = torch.zeros(rows, cols, device="cuda")     # moments are always float32
exp_avg_sq = torch.zeros(rows, cols, device="cuda")
for t in range(1, 11):                               # `step` starts at 1 (bias correction)
    grad = torch.randn_like(param)
    functional.adamw(param, grad, exp_avg, exp_avg_sq, lr=1e-3, step=t, weight_decay=0.1)
print("adamw: 10 steps on a bfloat16 tensor done")

# ---- QATState: one NVFP4 weight + its mlops_adamw_qat state -----------------------------------------------
w0 = torch.randn(rows, cols, device="cuda") * 0.02
st = functional.QATState(w0)           # quantizes w0; w0 itself is not kept
print("NVFP4 weight:", type(st.weight).__name__, tuple(st.weight.shape),
      f"| optimizer state {st.state.numel():,} B ({st.state.numel() / (rows * cols):.2f} B / weight)")

# Forward on the NVFP4 tensor; the gradient the step wants is dL/d(st.weight.dequantize()).
x = torch.randn(64, cols, device="cuda")
target = torch.randn(64, rows, device="cuda")
y = nvfp4effma.linear(x, st.weight, act="exact")
grad_q = 2 * (y - target).t() @ x / y.numel()       # d mean((x q^T - target)^2) / dq

# Gradients go in as FP8 (amax-scaled) with the factor that decodes them, or as float32 with grad_scale=1.
g8, gs = functional.fp8_grad(grad_q, torch.float8_e5m2)
before = st.weight.dequantize(torch.float32)
st.step(g8, lr=1e-3, step=1, weight_decay=0.1, grad_scale=gs, seed=123)   # rewrites st.weight in place
moved = (st.weight.dequantize(torch.float32) != before).float().mean().item()
print(f"QATState.step: {moved:.1%} of the NVFP4 codes changed after one step")

# master(): the continuous weight the NVFP4 codes stochastically round (never stored, decoded on demand).
print(f"master vs NVFP4 weight: max |diff| {(st.master() - st.weight.dequantize(torch.float32)).abs().max():.2e}")

# ---- reference.step: the same step in PyTorch, bit for bit ----------------------------------------------
# Clone the NVFP4 tensor and the state, step both ways with the same seed, compare every byte.
q_ref = nvfp4effma.NVFP4Tensor(st.weight.data.clone(), st.weight.scales.view(torch.uint8).clone(),
                               amax=st.weight.amax.clone(), scale_max=st.weight.scale_max)
state_ref = st.state.clone()
g8, gs = functional.fp8_grad(grad_q, torch.float8_e5m2)
st.step(g8, lr=1e-3, step=2, weight_decay=0.1, grad_scale=gs, seed=123)
reference.step(q_ref, state_ref, g8, lr=1e-3, step=2, weight_decay=0.1, grad_scale=gs, seed=123)

carve = lambda s: reference.carve(s, rows, cols)   # named views of the state buffer
a, b = carve(st.state), carve(state_ref)
same = torch.equal(st.weight.data, q_ref.data) and all(torch.equal(a[k], b[k]) for k in a)
print(f"reference.step == kernel (codes + state arrays {sorted(a)}): {same}")
