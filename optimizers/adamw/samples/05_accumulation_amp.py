"""Gradient accumulation, bfloat16 autocast, and the activation formats of QATLinear (`act`).

    python 05_accumulation_amp.py

* Accumulation: call backward() several times before step(), as usual. FP8 gradients (grad_fp8) are
  summed in float32 and re-encoded once per backward, so the sum is not limited by FP8 range.
* Autocast: QATLinear's GEMM takes activations in their own dtype (float32 / bfloat16 / float16) and
  returns that dtype, so it works inside an autocast region; autocast casts the dense layers around it.
* act: how activations enter the NVFP4 GEMM. "exact" (default, W4A16) keeps them as they are; "int8"
  (W4A8, int8 with an FP32 scale per 16 values) is about 2x faster on GPUs with DP4A (sm_61+); "nvfp4"
  (W4A4) is the smallest; "auto" is exact for up to 2 rows, int8 beyond. The backward uses the
  activations the GEMM actually read, and keeps them in that format (less activation memory).
* Dense parameters may be bfloat16 themselves: mlops_adamw.AdamW keeps FP32 moments for them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import mlops_adamw

dim, micro_batches, micro_size = 256, 4, 64
target = torch.randn(dim, dim, device="cuda", generator=torch.Generator("cuda").manual_seed(1)) / dim ** 0.5


def build(act):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.LayerNorm(dim),
        mlops_adamw.QATLinear(dim, 4 * dim, grad_dtype=torch.float8_e5m2, act=act),
        nn.GELU(),
        mlops_adamw.QATLinear(4 * dim, dim, grad_dtype=torch.float8_e5m2, act=act),
    ).cuda()
    model[0].to(torch.bfloat16)  # a bfloat16 dense parameter, to show AdamW handles it
    dense, qat = mlops_adamw.split_params(model)
    return model, mlops_adamw.AdamW(dense, lr=2e-3), mlops_adamw.AdamWQAT(qat, lr=2e-3, weight_decay=0.1)


for act in ("exact", "int8", "nvfp4"):
    model, opt, opt_qat = build(act)
    for step in range(1, 301):
        total = 0.0
        for _ in range(micro_batches):
            x = torch.randn(micro_size, dim, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x.bfloat16())
            loss = F.mse_loss(out.float(), x @ target) / micro_batches  # mean over the full batch
            loss.backward()                                             # accumulates into grad / grad_fp8
            total += loss.item()
        opt.step(); opt_qat.step()
        opt.zero_grad(); opt_qat.zero_grad()
    print(f"act={act:5s}  loss after 300 steps {total:.4f}  (batch {micro_batches} x {micro_size})")
