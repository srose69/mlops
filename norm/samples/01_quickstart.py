# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Quickstart: RMSNorm with a residual add, in Hugging Face's order of roundings, against PyTorch.

    python 01_quickstart.py
"""
import torch

from mlops.norm import RMSNorm, rms_norm

rows, hidden = 16, 4096
x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
h = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)      # the residual stream
w = torch.rand(hidden, device="cuda", dtype=torch.bfloat16) + 0.5

out = rms_norm(x, w, 1e-6, residual=h)                                   # out.y, out.residual = round(x + h)
print("y:", tuple(out.y.shape), out.y.dtype, " residual:", tuple(out.residual.shape))

# the reference: transformers' LlamaRMSNorm on v = x + h
v = (x + h)
f = v.float()
ref = w * (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + 1e-6)).to(v.dtype)
print("residual stream equal:", torch.equal(out.residual, v))
print("y vs LlamaRMSNorm: max abs diff %.3g, equal elements %.2f%%" % ((out.y.float() - ref.float()).abs().max(), 100 * (out.y == ref).float().mean()))

# an FP32 residual stream with BF16 weights: x and the residual in float32, y in bfloat16
out32 = rms_norm(x.float(), w, 1e-6, residual=h.float())
print("fp32 stream ->", out32.y.dtype, out32.residual.dtype)

print("module:", RMSNorm(hidden)(x).y.shape)
