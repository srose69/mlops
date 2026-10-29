# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""INT8 and NVFP4 outputs: the normalized rows quantized in the same kernel, as NVFP4EffMA's quantizer makes them.

    python 03_quantized_outputs.py
"""
import torch

from mlops.norm import rms_norm

rows, hidden = 16, 4096
x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
w = torch.rand(hidden, device="cuda", dtype=torch.bfloat16) + 0.5
full = rms_norm(x, w, 1e-6).y.float()

q = rms_norm(x, w, 1e-6, quant="int8")                                   # int8 [rows, hidden], float32 scale per 16
deq = (q.y.float().reshape(rows, hidden // 16, 16) * q.scales.unsqueeze(-1)).reshape(rows, hidden)
print("int8 :", tuple(q.y.shape), q.y.dtype, "scales", tuple(q.scales.shape), q.scales.dtype,
      " max error %.3g (half a step of the largest scale: %.3g)" % ((deq - full).abs().max(), q.scales.max() / 2))

g = torch.tensor([0.02], device="cuda")                                  # the decode-factor global of the NVFP4 tensor
n = rms_norm(x, w, 1e-6, quant="nvfp4", global_scale=g)                  # E2M1 nibbles + E4M3 scale per 16
print("nvfp4:", tuple(n.y.shape), n.y.dtype, "scales", tuple(n.scales.shape), n.scales.dtype)
