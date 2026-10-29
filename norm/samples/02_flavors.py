# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""The flavors of current models, each against PyTorch: Gemma's 1 + weight, LayerNorm with a bias, Mamba-2's gated norm over groups.

    python 02_flavors.py
"""
import torch
import torch.nn.functional as F

from mlops.norm import layer_norm, norm, rms_norm

rows, hidden = 8, 2048
x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
w = torch.rand(hidden, device="cuda", dtype=torch.bfloat16) - 0.5
b = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) * 0.1
z = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)


def report(name, got, ref):
    d = (got.float() - ref.float()).abs()
    print("%-36s equal %.1f %%, max relative difference %.3g (bf16 step: 2^-8)" % (name, 100 * (got == ref).float().mean(), (d / ref.float().abs().clamp_min(1e-3)).max()))


f = x.float()
r = torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + 1e-6)
report("Gemma: rms, 1 + weight", rms_norm(x, w, 1e-6, one_plus=True).y, (f * r * (1 + w.float())).to(x.dtype))
report("GPT / Falcon: layernorm + bias", layer_norm(x, w, b).y, F.layer_norm(f, (hidden,), w.float(), b.float(), 1e-5).to(x.dtype))
report("Qwen3-Next: rms, gate after", norm(x, w, gate=z, one_plus=True).y, (f * r * (1 + w.float()) * F.silu(z.float())).to(x.dtype))

g = (f * F.silu(z.float())).reshape(rows, 4, -1)                          # Mamba-2: gate before, 4 groups
ref = (g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + 1e-6)).reshape(rows, hidden) * w.float()
report("Mamba-2: rms, gate before, groups 4", norm(x, w, gate=z, gate_mode="before", groups=4).y, ref.to(x.dtype))
