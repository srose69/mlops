# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Context extension: YaRN (or Llama 3 / linear / dynamic NTK / LongRoPE) and very long positions.

    python 03_long_context_yarn.py
"""
import torch

from mlops.positional import RoPE, rope_frequencies

# a model trained at 32K, extended 4x with YaRN (the dict is transformers' rope_parameters / rope_scaling)
yarn = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
rope = RoPE(128, base=1000000.0, scaling=yarn)
print(rope)                                    # frequencies and the attention factor 0.1 ln(4) + 1

inv, att = rope_frequencies(128, 1000000.0, yarn)
print(f"highest / lowest frequency: {inv[0]:.3g} / {inv[-1]:.3g} rad per position, attention factor {att:.4f}")

# or straight from a model's config: RoPE.from_hf(AutoConfig.from_pretrained(...))

# accuracy does not degrade with the position: rotate at 10M and check against float64
q = torch.randn(1, 4, 8, 128, device="cuda")
pos = torch.arange(4, device="cuda") + 10_000_000
y, _ = rope(q, None, pos)
ang = pos.double()[:, None] * inv.cuda()[None]
c, s = (torch.cos(ang) * att)[None, :, None], (torch.sin(ang) * att)[None, :, None]
x1, x2 = q.double()[..., :64], q.double()[..., 64:]
ref = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), -1)
print(f"max error at position 10M (FP32): {(y.double() - ref).abs().max().item():.1e}")
