# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Inference: prefill, then decode one token at a time against a KV cache, RoPE in place.

    python 05_inference_decode.py
"""
import torch

from mlops.positional import RoPE

B, Hq, Hk, D, MAX = 4, 8, 2, 64, 1024
rope = RoPE(D, base=500000.0)
k_cache = torch.zeros(B, MAX, Hk, D, device="cuda", dtype=torch.float16)

# prefill: sequences of different lengths, left-padded; positions count real tokens only
lengths = torch.tensor([10, 7, 12, 3], device="cuda")
S = int(lengths.max())
pos = (torch.arange(S, device="cuda")[None] - (S - lengths)[:, None]).clamp_min(0)   # [B, S]
q = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, S, Hk, D, device="cuda", dtype=torch.float16)
with torch.no_grad():
    rope(q, k, pos, inplace=True)                   # no copies: q and k rotated where they are
k_cache[:, :S] = k
print("prefill positions per sequence:", pos[:, -1].tolist())

# decode: one new token per sequence, at its own next position
next_pos = lengths.clone()
for step in range(3):
    q1 = torch.randn(B, 1, Hq, D, device="cuda", dtype=torch.float16)
    k1 = torch.randn(B, 1, Hk, D, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        rope(q1, k1, next_pos[:, None], inplace=True)
    k_cache[:, S + step] = k1[:, 0]
    print(f"decode step {step}: positions {next_pos.tolist()}")
    next_pos += 1
