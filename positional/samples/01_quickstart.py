# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Quickstart: rotate q and k of an attention layer with RoPE.

    python 01_quickstart.py
"""
import torch

from mlops.positional import RoPE

B, S, Hq, Hk, D = 2, 16, 8, 2, 64          # batch, sequence, q heads, kv heads (GQA), head_dim
q = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, S, Hk, D, device="cuda", dtype=torch.bfloat16)

rope = RoPE(D, base=10000.0)                # pairs (i, i + D/2) as in Llama; interleaved=True for GPT-J pairs

q_rot, k_rot = rope(q, k)                   # positions 0..S-1, layout [B, S, H, D]
print("rotated:", tuple(q_rot.shape), tuple(k_rot.shape), q_rot.dtype)

# explicit positions: [S] shared by the batch, or [B, S] (packed sequences, left padding, ...)
pos = torch.arange(S, device="cuda") + 100
q_rot2, _ = rope(q, k, pos)
print("same as offset=100:", torch.equal(q_rot2, rope(q, k, offset=100)[0]))

# the same tensor in transformers' layout [B, H, S, D] gives the same values
q_bhsd, k_bhsd = rope(q.transpose(1, 2), k.transpose(1, 2), pos, layout="bhsd")
print("layout bhsd == bshd:", torch.equal(q_bhsd.transpose(1, 2), q_rot2))

# relative positions: <q at m, k at n> depends on m - n only (FP32, so the dtype's rounding does not blur it)
qs, ks = q[:1, :1, :1].float(), k[:1, :1, :1].float()
dot = lambda m, n: (rope(qs, None, torch.tensor([m], device="cuda"))[0] *
                    rope(ks, None, torch.tensor([n], device="cuda"))[0]).sum().item()
print(f"<q@10, k@3> = {dot(10, 3):.6f}   <q@1010, k@1003> = {dot(1010, 1003):.6f}   <q@10, k@4> = {dot(10, 4):.6f}")
