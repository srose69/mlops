# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Quickstart: encode a BF16 KV cache as FBLA, then decode one GQA query.

    python 01_decode.py
"""
import torch

from mlops.attention import encode, decode

k = torch.randn(1, 8, 16384, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
cache = encode(k, v, mode="3+3", capacity=32768)  # fixed strides leave room to append the next 16K tokens

q = torch.randn(1, 16, 128, device="cuda", dtype=torch.bfloat16)
out = decode(q, cache)                            # [1, 16, 128], default W_j > 3e-4
print(out.shape)
