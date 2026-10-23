# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Quickstart: encode a BF16 KV cache as FBLA, decode one query per head, check it against exact attention.

    python 01_decode.py [--mode 4+2 | 3+3]

What FBLA replaces: the KV cache and the attention over it.  This sample is the decode side: a cache made from
dense K / V and one query per head.  The prompt can go through FBLA as well (prefill: 02_streaming.py).

  encode(k, v, mode)  [batch, kv_heads, tokens, 128] BF16 K / V -> FBLACache (6.25 bits per element stored)
  decode(q, cache)    [batch, q_heads, 128] BF16, one query per head -> [batch, q_heads, 128] BF16

The quantization mode is chosen once, in encode (or empty); the cache records it, and append / prefill / decode
follow it:
  "4+2"  4-bit coarse plane + 2-bit fine plane, default refinement threshold tau = 1e-3
  "3+3"  3-bit coarse plane + 3-bit fine plane, default tau = 3e-4
Both store the same 6-bit code; the split decides how much every decode step reads (the coarse plane) and how
much only the refined blocks read (the fine plane).

q_heads / kv_heads must be 1, 2 or 4 (GQA), head dim 128.  q is the query after the projection, the q / k norm
and RoPE, exactly what goes into softmax(q k^T * scale) v.
"""
import argparse
import math

import torch

from mlops.attention import decode, encode

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="4+2", choices=["4+2", "3+3"], help="FBLA quantization mode")
mode = ap.parse_args().mode

torch.manual_seed(0)
B, Hq, Hkv, N, D = 1, 16, 8, 16384, 128  # Qwen3-0.6B geometry: GQA 2

# Synthetic K / V: uneven channel scales and a common offset in K, as in real keys.  Random data is a harsh
# case for any KV quantization (real keys are far more structured); 03_transformers.py measures a real model.
k = (torch.randn(B, Hkv, N, D, device="cuda") * (0.3 + 1.5 * torch.rand(D, device="cuda"))
     + torch.randn(D, device="cuda")).bfloat16()
v = torch.randn(B, Hkv, N, D, device="cuda").bfloat16()
q = torch.randn(B, Hq, D, device="cuda").bfloat16()  # one decode query per q head

cache = encode(k, v, mode=mode)  # the mode is fixed here; capacity=... leaves room for append (02_streaming.py)
out = decode(q, cache)           # [1, 16, 128] BF16; tau = 0 (default) selects the mode's threshold
print(f"mode {mode}: q {tuple(q.shape)} -> out {tuple(out.shape)} {out.dtype}")


def exact(q, k, v):
    """float64 softmax(q k^T / sqrt(d)) v with GQA."""
    g = q.shape[1] // k.shape[1]
    kk, vv = k.double().repeat_interleave(g, 1), v.double().repeat_interleave(g, 1)
    s = torch.einsum("bhd,bhnd->bhn", q.double(), kk) / math.sqrt(D)
    return torch.einsum("bhn,bhnd->bhd", torch.softmax(s, -1), vv)


# Error in units of the values' size: what the layer adds to the residual stream is W_O o, so an error of o
# matters relative to the vectors it mixes, not to |o| (a diffuse attention gives a small o).
ref = exact(q, k, v)
v_rms = v.double().pow(2).sum(-1).mean().sqrt()
err = lambda x: ((x.double() - ref).norm(dim=-1).mean() / v_rms).item()
print(f"\nerror |out - exact| / |v|, N = {N}  (BF16 rounding of the exact output: {err(ref.bfloat16()):.1e})")
# tau trades accuracy for traffic: a larger tau refines fewer blocks (1e30: coarse plane only), a smaller one more
# (1e-30: every block, the full 6-bit code)
for tau, what in ((0.0, "default tau"), (1e-4, "tau 1e-4"), (1e30, "coarse plane only"), (1e-30, "all refined")):
    print(f"  {mode}, {what:18s} {err(decode(q, cache, tau=tau)):.1e}")

# Storage: packed records + metadata + the two BF16 groups (attention sink, open group), against BF16 K + V.
stored = sum(t.numel() * t.element_size()
             for t in (cache.coarse, cache.fine, cache.meta, cache.sink_k, cache.sink_v, cache.open_k, cache.open_v))
print(f"\ncache {stored / 2**20:.1f} MiB vs BF16 K + V {2 * k.numel() * 2 / 2**20:.1f} MiB: "
      f"{8 * stored / (2 * k.numel()):.2f} bits per element (6.25 + the two BF16 groups)")
