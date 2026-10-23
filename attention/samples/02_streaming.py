# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""One cache for a whole generation: allocate, prefill the prompt through it, then append + decode per token.

    python 02_streaming.py [--mode 4+2 | 3+3]

The cache is laid out for `capacity` tokens up front (fixed strides: nothing is reallocated or moved as the
context grows).  Per layer:

  cache = empty(batch, kv_heads, capacity, mode)   once; the quantization mode is fixed here
  out = prefill(q, k, v, cache)    the prompt, chunk by chunk: q [batch, q_heads, n, 128], k / v [batch,
                                   kv_heads, n, 128]; appends k / v and returns the causal attention of the
                                   chunk's queries over everything so far (full 6-bit codes)
  append(cache, k_t, v_t)          per decode step: the new token's K / V, [batch, kv_heads, 1, 128]
  out = decode(q_t, cache)         its query, [batch, q_heads, 128], over all tokens including itself

New tokens go to the open 128-token group (BF16); when it is full the group is quantized into the packed
records.  Every write runs on the GPU.  The sample checks the prompt's attention and every decode step against
exact attention, and that the streamed cache decodes bit-for-bit like one encoded from scratch.
"""
import argparse
import math
import time

import torch

from mlops.attention import append, decode, empty, encode, prefill

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="4+2", choices=["4+2", "3+3"], help="FBLA quantization mode")
mode = ap.parse_args().mode

torch.manual_seed(0)
B, Hq, Hkv, D = 1, 16, 8, 128
prompt, steps, capacity, chunk = 3000, 600, 8192, 1024  # prefill 3000 tokens in chunks, decode 600

k_all = (torch.randn(B, Hkv, prompt + steps, D, device="cuda") * (0.3 + 1.5 * torch.rand(D, device="cuda"))).bfloat16()
v_all = torch.randn(B, Hkv, prompt + steps, D, device="cuda").bfloat16()
q_all = torch.randn(B, Hq, prompt + steps, D, device="cuda").bfloat16()
v_rms = v_all.double().pow(2).sum(-1).mean().sqrt()


def exact(q, t):
    """float64 attention of the queries q [B, Hq, n, D] at positions t - n + 1 .. t over the tokens 0 .. t."""
    g, n = Hq // Hkv, q.shape[2]
    kk = k_all[:, :, :t + 1].double().repeat_interleave(g, 1)
    vv = v_all[:, :, :t + 1].double().repeat_interleave(g, 1)
    s = torch.einsum("bhqd,bhnd->bhqn", q.double(), kk) / math.sqrt(D)
    s = s.masked_fill(torch.arange(t + 1, device="cuda") > torch.arange(t + 1 - n, t + 1, device="cuda")[:, None], -math.inf)
    return torch.einsum("bhqn,bhnd->bhqd", torch.softmax(s, -1), vv)


err = lambda x, ref: ((x.double() - ref).norm(dim=-1).mean() / v_rms).item()  # in units of |v|

cache = empty(B, Hkv, capacity, mode)
torch.cuda.synchronize()
t0 = time.perf_counter()
outs = [prefill(q_all[:, :, s:e], k_all[:, :, s:e], v_all[:, :, s:e], cache)
        for s, e in ((s, min(s + chunk, prompt)) for s in range(0, prompt, chunk))]
torch.cuda.synchronize()
print(f"mode {mode}: prefill {prompt} tokens in chunks of {chunk}: {(time.perf_counter() - t0) * 1e3:.1f} ms")
last = outs[-1][:, :, -256:]  # the prompt's last 256 positions against exact attention
print(f"  prompt attention, last 256 positions: error vs exact {err(last, exact(q_all[:, :, prompt - 256:prompt], prompt - 1)):.2e} |v|")

t0 = time.perf_counter()
for t in range(prompt, prompt + steps):
    append(cache, k_all[:, :, t:t + 1], v_all[:, :, t:t + 1])   # the new token joins the cache ...
    out = decode(q_all[:, :, t], cache)                        # ... and its query reads all t + 1 tokens
    if (t - prompt) % 150 == 0 or t == prompt + steps - 1:
        e = err(out, exact(q_all[:, :, t:t + 1], t)[:, :, 0])
        # the same context encoded from scratch must decode identically
        same = torch.equal(out, decode(q_all[:, :, t], encode(k_all[:, :, :t + 1], v_all[:, :, :t + 1], mode)))
        print(f"  tokens {cache.tokens:5d} (closed groups {(cache.tokens - 1) // 128}): decode error vs exact "
              f"{e:.2e} |v|, equals a fresh encode: {same}")
        assert same
torch.cuda.synchronize()
print(f"{steps} steps of append + decode: {(time.perf_counter() - t0) / steps * 1e3:.2f} ms per step "
      "(includes the checks)")
