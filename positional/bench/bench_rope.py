#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""RoPE of q and k: transformers' eager path (its rotary module's cos / sin + apply_rotary_pos_emb) against
mlops.positional -- apply_rotary_pos_emb on the same tables (a drop-in, bit for bit) and RoPE from the positions
(no tables). Llama-like heads (32 q / 8 kv, head_dim 128, BF16, transformers' [B, H, S, D] layout); prefill,
decode (one token per sequence) and a training step (forward + backward). Median time over EFFMA_BENCH_REPS
calls; DRAM throughput of q and k read and written once.

    python bench/bench_rope.py          (with mlops-positional and transformers installed)
"""
import os
import statistics

import torch
from transformers import LlamaConfig
from transformers.models.llama import modeling_llama

from mlops.positional import RoPE, apply_rotary_pos_emb

REPS = int(os.environ.get("EFFMA_BENCH_REPS", "200"))
dev = "cuda"


def timed(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = []
    for _ in range(REPS):
        a.record(); fn(); b.record(); b.synchronize()
        t.append(a.elapsed_time(b))
    return statistics.median(t)


def main():
    Hq, Hk, D = 32, 8, 128
    cfg = LlamaConfig(hidden_size=Hq * D, num_attention_heads=Hq, num_key_value_heads=Hk, head_dim=D,
                      max_position_embeddings=131072, rope_parameters={"rope_type": "llama3", "rope_theta": 500000.0,
                      "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192})
    rot = modeling_llama.LlamaRotaryEmbedding(cfg, device=dev)
    rope = RoPE.from_hf(cfg)
    print(f"{torch.cuda.get_device_name()}: RoPE of q [B, {Hq}, S, {D}] and k [B, {Hk}, S, {D}], BF16, llama3 scaling; "
          f"median of {REPS}\n")
    print(f"{'':44s} {'transformers':>13s} {'drop-in':>17s} {'RoPE(positions)':>23s}")
    for name, B, S, train in (("prefill B=1 S=4096", 1, 4096, False), ("prefill B=8 S=512", 8, 512, False),
                              ("decode B=1 (1 token)", 1, 1, False), ("decode B=64 (1 token each)", 64, 1, False),
                              ("training B=4 S=2048, forward + backward", 4, 2048, True)):
        q = torch.randn(B, Hq, S, D, device=dev, dtype=torch.bfloat16, requires_grad=train)
        k = torch.randn(B, Hk, S, D, device=dev, dtype=torch.bfloat16, requires_grad=train)
        pos = torch.arange(S, device=dev)[None].expand(B, S) + 1000
        gq, gk = torch.randn_like(q), torch.randn_like(k)

        def hf():
            cos, sin = rot(q, pos)
            y = modeling_llama.apply_rotary_pos_emb(q, k, cos, sin)
            if train:
                torch.autograd.backward(y, (gq, gk))

        def dropin():
            cos, sin = rot(q, pos)
            y = apply_rotary_pos_emb(q, k, cos, sin)
            if train:
                torch.autograd.backward(y, (gq, gk))

        def ours():
            y = rope(q, k, pos, layout="bhsd")
            if train:
                torch.autograd.backward(y, (gq, gk))

        t_hf, t_d, t_o = timed(hf), timed(dropin), timed(ours)
        gbs = (2 if train else 1) * 2 * 2 * (q.numel() + k.numel()) / (t_o * 1e-3) / 1e9
        print(f"{name:44s} {t_hf * 1e3:9.1f} us {t_d * 1e3:8.1f} us {t_hf / t_d:5.2f}x {t_o * 1e3:8.1f} us "
              f"{t_hf / t_o:5.2f}x  ({gbs:.0f} GB/s)", flush=True)


if __name__ == "__main__":
    main()
