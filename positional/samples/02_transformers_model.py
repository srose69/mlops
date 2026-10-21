# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Put the kernel into a transformers model: the outputs stay bit for bit the same, RoPE gets faster.

    python 02_transformers_model.py          (needs transformers)
"""
import time

import torch
from transformers import LlamaConfig, LlamaModel

from mlops.positional import patch_transformers

cfg = LlamaConfig(hidden_size=512, num_attention_heads=8, num_key_value_heads=2, intermediate_size=1024,
                  num_hidden_layers=4, vocab_size=1000, max_position_embeddings=8192)
model = LlamaModel(cfg).cuda().to(torch.bfloat16).eval()
ids = torch.randint(0, 1000, (1, 2048), device="cuda")


def run():
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        out = model(ids).last_hidden_state
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1e3


run()
ref, t_ref = run()
print("patched:", patch_transformers("llama"))   # replaces apply_rotary_pos_emb in modeling_llama
out, t_new = run()
print(f"identical outputs: {torch.equal(ref, out)}    forward {t_ref:.1f} ms -> {t_new:.1f} ms")
