#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Capture the q / k / v every attention layer of a transformers model sees (after its norms and RoPE, k / v before
the GQA repeat), on one text, for bench/fbla_vs_sdpa.py --qkv. The dump is an optional local benchmark input;
it is intentionally ignored by git and is not part of the repository.

    python bench/capture_qkv.py MODEL_DIR [TEXT_FILE ...] --tokens 2048 --out qkv.pt
"""
import argparse

import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

from fbla_lm import corpus

CAPTURED = []


def capture(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    CAPTURED.append((module.layer_idx, query[0].bfloat16().cpu(), key[0].bfloat16().cpu(), value[0].bfloat16().cpu()))
    g = query.shape[1] // key.shape[1]
    k, v = key.repeat_interleave(g, 1), value.repeat_interleave(g, 1)
    N, C = query.shape[2], 256                  # causal attention in query chunks (no N x N score matrix)
    out = torch.empty_like(query)
    for i in range(0, N, C):
        j = min(i + C, N)
        mask = torch.arange(j, device=query.device)[None, :] <= torch.arange(i, j, device=query.device)[:, None]
        out[:, :, i:j] = F.scaled_dot_product_attention(query[:, :, i:j], k[:, :, :j], v[:, :, :j], attn_mask=mask,
                                                        scale=scaling)
    return out.transpose(1, 2).contiguous(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("text", nargs="*", help="text files; none: the WikiText-2 test split")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    AttentionInterface.register("capture", capture)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="capture").cuda()
    ids = tok(corpus(a.text), return_tensors="pt").input_ids[:, : a.tokens].cuda()
    if ids.shape[1] < a.tokens:
        raise SystemExit(f"the text has only {ids.shape[1]} tokens")
    with torch.no_grad():
        if a.tokens <= 4096:
            loss = model(ids, labels=ids).loss.item()
        else:                                    # the LM head on every position would not fit: q / k / v only
            model.model(ids); loss = float("nan")
    CAPTURED.sort(key=lambda x: x[0])
    q = torch.stack([c[1] for c in CAPTURED]); k = torch.stack([c[2] for c in CAPTURED])
    v = torch.stack([c[3] for c in CAPTURED])
    cfg = model.config
    torch.save(dict(q=q, k=k, v=v, model=a.model, tokens=a.tokens, loss=loss,
                    scale=getattr(model.model.layers[0].self_attn, "scaling", None)), a.out)
    print(f"{a.out}: q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)} (layers, heads, tokens, head_dim); "
          f"LM loss on the text {loss:.3f}; {cfg.num_hidden_layers} layers")


if __name__ == "__main__":
    main()
