#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""KV-cache quantization as runtimes do it, against FBLA formats: K logit error per layer (bench/capture_qkv.py
tensors) and the model's LM loss with every attention layer reading the quantized cache.

* vLLM fp8        float8_e4m3fn, one scale per tensor (amax / 448), K and V
* llama.cpp q8_0  blocks of 32 values along head_dim (within a token), d = amax / 127, int8
* llama.cpp q4_0  blocks of 32 along head_dim, d = (value of largest |x|) / -8, codes 0..15 - 8
* llama.cpp q4_1  blocks of 32 along head_dim, min + d = (max - min) / 15, codes 0..15
* KIVI 4 / 2 bit  K per channel (groups of 32 tokens), V per token (groups of 32 channels), min + step; the
                  fp16 window of recent tokens is left out (every token quantized)
* FBLA            mu + step per channel per block of 16 tokens (two bf16 per channel per block), 4-bit codes

    python bench/kv_native.py QKV_FILE MODEL_DIR TEXT_FILE [TEXT_FILE ...]
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))


torch.backends.cuda.matmul.allow_tf32 = False


def fp8_tensor(x):
    s = x.float().abs().amax() / 448.0
    return (x.float() / s).to(torch.float8_e4m3fn).float() * s


def _blocks_hd(x, n=32):  # [..., D] -> [..., D/n, n]
    return x.float().reshape(*x.shape[:-1], x.shape[-1] // n, n)


def q8_0(x):
    b = _blocks_hd(x); d = (b.abs().amax(-1, keepdim=True) / 127).half().float()
    return (torch.round(b / d.clamp_min(1e-30)).clamp(-127, 127) * d).reshape(x.shape)


def q4_0(x):
    b = _blocks_hd(x)
    i = b.abs().argmax(-1, keepdim=True)
    d = (torch.gather(b, -1, i) / -8).half().float()
    q = torch.round(b / torch.where(d == 0, torch.ones_like(d), d) + 8).clamp(0, 15)
    return ((q - 8) * d).reshape(x.shape)


def q4_1(x):
    b = _blocks_hd(x)
    lo, hi = b.amin(-1, keepdim=True), b.amax(-1, keepdim=True)
    d, m = ((hi - lo) / 15).half().float(), lo.half().float()
    q = torch.round((b - m) / d.clamp_min(1e-30)).clamp(0, 15)
    return (q * d + m).reshape(x.shape)


def _asym(x, dim, bits):  # min + step along dim (groups already arranged on that dim)
    lo, hi = x.amin(dim, keepdim=True), x.amax(dim, keepdim=True)
    L = 2 ** bits - 1
    d, m = ((hi - lo) / L).half().float(), lo.half().float()
    return torch.round((x - m) / d.clamp_min(1e-30)).clamp(0, L) * d + m


def kivi_k(bits, g=32):  # x [..., N, D]: per channel over groups of g tokens
    def f(x):
        *lead, N, D = x.shape
        return _asym(x.float().reshape(*lead, N // g, g, D), -2, bits).reshape(x.shape)
    return f


def kivi_v(bits, g=32):  # per token over groups of g channels
    def f(x):
        b = _blocks_hd(x, g)
        return _asym(b, -1, bits).reshape(x.shape)
    return f


def fbla_k(x):  # mu + step per channel per block of 16 tokens
    *lead, N, D = x.shape
    xb = x.float().reshape(*lead, N // 16, 16, D)
    lo, hi = xb.amin(-2, keepdim=True), xb.amax(-2, keepdim=True)
    mu = ((lo + hi) / 2).bfloat16().float(); st = ((hi - lo) / 14).bfloat16().float()
    return (mu + torch.round((xb - mu) / st.clamp_min(1e-30)).clamp(-7, 7) * st).reshape(x.shape)


SCHEMES = [  # name, K, V, bits per element (K, V)
    ("control: fp32 attention, no quantization", lambda x: x.float(), lambda x: x.float(), "16 / 16"),
    ("vLLM fp8 e4m3, per tensor", fp8_tensor, fp8_tensor, "8 / 8"),
    ("llama.cpp q8_0", q8_0, q8_0, "8.5 / 8.5"),
    ("llama.cpp q4_1", q4_1, q4_1, "5.0 / 5.0"),
    ("llama.cpp q4_0", q4_0, q4_0, "4.5 / 4.5"),
    ("KIVI 4-bit", kivi_k(4), kivi_v(4), "5.0 / 5.0"),
    ("KIVI 2-bit", kivi_k(2), kivi_v(2), "3.0 / 3.0"),
    ("FBLA K: mu + step / channel / block-16, V: KIVI 4-bit", fbla_k, kivi_v(4), "6.0 / 5.0"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qkv"); ap.add_argument("model"); ap.add_argument("text", nargs="+")
    ap.add_argument("--layers", default="1,5,13,23,27"); ap.add_argument("--lm-only", action="store_true"); ap.add_argument("--windows", type=int, default=12); ap.add_argument("--win", type=int, default=1024)
    a = ap.parse_args()
    d = torch.load(a.qkv)
    layers = [int(x) for x in a.layers.split(",")]
    if a.lm_only:
        layers = []
    print("K logit error RMS (q . dk * scale, causal), per layer\n")
    from termtable import Table
    width = max(len(s[0]) for s in SCHEMES)
    t = Table([("scheme", width, "<"), ("K / V bits", 10, ">")] + [(f"L{L}", 6, ">") for L in layers])
    for name, fk, _, bits in SCHEMES:
        cells = []
        for L in layers:
            q, k = d["q"][L].cuda().double(), d["k"][L].cuda().double()
            g = q.shape[0] // k.shape[0]; H, N, D = k.shape
            dk = fk(k.float()).double() - k
            e = (q.view(H, g, N, D) @ dk[:, None].transpose(-1, -2)) * d["scale"]
            m = torch.tril(torch.ones(N, N, device="cuda", dtype=torch.bool)).expand_as(e)
            cells.append(f"{e[m].pow(2).mean().sqrt().item():.3f}")
        t.row(name, bits, *cells)

    cur = {}

    def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        g = query.shape[1] // key.shape[1]
        k, v = key.repeat_interleave(g, 1), value.repeat_interleave(g, 1)
        if cur["k"] is None:
            o = F.scaled_dot_product_attention(query, k, v, is_causal=True, scale=scaling)
        else:
            o = F.scaled_dot_product_attention(query.float(), cur["k"](k), cur["v"](v), is_causal=True, scale=scaling)
        return o.to(query.dtype).transpose(1, 2).contiguous(), None
    AttentionInterface.register("kv_native", attn)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="kv_native").cuda()
    ids = tok("\n\n".join(open(f).read() for f in a.text), return_tensors="pt").input_ids
    wins = [ids[:, i:i + a.win].cuda() for i in range(0, ids.shape[1] - a.win + 1, a.win)][: a.windows]
    print(f"\nLM loss, {len(wins)} windows of {a.win} tokens: delta vs bf16 SDPA, mean (min .. max over windows)\n")
    t = Table([("scheme", width, "<"), ("K / V bits", 10, ">"), ("delta loss (mean, min .. max)", 30, ">")])
    base = None
    for name, fk, fv, bits in [("bf16 SDPA", None, None, "16 / 16")] + SCHEMES:
        cur["k"], cur["v"] = fk, fv
        with torch.no_grad():
            ls = torch.tensor([model(w, labels=w).loss.item() for w in wins])
        base = ls if base is None else base
        dl = ls - base
        t.row(name, bits, f"{dl.mean():+.4f} ({dl.min():+.4f} .. {dl.max():+.4f})")


if __name__ == "__main__":
    main()
