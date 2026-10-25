#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""End to end: a transformers model's LM loss with every attention layer replaced by FBLA (the numerical model of
python/mlops/attention/fbla_reference.py) or by a runtime KV-cache quantization, against bf16 SDPA.

Every quantized scheme runs with decode semantics inside each window: a query reads the sink group (the first 128
tokens) and its own, still open, group of 128 in bf16, every other earlier group from the quantized cache (KIVI's
own runtime keeps the last 128 tokens in fp16 the same way). Reported per scheme: the loss delta over the windows (mean, min, max),
and for FBLA the share of quantized (query, 16-token sub-block) pairs whose refinement plane was read, which gives
the bits per element actually read: coarse + 32 / G + fine * share.

With --decode T the window is a long context whose first win - T positions are computed exactly (prefill writes
the cache) and whose last T positions decode against the quantized cache; the loss is that of the last T tokens.

    python bench/fbla_lm.py MODEL_DIR [TEXT_FILE ...] [--win 2048] [--windows 13]
    python bench/fbla_lm.py MODEL_DIR [TEXT_FILE ...] --win 16384 --decode 1024 --windows 11

Without text files the text is the WikiText-2 (raw) test split (about 299K Qwen3 tokens).

The "FBLA kernel" rows run the CUDA kernels (the installed mlops.attention package): decode needs --decode; the
prefill row runs the last T queries (or, without --decode, the whole window) through mlops_fbla_prefill, on tensor
cores where the device has them (MLOPS_FBLA_DP4A=1: the DP4A path).
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlops.attention import append as fbla_append, decode as fbla_decode, empty as fbla_empty  # noqa: E402  (installed package)
from mlops.attention import encode as fbla_encode, prefill as fbla_prefill  # noqa: E402
from mlops.attention import fbla_reference as fb  # noqa: E402
from mlops.attention.fbla import FBLACache  # noqa: E402
import kv_native as kn  # noqa: E402
from termtable import Table  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
CUR = {}


def chunked_sdpa(q, k, v, scale, C=512):
    N = q.shape[2]
    out = torch.empty(q.shape, dtype=torch.float32 if q.dtype == torch.float32 else q.dtype, device=q.device)
    for i in range(0, N, C):
        j = min(i + C, N)
        mask = torch.arange(j, device=q.device)[None, :] <= torch.arange(i, j, device=q.device)[:, None]
        out[:, :, i:j] = F.scaled_dot_product_attention(q[:, :, i:j], k[:, :, :j], v[:, :, :j], attn_mask=mask, scale=scale)
    return out


def decode_semantics(q, k, v, fk, fv, scale, G=128):
    """Causal attention where query i reads groups 0 and i // G exact and every other group through fk / fv."""
    kq, vq = fk(k), fv(v)
    N = q.shape[2]
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for i in range(0, N, G):
        j = min(i + G, N)
        kk, vv = kq[:, :, :j].clone(), vq[:, :, :j].clone()
        kk[:, :, :G], vv[:, :, :G] = k[:, :, :G].float(), v[:, :, :G].float()
        kk[:, :, i:j], vv[:, :, i:j] = k[:, :, i:j].float(), v[:, :, i:j].float()
        mask = torch.arange(j, device=q.device)[None, :] <= torch.arange(i, j, device=q.device)[:, None]
        out[:, :, i:j] = F.scaled_dot_product_attention(q[:, :, i:j].float(), kk, vv, attn_mask=mask, scale=scale)
    return out


def kernel_decode(query, key, value, T, fmt, tau, scale, G=128):
    """The last T queries through the CUDA kernel (mlops_fbla_decode), query p against a cache of p + 1 tokens:
    the closed groups encoded once, the open group (p // G) BF16 up to p.  query [1, Hq, N, D], key / value
    [1, Hkv, N, D] -> [1, Hq, T, D] BF16."""
    N = query.shape[2]
    c = fbla_encode(key, value, fmt)
    out = []
    for p in range(N - T, N):
        g0 = p // G * G
        ok = torch.zeros_like(c.open_k); ov = torch.zeros_like(c.open_v)
        ok[:, :, :p + 1 - g0] = key[:, :, g0:p + 1]; ov[:, :, :p + 1 - g0] = value[:, :, g0:p + 1]
        cp = FBLACache(c.coarse, c.fine, c.meta, c.sink_k, c.sink_v, ok, ov, p + 1, c.capacity, c.format)
        out.append(fbla_decode(query[:, :, p].contiguous(), cp, scale=scale, tau=tau))
    return torch.stack(out, 2)


def corpus(files):
    """The text of the evaluation: `files` joined, or without any the WikiText-2 (raw) test split (Salesforce/wikitext
    from the Hugging Face hub, cached by huggingface_hub; needs pyarrow), a fixed text every run and every GPU."""
    if files:
        return "\n\n".join(open(f).read() for f in files)
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    return "".join(pq.read_table(path).column("text").to_pylist())


def kernel_prefill(query, key, value, T, fmt, chunk, scale):
    """The last T queries through mlops_fbla_prefill (tensor cores where the device has them, MLOPS_FBLA_DP4A=1:
    DP4A) in chunks, the first N - T tokens appended to the cache without attention (T = N: the whole window from
    an empty cache).  query [1, Hq, N, D], key / value [1, Hkv, N, D] -> [1, Hq, T, D] BF16."""
    N = query.shape[2]
    c = fbla_empty(1, key.shape[1], N, fmt)
    if N > T:
        fbla_append(c, key[:, :, :N - T], value[:, :, :N - T])
    return torch.cat([fbla_prefill(query[:, :, i:i + chunk].contiguous(), key[:, :, i:i + chunk], value[:, :, i:i + chunk], c, scale=scale)
                      for i in range(N - T, N, chunk)], 2)


def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    g = query.shape[1] // key.shape[1]
    k, v = key.repeat_interleave(g, 1), value.repeat_interleave(g, 1)
    mode = CUR["mode"]
    T = CUR.get("decode", 0)
    if T and mode is not None:                   # prefill exact, the last T queries decode against the cache
        N = query.shape[2]
        o = torch.empty(query.shape, dtype=torch.float32, device=query.device)
        o[:, :, : N - T] = chunked_sdpa(query[:, :, : N - T].float(), k[:, :, : N - T].float(), v[:, :, : N - T].float(), scaling)
        if isinstance(mode, fb.FBLAConfig):
            o[:, :, N - T:] = fb.attention(query[0, :, N - T:], k[0], v[0], mode, scaling, CUR["stats"])[None]
        elif mode[0] == "kernel":
            o[:, :, N - T:] = kernel_decode(query, key, value, T, mode[1], mode[2], scaling).float()
        elif mode[0] == "prefill":
            o[:, :, N - T:] = kernel_prefill(query, key, value, T, mode[1], mode[2], scaling).float()
        else:
            fk, fv = mode
            kq, vq = fk(k), fv(v)
            G = 128
            kq[:, :, :G], vq[:, :, :G] = k[:, :, :G].float(), v[:, :, :G].float()
            for i in range(N - T, N, G):                                 # each query's own group exact
                j = min(i + G, N)
                kk, vv = kq[:, :, :j].clone(), vq[:, :, :j].clone()
                kk[:, :, i - i % G:j], vv[:, :, i - i % G:j] = k[:, :, i - i % G:j].float(), v[:, :, i - i % G:j].float()
                mask = torch.arange(j, device=query.device)[None, :] <= torch.arange(i, j, device=query.device)[:, None]
                o[:, :, i:j] = F.scaled_dot_product_attention(query[:, :, i:j].float(), kk, vv, attn_mask=mask, scale=scaling)
        return o.to(query.dtype).transpose(1, 2).contiguous(), None
    if mode is None:
        o = chunked_sdpa(query, k, v, scaling)
    elif isinstance(mode, fb.FBLAConfig):
        o = fb.attention(query[0], k[0], v[0], mode, scaling, CUR["stats"])[None]
    elif mode[0] == "prefill":
        o = kernel_prefill(query, key, value, query.shape[2], mode[1], mode[2], scaling).float()
    else:
        fk, fv = mode
        o = decode_semantics(query, k, v, fk, fv, scaling)
    return o.to(query.dtype).transpose(1, 2).contiguous(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("text", nargs="*", help="text files; none: the WikiText-2 test split")
    ap.add_argument("--win", type=int, default=2048); ap.add_argument("--windows", type=int, default=13)
    ap.add_argument("--decode", type=int, default=0)
    ap.add_argument("--only", help="comma-separated substrings: run only the schemes whose name contains one (and SDPA)")
    ap.add_argument("--absolute", action="store_true", help="with --only, skip SDPA and report absolute loss")
    ap.add_argument("--baseline-losses", help="tensor of cached per-window bf16 losses; do not run SDPA")
    ap.add_argument("--save-losses", help="save the single selected mode's per-window losses to this file")
    a = ap.parse_args()
    AttentionInterface.register("fbla_eval", attn)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="fbla_eval").cuda()
    ids = tok(corpus(a.text), return_tensors="pt").input_ids
    if a.decode:   # windows spread over the text (they may overlap)
        n = ids.shape[1] - a.win
        wins = [ids[:, s:s + a.win].cuda() for s in torch.linspace(0, n, a.windows).long().tolist()]
    else:
        wins = [ids[:, i:i + a.win].cuda() for i in range(0, ids.shape[1] - a.win + 1, a.win)][: a.windows]
    CUR["decode"] = a.decode

    def loss_of(w):
        if not a.decode:
            return model(w, labels=w).loss.item()
        h = model.model(w).last_hidden_state[:, -a.decode - 1:-1]
        return F.cross_entropy(model.lm_head(h).float().flatten(0, 1), w[:, -a.decode:].flatten()).item()
    G = fb.FBLAConfig().group
    modes = [("bf16 SDPA", None, "16"),
             ("vLLM fp8 e4m3, per tensor", (kn.fp8_tensor, kn.fp8_tensor), "8"),
             ("llama.cpp q8_0", (kn.q8_0, kn.q8_0), "8.5"),
             ("KIVI 4-bit (K per channel / V per token, groups of 32)", (kn.kivi_k(4), kn.kivi_v(4)), "5.0"),
             ("FBLA coarse only (4 b)", fb.FBLAConfig(refine="none"), None),
             ("FBLA refine p_blk > 0.01", fb.FBLAConfig(), None),
             ("FBLA refine p_blk > 0.001", fb.FBLAConfig(tau=1e-3), None),
             ("FBLA all refined (6 b)", fb.FBLAConfig(refine="all"), None),
             ("FBLA refine p_blk > 0.01, q16 (two dp4a)", fb.FBLAConfig(q_bits=16), None),
             ("FBLA refine W_j > 0.01 (normalized mass, two passes)", fb.FBLAConfig(refine="norm"), None),
             ("FBLA refine W_j > 0.001 (normalized mass, two passes)", fb.FBLAConfig(refine="norm", tau=1e-3), None),
             ("FBLA 3+3 refine W_j > 0.001", fb.FBLAConfig(coarse_bits=3, fine_bits=3,
                                                            refine="norm", tau=1e-3), None),
             ("FBLA 3+3 refine W_j > 0.0003", fb.FBLAConfig(coarse_bits=3, fine_bits=3,
                                                             refine="norm", tau=3e-4), None),
             # prefill (exact: every block through its full codes), P per 16-token block or per 128-token group
             ("FBLA prefill exact 4+2, P block 16", fb.FBLAConfig(refine="all"), None),
             ("FBLA prefill exact 4+2, P block 128", fb.FBLAConfig(refine="all", sub=128), None),
             # prefill chunk (--decode T): one refinement decision per 16 consecutive queries, coarse q / P bits
             ("FBLA prefill 4+2 W_j > 0.001, tile 16, coarse q8 p8", fb.FBLAConfig(refine="norm", tau=1e-3, tile=16), None),
             ("FBLA prefill 4+2 W_j > 0.001, tile 16, coarse q4 p8",
              fb.FBLAConfig(refine="norm", tau=1e-3, tile=16, q_bits_coarse=4), None),
             ("FBLA prefill 4+2 W_j > 0.001, tile 16, coarse q8 p4",
              fb.FBLAConfig(refine="norm", tau=1e-3, tile=16, p_bits_coarse=4), None),
             ("FBLA prefill 4+2 W_j > 0.001, tile 16, coarse q4 p4",
              fb.FBLAConfig(refine="norm", tau=1e-3, tile=16, q_bits_coarse=4, p_bits_coarse=4), None),
             ("FBLA prefill 3+3 W_j > 0.0003, tile 16, coarse q4 p4",
              fb.FBLAConfig(coarse_bits=3, fine_bits=3, refine="norm", tau=3e-4, tile=16, q_bits_coarse=4,
                            p_bits_coarse=4), None),
             ("FBLA kernel 3+3, W_j > 0.0003 (mlops_fbla_decode)", ("kernel", "3+3", 3e-4), "kernel"),
             ("FBLA kernel 4+2, W_j > 0.001 (mlops_fbla_decode)", ("kernel", "4+2", 1e-3), "kernel"),
             ("FBLA kernel prefill 4+2, chunks of 1024 (mlops_fbla_prefill)", ("prefill", "4+2", 1024), "kernel"),
             ("KIVI 2-bit", (kn.kivi_k(2), kn.kivi_v(2)), "3.0"),
             ("llama.cpp q4_0", (kn.q4_0, kn.q4_0), "4.5")]
    what = f"last {a.decode} tokens decoding against the cache of the first {a.win - a.decode}" if a.decode else "all tokens"
    metric = "absolute LM loss" if a.absolute else "delta vs bf16 SDPA"
    print(f"{a.model}: {len(wins)} windows of {a.win} tokens, loss of the {what}; {metric} (mean, min .. max)\n")
    loss_heading = "LM loss" if a.absolute else "delta loss"
    table = Table([("attention", 56, "<"), ("K / V bits per element read", 20, ">"),
                   (loss_heading + " (mean, min .. max)", 30, ">")])
    if a.only:
        keys = a.only.split(",")
        use_cached_base = a.baseline_losses is not None
        modes = [m for m in modes if any(x in m[0] for x in keys)
                 or (not a.absolute and not use_cached_base and m[1] is None)]
    if a.save_losses and len(modes) != 1:
        raise ValueError("--save-losses requires exactly one selected mode")
    base = torch.load(a.baseline_losses).float() if a.baseline_losses else None
    for name, mode, bits in modes:
        CUR["mode"], CUR["stats"] = mode, {}
        with torch.no_grad():
            ls = torch.tensor([loss_of(w) for w in wins])
        if a.save_losses:
            torch.save(ls, a.save_losses)
        base = ls if base is None else base
        dl = ls if a.absolute else ls - base
        if isinstance(mode, fb.FBLAConfig):
            st = CUR["stats"]; share = st["refined"] / max(st["quantized"], 1)
            bits = f"{mode.coarse_bits + 32 / G + mode.fine_bits * share:.2f} (refined {share:.1%})"
        table.row(name, bits, f"{dl.mean():+.4f} ({dl.min():+.4f} .. {dl.max():+.4f})")


if __name__ == "__main__":
    main()
