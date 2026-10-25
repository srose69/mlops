#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Token-distribution comparison of one FBLA configuration (or KIVI / llama.cpp KV quantization) against bf16
attention, on one window of the WikiText-2 test split unless text files are given.

The two model forwards retain only the evaluated hidden states on CPU.  The LM head is then applied in small
chunks, so a [tokens, vocabulary] logits tensor is never kept in GPU memory.
"""
import argparse
import gc
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlops.attention import fbla_reference as fb  # noqa: E402
from fbla_lm import CUR, attn, corpus  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False


def quantiles(x, ps):
    return {p: torch.quantile(x.float(), p).item() for p in ps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("text", nargs="*", help="text files; none: the WikiText-2 test split")
    ap.add_argument("--ids-from", help="take the window's token ids from a --bf16-hidden-cache file instead of tokenizing text")
    ap.add_argument("--win", type=int, default=16384)
    ap.add_argument("--start", type=int, default=0, help="first token of the window in the text")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--logit-chunk", type=int, default=16)
    ap.add_argument("--tau", type=float, default=3e-4)
    ap.add_argument("--coarse-bits", type=int, default=3)
    ap.add_argument("--fine-bits", type=int, default=3)
    ap.add_argument("--bf16-hidden-cache", help="reuse/save bf16 hidden states for this exact tokenized window")
    ap.add_argument("--kernel", action="store_true", help="decode through the CUDA kernel instead of the model")
    ap.add_argument("--prefill-kernel", action="store_true", help="the evaluated tokens through mlops_fbla_prefill in chunks")
    ap.add_argument("--chunk", type=int, default=1024, help="prefill chunk of --prefill-kernel")
    ap.add_argument("--kivi", type=int, default=0, help="KIVI with this many bits instead of FBLA")
    ap.add_argument("--llama", choices=["q8_0", "q4_0"], help="llama.cpp's KV cache quantization instead of FBLA")
    a = ap.parse_args()

    AttentionInterface.register("fbla_token_metrics", attn)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="fbla_token_metrics"
    ).cuda().eval()
    ids = (torch.load(a.ids_from)["input_ids"] if a.ids_from else
           tok(corpus(a.text), return_tensors="pt").input_ids)[:, a.start:a.start + a.win].cuda()
    if ids.shape[1] != a.win:
        raise ValueError(f"text has only {ids.shape[1]} tokens, need {a.win}")
    CUR["decode"] = a.tokens

    def hidden(mode, stats):
        CUR["mode"], CUR["stats"] = mode, stats
        with torch.no_grad():
            h = model.model(ids).last_hidden_state[:, -a.tokens - 1:-1].cpu()
        torch.cuda.empty_cache()
        return h

    h_bf16 = None
    if a.bf16_hidden_cache and os.path.exists(a.bf16_hidden_cache):
        cache = torch.load(a.bf16_hidden_cache)
        same = (cache["model"] == a.model and cache["win"] == a.win and cache["tokens"] == a.tokens
                and torch.equal(cache["input_ids"], ids.cpu()))
        if not same:
            raise ValueError("bf16 hidden cache does not match this model/window/text")
        h_bf16 = cache["hidden"]
    if h_bf16 is None:
        h_bf16 = hidden(None, {})
        if a.bf16_hidden_cache:
            torch.save(dict(model=a.model, win=a.win, tokens=a.tokens, input_ids=ids.cpu(), hidden=h_bf16),
                       a.bf16_hidden_cache)
    cfg = fb.FBLAConfig(coarse_bits=a.coarse_bits, fine_bits=a.fine_bits, refine="norm", tau=a.tau)
    stats = {}
    import kv_native as kn
    mode = ((kn.kivi_k(a.kivi), kn.kivi_v(a.kivi)) if a.kivi else
            (getattr(kn, a.llama),) * 2 if a.llama else
            ("kernel", f"{a.coarse_bits}+{a.fine_bits}", a.tau) if a.kernel else
            ("prefill", f"{a.coarse_bits}+{a.fine_bits}", a.chunk) if a.prefill_kernel else cfg)
    h_fbla = hidden(mode, stats)
    targets = ids[:, -a.tokens:].cpu().flatten()

    top1_b, top1_q, margins = [], [], []
    kl, true_delta, true_abs_delta = [], [], []
    with torch.no_grad():
        for i in range(0, a.tokens, a.logit_chunk):
            j = min(i + a.logit_chunk, a.tokens)
            lb = model.lm_head(h_bf16[:, i:j].cuda()).float().flatten(0, 1)
            lq = model.lm_head(h_fbla[:, i:j].cuda()).float().flatten(0, 1)
            vb, ib = lb.topk(2, dim=-1)
            top1_b.append(ib[:, 0].cpu())
            top1_q.append(lq.argmax(-1).cpu())
            margins.append((vb[:, 0] - vb[:, 1]).cpu())
            lpb, lpq = F.log_softmax(lb, -1), F.log_softmax(lq, -1)
            kl.append((lpb.exp() * (lpb - lpq)).sum(-1).cpu())
            y = targets[i:j].cuda()
            d = (lpq.gather(1, y[:, None]) - lpb.gather(1, y[:, None])).squeeze(1).cpu()
            true_delta.append(d)
            true_abs_delta.append(d.abs())
            del lb, lq, vb, ib, lpb, lpq
    top1_b, top1_q = torch.cat(top1_b), torch.cat(top1_q)
    margins, kl = torch.cat(margins), torch.cat(kl)
    true_delta, true_abs_delta = torch.cat(true_delta), torch.cat(true_abs_delta)
    flips = top1_b != top1_q

    name = (f"KIVI {a.kivi}-bit" if a.kivi else f"llama.cpp {a.llama}" if a.llama else
            f"FBLA {a.coarse_bits}+{a.fine_bits}, prefill chunks of {a.chunk} (CUDA kernel)" if a.prefill_kernel else
            f"FBLA {a.coarse_bits}+{a.fine_bits}, W_j > {a.tau:g}" + (" (CUDA kernel)" if a.kernel else ""))
    print(f"{name}; context {a.win}, evaluated tokens {a.tokens}")
    if not (a.kernel or a.kivi or a.llama or a.prefill_kernel):
        share = stats["refined"] / max(stats["quantized"], 1)
        bits = cfg.coarse_bits + 32 / cfg.group + cfg.fine_bits * share
        print(f"refined {share:.3%}; quantized traffic {bits:.4f} bits/element")
    print(f"top-1 agreement: {(~flips).float().mean().item():.4%} ({(~flips).sum().item()}/{a.tokens})")
    print("top-1 flips by bf16 top1-top2 margin:")
    bounds = [(0.0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, float("inf"))]
    for lo, hi in bounds:
        mask = (margins >= lo) & (margins < hi)
        rate = flips[mask].float().mean().item() if mask.any() else float("nan")
        print(f"  [{lo:g}, {hi:g}): n={mask.sum().item():4d}, flips={rate:.3%}")
    kq = quantiles(kl, [0.5, 0.95, 0.99, 0.999])
    print(f"KL(bf16 || FBLA), nats: mean={kl.mean().item():.6g}, p50={kq[0.5]:.6g}, "
          f"p95={kq[0.95]:.6g}, p99={kq[0.99]:.6g}, p99.9={kq[0.999]:.6g}, max={kl.max().item():.6g}")
    aq = quantiles(true_abs_delta, [0.5, 0.95, 0.99, 0.999])
    dq = quantiles(true_delta, [0.001, 0.01, 0.5, 0.99, 0.999])
    print("|delta log p(true)|, nats: "
          f"p50={aq[0.5]:.6g}, p95={aq[0.95]:.6g}, p99={aq[0.99]:.6g}, "
          f"p99.9={aq[0.999]:.6g}, max={true_abs_delta.max().item():.6g}")
    print("delta log p(true) = FBLA - bf16, nats: "
          f"mean={true_delta.mean().item():+.6g}, p0.1={dq[0.001]:+.6g}, p1={dq[0.01]:+.6g}, "
          f"p50={dq[0.5]:+.6g}, p99={dq[0.99]:+.6g}, p99.9={dq[0.999]:+.6g}")

    del h_bf16, h_fbla, model
    gc.collect()


if __name__ == "__main__":
    main()
