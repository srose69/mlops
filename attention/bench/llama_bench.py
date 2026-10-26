#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""llama-bench's tests for the mlops stack: a whole model on samples/04_generate.py's decoder (NVFP4EffMA linear
layers, mlops.positional RoPE, FBLA prefill / decode), so its numbers sit beside llama.cpp's.

    python bench/llama_bench.py MODEL_DIR [-p 512] [-n 128] [-d 0,4096] [-r 5] [--mode 4+2] [--nvfp4 mlp]

As llama-bench: random tokens; `ppP` processes P prompt tokens in one batch, `tgN` generates N tokens one at a
time; `@ dD` runs the test after a context of D tokens is already in the cache (not timed); one warm-up run,
then -r timed runs on a fresh cache each, reported as mean +- standard deviation of tokens/s.  The model is
placed per depth (the cache first, then the layers that fit); the rest stream from host memory (the column
`GPU layers`).
"""
import argparse
import importlib.util
import os
import statistics
import time

import torch

from termtable import Table

spec = importlib.util.spec_from_file_location(
    "generate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "samples", "04_generate.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("-p", "--n-prompt", default="512", help="prompt lengths, comma-separated (0: none)")
    ap.add_argument("-n", "--n-gen", default="128", help="generated lengths, comma-separated (0: none)")
    ap.add_argument("-d", "--n-depth", default="0", help="context already in the cache, comma-separated")
    ap.add_argument("-r", "--repetitions", type=int, default=5)
    ap.add_argument("--mode", default="4+2", choices=["4+2", "3+3"], help="FBLA quantization mode")
    ap.add_argument("--nvfp4", default="mlp", choices=["mlp", "all", "none"], help="see samples/04_generate.py")
    ap.add_argument("-ngl", "--gpu-layers", type=int, default=None, help="at most this many layers on the GPU")
    ap.add_argument("--nbuf", type=int, default=None, help="streamed layers on the GPU at a time (default: the decoder's)")
    ap.add_argument("--tc", choices=["exact", "int8"], help="NVFP4EffMA tensor cores (TCWeight: GEMV / GEMM), W4A16 / W4A8")
    ap.add_argument("--fill-chunk", type=int, default=512, help="prefill chunk of the (untimed) context (llama.cpp's ubatch)")
    a = ap.parse_args()
    ints = lambda s: [int(x) for x in s.split(",") if x]
    P, N, Dp = [x for x in ints(a.n_prompt) if x], [x for x in ints(a.n_gen) if x], ints(a.n_depth)

    model = gen.Decoder(a.model, a.nvfp4, "fbla", a.mode)
    if a.nbuf:
        model.NBUF = a.nbuf
    vocab = model.embed.shape[0]
    size = sum(x.numel() * x.element_size() for L in model.layers for n in model.LINEARS for x in L[n].t)
    size += sum(x.numel() * x.element_size() for x in model.lm_head.t)
    name = os.path.basename(os.path.normpath(a.model))
    print(f"{name}: {len(model.layers)} layers, FBLA {a.mode}, NVFP4EffMA linear layers{' (tensor cores, ' + a.tc + ')' if a.tc else ''} "
          f"({'checkpoint' if model.prequantized else a.nvfp4}), weights {size / 2**30:.2f} GiB (embedding table in host "
          f"memory); {torch.cuda.get_device_name()}\n")
    table = Table([("model", 24, "<"), ("GPU layers", 10, ">"), ("test", 16, ">"), ("t/s", 18, ">")])

    gen_tokens = torch.Generator().manual_seed(0)
    rand = lambda n: torch.randint(0, vocab, (n,), generator=gen_tokens)

    def fresh(depth):
        model.cache = None
        model.cache = model.new_cache(capacity)
        for s in range(0, depth, a.fill_chunk):
            model.forward(rand(min(a.fill_chunk, depth - s)), s)
        torch.cuda.synchronize()

    def pp(depth, n):
        fresh(depth)
        t0 = time.perf_counter()
        model.forward(rand(n), depth)
        torch.cuda.synchronize()
        return n / (time.perf_counter() - t0)

    def tg(depth, n):
        fresh(depth)
        toks = rand(n)
        t0 = time.perf_counter()
        for i in range(n):  # one token per step, as llama_decode with a batch of 1
            model.forward(toks[i:i + 1], depth + i)
        torch.cuda.synchronize()
        return n / (time.perf_counter() - t0)

    plans = []
    for d in Dp:
        capacity = d + max(P + N)
        resident = model.place(capacity, max(P + [a.fill_chunk]), a.gpu_layers, a.tc)
        if model.streamed:
            pl = model.plan
            plans.append(f"  d{d}: layers {model.streamed} streamed from host memory ({pl['layer_mb']:.0f} MiB each at "
                         f"{pl['bandwidth'] / 1e9:.1f} GB/s): {pl['copy_ms']:.1f} ms of copies per pass, at most "
                         f"{1e3 / pl['copy_ms']:.1f} tokens/s in tg")
        for kind, f, lens in (("pp", pp, P), ("tg", tg, N)):
            for n in lens:
                f(d, n)  # warm-up
                r = [f(d, n) for _ in range(a.repetitions)]
                test = f"{kind}{n}" + (f" @ d{d}" if d else "")
                sd = statistics.stdev(r) if len(r) > 1 else 0.0
                table.row(name, resident, test, f"{statistics.mean(r):.2f} ± {sd:.2f}")
                model.cache = None
                torch.cuda.empty_cache()


    if plans:
        print("\nstreaming plan (layers spread evenly, copies overlapping the compute of the layers before them):")
        print("\n".join(plans))


if __name__ == "__main__":
    with torch.no_grad():
        main()
