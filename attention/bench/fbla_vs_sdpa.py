#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""FBLA decode against PyTorch's scaled_dot_product_attention.

Accuracy (--qkv: q, k, v of every layer of a model at 16K, as captured for the format study): one decode query (the last position) per layer against
float64 attention, relative error of the 16 q heads' outputs, for the FBLA kernel, the PyTorch model of its
format (python/mlops/attention/fbla_reference.py, normalized-mass refinement, the same tau) and SDPA.

Speed (random cache, B = 1, 16 q heads / 8 KV heads, D = 128): on sm_61 SDPA's memory-efficient backend runs
only FP16 without GQA, so it gets K / V repeated to 16 heads (outside the timing); BF16 runs the math backend.

    python bench/fbla_vs_sdpa.py [--qkv qkv_16k.pt] [--sizes 16384,65536,131072,262144]
"""
import argparse
import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from mlops.attention import decode, encode
from mlops.attention import fbla_reference as fr

MODES = [("3+3", 3, 3, 3e-4), ("4+2", 4, 2, 1e-3)]


def rel(a, b):
    return ((a.double() - b).norm() / b.norm()).item()


def kernel_rows(q, k, v, mode, tau, scale, M):
    """The last M positions through the kernel, position p against a cache of p + 1 tokens (the closed groups
    encoded once, the open group BF16 up to p).  q [Hq, N, D], k / v [Hkv, N, D] -> [Hq, M, D]."""
    from mlops.attention.fbla import FBLACache
    N = k.shape[1]
    c = encode(k[None], v[None], mode)
    out = []
    for p in range(N - M, N):
        g0 = p // 128 * 128
        ok, ov = torch.zeros_like(c.open_k), torch.zeros_like(c.open_v)
        ok[0, :, :p + 1 - g0] = k[:, g0:p + 1]; ov[0, :, :p + 1 - g0] = v[:, g0:p + 1]
        cp = FBLACache(c.coarse, c.fine, c.meta, c.sink_k, c.sink_v, ok, ov, p + 1, c.capacity, c.format)
        out.append(decode(q[None, :, p].contiguous(), cp, scale=scale, tau=tau)[0])
    return torch.stack(out, 1)


def accuracy(path, layers, M):
    d = torch.load(path)
    scale = d.get("scale") or 1 / math.sqrt(128)
    L = d["q"].shape[0] if layers == "all" else None
    layers = range(L) if L else [int(x) for x in layers.split(",")]
    N = d["q"].shape[2]
    print(f"{d['model']}: decode queries at the last {M} positions of {N}, every one against its own prefix; "
          f"relative error vs float64 attention, mean over {len(layers)} layers x {M} positions\n")
    names = ["FBLA 3+3 kernel", "FBLA 3+3 reference", "kernel vs reference", "FBLA 4+2 kernel",
             "FBLA 4+2 reference", "kernel vs reference", "SDPA BF16"]
    print("| layer | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    sums = [0.0] * len(names)
    pos = torch.arange(N - M, N, device="cuda")
    for li in layers:
        q = d["q"][li].cuda()                  # [16, N, 128]
        k, v = d["k"][li].cuda(), d["v"][li].cuda()
        g = q.shape[0] // k.shape[0]
        kr, vr = k.repeat_interleave(g, 0), v.repeat_interleave(g, 0)
        s = (q[:, N - M:].double() @ kr.double().transpose(-1, -2)) * scale
        s = s.masked_fill(torch.arange(N, device="cuda")[None, None, :] > pos[None, :, None], -math.inf)
        exact = torch.softmax(s, -1) @ vr.double()
        row = []
        for mode, cb, fb, tau in MODES:
            kern = kernel_rows(q, k, v, mode, tau, scale, M)
            cfg = fr.FBLAConfig(coarse_bits=cb, fine_bits=fb, refine="norm", tau=tau)
            ref = fr.attention(q[:, N - M:].float(), kr.float(), vr.float(), cfg, scale)
            row += [rel(kern, exact), rel(ref, exact), rel(kern, ref.double())]
        mask = torch.arange(N, device="cuda")[None, :] <= pos[:, None]
        with sdpa_kernel([SDPBackend.MATH]):
            sd = F.scaled_dot_product_attention(q[None, :, N - M:], kr[None], vr[None], attn_mask=mask, scale=scale)[0]
        row.append(rel(sd, exact))
        sums = [x + y for x, y in zip(sums, row)]
        print(f"| {li} | " + " | ".join(f"{x:.2e}" for x in row) + " |", flush=True)
    print("| mean | " + " | ".join(f"{x / len(layers):.2e}" for x in sums) + " |\n")


def timed(fn, reps=20):
    try:
        for _ in range(3):
            fn()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan")
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = []
    for _ in range(reps):
        a.record(); fn(); b.record(); b.synchronize()
        t.append(a.elapsed_time(b))
    return sorted(t)[len(t) // 2] * 1e3


def speed(sizes):
    print("decode latency (us), B = 1, 16 q heads / 8 KV heads, D = 128\n")
    print("| context | SDPA FP16 efficient (KV repeated) | SDPA BF16 math | SDPA FP32 math | FBLA 3+3 | FBLA 4+2 | speed-up vs SDPA BF16 / FP32 |")
    print("|---|---|---|---|---|---|---|")
    torch.manual_seed(0)
    for N in sizes:
        k = torch.randn(1, 8, N, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        q = torch.randn(1, 16, 1, 128, device="cuda", dtype=torch.bfloat16)
        kr, vr, qh = k.repeat_interleave(2, 1).half(), v.repeat_interleave(2, 1).half(), q.half()
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            t_eff = timed(lambda: F.scaled_dot_product_attention(qh, kr, vr), reps=3 if N > 65536 else 10)
        del kr, vr
        torch.cuda.empty_cache()
        with sdpa_kernel([SDPBackend.MATH]):
            t_math = timed(lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True))
        qf, kf, vf = q.float(), k.float(), v.float()
        with sdpa_kernel([SDPBackend.MATH]):
            t_fp32 = timed(lambda: F.scaled_dot_product_attention(qf, kf, vf, enable_gqa=True))
        del qf, kf, vf
        res = []
        for mode, _, _, _ in MODES:
            c = encode(k, v, mode)
            res.append(timed(lambda: decode(q[:, :, 0], c)))
            del c
        print(f"| {N} | {t_eff:.1f} | {t_math:.1f} | {t_fp32:.1f} | {res[0]:.1f} | {res[1]:.1f} | "
              f"{t_math / res[0]:.2f}x / {t_math / res[1]:.2f}x; "
              f"{t_fp32 / res[0]:.2f}x / {t_fp32 / res[1]:.2f}x |",
              flush=True)
        del k, v
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qkv")
    ap.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    ap.add_argument("--positions", type=int, default=32, help="decode positions per layer (the last ones)")
    ap.add_argument("--sizes", default="16384,65536,131072,262144")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.qkv:
        accuracy(a.qkv, a.layers, a.positions)
    if a.sizes:
        speed([int(x) for x in a.sizes.split(",")])


if __name__ == "__main__":
    main()
