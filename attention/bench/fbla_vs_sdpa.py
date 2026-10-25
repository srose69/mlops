#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""FBLA decode and prefill against the fastest attention the machine has: flash-attn 2 if installed, else
PyTorch's scaled_dot_product_attention (the flash backend where it exists, the memory-efficient one otherwise).

Accuracy (--qkv: q, k, v of every layer of a model at 16K, as captured for the format study): one decode query (the last position) per layer against
float64 attention, relative error of the 16 q heads' outputs, for the FBLA kernel, the PyTorch model of its
format (python/mlops/attention/fbla_reference.py, normalized-mass refinement, the same tau) and SDPA.

Speed (random K / V, B = 1, KV heads 8, D = 128; the baseline reads the BF16 K / V of the whole context, FBLA its
6.25-bit cache).  Every call is timed alone with CUDA events, 50 calls a mode, medians; the SM clock is the NVML
median while the call is in flight.
    idle   200 ms of pause before each call: the GPU is at its boost clock
    warm   back-to-back calls after a second of load: the sustained clock under the board's power limit
Decode: one query per q head, the L2 flushed before each call (a decode step reads a cache that is not there), both
sides replayed from a CUDA graph where it captures (no launch overhead in the time).  Prefill: a chunk of queries
appended to a cache of `context` tokens, causal over all of them; FBLA is the whole mlops.attention.prefill call
(close, transcode, attention, finish) on a cache with persistent images.

    python bench/fbla_vs_sdpa.py [--qkv qkv_16k.pt] [--decode 16384,65536,131072,262144] [--prefill]
"""
import argparse
import math
import threading
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from mlops.attention import append, decode, empty, encode, prefill
from termtable import Table
from mlops.attention import fbla_reference as fr

try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_func = flash_attn_with_kvcache = None

MODES = [("3+3", 3, 3, 3e-4), ("4+2", 4, 2, 1e-3)]
D, KH, NRUN = 128, 8, 50


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
    t = Table([("layer", 5, ">")] + [(n, 9, ">") for n in names])
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
        t.row(li, *(f"{x:.2e}" for x in row))
    t.row("mean", *(f"{x / len(layers):.2e}" for x in sums))
    print()


class Clock:
    """NVML SM clock (MHz) sampled every 0.5 ms into the list of the call in flight (`call`); nan without NVML."""

    def __init__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        except Exception:
            self.nv = None
        self.call, self.samples, self.run = None, {}, True
        self.t = threading.Thread(target=self.loop)
        self.t.start()

    def loop(self):
        while self.run:
            c = self.call
            if c is not None and self.nv:
                self.samples.setdefault(c, []).append(self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM))
            time.sleep(0.0005)

    def stop(self):
        self.run = False
        self.t.join()


def med(x):
    s = sorted(x)
    n = len(s)
    return float("nan") if not n else s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def measure(f, clock, flush=None):
    """-> {'idle': (ms, MHz), 'warm': (ms, MHz)}: medians of NRUN calls timed alone, with `flush()` before each call
    (outside the timing).  nan when PyTorch runs out of memory."""
    try:
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        t0 = time.time()
        while time.time() - t0 < 1.0:
            f()
        torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {m: (float("nan"), float("nan")) for m in ("idle", "warm")}
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    res = {}
    for mode, gap in (("warm", 0.0), ("idle", 0.2)):
        clock.samples = {}
        ms = []
        for i in range(NRUN):
            if gap:
                time.sleep(gap)
            if flush:
                flush()
            clock.call = i
            e0.record()
            f()
            e1.record()
            e1.synchronize()
            clock.call = None
            ms.append(e0.elapsed_time(e1))
        res[mode] = (med(ms), med([med(clock.samples.get(i, [])) for i in range(NRUN)]))
        if mode == "warm":
            time.sleep(1.0)
    return res


def graphed(f):
    """f replayed from a CUDA graph (no launch overhead in the time); f itself when it cannot be captured."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                f()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            f()
        return g.replay
    except Exception:
        torch.cuda.synchronize()
        return f


def l2_flush():
    n = 2 * torch.cuda.get_device_properties(0).L2_cache_size
    buf = torch.empty(n, dtype=torch.uint8, device="cuda")
    return lambda: buf.fill_(1)


def decode_baseline(q, k, v):
    """(name, call) of the fastest decode attention: q [1, Hq, 1, D], k / v [1, Hkv, N, D] BF16."""
    if flash_attn_with_kvcache is not None:
        qa, ka, va = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        return "FA2", lambda: flash_attn_with_kvcache(qa, ka, va)
    if torch.cuda.get_device_capability() >= (8, 0):
        def flash():
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        try:
            flash()
            return "SDPA flash", flash
        except RuntimeError:
            pass
    kr, vr, qh = k.repeat_interleave(q.shape[1] // k.shape[1], 1).half(), v.repeat_interleave(q.shape[1] // k.shape[1], 1).half(), q.half()

    def efficient():
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(qh, kr, vr)
    return "SDPA mem-efficient FP16 (KV repeated)", efficient


def prefill_baseline(q, k, v, n0):
    """(name, call) of the fastest causal chunk attention: q [1, Hq, t, D] at positions n0 .. n0 + t - 1 over
    k / v [1, Hkv, n0 + t, D] BF16 (bottom-right causal: SDPA's flash backend cannot do that, its mask path can)."""
    if flash_attn_func is not None:
        qa, ka, va = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        return "FA2", lambda: flash_attn_func(qa, ka, va, causal=True)
    t, n = q.shape[2], k.shape[2]
    mask = torch.arange(n, device=q.device)[None, :] <= torch.arange(n0, n, device=q.device)[:, None]
    kr, vr = k.repeat_interleave(q.shape[1] // k.shape[1], 1), v.repeat_interleave(q.shape[1] // k.shape[1], 1)

    def efficient():
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask)
    return "SDPA mem-efficient (mask, KV repeated)", efficient


def mhz(x):
    return "-" if x != x else f"{x:.0f}"


def speed_decode(sizes, gqa):
    torch.manual_seed(0)
    clock, flush = Clock(), l2_flush()
    QH = gqa * KH
    rows, name = [], "baseline"
    for N in sizes:
        k = torch.randn(1, KH, N, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        q = torch.randn(1, QH, 1, D, device="cuda", dtype=torch.bfloat16)
        try:
            name, call = decode_baseline(q, k, v)
            base = measure(graphed(call), clock, flush)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            base = {m: (float("nan"), float("nan")) for m in ("idle", "warm")}
        fb = []
        for mode, _, _, _ in MODES:
            c = encode(k, v, mode)
            qq = q[:, :, 0].contiguous()
            fb.append(measure(graphed(lambda: decode(qq, c)), clock, flush))
            del c
        rows.append((N, base, fb))
        del k, v
        torch.cuda.empty_cache()
    clock.stop()
    print(f"decode latency (us), B = 1, {QH} q heads / {KH} KV heads, D = 128, L2 flushed before each call; baseline: {name}\n")
    t = Table([("context", 7, ">"), ("mode", 4, ">"), (f"{name[:22]} us", 22, ">"), ("MHz", 5, ">"), ("FBLA 3+3 us", 11, ">"),
               ("FBLA 4+2 us", 11, ">"), ("MHz", 5, ">"), ("x 3+3", 6, ">"), ("x 4+2", 6, ">")])
    for N, base, fb in rows:
        for mode in ("idle", "warm"):
            b = base[mode][0]
            t.row(N, mode, "OOM" if b != b else f"{b * 1e3:.1f}", mhz(base[mode][1]), f"{fb[0][mode][0] * 1e3:.1f}",
                  f"{fb[1][mode][0] * 1e3:.1f}", mhz(fb[1][mode][1]), "-" if b != b else f"{b / fb[0][mode][0]:.2f}",
                  "-" if b != b else f"{b / fb[1][mode][0]:.2f}")
    print()


def speed_prefill(cases, mode):
    torch.manual_seed(0)
    clock = Clock()
    rows, name = [], "baseline"
    for gqa, n0, t in cases:
        QH, n = gqa * KH, n0 + t
        q = torch.randn(1, QH, t, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, KH, n, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        try:
            name, call = prefill_baseline(q, k, v, n0)
            base = measure(call, clock)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            base = {m: (float("nan"), float("nan")) for m in ("idle", "warm")}
        cache = empty(1, KH, n, mode)
        if n0:
            append(cache, k[:, :, :n0], v[:, :, :n0])
        kc, vc = k[:, :, n0:].contiguous(), v[:, :, n0:].contiguous()

        def fbla():
            cache.tokens = n0
            prefill(q, kc, vc, cache)
        rows.append((gqa, n0, t, base, measure(fbla, clock)))
        del q, k, v, cache, kc, vc
        torch.cuda.empty_cache()
    clock.stop()
    print(f"prefill latency (ms): a chunk appended to a cache of `context` tokens and attended causally, {KH} KV heads, "
          f"D = 128; baseline: {name}; FBLA {mode}\n")
    tb = Table([("GQA", 3, ">"), ("context", 7, ">"), ("chunk", 6, ">"), ("mode", 4, ">"), (f"{name[:22]} ms", 22, ">"), ("MHz", 5, ">"),
                ("FBLA ms", 8, ">"), ("MHz", 5, ">"), ("x", 6, ">")])
    for gqa, n0, t, base, fb in rows:
        for m in ("idle", "warm"):
            b, f = base[m][0], fb[m][0]
            tb.row(gqa, n0, t, m, "OOM" if b != b else f"{b:.3f}", mhz(base[m][1]), f"{f:.3f}", mhz(fb[m][1]),
                   "-" if b != b else f"{b / f:.2f}")
    print()


PREFILL_CASES = [(2, 65536, 2048), (1, 65536, 2048), (4, 65536, 2048), (8, 65536, 2048), (2, 0, 2048), (2, 16384, 2048),
                 (2, 65536, 512), (2, 65536, 8192), (2, 262144, 2048)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qkv")
    ap.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    ap.add_argument("--positions", type=int, default=32, help="decode positions per layer (the last ones)")
    ap.add_argument("--decode", default="16384,65536,131072,262144", help="decode contexts, '' to skip")
    ap.add_argument("--gqa", type=int, default=2, help="q heads per KV head of the decode rows")
    ap.add_argument("--prefill", action="store_true", help="the prefill rows (GQA, context, chunk: PREFILL_CASES)")
    ap.add_argument("--prefill-mode", default="4+2", choices=["3+3", "4+2"])
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.qkv:
        accuracy(a.qkv, a.layers, a.positions)
    if a.decode:
        speed_decode([int(x) for x in a.decode.split(",")], a.gqa)
    if a.prefill:
        speed_prefill(PREFILL_CASES, a.prefill_mode)


if __name__ == "__main__":
    main()
