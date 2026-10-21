#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Accuracy of RoPE at long positions (128K .. 10M): transformers, torchtune and mlops.positional against the exact
rotation (float64 angles from the exact frequencies, float64 products).

The script stands alone: it computes transformers' and torchtune's results with their own arithmetic, written out
here, so it needs neither library and runs at any position (torchtune caches cos / sin for every position up to
max_seq_len: 5 GB at 10M). When the libraries are installed, it first checks that these copies give their bits:
transformers' LlamaRotaryEmbedding + apply_rotary_pos_emb at every position measured, torchtune's
RotaryPositionalEmbeddings up to TORCHTUNE_CHECK_MAX (default 1M) positions.

* transformers   angle = float32(position) * float32(inv_freq), cos / sin in FP32, rounded to the dtype, then
                 q * cos + rotate_half(q) * sin in the dtype (half-split pairs). inv_freq as computed on the CPU (a
                 model built there and moved); a rotary module built on a CUDA device gets it from the GPU's powf,
                 one float32 ulp off for some frequencies -- the same error class
* torchtune      the same FP32 angle, cos / sin in FP32 (its cache, built on the CPU), the rotation in FP32,
                 rounded to the dtype once (interleaved pairs)
* mlops          the phase reduced exactly in turns, sincospi, FP32 rotation, rounded once (both pair layouts)

Errors per pair, relative to the pair's magnitude: RMS and maximum over 64 consecutive positions from each start x
8 heads x 64 pairs; head_dim 128, base 500000 (Llama 3). The first column is the floor: the exact result rounded
once to the dtype.

    python bench/bench_rope_accuracy.py [--base 500000] [--head-dim 128]
"""
import argparse
import os

import torch

from mlops.positional import RoPE

STARTS = [128 << 10, 256 << 10, 512 << 10, 1 << 20, 2 << 20, 3 << 20, 4 << 20, 5 << 20, 6 << 20, 7 << 20, 8 << 20,
          10 << 20]
N, H = 64, 8
dev = "cuda"


def label(p):
    return f"{p >> 20}M" if p >= 1 << 20 else f"{p >> 10}K"


# ---- the three implementations ----------------------------------------------------------------------------
def inv_freq_fp32(dim, base):  # transformers and torchtune: float32 arithmetic
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))


def transformers_like(q, pos, inv32):
    """[B, S, H, D] q, pos [S]: LlamaRotaryEmbedding.forward + apply_rotary_pos_emb, half-split pairs."""
    freqs = pos.float()[:, None] * inv32.to(q.device)[None, :]           # float32 product (their K=1 matmul)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().to(q.dtype), emb.sin().to(q.dtype)
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    h = q.shape[-1] // 2
    rot = torch.cat((-q[..., h:], q[..., :h]), dim=-1)
    return q * c + rot * s


def torchtune_like(q, pos, inv32):
    """RotaryPositionalEmbeddings.forward with input_pos, interleaved pairs. Its cache is built at construction, on
    the CPU for a module built there and moved (as here), and indexed by the positions."""
    idx_theta = torch.einsum("i, j -> ij", pos.cpu().float(), inv32).float()
    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1).to(q.device)  # [S, D/2, 2]
    xs = q.float().reshape(*q.shape[:-1], -1, 2)
    rc = cache.view(-1, xs.size(1), 1, xs.size(3), 2)
    out = torch.stack([xs[..., 0] * rc[..., 0] - xs[..., 1] * rc[..., 1],
                       xs[..., 1] * rc[..., 0] + xs[..., 0] * rc[..., 1]], -1)
    return out.flatten(3).type_as(q)


def exact(q, pos, inv64, interleaved):
    """float64 rotation with exact angles: position (integer) times the exact frequency, reduced in float64."""
    x = q.double()
    ang = pos.double()[:, None] * inv64.to(q.device)[None, :]
    c, s = torch.cos(ang)[None, :, None, :], torch.sin(ang)[None, :, None, :]
    if interleaved:
        x1, x2 = x[..., 0::2], x[..., 1::2]
    else:
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
    return x1, x2, x1 * c - x2 * s, x2 * c + x1 * s


def pair_errors(y, q, pos, inv64, interleaved):
    """(RMS, max) of |y - exact| / |pair| over all pairs."""
    x1, x2, r1, r2 = exact(q, pos, inv64, interleaved)
    yd = y.double()
    if interleaved:
        y1, y2 = yd[..., 0::2], yd[..., 1::2]
    else:
        h = yd.shape[-1] // 2
        y1, y2 = yd[..., :h], yd[..., h:]
    e = torch.sqrt((y1 - r1) ** 2 + (y2 - r2) ** 2) / torch.sqrt(x1 ** 2 + x2 ** 2).clamp_min(1e-30)
    return float(torch.sqrt((e ** 2).mean())), float(e.max())


# ---- the copies against the libraries ----------------------------------------------------------------------
def check_against_libraries(D, base, inv32):
    q = torch.randn(1, N, H, D, device=dev).to(torch.bfloat16)
    try:
        from transformers import LlamaConfig
        from transformers.models.llama import modeling_llama
        cfg = LlamaConfig(hidden_size=D * H, num_attention_heads=H, head_dim=D, max_position_embeddings=1 << 24,
                          rope_parameters={"rope_type": "default", "rope_theta": float(base)})
        rot = modeling_llama.LlamaRotaryEmbedding(cfg).to(dev)  # built on the CPU and moved, as a model is
        for p in STARTS:
            pos = torch.arange(p, p + N, device=dev)
            cos, sin = rot(q, pos[None])
            ref, _ = modeling_llama.apply_rotary_pos_emb(q, q, cos, sin, unsqueeze_dim=2)
            assert torch.equal(ref, transformers_like(q, pos, inv32)), f"transformers copy differs at {p}"
        print(f"transformers {__import__('transformers').__version__}: its bits reproduced at every start")
    except ImportError as e:
        print(f"transformers not importable ({str(e).strip().splitlines()[0]}): its copy unchecked")
    try:
        from torchtune.modules import RotaryPositionalEmbeddings
        top = int(os.environ.get("TORCHTUNE_CHECK_MAX", str(1 << 20)))
        rope = RotaryPositionalEmbeddings(D, max_seq_len=top + N, base=base).to(dev)
        n = 0
        for p in STARTS:
            if p + N > top + N:
                continue
            pos = torch.arange(p, p + N, device=dev)
            assert torch.equal(rope(q, input_pos=pos[None]), torchtune_like(q, pos, inv32)), f"torchtune copy differs at {p}"
            n += 1
        del rope
        torch.cuda.empty_cache()
        print(f"torchtune {__import__('torchtune').__version__}: its bits reproduced at the {n} starts up to "
              f"{label(top)} (its cache holds every position up to max_seq_len)")
    except ImportError as e:
        print(f"torchtune not importable ({str(e).strip().splitlines()[0]}): its copy unchecked")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=float, default=500000.0)
    ap.add_argument("--head-dim", type=int, default=128)
    a = ap.parse_args()
    D, base = a.head_dim, a.base
    torch.manual_seed(0)
    inv32 = inv_freq_fp32(D, base)
    ours_half, ours_inter = RoPE(D, base), RoPE(D, base, interleaved=True)
    inv64 = ours_half.inv_freq  # the exact frequencies, float64
    print(f"RoPE accuracy, head_dim {D}, base {base:g}; {N} positions from each start x {H} heads; error per pair "
          f"relative to its magnitude, RMS / max\n")
    check_against_libraries(D, base, inv32)
    for dtype in (torch.bfloat16, torch.float32):
        name = {torch.bfloat16: "BF16", torch.float32: "FP32"}[dtype]
        print(f"\n{name} q\n")
        print("| position | exact, rounded once | transformers | torchtune | mlops (half split) | mlops (interleaved) |")
        print("|---|---|---|---|---|---|")
        for p in STARTS:
            pos = torch.arange(p, p + N, device=dev)
            q = torch.randn(1, N, H, D, device=dev).to(dtype)
            _, _, r1, r2 = exact(q, pos, inv64, False)
            floor = torch.cat((r1, r2), dim=-1).to(dtype)
            cells = []
            for y, inter in ((floor, False), (transformers_like(q, pos, inv32), False),
                             (torchtune_like(q, pos, inv32), True), (ours_half(q, None, pos)[0], False),
                             (ours_inter(q, None, pos)[0], True)):
                rms, mx = pair_errors(y, q, pos, inv64, inter)
                cells.append(f"{rms:.1e} / {mx:.1e}")
            print(f"| {label(p)} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
