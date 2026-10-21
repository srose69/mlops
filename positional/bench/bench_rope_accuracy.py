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

With --scaling the frequencies are those of a context-extension recipe (configurations of released models):
yarn (Qwen2.5 / DeepSeek style, factor 4 from 32K), llama3 (Llama 3.1, factor 8 from 8K), linear (factor 4),
dynamic (NTK, factor 4 from 32K: the base grows with the position, in float32 on the GPU), longrope (Phi-3 style per-frequency factors,
from 4K). transformers computes their frequencies in float32 (on the CPU for the static ones, on the GPU for
dynamic and LongRoPE, which it recomputes in the forward) and multiplies cos / sin by the attention factor in
float32; the exact reference uses float64 frequencies. torchtune is measured without scaling only.

    python bench/bench_rope_accuracy.py [--scaling default|yarn|llama3|linear|dynamic|longrope|all] [--base B]
"""
import argparse
import os

import torch

from mlops.positional import RoPE, rope_frequencies

SCALINGS = {
    "default": None,
    "yarn": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768},
    "llama3": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
               "original_max_position_embeddings": 8192},
    "linear": {"rope_type": "linear", "factor": 4.0},
    "dynamic": {"rope_type": "dynamic", "factor": 4.0},
    "longrope": {"rope_type": "longrope", "original_max_position_embeddings": 4096,
                 "short_factor": [1.0 + 0.02 * i for i in range(64)], "long_factor": [1.0 + 0.6 * i for i in range(64)]},
}
MAX_POS = 32768  # max_position_embeddings of the configs (dynamic grows the base beyond it)
STARTS = [128 << 10, 256 << 10, 512 << 10, 1 << 20, 2 << 20, 3 << 20, 4 << 20, 5 << 20, 6 << 20, 7 << 20, 8 << 20,
          10 << 20]
N, H = 64, 8
dev = "cuda"


def label(p):
    return f"{p >> 20}M" if p >= 1 << 20 else f"{p >> 10}K"


# ---- the three implementations ----------------------------------------------------------------------------
def inv_freq_fp32(dim, base):  # transformers and torchtune: float32 arithmetic
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))


def transformers_like(q, pos, inv32, att=1.0):
    """[B, S, H, D] q, pos [S]: LlamaRotaryEmbedding.forward + apply_rotary_pos_emb, half-split pairs."""
    freqs = pos.float()[:, None] * inv32.to(q.device)[None, :]           # float32 product (their K=1 matmul)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = (emb.cos() * att).to(q.dtype), (emb.sin() * att).to(q.dtype)
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


def exact(q, pos, inv64, interleaved, att=1.0):
    """float64 rotation with exact angles: position (integer) times the exact frequency, reduced in float64."""
    x = q.double()
    ang = pos.double()[:, None] * inv64.to(q.device)[None, :]
    c, s = (torch.cos(ang) * att)[None, :, None, :], (torch.sin(ang) * att)[None, :, None, :]
    if interleaved:
        x1, x2 = x[..., 0::2], x[..., 1::2]
    else:
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
    return x1, x2, x1 * c - x2 * s, x2 * c + x1 * s


def pair_errors(y, q, pos, inv64, interleaved, att=1.0):
    """(RMS, max) of |y - exact| / |att * pair| over all pairs."""
    x1, x2, r1, r2 = exact(q, pos, inv64, interleaved, att)
    yd = y.double()
    if interleaved:
        y1, y2 = yd[..., 0::2], yd[..., 1::2]
    else:
        h = yd.shape[-1] // 2
        y1, y2 = yd[..., :h], yd[..., h:]
    e = torch.sqrt((y1 - r1) ** 2 + (y2 - r2) ** 2) / (att * torch.sqrt(x1 ** 2 + x2 ** 2)).clamp_min(1e-30)
    return float(torch.sqrt((e ** 2).mean())), float(e.max())


def position_shifts(y, q, pos, inv64, interleaved):
    """The rotation error as a position error: each pair's angle error over its frequency, in tokens -- where that
    frequency puts the token. (max |shift|, RMS shift, max error of the distance m - n between the N positions)."""
    _, _, r1, r2 = exact(q, pos, inv64, interleaved)
    yd = y.double()
    y1, y2 = (yd[..., 0::2], yd[..., 1::2]) if interleaved else yd.split(yd.shape[-1] // 2, dim=-1)
    dth = torch.angle(torch.complex(y1, y2) * torch.complex(r1, -r2))    # [1, N, H, D/2]
    d = dth / inv64.to(y.device)
    dist = (d[0, :, None] - d[0, None, :]).abs().max()
    return float(d.abs().max()), float(d.pow(2).mean().sqrt()), float(dist)


def hf_config(D, base, scaling):
    from transformers import LlamaConfig
    params = dict(scaling or {"rope_type": "default"}, rope_theta=float(base))
    return LlamaConfig(hidden_size=D * H, num_attention_heads=H, head_dim=D, max_position_embeddings=MAX_POS,
                       rope_parameters=params)


def hf_frequencies(D, base, scaling, seq_len):
    """transformers' float32 inverse frequencies and attention factor for a forward at positions < seq_len: its
    own functions where installed (on the CPU for the static recipes, on the GPU for dynamic and LongRoPE, which it
    recomputes in the forward), else float32 of the float64 ones."""
    kind = (scaling or {}).get("rope_type", "default")
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        cfg = hf_config(D, base, scaling)
        if kind == "default":
            return inv_freq_fp32(D, base), 1.0
        if kind in ("dynamic", "longrope"):
            if kind == "longrope" and seq_len <= scaling["original_max_position_embeddings"]:
                inv, att = ROPE_INIT_FUNCTIONS[kind](cfg, "cpu")
            else:
                # dynamic: the forward passes seq_len = max(position_ids) + 1 as a CUDA tensor, so the grown base
                # is float32 tensor arithmetic on the GPU
                inv, att = ROPE_INIT_FUNCTIONS[kind](cfg, dev, seq_len=torch.tensor(seq_len, device=dev) if kind == "dynamic"
                                                     else scaling["original_max_position_embeddings"] + 1)
            if kind == "longrope":  # the attention factor is computed once, at construction
                att = ROPE_INIT_FUNCTIONS[kind](cfg, "cpu")[1]
            return inv.float().cpu(), float(att)
        inv, att = ROPE_INIT_FUNCTIONS[kind](cfg, "cpu")
        return inv.float(), float(att)
    except ImportError:
        inv, att = rope_frequencies(D, base, scaling, max_position_embeddings=MAX_POS, seq_len=seq_len)
        return inv.float(), att


# ---- the copies against the libraries ----------------------------------------------------------------------
def check_against_libraries(D, base, inv32, scaling):
    q = torch.randn(1, N, H, D, device=dev).to(torch.bfloat16)
    try:
        from transformers.models.llama import modeling_llama
        cfg = hf_config(D, base, scaling)
        rot = modeling_llama.LlamaRotaryEmbedding(cfg).to(dev)  # built on the CPU and moved, as a model is
        for p in STARTS:
            pos = torch.arange(p, p + N, device=dev)
            cos, sin = rot(q, pos[None])
            ref, _ = modeling_llama.apply_rotary_pos_emb(q, q, cos, sin, unsqueeze_dim=2)
            inv_p, att_p = hf_frequencies(D, base, scaling, p + N)
            assert torch.equal(ref, transformers_like(q, pos, inv_p, att_p)), f"transformers copy differs at {p}"
        print(f"transformers {__import__('transformers').__version__}: its bits reproduced at every start")
    except ImportError as e:
        print(f"transformers not importable ({str(e).strip().splitlines()[0]}): its copy unchecked")
    if scaling:
        return
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


def run(D, base, kind):
    scaling = SCALINGS[kind]
    torch.manual_seed(0)
    inv32 = inv_freq_fp32(D, base)
    ours_half = RoPE(D, base, scaling=scaling, max_position_embeddings=MAX_POS)
    ours_inter = RoPE(D, base, interleaved=True, scaling=scaling, max_position_embeddings=MAX_POS)
    short = {k: (f"[{len(v)} values]" if isinstance(v, list) else v) for k, v in (scaling or {}).items()}
    print(f"\n## {kind}: head_dim {D}, base {base:g}" + (f", {short}" if scaling else "") + "\n")
    check_against_libraries(D, base, inv32, scaling)
    for dtype in (torch.bfloat16, torch.float32):
        name = {torch.bfloat16: "BF16", torch.float32: "FP32"}[dtype]
        print(f"\n{name} q\n")
        tt = "torchtune" if not scaling else "torchtune (no scaling)"
        print(f"| position | exact, rounded once | transformers | {tt} | mlops (half split) | mlops (interleaved) |")
        print("|---|---|---|---|---|---|")
        for p in STARTS:
            pos = torch.arange(p, p + N, device=dev)
            q = torch.randn(1, N, H, D, device=dev).to(dtype)
            inv64, att = rope_frequencies(D, base, scaling, max_position_embeddings=MAX_POS, seq_len=p + N)
            inv_hf, att_hf = hf_frequencies(D, base, scaling, p + N)
            _, _, r1, r2 = exact(q, pos, inv64, False, att)
            floor = torch.cat((r1, r2), dim=-1).to(dtype)
            cells = []
            for y, inter in ((floor, False), (transformers_like(q, pos, inv_hf, att_hf), False),
                             (torchtune_like(q, pos, inv32) if not scaling else None, True),
                             (ours_half(q, None, pos)[0], False), (ours_inter(q, None, pos)[0], True)):
                if y is None:
                    cells.append("—")
                    continue
                rms, mx = pair_errors(y, q, pos, inv64, inter, att)
                cells.append(f"{rms:.1e} / {mx:.1e}")
            print(f"| {label(p)} | " + " | ".join(cells) + " |", flush=True)
    if scaling:
        return
    print(f"\nFP32 q, the error as a position: where each frequency puts the token, in tokens -- max / RMS of the "
          f"shift, and the largest error of the distance m - n between two of the {N} positions\n")
    print("| position | exact, rounded once | transformers | torchtune | mlops |")
    print("|---|---|---|---|---|")
    for p in STARTS:
        pos = torch.arange(p, p + N, device=dev)
        q = torch.randn(1, N, H, D, device=dev)
        inv64, _ = rope_frequencies(D, base, None, max_position_embeddings=MAX_POS, seq_len=p + N)
        _, _, r1, r2 = exact(q, pos, inv64, False)
        cells = [position_shifts(y, q, pos, inv64, inter) for y, inter in
                 ((torch.cat((r1, r2), dim=-1).float(), False), (transformers_like(q, pos, inv32), False), (torchtune_like(q, pos, inv32), True),
                  (ours_half(q, None, pos)[0], False))]
        print(f"| {label(p)} | " + " | ".join(f"{a:.3f} / {b:.3f} / {c:.3f}" for a, b, c in cells) + " |", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=float, default=500000.0)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--scaling", default="default", choices=list(SCALINGS) + ["all"])
    a = ap.parse_args()
    print(f"RoPE accuracy: {N} positions from each start x {H} heads; error per pair relative to its magnitude "
          f"(times the attention factor), RMS / max")
    for kind in (SCALINGS if a.scaling == "all" else [a.scaling]):
        run(a.head_dim, a.base, kind)


if __name__ == "__main__":
    main()
