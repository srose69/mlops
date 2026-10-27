# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""FBLA cache encoding (the layout of include/mlops_fbla.h), prefill and decode.

    cache = empty(batch, kv_heads, capacity, "4+2")   # or encode(k, v, "4+2", capacity) from dense K / V
    out = prefill(q, k, v, cache)          # a chunk: q [B, Hq, n, 128], k / v [B, Hkv, n, 128] BF16; appends
    append(cache, k_new, v_new)            # [B, Hkv, n, 128] without attention
    out = decode(q, cache)                 # one query per head: q [B, Hq, 128], Hq a multiple of the KV heads

``capacity`` lays the cache out for more tokens than it holds (one allocation for a growing context; the
decode reads the first ``tokens``).  Every write of the cache (encode, append, prefill) runs on the GPU
(mlops_fbla_prefill); ``_quantize`` here is the reference quantizer of the tests.
"""
from dataclasses import dataclass
from typing import Optional

import torch

from . import _C

_MODES = {"3+3": (3, 3, 0), "4+2": (4, 2, 1)}


def _bf16(x, toward):
    """x rounded to bf16 toward -inf (toward < 0) or +inf, as bf16."""
    y = x.to(torch.bfloat16)
    bad = y.float() > x if toward < 0 else y.float() < x
    edge = torch.full_like(y, -torch.inf if toward < 0 else torch.inf)
    return torch.where(bad, torch.nextafter(y, edge), y)


def _quantize(x, groups):
    """[B, H, N, 128] -> (base, step [B, H, groups, 128] bf16, codes [B, H, groups * 128, 128] uint8)."""
    B, H, N, D = x.shape
    xp = torch.zeros((B, H, groups * 128, D), device=x.device, dtype=torch.float32)
    xp[:, :, :N] = x.float()
    xb = xp.reshape(B, H, groups, 128, D)
    valid = (torch.arange(groups * 128, device=x.device) < N).reshape(groups, 128, 1)
    lo = torch.where(valid, xb, torch.inf).amin(-2)
    hi = torch.where(valid, xb, -torch.inf).amax(-2)
    lo, hi = torch.where(torch.isfinite(lo), lo, 0), torch.where(torch.isfinite(hi), hi, 0)
    base = _bf16(lo, -1)
    step = _bf16((hi - base.float()) * torch.tensor(1 / 63, dtype=torch.float32, device=x.device), +1)  # * fl(1/63)
    s = step.float()[..., None, :]
    inv = torch.where(s > 0, torch.reciprocal(s.clamp_min(1e-30)), torch.zeros_like(s))  # fl(1/step), one per channel
    code = torch.round((xb - base.float()[..., None, :]) * inv)
    code = code.clamp_(0, 63).to(torch.uint8).reshape(B, H, groups * 128, D)
    return base.contiguous(), step.contiguous(), code


@dataclass
class FBLACache:
    coarse: torch.Tensor  # [B, H, capacity_groups * 8, 2 CB 256] uint8
    fine: torch.Tensor    # [B, H, capacity_groups * 8, 2 FB 256] uint8
    meta: torch.Tensor    # [B, H, capacity_groups, 512] bfloat16
    sink_k: torch.Tensor
    sink_v: torch.Tensor
    open_k: torch.Tensor
    open_v: torch.Tensor
    tokens: int
    capacity: int
    format: int
    images: Optional[torch.Tensor] = None  # uint8, the tensor-core prefill's closed-group images (None: rebuilt every call)


def _check_kv(k, v):
    if k.shape != v.shape or k.ndim != 4 or k.shape[-1] != 128:
        raise ValueError(f"k and v must have the same [B, H, N, 128] shape, got {tuple(k.shape)} and {tuple(v.shape)}")
    if not k.is_cuda or k.device != v.device:
        raise ValueError("k and v must be CUDA tensors on the same device")


def empty(batch: int, kv_heads: int, capacity: int, mode: str = "3+3", device="cuda") -> FBLACache:
    """An empty cache laid out for ``capacity`` tokens (zero-filled: 0 tokens)."""
    if mode not in _MODES:
        raise ValueError(f"mode must be '3+3' or '4+2', got {mode!r}")
    cb, fb, fmt = _MODES[mode]
    cg = (capacity + 127) // 128
    u8 = dict(device=device, dtype=torch.uint8)
    bf = dict(device=device, dtype=torch.bfloat16)
    ex = [torch.zeros((batch, kv_heads, 128, 128), **bf) for _ in range(4)]
    dev = torch.device(device)
    with torch.cuda.device(dev):  # (the size depends on the device: 0 where the prefill does not use tensor cores)
        nimg = _C.images_bytes(batch, kv_heads, cg * 128)
    images = torch.zeros(nimg, **u8) if nimg else None
    return FBLACache(torch.zeros((batch, kv_heads, cg * 8, 2 * cb * 256), **u8),
                     torch.zeros((batch, kv_heads, cg * 8, 2 * fb * 256), **u8),
                     torch.zeros((batch, kv_heads, cg, 512), **bf), ex[0], ex[1], ex[2], ex[3], 0, capacity, fmt, images)


def _prefill(q, cache, k, v, scale):
    _check_kv(k, v)
    c = cache
    if k.shape[:2] != c.meta.shape[:2]:
        raise ValueError(f"k / v heads {tuple(k.shape[:2])} do not match the cache's {tuple(c.meta.shape[:2])}")
    n = k.shape[2]
    if c.tokens + n > c.capacity:
        raise ValueError(f"cache full: {c.tokens} + {n} tokens > capacity {c.capacity}")
    out = _C.prefill(q, k.to(torch.bfloat16).contiguous(), v.to(torch.bfloat16).contiguous(), c.coarse, c.fine,
                     c.meta, c.sink_k, c.sink_v, c.open_k, c.open_v, c.tokens, c.capacity,
                     q.shape[1] if q is not None else k.shape[1], c.format, scale, c.images)
    c.tokens += n
    return out


@torch.no_grad()
def encode(k: torch.Tensor, v: torch.Tensor, mode: str = "3+3", capacity: Optional[int] = None) -> FBLACache:
    """Encode dense ``[B, Hkv, N, 128]`` K / V into an :class:`FBLACache` laid out for ``capacity`` tokens."""
    _check_kv(k, v)
    B, H, N, _ = k.shape
    capacity = N if capacity is None else capacity
    if capacity < N:
        raise ValueError(f"capacity {capacity} < tokens {N}")
    cache = empty(B, H, capacity, mode, k.device)
    _prefill(None, cache, k, v, 0.0)
    return cache


@torch.no_grad()
def append(cache: FBLACache, k: torch.Tensor, v: torch.Tensor) -> FBLACache:
    """Append ``[B, Hkv, n, 128]`` K / V to ``cache`` in place (a decode loop: n = 1 per step).

    New tokens go to the open group (BF16); a full open group is quantized into its records and metadata when
    the next token arrives, exactly as :func:`encode` would.  The first 128 tokens are also the sink group."""
    _prefill(None, cache, k, v, 0.0)
    return cache


@torch.no_grad()
def prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cache: FBLACache, *, scale: float = 0.0) -> torch.Tensor:
    """Append a chunk ``k, v [B, Hkv, n, 128]`` to ``cache`` and return the attention of its queries
    ``q [B, Hq, n, 128]`` (BF16): position ``cache.tokens + i`` attends to the tokens up to itself through the
    full 6-bit codes, as :func:`decode` with ``tau`` -> 0 would after appending them (prefill is compute-bound:
    refining every block costs nothing).  Chunked prefill of a long prompt: call it chunk by chunk."""
    if q.dtype != torch.bfloat16 or q.ndim != 4 or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f"q must be BF16 [B, Hq, n, 128] matching k {tuple(k.shape)}, got {q.dtype} {tuple(q.shape)}")
    return _prefill(q.contiguous(), cache, k, v, scale)


_SCRATCH = {}  # (device, batch * kv_heads) -> allocated once, reused (mlops_fbla_decode_scratch_bytes)


def _scratch(device, heads, nbytes):
    key = (device, heads)
    s = _SCRATCH.get(key)
    if s is None or s.numel() < nbytes:
        s = torch.full((nbytes,), 0xFF, dtype=torch.uint8, device=device)
        _SCRATCH[key] = s
    return s


def decode(q: torch.Tensor, cache: FBLACache, *, scale: float = 0.0, tau: float = 0.0) -> torch.Tensor:
    """One decode query per q head ``[B, Hq, 128]`` BF16 against ``cache``; scale 0 = 1/sqrt(128), tau 0 = the
    format's default."""
    if q.dtype != torch.bfloat16:
        raise TypeError(f"q must be bfloat16, got {q.dtype}")
    c = cache
    kv_heads = c.meta.shape[1]
    nbytes = _C.scratch_bytes(q.shape[0], q.shape[1], kv_heads, c.tokens, c.format, c.capacity)
    scratch = _scratch(q.device, q.shape[0] * kv_heads, nbytes)
    return _C.decode(q.contiguous(), c.coarse, c.fine, c.meta, c.sink_k, c.sink_v, c.open_k, c.open_v, scratch,
                     c.tokens, c.capacity, kv_heads, c.format, scale, tau)
