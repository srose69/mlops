# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""FBLA cache encoding (the layout of include/mlops_fbla.h) and decode.

    cache = encode(k, v, "3+3")            # k, v: [batch, kv_heads, tokens, 128] BF16
    out = decode(q, cache)                 # q: [batch, q_heads, 128] BF16, q_heads / kv_heads = 1, 2 or 4

``capacity`` lays the cache out for more tokens than it holds (one allocation for a growing context; the
decode reads the first ``tokens``).
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
    step = _bf16((hi - base.float()) / 63.0, +1)
    s = step.float()[..., None, :]
    code = torch.where(s > 0, torch.round((xb - base.float()[..., None, :]) / s.clamp_min(1e-30)), torch.zeros_like(xb))
    code = code.clamp_(0, 63).to(torch.uint8).reshape(B, H, groups * 128, D)
    return base.contiguous(), step.contiguous(), code


def _lanes(code, value):
    """[B, H, n, 16, 128] codes of n sub-blocks -> [B, H, n, 32 lanes, 16 slots, 4 bytes] (mlops_fbla.h).
    K: lane = (token l % 16, half l / 16), slot s = channels 64 half + 4 s .. + 3.  V: lane = channels
    4 l .. + 3; returned in (token quad j, channel c) order, slot = 4 j + c (see _V_SLOTS)."""
    B, H, n = code.shape[:3]
    if value:
        x = code.reshape(B, H, n, 4, 4, 32, 4)            # j, b, lane, c
        return x.permute(0, 1, 2, 5, 3, 6, 4).reshape(B, H, n, 32, 16, 4)
    x = code.reshape(B, H, n, 16, 2, 16, 4)                # t, half, s, b
    return x.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, H, n, 32, 16, 4)


def _v_slot(bits, s):
    """V slot s -> (channel c, token quad j)."""
    if bits == 4:
        return (s % 2, s // 2) if s < 8 else (2 + s % 2, (s - 8) // 2)
    if bits == 2:
        return s // 4, s % 4
    if s < 4:
        return 0, s
    if 6 <= s < 10:
        return 1, s - 6
    if s >= 12:
        return 3, s - 12
    return 2, {4: 0, 5: 1, 10: 2, 11: 3}[s]


def _slots(code, value, bits):
    """[B, H, n, 16, 128] -> [B, H, n, 32, 16 slots, 4] in the slot order of mlops_fbla.h."""
    x = _lanes(code, value)
    if value:
        order = [4 * j + c for c, j in (_v_slot(bits, s) for s in range(16))]
        x = x[..., order, :]
    return x


def _pack(a, bits):
    """[..., 32, 16 slots, 4] codes < 2^bits -> [..., 32, 2 bits words, 4 bytes] (bitwise per byte)."""
    a = a.to(torch.int32)
    if bits == 4:
        return a[..., :8, :] | a[..., 8:, :] << 4
    if bits == 2:
        return sum(a[..., 4 * m:4 * m + 4, :] << (2 * m) for m in range(4))
    w = a[..., :6, :] | a[..., 6:12, :] << 3
    w = w.clone()
    for r in range(4):
        c = a[..., 12 + r, :]
        w[..., r, :] |= (c & 3) << 6
        w[..., 4 + r // 2, :] |= (c >> 2) << (6 + r % 2)
    return w


def _record(kw, vw, bits):
    """K and V words [..., 32, 2 bits, 4] -> [..., 2 bits 256] record bytes (mlops_fbla.h)."""
    lead = kw.shape[:-3]
    kb, vb = kw.to(torch.uint8), vw.to(torch.uint8)
    if bits == 4:
        parts = [kb[..., :4, :], kb[..., 4:, :], vb[..., :4, :], vb[..., 4:, :]]
        return torch.cat([x.reshape(*lead, 512) for x in parts], -1)
    if bits == 2:
        return torch.cat([kb.reshape(*lead, 512), vb.reshape(*lead, 512)], -1)
    klo = kb[..., 4:, :].reshape(*lead, 2, 16, 8).transpose(-3, -2).reshape(*lead, 256)  # 16 t + 8 h
    return torch.cat([kb[..., :4, :].reshape(*lead, 512), vb[..., :4, :].reshape(*lead, 512), klo,
                      vb[..., 4:, :].reshape(*lead, 256)], -1)


def _meta(kbase, kstep, vbase, vstep):
    """[B, H, G, 128] x 4 -> [B, H, G, 512] BF16: per lane 4 channels of K base, K step, then V's."""
    lead = kbase.shape[:-1]
    k = torch.stack((kbase.reshape(*lead, 32, 4), kstep.reshape(*lead, 32, 4)), -2).reshape(*lead, 256)
    v = torch.stack((vbase.reshape(*lead, 32, 4), vstep.reshape(*lead, 32, 4)), -2).reshape(*lead, 256)
    return torch.cat((k, v), -1)


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


@torch.no_grad()
def encode(k: torch.Tensor, v: torch.Tensor, mode: str = "3+3", capacity: Optional[int] = None) -> FBLACache:
    """Encode dense ``[B, Hkv, N, 128]`` K / V into an :class:`FBLACache`."""
    if k.shape != v.shape or k.ndim != 4 or k.shape[-1] != 128:
        raise ValueError(f"k and v must have the same [B, H, N, 128] shape, got {tuple(k.shape)} and {tuple(v.shape)}")
    if not k.is_cuda or k.device != v.device:
        raise ValueError("k and v must be CUDA tensors on the same device")
    if mode not in _MODES:
        raise ValueError(f"mode must be '3+3' or '4+2', got {mode!r}")
    cb, fb, fmt = _MODES[mode]
    B, H, N, _ = k.shape
    capacity = N if capacity is None else capacity
    if capacity < N:
        raise ValueError(f"capacity {capacity} < tokens {N}")
    groups, cgroups = (N + 127) // 128, (capacity + 127) // 128
    kb, ks, kcode = _quantize(k, groups)
    vb, vs, vcode = _quantize(v, groups)
    kslots = _lanes(kcode.reshape(B, H, groups * 8, 16, 128), False)
    planes = []
    for bits, shift, mask in ((cb, fb, None), (fb, 0, (1 << fb) - 1)):
        kc = kslots >> shift if mask is None else kslots & mask
        vc = _slots(vcode.reshape(B, H, groups * 8, 16, 128), True, bits)
        vc = vc >> shift if mask is None else vc & mask
        rec = torch.zeros((B, H, cgroups * 8, 2 * bits * 256), device=k.device, dtype=torch.uint8)
        rec[:, :, :groups * 8] = _record(_pack(kc, bits), _pack(vc, bits), bits)
        planes.append(rec)
    meta = torch.zeros((B, H, cgroups, 512), device=k.device, dtype=torch.bfloat16)
    meta[:, :, :groups] = _meta(kb, ks, vb, vs)
    exact = []
    last = (groups - 1) * 128
    for x in (k, v):
        s_ = torch.zeros((B, H, 128, 128), device=k.device, dtype=torch.bfloat16)
        o = torch.zeros_like(s_)
        s_[:, :, :min(N, 128)] = x[:, :, :128].to(torch.bfloat16)
        o[:, :, :N - last] = x[:, :, last:].to(torch.bfloat16)
        exact += [s_, o]
    return FBLACache(planes[0], planes[1], meta, exact[0], exact[2], exact[1], exact[3], N, capacity, fmt)


_SCRATCH = {}  # (device, batch * kv_heads) -> zero-filled once, reused (mlops_fbla_decode_scratch_bytes)


def _scratch(device, heads, nbytes):
    key = (device, heads)
    s = _SCRATCH.get(key)
    if s is None or s.numel() < nbytes:
        s = torch.zeros(nbytes, dtype=torch.uint8, device=device)
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
