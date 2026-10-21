# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""PyTorch reference of mlops_adamw_qat (include/mlops_adamw_qat.h, adamw/src/adamw_qat_kernels.cuh).

It steps the same objects the CUDA kernel steps -- an ``nvfp4effma.NVFP4Tensor`` (rowwise, 1x16 blocks,
E4M3 scales, linear or swizzled, amax or decode global) and the uint8 state buffer of
``mlops.optim.functional.QATState`` -- in the same arithmetic order, with the same stochastic-rounding stream, so the
two can be swapped on one state. FMA is emulated in float64 (a rare double rounding differs in the last
bit). Readable rather than fast: one tensor per quantity, the kernel's per-thread loops written out.

    st = mlops.optim.functional.QATState(w)
    reference.step(st.weight, st.state, grad, lr=1e-3, step=1)   # instead of st.step(grad, lr=1e-3, step=1)
"""
import math
from typing import Tuple

import torch

M32 = 0xFFFFFFFF
B = 16          # block size
Q = 4           # elements per kernel thread (4 threads per block)


# ---- float32 / uint32 primitives -----------------------------------------------------------------------
def _f(x, dev):
    return torch.as_tensor(x, dtype=torch.float32, device=dev)


def fma(a, b, c):
    """a * b + c rounded once to float32 (float64 intermediate: the product is exact)."""
    return (a.double() * torch.as_tensor(b).double().to(a.device) + torch.as_tensor(c).double().to(a.device)).float()


def bits(x):
    return x.contiguous().view(torch.int32).long() & M32


def flt(u):
    u = u & M32
    return torch.where(u >= 2 ** 31, u - 2 ** 32, u).to(torch.int32).view(torch.float32)


def pow2(e):
    return flt((e + 127) << 23)


def expo(x):
    return ((bits(x) >> 23) & 255) - 127


def mix(x):  # (out of place: callers keep their inputs)
    x = x & M32
    x = x ^ (x >> 16); x = (x * 0x7FEB352D) & M32
    x = x ^ (x >> 15); x = (x * 0x846CA68B) & M32
    return x ^ (x >> 16)


def xorshift(x):
    x = x ^ ((x << 13) & M32); x = x ^ (x >> 17)
    return x ^ ((x << 5) & M32)


def step_key(seed: int, step: int) -> int:
    return int(mix(torch.tensor((seed ^ ((step * 0x9E3779B9) & M32)) & M32)))


# ---- formats ---------------------------------------------------------------------------------------------
def dec_e4m3(b):
    return b.to(torch.uint8).view(torch.float8_e4m3fn).float()


def dec_e5m2(b):
    return b.to(torch.uint8).view(torch.float8_e5m2).float()


def dec_bf16(b):
    return flt((b.long() & 0xFFFF) << 16)


def e2m1_values(dev):
    return torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)


def sr_floor(t, U):
    f = torch.floor(t)
    return torch.where(t - f > U, f + 1, f)


def sr_fixed(x, one, lo, hi, U):
    return sr_floor(x * one, U).clamp(lo, hi)


def enc_e5m2_exact(a):
    """E5M2 byte of an exactly representable a >= 0."""
    e = expo(torch.where(a > 0, a, torch.ones_like(a)))
    normal = ((e + 15) << 2) | ((bits(a) >> 21) & 3)
    return torch.where(a == 0, torch.zeros_like(e), torch.where(a < 2 ** -14, (a * 65536).long(), normal))


def sr_e4m3(x, U):
    a = x.abs()
    s = torch.where(x < 0, 0x80, 0)
    lb = bits(a) & 0xFFF00000
    lo, hi = flt(lb), flt(lb + 0x100000)
    normal = (lb >> 20) - 960 + (U * (hi - lo) < a - lo).long()
    sub = (sr_floor(a * 512, U)).long()                       # subnormal grid 2^-9: code = value * 2^9
    code = torch.where(a >= 2 ** -6, normal, sub)
    return torch.where(a < 448, s | code, s | 0x7E)


def sr_e5m2_sq(x, vt, U):
    """SR into E5M2 with E[decoded^2] = vt (x ~ sqrt(vt) seeds the bracket)."""
    # normal bracket
    lb = bits(x) & 0xFFE00000
    lo, hi = flt(lb), flt(lb + 0x200000)
    down, upw = vt < lo * lo, vt > hi * hi
    lb = torch.where(down, lb - 0x200000, torch.where(upw, lb + 0x200000, lb))
    lo, hi = flt(lb), flt(lb + 0x200000)
    c = (lb >> 21) - 448
    lo2, hi2 = lo * lo, hi * hi
    fast = torch.where(c >= 123, torch.full_like(c, 123), c + (U * (hi2 - lo2) < vt - lo2).long())
    # small x: the generic bracket
    sub = x < 2 ** -14
    ex = expo(torch.where(sub, torch.ones_like(x), x))
    q = torch.where(sub, _f(2 ** -16, x.device), pow2(ex - 2))
    qi = torch.where(sub, _f(2 ** 16, x.device), pow2(2 - ex))
    c2 = enc_e5m2_exact(torch.floor(x * qi) * q)
    d = dec_e5m2(c2.clamp(0, 123))
    c2 = torch.where((c2 > 0) & (vt < d * d), c2 - 1, c2)
    d1 = dec_e5m2((c2 + 1).clamp(0, 123))
    c2 = torch.where((c2 < 123) & (vt > d1 * d1), c2 + 1, c2)
    lo, hi = dec_e5m2(c2.clamp(0, 123)), dec_e5m2((c2 + 1).clamp(0, 123))
    lo2, hi2 = lo * lo, hi * hi
    slow = torch.where(c2 >= 123, torch.full_like(c2, 123), c2 + (U * (hi2 - lo2) < vt - lo2).long())
    out = torch.where(x >= 2 ** -13, fast, slow)
    return torch.where(x < 57344, out, torch.full_like(out, 123))


def sr_bf16(x, U):
    b = bits(x)
    return (b >> 16) + ((b & 0xFFFF) > (U * 65536).long()).long()


def rsqrt_quake(x):
    y = flt(0x5F1FFFF9 - (bits(x) >> 1))
    return 0.703952253 * (y * fma(-x, y * y, 2.38924456))


def g_of(u):
    a = u.abs()
    return torch.copysign(torch.where(a < 4, 0.5 * a, torch.where(a < 6, a - 2, fma(_f(2, u.device).expand_as(a), a - 6, 4.0))), u)


def g_inv(y):
    a = y.abs()
    return torch.copysign(torch.where(a < 2, 2 * a, torch.where(a < 4, a + 2, fma(_f(.5, y.device).expand_as(a), a - 4, 6.0))), y)


def e2m1_round(a):
    k = torch.where(a < 2, 2.0, torch.where(a < 4, 1.0, 0.5)).float()
    return torch.clamp(((a * k + 8388608.0) - 8388608.0) * (1 / k), max=6.0)


def huber(e, d):
    a = e.abs()
    return torch.where(a <= d, (0.5 * a) * a, d * fma(_f(-0.5, e.device).expand_as(d), d, a))


# ---- tensor layout ---------------------------------------------------------------------------------------
def _align16(n):
    return (n + 15) & ~15


def carve(state: torch.Tensor, rows: int, cols: int) -> dict:
    """Views of the state buffer, in mlops_adamw_qat_state_bind's order."""
    n, nb = rows * cols, rows * cols // B
    out, p = {}, 0
    for name, size, dt in (("u", 2 * n, torch.int16), ("r", n, torch.uint8), ("s", n, torch.uint8),
                           ("qbar", n, torch.int8), ("s_exp", nb, torch.int8), ("cusum", 2 * nb, torch.int16),
                           ("hnum", 2 * nb, torch.int16), ("hden", 2 * nb, torch.int16), ("next", nb, torch.uint8),
                           ("gdec", 4 * rows, torch.float32)):
        out[name] = state[p:p + size].view(dt)
        p += _align16(size)
    return out


def scale_index(q, rows: int, nb: int, dev) -> torch.Tensor:
    """Flat byte index of block scale (r, b) for every block, row-major."""
    r = torch.arange(rows, device=dev).repeat_interleave(nb)
    b = torch.arange(nb, device=dev).repeat(rows)
    if not q.swizzled:
        return r * q.scales.shape[-1] + b
    pc = (nb + 3) // 4 * 4
    return ((r >> 7) * (pc >> 2) << 9) + ((r & 31) << 4) + (((r >> 5) & 3) << 2) + ((b >> 2) << 9) + (b & 3)


def gdec_of(q, rows: int, dev) -> torch.Tensor:
    """Decode factor of each row's global, as nvfp4a_quantize_ex encodes it."""
    g = q.global_scale if q.global_scale is not None else torch.ones(1, device=dev)
    g = (g if g.numel() > 1 else g.expand(rows)).float()
    if not q.global_is_amax:
        return g
    smax = float(q.scale_max or 448)
    s_enc = torch.clamp(_f(smax * 6.0, dev) / g, max=3.4028234663852886e38)
    return torch.where((g == 0) | (s_enc == 0), torch.ones_like(g), 1.0 / s_enc)


# ---- the step --------------------------------------------------------------------------------------------
@torch.no_grad()
def step(q, state: torch.Tensor, grad: torch.Tensor, lr: float, step: int, betas: Tuple[float, float] = (0.9, 0.999),
         eps: float = 1e-12, weight_decay: float = 0.0, grad_scale: float = 1.0, scale_z: float = 3.0,
         seed: int = 0) -> None:
    """One mlops_adamw_qat step on (q, state), in place. grad: float32 or float8 (times grad_scale)."""
    dev = state.device
    rows, cols = grad.shape
    nb, nblk = cols // B, rows * cols // B
    st = carve(state, rows, cols)
    F = lambda x: _f(x, dev)

    # host constants, as mlops_adamw_qat computes them (double, rounded to float once)
    b1d, b2d = float(F(betas[0])), float(F(betas[1]))
    z = scale_z if scale_z > 0 else 3.0
    period = max(1, round((1 + b1d) / (1 - b1d)))
    lr_, b1, omb1, b2, omb2 = F(lr), F(b1d), F(1 - b1d), F(b2d), F(1 - b2d)
    eps2 = F(float(F(eps)) ** 2)
    bc = F(math.sqrt(1 - b2d ** step) / (1 - b1d ** step))
    wd, gs = F(weight_decay), F(grad_scale if grad_scale != 0 else 1.0)
    rbc1 = F(1 / (1 - b1d ** step))
    rbc1p = F(1 / (1 - b1d ** (step - 1)) if step > 1 else 0.0)
    hq = round(4 * z * z * 256)
    sig = F(z * z * (1 - b2d) / (1 + b2d) * (1 + b2d ** step) / (1 - b2d ** step))
    phase = step % period
    key = step_key(seed, step)

    # lattice
    sidx = scale_index(q, rows, nb, dev)
    scales = q.scales.view(torch.uint8).reshape(-1)
    row = torch.arange(rows, device=dev).repeat_interleave(nb)
    bcol = torch.arange(nb, device=dev).repeat(rows)
    gd = gdec_of(q, rows, dev)[row]                                       # [nblk]
    kb = scales[sidx].long()
    sm = dec_e4m3(kb)
    sigma = sm * gd
    left = scales[scale_index(q, rows, nb, dev).view(rows, nb)[row, (bcol - 1).clamp(min=0)]].long()
    right = scales[scale_index(q, rows, nb, dev).view(rows, nb)[row, (bcol + 1).clamp(max=nb - 1)]].long()
    im = 1.0 / sigma
    i_left, i_right = 1.0 / (dec_e4m3(left) * gd), 1.0 / (dec_e4m3(right) * gd)

    # element state, [nblk, 16]
    data = q.data.reshape(-1)
    nib = torch.stack([data & 15, data >> 4], -1).reshape(nblk, B).long()
    qn = e2m1_values(dev)[nib]
    g = (grad.float().reshape(nblk, B)) * gs
    u = st["u"].reshape(nblk, B).float() * 2 ** -12
    s_old = dec_e5m2(st["s"].reshape(nblk, B)) * pow2(st["s_exp"].long())[:, None]
    r_old = dec_e4m3(st["r"].reshape(nblk, B))
    qb_old = st["qbar"].reshape(nblk, B).float() * 0.0625
    m_old = r_old * s_old
    d = sigma[:, None] * fma(-qb_old, rbc1p, qn)
    v = fma(b2.expand_as(s_old), s_old * s_old, omb2 * (g * g))
    y = rsqrt_quake(v + eps2)
    x = v * y
    rn = fma(b1.expand_as(m_old), m_old, omb1 * g) * y
    qb = fma(b1.expand_as(qb_old), qb_old, omb1 * qn)
    j = torch.arange(B, device=dev)
    f = torch.where(j < 8, (j + 8.5) * 0.0625, (j - 7.5) * 0.0625).float()
    i_lo = torch.where(j < 8, i_left[:, None], im[:, None])
    i_hi = torch.where(j < 8, im[:, None], i_right[:, None])
    inv_sig = fma(f.expand_as(i_lo), i_hi - i_lo, i_lo)
    au = u.abs()
    inv_slope = torch.where(au < 4, 2.0, torch.where(au < 6, 1.0, 0.5)).float()
    stp = lr_ * fma(rn, bc, wd * (sigma[:, None] * g_of(u)))
    uf = u - (stp * inv_slope) * inv_sig

    # per-thread partial sums, then the two shuffle adds: (p0 + p1) + (p2 + p3)
    def quad(p):  # p: [nblk, 4] per-thread partials
        return (p[:, 0] + p[:, 1]) + (p[:, 2] + p[:, 3])
    gq, dq, vq = g.view(nblk, 4, Q), d.view(nblk, 4, Q), v.view(nblk, 4, Q)
    ped = torch.zeros(nblk, 4, device=dev); pdd = torch.zeros_like(ped); pv = torch.zeros_like(ped)
    for i in range(Q):
        ped = fma(gq[..., i], dq[..., i], ped)
        pdd = fma(dq[..., i], dq[..., i], pdd)
        pv = pv + vq[..., i]
    sum_ed, sum_dd, sum_v = quad(ped), quad(pdd), quad(pv)
    xmax = x.max(1).values

    # stochastic-rounding stream: one per thread, three xorshift words per element
    tid = torch.arange(nblk * 4, device=dev).view(nblk, 4)
    rng = mix(torch.full_like(tid, key) ^ tid)
    Uu, Us, Ur, Uqb = [torch.empty(nblk, 4, Q, device=dev) for _ in range(4)]
    for i in range(Q):
        rng = xorshift(rng); w0 = rng
        rng = xorshift(rng); w1 = rng
        rng = xorshift(rng); w2 = rng
        Uu[..., i] = (w0 >> 8).float() * 2 ** -24
        Us[..., i] = (w1 >> 8).float() * 2 ** -24
        Ur[..., i] = (w2 & 0xFFFF).float() * 2 ** -16
        Uqb[..., i] = (w2 >> 16).float() * 2 ** -16
    Uu, Us, Ur, Uqb = (t.reshape(nblk, B) for t in (Uu, Us, Ur, Uqb))

    E = torch.where(xmax > 0, (expo(torch.where(xmax > 0, xmax, torch.ones_like(xmax))) + 1 - 12).clamp(-100, 100),
                    torch.zeros_like(xmax, dtype=torch.long))
    pi = pow2(-E)[:, None]
    s_new = sr_e5m2_sq(x * pi, (v * pi) * pi, Us)
    r_new = sr_e4m3(rn, Ur)

    # block scale
    num = fma(b2.expand_as(sum_ed), dec_bf16(st["hnum"]), omb2 * sum_ed)
    den = fma(b2.expand_as(sum_dd), dec_bf16(st["hden"]), omb2 * sum_dd)
    blk = torch.arange(nblk, device=dev)
    votes = (phase + (blk >> 6)) % period == 0
    ok = votes & (num > 0) & (den > 0) & (num * num > (sig * (sum_v * 0.0625)) * den)
    inv_h = den / num
    ws = (fma(qb, sigma[:, None], -((rn * x) * inv_h[:, None])) * rbc1).abs()
    best = torch.full((nblk,), float("inf"), device=dev)
    kbest = kb.clone()
    for o in range(-8, 9):
        c = (kb + o).clamp(1, 126)
        sc = dec_e4m3(c) * gd
        isc = 1.0 / sc
        dl = sigma * isc
        a = ws * isc[:, None]
        h = huber(a - e2m1_round(a), dl[:, None]).view(nblk, 4, Q)
        p = torch.zeros(nblk, 4, device=dev)
        for i in range(Q):
            p = p + h[..., i]
        D = quad(p) * (sc * sc)
        better = D < best
        best = torch.where(better, D, best)
        kbest = torch.where(better, c, kbest)
    vote = torch.where(ok, torch.sign(kbest - kb), torch.zeros_like(kb))
    S = st["cusum"].long() + 256 * period * vote
    mag = S.abs() - 16 * period
    S = torch.where(mag > 0, torch.sign(S) * mag, torch.zeros_like(S))
    move = votes & (S.abs() > hq)
    kn = torch.where(move, (kb + torch.sign(S)).clamp(1, 126), kb)
    S = torch.where(move, torch.zeros_like(S), S)
    st["cusum"].copy_(torch.where(votes, S, st["cusum"].long()).to(torch.int16))
    st["next"].copy_(torch.where(votes, kn, st["next"].long()).to(torch.uint8))

    # block draws (thread 0's stream after its four elements)
    w = xorshift(rng[:, 0] ^ 0xB10C5CA1)
    hn = sr_bf16(num, (w >> 8).float() * 2 ** -24)
    hd = sr_bf16(den, (xorshift(w) >> 8).float() * 2 ** -24)
    st["hnum"].copy_(torch.where(hn >= 2 ** 15, hn - 2 ** 16, hn).to(torch.int16))
    st["hden"].copy_(torch.where(hd >= 2 ** 15, hd - 2 ** 16, hd).to(torch.int16))
    st["s_exp"].copy_(E.to(torch.int8))

    # a move keeps the master values
    moved = (kn != kb)[:, None]
    ratio = (sm / dec_e4m3(kn))[:, None]
    uf = torch.where(moved, g_inv(g_of(uf) * ratio), uf)
    qb = torch.where(moved, qb * ratio, qb)
    un = sr_fixed(uf, 4096.0, -32768.0, 32767.0, Uu)
    qq = sr_fixed(qb, 16.0, -128.0, 127.0, Uqb)
    k = torch.clamp(torch.round(un * 2 ** -12).abs(), max=7.0).long()  # roundf: half away from zero
    k = torch.where((un * 2 ** -12).abs() - torch.floor((un * 2 ** -12).abs()) == 0.5,
                    torch.clamp(torch.floor((un * 2 ** -12).abs()) + 1, max=7.0).long(), k)
    nibble = torch.where(k == 0, 0, torch.where(un < 0, 8, 0)) | k

    st["s"].copy_(s_new.reshape(-1).to(torch.uint8))
    st["r"].copy_(r_new.reshape(-1).to(torch.uint8))
    st["u"].copy_(un.reshape(-1).to(torch.int16))
    st["qbar"].copy_(qq.reshape(-1).to(torch.int8))
    nibble = nibble.reshape(-1, 2)
    q.data.view(-1).copy_((nibble[:, 0] | (nibble[:, 1] << 4)).to(torch.uint8))
    scales[sidx] = torch.where(votes, kn, kb).to(torch.uint8)
