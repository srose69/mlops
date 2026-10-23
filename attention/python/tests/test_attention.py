# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""mlops.attention: the record layout against an independent decoder of include/mlops_fbla.h, the decode
kernel against a float64 model of the format's definition (as fbla/tests/test_fbla.cu), a cache laid out
for a larger capacity, and the distance to float64 attention."""
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _word_at(bits, tensor, lane, w):
    """byte offset of word w of K (0) or V (1) of a lane in a record, as mlops_fbla.h states it"""
    if bits == 4:
        return 1024 * tensor + (16 * lane + 4 * w if w < 4 else 512 + 16 * lane + 4 * (w - 4))
    if bits == 2:
        return 512 * tensor + 16 * lane + 4 * w
    if w < 4:
        return 512 * tensor + 16 * lane + 4 * w
    return 1280 + 8 * lane + 4 * (w - 4) if tensor else 1024 + 16 * (lane % 16) + 8 * (lane // 16) + 4 * (w - 4)


def _slot_bits(bits, s):
    """[(word, bit, nbits, code shift)] holding slot s"""
    if bits == 4:
        return [(s % 8, 4 * (s // 8), 4, 0)]
    if bits == 2:
        return [(s % 4, 2 * (s // 4), 2, 0)]
    if s < 12:
        return [(s % 6, 3 * (s // 6), 3, 0)]
    r = s - 12
    return [(r, 6, 2, 0), (4 + r // 2, 6 + r % 2, 1, 2)]


def _v_of(bits, s):
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


def _unpack(rec, bits):
    """[n, 2 bits 256] records -> K, V codes [n, 16 tokens, 128 channels]"""
    r = rec.long()
    n = r.shape[0]
    out = [torch.zeros(n, 16, 128, dtype=torch.long, device=r.device) for _ in range(2)]
    for tensor in range(2):
        for lane in range(32):
            for s in range(16):
                for b in range(4):
                    code = 0
                    for w, bit, nb, sh in _slot_bits(bits, s):
                        code = code | (((r[:, _word_at(bits, tensor, lane, w) + b] >> bit) & ((1 << nb) - 1)) << sh)
                    if tensor == 0:
                        t, c = lane % 16, 64 * (lane // 16) + 4 * s + b
                    else:
                        ch, j = _v_of(bits, s)
                        t, c = 4 * j + b, 4 * lane + ch
                    out[tensor][:, t, c] = code
    return out


@pytest.mark.parametrize("mode,cb,fb", [("3+3", 3, 3), ("4+2", 4, 2)])
def test_record_layout(mode, cb, fb):
    from mlops.attention import encode
    from mlops.attention.fbla import _quantize
    g = torch.Generator(device="cuda").manual_seed(1)
    k = torch.randn(1, 1, 512, 128, device="cuda", generator=g).bfloat16()
    v = torch.randn(1, 1, 512, 128, device="cuda", generator=g).bfloat16()
    c = encode(k, v, mode)  # 4 groups: 0 .. 2 closed (the GPU quantizer), 3 open
    for x, tensor in ((k, 0), (v, 1)):
        base, step, code = _quantize(x, 4)
        code = code[0, 0, :384].reshape(24, 16, 128).long()
        coarse, fine = _unpack(c.coarse[0, 0, :24], cb)[tensor], _unpack(c.fine[0, 0, :24], fb)[tensor]
        assert torch.equal(coarse, code >> fb) and torch.equal(fine, code & ((1 << fb) - 1))
        m = c.meta[0, 0, :3].reshape(3, 2, 32, 2, 4)[:, tensor]  # group, lane, (base, step), channel
        assert torch.equal(m[:, :, 0].reshape(3, 128), base[0, 0, :3]) and torch.equal(m[:, :, 1].reshape(3, 128), step[0, 0, :3])


def _case(N, kv_heads=2, q_heads=4, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    chan = 0.2 + 1.5 * torch.rand(128, device="cuda", generator=g)
    k = (torch.randn(1, kv_heads, N, 128, device="cuda", generator=g) * chan + (torch.arange(128, device="cuda") % 7 == 0) * 3)
    v = torch.randn(1, kv_heads, N, 128, device="cuda", generator=g)
    q = 1.5 * torch.randn(1, q_heads, 128, device="cuda", generator=g)
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


@pytest.mark.parametrize("mode,cb,fb", [("3+3", 3, 3), ("4+2", 4, 2)])
@pytest.mark.parametrize("tau", [1e30, 3e-3, 1e-30])
def test_decode_matches_model(mode, cb, fb, tau):
    from mlops.attention import decode, encode
    q, k, v = _case(1000)
    got = decode(q, encode(k, v, mode), tau=tau)
    assert _rel(got, _model(q, k, v, cb, fb, tau)) < 4e-3  # BF16 output rounding


def _model(q, k, v, cb, fb, tau):
    """float64 model of the definition (as fbla/tests/test_fbla.cu), decode at position N - 1."""
    B, Hq, D = q.shape
    _, Hk, N, _ = k.shape
    G, S, groups = 128, 16, (N + 127) // 128
    from mlops.attention.fbla import _quantize
    out = torch.zeros(B, Hq, D, dtype=torch.float64, device=q.device)
    sl = torch.tensor(1 / math.sqrt(D), dtype=torch.float32) * torch.tensor(1.4426950408889634, dtype=torch.float32)  # FP32, as the kernel
    hoff = .5 * ((1 << fb) - 1)
    for hk in range(Hk):
        kb, ks, kc = _quantize(k[:, hk:hk + 1], groups)
        vb, vs, vc = _quantize(v[:, hk:hk + 1], groups)
        for hq in range(hk * (Hq // Hk), (hk + 1) * (Hq // Hk)):
            qs = q[0, hq].float() * sl.to(q.device)
            sc = torch.full((groups * G,), -math.inf, dtype=torch.float64, device=q.device)
            sf = sc.clone()
            vals_c = torch.zeros(groups * G, D, dtype=torch.float64, device=q.device)
            vals_f = vals_c.clone()
            for g in range(groups):
                n = min(G, N - g * G)
                t = slice(g * G, g * G + n)
                if g == 0 or g == groups - 1:
                    s = (qs.double() @ k[0, hk, t].double().T)
                    sc[t] = sf[t] = s
                    vals_c[t] = vals_f[t] = v[0, hk, t].double()
                    continue
                z = qs * ks[0, 0, g].float()
                amax = z.abs().max()
                sq = amax * torch.tensor(1 / 127, dtype=torch.float32, device=z.device)
                qh = torch.round(z * (127 / amax)) if amax > 0 else torch.zeros_like(z)
                code = kc[0, 0, t].double()
                qb = (qs.double() * kb[0, 0, g].double()).sum()
                sc[t] = qb + sq.double() * ((1 << fb) * (qh.double() @ torch.floor(code / (1 << fb)).T) + hoff * qh.sum())
                sf[t] = qb + sq.double() * (qh.double() @ code.T)
                cv = vc[0, 0, t].double()
                vals_f[t] = vb[0, 0, g].double() + vs[0, 0, g].double() * cv
                vals_c[t] = vb[0, 0, g].double() + vs[0, 0, g].double() * (torch.floor(cv / (1 << fb)) * (1 << fb) + hoff)
            nb = groups * 8
            lse = torch.full((nb,), -math.inf, dtype=torch.float64, device=q.device)
            for j in range(nb):
                s = sc[j * S:(j + 1) * S]
                if torch.isfinite(s).any():
                    mb = s.max()
                    lse[j] = mb + torch.log2(torch.round(255 * torch.exp2(s - mb)).sum())
            R = torch.logsumexp(lse * math.log(2), 0) / math.log(2)
            O, L = torch.zeros(D, dtype=torch.float64, device=q.device), 0.0
            for j in range(nb):
                if not torch.isfinite(lse[j]):
                    continue
                g = j // 8
                fine = 0 < g < groups - 1 and 2 ** (lse[j] - R).item() > tau
                s = (sf if fine else sc)[j * S:(j + 1) * S]
                mb = s.max()
                p = torch.where(torch.isfinite(s), torch.round(255 * torch.exp2(s - mb)), torch.zeros_like(s))
                w = 2 ** (mb - R)
                L = L + w * p.sum()
                O += w * (p @ (vals_f if fine else vals_c)[j * S:(j + 1) * S])
            out[0, hq] = O / L
    return out


@pytest.mark.parametrize("mode", ["3+3", "4+2"])
def test_capacity_layout(mode):
    from mlops.attention import decode, encode
    q, k, v = _case(3000, seed=3)
    a = decode(q, encode(k, v, mode), tau=1e-3)
    b = decode(q, encode(k, v, mode, capacity=8192), tau=1e-3)
    assert torch.equal(a, b)


@pytest.mark.parametrize("mode", ["3+3", "4+2"])
def test_decode_close_to_exact_attention(mode):
    from mlops.attention import decode, encode
    q, k, v = _case(4096, seed=5)
    got = decode(q, encode(k, v, mode)).double()
    g = q.shape[1] // k.shape[1]
    s = (q.double()[:, :, None] @ k.double().repeat_interleave(g, 1).transpose(-1, -2))[:, :, 0] / math.sqrt(128)
    exact = torch.softmax(s, -1)[:, :, None] @ v.double().repeat_interleave(g, 1)
    assert _rel(got, exact[:, :, 0]) < 0.05  # the 6-bit format's error on random data


@pytest.mark.parametrize("mode", ["3+3", "4+2"])
def test_append_matches_encode(mode):
    """A cache grown token by token (closing groups on the way, starting inside the sink group) decodes exactly
    like one encoded from scratch."""
    from mlops.attention import append, decode, encode
    q, k, v = _case(700, seed=7)
    c = encode(k[:, :, :100], v[:, :, :100], mode, capacity=1024)
    for t in range(100, 700):
        append(c, k[:, :, t:t + 1], v[:, :, t:t + 1])
        if t in (127, 128, 255, 256, 699):
            assert torch.equal(decode(q, c, tau=1e-3), decode(q, encode(k[:, :, :t + 1], v[:, :, :t + 1], mode), tau=1e-3))
    with pytest.raises(ValueError):
        append(c, k[:, :, :400], v[:, :, :400])  # 700 + 400 > capacity 1024


@pytest.mark.parametrize("mode", ["3+3", "4+2"])
def test_prefill_matches_decode(mode):
    """Chunked prefill from an empty cache: every position's output against append + decode of that position
    with every block refined (the same attention; the two kernels round differently, so within BF16 output
    rounding), and the cache it leaves against encode's."""
    from mlops.attention import append, decode, empty, encode, prefill
    _, k, v = _case(900, seed=11)
    q = (1.5 * torch.randn(1, 4, 900, 128, device="cuda", generator=torch.Generator(device="cuda").manual_seed(12))).bfloat16()
    c = empty(1, 2, 1024, mode)
    outs = [prefill(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], c) for a, b in ((0, 300), (300, 384), (384, 900))]
    out = torch.cat(outs, 2)
    d = empty(1, 2, 1024, mode)
    for t in range(900):
        append(d, k[:, :, t:t + 1], v[:, :, t:t + 1])
        if t % 61 == 0 or t in (127, 128, 383, 384, 899):
            ref = decode(q[:, :, t], d, tau=1e-30)
            assert _rel(out[:, :, t], ref) < 4e-3, t
    e = encode(k, v, mode, capacity=1024)
    assert c.tokens == e.tokens == 900
    for x, y in ((c.coarse, e.coarse), (c.fine, e.fine), (c.meta, e.meta), (c.open_k, e.open_k), (c.sink_v, e.sink_v)):
        assert torch.equal(x, y)
