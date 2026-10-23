# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""FBLA (Flash-Block Attention): numerical model of the KV format and of the kernel's arithmetic, in PyTorch.

Format (K and V alike, per KV head). Tokens form groups of G = 128; per group and channel two bf16, the base (the
group's minimum) and the step, and per element one code c in 0 .. 2^(cb + fb) - 1 (cb = 4, fb = 2):

    x[t, c] = base[c] + code[t, c] * step[c]

stored as two planes: the coarse plane (the top cb bits of the code) and the refinement plane (the low fb bits).
Without its refinement an element decodes to the middle of its coarse cell. The first group (attention sink) and
the group still being filled (decode) are kept in bf16. Per element: cb + 32 / G bits always read, fb bits only
where they matter.

Kernel arithmetic, per 16-token sub-block j of KV group g, for every query (log2 domain):

    q' = q * scale * log2 e * step_K[g]  ->  int8 q_hat, sq        (once per query and group of 128)
    s[t] = q . base_K[g]  +  sq * sum_c q_hat[c] * code[t, c]       coarse: code = (hi << fb) + (2^fb - 1) / 2
    mb = max_t s;  m' = max(m, mb);  alpha = 2^(m - m');  p_blk = 2^(mb - m')
    refine (load the fb plane of K and V) when p_blk > tau: the sub-block's weight relative to the running maximum
    ("mass"), or, after a coarse pass over the whole cache ("norm"), when the sub-block's share of the softmax mass
    W_j = 2^(lse_j - LSE) > tau (split-K decode: coarse pass per chunk, a reduction, then the refining pass)
    p_hat = round(255 * 2^(s[t] - mb));  l = alpha l + p_blk / 255 * sum p_hat
    o[c] = alpha o[c] + p_blk / 255 * (sum_t p_hat * base_V[c] + step_V[c] * sum_t p_hat[t] * code_V[t, c])

Integer sums are float32 matmuls, exact below 2^24 (127 * 2 * 63 * D for QK^T, 255 * 63 * 16 for PV). The coarse
code's half-cell offset is kept integral by doubling. o / l at the end (weights normalized by their quantized sum).
"""
from dataclasses import dataclass
import math

import torch

LOG2E = 1.4426950408889634


@dataclass(frozen=True)
class FBLAConfig:
    group: int = 128                # tokens per (base, step) group
    sub: int = 16                   # tokens per refinement decision and per P block
    coarse_bits: int = 4
    fine_bits: int = 2
    sink_groups: int = 1            # leading groups kept in bf16
    open_group: bool = True         # keys in the query's own group (still being filled in decode) read in bf16
    refine: str = "mass"            # "mass" (p_blk > tau), "norm" (normalized mass > tau), "all", "none"
    tau: float = 0.01
    q_bits: int = 8
    p_bits: int = 8

    def name(self):
        r = {"all": "all refined", "none": "coarse only", "norm": f"refine W_j > {self.tau:g}"}.get(
            self.refine, f"refine p_blk > {self.tau:g}")
        return f"G={self.group} {self.coarse_bits}+{self.fine_bits} b, {r}"


def _bf16(x, toward):
    """x rounded to bf16 toward -inf (toward < 0) or +inf (toward > 0), as float32."""
    y = x.to(torch.bfloat16)
    bad = (y.float() > x) if toward < 0 else (y.float() < x)
    return torch.where(bad, torch.nextafter(y, torch.full_like(y, -math.inf if toward < 0 else math.inf)), y).float()


def encode(x: torch.Tensor, cfg: FBLAConfig) -> dict:
    """x [..., N, D] (N a multiple of G) -> base / step [..., N/G, 1, D] (bf16 values), code [..., N, D] (full).
    base is the minimum rounded down and step the range over the stored base rounded up, so every code is in
    0 .. 2^(cb + fb) - 1 without clipping."""
    *lead, N, D = x.shape
    G = cfg.group
    xb = x.float().reshape(*lead, N // G, G, D)
    lo, hi = xb.amin(-2, keepdim=True), xb.amax(-2, keepdim=True)
    L = 2 ** (cfg.coarse_bits + cfg.fine_bits) - 1
    base = _bf16(lo, -1)
    step = _bf16((hi - base) / torch.full_like(hi, L), +1)  # IEEE division (tensor / scalar multiplies by 1 / L)
    code = torch.where(step > 0, torch.round((xb - base) / step.clamp_min(1e-30)), torch.zeros_like(xb)).clamp(0, L)
    return dict(base=base, step=step, code=code.reshape(*lead, N, D))


def coarse2(code, fb):
    """Twice the coarse code: 2 * ((code >> fb) << fb) + 2^fb - 1 (an integer: the middle of the coarse cell)."""
    return 2 * torch.floor(code / 2 ** fb) * 2 ** fb + (2 ** fb - 1)


@torch.no_grad()
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cfg: FBLAConfig = FBLAConfig(),
              scale: float = None, stats: dict = None) -> torch.Tensor:
    """Causal FBLA forward. q [..., M, D] at positions N - M .. N - 1, k / v [..., N, D] (N a multiple of G)
    -> o [..., M, D] float32. `stats` (a dict) accumulates the refined / quantized (query, sub-block) pairs."""
    *lead, M, D = q.shape
    N, G, S, fb = k.shape[-2], cfg.group, cfg.sub, cfg.fine_bits
    if N % G:
        raise ValueError(f"N = {N} must be a multiple of the group ({G})")
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    K, V = encode(k, cfg), encode(v, cfg)
    kc2, vc2 = coarse2(K["code"], fb), coarse2(V["code"], fb)
    qs = q.float() * (scale * LOG2E)
    qm = 2 ** (cfg.q_bits - 1) - 1
    pq = float(2 ** cfg.p_bits - 1)
    dev = q.device
    qpos = torch.arange(N - M, N, device=dev)[:, None]                   # [M, 1]
    qgrp = qpos // G
    m = torch.full((*lead, M, 1), -math.inf, device=dev)
    l = torch.zeros((*lead, M, 1), device=dev)
    o = torch.zeros((*lead, M, D), device=dev)
    kf, vf = k.float(), v.float()
    n_ref = n_q = 0
    log2 = math.log(2)
    if cfg.refine == "norm":
        # Coarse pass: retain only the global row LSE. Keeping qh for every group and every sub-block LSE costs
        # O(MND/G + MN/S) memory; recomputing them in the refining pass makes this O(M) and mirrors split-K decode.
        global_lse = torch.full((*lead, M, 1), -math.inf, device=dev)
        for g in range(N // G):
            sK = K["step"][..., g, :, :]
            qp = qs * sK
            sq = qp.abs().amax(-1, keepdim=True) / qm
            qh = torch.where(sq > 0, torch.round(qp / sq.clamp_min(1e-30)), torch.zeros_like(qp)).clamp(-qm, qm)
            qb = qs @ K["base"][..., g, :, :].transpose(-1, -2)
            exact_g = (g < cfg.sink_groups) | (cfg.open_group & (qgrp == g))
            for j in range(g * G, (g + 1) * G, S):
                sl = slice(j, j + S)
                masked = torch.arange(j, j + S, device=dev)[None, :] > qpos
                s_c = qb + sq * 0.5 * (qh @ kc2[..., sl, :].transpose(-1, -2))
                s_e = qs @ kf[..., sl, :].transpose(-1, -2)
                sc = torch.where(exact_g, s_e, s_c).masked_fill(masked, -math.inf)
                block_lse = torch.logsumexp(sc * log2, -1, keepdim=True)
                global_lse = torch.logaddexp(global_lse, block_lse)
    for g in range(N // G):
        bK, sK = K["base"][..., g, :, :], K["step"][..., g, :, :]       # [.., 1, D]
        bV, sV = V["base"][..., g, :, :], V["step"][..., g, :, :]
        qp = qs * sK                                                     # q' = q step_K as int8, q . base_K
        sq = qp.abs().amax(-1, keepdim=True) / qm
        qh = torch.where(sq > 0, torch.round(qp / sq.clamp_min(1e-30)), torch.zeros_like(qp)).clamp(-qm, qm)
        qb = qs @ bK.transpose(-1, -2)
        exact_g = (g < cfg.sink_groups) | (cfg.open_group & (qgrp == g))  # [M, 1]
        for j in range(g * G, (g + 1) * G, S):
            sl = slice(j, j + S)
            masked = torch.arange(j, j + S, device=dev)[None, :] > qpos  # [M, S]
            s_f = qb + sq * (qh @ K["code"][..., sl, :].transpose(-1, -2))
            s_c = qb + sq * 0.5 * (qh @ kc2[..., sl, :].transpose(-1, -2))
            s_e = qs @ kf[..., sl, :].transpose(-1, -2)
            s_f, s_c, s_e = (x.masked_fill(masked, -math.inf) for x in (s_f, s_c, s_e))
            # refinement decision on the coarse scores, against the running maximum
            mbc = s_c.amax(-1, keepdim=True)
            pre = torch.exp2(mbc - torch.maximum(m, mbc)).nan_to_num(0.0)
            if cfg.refine == "norm":
                block_lse = torch.logsumexp(s_c * log2, -1, keepdim=True)
                ref = torch.exp(block_lse - global_lse) > cfg.tau
            else:
                ref = {"all": torch.ones_like(pre, dtype=torch.bool), "none": torch.zeros_like(pre, dtype=torch.bool)}.get(
                    cfg.refine, pre > cfg.tau)
            quant = torch.isfinite(mbc) & ~exact_g
            n_q += int(quant.sum()); n_ref += int((quant & ref).sum())
            sc = torch.where(exact_g, s_e, torch.where(ref, s_f, s_c))
            mb = sc.amax(-1, keepdim=True)
            live = torch.isfinite(mb)
            mn = torch.maximum(m, mb)
            alpha = torch.where(torch.isfinite(m), torch.exp2(m - mn), torch.zeros_like(m))
            pblk = torch.where(live, torch.exp2(mb - mn), torch.zeros_like(mb))
            e = torch.where(live, torch.exp2(sc - torch.where(live, mb, 0.0)), torch.zeros_like(sc))
            ph = torch.round(pq * e)
            w = pblk / pq
            sp = ph.sum(-1, keepdim=True)
            pv_f = sp * bV + sV * (ph @ V["code"][..., sl, :])
            pv_c = sp * bV + sV * 0.5 * (ph @ vc2[..., sl, :])
            pv_e = ph @ vf[..., sl, :]
            pv = torch.where(exact_g, pv_e, torch.where(ref, pv_f, pv_c))
            l = alpha * l + w * sp
            o = alpha * o + w * pv
            m = torch.where(live, mn, m)
    if stats is not None:
        stats["refined"] = stats.get("refined", 0) + n_ref
        stats["quantized"] = stats.get("quantized", 0) + n_q
    return o * (1.0 / l)


def exact_attention(q, k, v, scale=None):
    """float64 causal reference (q at positions N - M .. N - 1)."""
    D = q.shape[-1]
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    s = (q.double() @ k.double().transpose(-1, -2)) * scale
    M, N = q.shape[-2], k.shape[-2]
    mask = torch.arange(N, device=q.device)[None, :] > torch.arange(N - M, N, device=q.device)[:, None]
    return torch.softmax(s.masked_fill(mask, -math.inf), -1) @ v.double()
