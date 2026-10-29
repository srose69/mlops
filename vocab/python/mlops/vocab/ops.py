# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Embedding rows, the LM head and the sampler on the mlops_embed / mlops_lm_head / mlops_sample kernels."""
from typing import List, NamedTuple, Optional

import torch

from . import _C


class Embedding(NamedTuple):
    x: torch.Tensor                  # [n, hidden] float32
    xw: Optional[torch.Tensor]       # [n, hidden] bfloat16, bf16(x * norm_weight)
    parts: torch.Tensor              # [n, hidden / 256] float32, the sums of squares of 256 elements of x
    error: torch.Tensor              # [1] int32, 1 if an id was outside [0, vocab) (its column is zero)


class Head(NamedTuple):
    logits: Optional[torch.Tensor]     # [n, vocab] float32
    stats_max: Optional[torch.Tensor]  # [n, items] float32, the maximum of each element's rows
    stats_arg: Optional[torch.Tensor]  # [n, items] int32, its lowest index
    stats_hist: Optional[torch.Tensor] # [n, items, 256] int32, counts of the top byte of the rows' order keys


def embed_variants(vocab: int, hidden: int, n: int) -> List[tuple]:
    """(warps, threads, smem_bytes, regs, arch) of every way the current device runs the shape."""
    return _C.embed_variants(vocab, hidden, n)


def embed(table: torch.Tensor, ids: torch.Tensor, norm_weight: Optional[torch.Tensor] = None, *, variant: int = 0,
          items: int = 0) -> Embedding:
    """``table[ids]`` widened to float32 (table: bfloat16 [vocab, hidden], hidden a multiple of 256; ids: int32 [n]
    on the device), with ``xw = bf16(x * norm_weight)`` and the partial sums of squares of x that a deferred
    RMSNorm scalar is made of. ``variant`` indexes ``embed_variants``; ``items`` is the CTA count (0: resident)."""
    return Embedding(*_C.embed(table, ids, norm_weight, variant, items))


def lm_head_variants(vocab: int, hidden: int, n: int) -> List[tuple]:
    """(R, S, D, NB, tensor_cores, J, threads, smem_bytes, regs, arch) of every way the current device runs the
    shape: R rows a unit, S warps over hidden, D steps of loads in flight, NB columns carried, J pieces a lane."""
    return _C.lm_head_variants(vocab, hidden, n)


def lm_head(weight: torch.Tensor, x: torch.Tensor, *, norm_parts: Optional[torch.Tensor] = None, inv_n: float = 0.0,
            eps: float = 0.0, logit_scale: float = 1.0, variant: int = 0, items: int = 0, want_logits: bool = True,
            want_stats: bool = False) -> Head:
    """``logits[c] = s[c] * logit_scale * (weight @ x[c])`` for 1 .. 8 columns of x (bfloat16 [n, hidden]), FP32
    accumulation, with ``s[c] = 1 / sqrt(sum(norm_parts[c]) * inv_n + eps)`` when ``norm_parts`` (float32
    [n, parts]) is given. ``want_stats`` adds each element's maximum and lowest argmax per column, the input of
    ``Sampler(..., stats=...)``. ``variant`` indexes ``lm_head_variants``; ``items`` is the element count."""
    return Head(*_C.lm_head(weight, x, norm_parts, inv_n, eps, logit_scale, variant, items, want_logits, want_stats))


class Sampler:
    """The next token from a row of logits. ``seed`` and the step counter are device tensors, so a loop of calls (or a
    captured graph) needs no host value; every call advances the step. ``penalty`` != 1 needs ``seen`` (a packed bitmask,
    int32 [ceil(vocab / 32)], bit i of word i // 32 set for the tokens to penalize). With ``stats`` (from
    ``lm_head(..., want_stats=True)`` with ``logit_scale = 1 / temperature``, or 1 for greedy) the logits are not
    scanned for the maximum, and with a top-k not for the first radix pass either; that needs penalty 1."""

    def __init__(self, vocab: int, *, temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0, penalty: float = 1.0,
                 seed: int = 0, ctas: int = 0, device="cuda"):
        self.vocab, self.temperature, self.top_k, self.top_p, self.penalty, self.ctas = vocab, temperature, top_k, top_p, penalty, ctas
        self.state = torch.tensor([seed, 0], dtype=torch.int64, device=device)
        self.workspace = torch.zeros(_C.sample_workspace_bytes(vocab) + 256, dtype=torch.uint8, device=device)
        # the workspace must start on a 256-byte boundary
        off = (-self.workspace.data_ptr()) % 256
        self.workspace = self.workspace[off:off + _C.sample_workspace_bytes(vocab)]
        self.token = torch.zeros(1, dtype=torch.int32, device=device)

    @property
    def step(self) -> int:
        return int(self.state[1].item())

    def __call__(self, logits: torch.Tensor, seen: Optional[torch.Tensor] = None, stats: Optional[Head] = None, *,
                 column: int = 0) -> torch.Tensor:
        """The token (int32 [1], on the device, overwritten by the next call) drawn from ``logits`` ([vocab] float32,
        or a row ``column`` of a 2-D tensor)."""
        row = logits[column] if logits.dim() == 2 else logits
        smax = sarg = shist = None
        if stats is not None:
            smax, sarg = stats.stats_max[column].contiguous(), stats.stats_arg[column].contiguous()
            shist = stats.stats_hist[column].contiguous()
        _C.sample(row.contiguous(), seen, smax, sarg, shist, self.state, self.workspace, self.token, self.temperature, self.top_k,
                  self.top_p, self.penalty, self.ctas)
        return self.token
