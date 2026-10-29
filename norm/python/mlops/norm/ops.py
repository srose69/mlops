# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""RMSNorm and LayerNorm on the mlops_norm kernel."""
from typing import List, NamedTuple, Optional

import torch

from . import _C

_DT = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}
_OUT = {None: 0, "int8": 1, "nvfp4": 2}
_KIND = {"rms": 0, "layer": 1}
_ORDER = {"hf": 0, "fp32": 1}
_GATE = {None: 0, "after": 1, "before": 2}


class NormOut(NamedTuple):
    y: torch.Tensor                    # [rows, hidden] in y's dtype; int8 [rows, hidden]; uint8 [rows, hidden / 2] (nibbles)
    scales: Optional[torch.Tensor]     # quantized: float32 / uint8 (E4M3) [rows, hidden / 16]
    residual: Optional[torch.Tensor]   # round(x + residual), x's dtype


def _flavor(kind, has_weight, one_plus, has_bias, order, has_gate):
    if kind not in _KIND:
        raise ValueError(f"kind must be 'rms' or 'layer', got {kind!r}")
    if one_plus and not has_weight:
        raise ValueError("one_plus needs a weight")
    if order is None:  # Hugging Face's order is that of a plain-weight RMSNorm (Llama, Qwen); Gemma's 1 + w, biases and gates round once
        order = "hf" if kind == "rms" and has_weight and not one_plus and not has_bias and not has_gate else "fp32"
    if order not in _ORDER:
        raise ValueError(f"order must be 'hf' or 'fp32', got {order!r}")
    return _KIND[kind], (0 if not has_weight else 2 if one_plus else 1), _ORDER[order]


def norm_variants(rows: int, hidden: int, x_dtype: torch.dtype, y_dtype: torch.dtype, *, kind: str = "rms", weight: bool = True,
                  one_plus: bool = False, bias: bool = False, order: Optional[str] = None, gate_mode: Optional[str] = None,
                  groups: int = 1, quant: Optional[str] = None) -> List[tuple]:
    """(warps, threads, smem_bytes, regs, arch) of every way the current device runs the shape."""
    if gate_mode not in _GATE:
        raise ValueError(f"gate_mode must be None, 'after' or 'before', got {gate_mode!r}")
    k, s, o = _flavor(kind, weight, one_plus, bias, order, gate_mode is not None)
    return _C.norm_variants(rows, hidden, groups, k, s, int(bias), o, _GATE[gate_mode], _DT[x_dtype], _DT[y_dtype], _OUT[quant])


def norm(x: torch.Tensor, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None, eps: float = 1e-6, *,
         kind: str = "rms", one_plus: bool = False, order: Optional[str] = None, gate: Optional[torch.Tensor] = None,
         gate_mode: str = "after", groups: int = 1, residual: Optional[torch.Tensor] = None, quant: Optional[str] = None,
         global_scale: Optional[torch.Tensor] = None, y_dtype: Optional[torch.dtype] = None, variant: int = 0, items: int = 0) -> NormOut:
    """RMSNorm (``kind="rms"``) or LayerNorm (``"layer"``) of the rows of x [rows, hidden] (bfloat16 / float16 / float32,
    hidden a multiple of 256 * groups), after ``v = round(x + residual)``; groups normalize hidden / groups values each.

    ``weight`` / ``bias`` are [hidden] in y's dtype (``y_dtype``: the weight's, else x's); ``one_plus``: ``1 + weight``
    (Gemma, Qwen3-Next, LayerNorm1P); no weight: none. ``order="hf"`` (Llama, Qwen: ``round(round(v r) w)``, an RMSNorm
    with a plain weight, no bias and no gate) or ``"fp32"`` (one rounding at the end: Gemma, LayerNorm, biases, gates);
    the default is ``"hf"`` for exactly that RMSNorm and ``"fp32"`` otherwise.
    ``gate`` [rows, hidden] in x's dtype multiplies y by silu(gate) (``gate_mode="after"``) or v before the norm (``"before"``).

    ``quant="int8"``: y as int8 with a float32 scale per 16 values (amax / 127); ``quant="nvfp4"``: E2M1 nibbles with an E4M3
    scale per 16 values and ``global_scale`` (float32, one element, the decode factor; default 1) -- what NVFP4EffMA's
    ``nvfp4a_quantize`` makes of y. ``variant`` indexes ``norm_variants``."""
    if quant not in _OUT:
        raise ValueError(f"quant must be None, 'int8' or 'nvfp4', got {quant!r}")
    if gate_mode not in ("after", "before"):
        raise ValueError(f"gate_mode must be 'after' or 'before', got {gate_mode!r}")
    k, s, o = _flavor(kind, weight is not None, one_plus, bias is not None, order, gate is not None)
    yd = y_dtype or (weight.dtype if weight is not None else x.dtype)
    y, scales, new_res = _C.norm(x, residual, weight, bias, gate, eps, groups, k, s, o, _GATE[gate_mode] if gate is not None else 0, _OUT[quant],
                                 _DT[yd], global_scale, variant, items)
    return NormOut(y, scales if quant else None, new_res if residual is not None else None)


def rms_norm(x, weight=None, eps: float = 1e-6, **kw) -> NormOut:
    """:func:`norm` with ``kind="rms"``."""
    return norm(x, weight, kw.pop("bias", None), eps, kind="rms", **kw)


def layer_norm(x, weight=None, bias=None, eps: float = 1e-5, **kw) -> NormOut:
    """:func:`norm` with ``kind="layer"``."""
    return norm(x, weight, bias, eps, kind="layer", **kw)


class RMSNorm(torch.nn.Module):
    """A module around :func:`rms_norm` (``forward(x, residual=None, quant=None, global_scale=None)``)."""

    def __init__(self, hidden: int, eps: float = 1e-6, one_plus: bool = False, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.eps, self.one_plus = eps, one_plus
        self.weight = torch.nn.Parameter(torch.zeros(hidden, dtype=dtype, device=device) if one_plus else torch.ones(hidden, dtype=dtype, device=device))

    def forward(self, x, residual=None, quant=None, global_scale=None):
        return rms_norm(x, self.weight.data, self.eps, one_plus=self.one_plus, residual=residual, quant=quant, global_scale=global_scale)


class LayerNorm(torch.nn.Module):
    """A module around :func:`layer_norm` (weight and bias parameters; ``forward(x, residual=None, quant=None, global_scale=None)``)."""

    def __init__(self, hidden: int, eps: float = 1e-5, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden, dtype=dtype, device=device))
        self.bias = torch.nn.Parameter(torch.zeros(hidden, dtype=dtype, device=device))

    def forward(self, x, residual=None, quant=None, global_scale=None):
        return layer_norm(x, self.weight.data, self.bias.data, self.eps, residual=residual, quant=quant, global_scale=global_scale)
