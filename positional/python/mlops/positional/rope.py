# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Rotary position embedding (RoPE) on the mlops_rope kernel: q and k of every head in one pass, forward and
backward (the backward is the inverse rotation).

* ``RoPE``                    the module: angles from positions, exact (phase reduced in turns, sincospi), with
                              any frequency scaling -- YaRN, Llama 3, linear, dynamic NTK, LongRoPE (``scaling``,
                              Hugging Face's ``rope_parameters`` / ``rope_scaling`` dicts) -- and its attention
                              factor. ``RoPE.from_hf(config)`` reads a transformers config.
* ``rope_frequencies``        the inverse frequencies and attention factor of a scaling, in float64.
* ``apply_rotary_pos_emb``    Hugging Face's function of that name on the kernel, bit for bit (forward and autograd
                              backward), from the cos / sin tables its rotary modules make.
* ``patch_transformers``      puts it into transformers' modeling modules.
"""
import importlib
import math
from typing import Optional, Tuple

import torch

from . import _C

_DT = (torch.bfloat16, torch.float16, torch.float32)
_SCALINGS = ("default", "linear", "dynamic", "yarn", "llama3", "longrope")


# ---- frequencies ---------------------------------------------------------------------------------------------
def rope_frequencies(rotary_dim: int, base: float = 10000.0, scaling: Optional[dict] = None, *,
                     max_position_embeddings: Optional[int] = None,
                     seq_len: Optional[int] = None) -> Tuple[torch.Tensor, float]:
    """(inverse frequencies, float64 CPU [rotary_dim / 2], attention factor) of RoPE with an optional frequency
    scaling, as transformers computes them (``modeling_rope_utils``) but in float64:

    * ``None`` / ``{"rope_type": "default"}``: ``base^(-2i / rotary_dim)``;
    * ``"linear"`` (``factor``): position interpolation, every frequency / factor;
    * ``"dynamic"`` (``factor``): NTK-aware base growth with ``seq_len`` beyond ``max_position_embeddings``;
    * ``"yarn"`` (``factor``, ``original_max_position_embeddings``, optional ``beta_fast`` 32, ``beta_slow`` 1,
      ``attention_factor`` or ``mscale`` / ``mscale_all_dim``, ``truncate``): interpolation of the low frequencies,
      extrapolation of the high ones with a linear ramp between, attention factor 0.1 ln(factor) + 1;
    * ``"llama3"`` (``factor``, ``low_freq_factor``, ``high_freq_factor``, ``original_max_position_embeddings``);
    * ``"longrope"`` (``short_factor``, ``long_factor``, ``original_max_position_embeddings``, optional ``factor``,
      ``attention_factor``): per-frequency factors, the long ones when ``seq_len`` exceeds the original length.

    The attention factor multiplies cos and sin (so each of q, k), as in transformers."""
    if rotary_dim < 2 or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even and >= 2, got {rotary_dim}")
    s = dict(scaling or {})
    kind = s.get("rope_type", s.get("type", "default")) or "default"
    if kind not in _SCALINGS:
        raise ValueError(f"unsupported RoPE scaling {kind!r} (supported: {', '.join(_SCALINGS)})")
    dim = rotary_dim
    expo = torch.arange(0, dim, 2, dtype=torch.float64) / dim
    att = 1.0

    def need(key):
        if s.get(key) is None:
            raise ValueError(f"RoPE scaling {kind!r} needs {key!r}")
        return s[key]

    if kind == "default":
        inv = base ** -expo
    elif kind == "linear":
        inv = base ** -expo / float(need("factor"))
    elif kind == "dynamic":
        factor = float(need("factor"))
        if max_position_embeddings is None:
            raise ValueError("dynamic RoPE needs max_position_embeddings")
        n = max(seq_len or max_position_embeddings, max_position_embeddings)
        b = base * ((factor * n / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
        inv = b ** -expo
    elif kind == "yarn":
        orig = need("original_max_position_embeddings")
        factor = s.get("factor")
        if factor is None:
            if max_position_embeddings is None:
                raise ValueError("YaRN without a factor needs max_position_embeddings")
            factor = max_position_embeddings / orig

        def mscale(sc, m=1.0):
            return 1.0 if sc <= 1 else 0.1 * m * math.log(sc) + 1.0

        att = s.get("attention_factor")
        if att is None:
            m, m_all = s.get("mscale"), s.get("mscale_all_dim")
            att = mscale(factor, m) / mscale(factor, m_all) if m and m_all else mscale(factor)
        beta_fast, beta_slow = s.get("beta_fast") or 32, s.get("beta_slow") or 1

        def corr_dim(rot):  # the dimension whose wavelength fits `rot` rotations into the original context
            return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))

        low, high = corr_dim(beta_fast), corr_dim(beta_slow)
        if s.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
        extra = 1 - ramp
        pos_freqs = base ** expo
        inv = (1 / (factor * pos_freqs)) * (1 - extra) + (1 / pos_freqs) * extra
    elif kind == "llama3":
        factor, lo, hi = float(need("factor")), float(need("low_freq_factor")), float(need("high_freq_factor"))
        old = need("original_max_position_embeddings")
        inv = base ** -expo
        wavelen = 2 * math.pi / inv
        scaled = torch.where(wavelen > old / lo, inv / factor, inv)
        smooth = (old / wavelen - lo) / (hi - lo)
        smoothed = (1 - smooth) * scaled / factor + smooth * scaled
        medium = ~(wavelen < old / hi) & ~(wavelen > old / lo)
        inv = torch.where(medium, smoothed, scaled)
    else:  # longrope
        orig = need("original_max_position_embeddings")
        factor = s.get("factor")
        if factor is None:
            if max_position_embeddings is None:
                raise ValueError("LongRoPE without a factor needs max_position_embeddings")
            factor = max_position_embeddings / orig
        att = s.get("attention_factor")
        if att is None:
            att = 1.0 if factor <= 1 else math.sqrt(1 + math.log(factor) / math.log(orig))
        ext = torch.tensor(need("long_factor") if seq_len and seq_len > orig else need("short_factor"), dtype=torch.float64)
        if ext.numel() != dim // 2:
            raise ValueError(f"LongRoPE factors must have rotary_dim / 2 = {dim // 2} entries, got {ext.numel()}")
        inv = 1 / (ext * base ** expo)
    return inv.to(torch.float64), float(att)


# ---- the kernel on [B, S, H, D] views --------------------------------------------------------------------------
def _bshd(t: torch.Tensor, layout: str) -> torch.Tensor:
    """A view of t as [B, S, H, D]."""
    if layout == "bshd":
        return t
    if layout == "bhsd":
        return t.transpose(1, 2)
    return t.unsqueeze(0)  # thd


def _last_contiguous(t):
    return t if t.stride(-1) == 1 else t.contiguous()


def _run(q, k, layout, inverse, out_q=None, out_k=None, **angles):
    """mlops_rope of q (and k) in `layout`; out_q / out_k: destination tensors (None: new ones, same layout)."""
    q = _last_contiguous(q)
    k = _last_contiguous(k) if k is not None else None
    oq = torch.empty_like(q) if out_q is None else out_q
    ok = (torch.empty_like(k) if out_k is None else out_k) if k is not None else None
    _C.rope(_bshd(q, layout), _bshd(oq, layout), _bshd(k, layout) if k is not None else None,
            _bshd(ok, layout) if k is not None else None, inverse=inverse, **angles)
    return oq, ok


class _Rotate(torch.autograd.Function):
    """y = R(angles) x for q and k; the backward rotates the gradients back (mlops_rope with inverse)."""

    @staticmethod
    def forward(ctx, q, k, layout, angles):
        ctx.layout, ctx.angles = layout, angles
        return _run(q, k, layout, False, **angles)

    @staticmethod
    def backward(ctx, gq, gk):  # gq is materialized (zeros if unused); gk is None without k
        rq, rk = _run(gq, gk, ctx.layout, True, **ctx.angles)
        return rq, rk, None, None


def _shape_of(q: torch.Tensor, layout: str):
    if layout not in ("bshd", "bhsd", "thd"):
        raise ValueError(f"layout must be 'bshd', 'bhsd' or 'thd', got {layout!r}")
    want = 3 if layout == "thd" else 4
    if q.dim() != want:
        raise ValueError(f"q must be {want}-D for layout {layout!r}, got shape {tuple(q.shape)}")
    v = _bshd(q, layout)
    return v.shape[0], v.shape[1], v.shape[3]


class RoPE(torch.nn.Module):
    """Rotary position embedding of q and k, angles from the token positions.

    ``RoPE(head_dim, base=10000.0, *, rotary_dim=None, interleaved=False, scaling=None,
    max_position_embeddings=None)``: ``rotary_dim`` (default head_dim) leading elements of every head rotate, the
    rest pass through; pairs ``(i, i + rotary_dim / 2)`` (NeoX / Llama) or ``(2i, 2i + 1)`` with
    ``interleaved=True`` (GPT-J); ``scaling``: a transformers rope-scaling dict (``rope_frequencies``).

    ``rope(q, k=None, positions=None, *, layout="bshd", offset=0, inplace=False, seq_len=None)`` returns the rotated
    (q, k). ``layout``: ``"bshd"`` [batch, seq, heads, head_dim], ``"bhsd"`` (transformers' [batch, heads, seq,
    head_dim]) or ``"thd"`` [tokens, heads, head_dim]; k may have fewer heads (GQA). ``positions``: int32 / int64
    [batch, seq] or [seq] (shared by the batch), or None for ``offset + s``. ``inplace=True`` overwrites q and k
    (no autograd). ``seq_len`` selects the frequencies of the dynamic and LongRoPE scalings (default: from the
    positions).

    The phase is reduced exactly in turns and the sines come from sincospi in FP32, once per token and frequency
    for all heads; each output is rounded once to the dtype. At any position below 2^31, BF16 / FP16 outputs are the
    exact rotation rounded once (give or take the last bit), FP32 ones within a few FP32 ulps; transformers rounds
    the angle to FP32, which costs up to half a radian at 10M positions. Differentiable: the backward is the
    inverse rotation.
    """

    def __init__(self, head_dim: int, base: float = 10000.0, *, rotary_dim: Optional[int] = None,
                 interleaved: bool = False, scaling: Optional[dict] = None, max_position_embeddings: Optional[int] = None):
        super().__init__()
        self.head_dim, self.base = int(head_dim), float(base)
        self.rotary_dim = int(rotary_dim or head_dim)
        if self.rotary_dim > self.head_dim:
            raise ValueError(f"rotary_dim ({self.rotary_dim}) must not exceed head_dim ({self.head_dim})")
        self.interleaved, self.scaling, self.max_position_embeddings = bool(interleaved), scaling, max_position_embeddings
        self.kind = (dict(scaling or {}).get("rope_type") or dict(scaling or {}).get("type") or "default")
        self.inv_freq, self.attention_factor = rope_frequencies(self.rotary_dim, self.base, scaling,
                                                                max_position_embeddings=max_position_embeddings)
        self._tables = {}

    @classmethod
    def from_hf(cls, config, layer_type: Optional[str] = None, interleaved: bool = False) -> "RoPE":
        """From a transformers model config (``rope_parameters`` or the older ``rope_theta`` / ``rope_scaling``,
        ``head_dim`` or hidden_size / num_attention_heads, ``partial_rotary_factor``)."""
        params = getattr(config, "rope_parameters", None)
        if isinstance(params, dict) and layer_type is not None:
            params = params[layer_type]
        if not isinstance(params, dict):
            params = dict(getattr(config, "rope_scaling", None) or {})
            params.setdefault("rope_theta", getattr(config, "rope_theta", 10000.0))
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        partial = params.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0))
        return cls(head_dim, params.get("rope_theta", 10000.0), rotary_dim=int(head_dim * partial), interleaved=interleaved,
                   scaling=params, max_position_embeddings=getattr(config, "max_position_embeddings", None))

    def _freq(self, device, seq_len):
        dynamic = self.kind in ("dynamic", "longrope")
        key = (device, seq_len if dynamic else None)
        if key not in self._tables:
            if dynamic:
                inv, att = rope_frequencies(self.rotary_dim, self.base, self.scaling, seq_len=seq_len,
                                            max_position_embeddings=self.max_position_embeddings)
            else:
                inv, att = self.inv_freq, self.attention_factor
            if len(self._tables) > 8:
                self._tables.clear()
            self._tables[key] = (_C.freq_table(inv).to(device), att)
        return self._tables[key]

    def forward(self, q: torch.Tensor, k: Optional[torch.Tensor] = None, positions: Optional[torch.Tensor] = None, *,
                layout: str = "bshd", offset: int = 0, inplace: bool = False, seq_len: Optional[int] = None):
        B, S, D = _shape_of(q, layout)
        if D != self.head_dim:
            raise ValueError(f"q has head_dim {D}, this RoPE was made for {self.head_dim}")
        if k is not None and (k.dim() != q.dim() or _shape_of(k, layout) != (B, S, D)):
            raise ValueError(f"k {tuple(k.shape)} does not match q {tuple(q.shape)} in layout {layout!r}")
        pos = None
        if positions is not None:
            if positions.dtype not in (torch.int32, torch.int64):
                raise TypeError(f"positions must be int32 or int64, got {positions.dtype}")
            pos = positions.unsqueeze(0) if positions.dim() == 1 else positions
            if pos.dim() != 2 or pos.shape[-1] != S or pos.shape[0] not in (1, B):
                raise ValueError(f"positions must be [{B}, {S}] or [{S}], got {tuple(positions.shape)}")
            pos = pos.expand(B, S)
        if self.kind in ("dynamic", "longrope") and seq_len is None:
            seq_len = int(positions.max()) + 1 if positions is not None else offset + S
        freq, att = self._freq(q.device, seq_len)
        angles = dict(rotary_dim=self.rotary_dim, interleaved=self.interleaved, positions=pos, position_offset=int(offset),
                      freq=freq, attention_factor=att)
        if inplace:
            if torch.is_grad_enabled() and (q.requires_grad or (k is not None and k.requires_grad)):
                raise RuntimeError("RoPE(inplace=True) is not differentiable: use it under torch.no_grad()")
            if q.stride(-1) != 1 or (k is not None and k.stride(-1) != 1):
                raise ValueError("inplace RoPE needs q and k with a contiguous last dimension")
            _run(q, k, layout, False, out_q=q, out_k=k, **angles)
            return q, k
        return _Rotate.apply(q, k, layout, angles)

    def extra_repr(self):
        return (f"head_dim={self.head_dim}, rotary_dim={self.rotary_dim}, base={self.base:g}, "
                f"{'interleaved' if self.interleaved else 'half split'}, scaling={self.kind}, "
                f"attention_factor={self.attention_factor:.6g}")


# ---- transformers ----------------------------------------------------------------------------------------------
def _hf_reference(q, k, cos, sin, unsqueeze_dim, interleaved):
    def rot(x):
        if interleaved:
            return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)
    c, s = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return q * c + rot(q) * s, k * c + rot(k) * s


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         position_ids: Optional[torch.Tensor] = None, unsqueeze_dim: int = 1, *, interleaved: bool = False):
    """transformers' ``apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)`` on the kernel: one pass for q and k,
    the same bits as ``q * cos + rotate_half(q) * sin`` in the forward and in the autograd backward. q, k:
    [batch, heads, seq, head_dim] (``unsqueeze_dim=1``) or [batch, seq, heads, head_dim] (2); cos, sin: [batch or 1,
    seq, head_dim] in q's dtype, as the rotary modules make them. Anything else (CPU, other dtypes, gradients
    through cos / sin) runs the plain PyTorch formula."""
    fast = (q.is_cuda and q.dim() == 4 and k.dim() == 4 and q.dtype in _DT and k.dtype == q.dtype and
            cos.dtype == q.dtype and sin.dtype == q.dtype and cos.dim() == 3 and cos.shape == sin.shape and
            unsqueeze_dim in (1, 2, -3) and not (cos.requires_grad or sin.requires_grad) and
            cos.shape[-1] == q.shape[-1] and q.device == k.device == cos.device == sin.device)
    if not fast:
        return _hf_reference(q, k, cos, sin, unsqueeze_dim, interleaved)
    layout = "bhsd" if unsqueeze_dim in (1, -3) else "bshd"
    B, S = _bshd(q, layout).shape[:2]
    if cos.shape[0] not in (1, B) or cos.shape[1] != S:
        return _hf_reference(q, k, cos, sin, unsqueeze_dim, interleaved)
    c, s = _last_contiguous(cos).expand(B, -1, -1), _last_contiguous(sin).expand(B, -1, -1)
    angles = dict(rotary_dim=q.shape[-1], interleaved=interleaved, cos=c, sin=s)
    return _Rotate.apply(q, k, layout, angles)


def patch_transformers(*models: str) -> list:
    """Replace ``apply_rotary_pos_emb`` in transformers' modeling modules of the given model types (e.g. "llama",
    "qwen2", "mistral") by this one; returns the modules patched. Only for models with the standard signature
    (q, k, cos, sin, ..., unsqueeze_dim) and half-split pairs."""
    done = []
    for name in models:
        mod = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
        if not hasattr(mod, "apply_rotary_pos_emb"):
            raise ValueError(f"transformers.models.{name} has no apply_rotary_pos_emb")
        mod.apply_rotary_pos_emb = apply_rotary_pos_emb
        done.append(mod.__name__)
    return done
