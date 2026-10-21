# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""torch.optim front ends of the two kernels, and the NVFP4 layer AdamWQAT trains.

* ``AdamW``       dense CUDA parameters (bfloat16 / float16 / float32, FP32 moments): mlops_adamw.
* ``AdamWQAT``    NVFP4 weights of ``QATLinear`` layers only (``QATParameter``): mlops_adamw_qat. Dense
                  parameters are refused -- they belong to ``AdamW``.
* ``QATLinear``   linear layer whose weight is an NVFP4 tensor trained in place; forward is NVFP4EffMA's
                  GEMM (activations exact, int8 or NVFP4: ``act``), backward exact for it. Its NVFP4 tensor
                  and optimizer state are module buffers (``state_dict`` saves them). ``grad_dtype=torch.float8_e5m2`` hands the optimizer FP8
                  gradients (``weight.grad_fp8``) instead of float32 ``weight.grad``.
* ``split_params(model)``  -> (dense parameters, QAT parameters), for the two optimizers.

    from mlops.optim import AdamW, AdamWQAT, split_params
    dense, qat = split_params(model)
    opt = AdamW(dense, lr=1e-3)
    opt_qat = AdamWQAT(qat, lr=1e-3, weight_decay=0.1)
    loss.backward(); opt.step(); opt_qat.step(); opt.zero_grad(); opt_qat.zero_grad()
"""
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from . import _C

_DENSE = (torch.bfloat16, torch.float16, torch.float32)
_ACTS = ("exact", "int8", "nvfp4", "auto")
_FP8_MAX = {torch.float8_e5m2: 57344.0, torch.float8_e4m3fn: 448.0}


def _fp8_encode(grad: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, float]:
    """(grad * s rounded to ``dtype``, 1 / s), s mapping the tensor's amax to the format's maximum."""
    amax = float(grad.abs().max())
    s = _FP8_MAX[dtype] / amax if amax > 0 else 1.0
    return (grad * s).to(dtype), 1.0 / s


def _flat(q):
    """An nvfp4effma.NVFP4Tensor as the extension takes it."""
    return q.data, q.scales.view(torch.uint8), q.global_scale, q.flags, q.scale_max


def _init_qat(w: torch.Tensor, state: torch.Tensor, four_over_six: Optional[str]):
    """The NVFP4 weight of a QAT state: ``w`` quantized by nvfp4effma.quantize (1x16 blocks, E4M3 scales, amax
    global: Transformer Engine's current scaling, optionally 4over6), then ``state`` initialized from ``w`` and
    it (mlops_adamw_qat_init_quantized). The NVFP4 tensor is part of the state from then on."""
    import nvfp4effma
    w = w.contiguous()
    q = nvfp4effma.quantize(w, four_over_six=four_over_six)
    _C.qat_init_quantized(w, state, *_flat(q))
    return q


class QATParameter(nn.Parameter):
    """Handle of an NVFP4 weight for autograd and torch.optim: shape [out, in], float32, one element of
    storage (zero strides) -- the weight itself is ``.layer``'s NVFP4 tensor. dL/dq for the dequantized
    weight is ``.grad`` (float32) or, for a layer with an FP8 ``grad_dtype``, ``.grad_fp8`` = (FP8 bytes,
    grad_scale): torch refuses an FP8 ``.grad`` on a float32 tensor."""

    @classmethod
    def placeholder(cls, rows: int, cols: int, device) -> "QATParameter":
        p = cls(torch.zeros(1, device=device).expand(rows, cols), requires_grad=True)
        p.grad_fp8 = None
        return p

    def __repr__(self):
        return f"QATParameter({list(self.shape)}, NVFP4)"


def _act_mode(act: str, rows: int) -> str:
    """The activation format nvfp4effma.linear uses for ``act`` on ``rows`` rows ("auto": exact for up to
    AUTO_EXACT_MAX_ROWS rows, where both run at memory speed, int8 beyond)."""
    import nvfp4effma
    return ("exact" if rows <= nvfp4effma.AUTO_EXACT_MAX_ROWS else "int8") if act == "auto" else act


class _NVFP4LinearFn(torch.autograd.Function):
    """y = A(x) @ q.T, q the NVFP4 weight, A(x) the activations as the GEMM reads them: x itself ("exact",
    W4A16, in x's dtype), int8 with an FP32 scale per 16 values ("int8", W4A8) or NVFP4 ("nvfp4", W4A4).
    Backward exact for that product: grad_q = gy.T @ A(x); grad_x = gy @ q, straight through A. A(x) is
    what is kept for the backward (4 / 2 / 1.25 / 0.5625 B per value)."""

    @staticmethod
    def forward(ctx, x, handle, layer):
        import nvfp4effma
        w = layer.nvfp4_weight()
        x2 = x.reshape(-1, layer.in_features).contiguous()
        mode = _act_mode(layer.act, x2.shape[0])
        if mode == "exact":
            y = nvfp4effma.linear(x2, w, act="exact")
            ctx.save_for_backward(x2)
        elif mode == "int8":
            # nvfp4effma.linear(act="int8") in two calls, so the quantized activations can be kept
            xq = nvfp4effma.quantize_int8(x2)
            y = nvfp4effma.linear(xq, w)
            ctx.save_for_backward(xq.data, xq.scales)
        else:
            ctx.xq = nvfp4effma.quantize(x2)
            y = nvfp4effma.linear(ctx.xq, w, out_dtype=x.dtype)
        ctx.mode, ctx.layer, ctx.x_shape, ctx.x_dtype = mode, layer, x.shape, x.dtype
        return y.reshape(*x.shape[:-1], layer.out_features)

    @staticmethod
    def backward(ctx, gy):
        layer = ctx.layer
        n, k = layer.out_features, layer.in_features
        if ctx.mode == "exact":
            (x2,) = ctx.saved_tensors
            xa = x2.float()
        elif ctx.mode == "int8":
            xq, xs = ctx.saved_tensors
            xa = (xq.float().view(-1, k // 16, 16) * xs.view(-1, k // 16, 1)).view(-1, k)
        else:
            xa = ctx.xq.dequantize(torch.float32)
            del ctx.xq
        q = layer.nvfp4_weight().dequantize(torch.float32)
        gy2 = gy.reshape(-1, n).float()
        grad_q = gy2.t() @ xa
        if layer.grad_dtype is not None:
            h = layer.weight
            if h.grad_fp8 is not None:  # accumulation over several backwards: sum, then encode once
                g8, gs = h.grad_fp8
                grad_q += g8.float() * gs
            h.grad_fp8 = _fp8_encode(grad_q, layer.grad_dtype)
            grad_q = None
        return (gy2 @ q).reshape(ctx.x_shape).to(ctx.x_dtype), grad_q, None


class QATLinear(nn.Module):
    """Bias-free linear layer with an NVFP4 weight [out_features, in_features] trained by AdamWQAT.

    in_features must be a multiple of 64 (NVFP4EffMA's GEMM). The weight is initialized from ``weight``
    if given (any dense dtype, not kept), else from N(0, init_std^2). ``four_over_six``: None / "mae" /
    "mse", Transformer Engine's 4over6 scale selection for the initial quantization. ``grad_dtype``: None
    (float32 ``weight.grad``) or torch.float8_e5m2 / torch.float8_e4m3fn (``weight.grad_fp8``, amax-scaled).
    ``act``: how the activations enter the GEMM, as in nvfp4effma.linear -- "exact" (W4A16, x in its own
    bfloat16 / float16 / float32), "int8" (W4A8, about 2x faster from sm_61 on: DP4A), "nvfp4" (W4A4) or
    "auto" (exact up to 2 rows, int8 beyond); the output always has x's dtype."""

    def __init__(self, in_features: int, out_features: int, weight: Optional[torch.Tensor] = None,
                 init_std: float = 0.02, four_over_six: Optional[str] = None, grad_dtype: Optional[torch.dtype] = None,
                 act: str = "exact", device="cuda"):
        super().__init__()
        if in_features % 64:
            raise ValueError(f"in_features ({in_features}) must be a multiple of 64 (NVFP4EffMA's GEMM)")
        if act not in _ACTS:
            raise ValueError(f"act must be one of {_ACTS}, got {act!r}")
        self.act = act
        if grad_dtype is not None and grad_dtype not in _FP8_MAX:
            raise ValueError(f"grad_dtype must be None, torch.float8_e5m2 or torch.float8_e4m3fn, got {grad_dtype}")
        if four_over_six not in (None, "mae", "mse"):
            raise ValueError(f"four_over_six must be None, 'mae' or 'mse', got {four_over_six!r}")
        self.in_features, self.out_features, self.grad_dtype = in_features, out_features, grad_dtype
        dev = torch.device(device)
        if weight is None:
            weight = torch.empty(out_features, in_features, device=dev).normal_(std=init_std)
        if tuple(weight.shape) != (out_features, in_features):
            raise ValueError(f"weight must be [{out_features}, {in_features}], got {tuple(weight.shape)}")
        # zeros, not empty: the kernel's scratch and padding bytes are saved with state_dict and must be deterministic
        state = torch.zeros(_C.qat_state_bytes(out_features, in_features), dtype=torch.uint8, device=dev)
        q = _init_qat(weight.to(dev), state, four_over_six)
        self.register_buffer("nvfp4_data", q.data)
        self.register_buffer("nvfp4_scales", q.scales.view(torch.uint8))
        # the amax as 4 bytes: a module dtype cast (model.half()) must not touch it
        self.register_buffer("nvfp4_amax", q.global_scale.view(torch.uint8))
        self.register_buffer("qat_state", state)
        self._scale_max = q.scale_max
        self.weight = QATParameter.placeholder(out_features, in_features, dev)
        self.weight.layer = self

    def _apply(self, fn, recurse=True):
        # buffers follow device moves (uint8: dtype casts leave them alone); the weight stays a float32
        # one-element placeholder whatever fn did to it
        super()._apply(fn, recurse)
        self.weight.data = torch.zeros(1, device=self.nvfp4_data.device).expand(self.out_features, self.in_features)
        if self.weight.grad_fp8 is not None:
            g8, gs = self.weight.grad_fp8
            self.weight.grad_fp8 = (g8.to(self.nvfp4_data.device), gs)
        return self

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        for name, buf in self._buffers.items():  # the weight is these buffers; the placeholder is not saved
            destination[prefix + name] = buf if keep_vars else buf.detach()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        w = self._parameters.pop("weight")
        try:
            super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        finally:
            self._parameters["weight"] = w

    def nvfp4_weight(self):
        """The NVFP4 tensor the forward reads (views of this module's buffers)."""
        import nvfp4effma
        return nvfp4effma.NVFP4Tensor(self.nvfp4_data, self.nvfp4_scales, amax=self.nvfp4_amax.view(torch.float32),
                                      scale_max=self._scale_max)

    def _flat(self):
        return _flat(self.nvfp4_weight())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _NVFP4LinearFn.apply(x, self.weight, self)

    def extra_repr(self):
        g = "" if self.grad_dtype is None else f", grad_dtype={self.grad_dtype}"
        return f"in_features={self.in_features}, out_features={self.out_features}, NVFP4, act={self.act}{g}"


def split_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[QATParameter]]:
    """(dense parameters, QAT parameters) of a model, each parameter once."""
    dense, qat, seen = [], [], set()
    for p in model.parameters():
        if id(p) in seen or not p.requires_grad:
            continue
        seen.add(id(p))
        (qat if isinstance(p, QATParameter) else dense).append(p)
    return dense, qat


def _check_betas(lr, betas, eps, weight_decay):
    if lr < 0 or eps < 0 or weight_decay < 0 or not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
        raise ValueError(f"invalid hyperparameters: lr={lr}, betas={betas}, eps={eps}, weight_decay={weight_decay}")


class AdamW(torch.optim.Optimizer):
    """AdamW (decoupled weight decay) on dense CUDA parameters: bfloat16 / float16 / float32 parameters and
    gradients, FP32 moments, one fused mlops_adamw call per parameter. NVFP4 weights (QATParameter) are
    refused: they belong to AdamWQAT."""

    def __init__(self, params: Iterable, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def add_param_group(self, param_group: dict):
        super().add_param_group(param_group)  # the checks run here, so later add_param_group calls get them too
        g = self.param_groups[-1]
        try:
            _check_betas(g["lr"], g["betas"], g["eps"], g["weight_decay"])
        except ValueError:
            self.param_groups.pop()
            raise
        for p in g["params"]:
            if isinstance(p, QATParameter):
                self.param_groups.pop()
                raise TypeError("AdamW: an NVFP4 QATParameter belongs to AdamWQAT (see split_params)")
            if not p.is_cuda or p.dtype not in _DENSE or not p.is_contiguous():
                self.param_groups.pop()
                raise TypeError(f"AdamW: parameters must be contiguous CUDA bfloat16 / float16 / float32 tensors, "
                                f"got {p.dtype} on {p.device}{'' if p.is_contiguous() else ' (non-contiguous)'}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW: sparse gradients are not supported")
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.contiguous_format)
                    st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.contiguous_format)
                st["step"] += 1
                grad = p.grad if p.grad.dtype in _DENSE else p.grad.float()
                _C.adamw(p, grad.contiguous(), st["exp_avg"], st["exp_avg_sq"], g["lr"], b1, b2, g["eps"],
                         g["weight_decay"], st["step"], True, 1.0)
        return loss


class AdamWQAT(torch.optim.Optimizer):
    """mlops_adamw_qat on the NVFP4 weights of QATLinear layers (QATParameter): no master copy, 5 B of
    state per weight + 8 per 16-weight block, kept in the layer's buffers. Dense parameters are refused:
    they belong to AdamW.

    Gradients: ``p.grad`` (float32), or ``p.grad_fp8`` = (float8_e5m2 / float8_e4m3fn bytes, grad_scale) from
    a QATLinear with an FP8 ``grad_dtype``. ``scale_z``: significance and CUSUM threshold of the block-scale
    moves, in null standard deviations; ``seed``: the stochastic-rounding stream. Everything else AdamW
    does (dense, CPU, sparse or non-NVFP4 parameters, amsgrad, maximize, foreach) is refused."""

    def __init__(self, params: Iterable, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-12, weight_decay: float = 1e-2, scale_z: float = 3.0, seed: int = 0):
        self.seed = seed
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, scale_z=scale_z))

    def add_param_group(self, param_group: dict):
        super().add_param_group(param_group)
        g = self.param_groups[-1]
        try:
            _check_betas(g["lr"], g["betas"], g["eps"], g["weight_decay"])
        except ValueError:
            self.param_groups.pop()
            raise
        if not 0 < g["scale_z"] < 5.0:
            self.param_groups.pop()
            raise ValueError(f"scale_z must be in (0, 5), got {g['scale_z']}")
        for p in g["params"]:
            if not isinstance(p, QATParameter):
                self.param_groups.pop()
                raise TypeError(f"AdamWQAT: only NVFP4 weights of QATLinear layers (QATParameter); a dense "
                                f"{p.dtype} parameter {tuple(p.shape)} belongs to AdamW (see split_params)")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        k = 0
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                k += 1  # a stochastic-rounding stream per parameter, by its position in the optimizer
                if p.grad is not None and p.grad_fp8 is not None:
                    raise RuntimeError("AdamWQAT: both .grad and .grad_fp8 are set on one QATParameter")
                if p.grad is not None:
                    if p.grad.is_sparse or p.grad.dtype != torch.float32:
                        raise TypeError(f"AdamWQAT: .grad must be a dense float32 tensor, got {p.grad.dtype}"
                                        f"{' (sparse)' if p.grad.is_sparse else ''}")
                    grad, grad_scale = p.grad, 1.0
                elif p.grad_fp8 is not None:
                    grad, grad_scale = p.grad_fp8
                else:
                    continue
                st = self.state[p]
                st["step"] = st.get("step", 0) + 1
                layer = p.layer
                _C.qat_step(grad.contiguous(), layer.qat_state, *layer._flat(), lr=g["lr"], beta1=b1, beta2=b2,
                            eps=g["eps"], weight_decay=g["weight_decay"], step=st["step"],
                            grad_scale=float(grad_scale), scale_z=g["scale_z"],
                            seed=(self.seed + 0x9E3779B9 * k) & 0xFFFFFFFF)
        return loss

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none)
        for g in self.param_groups:
            for p in g["params"]:
                p.grad_fp8 = None
