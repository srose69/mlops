"""mlops_adamw: AdamW and NVFP4 AdamW-QAT CUDA kernels (../../include), bound for PyTorch.

* ``adamw``      plain AdamW: param/grad bfloat16/float16/float32, exp_avg/exp_avg_sq float32, all
                 contiguous CUDA tensors of one shape.
* ``QATState``   an NVFP4 weight trained by mlops_adamw_qat (include/mlops_adamw_qat.h): no master
                 copy, 5 bytes of optimizer state per element + 8 per 1x16 block. ``.weight`` is the
                 ``nvfp4effma.NVFP4Tensor`` the forward reads; ``step(grad, ...)`` takes dL/d(that weight,
                 dequantized) and rewrites it in place. lr and weight_decay are AdamW's.
* ``fp8_grad``   a gradient as FP8 bytes plus the grad_scale that decodes them (per-tensor amax scaling).
* ``reference``  the same step in PyTorch, on the same state (mlops_adamw/reference.py).
* ``AdamW``, ``AdamWQAT``, ``QATLinear``, ``split_params``: torch.optim front ends and the NVFP4 layer
                 (mlops_adamw/optim.py).
"""
from typing import Optional, Tuple

import torch

from . import _C, reference
from .optim import AdamW, AdamWQAT, QATLinear, QATParameter, _fp8_encode, split_params

__version__ = _C.__version__
__all__ = ["AdamW", "AdamWQAT", "QATLinear", "QATParameter", "split_params", "adamw", "QATState", "fp8_grad",
           "reference", "__version__"]

_FOS = {None: 0, "mae": 1, "mse": 2}


def adamw(param: torch.Tensor, grad: torch.Tensor, exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor, lr: float,
          betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0, step: int = 1,
          bias_correction: bool = True, grad_scale: float = 1.0) -> None:
    _C.adamw(param, grad, exp_avg, exp_avg_sq, lr, betas[0], betas[1], eps, weight_decay, step, bias_correction,
             grad_scale)


def fp8_grad(grad: torch.Tensor, dtype: torch.dtype = torch.float8_e5m2) -> Tuple[torch.Tensor, float]:
    """(grad * s rounded to `dtype`, 1 / s), s mapping the tensor's amax to the format's maximum."""
    return _fp8_encode(grad, dtype)


class QATState:
    """An NVFP4 weight [rows, cols] (cols a multiple of 16) and its mlops_adamw_qat state, initialized
    from ``w`` (any dense dtype; not kept). 1x16 blocks, E4M3 scales, amax global (raised 8x at init so
    every block scale can grow)."""

    def __init__(self, w: torch.Tensor, four_over_six: Optional[str] = None):
        import nvfp4effma
        if w.dim() != 2 or w.shape[1] % 16:
            raise ValueError(f"w must be 2-D with cols a multiple of 16, got {tuple(w.shape)}")
        rows, cols = w.shape
        dev = w.device
        self.shape = (rows, cols)
        self.weight = nvfp4effma.NVFP4Tensor(
            torch.zeros(rows, cols // 2, dtype=torch.uint8, device=dev),
            torch.zeros(rows, cols // 16, dtype=torch.uint8, device=dev),
            amax=torch.zeros(1, dtype=torch.float32, device=dev), scale_max=256 if four_over_six else 0)
        self.state = torch.zeros(_C.qat_state_bytes(rows, cols), dtype=torch.uint8, device=dev)  # deterministic scratch
        _C.qat_init(w.contiguous(), self.state, *self._q(), four_over_six=_FOS[four_over_six])

    def _q(self):
        t = self.weight
        return t.data, t.scales.view(torch.uint8), t.global_scale, t.flags, t.scale_max

    def step(self, grad: torch.Tensor, lr: float, step: int, betas: Tuple[float, float] = (0.9, 0.999),
             eps: float = 1e-12, weight_decay: float = 0.0, grad_scale: float = 1.0, scale_z: float = 3.0,
             seed: int = 0) -> None:
        """grad: float8_e5m2 / float8_e4m3fn (values times grad_scale, see fp8_grad) or float32."""
        _C.qat_step(grad.contiguous(), self.state, *self._q(), lr=lr, beta1=betas[0], beta2=betas[1], eps=eps,
                    weight_decay=weight_decay, step=step, grad_scale=grad_scale, scale_z=scale_z, seed=seed)

    def master(self) -> torch.Tensor:
        """The master weight sigma * g(u) the codes round, float32 (for inspection; never stored)."""
        rows, cols = self.shape
        st = reference.carve(self.state, rows, cols)
        u = st["u"].view(rows, cols).float() * 2 ** -12
        sidx = reference.scale_index(self.weight, rows, cols // 16, u.device)
        sigma = reference.dec_e4m3(self.weight.scales.view(torch.uint8).reshape(-1)[sidx]) * \
            reference.gdec_of(self.weight, rows, u.device).repeat_interleave(cols // 16)
        return reference.g_of(u) * sigma.view(rows, cols // 16).repeat_interleave(16, dim=1)
