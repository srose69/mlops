# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""mlops.norm: normalization layers on CUDA (the mlops-norm package).

* ``norm`` / ``rms_norm`` / ``layer_norm``  RMSNorm and LayerNorm of rows in the flavors of current models (weight, one plus
  weight, none; bias; Hugging Face's or the single-rounding order; silu gates; groups), with an optional residual add before
  and an INT8 / NVFP4 quantization after, the quantized outputs being what NVFP4EffMA's quantizer makes of y.

    out = rms_norm(x, weight, 1e-6, residual=h, quant="int8")        # out.y int8, out.scales FP32 [rows, hidden / 16]
    out = layer_norm(x, weight, bias)                                # out.y in the weight's dtype
"""
import torch  # noqa: F401  (loads libtorch / libc10, which _C links)

from . import _C
from .ops import LayerNorm, NormOut, RMSNorm, layer_norm, norm, norm_variants, rms_norm

__version__ = _C.__version__
__all__ = ["norm", "rms_norm", "layer_norm", "norm_variants", "RMSNorm", "LayerNorm", "NormOut", "__version__"]
