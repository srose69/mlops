# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""mlops.optim: AdamW and the NVFP4 AdamW-QAT on CUDA (the mlops-optimizers package).

* ``AdamW``         dense CUDA parameters (bfloat16 / float16 / float32), FP32 moments, matches torch.optim.AdamW.
* ``AdamWQAT``      the NVFP4 weights of ``QATLinear`` layers, trained in place: no master copy, 5 B of state per
                    weight + 8 per 16-weight block.
* ``QATLinear``     bias-free linear layer with an NVFP4 weight (NVFP4EffMA's GEMM), ``QATParameter`` its handle.
* ``split_params``  (dense, QAT) parameters of a model, for the two optimizers.
* ``functional``    the kernels as functions (``adamw``, ``QATState``, ``fp8_grad``); ``reference`` the QAT step in
                    PyTorch, bit-identical to the kernel.

    from mlops.optim import AdamW, AdamWQAT, QATLinear, split_params
"""
import torch  # noqa: F401  (loads libtorch / libc10, which _C links)

from . import _C

__version__ = _C.__version__


def _check_nvfp4effma():
    """The extension carries NVFP4EffMA's inline device code (E2M1 / E4M3 codecs, scale addressing) of the
    version it was built against, and runs on that version's NVFP4 tensors and GEMM: a different major.minor
    could disagree with it silently, so it is refused here."""
    import nvfp4effma
    have, built = nvfp4effma.__version__.split("+")[0], _C.built_nvfp4effma
    if have.split(".")[:2] != built.split(".")[:2]:
        raise ImportError(f"mlops-optimizers {__version__} was built against nvfp4effma {built}, but nvfp4effma {have} "
                          f"is installed: install nvfp4effma {'.'.join(built.split('.')[:2])}.x, or rebuild "
                          f"mlops-optimizers against {have}")


_check_nvfp4effma()

from . import functional, reference  # noqa: E402
from .optim import AdamW, AdamWQAT, QATLinear, QATParameter, split_params  # noqa: E402

__all__ = ["AdamW", "AdamWQAT", "QATLinear", "QATParameter", "split_params", "functional", "reference", "__version__"]
