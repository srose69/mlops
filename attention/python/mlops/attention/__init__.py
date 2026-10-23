# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""FBLA compressed-KV FlashAttention prefill and decode."""
import torch  # loads libtorch before the extension
from . import _C
from .fbla import FBLACache, append, decode, empty, encode, prefill

__version__ = _C.__version__
__all__ = ["FBLACache", "empty", "encode", "append", "prefill", "decode", "__version__"]
