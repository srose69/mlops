# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""mlops.positional: ``RoPE`` (q and k from token positions, no error growth with the position; YaRN / Llama 3 / linear / dynamic NTK /
LongRoPE scaling via ``RoPE.from_hf(config)``; differentiable), ``rope_frequencies`` (float64), ``apply_rotary_pos_emb`` (transformers', bit for
bit, forward and backward), ``patch_transformers``. Use: ``q, k = RoPE.from_hf(model.config).cuda()(q, k, position_ids, layout="bhsd")``."""
import torch  # noqa: F401  (loads libtorch / libc10, which _C links)

from . import _C
from .rope import RoPE, apply_rotary_pos_emb, patch_transformers, rope_frequencies

__version__ = _C.__version__
__all__ = ["RoPE", "rope_frequencies", "apply_rotary_pos_emb", "patch_transformers", "__version__"]
