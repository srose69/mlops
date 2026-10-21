# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""mlops.positional: positional encodings on CUDA (the mlops-positional package).

* ``RoPE``                    rotary position embedding of q and k from token positions, no error growth with the position,
                              with YaRN / Llama 3 / linear / dynamic NTK / LongRoPE frequency scaling
                              (``RoPE.from_hf(config)``), differentiable.
* ``rope_frequencies``        the inverse frequencies and attention factor of a scaling (float64).
* ``apply_rotary_pos_emb``    transformers' function of that name on the kernel, bit for bit, forward and backward.
* ``patch_transformers``      installs it into transformers' modeling modules.

    from mlops.positional import RoPE
    rope = RoPE.from_hf(model.config).cuda()
    q, k = rope(q, k, position_ids, layout="bhsd")
"""
import torch  # noqa: F401  (loads libtorch / libc10, which _C links)

from . import _C
from .rope import RoPE, apply_rotary_pos_emb, patch_transformers, rope_frequencies

__version__ = _C.__version__
__all__ = ["RoPE", "rope_frequencies", "apply_rotary_pos_emb", "patch_transformers", "__version__"]
