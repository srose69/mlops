# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""mlops.vocab: everything along the vocabulary axis on CUDA. ``embed``: rows for device-side token ids, with the first RMSNorm's inputs;
``lm_head``: the BF16 head for 1 .. 8 tokens, cut along hidden, with the sampler's statistics in its epilogue; ``Sampler``: greedy / temperature /
top-k / top-p / repetition penalty without sorting, seed and step on the device: ``Sampler(vocab, temperature=0.7, top_p=0.9, seed=1)(logits[0])``."""
import torch  # noqa: F401  (loads libtorch / libc10, which _C links)

from . import _C
from .ops import Embedding, Head, Sampler, embed, embed_variants, lm_head, lm_head_variants

__version__ = _C.__version__
__all__ = ["embed", "embed_variants", "Embedding", "lm_head", "lm_head_variants", "Head", "Sampler", "__version__"]
