# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""A sampling loop with no host values: the head's statistics feed the sampler, the seed and the step live on the device.

    python 02_decode_loop.py
"""
import torch

from mlops.vocab import Sampler, lm_head

vocab, hidden, temperature = 32000, 2048, 0.7
head = (torch.randn(vocab, hidden, device="cuda") * 0.02).to(torch.bfloat16)      # logits of std ~1: a spread-out distribution
x = torch.randn(1, hidden, device="cuda", dtype=torch.bfloat16)


def run(seed):
    sampler = Sampler(vocab, temperature=temperature, top_k=50, top_p=0.9, seed=seed)
    tokens = []
    for _ in range(8):                                                   # a real loop feeds the token back; here x is fixed
        h = lm_head(head, x, logit_scale=1 / temperature, want_stats=True)
        tokens.append(sampler(h.logits, stats=h).clone())
    return torch.cat(tokens).tolist(), sampler.step


a, steps = run(1)
print("seed 1:", a, "(step counter %d)" % steps)
print("seed 1 again:", run(1)[0] == a, " seed 2 differs:", run(2)[0] != a)
