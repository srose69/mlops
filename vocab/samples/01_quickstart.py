# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Quickstart: embedding rows, the LM head and a sampled token, against PyTorch.

    python 01_quickstart.py
"""
import torch

from mlops.vocab import Sampler, embed, lm_head

vocab, hidden = 32000, 2048
table = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16)
head = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16)
norm_weight = torch.rand(hidden, device="cuda", dtype=torch.bfloat16) + 0.5
ids = torch.tensor([17, 4242], dtype=torch.int32, device="cuda")         # token ids live on the device

e = embed(table, ids, norm_weight)                                       # e.x float32, e.xw = bf16(x * w), e.parts: sums of squares
print("embed rows equal table[ids]:", torch.equal(e.x, table[ids.long()].float()))

x = torch.randn(2, hidden, device="cuda", dtype=torch.bfloat16)          # final hidden states of 2 tokens
h = lm_head(head, x)                                                     # logits [2, vocab], float32
ref = x.float() @ head.float().T
print("logits: max abs diff %.3g" % (h.logits - ref).abs().max())

greedy = Sampler(vocab)(h.logits, column=0)                              # int32 [1] on the device
print("greedy token == argmax:", int(greedy.item()) == int(ref[0].argmax().item()))
