# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""mlops.vocab against PyTorch: embed (the rows, xw, the partial sums of squares, invalid ids), the head (every variant
against a float64 matmul, statistics against the logits, the norm scalar), the sampler (greedy against argmax with the
lowest index, determinism, the step counter, filters, the head's statistics giving the same tokens).

    pytest tests/test_vocab.py          (with mlops-vocab installed)
"""
import pytest
import torch

from mlops.vocab import Sampler, embed, embed_variants, lm_head, lm_head_variants

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"


def bf16(*shape, scale=1.0, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(*shape, device=DEV, generator=g) * scale).to(torch.bfloat16)


@pytest.mark.parametrize("hidden,n", [(1024, 1), (2560, 5), (4096, 8)])
def test_embed(hidden, n):
    vocab = 3000
    table, w = bf16(vocab, hidden, seed=1), bf16(hidden, seed=2)
    ids = torch.randint(0, vocab, (n,), dtype=torch.int32, device=DEV)
    for variant in range(len(embed_variants(vocab, hidden, n))):
        e = embed(table, ids, w, variant=variant)
        x = table[ids.long()].float()
        assert torch.equal(e.x, x)
        assert torch.equal(e.xw, (x * w.float()).to(torch.bfloat16))
        ref = x.double().pow(2).reshape(n, hidden // 256, 256).sum(-1)
        assert torch.allclose(e.parts.double(), ref, rtol=1e-5)
        assert int(e.error.item()) == 0


def test_embed_invalid_id():
    table = bf16(50, 512, seed=3)
    ids = torch.tensor([3, -1, 50, 7], dtype=torch.int32, device=DEV)
    e = embed(table, ids)
    assert int(e.error.item()) == 1
    assert torch.equal(e.x[0], table[3].float()) and torch.equal(e.x[3], table[7].float())
    assert not e.x[1].any() and not e.x[2].any()
    assert e.xw is None


@pytest.mark.parametrize("hidden,n", [(1024, 1), (2048, 3), (2048, 8)])
def test_head(hidden, n):
    vocab = 5003
    w, x = bf16(vocab, hidden, scale=0.05, seed=4), bf16(n, hidden, scale=3.0, seed=5)
    ref = x.double() @ w.double().T
    for variant in range(len(lm_head_variants(vocab, hidden, n))):
        h = lm_head(w, x, variant=variant, want_stats=True)
        mag = (x.double().abs() @ w.double().abs().T)
        assert ((h.logits.double() - ref).abs() <= (hidden / 16 + 8) * 2.0 ** -23 * mag + 1e-30).all()
        # the statistics: each element's maximum, and the best of them is the row's
        best = h.stats_max.max(1)
        assert torch.equal(best.values, h.logits.max(1).values)


def test_head_norm_scalar_and_scale():
    vocab, hidden, n = 2000, 2048, 2
    w, x = bf16(vocab, hidden, scale=0.05, seed=6), bf16(n, hidden, scale=3.0, seed=7)
    parts = torch.rand(n, 9, device=DEV) + 0.5
    plain = lm_head(w, x, variant=0).logits
    h = lm_head(w, x, norm_parts=parts, inv_n=1 / hidden, eps=1e-6, logit_scale=2.0, variant=0).logits
    s = torch.rsqrt(parts.sum(1, keepdim=True) / hidden + 1e-6)
    assert torch.allclose(h, plain * s * 2.0, rtol=1e-5)


def test_sampler_greedy_lowest_index():
    logits = torch.randn(151936, device=DEV)
    logits[[100, 900, 77000]] = 20.0
    tok = Sampler(logits.numel(), seed=1)(logits)
    assert int(tok.item()) == 100


def test_sampler_determinism_and_step():
    logits = torch.randn(30000, device=DEV) * 2
    a, b = Sampler(30000, temperature=0.8, top_k=40, top_p=0.9, seed=5), Sampler(30000, temperature=0.8, top_k=40, top_p=0.9, seed=5)
    ta = [int(a(logits).item()) for _ in range(20)]
    tb = [int(b(logits).item()) for _ in range(20)]
    assert ta == tb and a.step == 20
    assert len(set(ta)) > 1
    top = set(logits.topk(40).indices.tolist())
    assert set(ta) <= top


def test_sampler_penalty_needs_seen():
    logits = torch.randn(1000, device=DEV)
    with pytest.raises(ValueError):
        Sampler(1000, penalty=1.3)(logits)
    seen = torch.zeros(32, dtype=torch.int32, device=DEV)
    Sampler(1000, penalty=1.3)(logits, seen)


@pytest.mark.parametrize("temperature,top_k,top_p", [(0.0, 0, 1.0), (0.7, 30, 1.0), (1.0, 0, 0.9)])
def test_head_statistics_into_sampler(temperature, top_k, top_p):
    vocab, hidden = 20000, 1024
    w, x = bf16(vocab, hidden, scale=0.06, seed=8), bf16(1, hidden, scale=12.0, seed=9)
    plain = lm_head(w, x, variant=0).logits
    scale = 1.0 / temperature if temperature > 0 else 1.0
    fused = lm_head(w, x, logit_scale=scale, variant=0, want_stats=True)
    a = Sampler(vocab, temperature=temperature, top_k=top_k, top_p=top_p, seed=3)
    b = Sampler(vocab, temperature=temperature, top_k=top_k, top_p=top_p, seed=3)
    for _ in range(30):
        assert int(a(plain).item()) == int(b(fused.logits, stats=fused).item())
