# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""mlops.positional RoPE: against transformers (bit for bit with its tables, forward and backward; frequencies of
every scaling; a patched Llama model), against a float64 reference (positions, any layout, partial rotary,
interleaved pairs, large positions), autograd, argument checks.

    pytest tests/test_rope.py          (with mlops-positional installed; transformers for the comparisons with it)
"""
import math

import pytest
import torch

from mlops.positional import RoPE, apply_rotary_pos_emb, patch_transformers, rope_frequencies

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
try:
    import transformers
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.models.llama import modeling_llama
except ImportError:  # pragma: no cover
    transformers = None
needs_hf = pytest.mark.skipif(transformers is None, reason="needs transformers")

SCALINGS = {
    "default": {"rope_type": "default"},
    "linear": {"rope_type": "linear", "factor": 4.0},
    "dynamic": {"rope_type": "dynamic", "factor": 2.0},
    "yarn": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 2048},
    "yarn_mscale": {"rope_type": "yarn", "factor": 40.0, "original_max_position_embeddings": 4096, "mscale": 1.0,
                    "mscale_all_dim": 0.707, "beta_fast": 32, "beta_slow": 1},
    "llama3": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
               "original_max_position_embeddings": 8192},
    "longrope": {"rope_type": "longrope", "original_max_position_embeddings": 4096,
                 "short_factor": [1.0 + 0.01 * i for i in range(32)], "long_factor": [1.0 + 0.5 * i for i in range(32)]},
}


def hf_config(kind, head_dim=64, theta=10000.0, max_pos=16384):
    params = dict(SCALINGS[kind], rope_theta=theta)
    return LlamaConfig(hidden_size=head_dim * 4, num_attention_heads=4, num_key_value_heads=2, head_dim=head_dim,
                       max_position_embeddings=max_pos, rope_parameters=params, intermediate_size=64,
                       num_hidden_layers=1, vocab_size=64)


def ulp(x, dtype):
    mant = {torch.bfloat16: 8, torch.float16: 11, torch.float32: 24}[dtype]
    tiny = {torch.bfloat16: 1e-38, torch.float16: 6.1e-5, torch.float32: 1e-38}[dtype]
    return torch.exp2(torch.floor(torch.log2(x.abs().clamp_min(tiny))) - (mant - 1))


def ref_rope(x, pos, inv, rotary_dim, interleaved=False, att=1.0, inverse=False):
    """float64 reference on a [B, S, H, D] tensor, pos [B, S]."""
    x = x.double()
    P = rotary_dim // 2
    ang = pos.double()[..., None] * inv.to(x.device)[None, None, :]  # [B, S, P]
    c, s = (torch.cos(ang) * att)[:, :, None, :], (torch.sin(ang) * att)[:, :, None, :]
    if inverse:
        s = -s
    y = x.clone()
    if interleaved:
        x1, x2 = x[..., 0:2 * P:2], x[..., 1:2 * P:2]
        y[..., 0:2 * P:2], y[..., 1:2 * P:2] = x1 * c - x2 * s, x2 * c + x1 * s
    else:
        x1, x2 = x[..., :P], x[..., P:2 * P]
        y[..., :P], y[..., P:2 * P] = x1 * c - x2 * s, x2 * c + x1 * s
    return y


def rand(shape, dtype, seed, scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=g) * scale).to(DEV, dtype)


# ---- transformers, bit for bit ----------------------------------------------------------------------------------
@needs_hf
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("unsqueeze_dim", [1, 2])
@pytest.mark.parametrize("kind", ["default", "yarn", "llama3"])
def test_hf_apply_rotary_pos_emb_bitwise(dtype, unsqueeze_dim, kind):
    cfg = hf_config(kind, head_dim=128, theta=500000.0)
    rot = modeling_llama.LlamaRotaryEmbedding(cfg, device=DEV)
    B, S, Hq, Hk, D = 2, 77, 8, 2, 128
    pos = torch.arange(S, device=DEV)[None].expand(B, S) + torch.tensor([[0], [1000]], device=DEV)
    x = torch.empty(1, device=DEV, dtype=dtype)
    cos, sin = rot(x, pos)
    shape = (lambda h: (B, h, S, D)) if unsqueeze_dim == 1 else (lambda h: (B, S, h, D))
    q0, k0 = rand(shape(Hq), dtype, 1), rand(shape(Hk), dtype, 2)
    q1, k1 = q0.clone().requires_grad_(), k0.clone().requires_grad_()
    q2, k2 = q0.clone().requires_grad_(), k0.clone().requires_grad_()
    ya = modeling_llama.apply_rotary_pos_emb(q1, k1, cos, sin, unsqueeze_dim=unsqueeze_dim)
    yb = apply_rotary_pos_emb(q2, k2, cos, sin, unsqueeze_dim=unsqueeze_dim)
    assert torch.equal(ya[0], yb[0]) and torch.equal(ya[1], yb[1])
    gq, gk = rand(ya[0].shape, dtype, 3), rand(ya[1].shape, dtype, 4)
    torch.autograd.backward(ya, (gq, gk))
    torch.autograd.backward(yb, (gq, gk))
    assert torch.equal(q1.grad, q2.grad) and torch.equal(k1.grad, k2.grad)


@needs_hf
@pytest.mark.parametrize("kind", list(SCALINGS))
def test_frequencies_match_transformers(kind):
    cfg = hf_config(kind)
    seq_len = 8192 if kind in ("dynamic", "longrope") else None
    if kind == "default":
        inv_hf, att_hf = modeling_llama.LlamaRotaryEmbedding.compute_default_rope_parameters(cfg, "cpu")
    else:
        inv_hf, att_hf = ROPE_INIT_FUNCTIONS[cfg.rope_parameters["rope_type"]](cfg, "cpu", seq_len=seq_len)
    inv, att = rope_frequencies(64, 10000.0, cfg.rope_parameters, max_position_embeddings=cfg.max_position_embeddings,
                                seq_len=seq_len)
    assert inv.dtype == torch.float64 and inv.shape == inv_hf.shape
    assert torch.allclose(inv, inv_hf.double(), rtol=2e-6, atol=0), (inv - inv_hf.double()).abs().max()  # HF: float32
    assert math.isclose(att, float(att_hf), rel_tol=1e-12)


@needs_hf
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["default", "yarn", "longrope"])
def test_rope_from_hf_against_transformers(dtype, kind):
    """RoPE.from_hf from positions: at least as close to the exact rotation as transformers' pipeline (which rounds
    the FP32 angle and cos / sin to the dtype), and within an ulp of it."""
    cfg = hf_config(kind, head_dim=64, theta=10000.0)
    rope = RoPE.from_hf(cfg)
    rot = modeling_llama.LlamaRotaryEmbedding(cfg, device=DEV)
    B, S = 2, 300
    pos = torch.arange(S, device=DEV)[None].expand(B, S) * 13 + 3  # up to ~4000
    q, k = rand((B, 4, S, 64), dtype, 5), rand((B, 2, S, 64), dtype, 6)
    yq, yk = rope(q, k, pos, layout="bhsd")
    cos, sin = rot(q, pos)
    hq, hk = modeling_llama.apply_rotary_pos_emb(q, k, cos, sin)
    inv, att = rope_frequencies(64, 10000.0, cfg.rope_parameters, max_position_embeddings=cfg.max_position_embeddings,
                                seq_len=int(pos.max()) + 1)
    for x, y, h in ((q, yq, hq), (k, yk, hk)):
        ref = ref_rope(x.transpose(1, 2), pos, inv, 64, att=att).transpose(1, 2)
        err_ours, err_hf = (y.double() - ref).abs(), (h.double() - ref).abs()
        assert float(err_ours.max()) <= float(err_hf.max()) * 1.0001
        assert bool((err_ours <= 0.5 * ulp(ref, dtype) + 1e-6 * ref.abs().max()).all())


@needs_hf
def test_patch_transformers_llama_bitwise():
    cfg = hf_config("yarn", head_dim=32, theta=10000.0)
    cfg.hidden_size, cfg.num_hidden_layers = 128, 2
    torch.manual_seed(0)
    model = transformers.LlamaModel(cfg).to(DEV, torch.bfloat16).eval()
    ids = torch.randint(0, 64, (2, 50), device=DEV)
    with torch.no_grad():
        ref = model(ids).last_hidden_state
        orig = modeling_llama.apply_rotary_pos_emb
        try:
            assert patch_transformers("llama") == ["transformers.models.llama.modeling_llama"]
            out = model(ids).last_hidden_state
        finally:
            modeling_llama.apply_rotary_pos_emb = orig
    assert torch.equal(ref, out)


# ---- positions, against float64 ---------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["bshd", "bhsd", "thd"])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("rotary_dim", [64, 32, 20])
def test_rope_exact(dtype, layout, interleaved, rotary_dim):
    D, B, S, Hq, Hk = 64, (1 if layout == "thd" else 3), 41, 6, 2
    rope = RoPE(D, 10000.0, rotary_dim=rotary_dim, interleaved=interleaved)
    shape = {"bshd": lambda h: (B, S, h, D), "bhsd": lambda h: (B, h, S, D), "thd": lambda h: (S, h, D)}[layout]
    q, k = rand(shape(Hq), dtype, 7), rand(shape(Hk), dtype, 8, 4.0)
    pos = torch.randint(0, 100000, (B, S), device=DEV, dtype=torch.int32)
    yq, yk = rope(q, k, pos[0] if layout == "thd" else pos, layout=layout)
    to4 = {"bshd": lambda t: t, "bhsd": lambda t: t.transpose(1, 2), "thd": lambda t: t.unsqueeze(0)}[layout]
    for x, y in ((q, yq), (k, yk)):
        ref = ref_rope(to4(x), pos, rope.inv_freq, rotary_dim, interleaved)
        err = (to4(y).double() - ref).abs()
        bound = 0.5 * ulp(ref, dtype) + 6e-7 * to4(x).double().abs().amax(-1, keepdim=True)
        assert bool((err <= bound).all()), float((err - bound).max())
        assert torch.equal(to4(y)[..., rotary_dim:], to4(x)[..., rotary_dim:])  # pass-through


def test_positions_forms_and_inplace():
    rope = RoPE(128, 500000.0)
    B, S = 4, 33
    q, k = rand((B, S, 8, 128), torch.bfloat16, 9), rand((B, S, 2, 128), torch.bfloat16, 10)
    pos = (torch.arange(S, device=DEV) + 70)
    a = rope(q, k, pos[None].expand(B, S).contiguous())
    assert all(torch.equal(x, y) for x, y in zip(a, rope(q, k, pos)))                     # [S] broadcast
    assert all(torch.equal(x, y) for x, y in zip(a, rope(q, k, offset=70)))               # implicit positions
    assert all(torch.equal(x, y) for x, y in zip(a, rope(q, k, pos.to(torch.int32))))    # int32
    assert torch.equal(rope(q, None, pos)[0], a[0])                                       # without k
    qi, ki = q.clone(), k.clone()
    with torch.no_grad():
        rope(qi, ki, pos, inplace=True)
    assert torch.equal(qi, a[0]) and torch.equal(ki, a[1])
    # a strided q (a slice of a packed qkv projection)
    qkv = rand((B, S, 12, 128), torch.bfloat16, 11)
    q_s, k_s = qkv[:, :, :8], qkv[:, :, 8:10]
    assert torch.equal(rope(q_s, k_s, pos)[0], rope(q_s.contiguous(), k_s.contiguous(), pos)[0])


def test_large_positions_exact():
    """positions near 2^31: the phase is reduced exactly (transformers' FP32 angle would be off by up to ~100 rad)."""
    rope = RoPE(64, 10000.0)
    q = rand((1, 16, 4, 64), torch.float32, 12)
    pos = torch.arange(16, device=DEV, dtype=torch.int64)[None] + (2 ** 31 - 100)
    y, _ = rope(q, None, pos)
    ref = ref_rope(q, pos, rope.inv_freq, 64)
    assert float((y.double() - ref).abs().max()) < 4e-6


def test_scaling_attention_factor():
    s = {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096}
    rope = RoPE(64, 10000.0, scaling=s)
    assert math.isclose(rope.attention_factor, 0.1 * math.log(8.0) + 1)
    q = rand((2, 9, 3, 64), torch.float32, 13)
    pos = torch.arange(9, device=DEV)[None].expand(2, 9) + 20000
    y, _ = rope(q, None, pos)
    ref = ref_rope(q, pos, rope.inv_freq, 64, att=rope.attention_factor)
    assert float((y.double() - ref).abs().max()) < 1e-5


@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
def test_autograd_is_inverse_rotation(layout):
    rope = RoPE(64, 10000.0, rotary_dim=48)
    shp = (2, 17, 4, 64) if layout == "bshd" else (2, 4, 17, 64)
    q, k = rand(shp, torch.float32, 14).requires_grad_(), rand(shp, torch.float32, 15).requires_grad_()
    pos = torch.randint(0, 5000, (2, 17), device=DEV)
    yq, yk = rope(q, k, pos, layout=layout)
    gq, gk = rand(shp, torch.float32, 16), rand(shp, torch.float32, 17)
    torch.autograd.backward((yq, yk), (gq, gk))
    to4 = (lambda t: t) if layout == "bshd" else (lambda t: t.transpose(1, 2))
    for g, x in ((gq, q), (gk, k)):
        want = ref_rope(to4(g), pos, rope.inv_freq, 48, inverse=True)
        assert float((to4(x.grad).double() - want).abs().max()) < 2e-6


def test_rejects_bad_input():
    rope = RoPE(64)
    q = rand((2, 8, 4, 64), torch.bfloat16, 18)
    with pytest.raises(TypeError):
        rope(q.double())
    with pytest.raises(ValueError):
        rope(q.cpu())
    with pytest.raises(ValueError):
        rope(rand((2, 8, 4, 32), torch.bfloat16, 1))           # head_dim mismatch
    with pytest.raises(ValueError):
        rope(q, rand((2, 9, 2, 64), torch.bfloat16, 1))         # k with another sequence length
    with pytest.raises(TypeError):
        rope(q, None, torch.zeros(2, 8, device=DEV))           # float positions
    with pytest.raises(ValueError):
        rope(q, None, torch.zeros(3, 8, device=DEV, dtype=torch.int64))
    with pytest.raises(ValueError):
        rope(q, layout="sbhd")
    with pytest.raises(ValueError):
        RoPE(64, rotary_dim=66)
    with pytest.raises(ValueError):
        rope_frequencies(64, scaling={"rope_type": "su"})
    with pytest.raises(ValueError):
        rope_frequencies(64, scaling={"rope_type": "yarn", "factor": 4.0})  # no original_max_position_embeddings
    with pytest.raises(RuntimeError):
        rope(q.clone().requires_grad_(), inplace=True)
