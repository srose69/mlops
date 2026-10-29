# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""mlops.norm against PyTorch: RMSNorm in Hugging Face's order of roundings, the single-rounding flavors (one plus, bias, gates,
groups), LayerNorm; the quantized outputs against a dequantization within their grids; every variant; argument checks.

    pytest tests/test_norm.py          (with mlops-norm installed)
"""
import pytest
import torch
import torch.nn.functional as F

from mlops.norm import LayerNorm, RMSNorm, layer_norm, norm, norm_variants, rms_norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"
PAIRS = [(torch.bfloat16, torch.bfloat16), (torch.float32, torch.bfloat16), (torch.float16, torch.float16), (torch.float32, torch.float32)]


def tol(yd):  # the row sums add in an order of their own: a rounding flips only at the edge of a bf16 / fp16 step
    return dict(rtol=2 ** -7 if yd != torch.float32 else 1e-5, atol=1e-5)


def ref_norm(x, w, b, eps, *, kind="rms", one_plus=False, hf=False, residual=None, gate=None, gate_mode="after", groups=1, yd=None):
    v = x if residual is None else (x + residual).to(x.dtype)
    h = v.float()
    if gate is not None and gate_mode == "before":
        h = h * F.silu(gate.float())
    rows, hidden = h.shape
    g = h.reshape(rows, groups, hidden // groups)
    if kind == "layer":
        g = g - g.mean(-1, keepdim=True)
    n = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + eps)
    n = n.reshape(rows, hidden)
    if hf:
        n = n.to(x.dtype).float()
    wf = torch.ones(hidden, device=x.device) if w is None else (1 + w.float() if one_plus else w.float())
    y = n * wf
    if b is not None:
        y = y + b.float()
    if gate is not None and gate_mode == "after":
        y = y * F.silu(gate.float())
    return y.to(yd or (w.dtype if w is not None else x.dtype)), v


def data(rows, hidden, xd, wd, seed=1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = (torch.randn(rows, hidden, device=DEV, generator=g) * 3 + 0.5).to(xd)
    res = torch.randn(rows, hidden, device=DEV, generator=g).to(xd)
    w = (torch.rand(hidden, device=DEV, generator=g) + 0.5).to(wd)
    b = (torch.randn(hidden, device=DEV, generator=g) * 0.3).to(wd)
    z = torch.randn(rows, hidden, device=DEV, generator=g).to(xd)
    return x, res, w, b, z


@pytest.mark.parametrize("xd,wd", PAIRS)
@pytest.mark.parametrize("hidden", [1024, 2560, 4096])
def test_rms_hf_against_torch(xd, wd, hidden):
    x, res, w, _, _ = data(7, hidden, xd, wd)
    for variant in range(len(norm_variants(7, hidden, xd, wd))):
        out = rms_norm(x, w, 1e-6, residual=res, variant=variant)
        ref, v = ref_norm(x, w, None, 1e-6, hf=True, residual=res)
        assert torch.equal(out.residual, v)
        assert torch.allclose(out.y.float(), ref.float(), **tol(wd))
        if wd != torch.float32:
            assert (out.y != ref).float().mean() < 0.02
    plain = rms_norm(x, w, 1e-6)
    assert plain.residual is None and torch.allclose(plain.y.float(), ref_norm(x, w, None, 1e-6, hf=True)[0].float(), **tol(wd))


@pytest.mark.parametrize("xd,wd", PAIRS)
def test_flavors_against_torch(xd, wd):
    x, res, w, b, _ = data(6, 2048, xd, wd)
    cases = [
        (dict(kind="rms", one_plus=True), dict(kind="rms", one_plus=True), True, False),   # Gemma, Qwen3-Next
        (dict(kind="rms"), dict(kind="rms"), True, True),                                    # RMSNorm with a bias
        (dict(kind="rms"), dict(kind="rms"), False, False),                                  # no weight
        (dict(kind="layer"), dict(kind="layer"), True, True),                                # GPT, Falcon, Phi
        (dict(kind="layer"), dict(kind="layer"), False, False),                              # OLMo
        (dict(kind="layer", one_plus=True), dict(kind="layer", one_plus=True), True, True),  # LayerNorm1P
        (dict(kind="rms", groups=4), dict(kind="rms", groups=4), True, False),               # Mamba-2, Nemotron-H
        (dict(kind="layer", groups=2), dict(kind="layer", groups=2), True, True),
    ]
    for call, ref_kw, has_w, has_b in cases:
        out = norm(x, w if has_w else None, b if has_b else None, 1e-6, residual=res, y_dtype=wd, **call)
        ref, v = ref_norm(x, w if has_w else None, b if has_b else None, 1e-6, residual=res, yd=wd, **ref_kw)
        assert torch.equal(out.residual, v), call
        assert torch.allclose(out.y.float(), ref.float(), **tol(wd)), call


@pytest.mark.parametrize("mode,groups,kind", [("after", 1, "rms"), ("before", 4, "rms"), ("before", 1, "layer"), ("after", 1, "layer")])
def test_gates_against_torch(mode, groups, kind):
    x, res, w, b, z = data(5, 2048, torch.bfloat16, torch.bfloat16)
    bias = b if kind == "layer" else None
    out = norm(x, w, bias, 1e-6, kind=kind, gate=z, gate_mode=mode, groups=groups, residual=res)
    ref, _ = ref_norm(x, w, bias, 1e-6, kind=kind, gate=z, gate_mode=mode, groups=groups, residual=res)
    assert torch.allclose(out.y.float(), ref.float(), rtol=2 ** -6, atol=1e-3)


def test_variants_agree_bit_for_bit():
    x, res, w, b, _ = data(9, 2048, torch.bfloat16, torch.bfloat16)
    for call in (dict(kind="rms"), dict(kind="layer", one_plus=True)):
        bb = b if call["kind"] == "layer" else None
        n = len(norm_variants(9, 2048, torch.bfloat16, torch.bfloat16, bias=bb is not None, **call))
        outs = [norm(x, w, bb, residual=res, variant=i, items=k, **call).y for i in range(n) for k in (0, 1, 4)]
        assert all(torch.equal(outs[0], o) for o in outs)


@pytest.mark.parametrize("kind", ["rms", "layer"])
def test_int8_roundtrip(kind):
    x, _, w, b, _ = data(5, 2048, torch.bfloat16, torch.bfloat16)
    bb = b if kind == "layer" else None
    full = norm(x, w, bb, kind=kind).y.float()
    q = norm(x, w, bb, kind=kind, quant="int8")
    assert q.y.dtype == torch.int8 and q.scales.dtype == torch.float32 and q.scales.shape == (5, 128)
    deq = q.y.float().reshape(5, 128, 16) * q.scales.unsqueeze(-1)
    assert (deq.reshape(5, 2048) - full).abs().max() <= (q.scales.max() * 0.5 + 1e-6)


def test_nvfp4_shapes_and_range():
    x = torch.randn(4, 1024, device=DEV).to(torch.bfloat16)
    w = torch.ones(1024, device=DEV, dtype=torch.bfloat16)
    g = torch.tensor([0.02], device=DEV)
    q = layer_norm(x, w, torch.zeros_like(w), quant="nvfp4", global_scale=g)
    assert q.y.dtype == torch.uint8 and q.y.shape == (4, 512) and q.scales.shape == (4, 64) and q.scales.dtype == torch.uint8
    assert q.scales.any() and q.y.any()


def test_default_order():
    x, _, w, _, _ = data(8, 2048, torch.bfloat16, torch.bfloat16)
    # a plain weight rounds twice (Hugging Face's Llama), 1 + weight once (Gemma): the defaults are those
    assert torch.equal(rms_norm(x, w).y, rms_norm(x, w, order="hf").y)
    assert torch.equal(rms_norm(x, w, one_plus=True).y, rms_norm(x, w, one_plus=True, order="fp32").y)
    assert not torch.equal(rms_norm(x, w, order="hf").y, rms_norm(x, w, order="fp32").y)


def test_modules_and_errors():
    x = torch.randn(3, 1024, device=DEV).to(torch.bfloat16)
    assert RMSNorm(1024)(x).y.shape == (3, 1024)
    assert RMSNorm(1024, one_plus=True)(x).y.shape == (3, 1024)
    assert LayerNorm(1024)(x).y.shape == (3, 1024)
    w = torch.ones(1024, device=DEV, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        rms_norm(x, w, quant="fp8")
    with pytest.raises(ValueError):
        rms_norm(x, None, one_plus=True)
    with pytest.raises(ValueError):
        norm(x, w, kind="batch")
    with pytest.raises(ValueError):
        norm(x, w, gate=x, gate_mode="middle")
    with pytest.raises(ValueError):
        rms_norm(torch.randn(2, 1000, device=DEV).to(torch.bfloat16), torch.ones(1000, device=DEV, dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        layer_norm(x, w, w, order="hf")  # Hugging Face's order is RMSNorm's alone
    with pytest.raises((ValueError, TypeError)):
        rms_norm(x.to(torch.float16), w)  # fp16 x with a bf16 weight
