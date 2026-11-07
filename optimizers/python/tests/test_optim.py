# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""torch.optim front ends: mlops.optim.AdamW vs torch.optim.AdamW, the parameter checks of both optimizers, QATLinear + AdamWQAT (trains, equals
QATState's step, state_dict round trip, dtype casts leave it intact). Run: python tests/test_optim.py (with mlops-optimizers and nvfp4effma installed)"""
import io
import sys

import torch
import torch.nn as nn

from mlops import optim
from mlops.optim import functional, reference


def check(ok, what):
    print(f"[{'PASS' if ok else 'FAIL'}] {what}")
    return 0 if ok else 1


def test_adamw():
    f = 0
    torch.manual_seed(0)
    for dt in (torch.float32, torch.bfloat16):
        p = nn.Parameter(torch.randn(1000, 33, device="cuda", dtype=dt))
        ref = nn.Parameter(p.detach().float().clone())
        opt, opt_ref = optim.AdamW([p], lr=1e-2, weight_decay=0.1), torch.optim.AdamW([ref], lr=1e-2, weight_decay=0.1)
        for _ in range(20):
            g = torch.randn_like(ref).to(dt).float()
            p.grad, ref.grad = g.to(dt), g.clone()
            opt.step(); opt_ref.step()
            with torch.no_grad():
                ref.copy_(ref.to(dt).float())  # the parameter is stored in dt after every step (moments FP32)
        err = ((p.float() - ref).abs().max() / ref.abs().max()).item()
        f += check(err < (1e-6 if dt == torch.float32 else 2 ** -7), f"AdamW {dt} vs torch.optim.AdamW (parameter rounded to {dt} "
                   f"each step), 20 steps: max rel. err {err:.2e}")
    return f


def test_refusals():
    f = 0
    lin = optim.QATLinear(64, 32)
    dense = nn.Parameter(torch.randn(4, 4, device="cuda"))
    for opt, params, what in ((optim.AdamW, [lin.weight], "AdamW refuses an NVFP4 QATParameter"),
                              (optim.AdamWQAT, [dense], "AdamWQAT refuses a dense parameter"),
                              (optim.AdamWQAT, [lin.weight, dense], "AdamWQAT refuses a mixed parameter list"),
                              (optim.AdamW, [nn.Parameter(torch.randn(4, device="cpu"))], "AdamW refuses a CPU parameter")):
        try:
            opt(params)
            f += check(False, what)
        except TypeError as e:
            f += check(True, f"{what}: {e}")
    opt = optim.AdamWQAT([lin.weight])
    try:
        opt.add_param_group({"params": [dense]})
        f += check(False, "AdamWQAT.add_param_group refuses a dense parameter")
    except TypeError:
        f += check(len(opt.param_groups) == 1, "AdamWQAT.add_param_group refuses a dense parameter (groups unchanged)")
    try:
        optim.AdamWQAT([lin.weight], amsgrad=True)
        f += check(False, "AdamWQAT refuses AdamW options it does not implement (amsgrad)")
    except TypeError:
        f += check(True, "AdamWQAT refuses AdamW options it does not implement (amsgrad)")
    return f


def test_fp8():
    f = 0
    torch.manual_seed(2)
    w0 = torch.randn(64, 128, device="cuda") * 0.02
    x = torch.randn(256, 128, device="cuda")
    y = x @ (torch.randn(64, 128, device="cuda") * 0.02).t()
    lin = optim.QATLinear(128, 64, weight=w0, grad_dtype=torch.float8_e5m2)
    lin32 = optim.QATLinear(128, 64, weight=w0)
    opt = optim.AdamWQAT([lin.weight], lr=1e-3, seed=3)
    st = functional.QATState(w0)
    ok = True
    for t in range(1, 51):
        ((lin(x) - y) ** 2).mean().backward()
        lin32.load_state_dict(lin.state_dict())
        ((lin32(x) - y) ** 2).mean().backward()  # the same float32 gradient, encoded as the layer does
        g8, gs = functional.fp8_grad(lin32.weight.grad, torch.float8_e5m2)
        lin32.weight.grad = None
        ok &= lin.weight.grad is None and lin.weight.grad_fp8[0].dtype == torch.float8_e5m2 and \
            torch.equal(lin.weight.grad_fp8[0].view(torch.uint8), g8.view(torch.uint8)) and lin.weight.grad_fp8[1] == gs
        opt.step(); opt.zero_grad()
        st.step(g8, lr=1e-3, step=t, weight_decay=1e-2, grad_scale=gs, seed=(3 + 0x9E3779B9) & 0xFFFFFFFF)
        ok &= lin.weight.grad_fp8 is None
    ok &= torch.equal(lin.nvfp4_data, st.weight.data) and torch.equal(lin.nvfp4_scales, st.weight.scales.view(torch.uint8))
    f += check(ok, "grad_dtype=float8_e5m2: backward fills .grad_fp8 (== fp8_grad of the float32 gradient), "
                   "AdamWQAT == QATState.step on it, 50 steps, zero_grad clears it")

    ((lin(x[:128]) - y[:128]) ** 2).sum().backward()
    ((lin(x[128:]) - y[128:]) ** 2).sum().backward()
    lin32.load_state_dict(lin.state_dict())
    ((lin32(x) - y) ** 2).sum().backward()
    g8, gs = lin.weight.grad_fp8
    err = ((g8.float() * gs - lin32.weight.grad).abs().max() / lin32.weight.grad.abs().max()).item()
    f += check(err < 2 ** -2, f"grad_dtype=float8_e5m2: two backwards accumulate (max rel. err vs float32 sum {err:.2e}, E5M2)")
    return f


def test_qat():
    f = 0
    torch.manual_seed(1)
    w0 = torch.randn(64, 128, device="cuda") * 0.02
    target = torch.randn(64, 128, device="cuda") * 0.02
    x = torch.randn(512, 128, device="cuda")
    y = x @ target.t()

    for sdt in ("ue5m3", "e4m3"):  # E4M3 last: the checks below use its layer
        lin = optim.QATLinear(128, 64, weight=w0, scale_dtype=sdt)
        opt = optim.AdamWQAT([lin.weight], lr=1e-3, weight_decay=0.0, seed=5)
        st = functional.QATState(w0, scale_dtype=sdt)  # the same step through the low-level API, same stream (parameter 1)
        losses = []
        for t in range(1, 301):
            loss = ((lin(x) - y) ** 2).mean()
            loss.backward()
            g = lin.weight.grad.clone()
            opt.step(); opt.zero_grad()
            st.step(g, lr=1e-3, step=t, weight_decay=0.0, seed=(5 + 0x9E3779B9) & 0xFFFFFFFF)
            losses.append(loss.item())
        a, b = reference.carve(lin.qat_state, 64, 128), reference.carve(st.state, 64, 128)
        same = all(torch.equal(a[k], b[k]) for k in ("u", "r", "s", "qbar", "s_exp", "cusum", "hnum", "hden")) and \
            torch.equal(lin.nvfp4_data, st.weight.data) and torch.equal(lin.nvfp4_scales, st.weight.scales.view(torch.uint8))
        f += check(same, f"{sdt.upper()}: QATLinear + AdamWQAT == QATState.step on the same gradients, 300 steps (state and codes bit-identical)")
        f += check(losses[-1] < 0.2 * losses[0], f"{sdt.upper()}: QATLinear + AdamWQAT fits a linear map: loss {losses[0]:.3e} -> {losses[-1]:.3e}")
    f += check(lin.weight.untyped_storage().nbytes() == 4, "the weight handle holds 4 bytes (the weight is the NVFP4 buffers)")

    buf = io.BytesIO()
    torch.save(lin.state_dict(), buf)
    lin2 = optim.QATLinear(128, 64)
    lin2.load_state_dict(torch.load(io.BytesIO(buf.getvalue())))
    f += check(torch.equal(lin2(x), lin(x)) and lin2.scale_dtype == "e4m3",
               f"state_dict round trip, E4M3 checkpoint into a UE5M3 layer: same output, format restored "
               f"(keys {sorted(lin.state_dict())})")
    old = {k: v for k, v in lin.state_dict().items() if k != "nvfp4_format"}  # a checkpoint from before UE5M3
    lin3 = optim.QATLinear(128, 64)
    lin3.load_state_dict(old)
    f += check(torch.equal(lin3(x), lin(x)) and lin3.scale_dtype == "e4m3", "a checkpoint without nvfp4_format loads as E4M3")

    xh = x.half()
    ref = lin(xh)
    model = nn.Sequential(lin2).half()
    out = model(xh)
    f += check(out.dtype == torch.float16 and lin2.weight.dtype == torch.float32 and torch.equal(out, ref),
               "model.half(): NVFP4 buffers and the float32 handle untouched, same output")
    return f


def test_to_e4m3():
    """to_e4m3: a trained UE5M3 weight as E4M3 scales, the same values; refused when the scales do not fit."""
    import nvfp4effma
    f = 0
    torch.manual_seed(4)
    x = torch.randn(64, 256, device="cuda")
    lin = optim.QATLinear(256, 128)
    opt = optim.AdamWQAT([lin.weight], lr=1e-3)
    for _ in range(200):
        (lin(x) - x[:, :128]).pow(2).mean().backward(); opt.step(); opt.zero_grad()
    q = lin.nvfp4_weight()
    for swz in (False, True):
        qs = q.with_scale_layout(swz)
        e = optim.to_e4m3(qs)
        same = torch.equal(e.dequantize(torch.float32), q.dequantize(torch.float32)) and \
            torch.equal(nvfp4effma.linear(x, e, act="exact"), nvfp4effma.linear(x, q, act="exact"))
        f += check(not e.ue5m3 and e.swizzled == swz and same,
                   f"to_e4m3 of a trained UE5M3 weight ({'swizzled' if swz else 'linear'}): E4M3 scales, dequantized "
                   f"weight and nvfp4effma.linear output bit-identical")
    w = torch.randn(16, 64, device="cuda")
    w[0] *= 1e6; w[1] *= 1e-6                                          # 40 binades between two blocks
    wide = nvfp4effma.quantize(w, scale_dtype="ue5m3")
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kept = optim.to_e4m3(wide)
    f += check(kept is wide and kept.ue5m3 and len(caught) == 1 and "NOT E4M3" in str(caught[0].message),
               f"to_e4m3 of scales spanning more than E4M3 holds: returned as is (UE5M3) with a warning "
               f"({caught[0].message if caught else 'none'})")
    return f


def test_act():
    """Every act: forward == nvfp4effma.linear, output in x's dtype, backward exact for the GEMM performed
    (grad_q = gy.T @ A(x), grad_x = gy @ q) against dense autograd on the dequantized operands."""
    import nvfp4effma
    f = 0
    torch.manual_seed(3)
    for act in ("exact", "int8", "nvfp4", "auto"):
        for dt, rows in ((torch.float32, 96), (torch.bfloat16, 96), (torch.float16, 96), (torch.float32, 2)):
            lin = optim.QATLinear(128, 64, act=act)
            x = torch.randn(3, rows // 3 if rows > 2 else 1, 128, device="cuda", dtype=dt).requires_grad_(rows > 2)
            if rows == 2:
                x = torch.randn(2, 128, device="cuda", dtype=dt)
            y = lin(x)
            w = lin.nvfp4_weight()
            x2 = x.detach().reshape(-1, 128)
            mode = act if act != "auto" else ("exact" if x2.shape[0] <= nvfp4effma.AUTO_EXACT_MAX_ROWS else "int8")
            ref_y = nvfp4effma.linear(x2, w, act=mode) if mode != "nvfp4" else \
                nvfp4effma.linear(nvfp4effma.quantize(x2), w, out_dtype=dt)
            ok = y.dtype == dt and torch.equal(y.reshape(-1, 64), ref_y)
            if mode == "exact":
                xa = x2.float()
            elif mode == "int8":
                xa = nvfp4effma.quantize_int8(x2).dequantize(torch.float32)
            else:
                xa = nvfp4effma.quantize(x2).dequantize(torch.float32)
            gy = torch.randn_like(y)
            y.backward(gy)
            q = w.dequantize(torch.float32)
            gq = gy.reshape(-1, 64).float().t() @ xa
            ok &= torch.allclose(lin.weight.grad, gq, rtol=1e-5, atol=1e-5)
            if x.requires_grad:
                ok &= x.grad.dtype == dt and torch.allclose(x.grad.float().reshape(-1, 128), (gy.reshape(-1, 64).float() @ q).to(dt).float(),
                                                            rtol=1e-2, atol=1e-3)
            f += check(ok, f"act={act:5s} x {str(dt):14s} {x2.shape[0]:3d} rows ({mode}): forward == nvfp4effma.linear, "
                           f"output {y.dtype}, grad_q = gy.T @ A(x), grad_x = gy @ q")
    return f


def test_grouped():
    """QATGroupedLinear: the forward is one launch equal to nvfp4effma.linear per group (bit for bit where that runs the tiled kernel, within the GEMV
    kernel's summation order otherwise), the backward exact group by group, trains with AdamWQAT, state_dict round trip, FP8 gradients."""
    import nvfp4effma
    f = 0
    torch.manual_seed(7)
    G, N, K = 4, 96, 128
    for counts in ([70, 64, 130, 57], [0, 5, 300, 1], [200, 0, 0, 90]):
        for dt in (torch.bfloat16, torch.float32):
            lin = optim.QATGroupedLinear(G, K, N, scale_dtype="ue5m3")
            R = sum(counts)
            seg = torch.tensor([0] + list(torch.tensor(counts).cumsum(0)), dtype=torch.int32, device="cuda")
            x = torch.randn(R, K, device="cuda", dtype=dt, requires_grad=True)
            y = lin(x, seg)
            xq = nvfp4effma.quantize_int8(x.detach())
            q = lin.nvfp4_weight().dequantize(torch.float32).view(G, N, K)
            xa = xq.dequantize(torch.float32)
            ref = torch.zeros(R, N, device="cuda", dtype=dt)
            for g in range(G):
                a, b = int(seg[g]), int(seg[g + 1])
                if b == a:
                    continue
                ref[a:b] = (xa[a:b] @ q[g].t()).to(dt)
            err = ((y.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
            f += check(y.dtype == dt and err < (2 ** -6 if dt == torch.bfloat16 else 1e-5),
                       f"QATGroupedLinear {str(dt):14s} counts {counts}: forward vs per-group dense product on the dequantized operands, max rel. err {err:.1e}")
            gy = torch.randn_like(y)
            y.backward(gy)
            gq = torch.zeros(G * N, K, device="cuda")
            gx = torch.zeros(R, K, device="cuda")
            for g in range(G):
                a, b = int(seg[g]), int(seg[g + 1])
                if b > a:
                    gq[g * N:(g + 1) * N] = gy[a:b].float().t() @ xa[a:b]
                    gx[a:b] = gy[a:b].float() @ q[g]
            f += check(torch.allclose(lin.weight.grad, gq, rtol=1e-4, atol=1e-4) and
                       torch.allclose(x.grad.float(), gx.to(dt).float(), rtol=1e-2, atol=1e-3),
                       f"QATGroupedLinear {str(dt):14s} counts {counts}: grad_q = gy_g.T @ A(x)_g, grad_x = gy_g @ q_g per group")
    # the layer built from given weights computes the per-group products of those weights
    w = torch.randn(G, N, K, device="cuda") * 0.05
    counts = [70, 64, 130, 100]
    seg = torch.tensor([0] + list(torch.tensor(counts).cumsum(0)), dtype=torch.int32, device="cuda")
    x = torch.randn(sum(counts), K, device="cuda", dtype=torch.bfloat16)
    glin = optim.QATGroupedLinear(G, K, N, weight=w, scale_dtype="ue5m3")
    q = glin.nvfp4_weight()
    yg = glin(x, seg)
    xq = nvfp4effma.quantize_int8(x)
    qd = nvfp4effma.dequantize(q, torch.float32).view(G, N, K)
    xd = xq.dequantize(torch.float32)
    ref = torch.cat([(xd[int(seg[g]):int(seg[g + 1])] @ qd[g].t()).to(torch.bfloat16) for g in range(G)])
    f += check(torch.allclose(yg.float(), ref.float(), rtol=2 ** -6, atol=1e-3), "QATGroupedLinear == the per-group products of the same weights")
    # trains: fits G linear maps with AdamWQAT
    target = torch.randn(G, N, K, device="cuda") * 0.02
    lin = optim.QATGroupedLinear(G, K, N, weight=torch.randn(G, N, K, device="cuda") * 0.02)
    opt = optim.AdamWQAT([lin.weight], lr=1e-3, weight_decay=0.0, seed=3)
    x = torch.randn(sum(counts), K, device="cuda")
    yt = torch.cat([x[int(seg[g]):int(seg[g + 1])] @ target[g].t() for g in range(G)])
    losses = []
    for _ in range(200):
        loss = ((lin(x, seg) - yt) ** 2).mean()
        loss.backward(); opt.step(); opt.zero_grad()
        losses.append(loss.item())
    f += check(losses[-1] < 0.2 * losses[0], f"QATGroupedLinear + AdamWQAT fits {G} linear maps: loss {losses[0]:.3e} -> {losses[-1]:.3e}")
    buf = io.BytesIO()
    torch.save(lin.state_dict(), buf)
    lin2 = optim.QATGroupedLinear(G, K, N)
    lin2.load_state_dict(torch.load(io.BytesIO(buf.getvalue())))
    f += check(torch.equal(lin2(x, seg), lin(x, seg)), "QATGroupedLinear state_dict round trip: same output")
    # FP8 gradients
    l8 = optim.QATGroupedLinear(G, K, N, weight=torch.randn(G, N, K, device="cuda") * 0.02, grad_dtype=torch.float8_e5m2)
    o8 = optim.AdamWQAT([l8.weight], lr=1e-3, weight_decay=0.0)
    loss = ((l8(x, seg) - yt) ** 2).mean()
    loss.backward()
    f += check(l8.weight.grad is None and l8.weight.grad_fp8 is not None, "QATGroupedLinear with grad_dtype=float8_e5m2 hands AdamWQAT FP8 gradients")
    o8.step()
    return f


def test_grouped_kernels():
    """mlops_qat_grouped_dgrad / _wgrad against per-group torch matmuls: every dtype pair, empty groups, n and k that are not tile multiples, rows past seg[groups]."""
    f = 0
    torch.manual_seed(11)
    G = 5
    for n, k in ((96, 128), (130, 70)):
        counts = [0, 17, 64, 1, 90]
        seg = torch.tensor([0] + list(torch.tensor(counts).cumsum(0)), dtype=torch.int32, device="cuda")
        R = sum(counts) + 7
        for gdt in (torch.float32, torch.bfloat16, torch.float16):
            gy = torch.randn(R, n, device="cuda", dtype=gdt)
            x = torch.randn(R, k, device="cuda", dtype=gdt)
            w = torch.randn(G * n, k, device="cuda")
            dx = torch.ops.mlops_optim.grouped_dgrad(gy, w, seg, G, gdt)
            dw = torch.ops.mlops_optim.grouped_wgrad(gy, x, seg, G)
            ok = dx.shape == (R, k) and dw.shape == (G * n, k) and dx.dtype == gdt
            for g in range(G):
                a, b = int(seg[g]), int(seg[g + 1])
                ref_dx = gy[a:b].double() @ w[g * n:(g + 1) * n].double()
                ref_dw = gy[a:b].double().t() @ x[a:b].double()
                tol = 1e-5 if gdt == torch.float32 else 2 ** -6  # the output rounded once to bf16 / fp16
                ok &= b == a or torch.allclose(dx[a:b].double(), ref_dx, rtol=tol, atol=tol * float(ref_dx.abs().max()) + 1e-6)
                ok &= torch.allclose(dw[g * n:(g + 1) * n].double(), ref_dw, rtol=1e-4, atol=1e-4)
            ok &= not dx[int(seg[G]):].any()  # rows past seg[groups]
            f += check(ok, f"grouped dgrad / wgrad n={n} k={k} {str(gdt):14s}: per-group products, empty groups zero, rows past seg[groups] zero")
    ops = torch.ops.mlops_optim
    gy, w = torch.randn(40, 64, device="cuda"), torch.randn(128, 64, device="cuda")
    seg = torch.tensor([0, 10, 40], dtype=torch.int32, device="cuda")
    torch.library.opcheck(ops.grouped_dgrad, (gy, w, seg, 2, torch.float32))
    torch.library.opcheck(ops.grouped_wgrad, (gy, torch.randn(40, 64, device="cuda"), seg, 2))
    f += check(True, "opcheck of grouped_dgrad / grouped_wgrad: schema and Meta kernels agree with the CUDA ones")
    return f


def graph_step(lin, x, seg, gy, graph_static=None):
    y = lin(x, seg) if seg is not None else lin(x)
    y.backward(gy)
    return y


def test_graph_capture():
    """Forward and backward of QATLinear and QATGroupedLinear (every act, float32 and FP8 gradients) captured once in a CUDA graph and replayed with new
    inputs and new row counts: the outputs and gradients equal the eager ones. Nothing in them reads back to the host."""
    f = 0
    torch.manual_seed(13)
    G, N, K = 3, 96, 128
    w = torch.randn(G * N, K, device="cuda") * 0.05
    n = 200
    segs = [torch.tensor(s + [n], dtype=torch.int32, device="cuda") for s in ([0, 70, 70], [0, 20, 150])]
    for grouped in (False, True):
        for act in ("exact", "int8", "nvfp4"):
            for gd in (None, torch.float8_e5m2):
                make = (lambda: optim.QATGroupedLinear(G, K, N, weight=w, act=act, grad_dtype=gd, scale_dtype="ue5m3")) if grouped else \
                       (lambda: optim.QATLinear(K, N, weight=w[:N], act=act, grad_dtype=gd, scale_dtype="ue5m3"))
                lin, ref = make(), make()
                sx = torch.zeros(n, K, device="cuda", requires_grad=True)
                sseg = segs[0].clone() if grouped else None
                sgy = torch.zeros(n, N, device="cuda")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    sx.data.normal_()
                    for _ in range(2):
                        lin.weight.grad, lin.weight.grad_fp8, sx.grad = None, None, None
                        graph_step(lin, sx, sseg, sgy)
                torch.cuda.current_stream().wait_stream(stream)
                lin.weight.grad, lin.weight.grad_fp8, sx.grad = None, None, None
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    sy = graph_step(lin, sx, sseg, sgy)
                ok = True
                for i in range(2):
                    xin = torch.randn(n, K, device="cuda")
                    gyin = torch.randn(n, N, device="cuda")
                    sx.data.copy_(xin)
                    sgy.copy_(gyin)
                    if grouped:
                        sseg.copy_(segs[i])
                    graph.replay()
                    xr = xin.clone().requires_grad_()
                    ref.weight.grad, ref.weight.grad_fp8 = None, None
                    yr = graph_step(ref, xr, segs[i] if grouped else None, gyin)
                    ok &= torch.equal(sy, yr) and torch.allclose(sx.grad, xr.grad, rtol=1e-4, atol=1e-4)
                    if gd is None:
                        ok &= torch.allclose(lin.weight.grad, ref.weight.grad, rtol=1e-4, atol=1e-4)
                    else:
                        g8, gs = lin.weight.grad_fp8
                        g8r, gsr = ref.weight.grad_fp8
                        ok &= torch.allclose(g8.float() * gs, g8r.float() * gsr, rtol=0.13, atol=0.13 * float((g8r.float() * gsr).abs().max()))
                f += check(ok, f"{'QATGroupedLinear' if grouped else 'QATLinear       '} act={act:5s} grad_dtype={str(gd):20s}: CUDA graph replay == eager "
                               f"(new inputs{', new row counts' if grouped else ''})")
    return f


def test_operators():
    """torch.ops.mlops_optim.*: opcheck of the optimizer operators (what they mutate) and the layers' forward / backward under torch.compile (aot_eager,
    fullgraph: no graph break at the extension)."""
    f = 0
    ops = torch.ops.mlops_optim
    p = torch.randn(256, device="cuda")
    torch.library.opcheck(ops.adamw, (p, torch.randn_like(p), torch.zeros_like(p), torch.zeros_like(p), 1e-3, 0.9, 0.999, 1e-8, 0.0, 1, True, 1.0))
    f += check(True, "opcheck of adamw")
    G, N, K = 3, 64, 128
    lin = optim.QATGroupedLinear(G, K, N)
    seg = torch.tensor([0, 30, 30, 100], dtype=torch.int32, device="cuda")
    x = torch.randn(100, K, device="cuda", requires_grad=True)
    compiled = torch.compile(lambda a, s: lin(a, s), backend="aot_eager", fullgraph=True)
    y = compiled(x, seg)
    y.backward(torch.ones_like(y))
    gc, gw = x.grad.clone(), lin.weight.grad.clone()
    x.grad, lin.weight.grad = None, None
    lin(x, seg).backward(torch.ones_like(y))
    f += check(torch.allclose(y, lin(x, seg)) and torch.allclose(gc, x.grad, rtol=1e-4, atol=1e-4) and torch.allclose(gw, lin.weight.grad, rtol=1e-4, atol=1e-4),
               "QATGroupedLinear under torch.compile (aot_eager, fullgraph): forward and backward equal the eager ones")
    for act in ("exact", "int8", "nvfp4"):
        single = optim.QATLinear(K, N, act=act)
        xs = torch.randn(48, K, device="cuda", requires_grad=True)
        y = torch.compile(lambda a: single(a), backend="aot_eager", fullgraph=True)(xs)
        y.backward(torch.ones_like(y))
        gc, gw = xs.grad.clone(), single.weight.grad.clone()
        xs.grad, single.weight.grad = None, None
        single(xs).backward(torch.ones_like(y))
        f += check(torch.allclose(y, single(xs)) and torch.allclose(gc, xs.grad, rtol=1e-4, atol=1e-4) and torch.allclose(gw, single.weight.grad, rtol=1e-4, atol=1e-4),
                   f"QATLinear act={act} under torch.compile (aot_eager, fullgraph): forward and backward equal the eager ones")
    return f


def test_dequantized():
    f = 0
    torch.manual_seed(9)
    for gd in (None, torch.float8_e5m2):
        lin = optim.QATLinear(128, 64, grad_dtype=gd)
        w = lin.dequantized(torch.float32)
        f += check(w.shape == (64, 128) and torch.equal(w, lin.nvfp4_weight().dequantize(torch.float32)) and w.requires_grad,
                   f"QATLinear.dequantized: the NVFP4 values, tracked (grad_dtype={gd})")
        g = torch.randn(64, 128, device="cuda")
        (lin.dequantized(torch.bfloat16).float() * g.bfloat16().float()).sum().backward()
        got = lin.weight.grad if gd is None else lin.weight.grad_fp8
        ok = torch.equal(got, g.bfloat16().float()) if gd is None else got is not None
        f += check(ok, f"QATLinear.dequantized: the gradient of the dense weight is the NVFP4 weight's gradient (grad_dtype={gd})")
    return f


def main():
    f = test_adamw() + test_refusals() + test_qat() + test_to_e4m3() + test_fp8() + test_act() + test_grouped() + test_grouped_kernels() + test_graph_capture() + test_operators() + test_dequantized()
    print("\nall passed" if not f else f"\n{f} FAILED")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
