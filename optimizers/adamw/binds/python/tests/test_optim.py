"""torch.optim front ends: mlops_adamw.AdamW vs torch.optim.AdamW, the parameter checks of both optimizers,
QATLinear + AdamWQAT (trains, equals QATState's step, state_dict round trip, dtype casts leave it intact).

    python tests/test_optim.py          (with mlops_adamw and nvfp4effma installed)
"""
import io
import sys

import torch
import torch.nn as nn

import mlops_adamw


def check(ok, what):
    print(f"[{'PASS' if ok else 'FAIL'}] {what}")
    return 0 if ok else 1


def test_adamw():
    f = 0
    torch.manual_seed(0)
    for dt in (torch.float32, torch.bfloat16):
        p = nn.Parameter(torch.randn(1000, 33, device="cuda", dtype=dt))
        ref = nn.Parameter(p.detach().float().clone())
        opt, opt_ref = mlops_adamw.AdamW([p], lr=1e-2, weight_decay=0.1), torch.optim.AdamW([ref], lr=1e-2, weight_decay=0.1)
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
    lin = mlops_adamw.QATLinear(64, 32)
    dense = nn.Parameter(torch.randn(4, 4, device="cuda"))
    for opt, params, what in ((mlops_adamw.AdamW, [lin.weight], "AdamW refuses an NVFP4 QATParameter"),
                              (mlops_adamw.AdamWQAT, [dense], "AdamWQAT refuses a dense parameter"),
                              (mlops_adamw.AdamWQAT, [lin.weight, dense], "AdamWQAT refuses a mixed parameter list"),
                              (mlops_adamw.AdamW, [nn.Parameter(torch.randn(4, device="cpu"))], "AdamW refuses a CPU parameter")):
        try:
            opt(params)
            f += check(False, what)
        except TypeError as e:
            f += check(True, f"{what}: {e}")
    opt = mlops_adamw.AdamWQAT([lin.weight])
    try:
        opt.add_param_group({"params": [dense]})
        f += check(False, "AdamWQAT.add_param_group refuses a dense parameter")
    except TypeError:
        f += check(len(opt.param_groups) == 1, "AdamWQAT.add_param_group refuses a dense parameter (groups unchanged)")
    try:
        mlops_adamw.AdamWQAT([lin.weight], amsgrad=True)
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
    lin = mlops_adamw.QATLinear(128, 64, weight=w0, grad_dtype=torch.float8_e5m2)
    lin32 = mlops_adamw.QATLinear(128, 64, weight=w0)
    opt = mlops_adamw.AdamWQAT([lin.weight], lr=1e-3, seed=3)
    st = mlops_adamw.QATState(w0)
    ok = True
    for t in range(1, 51):
        ((lin(x) - y) ** 2).mean().backward()
        lin32.load_state_dict(lin.state_dict())
        ((lin32(x) - y) ** 2).mean().backward()  # the same float32 gradient, encoded as the layer does
        g8, gs = mlops_adamw.fp8_grad(lin32.weight.grad, torch.float8_e5m2)
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

    lin = mlops_adamw.QATLinear(128, 64, weight=w0)
    opt = mlops_adamw.AdamWQAT([lin.weight], lr=1e-3, weight_decay=0.0, seed=5)
    st = mlops_adamw.QATState(w0)  # the same step through the low-level API, same stream (parameter 1)
    losses = []
    for t in range(1, 301):
        loss = ((lin(x) - y) ** 2).mean()
        loss.backward()
        g = lin.weight.grad.clone()
        opt.step(); opt.zero_grad()
        st.step(g, lr=1e-3, step=t, weight_decay=0.0, seed=(5 + 0x9E3779B9) & 0xFFFFFFFF)
        losses.append(loss.item())
    a, b = mlops_adamw.reference.carve(lin.qat_state, 64, 128), mlops_adamw.reference.carve(st.state, 64, 128)
    same = all(torch.equal(a[k], b[k]) for k in ("u", "r", "s", "qbar", "s_exp", "cusum", "hnum", "hden")) and \
        torch.equal(lin.nvfp4_data, st.weight.data) and torch.equal(lin.nvfp4_scales, st.weight.scales.view(torch.uint8))
    f += check(same, "QATLinear + AdamWQAT == QATState.step on the same gradients, 300 steps (state and codes bit-identical)")
    f += check(losses[-1] < 0.2 * losses[0], f"QATLinear + AdamWQAT fits a linear map: loss {losses[0]:.3e} -> {losses[-1]:.3e}")
    f += check(lin.weight.untyped_storage().nbytes() == 4, "the weight handle holds 4 bytes (the weight is the NVFP4 buffers)")

    buf = io.BytesIO()
    torch.save(lin.state_dict(), buf)
    lin2 = mlops_adamw.QATLinear(128, 64)
    lin2.load_state_dict(torch.load(io.BytesIO(buf.getvalue())))
    f += check(torch.equal(lin2(x), lin(x)), f"state_dict round trip: same output (keys {sorted(lin.state_dict())})")

    xh = x.half()
    ref = lin(xh)
    model = nn.Sequential(lin2).half()
    out = model(xh)
    f += check(out.dtype == torch.float16 and lin2.weight.dtype == torch.float32 and torch.equal(out, ref),
               "model.half(): NVFP4 buffers and the float32 handle untouched, same output")
    return f


def test_act():
    """Every act: forward == nvfp4effma.linear, output in x's dtype, backward exact for the GEMM performed
    (grad_q = gy.T @ A(x), grad_x = gy @ q) against dense autograd on the dequantized operands."""
    import nvfp4effma
    from nvfp4effma import _C as nc
    f = 0
    torch.manual_seed(3)
    for act in ("exact", "int8", "nvfp4", "auto"):
        for dt, rows in ((torch.float32, 96), (torch.bfloat16, 96), (torch.float16, 96), (torch.float32, 2)):
            lin = mlops_adamw.QATLinear(128, 64, act=act)
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
                xq, xs = nc.quantize_int8(x2.contiguous())
                xa = (xq.float().view(-1, 8, 16) * xs.view(-1, 8, 1)).view(-1, 128)
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


def main():
    f = test_adamw() + test_refusals() + test_qat() + test_fp8() + test_act()
    print("\nall passed" if not f else f"\n{f} FAILED")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
