# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""CUDA mlops_adamw_qat vs the PyTorch reference (mlops/optim/reference.py), one step at a time from a
shared state: both copies of the state step once, every array is compared, then the CUDA state goes on.
FMA is float64-emulated in the reference, so a last-bit difference may flip a stochastic rounding: the
test wants >= 99.9% of every byte array identical and all u within one Q3.12 quantum.

    python tests/test_reference.py          (with mlops-optimizers and nvfp4effma installed)
"""
import sys

import nvfp4effma
import torch

from mlops.optim import functional, reference


def run(rows, cols, grad_dtype, steps, swizzled=False, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(rows, cols, device="cuda") * 0.02
    target = torch.randn(rows, cols, device="cuda") * 0.02
    st = functional.QATState(w)
    if swizzled:
        st.weight = st.weight.with_scale_layout(True)
    worst = {}
    for t in range(1, steps + 1):
        q = st.weight.dequantize(torch.float32)
        g = q - target + 1e-3 * torch.randn_like(q)
        if grad_dtype is torch.float32:
            gq, gs = g, 1.0
        else:
            gq, gs = functional.fp8_grad(g, grad_dtype)
        state_ref, data_ref, scales_ref = st.state.clone(), st.weight.data.clone(), st.weight.scales.clone()
        kw = dict(lr=1e-3, step=t, weight_decay=0.1, grad_scale=gs, seed=3)
        st.step(gq, **kw)
        qref = nvfp4effma.NVFP4Tensor(data_ref, scales_ref, amax=st.weight.global_scale.clone(),
                                      scale_max=st.weight.scale_max, swizzled=st.weight.swizzled)
        reference.step(qref, state_ref, gq, **kw)
        a, b = reference.carve(st.state, rows, cols), reference.carve(state_ref, rows, cols)
        for k in ("r", "s", "qbar", "s_exp", "cusum", "hnum", "hden"):
            eq = (a[k] == b[k]).float().mean().item()
            worst[k] = min(worst.get(k, 1.0), eq)
        du = (a["u"].long() - b["u"].long()).abs().max().item()
        worst["u within 1"] = min(worst.get("u within 1", 1.0), float(du <= 1))
        worst["u"] = min(worst.get("u", 1.0), (a["u"] == b["u"]).float().mean().item())
        worst["codes"] = min(worst.get("codes", 1.0), (st.weight.data == data_ref).float().mean().item())
        worst["scales"] = min(worst.get("scales", 1.0), (st.weight.scales.view(torch.uint8) == scales_ref.view(torch.uint8)).float().mean().item())
    return worst


def main():
    fails = 0
    for rows, cols, gdt, steps, swz in ((64, 1024, torch.float32, 60, False), (128, 512, torch.float8_e5m2, 60, False),
                                        (256, 256, torch.float8_e4m3fn, 40, True)):
        worst = run(rows, cols, gdt, steps, swz)
        ok = all(v >= 0.999 for k, v in worst.items() if k != "u within 1") and worst["u within 1"] == 1.0
        fails += not ok
        name = f"{rows}x{cols} grad={str(gdt).split('.')[-1]}{' swizzled' if swz else ''}, {steps} steps"
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: worst identical fraction " +
              ", ".join(f"{k} {v:.5f}" for k, v in worst.items()))
    print("\nall passed" if not fails else f"\n{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
