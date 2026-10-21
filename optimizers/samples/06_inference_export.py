# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""After training: inference, and exporting the NVFP4 weights.

    python 06_inference_export.py

* Inference with the trained model as is: QATLinear's forward is the NVFP4 GEMM, nothing to convert.
* layer.nvfp4_weight(): the nvfp4effma.NVFP4Tensor the forward reads (views of the layer's buffers);
  usable with nvfp4effma.linear directly, without mlops.optim.
* For deployment you only need codes, scales and amax; the optimizer state (`qat_state`) can be dropped.
* .dequantize() gives the exact dense weight the model computes with, e.g. to build a plain nn.Linear.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlops.optim import AdamWQAT, QATLinear, split_params
import nvfp4effma

torch.manual_seed(0)
dim = 128
model = nn.Sequential(QATLinear(dim, 256), nn.GELU(), QATLinear(256, dim)).cuda()
dense, qat = split_params(model)
opt_qat = AdamWQAT(qat, lr=1e-3)
for _ in range(100):
    x = torch.randn(128, dim, device="cuda")
    F.mse_loss(model(x), x.roll(1, -1)).backward()
    opt_qat.step(); opt_qat.zero_grad()

x = torch.randn(16, dim, device="cuda")
model.eval()
with torch.no_grad():
    y = model(x)

# 1. The NVFP4 tensor of a layer, used through nvfp4effma alone.
lin = model[0]
q = lin.nvfp4_weight()
with torch.no_grad():
    y0 = nvfp4effma.linear(x, q, act="exact")
    print("nvfp4effma.linear == QATLinear forward:", torch.equal(y0, lin(x)))

# 2. Inference-only export: weights without optimizer state.
export = {k: v for k, v in model.state_dict().items() if not k.endswith("qat_state")}
full = sum(v.numel() * v.element_size() for v in model.state_dict().values())
small = sum(v.numel() * v.element_size() for v in export.values())
print(f"state_dict: {full:,} B with optimizer state, {small:,} B for inference")
# Loading it back: load_state_dict(export, strict=False) skips the missing qat_state (fine for inference).
fresh = nn.Sequential(QATLinear(dim, 256), nn.GELU(), QATLinear(256, dim)).cuda()
fresh.load_state_dict(export, strict=False)
with torch.no_grad():
    print("reloaded inference model gives the same output:", torch.equal(fresh(x), y))

# 3. Dense copy (the exact weights the NVFP4 model uses), e.g. for tools that expect nn.Linear.
dense_model = nn.Sequential(nn.Linear(dim, 256, bias=False), nn.GELU(), nn.Linear(256, dim, bias=False)).cuda()
with torch.no_grad():
    for d, l in ((dense_model[0], model[0]), (dense_model[2], model[2])):
        d.weight.copy_(l.nvfp4_weight().dequantize(torch.float32))
    y_dense = dense_model(x)
rel = ((y_dense - y).norm() / y.norm()).item()
print(f"dense copy vs NVFP4 model: relative difference {rel:.1e} (float32 summation order only)")
