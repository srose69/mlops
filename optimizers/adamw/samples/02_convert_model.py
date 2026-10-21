"""Convert an existing model: replace its nn.Linear layers with QATLinear, initialized from their weights.

    python 02_convert_model.py

Typical for QAT fine-tuning of a pretrained model. QATLinear has no bias and needs in_features % 64 == 0,
so layers with a bias are handled by keeping the bias as a separate dense parameter, and layers of
another shape stay dense.
"""
import torch
import torch.nn as nn

import mlops_adamw


class QATLinearWithBias(nn.Module):
    """QATLinear (NVFP4 weight, AdamWQAT) + a dense bias (AdamW)."""

    def __init__(self, linear: nn.Linear, **qat_kwargs):
        super().__init__()
        self.qat = mlops_adamw.QATLinear(linear.in_features, linear.out_features,
                                         weight=linear.weight.detach(), device=linear.weight.device, **qat_kwargs)
        self.bias = nn.Parameter(linear.bias.detach().clone())

    def forward(self, x):
        return self.qat(x) + self.bias


def to_qat(module: nn.Module, **qat_kwargs) -> nn.Module:
    """Replace, recursively and in place, every eligible nn.Linear of `module`. Returns `module`."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and child.in_features % 64 == 0:
            if child.bias is None:
                new = mlops_adamw.QATLinear(child.in_features, child.out_features, weight=child.weight.detach(),
                                            device=child.weight.device, **qat_kwargs)
            else:
                new = QATLinearWithBias(child, **qat_kwargs)
            setattr(module, name, new)
        else:
            to_qat(child, **qat_kwargs)
    return module


torch.manual_seed(0)
# A "pretrained" model: a transformer-style MLP block plus a small head whose input (10) is not % 64.
model = nn.Sequential(
    nn.LayerNorm(128),
    nn.Linear(128, 512, bias=False),
    nn.GELU(),
    nn.Linear(512, 128),       # has a bias
    nn.Linear(128, 10),
    nn.Linear(10, 10),         # in_features 10: stays dense
).cuda()
x = torch.randn(64, 128, device="cuda")
with torch.no_grad():
    y_dense = model(x)

# four_over_six="mse": Transformer Engine's 4over6 choice of block scales for the initial quantization,
# lower quantization error than plain amax scaling (optional).
to_qat(model, four_over_six="mse", grad_dtype=torch.float8_e5m2)
print(model)

# The converted model differs from the original only by NVFP4 quantization error.
with torch.no_grad():
    y_qat = model(x)
print(f"relative output change after NVFP4 conversion: {((y_qat - y_dense).norm() / y_dense.norm()).item():.3f}")

dense, qat = mlops_adamw.split_params(model)
print(f"dense parameters: {sum(p.numel() for p in dense):,}   NVFP4 weights: {sum(p.numel() for p in qat):,}")
opt = mlops_adamw.AdamW(dense, lr=1e-4)
opt_qat = mlops_adamw.AdamWQAT(qat, lr=1e-4, weight_decay=0.01)
