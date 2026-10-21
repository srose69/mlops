"""Checkpoint and resume, verified bit-exact against an uninterrupted run.

    python 04_checkpoint_resume.py

Where the state lives:
  * model.state_dict()     NVFP4 codes, block scales, amax AND AdamWQAT's moments (QATLinear buffers),
                           plus dense parameters
  * opt.state_dict()       AdamW's FP32 moments and step counts (as for torch.optim.AdamW)
  * opt_qat.state_dict()   only step counts and hyperparameters; the AdamWQAT `seed` is a constructor
                           argument, pass the same one when resuming
"""
import io

import torch
import torch.nn as nn
import torch.nn.functional as F

import mlops_adamw

dim = 128
data = [(torch.randn(64, dim, device="cuda", generator=torch.Generator("cuda").manual_seed(i)),) for i in range(40)]


def build():
    torch.manual_seed(0)
    model = nn.Sequential(mlops_adamw.QATLinear(dim, 256, grad_dtype=torch.float8_e5m2), nn.GELU(),
                          mlops_adamw.QATLinear(256, dim), nn.LayerNorm(dim)).cuda()
    dense, qat = mlops_adamw.split_params(model)
    return model, mlops_adamw.AdamW(dense, lr=1e-3), mlops_adamw.AdamWQAT(qat, lr=1e-3, weight_decay=0.1, seed=7)


def train(model, opt, opt_qat, batches):
    for (x,) in batches:
        F.mse_loss(model(x), x.flip(-1)).backward()
        opt.step(); opt_qat.step()
        opt.zero_grad(); opt_qat.zero_grad()


# Run A: 40 steps without interruption.
model_a, opt_a, qopt_a = build()
train(model_a, opt_a, qopt_a, data)

# Run B: 20 steps, checkpoint, fresh process-like rebuild, load, 20 more steps.
model_b, opt_b, qopt_b = build()
train(model_b, opt_b, qopt_b, data[:20])
buf = io.BytesIO()  # a file path works the same: torch.save(ckpt, "ckpt.pt")
torch.save({"model": model_b.state_dict(), "opt": opt_b.state_dict(), "opt_qat": qopt_b.state_dict()}, buf)
del model_b, opt_b, qopt_b

model_b, opt_b, qopt_b = build()  # same architecture, same AdamWQAT seed
ckpt = torch.load(io.BytesIO(buf.getvalue()))
model_b.load_state_dict(ckpt["model"])
opt_b.load_state_dict(ckpt["opt"])
qopt_b.load_state_dict(ckpt["opt_qat"])
train(model_b, opt_b, qopt_b, data[20:])

same = all(torch.equal(a, b) for a, b in zip(model_a.state_dict().values(), model_b.state_dict().values()))
print(f"checkpoint size: {len(buf.getvalue()) / 1024:.1f} KiB")
print(f"resumed run bit-identical to the uninterrupted one: {same}")
assert same
