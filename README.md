# mlops

ML training ops in CUDA C++ for the GPUs NVFP4EffMA supports: sm_50 .. sm_90, those without native FP4. Each
op has a C API and PyTorch bindings. They build on [NVFP4EffMA](../NVFP4EffMA), and below sm_89 on
[E5M2EffMA](../E5M2EffMA); from sm_89 on, FP8 is native and they use the hardware `cvt`.

## Install

Neither package is on PyPI. Install NVFP4EffMA first: `mlops_adamw` depends on it. NVFP4EffMA's static
library must be built first with its own `make`.

```
pip install ./NVFP4EffMA/binds/python --no-build-isolation
pip install ./mlops/optimizers/adamw/binds/python --no-build-isolation
```

Both build for the GPUs visible at build time, for all of sm_50..sm_90 if none is visible, or for an
explicit list: `TORCH_CUDA_ARCH_LIST="6.1;8.9"`. sm_89 needs nvcc 12.1+. After that, `import mlops_adamw`
works from any script.

## Usage

Two `torch.optim` optimizers that split a model between them: `AdamW` for dense parameters and `AdamWQAT`
for the NVFP4 weights of `QATLinear` layers.

```python
import torch, torch.nn as nn
import mlops_adamw

model = nn.Sequential(
    mlops_adamw.QATLinear(512, 2048, grad_dtype=torch.float8_e5m2),  # NVFP4 weight, trained in place
    nn.GELU(),
    mlops_adamw.QATLinear(2048, 512, grad_dtype=torch.float8_e5m2),
    nn.LayerNorm(512),                                               # dense
).cuda()

dense, qat = mlops_adamw.split_params(model)
opt     = mlops_adamw.AdamW(dense, lr=1e-3, weight_decay=0.0)
opt_qat = mlops_adamw.AdamWQAT(qat, lr=1e-3, weight_decay=0.1)

for x, y in loader:
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step(); opt_qat.step()
    opt.zero_grad(); opt_qat.zero_grad()

torch.save(model.state_dict(), "ckpt.pt")   # holds the NVFP4 weights and AdamWQAT's moments
```

* **`AdamW`** works on dense CUDA parameters (bfloat16 / float16 / float32) with FP32 moments and matches
  `torch.optim.AdamW`. It refuses NVFP4 weights.
* **`AdamWQAT`** accepts only `QATLinear` weights. It refuses dense, CPU and sparse parameters, also when
  they come through `add_param_group`. It also refuses AdamW options it does not implement (`amsgrad`,
  `maximize`, `foreach`).
* **`QATLinear(in, out, act="exact")`** is a bias-free linear layer with `in` a multiple of 64. The forward is
  NVFP4EffMA's GEMM. `act` sets how activations enter it:
  * `"exact"`: as they are, bf16 / fp16 / fp32;
  * `"int8"`: about 2x faster with DP4A, sm_61+;
  * `"nvfp4"`;
  * `"auto"`.

  The output has x's dtype. The weight and the optimizer state are module buffers, so `state_dict`, `.to()` and
  `model.half()` work as usual.
* **Gradients**: `grad_dtype=None` gives a float32 `weight.grad`. `float8_e5m2` / `float8_e4m3fn` gives
  `weight.grad_fp8` instead (FP8 bytes plus their scale), because torch does not allow an FP8 `.grad` on
  a float32 tensor.

Parameter groups, LR schedulers and `state_dict` behave as with any `torch.optim` optimizer. Full API,
low-level calls and the PyTorch reference: [binds/python/README.md](optimizers/adamw/binds/python/README.md).
A complete training script: [`misc/train_nvfp4.py`](../misc/train_nvfp4.py).

Runnable examples to copy from, in [`optimizers/adamw/samples/`](optimizers/adamw/samples):

| sample | shows |
|---|---|
| `01_quickstart.py` | a minimal model, `split_params`, both optimizers, the training loop |
| `02_convert_model.py` | replacing the `nn.Linear` layers of an existing model with `QATLinear`, keeping biases |
| `03_groups_schedule_clip.py` | parameter groups, warm-up + cosine LR, global gradient clipping including FP8 gradients |
| `04_checkpoint_resume.py` | what is saved where; resuming that is bit-identical to an uninterrupted run |
| `05_accumulation_amp.py` | gradient accumulation, bfloat16 autocast, `act` = exact / int8 / nvfp4 |
| `06_inference_export.py` | inference, export without optimizer state, `nvfp4effma` access, dense copy |
| `07_low_level.py` | `mlops_adamw.adamw`, `QATState`, `fp8_grad`, the bit-identical PyTorch reference |

## optimizers/adamw

[`optimizers/adamw`](optimizers/adamw) ([README](optimizers/adamw/README.md): derivations, formats, tests):

* **`mlops_adamw`**: AdamW for BF16 / FP16 / FP32 parameters with FP32 moments, 88% of DRAM peak on a
  GTX 1080 (the only GPU measured). It knows nothing about NVFP4: NVFP4 training with it keeps an FP32 master and re-quantizes it
  after every step.
* **`mlops_adamw_qat`**: AdamW that trains an NVFP4 weight in place. No FP32 master: the master lives on
  the NVFP4 lattice as a 16-bit grid coordinate, the moments in FP8 (E4M3 `m / sqrt(v)`, E5M2 `sqrt(v)`),
  all stochastically rounded; gradients come in as FP8. Block scales move only when a block's Newton
  estimate of its optimum, with a statistically significant curvature, keeps asking for another scale,
  and a move never changes a weight. A PyTorch reference of the step is bit-identical to the kernel.

| on a 4096 x 4096 weight, GTX 1080 | step | NVFP4 weight + optimizer state |
|---|---|---|
| `mlops_adamw` on an FP32 master + NVFP4 re-quantization | 2.22 ms | 12.56 B / weight |
| `mlops_adamw_qat`, E5M2 gradients | 1.33 ms | 6.12 B / weight |

The QAT step is 1.67x faster than the path it replaces and on par with plain BF16 AdamW (1.31 ms). On the
GTX 1080 it is bound by arithmetic, not memory: its access pattern alone runs in 0.76 ms. Other GPUs are not
measured.

Training a small NVFP4 transformer ([`misc/`](../misc): 287k NVFP4 + 279k dense parameters, WikiText-2,
5000 steps, identical hyperparameters), validation loss over two seeds:

| optimizer | validation loss |
|---|---|
| `mlops_adamw`, FP32 master | 4.720 / 4.706 |
| `mlops_adamw_qat` | 4.716 / 4.707 |

The two optimizers differ by less than the two seeds do.

## Build and test (C / CUDA)

```
cd optimizers/adamw && make && make test && make qat && make bench
```
