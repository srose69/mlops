# mlops-optimizers

AdamW and an AdamW that trains NVFP4 weights in place, for NVIDIA GPUs without native FP4 (sm_50 .. sm_90):
a C API ([`include/`](include)) and a PyTorch package, `mlops.optim`. The math, the formats and the
measurements: [adamw/README.md](adamw/README.md).

| | weights | optimizer state per weight | |
|---|---|---|---|
| `AdamW` (`mlops_adamw`) | BF16 / FP16 / FP32 | 8 B (FP32 `m`, `v`) | matches `torch.optim.AdamW`, 88% of DRAM peak on a GTX 1080 |
| `AdamWQAT` (`mlops_adamw_qat`) | NVFP4, trained in place | 5 B + 8 B per 16-weight block | no master copy, FP8 gradients, PyTorch reference bit-identical to the kernel |

On a 4096 × 4096 weight on a GTX 1080 (the only GPU measured), the QAT step takes 1.33 ms against 2.22 ms
for AdamW on an FP32 master plus NVFP4 re-quantization, with half the memory (6.12 vs 12.56 B per weight),
and trains a small NVFP4 transformer to the same validation loss
([details](adamw/README.md#performance)).

## Block scales: UE5M3 by default

An NVFP4 weight's block scales are 8-bit. Transformer Engine uses E4M3 (`2^-9 .. 448`). `QATLinear` and
`QATState` default to **UE5M3**: unsigned, 5 exponent bits, the same 3 mantissa bits, `2^-17 .. 114688`
([Graphcore, arXiv:2609.02846](https://arxiv.org/abs/2609.02846)). There, on Nemotron-H 8B over 190B tokens,
Transformer Engine's native NVFP4 had repeated loss spikes once its BF16 layers and Hadamard transform were
removed. UE5M3 scales trained smoothly with every linear in FP4 and a tensor scale refreshed only every 50
steps; E4M3 leaves a tensor at its amax no room to grow (their section 5.2). Here the global scale never
changes after initialization, so the scale format sets how far a block's weights can grow during training:

| | growth of a block scale | when weights outgrow it |
|---|---|---|
| E4M3 | 8x | the scale stops at 448, the block's largest weights saturate |
| UE5M3 | 256x (23 binades left below the largest block) | — |

* **Nothing else changes.** While no E4M3 scale would leave its range, both formats train bit for bit the
  same: same values, moments and codes, only the scale bytes differ by a constant. The step is 0.5-1% faster.
* **Where it matters** (`make qat`): when weights grow 32x, every UE5M3 block still represents them exactly;
  about a third of the E4M3 blocks stop at the top with 18% error.
* **Export.** Blackwell's native NVFP4 (cuBLASLt, Transformer Engine, ModelOpt, vLLM) reads E4M3 scales.
  `mlops.optim.to_e4m3(layer.nvfp4_weight())` converts with the same values bit for bit: one power of two
  per row between the scales and the global. If training spread a row's scales over more than E4M3 spans,
  it returns the weight as is, still UE5M3 (`.ue5m3`), with a warning instead of changing it.
  `nvfp4effma` runs UE5M3 tensors directly.
* `scale_dtype="e4m3"` selects E4M3; `four_over_six` (an E4M3 recipe) selects it by itself. The format is
  saved in `state_dict` (`nvfp4_format`); checkpoints without it load as E4M3.

## Install

Nothing is downloaded during installation; install these first:

* PyTorch (a CUDA build);
* **nvfp4effma 0.4.x** ([releases](https://github.com/srose69/nvfp4effma/releases)): NVFP4 tensors and GEMM at
  run time, its headers at build time;
* **e5m2effma 0.2+** ([releases](https://github.com/srose69/e5m2effma/releases)): only to build for GPUs below
  sm_89, headers only; not needed at run time or for a prebuilt wheel.

From a release, in the directory with the wheel:

```sh
pip install --no-index -f . mlops-optimizers
```

From the sources (the mlops root):

```sh
./setup.sh optimizers                      # = pip install --no-build-isolation --no-index ./optimizers/python
```

The build compiles for the GPUs visible at build time, for all of sm_50..sm_90 if none is visible, or for
`MLOPS_ARCHS="61 89"` (or `TORCH_CUDA_ARCH_LIST="6.1;8.9"`); sm_89 needs nvcc 12.1+. More variables:
[`../build/torch_ext.py`](../build/torch_ext.py). The extension compiles NVFP4EffMA's device codecs into its
kernels, so `import mlops.optim` refuses an installed nvfp4effma of another minor version than the one it was
built against (`mlops.optim._C.built_nvfp4effma`).

## torch.optim

```python
import torch, torch.nn as nn
from mlops.optim import AdamW, AdamWQAT, QATLinear, split_params

model = nn.Sequential(
    QATLinear(512, 2048, grad_dtype=torch.float8_e5m2),  # NVFP4 weight, trained in place
    nn.GELU(),
    QATLinear(2048, 512, grad_dtype=torch.float8_e5m2),
    nn.LayerNorm(512),                                   # dense
).cuda()

dense, qat = split_params(model)
opt = AdamW(dense, lr=1e-3, weight_decay=0.0)
opt_qat = AdamWQAT(qat, lr=1e-3, weight_decay=0.1)

for x, y in loader:
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step(); opt_qat.step()
    opt.zero_grad(); opt_qat.zero_grad()

torch.save(model.state_dict(), "ckpt.pt")   # holds the NVFP4 weights and AdamWQAT's moments
```

Both are regular `torch.optim.Optimizer`s: parameter groups, `lr` changes through `param_groups`,
`state_dict`, LR schedulers.

* **`AdamW`**: `mlops_adamw` on dense CUDA parameters (bfloat16 / float16 / float32, contiguous) with FP32
  moments; equal to `torch.optim.AdamW` up to rounding. It refuses `QATParameter`s.
* **`AdamWQAT`**: `mlops_adamw_qat`, and only on the weights of `QATLinear` layers. It refuses everything
  `AdamW` handles (any dense parameter, also via `add_param_group`) and every `torch.optim.AdamW` option
  it does not implement (`amsgrad`, `maximize`, `foreach`, ...). Extra options: `scale_z` (threshold of
  the block-scale moves, in null standard deviations, default 3) and `seed` (stochastic rounding; each
  parameter gets its own stream from its position in the optimizer). Its moments live in the layers'
  buffers, so a checkpoint needs `model.state_dict()`; `opt_qat.state_dict()` carries only step counts.
* **`QATLinear(in, out, weight=None, init_std=0.02, four_over_six=None, scale_dtype=None, grad_dtype=None, act="exact")`**:
  bias-free linear layer, `in` a multiple of 64. The initial weight is quantized by `nvfp4effma.quantize`
  (1 × 16 blocks, amax global, optionally 4over6). `scale_dtype`: `"ue5m3"` (the default) or `"e4m3"`
  (the default with `four_over_six`), see [block scales](#block-scales-ue5m3-by-default). The forward is
  NVFP4EffMA's GEMM on the NVFP4 tensor; the backward is exact for the GEMM performed
  (`grad_q = gy.T @ A(x)`, `grad_x = gy @ q`). `act` is the format A of the activations, as in
  `nvfp4effma.linear`; the backward keeps A(x), not x:

  | `act` | A(x) | kept for backward |
  |---|---|---|
  | `"exact"` | x as it is, bf16 / fp16 / fp32 (W4A16) | 2-4 B / value |
  | `"int8"` | `nvfp4effma.Int8Tensor`: int8 with an FP32 scale per 16 values (W4A8), about 2x faster on DP4A GPUs (sm_61+) | 1.25 B |
  | `"nvfp4"` | NVFP4 (W4A4) | 0.5625 B |
  | `"auto"` | exact for up to 2 rows, int8 beyond | |

  The output has x's dtype in every mode. The NVFP4 codes, scales, amax and the optimizer state are module
  buffers, so `state_dict` / `load_state_dict`, `.to(device)` and `model.half()` work as usual (dtype casts
  leave the NVFP4 bytes alone). `layer.weight` is a `QATParameter`: a float32 `[out, in]` handle with 4 bytes
  of storage, for autograd and the optimizer only, never read as a weight.
* **Gradients**: `grad_dtype=None` gives a float32 `weight.grad`. `torch.float8_e5m2` /
  `torch.float8_e4m3fn` gives `weight.grad_fp8 = (FP8 bytes, grad_scale)` with amax scaling instead
  (PyTorch does not allow an FP8 `.grad` on a float32 tensor). Several backwards before a step accumulate.
  The backward computes the product in float32 and encodes it, so this saves optimizer traffic but no
  GEMM work. `AdamWQAT.zero_grad` clears both.
* **`split_params(model)`**: (dense parameters, QAT parameters), each parameter once.

## Low level: mlops.optim.functional

```python
from mlops.optim import functional, reference

functional.adamw(param, grad, exp_avg, exp_avg_sq, lr=1e-3, step=t, weight_decay=0.1)

st = functional.QATState(w)                  # w: dense [out, in] init, not kept; UE5M3 scales (scale_dtype)
y = nvfp4effma.linear(x, st.weight)          # forward on the NVFP4 tensor itself
g8, gs = functional.fp8_grad(grad_q)         # grad_q = dL/d(st.weight.dequantize()); E5M2 + decode factor
st.step(g8, lr=1e-3, step=t, weight_decay=0.1, grad_scale=gs)   # rewrites st.weight in place
```

`adamw`: `param` / `grad` bfloat16, float16 or float32, `exp_avg` / `exp_avg_sq` float32, in place.
`QATState.step` takes `float8_e5m2`, `float8_e4m3fn` or `float32` gradients. `st.state` is the whole
optimizer state (one uint8 buffer, 5 B per element + 8 per 16-element block). `st.master()` decodes the
master weight for inspection. On the same gradients and seed, `QATLinear` + `AdamWQAT` produce exactly
the same bits as `QATState.step`.

`reference.step(q, state, grad, ...)` is the same QAT step written in PyTorch on the same objects,
bit-identical to the kernel (same stochastic-rounding stream, same arithmetic order) — to read, modify or
debug the algorithm:

```python
reference.step(st.weight, st.state, g8, lr=1e-3, step=t, weight_decay=0.1, grad_scale=gs)
```

## Samples

Runnable examples to copy from, in [`samples/`](samples):

| sample | shows |
|---|---|
| `01_quickstart.py` | a minimal model, `split_params`, both optimizers, the training loop |
| `02_convert_model.py` | replacing the `nn.Linear` layers of an existing model with `QATLinear`, keeping biases |
| `03_groups_schedule_clip.py` | parameter groups, warm-up + cosine LR, global gradient clipping including FP8 gradients |
| `04_checkpoint_resume.py` | what is saved where; resuming that is bit-identical to an uninterrupted run |
| `05_accumulation_amp.py` | gradient accumulation, bfloat16 autocast, `act` = exact / int8 / nvfp4 |
| `06_inference_export.py` | inference, export without optimizer state, `nvfp4effma` access, dense copy, `to_e4m3` for Blackwell / TE |
| `07_low_level.py` | `functional.adamw`, `QATState`, `fp8_grad`, the bit-identical PyTorch reference |
| `08_transformer_training.py` | a 3-layer transformer with RoPE learns to sort: NVFP4 weights, FP8 gradients, 6 B instead of 16 B per weight, within a few points of the dense model; `ue5m3` for UE5M3 block scales (needs mlops-positional) |

## C / C++

[`include/mlops_optimizers.h`](include/mlops_optimizers.h) includes the whole package:
[`mlops_adamw.h`](include/mlops_adamw.h) and [`mlops_adamw_qat.h`](include/mlops_adamw_qat.h), which document
the exact order of operations. Raw device pointers, an explicit `cudaStream_t`, a status instead of
exceptions (`mlops_adamw_last_error()` describes the failure). `mlops_adamw_qat_init_quantized` initializes
the QAT state from a weight already quantized with NVFP4EffMA; `mlops_adamw_qat_init` quantizes it too and is
the only function that needs NVFP4EffMA's library at link time (`pkg-config --libs mlops-optimizers` adds
it).

## Build and test

```sh
make lib                # out/libmlops_optimizers.a: every architecture sm_50..sm_90 (ARCHS="61 89", ARCHS=native)
make test               # mlops_adamw vs its host emulation, every dtype pair, with and without bias correction
make qat                # mlops_adamw_qat: codecs, one step vs a double-precision reference (linear and swizzled
                        # scales), noise / scale-finding / cell-walking behaviour; then the same through the
                        # sm_89+ FP8 path (QAT_FP8_CVT_MODEL)
make bench              # step time and bandwidth of both, NVML clocks
make install PREFIX=..  # headers, library, pkg-config file
```

Tests and bench run on GPU `GPU` (default 0) and are built for its architecture. Each kernel instance is
its own object, compiled in parallel (`JOBS`); toolchain, architectures and where NVFP4EffMA / E5M2EffMA are
found: [`../build/arch.mk`](../build/arch.mk) (by default the installed Python packages, else checkouts next
to `mlops/`). The tests and the bench also link NVFP4EffMA's library (`NVFP4EFFMA_LIB`).

## Tests

```sh
python python/tests/test_optim.py      # AdamW vs torch.optim.AdamW, refusals, QATLinear + AdamWQAT vs QATState
                                       # (float32 and E5M2 gradients), every act, state_dict, model.half()
python python/tests/test_reference.py  # kernel vs reference, byte for byte on every state array
```

To run the reference test on the sm_89+ FP8 path on any GPU, build into a scratch directory with
`NVCC_APPEND_FLAGS=-DQAT_FP8_CVT_MODEL pip install ./python --no-build-isolation --no-deps --target /tmp/x`,
then run it with `PYTHONPATH=/tmp/x`.

## License

GNU AGPL v3 with the Linking and Attribution Exception ([NOTICE](../NOTICE)).
