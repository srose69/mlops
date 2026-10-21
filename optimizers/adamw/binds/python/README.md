# mlops_adamw (PyTorch bindings)

`mlops_adamw` and `mlops_adamw_qat` (`../../include`) for PyTorch: two `torch.optim` optimizers, the NVFP4
layer the second one trains, the low-level calls, and a PyTorch reference of the QAT step. NVFP4 weights
are [`nvfp4effma.NVFP4Tensor`](../../../../../NVFP4EffMA/binds/python) instances.

## Install

Neither package is on PyPI; install NVFP4EffMA first (pip resolves `mlops_adamw`'s dependency on it from
the environment), then this one, from the checkouts:

```
pip install ./NVFP4EffMA/binds/python --no-build-isolation
pip install ./mlops/optimizers/adamw/binds/python --no-build-isolation
```

Needs NVFP4EffMA's static library built (its own `make`, for the same architectures) and the NVFP4EffMA and
E5M2EffMA sources next to `mlops/` (`NVFP4EFFMA_ROOT`, `E5M2EFFMA_ROOT` to override; `MLOPS_ADAMW_CCBIN`
picks the host compiler).

Architectures are chosen in this order: `MLOPS_ADAMW_ARCHS="61 89"` or `TORCH_CUDA_ARCH_LIST="6.1;8.9"`,
else the GPUs visible at build time, else all of sm_50..sm_90. sm_89 needs nvcc 12.1+.
After that `import mlops_adamw` works from any script, no `sys.path` edits.

Runnable examples of everything below: [`../../samples/`](../../samples) (quickstart, converting a model,
groups / schedules / clipping, checkpoints, accumulation and autocast, inference export, low level).

## torch.optim

```python
import mlops_adamw

lin = mlops_adamw.QATLinear(512, 1024, grad_dtype=torch.float8_e5m2)  # NVFP4 weight, trained in place
...
dense, qat = mlops_adamw.split_params(model)
opt = mlops_adamw.AdamW(dense, lr=1e-3, weight_decay=0.0)
opt_qat = mlops_adamw.AdamWQAT(qat, lr=1e-3, weight_decay=0.1)

loss.backward()
opt.step(); opt_qat.step()
opt.zero_grad(); opt_qat.zero_grad()
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
* **`QATLinear(in, out, weight=None, init_std=0.02, four_over_six=None, grad_dtype=None, act="exact")`**:
  bias-free linear layer, `in` a multiple of 64. The forward is NVFP4EffMA's GEMM on the NVFP4 tensor; the
  backward is exact for the GEMM performed (`grad_q = gy.T @ A(x)`, `grad_x = gy @ q`). `act` is the
  format A of the activations, as in `nvfp4effma.linear`; the backward keeps A(x), not x:

  | `act` | A(x) | kept for backward |
  |---|---|---|
  | `"exact"` | x as it is, bf16 / fp16 / fp32 (W4A16) | 2-4 B / value |
  | `"int8"` | int8 with an FP32 scale per 16 values (W4A8): about 2x faster on DP4A GPUs (sm_61+) | 1.25 B |
  | `"nvfp4"` | NVFP4 (W4A4) | 0.5625 B |
  | `"auto"` | exact for up to 2 rows, int8 beyond | |

  The output has x's dtype in every mode. The NVFP4 codes, scales, amax and the optimizer state are module buffers, so
  `state_dict` / `load_state_dict`, `.to(device)` and `model.half()` work as usual (dtype casts leave the
  NVFP4 bytes alone). `layer.weight` is a `QATParameter`: a float32 `[out, in]` handle with 4 bytes of
  storage, for autograd and the optimizer only, never read as a weight.
* **Gradients**: `grad_dtype=None` gives a float32 `weight.grad`. `torch.float8_e5m2` /
  `torch.float8_e4m3fn` gives `weight.grad_fp8 = (FP8 bytes, grad_scale)` with amax scaling instead
  (PyTorch does not allow an FP8 `.grad` on a float32 tensor). Several backwards before a step accumulate.
  The backward computes the product in float32 and encodes it, so this saves optimizer traffic but no
  GEMM work. `AdamWQAT.zero_grad` clears both.

## Low level

```python
mlops_adamw.adamw(param, grad, exp_avg, exp_avg_sq, lr=1e-3, step=t, weight_decay=0.1)

st = mlops_adamw.QATState(w)                 # w: dense [out, in] init, not kept
y = nvfp4effma.linear(x, st.weight)          # forward on the NVFP4 tensor itself
g8, gs = mlops_adamw.fp8_grad(grad_q)        # grad_q = dL/d(st.weight.dequantize()); E5M2 + decode factor
st.step(g8, lr=1e-3, step=t, weight_decay=0.1, grad_scale=gs)   # rewrites st.weight in place
```

`adamw`: `param` / `grad` bfloat16, float16 or float32, `exp_avg` / `exp_avg_sq` float32, in place.
`QATState.step` takes `float8_e5m2`, `float8_e4m3fn` or `float32` gradients. `st.state` is the whole
optimizer state (one uint8 buffer, 5 B per element + 8 per 16-element block). `st.master()` decodes the
master weight for inspection. On the same gradients and seed, `QATLinear` + `AdamWQAT` produce exactly
the same bits as `QATState.step`.

## Reference

`mlops_adamw.reference.step(q, state, grad, ...)` is the same step written in PyTorch on the same
objects, bit-identical to the kernel (same stochastic-rounding stream, same arithmetic order):

```python
mlops_adamw.reference.step(st.weight, st.state, g8, lr=1e-3, step=t, weight_decay=0.1, grad_scale=gs)
```

Use it to read, modify or debug the algorithm.

## Tests

```
python tests/test_optim.py       # AdamW vs torch.optim.AdamW, refusals, QATLinear + AdamWQAT vs QATState
                                 # (float32 and E5M2 gradients), state_dict, model.half()
python tests/test_reference.py   # kernel vs reference, byte for byte on every state array
```

To run the reference test on the sm_89+ FP8 path on any GPU, build into a scratch directory with
`NVCC_APPEND_FLAGS=-DQAT_FP8_CVT_MODEL pip install . --no-build-isolation --no-deps --target /tmp/x`, then
run it with `PYTHONPATH=/tmp/x`.
