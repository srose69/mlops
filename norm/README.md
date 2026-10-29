# mlops-norm

RMSNorm and LayerNorm as CUDA kernels for NVIDIA GPUs sm_50 .. sm_90, with a residual add before and an INT8 / NVFP4
quantization after, for the places of a model where the norm is not folded into a GEMM's epilogue (NVFP4EffMA's deferred
scalar): a C API ([`include/`](include)), the work of each call as items for an executor of one's own
([`include/mlops_norm_items.h`](include/mlops_norm_items.h)), and a PyTorch package, `mlops.norm`. The arithmetic, the kernels and the
measurements: [layer/README.md](layer/README.md).

A 2048 x 4096 BF16 RMSNorm takes 160 µs on a GTX 1080 and 20 µs on an L40S, and the same call with an INT8 or NVFP4 output
takes 127 - 130 µs and 18 - 19 µs: the quantized bytes are what NVFP4EffMA's `nvfp4a_quantize` makes of the normalized rows,
bit for bit, with no pass over them in between.

| flavor | call | models |
|---|---|---|
| RMSNorm, Hugging Face's roundings | `rms_norm(x, w)` | Llama, Qwen, Mistral, DeepSeek |
| RMSNorm, zero-centered weight | `rms_norm(x, w, one_plus=True)` | Gemma, Qwen3-Next |
| LayerNorm, weight and bias; no parameters | `layer_norm(x, w, b)`, `layer_norm(x)` | GPT, Falcon, Phi; OLMo |
| gated, grouped RMSNorm | `norm(x, w, gate=z, gate_mode=..., groups=g)` | Mamba-2, Nemotron-H, Qwen3-Next |

## Install

Needs PyTorch (a CUDA build), the CUDA toolkit and the `nvfp4effma` package (its headers, to build; nothing at run time).
Nothing is downloaded.

```sh
pip install --no-index -f . mlops-norm              # from a release, in the directory with the wheel
./setup.sh norm                                     # from the sources (the mlops root)
```

The build compiles for the GPUs visible at build time, for all of sm_50..sm_90 if none is visible, or for
`MLOPS_ARCHS="61 89"`; other variables: [`../build/torch_ext.py`](../build/torch_ext.py).

## Python

```python
from mlops.norm import rms_norm, layer_norm, norm

out = rms_norm(x, weight, 1e-6, residual=h, quant="int8")              # out.y int8, out.scales float32 [rows, hidden / 16], out.residual
out = rms_norm(x, weight, 1e-6, one_plus=True)                         # Gemma
out = layer_norm(x, weight, bias)                                      # GPT, Falcon, Phi
out = norm(x, weight, eps=1e-6, gate=z, gate_mode="before", groups=4)  # Mamba-2 / Nemotron-H
out = layer_norm(x, weight, bias, quant="nvfp4", global_scale=g)       # out.y uint8 nibbles, out.scales uint8 (E4M3)
```

* **`norm(x, weight=None, bias=None, eps=1e-6, *, kind="rms", one_plus=False, order=None, gate=None, gate_mode="after",
  groups=1, residual=None, quant=None, global_scale=None, y_dtype=None, variant=0, items=0)`** → `NormOut(y, scales, residual)`
  * `x`: `[rows, hidden]` bfloat16 / float16 / float32, `hidden` a multiple of `256 * groups`, at most 32768.
  * `weight`, `bias`: `[hidden]` in `y_dtype` (default: the weight's dtype, else x's); x is `y_dtype` or float32.
  * `order=None`: `"hf"` for an RMSNorm with a plain weight, no bias and no gate (Llama, Qwen), `"fp32"` otherwise (Gemma's
    `1 + weight`, LayerNorm, biases, gates); `"hf"` is RMSNorm's alone.
  * `gate`: `[rows, hidden]` in x's dtype; `gate_mode` `"after"` or `"before"`.
  * `quant`: `None`, `"int8"` or `"nvfp4"`; `global_scale` is a float32 tensor of one element (default 1).
* **`rms_norm(x, weight, eps, **kw)`, `layer_norm(x, weight, bias, eps, **kw)`**: `norm` with the kind set.
* **`RMSNorm`, `LayerNorm`**: modules around them. **`norm_variants`**: the instances for a shape, as data.

A group is a multiple of 256 values; per-head norms of 64 or 128 values (QK-norm) belong to the attention library
(`mlops_fbla_qkv_at`).

## Samples

Short runnable examples in [`samples/`](samples):

| sample | shows |
|---|---|
| `01_quickstart.py` | RMSNorm with a residual add against transformers' Llama formula; an FP32 residual stream |
| `02_flavors.py` | Gemma's `1 + weight`, LayerNorm with a bias, Qwen3-Next's gate, Mamba-2's gated norm over groups, each against PyTorch |
| `03_quantized_outputs.py` | INT8 and NVFP4 outputs: shapes, scales, the INT8 dequantization error |
| `04_c_api.cu` | the C API: the variants of a shape, a call with a residual and an INT8 output, error reporting |


## C / C++

[`include/mlops_norm.h`](include/mlops_norm.h) includes the whole package:

* **Call.** `mlops_norm(config, variant, args, stream)` over raw device pointers, returning a status
  (`mlops_norm_last_error()` describes failures). `mlops_norm_config` names the flavor, the dtypes and the output; the pointers
  are in `mlops_norm_args`.
* **Variants.** `mlops_norm_variants` and `mlops_norm_item_count`: the instances for a shape and the resolved element count.
* **Items.** [`mlops_norm_items.h`](include/mlops_norm_items.h) describes a call as work items for an executor (`mlops_norm_describe`):
  the CTA geometry, the arguments, the entry compiled with NVRTC from [`mlops_norm_items.cuh`](include/mlops_norm_items.cuh), and the
  byte ranges each item reads and writes (`mlops_norm_item_access`).

## Build and test

```sh
make lib                # out/libmlops_norm.a (ARCHS="61 89", ARCHS=native; default every architecture sm_50..sm_90)
make test               # C tests (ARCHS= and TEST_ARCH= as in ../build/arch.mk); needs libnvfp4effma.a, the reference quantizer
make bench              # the norms, per output and residual
make install PREFIX=..  # headers, library, pkg-config file
pytest python/tests     # the PyTorch package against PyTorch's RMSNorm / LayerNorm formulas, gates and groups
```

`make test` checks, bit for bit against a host emulation: every flavor (RMS / LayerNorm; weight, one plus, none; bias; both
orders; groups), every dtype pair, with and without a residual (also in place), every variant and element count, INT8 and
NVFP4 outputs with padded scale rows against `nvfp4a_quantize` of the same y, the E2M1 conversion against NVFP4EffMA's for all
2^32 floats, and the invalid arguments; the gates, whose `exp` is libm's, within a bound. The items go through NVRTC: each entry is
compiled from its `entry` string and run on its described CTA, the outputs are the call's, and running every item alone on poisoned
outputs shows that only declared bytes change.

## License

GNU AGPL v3 with the Linking and Attribution Exception ([NOTICE](../NOTICE)).
