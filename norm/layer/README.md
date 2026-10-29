# mlops/norm/layer

RMSNorm and LayerNorm of rows in the flavors of current models, with a residual add before and an INT8 / NVFP4 quantization
after, on every GPU sm_50 .. sm_90. Installation and the PyTorch API: [the package README](../README.md).

## The computation

Per row (or per group of `hidden / groups` values, normalized independently), with `c = v` (RMS) or `v - mean(v)` (LayerNorm):

```
v  = round_x(x + residual)          # written to residual_out
v  = v * silu(gate)                 # gate BEFORE the norm (Mamba-2, Nemotron-H)
r  = 1 / sqrt(mean(c^2) + eps)
w' = weight | 1 + weight | 1        # the weight, zero-centered (Gemma, Qwen3-Next, LayerNorm1P), none
```

| order | formula | for |
|---|---|---|
| `ORDER_ROUND_THEN_SCALE` | `y = round_y(round_x(c * r) * w')` | RMSNorm of Llama, Qwen, Mistral, DeepSeek (Hugging Face's roundings) |
| `ORDER_FP32` | `y = round_y(fma(c * r, w', bias))`, then `y *= silu(gate)` for a gate AFTER (Qwen3-Next, Mamba-2) | Gemma, LayerNorm of PyTorch, GPT, Falcon, Phi, OLMo, any bias or gate |

x may be BF16, FP16 or FP32 (a residual stream kept in FP32); weight, bias and y are in y's dtype, which is x's or x is FP32. The
sums are taken in units of 256 elements (a lane's eight values in order, then a butterfly), the unit sums added lane-strided: the
same arithmetic as `mlops_embed`'s parts and `mlops_lm_head`'s norm scalar, and independent of the variant and the element count,
so every launch shape and every GPU gives the same bytes (the gates' `exp` aside). LayerNorm takes the mean first, then the sum
of `(v - mean)^2`.

| output | what is written |
|---|---|
| dtype | y `[rows][hidden]` |
| INT8 | int8 `[rows][hidden]` and FP32 scales `[rows][ld_scale]`, one per 16: `sb = amax / 127`, `q = RNE(y / sb)` |
| NVFP4 | E2M1 nibbles `[rows][hidden / 2]` and E4M3 scales `[rows][ld_scale]`, one per 16, with a decode-factor global `g` |

The E4M3 and E2M1 conversions are NVFP4EffMA's own (`nvfp4effma.cuh`), imported; the E2M1 one is rewritten in fewer
instructions and checked against it for every one of the 2^32 floats in `make test`.

## How it is built

The kernel is laid out as CUTLASS lays one out: `Element<dtype>` (load / round / store of a lane's 8 values), `OutputOp` (the
epilogue: dtype, INT8 or NVFP4), `Traits` (the flavor), `Params`, and `NormKernel::operator()`. A CTA of W warps takes rows; a
row is units of 256 elements (a lane holds 8 consecutive values: 16 bytes of BF16), the warps taking the units round robin. Pass
one reads x and the residual, writes v, and puts the unit sums in shared memory; pass two (and a second sum for LayerNorm) reads
v back (from `residual_out`, so the call may run in place), applies r and the weight, rounds, and quantizes: the 16 values of a
block are two lanes, one shuffle for their maximum. From sm_80 a BF16 pair is rounded by `cvt.rn.bf16x2.f32`; below it, by
integer instructions with the same results.

**Variants are data.** `mlops_norm_variants` lists the instances for a configuration (W = 1, 2, 4, 8 warps, with threads, shared
memory and registers from `cudaFuncAttributes`); the call and the items run the one the caller names, and the element count is the
caller's too (`items`, 0: the resident CTAs). There is one instance per (kind, order, gate, dtype pair, output): 105 kernels, no
dtype test in any of them, all free of spills on sm_61, 70, 75, 80 and 89 (up to 74 registers). Buffers should be 256-byte aligned.

## Measurements

`make bench`, the best variant of several runs, 2048 x 4096 BF16, in µs (GTX 1080 with the display on the same card; L40S):

| | dtype | dtype, residual | INT8 | INT8, residual | NVFP4 | NVFP4, residual |
|---|---:|---:|---:|---:|---:|---:|
| 1080, RMS (Hugging Face's order) | 160 | 307 | 127 | 252 | 130 | 208 |
| 1080, RMS one plus | 169 | 328 | 129 | 243 | 125 | 222 |
| 1080, LayerNorm + bias | 188 | 342 | 154 | 261 | 150 | 237 |
| L40S, RMS (Hugging Face's order) | 20.1 | 25.2 | 17.9 | 22.3 | 18.6 | 22.5 |
| L40S, RMS one plus | 19.9 | 24.9 | 19.6 | 23.9 | 19.8 | 23.6 |
| L40S, LayerNorm + bias | 22.3 | 27.6 | 22.5 | 26.6 | 23.3 | 27.0 |

On the 1080 these are 200 - 250 GB/s of traffic, near the 235 - 250 GB/s a device-to-device copy reaches on the same card. On an L40S the
shape sits in its 96 MB of L2; 8192 x 4096 with a residual (DRAM) takes 389 µs as dtype and 319 µs as NVFP4, 690 GB/s, about
80 % of the card's peak.
