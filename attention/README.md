# mlops-attention

FBLA (Flash Block Attention) - compressed-KV FlashAttention prefill and decode for NVIDIA GPUs with DP4A (sm_61 .. sm_90): a C API
([`include/`](include)) and a PyTorch package, `mlops.attention`. Every closed 128-token group is a 6-bit
per-channel affine quantization split into an always-read coarse plane and a selectively read fine plane;
the attention-sink group and the open group remain BF16. The format, arithmetic, kernels and full
measurements: [fbla/README.md](fbla/README.md).

| mode | coarse plane | fine plane | default refinement test | Qwen3-0.6B loss vs BF16 SDPA (CUDA kernel) |
|---|---:|---:|---:|---:|
| `3+3` | 3 bits | 3 bits | `W_j > 3e-4` | -0.0008 (within noise) |
| `4+2` | 4 bits | 2 bits | `W_j > 1e-3` | +0.0005 (within noise) |

Here `W_j` is a 16-token block's normalized coarse-softmax mass. Decode reads the coarse planes and refines
only the blocks above the threshold; prefill, compute-bound, reads the full codes. One packed record is shared
by all 1, 2 or 4 query heads of a KV head.

## Install

Needs PyTorch (a CUDA build), the CUDA toolkit and a GPU sm_61 or newer. Nothing is downloaded.

```sh
pip install --no-index -f . mlops-attention     # from a release, beside the wheel
./setup.sh attention                            # from the sources (the mlops root)
```

The build targets the visible GPUs, or `MLOPS_ARCHS="61 89"`; other variables:
[`../build/torch_ext.py`](../build/torch_ext.py).

## Python

```python
import torch
from mlops.attention import append, decode, empty, encode, prefill
from mlops.positional import RoPE                # mlops-positional, installed on its own

rope = RoPE(128, base=1e6)                       # Qwen3's rotary embedding

# One layer.  q / k / v: after the projections (and Qwen3's q / k norm), [batch, heads, n, 128] BF16.
cache = empty(1, 8, 65536, mode="4+2")           # laid out for 65,536 tokens; the mode is fixed here
q, k = rope(q, k, layout="bhsd")                 # the prompt at positions 0 .. n - 1
y = prefill(q, k, v, cache)                      # [1, 16, n, 128]; call it per chunk for a long prompt

q1, k1 = rope(q1, k1, layout="bhsd", offset=t)   # a new token at position t: [1, 16, 1, 128], [1, 8, 1, 128]
append(cache, k1, v1)                            # the cache grows in place
y1 = decode(q1[:, :, 0], cache)                  # [1, 16, 128]
y1 = decode(q1[:, :, 0], cache, tau=1e-4)        # more refinement

cache = encode(k_all, v_all, mode="3+3", capacity=65536)  # or a cache from dense K / V
```

* **`encode(k, v, mode="3+3", capacity=None)`** takes BF16 `[batch, kv_heads, tokens, 128]` tensors and
  produces the documented packed cache. `mode` is `"3+3"` or `"4+2"`. `capacity` fixes the per-head strides
  for a growing cache; an inference engine normally writes this layout directly instead of repacking a dense
  cache.
* **`append(cache, k, v)`** adds `[batch, kv_heads, n, 128]` tokens in place, up to `capacity`. A full open group
  is quantized when the next token arrives, exactly as `encode` would; the result decodes bit-for-bit like a
  fresh `encode` of the same tokens.
* **`decode(q, cache, scale=0, tau=0)`** takes one BF16 query per query head, `[batch, q_heads, 128]`, with
  `q_heads / kv_heads` equal to 1, 2 or 4. Zero scale means `1 / sqrt(128)`; zero tau selects the mode's
  default. Scratch is allocated once per device and number of KV heads, then reused.
* **`empty(batch, kv_heads, capacity, mode="3+3")`** is a zero-filled cache of 0 tokens.
* **`prefill(q, k, v, cache, scale=0)`** appends the chunk's K / V like `append` and returns the causal
  attention of its queries over the whole cache (position `tokens + i` reads tokens `0 .. tokens + i`). Closed
  groups are read through the full 6-bit codes: prefill is compute-bound, a DP4A over 6-bit codes costs what one
  over the coarse plane does, so every block is refined (`decode` with `tau` -> 0) and both modes give the same
  output. One streaming kernel, no score matrix, no scratch.
* The first 128 tokens and the currently open 128-token group are exact BF16. Closed groups use one BF16 base
  and step per channel and tensor, with a 6-bit code. Bases round down and steps round up, so all source values
  are representable without clipping.

## Quality and speed

On Qwen3-0.6B (16K contexts, 11,264 scored tokens, BF16 loss 2.323) both modes are within the noise of BF16
attention (3+3 -0.0008, 4+2 +0.0005), as is llama.cpp's `q8_0` at 8.5 bits per element (-0.0001); KIVI-4 costs
+0.0038 and `q4_0` +0.48. On Qwen3-8B the cache is within ±0.002 of exact attention. On a GTX 1080 (Qwen3
geometry):

| | FBLA | baseline |
|---|---:|---:|
| decode, 16K context | 163 .. 193 us on random data, 102 .. 104 us at Qwen3's refinement rates | PyTorch BF16 SDPA 6.8 ms (35–42×) |
| decode, 256K context | 0.8 .. 1.0 ms on random data | PyTorch SDPA: out of memory |
| decode, 1M context | 5.2 ms, 249 GB/s (88 % of a device copy) | |
| prefill attention | 8.0 equivalent FP32 TFLOPS (4+2, 64K context) | PyTorch FP32 SDPA 2× slower (Qwen3 data) |
| Qwen3-8B-NVFP4 end to end, 16K context | 361 tokens/s prefill, 29.5 decode | llama.cpp 192 / 20.0 (KV `q8_0`) |

Full tables, methodology and the token-level comparison: [docs/overview.md](docs/overview.md); against
llama.cpp: [docs/llamacpp-compare.md](docs/llamacpp-compare.md); results by GPU:
[benches/table.md](benches/table.md), controlled runs with [`bench/run_bench.sh`](bench/run_bench.sh).

## Samples

| sample | what it shows |
|---|---|
| [`01_decode.py`](samples/01_decode.py) | `encode` and `decode`: shapes, the refinement threshold, error against exact attention, storage |
| [`02_streaming.py`](samples/02_streaming.py) | one preallocated cache for a whole generation: `prefill` of the prompt in chunks, then `append` + `decode` per token; checks both against exact attention and that the cache decodes exactly like a fresh encode |
| [`03_transformers.py`](samples/03_transformers.py) | FBLA inside a Hugging Face model through `AttentionInterface`: FBLA prefill and decode; greedy generation and teacher-forced KL against SDPA |
| [`04_generate.py`](samples/04_generate.py) | a complete generator without transformers' model: Qwen3 / Qwen2 / Llama from safetensors (BF16, or NVFP4 from ModelOpt such as `nvidia/Qwen3-8B-NVFP4`), RoPE from `mlops.positional`, NVFP4 linear layers (NVFP4EffMA), FBLA prefill + decode, layers streamed from host memory when the GPU is small; streamed text, tokens/s and cache size, `--attn sdpa` as the baseline |

Where it plugs in: FBLA replaces the KV cache and the attention over it. The attention function receives
q, k, v after the projections, the q / k norm and RoPE; prefill calls `prefill(q, k, v, cache)` on an `empty`
cache (or runs ordinary attention and writes the prompt's K / V with `encode`), every decode step appends the new
token's K / V with `append(cache, k, v)` and calls `decode(q, cache)`. Every cache write runs on the GPU
(`mlops_fbla_prefill`), and no BF16 cache is kept; an inference engine can also write the layout of
`mlops_fbla.h` from its own kernels.

## C / C++

[`include/mlops_fbla.h`](include/mlops_fbla.h) (all of the package: `mlops_attention.h`) specifies the cache
layout and API. `mlops_fbla_decode` takes raw device pointers, explicit scratch and a `cudaStream_t`, and
returns a status; `mlops_fbla_last_error()` describes failures. Scratch must be zero-filled once and is reusable
across calls and context lengths that fit it. `mlops_fbla_prefill` appends a chunk (and with q attends its
queries) on the same cache and needs no scratch.

## Build and test

```sh
make lib                # out/libmlops_attention.a: every DP4A architecture sm_61..sm_90
make test               # both modes vs float64; GQA 1/2/4, capacity, all/none refinement, up to 256K
make bench              # uncontrolled local run: 36 rows, traffic and NVML clocks
bench/run_bench.sh      # controlled board run; writes benches/<gpu>.md and benches/table.md
make install PREFIX=..  # headers, library, pkg-config file
pytest python/tests     # record layout, kernel vs model, capacity and exact-attention distance
```

Tests and benchmark use GPU `GPU` (default 0) and build for its architecture. Kernel formats are separate
objects compiled in parallel (`JOBS`); toolchain and architecture selection:
[`../build/arch.mk`](../build/arch.mk).

## License

GNU AGPL v3 with the Linking and Attribution Exception ([NOTICE](../NOTICE)).
