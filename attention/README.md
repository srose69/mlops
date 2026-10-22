# mlops-attention

FBLA compressed-KV FlashAttention decode for NVIDIA GPUs with DP4A (sm_61 .. sm_90): a C API
([`include/`](include)) and a PyTorch package, `mlops.attention`. Every closed 128-token group is a 6-bit
per-channel affine quantization split into an always-read coarse plane and a selectively read fine plane;
the attention-sink group and the open group remain BF16. The format, arithmetic, kernels and full
measurements: [fbla/README.md](fbla/README.md).

| mode | coarse plane | fine plane | default refinement test | Qwen3-0.6B loss vs BF16 SDPA |
|---|---:|---:|---:|---:|
| `3+3` | 3 bits | 3 bits | `W_j > 3e-4` | +0.0003, CUDA kernel |
| `4+2` | 4 bits | 2 bits | `W_j > 1e-3` | +0.0025, CUDA kernel |

Here `W_j` is a 16-token block's normalized coarse-softmax mass. The coarse pass is already a complete
FlashAttention pass over K and V. A selection pass builds a compact work list; the refinement pass adds only
the `full − coarse` correction. One packed record is shared by all 1, 2 or 4 query heads of a KV head.

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
from mlops.attention import encode, decode

k = torch.randn(1, 8, 16384, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
cache = encode(k, v, mode="3+3", capacity=65536)

q = torch.randn(1, 16, 128, device="cuda", dtype=torch.bfloat16)
y = decode(q, cache)                            # [1, 16, 128]
y = decode(q, cache, tau=1e-4)                  # more refinement
```

* **`encode(k, v, mode="3+3", capacity=None)`** takes BF16 `[batch, kv_heads, tokens, 128]` tensors and
  produces the documented packed cache. `mode` is `"3+3"` or `"4+2"`. `capacity` fixes the per-head strides
  for a growing cache; an inference engine normally writes this layout directly instead of repacking a dense
  cache.
* **`decode(q, cache, scale=0, tau=0)`** takes one BF16 query per query head, `[batch, q_heads, 128]`, with
  `q_heads / kv_heads` equal to 1, 2 or 4. Zero scale means `1 / sqrt(128)`; zero tau selects the mode's
  default. Scratch is allocated once per device and number of KV heads, then reused.
* The first 128 tokens and the currently open 128-token group are exact BF16. Closed groups use one BF16 base
  and step per channel and tensor, with a 6-bit code. Bases round down and steps round up, so all source values
  are representable without clipping.

## Quality

Qwen3-0.6B, two windows of 16,384 tokens. Loss is evaluated on the last 1,024 tokens decoding against the
cache of the first 15,360; values are paired deltas against BF16 SDPA (mean, min .. max). The non-kernel FBLA
rows are the format study; kernel rows run `mlops_fbla_decode` and include its Q8/P8 arithmetic.

| attention | K / V bits per element read | delta loss |
|---|---:|---:|
| BF16 SDPA | 16 | +0.0000 (+0.0000 .. +0.0000) |
| FBLA 4+2, `W_j > 1e-3` | 4.34 (refined 4.4%) | +0.0008 (-0.0019 .. +0.0036) |
| KIVI 4-bit, K per channel / V per token, groups of 32 | 5.0 | +0.0019 (+0.0019 .. +0.0020) |
| FBLA kernel 3+3, `W_j > 3e-4` | measured by the native benchmark | +0.0003 (-0.0020 .. +0.0027) |
| llama.cpp `q8_0` | 8.5 | +0.0007 (+0.0002 .. +0.0011) |
| FBLA kernel 4+2, `W_j > 1e-3` | measured by the native benchmark | +0.0025 (+0.0005 .. +0.0044) |
| vLLM FP8 E4M3, per tensor | 8 | +0.0027 (-0.0014 .. +0.0068) |
| FBLA 4+2, coarse only | 4.25 (refined 0.0%) | +0.0161 (+0.0148 .. +0.0174) |
| FBLA 3+3, `W_j > 3e-4` | 3.60 (refined 11.7%) | +0.0040 (+0.0039 .. +0.0041) |

| KIVI 2-bit | 3.0 | +0.1031 (+0.1011 .. +0.1050) |

The CUDA kernel's physical traffic is not the format-study expression `coarse + fine × refined`: a selected
record is reread as coarse and fine to form an exact `full − coarse` correction. `make bench` counts those
bytes, metadata and exact groups from the actual kernel algorithm.

### Token distributions

One 16,384-token window, 2,048 evaluated tokens. The first FBLA row is the format model (4.3467 bits/element,
4.836% refined); the second is the CUDA kernel. `delta log p(true)` is approximation minus BF16, in nats.

| attention | top-1 agreement | KL mean | KL p99 | KL p99.9 | KL max | delta log p mean | p0.1 | p1 | p99 | p99.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FBLA 4+2 format | 96.7285% (1981/2048) | 0.00221755 | 0.0189053 | 0.0663226 | 0.0948992 | -0.000487557 | -0.508834 | -0.271406 | +0.310232 | +0.613736 |
| FBLA 4+2 CUDA kernel | 96.0449% (1967/2048) | 0.00228486 | 0.0171846 | 0.0541916 | 0.228847 | -0.000320436 | -0.425198 | -0.243340 | +0.342437 | +0.724149 |
| KIVI 4-bit | 95.5566% (1957/2048) | 0.00464367 | 0.0384911 | 0.152416 | 0.351163 | -0.000999796 | -1.02277 | -0.401161 | +0.382878 | +0.780131 |

The scripts that produced the two kinds of measurement are
[`bench/fbla_lm.py`](bench/fbla_lm.py) and [`bench/fbla_token_metrics.py`](bench/fbla_token_metrics.py).
They run from a model directory and input text. Optional `--baseline-losses` / `--bf16-hidden-cache` files let
later formats reuse the exact same baseline, but model, QKV and hidden-state dumps are local inputs or caches:
none are distributed in this repository.

## Performance

The native benchmark uses a single cache allocated for 1M tokens and decodes it at 16K .. 1M. It first warms
GPU Boost and measures a 256 MiB device-to-device copy, then idles for one second before each row and reports
actual implied DRAM traffic: coarse K/V and metadata, exact sink/open groups, and coarse plus fine K/V for
each selected block. Synthetic scores are thresholded to the Qwen3-0.6B per-query refinement rates (12.3%
for 3+3, 4.8% for 4+2); the reported rate is the union read once for all GQA queries.

GTX 1080, Qwen geometry (8 KV heads, two queries each), controlled run with a 282 GB/s D2D-copy baseline.
Default repeat counts are selected per scenario for about one second of timed work;
`MLOPS_BENCH_REPS` overrides them:

| context | 3+3 latency | effective GB/s | 4+2 latency | effective GB/s |
|---:|---:|---:|---:|---:|
| 16K | 104.4 us | 192.9 | 102.0 us | 204.7 |
| 64K | 318.5 us | 246.4 | 319.5 us | 254.9 |
| 128K | 598.0 us | 261.1 | 605.2 us | 268.1 |
| 256K | 1.159 ms | 268.9 | 1.179 ms | 274.6 |
| 512K | 2.506 ms | 248.5 | 2.553 ms | 253.2 |
| 1M | 5.197 ms | 239.4 | 5.184 ms | 249.4 |

Results by GPU: [benches/table.md](benches/table.md) (full GTX 1080 run: [benches/1080.md](benches/1080.md)).
The controlled [`bench/run_bench.sh`](bench/run_bench.sh) records board conditions and automatically updates
both files; methodology: [docs/performance.md](docs/performance.md).

## Sample

[`samples/01_decode.py`](samples/01_decode.py) builds both a packed cache and a decode query. It is deliberately
small; an inference engine should append directly to the C layout documented in `mlops_fbla.h`.

## C / C++

[`include/mlops_fbla.h`](include/mlops_fbla.h) (all of the package: `mlops_attention.h`) specifies the cache
layout and API. `mlops_fbla_decode` takes raw device pointers, explicit scratch and a `cudaStream_t`, and
returns a status; `mlops_fbla_last_error()` describes failures. Scratch must be zero-filled once and is reusable
across calls and context lengths that fit it.

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
