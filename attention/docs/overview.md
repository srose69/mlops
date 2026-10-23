# FBLA overview: quality and performance

The format, arithmetic and kernels: [fbla/README.md](../fbla/README.md). The comparison with llama.cpp on
Qwen3-8B: [llamacpp-compare.md](llamacpp-compare.md). Benchmark methodology: [performance.md](performance.md).

## How the cache is read

Every closed 128-token group is a 6-bit per-channel affine code (one BF16 base and step per channel and
tensor) split into a coarse plane (the top 3 or 4 bits) and a fine plane (the rest); the attention-sink group
and the open group are BF16.

* **Decode** reads every coarse plane: the coarse pass is already a complete FlashAttention pass over K and V.
  A selection pass lists the 16-token blocks whose normalized coarse-softmax mass `W_j` exceeds the mode's
  threshold, and the refinement pass adds only their `full − coarse` correction. At Qwen3-0.6B's refinement
  rates the format reads 4.34 (4+2) and 3.60 (3+3) bits per element (the kernel rereads a selected record as
  coarse and fine, see below), so decode runs near DRAM bandwidth.
* **Prefill** is compute-bound: a DP4A over 6-bit codes costs what one over the coarse plane does, so it reads
  every block through its full codes (decode with `tau` -> 0) in one streaming GEMM-tile kernel, and both modes
  give the same output.

## Quality

Qwen3-0.6B, [`bench/fbla_lm.py`](../bench/fbla_lm.py) `--win 16384 --decode 1024 --windows 11` over six of the
repository's READMEs: in each of 11 windows of 16,384 tokens the last 1,024 tokens decode against the cache of
the first 15,360, 11,264 scored tokens in all (the windows' contexts overlap, their scored tokens do not).
Paired deltas against BF16 SDPA (BF16 loss 2.323), mean and the range over the windows; the kernel rows run
`mlops_fbla_decode` with its Q8 / P8 arithmetic.

| KV cache | bits per element stored | delta loss |
|---|---:|---:|
| BF16 SDPA | 16 | +0.0000 |
| llama.cpp `q8_0` | 8.5 | -0.0001 (-0.0055 .. +0.0040) |
| FBLA 3+3, `W_j > 3e-4` (CUDA kernel) | 6.25 | -0.0008 (-0.0063 .. +0.0025) |
| FBLA 4+2, `W_j > 1e-3` (CUDA kernel) | 6.25 | +0.0005 (-0.0049 .. +0.0057) |
| KIVI 4-bit, K per channel / V per token, groups of 32 | 5.0 | +0.0038 (-0.0031 .. +0.0120) |
| llama.cpp `q4_0` | 4.5 | +0.4755 (+0.1292 .. +0.7936) |

The windows differ by up to ±0.006, so FBLA in either mode and `q8_0` are within the noise of BF16; KIVI-4 is
measurably worse and `q4_0` loses a fifth of the model.  An earlier study on two windows (2,048 scored
tokens) ranked the modes +0.0003 / +0.0025: within that run's noise, not a difference between them.

On Qwen3-8B (one 6,144-token window, every position scored, [llamacpp-compare.md](llamacpp-compare.md#quality))
the FBLA 4+2 cache is within ±0.002 of exact FP32 attention with BF16 or NVFP4 weights.

### Token distributions

One 16,384-token window, 2,048 evaluated tokens. The first FBLA row is the format model (4.3467 bits/element,
4.836% refined); the second is the CUDA kernel. `delta log p(true)` is approximation minus BF16, in nats.

| attention | top-1 agreement | KL mean | KL p99 | KL p99.9 | KL max | delta log p mean | p0.1 | p1 | p99 | p99.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FBLA 4+2 format | 96.7285% (1981/2048) | 0.00221755 | 0.0189053 | 0.0663226 | 0.0948992 | -0.000487557 | -0.508834 | -0.271406 | +0.310232 | +0.613736 |
| FBLA 4+2 CUDA kernel | 96.0449% (1967/2048) | 0.00228486 | 0.0171846 | 0.0541916 | 0.228847 | -0.000320436 | -0.425198 | -0.243340 | +0.342437 | +0.724149 |
| KIVI 4-bit | 95.5566% (1957/2048) | 0.00464367 | 0.0384911 | 0.152416 | 0.351163 | -0.000999796 | -1.02277 | -0.401161 | +0.382878 | +0.780131 |

The scripts that produced the two kinds of measurement are [`bench/fbla_lm.py`](../bench/fbla_lm.py) and
[`bench/fbla_token_metrics.py`](../bench/fbla_token_metrics.py). They run from a model directory and input
text. Optional `--baseline-losses` / `--bf16-hidden-cache` files let later formats reuse the exact same
baseline, but model, QKV and hidden-state dumps are local inputs or caches: none are distributed in this
repository.

## Performance

### Decode

The native benchmark uses a single cache allocated for 1M tokens and decodes it at 16K .. 1M. It first warms
GPU Boost and measures a 256 MiB device-to-device copy, then idles for one second before each row and reports
actual implied DRAM traffic: coarse K/V and metadata, exact sink/open groups, and coarse plus fine K/V for
each selected block. Synthetic scores are thresholded to the Qwen3-0.6B per-query refinement rates (12.3%
for 3+3, 4.8% for 4+2); the reported rate is the union read once for all GQA queries.

GTX 1080, Qwen geometry (8 KV heads, two queries each), controlled run with a 282 GB/s D2D-copy baseline.
Default repeat counts are selected per scenario for about one second of timed work; `MLOPS_BENCH_REPS`
overrides them:

| context | 3+3 latency | effective GB/s | 4+2 latency | effective GB/s |
|---:|---:|---:|---:|---:|
| 16K | 104.4 us | 192.9 | 102.0 us | 204.7 |
| 64K | 318.5 us | 246.4 | 319.5 us | 254.9 |
| 128K | 598.0 us | 261.1 | 605.2 us | 268.1 |
| 256K | 1.159 ms | 268.9 | 1.179 ms | 274.6 |
| 512K | 2.506 ms | 248.5 | 2.553 ms | 253.2 |
| 1M | 5.197 ms | 239.4 | 5.184 ms | 249.4 |

Against PyTorch ([`bench/fbla_vs_sdpa.py`](../bench/fbla_vs_sdpa.py): random K / V, so more blocks are refined
than at Qwen3's rates, and FBLA is slower here than in the table above):

| context | SDPA BF16 math | SDPA FP32 math | FBLA 3+3 | FBLA 4+2 | speed-up vs BF16 / FP32 |
|---:|---:|---:|---:|---:|---:|
| 16K | 6.80 ms | 5.66 ms | 193 us | 163 us | 35–42× / 29–35× |
| 64K | 30.5 ms | 27.3 ms | 375 us | 286 us | 81–107× / 73–95× |
| 128K | 63.0 ms | 56.8 ms | 429 us | 533 us | 118–147× / 107–132× |
| 256K | out of memory | out of memory | 805 us | 1.03 ms | |

At 256K PyTorch runs out of the GTX 1080's memory: SDPA repeats K / V to every query head (and FP32 copies them
first).
Results by GPU: [benches/table.md](../benches/table.md) (full GTX 1080 run: [benches/1080.md](../benches/1080.md)).

### Prefill

`make bench`, rows `PREFILL`: a chunk appended to a cache of `context` tokens and attended, 8 KV heads, two
queries each; work counted as attended (query, key) pairs, "equiv. TFLOPS" = 4 x 128 FLOP per pair (what
Q K^T and P V cost in floating point). GTX 1080 (FP32 peak about 9.2 TFLOPS at 1.8 GHz):

| context | chunk | 3+3 latency | equiv. TFLOPS | 4+2 latency | equiv. TFLOPS |
|---:|---:|---:|---:|---:|---:|
| 16K | 2048 | 42.3 ms | 6.90 | 41.4 ms | 7.05 |
| 64K | 2048 | 157.1 ms | 7.11 | 139.5 ms | 8.00 |
| 64K | 8192 | 647.0 ms | 7.22 | 581.7 ms | 8.03 |
| 256K | 2048 | 611.6 ms | 7.22 | 550.7 ms | 8.02 |

The kernel is warp-specialized (one or two producer warps decode the records into swizzled tiles, the rest
compute) and needs 8 warps per CTA; 3+3 takes a second producer warp to unpack its 3-bit planes, so it runs
with one consumer warp fewer than 4+2.

On Qwen3-0.6B's own q / k / v (2048 queries at context 14,336, 16 query heads): FBLA prefill 43 .. 47 ms in
either mode, PyTorch FP32 SDPA (memory-efficient kernel) 88 ms, cuBLAS FP32 `softmax(Q K^T) V` 200 ms.

### A whole model

[`samples/04_generate.py`](../samples/04_generate.py) and [`bench/llama_bench.py`](../bench/llama_bench.py) run
Qwen3-8B-NVFP4 end to end on a GTX 1080: 778 tokens/s prefill and 35.4 tokens/s decode at an empty context,
361 / 29.5 at 16K; [llamacpp-compare.md](llamacpp-compare.md).
