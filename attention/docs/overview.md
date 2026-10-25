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
* **Prefill** is compute-bound, so it reads every closed block through its full codes (decode with `tau` -> 0)
  in one streaming GEMM-tile kernel, and both modes give the same output. On sm_75 .. sm_89 the kernel runs on
  tensor cores (P blocks of 128 tokens in the closed groups); elsewhere on DP4A (P blocks of 16).

## Quality

Qwen3-0.6B on the WikiText-2 (raw) test split (299K tokens; a fixed public text, downloaded by the scripts),
[`bench/fbla_lm.py`](../bench/fbla_lm.py) `--win 16384 --decode 1024 --windows 11`: in each of 11 windows of
16,384 tokens the last 1,024 tokens run against the cache of the first 15,360, 11,264 scored tokens in all (the
windows' contexts overlap, their scored tokens do not). Paired deltas against BF16 SDPA (BF16 loss 2.784), mean
and the range over the windows, on an L40S (sm_89). The decode rows run `mlops_fbla_decode` with
its Q8 / P8 arithmetic; the prefill row runs the same queries through `mlops_fbla_prefill` on tensor cores
(chunks of 1,024, every closed block through its full codes, P blocks of 128 tokens).

| KV cache | bits per element stored | delta loss |
|---|---:|---:|
| BF16 SDPA | 16 | +0.0000 |
| llama.cpp `q8_0` | 8.5 | +0.0011 (-0.0047 .. +0.0067) |
| FBLA 3+3, `W_j > 3e-4` (decode kernel) | 6.25 | +0.0013 (-0.0048 .. +0.0046) |
| FBLA 4+2, `W_j > 1e-3` (decode kernel) | 6.25 | +0.0014 (-0.0040 .. +0.0042) |
| FBLA 4+2, prefill kernel (tensor cores) | 6.25 | -0.0013 (-0.0045 .. +0.0019) |
| FBLA 4+2, prefill kernel (DP4A, `MLOPS_FBLA_DP4A=1`) | 6.25 | +0.0003 (-0.0065 .. +0.0047) |
| KIVI 4-bit, K per channel / V per token, groups of 32 | 5.0 | +0.0010 (-0.0034 .. +0.0058) |
| llama.cpp `q4_0` | 4.5 | +0.2048 (+0.0230 .. +0.5086) |

The windows differ by up to ±0.005, so FBLA in every mode and path, `q8_0` and KIVI-4 are within the noise of
BF16; `q4_0` raises the loss by 7 % (+0.20).

On Qwen3-8B (one 6,144-token window, every position scored, [llamacpp-compare.md](llamacpp-compare.md#quality);
measured on this repository's documentation, not re-run on WikiText) the FBLA 4+2 cache is within ±0.002 of
exact FP32 attention with BF16 or NVFP4 weights.

### Token distributions

The first 16,384 tokens of the same text, 2,048 evaluated tokens (`bench/fbla_token_metrics.py`, one window: a
flip of one token moves top-1 agreement by 0.05 %). The first FBLA row is the format model (4.3424
bits/element, 4.622% refined); the next three are CUDA kernels. `delta log p(true)` is approximation minus
BF16, in nats.

| attention | top-1 agreement | KL mean | KL p99 | KL p99.9 | KL max | delta log p mean | p0.1 | p1 | p99 | p99.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FBLA 4+2 format | 96.0449% (1967/2048) | 0.00231438 | 0.0166290 | 0.0510556 | 0.165255 | +0.00188933 | -0.433487 | -0.213318 | +0.243424 | +0.449123 |
| FBLA 4+2 decode kernel | 96.3867% (1974/2048) | 0.00223486 | 0.0166284 | 0.0458892 | 0.0490008 | -0.00336936 | -0.433205 | -0.236322 | +0.239920 | +0.444697 |
| FBLA 3+3 decode kernel | 96.0449% (1967/2048) | 0.00259956 | 0.0226739 | 0.0560670 | 0.184098 | +0.00130370 | -0.397581 | -0.243113 | +0.326487 | +0.647485 |
| FBLA 4+2 prefill kernel (tensor cores) | 96.4844% (1976/2048) | 0.00205000 | 0.0131507 | 0.0303366 | 0.0365622 | +0.00155675 | -0.482885 | -0.217820 | +0.245980 | +0.471112 |
| llama.cpp `q8_0` | 96.5820% (1978/2048) | 0.00161858 | 0.0102092 | 0.0294826 | 0.0871680 | -0.00119162 | -0.434718 | -0.215168 | +0.195002 | +0.365658 |
| KIVI 4-bit | 95.4590% (1955/2048) | 0.00467099 | 0.0348574 | 0.105348 | 1.51841 | -0.00445893 | -0.739119 | -0.338186 | +0.325965 | +0.729860 |
| llama.cpp `q4_0` | 74.2676% (1521/2048) | 0.366470 | 3.73899 | 7.94368 | 13.8816 | -0.245661 | -9.23180 | -4.92260 | +2.29873 | +4.00413 |

`q8_0` is closer to BF16 than FBLA (KL 0.0016 against 0.0020 .. 0.0026), at 8.5 bits per element against 6.25;
KIVI-4 (5 bits) has about twice FBLA's KL and a heavier tail.

The scripts that produced the two kinds of measurement are [`bench/fbla_lm.py`](../bench/fbla_lm.py) and
[`bench/fbla_token_metrics.py`](../bench/fbla_token_metrics.py). They run from a model directory; without text
files they use the WikiText-2 test split (needs `pyarrow` and network once; `huggingface_hub` caches it).
Optional `--baseline-losses` / `--bf16-hidden-cache` files let later formats reuse the exact same baseline, but
model, QKV and hidden-state dumps are local inputs or caches: none are distributed in this repository.

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

Against flash-attn 2.8.3 on an L40S (sm_89, `python bench/fbla_vs_sdpa.py --decode 16384,65536,131072,262144,1048576`:
random K / V, so more blocks are refined than at Qwen3's rates, default `tau`; 8 KV heads, 16 q heads; the L2 is
flushed before each call and both sides replay from a CUDA graph; median of 50 calls after 200 ms idle, 2520 MHz;
back to back the times are within 0.3 % and the clock stays at 2520 MHz). FA2 reads the BF16 K / V
of the whole context, FBLA 6.25 bits per element:

| context | FA2 | FBLA 3+3 | FBLA 4+2 | speed-up 3+3 / 4+2 |
|---:|---:|---:|---:|---:|
| 16K | 137.7 us | 70.4 us | 64.8 us | 1.95× / 2.12× |
| 64K | 470.6 us | 164.3 us | 150.6 us | 2.86× / 3.12× |
| 128K | 832.3 us | 222.2 us | 282.3 us | 3.75× / 2.95× |
| 256K | 1.550 ms | 407.0 us | 496.0 us | 3.81× / 3.13× |
| 1M | 5.852 ms | 1.302 ms | 1.663 ms | 4.49× / 3.52× |

Without flash-attn 2 the script falls back to PyTorch's flash backend (sm_80 and newer), else to its
memory-efficient FP16 backend with K / V repeated to every query head. On a GTX 1080 against PyTorch
(an earlier version of the script; this table has not been re-run with the current one):

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
either mode, PyTorch FP32 SDPA (memory-efficient kernel) 88 ms, cuBLAS FP32 `softmax(Q K^T) V` 200 ms
(GTX 1080, DP4A).

On tensor cores (sm_75 .. sm_89) against flash-attn 2.8.3 BF16 on an L40S (`python bench/fbla_vs_sdpa.py
--decode '' --prefill`): a chunk of queries appended to a cache of `context` tokens, causal over all of them,
8 KV heads, 4+2 (3+3 gives the same output). FA2 reads the BF16 K / V; FBLA is the whole
`mlops.attention.prefill` call (close, transcode, attention, finish) with persistent images. Median of 50
calls: idle (200 ms before each call, 2520 MHz) and warm (back to back after a second of load, where both cards
sit at the 300 W limit: FA2 at about 1.2 .. 1.35 GHz, FBLA at 1.7 .. 1.9 GHz).

| GQA | context | chunk | FA2 idle | FBLA idle | speed-up | FA2 warm | FBLA warm | speed-up |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 64K | 2048 | 5.151 ms | 3.324 ms | 1.55× | 7.912 ms | 4.499 ms | 1.76× |
| 1 | 64K | 2048 | 2.672 ms | 1.785 ms | 1.50× | 4.107 ms | 2.206 ms | 1.86× |
| 4 | 64K | 2048 | 10.03 ms | 6.376 ms | 1.57× | 15.88 ms | 8.686 ms | 1.83× |
| 8 | 64K | 2048 | 18.85 ms | 12.48 ms | 1.51× | 31.56 ms | 17.66 ms | 1.79× |
| 2 | 0 | 2048 | 0.330 ms | 0.316 ms | 1.04× | 0.186 ms | 0.220 ms | 0.85× |
| 2 | 16K | 2048 | 1.521 ms | 1.063 ms | 1.43× | 2.072 ms | 1.213 ms | 1.71× |
| 2 | 64K | 512 | 1.466 ms | 1.152 ms | 1.27× | 2.100 ms | 1.574 ms | 1.33× |
| 2 | 64K | 8192 | 19.79 ms | 13.10 ms | 1.51× | 32.69 ms | 19.22 ms | 1.70× |
| 2 | 256K | 2048 | 19.77 ms | 12.27 ms | 1.61× | 32.75 ms | 17.73 ms | 1.85× |

With an empty cache there is nothing to read from the cache, and the fixed cost of the close kernel and the
extra launches shows (0.85× warm).

### A whole model

[`samples/04_generate.py`](../samples/04_generate.py) and [`bench/llama_bench.py`](../bench/llama_bench.py) run
Qwen3-8B-NVFP4 end to end on a GTX 1080: 778 tokens/s prefill and 35.4 tokens/s decode at an empty context,
361 / 29.5 at 16K; [llamacpp-compare.md](llamacpp-compare.md).
