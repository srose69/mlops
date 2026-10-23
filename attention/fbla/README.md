# mlops/attention/fbla

FBLA (Flash Block Attention) is a compressed-KV FlashAttention decode and prefill for head dimension 128. Closed
128-token groups use a 6-bit per-channel affine code split into coarse and fine planes; in decode the fine K/V
record of a 16-token block is read only when its normalized coarse-softmax mass is large enough, prefill reads
full codes ([Prefill](#prefill)). Installation, PyTorch API and model-quality results:
[the package README](../README.md). The normative physical layout and C contract are in
[`../include/mlops_fbla.h`](../include/mlops_fbla.h).

## Format

For one K or V channel in a group, FBLA stores BF16 `base` and `step` and a six-bit code `c`:

```
x = base + step c,     c in [0, 63]
```

`base` is the channel minimum rounded toward `-inf`; `step = (max - base) / 63` is rounded toward `+inf`.
The code is split as `c = 2^FB h + f`:

| mode | coarse `h` | fine `f` | coarse reconstruction | default threshold |
|---|---:|---:|---|---:|
| 3+3 | 3 bits | 3 bits | `base + step (8h + 3.5)` | `W_j > 3e-4` |
| 4+2 | 4 bits | 2 bits | `base + step (4h + 1.5)` | `W_j > 1e-3` |

The first group is an exact BF16 attention sink. The last group is also BF16 while it is being appended. A
closed group's metadata costs 0.25 bits per K/V element: BF16 base and step for K and V, per channel and 128
tokens.

For a 16-token block `j`, the coarse scores define

```
LSE_j = log2 sum_t 2^s_jt
R     = log2 sum_j 2^LSE_j
W_j   = 2^(LSE_j - R).
```

The fine plane is selected when `W_j > tau`. Selection is performed in the log2 domain, `LSE_j - R >
log2(tau)`, so it needs no exponential.

## Three kernels

### Coarse

The coarse pass is a complete FlashAttention pass, not a score-only prepass. Each warp keeps an online
`(m, l, o)` state and consumes both K and V of every coarse record once. It writes:

* `LSE_j` for every closed 16-token block;
* one `(m, l, o)` partial per persistent CTA and query.

The work unit is either one exact BF16 sub-block or one complete closed group. Persistent CTAs are divided
over `(batch, KV head)` pairs from the kernel's measured occupancy and the device's SM count. Warp `w` takes
units `w, w + P·8, ...`; neighbouring warps therefore visit the same context window while every warp gets the
same amount of work to within one unit. There is no fixed 512- or 2048-token split.

Inside a closed group, one warp handles two 16-token records at a time:

* lane `l` holds token `l mod 16` of one of the two blocks and all 128 K channels, so QK needs no 32-lane sum;
* lane `l` holds four V channels across 16 tokens;
* the next pair's vector loads are issued immediately after the current registers' last use;
* P8 bytes are written by the producing lanes and returned to every lane as one shared `LDS.128`.

### Select

One CTA scans a slice of 1,024 sub-blocks. A warp first reduces the coarse partials to the row LSE `R`, then
loads its whole share of block LSEs before consuming them. Ballots and prefix counts compact selected blocks
in context order into

```
item = subblock_index << 4 | query_mask.
```

The query mask unions the GQA queries: if several queries select a record, refinement reads it once. Separate
slice counts make the item order, and therefore the summation order, independent of CTA timing.

### Refine

All refinement warps of a KV head stride over the concatenated item list. For one item they read coarse and
fine K/V plus the group's metadata, recompute the exact same Q8 codes as the coarse pass, and accumulate only

```
delta_l = l_full - l_coarse
delta_o = o_full - o_coarse.
```

The next item's records and metadata are fetched before the current item is evaluated. Corrections are reduced
per CTA. The last CTA, detected by a per-head counter after `__threadfence`, combines both passes and writes
BF16 output; a separate fourth launch is unnecessary.

## Prefill

`mlops_fbla_prefill` appends a chunk of tokens and attends its queries, causally, over the whole cache. It runs
in three launches. `close_kernel` quantizes the groups the chunk closes, writing the encoder's bytes bit for
bit. `stream_kernel` computes the attention. `finish_kernel` updates the sink and the open group.

Prefill is compute-bound. A DP4A over 6-bit codes costs what one over the coarse plane does, so prefill reads
every closed group through its full codes `c 2^FB + f`. This is the decode with every block refined, and
the output does not depend on the mode. One FlashAttention pass then needs no selection, no second pass and
no scratch.

`stream_kernel` gives each CTA 8 query rows per consumer warp, as GEMM tiles. It first processes the BF16
groups (the sink and the row's own group), then streams the closed groups newest first through a two-stage
shared-memory ring:

* Producer warps turn the coarse and fine records into swizzled tiles of full codes, so every fragment is one
  conflict-free `LDS.128`. Each half group (64 tokens) gets K `[token][channel word]` and V
  `[token word][channel]`.
* Consumer warps quantize q per group (the decode's q-hat), then compute QK and PV by DP4A, with 4 rows × 4
  tokens and 4 rows × 8 channels per lane. They use an online softmax whose maximum moves only when a
  sub-block exceeds it by 2^8.

The two sides meet only at named barriers. The block is 8 warps: 4+2 uses 7 consumers and 1 producer, while
3+3 uses 6 consumers and 2 producers, because unpacking the 3-bit planes needs the second producer warp.

## Integer arithmetic

The record is the DP4A operand. Codes are not unpacked into an int8 tile and are not shifted down merely to
make them look conventional.

* QK uses inline `dp4a.s32.u32`: signed Q8 times unsigned packed codes.
* PV uses `dp4a.u32.u32`: unsigned P8 times unsigned packed codes.
* Slots whose codes remain multiplied by `2^n` accumulate separately; `2^-n` is folded into the FP32
  epilogue, where it is an exact power-of-two multiply.
* Every integer accumulator starts at the FP32 bits of `1.5 · 2^23`. Sums remain within `±2^22`, so
  reinterpret-as-float minus `1.5 · 2^23` is the exact integer result without `I2F`.
* `p = round(255 · 2^(s-m))` uses `MUFU.EX2`, an FFMA with the same magic constant and the result's low byte;
  no float-to-int conversion is issued.
* The online-softmax update uses one `2^-|delta m|`: depending on the sign, either the old state or the new
  block has weight one.
* Q8's reciprocal is `MUFU.RCP` followed by Markstein refinement. Rounding to int8 again uses the magic
  constant; four bytes are assembled with `PRMT`.

These are the same accumulator, conversion and software-pipeline patterns used by NVFP4EffMA's DP4A kernels.

## Record layout

One record contains K and V for 16 tokens. Its size is `2 · bits · 256` bytes: 1,536 bytes for a 3-bit plane,
2,048 for 4-bit, 1,024 for 2-bit. A warp-wide access is always a contiguous 256- or 512-byte segment made of
128-bit lane loads.

| operand | lane owns | DP4A word |
|---|---|---|
| K | token `lane % 16`, channel half `lane / 16` | four adjacent channels of that token |
| V | channels `4 lane .. 4 lane + 3` | four adjacent tokens of one channel |

Four-bit and two-bit slots are ordinary per-byte nibbles/pairs. The 3-bit record is deliberately non-linear:
the first twelve slots occupy two three-bit fields of six words; the remaining four split across the free top
bits. K's last two words are transposed so both channel halves of one token arrive in one `LDG.128`. Every hot
slot is then one mask, except the four split slots.

The exact byte offsets, V slot permutations, metadata and capacity strides are documented in
[`../include/mlops_fbla.h`](../include/mlops_fbla.h). Both the C test and Python test reconstruct the layout
independently from that text before comparing any attention result.

## GQA and storage

One CTA belongs to a KV head, not a query head. The same record is reused for GQA ratios 1, 2 and 4.

* GQA 1/2 keeps per-query online state in registers. Shared memory holds Q8 words, V affine values, P8 bytes
  and the final cross-warp merge; the query buffer aliases the merge buffer because their lifetimes do not
  overlap.
* GQA 4 has separate coarse/refine kernels. They process two queries at a time over the same code registers;
  query-indexed `(m,l,o)` and group state live in warp-local shared slots, which keeps two CTAs resident without
  spilling.
* Special-register values used once per group are reread through volatile PTX instead of being kept live
  across the loop. Streamed record loads are volatile so ptxas cannot sink the prefetch past the computation it
  is meant to overlap.

Shared memory and register count are consequences of reuse and occupancy, not targets by themselves. Launch
geometry calls `cudaOccupancyMaxActiveBlocksPerMultiprocessor` once per kernel/device, caches the result, and
creates one persistent wave subject to available work.

## Correctness

`tests/test_fbla.cu` is a float64 implementation of the definition, written independently of the device
helpers. It checks both formats at GQA 1/2/4, batch 2, partial open groups, an 8K-capacity cache decoded at 3K,
default/no/all refinement and contexts through 256K. Every case runs twice against the same zero-initialized
scratch and must reproduce its BF16 output bit for bit. The current sm_61 suite passes with relative RMS error
below `2.1e-3` and max absolute error below `3.9e-3`; the difference is P8 and approximate SFU arithmetic.

The Python tests independently unpack every code of every record, compare the kernel with a float64 model and
check a capacity-padded cache against a tightly laid-out one.

## Performance

GTX 1080, sm_61, 20 SMs, 320 GB/s nominal DRAM peak. `make bench` warms GPU Boost, measures a 256 MiB D2D
copy, idles for one second between rows and samples each row's SM clock through NVML. Repeat counts are chosen
per format/geometry/context for about one second of timed work on that GPU; this keeps both short and long
contexts under sustained load without making the 1M scenarios dominate the run. `MLOPS_BENCH_REPS`,
`MLOPS_BENCH_WARMUP` and `MLOPS_BENCH_COOL` override the defaults. One cache is allocated for 1M tokens and
decoded at each shorter length without repacking. Traffic includes exact groups, metadata, the coarse pass and
both planes reread by selected corrections.

Synthetic scores are thresholded to the Qwen3-0.6B per-query rates: 12.3% for 3+3 and 4.8% for 4+2. The table's
rate is the union of blocks selected by any GQA query, whose records are read once.

Controlled GTX 1080 run, Qwen geometry (8 KV heads, GQA 2), D2D-copy baseline 282 GB/s:

| context | 3+3 latency | effective GB/s | 4+2 latency | effective GB/s |
|---:|---:|---:|---:|---:|
| 16K | 104.4 us | 192.9 | 102.0 us | 204.7 |
| 64K | 318.5 us | 246.4 | 319.5 us | 254.9 |
| 128K | 598.0 us | 261.1 | 605.2 us | 268.1 |
| 256K | 1.159 ms | 268.9 | 1.179 ms | 274.6 |
| 512K | 2.506 ms | 248.5 | 2.553 ms | 253.2 |
| 1M | 5.197 ms | 239.4 | 5.184 ms | 249.4 |

The package README keeps the Qwen-geometry summary. Full tables and run conditions are in
[`../benches/table.md`](../benches/table.md), with one linked file per GPU; benchmark methodology and the
controlled runner are documented in [`../docs/performance.md`](../docs/performance.md). Single-KV-head rows
expose launch/parallelism costs and the GQA=4 shared-state specialization; they are not expected to saturate
DRAM at short contexts.

## Code

| file | |
|---|---|
| `../include/mlops_fbla.h` | C API and normative cache layout |
| `src/fbla_kernels.cuh` | coarse, select and refinement kernels; DP4A/layout primitives |
| `src/fbla_prefill.cuh` | prefill: group closing, the streaming attention kernel, sink / open update |
| `src/fbla.cu` | validation, scratch geometry and format dispatch |
| `src/fbla_inst.cu` | separate 3+3 and 4+2 instances |
| `tests/test_fbla.cu` | float64 definition, independent packer, full CUDA test matrix |
| `bench/bench_fbla.cu` | 16K .. 1M native benchmark and physical-traffic accounting |
| `../python/mlops/attention/fbla.py` | reference encoder and PyTorch-facing cache object |
