# mlops/vocab/lm_head

The BF16 LM head for 1 .. 8 tokens, on every GPU sm_50 .. sm_90: `logits = s * logit_scale * (W x)` with the deferred RMSNorm
scalar `s` of the producer, and each element's maximum and key-byte counts for the sampler. Installation and the PyTorch API:
[the package README](../README.md).

## The head

**The head is cut along hidden.** The S warps of a CTA take disjoint ranges of 256-column passes of the same rows, each
keeps its piece of `x`, and the S partial sums of a row are added in warp order in shared memory. No atomics, no global
partials, a fixed order: the result does not depend on how many elements the rows are split over. The loads of a step are
issued D steps ahead in a register ring, across units. On CUDA cores the instance carries NB = 1, 2, 4 or 8 columns.

**Every architecture has a path.** CUDA cores on all of sm_50 .. sm_90 (at n <= 8 the head is bound by memory, not by the
FMAs), and tensor cores from Volta:

| architecture | tensor-core instance | operands |
|---|---|---|
| sm_70, sm_72 | `mma.m8n8k4` (four independent 8 x 8 x 4 products per warp: a lane holds the 4 k of one row of W and of one token's x, so a warp takes 32 rows) | FP16, converted from BF16 in registers |
| sm_75 | `mma.m16n8k8` | FP16, converted the same way |
| sm_80, 86, 87, 89, 90 | `mma.m16n8k16`, W the A operand, a chunk order of k that makes every load a whole 16-byte piece of one row | BF16 |

The FP16 instances keep the mantissa (FP16 has more bits than BF16) and take FP16's range: |x| must stay below 65504, and a
weight below 2^-14 loses at most 2^-25 of absolute precision per unit of x. All of the tensor-core instances run on an L40S
(`m8n8k4` at a lower rate there: 401 GB/s against 750), which is where their layouts are checked; the Volta and Turing ones
have not been timed on their own GPUs. Every instance is offered as a variant on a device that can run it, and the caller's
measurement picks.

**Results stay in L2.** The head writes logits and statistics with `st.global.cg` and reads the weights with streaming
loads, so the 1.2 GB of weights do not push the results out: on an L40S a scan of the logits right after the head costs
what a repeated one does (3.4 µs against 4.0 µs for n = 1). On a GTX 1080 (2 MB of L2) they fit for n = 1 and not for
n = 8. A call that needs only a token never writes them: `want_logits` off leaves the statistics.

**Variants are data.** `mlops_lm_head_variants` lists the instances the device accepts for a shape, without local memory:
R rows a unit, S warps, D steps in flight, NB columns, tensor cores or not, J pieces, with threads, shared memory and
registers from `cudaFuncGetAttributes`. The call and the items run the one the caller names; choosing is the caller's (an
executor measures on its machine). The element count is the caller's too (`items`, 0: the resident CTAs), and the rows are
split over it in units, balanced to one unit.

## Measurements

GTX 1080 (320 GB/s DRAM peak, the display on the same card) and L40S (864 GB/s), the head at 151936 x 4096 BF16, `make bench`:

| | read | of DRAM peak |
|---|---:|---:|
| L40S, the best variants (n = 1 on CUDA cores, n = 8 on tensor cores) | 750 GB/s, 1.66 ms | 98 % (peak 768 GB/s as the card reports it) |
| GTX 1080, the best variants | 250 - 255 GB/s, 4.9 - 5.0 ms | simple streaming reads reach 254 GB/s |
