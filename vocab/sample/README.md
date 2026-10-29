# mlops/vocab/sample

Token sampling without sorting the vocabulary: greedy, temperature, top-k, top-p and a repetition penalty, on every GPU
sm_50 .. sm_90. Installation and the PyTorch API: [the package README](../README.md).

## The sampler

**The sampler is exact.** Masses `exp(l - max)` are counted in fixed point (units of 2^-40), so sums are integers in any
order: a token depends on (logits, configuration, seed, step) alone, not on the grid or the device. The top-k / top-p
threshold is found by four radix passes over the order keys, the draw is one Philox4x32-10 block. The radix histograms
add into 256 counters in global memory, the buckets are two 20-bit limbs added with 32-bit shared atomics, and the closing
steps of a pass reduce across the CTA's threads.

**The head's statistics replace the sampler's scan.** With `logit_scale = 1 / temperature` (1 for greedy) and penalty 1,
the head's per-element maxima are enough for the sampler: greedy needs no pass over the logits at all. The head also counts
the top byte of the order key of its rows, which is the first radix pass of a top-k; top-p's pass weighs the rows by
`exp(l - max)` and needs the global maximum first, so it stays a pass.

## Measurements

The sampler per call at 151936 tokens, scanning the logits, with the head's statistics, and with the head's key-byte counts
too (a top-k's first radix pass is then a sum over 64 histograms):

| | passes | 1080 scan | 1080 stats | 1080 + counts | L40S scan | L40S stats | L40S + counts |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy | 1 | 4.6 µs | 2.0 µs | - | 3.6 µs | 2.7 µs | - |
| temperature | 2 | 19.3 µs | 10.7 µs | - | 10.1 µs | 8.6 µs | - |
| top-k 20, top-p 0.95 | 7 | 66.0 µs | 60.1 µs | 55.6 µs | 36.9 µs | 35.1 µs | 34.6 µs |
| top-p 0.9 | 7 | 66.1 µs | 66.1 µs | - | 38.9 µs | 37.4 µs | - |
| top-k 2000 | 6 | 52.1 µs | 44.3 µs | 40.5 µs | 29.5 µs | 27.9 µs | 27.5 µs |

A pass costs 7 - 12 µs on the 1080 and 3 - 5 µs on the L40S, nearly all of it the pass's fixed work (the last-arrival
fence, the closing step) rather than the logits.
