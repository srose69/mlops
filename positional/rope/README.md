# mlops/positional/rope

Rotary position embedding (RoPE; Su et al., RoFormer) of q and k in one pass, for training and inference, on
every GPU sm_50 .. sm_90. Installation and the PyTorch API: [the package README](../README.md).

## The rotation

Pair `i` of a head (`i < rotary_dim / 2`) is rotated by the angle `a = m · θ_i` of its token's position `m`:

```
y1 = x1 cos a − x2 sin a            y2 = x2 cos a + x1 sin a
```

* **Pairs.** `(x[i], x[i + rotary_dim/2])` (half split: NeoX, Llama, Qwen, Mistral, …) or `(x[2i], x[2i+1])`
  (`interleaved`: GPT-J and its descendants). Elements from `rotary_dim` to `head_dim` pass through (partial
  rotary: NeoX, Phi, StableLM).
* **Relative position.** The rotation is orthogonal, so in `q_m · k_n` the angles subtract: attention sees `m − n` only.
* **Frequencies.** `θ_i = base^(−2i / rotary_dim)` (base 10 000, 500 000 for Llama 3, …), or a scaled set:
  YaRN, Llama 3, linear, dynamic NTK, LongRoPE. All of them are a vector `inv_freq` plus an attention factor
  (YaRN's mscale, LongRoPE's factor) that multiplies cos and sin. `rope_frequencies` computes them as
  transformers does, in float64.
* **Backward.** The gradient of a rotation is the rotation by `−a`: the backward is the same kernel.

## Exact angles

transformers computes `a = m · θ_i` as one FP32 product, then `cos` / `sin`, then rounds them to the model's dtype.
At position 100 000 the FP32 angle is already off by up to ~4·10⁻³ rad; near 2³¹ by ~100 rad.

mlops_rope reduces the **phase in turns**, `t = frac(m · θ_i / 2π)`:

* `f = θ_i / 2π` is held as a sum of two floats (`mlops_rope_freq`, from the float64 frequencies).
* The position is split, `m = 2¹⁶ a + b`, with `g = frac(2¹⁶ f)` precomputed, so every partial product has an
  integer factor below 2¹⁶.
* The rounding error of each product is recovered exactly with an FMA, and integer parts are removed exactly
  before the small terms join.

The result is within a few 2⁻²⁴ turns for every `|m| < 2³¹`. `sincospif` then gives cos and sin to an FP32 ulp.
They are computed once per token and frequency and shared by all heads of q and k, so the special-function
work is negligible. Each output is computed in FP32 (one FMA) and rounded once to the dtype.

Measured against a long-double reference (`make test`), every output of every dtype is within half an ulp of the
dtype plus the FP32 computation error, at positions up to 2³¹ − 1 and negative ones. The tests compare the
pipelines against the exact rotation: `RoPE.from_hf(config)` is never further from it than transformers' own
pipeline.

### At long positions

[`bench/bench_rope_accuracy.py`](../bench/bench_rope_accuracy.py) compares three implementations with the exact
rotation: float64 angles from the exact frequencies and float64 products. Setup: head_dim 128, base 500 000,
64 consecutive positions from each start × 8 heads. The error is per pair, relative to its magnitude, RMS / max.

* **The other two.** transformers and torchtune are computed with their own arithmetic, written out in the
  script, so it runs at any position. torchtune's cache alone would take 5 GB at 10M positions.
* **Checked against the real libraries.** The copies reproduce the libraries' bits: transformers 5.13 at every
  start, torchtune 0.6.1 at the starts its cache fits (1M by default).
* **The floor.** The first column is the exact result rounded once to the dtype.

BF16 q:

| position | exact, rounded once | transformers | torchtune | mlops |
|---|---|---|---|---|
| 128K | 1.7e-3 / 3.8e-3 | 2.9e-3 / 1.3e-2 | 2.0e-3 / 1.1e-2 | 1.7e-3 / 3.8e-3 |
| 512K | 1.7e-3 / 3.9e-3 | 5.2e-3 / 3.9e-2 | 4.7e-3 / 3.9e-2 | 1.7e-3 / 3.9e-3 |
| 1M | 1.7e-3 / 3.8e-3 | 9.1e-3 / 7.4e-2 | 8.8e-3 / 7.4e-2 | 1.7e-3 / 3.8e-3 |
| 2M | 1.6e-3 / 3.8e-3 | 1.8e-2 / 1.5e-1 | 1.8e-2 / 1.5e-1 | 1.6e-3 / 3.8e-3 |
| 4M | 1.7e-3 / 3.8e-3 | 3.6e-2 / 3.0e-1 | 3.5e-2 / 3.0e-1 | 1.7e-3 / 3.8e-3 |
| 6M | 1.7e-3 / 3.8e-3 | 5.3e-2 / 3.8e-1 | 5.3e-2 / 3.8e-1 | 1.7e-3 / 3.8e-3 |
| 8M | 1.7e-3 / 3.9e-3 | 7.0e-2 / 5.9e-1 | 7.0e-2 / 5.9e-1 | 1.7e-3 / 3.9e-3 |
| 10M | 1.7e-3 / 3.8e-3 | 9.0e-2 / 6.7e-1 | 9.0e-2 / 6.7e-1 | 1.7e-3 / 3.8e-3 |

FP32 q:

| position | exact, rounded once | transformers | torchtune | mlops |
|---|---|---|---|---|
| 128K | 2.5e-8 / 5.8e-8 | 1.1e-3 / 9.2e-3 | 1.1e-3 / 9.2e-3 | 8.1e-8 / 3.9e-7 |
| 1M | 2.5e-8 / 5.9e-8 | 8.7e-3 / 7.3e-2 | 8.7e-3 / 7.3e-2 | 8.1e-8 / 3.5e-7 |
| 4M | 2.5e-8 / 5.8e-8 | 3.5e-2 / 3.0e-1 | 3.5e-2 / 3.0e-1 | 8.1e-8 / 3.9e-7 |
| 10M | 2.5e-8 / 5.9e-8 | 9.0e-2 / 6.7e-1 | 9.0e-2 / 6.7e-1 | 8.1e-8 / 4.2e-7 |

(every start from 128K to 10M, and both pair layouts of mlops, in the script's output.)

FP32 does not save transformers and torchtune: at 128K their error is already 1.1e-3, 44 000× the float32
floor. **As a position.** A frequency's angle error divided by the frequency is where that frequency places the
token. With FP32 q, in tokens: the shift max / RMS, and the largest error of the distance between two of the 64
positions:

| position | exact, rounded once | transformers | torchtune | mlops |
|---|---|---|---|---|
| 128K | ±0.018 / 0.001, 0.031 | ±0.035 / 0.006, 0.050 | ±0.034 / 0.006, 0.055 | ±0.030 / 0.003, 0.045 |
| 1M | ±0.015 / 0.001, 0.027 | ±0.13 / 0.048, 0.13 | ±0.13 / 0.048, 0.13 | ±0.087 / 0.007, 0.115 |
| 4M | ±0.016 / 0.001, 0.029 | ±0.54 / 0.19, 0.49 | ±0.54 / 0.19, 0.49 | ±0.073 / 0.005, 0.130 |
| 10M | ±0.018 / 0.001, 0.032 | **±1.38 / 0.48, 1.20** | **±1.38 / 0.48, 1.20** | ±0.047 / 0.004, 0.066 |

At 10M every token is placed up to 1.4 positions off, and the distance between two tokens is wrong by up to
1.2 positions. The pairs that tell neighbouring tokens apart (periods of 7-12 tokens) are turned up to
0.69 rad (39°) the wrong way. The error grows linearly with the position; above 2^24 = 16.7M FP32 cannot even hold
the position exactly, and at 32M the shift reaches ±6 tokens. mlops stays at a few hundredths of a token at every position: the float32 floor of the lowest
frequencies.

With the context-extension recipes (`--scaling all`; YaRN factor 4 from 32K, Llama 3.1, linear 4, dynamic NTK 4,
LongRoPE from 4K; transformers' copies again checked bit for bit against `LlamaRotaryEmbedding` at every start,
with its frequencies computed where it computes them: on the CPU for the static recipes, as float32 tensor
arithmetic on the GPU for dynamic NTK and LongRoPE), at 10M positions, RMS / max:

| scaling | BF16: floor | BF16: transformers | BF16: mlops | FP32: transformers | FP32: mlops |
|---|---|---|---|---|---|
| none | 1.7e-3 / 3.8e-3 | 9.0e-2 / 6.7e-1 | 1.7e-3 / 3.8e-3 | 9.0e-2 / 6.7e-1 | 8.1e-8 / 4.2e-7 |
| YaRN | 1.7e-3 / 3.8e-3 | 9.0e-2 / 6.7e-1 | 1.7e-3 / 3.8e-3 | 9.0e-2 / 6.7e-1 | 8.5e-8 / 4.2e-7 |
| Llama 3.1 | 1.7e-3 / 3.8e-3 | 9.0e-2 / 6.7e-1 | 1.7e-3 / 3.8e-3 | 9.0e-2 / 6.7e-1 | 8.2e-8 / 4.2e-7 |
| dynamic NTK | 1.7e-3 / 3.8e-3 | 7.4e-2 / 7.2e-1 | 1.7e-3 / 3.8e-3 | 7.4e-2 / 7.2e-1 | 7.3e-8 / 3.6e-7 |
| LongRoPE | 1.7e-3 / 3.8e-3 | 2.9e-2 / 3.7e-1 | 1.7e-3 / 3.8e-3 | 2.9e-2 / 3.7e-1 | 8.3e-8 / 3.7e-7 |
| linear 4 | 1.7e-3 / 3.8e-3 | 2.3e-2 / 1.7e-1 | 1.7e-3 / 3.8e-3 | 2.3e-2 / 1.7e-1 | 7.5e-8 / 3.7e-7 |

YaRN, Llama 3 and dynamic NTK leave the highest frequency (1 rad per position) unscaled, so transformers' FP32
angle error is the unscaled one. Linear and LongRoPE divide it, and its error with it. With every recipe, mlops
stays at the floor in BF16 and at its constant few FP32 ulps.

* **transformers and torchtune.** Their error grows linearly with the position. Both take the angle as one FP32
  product, whose rounding is half an ulp of the angle: up to 0.5 rad at 10M positions, where the RMS error is
  9 % and the worst pair is rotated ~0.7 rad off. In BF16, transformers additionally rounds cos / sin to BF16,
  which is why it is worse than torchtune at short positions.
* **mlops in BF16.** It sits on the floor at every position: the result is the exact rotation rounded once.
* **mlops in FP32.** The error is constant over the whole range, about 3× the floor in RMS. This is the FP32
  arithmetic of the step itself: the phase within ~2⁻²⁴ turns, `sincospi`, two FMAs.

## Hugging Face's arithmetic

With cos / sin tables (`apply_rotary_pos_emb`, `mlops_rope_config.cos / sin`) the kernel reproduces
`q * cos + rotate_half(q) * sin` exactly, every product and sum rounded to the dtype as PyTorch does. Its
autograd backward is reproduced too (`inverse`, with the sine indices of the two outputs swapped), bit for bit
against transformers 5.13 in every dtype and both layouts. `patch_transformers("llama")` leaves a Llama model's
output unchanged to the last bit.

## Kernel

* **Blocks.** One thread block per token. The block first writes the factors `(cos₁, sin₁, cos₂, sin₂)` of every
  pair to shared memory: computed from the position, or loaded from the tables. Then it rotates every head of q
  and k with them.
* **Vectorized access.** Each thread handles V = 4, 2 or 1 pairs per access: one vector load and store per half
  (half split), or one of 2V elements (interleaved). V is the widest the alignment of the tensors allows.
* **Layouts.** `[B, S, H, D]` with any strides and a contiguous last dimension. This covers transformers'
  `[B, H, S, D]`, packed qkv projections and `[T, H, D]`.
* **Compiled variants.** The dtype, V, the pair layout and the arithmetic are template parameters, one object
  per dtype. The angle source and the direction only change how shared memory is filled, never the rotation loop.

## Performance

GTX 1080 (320 GB/s DRAM peak), Llama-like heads (32 q, 8 kv, head_dim 128), BF16, transformers' `[B, H, S, D]`,
llama3 scaling, median of 200 calls (`make bench`):

| | transformers (rotary module + `apply_rotary_pos_emb`) | drop-in `apply_rotary_pos_emb` (same tables) | `RoPE` from positions |
|---|---|---|---|
| prefill, 1 × 4096 tokens | 2652 µs | 458 µs (5.8×) | 324 µs (8.2×, 259 GB/s) |
| prefill, 8 × 512 | 2587 µs | 469 µs (5.5×) | 325 µs (8.0×) |
| decode, 1 token | 92 µs | 77 µs | 22 µs (4.2×) |
| decode, 64 × 1 token | 107 µs | 79 µs | 23 µs (4.7×) |
| training step 4 × 2048, forward + backward | 11.9 ms | 2.43 ms (4.9×) | 2.14 ms (5.5×) |

From positions the kernel reads and writes q and k once, at 81% of the DRAM peak. The drop-in also saves
transformers' four to five elementwise passes, but the tables are still built by the rotary module.

In decode almost all of the remaining time is launch and Python overhead. Removing it needs fusion: RoPE on
the input of attention, or RoPE plus the KV-cache write. That is the purpose of the header-only device code in
`src/rope.cuh`.

## Code

| file | |
|---|---|
| `../include/mlops_rope.h` | C API, the exact arithmetic of both modes |
| `src/rope.cuh` | device code for fused kernels: phase in turns, sincospi, dtype codecs, the rotation in both arithmetics |
| `src/rope_kernels.cuh`, `src/rope_inst.cu` | the kernel and its instances (one object per dtype) |
| `src/rope.cu` | host side: argument checks, vector width, `mlops_rope_freq` |
| `tests/test_rope.cu` | `make test` (in `positional/`) |
