# mlops/optimizers/adamw

The two optimizers of [mlops-optimizers](..) (`mlops.optim` in Python), for every GPU NVFP4EffMA supports:
sm_50 .. sm_90, the ones without native FP4. This page has the math, the formats and the measurements;
installation, the PyTorch API and the build are in [the package README](../README.md).

| | weights | state per element | step |
|---|---|---|---|
| `mlops_adamw` | BF16 / FP16 / FP32 | 8 B (FP32 `m`, `v`) | AdamW |
| `mlops_adamw_qat` | NVFP4, trained in place | 5 B (+ 8 B per 16-element block) | AdamW on the NVFP4 lattice, no master copy |

`mlops_adamw_qat` builds on [NVFP4EffMA](https://github.com/srose69/nvfp4effma) (NVFP4 tensors, E2M1 / E4M3
codecs, the initial quantization) and, in the device code below sm_89,
[E5M2EffMA](https://github.com/srose69/e5m2effma) (FP8 codecs); from sm_89 on, where FP8 is native, it uses the
hardware `cvt` instead ([FP8 codecs](#fp8-codecs)). A PyTorch reference of its step, bit-identical to the
kernel, is in [`mlops/optim/reference.py`](../python/mlops/optim/reference.py).

## `mlops_adamw`

Elementwise AdamW (Loshchilov & Hutter) over raw device pointers, in FP32:

```
g = grad * grad_scale
m = b1 m + (1 - b1) g                     v = b2 v + (1 - b2) g^2
p = p - lr wd p - lr (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)
```

Parameters and gradients BF16, FP16 or FP32 (any pair), moments FP32, 128-bit accesses; bit-reproducible
against the host emulation in `tests/test_adamw.cu`. It does not touch NVFP4: QAT with it means an FP32
master, this step, and a fresh `nvfp4a_quantize_ex` of the master afterwards (the `adamw` baseline in
`misc/`).

## `mlops_adamw_qat`

### Why

With an FP32 master the master moves in steps far below the NVFP4 cell it sits in; the forward sees
nothing until the master crosses a rounding boundary, and the 4 bytes that resolve the master below the
cell are bytes the weight never uses. Storing the master *on* the lattice, with just enough bits for
sub-cell progress, and keeping AdamW's moments in 8 bits, halves the memory of NVFP4 training. FP16 for
the state is ruled out (`v` spans more binades than FP16 has); the formats below are chosen per quantity
from its range and the precision the step needs.

The other half of NVFP4 QAT is the block scale. Re-deriving it from the amax every step makes the lattice
jump under the elements; freezing it loses the resolution a block gains when its weights settle. Here a
scale moves only when the block's own statistics ask for it, persistently, and a move never changes a
weight value.

### The lattice

E2M1's magnitudes `{0, .5, 1, 1.5, 2, 3, 4, 6}` are anchors at the integer coordinates 0..7; joining
them with secants gives a piecewise-linear `g(u)` with slope `g' = 0.5 / 1 / 2` (the 4-bit code read as a
continuous integer, Quake's float-bits-as-int at 4 bits). Element `i` of block `b`:

```
w_i = sigma_b g(u_i)            master (never stored as a float)
q_i = sigma_b g(round(u_i))     what the forward reads: the NVFP4 nibble is round(u_i)
sigma_b = E4M3(k_b) gdec        k_b the block's scale byte, gdec the global decode factor
```

`u` is stored as Q3.12: twelve fractional bits below a cell, enough for a step of `lr / J` with
stochastic rounding to stay unbiased (an E2M1 value times its scale has at most 6 significant bits, so
the lattice never needs more than the integer part plus sub-cell progress).

### Element step

AdamW on `q` in weight units, mapped onto `u` by the Jacobian `J = dw/du = sigma~ g'(u)`:

```
v    = b2 s^2 + (1 - b2) grad^2
y    = rsqrt(v + eps^2)
s   <- SR_E5M2(v y)                     E[s^2] = v
r   <- SR_E4M3((b1 r s + (1 - b1) grad) y)
u   <- SR_Q3.12(u - lr (r sqrt(1 - b2^t) / (1 - b1^t) + wd w) / J)
qbar <- SR_Q3.4(b1 qbar + (1 - b1) q / sigma)
```

* **`r = m / sqrt(v)` in E4M3.** Scale-free, and bounded: by Cauchy-Schwarz on the two EMAs,
  `|m| / sqrt(v) <= (1 - b1) / sqrt((1 - b2)(1 - b1^2 / b2)) = 7.27` for (0.9, 0.999), inside E4M3's range
  with its 3 mantissa bits where Adam's update needs them.
* **`s = sqrt(v)` in E5M2** with a power-of-two exponent per block (the block's largest `s` at `<= 2^12`).
  The stochastic rounding is done in the squared domain, so `E[s^2] = v` exactly and `v`'s EMA stays
  unbiased.
* **`y` is Quake's rsqrt**: magic `0x5F1FFFF9` plus one Newton step with refitted coefficients,
  `|rel. err| <= 6.5e-4`. `s` only seeds its rounding bracket from `v y`, so the approximation never
  biases `s`.
* **The moments never see `J`.** `g'` jumps by 2x at cell boundaries and `sigma` changes when the block
  scale moves; inside `v` those jumps would be remembered for `1 / (1 - b2)` steps. Applying `J` to the
  step instead makes this, to first order in `lr`, AdamW on `q`: `lr` and `weight_decay` mean what they
  mean for AdamW, and the decay acts on the master `w`.
* **`1 / sigma~`** is interpolated linearly between the centers of the element's block and its neighbour
  along the row, so the step multiplier has no seam at block boundaries.
* **Stochastic rounding** (SR) everywhere a value is stored: `E[stored] = exact`, so progress far below a
  quantum accumulates in expectation. The uniforms come from a counter-based stream (one lowbias32 mix
  per thread, xorshift steps after it).

### Block scale

Under QAT an element does not settle on its optimum `w*`: it oscillates between the two codes around it,
and its master sits at the rounding boundary between them. Neither `u` nor `q` locates `w*` inside the
cell; the mean of the oscillation does. With `H` the curvature, `m = H (qbar - w*)` (`qbar` the EMA of
`q` over the same window as `m`), so

```
w*_i = qbar_i - m_i / H_b
```

**`H_b` is a secant**, the regression of the gradient on the code moves `d = q - qbar(t-1)` over the
block: `E[grad | past] = H (q - w*)`, and the batch noise in `grad` is independent of `q`, which was fixed
before the batch was drawn:

```
H_b = EMA_b2[sum_i grad_i d_i] / EMA_b2[sum_i d_i^2]        two BF16 accumulators, SR
```

It measures the curvature along the directions the elements actually move (coupled moves included),
which is what the Newton estimate needs. (Centering `grad` on `m(t-1)` would bias it: `m(t-1)` drove the
moves.)

**`H_b` is used only when significant.** With `H = 0` the numerator is a sum of independent terms with
`E[grad^2] = v`, so its null variance is `(1 - b2)/(1 + b2) * (1 + b2^t)/(1 - b2^t) * mean(v) * EMA[sum d^2]`
(no extra state). A block votes only when the numerator is `z` of those standard deviations above zero;
otherwise `m / H` would be noise over noise.

**The preferred scale** minimizes the representation loss of `w*` over one E4M3 octave each way:

```
k* = argmin_{|k - k_b| <= 8} sum_i huber(w*_i - Q_k(w*_i), delta = sigma_b)
```

The window is two octaves because the loss is not convex in `k` (several local minima per octave: E2M1
is self-similar under x2 only between 1 and 6), and a local search stops in them. Huber keeps an outlier
from dragging the block scale by more than its linear share; `delta = sigma_b` makes every in-range
rounding error quadratic (no cell is wider than `2 sigma`), so only clipping is linearized.

**A CUSUM decides the move.** The vote `x = sign(k* - k_b)` enters a two-sided CUSUM with drift 1/16 per
step (one element's share of the block), threshold `h = 4 z^2`:

```
S <- sign(S + P x) max(0, |S + P x| - P / 16);    |S| > h  ->  k_b += sign(S), S = 0
```

The vote is made of `b1`-EMAs, so it is evaluated once per `P = (1 + b1) / (1 - b1)` steps (19 for
`b1 = 0.9`, the EMAs' integrated autocorrelation time) and held for those `P` steps. A move takes about
`h / (1 - 1/16) = 38` steps of a unanimous vote; a vote that stops recurring drains away.

**A move keeps every weight.** `u` and `qbar` are re-expressed against the new `sigma` (with SR); only the
representation changes. At a fixed `w*` the argmin never returns to a byte it left, so a block cannot
oscillate between two scales.

**Initialization** quantizes the dense initial weight with amax scaling (`nvfp4a_quantize_ex`), then
raises the global `2^H` and lowers every block byte by `8 H` codes (exact for normals): the same values, but
no block starts pinned at the top of its scale format. With E4M3 scales (`2^-9 .. 448`) `H = 3`: a block
scale can grow 8x, then it stays at 448 and the block's largest weights saturate. With UE5M3 scales
(unsigned, `2^-17 .. 114688`, `NVFP4A_SCALES_UE5M3`) `H = 8`: 256x of growth with 23 binades still below the
largest block. That is the room Graphcore's UE5M3 recipe ([arXiv:2609.02846](https://arxiv.org/abs/2609.02846))
gets from a tensor scale refreshed every 50 steps; here no amax is ever recomputed. While no E4M3 scale
would leave its range, both formats train bit-identically (same values, moments and codes; the scale bytes
differ by a constant). `make qat` checks the difference: when the weights grow 32x, every UE5M3 block follows
them exactly, while about a third of the E4M3 blocks stop at the top with 18% error.

### State

| | per element | per 16-element block |
|---|---|---|
| NVFP4 weight | 0.5 B code | 1 B scale |
| `u` | 2 B, Q3.12 | |
| `r`, `s` | 1 B each, E4M3 / E5M2 | 1 B exponent of `s` |
| `qbar` | 1 B, Q3.4 | |
| secant | | 2 x 2 B, BF16 |
| CUSUM | | 2 B, Q7.8 |
| scratch | | 1 B next scale byte (+ 4 B per row) |

**6.12 B per weight** in total, against 12.56 B for an FP32 master with FP32 moments plus the NVFP4 copy
the forward reads.

### Gradients

E5M2 or E4M3 with a per-tensor decode factor (`grad_scale`: the FP8 output scale of the backward GEMM),
or FP32. An FP8 gradient's rounding error, at most 2^-3 relative, is small next to the minibatch noise
Adam averages over; E5M2's 32 binades cover a tensor's gradient range with per-tensor scaling, which is
why Transformer Engine uses it for gradients.

### FP8 codecs

The FP8 operations of the step are the gradient decode (E5M2 / E4M3), the decode of the stored `r`
(E4M3) and `s` (E5M2), and one E5M2 RNE encode (SR's lower neighbour, always an exact value). Their
source depends on the target:

| target | decode | encode |
|---|---|---|
| sm_50 .. sm_87 | E5M2EffMA: raw bits + one FMUL (`decode_fp8x4`) | E5M2EffMA `f32_to_e5m2` |
| sm_89, sm_90 | `cvt.rn.f16x2.{e4m3x2,e5m2x2}` + `cvt.f32.f16` | `cvt.rn.satfinite.e5m2x2.f32` |

All of them are exact on finite codes, so every architecture computes the same bits:

* `f32_to_e5m2` against NVIDIA's model of `cvt.rn.satfinite.e5m2x2.f32` (`cuda_fp8.h`'s
  `__nv_cvt_float_to_fp8`): 0 mismatches over all 2^32 FP32 inputs. The host code (the tests' side of
  the codecs) therefore uses `cuda_fp8.h`, so E5M2EffMA is needed only to build for GPUs below sm_89. Both decodes against the f16 that
  `cvt.rn.f16x2.*` returns: 0 mismatches over all 256 codes.
* `make qat` runs the whole QAT test twice: once on the GPU's own path, once with `QAT_FP8_CVT_MODEL`,
  which routes the sm_89+ wrappers through `cuda_fp8.h` (NVIDIA's software model of the same
  instructions) on any GPU. `python/tests/test_reference.py` with the extension built that way
  (`NVCC_APPEND_FLAGS=-DQAT_FP8_CVT_MODEL`, see [the package README](../README.md#tests)) is
  byte-identical to the PyTorch reference.
* On sm_90, the wrappers and `cuda_fp8.h`'s intrinsics compile to the same SASS (`F2FP.F16.E5M2.UNPACK_B`,
  `F2FP.F16.E4M3.UNPACK_B`, `F2FP.SATFINITE.E5M2.F32.PACK_AB_MERGE_C`). The sm_89 kernels assemble with
  ptxas 12.6 into the same instructions.
* Not run on Ada or Hopper hardware.

They differ only on non-finite codes, which the step never stores (`r` saturates at 448, `s` at 57344).
A gradient that holds Inf / NaN codes is outside the contract: `cvt` (like torch and the reference)
decodes them as Inf / NaN, E5M2EffMA as large finite values.

sm_89's `cvt` needs PTX ISA 8.1 (CUDA 12.1); with an older nvcc the build drops sm_89
(`build/arch.mk`, `build/torch_ext.py`) and `qat_codec.cuh` refuses it with an `#error`. sm_90 needs PTX ISA 7.8 (CUDA 11.8).

### Performance

Measured on a GTX 1080 (Pascal, 320 GB/s DRAM peak) only. A 4096 x 4096 weight, median of 100-200 steps
after a boost warm-up (`make bench`):

| step | time | bytes / element | DRAM throughput |
|---|---|---|---|
| `mlops_adamw`, FP32 param + grad | 1.66 ms | 28 | 283 GB/s (88% of peak) |
| `mlops_adamw`, BF16 param + grad | 1.31 ms | 22 | 281 GB/s (88%) |
| `mlops_adamw` FP32 + NVFP4 re-quantization (0.56 ms) | 2.22 ms | | |
| `mlops_adamw_qat`, E5M2 grad | 1.33 ms | 13.1 | 166 GB/s (52%) |
| `mlops_adamw_qat`, FP32 grad | 1.34 ms | 16.1 | 202 GB/s (63%) |
| `mlops_adamw_qat`, UE5M3 scales, E5M2 / FP32 grad | 1.30 / 1.33 ms | 13.1 / 16.1 | 169 / 203 GB/s |

UE5M3 scales cost nothing: their decode has no sign bit to handle (the step kernel is 6 SASS instructions
shorter), 0.5-1% faster on the GTX 1080 and the Tesla P4 alike.

Against the NVFP4 baseline it replaces (AdamW on an FP32 master plus re-quantization) the QAT step is
1.67x faster; against plain BF16 AdamW it is on par while moving 40% fewer bytes. On the GTX 1080 it is not
bandwidth-bound: the same access pattern with no arithmetic runs in 0.76 ms (290 GB/s). The difference is
arithmetic: the scale vote's 17-candidate window (~0.25 ms), the stochastic roundings of `s` and `r`
(~0.2 ms) and the uniforms (~0.15 ms). Newer GPUs have more FP32 throughput per byte of DRAM bandwidth,
so there the step should be closer to its memory bound; that is not measured.

### Training

`misc/train_nvfp4.py` (a training script outside this repository): a 7-layer decoder (dim 64, SwiGLU 128, 4 heads,
287k NVFP4 + 279k dense parameters) on WikiText-2's `wiki.train.raw` (byte-level BPE 4096, last 5% held
out), NVFP4 forward through NVFP4EffMA's GEMM, 5000 steps of 20 x 256 tokens, lr 1e-3 with 100 warm-up
steps and cosine decay to 1e-4, weight decay 0.1; the same hyperparameters for both optimizers:

| NVFP4 linears optimized by | validation loss, seed 0 / 1 | NVFP4 weight + optimizer state |
|---|---|---|
| `mlops_adamw` on an FP32 master, re-quantized every step | 4.720 / 4.706 | 12.56 B / weight |
| `mlops_adamw_qat`, E5M2 gradients | 4.716 / 4.707 | 6.12 B / weight |

The difference between the optimizers (at most 0.003) is below the spread between seeds (0.01-0.014): with
half the memory, QAT trains as well as an FP32 master here. Activations enter the GEMM exactly in FP32
(`act="exact"`).

### Limits

* Rowwise NVFP4 with 1x16 blocks and E4M3 or UE5M3 scales (linear or swizzled), global per tensor or per row; no
  columnwise copy (a backward that needs `W^T` dequantizes or transposes).
* The scale of a block whose curvature never becomes significant stays where initialization put it.
* Under pure-noise gradients about 1% of the block scales move in 2000 steps (`make qat`).

## Code

| file | |
|---|---|
| `../include/mlops_adamw.h`, `../include/mlops_adamw_qat.h` | C API, the exact order of operations |
| `src/adamw_kernels.cuh`, `src/adamw_inst.cu`, `src/adamw.cu` | AdamW kernel, its 9 (param, grad) dtype instances, host side |
| `src/adamw_qat_kernels.cuh`, `src/adamw_qat_inst.cu`, `src/adamw_qat.cu` | QAT step kernel, its 6 instances (gradient type x scale format), host side (state, init, scale commit) |
| `src/adamw_qat_init.cu` | `mlops_adamw_qat_init`: NVFP4EffMA's quantization + init (the only code that links NVFP4EffMA) |
| `src/qat_codec.cuh` | SR encodes, secant maps, Quake rsqrt, uniforms, the FP8 codec selection |
| `tests/`, `bench/` | `make test`, `make qat`, `make bench` (in `optimizers/`) |

Each kernel instance is its own object, so a build for all architectures runs in parallel.
