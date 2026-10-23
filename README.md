# mlops

ML ops in CUDA C++ for training and inference on NVIDIA GPUs without native FP4: sm_50 .. sm_90, from Maxwell
to Hopper. Each op has a C API and PyTorch bindings. The NVFP4 ones build on
[NVFP4EffMA](https://github.com/srose69/nvfp4effma) (NVFP4 tensors and GEMM) and, for GPUs below sm_89, on the
FP8 codecs of [E5M2EffMA](https://github.com/srose69/e5m2effma); from sm_89 on, FP8 is native and they use the
hardware `cvt`.

mlops is a set of packages, each installed on its own and with its own version; the meta package `mlops`
installs all of them.

| package | Python | version | contents |
|---|---|---|---|
| [mlops-optimizers](optimizers) | `mlops.optim` | 0.2.0 | AdamW, and AdamWQAT: AdamW that trains NVFP4 weights in place, on GPUs from Maxwell to Hopper. Transformer Engine trains NVFP4 only on Blackwell (sm_100+), through an FP32 or FP16 master copy re-quantized every step. AdamWQAT keeps no such copy: each weight's position lives on the NVFP4 lattice as a 16-bit coordinate, the moments in FP8, and block scales move only on significant evidence and, UE5M3 by default, can grow 256x without an amax recompute (`to_e4m3` exports Transformer Engine's E4M3 scales bit-exact when they fit, else keeps UE5M3 and warns). 6 B per weight for weight + optimizer state, against 12.6 B for an FP32 master with FP32 moments. |
| [mlops-positional](positional) | `mlops.positional` | 0.1.0 | RoPE of q and k in one pass, 4–8× transformers' eager path, with YaRN / Llama 3 / LongRoPE scaling built in. Its error does not grow with the position: at the floor of the dtype up to 10M tokens. transformers and torchtune take the angle as one FP32 product: in FP32 their error is 1.1e-3 already at 128K, 44 000× the float32 floor, and 9 % RMS at 10M, where they place every token up to ±1.4 positions off. A bit-exact drop-in for transformers' `apply_rotary_pos_emb`. |
| [mlops-attention](attention) | `mlops.attention` | 0.2.1 | FBLA (Flash Block Attention): compressed-KV FlashAttention prefill and decode for DP4A GPUs (sm_61+), 3+3 and 4+2 modes, 6.25 bits per KV element. On GTX 1080: decode 35–42× faster than PyTorch BF16 SDPA at 16K (at 256K SDPA runs out of memory, FBLA takes 1 ms), prefill 2× faster than PyTorch FP32 SDPA. As accurate as llama.cpp's `q8_0` cache (both within the noise of BF16 on Qwen3-0.6B) at 74 % of its size; `q4_0` costs 20 % of the loss. Details: [attention/docs/overview.md](attention/docs/overview.md). |
| `mlops` | | 0.5.0 | meta package: every package above |

**Why UE5M3 block scales by default.** NVFP4 as Transformer Engine stores it has E4M3 block scales
(`2^-9 .. 448`). NVIDIA's NVFP4 pretraining ([arXiv:2509.25149](https://arxiv.org/abs/2509.25149)) diverges
with every linear layer in FP4: the last layers need more range. The recipe keeps 16% of them in BF16, adds a
Hadamard transform and recomputes every amax before each quantization. Graphcore
([arXiv:2609.02846](https://arxiv.org/abs/2609.02846), section 5.2) shows that E4M3 scales leave a tensor at
its amax no room to grow, while UE5M3 leaves about 137x. On Nemotron-H 8B over 190B tokens, Transformer
Engine's native NVFP4 without the BF16 layers and the transform had repeated loss spikes. UE5M3 scales
trained smoothly with every linear in FP4, refreshed the tensor scale only every 50 steps, and ended below
Transformer Engine's full recipe. AdamWQAT never recomputes an amax, so the scale format alone sets how far
a block can grow: 8x with E4M3, 256x with UE5M3. The two train bit-identically until E4M3 would run out
([details](optimizers/README.md#block-scales-ue5m3-by-default)).

**What FP32 angles cost at long context.** The angle error of a RoPE frequency, divided by the frequency,
is a position error: where that frequency places the token. transformers and torchtune compute the angle as one
FP32 product, and the error grows with the position. With FP32 q, measured by `make accuracy` in
mlops-positional (Llama 3 frequencies):

| context | token placed off by, max / RMS | distance between two tokens off by | mlops.positional |
|---|---|---|---|
| 1M | ±0.13 / 0.05 | 0.13 | ±0.09 / 0.007 |
| 4M | ±0.54 / 0.19 | 0.49 | ±0.07 / 0.005 |
| 10M | ±1.38 / 0.48 | 1.20 | ±0.05 / 0.004 |

At 10M the pairs that tell neighbouring tokens apart (periods of 7-12 tokens) are turned up to 0.69 rad (39°)
the wrong way. mlops' column does not grow: it is the float32 floor of the lowest frequencies (the exact
rotation rounded once gives ±0.02).

**What selective KV refinement buys.** FBLA keeps the attention sink and open group in BF16. Every closed
group has one BF16 base/step per channel and a 6-bit code; only its high 3 or 4 bits are read initially. A
16-token block's low plane is fetched when its coarse-softmax share `W_j` exceeds the mode's threshold. Unlike
materialized blockwise attention, the coarse kernel already computes FA2 `(m,l,o)`; refinement adds a sparse
`full - coarse` correction, so decode reads about 4.3 bits per element and runs near DRAM bandwidth (16K
context: 0.1 .. 0.2 ms on a GTX 1080, 35–42× PyTorch BF16 SDPA; 1M: 5.2 ms at 88 % of a device copy). Prefill is
compute-bound and reads the full codes through DP4A GEMM tiles at 8 equivalent FP32 TFLOPS. On Qwen3-0.6B,
16K contexts, 11,264 scored tokens (11 windows), BF16 loss 2.323:

| KV cache | bits per element | delta loss vs BF16 SDPA |
|---|---:|---:|
| BF16 SDPA | 16 | +0.0000 |
| llama.cpp `q8_0` | 8.5 | -0.0001 |
| FBLA 3+3, CUDA kernel | 6.25 | -0.0008 |
| FBLA 4+2, CUDA kernel | 6.25 | +0.0005 |
| KIVI 4-bit | 5.0 | +0.0038 |
| llama.cpp `q4_0` | 4.5 | +0.4755 |

The windows differ by up to ±0.006: FBLA and `q8_0` are within the noise of BF16.

On one 16K window / 2,048 tokens, FBLA 4+2's CUDA kernel has 96.04% top-1 agreement and mean
`KL(BF16 || FBLA) = 0.00228`, against 95.56% and 0.00464 for KIVI-4. Full tails, loss ranges, format traffic
and the 16K .. 1M native benchmark: [attention/docs/overview.md](attention/docs/overview.md).

**Against llama.cpp.** Qwen3-8B on one GTX 1080, the same NVFP4 weights (`nvidia/Qwen3-8B-NVFP4`; llama.cpp
runs its own GGUF conversion of them), `llama-bench`'s tests: the mlops stack (NVFP4EffMA linear layers,
mlops.positional, FBLA) prefills 1.9× and decodes 1.5× faster at a 16K context against its `q8_0` cache, and
prefills 2.1× faster at 32K, where llama.cpp fits only a `q4_0` cache. The FBLA
cache is as accurate as `q8_0` at 74 % of its size (6.25 bits per element against 8.5), and far more accurate
than `q4_0`. Full table: [attention/docs/llamacpp-compare.md](attention/docs/llamacpp-compare.md).

```python
from mlops.optim import AdamW, AdamWQAT, QATLinear, split_params
from mlops.positional import RoPE
from mlops.attention import empty, prefill, append, decode
```

## Install

None of the packages is on PyPI, and installing them never downloads anything: dependencies are installed
by you, beforehand.

* PyTorch (a CUDA build), for every package;
* **nvfp4effma 0.4.x** ([releases](https://github.com/srose69/nvfp4effma/releases)): mlops-optimizers, at build
  and at run time;
* **e5m2effma 0.2+** ([releases](https://github.com/srose69/e5m2effma/releases)): mlops-optimizers, only to build
  for GPUs below sm_89, and only its headers; not needed at run time or with prebuilt wheels.

**Prebuilt wheels** (one per Python / PyTorch / CUDA combination), from the directory they are in:

```sh
pip install --no-index -f . mlops-optimizers     # one package
pip install --no-index -f . mlops                # every package (all their wheels in the directory)
```

`--no-index` takes the packages from that directory and their dependencies from what is installed.

**From the sources:**

```sh
./setup.sh --list           # packages and their versions
./setup.sh optimizers       # one package or several
./setup.sh all              # every package and the meta package
```

The build compiles for the GPUs visible at build time, for all of sm_50..sm_90 if none is visible, or for
`MLOPS_ARCHS="61 89"`; mlops-optimizers needs nvcc 12.1+ for sm_89 (its FP8 `cvt`). Other variables
(`MAX_JOBS`, `MLOPS_CCBIN`, `CUDA_HOME`): [`build/torch_ext.py`](build/torch_ext.py).

## Build and test (C / CUDA)

Every package has its own Makefile; the one at the root runs a target in each of them (`PKGS`):

```sh
make lib                    # lib<package>.a for every architecture sm_50..sm_90 (ARCHS="61 89", ARCHS=native)
make test qat               # the packages' tests on GPU 0 (GPU=n)
make bench
make install PREFIX=/opt/mlops
```

Builds run in parallel within half of the available RAM (`JOBS=n` overrides), one object per kernel
instance. Toolchain, architectures and the dependencies' headers (installed packages, else checkouts next
to `mlops/`): [`build/arch.mk`](build/arch.mk).

## Layout

```
build/          arch.mk (shared by the package Makefiles), torch_ext.py (shared by their setup.py)
optimizers/     mlops-optimizers: include/ (C API), adamw/ (kernels, tests, bench, math), python/ (mlops.optim),
                samples/
positional/     mlops-positional: include/ (C API), rope/ (kernels, tests, math), python/ (mlops.positional), bench/
attention/      mlops-attention: include/ (C API), fbla/ (kernels, tests, bench), python/ (mlops.attention), samples/
meta/           the meta package "mlops"
setup.sh        source installation
```

## License

GNU AGPL v3 with the Linking and Attribution Exception: programs using mlops may be under any license, but
must show the attribution in [NOTICE](NOTICE). See [LICENSE](LICENSE) and [LICENSE-EXCEPTION](LICENSE-EXCEPTION).
