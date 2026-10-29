# mlops-vocab

Everything along the vocabulary axis as CUDA kernels for NVIDIA GPUs sm_50 .. sm_90, for inference: the embedding rows for
token ids that live on the device, the BF16 LM head for 1 .. 8 tokens, and token sampling without sorting the vocabulary. A
C API ([`include/`](include)), the work of each call as items for an executor of one's own
([`include/mlops_vocab_items.h`](include/mlops_vocab_items.h)), and a PyTorch package, `mlops.vocab`. The kernels, the arithmetic
and the measurements: [lm_head/README.md](lm_head/README.md), [sample/README.md](sample/README.md).

The head reads Qwen3-8B's 151936 x 4096 BF16 matrix (1245 MB) at 750 GB/s on an L40S, 98 % of its DRAM peak, and at the
speed of simple streaming reads of the same buffer on a GTX 1080 (250 - 255 GB/s). A sampled token costs 2 - 66 µs on a GTX 1080
and 3 - 39 µs on an L40S, 1 - 3 % of the head.

| call | header | what it does |
|---|---|---|
| `mlops_embed` | [`mlops_embed.h`](include/mlops_embed.h) | `x = table[ids]` in FP32, `xw = bf16(x * w)` and the sums of squares of 256 elements: the inputs of the first RMSNorm. The table in VRAM or in mapped host memory. |
| `mlops_lm_head` | [`mlops_lm_head.h`](include/mlops_lm_head.h) | `logits = s * logit_scale * (W x)` for n <= 8 columns, the deferred RMSNorm scalar `s` from the producer's partial sums, and each element's maximum and lowest argmax for the sampler. |
| `mlops_sample` | [`mlops_sample.h`](include/mlops_sample.h) | greedy, temperature, top-k, top-p and a repetition penalty. The seed and the step live on the device. |

## Install

Needs only PyTorch (a CUDA build) and, to build from the sources, the CUDA toolkit. Nothing is downloaded.

```sh
pip install --no-index -f . mlops-vocab             # from a release, in the directory with the wheel
./setup.sh vocab                                    # from the sources (the mlops root)
```

The build compiles for the GPUs visible at build time, for all of sm_50..sm_90 if none is visible, or for
`MLOPS_ARCHS="61 89"`; other variables: [`../build/torch_ext.py`](../build/torch_ext.py).

## Python

```python
from mlops.vocab import Sampler, embed, lm_head

e = embed(table, ids, norm_weight)                  # e.x, e.xw, e.parts
h = lm_head(weight, hidden, want_stats=True, logit_scale=1 / 0.7)
token = Sampler(vocab, temperature=0.7, top_p=0.9, seed=1)(h.logits, stats=h)
```

* **`embed(table, ids, norm_weight=None, *, variant=0, items=0)`**: `table` bfloat16 `[vocab, hidden]`, `ids` int32 `[n]` on
  the device; `hidden` a multiple of 256.
* **`lm_head(weight, x, *, norm_parts=None, inv_n=0.0, eps=0.0, logit_scale=1.0, want_logits=True, want_stats=False,
  variant=0, items=0)`**: `x` bfloat16 `[n, hidden]`, `n` in 1 .. 8; the logits are float32 `[n, vocab]`.
* **`Sampler(vocab, *, temperature=0.0, top_k=0, top_p=1.0, penalty=1.0, seed=0)`**: `sampler(logits, seen=None, stats=None)`
  returns the token as an int32 `[1]` on the device. With `stats` from the head and penalty 1, greedy needs no pass over
  the logits.
* **`embed_variants`, `lm_head_variants`**: the instances the device accepts for a shape, as data.

## Samples

Short runnable examples in [`samples/`](samples):

| sample | shows |
|---|---|
| `01_quickstart.py` | `embed`, `lm_head` and a greedy `Sampler` against PyTorch |
| `02_decode_loop.py` | a sampling loop with no host values: the head's statistics feed the sampler, the seed and the step live on the device |
| `03_c_api.cu` | the C API: the variants of a head, one head call and a greedy token on a stream |

## C / C++

[`include/mlops_vocab.h`](include/mlops_vocab.h) includes the whole package:

* **Calls.** `mlops_embed`, `mlops_lm_head` and `mlops_sample` over raw device pointers, each returning a status
  (`mlops_*_last_error()` describes failures). The sampler's workspace (`mlops_sample_workspace_bytes`) is zero before the
  first call.
* **Variants.** `mlops_embed_variants`, `mlops_lm_head_variants`: the instances for a shape, as data; the element count is
  the caller's.
* **Items.** [`mlops_vocab_items.h`](include/mlops_vocab_items.h) describes a call as work items for an executor: the CTA
  geometry, the arguments, the entry compiled with NVRTC from [`mlops_vocab_items.cuh`](include/mlops_vocab_items.cuh), and the
  byte ranges each item reads and writes.

## Build and test

```sh
make lib                # out/libmlops_vocab.a (ARCHS="61 89", ARCHS=native; default every architecture sm_50..sm_90)
make test               # C tests (ARCHS= and TEST_ARCH= as in ../build/arch.mk)
make bench              # the head and the sampler (per-pass timings)
make install PREFIX=..  # headers, library, pkg-config file
pytest python/tests     # the PyTorch package against PyTorch
```

`make test` checks, bit for bit against host emulations of the arithmetic: the embedding rows, `xw` and the partial sums; every
head variant on CUDA cores, the statistics and key-byte counts (against the logits the call wrote), the lowest index among equal
maxima, the norm scalar and `logit_scale`; the sampler's tokens against the sorted order (kept sets, draw frequencies within
5 sigma, the same token on any grid) and with the head's statistics. The tensor-core variants are checked against the exact
(double) sum within a bound that grows with the number of mma adds, and for repeatability whatever the element count. The
items go through NVRTC: each entry is compiled from its `entry` string and run on its described CTA, the outputs are the
call's, and running every item alone on poisoned outputs shows that only declared bytes change and that together the items
write every output byte.

## License

GNU AGPL v3 with the Linking and Attribution Exception ([NOTICE](../NOTICE)).
