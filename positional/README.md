# mlops-positional

Positional encodings as CUDA kernels for NVIDIA GPUs sm_50 .. sm_90, for training and inference: a C API
([`include/`](include)) and a PyTorch package, `mlops.positional`. Its first member is RoPE: q and k of every
head in one pass, as accurate at 10M positions as at 0, with every frequency scaling transformers knows (YaRN, Llama 3, linear,
dynamic NTK, LongRoPE). How it works, accuracy and measurements: [rope/README.md](rope/README.md).

On a GTX 1080 it rotates a 4096-token prefill (32 q / 8 kv heads of 128, BF16) in 324 µs against 2.65 ms for
transformers' eager path (8.2×), and a decode step in 22 µs against 92-107 µs.

**Accuracy at long positions.** In BF16 it stays at the floor of the dtype from 128K to 10M positions: the exact
rotation rounded once. transformers and torchtune take the angle as one FP32 product, so their error grows with
the position, to 9 % RMS and 0.67 at the worst pair at 10M
([table](rope/README.md#at-long-positions)).

## Install

Needs only PyTorch (a CUDA build) and, to build from the sources, the CUDA toolkit. Nothing is downloaded.

```sh
pip install --no-index -f . mlops-positional        # from a release, in the directory with the wheel
./setup.sh positional                               # from the sources (the mlops root)
```

The build compiles for the GPUs visible at build time, for all of sm_50..sm_90 if none is visible, or for
`MLOPS_ARCHS="61 89"`; other variables: [`../build/torch_ext.py`](../build/torch_ext.py).

## RoPE

```python
import torch
from mlops.positional import RoPE

rope = RoPE(128, 500000.0, scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192})
rope = RoPE.from_hf(model.config)                          # or from a transformers config

q, k = rope(q, k, position_ids, layout="bhsd")             # transformers' [B, H, S, D]; differentiable
q, k = rope(q, k, offset=past_len)                         # [B, S, H, D], positions past_len + s
with torch.no_grad():
    rope(q, k, position_ids, inplace=True)                 # inference: in place
```

* **`RoPE(head_dim, base=10000.0, *, rotary_dim=None, interleaved=False, scaling=None,
  max_position_embeddings=None)`**
  * `rotary_dim` (default `head_dim`): how many leading elements of each head rotate; the rest pass through.
  * Pairs are `(i, i + rotary_dim/2)`, or `(2i, 2i+1)` with `interleaved=True` (GPT-J).
  * `scaling`: a transformers rope-scaling dict — `"linear"`, `"dynamic"`, `"yarn"`, `"llama3"`, `"longrope"`
    with their keys. Its attention factor (YaRN's mscale, …) is applied too.
  * `RoPE.from_hf(config)` reads `rope_parameters` (or `rope_theta` / `rope_scaling`), `head_dim` and
    `partial_rotary_factor` from a transformers config.
* **`rope(q, k=None, positions=None, *, layout="bshd", offset=0, inplace=False, seq_len=None)`** → `(q, k)`
  * `layout`: `"bshd"` `[B, S, H, D]`, `"bhsd"` `[B, H, S, D]` or `"thd"` `[T, H, D]`.
  * k may have fewer heads than q (GQA).
  * Any strides with a contiguous last dimension work, e.g. slices of a packed qkv projection.
  * `positions`: int32 / int64 `[B, S]` or `[S]`, or `None` for `offset + s`.
  * bf16 / fp16 / fp32.
  * The backward is the inverse rotation.
  * `seq_len` selects the frequencies of the dynamic and LongRoPE scalings; by default it comes from the positions.
* **Accuracy.** At any position below 2³¹ (phase reduced in turns, `sincospi`, one rounding to the dtype), BF16 /
  FP16 outputs are the exact rotation rounded once, give or take the last bit; FP32 outputs are within a few FP32
  ulps (≤ 4·10⁻⁷ relative).
* **`rope_frequencies(rotary_dim, base, scaling, *, max_position_embeddings=None, seq_len=None)`** →
  `(inv_freq, attention_factor)`: the float64 frequencies of a scaling, as transformers computes them.

## transformers

```python
from mlops.positional import apply_rotary_pos_emb, patch_transformers

q, k = apply_rotary_pos_emb(q, k, cos, sin)       # transformers' function, on the kernel, bit for bit
patch_transformers("llama", "qwen2", "mistral")   # installs it into those modeling modules
```

* **Drop-in `apply_rotary_pos_emb`.** It takes the cos / sin tables of the model's rotary module and returns the
  same bits as transformers in the forward and in the autograd backward, in one pass.
* **Fallback.** CPU tensors, mixed dtypes and gradients through cos / sin run the plain PyTorch formula.
* **`patch_transformers`.** It is for models with the standard signature and half-split pairs. A patched Llama
  produces identical outputs.

## Samples

Short runnable examples in [`samples/`](samples):

| sample | shows |
|---|---|
| `01_quickstart.py` | `RoPE` on q and k (GQA), positions vs offset, both layouts, `<q_m, k_n>` depending on `m − n` only |
| `02_transformers_model.py` | `patch_transformers("llama")` on a model: identical outputs, faster forward |
| `03_long_context_yarn.py` | a YaRN-extended model's frequencies and attention factor; accuracy at position 10M |
| `04_training.py` | an attention layer with RoPE on a packed qkv projection, trained (autograd) |
| `05_inference_decode.py` | prefill with left padding, then decode against a KV cache, in place |
| `06_c_api.cu` | the C API: frequency table, rotate, rotate back, error reporting |

## C / C++

[`include/mlops_rope.h`](include/mlops_rope.h) (all of the package: `mlops_positional.h`):

* **Call.** `mlops_rope(cfg, q_in, q_out, k_in, k_out, stream)` over raw device pointers, returning a status
  (`mlops_rope_last_error()` describes failures).
* **Angles.** From positions plus the frequency table of `mlops_rope_freq(inv_freq, n, out)`, or from cos / sin
  tables.
* **Direction.** `inverse` gives the backward.
* **Header-only device code.** [`rope/src/rope.cuh`](rope/src/rope.cuh) holds the phase, sincospi and rotation
  for kernels that fuse RoPE into something else.

## Build and test

```sh
make lib                # out/libmlops_positional.a: every architecture sm_50..sm_90 (ARCHS="61 89", ARCHS=native)
make test               # exact mode vs long double, every dtype, layout, vector width, positions up to 2^31;
                        # tables vs Hugging Face's arithmetic bit for bit, forward and backward; argument checks
make bench              # speed vs transformers (needs mlops-positional and transformers installed)
make accuracy           # accuracy at 128K .. 10M positions vs transformers and torchtune (stand-alone; checks
                        # its copies of their arithmetic against them when they are installed)
make install PREFIX=..  # headers, library, pkg-config file
pytest python/tests     # vs transformers 5.x (bit for bit, frequencies of every scaling, a patched Llama),
                        # vs float64, autograd, argument checks
```

## License

GNU AGPL v3 with the Linking and Attribution Exception ([NOTICE](../NOTICE)).
