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
| [mlops-optimizers](optimizers) | `mlops.optim` | 0.1.0 | AdamW, and AdamWQAT: AdamW that trains NVFP4 weights in place, without a master copy |
| [mlops-positional](positional) | `mlops.positional` | 0.1.0 | RoPE of q and k in one pass, no error growth with the position (BF16 at the floor up to 10M), YaRN / Llama 3 / LongRoPE scaling; a bit-exact drop-in for transformers' `apply_rotary_pos_emb` |
| `mlops` | | 0.2.0 | meta package: every package above |

```python
from mlops.optim import AdamW, AdamWQAT, QATLinear, split_params
from mlops.positional import RoPE
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
meta/           the meta package "mlops"
setup.sh        source installation
```

## License

GNU AGPL v3 with the Linking and Attribution Exception: programs using mlops may be under any license, but
must show the attribution in [NOTICE](NOTICE). See [LICENSE](LICENSE) and [LICENSE-EXCEPTION](LICENSE-EXCEPTION).
