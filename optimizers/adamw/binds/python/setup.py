"""Build of the mlops_adamw PyTorch extension.

    pip install ./binds/python --no-build-isolation        (from the adamw/ repository root)

Needs NVFP4EffMA's static library already built (its own `make`, for every target architecture) and the
NVFP4EffMA / E5M2EffMA sources checked out next to this repo (headers only from E5M2EffMA): set
NVFP4EFFMA_ROOT / E5M2EFFMA_ROOT if they aren't siblings of mlops/.

Architectures, as NVFP4EffMA's own build chooses them:
  MLOPS_ADAMW_ARCHS="61 89"         explicit list of compute capabilities, or
  TORCH_CUDA_ARCH_LIST="6.1;8.9"   the usual PyTorch variable, or
  the GPUs visible to PyTorch at build time, or, without any, every supported one (sm_50..sm_90).
Architectures outside sm_50..sm_90 are dropped (native FP4, or too old); sm_89 needs nvcc 12.1+ (PTX ISA
8.1 for its FP8 cvt) and is dropped with an older one.
"""
import os
import re
import subprocess
import sys

from setuptools import find_packages, setup

try:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
except ImportError as e:
    sys.exit(f"mlops_adamw needs PyTorch at build time ({e}); install torch first and use "
             "`pip install --no-build-isolation ...`")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MLOPS_ADAMW_ROOT", os.path.normpath(os.path.join(HERE, "..", "..")))
NVFP4EFFMA = os.environ.get("NVFP4EFFMA_ROOT", os.path.normpath(os.path.join(ROOT, "..", "..", "..", "NVFP4EffMA")))
E5M2EFFMA = os.environ.get("E5M2EFFMA_ROOT", os.path.normpath(os.path.join(ROOT, "..", "..", "..", "E5M2EffMA")))

HEADER = os.path.join(ROOT, "include", "mlops_adamw.h")
if not os.path.isfile(HEADER):
    sys.exit(f"mlops_adamw sources not found at {ROOT}; build from a repository checkout or set MLOPS_ADAMW_ROOT")
if not os.path.isfile(os.path.join(NVFP4EFFMA, "out", "libnvfp4effma.a")):
    sys.exit(f"NVFP4EffMA static library not found under {NVFP4EFFMA}/out; run `make` there first "
             "(set NVFP4EFFMA_ROOT if it isn't a sibling of this repository)")
if not os.path.isfile(os.path.join(E5M2EFFMA, "src", "e5m2effma.cuh")):
    sys.exit(f"E5M2EffMA sources not found at {E5M2EFFMA}; set E5M2EFFMA_ROOT")
VERSION = re.search(r'#define MLOPS_ADAMW_VERSION "([^"]+)"', open(HEADER).read()).group(1)

if CUDA_HOME is None:
    sys.exit("mlops_adamw: CUDA toolkit not found; set CUDA_HOME")
SUPPORTED = [50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90]  # NVFP4EffMA's range


def nvcc_version():
    out = subprocess.check_output([os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"], text=True)
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def target_archs():
    env = os.environ.get("MLOPS_ADAMW_ARCHS") or os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env:
        archs = set()
        for tok in re.split(r"[;,\s]+", env.strip()):
            m = re.fullmatch(r"(\d+)\.?(\d)?(\+PTX)?", tok)
            if not m:
                sys.exit(f"mlops_adamw: cannot parse architecture '{tok}' (expected e.g. 61 or 6.1)")
            archs.add(int(m.group(1) + (m.group(2) or "")))
        source = "MLOPS_ADAMW_ARCHS / TORCH_CUDA_ARCH_LIST"
    elif torch.cuda.is_available() and torch.cuda.device_count():
        archs = {10 * a + b for a, b in (torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))}
        source = "visible GPUs"
    else:
        archs, source = set(SUPPORTED), "default (no GPU visible)"
    dropped = sorted(a for a in archs if a not in SUPPORTED)
    if dropped:
        print(f"mlops_adamw: skipping sm_{', sm_'.join(map(str, dropped))} (native FP4 or unsupported)", file=sys.stderr)
    archs = sorted(a for a in archs if a in SUPPORTED)
    if 89 in archs and nvcc_version() < (12, 1):
        print("mlops_adamw: skipping sm_89: its FP8 cvt needs nvcc 12.1+ (PTX ISA 8.1), found %d.%d" % nvcc_version(),
              file=sys.stderr)
        archs.remove(89)
    if not archs:
        sys.exit(f"mlops_adamw: no supported architecture to build for ({source}); it targets sm_50..sm_90")
    print(f"mlops_adamw: building for sm_{', sm_'.join(map(str, archs))} ({source})", file=sys.stderr)
    return archs


# BuildExtension reads the target list from TORCH_CUDA_ARCH_LIST: SASS for each, no PTX (never JIT-ed onto
# sm_100+, which has native FP4)
os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(f"{a // 10}.{a % 10}" for a in target_archs())

setup(
    name="mlops_adamw",
    version=VERSION,
    description="AdamW and NVFP4 AdamW-QAT CUDA kernels (mlops/optimizers/adamw)",
    packages=find_packages(include=["mlops_adamw", "mlops_adamw.*"]),
    ext_modules=[CUDAExtension(
        "mlops_adamw._C",
        ["csrc/mlops_adamw_torch.cpp",
         os.path.join(ROOT, "src", "adamw.cu"),
         os.path.join(ROOT, "src", "adamw_qat.cu")],
        include_dirs=[os.path.join(ROOT, "include"), os.path.join(ROOT, "src"),
                     os.path.join(NVFP4EFFMA, "include"), os.path.join(NVFP4EFFMA, "src"),
                     os.path.join(E5M2EFFMA, "include"), os.path.join(E5M2EFFMA, "src")],
        library_dirs=[os.path.join(NVFP4EFFMA, "out")],
        libraries=["nvfp4effma"],
        extra_compile_args={"cxx": ["-O2"],
                           "nvcc": ["-O3", "-std=c++17", "-ccbin", os.environ.get("MLOPS_ADAMW_CCBIN", "g++-12")]},
    )],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch>=2.1", "nvfp4effma"],
    zip_safe=False,
)
