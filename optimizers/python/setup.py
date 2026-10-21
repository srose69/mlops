# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Build of mlops-optimizers (import mlops.optim): AdamW and the NVFP4 AdamW-QAT for PyTorch.

    pip install ./optimizers/python --no-build-isolation        (from the mlops root; or ./setup.sh optimizers)

Needs, installed beforehand (nothing is downloaded):
  * PyTorch;
  * nvfp4effma 0.4+ (its headers; at run time its NVFP4 tensors and GEMM);
  * e5m2effma 0.2+ only to build for GPUs below sm_89 (headers only, not needed at run time).
Architectures and the other variables: ../../build/torch_ext.py.
"""
import os
import sys

from setuptools import setup

try:
    import torch  # noqa: F401
    from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension
except ImportError as e:  # build isolation hides the installed torch
    sys.exit(f"mlops-optimizers needs PyTorch at build time ({e}); install torch first and use "
             "`pip install --no-build-isolation ...`")

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)  # optimizers/
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "build"))
import torch_ext  # noqa: E402

NAME = "mlops-optimizers"
VERSION = torch_ext.header_version(os.path.join(PKG, "include", "mlops_optimizers.h"), "MLOPS_OPTIMIZERS")
archs = torch_ext.target_archs(NAME, CUDA_HOME)
nv_dir, nv_ver = torch_ext.dependency(NAME, "nvfp4effma", "to build and to run")
e5_dir = e5_ver = None
if any(a < 89 for a in archs):
    e5_dir, e5_ver = torch_ext.dependency(NAME, "e5m2effma", "to build for GPUs below sm_89")
major, minor = (int(x) for x in nv_ver.split(".")[:2])
print(f"{NAME}: against nvfp4effma {nv_ver}" + (f", e5m2effma {e5_ver} (headers)" if e5_ver else ""), file=sys.stderr)

setup(
    name=NAME,
    version=torch_ext.local_version(VERSION),
    description="AdamW and NVFP4 AdamW-QAT (no master copy) CUDA kernels for PyTorch, GPUs sm_50..sm_90",
    long_description=open(os.path.join(PKG, "README.md")).read(),
    long_description_content_type="text/markdown",
    packages=["mlops.optim"],
    ext_modules=[CUDAExtension(
        "mlops.optim._C", ["csrc/mlops_optim_torch.cpp"],
        include_dirs=[os.path.join(PKG, "include"), os.path.join(nv_dir, "include")],
        # the inline device code of these versions is compiled into the library: mlops.optim checks at import
        # that the installed nvfp4effma is the same major.minor
        define_macros=[("MLOPS_BUILT_NVFP4EFFMA", f'"{nv_ver}"'), ("MLOPS_BUILT_E5M2EFFMA", f'"{e5_ver or ""}"')],
        extra_compile_args={"cxx": ["-O2"]})],
    cmdclass={"build_ext": torch_ext.make_build(NAME, PKG, "mlops_optimizers", archs,
                                                {"NVFP4EFFMA_DIR": nv_dir, "E5M2EFFMA_DIR": e5_dir})},
    python_requires=">=3.9",
    install_requires=["torch>=2.1", f"nvfp4effma>={major}.{minor},<{major}.{minor + 1}"],
    license_files=torch_ext.copy_license_files(HERE),
    zip_safe=False,
)
