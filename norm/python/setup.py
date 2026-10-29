# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Build of mlops-norm (import mlops.norm): RMSNorm and LayerNorm with a residual add, gates and INT8 / NVFP4 quantization.

    pip install ./norm/python --no-build-isolation        (from the mlops root; or ./setup.sh norm)

Needs, installed beforehand (nothing is downloaded): PyTorch; nvfp4effma 0.6+ (its headers: the quantizer's primitives
are imported from it; compiled in, not needed at run time). Architectures and the other variables: ../../build/torch_ext.py.
"""
import os
import sys

from setuptools import setup

try:
    import torch  # noqa: F401
    from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension
except ImportError as e:  # build isolation hides the installed torch
    sys.exit(f"mlops-norm needs PyTorch at build time ({e}); install torch first and use "
             "`pip install --no-build-isolation ...`")

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)  # norm/
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "build"))
import torch_ext  # noqa: E402

NAME = "mlops-norm"
VERSION = torch_ext.header_version(os.path.join(PKG, "include", "mlops_norm.h"), "MLOPS_NORM")
archs = torch_ext.target_archs(NAME, CUDA_HOME)
nv_dir, nv_ver = torch_ext.dependency(NAME, "nvfp4effma", "to build")
print(f"{NAME}: against nvfp4effma {nv_ver} (headers)", file=sys.stderr)

setup(
    name=NAME,
    version=torch_ext.local_version(VERSION),
    description="RMSNorm and LayerNorm with residual add, gates and INT8 / NVFP4 quantization as CUDA kernels for PyTorch, GPUs sm_50..sm_90",
    long_description=open(os.path.join(PKG, "README.md")).read() if os.path.isfile(os.path.join(PKG, "README.md")) else "",
    long_description_content_type="text/markdown",
    packages=["mlops.norm"],
    ext_modules=[CUDAExtension("mlops.norm._C", ["csrc/mlops_norm_torch.cpp"],
                               include_dirs=[os.path.join(PKG, "include")], extra_compile_args={"cxx": ["-O2"]})],
    cmdclass={"build_ext": torch_ext.make_build(NAME, PKG, "mlops_norm", archs, {"NVFP4EFFMA_DIR": nv_dir})},
    python_requires=">=3.9",
    install_requires=["torch>=2.1"],
    license_files=torch_ext.copy_license_files(HERE),
    zip_safe=False,
)
