# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Build of mlops-positional (import mlops.positional): positional encodings (RoPE) for PyTorch.

    pip install ./positional/python --no-build-isolation        (from the mlops root; or ./setup.sh positional)

Needs only PyTorch and the CUDA toolkit. Architectures and the other variables: ../../build/torch_ext.py.
"""
import os
import sys

from setuptools import setup

try:
    import torch  # noqa: F401
    from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension
except ImportError as e:  # build isolation hides the installed torch
    sys.exit(f"mlops-positional needs PyTorch at build time ({e}); install torch first and use "
             "`pip install --no-build-isolation ...`")

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)  # positional/
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "build"))
import torch_ext  # noqa: E402

NAME = "mlops-positional"
VERSION = torch_ext.header_version(os.path.join(PKG, "include", "mlops_positional.h"), "MLOPS_POSITIONAL")
archs = torch_ext.target_archs(NAME, CUDA_HOME)

setup(
    name=NAME,
    version=torch_ext.local_version(VERSION),
    description="Positional encodings (RoPE with YaRN / Llama 3 / LongRoPE scaling) as CUDA kernels for PyTorch, GPUs sm_50..sm_90",
    long_description=open(os.path.join(PKG, "README.md")).read() if os.path.isfile(os.path.join(PKG, "README.md")) else "",
    long_description_content_type="text/markdown",
    packages=["mlops.positional"],
    ext_modules=[CUDAExtension("mlops.positional._C", ["csrc/mlops_positional_torch.cpp"],
                               include_dirs=[os.path.join(PKG, "include")], extra_compile_args={"cxx": ["-O2"]})],
    cmdclass={"build_ext": torch_ext.make_build(NAME, PKG, "mlops_positional", archs, {})},
    python_requires=">=3.9",
    install_requires=["torch>=2.1"],
    license_files=torch_ext.copy_license_files(HERE),
    zip_safe=False,
)
