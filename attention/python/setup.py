# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Build mlops-attention (import mlops.attention): FBLA compressed-KV FlashAttention prefill and decode.

    pip install ./attention/python --no-build-isolation        (from the mlops root; or ./setup.sh attention)

Needs PyTorch, the CUDA toolkit and a target GPU sm_61 or newer. Architectures and the other variables:
../../build/torch_ext.py.
"""
import os
import sys
from setuptools import setup

try:
    import torch  # noqa: F401
    from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension
except ImportError as e:
    sys.exit(f"mlops-attention needs PyTorch at build time ({e}); install torch first and use "
             "`pip install --no-build-isolation ...`")

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "build"))
import torch_ext  # noqa: E402

NAME = "mlops-attention"
VERSION = torch_ext.header_version(os.path.join(PKG, "include", "mlops_attention.h"), "MLOPS_ATTENTION")
archs = [a for a in torch_ext.target_archs(NAME, CUDA_HOME) if a >= 61]
if not archs:
    sys.exit("mlops-attention needs sm_61 or newer (DP4A)")

setup(
    name=NAME,
    version=torch_ext.local_version(VERSION),
    description="FBLA compressed-KV FlashAttention prefill and decode for NVIDIA GPUs (DP4A; tensor-core prefill on sm_75 .. sm_89)",
    long_description=open(os.path.join(PKG, "README.md")).read(),
    long_description_content_type="text/markdown",
    packages=["mlops.attention"],
    ext_modules=[CUDAExtension("mlops.attention._C", ["csrc/mlops_attention_torch.cpp"],
                               include_dirs=[os.path.join(PKG, "include")],
                               extra_compile_args={"cxx": ["-O2"]})],
    cmdclass={"build_ext": torch_ext.make_build(NAME, PKG, "mlops_attention", archs, {})},
    python_requires=">=3.9",
    install_requires=["torch>=2.1"],
    license_files=torch_ext.copy_license_files(HERE),
    zip_safe=False,
)
