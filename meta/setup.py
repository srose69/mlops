# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""The meta distribution "mlops": no code of its own, it depends on every mlops package.

    pip install --no-index -f <directory with the wheels> mlops        (or ../setup.sh all)

It must not contain an mlops/__init__.py: mlops is a namespace package the others share (mlops.optim, ...).
"""
import os
import sys

from setuptools import setup

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "build"))
import torch_ext  # noqa: E402

VERSION = "0.7.0"
PACKAGES = ["mlops-optimizers~=0.2.0", "mlops-positional~=0.1.0", "mlops-attention~=0.4.0", "mlops-vocab~=0.1.0", "mlops-norm~=0.1.0"]  # one entry per package, at the minor version this release was made with

setup(
    name="mlops",
    version=VERSION,
    description="CUDA training ops for GPUs without native FP4 (sm_50..sm_90): every mlops package",
    long_description=open(os.path.join(os.path.dirname(HERE), "README.md")).read(),
    long_description_content_type="text/markdown",
    packages=[],
    python_requires=">=3.9",
    install_requires=PACKAGES,
    license_files=torch_ext.copy_license_files(HERE),
)
