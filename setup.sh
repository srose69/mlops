#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
# Install mlops packages from this source tree into the current Python environment.
#
#   ./setup.sh --list              packages and their versions
#   ./setup.sh optimizers [...]    one package or several
#   ./setup.sh all                 every package and the meta package "mlops"
#
# Nothing is downloaded (pip --no-index): PyTorch, and for optimizers the EffMA packages, must be installed first --
#   nvfp4effma   optimizers (https://github.com/srose69/nvfp4effma/releases)
#   e5m2effma    optimizers, only to build for GPUs below sm_89, headers only (https://github.com/srose69/e5m2effma/releases)
# Variables: PYTHON (default python3), MLOPS_ARCHS="61 89" (default: the visible GPUs), MAX_JOBS, MLOPS_CCBIN,
# CUDA_HOME; see build/torch_ext.py.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
PACKAGES=(optimizers positional attention)  # directories with include/mlops_<name>.h and python/setup.py

version() { sed -n "s/^#define MLOPS_${1^^}_VERSION \"\(.*\)\"/\1/p" "$1/include/mlops_$1.h"; }
meta_version() { sed -n 's/^VERSION = "\(.*\)"/\1/p' meta/setup.py; }

if [ $# -eq 0 ] || [ "$1" = -h ] || [ "$1" = --help ]; then
  sed -n '4,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0
fi
if [ "$1" = --list ]; then
  for p in "${PACKAGES[@]}"; do printf '%-12s mlops-%-12s %s\n' "$p" "$p" "$(version "$p")"; done
  printf '%-12s %-18s %s\n' all mlops "$(meta_version) (meta package: all of the above)"
  exit 0
fi

SELECTED=()
for a in "$@"; do
  if [ "$a" = all ]; then SELECTED=("${PACKAGES[@]}"); META=1; continue; fi
  [[ " ${PACKAGES[*]} " == *" $a "* ]] || { echo "unknown package '$a' (./setup.sh --list)"; exit 1; }
  SELECTED+=("$a")
done

"$PYTHON" -c "import torch" 2> /dev/null || {
  echo "$PYTHON has no PyTorch: install torch first (or set PYTHON to the interpreter that has it)"; exit 1; }
"$PYTHON" -c "import torch; assert torch.version.cuda" 2> /dev/null || { echo "PyTorch without CUDA: a CUDA build of torch is needed"; exit 1; }
"$PYTHON" -c "from torch.utils.cpp_extension import CUDA_HOME; assert CUDA_HOME" 2> /dev/null || {
  echo "CUDA toolkit not found: set CUDA_HOME"; exit 1; }
if [[ " ${SELECTED[*]} " == *" optimizers "* ]]; then
  "$PYTHON" -c "import importlib.util as u; assert u.find_spec('nvfp4effma')" 2> /dev/null || {
    echo "nvfp4effma (needed by optimizers) is not installed: install it first from https://github.com/srose69/nvfp4effma/releases"; exit 1; }
fi

for p in "${SELECTED[@]}"; do
  echo "== mlops-$p $(version "$p")"
  "$PYTHON" -m pip install --no-build-isolation --no-index "./$p/python"
  rm -rf "$p/python/build" "$p"/python/*.egg-info
done
if [ "${META:-0}" = 1 ]; then
  echo "== mlops $(meta_version)"
  "$PYTHON" -m pip install --no-build-isolation --no-index ./meta
  rm -rf meta/build meta/*.egg-info
fi
