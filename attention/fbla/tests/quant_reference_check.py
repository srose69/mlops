#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""The python reference's q-hat (fbla_reference._quantize_q) against the C++ definition's codes, both on the inputs
tc_fn_bench writes with MLOPS_QUANT_DUMP=<file> (its quant cases: q [case][16][128] BF16, K steps [case][128] BF16,
sl [case], the definition's q-hat [case][16][128] int8).  Prints the codes that differ (0 expected)."""
import importlib.util
import os
import sys

import numpy as np
import torch

# the module alone (the package's __init__ loads the compiled extension)
_spec = importlib.util.spec_from_file_location(
    "fbla_reference", os.path.join(os.path.dirname(__file__), "..", "..", "python", "mlops", "attention", "fbla_reference.py"))
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
_quantize_q = _ref._quantize_q


def bf16(u16):
    return torch.from_numpy((u16.astype(np.uint32) << 16).view(np.float32))


def main(path):
    raw = open(path, "rb").read()
    nc = int(np.frombuffer(raw, np.int32, 1)[0])
    off = 4
    q = np.frombuffer(raw, np.uint16, nc * 16 * 128, off).reshape(nc, 16, 128); off += q.nbytes
    st = np.frombuffer(raw, np.uint16, nc * 128, off).reshape(nc, 1, 128); off += st.nbytes
    sl = np.frombuffer(raw, np.float32, nc, off); off += sl.nbytes
    ref = np.frombuffer(raw, np.int8, nc * 16 * 128, off).reshape(nc, 16, 128)
    qh, sq = _quantize_q(bf16(q) * bf16(st), 8, torch.from_numpy(sl.copy()).view(nc, 1, 1))
    diff = int((qh.numpy().astype(np.int8) != ref).sum())
    print(f"python reference: {nc} cases x 16 rows x 128 channels, q-hat codes differing from the C++ definition: {diff}")
    return diff == 0


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
