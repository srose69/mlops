// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE, LICENSE-EXCEPTION (additional terms).
// Device entries of the work items (named in include/mlops_fbla_items.h), for an executor's kernels, compiled ahead of time with nvcc (-std=c++17 -gencode
// arch=compute_<arch>,code=sm_<the device's>) or at run time with NVRTC (--std=c++17 --device-as-default-execution-space, compute_<arch>, the PTX JIT-compiled on the
// device); include paths: attention's include/, positional's include/ and the CUDA toolkit's include/ (CUDA 13 also include/cccl). Every thread of the
// described CTA calls the entry with the executor's position policy (fbla_kernels.cuh's Grid is the launches'; x / y / z, nx / ny / nz, tid, ro, the device length's read) and geom[0..].
#pragma once
#include "../fbla/src/fbla_kernels.cuh"
#include "../fbla/src/fbla_prefill.cuh"
#include "../fbla/src/fbla_qkv.cuh"
