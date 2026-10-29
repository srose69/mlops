// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE, LICENSE-EXCEPTION (additional terms).
// The device entries of the work items (include/mlops_fbla_items.h names them and their argument types), for an
// executor's kernels. NVRTC: --std=c++17 --device-as-default-execution-space, include paths: attention's include/
// and the CUDA toolkit's include/ (CUDA 13 also include/cccl). Compile for the plan's compute_<arch> and JIT on the
// current device.  Every thread of the described CTA calls the entry with the executor's own position policy
// (fbla_kernels.cuh's Grid is the launches'; x / y / z, nx / ny / nz, tid and the device length's read, ro, are
// its interface) and the plan's geom[0..] as the entry's own arguments after the args blob; synchronize the CTA
// before reusing the entry's shared memory.
#pragma once
#include "../fbla/src/fbla_kernels.cuh"
#include "../fbla/src/fbla_prefill.cuh"
#include "../fbla/src/fbla_qkv.cuh"
