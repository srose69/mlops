// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// Device entries of the work items (named in include/mlops_norm_items.h), for an executor's kernels, compiled ahead of time with nvcc
// (-std=c++17 -gencode arch=compute_<arch>,code=sm_<the device's>) or at run time with NVRTC (--std=c++17
// --device-as-default-execution-space, compute_<arch>); include paths: norm's include/, NVFP4EffMA's include/ and src/, the CUDA
// toolkit's include/ (CUDA 13 also include/cccl). Every thread of the described CTA calls the entry (a norm's warps are blockDim.x / 32).
#pragma once
#include "../layer/src/norm.cuh"

namespace mlops_norm_entries {

// KIND, ORDER, GATE: mlops_norm_kind, _order, _gate; XT, YT: mlops_norm_dtype of x / of weight, bias and y; OUT: mlops_norm_out.
template <int KIND, int ORDER, int GATE, int XT, int YT, int OUT>
__device__ __forceinline__ void norm(const mlops_norm_detail::Params &params, int item, unsigned tid, void *smem) {
  using Kernel = mlops_norm_detail::NormKernel<mlops_norm_detail::Traits<KIND, ORDER, GATE, XT, YT, OUT>>;
  Kernel(params, *static_cast<typename Kernel::SharedStorage *>(smem), tid)(item);
}

}  // namespace mlops_norm_entries
