// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The device entries of the work items (include/mlops_norm_items.h names them and their argument types), for an
// executor's kernels. NVRTC: --std=c++17 --device-as-default-execution-space, include paths: norm's include/, NVFP4EffMA's
// include/ and src/ (the quantizer's primitives are imported from it), and the CUDA toolkit's include/ (CUDA 13 also
// include/cccl). Every thread of the described CTA calls the entry (the CTA is the described number of threads: the warps of
// a norm are blockDim.x / 32), and the CTA is synchronized before the entry's shared memory is reused.
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
