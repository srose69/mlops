// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// One adamw_kernel_vec4 instance: MLOPS_PDT / MLOPS_GDT = param / grad dtype codes (mlops_adamw_dtype).
#include "adamw_kernels.cuh"

#if !defined(MLOPS_PDT) || !defined(MLOPS_GDT)
#error "build with -DMLOPS_PDT=<param dtype> -DMLOPS_GDT=<grad dtype> (optimizers/Makefile)"
#endif

namespace mlops_adamw_detail {

template <mlops_adamw_dtype PDT, mlops_adamw_dtype GDT>
void launch_vec4(const AdamArgs &a, size_t count4, void *param, const void *grad, float *exp_avg, float *exp_avg_sq,
                 cudaStream_t stream) {
  adamw_kernel_vec4<PDT, GDT><<<grid_for(count4), kThreads, 0, stream>>>(
      a, count4, param, grad, reinterpret_cast<float4 *>(exp_avg), reinterpret_cast<float4 *>(exp_avg_sq));
}

template void launch_vec4<(mlops_adamw_dtype)MLOPS_PDT, (mlops_adamw_dtype)MLOPS_GDT>(
    const AdamArgs &, size_t, void *, const void *, float *, float *, cudaStream_t);

} // namespace mlops_adamw_detail
