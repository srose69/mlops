// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// One qat_step_kernel instance: MLOPS_QAT_GT = gradient dtype code (mlops_adamw_dtype: E4M3, E5M2 or FP32).
#include "adamw_qat_kernels.cuh"

#ifndef MLOPS_QAT_GT
#error "build with -DMLOPS_QAT_GT=<grad dtype> (optimizers/Makefile)"
#endif

namespace mlops_qat_detail {

template <int GT>
void launch_step(const Lattice &L, const Hyper &h, const void *grad, const mlops_adamw_qat_state &st, int grid,
                 cudaStream_t stream) {
  qat_step_kernel<GT><<<grid, 256, 0, stream>>>(L, h, grad, st);
}

template void launch_step<MLOPS_QAT_GT>(const Lattice &, const Hyper &, const void *, const mlops_adamw_qat_state &, int,
                                        cudaStream_t);

}  // namespace mlops_qat_detail
