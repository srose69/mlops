/* SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
 * Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
 */
/* mlops/optimizers/qat_grouped: the dense grouped GEMMs of the backward of a mixture-of-experts layer whose weights are one stacked NVFP4 tensor
 * (QATGroupedLinear; its forward is nvfp4a_gemm_grouped).  Group g owns the rows seg[g] .. seg[g + 1] of the row-major activations and gradients; seg
 * is a device array of groups + 1 non-decreasing int32 (seg[0] = 0, seg[groups] <= rows), so the row counts never reach the host: no synchronization,
 * the calls can be captured in a CUDA graph.  Stateless C calls over raw device pointers on an explicit cudaStream_t, a status instead of throwing.
 * The products are FP32 FMAs summed over the reduction index in ascending order: bit-reproducible, independent of the other groups and of the GPU.
 * Types and statuses are those of mlops_adamw.h. */
#ifndef MLOPS_QAT_GROUPED_H
#define MLOPS_QAT_GROUPED_H

#include "mlops_adamw.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Input gradient: dx[r][0 .. k) = sum over i < n of gy[r][i] * w[g n + i][0 .. k) for the rows r of group g, dx rounded once to dx_dtype.  gy [rows][n]
 * (BF16 / FP16 / FP32, row stride ldgy), w the dequantized stacked weight [groups * n][k] FP32 (row stride ldw), dx [rows][k] (ldx).  Rows past
 * seg[groups] are not written.  k a multiple of 4 is not required. */
mlops_adamw_status mlops_qat_grouped_dgrad(int groups, int n, int k, int rows, const void *gy, mlops_adamw_dtype gy_dtype, int ldgy,
                                           const float *w, int ldw, void *dx, mlops_adamw_dtype dx_dtype, int ldx, const int *seg,
                                           cudaStream_t stream);

/* Weight gradient (of the dequantized weight): dw[g n + i][j] = sum over the rows r of group g of gy[r][i] * x[r][j], FP32, [groups * n][k] (row stride
 * lddw): every row of dw is written (zeros for a group without rows).  gy [rows][n] (ldgy) and x [rows][k] (ldx) BF16 / FP16 / FP32. */
mlops_adamw_status mlops_qat_grouped_wgrad(int groups, int n, int k, int rows, const void *gy, mlops_adamw_dtype gy_dtype, int ldgy,
                                           const void *x, mlops_adamw_dtype x_dtype, int ldx, float *dw, int lddw, const int *seg,
                                           cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif /* MLOPS_QAT_GROUPED_H */
