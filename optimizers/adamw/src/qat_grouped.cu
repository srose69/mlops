// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_qat_grouped_dgrad / _wgrad (include/mlops_qat_grouped.h): FP32 shared-memory tiled GEMMs, 64 x 64 tiles, 16-deep k steps, 256 threads of
// 4 x 4 outputs.  Every output is one thread's chain of FMAs over the reduction index in ascending order (zero-padded past the group's rows / n).
#include "common.cuh"
#include "mlops_qat_grouped.h"
#include <climits>

using namespace mlops_adamw_detail;

namespace {

constexpr int kTile = 64, kStep = 16, kBlock = 256;

// A group's tiles: block x of the dgrad grid runs one row tile of the first group whose tiles contain it.
__device__ __forceinline__ bool group_row_tile(const int *seg, int groups, int &g, int &row0, int &row_end) {
  int tile = blockIdx.x;
  for (g = 0; g < groups; ++g) {
    row0 = __ldg(seg + g);
    row_end = __ldg(seg + g + 1);
    const int tiles = (row_end - row0 + kTile - 1) / kTile;
    if (tile < tiles) {
      row0 += tile * kTile;
      return true;
    }
    tile -= tiles;
  }
  return false;
}

__global__ void __launch_bounds__(kBlock) dgrad_kernel(int groups, int n, int k, const void *__restrict__ gy, mlops_adamw_dtype gy_dtype, int ldgy,
                                                       const float *__restrict__ w, int ldw, void *__restrict__ dx, mlops_adamw_dtype dx_dtype,
                                                       int ldx, const int *__restrict__ seg) {
  __shared__ float sg[kStep][kTile + 4], sw[kStep][kTile + 4];
  int g, row0, row_end;
  if (!group_row_tile(seg, groups, g, row0, row_end)) return;
  const int j0 = blockIdx.y * kTile, tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  float acc[4][4] = {};
  for (int i0 = 0; i0 < n; i0 += kStep) {
    // gy tile [kTile rows][kStep]: 16 consecutive threads read one row's 16 values
    for (int e = threadIdx.x; e < kTile * kStep; e += kBlock) {
      const int rr = e / kStep, kk = e % kStep, r = row0 + rr;
      sg[kk][rr] = r < row_end && i0 + kk < n ? mlops_adamw_load_f32(gy, gy_dtype, (size_t)r * ldgy + i0 + kk) : 0.f;
    }
    // w tile [kStep rows of w_g][kTile columns]: 64 consecutive threads read one row
    for (int e = threadIdx.x; e < kTile * kStep; e += kBlock) {
      const int kk = e / kTile, cc = e % kTile;
      sw[kk][cc] = i0 + kk < n && j0 + cc < k ? w[((size_t)g * n + i0 + kk) * ldw + j0 + cc] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kStep; ++kk)
#pragma unroll
      for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = fmaf(sg[kk][ty * 4 + a], sw[kk][tx * 4 + b], acc[a][b]);
    __syncthreads();
  }
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const int r = row0 + ty * 4 + a, j = j0 + tx * 4 + b;
      if (r < row_end && j < k) mlops_adamw_store_f32(dx, dx_dtype, (size_t)r * ldx + j, acc[a][b]);
    }
}

__global__ void __launch_bounds__(kBlock) wgrad_kernel(int n, int k, const void *__restrict__ gy, mlops_adamw_dtype gy_dtype, int ldgy,
                                                       const void *__restrict__ x, mlops_adamw_dtype x_dtype, int ldx, float *__restrict__ dw,
                                                       int lddw, const int *__restrict__ seg) {
  __shared__ float sg[kStep][kTile + 4], sx[kStep][kTile + 4];
  const int g = blockIdx.y, ti = (n + kTile - 1) / kTile;
  const int i0 = blockIdx.x % ti * kTile, j0 = blockIdx.x / ti * kTile, tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  const int r_begin = __ldg(seg + g), r_end = __ldg(seg + g + 1);
  float acc[4][4] = {};
  for (int r0 = r_begin; r0 < r_end; r0 += kStep) {
    for (int e = threadIdx.x; e < kTile * kStep; e += kBlock) {
      const int kk = e / kTile, cc = e % kTile, r = r0 + kk;
      sg[kk][cc] = r < r_end && i0 + cc < n ? mlops_adamw_load_f32(gy, gy_dtype, (size_t)r * ldgy + i0 + cc) : 0.f;
      sx[kk][cc] = r < r_end && j0 + cc < k ? mlops_adamw_load_f32(x, x_dtype, (size_t)r * ldx + j0 + cc) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kStep; ++kk)
#pragma unroll
      for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = fmaf(sg[kk][ty * 4 + a], sx[kk][tx * 4 + b], acc[a][b]);
    __syncthreads();
  }
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const int i = i0 + ty * 4 + a, j = j0 + tx * 4 + b;
      if (i < n && j < k) dw[((size_t)g * n + i) * lddw + j] = acc[a][b];
    }
}

bool dense_dtype(mlops_adamw_dtype t) { return t == MLOPS_ADAMW_BF16 || t == MLOPS_ADAMW_FP16 || t == MLOPS_ADAMW_FP32; }

}  // namespace

extern "C" mlops_adamw_status mlops_qat_grouped_dgrad(int groups, int n, int k, int rows, const void *gy, mlops_adamw_dtype gy_dtype, int ldgy,
                                                      const float *w, int ldw, void *dx, mlops_adamw_dtype dx_dtype, int ldx, const int *seg,
                                                      cudaStream_t stream) {
  if (!gy || !w || !dx || !seg) return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_qat_grouped_dgrad: null pointer argument");
  if (groups < 0 || n < 0 || k < 0 || rows < 0) return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_qat_grouped_dgrad: negative size");
  if (!dense_dtype(gy_dtype) || !dense_dtype(dx_dtype))
    return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_qat_grouped_dgrad: gy and dx must be BF16, FP16 or FP32");
  if (ldgy < n || ldw < k || ldx < k) return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_qat_grouped_dgrad: a row stride is below its row length");
  if (groups == 0 || n == 0 || k == 0 || rows == 0) return MLOPS_ADAMW_OK_();
  const long long tiles = (long long)((rows + kTile - 1) / kTile + groups);
  if (tiles > INT_MAX) return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_qat_grouped_dgrad: too many rows");
  dgrad_kernel<<<dim3((unsigned)tiles, (unsigned)((k + kTile - 1) / kTile)), kBlock, 0, stream>>>(groups, n, k, gy, gy_dtype, ldgy, w, ldw, dx,
                                                                                                  dx_dtype, ldx, seg);
  MLOPS_ADAMW_CUDA_CHECK(cudaGetLastError());
  return MLOPS_ADAMW_OK_();
}

extern "C" mlops_adamw_status mlops_qat_grouped_wgrad(int groups, int n, int k, int rows, const void *gy, mlops_adamw_dtype gy_dtype, int ldgy,
                                                      const void *x, mlops_adamw_dtype x_dtype, int ldx, float *dw, int lddw, const int *seg,
                                                      cudaStream_t stream) {
  if (!gy || !x || !dw || !seg) return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_qat_grouped_wgrad: null pointer argument");
  if (groups < 0 || n < 0 || k < 0 || rows < 0) return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_qat_grouped_wgrad: negative size");
  if (!dense_dtype(gy_dtype) || !dense_dtype(x_dtype))
    return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_qat_grouped_wgrad: gy and x must be BF16, FP16 or FP32");
  if (ldgy < n || ldx < k || lddw < k) return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_qat_grouped_wgrad: a row stride is below its row length");
  if (groups == 0 || n == 0 || k == 0) return MLOPS_ADAMW_OK_();
  const long long tiles = (long long)((n + kTile - 1) / kTile) * ((k + kTile - 1) / kTile);
  if (tiles > INT_MAX || groups > 65535) return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_qat_grouped_wgrad: n * k or groups too large");
  wgrad_kernel<<<dim3((unsigned)tiles, (unsigned)groups), kBlock, 0, stream>>>(n, k, gy, gy_dtype, ldgy, x, x_dtype, ldx, dw, lddw, seg);
  MLOPS_ADAMW_CUDA_CHECK(cudaGetLastError());
  return MLOPS_ADAMW_OK_();
}
