// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The C API: list the variants of a shape, run RMSNorm with a residual and an INT8 output on a stream, read the result.
//
//   nvcc -std=c++17 04_c_api.cu -I ../include -L ../out -lmlops_norm -o c_api && ./c_api
//   (after `make lib` in norm/, or with the release's C archive)
#include <cstdio>
#include <vector>
#include <cuda_bf16.h>
#include "mlops_norm.h"

int main() {
  const int rows = 4, hidden = 1024;
  std::vector<__nv_bfloat16> hx((size_t)rows * hidden), hw(hidden);
  for (size_t i = 0; i < hx.size(); ++i) hx[i] = __float2bfloat16(sinf(0.01f * i));
  for (int i = 0; i < hidden; ++i) hw[i] = __float2bfloat16(1.0f);
  void *x, *res, *res_out, *w, *y, *scales;
  cudaMalloc(&x, hx.size() * 2); cudaMalloc(&res, hx.size() * 2); cudaMalloc(&res_out, hx.size() * 2);
  cudaMalloc(&w, hidden * 2); cudaMalloc(&y, (size_t)rows * hidden); cudaMalloc(&scales, (size_t)rows * hidden / 16 * 4);
  cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(res, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(w, hw.data(), hidden * 2, cudaMemcpyHostToDevice);

  mlops_norm_config c = {};
  c.rows = rows; c.hidden = hidden; c.eps = 1e-6f;
  c.kind = MLOPS_NORM_RMS; c.scale = MLOPS_NORM_SCALE_WEIGHT; c.order = MLOPS_NORM_ORDER_ROUND_THEN_SCALE;
  c.x_dtype = c.y_dtype = MLOPS_NORM_BF16; c.out = MLOPS_NORM_OUT_INT8;
  mlops_norm_variant v[8];
  const int n = mlops_norm_variants(&c, v, 8);        // the instances for this shape: warps, threads, shared memory, registers
  for (int i = 0; i < n; ++i) printf("variant %d: %d warps, %d registers\n", i, v[i].warps, v[i].regs);

  mlops_norm_args a = {};
  a.x = x; a.residual = res; a.residual_out = res_out; a.weight = w; a.y = y; a.scales = scales;
  if (mlops_norm(&c, &v[n - 1], &a, 0) != MLOPS_NORM_OK) { printf("error: %s\n", mlops_norm_last_error()); return 1; }

  std::vector<signed char> hy((size_t)rows * hidden);
  std::vector<float> hs((size_t)rows * hidden / 16);
  cudaMemcpy(hy.data(), y, hy.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hs.data(), scales, hs.size() * 4, cudaMemcpyDeviceToHost);
  printf("row 0: q[0..3] = %d %d %d %d, scale[0] = %g\n", hy[0], hy[1], hy[2], hy[3], hs[0]);

  a.x = nullptr;                                      // an invalid call reports why
  if (mlops_norm(&c, &v[0], &a, 0) != MLOPS_NORM_OK) printf("rejected: %s\n", mlops_norm_last_error());
  return 0;
}
