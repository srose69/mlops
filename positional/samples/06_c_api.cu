// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The C API: rotate q and k ([B, S, H, D], BF16) from positions, then back (the backward), on a stream.
//
//   nvcc -std=c++17 06_c_api.cu -I ../include -L ../out -lmlops_positional -o c_api && ./c_api
//   (after `make lib` in positional/, or with the release's C archive)
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_bf16.h>
#include "mlops_positional.h"

int main() {
  const int B = 2, S = 8, Hq = 4, Hk = 2, D = 64, R = D;
  std::vector<__nv_bfloat16> hq((size_t)B * S * Hq * D);
  for (size_t i = 0; i < hq.size(); ++i) hq[i] = __float2bfloat16(sinf(0.1f * i));
  void *q, *k;
  cudaMalloc(&q, hq.size() * 2);
  cudaMalloc(&k, (size_t)B * S * Hk * D * 2);
  cudaMemcpy(q, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice);
  cudaMemset(k, 0, (size_t)B * S * Hk * D * 2);

  // frequencies (any scaling: compute inv_freq on the host, e.g. YaRN's), as the kernel's table
  std::vector<double> inv(R / 2);
  for (int i = 0; i < R / 2; ++i) inv[i] = pow(10000.0, -2.0 * i / R);
  std::vector<float> table(4 * R / 2);
  mlops_rope_freq(inv.data(), R / 2, table.data());
  float *freq;
  cudaMalloc(&freq, table.size() * 4);
  cudaMemcpy(freq, table.data(), table.size() * 4, cudaMemcpyHostToDevice);

  // [B, S, H, D] contiguous: strides in elements
  const mlops_rope_tensor tq = {q, Hq, (int64_t)S * Hq * D, Hq * D, D}, tk = {k, Hk, (int64_t)S * Hk * D, Hk * D, D};
  mlops_rope_config cfg = {};
  cfg.batch = B; cfg.seq = S; cfg.head_dim = D; cfg.rotary_dim = R; cfg.dtype = MLOPS_ROPE_BF16;
  cfg.freq = freq; cfg.position_offset = 1000;  // positions 1000 + s (or cfg.positions: an int32 / int64 [B, S] array)

  if (mlops_rope(&cfg, &tq, &tq, &tk, &tk, 0) != MLOPS_ROPE_OK) { printf("error: %s\n", mlops_rope_last_error()); return 1; }
  cfg.inverse = 1;                              // rotate back: the backward
  mlops_rope(&cfg, &tq, &tq, &tk, &tk, 0);
  std::vector<__nv_bfloat16> back(hq.size());
  cudaMemcpy(back.data(), q, back.size() * 2, cudaMemcpyDeviceToHost);
  float worst = 0;
  for (size_t i = 0; i < hq.size(); ++i)
    worst = fmaxf(worst, fabsf(__bfloat162float(back[i]) - __bfloat162float(hq[i])));
  printf("rotate + inverse: max deviation %.4f (two BF16 roundings)\n", worst);
  // misuse is reported, not crashed on:
  cfg.rotary_dim = 63;
  printf("rotary_dim 63: %s -- %s\n", mlops_rope_status_string(mlops_rope(&cfg, &tq, &tq, nullptr, nullptr, 0)),
         mlops_rope_last_error());
  return 0;
}
