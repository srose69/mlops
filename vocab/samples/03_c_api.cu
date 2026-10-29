// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The C API: the LM head for one token (the variants of the shape, as data) and greedy sampling of its logits on a stream.
//
//   nvcc -std=c++17 03_c_api.cu -I ../include -L ../out -lmlops_vocab -o c_api && ./c_api
//   (after `make lib` in vocab/, or with the release's C archive)
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_bf16.h>
#include "mlops_vocab.h"

int main() {
  const int vocab = 4096, hidden = 512;
  std::vector<__nv_bfloat16> hw((size_t)vocab * hidden), hx(hidden);
  for (auto &v : hw) v = __float2bfloat16(rand() / (float)RAND_MAX - 0.5f);
  for (auto &v : hx) v = __float2bfloat16(rand() / (float)RAND_MAX - 0.5f);
  void *w, *x, *logits;
  cudaMalloc(&w, hw.size() * 2); cudaMalloc(&x, hidden * 2); cudaMalloc(&logits, vocab * 4);
  cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(x, hx.data(), hidden * 2, cudaMemcpyHostToDevice);

  mlops_lm_head_config hc = {};
  hc.vocab = vocab; hc.hidden = hidden; hc.n = 1; hc.logit_scale = 1.f;
  mlops_lm_head_variant v[64];
  const int n = mlops_lm_head_variants(&hc, v, 64);   // the instances this device accepts for the shape
  for (int i = 0; i < n && i < 4; ++i)
    printf("variant %d: R %d, S %d, D %d, NB %d, tensor cores %d, %d regs\n", i, v[i].rows_per_unit, v[i].k_pieces, v[i].depth,
           v[i].columns, v[i].tensor_cores, v[i].regs);
  if (mlops_lm_head(&hc, &v[0], (const uint16_t *)w, (const uint16_t *)x, nullptr, (float *)logits, nullptr, nullptr, nullptr, 0)) {
    printf("error: %s\n", mlops_lm_head_last_error());
    return 1;
  }

  mlops_sample_config sc = {};
  sc.vocab = vocab; sc.temperature = 0.f; sc.top_p = 1.f; sc.penalty = 1.f;     // greedy
  mlops_sample_state st = {1234, 0}, *d_state;
  void *ws;
  int *d_token, token = -1;
  const size_t wb = mlops_sample_workspace_bytes(&sc);
  cudaMalloc(&d_state, sizeof st); cudaMemcpy(d_state, &st, sizeof st, cudaMemcpyHostToDevice);
  cudaMalloc(&ws, wb); cudaMemset(ws, 0, wb);                                   // zero before the first call
  cudaMalloc(&d_token, sizeof(int));
  if (mlops_sample(&sc, (const float *)logits, nullptr, nullptr, d_state, ws, wb, d_token, 0)) {
    printf("error: %s\n", mlops_sample_last_error());
    return 1;
  }
  cudaMemcpy(&token, d_token, sizeof token, cudaMemcpyDeviceToHost);
  printf("greedy token: %d\n", token);
  return 0;
}
