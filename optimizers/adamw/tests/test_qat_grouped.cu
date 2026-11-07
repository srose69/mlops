// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
/* mlops_qat_grouped_dgrad / _wgrad against their host emulation, bit for bit: every output is a chain of FP32 FMAs over the reduction index in ascending
 * order (include/mlops_qat_grouped.h), the dgrad output rounded once to its dtype.  Every dtype of gy / x / dx, groups without rows, n and k that are not
 * tile multiples, rows past seg[groups] (dgrad leaves them alone, here poisoned), and the argument checks. */
#include "mlops_qat_grouped.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(expr)                                                                                   \
  do {                                                                                              \
    cudaError_t _e = (expr);                                                                        \
    if (_e != cudaSuccess) {                                                                        \
      fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(_e)); \
      exit(1);                                                                                      \
    }                                                                                               \
  } while (0)

static size_t esize(mlops_adamw_dtype t) { return t == MLOPS_ADAMW_FP32 ? 4 : 2; }
static float load(mlops_adamw_dtype t, const void *p, size_t i) {
  switch (t) {
    case MLOPS_ADAMW_BF16: return __bfloat162float(((const __nv_bfloat16 *)p)[i]);
    case MLOPS_ADAMW_FP16: return __half2float(((const __half *)p)[i]);
    default: return ((const float *)p)[i];
  }
}
static void store(mlops_adamw_dtype t, void *p, size_t i, float v) {
  switch (t) {
    case MLOPS_ADAMW_BF16: ((__nv_bfloat16 *)p)[i] = __float2bfloat16_rn(v); break;
    case MLOPS_ADAMW_FP16: ((__half *)p)[i] = __float2half_rn(v); break;
    default: ((float *)p)[i] = v;
  }
}
template <class T> static T *to_device(const std::vector<T> &h) {
  T *d;
  CK(cudaMalloc(&d, h.size() * sizeof(T) + 16));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

static int run(int groups, int n, int k, const std::vector<int> &counts, int tail, mlops_adamw_dtype gt, mlops_adamw_dtype xt, mlops_adamw_dtype dt) {
  std::mt19937 rng(1234 + n + k);
  std::normal_distribution<float> normal;
  std::vector<int> seg(groups + 1, 0);
  for (int g = 0; g < groups; ++g) seg[g + 1] = seg[g] + counts[g];
  const int used = seg[groups], rows = used + tail;
  std::vector<uint8_t> gy(rows * (size_t)n * esize(gt)), x(rows * (size_t)k * esize(xt)), dx(rows * (size_t)k * esize(dt), 0xA5);
  std::vector<float> w((size_t)groups * n * k);
  for (size_t i = 0; i < (size_t)rows * n; ++i) store(gt, gy.data(), i, normal(rng));
  for (size_t i = 0; i < (size_t)rows * k; ++i) store(xt, x.data(), i, normal(rng));
  for (float &v : w) v = normal(rng);
  void *dgy = to_device(gy), *dxx = to_device(x), *ddx = to_device(dx);
  float *dw = to_device(w), *ddw;
  int *dseg = to_device(seg);
  CK(cudaMalloc(&ddw, (size_t)groups * n * k * 4 + 16));
  CK(cudaMemset(ddw, 0xFF, (size_t)groups * n * k * 4));
  if (mlops_qat_grouped_dgrad(groups, n, k, rows, dgy, gt, n, dw, k, ddx, dt, k, dseg, 0) != MLOPS_ADAMW_OK ||
      mlops_qat_grouped_wgrad(groups, n, k, rows, dgy, gt, n, dxx, xt, k, ddw, k, dseg, 0) != MLOPS_ADAMW_OK) {
    printf("  call failed: %s\n", mlops_adamw_last_error());
    return 1;
  }
  std::vector<uint8_t> got_dx(dx.size());
  std::vector<float> got_dw((size_t)groups * n * k);
  CK(cudaMemcpy(got_dx.data(), ddx, got_dx.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(got_dw.data(), ddw, got_dw.size() * 4, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int g = 0; g < groups; ++g)
    for (int r = seg[g]; r < seg[g + 1]; ++r)
      for (int j = 0; j < k; ++j) {
        float acc = 0.f;
        for (int i = 0; i < n; ++i) acc = fmaf(load(gt, gy.data(), (size_t)r * n + i), w[((size_t)g * n + i) * k + j], acc);
        std::vector<uint8_t> one(4, 0);
        store(dt, one.data(), 0, acc);
        if (memcmp(one.data(), &got_dx[((size_t)r * k + j) * esize(dt)], esize(dt))) ++bad;
      }
  for (size_t b = (size_t)used * k * esize(dt); b < dx.size(); ++b) bad += got_dx[b] != 0xA5;  // rows past seg[groups]
  for (int g = 0; g < groups; ++g)
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < k; ++j) {
        float acc = 0.f;
        for (int r = seg[g]; r < seg[g + 1]; ++r) acc = fmaf(load(gt, gy.data(), (size_t)r * n + i), load(xt, x.data(), (size_t)r * k + j), acc);
        const float v = got_dw[((size_t)g * n + i) * k + j];
        bad += memcmp(&acc, &v, 4) != 0;
      }
  for (void *p : {dgy, dxx, ddx, (void *)dw, (void *)ddw, (void *)dseg}) CK(cudaFree(p));
  return bad;
}

int main() {
  const mlops_adamw_dtype T[] = {MLOPS_ADAMW_FP32, MLOPS_ADAMW_BF16, MLOPS_ADAMW_FP16};
  const char *name[] = {"FP32", "BF16", "FP16"};
  int failed = 0;
  struct Shape { int groups, n, k; std::vector<int> counts; int tail; };
  const std::vector<Shape> shapes = {{5, 96, 128, {0, 17, 64, 1, 90}, 7}, {3, 130, 70, {33, 0, 129}, 0}, {1, 64, 64, {200}, 3}, {4, 70, 130, {0, 0, 5, 0}, 2}};
  for (const Shape &s : shapes)
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        const int bad = run(s.groups, s.n, s.k, s.counts, s.tail, T[a], T[b], T[(a + b) % 3]);
        printf("groups %d n %3d k %3d gy %s x %s dx %s: %s\n", s.groups, s.n, s.k, name[a], name[b], name[(a + b) % 3], bad ? "FAILED" : "ok");
        failed += bad != 0;
      }
  // argument checks
  float f;
  int seg[2] = {0, 1};
  if (mlops_qat_grouped_dgrad(1, 4, 4, 1, nullptr, MLOPS_ADAMW_FP32, 4, &f, 4, &f, MLOPS_ADAMW_FP32, 4, seg, 0) != MLOPS_ADAMW_INVALID_VALUE ||
      mlops_qat_grouped_wgrad(1, 4, 4, 1, &f, MLOPS_ADAMW_E5M2, 4, &f, MLOPS_ADAMW_FP32, 4, &f, 4, seg, 0) != MLOPS_ADAMW_INVALID_VALUE ||
      mlops_qat_grouped_dgrad(1, 4, 4, 1, &f, MLOPS_ADAMW_FP32, 3, &f, 4, &f, MLOPS_ADAMW_FP32, 4, seg, 0) != MLOPS_ADAMW_INVALID_VALUE) {
    printf("argument checks: FAILED\n");
    ++failed;
  } else {
    printf("argument checks: ok\n");
  }
  printf("%s\n", failed ? "FAILED" : "ALL PASS");
  return failed != 0;
}
