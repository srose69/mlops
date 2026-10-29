// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_lm_head against a host emulation of its arithmetic (lane order, butterfly, k-pieces in warp order):
// logits bit for bit for every variant, the per-element statistics, the lowest index among equal maxima.
// `test_lm_head bench`: every variant at Qwen3-8B's head against a device-to-device copy of the same bytes.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_fp16.h>

#include "mlops_lm_head.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static std::mt19937 rng(4321);

static uint16_t bf16(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float f32(uint16_t v) {
  const uint32_t u = (uint32_t)v << 16;
  float x;
  memcpy(&x, &u, 4);
  return x;
}
template <class T> static T *up(const std::vector<T> &h) {
  T *d;
  CK(cudaMalloc(&d, h.size() * sizeof(T) + 16));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
template <class T> static std::vector<T> down(const T *d, size_t n) {
  std::vector<T> h(n);
  CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

// The sum of row `row`'s k-piece [p0, p1) as one warp forms it.
static float warp_sum(const std::vector<uint16_t> &w, const std::vector<uint16_t> &x, int hidden, int row, int p0, int p1) {
  float acc[32] = {};
  for (int lane = 0; lane < 32; ++lane)
    for (int p = p0; p < p1; ++p)
      for (int e = 0; e < 8; ++e) {
        const int k = 256 * p + 8 * lane + e;
        acc[lane] = fmaf(f32(w[(size_t)row * hidden + k]), f32(x[k]), acc[lane]);
      }
  for (int o = 16; o; o >>= 1) {
    float nxt[32];
    for (int l = 0; l < 32; ++l) nxt[l] = acc[l] + acc[l ^ o];
    memcpy(acc, nxt, sizeof acc);
  }
  return acc[0];
}

// The exact (double) row sums and the sums of the absolute products, for the tolerance of the tensor-core path.
static float through_f16(uint16_t v) { return __half2float(__float2half_rn(f32(v))); }
static void exact_rows(const std::vector<uint16_t> &w, const std::vector<uint16_t> &x, int vocab, int hidden, int n_cols, bool f16,
                       std::vector<double> &sum, std::vector<double> &mag) {
  sum.assign((size_t)n_cols * vocab, 0.);
  mag.assign((size_t)n_cols * vocab, 0.);
  for (int col = 0; col < n_cols; ++col)
    for (int row = 0; row < vocab; ++row) {
      double s = 0., m = 0.;
      for (int k = 0; k < hidden; ++k) {
        const double p = f16 ? (double)through_f16(w[(size_t)row * hidden + k]) * through_f16(x[(size_t)col * hidden + k])
                             : (double)f32(w[(size_t)row * hidden + k]) * f32(x[(size_t)col * hidden + k]);
        s += p;
        m += std::fabs(p);
      }
      sum[(size_t)col * vocab + row] = s;
      mag[(size_t)col * vocab + row] = m;
    }
}

static float host_scale(const std::vector<float> &parts, float inv_n, float eps) {
  float lane[32] = {};
  for (size_t i = 0; i < parts.size(); ++i) lane[i % 32] += parts[i];
  for (int o = 16; o; o >>= 1) {
    float nxt[32];
    for (int l = 0; l < 32; ++l) nxt[l] = lane[l] + lane[l ^ o];
    memcpy(lane, nxt, sizeof lane);
  }
  const float v = lane[0] * inv_n + eps;
  return (float)(1.0 / sqrt((double)v));
}

static int report(const char *what, long bad, long n) {
  printf("[%s] %s (%ld of %ld differ)\n", bad ? "FAIL" : "PASS", what, bad, n);
  return bad != 0;
}

static int run_case(int vocab, int hidden, int n_cols, int items, bool norm) {
  std::normal_distribution<float> dw(0.f, 0.05f), dx(0.f, 1.f);
  std::vector<uint16_t> w((size_t)vocab * hidden), x((size_t)n_cols * hidden);
  for (auto &v : w) v = bf16(dw(rng));
  for (auto &v : x) v = bf16(std::fabs(dx(rng)) + 0.1f);
  const int a = vocab / 3, b = vocab / 2;  // two equal maxima: the lower index wins
  for (int k = 0; k < hidden; ++k) w[(size_t)a * hidden + k] = w[(size_t)b * hidden + k] = bf16(0.9f);
  std::vector<float> parts;
  const int np = 37;
  if (norm)
    for (int i = 0; i < n_cols * np; ++i) parts.push_back(0.5f + (float)(rng() % 1000) / 100.f);
  const float inv_n = 1.f / hidden, eps = 1e-6f;
  std::vector<float> scale(n_cols, 1.f);
  if (norm)
    for (int c = 0; c < n_cols; ++c) scale[c] = host_scale(std::vector<float>(parts.begin() + c * np, parts.begin() + (c + 1) * np), inv_n, eps);

  const mlops_lm_head_config c{vocab, hidden, n_cols, items, norm ? np : 0, inv_n, eps, 1.f};
  const int nv = mlops_lm_head_variants(&c, nullptr, 0);
  std::vector<mlops_lm_head_variant> vs(nv);
  mlops_lm_head_variants(&c, vs.data(), nv);
  uint16_t *dw_ = up(w), *dx_ = up(x);
  float *dparts = norm ? up(parts) : nullptr, *dlogits;
  CK(cudaMalloc(&dlogits, (size_t)n_cols * vocab * 4));
  long bad = 0, n = 0, bad_stats = 0, n_stats = 0, bad_tc = 0, n_tc = 0, bad_hist = 0, n_hist = 0;
  std::vector<std::vector<float>> by_pieces(9);  // the arithmetic depends on S only
  std::vector<double> exact_sum[2], exact_mag[2];  // BF16 operands, and the FP16-converted ones
  for (const auto &v : vs) {
    const int ni = mlops_lm_head_item_count(&c, &v);
    float *dmax;
    int *darg;
    unsigned *dhist;
    CK(cudaMalloc(&dmax, (size_t)n_cols * ni * 4));
    CK(cudaMalloc(&darg, (size_t)n_cols * ni * 4));
    CK(cudaMalloc(&dhist, (size_t)n_cols * ni * 256 * 4));
    CK(cudaMemset(dlogits, 0xff, (size_t)n_cols * vocab * 4));
    if (mlops_lm_head(&c, &v, dw_, dx_, dparts, dlogits, dmax, darg, dhist, 0)) {
      fprintf(stderr, "%s\n", mlops_lm_head_last_error());
      return 1;
    }
    CK(cudaDeviceSynchronize());
    const auto logits = down(dlogits, (size_t)n_cols * vocab), smax = down(dmax, (size_t)n_cols * ni);
    const auto sarg = down(darg, (size_t)n_cols * ni);
    {  // the key-byte counts of each element's rows, from the logits the call itself wrote
      const auto hist = down(dhist, (size_t)n_cols * ni * 256);
      const long long hunits = (vocab + v.rows_per_unit - 1) / v.rows_per_unit;
      for (int col = 0; col < n_cols; ++col)
        for (int it = 0; it < ni; ++it) {
          const int lo = (int)(it * hunits / ni) * v.rows_per_unit, hi = std::min((int)(((it + 1) * hunits / ni) * v.rows_per_unit), vocab);
          std::vector<unsigned> want(256, 0u);
          for (int row = lo; row < hi; ++row) {
            uint32_t u;
            memcpy(&u, &logits[(size_t)col * vocab + row], 4);
            ++want[((u & 0x80000000u ? ~u : u | 0x80000000u) >> 24)];
          }
          for (int b = 0; b < 256; ++b) { bad_hist += hist[((size_t)col * ni + it) * 256 + b] != want[b]; ++n_hist; }
        }
    }
    const int passes = hidden / 256;
    if (v.tensor_cores) {  // FP32 accumulation in the tensor cores' own order: within a relative bound of the exact sum
      const int fi = v.tensor_cores >= 2;
      if (exact_sum[fi].empty()) exact_rows(w, x, vocab, hidden, n_cols, fi, exact_sum[fi], exact_mag[fi]);
      for (int col = 0; col < n_cols; ++col)
        for (int row = 0; row < vocab; ++row) {
          const size_t i = (size_t)col * vocab + row;
          // Each mma adds its 16 products to the accumulator with a truncation of 2^-23 of the partial sum at worst, and the
          // S pieces are added once more: the bound grows with the number of adds, not with the sum alone.
          const double bound = (hidden / (v.tensor_cores == 1 ? 16 : v.tensor_cores == 2 ? 8 : 4) + 8) * 0x1p-23 * exact_mag[fi][i] * scale[col];
          bad_tc += std::fabs((double)logits[i] - exact_sum[fi][i] * scale[col]) > bound + 1e-30;
          ++n_tc;
        }
      float *dl2;
      CK(cudaMalloc(&dl2, (size_t)n_cols * vocab * 4));
      mlops_lm_head_config c2 = c;
      c2.items = 3;  // the element count does not change the bits
      if (mlops_lm_head(&c2, &v, dw_, dx_, dparts, dl2, nullptr, nullptr, nullptr, 0)) return 1;
      const auto again = down(dl2, (size_t)n_cols * vocab);
      CK(cudaFree(dl2));
      bad_tc += memcmp(again.data(), logits.data(), again.size() * 4) != 0;
      ++n_tc;
      for (int col = 0; col < n_cols; ++col)
        for (int it = 0; it < ni; ++it) {
          const long long units = (vocab + v.rows_per_unit - 1) / v.rows_per_unit;
          const int lo = (int)(it * units / ni) * v.rows_per_unit, hi = std::min((int)(((it + 1) * units / ni) * v.rows_per_unit), vocab);
          float bm = -INFINITY;
          int ba = 0x7fffffff;
          for (int row = lo; row < hi; ++row)
            if (logits[(size_t)col * vocab + row] > bm) { bm = logits[(size_t)col * vocab + row]; ba = row; }
          bad_stats += (smax[(size_t)col * ni + it] != bm && !(std::isinf(bm) && std::isinf(smax[(size_t)col * ni + it]))) || sarg[(size_t)col * ni + it] != ba;
          ++n_stats;
        }
      CK(cudaFree(dmax));
      CK(cudaFree(darg));
      continue;
    }
    std::vector<float> &ref = by_pieces[v.k_pieces];
    if (ref.empty()) {
      ref.resize((size_t)n_cols * vocab);
      for (int col = 0; col < n_cols; ++col) {
        const std::vector<uint16_t> xc(x.begin() + (size_t)col * hidden, x.begin() + (size_t)(col + 1) * hidden);
        for (int row = 0; row < vocab; ++row) {
          float sum = warp_sum(w, xc, hidden, row, 0, passes / v.k_pieces);
          for (int p = 1; p < v.k_pieces; ++p) sum = sum + warp_sum(w, xc, hidden, row, p * passes / v.k_pieces, (p + 1) * passes / v.k_pieces);
          ref[(size_t)col * vocab + row] = sum * scale[col];
        }
      }
    }
    for (size_t i = 0; i < ref.size(); ++i) {
      uint32_t u0, u1;
      memcpy(&u0, &logits[i], 4);
      memcpy(&u1, &ref[i], 4);
      bad += u0 != u1;
      ++n;
    }
    const long long units = (vocab + v.rows_per_unit - 1) / v.rows_per_unit;
    for (int col = 0; col < n_cols; ++col)
      for (int it = 0; it < ni; ++it) {
        const int lo = (int)(it * units / ni) * v.rows_per_unit, hi = std::min((int)(((it + 1) * units / ni) * v.rows_per_unit), vocab);
        float bm = -INFINITY;
        int ba = 0x7fffffff;
        for (int row = lo; row < hi; ++row)
          if (ref[(size_t)col * vocab + row] > bm) { bm = ref[(size_t)col * vocab + row]; ba = row; }
        bad_stats += (smax[(size_t)col * ni + it] != bm && !(std::isinf(bm) && std::isinf(smax[(size_t)col * ni + it]))) || sarg[(size_t)col * ni + it] != ba;
        ++n_stats;
      }
    CK(cudaFree(dmax));
    CK(cudaFree(darg));
  }
  char what[160];
  snprintf(what, sizeof what, "lm_head %d x %d, n %d, items %d%s: %d variants, logits", vocab, hidden, n_cols, items, norm ? " norm" : "", nv);
  int f = report(what, bad, n);
  if (n_tc) {
    snprintf(what, sizeof what, "lm_head %d x %d, n %d, items %d%s: tensor cores, bound and repeat", vocab, hidden, n_cols, items, norm ? " norm" : "");
    f += report(what, bad_tc, n_tc);
  }
  snprintf(what, sizeof what, "lm_head %d x %d, n %d, items %d%s: statistics", vocab, hidden, n_cols, items, norm ? " norm" : "");
  f += report(what, bad_stats, n_stats);
  snprintf(what, sizeof what, "lm_head %d x %d, n %d, items %d%s: key-byte counts", vocab, hidden, n_cols, items, norm ? " norm" : "");
  f += report(what, bad_hist, n_hist);
  CK(cudaFree(dw_)); CK(cudaFree(dx_)); CK(cudaFree(dlogits));
  if (dparts) CK(cudaFree(dparts));
  return f;
}

static int run_invalid() {
  long bad = 0, n = 0;
  uint16_t *w;
  CK(cudaMalloc(&w, 1 << 20));
  float *out;
  CK(cudaMalloc(&out, 1 << 20));
  const mlops_lm_head_config good{1000, 1024, 1, 0, 0, 0.f, 0.f, 1.f};
  mlops_lm_head_variant v[16];
  const int nv = mlops_lm_head_variants(&good, v, 16);
  bad += nv == 0;
  const mlops_lm_head_config broken[] = {{0, 1024, 1, 0, 0, 0, 0, 1}, {1000, 1000, 1, 0, 0, 0, 0, 1}, {1000, 1024, 1, -1, 0, 0, 0, 1}, {1000, 1024, 0, 0, 0, 0, 0, 1}, {1000, 1024, 9, 0, 0, 0, 0, 1}, {1000, 1024, 1, 0, 0, 0, 0, 0}};
  for (const auto &c : broken) {
    bad += mlops_lm_head_variants(&c, nullptr, 0) != 0;
    ++n;
  }
  bad += mlops_lm_head(&good, &v[0], nullptr, w, nullptr, out, nullptr, nullptr, nullptr, 0) != MLOPS_LM_HEAD_INVALID_VALUE;
  bad += mlops_lm_head(&good, &v[0], w, w, nullptr, nullptr, nullptr, nullptr, nullptr, 0) != MLOPS_LM_HEAD_INVALID_VALUE;  // no output
  bad += mlops_lm_head(&good, &v[0], w + 1, w, nullptr, out, nullptr, nullptr, nullptr, 0) != MLOPS_LM_HEAD_INVALID_VALUE;  // misaligned
  unsigned *hh;
  CK(cudaMalloc(&hh, 4096));
  bad += mlops_lm_head(&good, &v[0], w, w, nullptr, out, nullptr, nullptr, hh, 0) != MLOPS_LM_HEAD_INVALID_VALUE;  // hist without stats
  mlops_lm_head_variant odd{3, 4, 1, 1, 0, 0, 128, 0, 0, 0};
  bad += mlops_lm_head(&good, &odd, w, w, nullptr, out, nullptr, nullptr, nullptr, 0) != MLOPS_LM_HEAD_INVALID_VALUE;
  const mlops_lm_head_config tiny{1000, 256, 0, 0, 0, 0};  // one pass: no variant with 2 or more pieces
  bad += mlops_lm_head_variants(&tiny, nullptr, 0) != 0;
  n += 6 + 1 + 1 + 1;
  return report("invalid arguments", bad, n);
}

static void bench(int vocab, int hidden, int n_cols) {
  const size_t bytes = (size_t)vocab * hidden * 2;
  uint16_t *w, *x;
  float *logits;
  CK(cudaMalloc(&w, bytes));
  CK(cudaMemset(w, 0x3c, bytes));
  CK(cudaMalloc(&x, (size_t)n_cols * hidden * 2));
  CK(cudaMemset(x, 0x3c, (size_t)n_cols * hidden * 2));
  CK(cudaMalloc(&logits, (size_t)n_cols * vocab * 4));
  void *dst;
  CK(cudaMalloc(&dst, bytes));
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  float ms;
  for (int i = 0; i < 3; ++i) CK(cudaMemcpyAsync(dst, w, bytes, cudaMemcpyDeviceToDevice));
  CK(cudaEventRecord(a));
  for (int i = 0; i < 20; ++i) CK(cudaMemcpyAsync(dst, w, bytes, cudaMemcpyDeviceToDevice));
  CK(cudaEventRecord(b));
  CK(cudaEventSynchronize(b));
  CK(cudaEventElapsedTime(&ms, a, b));
  int dev = 0, mem_khz = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, dev));
  CK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev));
  const double copy_gbs = 2.0 * bytes / (ms / 20 * 1e-3) * 1e-9, peak = 2.0 * mem_khz * 1e3 * (prop.memoryBusWidth / 8) * 1e-9;
  printf("head %d x %d BF16, n %d (%.0f MB): device-to-device copy %.0f GB/s of traffic (read + write), DRAM peak %.0f GB/s\n", vocab, hidden, n_cols,
         bytes * 1e-6, copy_gbs, peak);
  printf("%4s %3s %3s %3s %3s %7s %5s %6s %12s %12s %9s %9s\n", "path", "R", "S", "D", "J", "threads", "regs", "items", "latency us", "read GB/s", "of copy", "of peak");
  const mlops_lm_head_config c{vocab, hidden, n_cols, 0, 0, 0.f, 0.f, 1.f};
  const int nv = mlops_lm_head_variants(&c, nullptr, 0);
  std::vector<mlops_lm_head_variant> vs(nv);
  mlops_lm_head_variants(&c, vs.data(), nv);
  for (const auto &v : vs) {
    for (int i = 0; i < 3; ++i) mlops_lm_head(&c, &v, w, x, nullptr, logits, nullptr, nullptr, nullptr, 0);
    CK(cudaEventRecord(a));
    for (int i = 0; i < 20; ++i) mlops_lm_head(&c, &v, w, x, nullptr, logits, nullptr, nullptr, nullptr, 0);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&ms, a, b));
    const double us = ms / 20 * 1e3, gbs = bytes / (us * 1e-6) * 1e-9;
    printf("%4s %3d %3d %3d %3d %7d %5d %6d %12.1f %12.1f %8.0f%% %8.0f%%\n", v.tensor_cores == 3 ? "tc884" : v.tensor_cores == 2 ? "tc16" : v.tensor_cores ? "tc" : "cuda", v.rows_per_unit, v.k_pieces, v.depth, v.chunks, v.threads, v.regs,
           mlops_lm_head_item_count(&c, &v), us, gbs, 100.0 * gbs / copy_gbs, 100.0 * gbs / peak);
  }
}

int main(int argc, char **argv) {
  if (argc > 1 && !strcmp(argv[1], "bench")) {
    bench(argc > 2 ? atoi(argv[2]) : 151936, argc > 3 ? atoi(argv[3]) : 4096, argc > 4 ? atoi(argv[4]) : 1);
    return 0;
  }
  int f = 0;
  f += run_case(1000, 1024, 1, 0, false);
  f += run_case(777, 2048, 1, 5, true);       // a vocabulary that is not a multiple of R, few elements
  f += run_case(4099, 1280, 1, 3, true);      // 5 passes: uneven k-pieces
  f += run_case(151936, 1024, 1, 0, false);   // Qwen3-0.6B
  f += run_case(20011, 4096, 1, 0, true);     // Qwen3-8B's hidden size
  f += run_case(100, 2048, 1, 64, false);     // more elements than units: the extra ones are empty
  for (int cols : {2, 3, 4, 5, 8}) f += run_case(3001, 2048, cols, 0, cols % 2 == 0);  // columns; 3 and 5 pad to 4 and 8
  f += run_case(40000, 1024, 8, 17, true);
  f += run_invalid();
  printf(f ? "\n%d FAILED\n" : "\nALL PASS\n", f);
  return f != 0;
}
