// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_fbla_decode latency and the DRAM traffic it implies, streaming: one cache laid out for 1M tokens
// (capacity), filled on the device, decoded at a growing context 16K .. 1M.  B = 1, D = 128, one decode query
// per head: one KV head with 2 or 4 q heads, and the Qwen3-0.6B geometry (8 KV heads, 16 q heads).
//
// The cache holds random codes, whose scores are flatter than a language model's: tau is chosen per context
// so that the share of refined (q head, sub-block) pairs is the one measured on Qwen3-0.6B at 16K (3+3:
// 12.3 %, 4+2: 4.8 %); the printed share is that of (KV head, sub-block) pairs, which the GQA queries read
// once.  Bytes read per call: the coarse planes of every closed group, the metadata (4 x 256 bytes per group),
// the BF16 sink and open groups, and for every refined sub-block both planes of K and V once more; "BF16
// equiv." is the K + V of a BF16 cache of that context.
#include "mlops_fbla.h"
#include "bench_util.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static uint32_t rng(uint32_t &s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
static size_t up256(size_t n) { return (n + 255) & ~(size_t)255; }
template <class T> static T *upload(const std::vector<T> &h) {
  T *p; CUDA_OK(cudaMalloc(&p, h.size() * sizeof(T)));
  CUDA_OK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
static uint16_t bf16(float x) { uint32_t u; memcpy(&u, &x, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

// About one second of timed work per row on the reference GTX 1080.  A single repeat count cannot cover
// both the 40 us one-head/16K row and the 7 ms eight-head/1M row.  As in the other mlops benchmarks these
// are only defaults: MLOPS_BENCH_REPS overrides every row for another GPU or a quick run.
static int default_reps(int cb, int KH, int gqa, int tokens) {
  static constexpr int reps[3][2][6] = {
    {{18000, 14500, 10000, 5500, 3000, 1600}, {25000, 16000, 10000, 5500, 3000, 1600}}, // KV1 GQA2
    {{ 9000,  8000,  6000, 4000, 2400, 1300}, {14000, 11000,  7500, 4500, 2600, 1400}}, // KV1 GQA4
    {{ 9000,  3000,  1700,  850,  300,  150}, {10000,  3000,  1600,  850,  300,  150}}, // KV8 GQA2
  };
  const int scenario = KH == 8 ? 2 : gqa == 4 ? 1 : 0;
  const int size = tokens <= 16384 ? 0 : tokens <= 65536 ? 1 : tokens <= 131072 ? 2
                 : tokens <= 262144 ? 3 : tokens <= 524288 ? 4 : 5;
  return reps[scenario][cb == 4][size];
}

__device__ uint32_t hash(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
__global__ void fill_bytes(uint8_t *p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n / 4; i += (size_t)gridDim.x * blockDim.x)
    reinterpret_cast<uint32_t *>(p)[i] = hash((uint32_t)i * 2654435761u + seed);
}
__global__ void fill_bf16(uint16_t *p, size_t n, uint32_t seed, float lo, float span) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const float x = lo + span * (hash((uint32_t)i * 2246822519u + seed) & 0xffffu) * (1.f / 65535.f);
    p[i] = (uint16_t)(__float_as_uint(x) >> 16);
  }
}

__global__ void fill_meta(uint16_t *p, size_t n) {  // bases in [-0.25, 0.25), steps in [0.02, 0.05)
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const bool step = (i / 4) % 2;
    const float u = (hash((uint32_t)i * 2246822519u + 3) & 0xffffu) * (1.f / 65535.f);
    p[i] = (uint16_t)(__float_as_uint(step ? .02f + .03f * u : -.25f + .5f * u) >> 16);
  }
}

struct Cache {
  int cb, fb, cap;
  uint8_t *coarse{}, *fine{};
  uint16_t *meta{}, *ex{}, *ex2{};
  Cache(int cb_, int fb_, int cap_, int KH) : cb(cb_), fb(fb_), cap(cap_) {
    const size_t sub = (size_t)KH * (cap / 16), groups = (size_t)KH * (cap / 128);
    CUDA_OK(cudaMalloc(&coarse, sub * 2 * cb * 256)); fill_bytes<<<1024, 256>>>(coarse, sub * 2 * cb * 256, 7u + cb);
    CUDA_OK(cudaMalloc(&fine, sub * 2 * fb * 256)); fill_bytes<<<1024, 256>>>(fine, sub * 2 * fb * 256, 9u + fb);
    // per group: K base, K step, V base, V step of every lane's four channels
    CUDA_OK(cudaMalloc(&meta, groups * 1024));
    fill_meta<<<512, 256>>>(meta, groups * 512);
    CUDA_OK(cudaMalloc(&ex, (size_t)KH * 128 * 128 * 2)); fill_bf16<<<64, 256>>>(ex, (size_t)KH * 128 * 128, 11, -.5f, 1.f);
    // prefill writes the sink and the open group: K / V of both, apart from the decode rows' shared `ex`
    CUDA_OK(cudaMalloc(&ex2, (size_t)KH * 128 * 128 * 2 * 2)); fill_bf16<<<64, 256>>>(ex2, (size_t)KH * 128 * 128 * 2, 12, -.5f, 1.f);
    CUDA_OK(cudaFree(ex)); CUDA_OK(cudaMalloc(&ex, (size_t)KH * 128 * 128 * 2 * 2));
    fill_bf16<<<64, 256>>>(ex, (size_t)KH * 128 * 128 * 2, 11, -.5f, 1.f);
    CUDA_OK(cudaDeviceSynchronize());
  }
  ~Cache() { for (void *p : {(void *)coarse, (void *)fine, (void *)meta, (void *)ex, (void *)ex2}) cudaFree(p); }
};

static void run_case(Bench &bench, double peak_gbs, const Cache &cc, int KH, int gqa, int tokens, void *scratch,
                     size_t scratch_bytes) {
  constexpr int B = 1, D = 128, G = 128, S = 16;
  const int QH = gqa * KH;
  const int cb = cc.cb, fb = cc.fb, groups = (tokens + G - 1) / G, nb = groups * 8;
  const size_t heads = (size_t)B * KH;
  uint32_t seed = 0x91ac35u + tokens + cb;
  std::vector<uint16_t> q((size_t)B * QH * D);
  for (auto &x : q) x = bf16(((int)(rng(seed) & 1023) - 511) * (2.f / 511.f));
  uint16_t *dq = upload(q), *dout;
  CUDA_OK(cudaMalloc(&dout, q.size() * 2));
  mlops_fbla_config cfg{B, QH, KH, tokens, 0.f, 1e30f, cb == 3 ? MLOPS_FBLA_3P3 : MLOPS_FBLA_4P2, cc.cap};
  const size_t sz = mlops_fbla_decode_scratch_bytes(&cfg);
  if (sz > scratch_bytes) { fprintf(stderr, "scratch too small\n"); exit(1); }
  mlops_fbla_cache cache{cc.coarse, cc.fine, cc.meta, cc.ex, cc.ex, cc.ex, cc.ex};
  auto call = [&] {
    if (mlops_fbla_decode(&cfg, dq, &cache, dout, scratch, sz, 0)) { fprintf(stderr, "%s\n", mlops_fbla_last_error()); exit(1); }
  };
  // Tau for the target share. The first call leaves closed-block LSEs and the complete row LSE in scratch.
  // Use row_lse rather than reconstructing it from the block array: exact sink/open blocks deliberately have
  // no entries in that array; their mass is already present in row_lse through the coarse partials.
  call(); CUDA_OK(cudaDeviceSynchronize());
  std::vector<float> lse((size_t)B * QH * nb);
  CUDA_OK(cudaMemcpy(lse.data(), (char *)scratch + 256, lse.size() * 4, cudaMemcpyDeviceToHost));
  int dev = 0, sms = 1; CUDA_OK(cudaGetDevice(&dev));
  CUDA_OK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  const int max_ctas = std::max(1, (8 * sms + KH - 1) / KH);
  const size_t off_lse = up256((size_t)KH * sizeof(int));
  const size_t off_part = off_lse + up256((size_t)QH * nb * sizeof(float));
  const size_t off_dpart = off_part + up256((size_t)QH * max_ctas * (D + 2) * sizeof(float));
  const size_t off_row_lse = off_dpart + up256((size_t)QH * max_ctas * (D + 1) * sizeof(float));
  std::vector<float> row_lse(QH);
  CUDA_OK(cudaMemcpy(row_lse.data(), (char *)scratch + off_row_lse, row_lse.size() * sizeof(float), cudaMemcpyDeviceToHost));
  const int first = 8, last = (groups - 1) * 8, per = last - first;  // closed groups' sub-blocks
  std::vector<float> w;
  w.reserve((size_t)B * QH * per);
  for (int r = 0; r < B * QH; ++r) {
    for (int j = first; j < last; ++j) w.push_back((float)exp2(lse[(size_t)r * nb + j] - row_lse[r]));
  }
  std::vector<float> sorted = w;
  const size_t k = std::max<size_t>(1, std::min(sorted.size() - 1, (size_t)llround((cb == 3 ? .123 : .048) * sorted.size())));
  std::nth_element(sorted.begin(), sorted.begin() + k, sorted.end(), std::greater<float>());
  cfg.tau = sorted[k];
  size_t refined = 0;
  for (int kh = 0; kh < KH; ++kh)
    for (int j = 0; j < per; ++j) {
      bool any = false;
      for (int qi = 0; qi < QH / KH; ++qi) any |= w[(size_t)(kh * (QH / KH) + qi) * per + j] > cfg.tau;
      refined += any;
    }
  const double share = (double)refined / ((size_t)KH * per);
  const double closed_tokens = std::max(0, groups - 2) * (double)G, exact_tokens = tokens - closed_tokens;
  const double bytes = heads * (closed_tokens * D * cb / 8.0 * 2 + std::max(0, groups - 2) * 4.0 * 256 +
                                exact_tokens * D * 2 * 2 + share * closed_tokens / S * (cb + fb) * 256 * 2);
  const double bf16_bytes = heads * (double)tokens * D * 2 * 2;
  const BenchRow row = bench.run(call, default_reps(cb, KH, gqa, tokens));
  char fmt[8]; snprintf(fmt, sizeof fmt, "%d+%d", cb, fb);
  printf("%-6s %3d %3d %9d %12.1f %14.1f %9.1f%% %10.1f %12.1f %8.1f%% %8.0f\n",
         fmt, KH, gqa, tokens, row.ms * 1e3, bytes / (row.ms * 1e-3) * 1e-9,
         100.0 * bytes / (row.ms * 1e-3) * 1e-9 / peak_gbs, bytes * 1e-6, bf16_bytes * 1e-6,
         100.0 * share, row.mhz);
  fflush(stdout);
  cudaFree(dq); cudaFree(dout);
}

// Prefill, streaming: the cache holds `ctx` tokens, a chunk of `chunk` new tokens (queries, keys, values) is
// appended and attended.  Work = attended (query row, key) pairs: q_heads x sum over the chunk of (position + 1);
// "equiv. TFLOPS" counts the 4 D floating-point operations a BF16 attention spends per pair (Q K^T and P V).
static void run_prefill_case(Bench &bench, const Cache &cc, int KH, int gqa, int ctx, int chunk) {
  const int QH = gqa * KH, D = 128;
  const mlops_fbla_format fmt = cc.cb == 3 ? MLOPS_FBLA_3P3 : MLOPS_FBLA_4P2;
  uint16_t *q, *k, *v, *out;
  CUDA_OK(cudaMalloc(&q, (size_t)QH * chunk * D * 2)); CUDA_OK(cudaMalloc(&out, (size_t)QH * chunk * D * 2));
  CUDA_OK(cudaMalloc(&k, (size_t)KH * chunk * D * 2)); CUDA_OK(cudaMalloc(&v, (size_t)KH * chunk * D * 2));
  fill_bf16<<<256, 256>>>(q, (size_t)QH * chunk * D, 21, -2.f, 4.f);
  fill_bf16<<<256, 256>>>(k, (size_t)KH * chunk * D, 22, -2.f, 4.f);
  fill_bf16<<<256, 256>>>(v, (size_t)KH * chunk * D, 23, -1.f, 2.f);
  mlops_fbla_config cfg{1, QH, KH, ctx, 0.f, 0.f, fmt, cc.cap};
  mlops_fbla_cache cache{cc.coarse, cc.fine, cc.meta, cc.ex, cc.ex + KH * 128 * 128, cc.ex2, cc.ex2 + KH * 128 * 128};
  auto call = [&] {
    if (mlops_fbla_prefill(&cfg, chunk, q, k, v, &cache, out, 0)) {
      fprintf(stderr, "%s\n", mlops_fbla_last_error()); exit(1);
    }
  };
  call(); CUDA_OK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0); call(); cudaEventRecord(e1); cudaEventSynchronize(e1);
  float est; cudaEventElapsedTime(&est, e0, e1);
  const BenchRow row = bench.run(call, std::max(3, std::min(300, (int)(800.f / std::max(est, 1e-3f)))));
  const double pairs = (double)QH * chunk * (ctx + 0.5 * (chunk + 1));
  printf("%-6s %3d %3d %9d %7d %12.3f %12.2f %12.2f %8.0f\n", cc.cb == 3 ? "3+3" : "4+2", KH, gqa, ctx, chunk,
         row.ms, pairs / (row.ms * 1e-3) * 1e-9, 4.0 * D * pairs / (row.ms * 1e-3) * 1e-12, row.mhz);
  fflush(stdout);
  for (void *x : {(void *)q, (void *)k, (void *)v, (void *)out}) cudaFree(x);
}

int main() {
  int dev = 0; CUDA_OK(cudaGetDevice(&dev));
  cudaDeviceProp p{}; CUDA_OK(cudaGetDeviceProperties(&p, dev));
  int memory_clock_khz = 0;  // cudaDeviceProp::memoryClockRate is gone in CUDA 13
  CUDA_OK(cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, dev));
  const double peak = 2.0 * memory_clock_khz * 1e3 * (p.memoryBusWidth / 8) * 1e-9;
  const int cap = 1 << 20;
  printf("%s sm_%d%d, %d SMs: warming GPU (%d x 2048^3 FP32) and measuring D2D copy ...\n",
         p.name, p.major, p.minor, p.multiProcessorCount, Bench::env("MLOPS_BENCH_WARMUP", 4));
  fflush(stdout);
  Bench bench;

  constexpr size_t copy_bytes = (size_t)256 << 20;
  void *copy_src, *copy_dst;
  CUDA_OK(cudaMalloc(&copy_src, copy_bytes)); CUDA_OK(cudaMalloc(&copy_dst, copy_bytes));
  CUDA_OK(cudaMemset(copy_src, 0, copy_bytes));
  const BenchRow copy = bench.run([&] { CUDA_OK(cudaMemcpyAsync(copy_dst, copy_src, copy_bytes,
                                                               cudaMemcpyDeviceToDevice)); }, 500);
  const double copy_gbs = 2.0 * copy_bytes / (copy.ms * 1e-3) * 1e-9;
  cudaFree(copy_src); cudaFree(copy_dst);

  const int fixed_reps = Bench::env("MLOPS_BENCH_REPS", -1);
  printf("DRAM peak %.0f GB/s (%d-bit bus), D2D copy %.0f GB/s read+write; B = 1, D = 128; "
         "cache capacity %d\n", peak, p.memoryBusWidth, copy_gbs, cap);
  if (fixed_reps >= 0)
    printf("median of %d calls per row after %d s idle; SM clock sampled through NVML\n", fixed_reps,
           Bench::env("MLOPS_BENCH_COOL", 1));
  else
    printf("median after %d s idle; default repeats give about 1 s timed work per row; "
           "SM clock sampled through NVML\n", Bench::env("MLOPS_BENCH_COOL", 1));
  printf("\n%-6s %3s %3s %9s %12s %14s %10s %10s %12s %9s %8s\n"
         "%-6s %3s %3s %9s %12s %14s %10s %10s %12s %9s %8s\n",
         "format", "KV", "GQA", "context", "latency us", "effective GB/s", "DRAM peak", "read MB",
         "BF16 KV MB", "refined", "SM MHz",
         "------", "---", "---", "---------", "----------", "--------------", "----------", "----------",
         "------------", "---------", "--------");
  fflush(stdout);
  const int heads[][2] = {{1, 2}, {1, 4}, {8, 2}};  // (KV heads, q heads per KV head)
  for (const auto &hg : heads)
    for (int fmt = 0; fmt < 2; ++fmt) {
      const int KH = hg[0], gqa = hg[1], cb = fmt ? 4 : 3, fb = fmt ? 2 : 3;
      const int sizes[] = {16384, 65536, 131072, 262144, 524288, 1048576};
      auto name = [&](char *s, size_t n, int t) { snprintf(s, n, "FBLA %d+%d KV=%d GQA=%d N=%d", cb, fb, KH, gqa, t); };
      bool any = false;
      for (int n : sizes) { char s[64]; name(s, sizeof s, n); any |= bench_want(s); }
      if (!any) continue;
      Cache cc(cb, fb, cap, KH);
      mlops_fbla_config big{1, gqa * KH, KH, cap, 0.f, 0.f, fmt ? MLOPS_FBLA_4P2 : MLOPS_FBLA_3P3, cap};
      const size_t sz = mlops_fbla_decode_scratch_bytes(&big);
      void *scratch; CUDA_OK(cudaMalloc(&scratch, sz)); CUDA_OK(cudaMemset(scratch, 0, sz));
      for (int n : sizes) {
        char s[64]; name(s, sizeof s, n);
        if (bench_want(s)) run_case(bench, peak, cc, KH, gqa, n, scratch, sz);
      }
      cudaFree(scratch);
    }

  // ---- prefill -----------------------------------------------------------------------------------------
  const int pcases[][2] = {{0, 2048}, {16384, 2048}, {65536, 512}, {65536, 2048}, {65536, 8192}, {262144, 2048}};
  bool header = false;
  for (int fmt = 0; fmt < 2; ++fmt) {
    const int KH = 8, gqa = 2, cb = fmt ? 4 : 3, fb = fmt ? 2 : 3;
    auto name = [&](char *s, size_t n, int ctx, int chunk) {
      snprintf(s, n, "PREFILL %d+%d KV=%d GQA=%d ctx=%d chunk=%d", cb, fb, KH, gqa, ctx, chunk);
    };
    bool any = false;
    for (const auto &pc : pcases) { char s[96]; name(s, sizeof s, pc[0], pc[1]); any |= bench_want(s); }
    if (!any) continue;
    if (!header) {
      printf("\nprefill: a chunk appended to a cache of `context` tokens and attended (Qwen3-0.6B heads)\n"
             "%-6s %3s %3s %9s %7s %12s %12s %12s %8s\n", "format", "KV", "GQA", "context", "chunk", "latency ms",
             "Gpairs/s", "equiv. TFLOPS", "SM MHz");
      header = true;
    }
    Cache cc(cb, fb, 1 << 19, KH);
    for (const auto &pc : pcases) {
      char s[96]; name(s, sizeof s, pc[0], pc[1]);
      if (bench_want(s)) run_prefill_case(bench, cc, KH, gqa, pc[0], pc[1]);
    }
  }
  return 0;
}
