// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The work items of mlops-vocab (include/mlops_vocab_items.h) through NVRTC and an executor of the test's own: each
// item's entry is compiled from its `entry` string and run on the described CTA; the outputs are the call's, bit for
// bit. The declared writes are checked by running every item alone on poisoned outputs: only declared bytes change,
// and together the items write every output byte.
#include <cuda.h>
#include <nvrtc.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "mlops_vocab_items.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CU(x) do { CUresult r_ = (x); if (r_) { const char *m_; cuGetErrorString(r_, &m_); fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, m_); exit(1); } } while (0)

#ifndef VOCAB_INCLUDE
#define VOCAB_INCLUDE "include"
#endif
#ifndef CUDA_INCLUDE
#define CUDA_INCLUDE "/usr/local/cuda/include"
#endif

static std::mt19937 rng(7);
static uint16_t bf16(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
template <class T> static T *up(const std::vector<T> &h) {
  T *d;
  CK(cudaMalloc(&d, h.size() * sizeof(T) + 16));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
static std::vector<unsigned char> download(const void *d, size_t n) {
  std::vector<unsigned char> h(n);
  CK(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost));
  return h;
}
static int report(const char *what, long bad, long n) {
  printf("[%s] %s (%ld of %ld differ)\n", bad ? "FAIL" : "PASS", what, bad, n);
  return bad != 0;
}

// ---- NVRTC ------------------------------------------------------------------------------------------------------
struct Kernel {
  CUmodule mod;
  CUfunction fn;
};
static Kernel compile(const mlops_vocab_items &it) {
  const std::string src = std::string("#include \"mlops_vocab_items.cuh\"\nextern \"C\" __global__ void run(const ") + it.args_type +
                          " *args, int first) {\n  extern __shared__ __align__(16) unsigned char smem[];\n  " + it.entry +
                          "(*args, first + blockIdx.x, threadIdx.x, smem);\n}\n";
  nvrtcProgram prog;
  if (nvrtcCreateProgram(&prog, src.c_str(), "items.cu", 0, nullptr, nullptr)) exit(1);
  char arch[32];
  snprintf(arch, sizeof arch, "--gpu-architecture=compute_%d", it.arch);
  const char *opts[] = {"--std=c++17", "--device-as-default-execution-space", arch, "-I" VOCAB_INCLUDE, "-I" CUDA_INCLUDE, "-I" CUDA_INCLUDE "/cccl"};
  const nvrtcResult r = nvrtcCompileProgram(prog, 6, opts);
  size_t log_n;
  nvrtcGetProgramLogSize(prog, &log_n);
  std::string log(log_n, 0);
  nvrtcGetProgramLog(prog, &log[0]);
  if (r) {
    fprintf(stderr, "NVRTC failed for %s:\n%s\n", it.entry, log.c_str());
    exit(1);
  }
  size_t ptx_n;
  nvrtcGetPTXSize(prog, &ptx_n);
  std::string ptx(ptx_n, 0);
  nvrtcGetPTX(prog, &ptx[0]);
  nvrtcDestroyProgram(&prog);
  Kernel k;
  CU(cuModuleLoadData(&k.mod, ptx.c_str()));
  CU(cuModuleGetFunction(&k.fn, k.mod, "run"));
  return k;
}
static void run(const Kernel &k, const mlops_vocab_items &it, int first, int count) {
  void *d_args;
  CK(cudaMalloc(&d_args, it.args_bytes));
  CK(cudaMemcpy(d_args, it.args.bytes, it.args_bytes, cudaMemcpyHostToDevice));
  void *params[] = {&d_args, &first};
  CU(cuLaunchKernel(k.fn, count, 1, 1, it.threads, 1, 1, it.smem_bytes, 0, params, nullptr));
  CK(cudaDeviceSynchronize());
  CK(cudaFree(d_args));
}

// ---- declared writes --------------------------------------------------------------------------------------------
struct Buffer {
  int arg;
  void *ptr;
  size_t bytes;
};
static void poison(const std::vector<Buffer> &bufs) {
  for (const auto &b : bufs) CK(cudaMemset(b.ptr, 0xcd, b.bytes));
}
// Runs every item alone, compares what changed with what it declared.
static int check_writes(const char *what, const Kernel &k, const mlops_vocab_items &it, const std::vector<Buffer> &outs) {
  long bad = 0, n = 0;
  std::vector<std::vector<char>> covered;
  for (const auto &b : outs) covered.emplace_back(b.bytes, 0);
  for (int item = 0; item < it.items; ++item) {
    poison(outs);
    run(k, it, item, 1);
    std::vector<mlops_vocab_item_range> ranges(mlops_vocab_item_access(&it, item, nullptr, 0));
    mlops_vocab_item_access(&it, item, ranges.data(), (int)ranges.size());
    for (size_t o = 0; o < outs.size(); ++o) {
      std::vector<char> declared(outs[o].bytes, 0);
      for (const auto &r : ranges)
        if (r.arg == outs[o].arg && r.write == 1)
          for (int row = 0; row < r.rows; ++row)
            for (long long i = 0; i < r.bytes; ++i) {
              const long long at = r.offset + row * r.stride + i;
              if (at >= 0 && (size_t)at < outs[o].bytes) declared[at] = 1;
              else ++bad;
            }
      const auto now = download(outs[o].ptr, outs[o].bytes);
      for (size_t i = 0; i < now.size(); ++i) {
        const bool changed = now[i] != 0xcd;
        bad += changed && !declared[i];  // a store outside the declared ranges
        covered[o][i] |= declared[i];
        ++n;
      }
    }
  }
  for (size_t o = 0; o < outs.size(); ++o)
    for (char c : covered[o]) { bad += !c; ++n; }  // an output byte no item declares
  return report(what, bad, n);
}

// ---- embed ------------------------------------------------------------------------------------------------------
static int test_embed() {
  const int vocab = 3000, hidden = 2560, n_tok = 3;
  std::vector<uint16_t> table((size_t)vocab * hidden), w(hidden);
  std::normal_distribution<float> d(0.f, 1.f);
  for (auto &v : table) v = bf16(d(rng));
  for (auto &v : w) v = bf16(d(rng));
  std::vector<int> ids = {5, 2999, 17};
  uint16_t *dt = up(table), *dw = up(w), *dxw;
  int *did = up(ids), *derr;
  float *dx, *dp;
  const size_t nx = (size_t)n_tok * hidden, np = (size_t)n_tok * hidden / 256;
  CK(cudaMalloc(&dx, nx * 4)); CK(cudaMalloc(&dxw, nx * 2)); CK(cudaMalloc(&dp, np * 4)); CK(cudaMalloc(&derr, 4));
  const mlops_embed_config c{vocab, hidden, n_tok, 0};
  mlops_embed_variant v[4];
  mlops_embed_variants(&c, v, 4);
  int f = 0;
  for (int vi : {0, 3}) {
    CK(cudaMemset(derr, 0, 4));
    if (mlops_embed(&c, &v[vi], dt, did, dw, dx, dxw, dp, derr, 0)) return 1;
    CK(cudaDeviceSynchronize());
    const auto rx = download(dx, nx * 4), rxw = download(dxw, nx * 2), rp = download(dp, np * 4);
    mlops_vocab_items it;
    if (mlops_embed_items(&c, &v[vi], dt, did, dw, dx, dxw, dp, derr, &it) != 1) return 1;
    const Kernel k = compile(it);
    const std::vector<Buffer> outs = {{MLOPS_VOCAB_ARG_X, dx, nx * 4}, {MLOPS_VOCAB_ARG_XW, dxw, nx * 2}, {MLOPS_VOCAB_ARG_PARTS, dp, np * 4}};
    poison(outs);
    run(k, it, 0, it.items);
    long bad = download(dx, nx * 4) != rx, n = 3;
    bad += download(dxw, nx * 2) != rxw;
    bad += download(dp, np * 4) != rp;
    char what[160];
    snprintf(what, sizeof what, "embed items (%d warps, %d items): outputs of the call", v[vi].warps, it.items);
    f += report(what, bad, n);
    snprintf(what, sizeof what, "embed items (%d warps): declared writes", v[vi].warps);
    f += check_writes(what, k, it, outs);
  }
  return f;
}

// ---- head -------------------------------------------------------------------------------------------------------
static int test_head(int vocab, int hidden, int n_cols, bool norm) {
  std::normal_distribution<float> d(0.f, 0.05f);
  std::vector<uint16_t> w((size_t)vocab * hidden), x((size_t)n_cols * hidden);
  for (auto &v : w) v = bf16(d(rng));
  for (auto &v : x) v = bf16(d(rng) * 10.f);
  std::vector<float> parts;
  if (norm) for (int i = 0; i < n_cols * 9; ++i) parts.push_back(1.f + (rng() % 100) / 50.f);
  uint16_t *dw = up(w), *dx = up(x);
  float *dparts = norm ? up(parts) : nullptr, *dl, *dmax;
  int *darg;
  unsigned *dhist;
  const mlops_lm_head_config c{vocab, hidden, n_cols, 11, norm ? 9 : 0, 1.f / hidden, 1e-6f, 0.8f};
  const int nv = mlops_lm_head_variants(&c, nullptr, 0);
  std::vector<mlops_lm_head_variant> vs(nv);
  mlops_lm_head_variants(&c, vs.data(), nv);
  const int ni = mlops_lm_head_item_count(&c, &vs[0]);
  CK(cudaMalloc(&dl, (size_t)n_cols * vocab * 4)); CK(cudaMalloc(&dmax, (size_t)n_cols * ni * 4)); CK(cudaMalloc(&darg, (size_t)n_cols * ni * 4));
  int f = 0;
  std::vector<int> picks = {0, nv / 2, nv - 1};
  for (int pi : picks) {
    const auto &v = vs[pi];
    const int items_n = mlops_lm_head_item_count(&c, &v);
    CK(cudaMalloc(&dmax, (size_t)n_cols * items_n * 4)); CK(cudaMalloc(&darg, (size_t)n_cols * items_n * 4));
    CK(cudaMalloc(&dhist, (size_t)n_cols * items_n * 256 * 4));
    if (mlops_lm_head(&c, &v, dw, dx, dparts, dl, dmax, darg, dhist, 0)) return 1;
    CK(cudaDeviceSynchronize());
    const size_t nl = (size_t)n_cols * vocab, ns = (size_t)n_cols * items_n;
    const auto rl = download(dl, nl * 4), rm = download(dmax, ns * 4), ra = download(darg, ns * 4), rh = download(dhist, ns * 256 * 4);
    mlops_vocab_items it;
    if (mlops_lm_head_items(&c, &v, dw, dx, dparts, dl, dmax, darg, dhist, &it) != 1) return 1;
    const Kernel k = compile(it);
    const std::vector<Buffer> outs = {{MLOPS_VOCAB_ARG_LOGITS, dl, nl * 4}, {MLOPS_VOCAB_ARG_STATS_MAX, dmax, ns * 4}, {MLOPS_VOCAB_ARG_STATS_ARG, darg, ns * 4}, {MLOPS_VOCAB_ARG_STATS_HIST, dhist, ns * 256 * 4}};
    poison(outs);
    run(k, it, 0, it.items);
    long bad = download(dl, nl * 4) != rl;
    bad += download(dmax, ns * 4) != rm;
    bad += download(darg, ns * 4) != ra;
    bad += download(dhist, ns * 256 * 4) != rh;
    char what[170];
    snprintf(what, sizeof what, "head items (%s R%d S%d D%d, n %d%s, %d items): outputs of the call", v.tensor_cores == 3 ? "tc884" : v.tensor_cores == 2 ? "tc16" : v.tensor_cores ? "tc" : "cuda", v.rows_per_unit, v.k_pieces, v.depth, n_cols, norm ? " norm" : "", it.items);
    f += report(what, bad, 4);
    snprintf(what, sizeof what, "head items (%s R%d S%d D%d, n %d): declared writes", v.tensor_cores == 3 ? "tc884" : v.tensor_cores == 2 ? "tc16" : v.tensor_cores ? "tc" : "cuda", v.rows_per_unit, v.k_pieces, v.depth, n_cols);
    f += check_writes(what, k, it, outs);
  }
  return f;
}

// ---- sample -----------------------------------------------------------------------------------------------------
static int test_sample(const char *name, mlops_sample_config c, bool with_stats) {
  std::normal_distribution<float> d(0.f, 2.f);
  std::vector<float> logits(c.vocab);
  for (auto &x : logits) x = d(rng);
  std::vector<uint32_t> seen((c.vocab + 31) / 32, 0x01010101u);
  float *dl = up(logits);
  uint32_t *dseen = up(seen);
  const size_t ws_bytes = mlops_sample_workspace_bytes(&c);
  void *ws_a, *ws_b;
  CK(cudaMalloc(&ws_a, ws_bytes)); CK(cudaMemset(ws_a, 0, ws_bytes));
  CK(cudaMalloc(&ws_b, ws_bytes)); CK(cudaMemset(ws_b, 0, ws_bytes));
  mlops_sample_state *sa, *sb;
  const mlops_sample_state s0{11, 5};
  CK(cudaMalloc(&sa, sizeof s0)); CK(cudaMemcpy(sa, &s0, sizeof s0, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&sb, sizeof s0)); CK(cudaMemcpy(sb, &s0, sizeof s0, cudaMemcpyHostToDevice));
  int *ta, *tb;
  CK(cudaMalloc(&ta, 4)); CK(cudaMalloc(&tb, 4));
  // statistics: the maximum and lowest index of each of 4 ranges of the (sampled) logits, as a head would give them
  std::vector<float> smax(4, -1e30f);
  std::vector<int> sarg(4, 0);
  std::vector<float> sampled(logits);
  if (with_stats) {
    for (auto &x : sampled) x = c.temperature > 0.f ? x * (1.f / c.temperature) : x;
    const int per = (c.vocab + 3) / 4;
    for (int i = 0; i < c.vocab; ++i)
      if (sampled[i] > smax[i / per]) { smax[i / per] = sampled[i]; sarg[i / per] = i; }
    CK(cudaMemcpy(dl, sampled.data(), sampled.size() * 4, cudaMemcpyHostToDevice));
  }
  float *dsm = up(smax);
  int *dsa = up(sarg);
  const mlops_sample_stats stats{dsm, dsa, 4};
  const mlops_sample_stats *sp = with_stats ? &stats : nullptr;
  const uint32_t *seen_p = c.penalty != 1.f ? dseen : nullptr;
  if (mlops_sample(&c, dl, seen_p, sp, sa, ws_a, ws_bytes, ta, 0)) { fprintf(stderr, "%s\n", mlops_sample_last_error()); return 1; }
  CK(cudaDeviceSynchronize());
  mlops_vocab_items passes[8];
  const int np = mlops_sample_items(&c, dl, seen_p, sp, sb, ws_b, ws_bytes, tb, passes, 8);
  if (np < 1) { fprintf(stderr, "%s\n", mlops_sample_last_error()); return 1; }
  for (int p = 0; p < np; ++p) {
    const Kernel k = compile(passes[p]);
    run(k, passes[p], 0, passes[p].items);
  }
  int ra, rb;
  CK(cudaMemcpy(&ra, ta, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&rb, tb, 4, cudaMemcpyDeviceToHost));
  mlops_sample_state ea, eb;
  CK(cudaMemcpy(&ea, sa, sizeof ea, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&eb, sb, sizeof eb, cudaMemcpyDeviceToHost));
  long bad = ra != rb || ea.step != eb.step || eb.step != s0.step + 1;
  // every pass declares ranges inside its arguments' sizes
  for (int p = 0; p < np; ++p)
    for (int item = 0; item < passes[p].items; ++item) {
      std::vector<mlops_vocab_item_range> r(mlops_vocab_item_access(&passes[p], item, nullptr, 0));
      mlops_vocab_item_access(&passes[p], item, r.data(), (int)r.size());
      for (const auto &q : r) {
        const long long last = q.offset + (q.rows - 1) * q.stride + q.bytes;
        const long long limit = q.arg == MLOPS_VOCAB_ARG_WORKSPACE ? (long long)ws_bytes : q.arg == MLOPS_VOCAB_ARG_LOGITS ? 4LL * c.vocab : 1LL << 40;
        bad += q.offset < 0 || last > limit;
      }
    }
  char what[170];
  snprintf(what, sizeof what, "sample items (%s%s): %d passes, the call's token and state", name, with_stats ? ", head statistics" : "", np);
  return report(what, bad, 1);
}

int main() {
  CK(cudaFree(0));
  int f = 0;
  f += test_embed();
  f += test_head(1000, 1024, 1, false);
  f += test_head(2500, 2048, 1, true);
  f += test_head(3001, 2048, 8, true);
  f += test_head(1500, 1024, 3, false);
  f += test_sample("greedy", {151936, 0.f, 0, 1.f, 1.2f, 0}, false);
  f += test_sample("greedy", {151936, 0.f, 0, 1.f, 1.f, 0}, true);
  f += test_sample("top-k, top-p", {151936, 0.6f, 20, 0.95f, 1.1f, 0}, false);
  f += test_sample("top-p", {50000, 1.f, 0, 0.9f, 1.f, 0}, false);
  f += test_sample("large top-k", {30000, 1.f, 2000, 1.f, 1.f, 0}, true);
  f += test_sample("temperature", {9000, 0.5f, 0, 1.f, 1.f, 0}, true);
  printf(f ? "\n%d FAILED\n" : "\nALL PASS\n", f);
  return f != 0;
}
