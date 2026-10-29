// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The work items of mlops-norm (include/mlops_norm_items.h) through NVRTC and an executor of the test's own: each item's
// entry is compiled from its `entry` string and run on the described CTA; the outputs are the call's, bit for bit. The
// declared writes are checked by running every item alone on poisoned outputs: only declared bytes change, and
// together the items write every output byte.
#include <cuda.h>
#include <cuda_fp16.h>
#include <nvrtc.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "mlops_norm_items.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CU(x) do { CUresult r_ = (x); if (r_) { const char *m_; cuGetErrorString(r_, &m_); fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, m_); exit(1); } } while (0)

static std::mt19937 rng(11);
static int report(const char *what, long bad, long n) {
  printf("[%s] %s (%ld of %ld differ)\n", bad ? "FAIL" : "PASS", what, bad, n);
  return bad != 0;
}
static std::vector<unsigned char> download(const void *d, size_t n) {
  std::vector<unsigned char> h(n);
  CK(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost));
  return h;
}

struct Kernel { CUmodule mod; CUfunction fn; };
static Kernel compile(const mlops_norm_items &it) {
  const std::string src = std::string("#include \"mlops_norm_items.cuh\"\nextern \"C\" __global__ void run(const ") + it.args_type +
                          " *args, int first) {\n  extern __shared__ __align__(16) unsigned char smem[];\n  " + it.entry +
                          "(*args, first + blockIdx.x, threadIdx.x, smem);\n}\n";
  nvrtcProgram prog;
  if (nvrtcCreateProgram(&prog, src.c_str(), "items.cu", 0, nullptr, nullptr)) exit(1);
  char arch[32];
  snprintf(arch, sizeof arch, "--gpu-architecture=compute_%d", it.arch);
  const char *opts[] = {"--std=c++17", "--device-as-default-execution-space", arch, "-I" NORM_INCLUDE, "-I" NVFP4_INCLUDE, "-I" NVFP4_SRC,
                        "-I" CUDA_INCLUDE, "-I" CUDA_INCLUDE "/cccl"};
  const nvrtcResult r = nvrtcCompileProgram(prog, 8, opts);
  size_t log_n;
  nvrtcGetProgramLogSize(prog, &log_n);
  std::string log(log_n, 0);
  nvrtcGetProgramLog(prog, &log[0]);
  if (r) { fprintf(stderr, "NVRTC failed for %s:\n%s\n", it.entry, log.c_str()); exit(1); }
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
static void run(const Kernel &k, const mlops_norm_items &it, int first, int count) {
  void *d_args;
  CK(cudaMalloc(&d_args, it.args_bytes));
  CK(cudaMemcpy(d_args, it.args.bytes, it.args_bytes, cudaMemcpyHostToDevice));
  void *params[] = {&d_args, &first};
  CU(cuLaunchKernel(k.fn, count, 1, 1, it.threads, 1, 1, it.smem_bytes, 0, params, nullptr));
  CK(cudaDeviceSynchronize());
  CK(cudaFree(d_args));
}

struct Buffer { int arg; void *ptr; size_t bytes; };

struct Case {
  const char *name;
  int rows, hidden, groups;
  mlops_norm_kind kind;
  mlops_norm_scale scale;
  int bias;
  mlops_norm_order order;
  mlops_norm_gate gate;
  mlops_norm_dtype xd, yd;
  mlops_norm_out out;
  bool residual;
  int ld_scale;
};

static int test_case(const Case &k) {
  const int xb = k.xd == MLOPS_NORM_FP32 ? 4 : 2, yb = k.yd == MLOPS_NORM_FP32 ? 4 : 2, ld = k.ld_scale ? k.ld_scale : k.hidden / 16;
  std::vector<unsigned char> hx((size_t)k.rows * k.hidden * xb), hr(hx.size()), hz(hx.size()), hw((size_t)k.hidden * yb), hb(hw.size());
  std::normal_distribution<float> d(0.f, 1.f);
  auto fill = [&](std::vector<unsigned char> &buf, int dt, size_t n, float scale) {
    for (size_t i = 0; i < n; ++i) {
      const float v = d(rng) * scale;
      if (dt == MLOPS_NORM_FP32) memcpy(&buf[4 * i], &v, 4);
      else if (dt == MLOPS_NORM_FP16) { const __half hh = __float2half_rn(v); memcpy(&buf[2 * i], &hh, 2); }
      else { unsigned u; memcpy(&u, &v, 4); const unsigned short h = (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); memcpy(&buf[2 * i], &h, 2); }
    }
  };
  fill(hx, k.xd, (size_t)k.rows * k.hidden, 2.f); fill(hr, k.xd, (size_t)k.rows * k.hidden, 1.f); fill(hz, k.xd, (size_t)k.rows * k.hidden, 1.f);
  fill(hw, k.yd, k.hidden, 0.5f); fill(hb, k.yd, k.hidden, 0.2f);
  void *dx, *dr, *dz, *dw, *db, *dro, *dy, *dsc;
  float *dg;
  const size_t ybytes = k.out == MLOPS_NORM_OUT_DTYPE ? (size_t)k.rows * k.hidden * yb : k.out == MLOPS_NORM_OUT_INT8 ? (size_t)k.rows * k.hidden : (size_t)k.rows * k.hidden / 2;
  const size_t sbytes = k.out == MLOPS_NORM_OUT_INT8 ? (size_t)k.rows * ld * 4 : (size_t)k.rows * ld;
  CK(cudaMalloc(&dx, hx.size())); CK(cudaMalloc(&dr, hx.size())); CK(cudaMalloc(&dz, hx.size())); CK(cudaMalloc(&dw, hw.size())); CK(cudaMalloc(&db, hw.size()));
  CK(cudaMalloc(&dro, hx.size())); CK(cudaMalloc(&dy, ybytes)); CK(cudaMalloc(&dsc, sbytes + 4)); CK(cudaMalloc(&dg, 4));
  CK(cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dr, hr.data(), hr.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dz, hz.data(), hz.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, hb.data(), hb.size(), cudaMemcpyHostToDevice));
  const float g = 0.02f;
  CK(cudaMemcpy(dg, &g, 4, cudaMemcpyHostToDevice));
  const mlops_norm_config c{k.rows, k.hidden, k.groups, 1e-6f, k.kind, k.scale, k.bias, k.order, k.gate, k.xd, k.yd, k.out, k.ld_scale, 5};
  const int nv = mlops_norm_variants(&c, nullptr, 0);
  std::vector<mlops_norm_variant> vs(nv);
  mlops_norm_variants(&c, vs.data(), nv);
  mlops_norm_args a{};
  a.x = dx; a.residual = k.residual ? dr : nullptr; a.weight = k.scale != MLOPS_NORM_SCALE_NONE ? dw : nullptr; a.bias = k.bias ? db : nullptr;
  a.gate = k.gate != MLOPS_NORM_GATE_NONE ? dz : nullptr; a.residual_out = k.residual ? dro : nullptr; a.y = dy; a.scales = dsc;
  a.global = k.out == MLOPS_NORM_OUT_NVFP4 ? dg : nullptr;
  int f = 0;
  for (size_t vi : {size_t(0), vs.size() - 1}) {
    const auto &v = vs[vi];
    CK(cudaMemset(dy, 0xcd, ybytes)); CK(cudaMemset(dsc, 0xcd, sbytes)); CK(cudaMemset(dro, 0xcd, hx.size()));
    if (mlops_norm(&c, &v, &a, 0)) { fprintf(stderr, "%s\n", mlops_norm_last_error()); return 1; }
    CK(cudaDeviceSynchronize());
    const auto ry = download(dy, ybytes), rs = download(dsc, sbytes), rro = download(dro, hx.size());
    mlops_norm_items it;
    if (mlops_norm_describe(&c, &v, &a, &it) != 1) { fprintf(stderr, "%s\n", mlops_norm_last_error()); return 1; }
    const Kernel kern = compile(it);
    std::vector<Buffer> outs = {{MLOPS_NORM_ARG_Y, dy, ybytes}};
    if (k.out != MLOPS_NORM_OUT_DTYPE) outs.push_back({MLOPS_NORM_ARG_SCALES, dsc, sbytes});
    if (k.residual) outs.push_back({MLOPS_NORM_ARG_RESIDUAL_OUT, dro, hx.size()});
    for (const auto &b : outs) CK(cudaMemset(b.ptr, 0xcd, b.bytes));
    run(kern, it, 0, it.items);
    long bad = download(dy, ybytes) != ry;
    if (k.out != MLOPS_NORM_OUT_DTYPE) {  // only the first hidden / 16 scales of a row are written
      const auto now = download(dsc, sbytes);
      const size_t sz = k.out == MLOPS_NORM_OUT_INT8 ? 4 : 1;
      for (int r = 0; r < k.rows; ++r) bad += memcmp(&now[(size_t)r * ld * sz], &rs[(size_t)r * ld * sz], (size_t)k.hidden / 16 * sz) != 0;
    }
    if (k.residual) bad += download(dro, hx.size()) != rro;
    char what[220];
    snprintf(what, sizeof what, "items %s (%d warps, %d items): outputs of the call", k.name, v.warps, it.items);
    f += report(what, bad, 1);

    // declared writes: each item alone on poisoned outputs
    long wbad = 0, wn = 0;
    std::vector<std::vector<char>> covered;
    for (const auto &b : outs) covered.emplace_back(b.bytes, 0);
    for (int item = 0; item < it.items; ++item) {
      for (const auto &b : outs) CK(cudaMemset(b.ptr, 0xcd, b.bytes));
      run(kern, it, item, 1);
      std::vector<mlops_norm_item_range> ranges(mlops_norm_item_access(&it, item, nullptr, 0));
      mlops_norm_item_access(&it, item, ranges.data(), (int)ranges.size());
      for (size_t o = 0; o < outs.size(); ++o) {
        std::vector<char> declared(outs[o].bytes, 0);
        for (const auto &r : ranges)
          if (r.arg == outs[o].arg && r.write == 1)
            for (int row = 0; row < r.rows; ++row)
              for (long long i = 0; i < r.bytes; ++i) {
                const long long at = r.offset + row * r.stride + i;
                if (at >= 0 && (size_t)at < outs[o].bytes) declared[at] = 1; else ++wbad;
              }
        const auto now = download(outs[o].ptr, outs[o].bytes);
        for (size_t i = 0; i < now.size(); ++i) { wbad += now[i] != 0xcd && !declared[i]; covered[o][i] |= declared[i]; ++wn; }
      }
    }
    // every byte the call writes is declared by some item (the scale rows' padding is never written)
    for (size_t o = 0; o < outs.size(); ++o) {
      if (outs[o].arg == MLOPS_NORM_ARG_SCALES) continue;
      for (char cv : covered[o]) { wbad += !cv; ++wn; }
    }
    snprintf(what, sizeof what, "items %s (%d warps): declared writes", k.name, v.warps);
    f += report(what, wbad, wn);
  }
  return f;
}

int main() {
  CK(cudaFree(0));
  const auto B = MLOPS_NORM_BF16, H = MLOPS_NORM_FP16, F = MLOPS_NORM_FP32;
  const auto RMS = MLOPS_NORM_RMS, LN = MLOPS_NORM_LAYER;
  const auto W = MLOPS_NORM_SCALE_WEIGHT, ONE = MLOPS_NORM_SCALE_ONE_PLUS_WEIGHT;
  const auto HF = MLOPS_NORM_ORDER_ROUND_THEN_SCALE, F32 = MLOPS_NORM_ORDER_FP32;
  const auto NG = MLOPS_NORM_GATE_NONE, AFTER = MLOPS_NORM_GATE_AFTER, BEFORE = MLOPS_NORM_GATE_BEFORE;
  const auto D = MLOPS_NORM_OUT_DTYPE, I8 = MLOPS_NORM_OUT_INT8, N4 = MLOPS_NORM_OUT_NVFP4;
  int f = 0;
  f += test_case({"rms hf bf16", 11, 1024, 1, RMS, W, 0, HF, NG, B, B, D, false, 0});
  f += test_case({"rms hf, fp32 stream, residual", 7, 2048, 1, RMS, W, 0, HF, NG, F, B, D, true, 0});
  f += test_case({"rms gemma (one plus, fp32 order)", 6, 1536, 1, RMS, ONE, 0, F32, NG, H, H, D, true, 0});
  f += test_case({"layernorm + bias", 6, 1024, 1, LN, W, 1, F32, NG, B, B, D, false, 0});
  f += test_case({"layernorm 1P + bias, groups 2, residual", 5, 2048, 2, LN, ONE, 1, F32, NG, F, B, D, true, 0});
  f += test_case({"rms gate after", 6, 1024, 1, RMS, ONE, 0, F32, AFTER, B, B, D, false, 0});
  f += test_case({"rms gate before, groups 4", 5, 2048, 4, RMS, W, 0, F32, BEFORE, B, B, D, true, 0});
  f += test_case({"rms hf -> int8, residual, padded", 9, 2048, 1, RMS, W, 0, HF, NG, F, B, I8, true, 160});
  f += test_case({"layernorm + bias -> nvfp4", 8, 4096, 1, LN, W, 1, F32, NG, B, B, N4, false, 0});
  f += test_case({"rms gemma -> nvfp4, padded", 5, 1024, 1, RMS, ONE, 0, F32, NG, F, H, N4, true, 80});
  printf(f ? "\n%d FAILED\n" : "\nALL PASS\n", f);
  return f != 0;
}
