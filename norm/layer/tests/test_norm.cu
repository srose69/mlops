// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_norm against a host emulation of its arithmetic, bit for bit: every flavor (RMSNorm and LayerNorm; the weight, one plus
// it, none; a bias; Hugging Face's order and the single-rounding one; groups), every variant and element count, every dtype pair,
// with and without a residual (also in place). The gates, whose exp is libm's, within a bound. The INT8 and NVFP4 outputs are
// the bytes nvfp4a_quantize makes of the call's own y.
// `test_norm bench`: the call against a device-to-device copy of the same bytes.
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "mlops_norm_layer.h"
#include "norm.cuh"
#include "nvfp4effma.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static std::mt19937 rng(2718);

// ---- host dtypes ------------------------------------------------------------------------------------------------
static float round_to(float x, int dt) {
  if (dt == MLOPS_NORM_FP32) return x;
  if (dt == MLOPS_NORM_FP16) return __half2float(__float2half_rn(x));
  uint32_t u;
  memcpy(&u, &x, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) u = 0x7fc00000u;
  u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
  memcpy(&x, &u, 4);
  return x;
}
static int dsize(int dt) { return dt == MLOPS_NORM_FP32 ? 4 : 2; }
static void put(std::vector<unsigned char> &b, size_t i, float v, int dt) {  // v already exactly representable
  if (dt == MLOPS_NORM_FP32) memcpy(&b[4 * i], &v, 4);
  else if (dt == MLOPS_NORM_FP16) { const __half h = __float2half_rn(v); memcpy(&b[2 * i], &h, 2); }
  else { uint32_t u; memcpy(&u, &v, 4); const uint16_t h = (uint16_t)(u >> 16); memcpy(&b[2 * i], &h, 2); }
}
static float get(const std::vector<unsigned char> &b, size_t i, int dt) {
  if (dt == MLOPS_NORM_FP32) { float v; memcpy(&v, &b[4 * i], 4); return v; }
  if (dt == MLOPS_NORM_FP16) { __half h; memcpy(&h, &b[2 * i], 2); return __half2float(h); }
  uint16_t h; memcpy(&h, &b[2 * i], 2);
  const uint32_t u = (uint32_t)h << 16;
  float v; memcpy(&v, &u, 4);
  return v;
}

// ---- the sums of the call -----------------------------------------------------------------------------------------
static void butterfly(float *l) {
  for (int o = 16; o; o >>= 1) {
    float n[32];
    for (int i = 0; i < 32; ++i) n[i] = l[i] + l[i ^ o];
    memcpy(l, n, sizeof n);
  }
}
// the sum of a group of units (256 values each): a lane's eight in order, a butterfly, the unit sums lane-strided, a butterfly
static float group_sum(const std::vector<float> &q, int first_unit, int units) {
  std::vector<float> part(units);
  for (int u = 0; u < units; ++u) {
    float lane[32];
    for (int l = 0; l < 32; ++l) {
      float s = 0.f;
      for (int e = 0; e < 8; ++e) s = s + q[256 * (first_unit + u) + 8 * l + e];
      lane[l] = s;
    }
    butterfly(lane);
    part[u] = lane[0];
  }
  float lane[32] = {};
  for (int u = 0; u < units; ++u) lane[u % 32] = lane[u % 32] + part[u];
  butterfly(lane);
  return lane[0];
}
static float host_rsqrt(float a) { return (float)(1.0 / sqrt((double)a)); }
static float host_silu(float g) { return g / (1.f + expf(-g)); }

struct Case {
  const char *name;
  int rows, hidden, groups;
  mlops_norm_kind kind;
  mlops_norm_scale scale;
  bool bias;
  mlops_norm_order order;
  mlops_norm_gate gate;
  int xd, yd, out, ld_scale;
  bool residual, in_place;
  float g;
};

static int report(const char *what, long bad, long n) {
  printf("[%s] %s (%ld of %ld differ)\n", bad ? "FAIL" : "PASS", what, bad, n);
  return bad != 0;
}

static int run_case(const Case &k, int items) {
  const int R = k.rows, H = k.hidden, G = k.groups ? k.groups : 1, GS = H / G, ld = k.ld_scale ? k.ld_scale : H / 16;
  std::normal_distribution<float> d(0.f, 1.5f);
  const size_t xb = (size_t)R * H * dsize(k.xd);
  std::vector<unsigned char> x(xb), res(xb), gate(xb), w((size_t)H * dsize(k.yd)), bs((size_t)H * dsize(k.yd));
  for (size_t i = 0; i < (size_t)R * H; ++i) {
    put(x, i, round_to(d(rng) * (1.f + (i % 7)) + 0.5f, k.xd), k.xd);
    put(res, i, round_to(d(rng), k.xd), k.xd);
    put(gate, i, round_to(d(rng), k.xd), k.xd);
  }
  for (int i = 0; i < H; ++i) { put(w, i, round_to(0.5f + d(rng) * 0.2f, k.yd), k.yd); put(bs, i, round_to(d(rng) * 0.3f, k.yd), k.yd); }
  const float eps = 1e-6f;

  // the reference
  std::vector<float> yref((size_t)R * H);
  std::vector<unsigned char> vref(xb);
  for (int r = 0; r < R; ++r) {
    std::vector<float> v(H), vv(H);  // v as stored; vv after a gate before the norm
    for (int i = 0; i < H; ++i) {
      v[i] = get(x, (size_t)r * H + i, k.xd);
      if (k.residual) v[i] = round_to(v[i] + get(res, (size_t)r * H + i, k.xd), k.xd);
      put(vref, (size_t)r * H + i, v[i], k.xd);
      vv[i] = k.gate == MLOPS_NORM_GATE_BEFORE ? v[i] * host_silu(get(gate, (size_t)r * H + i, k.xd)) : v[i];
    }
    for (int g = 0; g < G; ++g) {
      const int u0 = g * GS / 256, units = GS / 256;
      std::vector<float> sq(H), seg(GS);
      float mean = 0.f;
      if (k.kind == MLOPS_NORM_LAYER) mean = group_sum(vv, u0, units) * (1.f / (float)GS);
      for (int i = 0; i < H; ++i) { const float c = k.kind == MLOPS_NORM_LAYER ? vv[i] - mean : vv[i]; sq[i] = (i / GS == g) ? c * c : 0.f; }
      std::vector<float> sg(vv.begin() + g * GS, vv.begin() + (g + 1) * GS), sqg(GS);
      for (int i = 0; i < GS; ++i) { const float c = k.kind == MLOPS_NORM_LAYER ? sg[i] - mean : sg[i]; sqg[i] = c * c; }
      const float rstd = host_rsqrt(group_sum(sqg, 0, units) * (1.f / (float)GS) + eps);
      for (int i = g * GS; i < (g + 1) * GS; ++i) {
        const float c = k.kind == MLOPS_NORM_LAYER ? vv[i] - mean : vv[i], n = c * rstd;
        const float wi = k.scale == MLOPS_NORM_SCALE_NONE ? 1.f : k.scale == MLOPS_NORM_SCALE_WEIGHT ? get(w, i, k.yd) : 1.f + get(w, i, k.yd);
        float y;
        if (k.order == MLOPS_NORM_ORDER_ROUND_THEN_SCALE) y = round_to(round_to(n, k.xd) * wi, k.yd);
        else {
          y = std::fmaf(n, wi, k.bias ? get(bs, i, k.yd) : 0.f);
          if (k.gate == MLOPS_NORM_GATE_AFTER) y *= host_silu(get(gate, (size_t)r * H + i, k.xd));
          y = round_to(y, k.yd);
        }
        yref[(size_t)r * H + i] = y;
      }
    }
  }

  void *dx, *dres, *dgate, *dw, *dbs, *dro, *dy, *dsc;
  float *dg;
  const size_t ybytes = k.out == MLOPS_NORM_OUT_DTYPE ? (size_t)R * H * dsize(k.yd) : k.out == MLOPS_NORM_OUT_INT8 ? (size_t)R * H : (size_t)R * H / 2;
  const size_t sbytes = k.out == MLOPS_NORM_OUT_INT8 ? (size_t)R * ld * 4 : (size_t)R * ld;
  CK(cudaMalloc(&dx, xb)); CK(cudaMalloc(&dres, xb)); CK(cudaMalloc(&dgate, xb)); CK(cudaMalloc(&dw, w.size())); CK(cudaMalloc(&dbs, bs.size()));
  CK(cudaMalloc(&dro, xb)); CK(cudaMalloc(&dy, ybytes)); CK(cudaMalloc(&dsc, sbytes + 4)); CK(cudaMalloc(&dg, 4));
  CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dbs, bs.data(), bs.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dgate, gate.data(), xb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dg, &k.g, 4, cudaMemcpyHostToDevice));

  mlops_norm_config c{R, H, k.groups, eps, k.kind, k.scale, k.bias, k.order, k.gate, (mlops_norm_dtype)k.xd, (mlops_norm_dtype)k.yd,
                      (mlops_norm_out)k.out, k.ld_scale, items};
  const int nv = mlops_norm_variants(&c, nullptr, 0);
  std::vector<mlops_norm_variant> vs(nv);
  mlops_norm_variants(&c, vs.data(), nv);
  long bad = 0, n = 0;
  auto args = [&](void *ro, void *y, void *sc) {
    mlops_norm_args a{};
    a.x = dx; a.residual = k.residual ? dres : nullptr; a.weight = k.scale != MLOPS_NORM_SCALE_NONE ? dw : nullptr; a.bias = k.bias ? dbs : nullptr;
    a.gate = k.gate != MLOPS_NORM_GATE_NONE ? dgate : nullptr; a.residual_out = k.residual ? ro : nullptr; a.y = y; a.scales = sc;
    a.global = k.out == MLOPS_NORM_OUT_NVFP4 ? dg : nullptr;
    return a;
  };
  for (const auto &v : vs) {
    CK(cudaMemcpy(dx, x.data(), xb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dres, res.data(), xb, cudaMemcpyHostToDevice));
    CK(cudaMemset(dy, 0xcd, ybytes)); CK(cudaMemset(dsc, 0xcd, sbytes)); CK(cudaMemset(dro, 0xcd, xb));
    void *ro = k.in_place ? dx : dro;
    mlops_norm_args a = args(ro, dy, dsc);
    if (mlops_norm(&c, &v, &a, 0)) { fprintf(stderr, "%s\n", mlops_norm_last_error()); return 1; }
    CK(cudaDeviceSynchronize());
    std::vector<unsigned char> got(ybytes), gsc(sbytes), gro(xb);
    CK(cudaMemcpy(got.data(), dy, ybytes, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(gsc.data(), dsc, sbytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gro.data(), ro, xb, cudaMemcpyDeviceToHost));
    if (k.residual) { bad += memcmp(gro.data(), vref.data(), vref.size()) != 0; ++n; }
    if (k.out == MLOPS_NORM_OUT_DTYPE) {
      for (size_t i = 0; i < (size_t)R * H; ++i) {
        const float g = get(got, i, k.yd), want = yref[i];
        if (k.gate == MLOPS_NORM_GATE_NONE) bad += g != want;
        else bad += std::fabs(g - want) > 3e-2f * std::fabs(want) + 1e-3f;  // libm's exp on the device, the host's
        ++n;
      }
    }
    if (k.out != MLOPS_NORM_OUT_DTYPE) {
      // y of the call (the same call in the output dtype), quantized by NVFP4EffMA: the same bytes
      void *dyy;
      CK(cudaMalloc(&dyy, (size_t)R * H * dsize(k.yd)));
      mlops_norm_config cy = c;
      cy.out = MLOPS_NORM_OUT_DTYPE;
      mlops_norm_variant vy;
      mlops_norm_variants(&cy, &vy, 1);
      CK(cudaMemcpy(dx, x.data(), xb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dres, res.data(), xb, cudaMemcpyHostToDevice));
      mlops_norm_args ay = args(dro, dyy, nullptr);
      if (mlops_norm(&cy, &vy, &ay, 0)) return 1;
      nvfp4a_tensor src{};
      src.data = dyy;
      src.dtype = k.yd == MLOPS_NORM_BF16 ? NVFP4A_BF16 : k.yd == MLOPS_NORM_FP16 ? NVFP4A_FP16 : NVFP4A_FP32;
      nvfp4a_tensor dst{};
      void *qd, *qs;
      CK(cudaMalloc(&qd, ybytes)); CK(cudaMalloc(&qs, sbytes + 16));
      dst.data = qd; dst.block_scale = qs; dst.ld_scale = ld; dst.global_scale = k.out == MLOPS_NORM_OUT_NVFP4 ? dg : nullptr;
      dst.dtype = k.out == MLOPS_NORM_OUT_INT8 ? NVFP4A_INT8 : NVFP4A_NVFP4;
      if (nvfp4a_quantize(&src, H, &dst, H, R, H, 0) != NVFP4A_OK) { fprintf(stderr, "nvfp4a_quantize failed\n"); return 1; }
      CK(cudaDeviceSynchronize());
      std::vector<unsigned char> rd(ybytes), rs(sbytes);
      CK(cudaMemcpy(rd.data(), qd, ybytes, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(rs.data(), qs, sbytes, cudaMemcpyDeviceToHost));
      bad += memcmp(rd.data(), got.data(), ybytes) != 0;
      const size_t sz = k.out == MLOPS_NORM_OUT_INT8 ? 4 : 1;
      for (int r = 0; r < R; ++r) bad += memcmp(&rs[(size_t)r * ld * sz], &gsc[(size_t)r * ld * sz], (size_t)H / 16 * sz) != 0;
      n += 1 + R;
      // and y itself is the reference's (bit for bit when there is no gate)
      if (k.gate == MLOPS_NORM_GATE_NONE) {
        std::vector<unsigned char> yy((size_t)R * H * dsize(k.yd));
        CK(cudaMemcpy(yy.data(), dyy, yy.size(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < (size_t)R * H; ++i) { bad += get(yy, i, k.yd) != yref[i]; ++n; }
      }
      CK(cudaFree(dyy)); CK(cudaFree(qd)); CK(cudaFree(qs));
    }
  }
  char what[220];
  snprintf(what, sizeof what, "%s: %d x %d x%d y%d%s%s%s, %d variants, items %d", k.name, R, H, k.xd, k.yd, k.groups > 1 ? " groups" : "", k.residual ? ", residual" : "", k.in_place ? " in place" : "", nv, items);
  CK(cudaFree(dx)); CK(cudaFree(dres)); CK(cudaFree(dgate)); CK(cudaFree(dw)); CK(cudaFree(dbs)); CK(cudaFree(dro)); CK(cudaFree(dy)); CK(cudaFree(dsc)); CK(cudaFree(dg));
  return report(what, bad, n);
}

static mlops_norm_config base(int rows, int hidden) {
  return mlops_norm_config{rows, hidden, 0, 1e-6f, MLOPS_NORM_RMS, MLOPS_NORM_SCALE_WEIGHT, 0, MLOPS_NORM_ORDER_ROUND_THEN_SCALE, MLOPS_NORM_GATE_NONE,
                           MLOPS_NORM_BF16, MLOPS_NORM_BF16, MLOPS_NORM_OUT_DTYPE, 0, 0};
}

// the E2M1 conversion of the kernels against NVFP4EffMA's, for every float
__global__ void e2m1_check(unsigned long long *bad) {
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < (1ull << 32); i += (unsigned long long)gridDim.x * blockDim.x) {
    const float x = __uint_as_float((unsigned)i);
    if (mlops_norm_detail::e2m1(x) != nvfp4a::f32_to_e2m1(x)) atomicAdd(bad, 1ull);
  }
}
static int run_e2m1() {
  unsigned long long *d, h = 0;
  CK(cudaMalloc(&d, 8)); CK(cudaMemset(d, 0, 8));
  e2m1_check<<<4096, 256>>>(d);
  CK(cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost)); CK(cudaFree(d));
  return report("e2m1 conversion against nvfp4a::f32_to_e2m1, all 2^32 floats", (long)h, 1l << 32);
}

static int run_invalid() {
  long bad = 0, n = 0;
  void *p;
  CK(cudaMalloc(&p, 1 << 20));
  const mlops_norm_config good = base(4, 1024);
  mlops_norm_variant v[4];
  bad += mlops_norm_variants(&good, v, 4) < 1;
  const auto with = [&](auto edit) { mlops_norm_config c = good; edit(c); return c; };
  const mlops_norm_config broken[] = {
      with([](auto &c) { c.rows = 0; }), with([](auto &c) { c.hidden = 1000; }), with([](auto &c) { c.eps = -1.f; }),
      with([](auto &c) { c.y_dtype = MLOPS_NORM_FP16; }), with([](auto &c) { c.out = (mlops_norm_out)3; }), with([](auto &c) { c.out = MLOPS_NORM_OUT_INT8; c.ld_scale = 10; }),
      with([](auto &c) { c.hidden = 65536; }), with([](auto &c) { c.groups = 3; }), with([](auto &c) { c.hidden = 768; c.groups = 2; }),
      with([](auto &c) { c.kind = MLOPS_NORM_LAYER; }),                                   // Hugging Face's order is RMSNorm's
      with([](auto &c) { c.bias = 1; }),                                                  // a bias is a single-rounding thing
      with([](auto &c) { c.gate = MLOPS_NORM_GATE_AFTER; }),                              // so is a gate
      with([](auto &c) { c.groups = -1; })};
  for (const auto &b : broken) { bad += mlops_norm_variants(&b, nullptr, 0) != 0; ++n; }
  mlops_norm_args a{};
  a.x = p; a.weight = p; a.y = p;
  bad += mlops_norm(&good, &v[0], &a, 0) != MLOPS_NORM_OK;
  mlops_norm_args none = a;
  none.x = nullptr;
  bad += mlops_norm(&good, &v[0], &none, 0) != MLOPS_NORM_INVALID_VALUE;
  mlops_norm_args res = a;
  res.residual = p;  // residual without residual_out
  bad += mlops_norm(&good, &v[0], &res, 0) != MLOPS_NORM_INVALID_VALUE;
  mlops_norm_args mis = a;
  mis.x = (char *)p + 2;
  bad += mlops_norm(&good, &v[0], &mis, 0) != MLOPS_NORM_INVALID_VALUE;
  mlops_norm_args noweight = a;
  noweight.weight = nullptr;  // scale WEIGHT needs it
  bad += mlops_norm(&good, &v[0], &noweight, 0) != MLOPS_NORM_INVALID_VALUE;
  mlops_norm_config q = good;
  q.out = MLOPS_NORM_OUT_INT8;
  mlops_norm_variant vq[4];
  mlops_norm_variants(&q, vq, 4);
  bad += mlops_norm(&q, &vq[0], &a, 0) != MLOPS_NORM_INVALID_VALUE;  // quantized, no scales
  mlops_norm_config g = good;
  g.order = MLOPS_NORM_ORDER_FP32;
  g.gate = MLOPS_NORM_GATE_AFTER;
  mlops_norm_variant vg[4];
  mlops_norm_variants(&g, vg, 4);
  bad += mlops_norm(&g, &vg[0], &a, 0) != MLOPS_NORM_INVALID_VALUE;  // a gate without its tensor
  const mlops_norm_config tiny = base(4, 256);  // one unit: only 1 warp fits
  bad += mlops_norm_variants(&tiny, nullptr, 0) != 1;
  mlops_norm_variant odd{3, 96, 0, 0, 0};
  bad += mlops_norm(&good, &odd, &a, 0) != MLOPS_NORM_INVALID_VALUE;
  n += 9;
  return report("invalid arguments", bad, n);
}

static void bench() {
  struct Row { int rows, hidden, xd, yd; };
  const Row rows[] = {{1, 4096, MLOPS_NORM_FP32, MLOPS_NORM_BF16}, {8, 4096, MLOPS_NORM_FP32, MLOPS_NORM_BF16}, {512, 4096, MLOPS_NORM_FP32, MLOPS_NORM_BF16},
                      {2048, 4096, MLOPS_NORM_BF16, MLOPS_NORM_BF16}, {2048, 1024, MLOPS_NORM_BF16, MLOPS_NORM_BF16}, {8192, 4096, MLOPS_NORM_BF16, MLOPS_NORM_BF16}};
  struct Flavor { const char *name; mlops_norm_kind kind; mlops_norm_scale scale; int bias; mlops_norm_order order; };
  const Flavor flavors[] = {{"rms hf", MLOPS_NORM_RMS, MLOPS_NORM_SCALE_WEIGHT, 0, MLOPS_NORM_ORDER_ROUND_THEN_SCALE},
                            {"rms gemma", MLOPS_NORM_RMS, MLOPS_NORM_SCALE_ONE_PLUS_WEIGHT, 0, MLOPS_NORM_ORDER_FP32},
                            {"layernorm", MLOPS_NORM_LAYER, MLOPS_NORM_SCALE_WEIGHT, 1, MLOPS_NORM_ORDER_FP32}};
  printf("%-10s %6s %6s %5s %5s %-6s %-8s %12s %12s %9s\n", "flavor", "rows", "hidden", "x", "y", "output", "residual", "latency us", "traffic GB/s", "of copy");
  void *a, *b;
  const size_t big = (size_t)256 << 20;
  CK(cudaMalloc(&a, big)); CK(cudaMalloc(&b, big));
  CK(cudaMemset(a, 0, big));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  float ms;
  // the best of several windows: the clocks of a card that also drives a display move by tens of percent
  float copy_ms = 1e9f;
  for (int w = 0; w < 8; ++w) {
    CK(cudaEventRecord(e0));
    for (int i = 0; i < 20; ++i) CK(cudaMemcpyAsync(b, a, 64 << 20, cudaMemcpyDeviceToDevice));
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1));
    copy_ms = std::min(copy_ms, ms / 20);
  }
  const double copy = 2.0 * (64 << 20) / (copy_ms * 1e-3) * 1e-9;
  for (const Flavor &f : flavors)
    for (const Row &r : rows) {
      if (f.kind == MLOPS_NORM_LAYER && r.xd != r.yd) continue;
      for (int res = 0; res < 2; ++res)
        for (int out = 0; out < 3; ++out) {
          mlops_norm_config c = base(r.rows, r.hidden);
          c.kind = f.kind; c.scale = f.scale; c.bias = f.bias; c.order = f.order; c.x_dtype = (mlops_norm_dtype)r.xd; c.y_dtype = (mlops_norm_dtype)r.yd; c.out = (mlops_norm_out)out;
          const int nv = mlops_norm_variants(&c, nullptr, 0);
          std::vector<mlops_norm_variant> vs(nv);
          mlops_norm_variants(&c, vs.data(), nv);
          const size_t xb = (size_t)r.rows * r.hidden * dsize(r.xd), yb = out == 0 ? (size_t)r.rows * r.hidden * dsize(r.yd) : out == 1 ? (size_t)r.rows * r.hidden : (size_t)r.rows * r.hidden / 2;
          mlops_norm_args args{};
          args.x = a; args.residual = res ? (char *)a + xb : nullptr; args.weight = (char *)a + 2 * xb; args.bias = f.bias ? (char *)a + 2 * xb + 65536 : nullptr;
          args.residual_out = res ? b : nullptr; args.y = (char *)b + xb; args.scales = (char *)b + xb + ((yb + 255) & ~(size_t)255);  // 256-byte aligned, as a caller's buffers are
          double t_best = 1e9;
          int warps = 0;
          for (int i = 0; i < 300; ++i) mlops_norm(&c, &vs[0], &args, 0);  // clocks up
          for (const auto &v : vs) {
            for (int i = 0; i < 3; ++i) mlops_norm(&c, &v, &args, 0);
            for (int w = 0; w < 6; ++w) {
              CK(cudaEventRecord(e0));
              for (int i = 0; i < 50; ++i) mlops_norm(&c, &v, &args, 0);
              CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1));
              if (ms / 50 * 1e3 < t_best) { t_best = ms / 50 * 1e3; warps = v.warps; }
            }
          }
          const double traffic = (double)xb * (res ? 3 : 1) + yb + (out ? (double)r.rows * r.hidden / 16 * (out == 1 ? 4 : 1) : 0.);
          printf("%-10s %6d %6d %5s %5s %-6s %-8s %12.1f %12.1f %8.0f%%   (%d warps)\n", f.name, r.rows, r.hidden, r.xd == 2 ? "fp32" : "bf16", r.yd == 2 ? "fp32" : "bf16",
                 out == 0 ? "dtype" : out == 1 ? "int8" : "nvfp4", res ? "yes" : "no", t_best, traffic / (t_best * 1e-6) * 1e-9, 100.0 * traffic / (t_best * 1e-6) * 1e-9 / copy, warps);
        }
    }
}

int main(int argc, char **argv) {
  if (argc > 1 && !strcmp(argv[1], "bench")) { bench(); return 0; }
  int f = 0;
  const int B = MLOPS_NORM_BF16, H = MLOPS_NORM_FP16, F = MLOPS_NORM_FP32;
  const auto RMS = MLOPS_NORM_RMS, LN = MLOPS_NORM_LAYER;
  const auto W = MLOPS_NORM_SCALE_WEIGHT, ONE = MLOPS_NORM_SCALE_ONE_PLUS_WEIGHT, NO = MLOPS_NORM_SCALE_NONE;
  const auto HF = MLOPS_NORM_ORDER_ROUND_THEN_SCALE, F32 = MLOPS_NORM_ORDER_FP32;
  const auto NG = MLOPS_NORM_GATE_NONE, AFTER = MLOPS_NORM_GATE_AFTER, BEFORE = MLOPS_NORM_GATE_BEFORE;
  const auto D = MLOPS_NORM_OUT_DTYPE, I8 = MLOPS_NORM_OUT_INT8, N4 = MLOPS_NORM_OUT_NVFP4;
  for (int items : {0, 1, 3}) {
    f += run_case({"rms hf (Llama, Qwen)", 5, 1024, 1, RMS, W, false, HF, NG, B, B, D, 0, false, false, 1.f}, items);
    f += run_case({"rms hf, fp32 stream", 5, 2048, 1, RMS, W, false, HF, NG, F, B, D, 0, true, false, 1.f}, items);
    f += run_case({"rms hf, fp16", 4, 4096, 1, RMS, W, false, HF, NG, F, H, D, 0, true, true, 1.f}, items);
  }
  f += run_case({"rms hf, fp16 / fp16", 7, 1280, 1, RMS, W, false, HF, NG, H, H, D, 0, true, false, 1.f}, 0);
  f += run_case({"rms hf, fp32", 3, 1024, 1, RMS, W, false, HF, NG, F, F, D, 0, true, false, 1.f}, 0);
  f += run_case({"rms hf, no weight", 4, 1024, 1, RMS, NO, false, HF, NG, B, B, D, 0, false, false, 1.f}, 0);
  f += run_case({"rms hf, one plus (zero-centered)", 4, 1024, 1, RMS, ONE, false, HF, NG, B, B, D, 0, false, false, 1.f}, 0);
  f += run_case({"rms fp32 order, one plus (Gemma)", 5, 2048, 1, RMS, ONE, false, F32, NG, B, B, D, 0, true, false, 1.f}, 0);
  f += run_case({"rms fp32 order, weight + bias", 5, 1024, 1, RMS, W, true, F32, NG, B, B, D, 0, false, false, 1.f}, 0);
  f += run_case({"layernorm, weight + bias (GPT, Falcon, Phi)", 6, 1024, 1, LN, W, true, F32, NG, B, B, D, 0, false, false, 1.f}, 0);
  f += run_case({"layernorm, weight, residual", 6, 2048, 1, LN, W, false, F32, NG, H, H, D, 0, true, false, 1.f}, 0);
  f += run_case({"layernorm, no parameters (OLMo)", 4, 1024, 1, LN, NO, false, F32, NG, F, F, D, 0, false, false, 1.f}, 0);
  f += run_case({"layernorm, one plus + bias (LayerNorm1P)", 5, 2048, 1, LN, ONE, true, F32, NG, F, B, D, 0, true, false, 1.f}, 3);
  f += run_case({"rms groups 4 (Mamba-2, Nemotron-H)", 4, 2048, 4, RMS, W, false, F32, NG, B, B, D, 0, false, false, 1.f}, 0);
  f += run_case({"layernorm groups 2", 4, 1024, 2, LN, W, true, F32, NG, B, B, D, 0, false, false, 1.f}, 0);
  f += run_case({"rms, gate after (Qwen3-Next)", 5, 1024, 1, RMS, ONE, false, F32, AFTER, B, B, D, 0, false, false, 1.f}, 0);
  f += run_case({"rms, gate before, groups 4 (Mamba-2)", 4, 2048, 4, RMS, W, false, F32, BEFORE, B, B, D, 0, true, false, 1.f}, 0);
  f += run_case({"layernorm, gate before", 4, 1024, 1, LN, W, true, F32, BEFORE, F, F, D, 0, false, false, 1.f}, 0);
  f += run_case({"rms hf -> int8", 9, 2048, 1, RMS, W, false, HF, NG, B, B, I8, 0, false, false, 1.f}, 0);
  f += run_case({"rms hf, fp32 stream -> int8, padded scales", 9, 2048, 1, RMS, W, false, HF, NG, F, B, I8, 160, true, false, 1.f}, 3);
  f += run_case({"rms hf -> nvfp4, g 0.0123", 6, 4096, 1, RMS, W, false, HF, NG, B, B, N4, 0, true, false, 0.0123f}, 0);
  f += run_case({"rms gemma -> nvfp4, padded scales", 6, 1024, 1, RMS, ONE, false, F32, NG, F, B, N4, 80, false, false, 1.f}, 2);
  f += run_case({"layernorm + bias -> nvfp4 from fp16", 4, 2048, 1, LN, W, true, F32, NG, H, H, N4, 0, true, false, 0.002f}, 0);
  f += run_case({"layernorm + bias -> int8", 4, 1024, 1, LN, W, true, F32, NG, B, B, I8, 0, false, false, 1.f}, 0);
  f += run_case({"rms, gate after -> nvfp4", 4, 1024, 1, RMS, W, false, F32, AFTER, B, B, N4, 0, false, false, 0.5f}, 0);
  f += run_case({"rms gemma, fp16 / fp16, 6 units", 6, 1536, 1, RMS, ONE, false, F32, NG, H, H, D, 0, true, false, 1.f}, 5);
  f += run_case({"layernorm + bias, 4 units", 6, 1024, 1, LN, W, true, F32, NG, B, B, D, 0, false, false, 1.f}, 5);
  f += run_e2m1();
  f += run_invalid();
  printf(f ? "\n%d FAILED\n" : "\nALL PASS\n", f);
  return f != 0;
}
