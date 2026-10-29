// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The norm kernel (include/mlops_norm_layer.h), built as CUTLASS builds a kernel: Element traits (the dtypes), Traits (the
// flavor), OutputOp (the epilogue: dtype, INT8 or NVFP4), Params and SharedStorage, and NormKernel::operator(). A CTA of
// warps takes rows; a row is units of 256 elements, a lane holding 8 consecutive ones; a group is gu = hidden / (256 groups)
// units. The CTA's warps are the launch's (blockDim.x / 32).
#pragma once
#include "mlops_norm_items.h"
#include "nvfp4effma.cuh"  // the exact E2M1 / E4M3 primitives of the quantizer (nvfp4a::), NVRTC-safe

#define MLOPS_NORM_DEVICE __device__ __forceinline__
#define MLOPS_NORM_UNROLL _Pragma("unroll")

namespace mlops_norm_detail {

constexpr int kMaxUnits = 128;      // hidden <= 32768
constexpr int kUnitElements = 256;  // a warp's access: 32 lanes
constexpr int kAccess = 8;          // elements of a lane's access
enum { kBf16 = 0, kFp16 = 1, kFp32 = 2 };

template <typename T, int N> struct Array {
  T v[N];
  MLOPS_NORM_DEVICE T &operator[](int i) { return v[i]; }
  MLOPS_NORM_DEVICE const T &operator[](int i) const { return v[i]; }
};
using Fragment = Array<float, kAccess>;  // a lane's access, widened
using Words = Array<unsigned, 4>;        // the same in a 16-bit dtype: eight halves

// ---- Element: one dtype -------------------------------------------------------------------------------------------
// load widens an access to FP32; round rounds a fragment to nearest even in place and returns its packed words (FP32: none);
// store writes a fragment already rounded. From sm_80 a pair is rounded by one instruction.
template <int Dtype> struct Element;

template <> struct Element<kFp32> {
  static constexpr int kBytes = 4;
  static MLOPS_NORM_DEVICE Fragment load(const void *p, size_t i) {
    const float4 a = __ldcg(reinterpret_cast<const float4 *>(static_cast<const float *>(p) + i));
    const float4 b = __ldcg(reinterpret_cast<const float4 *>(static_cast<const float *>(p) + i) + 1);
    return {{a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w}};
  }
  static MLOPS_NORM_DEVICE Words round(Fragment &) { return {}; }
  static MLOPS_NORM_DEVICE void store(void *p, size_t i, const Fragment &v, const Words &) {
    float4 *q = reinterpret_cast<float4 *>(static_cast<float *>(p) + i);
    q[0] = make_float4(v[0], v[1], v[2], v[3]);
    q[1] = make_float4(v[4], v[5], v[6], v[7]);
  }
};

template <int Dtype> struct Element16 {
  static constexpr int kBytes = 2;
  static MLOPS_NORM_DEVICE Fragment load(const void *p, size_t i) {
    const uint4 t = __ldcg(reinterpret_cast<const uint4 *>(static_cast<const unsigned short *>(p) + i));
    Fragment v;
    unpack(t.x, v[0], v[1]); unpack(t.y, v[2], v[3]); unpack(t.z, v[4], v[5]); unpack(t.w, v[6], v[7]);
    return v;
  }
  static MLOPS_NORM_DEVICE Words round(Fragment &v) {
    Words w;
    MLOPS_NORM_UNROLL
    for (int e = 0; e < kAccess; e += 2) {
      w[e / 2] = pack(v[e], v[e + 1]);
      unpack(w[e / 2], v[e], v[e + 1]);
    }
    return w;
  }
  static MLOPS_NORM_DEVICE void store(void *p, size_t i, const Fragment &, const Words &w) {
    *reinterpret_cast<uint4 *>(static_cast<unsigned short *>(p) + i) = make_uint4(w[0], w[1], w[2], w[3]);
  }
  static MLOPS_NORM_DEVICE void unpack(unsigned w, float &a, float &b);  // the low half first
  static MLOPS_NORM_DEVICE unsigned pack(float a, float b);              // a low
};

template <> struct Element<kBf16> : Element16<kBf16> {};
template <> struct Element<kFp16> : Element16<kFp16> {};

template <> MLOPS_NORM_DEVICE void Element16<kBf16>::unpack(unsigned w, float &a, float &b) {
  a = __uint_as_float(w << 16);
  b = __uint_as_float(w & 0xffff0000u);
}
template <> MLOPS_NORM_DEVICE unsigned Element16<kBf16>::pack(float a, float b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned w;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(w) : "f"(b), "f"(a));  // the first operand is the high half
  return w;
#else
  unsigned ua = __float_as_uint(a), ub = __float_as_uint(b);
  ua = (ua & 0x7fffffffu) > 0x7f800000u ? 0x7fc00000u : ua + 0x7fffu + ((ua >> 16) & 1u);
  ub = (ub & 0x7fffffffu) > 0x7f800000u ? 0x7fc00000u : ub + 0x7fffu + ((ub >> 16) & 1u);
  return (ua >> 16) | (ub & 0xffff0000u);
#endif
}
template <> MLOPS_NORM_DEVICE void Element16<kFp16>::unpack(unsigned w, float &a, float &b) {
  const unsigned short l = (unsigned short)w, h = (unsigned short)(w >> 16);
  asm("cvt.f32.f16 %0, %1;" : "=f"(a) : "h"(l));
  asm("cvt.f32.f16 %0, %1;" : "=f"(b) : "h"(h));
}
template <> MLOPS_NORM_DEVICE unsigned Element16<kFp16>::pack(float a, float b) {
  unsigned short l, h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(l) : "f"(a));
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(b));
  return (unsigned)l | ((unsigned)h << 16);
}

// ---- Params: the arguments of a launch ----------------------------------------------------------------------------
struct Params {
  const void *x, *residual, *weight, *bias, *gate;
  void *residual_out, *y, *scales;
  const float *global;
  int rows, hidden, items, ld_scale, groups;
  float eps, inv_n;        // inv_n = 1 / (values of a group)
  float w_alpha, w_beta;   // w' = fma(weight, alpha, beta): (1, 0) the weight, (1, 1) one plus it, (0, 1) none
  int has_weight, has_bias;
};

// ---- OutputOp: what becomes of a lane's rounded y -------------------------------------------------------------------
// operator()(params, y, words, at, block, lane): at is the element index of the access, block the index of the access's
// scale block in the row's scales. NVFP4 and INT8 follow nvfp4a_quantize (nvfp4effma.cu, quant_nvfp4effma.cuh).
// nvfp4a::f32_to_e2m1 in fewer instructions: below 1 an integer of 2|x| (a magic add), from 1 the mantissa rounded to one bit
// and the exponent field as the code, clamped to 6; equal to it for every float (tests/test_norm.cu checks all 2^32).
MLOPS_NORM_DEVICE unsigned e2m1(float x) {
  const unsigned u = __float_as_uint(x), a = u & 0x7fffffffu, s = (u >> 28) & 8u;
  const unsigned small = __float_as_uint(fmaf(__uint_as_float(a), 2.f, 8388608.f)) - 0x4B000000u;
  const unsigned big = ((a + 0x1fffffu + ((a >> 22) & 1u)) >> 22) - 252u;
  return s | min(a < 0x3f800000u ? small : big, 7u);
}
template <int Out, int Dtype> struct OutputOp;

template <int Dtype> struct OutputOp<MLOPS_NORM_OUT_DTYPE, Dtype> {
  MLOPS_NORM_DEVICE explicit OutputOp(const Params &) {}
  MLOPS_NORM_DEVICE void operator()(const Params &p, const Fragment &y, const Words &w, size_t at, size_t, int) const {
    Element<Dtype>::store(p.y, at, y, w);
  }
};

// the maximum of |y| over a block of 16 values: the access is half of it, the other half is the neighbour lane
MLOPS_NORM_DEVICE float block_amax(const Fragment &y) {
  float m = 0.f;
  MLOPS_NORM_UNROLL
  for (int e = 0; e < kAccess; ++e) m = fmaxf(m, fabsf(y[e]));
  return fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
}

template <int Dtype> struct OutputOp<MLOPS_NORM_OUT_INT8, Dtype> {
  MLOPS_NORM_DEVICE explicit OutputOp(const Params &) {}
  MLOPS_NORM_DEVICE void operator()(const Params &p, const Fragment &y, const Words &, size_t at, size_t block, int lane) const {
    const float sb = __fdiv_rn(block_amax(y), 127.f);  // sb = amax / 127, q = RNE(y / sb)
    unsigned w[2] = {0u, 0u};
    MLOPS_NORM_UNROLL
    for (int e = 0; e < kAccess; ++e) {
      const int q = sb > 0.f ? (int)fminf(fmaxf(rintf(__fdiv_rn(y[e], sb)), -127.f), 127.f) : 0;
      w[e / 4] |= (unsigned)(unsigned char)(signed char)q << (8 * (e % 4));
    }
    *reinterpret_cast<uint2 *>(static_cast<signed char *>(p.y) + at) = make_uint2(w[0], w[1]);
    if (!(lane & 1)) static_cast<float *>(p.scales)[block] = sb;
  }
};

template <int Dtype> struct OutputOp<MLOPS_NORM_OUT_NVFP4, Dtype> {
  float s_enc, s_dec;  // the encode and the decode factor of the global g: S_enc = min(1 / g, FLT_MAX), S_dec = g
  // (quant_global of quant_nvfp4effma.cuh, whose header includes <type_traits>, which NVRTC lacks)
  MLOPS_NORM_DEVICE explicit OutputOp(const Params &p) {
    s_dec = p.global ? __ldg(p.global) : 1.f;
    s_enc = fminf(__fdiv_rn(1.f, s_dec), 3.40282347e38f);
  }
  MLOPS_NORM_DEVICE void operator()(const Params &p, const Fragment &y, const Words &, size_t at, size_t block, int lane) const {
    constexpr float kMax = 3.40282347e38f;
    const float d = fminf(__fmul_rn(block_amax(y), __fmul_rn(s_enc, 1.f / 6.f)), kMax);
    const unsigned code = nvfp4a::f32_to_e4m3(d);
    const float inv = fminf(__frcp_rn(__fmul_rn(nvfp4a::e4m3_value(code), s_dec)), kMax);
    unsigned nibbles = 0u;
    MLOPS_NORM_UNROLL
    for (int e = 0; e < kAccess; ++e) nibbles |= e2m1(__fmul_rn(y[e], inv)) << (4 * e);
    *reinterpret_cast<unsigned *>(static_cast<unsigned char *>(p.y) + at / 2) = nibbles;
    if (!(lane & 1)) static_cast<unsigned char *>(p.scales)[block] = (unsigned char)code;
  }
};

// ---- warp reductions: the order that defines the sums ---------------------------------------------------------------------
// A lane's eight values in order, then an xor butterfly over the warp: a unit's sum.
MLOPS_NORM_DEVICE float unit_sum(const Fragment &q) {
  float s = 0.f;
  MLOPS_NORM_UNROLL
  for (int e = 0; e < kAccess; ++e) s = __fadd_rn(s, q[e]);
  MLOPS_NORM_UNROLL
  for (int o = 16; o; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
  return s;
}
// The unit sums of a group added lane-strided, then by a butterfly.
MLOPS_NORM_DEVICE float group_sum(const float *parts, int count, int lane) {
  float s = 0.f;
  for (int i = lane; i < count; i += 32) s = __fadd_rn(s, parts[i]);
  MLOPS_NORM_UNROLL
  for (int o = 16; o; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
  return s;
}
MLOPS_NORM_DEVICE float silu(float g) { return __fdiv_rn(g, 1.f + expf(-g)); }

// ---- Traits: one flavor on one pair of dtypes with one output -------------------------------------------------------------
template <int Kind, int Order, int Gate, int XDtype, int YDtype, int Out> struct Traits {
  static constexpr bool kLayer = Kind == MLOPS_NORM_LAYER;
  static constexpr bool kRoundThenScale = Order == MLOPS_NORM_ORDER_ROUND_THEN_SCALE;
  static constexpr bool kGateBefore = Gate == MLOPS_NORM_GATE_BEFORE, kGateAfter = Gate == MLOPS_NORM_GATE_AFTER;
  using ElementX = Element<XDtype>;
  using ElementY = Element<YDtype>;
  using Output = OutputOp<Out, YDtype>;
};

// ---- the kernel --------------------------------------------------------------------------------------------------------------
template <class Traits> struct NormKernel {
  using ElementX = typename Traits::ElementX;
  using ElementY = typename Traits::ElementY;

  struct SharedStorage {
    float parts[kMaxUnits];  // the unit sums of the row
    float mean[kMaxUnits];   // per unit: its group's
    float rstd[kMaxUnits];
  };

  const Params &p;
  SharedStorage &smem;
  int warps, warp, lane, units, group_units;

  MLOPS_NORM_DEVICE NormKernel(const Params &params, SharedStorage &shared, unsigned tid)
      : p(params), smem(shared), warps(blockDim.x >> 5), warp(tid >> 5), lane(tid & 31), units(params.hidden / kUnitElements),
        group_units(params.hidden / kUnitElements / params.groups) {}

  // v of an access as the passes see it: the rounded sum x + residual (written to residual_out in the first pass, read from it
  // after), times silu(gate) when the gate comes before the norm
  MLOPS_NORM_DEVICE Fragment value(size_t at, bool first) const {
    Fragment v = ElementX::load(first || !p.residual ? p.x : p.residual_out, at);
    if (first && p.residual) {
      const Fragment r = ElementX::load(p.residual, at);
      MLOPS_NORM_UNROLL
      for (int e = 0; e < kAccess; ++e) v[e] = __fadd_rn(v[e], r[e]);
      const Words w = ElementX::round(v);
      ElementX::store(p.residual_out, at, v, w);
    }
    if constexpr (Traits::kGateBefore) {
      const Fragment z = ElementX::load(p.gate, at);
      MLOPS_NORM_UNROLL
      for (int e = 0; e < kAccess; ++e) v[e] = __fmul_rn(v[e], silu(z[e]));
    }
    return v;
  }

  // The unit sums of a row into smem.parts: of v (the mean) or v^2 (RMS), then of (v - mean)^2 (LayerNorm's second pass)
  template <bool Centered> MLOPS_NORM_DEVICE void sum_units(size_t base, bool first) {
    for (int u = warp; u < units; u += warps) {
      Fragment v = value(base + kUnitElements * u + kAccess * lane, first), q;
      const float m = Centered ? smem.mean[u] : 0.f;
      MLOPS_NORM_UNROLL
      for (int e = 0; e < kAccess; ++e) {
        const float c = Centered ? __fsub_rn(v[e], m) : v[e];
        q[e] = Traits::kLayer && !Centered ? c : __fmul_rn(c, c);
      }
      const float s = unit_sum(q);
      if (!lane) smem.parts[u] = s;
    }
    __syncthreads();
  }

  // Per group, a warp each: the mean (LayerNorm's first sum) or 1 / sqrt(mean of squares + eps), written for each unit of the group
  template <bool Mean> MLOPS_NORM_DEVICE void group_statistics() {
    for (int g = warp; g < p.groups; g += warps) {
      const float s = group_sum(smem.parts + g * group_units, group_units, lane);
      const float t = Mean ? __fmul_rn(s, p.inv_n) : __frsqrt_rn(__fadd_rn(__fmul_rn(s, p.inv_n), p.eps));
      for (int i = lane; i < group_units; i += 32) (Mean ? smem.mean : smem.rstd)[g * group_units + i] = t;
    }
    __syncthreads();
  }

  // y of an access, rounded to ElementY's dtype
  MLOPS_NORM_DEVICE Fragment normalize(size_t at, int u, Words &words) const {
    Fragment v = value(at, false), w = {}, b = {}, n, y;
    if (p.has_weight) w = ElementY::load(p.weight, (size_t)kUnitElements * u + kAccess * lane);
    if (p.has_bias) b = ElementY::load(p.bias, (size_t)kUnitElements * u + kAccess * lane);
    const float r = smem.rstd[u], m = Traits::kLayer ? smem.mean[u] : 0.f;
    MLOPS_NORM_UNROLL
    for (int e = 0; e < kAccess; ++e) n[e] = __fmul_rn(Traits::kLayer ? __fsub_rn(v[e], m) : v[e], r);
    if constexpr (Traits::kRoundThenScale) {
      ElementX::round(n);  // Hugging Face: the normalized value rounds to x's dtype before the weight is applied
      MLOPS_NORM_UNROLL
      for (int e = 0; e < kAccess; ++e) y[e] = __fmul_rn(n[e], __fmaf_rn(w[e], p.w_alpha, p.w_beta));
    } else {
      Fragment z = {};
      if constexpr (Traits::kGateAfter) z = ElementX::load(p.gate, at);
      MLOPS_NORM_UNROLL
      for (int e = 0; e < kAccess; ++e) {
        y[e] = __fmaf_rn(n[e], __fmaf_rn(w[e], p.w_alpha, p.w_beta), b[e]);
        if constexpr (Traits::kGateAfter) y[e] = __fmul_rn(y[e], silu(z[e]));
      }
    }
    words = ElementY::round(y);
    return y;
  }

  // Rows [item rows / items, (item + 1) rows / items); the CTA is synchronized on return
  MLOPS_NORM_DEVICE void operator()(int item) {
    const typename Traits::Output output(p);
    const long long lo = (long long)item * p.rows / p.items, hi = (long long)(item + 1) * p.rows / p.items;
    for (long long row = lo; row < hi; ++row) {
      const size_t base = (size_t)row * p.hidden;
      sum_units<false>(base, true);
      group_statistics<Traits::kLayer>();
      if constexpr (Traits::kLayer) {
        sum_units<true>(base, false);
        group_statistics<false>();
      }
      for (int u = warp; u < units; u += warps) {
        const size_t at = base + kUnitElements * u + kAccess * lane;
        Words words;
        const Fragment y = normalize(at, u, words);
        output(p, y, words, at, (size_t)row * p.ld_scale + 16 * u + (lane >> 1), lane);
      }
      __syncthreads();  // the row's sums are rewritten by the next one
    }
  }
};

}  // namespace mlops_norm_detail
