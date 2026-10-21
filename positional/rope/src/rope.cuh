// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// RoPE device code (include/mlops_rope.h): the phase of a position in turns, the dtype codecs and the rotation
// of one pair in both arithmetics. Header-only and free of the kernel's layout, so fused kernels (attention,
// cache writes) rotate in registers with the same bits as mlops_rope.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include "mlops_rope.h"

namespace mlops_rope_detail {

struct Freq { float fh, fl, gh, gl; };  // f = inv_freq / 2pi = fh + fl, g = frac(2^16 f) = gh + gl (mlops_rope_freq)

// frac(x * (hi + lo)) for an integer-valued |x| <= 2^16, within [-1/2, 1/2] up to a few 2^-24: the product's
// rounding error is recovered exactly by the FMA, the integer part removed exactly before the small terms join
__device__ __forceinline__ float frac_mul(float x, float hi, float lo) {
  const float p = __fmul_rn(x, hi), e = __fmaf_rn(x, hi, -p);
  return __fadd_rn(__fsub_rn(p, rintf(p)), __fmaf_rn(x, lo, e));
}

// the phase m * f in turns, reduced to [-1/2, 1/2], for every |m| < 2^31: m = 2^16 a + b
__device__ __forceinline__ float phase_turns(int32_t m, const Freq &f) {
  const float a = (float)(m >> 16), b = (float)(m & 0xFFFF);
  const float t = __fadd_rn(frac_mul(a, f.gh, f.gl), frac_mul(b, f.fh, f.fl));
  return __fsub_rn(t, rintf(t));
}

// cos and sin of the angle, times the attention factor (exact: sincospi on the reduced phase)
__device__ __forceinline__ void angle_cos_sin(int32_t m, const Freq &f, float att, float &c, float &s) {
  sincospif(__fmul_rn(2.0f, phase_turns(m, f)), &s, &c);
  c = __fmul_rn(c, att);
  s = __fmul_rn(s, att);
}

// ---- dtypes: storage type, decode, rounding to the dtype (as a float), encode ---------------------------
// FP32 bits -> BF16 bits in the high half, round to nearest even; NaN -> canonical 0x7FC0 (as torch)
__device__ __forceinline__ uint32_t bf16_rne_hi(uint32_t u) {
  u = (u & 0x7FFFFFFFu) > 0x7F800000u ? 0x7FC00000u : u;
  return (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
}

template <int DT> struct Elem;
template <> struct Elem<MLOPS_ROPE_BF16> {
  using S = uint16_t;
  static __device__ __forceinline__ float load(S v) { return __uint_as_float((uint32_t)v << 16); }
  static __device__ __forceinline__ float round(float v) { return __uint_as_float(bf16_rne_hi(__float_as_uint(v))); }
  static __device__ __forceinline__ S store(float v) { return (S)(bf16_rne_hi(__float_as_uint(v)) >> 16); }
};
template <> struct Elem<MLOPS_ROPE_FP16> {
  using S = __half;
  static __device__ __forceinline__ float load(S v) { return __half2float(v); }
  static __device__ __forceinline__ float round(float v) { return __half2float(__float2half_rn(v)); }
  static __device__ __forceinline__ S store(float v) { return __float2half_rn(v); }
};
template <> struct Elem<MLOPS_ROPE_FP32> {
  using S = float;
  static __device__ __forceinline__ float load(S v) { return v; }
  static __device__ __forceinline__ float round(float v) { return v; }
  static __device__ __forceinline__ S store(float v) { return v; }
};

// ---- the rotation of one pair; cs = (cos1, sin1, cos2, sin2): the factors of the first and the second output --
// exact arithmetic: FP32, one rounding to the dtype at the store
__device__ __forceinline__ void rotate_exact(float x1, float x2, const float4 &cs, float &y1, float &y2) {
  y1 = __fmaf_rn(x1, cs.x, __fmul_rn(-x2, cs.y));
  y2 = __fmaf_rn(x2, cs.z, __fmul_rn(x1, cs.w));
}
// Hugging Face's apply_rotary_pos_emb, q * cos + rotate_half(q) * sin, every operation rounded to the dtype
template <int DT>
__device__ __forceinline__ void rotate_hf(float x1, float x2, const float4 &cs, float &y1, float &y2) {
  using E = Elem<DT>;
  y1 = E::round(__fadd_rn(E::round(__fmul_rn(x1, cs.x)), E::round(__fmul_rn(-x2, cs.y))));
  y2 = E::round(__fadd_rn(E::round(__fmul_rn(x2, cs.z)), E::round(__fmul_rn(x1, cs.w))));
}

}  // namespace mlops_rope_detail
