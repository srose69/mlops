// mlops_adamw_qat number formats the sibling repos do not have: stochastic rounding (SR) into E4M3,
// E5M2 (E[s^2] exact), BF16 and fixed point, the secant maps of the E2M1 lattice, the Quake rsqrt and
// the counter-based uniforms SR draws from. Decodes and RNE encodes are NVFP4EffMA's (E2M1, E4M3); the
// FP8 ones (E5M2 encode, E4M3 / E5M2 x4 decode) are E5M2EffMA's below sm_89 and the native cvt from
// sm_89 on (QAT_FP8_CVT). Host + device, so the tests run the same code on the CPU; arithmetic goes
// through mul / add / sub / fmar so neither side contracts it differently.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <nvfp4effma.cuh>

// FP8 codec source of this compilation pass:
//   QAT_FP8_CVT        device pass for sm_89+: cvt.rn.f16x2.{e4m3x2,e5m2x2} / cvt.rn.satfinite.e5m2x2.f32
//                      (PTX ISA 7.8 on sm_90, 8.1 = CUDA 12.1 on sm_89; the gating of CUTLASS's float8.h)
//   QAT_FP8_CVT_MODEL  tests only: the same wrappers through cuda_fp8.h, NVIDIA's model of those cvt
//                      instructions below sm_90, so the sm_89+ code path runs on any GPU
//   otherwise          E5M2EffMA (sm_50..sm_87 device passes and the host pass)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
#if __CUDA_ARCH__ == 890 && (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 1))
#error "mlops_adamw_qat for sm_89 needs CUDA 12.1+ (PTX ISA 8.1: cvt with .e4m3x2 / .e5m2x2); drop sm_89 or upgrade nvcc"
#endif
#define QAT_FP8_CVT 1
#elif defined(__CUDA_ARCH__) && defined(QAT_FP8_CVT_MODEL)
#include <cuda_fp8.h>
#else
#include <e5m2effma.cuh>
#endif

#if defined(__CUDACC__)
#define QAT_HD __host__ __device__ __forceinline__
#else
#define QAT_HD inline
#endif

namespace qat {

// ---- FP8 codecs ---------------------------------------------------------------------------------------
// Finite codes decode exactly on every path (E5M2EffMA's raw-bits FMUL == cvt, checked on all 256 codes);
// non-finite codes are outside the contract: cvt gives Inf / NaN (as torch and the PyTorch reference do),
// E5M2EffMA large finite values. The stored r (E4M3, <= 0x7E) and s (E5M2, <= 123) are always finite.
#if defined(__CUDA_ARCH__) && (defined(QAT_FP8_CVT) || defined(QAT_FP8_CVT_MODEL))
// bytes 0, 1 of the low half of w -> f16 halves 0, 1 of the result (cvt: byte k -> half k)
template <bool E5M2> __device__ __forceinline__ uint32_t cvt_f16x2_fp8x2(uint32_t w16) {
  uint32_t h;
#if defined(QAT_FP8_CVT)
  if (E5M2) asm("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(h) : "h"((uint16_t)w16));
  else asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((uint16_t)w16));
#else
  const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)w16, E5M2 ? __NV_E5M2 : __NV_E4M3);
  h = (uint32_t)r.x | ((uint32_t)r.y << 16);
#endif
  return h;
}
__device__ __forceinline__ void f16x2_to_f32(uint32_t h, float &lo, float &hi) {  // exact
  asm("cvt.f32.f16 %0, %1;" : "=f"(lo) : "h"((uint16_t)(h & 0xFFFFu)));
  asm("cvt.f32.f16 %0, %1;" : "=f"(hi) : "h"((uint16_t)(h >> 16)));
}
template <bool E5M2> __device__ __forceinline__ void dec_fp8x4(uint32_t w, float &v0, float &v1, float &v2, float &v3) {
  f16x2_to_f32(cvt_f16x2_fp8x2<E5M2>(w & 0xFFFFu), v0, v1);
  f16x2_to_f32(cvt_f16x2_fp8x2<E5M2>(w >> 16), v2, v3);
}
#else
template <bool E5M2> __device__ __forceinline__ void dec_fp8x4(uint32_t w, float &v0, float &v1, float &v2, float &v3) {
  if (E5M2) e5ma::decode_e5m2x4(w, v0, v1, v2, v3);
  else e5ma::decode_e4m3x4(w, v0, v1, v2, v3);
}
#endif

// FP32 -> E5M2 byte, RNE, saturating to +-57344 (cvt.rn.satfinite.e5m2x2.f32: b -> the low byte)
QAT_HD uint32_t enc_e5m2(float x) {
#if defined(__CUDA_ARCH__) && defined(QAT_FP8_CVT)
  uint16_t d;
  asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(d) : "f"(0.0f), "f"(x));
  return d & 0xFFu;
#elif defined(__CUDA_ARCH__) && defined(QAT_FP8_CVT_MODEL)
  return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E5M2);
#else
  return e5ma::f32_to_e5m2(x);
#endif
}

#ifdef __CUDA_ARCH__
QAT_HD float mul(float a, float b) { return nvfp4a::mul_rn(a, b); }
QAT_HD float fmar(float a, float b, float c) { return nvfp4a::fma_rn(a, b, c); }
QAT_HD float add(float a, float b) { return __fadd_rn(a, b); }
QAT_HD float sub(float a, float b) { return __fsub_rn(a, b); }
QAT_HD uint32_t bits(float x) { return __float_as_uint(x); }
QAT_HD float flt(uint32_t u) { return __uint_as_float(u); }
#else
QAT_HD float mul(float a, float b) { return a * b; }
QAT_HD float fmar(float a, float b, float c) { return std::fmaf(a, b, c); }
QAT_HD float add(float a, float b) { return a + b; }
QAT_HD float sub(float a, float b) { return a - b; }
QAT_HD uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }
QAT_HD float flt(uint32_t u) { float x; std::memcpy(&x, &u, 4); return x; }
#endif

QAT_HD float pow2(int e) { return flt((uint32_t)(e + 127) << 23); }     // normal range only
QAT_HD int expo(float x) { return (int)((bits(x) >> 23) & 255u) - 127; } // x > 0 normal

// ---- counter-based uniforms (lowbias32 mixer) ------------------------------------------------------
QAT_HD uint32_t mix(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return x;
}
QAT_HD uint32_t step_key(uint32_t seed, int step) { return mix(seed ^ ((uint32_t)step * 0x9E3779B9U)); }
QAT_HD uint32_t xorshift(uint32_t x) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }
// Per element, from a stream seeded once per thread with one mix (two 32-bit multiplies, emulated on
// Pascal) and advanced by multiply-free xorshift steps: u and s get 24-bit uniforms (their updates are
// small fractions of a quantum, so SR needs fine probabilities), r and qbar 16-bit ones; all independent.
struct Draws { float u, s, r, qbar; };
QAT_HD Draws draws(uint32_t &x) {
  const uint32_t w0 = x = xorshift(x), w1 = x = xorshift(x), w2 = x = xorshift(x);
  return {(float)(w0 >> 8) * 0x1p-24f, (float)(w1 >> 8) * 0x1p-24f, (float)(w2 & 0xFFFFu) * 0x1p-16f,
          (float)(w2 >> 16) * 0x1p-16f};
}
QAT_HD float uniform24(uint32_t w) { return (float)(w >> 8) * 0x1p-24f; }

// floor(t) + 1 with probability frac(t): exact (frac is exact, compared on U's grid)
QAT_HD float sr_floor(float t, float U) {
  const float f = floorf(t);
  return sub(t, f) > U ? add(f, 1.0f) : f;
}
QAT_HD float sr_fixed(float x, float one, float lo, float hi, float U) {  // SR(x * one), clamped
  return fminf(fmaxf(sr_floor(mul(x, one), U), lo), hi);
}

// one finite E5M2 byte -> exact FP32: the device FP8 decode, the raw-bits form on the host
QAT_HD float dec_e5m2(uint32_t b) {
#ifdef __CUDA_ARCH__
  float v, d1, d2, d3;
  dec_fp8x4<true>(b & 0xFFu, v, d1, d2, d3);
  return v;
#else
  return mul(flt(((b & 0x80u) << 24) | ((b & 0x7Fu) << 21)), 0x1p112f);
#endif
}

// SR of a signed value into E4M3, saturating at 448: E[decoded] = x for |x| < 448
QAT_HD uint32_t sr_e4m3(float x, float U) {
  const float a = fabsf(x);
  const uint32_t s = x < 0.0f ? 0x80u : 0u;
  if (!(a < 448.0f)) return s | 0x7Eu;
  if (a >= 0x1p-6f) {  // normal bracket: a's bits truncated to 3 mantissa bits, the upper one 2^20 away
    const uint32_t lb = bits(a) & 0xFFF00000u;
    const float lo = flt(lb), hi = flt(lb + 0x100000u);
    return s | ((lb >> 20) - 960u + (mul(U, sub(hi, lo)) < sub(a, lo) ? 1u : 0u));  // byte: FP32 exponent - 120
  }
  const bool sub_ = a < 0x1p-6f;
  const float q = sub_ ? 0x1p-9f : pow2(expo(a) - 3), qi = sub_ ? 0x1p9f : pow2(3 - expo(a));
  return s | nvfp4a::f32_to_e4m3(mul(sr_floor(mul(a, qi), U), q));  // exact value: RNE encodes it as is
}

// SR into E5M2 between the neighbours lo <= sqrt(vt) <= hi, P(hi) chosen so E[decoded^2] = vt exactly.
// x ~ sqrt(vt) (e.g. vt * rsqrt(vt)) only seeds the bracket; it is corrected in the squared domain, so
// an approximate x never biases the result.
QAT_HD uint32_t sr_e5m2_sq(float x, float vt, float U) {
  if (!(x < 57344.0f)) return 123u;  // 57344
  if (x >= 0x1p-13f) {  // normal bracket: x's bits truncated to 2 mantissa bits, neighbours one 2^21 step away
    uint32_t lb = bits(x) & 0xFFE00000u;
    float lo = flt(lb), hi = flt(lb + 0x200000u);
    if (vt < mul(lo, lo)) { lb -= 0x200000u; hi = lo; lo = flt(lb); }
    else if (vt > mul(hi, hi)) { lb += 0x200000u; lo = hi; hi = flt(lb + 0x200000u); }
    const uint32_t c = (lb >> 21) - 448u;  // E5M2 byte: FP32 exponent - 112, 2 mantissa bits
    if (c >= 123u) return 123u;
    const float lo2 = mul(lo, lo), hi2 = mul(hi, hi);
    return mul(U, sub(hi2, lo2)) < sub(vt, lo2) ? c + 1 : c;
  }
  const bool sub_ = x < 0x1p-14f;
  const float q = sub_ ? 0x1p-16f : pow2(expo(x) - 2), qi = sub_ ? 0x1p16f : pow2(2 - expo(x));
  uint32_t c = enc_e5m2(mul(floorf(mul(x, qi)), q));  // lower neighbour of x (exact: RNE keeps it)
  if (c > 0 && vt < mul(dec_e5m2(c), dec_e5m2(c))) --c;      // x was above sqrt(vt)'s bracket
  if (c < 123 && vt > mul(dec_e5m2(c + 1), dec_e5m2(c + 1))) ++c;  // ... or below it
  if (c >= 123) return 123u;
  const float lo = dec_e5m2(c), hi = dec_e5m2(c + 1);
  const float lo2 = mul(lo, lo), hi2 = mul(hi, hi);
  return mul(U, sub(hi2, lo2)) < sub(vt, lo2) ? c + 1 : c;
}

// SR into BF16: the magnitude is rounded up with probability (low 16 bits) / 2^16
QAT_HD float dec_bf16(uint32_t b) { return flt(b << 16); }
QAT_HD uint32_t sr_bf16(float x, float U) {
  const uint32_t b = bits(x);
  return (b >> 16) + ((b & 0xFFFFu) > (uint32_t)mul(U, 65536.0f) ? 1u : 0u);
}

// ---- Quake rsqrt: magic seed + one Newton step with free coefficients -------------------------------
QAT_HD float rsqrt_quake(float x) {
  const float y = flt(0x5F1FFFF9u - (bits(x) >> 1));
  return mul(0.703952253f, mul(y, fmar(-x, mul(y, y), 2.38924456f)));
}

// ---- E2M1 secant map g (grid coordinate u -> value / sigma), its inverse and slope ---------------
QAT_HD float g_of(float u) {
  const float a = fabsf(u);
  const float y = a < 4.0f ? mul(0.5f, a) : a < 6.0f ? sub(a, 2.0f) : fmar(2.0f, sub(a, 6.0f), 4.0f);
  return copysignf(y, u);
}
QAT_HD float g_inv(float y) {
  const float a = fabsf(y);
  const float u = a < 2.0f ? mul(2.0f, a) : a < 4.0f ? add(a, 2.0f) : fmar(0.5f, sub(a, 4.0f), 6.0f);
  return copysignf(u, y);
}
// nearest E2M1 magnitude of a >= 0 as a value, saturating at 6: nvfp4a::f32_to_e2m1_sel's arithmetic
// (grid 0.5 / 1 / 2 below 2 / 4 / 6, RNE by one FP32 addition of 2^23) without the code round trip
QAT_HD float e2m1_round(float a) {
  const float k = a < 2.0f ? 2.0f : a < 4.0f ? 1.0f : 0.5f, ik = a < 2.0f ? 0.5f : a < 4.0f ? 1.0f : 2.0f;
  return fminf(mul(sub(fmar(a, k, 8388608.0f), 8388608.0f), ik), 6.0f);
}
QAT_HD float g_slope(float u) {
  const float a = fabsf(u);
  return a < 4.0f ? 0.5f : a < 6.0f ? 1.0f : 2.0f;
}

}  // namespace qat
