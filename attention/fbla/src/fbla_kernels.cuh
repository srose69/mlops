// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// FBLA flash decoding, sm_61+: three launches.
//
//   coarse_kernel  one flash-decoding pass over the coarse records (K and V of a sub-block, read once).
//                  Warp w of a (batch, KV head) owns a contiguous, balanced run of sub-blocks; per group it
//                  quantizes q' = q * step_K to int8, per sub-block QK^T is a DP4A chain over the 64 channels
//                  of the lane's (token, half) and one xor-16 shuffle, P is quantized to u8 against the
//                  sub-block maximum and broadcast through shared memory, P V is a DP4A chain over the lane's
//                  four channels x 16 tokens.  It writes every closed sub-block's log2-mass and one (m, l, o)
//                  per CTA.  No barrier, atomic or address multiply in the loop: one pointer per lane.
//   select_kernel  R = log2 of the row's coarse mass from the partials, and the list of the sub-blocks with
//                  W_j = 2^(lse_j - R) > tau for some query of the KV head, in sub-block order per slice.
//   refine_kernel  the list's items, strided over all warps of the (batch, KV head): the correction
//                  exact - coarse of K and V, weighted 2^(m_block - R); the last CTA writes the BF16 output.
//
// Codes are never shifted into place where a mask suffices: a masked slot is the code times 2^n, and n is
// fixed per K accumulator or per V channel, so the factor 2^-n is folded into the FP32 epilogue.  Integer
// sums start at the bits of 1.5 * 2^23 (NVFP4EffMA's DP4A epilogue): every sum stays within +-2^22, so the
// accumulator read as FP32 minus 1.5 * 2^23 is the exact sum without an I2F.
#ifndef MLOPS_FBLA_KERNELS_CUH
#define MLOPS_FBLA_KERNELS_CUH

#include "common.cuh"
#include <algorithm>
#include <cmath>

namespace mlops_fbla_detail {

constexpr int D = 128, G = 128, S = 16, SBG = G / S, W = 8;
constexpr int SLICE = 1024, MAX_SLICES = 4096;  // select pass: sub-blocks per slice; tokens <= 2^26
constexpr uint32_t FULL = 0xffffffffu;
constexpr uint32_t MAGIC = 0x4B400000u;         // FP32 bits of 1.5 * 2^23
constexpr float MAGICF = 12582912.f;

__device__ __forceinline__ float ex2(float x) {  // MUFU.EX2: 2^-inf = 0
  float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
__device__ __forceinline__ float lg2(float x) {
  float y; asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r;
}
__device__ __forceinline__ uint32_t dp4a_su(uint32_t a, uint32_t b, uint32_t c) {  // s8 x u8
  uint32_t d; asm("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
}
__device__ __forceinline__ uint32_t dp4a_uu(uint32_t a, uint32_t b, uint32_t c) {  // u8 x u8
  uint32_t d; asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
}
__device__ __forceinline__ float isum(uint32_t acc) { return __uint_as_float(acc) - MAGICF; }
__device__ __forceinline__ float max16(float x) {  // lanes l and l ^ 16 hold the same token
#pragma unroll
  for (int o = 8; o; o >>= 1) x = fmaxf(x, __shfl_xor_sync(FULL, x, o));
  return x;
}
__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
  for (int o = 16; o; o >>= 1) x = fmaxf(x, __shfl_xor_sync(FULL, x, o));
  return x;
}
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(FULL, x, o);
  return x;
}
__device__ __forceinline__ void bf16x4(uint32_t a, uint32_t b, float x[4]) {
  x[0] = __uint_as_float(a << 16); x[1] = __uint_as_float(a & 0xffff0000u);
  x[2] = __uint_as_float(b << 16); x[3] = __uint_as_float(b & 0xffff0000u);
}
// Re-read special registers: volatile, so a value derived from them is recomputed where it is used (once
// per group) instead of being kept live across the whole loop (and spilled).
__device__ __forceinline__ int tid_x() { int t; asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t)); return t; }
__device__ __forceinline__ int ctaid_y() { int t; asm volatile("mov.u32 %0, %%ctaid.y;" : "=r"(t)); return t; }
__device__ __forceinline__ int ctaid_x() { int t; asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(t)); return t; }
__device__ __forceinline__ uint4 lds128v(const void *p) {  // volatile: ordered with the record loads
  uint4 v;
  asm volatile("ld.volatile.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"((uint32_t)__cvta_generic_to_shared(p)));
  return v;
}
__device__ __forceinline__ uint4 ld128(const uint8_t *p) { return __ldg(reinterpret_cast<const uint4 *>(p)); }
// Streamed record loads, volatile: ptxas keeps them in program order with the other volatile memory
// operations (the P bytes in shared memory), so the next record's load is issued where it is written,
// right after the registers it overwrites are read, instead of being sunk to the end of the loop.
__device__ __forceinline__ uint4 ldv128(const uint8_t *p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint2 ldv64(const uint8_t *p) {
  uint2 v;
  asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}

// ---- records (mlops_fbla.h) ---------------------------------------------------------------------
// A lane's 2 * BITS words of K or of V, every load a fixed offset from a per-lane base and every warp-wide
// load one contiguous 256 or 512 bytes.  V: p = record + 16 lane (128-bit words) and p8 = record + 8 lane
// (the 3-bit low words).  K by token (coarse pass): p = record + 16 t serves the high words of both halves
// (+ 0, + 256) and the 3-bit low words of both (+ 1024, one 128-bit load).
template <int BITS> struct Codes { uint32_t w[2 * BITS]; };
template <int BITS> __host__ __device__ constexpr int record_bytes() { return 2 * BITS * 256; }
struct Rec { const uint8_t *p, *p8; };
__device__ __forceinline__ Rec rec_at(const uint8_t *record, int lane) { return {record + 16 * lane, record + 8 * lane}; }
__device__ __forceinline__ Rec operator+(Rec r, size_t d) { return {r.p + d, r.p8 + d}; }

// K of lane (token t, half h) for the (token, half) mapping: base = record + 16 lane
template <int BITS> __device__ __forceinline__ void load_k(Codes<BITS> &c, const uint8_t *p, int lane) {
  const uint4 a = ldv128(p);
  c.w[0] = a.x; c.w[1] = a.y; c.w[2] = a.z; c.w[3] = a.w;
  if (BITS == 4) { const uint4 b = ldv128(p + 512); c.w[4] = b.x; c.w[5] = b.y; c.w[6] = b.z; c.w[7] = b.w; }
  if (BITS == 3) {  // low words of (t, h) at 1024 + 16 t + 8 h = p + 1024 - 8 h - ... relative to 16 lane
    const uint2 b = ldv64(p - 16 * lane + 1024 + 16 * (lane & 15) + 8 * (lane >> 4));
    c.w[4] = b.x; c.w[5] = b.y;
  }
}
// K of token t, both halves (coarse pass): p = record + 16 t
template <int BITS> __device__ __forceinline__ void load_k_token(Codes<BITS> (&k)[2], const uint8_t *p) {
  const uint4 a0 = ldv128(p), a1 = ldv128(p + 256);
  k[0].w[0] = a0.x; k[0].w[1] = a0.y; k[0].w[2] = a0.z; k[0].w[3] = a0.w;
  k[1].w[0] = a1.x; k[1].w[1] = a1.y; k[1].w[2] = a1.z; k[1].w[3] = a1.w;
  if (BITS == 4) {
    const uint4 b0 = ldv128(p + 512), b1 = ldv128(p + 768);
    k[0].w[4] = b0.x; k[0].w[5] = b0.y; k[0].w[6] = b0.z; k[0].w[7] = b0.w;
    k[1].w[4] = b1.x; k[1].w[5] = b1.y; k[1].w[6] = b1.z; k[1].w[7] = b1.w;
  }
  if (BITS == 3) {
    const uint4 b = ldv128(p + 1024);
    k[0].w[4] = b.x; k[0].w[5] = b.y; k[1].w[4] = b.z; k[1].w[5] = b.w;
  }
}
template <int BITS> __device__ __forceinline__ void load_v(Codes<BITS> &c, Rec r) {
  const uint4 a = ldv128(r.p + (BITS == 4 ? 1024 : 512));
  c.w[0] = a.x; c.w[1] = a.y; c.w[2] = a.z; c.w[3] = a.w;
  if (BITS == 4) { const uint4 b = ldv128(r.p + 1536); c.w[4] = b.x; c.w[5] = b.y; c.w[6] = b.z; c.w[7] = b.w; }
  if (BITS == 3) { const uint2 b = ldv64(r.p8 + 1280); c.w[4] = b.x; c.w[5] = b.y; }
}

// 3-bit slot 12 + r: two bits from W[r], one from W[4 + r / 2] (the only slots that need shifts)
__device__ __forceinline__ uint32_t split3(const Codes<3> &c, int r) {
  return ((c.w[r] >> 6) & 0x03030303u) | ((c.w[4 + (r >> 1)] >> (4 + (r & 1))) & 0x04040404u);
}

// K slot s: the four codes, times 2^n with n = 4 (4-bit, s >= 8), 3 (3-bit, 6 <= s < 12), else 0.
template <int BITS> __host__ __device__ constexpr bool k_scaled(int s) {
  return BITS == 4 ? s >= 8 : BITS == 3 ? (s >= 6 && s < 12) : false;
}
template <int BITS> __device__ __forceinline__ uint32_t k_slot(const Codes<BITS> &c, int s) {
  if constexpr (BITS == 4) return s < 8 ? c.w[s] & 0x0f0f0f0fu : c.w[s - 8] & 0xf0f0f0f0u;
  else if constexpr (BITS == 2) return (c.w[s & 3] >> (2 * (s >> 2))) & 0x03030303u;
  else return s < 6 ? c.w[s] & 0x07070707u : s < 12 ? c.w[s - 6] & 0x38383838u : split3(c, s - 12);
}
template <int BITS> __host__ __device__ constexpr float k_unscale() { return BITS == 4 ? 1.f / 16 : 1.f / 8; }

// V slot s: (channel, token quad) and the four codes, times 2^n with n fixed per channel (v_unscale).
template <int BITS> __host__ __device__ constexpr int v_channel(int s) {
  return BITS == 4 ? (s < 8 ? s % 2 : 2 + s % 2)
       : BITS == 2 ? s >> 2
       : s < 4 ? 0 : s < 6 ? 2 : s < 10 ? 1 : s < 12 ? 2 : 3;
}
template <int BITS> __host__ __device__ constexpr int v_quad(int s) {
  return BITS == 4 ? (s < 8 ? s / 2 : (s - 8) / 2)
       : BITS == 2 ? s & 3
       : s < 4 ? s : s < 6 ? s - 4 : s < 10 ? s - 6 : s < 12 ? s - 8 : s - 12;
}
template <int BITS> __device__ __forceinline__ uint32_t v_slot(const Codes<BITS> &c, int s) {
  if constexpr (BITS == 4) return s < 8 ? c.w[s] & 0x0f0f0f0fu : c.w[s - 8] & 0xf0f0f0f0u;
  else if constexpr (BITS == 2) return c.w[s & 3] & (0x03030303u << (2 * (s >> 2)));
  else return s < 6 ? c.w[s] & 0x07070707u : s < 10 ? c.w[s - 6] & 0x38383838u
            : s < 12 ? (c.w[s - 6] >> 3) & 0x07070707u : split3(c, s - 12);
}
template <int BITS> __host__ __device__ constexpr float v_unscale(int ch) {
  return BITS == 4 ? (ch < 2 ? 1.f : 1.f / 16) : BITS == 2 ? 1.f / float(1 << (2 * ch)) : ch == 1 ? 1.f / 8 : 1.f;
}

// ---- products ---------------------------------------------------------------------------------
// q-hat . codes over the lane's 64 channels, both halves added; q-hat words from the warp's stage
// ([QG][32] in shared memory: the 16 lanes of a half read the same 16 bytes, a broadcast).
template <int BITS, int QG>
__device__ __forceinline__ void qk(const uint32_t (*qs)[32], int half, const Codes<BITS> &c, float x[QG]) {
  uint32_t lo[QG], hi[QG];
#pragma unroll
  for (int qi = 0; qi < QG; ++qi) lo[qi] = hi[qi] = MAGIC;
#pragma unroll
  for (int v = 0; v < 4; ++v) {
    uint4 q[QG];
#pragma unroll
    for (int qi = 0; qi < QG; ++qi) q[qi] = reinterpret_cast<const uint4 *>(&qs[qi][16 * half])[v];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int s = 4 * v + e;
      const uint32_t cw = k_slot<BITS>(c, s);
#pragma unroll
      for (int qi = 0; qi < QG; ++qi) {
        const uint32_t qw = e == 0 ? q[qi].x : e == 1 ? q[qi].y : e == 2 ? q[qi].z : q[qi].w;
        if (k_scaled<BITS>(s)) hi[qi] = dp4a_su(qw, cw, hi[qi]);
        else lo[qi] = dp4a_su(qw, cw, lo[qi]);
      }
    }
  }
#pragma unroll
  for (int qi = 0; qi < QG; ++qi) {
    x[qi] = BITS == 2 ? isum(lo[qi]) : fmaf(isum(hi[qi]), k_unscale<BITS>(), isum(lo[qi]));
    x[qi] += __shfl_xor_sync(FULL, x[qi], 16);
  }
}

// q-hat . codes over all 128 channels of the lane's token: halves k[0], k[1] (no cross-lane sum).
template <int BITS, int QG>
__device__ __forceinline__ void qk_token(const uint32_t (*qs)[32], const Codes<BITS> (&k)[2], float x[QG]) {
  uint32_t lo[QG], hi[QG];
#pragma unroll
  for (int qi = 0; qi < QG; ++qi) lo[qi] = hi[qi] = MAGIC;
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      uint4 q[QG];
#pragma unroll
      for (int qi = 0; qi < QG; ++qi) q[qi] = lds128v(&qs[qi][16 * h + 4 * v]);  // broadcast
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int s = 4 * v + e;
        const uint32_t cw = k_slot<BITS>(k[h], s);
#pragma unroll
        for (int qi = 0; qi < QG; ++qi) {
          const uint32_t qw = e == 0 ? q[qi].x : e == 1 ? q[qi].y : e == 2 ? q[qi].z : q[qi].w;
          if (k_scaled<BITS>(s)) hi[qi] = dp4a_su(qw, cw, hi[qi]);
          else lo[qi] = dp4a_su(qw, cw, lo[qi]);
        }
      }
    }
#pragma unroll
  for (int qi = 0; qi < QG; ++qi) x[qi] = BITS == 2 ? isum(lo[qi]) : fmaf(isum(hi[qi]), k_unscale<BITS>(), isum(lo[qi]));
}

// sum_t p_t code_t of the lane's four channels (times the channel's 2^n), for every query.
template <int BITS, int QG>
__device__ __forceinline__ void pv(const uint32_t (*P)[4], const Codes<BITS> &c, float y[QG][4]) {
  uint32_t acc[QG][4];
#pragma unroll
  for (int qi = 0; qi < QG; ++qi)
#pragma unroll
    for (int ch = 0; ch < 4; ++ch) acc[qi][ch] = MAGIC;
#pragma unroll
  for (int s = 0; s < 16; ++s) {
    const uint32_t cw = v_slot<BITS>(c, s);
#pragma unroll
    for (int qi = 0; qi < QG; ++qi)
      acc[qi][v_channel<BITS>(s)] = dp4a_uu(P[qi][v_quad<BITS>(s)], cw, acc[qi][v_channel<BITS>(s)]);
  }
#pragma unroll
  for (int qi = 0; qi < QG; ++qi)
#pragma unroll
    for (int ch = 0; ch < 4; ++ch) y[qi][ch] = isum(acc[qi][ch]);
}

// ---- softmax of a 16-token block --------------------------------------------------------------
// p = round(255 * 2^(s - mb)) of the lane's token (lane % 16) into the block's 16 bytes in shared memory;
// every lane then reads all 16 (one broadcast LDS.128 instead of six shuffles).
__device__ __forceinline__ void put_p(uint32_t *pb, float s, float mb, int lane) {
  const uint32_t bits = __float_as_uint(fmaf(ex2(s - mb), 255.f, MAGICF));  // low byte = round(255 p)
  asm volatile("st.volatile.shared.u8 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(pb) + (lane & 15)), "r"(bits) : "memory");
}
__device__ __forceinline__ void put_p_byte(uint32_t *pb, float s, float mb, int byte) {
  const uint32_t bits = __float_as_uint(fmaf(ex2(s - mb), 255.f, MAGICF));
  asm volatile("st.volatile.shared.u8 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(pb) + byte), "r"(bits) : "memory");
}
__device__ __forceinline__ float get_p(const uint32_t *pb, uint32_t P[4]) {
  asm volatile("ld.volatile.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(P[0]), "=r"(P[1]), "=r"(P[2]), "=r"(P[3])
               : "r"((uint32_t)__cvta_generic_to_shared(pb)) : "memory");
  uint32_t acc = MAGIC;
#pragma unroll
  for (int j = 0; j < 4; ++j) acc = dp4a_uu(P[j], 0x01010101u, acc);
  return isum(acc);  // sum p, exact
}
// FA2 online update with one 2^x: the rescale of the running sums (alpha) or the block weight (w) is 1.
__device__ __forceinline__ void online(float mb, float &m, float &alpha, float &w) {
  const float d = mb - m, e = ex2(-fabsf(d));
  const bool up = d > 0.f;
  alpha = up ? e : 1.f; w = up ? 1.f : e;
  m = fmaxf(m, mb);
}

// ---- per-group query quantization -------------------------------------------------------------
// The lane's channels 4 l .. + 3: z = q * scale * log2 e * step_K, q-hat = rint(z * 127 / max|z|) with
// sq = max|z| / 127 (both IEEE divisions, once per group).  The coarse code c of a slot decodes to
// base + step (2^FB c + hoff): s = c0 + sq 2^FB (q-hat . c), c0 = q . base + sq hoff sum q-hat.
// The group's state lives in the warp's shared slots, not in registers: q-hat words (stage), per query
// (c0, sq 2^FB), per lane the V affine (vb, vs) with 2^FB and the channel's 2^-n folded in.
template <int QG> struct WarpSlots {
  alignas(16) uint32_t stage[QG][32];  // q-hat words of the group
  alignas(16) float4 vb[32], vs[32];   // the lanes' V affine of the group (coarse code -> value)
  alignas(16) uint32_t pb[QG][8];      // P bytes of the sub-block pair (16 + 16)
  alignas(16) float2 ls[QG * 8];       // the group's (m_block, sum p) per query and sub-block
};
template <int QG> struct GroupState {
  float c0[QG], sqc[QG];  // coarse score = c0 + sqc * x
  float sq[QG], qb[QG];   // full score = qb + sq * (q-hat . code) (refinement pass)
};

// km, vm: the group's K and V metadata of the lane; qs: q * scale * log2 e.  r = 127 / max|z| correctly rounded
// from MUFU.RCP (no IEEE-division slow path); rint(z r) is the low byte of the bits of z r + 1.5 * 2^23 (no F2I).  The coarse
// and the refinement pass both run this on the same inputs, so their q-hat agree bit for bit.
template <int CB, int FB, int QG>
__device__ __forceinline__ void setup_group(int lane, uint4 km, uint4 vm, const float (*qs)[D], uint32_t (*stage)[32],
                                            GroupState<QG> &gs, float4 *vbs = nullptr, float4 *vss = nullptr) {
  constexpr float HOFF = .5f * ((1 << FB) - 1);
  float kb[4], ks[4];
  bf16x4(km.x, km.y, kb); bf16x4(km.z, km.w, ks);
#pragma unroll
  for (int qi = 0; qi < QG; ++qi) {
    const float4 q4 = reinterpret_cast<const float4 *>(qs[qi])[lane];
    const float qf[4] = {q4.x, q4.y, q4.z, q4.w};
    float z[4], amax = 0.f, qb = 0.f;
#pragma unroll
    for (int x = 0; x < 4; ++x) { z[x] = qf[x] * ks[x]; amax = fmaxf(amax, fabsf(z[x])); qb = fmaf(qf[x], kb[x], qb); }
    amax = warp_max(amax);
    // r = fl(127 / amax), correctly rounded (Markstein: MUFU.RCP, one reciprocal refinement, quotient and its
    // residual correction; exact for normal amax), without IEEE division's slow-path check: the q-hat codes
    // are those of the definition
    float rc;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rc) : "f"(amax));
    rc = fmaf(fmaf(-amax, rc, 1.f), rc, rc);
    const float q0 = 127.f * rc;
    const float sq = amax * (1.f / 127.f), r = amax > 0.f ? fmaf(fmaf(-amax, q0, 127.f), rc, q0) : 0.f;
    uint32_t bq[4];
    float qsum = 0.f;
#pragma unroll
    for (int x = 0; x < 4; ++x) {
      const float t = __fadd_rn(__fmul_rn(z[x], r), MAGICF);  // rint(fl(z r)): no FMA contraction; low byte = code
      bq[x] = __float_as_uint(t);
      qsum += t - MAGICF;
    }
    const uint32_t wq = prmt(prmt(bq[0], bq[1], 0x0040u), prmt(bq[2], bq[3], 0x0040u), 0x5410u);
    stage[qi][lane] = wq;
    qb = warp_sum(qb);
    gs.c0[qi] = fmaf(sq * HOFF, warp_sum(qsum), qb); gs.sqc[qi] = sq * (1 << FB);
    gs.sq[qi] = sq; gs.qb[qi] = qb;
  }
  float vb[4], vs[4];
  bf16x4(vm.x, vm.y, vb); bf16x4(vm.z, vm.w, vs);
  if (vbs) {  // coarse V value = vb * p + vs * y (vs with 2^FB and the channel's 2^-n folded in)
    vbs[lane] = make_float4(fmaf(vs[0], HOFF, vb[0]), fmaf(vs[1], HOFF, vb[1]), fmaf(vs[2], HOFF, vb[2]), fmaf(vs[3], HOFF, vb[3]));
    vss[lane] = make_float4(vs[0] * ((1 << FB) * v_unscale<CB>(0)), vs[1] * ((1 << FB) * v_unscale<CB>(1)),
                            vs[2] * ((1 << FB) * v_unscale<CB>(2)), vs[3] * ((1 << FB) * v_unscale<CB>(3)));
  }
  __syncwarp();  // the slots are read by every lane
}

// ---- BF16 groups (sink, open) -----------------------------------------------------------------
template <int QG>
__device__ __forceinline__ void exact_scores(const uint16_t *k, int t0, int count, const float (*qs)[D], int lane,
                                             float s[QG]) {
  const int t = lane & 15, h = lane >> 4, tok = t0 + t;
  const uint4 *kp = reinterpret_cast<const uint4 *>(k + (size_t)tok * D + 64 * h);
  float x[QG];
#pragma unroll
  for (int qi = 0; qi < QG; ++qi) x[qi] = 0.f;
#pragma unroll
  for (int m = 0; m < 8; ++m) {
    const uint4 w = __ldg(kp + m);
    float kv[8];
    bf16x4(w.x, w.y, kv); bf16x4(w.z, w.w, kv + 4);
#pragma unroll
    for (int qi = 0; qi < QG; ++qi)
#pragma unroll
      for (int e = 0; e < 8; ++e) x[qi] = fmaf(qs[qi][64 * h + 8 * m + e], kv[e], x[qi]);
  }
#pragma unroll
  for (int qi = 0; qi < QG; ++qi) {
    x[qi] += __shfl_xor_sync(FULL, x[qi], 16);
    s[qi] = tok < count ? x[qi] : -INFINITY;
  }
}

template <int QG>
__device__ __forceinline__ void exact_block(const uint16_t *ek, const uint16_t *ev, int j, int count,
                                            const float (*qs)[D], uint32_t (*pb)[8], int lane, float m[QG],
                                            float l[QG], float o[QG][4]) {
  float s[QG], al[QG], w[QG];
  exact_scores<QG>(ek, j * S, count, qs, lane, s);
#pragma unroll
  for (int qi = 0; qi < QG; ++qi) {
    const float mb = max16(s[qi]);
    online(mb, m[qi], al[qi], w[qi]);
    put_p(pb[qi], s[qi], mb, lane);
  }
  __syncwarp();
#pragma unroll
  for (int qi = 0; qi < QG; ++qi) {
    uint32_t P[4];
    const float sp = get_p(pb[qi], P);
    float y[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int t = 0; t < S; ++t) {
      float vv[4];
      const uint2 x = j * S + t < count ? __ldg(reinterpret_cast<const uint2 *>(ev + (size_t)(j * S + t) * D) + lane)
                                        : make_uint2(0u, 0u);
      bf16x4(x.x, x.y, vv);
      const float p = __uint_as_float(MAGIC | ((P[t >> 2] >> (8 * (t & 3))) & 255u)) - MAGICF;
#pragma unroll
      for (int c = 0; c < 4; ++c) y[c] = fmaf(p, vv[c], y[c]);
    }
    l[qi] = fmaf(al[qi], l[qi], w[qi] * sp);
#pragma unroll
    for (int c = 0; c < 4; ++c) o[qi][c] = fmaf(al[qi], o[qi][c], w[qi] * y[c]);
  }
  __syncwarp();
}

// ---- coarse pass --------------------------------------------------------------------------------
// Units of a (batch, KV head): first the BF16 sub-blocks (sink group, then open group), then the closed
// groups 1 .. groups - 2.  Warp w of the head's P W warps takes the units w, w + P W, ...: at any moment all
// warps read one window of the cache (DRAM rows and TLB stay hot), and no warp has more than one unit more
// than another.
__device__ __forceinline__ int units_of(const Args &a) {  // BF16 sub-blocks + closed groups of a head
  const int groups = a.groups;
  const int na = groups == 1 ? (a.tokens + S - 1) / S : SBG;
  const int no = groups >= 2 ? (a.tokens - G * (groups - 1) + S - 1) / S : 0;
  return na + no + (groups >= 3 ? groups - 2 : 0);
}

template <int QG> struct SmemCoarse {
  WarpSlots<QG> ws[W];
  union {
    alignas(16) float qs[W][QG][D];  // q * scale * log2 e, per warp
    struct { alignas(16) float o[W][QG][D]; float m[W][QG], l[W][QG]; } merge;  // same index: no overlap
  } u;
};

template <int QG> struct Run {  // a warp's running softmax
  float m[QG], l[QG], o[QG][4];
};

// Pair J of a group: sub-blocks A = 2J and B = 2J + 1.  K: lane l holds token l % 16 of A
// (l < 16) or of B, all 128 channels (halves kc[0], kc[1]), so a token's score needs no cross-lane sum and the
// two blocks' maxima come from one 16-lane reduction.  V: lane l holds channels 4 l .. + 3 of A (va) and B (vb).
// The registers just read are refilled from the next pair (nk, nv) at once.  The pair's (m_block, sum p) of
// query qi goes to the warp's slots (lane 0), the group's lse = m + log2 sum p is stored once per group.

template <int CB, int QG>
__device__ __forceinline__ void coarse_pair(Codes<CB> (&kc)[2], Codes<CB> &va, Codes<CB> &vb, const uint8_t *nk, Rec nv,
                                            WarpSlots<QG> &ws, const GroupState<QG> &gs, int lane, int J, Run<QG> &r) {
  constexpr int RB = record_bytes<CB>();
  const bool upper = (tid_x() & 31) >= 16;  // this lane's token is in B (re-read: not kept live)
  float x[QG];
  qk_token<CB, QG>(ws.stage, kc, x);
  load_k_token<CB>(kc, nk);
  float mA[QG], mB[QG], al[QG], wA[QG], wB[QG];
#pragma unroll
  for (int qi = 0; qi < QG; ++qi) {
    const float s = fmaf(gs.sqc[qi], x[qi], gs.c0[qi]);
    const float mh = max16(s), mo = __shfl_xor_sync(FULL, mh, 16);
    mA[qi] = upper ? mo : mh; mB[qi] = upper ? mh : mo;
    const float mn = fmaxf(r.m[qi], fmaxf(mA[qi], mB[qi]));
    al[qi] = ex2(r.m[qi] - mn); wA[qi] = ex2(mA[qi] - mn); wB[qi] = ex2(mB[qi] - mn);
    r.m[qi] = mn;
    put_p_byte(ws.pb[qi], s, mh, lane);
  }
  __syncwarp();
  // P V by query pairs (QC at a time): the V codes stay, their decode repeats for QG = 4 instead of holding
  // P and y of all four queries
  constexpr int QC = QG < 2 ? QG : 2;
  const float4 vb4 = ws.vb[lane], vs4 = ws.vs[lane];
  const float gvb[4] = {vb4.x, vb4.y, vb4.z, vb4.w}, gvs[4] = {vs4.x, vs4.y, vs4.z, vs4.w};
#pragma unroll
  for (int q0 = 0; q0 < QG; q0 += QC) {
    uint32_t PA[QC][4], PB[QC][4];
    float sA[QC], sB[QC];
#pragma unroll
    for (int j = 0; j < QC; ++j) {
      const int qi = q0 + j;
      sA[j] = get_p(ws.pb[qi], PA[j]);
      sB[j] = get_p(ws.pb[qi] + 4, PB[j]);
      if (!lane) { ws.ls[8 * qi + 2 * J] = make_float2(mA[qi], sA[j]); ws.ls[8 * qi + 2 * J + 1] = make_float2(mB[qi], sB[j]); }
    }
    float yA[QC][4], yB[QC][4];
    pv<CB, QC>(PA, va, yA);
    pv<CB, QC>(PB, vb, yB);
#pragma unroll
    for (int j = 0; j < QC; ++j) {
      const int qi = q0 + j;
      const float wsp = fmaf(wA[qi], sA[j], wB[qi] * sB[j]);
      r.l[qi] = fmaf(al[qi], r.l[qi], wsp);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float t = fmaf(wA[qi], yA[j][c], wB[qi] * yB[j][c]);
        r.o[qi][c] = fmaf(al[qi], r.o[qi][c], fmaf(gvs[c], t, gvb[c] * wsp));
      }
    }
  }
  load_v<CB>(va, nv);
  load_v<CB>(vb, nv + RB);
  __syncwarp();  // pb is rewritten by the next pair
}

template <int CB, int FB, int QG>
__global__ void __launch_bounds__(32 * W, 2) coarse_kernel(const Args a) {
  static_assert(QG <= 2, "four queries per KV head: coarse_kernel4");
  __shared__ SmemCoarse<QG> sm;
  constexpr int RB = record_bytes<CB>();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int bh = blockIdx.y, b = bh / a.kv_heads, kh = bh - b * a.kv_heads;
  const int row0 = b * a.q_heads + kh * QG, cta = blockIdx.x, P = gridDim.x, nw = P * W;
  const int groups = a.groups;
  const int na = groups == 1 ? (a.tokens + S - 1) / S : SBG;          // sink sub-blocks
  const int cnt_open = groups >= 2 ? a.tokens - G * (groups - 1) : 0;  // open group's tokens
  const int ne = na + (cnt_open + S - 1) / S;                          // BF16 units
  const int units = ne + (groups >= 3 ? groups - 2 : 0);
  WarpSlots<QG> &ws = sm.ws[warp];
  const float (*qs)[D] = sm.u.qs[warp];

#pragma unroll
  for (int qi = 0; qi < QG; ++qi) {  // this warp's q, scaled, in shared memory
    const uint2 qw = __ldg(reinterpret_cast<const uint2 *>(a.q + (size_t)(row0 + qi) * D) + lane);
    float qf[4];
    bf16x4(qw.x, qw.y, qf);
    reinterpret_cast<float4 *>(sm.u.qs[warp][qi])[lane] =
        make_float4(qf[0] * a.scale_log2, qf[1] * a.scale_log2, qf[2] * a.scale_log2, qf[3] * a.scale_log2);
  }
  __syncwarp();

  Run<QG> r;
#pragma unroll
  for (int qi = 0; qi < QG; ++qi) {
    r.m[qi] = -INFINITY; r.l[qi] = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) r.o[qi][c] = 0.f;
  }
  int u = cta * W + warp;
  for (; u < ne; u += nw) {
    const bool sink = u < na;
    const size_t eo = (size_t)bh * G * D;
    exact_block<QG>((sink ? a.c.exact_sink_k : a.c.exact_open_k) + eo, (sink ? a.c.exact_sink_v : a.c.exact_open_v) + eo,
                    sink ? u : u - na, sink ? (groups == 1 ? a.tokens : G) : cnt_open, qs, ws.pb, lane, r.m, r.l, r.o);
  }

  if (u < units) {
    // closed groups g = 1 + u - ne, stepping by nw, as pairs of records: the cursors advance by 2 RB within a
    // group and by `step` from a group's last pair to the warp's next group; K and V of a pair have their own
    // lane bases (token-major, channel-major), every load an immediate offset from one of them
    int g = 1 + u - ne;
    const uint8_t *g0 = a.c.coarse + ((size_t)bh * a.cgroups + g) * SBG * RB;
    const uint8_t *pk = g0 + (lane >> 4) * RB + 16 * (lane & 15);
    Rec pv_ = rec_at(g0, lane);
    Codes<CB> kc[2], va, vb;
    load_k_token<CB>(kc, pk);
    load_v<CB>(va, pv_);
    load_v<CB>(vb, pv_ + RB);
    // metadata and lse addresses are derived from g once per group (nothing but the record cursors and the
    // running softmax is carried across groups)
    auto meta_at = [&](int gg) {
      return reinterpret_cast<const uint8_t *>(a.c.meta) + ((size_t)ctaid_y() * a.cgroups + gg) * 1024 + 16 * (tid_x() & 31);
    };
    uint4 km = ld128(meta_at(g)), vm = ld128(meta_at(g) + 512);
    GroupState<QG> gs;
    for (;;) {
      setup_group<CB, FB, QG>(lane, km, vm, qs, ws.stage, gs, ws.vb, ws.vs);
      u += nw;
      const bool more = u < units_of(a);
      // from the group's last pair: the warp's next group, or (after its last group) the following records,
      // which exist (at worst the open group's first two); the next group's metadata comes with them
      const size_t step = more ? (size_t)(nw * SBG - (SBG - 2)) * RB : 2 * RB;  // from pair 2 to the next pair 3's
      // pairs 0..2 as a loop: an unrolled group lets ptxas overlap neighbouring pairs, whose live state then
      // exceeds the register budget
#pragma unroll 1
      for (int J = 0; J < 3; ++J) {
        coarse_pair<CB, QG>(kc, va, vb, pk + 2 * RB, pv_ + 2 * RB, ws, gs, lane, J, r);
        pk += 2 * RB; pv_ = pv_ + 2 * RB;
      }
      pk += step; pv_ = pv_ + step;
      {
        const uint8_t *mn = meta_at(more ? g + nw : g);
        km = ld128(mn); vm = ld128(mn + 512);
      }
      coarse_pair<CB, QG>(kc, va, vb, pk, pv_, ws, gs, lane, 3, r);
      // lane 8 qi + j stores the group's lse of sub-block j, query qi: one 32-byte sector per query
      {
        const int l2 = tid_x() & 31, b2 = ctaid_y() / a.kv_heads, r2 = b2 * a.q_heads + (ctaid_y() - b2 * a.kv_heads) * QG;
        __syncwarp();
        if (l2 < 8 * QG) {
          const float2 e = ws.ls[l2];
          a.lse[(size_t)(r2 + (l2 >> 3)) * a.nblocks + (size_t)g * SBG + (l2 & 7)] = e.x + lg2(e.y);
        }
      }
      if (!more) break;
      g += nw;
    }
  }

  // the CTA's warps merged into one partial (m, l, o) per query
#pragma unroll
  for (int qi = 0; qi < QG; ++qi) {
    if (!lane) { sm.u.merge.m[warp][qi] = r.m[qi]; sm.u.merge.l[warp][qi] = r.l[qi]; }
    *reinterpret_cast<float4 *>(&sm.u.merge.o[warp][qi][4 * lane]) = make_float4(r.o[qi][0], r.o[qi][1], r.o[qi][2], r.o[qi][3]);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < QG * D; i += blockDim.x) {
    const int qi = i / D, ch = i - qi * D;
    float mx = -INFINITY;
#pragma unroll
    for (int w = 0; w < W; ++w) mx = fmaxf(mx, sm.u.merge.m[w][qi]);
    float ls = 0.f, os = 0.f;
#pragma unroll
    for (int w = 0; w < W; ++w) {
      const float e = sm.u.merge.m[w][qi] == -INFINITY ? 0.f : ex2(sm.u.merge.m[w][qi] - mx);
      ls = fmaf(e, sm.u.merge.l[w][qi], ls); os = fmaf(e, sm.u.merge.o[w][qi][ch], os);
    }
    const int by = ctaid_y(), bb = by / a.kv_heads, r0 = bb * a.q_heads + (by - bb * a.kv_heads) * QG;
    float *part = a.part + ((size_t)(r0 + qi) * gridDim.x + ctaid_x()) * (D + 2);
    part[2 + ch] = os;
    if (!ch) { part[0] = mx; part[1] = ls; }
  }
}

// ---- coarse pass, four queries per KV head -------------------------------------------------------
// One warp serves all four queries of its KV head (every record read once, every code decoded once), in two
// halves of two queries over the same code registers.  The state indexed by query -- running (m, l), o, the
// group's (c0, sq 2^FB) -- lives in the warp's shared slots, so the halves are a loop and nothing indexed by
// query sits in registers: two CTAs per SM without spills.
struct WarpSlots4 {
  alignas(16) uint32_t stage[4][32];  // q-hat words of the group
  alignas(16) float4 vb[32], vs[32];  // the lanes' V affine of the group
  alignas(16) uint32_t pb[4][8];      // P bytes of the pair, per query
  alignas(16) float2 ls[32];          // the group's (m_block, sum p), lane 8 qi + j
  alignas(16) float2 cs[4];           // (c0, sq 2^FB) per query
  alignas(16) float2 ml[4];           // running (m, l) per query
  alignas(16) float o[4][D];          // running o per query
};
struct SmemCoarse4 {
  WarpSlots4 ws[W];
  alignas(16) float qs[4][D];  // q * scale * log2 e (the CTA's KV head: the same for every warp)
};

template <int CB>
__device__ __forceinline__ void coarse_pair4(Codes<CB> (&kc)[2], Codes<CB> &va, Codes<CB> &vb, const uint8_t *nk, Rec nv,
                                             WarpSlots4 &ws, int lane, int J) {
  constexpr int RB = record_bytes<CB>();
  const bool upper = (tid_x() & 31) >= 16;  // this lane's token is in B
#pragma unroll 1
  for (int h = 0; h < 2; ++h) {
    const int q0 = 2 * h;
    float x[2];
    qk_token<CB, 2>(ws.stage + q0, kc, x);
    if (h) load_k_token<CB>(kc, nk);  // after the last half's QK: the next pair's K into the same registers
    float mA[2], mB[2], al[2], wA[2], wB[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float2 cs = ws.cs[q0 + j], ml = ws.ml[q0 + j];
      const float s = fmaf(cs.y, x[j], cs.x);
      const float mh = max16(s), mo = __shfl_xor_sync(FULL, mh, 16);
      mA[j] = upper ? mo : mh; mB[j] = upper ? mh : mo;
      const float mn = fmaxf(ml.x, fmaxf(mA[j], mB[j]));
      al[j] = ex2(ml.x - mn); wA[j] = ex2(mA[j] - mn); wB[j] = ex2(mB[j] - mn);
      put_p_byte(ws.pb[q0 + j], s, mh, lane);
    }
    __syncwarp();
    uint32_t PA[2][4], PB[2][4];
    float sA[2], sB[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      sA[j] = get_p(ws.pb[q0 + j], PA[j]);
      sB[j] = get_p(ws.pb[q0 + j] + 4, PB[j]);
      if (!lane) { ws.ls[8 * (q0 + j) + 2 * J] = make_float2(mA[j], sA[j]); ws.ls[8 * (q0 + j) + 2 * J + 1] = make_float2(mB[j], sB[j]); }
    }
    float yA[2][4], yB[2][4];
    pv<CB, 2>(PA, va, yA);
    pv<CB, 2>(PB, vb, yB);
    const float4 vb4 = ws.vb[lane], vs4 = ws.vs[lane];
    const float gvb[4] = {vb4.x, vb4.y, vb4.z, vb4.w}, gvs[4] = {vs4.x, vs4.y, vs4.z, vs4.w};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int qi = q0 + j;
      const float2 ml = ws.ml[qi];
      const float wsp = fmaf(wA[j], sA[j], wB[j] * sB[j]);
      float4 o = reinterpret_cast<float4 *>(ws.o[qi])[lane];
      float ov[4] = {o.x, o.y, o.z, o.w};
#pragma unroll
      for (int c = 0; c < 4; ++c) ov[c] = fmaf(al[j], ov[c], fmaf(gvs[c], fmaf(wA[j], yA[j][c], wB[j] * yB[j][c]), gvb[c] * wsp));
      reinterpret_cast<float4 *>(ws.o[qi])[lane] = make_float4(ov[0], ov[1], ov[2], ov[3]);
      __syncwarp();  // every lane has read ml before lane 0 rewrites it
      if (!lane) ws.ml[qi] = make_float2(fmaxf(ml.x, fmaxf(mA[j], mB[j])), fmaf(al[j], ml.y, wsp));
    }
    __syncwarp();  // pb, ml and o are read again by the next half or pair
  }
  load_v<CB>(va, nv);
  load_v<CB>(vb, nv + RB);
}

template <int CB, int FB>
__global__ void __launch_bounds__(32 * W, 2) coarse_kernel4(const Args a) {
  __shared__ SmemCoarse4 sm;
  constexpr int QG = 4, RB = record_bytes<CB>();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int bh = blockIdx.y, b = bh / a.kv_heads, kh = bh - b * a.kv_heads;
  const int row0 = b * a.q_heads + kh * QG, cta = blockIdx.x, P = gridDim.x, nw = P * W;
  const int groups = a.groups;
  const int na = groups == 1 ? (a.tokens + S - 1) / S : SBG;
  const int cnt_open = groups >= 2 ? a.tokens - G * (groups - 1) : 0;
  const int ne = na + (cnt_open + S - 1) / S, units = units_of(a);
  WarpSlots4 &ws = sm.ws[warp];
  if (warp < QG) {  // the CTA's q, scaled
    const uint2 qw = __ldg(reinterpret_cast<const uint2 *>(a.q + (size_t)(row0 + warp) * D) + lane);
    float qf[4];
    bf16x4(qw.x, qw.y, qf);
    reinterpret_cast<float4 *>(sm.qs[warp])[lane] =
        make_float4(qf[0] * a.scale_log2, qf[1] * a.scale_log2, qf[2] * a.scale_log2, qf[3] * a.scale_log2);
  }
  __syncthreads();

  {  // BF16 sub-blocks with the state in registers, then into the slots
    Run<QG> r;
#pragma unroll
    for (int qi = 0; qi < QG; ++qi) {
      r.m[qi] = -INFINITY; r.l[qi] = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c) r.o[qi][c] = 0.f;
    }
    for (int u = cta * W + warp; u < ne; u += nw) {
      const bool sink = u < na;
      const size_t eo = (size_t)bh * G * D;
      exact_block<QG>((sink ? a.c.exact_sink_k : a.c.exact_open_k) + eo, (sink ? a.c.exact_sink_v : a.c.exact_open_v) + eo,
                      sink ? u : u - na, sink ? (groups == 1 ? a.tokens : G) : cnt_open, sm.qs, ws.pb, lane, r.m, r.l, r.o);
    }
#pragma unroll
    for (int qi = 0; qi < QG; ++qi) {
      if (!lane) ws.ml[qi] = make_float2(r.m[qi], r.l[qi]);
      reinterpret_cast<float4 *>(ws.o[qi])[lane] = make_float4(r.o[qi][0], r.o[qi][1], r.o[qi][2], r.o[qi][3]);
    }
    __syncwarp();
  }

  int u = cta * W + warp;
  while (u < ne) u += nw;
  if (u < units) {
    int g = 1 + u - ne;
    const uint8_t *g0 = a.c.coarse + ((size_t)bh * a.cgroups + g) * SBG * RB;
    const uint8_t *pk = g0 + (lane >> 4) * RB + 16 * (lane & 15);
    Rec pv_ = rec_at(g0, lane);
    Codes<CB> kc[2], va, vb;
    load_k_token<CB>(kc, pk);
    load_v<CB>(va, pv_);
    load_v<CB>(vb, pv_ + RB);
    auto meta_at = [&](int gg) {
      return reinterpret_cast<const uint8_t *>(a.c.meta) + ((size_t)ctaid_y() * a.cgroups + gg) * 1024 + 16 * (tid_x() & 31);
    };
    uint4 km = ld128(meta_at(g)), vm = ld128(meta_at(g) + 512);
    for (;;) {
      {
        GroupState<QG> gs;
        setup_group<CB, FB, QG>(lane, km, vm, sm.qs, ws.stage, gs, ws.vb, ws.vs);
        if (!lane)
#pragma unroll
          for (int qi = 0; qi < QG; ++qi) ws.cs[qi] = make_float2(gs.c0[qi], gs.sqc[qi]);
        __syncwarp();
      }
      u += nw;
      const bool more = u < units_of(a);
      const size_t step = more ? (size_t)(nw * SBG - (SBG - 2)) * RB : 2 * RB;
#pragma unroll 1
      for (int J = 0; J < 3; ++J) {
        coarse_pair4<CB>(kc, va, vb, pk + 2 * RB, pv_ + 2 * RB, ws, lane, J);
        pk += 2 * RB; pv_ = pv_ + 2 * RB;
      }
      pk += step; pv_ = pv_ + step;
      {
        const uint8_t *mn = meta_at(more ? g + nw : g);
        km = ld128(mn); vm = ld128(mn + 512);
      }
      coarse_pair4<CB>(kc, va, vb, pk, pv_, ws, lane, 3);
      {
        const int l2 = tid_x() & 31, b2 = ctaid_y() / a.kv_heads, r2 = b2 * a.q_heads + (ctaid_y() - b2 * a.kv_heads) * QG;
        const float2 e = ws.ls[l2];
        a.lse[(size_t)(r2 + (l2 >> 3)) * a.nblocks + (size_t)g * SBG + (l2 & 7)] = e.x + lg2(e.y);
      }
      if (!more) break;
      g += nw;
    }
  }

  // the CTA's warps merged into one partial (m, l, o) per query, from the slots
  __syncthreads();
  for (int i = threadIdx.x; i < QG * D; i += blockDim.x) {
    const int qi = i / D, ch = i - qi * D;
    float mx = -INFINITY;
#pragma unroll
    for (int w = 0; w < W; ++w) mx = fmaxf(mx, sm.ws[w].ml[qi].x);
    float ls = 0.f, os = 0.f;
#pragma unroll
    for (int w = 0; w < W; ++w) {
      const float2 ml = sm.ws[w].ml[qi];
      const float e = ml.x == -INFINITY ? 0.f : ex2(ml.x - mx);
      ls = fmaf(e, ml.y, ls); os = fmaf(e, sm.ws[w].o[qi][ch], os);
    }
    float *part = a.part + ((size_t)(row0 + qi) * P + cta) * (D + 2);
    part[2 + ch] = os;
    if (!ch) { part[0] = mx; part[1] = ls; }
  }
}

// ---- select pass ------------------------------------------------------------------------------------
// R = log2 sum of the whole row in the units of the partials (sum p, not / 255): the coarse partials of
// every CTA rescaled to R sum to l = 1, and every correction is weighted 2^(m_block - R) in the same units.
// CTA (s, head) lists the refined sub-blocks of slice s in sub-block order: the item order, and so the
// refinement's sums, do not depend on timing.
template <int QG> __device__ __forceinline__ float row_r(const Args &a, int row, int lane) {
  const float *part = a.part + (size_t)row * a.ctas * (D + 2);
  float mx = -INFINITY;
  for (int c = lane; c < a.ctas; c += 32) mx = fmaxf(mx, __ldg(part + (size_t)c * (D + 2)));
  mx = warp_max(mx);
  float ls = 0.f;
  for (int c = lane; c < a.ctas; c += 32) {
    const float mc = __ldg(part + (size_t)c * (D + 2));
    if (mc != -INFINITY) ls = fmaf(__ldg(part + (size_t)c * (D + 2) + 1), ex2(mc - mx), ls);
  }
  return mx + lg2(warp_sum(ls));
}

template <int QG>
__global__ void __launch_bounds__(32 * W) select_kernel(const Args a) {
  constexpr int PER = SLICE / W / 32;  // sub-blocks per lane: warp w scans SLICE / W of the slice at once
  __shared__ float R[QG];
  __shared__ int wc[W];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int bh = blockIdx.y, b = bh / a.kv_heads, kh = bh - b * a.kv_heads;
  const int row0 = b * a.q_heads + kh * QG, slice = blockIdx.x;
  if (warp < QG) {
    const float r = row_r<QG>(a, row0 + warp, lane);
    if (!lane) { R[warp] = r; if (!slice) a.row_lse[row0 + warp] = r; }
  }
  const int lo = SBG + slice * SLICE + warp * (SLICE / W), hi = SBG * (a.groups - 1);
  float x[QG][PER];  // all of the warp's loads in flight together
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int sb = lo + 32 * i + lane;
#pragma unroll
    for (int qi = 0; qi < QG; ++qi) x[qi][i] = sb < hi ? __ldg(a.lse + (size_t)(row0 + qi) * a.nblocks + sb) : -INFINITY;
  }
  __syncthreads();  // R
  uint32_t bal[PER];
  int qm[PER], n = 0;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    qm[i] = 0;
#pragma unroll
    for (int qi = 0; qi < QG; ++qi) qm[i] |= (x[qi][i] - R[qi] > a.log2_tau) << qi;
    bal[i] = __ballot_sync(FULL, qm[i] != 0);
    n += __popc(bal[i]);
  }
  if (!lane) wc[warp] = n;
  __syncthreads();
  int off = 0, total = 0;
#pragma unroll
  for (int w = 0; w < W; ++w) { off += w < warp ? wc[w] : 0; total += wc[w]; }
  int *items = a.items + ((size_t)bh * a.nslices + slice) * SLICE;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    if (qm[i]) items[off + __popc(bal[i] & ((1u << lane) - 1u))] = (lo + 32 * i + lane) << 4 | qm[i];
    off += __popc(bal[i]);
  }
  if (!threadIdx.x) a.counts[(size_t)bh * a.nslices + slice] = total;
}

// ---- refinement pass --------------------------------------------------------------------------------
// Item k of the head's list (slices concatenated) goes to warp k mod (gridDim.x W): every warp gets the same
// number of items +-1 wherever the refined sub-blocks cluster.  Per item both records (coarse, fine), the
// group's metadata, from which the q-hat of the coarse pass is recomputed (same code, same inputs: the same
// codes); corrections accumulate in registers, one partial per CTA.
template <int QG> struct SmemRefine {
  alignas(16) float qs[QG][D];  // q * scale * log2 e
  alignas(16) uint32_t stage[W][QG][32];
  alignas(16) uint32_t pb[W][2][QG][4];
  int pre[MAX_SLICES + 1];
  alignas(16) float o[W][QG][D];
  float l[W][QG];
  int last;
};

template <int CB, int FB, int QG>
__global__ void __launch_bounds__(32 * W, 2) refine_kernel(const Args a) {
  static_assert(QG <= 2, "four queries per KV head: refine_kernel4");
  __shared__ SmemRefine<QG> sm;
  constexpr int RC = record_bytes<CB>(), RF = record_bytes<FB>();
  constexpr float HOFF = .5f * ((1 << FB) - 1);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, half = lane >> 4;
  const int bh = blockIdx.y, b = bh / a.kv_heads, kh = bh - b * a.kv_heads;
  const int row0 = b * a.q_heads + kh * QG, cta = blockIdx.x, P = gridDim.x, ns = P * W;
  float R[QG];
#pragma unroll
  for (int qi = 0; qi < QG; ++qi) R[qi] = __ldg(a.row_lse + row0 + qi);

  if (warp == 1)
#pragma unroll
    for (int qi = 0; qi < QG; ++qi) {
      const uint2 qw = __ldg(reinterpret_cast<const uint2 *>(a.q + (size_t)(row0 + qi) * D) + lane);
      float qf[4];
      bf16x4(qw.x, qw.y, qf);
      reinterpret_cast<float4 *>(sm.qs[qi])[lane] =
          make_float4(qf[0] * a.scale_log2, qf[1] * a.scale_log2, qf[2] * a.scale_log2, qf[3] * a.scale_log2);
    }
  if (!warp) {  // prefix sums of the slice counts
    int run = 0;
    for (int s0 = 0; s0 < a.nslices; s0 += 32) {
      const int s = s0 + lane, n = s < a.nslices ? __ldg(a.counts + (size_t)bh * a.nslices + s) : 0;
      int x = n;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(FULL, x, o); x += lane >= o ? y : 0; }
      if (s < a.nslices) sm.pre[s] = run + x - n;
      run += __shfl_sync(FULL, x, 31);
    }
    if (!lane) sm.pre[a.nslices] = run;
  }
  __syncthreads();
  const int total = sm.pre[a.nslices];

  float dl[QG], dob[QG][4];
#pragma unroll
  for (int qi = 0; qi < QG; ++qi) {
    dl[qi] = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) dob[qi][c] = 0.f;
  }
  const size_t head = (size_t)bh * a.cgroups * SBG;
  const int *items = a.items + (size_t)bh * a.nslices * SLICE;
  const uint8_t *meta = reinterpret_cast<const uint8_t *>(a.c.meta) + (size_t)bh * a.cgroups * 1024 + 16 * lane;
  // software pipeline: the stream's next item (its index, records and metadata) is requested before this one
  // is computed
  struct Load { int it; Codes<CB> kc, vc; Codes<FB> kf, vf; uint4 km, vm; };
  int s = 0;
  auto item_at = [&](int k) {
    while (sm.pre[s + 1] <= k) ++s;
    return __ldg(items + (size_t)s * SLICE + (k - sm.pre[s]));
  };
  auto fetch = [&](Load &L, int it) {
    const int sb = it >> 4;
    const Rec cp = rec_at(a.c.coarse + (head + sb) * RC, lane), fp = rec_at(a.c.fine + (head + sb) * RF, lane);
    load_k<CB>(L.kc, cp.p, lane); load_k<FB>(L.kf, fp.p, lane);
    load_v<CB>(L.vc, cp); load_v<FB>(L.vf, fp);
    L.km = ld128(meta + (size_t)(sb / SBG) * 1024); L.vm = ld128(meta + (size_t)(sb / SBG) * 1024 + 512);
    L.it = it;
  };
  Load nx;
  int k = cta * W + warp;
  if (k < total) fetch(nx, item_at(k));
  for (; k < total; k += ns) {
    const Load cur = nx;
    if (k + ns < total) fetch(nx, item_at(k + ns));
    const int it = cur.it;
    const Codes<CB> &kc = cur.kc, &vc = cur.vc;
    const Codes<FB> &kf = cur.kf, &vf = cur.vf;
    const uint4 vm = cur.vm;
    GroupState<QG> gs;
    setup_group<CB, FB, QG>(lane, cur.km, vm, sm.qs, sm.stage[warp], gs);
    float xh[QG], xl[QG];
    qk<CB, QG>(sm.stage[warp], half, kc, xh);
    qk<FB, QG>(sm.stage[warp], half, kf, xl);
    float wc[QG], wf[QG];
#pragma unroll
    for (int qi = 0; qi < QG; ++qi) {
      const float sc = fmaf(gs.sqc[qi], xh[qi], gs.c0[qi]);
      const float sf = fmaf(gs.sq[qi], fmaf((float)(1 << FB), xh[qi], xl[qi]), gs.qb[qi]);
      const float mc = max16(sc), mf = max16(sf);
      put_p(sm.pb[warp][0][qi], sc, mc, lane);
      put_p(sm.pb[warp][1][qi], sf, mf, lane);
      const bool on = (it >> qi) & 1;  // this query's sub-block is refined
      wc[qi] = on ? ex2(mc - R[qi]) : 0.f;
      wf[qi] = on ? ex2(mf - R[qi]) : 0.f;
    }
    __syncwarp();
    uint32_t Pc[QG][4], Pf[QG][4];
    float spc[QG], spf[QG];
#pragma unroll
    for (int qi = 0; qi < QG; ++qi) { spc[qi] = get_p(sm.pb[warp][0][qi], Pc[qi]); spf[qi] = get_p(sm.pb[warp][1][qi], Pf[qi]); }
    float yc[QG][4], yhf[QG][4], ylf[QG][4];
    pv<CB, QG>(Pc, vc, yc);
    pv<CB, QG>(Pf, vc, yhf);
    pv<FB, QG>(Pf, vf, ylf);
    float vb[4], vs[4];
    bf16x4(vm.x, vm.y, vb); bf16x4(vm.z, vm.w, vs);
#pragma unroll
    for (int qi = 0; qi < QG; ++qi) {
      dl[qi] += fmaf(wf[qi], spf[qi], -wc[qi] * spc[qi]);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float vhi = vs[c] * ((1 << FB) * v_unscale<CB>(c));
        const float fine = fmaf(vb[c], spf[qi], fmaf(vhi, yhf[qi][c], vs[c] * v_unscale<FB>(c) * ylf[qi][c]));
        const float coarse = fmaf(fmaf(vs[c], HOFF, vb[c]), spc[qi], vhi * yc[qi][c]);
        dob[qi][c] += fmaf(wf[qi], fine, -wc[qi] * coarse);
      }
    }
    __syncwarp();  // stage and pb are rewritten by the next item
  }

  // the CTA's corrections, then the last CTA of the (batch, KV head) writes the output
#pragma unroll
  for (int qi = 0; qi < QG; ++qi) {
    if (!lane) sm.l[warp][qi] = dl[qi];
    *reinterpret_cast<float4 *>(&sm.o[warp][qi][4 * lane]) = make_float4(dob[qi][0], dob[qi][1], dob[qi][2], dob[qi][3]);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < QG * D; i += blockDim.x) {
    const int qi = i / D, ch = i - qi * D;
    float os = 0.f, ls = 0.f;
#pragma unroll
    for (int w = 0; w < W; ++w) { os += sm.o[w][qi][ch]; ls += sm.l[w][qi]; }
    float *dp = a.dpart + ((size_t)(row0 + qi) * P + cta) * (D + 1);
    dp[1 + ch] = os;
    if (!ch) dp[0] = ls;
  }
  __threadfence();
  __syncthreads();
  if (!threadIdx.x) sm.last = atomicAdd(a.counter + bh, 1) == P - 1;
  __syncthreads();
  if (!sm.last) return;
  __threadfence();
  // final combine: warp w sums the CTAs w, w + W, ... of both passes; lane l owns channels 4 l .. + 3
#pragma unroll
  for (int qi = 0; qi < QG; ++qi) {
    const float Rq = __ldg(a.row_lse + row0 + qi);
    const float *part = a.part + (size_t)(row0 + qi) * a.ctas * (D + 2);
    const float *dp = a.dpart + (size_t)(row0 + qi) * P * (D + 1);
    float O[4] = {0.f, 0.f, 0.f, 0.f}, L = 0.f;
    // four partials per round, all loads in flight together (the combine is a latency chain otherwise)
    for (int c0 = warp; c0 < a.ctas; c0 += 4 * W) {
      float mc[4], lc[4];
      float2 o0[4], o1[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int c = c0 + u * W;
        const float *pc = part + (size_t)min(c, a.ctas - 1) * (D + 2);
        mc[u] = c < a.ctas ? __ldcg(pc) : -INFINITY; lc[u] = __ldcg(pc + 1);
        o0[u] = __ldcg(reinterpret_cast<const float2 *>(pc + 2 + 4 * lane));  // 8-byte aligned
        o1[u] = __ldcg(reinterpret_cast<const float2 *>(pc + 4 + 4 * lane));
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const float e = mc[u] == -INFINITY ? 0.f : ex2(mc[u] - Rq);
        L = fmaf(e, lc[u], L);
        O[0] = fmaf(e, o0[u].x, O[0]); O[1] = fmaf(e, o0[u].y, O[1]); O[2] = fmaf(e, o1[u].x, O[2]); O[3] = fmaf(e, o1[u].y, O[3]);
      }
    }
    for (int c0 = warp; c0 < P; c0 += 4 * W) {
      float lc[4], oc[4][4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int c = c0 + u * W;
        const float *dc = dp + (size_t)min(c, P - 1) * (D + 1);
        const float f = c < P ? 1.f : 0.f;
        lc[u] = f * __ldcg(dc);
#pragma unroll
        for (int x2 = 0; x2 < 4; ++x2) oc[u][x2] = f * __ldcg(dc + 1 + 4 * lane + x2);
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        L += lc[u];
#pragma unroll
        for (int x2 = 0; x2 < 4; ++x2) O[x2] += oc[u][x2];
      }
    }
    if (!lane) sm.l[warp][qi] = L;
    *reinterpret_cast<float4 *>(&sm.o[warp][qi][4 * lane]) = make_float4(O[0], O[1], O[2], O[3]);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < QG * D; i += blockDim.x) {
    const int qi = i / D, ch = i - qi * D;
    float O = 0.f, L = 0.f;
#pragma unroll
    for (int w = 0; w < W; ++w) { O += sm.o[w][qi][ch]; L += sm.l[w][qi]; }
    a.out[(size_t)(row0 + qi) * D + ch] = to_bf16(O / L);
  }
  if (!threadIdx.x) a.counter[bh] = 0;  // the next call counts from zero again
}

// ---- refinement pass, four queries per KV head ---------------------------------------------------
// One warp serves all four queries of an item (its records read and decoded once), in two halves of two
// queries; the corrections and the group's per-query parameters live in the warp's shared slots.
struct RefineSlots4 {
  alignas(16) uint32_t stage[4][32];
  alignas(16) uint32_t pb[2][4][4];  // P bytes, coarse / full scores, per query
  alignas(16) float4 gq[4];          // (sq, q . base, c0, sq 2^FB) per query
  float dl[4];
  alignas(16) float o[4][D];         // corrections do per query (the final combine's buffer as well)
};
struct SmemRefine4 {
  alignas(16) float qs[4][D];
  RefineSlots4 ws[W];
  int pre[MAX_SLICES + 1];
  int last;
};

template <int CB, int FB>
__global__ void __launch_bounds__(32 * W, 2) refine_kernel4(const Args a) {
  __shared__ SmemRefine4 sm;
  constexpr int QG = 4, RC = record_bytes<CB>(), RF = record_bytes<FB>();
  constexpr float HOFF = .5f * ((1 << FB) - 1);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, half = lane >> 4;
  const int bh = blockIdx.y, b = bh / a.kv_heads, kh = bh - b * a.kv_heads;
  const int row0 = b * a.q_heads + kh * QG, cta = blockIdx.x, P = gridDim.x;
  RefineSlots4 &ws = sm.ws[warp];

  if (warp < QG) {
    const uint2 qw = __ldg(reinterpret_cast<const uint2 *>(a.q + (size_t)(row0 + warp) * D) + lane);
    float qf[4];
    bf16x4(qw.x, qw.y, qf);
    reinterpret_cast<float4 *>(sm.qs[warp])[lane] =
        make_float4(qf[0] * a.scale_log2, qf[1] * a.scale_log2, qf[2] * a.scale_log2, qf[3] * a.scale_log2);
  }
  if (warp == QG) {  // prefix sums of the slice counts
    int run = 0;
    for (int s0 = 0; s0 < a.nslices; s0 += 32) {
      const int s = s0 + lane, n = s < a.nslices ? __ldg(a.counts + (size_t)bh * a.nslices + s) : 0;
      int x = n;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(FULL, x, o); x += lane >= o ? y : 0; }
      if (s < a.nslices) sm.pre[s] = run + x - n;
      run += __shfl_sync(FULL, x, 31);
    }
    if (!lane) sm.pre[a.nslices] = run;
  }
#pragma unroll
  for (int qi = 0; qi < QG; ++qi) reinterpret_cast<float4 *>(ws.o[qi])[lane] = make_float4(0.f, 0.f, 0.f, 0.f);
  if (lane < QG) ws.dl[lane] = 0.f;
  __syncthreads();
  const int total = sm.pre[a.nslices];

  const size_t head = (size_t)bh * a.cgroups * SBG;
  const int *items = a.items + (size_t)bh * a.nslices * SLICE;
  const uint8_t *meta = reinterpret_cast<const uint8_t *>(a.c.meta) + (size_t)bh * a.cgroups * 1024 + 16 * lane;
  struct Load { int it; Codes<CB> kc, vc; Codes<FB> kf, vf; uint4 km, vm; };
  int s = 0;
  auto item_at = [&](int k) {
    while (sm.pre[s + 1] <= k) ++s;
    return __ldg(items + (size_t)s * SLICE + (k - sm.pre[s]));
  };
  auto fetch = [&](Load &L, int it) {
    const int sb = it >> 4;
    const Rec cp = rec_at(a.c.coarse + (head + sb) * RC, lane), fp = rec_at(a.c.fine + (head + sb) * RF, lane);
    load_k<CB>(L.kc, cp.p, lane); load_k<FB>(L.kf, fp.p, lane);
    load_v<CB>(L.vc, cp); load_v<FB>(L.vf, fp);
    L.km = ld128(meta + (size_t)(sb / SBG) * 1024); L.vm = ld128(meta + (size_t)(sb / SBG) * 1024 + 512);
    L.it = it;
  };
  Load nx;
  int k = cta * W + warp;
  if (k < total) fetch(nx, item_at(k));
  for (; k < total; k += P * W) {
    const Load cur = nx;
    if (k + P * W < total) fetch(nx, item_at(k + P * W));
    {
      GroupState<QG> gs;
      setup_group<CB, FB, QG>(lane, cur.km, cur.vm, sm.qs, ws.stage, gs);
      if (!lane)
#pragma unroll
        for (int qi = 0; qi < QG; ++qi) ws.gq[qi] = make_float4(gs.sq[qi], gs.qb[qi], gs.c0[qi], gs.sqc[qi]);
      __syncwarp();
    }
    float vb[4], vs[4];
    bf16x4(cur.vm.x, cur.vm.y, vb); bf16x4(cur.vm.z, cur.vm.w, vs);
#pragma unroll 1
    for (int h = 0; h < 2; ++h) {
      const int q0 = 2 * h;
      if (!((cur.it >> q0) & 3)) continue;  // warp-uniform: neither query of this half refines the item
      float xh[2], xl[2];
      qk<CB, 2>(ws.stage + q0, half, cur.kc, xh);
      qk<FB, 2>(ws.stage + q0, half, cur.kf, xl);
      float wc[2], wf[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int qi = q0 + j;
        const float4 gq = ws.gq[qi];
        const float R = __ldg(a.row_lse + row0 + qi);
        const float sc = fmaf(gq.w, xh[j], gq.z), sf = fmaf(gq.x, fmaf((float)(1 << FB), xh[j], xl[j]), gq.y);
        const float mc = max16(sc), mf = max16(sf);
        put_p(ws.pb[0][qi], sc, mc, lane);
        put_p(ws.pb[1][qi], sf, mf, lane);
        const bool on = (cur.it >> qi) & 1;
        wc[j] = on ? ex2(mc - R) : 0.f;
        wf[j] = on ? ex2(mf - R) : 0.f;
      }
      __syncwarp();
      uint32_t Pc[2][4], Pf[2][4];
      float spc[2], spf[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) { spc[j] = get_p(ws.pb[0][q0 + j], Pc[j]); spf[j] = get_p(ws.pb[1][q0 + j], Pf[j]); }
      float yc[2][4], yhf[2][4], ylf[2][4];
      pv<CB, 2>(Pc, cur.vc, yc);
      pv<CB, 2>(Pf, cur.vc, yhf);
      pv<FB, 2>(Pf, cur.vf, ylf);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int qi = q0 + j;
        float4 o = reinterpret_cast<float4 *>(ws.o[qi])[lane];
        float ov[4] = {o.x, o.y, o.z, o.w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const float vhi = vs[c] * ((1 << FB) * v_unscale<CB>(c));
          const float fine = fmaf(vb[c], spf[j], fmaf(vhi, yhf[j][c], vs[c] * v_unscale<FB>(c) * ylf[j][c]));
          const float coarse = fmaf(fmaf(vs[c], HOFF, vb[c]), spc[j], vhi * yc[j][c]);
          ov[c] += fmaf(wf[j], fine, -wc[j] * coarse);
        }
        reinterpret_cast<float4 *>(ws.o[qi])[lane] = make_float4(ov[0], ov[1], ov[2], ov[3]);
        if (!lane) ws.dl[qi] += fmaf(wf[j], spf[j], -wc[j] * spc[j]);
      }
      __syncwarp();  // stage, pb and the slots are rewritten by the next half or item
    }
  }

  // the CTA's corrections, then the last CTA of the (batch, KV head) writes the output
  __syncthreads();
  for (int i = threadIdx.x; i < QG * D; i += blockDim.x) {
    const int qi = i / D, ch = i - qi * D;
    float os = 0.f, ls = 0.f;
#pragma unroll
    for (int w = 0; w < W; ++w) { os += sm.ws[w].o[qi][ch]; ls += sm.ws[w].dl[qi]; }
    float *dp = a.dpart + ((size_t)(row0 + qi) * P + cta) * (D + 1);
    dp[1 + ch] = os;
    if (!ch) dp[0] = ls;
  }
  __threadfence();
  __syncthreads();
  if (!threadIdx.x) sm.last = atomicAdd(a.counter + bh, 1) == P - 1;
  __syncthreads();
  if (!sm.last) return;
  __threadfence();
#pragma unroll 1
  for (int qi = 0; qi < QG; ++qi) {
    const float Rq = __ldg(a.row_lse + row0 + qi);
    const float *part = a.part + (size_t)(row0 + qi) * a.ctas * (D + 2);
    const float *dp = a.dpart + (size_t)(row0 + qi) * P * (D + 1);
    float O[4] = {0.f, 0.f, 0.f, 0.f}, L = 0.f;
    for (int c0 = warp; c0 < a.ctas; c0 += 4 * W) {
      float mc[4], lc[4];
      float2 o0[4], o1[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int c = c0 + u * W;
        const float *pc = part + (size_t)min(c, a.ctas - 1) * (D + 2);
        mc[u] = c < a.ctas ? __ldcg(pc) : -INFINITY; lc[u] = __ldcg(pc + 1);
        o0[u] = __ldcg(reinterpret_cast<const float2 *>(pc + 2 + 4 * lane));
        o1[u] = __ldcg(reinterpret_cast<const float2 *>(pc + 4 + 4 * lane));
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const float e = mc[u] == -INFINITY ? 0.f : ex2(mc[u] - Rq);
        L = fmaf(e, lc[u], L);
        O[0] = fmaf(e, o0[u].x, O[0]); O[1] = fmaf(e, o0[u].y, O[1]); O[2] = fmaf(e, o1[u].x, O[2]); O[3] = fmaf(e, o1[u].y, O[3]);
      }
    }
    for (int c0 = warp; c0 < P; c0 += 4 * W) {
      float lc[4], oc[4][4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int c = c0 + u * W;
        const float *dc = dp + (size_t)min(c, P - 1) * (D + 1);
        const float f = c < P ? 1.f : 0.f;
        lc[u] = f * __ldcg(dc);
#pragma unroll
        for (int x2 = 0; x2 < 4; ++x2) oc[u][x2] = f * __ldcg(dc + 1 + 4 * lane + x2);
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        L += lc[u];
#pragma unroll
        for (int x2 = 0; x2 < 4; ++x2) O[x2] += oc[u][x2];
      }
    }
    __syncwarp();
    if (!lane) ws.dl[qi] = L;
    reinterpret_cast<float4 *>(ws.o[qi])[lane] = make_float4(O[0], O[1], O[2], O[3]);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < QG * D; i += blockDim.x) {
    const int qi = i / D, ch = i - qi * D;
    float O = 0.f, L = 0.f;
#pragma unroll
    for (int w = 0; w < W; ++w) { O += sm.ws[w].o[qi][ch]; L += sm.ws[w].dl[qi]; }
    a.out[(size_t)(row0 + qi) * D + ch] = to_bf16(O / L);
  }
  if (!threadIdx.x) a.counter[bh] = 0;  // the next call counts from zero again
}

// ---- launch ---------------------------------------------------------------------------------------
// CTAs per (batch, KV head): the device's resident CTAs split over the heads (persistent, one wave), at
// most bound (the scratch's partials), and no more than gives every warp 16 sub-blocks of work.
inline int blocks_per_sm(const void *kernel) {
  static struct { const void *k; int dev, occ; } cache[64];
  static int n = 0;
  int dev = 0;
  cudaGetDevice(&dev);
  for (int i = 0; i < n; ++i) if (cache[i].k == kernel && cache[i].dev == dev) return cache[i].occ;
  int occ = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, 32 * W, 0);
  if (n < 64) cache[n++] = {kernel, dev, std::max(occ, 1)};
  return std::max(occ, 1);
}
inline int per_head(const void *kernel, int bound, const Args &a, long long work) {
  int dev = 0, sms = 1;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const int heads = a.batch * a.kv_heads, slots = blocks_per_sm(kernel) * sms;
  const long long by_work = std::max<long long>(1, work / (16LL * W));
  return (int)std::max<long long>(1, std::min<long long>({(long long)std::max(1, slots / heads), (long long)bound, by_work}));
}

template <int CB, int FB, int QG> cudaError_t launch_qg(Args a, int bound, cudaStream_t stream) {
  const int heads = a.batch * a.kv_heads;
  const long long work = (long long)a.groups * SBG;
  const void *ck, *rk;
  if constexpr (QG == 4) {
    ck = reinterpret_cast<const void *>(coarse_kernel4<CB, FB>);
    rk = reinterpret_cast<const void *>(refine_kernel4<CB, FB>);
  } else {
    ck = reinterpret_cast<const void *>(coarse_kernel<CB, FB, QG>);
    rk = reinterpret_cast<const void *>(refine_kernel<CB, FB, QG>);
  }
  const int p1 = per_head(ck, bound, a, work);
  const int p2 = per_head(rk, bound, a, work / 4);
  a.ctas = p1;
  if constexpr (QG == 4) coarse_kernel4<CB, FB><<<dim3(p1, heads), 32 * W, 0, stream>>>(a);
  else coarse_kernel<CB, FB, QG><<<dim3(p1, heads), 32 * W, 0, stream>>>(a);
  select_kernel<QG><<<dim3(a.nslices, heads), 32 * W, 0, stream>>>(a);
  if constexpr (QG == 4) refine_kernel4<CB, FB><<<dim3(p2, heads), 32 * W, 0, stream>>>(a);
  else refine_kernel<CB, FB, QG><<<dim3(p2, heads), 32 * W, 0, stream>>>(a);
  return cudaGetLastError();
}

template <int CB, int FB> cudaError_t launch(const Args &a, int bound, cudaStream_t stream) {
  switch (a.q_heads / a.kv_heads) {
    case 1: return launch_qg<CB, FB, 1>(a, bound, stream);
    case 2: return launch_qg<CB, FB, 2>(a, bound, stream);
    default: return launch_qg<CB, FB, 4>(a, bound, stream);
  }
}

}  // namespace mlops_fbla_detail
#endif
