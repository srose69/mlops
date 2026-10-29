// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// FBLA prefill, sm_61+: `count` new tokens are appended and the query at position p attends to tokens 0 .. p,
// closed groups through their 6-bit codes, the sink and own group in BF16.  Launches:
//
//   close_kernel     quantizes the groups the chunk closes into records and metadata, one CTA per (group, head)
//   attention        sm_75 .. sm_89: transcode_kernel + prefill_tc_kernel (tensors/); else stream_kernel (DP4A)
//   finish_kernel    the sink and open group's BF16 tokens, after the attention has read the old ones
#ifndef MLOPS_FBLA_PREFILL_CUH
#define MLOPS_FBLA_PREFILL_CUH

#include "fbla_kernels.cuh"
#ifndef __CUDACC_RTC__
#include <algorithm>
#include <cstdlib>
#endif

namespace mlops_fbla_detail {


// The BF16 row of the token at position pos of this head: the chunk's input (pos >= n0), the sink (pos < 128),
// or the group that was open before the chunk.
__device__ __forceinline__ const uint16_t *token_row(const uint16_t *sink, const uint16_t *open, const uint16_t *in,
                                                    int n0, int pos) {
  if (pos >= n0) return in + (size_t)(pos - n0) * D;
  if (pos < G) return sink + (size_t)pos * D;
  return open + (size_t)(pos - (n0 - 1) / G * G) * D;
}

// x rounded to BF16 toward -inf (up = false) or +inf: round to nearest, then one step if that went the wrong way
// (the encoder's definition: python/mlops/attention/fbla.py, _bf16)
__device__ __forceinline__ uint16_t bf16_dir(float x, bool up) {
  uint16_t u = to_bf16(x);
  const float y = bf16(u);
  if (up ? y < x : y > x) u = y > 0.f ? (up ? u + 1 : u - 1) : y < 0.f ? (up ? u - 1 : u + 1) : (up ? 0x0001 : 0x8001);
  return u;
}

// ---- closing groups ---------------------------------------------------------------------------------
// Byte o of a BITS-plane record (mlops_fbla.h) from part(tensor, lane, slot, byte) = that code's bits in this
// plane: the inverse of the record layout, so every thread writes whole words and no write is shared.
template <int BITS, class F> __device__ __forceinline__ uint32_t record_byte(int o, F part) {
  if (BITS == 4) {
    const int t = o >> 10, u = o & 1023, l = (u & 511) >> 4, w = 4 * (u >> 9) + ((u & 15) >> 2), b = u & 3;
    return part(t, l, w, b) | part(t, l, w + 8, b) << 4;
  } else if (BITS == 2) {
    const int t = o >> 9, l = (o & 511) >> 4, k = (o & 15) >> 2, b = o & 3;
    return part(t, l, k, b) | part(t, l, 4 + k, b) << 2 | part(t, l, 8 + k, b) << 4 | part(t, l, 12 + k, b) << 6;
  } else {
    if (o < 1024) {  // words 0 .. 3 of K (bytes 0 .. 511) and V (512 .. 1023)
      const int t = o >> 9, l = (o & 511) >> 4, w = (o & 15) >> 2, b = o & 3;
      return part(t, l, w, b) | part(t, l, w + 6, b) << 3 | (part(t, l, 12 + w, b) & 3u) << 6;
    }
    int t, l;
    const int u = o < 1280 ? o - 1024 : o - 1280, i = (u & 7) >> 2, b = u & 3;
    if (o < 1280) { t = 0; l = (u >> 4) + 16 * ((u & 15) >> 3); }  // K words 4, 5: 1024 + 16 (l % 16) + 8 (l / 16)
    else { t = 1; l = u >> 3; }                                      // V words 4, 5: 1280 + 8 l
    const int w = 4 + i;
    return part(t, l, w, b) | part(t, l, w + 6, b) << 3 | (part(t, l, 12 + 2 * i, b) >> 2) << 6 |
           (part(t, l, 13 + 2 * i, b) >> 2) << 7;
  }
}

// Group g of head bh, the chunk starting at n0: per channel (thread c < 128 of T) the BF16 base (minimum rounded down)
// and step (range / 63 rounded up) of K and V, every code rint((x - base) / step), then the coarse and fine records.
template <int CB, int FB, int T, class Pos = Grid>
__device__ __forceinline__ void close_group(const PrefillArgs &a, int n0, int g, int bh, uint8_t (&code)[2][G][D]) {
  constexpr int RC = record_bytes<CB>(), RF = record_bytes<FB>();
  const int c = Pos::tid();
  const size_t eh = (size_t)bh * G * D, ih = (size_t)bh * a.count * D;
  const size_t unit = (size_t)bh * a.hs + (size_t)g * a.gs;
  uint16_t *meta = const_cast<uint16_t *>(a.c.meta) + unit * 512;
  if (a.c.tc_images && c == 0) *image_flag(a.c, unit) = 0u;  // its image is stale
#pragma unroll
  for (int x = 0; x < (T == D || c < D ? 2 : 0); ++x) {
    const uint16_t *sink = (x ? a.c.exact_sink_v : a.c.exact_sink_k) + eh;
    const uint16_t *open = (x ? a.c.exact_open_v : a.c.exact_open_k) + eh;
    const uint16_t *in = (x ? a.v : a.k) + ih;
    float lo = INFINITY, hi = -INFINITY;
    for (int t = 0; t < G; ++t) {
      const float e = bf16(InputLoad<Pos>::get(token_row(sink, open, in, n0, g * G + t) + c));
      lo = fminf(lo, e); hi = fmaxf(hi, e);
    }
    const uint16_t ub = bf16_dir(lo, false);
    const float base = bf16(ub);
    const uint16_t us = bf16_dir(__fmul_rn(__fsub_rn(hi, base), 1.f / 63.f), true);
    const float step = bf16(us), inv = step > 0.f ? __frcp_rn(step) : 0.f;
    for (int t = 0; t < G; ++t) {
      const float e = bf16(InputLoad<Pos>::get(token_row(sink, open, in, n0, g * G + t) + c));
      const float r = rintf(__fmul_rn(__fsub_rn(e, base), inv));
      code[x][t][c] = (uint8_t)fminf(fmaxf(r, 0.f), 63.f);
    }
    meta[256 * x + 8 * (c >> 2) + (c & 3)] = ub;      // lane c / 4: four bases, four steps
    meta[256 * x + 8 * (c >> 2) + 4 + (c & 3)] = us;
  }
  __syncthreads();
  // records: K code (lane l, slot s, byte b) = channel 64 (l / 16) + 4 s + b of token l % 16; V code = channel
  // 4 l + v_channel(s) of token 4 v_quad(s) + b (the plane's own slot order)
  const size_t sb0 = unit * SBG;
  for (int wi = Pos::tid(); wi < SBG * (RC + RF) / 4; wi += blockDim.x) {
    const bool fine = wi >= SBG * RC / 4;
    const int ow = fine ? 4 * wi - SBG * RC : 4 * wi, jj = ow / (fine ? RF : RC), o = ow - jj * (fine ? RF : RC);
    const uint8_t (*ck)[D] = code[0] + 16 * jj, (*cv)[D] = code[1] + 16 * jj;
    uint32_t word = 0;
#pragma unroll
    for (int b4 = 0; b4 < 4; ++b4) {
      uint32_t byte;
      if (fine) {
        auto part = [&](int t, int l, int s, int b) -> uint32_t {
          const uint8_t e = t ? cv[4 * v_quad<FB>(s) + b][4 * l + v_channel<FB>(s)] : ck[l & 15][64 * (l >> 4) + 4 * s + b];
          return e & ((1u << FB) - 1u);
        };
        byte = record_byte<FB>(o + b4, part);
      } else {
        auto part = [&](int t, int l, int s, int b) -> uint32_t {
          const uint8_t e = t ? cv[4 * v_quad<CB>(s) + b][4 * l + v_channel<CB>(s)] : ck[l & 15][64 * (l >> 4) + 4 * s + b];
          return (uint32_t)e >> FB;
        };
        byte = record_byte<CB>(o + b4, part);
      }
      word |= byte << (8 * b4);
    }
    uint8_t *plane = const_cast<uint8_t *>(fine ? a.c.fine : a.c.coarse) + (sb0 + jj) * (fine ? RF : RC);
    *reinterpret_cast<uint32_t *>(plane + o) = word;
  }
}

// The item of close_kernel: the Pos::x()'th group the chunk closes, of head Pos::y() (a CTA past them returns)
template <int CB, int FB, class Pos = Grid>
__device__ __forceinline__ void close_item(const PrefillArgs &a, uint8_t (&code)[2][G][D]) {
  const int n0 = n0_of<Pos>(a), g = (n0 == 0 ? 0 : (n0 - 1) / G) + Pos::x();
  if (g >= (n0 + count_of<Pos>(a) - 1) / G) return;  // (CTA-uniform)
  close_group<CB, FB, D, Pos>(a, n0, g, Pos::y(), code);
}
template <int CB, int FB>
__global__ void __launch_bounds__(128) close_kernel(const PrefillArgs a) {
  __shared__ uint8_t code[2][G][D];  // [K, V][token][channel]
  close_item<CB, FB, Grid>(a, code);
}

// The sink (positions < 128) and the open group take the chunk's tokens; a new group's unused rows are cleared.
// Format-independent: one object (MLOPS_FBLA_DEFINE_SHARED), launched through launch_finish.
void launch_finish(const PrefillArgs &a, int heads, cudaStream_t stream);
#if defined(MLOPS_FBLA_DEFINE_SHARED) || defined(__CUDACC_RTC__)
// The item of finish_kernel: head Pos::x()'s sink and open group take the chunk's tokens
template <class Pos = Grid>
__device__ __forceinline__ void finish_item(const PrefillArgs &a) {
  const int bh = Pos::x(), n0 = n0_of<Pos>(a), n1 = n0 + count_of<Pos>(a), gl = (n1 - 1) / G;
  const bool fresh = n0 == 0 || (n0 - 1) / G != gl;  // the open group is a new one
  const size_t eh = (size_t)bh * G * D, ih = (size_t)bh * a.count * D;
#pragma unroll
  for (int x = 0; x < 2; ++x) {
    const uint4 *in = reinterpret_cast<const uint4 *>((x ? a.v : a.k) + ih);
    uint4 *open = reinterpret_cast<uint4 *>(const_cast<uint16_t *>(x ? a.c.exact_open_v : a.c.exact_open_k) + eh);
    uint4 *sink = reinterpret_cast<uint4 *>(const_cast<uint16_t *>(x ? a.c.exact_sink_v : a.c.exact_sink_k) + eh);
    for (int e = threadIdx.x; e < G * 16; e += blockDim.x) {  // 16 uint4 per token row
      const int t = e >> 4, pos = gl * G + t;
      if (pos >= n0 && pos < n1) open[e] = InputLoad<Pos>::get(in + (size_t)(pos - n0) * 16 + (e & 15));
      else if (fresh && pos >= n1) open[e] = make_uint4(0u, 0u, 0u, 0u);
      if (t >= n0 && t < n1) sink[e] = InputLoad<Pos>::get(in + (size_t)(t - n0) * 16 + (e & 15));
    }
  }
}
#ifndef __CUDACC_RTC__
__global__ void __launch_bounds__(128) finish_kernel(const PrefillArgs a) { finish_item<Grid>(a); }
void launch_finish(const PrefillArgs &a, int heads, cudaStream_t stream) { finish_kernel<<<heads, 128, 0, stream>>>(a); }
#endif
#endif

// ---- attention: streaming GEMM tiles -------------------------------------------------------------------
// PROWS = 8 x consumer warps rows per CTA: BF16 groups first, then closed groups newest first, online softmax.
// Producer warps decode records into swizzled shared tiles (a ring of half-group stages); consumers compute.
// They meet at named barriers (stage full / empty) only.

// Kd[64 tokens][32 words]: word = 4 channels; quad q of a token at q ^ ((token / 4) % 8).
// Vd[16 token words][128 ch]: word = 4 tokens; quad q at q ^ ((q / 8) % 2).  Every fragment is one LDS.128.

// Consumer lane (r = lane / 16, b = lane % 16), rows 8 warp + 4 r .. + 3: QK tokens 4 b .. + 3, PV channels 8 b .. + 7.
// 8 warps per CTA: 3-bit planes need two producer warps, 4+2 one.
constexpr int NST = 2, PTH = 256;  // ring stages, threads per CTA
template <int CB> struct Warps {
  static constexpr int NCW = CB == 3 ? 6 : 7, NPW = 8 - NCW;  // consumer / producer warps
  static constexpr int PROWS = 8 * NCW, CTH = 32 * NCW, NLR = 256 / (32 * NPW);  // rows; lane-records per producer thread
};
enum { BAR_FULL = 1, BAR_EMPTY = 1 + NST, BAR_CONS = 1 + 2 * NST };  // named barrier ids (0: __syncthreads)

// bar.sync / bar.arrive order shared memory among the participating threads; the empty asm with the memory
// clobber only keeps the compiler from moving accesses across them (a clobber on the barrier itself makes
// ptxas add a MEMBAR.CTA, which waits for every outstanding load)
__device__ __forceinline__ void bar_sync(int id, int n) {
  asm volatile("" ::: "memory");
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n));
  asm volatile("" ::: "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int n) {
  asm volatile("" ::: "memory");
  asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n));
}

struct Stage {
  struct ClosedStorage {
    alignas(16) uint32_t kd[64 * 32];
    alignas(16) uint32_t vd[16 * 128];
  };
  struct ExactStorage {
    alignas(16) uint16_t kx[32 * D], vx[32 * D];
  };
  union {
    ClosedStorage c;  // closed half group
    ExactStorage x;   // a BF16 quarter group (stage 1, before the stream)
  };
  alignas(16) uint16_t meta[512];
};
template <int PROWS> struct StreamSmem {
  Stage st[NST];
  alignas(16) uint32_t qhat[PROWS][32];  // q-hat of the current group, per row
  alignas(16) uint32_t p[PROWS][16];     // P of the stage: token word u of a row = tokens 4 u .. + 3
  alignas(16) float2 wsp[PROWS][4];      // per row and sub-block of the stage: (2^(m_j - m), sum P_j)
};

__device__ __forceinline__ int kd_at(int tok, int w) { return tok * 32 + ((((w >> 2) ^ (tok >> 2)) & 7) << 2) + (w & 3); }
__device__ __forceinline__ int vd_at(int u, int ch) { const int q = ch >> 2; return u * 128 + ((q ^ ((q >> 3) & 1)) << 2) + (ch & 3); }

// a record's K / V slot word as the codes' values (the slot's 2^n factor shifted out)
template <int BITS> __device__ __forceinline__ uint32_t k_value(const Codes<BITS> &c, int s) {
  return k_slot<BITS>(c, s) >> (k_scaled<BITS>(s) ? (BITS == 4 ? 4 : 3) : 0);
}
template <int BITS> __device__ __forceinline__ uint32_t v_value(const Codes<BITS> &c, int s) {
  const int ch = v_channel<BITS>(s);
  const int n = BITS == 4 ? (ch < 2 ? 0 : 4) : BITS == 2 ? 2 * ch : (ch == 1 ? 3 : 0);
  return v_slot<BITS>(c, s) >> n;
}

// Producer: record lanes of a half group (of unit u: the group's place in the head's planes and metadata).  Lane-record
// index lr: 0 .. 127 the K lanes (sub-block lr / 32, lane lr % 32), 128 .. 255 the V lanes; producer lane pt owns
// lr = pt + 32 k (the K, then the V lanes it serves).
template <int CB, int FB> struct HalfRaw { Codes<CB> w[Warps<CB>::NLR]; Codes<FB> f[Warps<CB>::NLR]; uint4 meta[2]; };
template <int CB, int FB, class Pos = Grid>
__device__ __forceinline__ void half_load(HalfRaw<CB, FB> &h, const uint8_t *coarse_h, const uint8_t *fine_h,
                                          const uint16_t *meta_h, size_t u, int half, int pt) {
#pragma unroll
  for (int k = 0; k < Warps<CB>::NLR; ++k) {
    const int lr = pt + 32 * Warps<CB>::NPW * k, j4 = (lr & 127) >> 5, l = lr & 31;
    const size_t sb = u * SBG + 4 * half + j4;
    const uint8_t *rc = coarse_h + sb * record_bytes<CB>(), *rf = fine_h + sb * record_bytes<FB>();
    if (lr < 128) { load_k<CB>(h.w[k], rc + 16 * l, l); load_k<FB>(h.f[k], rf + 16 * l, l); }
    else { load_v<CB>(h.w[k], rec_at(rc, l)); load_v<FB>(h.f[k], rec_at(rf, l)); }
  }
#pragma unroll
  for (int k = 0; k < 2 / Warps<CB>::NPW; ++k)  // 64 words
    h.meta[k] = Pos::ro(reinterpret_cast<const uint4 *>(meta_h + u * 512) + pt + 32 * Warps<CB>::NPW * k);
}
// the fine plane's V slot holding the same (channel, token quad) as coarse slot s
template <int CB, int FB> __host__ __device__ constexpr int fine_v_slot(int s) {
  if (CB == FB) return s;
  int f = 0;
  while (v_channel<FB>(f) != v_channel<CB>(s) || v_quad<FB>(f) != v_quad<CB>(s)) ++f;
  return f;
}
// the codes c 2^FB + f of a slot: bytes < 64, no carries between them
template <int CB, int FB> __device__ __forceinline__ uint32_t k_code(const Codes<CB> &c, const Codes<FB> &f, int s) {
  return k_value<CB>(c, s) << FB | k_value<FB>(f, s);
}
template <int CB, int FB> __device__ __forceinline__ uint32_t v_code(const Codes<CB> &c, const Codes<FB> &f, int s) {
  return v_value<CB>(c, s) << FB | v_value<FB>(f, fine_v_slot<CB, FB>(s));
}
template <int CB, int FB> __device__ __forceinline__ void half_store(Stage &st, const HalfRaw<CB, FB> &h, int pt) {
#pragma unroll
  for (int k = 0; k < Warps<CB>::NLR; ++k) {
    const int lr = pt + 32 * Warps<CB>::NPW * k, j4 = (lr & 127) >> 5, l = lr & 31;
    if (lr < 128) {  // K lane l: token 16 j4 + l % 16, channels 64 (l / 16) .. + 63 = words 16 (l / 16) ..
      const int tok = 16 * j4 + (l & 15), hh = l >> 4;
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4)
        *reinterpret_cast<uint4 *>(st.c.kd + kd_at(tok, 16 * hh + 4 * q4)) =
            make_uint4(k_code(h.w[k], h.f[k], 4 * q4), k_code(h.w[k], h.f[k], 4 * q4 + 1),
                       k_code(h.w[k], h.f[k], 4 * q4 + 2), k_code(h.w[k], h.f[k], 4 * q4 + 3));
    } else {  // V lane l: channels 4 l .. + 3 x the sub-block's tokens
#pragma unroll
      for (int sl = 0; sl < 16; ++sl) st.c.vd[vd_at(4 * j4 + v_quad<CB>(sl), 4 * l + v_channel<CB>(sl))] = v_code(h.w[k], h.f[k], sl);
    }
  }
#pragma unroll
  for (int k = 0; k < 2 / Warps<CB>::NPW; ++k) reinterpret_cast<uint4 *>(st.meta)[pt + 32 * Warps<CB>::NPW * k] = h.meta[k];
}

// q-hat of a group for the warp's 8 rows (exactly setup_group's codes): lanes (row 8 warp + lane / 4, channels
// 32 (lane % 4) .. + 31).  Returns the lane's row's (sq, sl q . base_K): score = sq (q-hat . codes) + sl q . base_K.
// qv4: the lane's 32 channels of its row's q (BF16 pairs), loaded once per CTA
__device__ __forceinline__ float2 quantize_warp(const PrefillArgs &a, const uint16_t *meta, uint32_t (*qhat)[32], int warp,
                                                int lane, const uint4 (&qv4)[4]) {
  const int row = 8 * warp + (lane >> 2), qt = lane & 3;
  float z[32], amax = 0.f, qb = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {  // 8 channels per word quad
    const uint4 qv = qv4[i];
    float q8[8];
    bf16x4(qv.x, qv.y, q8); bf16x4(qv.z, qv.w, q8 + 4);
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const uint4 km = *reinterpret_cast<const uint4 *>(meta + 8 * (8 * qt + 2 * i + hh));  // 4 channels
      float kb[4], ks[4];
      bf16x4(km.x, km.y, kb); bf16x4(km.z, km.w, ks);
#pragma unroll
      for (int x = 0; x < 4; ++x) {  // z = q step_K, exact (mlops_fbla.h)
        const float qf = q8[4 * hh + x];
        z[8 * i + 4 * hh + x] = qf * ks[x];
        amax = fmaxf(amax, fabsf(z[8 * i + 4 * hh + x]));
        qb = fmaf(qf, kb[x], qb);
      }
    }
  }
#pragma unroll
  for (int o = 1; o < 4; o <<= 1) { amax = fmaxf(amax, __shfl_xor_sync(FULL, amax, o)); qb += __shfl_xor_sync(FULL, qb, o); }
  float rc;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rc) : "f"(amax));
  rc = fmaf(fmaf(-amax, rc, 1.f), rc, rc);
  const float q0 = 127.f * rc, rq = amax > 0.f ? fmaf(fmaf(-amax, q0, 127.f), rc, q0) : 0.f;
  const float sq = amax * (a.scale_log2 * (1.f / 127.f));
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    uint32_t bq[4];
#pragma unroll
    for (int x = 0; x < 4; ++x) bq[x] = __float_as_uint(fmaf(z[4 * w + x], rq, MAGICF));  // low byte = rint(z r), one rounding
    const uint32_t wq = prmt(prmt(bq[0], bq[1], 0x0040u), prmt(bq[2], bq[3], 0x0040u), 0x5410u);
    qhat[row][8 * qt + w] = wq;
  }
  __syncwarp();
  return make_float2(sq, qb * a.scale_log2);
}

// 4 rows x 4 tokens: q-hat . code values over the 32 channel words of a Kd tile.  The fragments of word quad
// q + 1 are loaded before the DP4A of quad q (registers double-buffered), so the LDS latency is covered.
__device__ __forceinline__ void qk_tile(const uint32_t (*qhat)[32], const uint32_t *kd, int rr, int b, float (&x)[4][4]) {
  uint32_t acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int t = 0; t < 4; ++t) acc[i][t] = MAGIC;
  const uint32_t *kt[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) kt[t] = kd + (4 * b + t) * 32;  // token row; quad q at (q ^ (token / 4 % 8)) * 4
  const int sw = b & 7;  // (4 b + t) / 4 % 8 for every t
  uint4 qa[2][4], kb[2][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) qa[0][i] = *reinterpret_cast<const uint4 *>(&qhat[rr + i][0]);
#pragma unroll
  for (int t = 0; t < 4; ++t) kb[0][t] = *reinterpret_cast<const uint4 *>(kt[t] + 4 * sw);
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int c = q & 1, n = c ^ 1;
    if (q + 1 < 8) {
#pragma unroll
      for (int i = 0; i < 4; ++i) qa[n][i] = *reinterpret_cast<const uint4 *>(&qhat[rr + i][4 * (q + 1)]);
#pragma unroll
      for (int t = 0; t < 4; ++t) kb[n][t] = *reinterpret_cast<const uint4 *>(kt[t] + 4 * ((q + 1) ^ sw));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        acc[i][t] = dp4a_su(qa[c][i].x, kb[c][t].x, acc[i][t]); acc[i][t] = dp4a_su(qa[c][i].y, kb[c][t].y, acc[i][t]);
        acc[i][t] = dp4a_su(qa[c][i].z, kb[c][t].z, acc[i][t]); acc[i][t] = dp4a_su(qa[c][i].w, kb[c][t].w, acc[i][t]);
      }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int t = 0; t < 4; ++t) x[i][t] = isum(acc[i][t]);
}

// P word of four tokens (byte t = round(255 2^(s_t - mb))) and its sum
__device__ __forceinline__ uint32_t p_quad(const float (&s)[4], float mb, float &sum) {
  uint32_t bq[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) bq[t] = __float_as_uint(fmaf(ex2(s[t] - mb), 255.f, MAGICF));
  const uint32_t w = prmt(prmt(bq[0], bq[1], 0x0040u), prmt(bq[2], bq[3], 0x0040u), 0x5410u);
  sum = isum(dp4a_uu(w, 0x01010101u, MAGIC));
  return w;
}

__device__ __forceinline__ size_t qrow_of(const PrefillArgs &a, int bh, int r) {  // row -> its q / out index
  const int bb = bh / a.kv_heads, kh = bh - bb * a.kv_heads, i = r / a.qg;
  return (size_t)(bb * a.q_heads + kh * a.qg + (r - i * a.qg)) * a.count + i;
}

// The attention (see above).
// The item of stream_kernel: Pos::x()'s block of PROWS query rows of head Pos::y() (a CTA past them returns)
template <int CB, int FB, class Pos = Grid>
__device__ __forceinline__ void stream_item(const PrefillArgs &a, StreamSmem<Warps<CB>::PROWS> &sm) {
  constexpr int NCW = Warps<CB>::NCW, PROWS = Warps<CB>::PROWS, CTH = Warps<CB>::CTH;
  constexpr int RC = record_bytes<CB>();
  const int t = Pos::tid(), lane = t & 31, warp = t >> 5;
  const int n0 = n0_of<Pos>(a), count = count_of<Pos>(a);
  const int bh = Pos::y(), rows = count * a.qg, r0 = Pos::x() * PROWS, n1 = n0 + count;
  if (r0 >= rows) return;  // (CTA-uniform)
  const int p_first = n0 + r0 / a.qg, p_last = n0 + (min(r0 + PROWS, rows) - 1) / a.qg;
  const int g_lo = p_first / G, g_hi = p_last / G, nhalf = 2 * max(0, g_hi - 1);
  const size_t head = (size_t)bh * a.hs;
  const uint8_t *coarse_h = a.c.coarse + head * SBG * RC, *fine_h = a.c.fine + head * SBG * record_bytes<FB>();
  const uint16_t *meta_h = a.c.meta + head * 512;
  auto group_of = [&](int it) { return g_hi - 1 - (it >> 1); };  // newest closed group first

  if (warp >= NCW) {  // ---- producer: the closed groups' half groups into the ring -------------------------
    const int pt = t - CTH;
    HalfRaw<CB, FB> raw;
    if (nhalf) half_load<CB, FB, Pos>(raw, coarse_h, fine_h, meta_h, (size_t)group_of(0) * a.gs, 0, pt);
    for (int it = 0; it < nhalf; ++it) {
      if (it == 1 || it >= NST) bar_sync(BAR_EMPTY + it % NST, PTH);  // stage 1 first holds the BF16 groups
      half_store(sm.st[it % NST], raw, pt);
      bar_arrive(BAR_FULL + it % NST, PTH);
      if (it + 1 < nhalf) half_load<CB, FB, Pos>(raw, coarse_h, fine_h, meta_h, (size_t)group_of(it + 1) * a.gs, (it + 1) & 1, pt);
    }
    return;
  }

  // ---- consumers ---------------------------------------------------------------------------------------
  const int rr = 8 * warp + 4 * (lane >> 4), b = lane & 15;  // rows rr .. rr + 3; token / channel block b
  bool valid[4];
  int pos[4], own[4];
  float m[4], l[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    valid[i] = r0 + rr + i < rows;
    pos[i] = n0 + (valid[i] ? (r0 + rr + i) / a.qg : 0);
    own[i] = pos[i] / G;
    m[i] = -INFINITY; l[i] = 0.f;
  }
  // o, and the current group's A = sum w y (codes) and B = sum w sum P
  float o[4][8], A[4][8], Bg[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    Bg[i] = 0.f;
#pragma unroll
    for (int c = 0; c < 8; ++c) o[i][c] = A[i][c] = 0.f;
  }
  // FA update: block maxima mb, sums sp.  Row m moves only when a block exceeds it by more than 8 (log2, one
  // ballot), so w = 2^(mb - m) <= 2^8 and the rescale is rare.  l is per lane, reduced over 16 lanes in row_l.
  auto fa_step = [&](const float (&mb)[4], const float (&sp)[4], float (&w)[4], int o0) {
    const uint32_t half_mask = 0xffffu << (lane & 16);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t up = __ballot_sync(FULL, mb[i] > m[i] + 8.f);
      if (up) {  // warp-uniform, so the whole warp shuffles (sm_70+ schedules its halves independently)
        float mx = mb[i];
#pragma unroll
        for (int o2 = 1; o2 < 16; o2 <<= 1) mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, o2));  // within the half
        if (up & half_mask) {  // uniform over the row's lanes
          if (m[i] != -INFINITY) {
            const float al = ex2(m[i] - mx);
            l[i] *= al; Bg[i] *= al;
#pragma unroll
            for (int c = 0; c < 8; ++c) { o[i][c] *= al; A[i][c] *= al; }
          }
          m[i] = mx;
        }
      }
      w[i] = mb[i] == -INFINITY ? 0.f : ex2(mb[i] - m[i]);
      if (!(b & (o0 - 1))) l[i] = fmaf(w[i], sp[i], l[i]);  // the block's leading lane
    }
  };
  auto row_l = [&](int i) {  // the row's l (every lane)
    float t2 = l[i];
#pragma unroll
    for (int o2 = 1; o2 < 16; o2 <<= 1) t2 += __shfl_xor_sync(FULL, t2, o2);
    return t2;
  };

  // ---- the BF16 groups: the sink (group 0, for rows whose own group is later) and the rows' own groups, by
  // quarter groups (32 tokens, 2 sub-blocks) in stage 1's memory; QK lanes take tokens 2 b, 2 b + 1 -----------
  {
    const size_t eh = (size_t)bh * G * D, ih = (size_t)bh * a.count * D;
    const uint16_t *sink_k = a.c.exact_sink_k + eh, *sink_v = a.c.exact_sink_v + eh;
    const uint16_t *open_k = a.c.exact_open_k + eh, *open_v = a.c.exact_open_v + eh;
    const uint16_t *in_k = a.k + ih, *in_v = a.v + ih;
    uint16_t *kx = sm.st[1].x.kx, *vx = sm.st[1].x.vx;
    const uint4 *qrow[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) qrow[i] = reinterpret_cast<const uint4 *>(a.q + qrow_of(a, bh, valid[i] ? r0 + rr + i : r0) * D);
    for (int gx = 0; gx <= g_hi; ++gx) {
      if (gx > 0 && gx < g_lo) continue;  // neither the sink nor any row's own group
      for (int qtr = 0; qtr < 4; ++qtr) {
        const int t0 = gx * G + 32 * qtr;
        if (t0 >= n1 || t0 > p_last) break;
        bar_sync(BAR_CONS, CTH);
        for (int e = t; e < 32 * 16 * 2; e += CTH) {  // K, V rows of the quarter; K quads swizzled by token
          const int x = e >> 9, tk = (e >> 4) & 31, qd = e & 15, pt = min(t0 + tk, n1 - 1);
          const uint4 v = Pos::ro(reinterpret_cast<const uint4 *>(token_row(x ? sink_v : sink_k, x ? open_v : open_k,
                                                                           x ? in_v : in_k, n0, pt)) + qd);
          if (x) reinterpret_cast<uint4 *>(vx)[tk * 16 + qd] = v;
          else reinterpret_cast<uint4 *>(kx)[tk * 16 + (qd ^ (tk & 15))] = v;
        }
        bar_sync(BAR_CONS, CTH);
        bool inc[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) inc[i] = valid[i] && (gx == own[i] || (gx == 0 && own[i] > 0)) && t0 <= pos[i];
        if (!__any_sync(FULL, inc[0] || inc[1] || inc[2] || inc[3])) continue;
        float sc[4][2];
#pragma unroll
        for (int i = 0; i < 4; ++i) sc[i][0] = sc[i][1] = 0.f;
#pragma unroll 2
        for (int e = 0; e < 16; ++e) {  // 8 channels per step
          float k8[2][8];
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            const int tk = 2 * b + k;
            const uint4 kk = reinterpret_cast<const uint4 *>(kx)[tk * 16 + (e ^ (tk & 15))];
            bf16x4(kk.x, kk.y, k8[k]); bf16x4(kk.z, kk.w, k8[k] + 4);
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint4 qq = Pos::ro(qrow[i] + e);
            float q8[8];
            bf16x4(qq.x, qq.y, q8); bf16x4(qq.z, qq.w, q8 + 4);
#pragma unroll
            for (int x = 0; x < 8; ++x) {
              const float qs = q8[x] * a.scale_log2;
              sc[i][0] = fmaf(qs, k8[0][x], sc[i][0]); sc[i][1] = fmaf(qs, k8[1][x], sc[i][1]);
            }
          }
        }
        float mb[4], sp[4], w[4];
        uint32_t pw[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            const int pt = t0 + 2 * b + k;
            if (!(inc[i] && pt <= pos[i] && pt < n1)) sc[i][k] = -INFINITY;
          }
          mb[i] = fmaxf(sc[i][0], sc[i][1]);
        }
#pragma unroll
        for (int o2 = 1; o2 < 8; o2 <<= 1)
#pragma unroll
          for (int i = 0; i < 4; ++i) mb[i] = fmaxf(mb[i], __shfl_xor_sync(FULL, mb[i], o2));
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float mz = mb[i] == -INFINITY ? 0.f : mb[i];
          const uint32_t b0 = __float_as_uint(fmaf(ex2(sc[i][0] - mz), 255.f, MAGICF));
          const uint32_t b1 = __float_as_uint(fmaf(ex2(sc[i][1] - mz), 255.f, MAGICF));
          pw[i] = prmt(b0, b1, 0x0040u) & 0xffffu;
          sp[i] = isum(dp4a_uu(pw[i], 0x01010101u, MAGIC));
        }
#pragma unroll
        for (int o2 = 1; o2 < 8; o2 <<= 1)
#pragma unroll
          for (int i = 0; i < 4; ++i) sp[i] += __shfl_xor_sync(FULL, sp[i], o2);
        fa_step(mb, sp, w, 8);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          asm volatile("st.shared.u16 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&sm.p[rr + i][0]) + 2 * b),
                       "h"((uint16_t)pw[i]) : "memory");
          if (!(b & 7)) sm.wsp[rr + i][b >> 3] = make_float2(w[i], sp[i]);
        }
        __syncwarp();
#pragma unroll 1
        for (int j = 0; j < 2; ++j) {
          float y[4][8];
#pragma unroll
          for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int c = 0; c < 8; ++c) y[i][c] = 0.f;
#pragma unroll 4
          for (int tk = 16 * j; tk < 16 * j + 16; ++tk) {
            const uint4 vv = reinterpret_cast<const uint4 *>(vx)[tk * 16 + b];
            float v8[8];
            bf16x4(vv.x, vv.y, v8); bf16x4(vv.z, vv.w, v8 + 4);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const float p = __uint_as_float(MAGIC | ((sm.p[rr + i][tk >> 2] >> (8 * (tk & 3))) & 255u)) - MAGICF;
#pragma unroll
              for (int c = 0; c < 8; ++c) y[i][c] = fmaf(p, v8[c], y[i][c]);
            }
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float wj = sm.wsp[rr + i][j].x;
#pragma unroll
            for (int c = 0; c < 8; ++c) o[i][c] = fmaf(wj, y[i][c], o[i][c]);
          }
        }
        __syncwarp();
      }
    }
  }
  if (nhalf) bar_arrive(BAR_EMPTY + 1, PTH);  // stage 1 is the producer's

  // ---- the closed groups, streamed newest first -------------------------------------------------------
  uint4 qv4[4];  // the quantizing lane's 32 channels of row 8 warp + lane / 4
  {
    const int qrow_q = 8 * warp + (lane >> 2);
    const uint4 *qp = reinterpret_cast<const uint4 *>(a.q + qrow_of(a, bh, min(r0 + qrow_q, rows - 1)) * D) + 4 * (lane & 3);
#pragma unroll
    for (int i = 0; i < 4; ++i) qv4[i] = r0 + qrow_q < rows ? Pos::ro(qp + i) : make_uint4(0u, 0u, 0u, 0u);
  }
  float2 rcv[4];  // per row (sq, q . base_K) of the current group
  for (int it = 0; it < nhalf; ++it) {
    const int g = group_of(it), half = it & 1;
    const Stage &st = sm.st[it % NST];
    bar_sync(BAR_FULL + it % NST, PTH);
    bool inc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) inc[i] = valid[i] && g < own[i];
    if (__any_sync(FULL, inc[0] || inc[1] || inc[2] || inc[3])) {
      if (!half) {
        const float2 rq = quantize_warp(a, st.meta, sm.qhat, warp, lane, qv4);
#pragma unroll
        for (int i = 0; i < 4; ++i) {  // row rr + i was quantized by lane 4 (rr + i - 8 warp)
          const int src = 4 * (rr + i - 8 * warp);
          rcv[i] = make_float2(__shfl_sync(FULL, rq.x, src), __shfl_sync(FULL, rq.y, src));
        }
      }
      float x[4][4];
      qk_tile(sm.qhat, st.c.kd, rr, b, x);
      // softmax of the stage, every step over the 4 rows at once (independent chains)
      float sc[4][4], mb[4], sp[4], w[4];
      uint32_t pw[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int k = 0; k < 4; ++k) sc[i][k] = inc[i] ? fmaf(rcv[i].x, x[i][k], rcv[i].y) : -INFINITY;
        mb[i] = fmaxf(fmaxf(sc[i][0], sc[i][1]), fmaxf(sc[i][2], sc[i][3]));
      }
#pragma unroll
      for (int o2 = 1; o2 < 4; o2 <<= 1)
#pragma unroll
        for (int i = 0; i < 4; ++i) mb[i] = fmaxf(mb[i], __shfl_xor_sync(FULL, mb[i], o2));
#pragma unroll
      for (int i = 0; i < 4; ++i) pw[i] = p_quad(sc[i], mb[i] == -INFINITY ? 0.f : mb[i], sp[i]);
#pragma unroll
      for (int o2 = 1; o2 < 4; o2 <<= 1)
#pragma unroll
        for (int i = 0; i < 4; ++i) sp[i] += __shfl_xor_sync(FULL, sp[i], o2);
      fa_step(mb, sp, w, 4);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        sm.p[rr + i][b] = pw[i];
        if (!(b & 3)) sm.wsp[rr + i][b >> 2] = make_float2(w[i], sp[i]);
      }
      __syncwarp();
      // P V of the stage's 4 sub-blocks into the group's A and B
#pragma unroll 1
      for (int j = 0; j < 4; ++j) {
        uint32_t acc[4][8];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int c = 0; c < 8; ++c) acc[i][c] = MAGIC;
        uint4 pq[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) pq[i] = *reinterpret_cast<const uint4 *>(&sm.p[rr + i][4 * j]);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const uint4 v0 = *reinterpret_cast<const uint4 *>(st.c.vd + vd_at(4 * j + u, 8 * b));
          const uint4 v1 = *reinterpret_cast<const uint4 *>(st.c.vd + vd_at(4 * j + u, 8 * b + 4));
          const uint32_t vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint32_t pwd = u == 0 ? pq[i].x : u == 1 ? pq[i].y : u == 2 ? pq[i].z : pq[i].w;
#pragma unroll
            for (int c = 0; c < 8; ++c) acc[i][c] = dp4a_uu(pwd, vv[c], acc[i][c]);
          }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 ws = sm.wsp[rr + i][j];
          Bg[i] = fmaf(ws.x, ws.y, Bg[i]);
#pragma unroll
          for (int c = 0; c < 8; ++c) A[i][c] = fmaf(ws.x, isum(acc[i][c]), A[i][c]);
        }
      }
      if (half) {  // group end: o += vs A + vb B
        const uint4 m0 = *reinterpret_cast<const uint4 *>(st.meta + 256 + 8 * (2 * b));
        const uint4 m1 = *reinterpret_cast<const uint4 *>(st.meta + 256 + 8 * (2 * b + 1));
        float vb[8], vs[8];
        bf16x4(m0.x, m0.y, vb); bf16x4(m0.z, m0.w, vs); bf16x4(m1.x, m1.y, vb + 4); bf16x4(m1.z, m1.w, vs + 4);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
          for (int c = 0; c < 8; ++c) {
            o[i][c] = fmaf(vs[c], A[i][c], fmaf(vb[c], Bg[i], o[i][c]));
            A[i][c] = 0.f;
          }
          Bg[i] = 0.f;
        }
      }
    }  // (a warp takes both halves of a group or neither: A and B are always closed by the group's end)
    if (it < nhalf - NST) bar_arrive(BAR_EMPTY + it % NST, PTH);
  }

  float lt[4];  // every lane reduces every row (row_l shuffles over the whole warp), before the validity test
#pragma unroll
  for (int i = 0; i < 4; ++i) lt[i] = row_l(i);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (!valid[i]) continue;
    const size_t qr = qrow_of(a, bh, r0 + rr + i);
    const float inv = 1.f / lt[i];
    uint32_t w[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) w[c] = to_bf16(o[i][2 * c] * inv) | (uint32_t)to_bf16(o[i][2 * c + 1] * inv) << 16;
    reinterpret_cast<uint4 *>(a.out + qr * D)[b] = make_uint4(w[0], w[1], w[2], w[3]);
  }
}
template <int CB, int FB>
__global__ void __launch_bounds__(PTH) stream_kernel(const PrefillArgs a) {
  __shared__ StreamSmem<Warps<CB>::PROWS> sm;
  stream_item<CB, FB, Grid>(a, sm);
}

}  // namespace mlops_fbla_detail
#include "tensors/prefill_tc.cuh"
namespace mlops_fbla_detail {

// Tensor-core attention, format-independent (MLOPS_FBLA_DEFINE_SHARED): GQA templates 1, 2, 4, 8 (else the
// ratio at run time), split-K below a wave, the merge.  Not stream-K: it spreads an image's reads over the call.
// tiles: group g of KV head bh at tiles + bh hstep + (g - toff) gstep (bytes)
#ifndef __CUDACC_RTC__
// Host-only launch state. Both the stream API and item plans use the same split decision and byte layout.
struct PrefillGrid {
  int attention_blocks, close_blocks;
  static PrefillGrid from_ceiling(int count) {
    const int attention = (count + G - 1) / G + 1;
    return {attention, attention + 1};
  }
};

struct TcSplitLayout {
  size_t ml_bytes, output_bytes, output_offset, bytes;
  static TcSplitLayout make(int heads, int count, int splits) {
    const size_t rows = (size_t)heads * count * splits;
    const size_t ml = rows * sizeof(float2), output = rows * D * sizeof(float);
    const size_t offset = (ml + 255) & ~size_t(255);
    return {ml, output, offset, offset + output};
  }
};

inline int tc_split_count(int blocks, int heads, int groups, int sms) {
  const long long base = (long long)blocks * heads;
  return base < sms ? std::max(1, std::min(std::min((int)(sms / base), 8), groups / 16)) : 1;
}

// Stream-ordered ownership also covers an early return after a failed launch or second allocation.
class StreamWorkspace {
 public:
  explicit StreamWorkspace(cudaStream_t stream) : stream_(stream) {}
  StreamWorkspace(const StreamWorkspace &) = delete;
  StreamWorkspace &operator=(const StreamWorkspace &) = delete;
  ~StreamWorkspace() { if (data_) cudaFreeAsync(data_, stream_); }
  cudaError_t allocate(size_t bytes) { return cudaMallocAsync(&data_, bytes, stream_); }
  template <class T> T *get(size_t offset = 0) const {
    return data_ ? reinterpret_cast<T *>(static_cast<uint8_t *>(data_) + offset) : nullptr;
  }
  cudaError_t finish(cudaError_t status) {
    const cudaError_t freed = data_ ? cudaFreeAsync(data_, stream_) : cudaSuccess;
    data_ = nullptr;
    return status != cudaSuccess ? status : freed;
  }
 private:
  cudaStream_t stream_;
  void *data_ = nullptr;
};

cudaError_t launch_tc_attention(const PrefillArgs &a, const uint8_t *tiles, size_t hstep, size_t gstep, int toff, int groups,
                                int nblk, cudaStream_t stream);
#ifdef MLOPS_FBLA_DEFINE_SHARED
// The device's dynamic shared memory for the TC kernel (0: DP4A).  The specialization is the compiled code's PTX version
// (a compute_75 build JIT-compiled on a newer card runs the sm_75 path); sm_75 .. sm_89 with tc::smem_bytes available.
// MLOPS_FBLA_DP4A=1 forces DP4A.  Cached per device.
static int prefill_tc_smem() {
  static thread_local int cache[16];
  static const bool dp4a = getenv("MLOPS_FBLA_DP4A") && getenv("MLOPS_FBLA_DP4A")[0] == '1';
  int dev = 0;
  if (dp4a || cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return 0;
  if (!cache[dev]) {
    int major = 0, minor = 0, optin = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa{};
    const int cc = 10 * major + minor;
    const bool ok = cc >= 75 && cc <= 89 && cudaFuncGetAttributes(&fa, tc::prefill_tc_kernel<1>) == cudaSuccess && fa.ptxVersion >= 75 &&
                    optin >= tc::smem_bytes(fa.ptxVersion);
    cache[dev] = ok ? tc::smem_bytes(fa.ptxVersion) : -1;
  }
  return max(cache[dev], 0);
}
bool prefill_tc_device() { return prefill_tc_smem() > 0; }

template <int GQA> static cudaError_t tc_attention(const PrefillArgs &a, const uint8_t *tiles, size_t hstep, size_t gstep, int toff, int groups,
                                                   int nblk, int dev, int sms, cudaStream_t stream) {
  static thread_local bool ready[16];  // above 48 KiB: configure once per device and calling thread
  const int smem = prefill_tc_smem();
  if (!ready[dev]) {
    const cudaError_t e = cudaFuncSetAttribute(tc::prefill_tc_kernel<GQA>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) return e;
    ready[dev] = true;
  }
  const int heads = a.batch * a.q_heads;
  const int nsplit = tc_split_count(nblk, heads, groups, sms);
  const TcSplitLayout layout = TcSplitLayout::make(heads, a.count, nsplit);
  StreamWorkspace workspace(stream);
  if (nsplit > 1) {
    const cudaError_t e = workspace.allocate(layout.bytes);
    if (e != cudaSuccess) return e;
  }
  float2 *part_ml = workspace.get<float2>();
  float *part_o = workspace.get<float>(layout.output_offset);
  tc::prefill_tc_kernel<GQA><<<dim3(nblk, heads, nsplit), tc::THREADS, smem, stream>>>(a, tiles, hstep, gstep, toff, part_ml, part_o);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return workspace.finish(status);
  if (nsplit > 1) {
    tc::merge_kernel<><<<dim3((a.count + 15) / 16, heads), 256, 0, stream>>>(a, part_ml, part_o, nsplit);
    status = cudaGetLastError();
  }
  return workspace.finish(status);
}
cudaError_t launch_tc_attention(const PrefillArgs &a, const uint8_t *tiles, size_t hstep, size_t gstep, int toff, int groups,
                                int nblk, cudaStream_t stream) {
  static thread_local int sms[16];
  int dev = 0;
  cudaGetDevice(&dev);  // (prefill_tc_device: dev < 16)
  if (!sms[dev]) cudaDeviceGetAttribute(&sms[dev], cudaDevAttrMultiProcessorCount, dev);
  switch (a.qg) {
    case 1: return tc_attention<1>(a, tiles, hstep, gstep, toff, groups, nblk, dev, sms[dev], stream);
    case 2: return tc_attention<2>(a, tiles, hstep, gstep, toff, groups, nblk, dev, sms[dev], stream);
    case 4: return tc_attention<4>(a, tiles, hstep, gstep, toff, groups, nblk, dev, sms[dev], stream);
    case 8: return tc_attention<8>(a, tiles, hstep, gstep, toff, groups, nblk, dev, sms[dev], stream);
    default: return tc_attention<0>(a, tiles, hstep, gstep, toff, groups, nblk, dev, sms[dev], stream);
  }
}
#endif

// Images of the closed groups 1 .. last position's group - 1, then the attention.  With tc_images only the
// groups close_kernel invalidated are transcoded; otherwise all, into a per-call allocation.
// nblk and groups derive from the chunk's tokens, or from the ceiling a.count for launch_prefill_at.
template <int CB, int FB> cudaError_t launch_prefill_tc(const PrefillArgs &a, int groups, int nblk, cudaStream_t stream) {
  const int heads = a.batch * a.kv_heads;
  if (a.c.tc_images) {
    const size_t hstep = (size_t)a.hs * IMAGE_SLOT, gstep = (size_t)a.gs * IMAGE_SLOT;
    if (groups) tc::transcode_kernel<CB, FB><<<dim3(groups, heads), tc::XT, 0, stream>>>(a, a.c.tc_images, hstep, gstep, 0, true);
    return launch_tc_attention(a, a.c.tc_images, hstep, gstep, 0, groups, nblk, stream);
  }
  static thread_local bool ready[16];
  int dev = 0;
  cudaGetDevice(&dev);
  if (!ready[dev]) {
    cudaMemPool_t pool;  // (by default its memory goes back at every sync)
    cudaError_t e = cudaDeviceGetDefaultMemPool(&pool, dev);
    uint64_t keep = ~0ull;
    if (e == cudaSuccess) e = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep);
    if (e != cudaSuccess) return e;
    ready[dev] = true;
  }
  StreamWorkspace workspace(stream);
  if (groups) {
    const cudaError_t e = workspace.allocate((size_t)heads * groups * tc::TGB);
    if (e != cudaSuccess) return e;
  }
  uint8_t *tiles = workspace.get<uint8_t>();
  if (groups) {
    tc::transcode_kernel<CB, FB><<<dim3(groups, heads), tc::XT, 0, stream>>>(a, tiles, (size_t)groups * tc::TGB, tc::TGB, 1, false);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return workspace.finish(e);
  }
  const cudaError_t e = launch_tc_attention(a, tiles, (size_t)groups * tc::TGB, tc::TGB, 1, groups, nblk, stream);
  return workspace.finish(e);
}

// Groups closed by the chunk: those with 128 (g + 1) < n0 + count that were open before (the group of
// position n0 - 1 onwards); a full open group closes when the next token arrives.
template <int CB, int FB> cudaError_t launch_prefill(const PrefillArgs &a, cudaStream_t stream) {
  const int heads = a.batch * a.kv_heads, n1 = a.n0 + a.count;
  const int g_first = a.n0 == 0 ? 0 : (a.n0 - 1) / G, g_end = (n1 - 1) / G;
  if (g_end > g_first) close_kernel<CB, FB><<<dim3(g_end - g_first, heads), 128, 0, stream>>>(a);
  if (a.q) {
    const int groups = max(0, g_end - 1), nblk = g_end - a.n0 / G + 1;
    if (prefill_tc_device()) {
      const cudaError_t e = launch_prefill_tc<CB, FB>(a, groups, nblk, stream);
      if (e != cudaSuccess) return e;
    } else {
      constexpr int PROWS = Warps<CB>::PROWS;
      const dim3 grid((a.count * a.qg + PROWS - 1) / PROWS, heads);
      stream_kernel<CB, FB><<<grid, PTH, 0, stream>>>(a);
    }
  }
  launch_finish(a, heads, stream);
  return cudaGetLastError();
}

// mlops_fbla_prefill_at: every grid comes from the ceiling a.count alone, so launches are capturable ahead of
// any start and count; kernels exit on the device length and count (worst grid: max_count / G + 2 blocks).
// The tensor-core attention needs the cache's tc_images.
template <int CB, int FB> cudaError_t launch_prefill_at(const PrefillArgs &a, cudaStream_t stream) {
  const int heads = a.batch * a.kv_heads;
  const PrefillGrid grid = PrefillGrid::from_ceiling(a.count);
  close_kernel<CB, FB><<<dim3(grid.close_blocks, heads), 128, 0, stream>>>(a);
  if (a.q) {
    if (prefill_tc_device()) {
      const cudaError_t e = launch_prefill_tc<CB, FB>(a, grid.close_blocks, grid.attention_blocks, stream);
      if (e != cudaSuccess) return e;
    } else {
      constexpr int PROWS = Warps<CB>::PROWS;
      stream_kernel<CB, FB><<<dim3((a.count * a.qg + PROWS - 1) / PROWS, heads), PTH, 0, stream>>>(a);
    }
  }
  launch_finish(a, heads, stream);
  return cudaGetLastError();
}

// One token per KV head at the device length: the group it closes, if any (one CTA per head, which returns when
// none), then the BF16 windows.
template <int CB, int FB> cudaError_t launch_append(const PrefillArgs &a, cudaStream_t stream) {
  const int heads = a.batch * a.kv_heads;
  close_kernel<CB, FB><<<dim3(1, heads), 128, 0, stream>>>(a);
  launch_finish(a, heads, stream);
  return cudaGetLastError();
}

// ---- the work items (mlops_fbla_items.h) -----------------------------------------------------------------------------
// Host descriptors of the format-independent passes: finish, and the tensor-core attention (GQA instance by
// a.qg) with its merge.  tc: split partials belong to the executor (geom[8], geom[9], geom[10] bytes each).
bool finish_item_of(int heads, const PrefillArgs &a, mlops_fbla_items &p);
int tc_item_of(const PrefillArgs &a, int nblk, int groups, const uint8_t *tiles, size_t hstep, size_t gstep, mlops_fbla_items &tc,
               mlops_fbla_items &merge);
#ifdef MLOPS_FBLA_DEFINE_SHARED
bool finish_item_of(int heads, const PrefillArgs &a, mlops_fbla_items &p) {
  char e[64];
  snprintf(e, sizeof e, "mlops_fbla_detail::finish_item<mlops_fbla::Pos>");
  return item_of(finish_kernel, MLOPS_FBLA_ITEM_FINISH, heads, 1, 1, 128, 0, e, a, p);
}
int tc_item_of(const PrefillArgs &a, int nblk, int groups, const uint8_t *tiles, size_t hstep, size_t gstep, mlops_fbla_items &tc,
               mlops_fbla_items &merge) {
  const int heads = a.batch * a.q_heads, smem = prefill_tc_smem();
  static thread_local int sms[16];
  int dev = 0;
  cudaGetDevice(&dev);
  if (!sms[dev]) cudaDeviceGetAttribute(&sms[dev], cudaDevAttrMultiProcessorCount, dev);
  const int nsplit = tc_split_count(nblk, heads, groups, sms[dev]);
  const TcSplitLayout layout = TcSplitLayout::make(heads, a.count, nsplit);
  auto one = [&](auto gqa) {
    using K = decltype(gqa);
    char e[80];
    snprintf(e, sizeof e, "mlops_fbla_detail::tc::prefill_tc_item<%d, mlops_fbla::Pos>", K::value);
    if (!item_of(tc::prefill_tc_kernel<K::value>, MLOPS_FBLA_ITEM_TC, nblk, heads, nsplit, tc::THREADS, smem, e, a, tc))
      return false;
    tc.geom[3] = (long long)tiles;
    tc.geom[4] = (long long)hstep;
    tc.geom[5] = (long long)gstep;
    tc.geom[6] = 0;  // toff
    tc.geom[7] = 0;  // part_ml: the executor's
    tc.geom[8] = 0;  // part_o
    tc.geom[9] = (long long)layout.ml_bytes;
    tc.geom[10] = (long long)layout.output_bytes;
    tc.geom[11] = nsplit;
    return true;
  };
  bool ok;
  switch (a.qg) {
    case 1: ok = one(std::integral_constant<int, 1>{}); break;
    case 2: ok = one(std::integral_constant<int, 2>{}); break;
    case 4: ok = one(std::integral_constant<int, 4>{}); break;
    case 8: ok = one(std::integral_constant<int, 8>{}); break;
    default: ok = one(std::integral_constant<int, 0>{}); break;
  }
  if (!ok) return -1;
  if (nsplit == 1) return nsplit;
  char em[64];
  snprintf(em, sizeof em, "mlops_fbla_detail::tc::merge_item<mlops_fbla::Pos>");
  ok &= item_of(tc::merge_kernel<0>, MLOPS_FBLA_ITEM_TC_MERGE, (a.count + 15) / 16, heads, 1, 256, 0, em, a, merge);
  merge.geom[3] = 0;  // part_ml, part_o: the executor's (the tc pass's)
  merge.geom[4] = 0;
  merge.geom[5] = nsplit;
  return ok ? nsplit : -1;
}
#endif  // MLOPS_FBLA_DEFINE_SHARED

// (host) mlops_fbla_prefill_at / mlops_fbla_append_at as the executor's items, the launches' grids from the chunk's
// ceiling (a.count) alone, as launch_prefill_at sizes them.
template <int CB, int FB>
inline int prefill_at_passes(const PrefillArgs &a, bool append, mlops_fbla_items *p, int max) {
  const int heads = a.batch * a.kv_heads;
  const PrefillGrid grid = PrefillGrid::from_ceiling(a.count);
  const int n = a.q && prefill_tc_device() ? 5 : a.q ? 3 : 2;
  if (max < n) return -1;
  char e[96];
  bool ok = true;
  snprintf(e, sizeof e, "mlops_fbla_detail::close_item<%d, %d, mlops_fbla::Pos>", CB, FB);
  ok &= item_of(close_kernel<CB, FB>, MLOPS_FBLA_ITEM_CLOSE, append ? 1 : grid.close_blocks, heads, 1, 128, 2 * G * D, e, a, p[0]);
  int i = 1;
  if (a.q) {
    if (prefill_tc_device()) {
      const size_t hstep = (size_t)a.hs * IMAGE_SLOT, gstep = (size_t)a.gs * IMAGE_SLOT;
      snprintf(e, sizeof e, "mlops_fbla_detail::tc::transcode_item<%d, %d, mlops_fbla::Pos>", CB, FB);
      ok &= item_of(tc::transcode_kernel<CB, FB>, MLOPS_FBLA_ITEM_TRANSCODE, grid.close_blocks, heads, 1, tc::XT,
                    tc::KT + tc::VT, e, a, p[i]);
      p[i].geom[3] = (long long)a.c.tc_images;
      p[i].geom[4] = (long long)hstep;
      p[i].geom[5] = (long long)gstep;
      p[i].geom[6] = 0;  // toff
      p[i].geom[7] = 1;  // flagged
      ++i;
      const int nsplit = tc_item_of(a, grid.attention_blocks, grid.close_blocks, a.c.tc_images, hstep, gstep, p[i], p[i + 1]);
      if (nsplit < 0) return -1;
      i += nsplit > 1 ? 2 : 1;
    } else {
      constexpr int PROWS = Warps<CB>::PROWS;
      snprintf(e, sizeof e, "mlops_fbla_detail::stream_item<%d, %d, mlops_fbla::Pos>", CB, FB);
      ok &= item_of(stream_kernel<CB, FB>, MLOPS_FBLA_ITEM_STREAM, (a.count * a.qg + PROWS - 1) / PROWS, heads, 1, PTH,
                    (int)sizeof(StreamSmem<PROWS>), e, a, p[i]);
      ++i;
    }
  }
  ok &= finish_item_of(heads, a, p[i]);
  return ok ? i + 1 : -1;
}
#endif  // !__CUDACC_RTC__

}  // namespace mlops_fbla_detail
#endif
