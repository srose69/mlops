// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// FBLA prefill, sm_61+: a chunk of `count` new tokens is appended to the cache and its queries attend to the
// cache, query at position p reading the tokens 0 .. p: the closed groups through their full 6-bit codes (every
// block refined: a decode with tau -> 0), the sink and its own group in BF16.  Three launches:
//
//   close_kernel     quantizes the groups the chunk closes into their records and metadata (6-bit codes,
//                    split into the coarse and fine planes), one CTA per (group, head), thread = channel
//   stream_kernel    the attention of every query row: one FlashAttention pass over the BF16 groups and the
//                    closed groups, streamed as GEMM tiles of full codes
//   finish_kernel    the sink and the open group's BF16 tokens (after the attention, which reads the old ones)
//
// Prefill is compute-bound: a DP4A over 6-bit codes costs what one over the coarse plane does, so the refinement
// rule, which saves the decode its fine-plane reads, would only cost a second pass here.
#ifndef MLOPS_FBLA_PREFILL_CUH
#define MLOPS_FBLA_PREFILL_CUH

#include "fbla_kernels.cuh"

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

// Group g = g0 + blockIdx.x of head blockIdx.y: per channel (thread) the BF16 base (minimum rounded down) and
// step (range / 63 rounded up) of K and V, every code rint((x - base) / step), then the coarse and fine records.
template <int CB, int FB>
__global__ void __launch_bounds__(128) close_kernel(const PrefillArgs a, int g0) {
  __shared__ uint8_t code[2][G][D];  // [K, V][token][channel]
  constexpr int RC = record_bytes<CB>(), RF = record_bytes<FB>();
  const int g = g0 + blockIdx.x, bh = blockIdx.y, c = threadIdx.x;
  const size_t eh = (size_t)bh * G * D, ih = (size_t)bh * a.count * D;
  uint16_t *meta = const_cast<uint16_t *>(a.c.meta) + ((size_t)bh * a.cgroups + g) * 512;
#pragma unroll
  for (int x = 0; x < 2; ++x) {
    const uint16_t *sink = (x ? a.c.exact_sink_v : a.c.exact_sink_k) + eh;
    const uint16_t *open = (x ? a.c.exact_open_v : a.c.exact_open_k) + eh;
    const uint16_t *in = (x ? a.v : a.k) + ih;
    float lo = INFINITY, hi = -INFINITY;
    for (int t = 0; t < G; ++t) {
      const float e = bf16(token_row(sink, open, in, a.n0, g * G + t)[c]);
      lo = fminf(lo, e); hi = fmaxf(hi, e);
    }
    const uint16_t ub = bf16_dir(lo, false);
    const float base = bf16(ub);
    const uint16_t us = bf16_dir(__fdiv_rn(__fsub_rn(hi, base), 63.f), true);
    const float step = bf16(us);
    for (int t = 0; t < G; ++t) {
      const float e = bf16(token_row(sink, open, in, a.n0, g * G + t)[c]);
      const float r = step > 0.f ? rintf(__fdiv_rn(__fsub_rn(e, base), step)) : 0.f;
      code[x][t][c] = (uint8_t)fminf(fmaxf(r, 0.f), 63.f);
    }
    meta[256 * x + 8 * (c >> 2) + (c & 3)] = ub;      // lane c / 4: four bases, four steps
    meta[256 * x + 8 * (c >> 2) + 4 + (c & 3)] = us;
  }
  __syncthreads();
  // records: K code (lane l, slot s, byte b) = channel 64 (l / 16) + 4 s + b of token l % 16; V code = channel
  // 4 l + v_channel(s) of token 4 v_quad(s) + b (the plane's own slot order)
  const size_t sb0 = ((size_t)bh * a.cgroups + g) * SBG;
  for (int wi = threadIdx.x; wi < SBG * (RC + RF) / 4; wi += blockDim.x) {
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

// The sink (positions < 128) and the open group (the chunk's last group) take the chunk's tokens; a newly
// opened group's unused rows are cleared.
// Format-independent: compiled into one object only (fbla_inst.cu, MLOPS_FBLA_DEFINE_SHARED) and launched from
// both through launch_finish, as the select pass.
void launch_finish(const PrefillArgs &a, int heads, cudaStream_t stream);
#ifdef MLOPS_FBLA_DEFINE_SHARED
__global__ void __launch_bounds__(128) finish_kernel(const PrefillArgs a) {
  const int bh = blockIdx.x, n1 = a.n0 + a.count, gl = (n1 - 1) / G;
  const bool fresh = a.n0 == 0 || (a.n0 - 1) / G != gl;  // the open group is a new one
  const size_t eh = (size_t)bh * G * D, ih = (size_t)bh * a.count * D;
#pragma unroll
  for (int x = 0; x < 2; ++x) {
    const uint4 *in = reinterpret_cast<const uint4 *>((x ? a.v : a.k) + ih);
    uint4 *open = reinterpret_cast<uint4 *>(const_cast<uint16_t *>(x ? a.c.exact_open_v : a.c.exact_open_k) + eh);
    uint4 *sink = reinterpret_cast<uint4 *>(const_cast<uint16_t *>(x ? a.c.exact_sink_v : a.c.exact_sink_k) + eh);
    for (int e = threadIdx.x; e < G * 16; e += blockDim.x) {  // 16 uint4 per token row
      const int t = e >> 4, pos = gl * G + t;
      if (pos >= a.n0 && pos < n1) open[e] = in[(size_t)(pos - a.n0) * 16 + (e & 15)];
      else if (fresh && pos >= n1) open[e] = make_uint4(0u, 0u, 0u, 0u);
      if (t >= a.n0 && t < n1) sink[e] = in[(size_t)(t - a.n0) * 16 + (e & 15)];
    }
  }
}
void launch_finish(const PrefillArgs &a, int heads, cudaStream_t stream) { finish_kernel<<<heads, 128, 0, stream>>>(a); }
#endif

// ---- attention: streaming GEMM tiles -------------------------------------------------------------------
// PROWS = 8 x (consumer warps) query rows per CTA; the BF16 groups first (sink, own group), then the closed
// groups newest first, with an online softmax.
// The closed groups stream through a ring of shared-memory stages, warp-specialized: producer warps turn the
// coarse and fine records into tiles of full codes c 2^FB + f (global -> registers -> swizzled stores), the
// consumer warps compute; they meet only at named barriers (stage full / stage empty), the consumers never at a CTA
// barrier.  A stage is a half group (64 tokens, 4 sub-blocks) in tiles that make every GEMM fragment one
// conflict-free LDS.128:
//   Kd[64 tokens][32 words]       word w = channels 4 w .. + 3 of a token (u8 codes); quad q of a token
//                                 stored at quad q ^ ((token / 4) % 8)
//   Vd[16 token words][128 ch]    word = tokens 4 u .. + 3 of a channel; quad q (channels 4 q .. + 3) stored
//                                 at q ^ ((q / 8) % 2)
// Consumer lane = (r = lane / 16, b = lane % 16), rows 8 warp + 4 r .. + 3:
//   QK  tokens 4 b .. + 3 (sub-block b / 4): 4 x 4 scores, DP4A over 32 channel words
//   PV  channels 8 b .. + 7: 4 x 8 outputs, DP4A over the 16 token words
// A warp quantizes its own rows' q per group and produces and consumes its own rows' P: __syncwarp only.
// 8 warps (256 threads: a larger block gets a lower register cap), split by the producer's work per stage:
// the 3-bit planes take two producer warps to keep up, 4+2 one
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
  union {
    struct { alignas(16) uint32_t kd[64 * 32]; alignas(16) uint32_t vd[16 * 128]; } c;  // closed half group
    struct { alignas(16) uint16_t kx[32 * D], vx[32 * D]; } x;  // a BF16 quarter group (stage 1, before the stream)
  };
  alignas(16) uint16_t meta[512];
};
template <int PROWS> struct StreamSmem {
  union {
    Stage st[NST];
  } u;
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

// Producer: record lanes of a half group.  Lane-record index lr: 0 .. 127 the K lanes (sub-block lr / 32, lane
// lr % 32), 128 .. 255 the V lanes; producer lane pt owns lr = pt + 32 k (the K, then the V lanes it serves).
template <int CB, int FB> struct HalfRaw { Codes<CB> w[Warps<CB>::NLR]; Codes<FB> f[Warps<CB>::NLR]; uint4 meta[2]; };
template <int CB, int FB>
__device__ __forceinline__ void half_load(HalfRaw<CB, FB> &h, const uint8_t *coarse_h, const uint8_t *fine_h,
                                          const uint16_t *meta_h, int g, int half, int pt) {
#pragma unroll
  for (int k = 0; k < Warps<CB>::NLR; ++k) {
    const int lr = pt + 32 * Warps<CB>::NPW * k, j4 = (lr & 127) >> 5, l = lr & 31;
    const size_t sb = (size_t)g * SBG + 4 * half + j4;
    const uint8_t *rc = coarse_h + sb * record_bytes<CB>(), *rf = fine_h + sb * record_bytes<FB>();
    if (lr < 128) { load_k<CB>(h.w[k], rc + 16 * l, l); load_k<FB>(h.f[k], rf + 16 * l, l); }
    else { load_v<CB>(h.w[k], rec_at(rc, l)); load_v<FB>(h.f[k], rec_at(rf, l)); }
  }
#pragma unroll
  for (int k = 0; k < 2 / Warps<CB>::NPW; ++k)  // 64 words
    h.meta[k] = __ldg(reinterpret_cast<const uint4 *>(meta_h + (size_t)g * 512) + pt + 32 * Warps<CB>::NPW * k);
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
// 32 (lane % 4) .. + 31).  Returns the lane's row's (sq, q . base_K): score = sq (q-hat . codes) + q . base_K.
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
      for (int x = 0; x < 4; ++x) {
        const float qf = q8[4 * hh + x] * a.scale_log2;
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
  const float sq = amax * (1.f / 127.f);
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    uint32_t bq[4];
#pragma unroll
    for (int x = 0; x < 4; ++x) bq[x] = __float_as_uint(__fadd_rn(__fmul_rn(z[4 * w + x], rq), MAGICF));  // low byte = rint(fl(z r))
    const uint32_t wq = prmt(prmt(bq[0], bq[1], 0x0040u), prmt(bq[2], bq[3], 0x0040u), 0x5410u);
    qhat[row][8 * qt + w] = wq;
  }
  __syncwarp();
  return make_float2(sq, qb);
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
template <int CB, int FB>
__global__ void __launch_bounds__(PTH) stream_kernel(const PrefillArgs a) {
  constexpr int NCW = Warps<CB>::NCW, PROWS = Warps<CB>::PROWS, CTH = Warps<CB>::CTH;
  __shared__ StreamSmem<PROWS> sm;
  constexpr int RC = record_bytes<CB>();
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int bh = blockIdx.y, rows = a.count * a.qg, r0 = blockIdx.x * PROWS, n1 = a.n0 + a.count;
  const int p_first = a.n0 + r0 / a.qg, p_last = a.n0 + (min(r0 + PROWS, rows) - 1) / a.qg;
  const int g_lo = p_first / G, g_hi = p_last / G, nhalf = 2 * max(0, g_hi - 1);
  const size_t head = (size_t)bh * a.cgroups;
  const uint8_t *coarse_h = a.c.coarse + head * SBG * RC, *fine_h = a.c.fine + head * SBG * record_bytes<FB>();
  const uint16_t *meta_h = a.c.meta + head * 512;
  auto group_of = [&](int it) { return g_hi - 1 - (it >> 1); };  // newest closed group first

  if (warp >= NCW) {  // ---- producer: the closed groups' half groups into the ring -------------------------
    const int pt = t - CTH;
    HalfRaw<CB, FB> raw;
    if (nhalf) half_load(raw, coarse_h, fine_h, meta_h, group_of(0), 0, pt);
    for (int it = 0; it < nhalf; ++it) {
      if (it == 1 || it >= NST) bar_sync(BAR_EMPTY + it % NST, PTH);  // stage 1 first holds the BF16 groups
      half_store(sm.u.st[it % NST], raw, pt);
      bar_arrive(BAR_FULL + it % NST, PTH);
      if (it + 1 < nhalf) half_load(raw, coarse_h, fine_h, meta_h, group_of(it + 1), (it + 1) & 1, pt);
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
    pos[i] = a.n0 + (valid[i] ? (r0 + rr + i) / a.qg : 0);
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
  // FA update of a step: lane block maxima mb, sums sp (a block's lanes differ below bit o0).  m of a row moves
  // only when a block exceeds it by more than 8 (in log2; one ballot): weights w = 2^(mb - m) <= 2^8 are exact
  // enough, and the row's max reduction and rescale are rare.  l is kept per lane (the blocks the lane leads)
  // and reduced over the row's 16 lanes only when needed (row_l).
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
    uint16_t *kx = sm.u.st[1].x.kx, *vx = sm.u.st[1].x.vx;
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
          const uint4 v = __ldg(reinterpret_cast<const uint4 *>(token_row(x ? sink_v : sink_k, x ? open_v : open_k,
                                                                           x ? in_v : in_k, a.n0, pt)) + qd);
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
            const uint4 qq = __ldg(qrow[i] + e);
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
    for (int i = 0; i < 4; ++i) qv4[i] = r0 + qrow_q < rows ? __ldg(qp + i) : make_uint4(0u, 0u, 0u, 0u);
  }
  float2 rcv[4];  // per row (sq, q . base_K) of the current group
  for (int it = 0; it < nhalf; ++it) {
    const int g = group_of(it), half = it & 1;
    const Stage &st = sm.u.st[it % NST];
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

// Groups closed by the chunk: those with 128 (g + 1) < n0 + count that were open before (the group of
// position n0 - 1 onwards); a full open group closes when the next token arrives.
template <int CB, int FB> cudaError_t launch_prefill(const PrefillArgs &a, cudaStream_t stream) {
  const int heads = a.batch * a.kv_heads, n1 = a.n0 + a.count;
  const int g_first = a.n0 == 0 ? 0 : (a.n0 - 1) / G, g_end = (n1 - 1) / G;
  if (g_end > g_first) close_kernel<CB, FB><<<dim3(g_end - g_first, heads), 128, 0, stream>>>(a, g_first);
  if (a.q) {
    constexpr int PROWS = Warps<CB>::PROWS;
    const dim3 grid((a.count * a.qg + PROWS - 1) / PROWS, heads);
    stream_kernel<CB, FB><<<grid, PTH, 0, stream>>>(a);
  }
  launch_finish(a, heads, stream);
  return cudaGetLastError();
}

}  // namespace mlops_fbla_detail
#endif
