// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The tensor-core prefill's per-warp arithmetic over a group image (tensors/transcode.cuh).  A warp owns 16 query rows;
// lane = 4 g + t holds rows g and g + 8.  q stays in 32 registers: q[r][c] = the BF16 pair of channels 8 c + 2 t, + 1
// of row g + 8 r, the A fragments of the BF16 mma as they are and the source of every closed group's q-hat.
//
// Closed group (128 tokens, one P block): q-hat (the format's: z = q step exact, r = fl(127 / max|z|), q-hat = rint(z r),
// sq = max|z| sl / 127, qb = sl q . base), QK by s8 x u8 mma, one maximum, P = rint(255 2^(s - m)) in bytes, PV by
// u8 x u8 mma, the fold o += w (step X + base sum P) in FP32.  Integer accumulators carry MAGIC (FP32 bits of 1.5 2^23).
//
// Scheduling: ptxas hoists independent loads and products and front-loads mma; dependencies through a hidden zero
// (zero_of: an integer 0 the compilers cannot see through) keep each step's live range short.  Every such gate below is
// measured (nvdisasm -plr, L40S cycles).
#ifndef MLOPS_FBLA_TENSORS_FN_CUH
#define MLOPS_FBLA_TENSORS_FN_CUH

#include "transcode.cuh"

namespace mlops_fbla_detail {
namespace tc {
#ifdef MLOPS_FBLA_TC75

__device__ __forceinline__ float lo_bf16(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi_bf16(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ float bf16_half(uint32_t w, int e) { return e ? hi_bf16(w) : lo_bf16(w); }
// KM / VM: shared memory (the slot, sm_80+) or global (the image, sm_75)
template <class Pos, bool Shared>
__device__ __forceinline__ uint4 meta128(const uint8_t *p) {
  if constexpr (Shared) return *reinterpret_cast<const uint4 *>(p);
  else return InputLoad<Pos>::get(reinterpret_cast<const uint4 *>(p));
}

// 0, data-dependent on b: min(|b|, +0) is +0 for every bit pattern (one FMNMX)
__device__ __forceinline__ int zero_of(uint32_t b) {
  int z;
  asm("{ .reg .f32 t; abs.f32 t, %1; min.f32 t, t, 0f00000000; mov.b32 %0, t; }" : "=r"(z) : "f"(__uint_as_float(b)));
  return z;
}
__device__ __forceinline__ float q_half(uint32_t w, int e, int zk) {  // BF16 half e of w as FP32 (zk = 0)
  return __uint_as_float(prmt(w, (uint32_t)zk, e ? 0x3244u : 0x1044u));
}

// q rows g and g + 8 (row0, row1), lane t's words 4 c + t
template <class Pos = Grid>
__device__ __forceinline__ void q_load(uint32_t (&q)[2][16], const uint16_t *row0, const uint16_t *row1, int t) {
  const uint32_t *p0 = reinterpret_cast<const uint32_t *>(row0) + t, *p1 = reinterpret_cast<const uint32_t *>(row1) + t;
  unroll<16>([&](auto c) { q[0][c] = Pos::ro(p0 + 4 * c); q[1][c] = Pos::ro(p1 + 4 * c); });
}

// ---- q-hat: quant_p1<0 .. 3>, quant_scale, quant_p2<0 .. 3> in this order -----------------------------------------
// km = KM + 48 t (record (KS, t) at + 192 KS: 8 FP32 steps, 8 BF16 bases).  Each step's loads and q unpacks wait for the
// previous step through zk (else LICM hoists the unpacks out of the group loop and ptxas merges pass 2's loads into
// pass 1's).  Pass 1: max|z| (am) and q . base (qp, two chains a row).
template <int KS, class Pos, bool Shared>
__device__ __forceinline__ void quant_p1(const uint32_t (&q)[2][16], const uint8_t *&km, int &zk, float (&am)[2], float (&qp)[2][2]) {
  km += zk;
  const uint4 s0 = meta128<Pos, Shared>(km + 192 * KS), s1 = meta128<Pos, Shared>(km + 192 * KS + 16),
              bw = meta128<Pos, Shared>(km + 192 * KS + 32);
  const uint32_t st[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w}, b[4] = {bw.x, bw.y, bw.z, bw.w};
  unroll<2>([&](auto r) {
    float az[8];
    unroll<4>([&](auto j) {
      unroll<2>([&](auto e) {
        const float qv = q_half(q[r][4 * KS + j], e, zk);
        az[2 * j + e] = fabsf(qv * __uint_as_float(st[2 * j + e]));
        qp[r][e] = fmaf(qv, bf16_half(b[j], e), qp[r][e]);
      });
    });
    am[r] = fmaxf(am[r], fmaxf(fmaxf(fmaxf(az[0], az[1]), fmaxf(az[2], az[3])), fmaxf(fmaxf(az[4], az[5]), fmaxf(az[6], az[7]))));
  });
  zk = zero_of(__float_as_uint(am[0]) | __float_as_uint(am[1]) | __float_as_uint(qp[0][0]) | __float_as_uint(qp[0][1]) |
               __float_as_uint(qp[1][0]) | __float_as_uint(qp[1][1]));
}
// The quad's max|z| and qb; r = fl(127 / max|z|) (correctly rounded: RCP, one refinement, residual correction; 0 for a
// zero row), sq
__device__ __forceinline__ void quant_scale(int &zk, float sl, float (&am)[2], const float (&qp)[2][2], float (&rq)[2], float (&sq)[2],
                                            float (&qb)[2]) {
  unroll<2>([&](auto r) {
    am[r] = __int_as_float(__float_as_int(am[r]) + zk);
    am[r] = fmaxf(am[r], __shfl_xor_sync(FULL, am[r], 1));
    qb[r] = qp[r][0] + qp[r][1];
    qb[r] += __shfl_xor_sync(FULL, qb[r], 1);
  });
  unroll<2>([&](auto r) {
    am[r] = fmaxf(am[r], __shfl_xor_sync(FULL, am[r], 2));
    qb[r] += __shfl_xor_sync(FULL, qb[r], 2);
  });
  const float s127 = sl * (1.f / 127.f);
  unroll<2>([&](auto r) {
    float rc;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rc) : "f"(am[r]));
    rc = fmaf(fmaf(-am[r], rc, 1.f), rc, rc);
    const float q0 = 127.f * rc;
    rq[r] = am[r] > 0.f ? fmaf(fmaf(-am[r], q0, 127.f), rc, q0) : 0.f;
    sq[r] = am[r] * s127;
    qb[r] *= sl;
  });
  zk = zero_of(__float_as_uint(rq[0]) | __float_as_uint(rq[1]));
}
// Pass 2: the s8 A fragment of k step KS (z recomputed), qa[2 h + r] = row r's bytes of channels j = 2 h, 2 h + 1
template <int KS, class Pos, bool Shared>
__device__ __forceinline__ void quant_p2(const uint32_t (&q)[2][16], const uint8_t *&km, int &zk, const float (&rq)[2], uint32_t (&qa)[4]) {
  km += zk;
  const uint4 s0 = meta128<Pos, Shared>(km + 192 * KS), s1 = meta128<Pos, Shared>(km + 192 * KS + 16);
  const uint32_t st[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
  unroll<2>([&](auto h) {
    unroll<2>([&](auto r) {
      uint32_t bq[4];
      unroll<4>([&](auto x) {
        constexpr int J = 2 * decltype(h)::value + decltype(x)::value / 2, E = decltype(x)::value % 2;
        const float z = q_half(q[r][4 * KS + J], E, zk) * __uint_as_float(st[2 * J + E]);
        bq[x] = __float_as_uint(fmaf(z, rq[r], MAGICF));  // low byte = rint(z r)
      });
      qa[2 * h + r] = prmt(prmt(bq[0], bq[1], 0x0040u), prmt(bq[2], bq[3], 0x0040u), 0x5410u);
    });
  });
  zk = zero_of(qa[0] | qa[1] | qa[2] | qa[3]);
}
template <class Pos>
__device__ __forceinline__ void quantize(const uint32_t (&q)[2][16], const uint8_t *km, float sl, uint32_t (&qa)[4][4], float (&sq)[2],
                                         float (&qb)[2]) {
  float qp[2][2] = {{0.f, 0.f}, {0.f, 0.f}}, am[2] = {0.f, 0.f}, rq[2];
  int zk = 0;
  unroll<4>([&](auto ks) { quant_p1<decltype(ks)::value, Pos, false>(q, km, zk, am, qp); });
  quant_scale(zk, sl, am, qp, rq, sq, qb);
  unroll<4>([&](auto ks) { quant_p2<decltype(ks)::value, Pos, false>(q, km, zk, rq, qa[ks]); });
}

// ---- QK -------------------------------------------------------------------------------------------------------
// A half group (64 tokens, kt: its K rows; ko = k_lane_off): acc[n] = n-tile n, c0 c1 row g, c2 c3 row g + 8, MAGIC +
// the score.  Two row blocks at a time (four independent IMMA chains); a pair's maxima go into mx (biased integers are
// monotonic) and gate the next pair's loads (else all 16 ldmatrix start at once: 82 against 62 live registers).  RED:
// the quad's reduction of mx.
template <bool RED>
__device__ __forceinline__ void qk_half(uint32_t (&acc)[8][4], const uint32_t (&qa)[4][4], uint32_t kt, uint32_t ko, uint32_t (&mx)[2]) {
  unroll<2>([&](auto gi) {
    unroll<4>([&](auto ks) {
      constexpr int KS = decltype(ks)::value;
      const uint32_t base = kt + (ko ^ (KS << 5));
      unroll<2>([&](auto j) {
        constexpr int NP = 2 * decltype(gi)::value + decltype(j)::value;
        uint32_t b[4];
        ldsm_x4<2048 * NP>(b, base);
        if constexpr (KS == 0) {
          mma_s8u8_c(acc[2 * NP], qa[0], b[0], b[1], MAGIC);
          mma_s8u8_c(acc[2 * NP + 1], qa[0], b[2], b[3], MAGIC);
        } else {
          mma_s8u8(acc[2 * NP], qa[KS], b[0], b[1]);
          mma_s8u8(acc[2 * NP + 1], qa[KS], b[2], b[3]);
        }
      });
    });
    unroll<4>([&](auto j) {
      constexpr int N = 4 * decltype(gi)::value + decltype(j)::value;
      mx[0] = max(mx[0], max(acc[N][0], acc[N][1]));
      mx[1] = max(mx[1], max(acc[N][2], acc[N][3]));
    });
    kt += zero_of(mx[0] | mx[1]);
  });
  if constexpr (RED)
    unroll<2>([&](auto r) {
      mx[r] = max(mx[r], __shfl_xor_sync(FULL, mx[r], 1));
      mx[r] = max(mx[r], __shfl_xor_sync(FULL, mx[r], 2));
    });
}
// Both halves (half h's K at kt + KT / 2 h), the maxima over the 128 tokens
__device__ __forceinline__ void qk_group(uint32_t (&acc)[2][8][4], const uint32_t (&qa)[4][4], uint32_t kt, uint32_t ko, uint32_t (&mx)[2]) {
  mx[0] = mx[1] = 0u;
  qk_half<false>(acc[0], qa, kt, ko, mx);
  qk_half<true>(acc[1], qa, kt + KT / 2, ko, mx);
}

// ---- P ------------------------------------------------------------------------------------------------------------
// d = sq acc + nb <= 0 in one FFMA (nb = -sq max rounded down; res, the exact rest, goes into the block's weight), P =
// rint(255 2^d) in the low byte of the magic FFMA
__device__ __forceinline__ void p_bias(const uint32_t (&mx)[2], const float (&sq)[2], float (&nb)[2], float (&res)[2]) {
  unroll<2>([&](auto r) {
    const float n = -sq[r] * __uint_as_float(mx[r]), e = fmaf(-sq[r], __uint_as_float(mx[r]), -n);
    nb[r] = __uint_as_float(__float_as_uint(n) + (e < 0.f));
    res[r] = fmaf(-sq[r], __uint_as_float(mx[r]), -nb[r]);
  });
}
__device__ __forceinline__ uint32_t p_byte(uint32_t acc, float sq, float nb) {
  return __float_as_uint(fmaf(ex2(fmaf(sq, __uint_as_float(acc), nb)), 255.f, MAGICF));
}
// P word 2 h + r of PV block B (tokens 32 B ..): row r of n-tiles 4 B + 2 h, + 1 (k_row: PV's A fragment as it is)
template <int B, int H, int R>
__device__ __forceinline__ uint32_t p_word(const uint32_t (&acc)[8][4], float sq, float nb) {
  constexpr int N = 4 * B + 2 * H;
  return prmt(prmt(p_byte(acc[N][2 * R], sq, nb), p_byte(acc[N][2 * R + 1], sq, nb), 0x0040u),
              prmt(p_byte(acc[N + 1][2 * R], sq, nb), p_byte(acc[N + 1][2 * R + 1], sq, nb), 0x0040u), 0x5410u);
}
// pa[Q] = the A fragment of PV block Q (V quarter Q)
__device__ __forceinline__ void p_words128(const uint32_t (&acc)[2][8][4], const float (&sq)[2], const float (&nb)[2], uint32_t (&pa)[4][4]) {
  unroll<4>([&](auto qq) {
    constexpr int Q = decltype(qq)::value;
    unroll<2>([&](auto hh) {
      unroll<2>([&](auto rr) {
        constexpr int H = decltype(hh)::value, R = decltype(rr)::value;
        pa[Q][2 * H + R] = p_word<Q % 2, H, R>(acc[Q / 2], sq[R], nb[R]);
      });
    });
  });
}
// The rows' P sums (an mma with ones): c[0] row g, c[2] row g + 8, MAGIC + the sum
__device__ __forceinline__ void p_rowsum128(const uint32_t (&pa)[4][4], uint32_t (&c)[4]) {
  mma_u8u8_c(c, pa[0], 0x01010101u, 0x01010101u, MAGIC);
  unroll<3>([&](auto qq) { mma_u8u8(c, pa[decltype(qq)::value + 1], 0x01010101u, 0x01010101u); });
}
// The running maximum moves only when a block's mb > m + 8 (warp-uniform, rare): o and l rescale by 2^(m - mb)
__device__ __forceinline__ void lazy_max(const float (&mb)[2], float (&m)[2], float (&l)[2], float (&o)[16][4]) {
  const bool up0 = mb[0] > m[0] + 8.f, up1 = mb[1] > m[1] + 8.f;
  if (__any_sync(FULL, up0 || up1)) {
    float al[2];
    unroll<2>([&](auto r) {
      const bool up = r ? up1 : up0;
      al[r] = up ? (m[r] == -INFINITY ? 0.f : ex2(m[r] - mb[r])) : 1.f;
      m[r] = up ? mb[r] : m[r];
      l[r] *= al[r];
    });
    unroll<16>([&](auto n) { unroll<4>([&](auto e) { o[n][e] *= al[e >> 1]; }); });
  }
}
// The group's mb = qb + sq x_max (-inf: not row r's group) and weight w = 2^(mb + res - m)
__device__ __forceinline__ void p128_weight(const uint32_t (&mx)[2], const float (&sq)[2], const float (&qb)[2], const float (&res)[2],
                                            const bool (&on)[2], float (&m)[2], float (&l)[2], float (&o)[16][4], float (&w)[2]) {
  float mb[2];
  unroll<2>([&](auto r) { mb[r] = on[r] ? fmaf(sq[r], __uint_as_float(mx[r]) - MAGICF, qb[r]) : -INFINITY; });
  lazy_max(mb, m, l, o);
  unroll<2>([&](auto r) { w[r] = ex2(mb[r] + res[r] - m[r]); });
}

// ---- PV and the fold -----------------------------------------------------------------------------------------------
// Microtile CB's fold: o += w (step acc - step MAGIC) + ws base, three FFMA an output, exact inner product (step BF16);
// vm = VM + 16 t, record (CB, t) at + 64 CB.  Returns the OR of its outputs: the gate for the next microtiles.
template <int CB, class Pos, bool Shared>
__device__ __forceinline__ uint32_t fold_tile(float (&o)[16][4], const uint32_t (&acc)[2][4], const uint8_t *vm, const float (&w)[2],
                                              const float (&ws)[2]) {
  const uint4 m = meta128<Pos, Shared>(vm + 64 * CB);
  const uint32_t bw[2] = {m.x, m.z}, sw[2] = {m.y, m.w};
  uint32_t u = 0;
  unroll<2>([&](auto j) {
    unroll<4>([&](auto e) {
      const float st = bf16_half(sw[j], e & 1), b = bf16_half(bw[j], e & 1);
      o[2 * CB + j][e] = fmaf(w[e >> 1], fmaf(st, __uint_as_float(acc[j][e]), -st * MAGICF), fmaf(ws[e >> 1], b, o[2 * CB + j][e]));
      u |= __float_as_uint(o[2 * CB + j][e]);
    });
  });
  return u;
}
// PV of the group, microtile cb = channels 16 cb .. + 15 over the 4 V quarters (vt + VQ Q; vo = v_lane_off), then its
// fold.  Microtile cb + 2's first IMMA wait for fold cb (B operand + zero: at most 16 accumulators live), microtile
// cb + 3's loads too (vt + zero).  step(cb): work placed after microtile cb (the next group's q-hat).
template <class Pos, bool Shared, class F>
__device__ __forceinline__ void pv_group(float (&o)[16][4], const uint32_t (&pa)[4][4], const float (&w)[2], const float (&ws)[2],
                                         uint32_t vt, uint32_t vo, const uint8_t *vm, F step) {
  constexpr int ILAG = 2, LAG = 3;
  uint32_t u[8], zc = 0, z = 0;
  unroll<8>([&](auto cb) {
    constexpr int CB = decltype(cb)::value;
    uint32_t acc[2][4];
    unroll<4>([&](auto qq) {
      constexpr int Q = decltype(qq)::value;
      uint32_t v[4];
      ldsm_x4<VQ * Q + 512 * CB>(v, vt + vo);
      if constexpr (Q == 0) {
        mma_u8u8_c(acc[0], pa[0], v[0] + zc, v[1], MAGIC);
        mma_u8u8_c(acc[1], pa[0], v[2] + zc, v[3], MAGIC);
      } else {
        mma_u8u8(acc[0], pa[Q], v[0], v[1]);
        mma_u8u8(acc[1], pa[Q], v[2], v[3]);
      }
    });
    u[CB] = fold_tile<CB, Pos, Shared>(o, acc, vm + z, w, ws);
    if constexpr (CB + 1 - ILAG >= 0 && CB < 7) zc = zero_of(u[CB + 1 - ILAG]);
    if constexpr (CB + 1 - LAG >= 0 && CB < 7) {
      z = zero_of(u[CB + 1 - LAG]);
      vt += z;
    }
    step(cb);
  });
}
// One closed group for the warp's rows (kt, vt: the slot's K and V; vm = VM + 16 t; on[r]: the group is row r's), the
// next group's q-hat (kmn = its KM + 48 t) under the PV: qa, sq, qb in this group's, out the next one's.
template <class Pos>
__device__ __forceinline__ void closed_group(float (&o)[16][4], float (&m)[2], float (&l)[2], const uint32_t (&q)[2][16], float sl,
                                             const uint8_t *kmn, const uint8_t *vm, uint32_t kt, uint32_t vt, uint32_t ko, uint32_t vo,
                                             const bool (&on)[2], uint32_t (&qa)[4][4], float (&sq)[2], float (&qb)[2]) {
  uint32_t acc[2][8][4], mx[2], pa[4][4], rs[4];
  float nb[2], res[2], w[2], ws[2];
  qk_group(acc, qa, kt, ko, mx);
  p_bias(mx, sq, nb, res);
  p128_weight(mx, sq, qb, res, on, m, l, o, w);
  p_words128(acc, sq, nb, pa);
  p_rowsum128(pa, rs);
  unroll<2>([&](auto r) {
    ws[r] = w[r] * (__uint_as_float(rs[2 * r]) - MAGICF);
    l[r] += ws[r];
  });
  float qp[2][2] = {{0.f, 0.f}, {0.f, 0.f}}, am[2] = {0.f, 0.f}, rq[2];
  int zk = 0;
  const uint8_t *km = kmn;
#ifdef MLOPS_FBLA_TC80
  constexpr bool Shared = true;
#else
  constexpr bool Shared = false;
#endif
  pv_group<Pos, Shared>(o, pa, w, ws, vt, vo, vm, [&](auto cb) {
    constexpr int CB = decltype(cb)::value;
    if constexpr (CB < 4) quant_p1<CB, Pos, Shared>(q, km, zk, am, qp);
    if constexpr (CB == 4) quant_scale(zk, sl, am, qp, rq, sq, qb);
    if constexpr (CB >= 5) quant_p2<CB - 5, Pos, Shared>(q, km, zk, rq, qa[CB - 5]);
    if constexpr (CB == 7) quant_p2<3, Pos, Shared>(q, km, zk, rq, qa[3]);
  });
}

// ---- BF16 blocks (the sink, the rows' own group): 64 tokens a P block -------------------------------------------------
// K at kx, V at kx + KT, [64 tokens][256 B], chunk c of row r at c ^ (r & 7).  s[j][e]: row g + 8 (e >> 1), token
// 8 j + 2 t + (e & 1).  P = 255 2^(s - mb) rounded to BF16, not to an integer: the sink's block carries the row's
// largest weight, and integer P rounds the block's other tokens to 0 / 1 of it (3x the error against exact attention,
// Qwen3-0.6B).
__host__ __device__ constexpr uint32_t bf_off(int row, int chunk) { return row * 256 + ((chunk ^ (row & 7)) << 4); }

// P in place of s (tokens > tlim[r] masked, tlim < 0: none), the block's maximum mb and sum sp
__device__ __forceinline__ void exact_p(float (&s)[8][4], float sl, const int (&tlim)[2], int t, float (&mb)[2], float (&sp)[2]) {
  float mx[2] = {static_cast<float>(-INFINITY), static_cast<float>(-INFINITY)};
  unroll<8>([&](auto j) {
    unroll<4>([&](auto e) {
      const int r = e >> 1, tk = 8 * decltype(j)::value + 2 * t + (e & 1);
      s[j][e] = tk <= tlim[r] ? s[j][e] * sl : -INFINITY;
      mx[r] = fmaxf(mx[r], s[j][e]);
    });
  });
  float mz[2], ps[2] = {0.f, 0.f};
  unroll<2>([&](auto r) {
    mx[r] = fmaxf(mx[r], __shfl_xor_sync(FULL, mx[r], 1));
    mb[r] = fmaxf(mx[r], __shfl_xor_sync(FULL, mx[r], 2));
    mz[r] = mb[r] == -INFINITY ? 0.f : mb[r];
  });
  unroll<8>([&](auto j) {
    unroll<4>([&](auto e) {
      const uint32_t u = __float_as_uint(255.f * ex2(s[j][e] - mz[e >> 1]));
      s[j][e] = __uint_as_float((u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u);  // to BF16, nearest even
      ps[e >> 1] += s[j][e];
    });
  });
  unroll<2>([&](auto r) {
    ps[r] += __shfl_xor_sync(FULL, ps[r], 1);
    sp[r] = ps[r] + __shfl_xor_sync(FULL, ps[r], 2);
  });
}
// w = 2^(mb - m) after the lazy maximum (0: nothing of the row); l += w sp
__device__ __forceinline__ void exact_weight(const float (&mb)[2], const float (&sp)[2], float (&m)[2], float (&l)[2], float (&o)[16][4],
                                             float (&w)[2]) {
  lazy_max(mb, m, l, o);
  unroll<2>([&](auto r) {
    w[r] = mb[r] == -INFINITY ? 0.f : ex2(mb[r] - m[r]);
    l[r] = fmaf(w[r], sp[r], l[r]);
  });
}

#ifdef MLOPS_FBLA_TC80
// q.k on the BF16 mma, k step kk = channels 16 kk ..
__device__ __forceinline__ void exact_scores(float (&s)[8][4], const uint32_t (&q)[2][16], uint32_t kx, int lane) {
  const int rm = lane & 7, mi = lane >> 3;
  const uint32_t lo = bf_off(8 * (mi >> 1) + rm, mi & 1);  // (chunk 2 kk + mi % 2: lo ^ 32 kk)
  unroll<8>([&](auto j) { unroll<4>([&](auto e) { s[j][e] = 0.f; }); });
  unroll<8>([&](auto kk) {
    constexpr int KK = decltype(kk)::value;
    const uint32_t a[4] = {q[0][2 * KK], q[1][2 * KK], q[0][2 * KK + 1], q[1][2 * KK + 1]};
    unroll<4>([&](auto np) {
      constexpr int NP = decltype(np)::value;
      uint32_t b[4];
      ldsm_x4<4096 * NP>(b, kx + (lo ^ (32 * KK)));
      mma_bf16(s[2 * NP], a, b[0], b[1]);
      mma_bf16(s[2 * NP + 1], a, b[2], b[3]);
    });
  });
}
// o += w P V (P as BF16 A fragments, V read transposed), one channel half at a time
__device__ __forceinline__ void exact_pv(float (&o)[16][4], const float (&p)[8][4], uint32_t vx, const float (&w)[2], int lane) {
  const int rm = lane & 7, mi = lane >> 3;
  const uint32_t lo = bf_off(8 * (mi & 1) + rm, mi >> 1);
  uint32_t pa[4][4];
  unroll<8>([&](auto j) {
    unroll<2>([&](auto r) { pa[j >> 1][2 * (j & 1) + r] = prmt(__float_as_uint(p[j][2 * r]), __float_as_uint(p[j][2 * r + 1]), 0x7632u); });
  });
  unroll<2>([&](auto h) {
    constexpr int H = decltype(h)::value;
    float y[8][4];
    unroll<8>([&](auto n) { unroll<4>([&](auto e) { y[n][e] = 0.f; }); });
    unroll<4>([&](auto kk) {
      unroll<4>([&](auto np) {
        constexpr int NP = decltype(np)::value;
        uint32_t v[4];
        ldsm_x4_t<4096 * decltype(kk)::value>(v, vx + (lo ^ (16 * (8 * H + 2 * NP))));
        mma_bf16(y[2 * NP], pa[kk], v[0], v[1]);
        mma_bf16(y[2 * NP + 1], pa[kk], v[2], v[3]);
      });
    });
    unroll<8>([&](auto n) { unroll<4>([&](auto e) { o[8 * H + n][e] = fmaf(w[e >> 1], y[n][e], o[8 * H + n][e]); }); });
  });
}
#else
// sm_75 (no BF16 mma): FFMA over the lane's 32 channels, then the quad's sum.  (Loops, not lambdas: NVVM does not
// inline lambdas this large, and their arguments then go to local memory.)
__device__ __forceinline__ void exact_scores(float (&s)[8][4], const uint32_t (&q)[2][16], const uint8_t *kx, int lane) {
  const int t = lane & 3;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float d[8][2];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      d[i][0] = d[i][1] = 0.f;
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        const uint32_t kw = *reinterpret_cast<const uint32_t *>(kx + bf_off(8 * j + i, c) + 4 * t);
#pragma unroll
        for (int r = 0; r < 2; ++r) d[i][r] = fmaf(hi_bf16(q[r][c]), hi_bf16(kw), fmaf(lo_bf16(q[r][c]), lo_bf16(kw), d[i][r]));
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        d[i][r] += __shfl_xor_sync(FULL, d[i][r], 1);
        d[i][r] += __shfl_xor_sync(FULL, d[i][r], 2);
      }
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int E = e & 1, R = e >> 1;
      s[j][e] = t == 0 ? d[E][R] : t == 1 ? d[2 + E][R] : t == 2 ? d[4 + E][R] : d[6 + E][R];
    }
  }
}
__device__ __forceinline__ void exact_pv(float (&o)[16][4], const float (&p)[8][4], const uint8_t *vx, const float (&w)[2], int lane) {
  const int t = lane & 3, q0 = lane & ~3;
#pragma unroll
  for (int T = 0; T < 64; ++T) {
    const int J = T >> 3, E = T & 1;
    const float p0 = __shfl_sync(FULL, p[J][E], q0 + ((T >> 1) & 3)) * w[0], p1 = __shfl_sync(FULL, p[J][2 + E], q0 + ((T >> 1) & 3)) * w[1];
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      const uint32_t vw = *reinterpret_cast<const uint32_t *>(vx + bf_off(T, n) + 4 * t);
      o[n][0] = fmaf(p0, lo_bf16(vw), o[n][0]);
      o[n][1] = fmaf(p0, hi_bf16(vw), o[n][1]);
      o[n][2] = fmaf(p1, lo_bf16(vw), o[n][2]);
      o[n][3] = fmaf(p1, hi_bf16(vw), o[n][3]);
    }
  }
}
#endif
#endif  // MLOPS_FBLA_TC75

}  // namespace tc
}  // namespace mlops_fbla_detail
#endif
