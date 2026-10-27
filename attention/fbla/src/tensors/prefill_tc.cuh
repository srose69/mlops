// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// FBLA prefill attention on tensor cores (sm_75 .. sm_89), format-independent (the images carry full codes).
//
// CTA = one q head x the positions of one group (warp w: positions 16 w ..), 8 warps, two slots of shared memory.  The
// BF16 blocks first (sink, own group; CTA barriers), then the closed groups in ascending order, group j in slot j & 1:
// every CTA of a wave reads the same image at the same time, so the L2 needs only a few images whatever its size.
//   sm_80+  slot = image (K, V, own VM, next group's KM).  Owner warp j & 7 copies group j (cp.async) a group ahead,
//           waits right before it and arrives on the slot's 7 pairwise READY barriers (ready_id); after group j the
//           slot's EMPTY barrier: the owner of j + 2 and of j sync (the latter is about to take over the id the new
//           owner leaves), the others arrive.
//   sm_75   no cp.async: a CTA barrier a group, every thread refills; KM / VM from global, prefetched into L1.
// GQA = 0: the ratio at run time.  Split-K (gridDim.z > 1) for grids below a wave, merged by merge_kernel.
#ifndef MLOPS_FBLA_TENSORS_PREFILL_TC_CUH
#define MLOPS_FBLA_TENSORS_PREFILL_TC_CUH

#include "fn.cuh"
#include <climits>

namespace mlops_fbla_detail {
namespace tc {

constexpr int NW = 8, THREADS = 32 * NW;
#ifdef MLOPS_FBLA_TC80
constexpr int SLOT = TGB;
#else
constexpr int SLOT = KT + VT;  // (sm_75: 64 KiB a block)
#endif
// the dynamic shared memory of the specialization compiled for virtual architecture ptx (x 10)
__host__ __device__ constexpr int smem_bytes(int ptx) { return 2 * (ptx >= 80 ? TGB : KT + VT); }
constexpr uint32_t BAR_EMPTY = 14;  // + slot; READY: 0 .. 13
static_assert(KT + VT == 2 * 64 * 256 && SLOT % 128 == 0, "a slot holds a BF16 block's K and V");

#ifdef MLOPS_FBLA_TC80
// One warp copies image img's K, V, VM and the next image's KM (nxt; null: none) into the slot, one commit
__device__ __forceinline__ void copy_group(uint32_t slot, const uint8_t *img, const uint8_t *nxt, int lane) {
  const uint32_t d = slot + 16 * lane;
  const uint8_t *s = img + 16 * lane;
  unroll<(KT + VT) / 512>([&](auto k) { cp_async16<512 * decltype(k)::value, 512 * decltype(k)::value>(d, s); });
  cp_async16<O_VM, O_VM>(d, s);
  if (nxt) {
    cp_async16<O_KM>(d, nxt + O_KM + 16 * lane);
    if (lane < 16) cp_async16<O_KM + 512>(d, nxt + O_KM + 512 + 16 * lane);
  }
  cp_async_commit();
}
#endif

// tiles: group g of KV head bh at tiles + bh hstep + (g - toff) gstep (bytes)
template <int GQA>
__global__ void __launch_bounds__(THREADS, 1) prefill_tc_kernel(const PrefillArgs a, const uint8_t *tiles, size_t hstep, size_t gstep, int toff,
                                                                float2 *part_ml, float *part_o) {
#ifdef MLOPS_FBLA_TC75
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t = lane & 3;
  const int bq = blockIdx.y, bb = bq / a.q_heads, hq = bq - bb * a.q_heads;
  const int bh = bb * a.kv_heads + (GQA ? hq / GQA : hq / a.qg);
  const int n1 = a.n0 + a.count, gb = a.n0 / G + blockIdx.x;  // the CTA's own group
  const float sl = a.scale_log2;
  const uint32_t s0 = smem_u32(smem);

  bool valid[2];
  int pos[2];
  unroll<2>([&](auto r) {
    pos[r] = gb * G + 16 * warp + g8 + 8 * r;
    valid[r] = pos[r] >= a.n0 && pos[r] < n1;
  });
  uint32_t q[2][16];
  {
    const uint16_t *qh = a.q + (size_t)bq * a.count * D;
    q_load(q, qh + (size_t)min(max(pos[0] - a.n0, 0), a.count - 1) * D, qh + (size_t)min(max(pos[1] - a.n0, 0), a.count - 1) * D, t);
  }
  float o[16][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
  unroll<16>([&](auto n) { unroll<4>([&](auto e) { o[n][e] = 0.f; }); });

  // ---- BF16 blocks (split piece 0): the sink's two (gb > 0), the own group's first and, if a row reaches it, second
  const int z = blockIdx.z, nsplit = gridDim.z;
  if (z == 0) {
    const size_t eh = (size_t)bh * G * D, ih = (size_t)bh * a.count * D;
    const uint16_t *sink_k = a.c.exact_sink_k + eh, *sink_v = a.c.exact_sink_v + eh;
    const uint16_t *open_k = a.c.exact_open_k + eh, *open_v = a.c.exact_open_v + eh;
    const uint16_t *in_k = a.k + ih, *in_v = a.v + ih;
    const int nsink = gb > 0 ? 2 : 0, nb_all = nsink + 1 + (min(n1 - 1, gb * G + G - 1) >= gb * G + 64);
    auto t0_of = [&](int nb) { return nb < nsink ? 64 * nb : gb * G + 64 * (nb - nsink); };
    auto load = [&](int nb) {  // block nb into slot nb & 1, 8 chunks a thread
      const int t0 = t0_of(nb), c = tid & 15;
      uint8_t *sl8 = smem + SLOT * (nb & 1);
      unroll<8>([&](auto k) {
        constexpr int X = decltype(k)::value >> 2;
        const int tk = (tid >> 4) + 16 * (decltype(k)::value & 3), pt = min(t0 + tk, n1 - 1);
        const uint16_t *src = token_row(X ? sink_v : sink_k, X ? open_v : open_k, X ? in_v : in_k, a.n0, pt) + 8 * c;
#ifdef MLOPS_FBLA_TC80
        cp_async16(smem_u32(sl8) + X * KT + bf_off(tk, c), src);
#else
        *reinterpret_cast<uint4 *>(sl8 + X * KT + bf_off(tk, c)) = __ldg(reinterpret_cast<const uint4 *>(src));
#endif
      });
#ifdef MLOPS_FBLA_TC80
      cp_async_commit();
#endif
    };
    load(0);
    for (int nb = 0; nb < nb_all; ++nb) {
      if (nb + 1 < nb_all) load(nb + 1);
#ifdef MLOPS_FBLA_TC80
      if (nb + 1 < nb_all) cp_async_wait<1>();
      else cp_async_wait<0>();
#endif
      __syncthreads();
      const int t0 = t0_of(nb);
      int tlim[2];
      unroll<2>([&](auto r) { tlim[r] = (valid[r] ? (nb < nsink ? INT_MAX / 2 : pos[r]) : -1) - t0; });
      if (__any_sync(FULL, tlim[0] >= 0 || tlim[1] >= 0)) {
        float s[8][4], mb[2], sp[2], w[2];
#ifdef MLOPS_FBLA_TC80
        exact_scores(s, q, s0 + SLOT * (nb & 1), lane);
#else
        exact_scores(s, q, smem + SLOT * (nb & 1), lane);
#endif
        exact_p(s, sl, tlim, t, mb, sp);
        exact_weight(mb, sp, m, l, o, w);
#ifdef MLOPS_FBLA_TC80
        exact_pv(o, s, s0 + SLOT * (nb & 1) + KT, w, lane);
#else
        exact_pv(o, s, smem + SLOT * (nb & 1) + KT, w, lane);
#endif
      }
      __syncthreads();
    }
  }

  // ---- closed groups 1 .. gb - 1, ascending; piece z: groups 1 + i0 .. i0 + ng ---------------------------------------
  const int nall = max(0, gb - 1), i0 = (int)((long long)nall * z / nsplit), ng = (int)((long long)nall * (z + 1) / nsplit) - i0;
  if (ng) {
    const uint8_t *img0 = tiles + bh * hstep + (1 + i0 - toff) * gstep;
    const uint32_t ko = k_lane_off(lane), vo = v_lane_off(lane);
    const bool on[2] = {valid[0], valid[1]};
#ifdef MLOPS_FBLA_TC80
    if (warp < min(ng, 2)) copy_group(s0 + SLOT * warp, img0 + warp * gstep, warp + 1 < ng ? img0 + (warp + 1) * gstep : nullptr, lane);
#else
    for (int i = tid; i < min(ng, 2) * SLOT / 16; i += THREADS)
      reinterpret_cast<uint4 *>(smem)[i] = __ldg(reinterpret_cast<const uint4 *>(img0 + (i / (SLOT / 16)) * gstep) + i % (SLOT / 16));
    __syncthreads();
    auto meta_pf = [&](int jk, int jv) {  // KM of group jk, VM of jv into L1 (6 + 4 lines)
      if (lane < 6 && jk < ng) prefetch_l1(img0 + jk * gstep + O_KM + 128 * lane);
      else if (lane >= 6 && lane < 10 && jv < ng) prefetch_l1(img0 + jv * gstep + O_VM + 128 * (lane - 6));
    };
    meta_pf(1, 0);
#endif
    uint32_t qa[4][4];
    float sq[2], qb[2];
    quantize(q, img0 + O_KM + 48 * t, sl, qa, sq, qb);
    for (int j = 0; j < ng; ++j) {
      const uint8_t *im = img0 + j * gstep;
      const uint32_t kt = s0 + SLOT * (j & 1);
#ifdef MLOPS_FBLA_TC80
      const uint8_t *kmn = smem + SLOT * (j & 1) + O_KM, *vmj = smem + SLOT * (j & 1) + O_VM;  // (KM: stale at the last)
      if (warp == (j & 7)) {
        cp_async_wait<0>();
        warp_sync();
        unroll<7>([&](auto i) { bar_arrive<64>(7u * (j & 1) + decltype(i)::value); });
      } else {
        bar_sync<64>(ready_id(warp, j));
      }
#else
      const uint8_t *kmn = j + 1 < ng ? im + gstep + O_KM : im + O_KM, *vmj = im + O_VM;
      meta_pf(j + 2, j + 1);
#endif
      closed_group(o, m, l, q, sl, kmn + 48 * t, vmj + 16 * t, kt, kt + O_V, ko, vo, on, qa, sq, qb);
#ifdef MLOPS_FBLA_TC80
      if (j + 2 < ng) {
        const uint32_t e = BAR_EMPTY + (j & 1);
        if (warp == ((j + 2) & 7)) {
          bar_sync<THREADS>(e);
          copy_group(kt, im + 2 * gstep, j + 3 < ng ? im + 3 * gstep : nullptr, lane);
        } else if (warp == (j & 7)) {
          bar_sync<THREADS>(e);
        } else {
          bar_arrive<THREADS>(e);
        }
      }
#else
      if (j + 1 < ng) __syncthreads();  // group j + 1 published, slot j & 1 free
      if (j + 2 < ng) {
        uint4 x[SLOT / 16 / THREADS];
        unroll<SLOT / 16 / THREADS>([&](auto k) { x[k] = __ldg(reinterpret_cast<const uint4 *>(im + 2 * gstep) + tid + THREADS * k); });
        unroll<SLOT / 16 / THREADS>([&](auto k) { reinterpret_cast<uint4 *>(smem + SLOT * (j & 1))[tid + THREADS * k] = x[k]; });
      }
#endif
    }
  }

  // ---- out: o / l in BF16 (lane t: channels 8 n + 2 t, + 1), or the piece's (m, l), o / l in FP32 ------------------------
  unroll<2>([&](auto r) {
    if (!valid[r]) return;
    const float inv = l[r] > 0.f ? 1.f / l[r] : 0.f;
    const int i = pos[r] - a.n0;
    if (nsplit == 1) {
      uint32_t *dst = reinterpret_cast<uint32_t *>(a.out + ((size_t)bq * a.count + i) * D) + t;
      unroll<16>([&](auto n) { dst[4 * n] = to_bf16(o[n][2 * r] * inv) | (uint32_t)to_bf16(o[n][2 * r + 1] * inv) << 16; });
    } else {
      const size_t pr = ((size_t)z * gridDim.y + bq) * a.count + i;  // [piece][q head][position]
      if (t == 0) part_ml[pr] = make_float2(m[r], l[r]);
      float2 *dst = reinterpret_cast<float2 *>(part_o + pr * D) + t;
      unroll<16>([&](auto n) { dst[4 * n] = make_float2(o[n][2 * r] * inv, o[n][2 * r + 1] * inv); });
    }
  });
#endif
}

// Split pieces merged: o = sum_s w_s o_s / sum_s w_s, w_s = l_s 2^(m_s - M); 8 channels a thread, nsplit <= 8
template <int = 0>
__global__ void __launch_bounds__(256) merge_kernel(const PrefillArgs a, const float2 *part_ml, const float *part_o, int nsplit) {
  constexpr int MAXS = 8;
  const int heads = gridDim.y, bq = blockIdx.y, i = blockIdx.x * 16 + (threadIdx.x >> 4), c = threadIdx.x & 15;
  if (i >= a.count) return;
  const size_t hs = (size_t)heads * a.count, r0 = (size_t)bq * a.count + i;
  float2 ml[MAXS];
  unroll<MAXS>([&](auto s) { ml[s] = s < nsplit ? part_ml[r0 + s * hs] : make_float2(-INFINITY, 0.f); });
  float M = ml[0].x;
  unroll<MAXS - 1>([&](auto s) { M = fmaxf(M, ml[s + 1].x); });
  float acc[8] = {}, L = 0.f;
  unroll<MAXS>([&](auto s) {
    const float w = ml[s].y > 0.f ? ml[s].y * ex2(ml[s].x - M) : 0.f;
    if (w > 0.f) {
      const float4 *p = reinterpret_cast<const float4 *>(part_o + (r0 + s * hs) * D) + 2 * c;
      const float4 x0 = p[0], x1 = p[1];
      const float x[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
      unroll<8>([&](auto k) { acc[k] = fmaf(w, x[k], acc[k]); });
    }
    L += w;
  });
  const float inv = L > 0.f ? 1.f / L : 0.f;
  uint32_t ow[4];
  unroll<4>([&](auto k) { ow[k] = to_bf16(acc[2 * k] * inv) | (uint32_t)to_bf16(acc[2 * k + 1] * inv) << 16; });
  reinterpret_cast<uint4 *>(a.out + r0 * D)[c] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
}

}  // namespace tc
}  // namespace mlops_fbla_detail
#endif
