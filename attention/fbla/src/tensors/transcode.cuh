// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// Closed groups' records -> attention images (full 6-bit codes, metadata in consumption order): one CTA per (group,
// KV head), warp j decodes sub-block j into shared memory, 16-byte coalesced stores.  sm_75+.
//
// Image (TGB = 34048 bytes, sections on 128):
//   K   [0, 16384)      [128 rows][128 B].  Token 16 b + tau at row 16 b + k_row(tau) (QK's accumulators of lane t
//                       are tokens 4t .. + 3: PV's A fragment as it is); row chunk c = channels 16 c + KPERM (lane t's
//                       s8 k positions = its q register channels); chunk c stored at c ^ k_swz(row).
//   V   [16384, 32768)  4 token quarters of [128 channel rows][32 B] (PV's B fragments), chunk j at j ^ v_swz(row).
//   KM  [32768, 33536)  record (ks, t) at 48 (4 ks + t): FP32 steps of channels 32 ks + 8 j + 2 t + e, then BF16 bases.
//   VM  [33536, 34048)  record (cb, t) at 16 (4 cb + t): BF16 (base, step) of channels 16 cb + 2 t, + 1, + 8, + 9.
// The swizzles' row-block parts are constant: an ldmatrix address is a lane offset ^ a k-step constant + an immediate.
#ifndef MLOPS_FBLA_TENSORS_TRANSCODE_CUH
#define MLOPS_FBLA_TENSORS_TRANSCODE_CUH

#include "mma.cuh"

namespace mlops_fbla_detail {
namespace tc {

constexpr int KT = 128 * 128, VT = 128 * 128, VQ = 32 * 128;  // K, V tiles; a V quarter
constexpr int KMB = 16 * 48, VMB = 32 * 16;                    // K, V metadata
constexpr int O_V = KT, O_KM = KT + VT, O_VM = O_KM + KMB, TGB = O_VM + VMB;
static_assert(TGB == 34048 && TGB == IMAGE_BYTES && TGB % 128 == 0 && O_KM % 128 == 0 && O_VM % 128 == 0, "the image's sections on 128 bytes");
constexpr int XT = 256;  // transcoder threads: a warp per sub-block

__host__ __device__ constexpr int k_row(int tau) { return ((tau & 2) << 2) + 2 * (tau >> 2) + (tau & 1); }
__host__ __device__ constexpr int k_swz(int row) { return (row ^ (row >> 1)) & 7; }
__host__ __device__ constexpr int v_swz(int row) { return (row >> 2) & 1; }
// the ldmatrix lane offsets: row r' = 8 (lane / 16) + lane % 8 of a 16-row block, logical chunk lane / 8 % 2
__host__ __device__ constexpr uint32_t k_lane_off(int lane) {
  return (((lane >> 4) << 3) | (lane & 7)) * 128 + ((((lane >> 3) & 1) ^ k_swz(((lane >> 4) << 3) | (lane & 7))) << 4);
}
__host__ __device__ constexpr uint32_t v_lane_off(int lane) {
  return (((lane >> 4) << 3) | (lane & 7)) * 32 + ((((lane >> 3) & 1) ^ v_swz(((lane >> 4) << 3) | (lane & 7))) << 4);
}

// The V slot of (channel c of a lane's four, token quad u)
template <int BITS> __host__ __device__ constexpr int v_slot_of(int c, int u) {
  int s = 0;
  while (v_channel<BITS>(s) != c || v_quad<BITS>(s) != u) ++s;
  return s;
}
__device__ __forceinline__ uint2 ld64(const uint8_t *p) { return __ldg(reinterpret_cast<const uint2 *>(p)); }
// a record lane's words: p = record + 16 l; p3 (3-bit low words) = record + 16 (l % 16) + 8 (l / 16) (K), + 8 l (V)
template <int BITS> __device__ __forceinline__ void rec_k(Codes<BITS> &c, const uint8_t *p, const uint8_t *p3) {
  const uint4 a = ld128<Grid>(p);
  c.w[0] = a.x; c.w[1] = a.y; c.w[2] = a.z; c.w[3] = a.w;
  if constexpr (BITS == 4) { const uint4 b = ld128<Grid>(p + 512); c.w[4] = b.x; c.w[5] = b.y; c.w[6] = b.z; c.w[7] = b.w; }
  if constexpr (BITS == 3) { const uint2 b = ld64(p3 + 1024); c.w[4] = b.x; c.w[5] = b.y; }
}
template <int BITS> __device__ __forceinline__ void rec_v(Codes<BITS> &c, const uint8_t *p, const uint8_t *p3) {
  const uint4 a = ld128<Grid>(p + (BITS == 4 ? 1024 : 512));
  c.w[0] = a.x; c.w[1] = a.y; c.w[2] = a.z; c.w[3] = a.w;
  if constexpr (BITS == 4) { const uint4 b = ld128<Grid>(p + 1536); c.w[4] = b.x; c.w[5] = b.y; c.w[6] = b.z; c.w[7] = b.w; }
  if constexpr (BITS == 3) { const uint2 b = ld64(p3 + 1280); c.w[4] = b.x; c.w[5] = b.y; }
}

// Sub-block j's K and V (coarse record rc, fine rf) into the shared image.  K lane l: token 16 j + l % 16, chunk cc
// of its channel half = slots 4 cc, + 2, + 1, + 3 in KPERM order (two PRMT a word; stores conflict-free).  V lane l:
// channels 4 l + cc, row 4 l + cc of quarter j / 2.
template <int CB, int FB>
__device__ __forceinline__ void decode_sub_block(uint8_t *img, const uint8_t *rc, const uint8_t *rf, int j, int l) {
  const uint8_t *pc = rc + 16 * l, *pf = rf + 16 * l;
  const int k3 = 16 * (l & 15) + 8 * (l >> 4);  // (3-bit planes only)
  {
    Codes<CB> c; Codes<FB> f;
    rec_k<CB>(c, pc, rc + k3); rec_k<FB>(f, pf, rf + k3);
    uint32_t kc[16];
    unroll<16>([&](auto s) { kc[s] = k_code<CB, FB>(c, f, s); });
    const int row = 16 * j + k_row(l & 15), hh = l >> 4;
    uint8_t *base = img + row * 128;
    unroll<4>([&](auto cc) {
      const uint32_t a = kc[4 * cc], b = kc[4 * cc + 2], c2 = kc[4 * cc + 1], d = kc[4 * cc + 3];
      *reinterpret_cast<uint4 *>(base + (((4 * hh + cc) ^ k_swz(row)) << 4)) =
          make_uint4(prmt(a, b, 0x5410u), prmt(a, b, 0x7632u), prmt(c2, d, 0x5410u), prmt(c2, d, 0x7632u));
    });
  }
  {
    Codes<CB> c; Codes<FB> f;
    rec_v<CB>(c, pc, rc + 8 * l); rec_v<FB>(f, pf, rf + 8 * l);
    uint8_t *base = img + O_V + (j >> 1) * VQ + 128 * l + (((j ^ l) & 1) << 4);
    unroll<4>([&](auto cc) {
      constexpr int CC = decltype(cc)::value;
      *reinterpret_cast<uint4 *>(base + 32 * CC) =
          make_uint4(v_code<CB, FB>(c, f, v_slot_of<CB>(CC, 0)), v_code<CB, FB>(c, f, v_slot_of<CB>(CC, 1)),
                     v_code<CB, FB>(c, f, v_slot_of<CB>(CC, 2)), v_code<CB, FB>(c, f, v_slot_of<CB>(CC, 3)));
    });
  }
}

// Metadata chunk i (0 .. 79) from the cache metadata mw (256 words): i < 32 KM steps (FP32) of record i % 16, chunk
// i / 16; i < 48 its bases; then VM record i - 48
__device__ __forceinline__ uint4 meta_chunk(const uint32_t *mw, int i) {
  const int r = i & 15, c = i < 32 ? 32 * (r >> 2) + 16 * (i >> 4) + 2 * (r & 3)  // KM steps, j = 2 p, 2 p + 1
                          : i < 48 ? 32 * (r >> 2) + 2 * (r & 3)                   // KM bases, j = 0 .. 3
                                   : 128 + 16 * ((i - 48) >> 2) + 2 * (i & 3);     // VM (cb, t)
  const uint32_t *p = mw + 4 * (c >> 2) + ((c >> 1) & 1);
  if (i < 32) {
    const uint32_t s0 = __ldg(p + 2), s1 = __ldg(p + 10);
    return make_uint4(s0 << 16, s0 & 0xffff0000u, s1 << 16, s1 & 0xffff0000u);
  }
  if (i < 48) return make_uint4(__ldg(p), __ldg(p + 8), __ldg(p + 16), __ldg(p + 24));
  return make_uint4(__ldg(p), __ldg(p + 2), __ldg(p + 8), __ldg(p + 10));
}

// Group g = 1 + blockIdx.x of KV head blockIdx.y to tiles + bh hstep + (g - toff) gstep.  flagged (persistent images:
// the slot's word after the image; close_kernel clears it): a group whose flag is set is skipped, a transcoded one sets it.
template <int CB, int FB>
__global__ void __launch_bounds__(XT) transcode_kernel(const PrefillArgs a, uint8_t *tiles, size_t hstep, size_t gstep, int toff, bool flagged) {
  constexpr int RC = record_bytes<CB>(), RF = record_bytes<FB>();
  __shared__ __align__(128) uint8_t img[KT + VT];
  const int tid = threadIdx.x, lane = tid & 31, j = tid >> 5, bh = blockIdx.y, g = 1 + blockIdx.x;
  const size_t hg = (size_t)bh * a.hs + (size_t)g * a.gs;
  uint8_t *slot = tiles + bh * hstep + (g - toff) * gstep;
  uint32_t *flag = reinterpret_cast<uint32_t *>(slot + TGB);
  if (flagged && *flag) return;  // (CTA-uniform)
  uint4 *dst = reinterpret_cast<uint4 *>(slot);
  if (tid < (KMB + VMB) / 16)
    dst[O_KM / 16 + (tid < KMB / 16 ? 3 * (tid & 15) + (tid >> 4) : tid)] =
        meta_chunk(reinterpret_cast<const uint32_t *>(a.c.meta + hg * 512), tid);
  decode_sub_block<CB, FB>(img, a.c.coarse + (hg * SBG + j) * RC, a.c.fine + (hg * SBG + j) * RF, j, lane);
  __syncthreads();
  unroll<(KT + VT) / (16 * XT)>([&](auto k) { dst[tid + XT * k] = *reinterpret_cast<const uint4 *>(img + 16 * (tid + XT * k)); });
  if (flagged && tid == 0) *flag = 1u;  // (read by later launches only)
}

}  // namespace tc
}  // namespace mlops_fbla_detail
#endif
