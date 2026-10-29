// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The embedding element (include/mlops_embed.h). A unit is 256 elements of one column on one warp, 16 bytes a
// lane; the units of the call are split over the elements in order and over a CTA's warps round robin. A unit's
// outputs do not depend on the split.
#pragma once
#include "mlops_embed.h"

namespace mlops_vocab_detail {

struct EmbedOp {
  const uint16_t *table;  // [vocab][hidden], device or mapped host memory
  const int *ids;         // [n]
  const uint16_t *w;      // [hidden], or null
  float *x;               // [n][hidden], or null
  uint16_t *xw;           // [n][hidden], or null
  float *parts;           // [n][hidden / 256], or null
  int *error;             // or null
  int vocab, hidden, n, items;
};

// bf16(x) rounded to nearest even, NaN canonical.
__device__ __forceinline__ unsigned short round_bf16(float x) {
  unsigned u = __float_as_uint(x);
  if ((u & 0x7fffffffu) > 0x7f800000u) u = 0x7fc00000u;
  return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// Element `item` of op.items on a CTA of W warps; all of its threads call it.
template <int W>
__device__ __forceinline__ void embed_item(const EmbedOp &op, int item, unsigned tid) {
  const int warp = tid >> 5, lane = tid & 31, per_col = op.hidden / 256;
  const long long total = (long long)op.n * per_col;
  const long long lo = item * total / op.items, hi = (item + 1) * total / op.items;
  for (long long u = lo + warp; u < hi; u += W) {
    const int col = (int)(u / per_col), k0 = (int)(u % per_col) * 256;
    const int id = __ldcg(op.ids + col);
    const bool ok = (unsigned)id < (unsigned)op.vocab;
    if (!ok && lane == 0 && op.error) *op.error = 1;
    uint4 t = make_uint4(0u, 0u, 0u, 0u);
    if (ok) t = __ldcg(reinterpret_cast<const uint4 *>(op.table + (size_t)id * op.hidden + k0 + 8 * lane));
    const unsigned ts[4] = {t.x, t.y, t.z, t.w};
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = __uint_as_float((ts[e / 2] << (16 * (1 - (e & 1)))) & 0xffff0000u);
    const size_t at = (size_t)col * op.hidden + k0 + 8 * lane;
    if (op.x) {
      reinterpret_cast<float4 *>(op.x + at)[0] = make_float4(v[0], v[1], v[2], v[3]);
      reinterpret_cast<float4 *>(op.x + at)[1] = make_float4(v[4], v[5], v[6], v[7]);
    }
    if (op.xw && op.w) {
      const uint4 wq = __ldg(reinterpret_cast<const uint4 *>(op.w + k0 + 8 * lane));
      const unsigned ws[4] = {wq.x, wq.y, wq.z, wq.w};
      unsigned o[4];
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        const float w0 = __uint_as_float((ws[e / 2] << 16) & 0xffff0000u), w1 = __uint_as_float(ws[e / 2] & 0xffff0000u);
        o[e / 2] = (unsigned)round_bf16(__fmul_rn(v[e], w0)) | ((unsigned)round_bf16(__fmul_rn(v[e + 1], w1)) << 16);
      }
      *reinterpret_cast<uint4 *>(op.xw + at) = make_uint4(o[0], o[1], o[2], o[3]);
    }
    if (op.parts) {
      float sq = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) sq = __fadd_rn(sq, __fmul_rn(v[e], v[e]));
#pragma unroll
      for (int o = 16; o; o >>= 1) sq = __fadd_rn(sq, __shfl_xor_sync(0xffffffffu, sq, o));
      if (!lane) op.parts[(size_t)col * per_col + k0 / 256] = sq;
    }
  }
}

}  // namespace mlops_vocab_detail
