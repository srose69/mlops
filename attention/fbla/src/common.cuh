// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
#ifndef MLOPS_FBLA_COMMON_CUH
#define MLOPS_FBLA_COMMON_CUH

#include "mlops_fbla.h"
#include <cstdint>

namespace mlops_fbla_detail {

// One decode: coarse, select and refine launches.  Warps of 8 per CTA.
struct Args {
  mlops_fbla_cache c;
  const uint16_t *q;
  uint16_t *out;
  float *lse;      // [rows, nblocks]: log2 mass of every closed sub-block (coarse scores)
  float *part;     // [rows, ctas, 2 + 128]: coarse (m, l, o) of every coarse CTA
  float *dpart;    // [rows, ctas, 1 + 128]: corrections (dl, do) of every refine CTA
  int *counter;    // [batch * kv_heads]: refine CTAs finished
  float *row_lse;  // [rows]: log2 sum of the row's coarse mass, written by the select pass
  int *items;      // [batch * kv_heads, nslices * SLICE]: refined sub-blocks (sub-block << 4 | query mask)
  int *counts;     // [batch * kv_heads, nslices]: items of every slice
  int batch, q_heads, kv_heads, tokens, groups, nblocks, nslices;
  int cgroups;     // groups of the cache's capacity: the per-head stride of the planes and the metadata
  int ctas;        // coarse CTAs per (batch, KV head)
  float scale_log2, log2_tau;
};

template <int CB, int FB> cudaError_t launch(const Args &, int bound, cudaStream_t);

__device__ __forceinline__ float bf16(uint16_t x) { return __uint_as_float((uint32_t)x << 16); }
__device__ __forceinline__ uint16_t to_bf16(float x) {
  uint32_t u = __float_as_uint(x);
  if ((u & 0x7fffffffu) > 0x7f800000u) u = 0x7fc00000u;
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

}  // namespace mlops_fbla_detail
#endif
