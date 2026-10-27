// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
#ifndef MLOPS_FBLA_COMMON_CUH
#define MLOPS_FBLA_COMMON_CUH

#include "mlops_fbla.h"
#include <cstdint>

namespace mlops_fbla_detail {

// n / d for 0 <= n < 2^31 and a fixed d >= 1 as a multiply: (umulhi(n, m) + n) >> l with l = ceil(log2 d) and
// m = 2^32 (2^l - d) / d + 1 (Granlund, Montgomery)
struct FastDiv {
  uint32_t m, l;
  static FastDiv of(int d) {
    uint32_t l = 0;
    while ((1ull << l) < (uint64_t)d) ++l;
    return {(uint32_t)(((1ull << 32) * ((1ull << l) - (uint64_t)d)) / (uint64_t)d + 1), l};
  }
  __device__ __forceinline__ int div(int n) const { return (int)((__umulhi((uint32_t)n, m) + (uint32_t)n) >> l); }
};

// Queries per (virtual) KV head of the decode for r q heads per KV head: r itself for 1, 2, 4; else the largest of 4, 2,
// 1 that divides r (r / that virtual heads per KV head, Args.vg)
inline int decode_group(int r) { return r == 1 || r == 2 || r == 4 ? r : r % 4 == 0 ? 4 : r % 2 == 0 ? 2 : 1; }

// One decode: coarse, select and refine launches.  Warps of 8 per CTA.
struct Args {
  mlops_fbla_cache c;
  const uint16_t *q;
  uint16_t *out;
  float *lse;      // [rows, nblocks]: log2 mass of every closed sub-block (coarse scores)
  float *part;     // [rows, ctas, 2 + 128]: coarse (m, l, o) of every coarse CTA
  float *dpart;    // [rows, ctas, 1 + 128]: corrections (dl, do) of every refine CTA
  float *row_lse;  // [rows]: log2 sum of the row's coarse mass, written by the select pass
  int *items;      // [batch * kv_heads * vg, nslices * SLICE]: refined sub-blocks (sub-block << 4 | query mask)
  int *counts;     // [batch * kv_heads * vg, nslices]: items of every slice
  const int *length;  // null, or the cache length in device memory: the length is *length + tokens
  int batch, q_heads, kv_heads, tokens, nblocks, nslices;  // nblocks, nslices: the scratch's layout
  int hs, gs;      // units (a group's records, metadata) between consecutive KV heads and groups
  int ctas, rctas; // coarse and refine CTAs per (batch, KV head)
  float scale_log2, log2_tau;
  // More than 4 queries per KV head: the passes see vg virtual KV heads per head, vh = bh vg + j, each with the queries
  // j QG' .. + QG' - 1 of its head (QG' = 4, 2 or 1); the items, counts and rows are the virtual heads', the cache the
  // head's (vdiv: / vg).  vg 1 otherwise.
  int vg;
  FastDiv vdiv;
};

// One prefill (fbla_prefill.cuh): close, stream and finish launches.
struct PrefillArgs {
  mlops_fbla_cache c;
  const uint16_t *q;      // [batch, q_heads, count, 128] BF16, or null (append only)
  const uint16_t *k, *v;  // [batch, kv_heads, count, 128] BF16: the chunk's tokens
  uint16_t *out;          // [batch, q_heads, count, 128] BF16
  const int *length;  // null, or a length in device memory: the chunk starts at *length + n0 (close_kernel,
                      // finish_kernel; the attention kernels run with null only)
  int batch, q_heads, kv_heads, qg, n0, count;
  int hs, gs;  // units between consecutive KV heads and groups, as Args
  float scale_log2;
};

// The cache length, read where it is used (a parameter, or one load of the device length through Pos::ro -- the
// position policy's, fbla_kernels.cuh): not kept live through the kernels' loops.
template <class Pos> __device__ __forceinline__ int tokens_of(const Args &a) { return a.length ? Pos::ro(a.length) + a.tokens : a.tokens; }
template <class Pos> __device__ __forceinline__ int n0_of(const PrefillArgs &a) { return a.length ? Pos::ro(a.length) + a.n0 : a.n0; }

// instantiated per format in fbla_inst.cu (one object each); fbla.cu sees only these declarations
template <int CB, int FB> cudaError_t launch(const Args &, int bound, cudaStream_t);
template <int CB, int FB> cudaError_t launch_prefill(const PrefillArgs &, cudaStream_t);
template <int CB, int FB> cudaError_t launch_append(const PrefillArgs &, cudaStream_t);
struct QkvArgs;
template <int CB, int FB> cudaError_t launch_qkv(const QkvArgs &, cudaStream_t);
bool prefill_tc_device();
// For a program that runs the passes' items itself (fbla_kernels.cuh, fbla_qkv.cuh): the Args of a decode and the
// QkvArgs of a step's input, checked and prepared as mlops_fbla_decode_at / mlops_fbla_qkv_at prepare them -- the layout
// of cfg->capacity (> 0), the length *length + cfg->tokens or, length null, cfg->tokens (Args.tokens, QkvArgs.p.n0: the
// program sets them per step).  decode_args: the coarse and refine passes over per_head (>= 1) CTAs per KV head, the
// scratch (decode_scratch_bytes of the same per_head) laid out for them.  Errors: mlops_fbla_last_error().
size_t decode_scratch_bytes(const mlops_fbla_config *cfg, int per_head);
mlops_fbla_status decode_args(const mlops_fbla_config *cfg, const int *length, const uint16_t *q, const mlops_fbla_cache *cache,
                              uint16_t *output, void *scratch, size_t scratch_bytes, int per_head, Args &a);
mlops_fbla_status qkv_args(const mlops_fbla_config *cfg, const int *length, const mlops_fbla_token *token, const mlops_fbla_cache *cache,
                           QkvArgs &a);  // the prefill's attention runs on tensor cores here (any head ratio), else on DP4A (1, 2, 4)
// mlops_fbla_cache.tc_images: a slot per unit (head and group, by hs and gs): the image of IMAGE_BYTES
// (tensors/transcode.cuh TGB), then a uint32 flag (nonzero: the image is the group's records'), 128-byte aligned
constexpr size_t IMAGE_BYTES = 34048, IMAGE_SLOT = IMAGE_BYTES + 128;
__host__ __device__ inline uint32_t *image_flag(const mlops_fbla_cache &c, size_t unit) {
  return reinterpret_cast<uint32_t *>(c.tc_images + unit * IMAGE_SLOT + IMAGE_BYTES);
}

__device__ __forceinline__ float bf16(uint16_t x) { return __uint_as_float((uint32_t)x << 16); }
__device__ __forceinline__ uint16_t to_bf16(float x) {
  uint32_t u = __float_as_uint(x);
  if ((u & 0x7fffffffu) > 0x7f800000u) u = 0x7fc00000u;
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

}  // namespace mlops_fbla_detail
#endif
