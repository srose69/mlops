// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
#include "common.cuh"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>

using mlops_fbla_detail::Args;

namespace {
thread_local char error_buf[256] = {0};
mlops_fbla_status fail(mlops_fbla_status st, const char *fmt, ...) {
  va_list ap; va_start(ap, fmt); vsnprintf(error_buf, sizeof(error_buf) - 1, fmt, ap); va_end(ap); return st;
}
mlops_fbla_status ok() { error_buf[0] = 0; return MLOPS_FBLA_OK; }
bool aligned(const void *p, size_t n) { return (reinterpret_cast<uintptr_t>(p) & (n - 1)) == 0; }
size_t up256(size_t n) { return (n + 255) & ~(size_t)255; }

int sm_count() {
  static int cached[64] = {0};
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0) return 20;
  if (dev < 64 && cached[dev]) return cached[dev];
  int n = 20;
  cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  if (dev < 64) cached[dev] = n;
  return n;
}

// Scratch carve-up: CTA counters first and of a size independent of the context (a zero-filled scratch
// serves any length), then the per-call arrays.  CTAs per (batch, KV head) are at most 8 per SM over the heads.
constexpr int SLICE = 1024, MAX_SLICES = 4096;  // fbla_kernels.cuh
struct Geometry {
  int groups, nblocks, nslices, ctas;
  size_t counter, lse, part, dpart, row_lse, counts, items, bytes;
};
Geometry geometry(const mlops_fbla_config *c) {
  Geometry g{};
  const size_t heads = (size_t)c->batch * c->kv_heads, rows = (size_t)c->batch * c->q_heads;
  g.groups = (c->tokens + MLOPS_FBLA_GROUP - 1) / MLOPS_FBLA_GROUP;
  g.nblocks = g.groups * (MLOPS_FBLA_GROUP / MLOPS_FBLA_SUBBLOCK);
  g.nslices = std::max(1, (g.nblocks + SLICE - 1) / SLICE);
  g.ctas = (int)std::max<size_t>(1, (8 * (size_t)sm_count() + heads - 1) / heads);
  g.counter = 0;
  g.lse = up256(heads * sizeof(int));
  g.part = g.lse + up256(rows * g.nblocks * sizeof(float));
  g.dpart = g.part + up256(rows * g.ctas * (MLOPS_FBLA_HEAD_DIM + 2) * sizeof(float));
  g.row_lse = g.dpart + up256(rows * g.ctas * (MLOPS_FBLA_HEAD_DIM + 1) * sizeof(float));
  g.counts = g.row_lse + up256(rows * sizeof(float));
  g.items = g.counts + up256(heads * g.nslices * sizeof(int));
  g.bytes = g.items + up256(heads * g.nslices * SLICE * sizeof(int));
  return g;
}
bool valid_shape(const mlops_fbla_config *c) {
  if (!c || c->batch <= 0 || c->q_heads <= 0 || c->kv_heads <= 0 || c->tokens <= 0) return false;
  if (c->q_heads % c->kv_heads) return false;
  if (c->capacity != 0 && c->capacity < c->tokens) return false;
  if (c->tokens > MAX_SLICES * SLICE * MLOPS_FBLA_SUBBLOCK) return false;
  const int r = c->q_heads / c->kv_heads;
  return r == 1 || r == 2 || r == 4;
}
}  // namespace

extern "C" size_t mlops_fbla_decode_scratch_bytes(const mlops_fbla_config *c) {
  return valid_shape(c) ? geometry(c).bytes : 0;
}

extern "C" mlops_fbla_status mlops_fbla_decode(const mlops_fbla_config *c, const uint16_t *q,
                                                 const mlops_fbla_cache *k, uint16_t *out, void *scratch,
                                                 size_t scratch_bytes, cudaStream_t stream) {
  if (!c || !k || !q || !out || !scratch) return fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_decode: null argument");
  if (!valid_shape(c))
    return fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_decode: batch, heads and tokens must be positive, "
                "q_heads / kv_heads 1, 2 or 4, tokens <= 2^26, and capacity 0 or >= tokens");
  if (c->format != MLOPS_FBLA_3P3 && c->format != MLOPS_FBLA_4P2)
    return fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_decode: invalid format");
  if (!(c->scale >= 0.f) || !(c->tau >= 0.f))
    return fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_decode: scale and tau must be nonnegative");
  if (!k->coarse || !k->fine || !k->meta || !k->exact_sink_k || !k->exact_sink_v || !k->exact_open_k || !k->exact_open_v)
    return fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_decode: every cache pointer is required");
  const void *ptrs[] = {q, out, scratch, k->coarse, k->fine, k->meta, k->exact_sink_k, k->exact_sink_v,
                        k->exact_open_k, k->exact_open_v};
  for (const void *p : ptrs)
    if (!aligned(p, 16)) return fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_decode: pointers must be 16-byte aligned");
  const Geometry g = geometry(c);
  if (scratch_bytes < g.bytes)
    return fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_decode: scratch has %zu bytes, need %zu", scratch_bytes, g.bytes);

  Args a{};
  a.c = *k; a.q = q; a.out = out;
  char *base = static_cast<char *>(scratch);
  a.lse = reinterpret_cast<float *>(base + g.lse);
  a.part = reinterpret_cast<float *>(base + g.part);
  a.dpart = reinterpret_cast<float *>(base + g.dpart);
  a.counter = reinterpret_cast<int *>(base + g.counter);
  a.row_lse = reinterpret_cast<float *>(base + g.row_lse);
  a.counts = reinterpret_cast<int *>(base + g.counts);
  a.items = reinterpret_cast<int *>(base + g.items);
  a.batch = c->batch; a.q_heads = c->q_heads; a.kv_heads = c->kv_heads; a.tokens = c->tokens;
  a.groups = g.groups; a.nblocks = g.nblocks; a.nslices = g.nslices;
  a.cgroups = c->capacity ? (c->capacity + MLOPS_FBLA_GROUP - 1) / MLOPS_FBLA_GROUP : g.groups;
  const float scale = c->scale == 0.f ? 1.f / sqrtf((float)MLOPS_FBLA_HEAD_DIM) : c->scale;
  a.scale_log2 = scale * 1.4426950408889634f;
  const float tau = c->tau != 0.f ? c->tau : c->format == MLOPS_FBLA_3P3 ? 3e-4f : 1e-3f;
  a.log2_tau = log2f(tau);
  const cudaError_t e = c->format == MLOPS_FBLA_3P3 ? mlops_fbla_detail::launch<3, 3>(a, g.ctas, stream)
                                                     : mlops_fbla_detail::launch<4, 2>(a, g.ctas, stream);
  if (e != cudaSuccess) return fail(MLOPS_FBLA_CUDA_ERROR, "mlops_fbla_decode: %s", cudaGetErrorString(e));
  return ok();
}

extern "C" const char *mlops_fbla_status_string(mlops_fbla_status s) {
  switch (s) { case MLOPS_FBLA_OK: return "ok"; case MLOPS_FBLA_INVALID_VALUE: return "invalid value";
               case MLOPS_FBLA_CUDA_ERROR: return "cuda error"; default: return "unknown"; }
}
extern "C" const char *mlops_fbla_last_error(void) { return error_buf; }
