// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
#include "common.cuh"
#include "fbla_qkv.cuh"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>

using mlops_fbla_detail::Args;
using mlops_fbla_detail::FastDiv;
using mlops_fbla_detail::decode_group;

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

// Scratch carve-up: per-call arrays only, every one written before it is read (no initial contents needed).  CTAs
// per (batch, KV head) are at most 8 per SM over the heads.
constexpr int SLICE = 1024, MAX_SLICES = 4096;  // fbla_kernels.cuh
struct Geometry {
  int groups, nblocks, nslices, ctas;
  size_t lse, part, dpart, row_lse, counts, items, bytes;
};
// per_head: the coarse and refine CTAs per (batch, virtual KV head) the partials are laid out for; 0: the device's resident
// CTAs over the virtual heads (decode_group)
Geometry geometry(const mlops_fbla_config *c, int per_head = 0) {
  Geometry g{};
  const int r = c->q_heads / c->kv_heads;
  const size_t heads = (size_t)c->batch * c->kv_heads * (r / decode_group(r)), rows = (size_t)c->batch * c->q_heads;
  g.groups = (c->tokens + MLOPS_FBLA_GROUP - 1) / MLOPS_FBLA_GROUP;
  g.nblocks = g.groups * (MLOPS_FBLA_GROUP / MLOPS_FBLA_SUBBLOCK);
  g.nslices = std::max(1, (g.nblocks + SLICE - 1) / SLICE);
  g.ctas = per_head > 0 ? per_head : (int)std::max<size_t>(1, (8 * (size_t)sm_count() + heads - 1) / heads);
  g.lse = 0;
  g.part = g.lse + up256(rows * g.nblocks * sizeof(float));
  g.dpart = g.part + up256(rows * g.ctas * (MLOPS_FBLA_HEAD_DIM + 2) * sizeof(float));
  g.row_lse = g.dpart + up256(rows * g.ctas * (MLOPS_FBLA_HEAD_DIM + 1) * sizeof(float));
  g.counts = g.row_lse + up256(rows * sizeof(float));
  g.items = g.counts + up256(heads * g.nslices * sizeof(int));
  g.bytes = g.items + up256(heads * g.nslices * SLICE * sizeof(int));
  return g;
}
// both strides given (positive) or neither
bool valid_strides(const mlops_fbla_config *c) {
  return (c->head_stride == 0 && c->group_stride == 0) || (c->head_stride > 0 && c->group_stride > 0);
}
bool valid_shape(const mlops_fbla_config *c) {
  if (!c || c->batch <= 0 || c->q_heads <= 0 || c->kv_heads <= 0 || c->tokens <= 0 || !valid_strides(c)) return false;
  if (c->q_heads % c->kv_heads) return false;
  if (c->capacity != 0 && c->capacity < c->tokens) return false;
  return c->tokens <= MAX_SLICES * SLICE * MLOPS_FBLA_SUBBLOCK;
}
}  // namespace

extern "C" size_t mlops_fbla_decode_scratch_bytes(const mlops_fbla_config *c) {
  return valid_shape(c) ? geometry(c).bytes : 0;
}

namespace {
// by_capacity false: cfg->tokens tokens, the layout theirs; true: the layout of cfg->capacity, the length *length +
// cfg->tokens (length null: cfg->tokens).  The scratch laid out for per_head (0: the device's); `bound`: its CTAs per KV head.
mlops_fbla_status prepare_decode(const char *fn, const mlops_fbla_config *c, bool by_capacity, const int *length, const uint16_t *q,
                                 const mlops_fbla_cache *k, uint16_t *out, void *scratch, size_t scratch_bytes, int per_head, Args &a,
                                 int &bound) {
  if (!c || !k || !q || !out || !scratch) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: null argument", fn);
  mlops_fbla_config lc = *c;  // the layout's
  if (by_capacity) lc.tokens = c->capacity;
  if (!valid_shape(&lc) || (by_capacity && c->tokens < 0))
    return fail(MLOPS_FBLA_INVALID_VALUE, "%s: batch, heads and tokens must be positive (with a device length: capacity "
                "positive, tokens >= 0), q_heads a multiple of kv_heads, tokens <= 2^26, capacity 0 or >= tokens, strides "
                "both 0 or both positive", fn);
  if (c->format != MLOPS_FBLA_3P3 && c->format != MLOPS_FBLA_4P2) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: invalid format", fn);
  if (!(c->scale >= 0.f) || !(c->tau >= 0.f)) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: scale and tau must be nonnegative", fn);
  if (!k->coarse || !k->fine || !k->meta || !k->exact_sink_k || !k->exact_sink_v || !k->exact_open_k || !k->exact_open_v)
    return fail(MLOPS_FBLA_INVALID_VALUE, "%s: every cache pointer is required", fn);
  const void *ptrs[] = {q, out, scratch, k->coarse, k->fine, k->meta, k->exact_sink_k, k->exact_sink_v,
                        k->exact_open_k, k->exact_open_v};
  for (const void *p : ptrs)
    if (!aligned(p, 16)) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: pointers must be 16-byte aligned", fn);
  if (!aligned(length, 4)) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: length must be 4-byte aligned", fn);
  const Geometry g = geometry(&lc, per_head);
  if (scratch_bytes < g.bytes) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: scratch has %zu bytes, need %zu", fn, scratch_bytes, g.bytes);

  a = Args{};
  a.c = *k; a.q = q; a.out = out; a.length = length;
  char *base = static_cast<char *>(scratch);
  a.lse = reinterpret_cast<float *>(base + g.lse);
  a.part = reinterpret_cast<float *>(base + g.part);
  a.dpart = reinterpret_cast<float *>(base + g.dpart);
  a.row_lse = reinterpret_cast<float *>(base + g.row_lse);
  a.counts = reinterpret_cast<int *>(base + g.counts);
  a.items = reinterpret_cast<int *>(base + g.items);
  a.batch = c->batch; a.q_heads = c->q_heads; a.kv_heads = c->kv_heads; a.tokens = c->tokens;
  a.nblocks = g.nblocks; a.nslices = g.nslices;
  a.hs = c->head_stride ? c->head_stride : c->capacity ? (c->capacity + MLOPS_FBLA_GROUP - 1) / MLOPS_FBLA_GROUP : g.groups;
  a.gs = c->group_stride ? c->group_stride : 1;
  const float scale = c->scale == 0.f ? 1.f / sqrtf((float)MLOPS_FBLA_HEAD_DIM) : c->scale;
  a.scale_log2 = scale * 1.4426950408889634f;
  const float tau = c->tau != 0.f ? c->tau : c->format == MLOPS_FBLA_3P3 ? 3e-4f : 1e-3f;
  a.log2_tau = log2f(tau);
  const int r = c->q_heads / c->kv_heads;
  a.vg = r / decode_group(r);
  a.vdiv = FastDiv::of(a.vg);
  bound = g.ctas;
  return ok();
}

mlops_fbla_status decode(const char *fn, const mlops_fbla_config *c, const int *length, const uint16_t *q, const mlops_fbla_cache *k,
                         uint16_t *out, void *scratch, size_t scratch_bytes, cudaStream_t stream) {
  Args a;
  int bound;
  const mlops_fbla_status st = prepare_decode(fn, c, length != nullptr, length, q, k, out, scratch, scratch_bytes, 0, a, bound);
  if (st != MLOPS_FBLA_OK) return st;
  const cudaError_t e = c->format == MLOPS_FBLA_3P3 ? mlops_fbla_detail::launch<3, 3>(a, bound, stream)
                                                     : mlops_fbla_detail::launch<4, 2>(a, bound, stream);
  if (e != cudaSuccess) return fail(MLOPS_FBLA_CUDA_ERROR, "%s: %s", fn, cudaGetErrorString(e));
  return ok();
}
}  // namespace

size_t mlops_fbla_detail::decode_scratch_bytes(const mlops_fbla_config *c, int per_head) {
  if (!c || c->capacity <= 0 || per_head < 1) return 0;
  mlops_fbla_config lc = *c;
  lc.tokens = c->capacity;
  return valid_shape(&lc) ? geometry(&lc, per_head).bytes : 0;
}

mlops_fbla_status mlops_fbla_detail::decode_args(const mlops_fbla_config *c, const int *length, const uint16_t *q,
                                                 const mlops_fbla_cache *k, uint16_t *out, void *scratch, size_t scratch_bytes,
                                                 int per_head, Args &a) {
  constexpr const char *fn = "decode_args";
  if (!c || c->capacity <= 0) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: capacity must be positive", fn);
  if (per_head < 1) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: per_head must be positive", fn);
  int bound;
  const mlops_fbla_status st = prepare_decode(fn, c, true, length, q, k, out, scratch, scratch_bytes, per_head, a, bound);
  if (st != MLOPS_FBLA_OK) return st;
  a.ctas = a.rctas = per_head;
  return ok();
}

extern "C" mlops_fbla_status mlops_fbla_decode(const mlops_fbla_config *c, const uint16_t *q, const mlops_fbla_cache *k,
                                                 uint16_t *out, void *scratch, size_t scratch_bytes, cudaStream_t stream) {
  return decode("mlops_fbla_decode", c, nullptr, q, k, out, scratch, scratch_bytes, stream);
}

extern "C" mlops_fbla_status mlops_fbla_decode_at(const mlops_fbla_config *c, const int *length, const uint16_t *q,
                                                    const mlops_fbla_cache *k, uint16_t *out, void *scratch, size_t scratch_bytes,
                                                    cudaStream_t stream) {
  if (!length) return fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_decode_at: null length");
  return decode("mlops_fbla_decode_at", c, length, q, k, out, scratch, scratch_bytes, stream);
}

namespace {
bool valid_prefill(const mlops_fbla_config *c, int count) {
  if (!c || c->batch <= 0 || c->q_heads <= 0 || c->kv_heads <= 0 || c->tokens < 0 || count <= 0 || !valid_strides(c)) return false;
  if (c->q_heads % c->kv_heads) return false;
  const long long n1 = (long long)c->tokens + count;
  return (c->capacity == 0 || c->capacity >= n1) && n1 <= MAX_SLICES * SLICE * MLOPS_FBLA_SUBBLOCK;
}
}  // namespace

extern "C" mlops_fbla_status mlops_fbla_prefill(const mlops_fbla_config *c, int count, const uint16_t *q,
                                                  const uint16_t *k, const uint16_t *v, const mlops_fbla_cache *cache,
                                                  uint16_t *out, cudaStream_t stream) {
  if (!c || !cache || !k || !v) return fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_prefill: null argument");
  if (q && !out) return fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_prefill: q needs output");
  if (!valid_prefill(c, count))
    return fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_prefill: batch, heads and count must be positive, tokens >= 0, "
                "q_heads a multiple of kv_heads, tokens + count <= capacity "
                "(0: tokens + count) and <= 2^26, strides both 0 or both positive");
  if (c->format != MLOPS_FBLA_3P3 && c->format != MLOPS_FBLA_4P2)
    return fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_prefill: invalid format");
  if (!(c->scale >= 0.f)) return fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_prefill: scale must be nonnegative");
  const mlops_fbla_cache *x = cache;
  if (!x->coarse || !x->fine || !x->meta || !x->exact_sink_k || !x->exact_sink_v || !x->exact_open_k || !x->exact_open_v)
    return fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_prefill: every cache pointer is required");
  const void *ptrs[] = {k, v, q, out, x->coarse, x->fine, x->meta, x->exact_sink_k, x->exact_sink_v,
                        x->exact_open_k, x->exact_open_v};
  for (const void *p : ptrs)
    if (!aligned(p, 16)) return fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_prefill: pointers must be 16-byte aligned");
  if (!aligned(x->tc_images, 128)) return fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_prefill: tc_images must be 128-byte aligned");

  mlops_fbla_detail::PrefillArgs a{};
  a.c = *cache; a.q = q; a.k = k; a.v = v; a.out = out;
  if (!c->capacity && !c->head_stride) a.c.tc_images = nullptr;  // (with neither, the heads' stride moves with tokens)
  a.batch = c->batch; a.q_heads = c->q_heads; a.kv_heads = c->kv_heads; a.qg = c->q_heads / c->kv_heads;
  a.n0 = c->tokens; a.count = count;
  const int capacity = c->capacity ? c->capacity : c->tokens + count;
  a.hs = c->head_stride ? c->head_stride : (capacity + MLOPS_FBLA_GROUP - 1) / MLOPS_FBLA_GROUP;
  a.gs = c->group_stride ? c->group_stride : 1;
  const float scale = c->scale == 0.f ? 1.f / sqrtf((float)MLOPS_FBLA_HEAD_DIM) : c->scale;
  a.scale_log2 = scale * 1.4426950408889634f;
  const cudaError_t e = c->format == MLOPS_FBLA_3P3 ? mlops_fbla_detail::launch_prefill<3, 3>(a, stream)
                                                     : mlops_fbla_detail::launch_prefill<4, 2>(a, stream);
  if (e != cudaSuccess) return fail(MLOPS_FBLA_CUDA_ERROR, "mlops_fbla_prefill: %s", cudaGetErrorString(e));
  return ok();
}

extern "C" mlops_fbla_status mlops_fbla_append_at(const mlops_fbla_config *c, const int *length, const uint16_t *k,
                                                    const uint16_t *v, const mlops_fbla_cache *cache, cudaStream_t stream) {
  constexpr const char *fn = "mlops_fbla_append_at";
  if (!c || !length || !cache || !k || !v) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: null argument", fn);
  if (c->batch <= 0 || c->kv_heads <= 0 || c->q_heads <= 0 || c->q_heads % c->kv_heads || c->tokens < 0 || c->capacity <= 0 ||
      !valid_strides(c))
    return fail(MLOPS_FBLA_INVALID_VALUE, "%s: batch and heads must be positive, q_heads a multiple of kv_heads, tokens >= 0, "
                "capacity positive, strides both 0 or both positive", fn);
  if (c->format != MLOPS_FBLA_3P3 && c->format != MLOPS_FBLA_4P2) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: invalid format", fn);
  const mlops_fbla_cache *x = cache;
  if (!x->coarse || !x->fine || !x->meta || !x->exact_sink_k || !x->exact_sink_v || !x->exact_open_k || !x->exact_open_v)
    return fail(MLOPS_FBLA_INVALID_VALUE, "%s: every cache pointer is required", fn);
  const void *ptrs[] = {k, v, x->coarse, x->fine, x->meta, x->exact_sink_k, x->exact_sink_v, x->exact_open_k, x->exact_open_v};
  for (const void *p : ptrs)
    if (!aligned(p, 16)) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: pointers must be 16-byte aligned", fn);
  if (!aligned(length, 4) || !aligned(x->tc_images, 128))
    return fail(MLOPS_FBLA_INVALID_VALUE, "%s: length must be 4-byte, tc_images 128-byte aligned", fn);

  mlops_fbla_detail::PrefillArgs a{};
  a.c = *cache; a.k = k; a.v = v; a.length = length;
  a.batch = c->batch; a.q_heads = c->q_heads; a.kv_heads = c->kv_heads; a.qg = c->q_heads / c->kv_heads;
  a.n0 = c->tokens; a.count = 1;
  a.hs = c->head_stride ? c->head_stride : (c->capacity + MLOPS_FBLA_GROUP - 1) / MLOPS_FBLA_GROUP;
  a.gs = c->group_stride ? c->group_stride : 1;
  const cudaError_t e = c->format == MLOPS_FBLA_3P3 ? mlops_fbla_detail::launch_append<3, 3>(a, stream)
                                                     : mlops_fbla_detail::launch_append<4, 2>(a, stream);
  if (e != cudaSuccess) return fail(MLOPS_FBLA_CUDA_ERROR, "%s: %s", fn, cudaGetErrorString(e));
  return ok();
}

mlops_fbla_status mlops_fbla_detail::qkv_args(const mlops_fbla_config *c, const int *length, const mlops_fbla_token *t,
                                              const mlops_fbla_cache *cache, QkvArgs &a) {
  constexpr const char *fn = "mlops_fbla_qkv_at";
  if (!c || !t || !cache || !t->q || !t->k || !t->v || !t->freq || !t->q_out) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: null argument", fn);
  if (c->batch <= 0 || c->kv_heads <= 0 || c->q_heads <= 0 || c->q_heads % c->kv_heads || c->tokens < 0 || c->capacity <= 0 ||
      !valid_strides(c))
    return fail(MLOPS_FBLA_INVALID_VALUE, "%s: batch and heads must be positive, q_heads a multiple of kv_heads, tokens >= 0, "
                "capacity positive, strides both 0 or both positive", fn);
  if (c->format != MLOPS_FBLA_3P3 && c->format != MLOPS_FBLA_4P2) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: invalid format", fn);
  if (!t->q_norm != !t->k_norm) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: q_norm and k_norm both or neither", fn);
  if (!(t->eps >= 0.f) || !(t->attention_factor >= 0.f)) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: eps and attention_factor nonnegative", fn);
  const mlops_fbla_cache *x = cache;
  if (!x->coarse || !x->fine || !x->meta || !x->exact_sink_k || !x->exact_sink_v || !x->exact_open_k || !x->exact_open_v)
    return fail(MLOPS_FBLA_INVALID_VALUE, "%s: every cache pointer is required", fn);
  const void *p16[] = {x->coarse, x->fine, x->meta, x->exact_sink_k, x->exact_sink_v, x->exact_open_k, x->exact_open_v, t->freq};
  const void *p4[] = {length, t->q, t->k, t->v, t->q_norm, t->k_norm, t->q_out};
  for (const void *p : p16)
    if (!aligned(p, 16)) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: cache pointers and freq must be 16-byte aligned", fn);
  for (const void *p : p4)
    if (!aligned(p, 4)) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: length and the token's pointers must be 4-byte aligned", fn);
  if (t->batch_stride % 2 || t->q_group_stride % 2 || t->kv_group_stride % 2)
    return fail(MLOPS_FBLA_INVALID_VALUE, "%s: strides must be even (4-byte rows)", fn);
  if (!aligned(x->tc_images, 128)) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: tc_images must be 128-byte aligned", fn);

  a = QkvArgs{};
  a.p.c = *cache; a.p.length = length;
  a.p.batch = c->batch; a.p.q_heads = c->q_heads; a.p.kv_heads = c->kv_heads; a.p.qg = c->q_heads / c->kv_heads;
  a.p.n0 = c->tokens; a.p.count = 1;
  a.p.hs = c->head_stride ? c->head_stride : (c->capacity + MLOPS_FBLA_GROUP - 1) / MLOPS_FBLA_GROUP;
  a.p.gs = c->group_stride ? c->group_stride : 1;
  a.q = t->q; a.k = t->k; a.v = t->v;
  a.batch_stride = t->batch_stride; a.q_group_stride = t->q_group_stride; a.kv_group_stride = t->kv_group_stride;
  a.q_norm = t->q_norm; a.k_norm = t->k_norm; a.eps = t->eps;
  a.att = t->attention_factor == 0.f ? 1.f : t->attention_factor;
  a.freq = reinterpret_cast<const mlops_rope_detail::Freq *>(t->freq);
  a.q_out = t->q_out;
  a.kv = FastDiv::of(c->kv_heads);
  return ok();
}

extern "C" mlops_fbla_status mlops_fbla_qkv_at(const mlops_fbla_config *c, const int *length, const mlops_fbla_token *t,
                                                 const mlops_fbla_cache *cache, cudaStream_t stream) {
  constexpr const char *fn = "mlops_fbla_qkv_at";
  if (!length) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: null argument", fn);
  mlops_fbla_detail::QkvArgs a;
  const mlops_fbla_status st = mlops_fbla_detail::qkv_args(c, length, t, cache, a);
  if (st != MLOPS_FBLA_OK) return st;
  const cudaError_t e = c->format == MLOPS_FBLA_3P3 ? mlops_fbla_detail::launch_qkv<3, 3>(a, stream)
                                                     : mlops_fbla_detail::launch_qkv<4, 2>(a, stream);
  if (e != cudaSuccess) return fail(MLOPS_FBLA_CUDA_ERROR, "%s: %s", fn, cudaGetErrorString(e));
  return ok();
}

extern "C" size_t mlops_fbla_images_bytes(const mlops_fbla_config *c) {
  if (!c || c->batch <= 0 || c->kv_heads <= 0 || c->capacity <= 0 || !mlops_fbla_detail::prefill_tc_device()) return 0;
  const size_t cg = (size_t)((c->capacity + MLOPS_FBLA_GROUP - 1) / MLOPS_FBLA_GROUP);
  return (size_t)c->batch * c->kv_heads * cg * mlops_fbla_detail::IMAGE_SLOT;
}

extern "C" const char *mlops_fbla_status_string(mlops_fbla_status s) {
  switch (s) { case MLOPS_FBLA_OK: return "ok"; case MLOPS_FBLA_INVALID_VALUE: return "invalid value";
               case MLOPS_FBLA_CUDA_ERROR: return "cuda error"; default: return "unknown"; }
}
extern "C" const char *mlops_fbla_last_error(void) { return error_buf; }
