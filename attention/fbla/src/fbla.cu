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
  static thread_local int cached[64] = {0};
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
constexpr int SLICE = mlops_fbla_detail::SLICE, MAX_SLICES = mlops_fbla_detail::MAX_SLICES;
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

namespace {
// the device-length prefill's arguments and checks, shared by mlops_fbla_prefill_at (count's ceiling max_count),
// mlops_fbla_append_at (max_count 1) and their work-item plans (mlops_fbla_items.h)
mlops_fbla_status prefill_at_args(const char *fn, const mlops_fbla_config *c, const int *length, const int *dcount, int max_count,
                                  const uint16_t *q, const uint16_t *k, const uint16_t *v, const mlops_fbla_cache *cache,
                                  uint16_t *out, mlops_fbla_detail::PrefillArgs &a) {
  if (!c || !length || !cache || !k || !v || (dcount && !length)) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: null argument", fn);
  if (q && !out) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: q needs output", fn);
  if (c->batch <= 0 || c->kv_heads <= 0 || c->q_heads <= 0 || c->q_heads % c->kv_heads || c->tokens < 0 || c->capacity <= 0 ||
      (long long)c->tokens + max_count > c->capacity || max_count <= 0 || c->capacity > (1 << 26) || !valid_strides(c))
    return fail(MLOPS_FBLA_INVALID_VALUE, "%s: batch, heads, max_count and capacity must be positive, tokens >= 0, "
                "tokens + max_count <= capacity and <= 2^26, q_heads a multiple of kv_heads, strides both 0 or both positive", fn);
  if (c->format != MLOPS_FBLA_3P3 && c->format != MLOPS_FBLA_4P2) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: invalid format", fn);
  if (!(c->scale >= 0.f)) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: scale must be nonnegative", fn);
  const mlops_fbla_cache *x = cache;
  if (!x->coarse || !x->fine || !x->meta || !x->exact_sink_k || !x->exact_sink_v || !x->exact_open_k || !x->exact_open_v)
    return fail(MLOPS_FBLA_INVALID_VALUE, "%s: every cache pointer is required", fn);
  const void *ptrs[] = {k, v, q, out, x->coarse, x->fine, x->meta, x->exact_sink_k, x->exact_sink_v,
                        x->exact_open_k, x->exact_open_v};
  for (const void *p : ptrs)
    if (!aligned(p, 16)) return fail(MLOPS_FBLA_INVALID_VALUE, "%s: pointers must be 16-byte aligned", fn);
  if (!aligned(length, 4) || !aligned(dcount, 4) || !aligned(x->tc_images, 128))
    return fail(MLOPS_FBLA_INVALID_VALUE, "%s: length and count must be 4-byte, tc_images 128-byte aligned", fn);
  if (q && mlops_fbla_detail::prefill_tc_device() && !x->tc_images)
    return fail(MLOPS_FBLA_INVALID_VALUE, "%s: the tensor-core attention needs the cache's tc_images "
                "(mlops_fbla_images_bytes); without them pass q null, appending only", fn);

  a.c = *cache; a.q = q; a.k = k; a.v = v; a.out = out;
  a.length = length; a.dcount = dcount;
  a.batch = c->batch; a.q_heads = c->q_heads; a.kv_heads = c->kv_heads; a.qg = c->q_heads / c->kv_heads;
  a.n0 = c->tokens; a.count = max_count;
  a.hs = c->head_stride ? c->head_stride : (c->capacity + MLOPS_FBLA_GROUP - 1) / MLOPS_FBLA_GROUP;
  a.gs = c->group_stride ? c->group_stride : 1;
  const float scale = c->scale == 0.f ? 1.f / sqrtf((float)MLOPS_FBLA_HEAD_DIM) : c->scale;
  a.scale_log2 = scale * 1.4426950408889634f;
  return MLOPS_FBLA_OK;
}
}  // namespace

extern "C" mlops_fbla_status mlops_fbla_prefill_at(const mlops_fbla_config *c, const int *length, const int *count, int max_count,
                                                    const uint16_t *q, const uint16_t *k, const uint16_t *v,
                                                    const mlops_fbla_cache *cache, uint16_t *out, cudaStream_t stream) {
  mlops_fbla_detail::PrefillArgs a{};
  const mlops_fbla_status st = prefill_at_args("mlops_fbla_prefill_at", c, length, count, max_count, q, k, v, cache, out, a);
  if (st != MLOPS_FBLA_OK) return st;
  const cudaError_t e = c->format == MLOPS_FBLA_3P3 ? mlops_fbla_detail::launch_prefill_at<3, 3>(a, stream)
                                                    : mlops_fbla_detail::launch_prefill_at<4, 2>(a, stream);
  if (e != cudaSuccess) return fail(MLOPS_FBLA_CUDA_ERROR, "mlops_fbla_prefill_at: %s", cudaGetErrorString(e));
  return ok();
}

extern "C" mlops_fbla_status mlops_fbla_append_at(const mlops_fbla_config *c, const int *length, const uint16_t *k,
                                                    const uint16_t *v, const mlops_fbla_cache *cache, cudaStream_t stream) {
  mlops_fbla_detail::PrefillArgs a{};
  const mlops_fbla_status st = prefill_at_args("mlops_fbla_append_at", c, length, nullptr, 1, nullptr, k, v, cache, nullptr, a);
  if (st != MLOPS_FBLA_OK) return st;
  const cudaError_t e = c->format == MLOPS_FBLA_3P3 ? mlops_fbla_detail::launch_append<3, 3>(a, stream)
                                                    : mlops_fbla_detail::launch_append<4, 2>(a, stream);
  if (e != cudaSuccess) return fail(MLOPS_FBLA_CUDA_ERROR, "mlops_fbla_append_at: %s", cudaGetErrorString(e));
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

// ---- the work items (mlops_fbla_items.h) -----------------------------------------------------------------------------
namespace {
bool valid_plan_output(mlops_fbla_items *passes, int max) {
  if (max < 0 || (max && !passes)) {
    fail(MLOPS_FBLA_INVALID_VALUE, "FBLA items: invalid pass output");
    return false;
  }
  return true;
}
int copy_plan(mlops_fbla_items *out, int max, mlops_fbla_items *p, int n, const mlops_fbla_config *c) {
  if (n < 0) {
    fail(MLOPS_FBLA_CUDA_ERROR, "FBLA items: cannot describe the current device's kernels");
    return -1;
  }
  for (int i = 0; i < n; ++i) {
    p[i].geom[MLOPS_FBLA_GEOM_CAPACITY_GROUPS] = (c->capacity + MLOPS_FBLA_GROUP - 1) / MLOPS_FBLA_GROUP;
    p[i].geom[MLOPS_FBLA_GEOM_HEAD_STRIDE] = c->head_stride ? c->head_stride : p[i].geom[MLOPS_FBLA_GEOM_CAPACITY_GROUPS];
    p[i].geom[MLOPS_FBLA_GEOM_GROUP_STRIDE] = c->group_stride ? c->group_stride : 1;
    p[i].geom[MLOPS_FBLA_GEOM_COARSE_BYTES] = c->format == MLOPS_FBLA_3P3 ? mlops_fbla_detail::record_bytes<3>() : mlops_fbla_detail::record_bytes<4>();
    p[i].geom[MLOPS_FBLA_GEOM_FINE_BYTES] = c->format == MLOPS_FBLA_3P3 ? mlops_fbla_detail::record_bytes<3>() : mlops_fbla_detail::record_bytes<2>();
  }
  if (max) memcpy(out, p, (size_t)std::min(max, n) * sizeof *p);
  ok();
  return n;
}
static_assert(sizeof(mlops_fbla_detail::Args) <= MLOPS_FBLA_ITEM_ARGS_BYTES, "decode item arguments");
static_assert(sizeof(mlops_fbla_detail::PrefillArgs) <= MLOPS_FBLA_ITEM_ARGS_BYTES, "prefill item arguments");
static_assert(sizeof(mlops_fbla_detail::QkvArgs) <= MLOPS_FBLA_ITEM_ARGS_BYTES, "qkv item arguments");
}  // namespace

extern "C" int mlops_fbla_prefill_at_items(const mlops_fbla_config *c, const int *length, const int *count, int max_count,
                                           const uint16_t *q, const uint16_t *k, const uint16_t *v,
                                           const mlops_fbla_cache *cache, uint16_t *output, mlops_fbla_items *passes, int max) {
  if (!valid_plan_output(passes, max)) return -1;
  mlops_fbla_items p[5]{};
  mlops_fbla_detail::PrefillArgs a{};
  const mlops_fbla_status st = prefill_at_args("mlops_fbla_prefill_at_items", c, length, count, max_count, q, k, v, cache,
                                               output, a);
  if (st != MLOPS_FBLA_OK) return -1;
  const int n = c->format == MLOPS_FBLA_3P3 ? mlops_fbla_detail::prefill_at_passes<3, 3>(a, false, p, 5)
                                           : mlops_fbla_detail::prefill_at_passes<4, 2>(a, false, p, 5);
  return copy_plan(passes, max, p, n, c);
}

extern "C" int mlops_fbla_append_at_items(const mlops_fbla_config *c, const int *length, const uint16_t *k, const uint16_t *v,
                                          const mlops_fbla_cache *cache, mlops_fbla_items *passes, int max) {
  if (!valid_plan_output(passes, max)) return -1;
  mlops_fbla_items p[5]{};
  mlops_fbla_detail::PrefillArgs a{};
  const mlops_fbla_status st = prefill_at_args("mlops_fbla_append_at_items", c, length, nullptr, 1, nullptr, k, v, cache,
                                               nullptr, a);
  if (st != MLOPS_FBLA_OK) return -1;
  const int n = c->format == MLOPS_FBLA_3P3 ? mlops_fbla_detail::prefill_at_passes<3, 3>(a, true, p, 5)
                                           : mlops_fbla_detail::prefill_at_passes<4, 2>(a, true, p, 5);
  return copy_plan(passes, max, p, n, c);
}

extern "C" int mlops_fbla_decode_at_items(const mlops_fbla_config *c, const int *length, const uint16_t *q,
                                          const mlops_fbla_cache *cache, uint16_t *output, void *scratch, size_t scratch_bytes,
                                          int per_head, mlops_fbla_items *passes, int max) {
  if (!valid_plan_output(passes, max)) return -1;
  if (!length) { fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_decode_at_items: null length"); return -1; }
  mlops_fbla_items p[4]{};
  mlops_fbla_detail::Args a{};
  if (per_head < 0) { fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_decode_at_items: negative per_head"); return -1; }
  if (per_head) {
    if (decode_args(c, length, q, cache, output, scratch, scratch_bytes, per_head, a) != MLOPS_FBLA_OK) return -1;
  } else {
    int bound;
    if (prepare_decode("mlops_fbla_decode_at_items", c, true, length, q, cache, output, scratch, scratch_bytes, 0, a, bound)
        != MLOPS_FBLA_OK) return -1;
    a.ctas = bound;  // decode_passes resolves each kernel's occupancy, as launch_qg does.
    a.rctas = 0;
  }
  const int n = c->format == MLOPS_FBLA_3P3 ? mlops_fbla_detail::decode_passes<3, 3>(a, scratch, p, 4)
                                           : mlops_fbla_detail::decode_passes<4, 2>(a, scratch, p, 4);
  return copy_plan(passes, max, p, n, c);
}

extern "C" int mlops_fbla_qkv_at_items(const mlops_fbla_config *c, const int *length, const mlops_fbla_token *token,
                                       const mlops_fbla_cache *cache, mlops_fbla_items *passes, int max) {
  if (!valid_plan_output(passes, max)) return -1;
  if (!length) { fail(MLOPS_FBLA_INVALID_VALUE, "mlops_fbla_qkv_at_items: null length"); return -1; }
  mlops_fbla_items p[1]{};
  mlops_fbla_detail::QkvArgs a{};
  if (mlops_fbla_detail::qkv_args(c, length, token, cache, a) != MLOPS_FBLA_OK) return -1;
  const int n = c->format == MLOPS_FBLA_3P3 ? mlops_fbla_detail::qkv_passes<3, 3>(a, p, 1)
                                           : mlops_fbla_detail::qkv_passes<4, 2>(a, p, 1);
  return copy_plan(passes, max, p, n, c);
}

// The ranges of one item of one pass (mlops_fbla_items.h).  Length and count live in device memory, so ranges
// are the envelope over every count up to the ceiling and every start the capacity allows.
// geom[20..]: hs, gs, groups, coarse/fine record bytes; decode 25..30: scratch lse/part/dpart/row_lse/items/counts.
extern "C" int mlops_fbla_item_access(const mlops_fbla_items *passes, int npasses, int pass, int item, mlops_fbla_item_range *ranges,
                                      int max) {
  if (!passes || pass < 0 || pass >= npasses || item < 0 || item >= passes[pass].items || max < 0 || (max && !ranges))
    return -1;
  const mlops_fbla_items &p = passes[pass];
  const int gx = (int)p.geom[0];
  if (gx <= 0 || p.geom[1] <= 0 || p.geom[2] <= 0) return -1;
  const int x = item % gx, z = item / (gx * (int)p.geom[1]);
  const int y = p.kind == MLOPS_FBLA_ITEM_FINISH || p.kind == MLOPS_FBLA_ITEM_QKV ? x : (item / gx) % (int)p.geom[1];
  const int G = MLOPS_FBLA_GROUP, D = MLOPS_FBLA_HEAD_DIM, SBG = G / MLOPS_FBLA_SUBBLOCK;
  const long long row = 2 * D, cgu = (int)p.geom[MLOPS_FBLA_GEOM_CAPACITY_GROUPS], RC = p.geom[MLOPS_FBLA_GEOM_COARSE_BYTES], RF = p.geom[MLOPS_FBLA_GEOM_FINE_BYTES], hs = p.geom[MLOPS_FBLA_GEOM_HEAD_STRIDE] ? p.geom[MLOPS_FBLA_GEOM_HEAD_STRIDE] : cgu,
                gs = p.geom[MLOPS_FBLA_GEOM_GROUP_STRIDE];
  int n = 0;
  auto put = [&](int arg, int write, long long offset, long long bytes, long long stride, int rows) {
    if (bytes <= 0 || rows <= 0) return;
    if (n < max) ranges[n] = mlops_fbla_item_range{arg, write, offset, bytes, stride, rows};
    ++n;
  };
  auto cache_units = [&](int head, int write, int arg, long long unit_bytes) {
    put(arg, write, (head * hs) * unit_bytes, unit_bytes, gs * unit_bytes, cgu);
  };
  constexpr int SLICE = mlops_fbla_detail::SLICE;
  mlops_fbla_detail::PrefillArgs a{};
  if (p.kind <= MLOPS_FBLA_ITEM_FINISH) {
    memcpy(&a, p.args.bytes, sizeof a);
    const long long cnt = a.count;
    auto position = [&] {
      if (a.length) put(MLOPS_FBLA_ARG_LENGTH, 0, 0, 4, 0, 1);
      if (a.dcount) put(MLOPS_FBLA_ARG_COUNT, 0, 0, 4, 0, 1);
    };
    switch (p.kind) {
      case MLOPS_FBLA_ITEM_CLOSE:
        put(MLOPS_FBLA_ARG_K, 0, y * cnt * row, row, row, (int)cnt);
        put(MLOPS_FBLA_ARG_V, 0, y * cnt * row, row, row, (int)cnt);
        put(MLOPS_FBLA_ARG_SINK_K, 0, (long long)y * G * row, row, row, G);
        put(MLOPS_FBLA_ARG_SINK_V, 0, (long long)y * G * row, row, row, G);
        put(MLOPS_FBLA_ARG_OPEN_K, 0, (long long)y * G * row, row, row, G);
        put(MLOPS_FBLA_ARG_OPEN_V, 0, (long long)y * G * row, row, row, G);
        cache_units(y, 1, MLOPS_FBLA_ARG_COARSE, SBG * RC);
        cache_units(y, 1, MLOPS_FBLA_ARG_FINE, SBG * RF);
        cache_units(y, 1, MLOPS_FBLA_ARG_META, 512 * 2);
        if (a.c.tc_images)
          put(MLOPS_FBLA_ARG_IMAGES, 1, (y * hs) * mlops_fbla_detail::IMAGE_SLOT + mlops_fbla_detail::IMAGE_BYTES, 4,
              gs * mlops_fbla_detail::IMAGE_SLOT, cgu);
        position();
        break;
      case MLOPS_FBLA_ITEM_TRANSCODE:
        cache_units(y, 0, MLOPS_FBLA_ARG_COARSE, SBG * RC);
        cache_units(y, 0, MLOPS_FBLA_ARG_FINE, SBG * RF);
        cache_units(y, 0, MLOPS_FBLA_ARG_META, 512 * 2);
        put(MLOPS_FBLA_ARG_IMAGES, 0, (y * hs) * mlops_fbla_detail::IMAGE_SLOT + mlops_fbla_detail::IMAGE_BYTES,
            4, gs * mlops_fbla_detail::IMAGE_SLOT, cgu);
        cache_units(y, 1, MLOPS_FBLA_ARG_IMAGES, mlops_fbla_detail::IMAGE_SLOT);
        position();
        break;
      case MLOPS_FBLA_ITEM_STREAM: {  // head y's q rows in and out, its K/V and records
        const int rows = a.qg * (int)cnt;
        put(MLOPS_FBLA_ARG_Q, 0, (long long)(y * a.qg) * cnt * row, row, row, rows);
        put(MLOPS_FBLA_ARG_K, 0, y * cnt * row, row, row, (int)cnt);
        put(MLOPS_FBLA_ARG_V, 0, y * cnt * row, row, row, (int)cnt);
        put(MLOPS_FBLA_ARG_SINK_K, 0, (long long)y * G * row, row, row, G);
        put(MLOPS_FBLA_ARG_SINK_V, 0, (long long)y * G * row, row, row, G);
        put(MLOPS_FBLA_ARG_OPEN_K, 0, (long long)y * G * row, row, row, G);
        put(MLOPS_FBLA_ARG_OPEN_V, 0, (long long)y * G * row, row, row, G);
        cache_units(y, 0, MLOPS_FBLA_ARG_COARSE, SBG * RC);
        cache_units(y, 0, MLOPS_FBLA_ARG_FINE, SBG * RF);
        cache_units(y, 0, MLOPS_FBLA_ARG_META, 512 * 2);
        put(MLOPS_FBLA_ARG_OUT, 1, (long long)(y * a.qg) * cnt * row, row, row, rows);
        position();
        break;
      }
      case MLOPS_FBLA_ITEM_TC: {  // q head y (its KV head y / qg): q and out rows, the head's images; the sink, the
        // own group's K / V, and (a split's partials, geom[7] / geom[8]) aside, that is all it touches
        const int kh = y / a.qg;
        put(MLOPS_FBLA_ARG_Q, 0, (long long)y * cnt * row, row, row, (int)cnt);
        put(MLOPS_FBLA_ARG_K, 0, kh * cnt * row, row, row, (int)cnt);
        put(MLOPS_FBLA_ARG_V, 0, kh * cnt * row, row, row, (int)cnt);
        put(MLOPS_FBLA_ARG_SINK_K, 0, (long long)kh * G * row, row, row, G);
        put(MLOPS_FBLA_ARG_SINK_V, 0, (long long)kh * G * row, row, row, G);
        put(MLOPS_FBLA_ARG_OPEN_K, 0, (long long)kh * G * row, row, row, G);
        put(MLOPS_FBLA_ARG_OPEN_V, 0, (long long)kh * G * row, row, row, G);
        cache_units(kh, 0, MLOPS_FBLA_ARG_IMAGES, mlops_fbla_detail::IMAGE_SLOT);
        if (p.geom[11] <= 1) put(MLOPS_FBLA_ARG_OUT, 1, (long long)y * cnt * row, row, row, (int)cnt);
        else {
          const long long pr = ((long long)z * a.batch * a.q_heads + y) * cnt;
          put(MLOPS_FBLA_ARG_PART_ML, 1, pr * 8, 8, 8, (int)cnt);
          put(MLOPS_FBLA_ARG_PART_O, 1, pr * D * 4, D * 4, D * 4, (int)cnt);
        }
        position();
        break;
      }
      case MLOPS_FBLA_ITEM_TC_MERGE:  // q head y's rows out (its inputs the split partials, geom[3] / geom[4])
        put(MLOPS_FBLA_ARG_PART_ML, 0, (long long)y * cnt * 8, cnt * 8,
            (long long)a.batch * a.q_heads * cnt * 8, (int)p.geom[5]);
        put(MLOPS_FBLA_ARG_PART_O, 0, (long long)y * cnt * D * 4, cnt * D * 4,
            (long long)a.batch * a.q_heads * cnt * D * 4, (int)p.geom[5]);
        put(MLOPS_FBLA_ARG_OUT, 1, (long long)y * cnt * row, row, row, (int)cnt);
        if (a.dcount) put(MLOPS_FBLA_ARG_COUNT, 0, 0, 4, 0, 1);
        break;
      case MLOPS_FBLA_ITEM_FINISH:
        put(MLOPS_FBLA_ARG_K, 0, y * cnt * row, row, row, (int)cnt);
        put(MLOPS_FBLA_ARG_V, 0, y * cnt * row, row, row, (int)cnt);
        put(MLOPS_FBLA_ARG_SINK_K, 1, (long long)y * G * row, row, row, G);
        put(MLOPS_FBLA_ARG_SINK_V, 1, (long long)y * G * row, row, row, G);
        put(MLOPS_FBLA_ARG_OPEN_K, 1, (long long)y * G * row, row, row, G);
        put(MLOPS_FBLA_ARG_OPEN_V, 1, (long long)y * G * row, row, row, G);
        position();
        break;
      default:
        return -1;
    }
  } else if (p.kind == MLOPS_FBLA_ITEM_QKV) {
    mlops_fbla_detail::QkvArgs q{};
    memcpy(&q, p.args.bytes, sizeof q);
    const int b = y / q.p.kv_heads, g = y - b * q.p.kv_heads;
    const long long qs = (b * q.batch_stride + g * q.q_group_stride) * 2, kvs = (b * q.batch_stride + g * q.kv_group_stride) * 2;
    put(MLOPS_FBLA_ARG_TQ, 0, qs, row, row, q.p.qg);  // the token's q rows of head y
    put(MLOPS_FBLA_ARG_TK, 0, kvs, row, 0, 1);        // its k and v rows (a KV head's, one each)
    put(MLOPS_FBLA_ARG_TV, 0, kvs, row, 0, 1);
    if (q.q_norm) {
      put(MLOPS_FBLA_ARG_TQN, 0, 0, row, 0, 1);  // (the norm weights and the frequencies: head-invariant)
      put(MLOPS_FBLA_ARG_TKN, 0, 0, row, 0, 1);
    }
    put(MLOPS_FBLA_ARG_TFREQ, 0, 0, (D / 2) * sizeof(mlops_rope_detail::Freq), 0, 1);
    put(MLOPS_FBLA_ARG_TQ_OUT, 1, (long long)y * q.p.qg * row, row, row, q.p.qg);
    put(MLOPS_FBLA_ARG_OPEN_K, 0, (long long)y * G * row, row, row, G);
    put(MLOPS_FBLA_ARG_OPEN_V, 0, (long long)y * G * row, row, row, G);
    put(MLOPS_FBLA_ARG_SINK_K, 0, (long long)y * G * row, row, row, G);
    put(MLOPS_FBLA_ARG_SINK_V, 0, (long long)y * G * row, row, row, G);
    put(MLOPS_FBLA_ARG_SINK_K, 1, (long long)y * G * row, row, row, G);
    put(MLOPS_FBLA_ARG_SINK_V, 1, (long long)y * G * row, row, row, G);
    put(MLOPS_FBLA_ARG_OPEN_K, 1, (long long)y * G * row, row, row, G);
    put(MLOPS_FBLA_ARG_OPEN_V, 1, (long long)y * G * row, row, row, G);
    cache_units(y, 1, MLOPS_FBLA_ARG_COARSE, SBG * RC);
    cache_units(y, 1, MLOPS_FBLA_ARG_FINE, SBG * RF);
    cache_units(y, 1, MLOPS_FBLA_ARG_META, 512 * 2);
    if (q.p.c.tc_images)
      put(MLOPS_FBLA_ARG_IMAGES, 1, (y * hs) * mlops_fbla_detail::IMAGE_SLOT + mlops_fbla_detail::IMAGE_BYTES, 4,
          gs * mlops_fbla_detail::IMAGE_SLOT, cgu);
    put(MLOPS_FBLA_ARG_LENGTH, 0, 0, 4, 0, 1);
  } else {  // the decode's passes (the args an Args)
    mlops_fbla_detail::Args d{};
    memcpy(&d, p.args.bytes, sizeof d);
    const int qg = d.q_heads / d.kv_heads / d.vg;
    const int row0 = y * qg;  // the (virtual) head's q rows
    const int kh = y / d.vg;  // (the virtual head's KV head; FastDiv::div is the device form)
    const long long o_lse = p.geom[MLOPS_FBLA_GEOM_LSE], o_part = p.geom[MLOPS_FBLA_GEOM_PART], o_dpart = p.geom[MLOPS_FBLA_GEOM_DPART], o_row = p.geom[MLOPS_FBLA_GEOM_ROW_LSE], o_items = p.geom[MLOPS_FBLA_GEOM_ITEMS],
                  o_counts = p.geom[MLOPS_FBLA_GEOM_COUNTS];
    const int hs_d = d.hs ? d.hs : cgu, gs_d = d.gs;
    auto units_d = [&](int write, int arg, long long unit_bytes) {
      put(arg, write, (kh * hs_d) * unit_bytes, unit_bytes, gs_d * unit_bytes, cgu);
    };
    switch (p.kind) {
      case MLOPS_FBLA_ITEM_COARSE:
        put(MLOPS_FBLA_ARG_Q, 0, (long long)row0 * row, row, row, qg);
        units_d(0, MLOPS_FBLA_ARG_COARSE, SBG * RC);
        units_d(0, MLOPS_FBLA_ARG_FINE, SBG * RF);
        units_d(0, MLOPS_FBLA_ARG_META, 512 * 2);
        for (int arg : {MLOPS_FBLA_ARG_SINK_K, MLOPS_FBLA_ARG_SINK_V, MLOPS_FBLA_ARG_OPEN_K, MLOPS_FBLA_ARG_OPEN_V})
          put(arg, 0, (long long)kh * G * row, row, row, G);
        put(MLOPS_FBLA_ARG_SCRATCH, 1, o_lse + (long long)row0 * d.nblocks * 4, 4, 4, qg * d.nblocks);
        put(MLOPS_FBLA_ARG_SCRATCH, 1, o_part + (long long)row0 * d.ctas * (2 + D) * 4, (2 + D) * 4, (2 + D) * 4, qg * d.ctas);
        put(MLOPS_FBLA_ARG_LENGTH, 0, 0, 4, 0, 1);
        break;
      case MLOPS_FBLA_ITEM_SELECT:
        put(MLOPS_FBLA_ARG_SCRATCH, 0, o_lse + (long long)row0 * d.nblocks * 4, 4, 4, qg * d.nblocks);
        put(MLOPS_FBLA_ARG_SCRATCH, 0, o_part + (long long)row0 * d.ctas * (2 + D) * 4,
            (2 + D) * 4, (2 + D) * 4, qg * d.ctas);
        if (!x) put(MLOPS_FBLA_ARG_SCRATCH, 1, o_row + (long long)row0 * 4, 4, 4, qg);
        put(MLOPS_FBLA_ARG_SCRATCH, 1, o_items + (long long)(y * d.nslices + x) * SLICE * 4, SLICE * 4, 0, 1);
        put(MLOPS_FBLA_ARG_SCRATCH, 1, o_counts + (long long)(y * d.nslices + x) * 4, 4, 0, 1);
        if (d.length) put(MLOPS_FBLA_ARG_LENGTH, 0, 0, 4, 0, 1);
        break;
      case MLOPS_FBLA_ITEM_REFINE:
        put(MLOPS_FBLA_ARG_Q, 0, (long long)row0 * row, row, row, qg);
        units_d(0, MLOPS_FBLA_ARG_COARSE, SBG * RC);
        units_d(0, MLOPS_FBLA_ARG_FINE, SBG * RF);
        units_d(0, MLOPS_FBLA_ARG_META, 512 * 2);
        put(MLOPS_FBLA_ARG_SCRATCH, 0, o_items + (long long)y * d.nslices * SLICE * 4, 4, 4, d.nslices * SLICE);
        put(MLOPS_FBLA_ARG_SCRATCH, 0, o_counts + (long long)y * d.nslices * 4, 4, 4, d.nslices);
        put(MLOPS_FBLA_ARG_SCRATCH, 0, o_part + (long long)row0 * d.ctas * (2 + D) * 4, (2 + D) * 4, (2 + D) * 4, qg * d.ctas);
        put(MLOPS_FBLA_ARG_SCRATCH, 0, o_row + (long long)row0 * 4, 4, 4, qg);
        put(MLOPS_FBLA_ARG_SCRATCH, 1, o_dpart + (long long)row0 * d.rctas * (1 + D) * 4, (1 + D) * 4, (1 + D) * 4, qg * d.rctas);
        put(MLOPS_FBLA_ARG_LENGTH, 0, 0, 4, 0, 1);
        break;
      case MLOPS_FBLA_ITEM_MERGE: {
        const int row_m = row0 + x / (D / 16);  // (MC = 16: merge_kernel's column blocks)
        put(MLOPS_FBLA_ARG_SCRATCH, 0, o_part + (long long)row_m * d.ctas * (2 + D) * 4, (2 + D) * 4, (2 + D) * 4, d.ctas);
        put(MLOPS_FBLA_ARG_SCRATCH, 0, o_dpart + (long long)row_m * d.rctas * (1 + D) * 4, (1 + D) * 4, (1 + D) * 4, d.rctas);
        put(MLOPS_FBLA_ARG_SCRATCH, 0, o_row + (long long)row_m * 4, 4, 0, 1);
        put(MLOPS_FBLA_ARG_OUT, 1, (long long)row_m * row + (x % (D / 16)) * 32, 32, 0, 1);
        break;
      }
      default:
        return -1;
    }
  }
  return n;
}

extern "C" const char *mlops_fbla_status_string(mlops_fbla_status s) {
  switch (s) { case MLOPS_FBLA_OK: return "ok"; case MLOPS_FBLA_INVALID_VALUE: return "invalid value";
               case MLOPS_FBLA_CUDA_ERROR: return "cuda error"; default: return "unknown"; }
}
extern "C" const char *mlops_fbla_last_error(void) { return error_buf; }
