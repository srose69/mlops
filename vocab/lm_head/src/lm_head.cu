// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_lm_head host side (include/mlops_lm_head.h): argument checks, the variants, the resolved element count.
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>

#include <cstring>

#include "items.h"
#include "instances.h"

using namespace mlops_vocab_detail;

namespace {

char *last_error_buf() {
  static thread_local char buf[256] = {0};
  return buf;
}
mlops_lm_head_status fail(mlops_lm_head_status st, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(last_error_buf(), 255, fmt, ap);
  va_end(ap);
  return st;
}

// Shared memory of an instance for n columns of x of `hidden` elements, with the key-byte counts or without.
size_t instance_smem(const Instance &in, int n, int hidden, bool hist) { return smem_bytes(in.tables, in.xs ? smem_x_bytes(n, hidden) : 0, in.nb, hist); }
const Instance *instances(int &count) {
  static Instance table[kMaxInstances];
  static int n = 0;
  static const bool init = [] {
    register_slice_0(table, n);
    register_slice_1(table, n);
    register_slice_2(table, n);
    register_slice_3(table, n);
    register_slice_4(table, n);
    register_slice_5(table, n);
    register_slice_6(table, n);
    return true;
  }();
  (void)init;
  count = n;
  return table;
}
int columns_for(int n) { return n <= 1 ? 1 : n <= 2 ? 2 : n <= 4 ? 4 : 8; }
const Instance *find(const mlops_lm_head_variant &v) {
  int count;
  const Instance *t = instances(count);
  for (int i = 0; i < count; ++i)
    if (t[i].r == v.rows_per_unit && t[i].s == v.k_pieces && t[i].d == v.depth && t[i].nb == v.columns && t[i].tc == v.tensor_cores &&
        t[i].j == v.chunks && t[i].xs == (v.x_staged != 0))
      return t + i;
  return nullptr;
}
// Tensor-core instances need sm_80 (BF16 mma), sm_75 (FP16 m16n8k8) or sm_70 (FP16 m8n8k4) and a hidden size that
// their blocks (32 J elements; 8 J for m8n8k4) tile into at least S pieces. Every instance needs its shared memory (x
// among it) to fit a CTA of the device.
bool structurally_fits(const Instance &in, const mlops_lm_head_config &c) {
  if (in.nb != columns_for(c.n)) return false;
  int dev = 0, major = 0, minor = 0, smem_max = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (instance_smem(in, c.n, c.hidden, false) > static_cast<size_t>(smem_max)) return false;
  if (!in.tc) return c.hidden / 256 >= in.s;
  const int cc = 10 * major + minor;
  const int block = (in.tc == 3 ? 8 : 32) * in.j;
  return cc >= (in.tc == 3 ? 70 : in.tc == 2 ? 75 : 80) && c.hidden % block == 0 && c.hidden / block >= in.s;
}

bool valid(const mlops_lm_head_config *c) {
  return c && c->vocab > 0 && c->hidden > 0 && c->hidden % 256 == 0 && c->n >= 1 && c->n <= 8 && c->items >= 0 && c->norm_parts_n >= 0 && c->logit_scale > 0.f;
}
bool aligned(const void *p, uintptr_t bytes) { return reinterpret_cast<uintptr_t>(p) % bytes == 0; }
const Instance *accepted(const mlops_lm_head_config *c, const mlops_lm_head_variant *v) {
  const Instance *in = v ? find(*v) : nullptr;
  if (!in || !structurally_fits(*in, *c)) return nullptr;
  cudaFuncAttributes a;
  return cudaFuncGetAttributes(&a, reinterpret_cast<const void *>(in->kernel)) == cudaSuccess && !a.localSizeBytes ? in : nullptr;
}

}  // namespace

extern "C" int mlops_lm_head_variants(const mlops_lm_head_config *c, mlops_lm_head_variant *out, int max) {
  if (!valid(c) || max < 0 || (max > 0 && !out)) return 0;
  int n = 0, count;
  const Instance *table = instances(count);
  for (int i = 0; i < count; ++i) {
    const Instance &in = table[i];
    if (!structurally_fits(in, *c)) continue;
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, reinterpret_cast<const void *>(in.kernel)) != cudaSuccess) return 0;
    if (a.localSizeBytes) continue;  // spills (or a stack): not offered
    if (n < max) out[n] = {in.r, in.s, in.d, in.nb, in.tc, in.j, 32 * in.s, static_cast<int>(instance_smem(in, c->n, c->hidden, false)), a.numRegs, a.binaryVersion, in.xs};
    ++n;
  }
  return n;
}

extern "C" int mlops_lm_head_item_count(const mlops_lm_head_config *c, const mlops_lm_head_variant *v) {
  if (!valid(c)) return 0;
  const Instance *in = accepted(c, v);
  if (!in) return 0;
  int items = c->items;
  if (!items) {
    int dev = 0, sms = 1, occ = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    const size_t bytes = instance_smem(*in, c->n, c->hidden, false);
    cudaFuncSetAttribute(reinterpret_cast<const void *>(in->kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, reinterpret_cast<const void *>(in->kernel), 32 * in->s, bytes);
    items = sms * std::max(occ, 1);
  }
  return std::min(items, (c->vocab + in->r - 1) / in->r);
}

extern "C" const char *mlops_lm_head_last_error(void) { return last_error_buf(); }

namespace {

// Validates a call's arguments and builds its device arguments; shared by the call and its description.
mlops_lm_head_status prepare(const mlops_lm_head_config *c, const mlops_lm_head_variant *v, const uint16_t *w, const uint16_t *x,
                             const float *norm_parts, float *logits, float *stats_max, int *stats_arg, unsigned *stats_hist,
                             const Instance *&in, HeadOp &op) {
  if (!valid(c)) return fail(MLOPS_LM_HEAD_INVALID_VALUE, "mlops_lm_head: vocab > 0, hidden a positive multiple of 256, 1 <= n <= 8, items >= 0, logit_scale > 0");
  in = accepted(c, v);
  if (!in) return fail(MLOPS_LM_HEAD_INVALID_VALUE, "mlops_lm_head: variant not among mlops_lm_head_variants of this shape");
  if (stats_hist) {  // the counts come on top of the variant's shared memory
    int dev = 0, smem_max = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (instance_smem(*in, c->n, c->hidden, true) > static_cast<size_t>(smem_max))
      return fail(MLOPS_LM_HEAD_INVALID_VALUE, "mlops_lm_head: stats_hist with this variant needs %zu bytes of shared memory, a CTA has %d (use a variant with x_staged = 0)",
                  instance_smem(*in, c->n, c->hidden, true), smem_max);
  }
  if (!w || !x) return fail(MLOPS_LM_HEAD_INVALID_VALUE, "mlops_lm_head: null w or x");
  if (!aligned(w, 128) || !aligned(x, 128)) return fail(MLOPS_LM_HEAD_INVALID_VALUE, "mlops_lm_head: w and x must be 128-byte aligned");
  if (!logits && !stats_max) return fail(MLOPS_LM_HEAD_INVALID_VALUE, "mlops_lm_head: neither logits nor stats requested");
  if ((stats_max == nullptr) != (stats_arg == nullptr)) return fail(MLOPS_LM_HEAD_INVALID_VALUE, "mlops_lm_head: stats_max and stats_arg go together");
  if (stats_hist && !stats_max) return fail(MLOPS_LM_HEAD_INVALID_VALUE, "mlops_lm_head: stats_hist needs stats_max");
  if (c->norm_parts_n > 0 && !norm_parts) return fail(MLOPS_LM_HEAD_INVALID_VALUE, "mlops_lm_head: norm_parts_n > 0 needs norm_parts");
  op = HeadOp{};
  op.w = w;
  op.x = x;
  op.norm_parts = c->norm_parts_n > 0 ? norm_parts : nullptr;
  op.logits = logits;
  op.stats_max = stats_max;
  op.stats_arg = stats_arg;
  op.hist = stats_hist;
  op.vocab = c->vocab;
  op.hidden = c->hidden;
  op.items = mlops_lm_head_item_count(c, v);
  op.n = c->n;
  op.norm_parts_n = c->norm_parts_n;
  op.inv_n = c->inv_n;
  op.eps = c->eps;
  op.logit_scale = c->logit_scale;
  return MLOPS_LM_HEAD_OK;
}

}  // namespace

extern "C" mlops_lm_head_status mlops_lm_head(const mlops_lm_head_config *c, const mlops_lm_head_variant *v, const uint16_t *w,
                                              const uint16_t *x, const float *norm_parts, float *logits, float *stats_max,
                                              int *stats_arg, unsigned *stats_hist, cudaStream_t stream) {
  const Instance *in;
  HeadOp op;
  if (const mlops_lm_head_status st = prepare(c, v, w, x, norm_parts, logits, stats_max, stats_arg, stats_hist, in, op)) return st;
  const size_t bytes = instance_smem(*in, op.n, op.hidden, op.hist != nullptr);
  if (cudaFuncSetAttribute(reinterpret_cast<const void *>(in->kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)) != cudaSuccess)
    return fail(MLOPS_LM_HEAD_CUDA_ERROR, "mlops_lm_head: %zu bytes of shared memory: %s", bytes, cudaGetErrorString(cudaGetLastError()));
  in->kernel<<<op.items, 32 * in->s, bytes, stream>>>(op);
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? MLOPS_LM_HEAD_OK : fail(MLOPS_LM_HEAD_CUDA_ERROR, "mlops_lm_head: %s", cudaGetErrorString(e));
}

extern "C" int mlops_lm_head_items(const mlops_lm_head_config *c, const mlops_lm_head_variant *v, const uint16_t *w, const uint16_t *x,
                                   const float *norm_parts, float *logits, float *stats_max, int *stats_arg, unsigned *stats_hist,
                                   mlops_vocab_items *out) {
  const Instance *in;
  HeadOp op;
  if (!out || prepare(c, v, w, x, norm_parts, logits, stats_max, stats_arg, stats_hist, in, op)) return -1;
  cudaFuncAttributes a;
  if (cudaFuncGetAttributes(&a, reinterpret_cast<const void *>(in->kernel)) != cudaSuccess) return -1;
  *out = mlops_vocab_items{};
  out->kind = in->tc ? MLOPS_VOCAB_ITEM_HEAD_TC : MLOPS_VOCAB_ITEM_HEAD;
  out->items = op.items;
  out->threads = 32 * in->s;
  out->smem_bytes = static_cast<int>(instance_smem(*in, op.n, op.hidden, op.hist != nullptr));
  out->regs = a.numRegs;
  out->arch = a.ptxVersion;
  if (in->tc) snprintf(out->entry, sizeof out->entry, "mlops_vocab_items::head_tc<%d, %d, %d, %d, %s>", in->s, in->j, in->d, in->tc - 1, in->xs ? "true" : "false");
  else snprintf(out->entry, sizeof out->entry, "mlops_vocab_items::head<%d, %d, %d, %d, %s>", in->r, in->s, in->d, in->nb, in->xs ? "true" : "false");
  snprintf(out->args_type, sizeof out->args_type, "mlops_vocab_detail::HeadOp");
  out->args_bytes = (int)sizeof op;
  mlops_vocab_detail::put_args(out->args.bytes, op);
  out->geom[0] = op.vocab; out->geom[1] = op.hidden; out->geom[2] = op.n; out->geom[3] = in->r; out->geom[4] = op.items;
  out->geom[5] = op.norm_parts_n; out->geom[6] = (logits != nullptr) | (stats_max != nullptr) << 1 | (op.norm_parts != nullptr) << 2 | (stats_hist != nullptr) << 3;
  return 1;
}

extern "C" int mlops_lm_head_items_column(mlops_vocab_items *items, const int *column, int columns) {
  if (!items || !column || columns < 1 || (items->kind != MLOPS_VOCAB_ITEM_HEAD && items->kind != MLOPS_VOCAB_ITEM_HEAD_TC) || items->geom[2] != 1) {
    fail(MLOPS_LM_HEAD_INVALID_VALUE, "mlops_lm_head_items_column: items of a one-column head, a column index and its count are needed");
    return -1;
  }
  if (reinterpret_cast<uintptr_t>(column) & 3) {
    fail(MLOPS_LM_HEAD_INVALID_VALUE, "mlops_lm_head_items_column: the column index must be 4-byte aligned");
    return -1;
  }
  mlops_vocab_detail::HeadOp op;
  memcpy(&op, items->args.bytes, sizeof op);
  op.column = column;
  mlops_vocab_detail::put_args(items->args.bytes, op);
  items->geom[7] = columns;
  return 0;
}

int mlops_vocab_detail::head_access(const mlops_vocab_items &it, int item, mlops_vocab_item_range *out, int max) {
  const long long vocab = it.geom[0], hidden = it.geom[1], n = it.geom[2], r = it.geom[3], items = it.geom[4];
  const long long in_columns = it.geom[7] ? it.geom[7] : n;  // x and norm_parts: a selected column is one of geom[7]
  const long long units = (vocab + r - 1) / r;
  const long long lo = item * units / items * r, hi = std::min((item + 1) * units / items * r, vocab);
  int count = 0;
  if (hi <= lo) return 0;
  put_range(out, max, count, MLOPS_VOCAB_ARG_W, 0, lo * hidden * 2, (hi - lo) * hidden * 2);
  put_range(out, max, count, MLOPS_VOCAB_ARG_HEAD_X, 0, 0, in_columns * hidden * 2);
  if (it.geom[6] & 4) put_range(out, max, count, MLOPS_VOCAB_ARG_NORM_PARTS, 0, 0, in_columns * it.geom[5] * 4);
  if (it.geom[7]) put_range(out, max, count, MLOPS_VOCAB_ARG_COLUMN, 0, 0, 4);
  if (it.geom[6] & 1) put_range(out, max, count, MLOPS_VOCAB_ARG_LOGITS, 1, lo * 4, (hi - lo) * 4, vocab * 4, (int)n);
  if (it.geom[6] & 2) {
    put_range(out, max, count, MLOPS_VOCAB_ARG_STATS_MAX, 1, item * 4, 4, items * 4, (int)n);
    put_range(out, max, count, MLOPS_VOCAB_ARG_STATS_ARG, 1, item * 4, 4, items * 4, (int)n);
  }
  if (it.geom[6] & 8) put_range(out, max, count, MLOPS_VOCAB_ARG_STATS_HIST, 1, item * 256 * 4, 256 * 4, items * 256 * 4, (int)n);
  return count;
}
