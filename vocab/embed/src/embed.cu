// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_embed host side (include/mlops_embed.h): argument checks, the variants, the resolved element count.
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>

#include <cstring>

#include "embed.cuh"
#include "items.h"

using namespace mlops_vocab_detail;

namespace {

char *last_error_buf() {
  static thread_local char buf[256] = {0};
  return buf;
}
mlops_embed_status fail(mlops_embed_status st, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(last_error_buf(), 255, fmt, ap);
  va_end(ap);
  return st;
}

template <int W> __global__ void __launch_bounds__(32 * W) embed_kernel(EmbedOp op) { embed_item<W>(op, blockIdx.x, threadIdx.x); }

struct Instance {
  int warps;
  void (*kernel)(EmbedOp);
};
const Instance kInstances[] = {{1, embed_kernel<1>}, {2, embed_kernel<2>}, {4, embed_kernel<4>}, {8, embed_kernel<8>}};

bool valid(const mlops_embed_config *c) { return c && c->vocab > 0 && c->hidden > 0 && c->hidden % 256 == 0 && c->n >= 1 && c->items >= 0; }
bool aligned(const void *p, uintptr_t bytes) { return reinterpret_cast<uintptr_t>(p) % bytes == 0; }
const Instance *accepted(const mlops_embed_variant *v) {
  if (!v) return nullptr;
  for (const Instance &in : kInstances)
    if (in.warps == v->warps) return &in;
  return nullptr;
}

}  // namespace

extern "C" int mlops_embed_variants(const mlops_embed_config *c, mlops_embed_variant *out, int max) {
  if (!valid(c) || max < 0 || (max > 0 && !out)) return 0;
  int n = 0;
  for (const Instance &in : kInstances) {
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, reinterpret_cast<const void *>(in.kernel)) != cudaSuccess) return 0;
    if (n < max) out[n] = {in.warps, 32 * in.warps, 0, a.numRegs, a.binaryVersion};
    ++n;
  }
  return n;
}

extern "C" int mlops_embed_item_count(const mlops_embed_config *c, const mlops_embed_variant *v) {
  const Instance *in = valid(c) ? accepted(v) : nullptr;
  if (!in) return 0;
  int items = c->items;
  if (!items) {
    int dev = 0, sms = 1, occ = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, reinterpret_cast<const void *>(in->kernel), 32 * in->warps, 0);
    items = sms * std::max(occ, 1);
  }
  return (int)std::min<long long>(items, (long long)c->n * (c->hidden / 256));
}

extern "C" const char *mlops_embed_last_error(void) { return last_error_buf(); }

namespace {

// Validates a call's arguments and builds its device arguments; shared by the call and its description.
mlops_embed_status prepare(const mlops_embed_config *c, const mlops_embed_variant *v, const uint16_t *table, const int *ids,
                           const uint16_t *norm_w, float *x, uint16_t *xw, float *parts, int *error, const Instance *&in, EmbedOp &op) {
  if (!valid(c)) return fail(MLOPS_EMBED_INVALID_VALUE, "mlops_embed: vocab > 0, hidden a positive multiple of 256, n >= 1, items >= 0");
  in = accepted(v);
  if (!in) return fail(MLOPS_EMBED_INVALID_VALUE, "mlops_embed: variant not among mlops_embed_variants");
  if (!table || !ids) return fail(MLOPS_EMBED_INVALID_VALUE, "mlops_embed: null table or ids");
  if (!x && !xw && !parts) return fail(MLOPS_EMBED_INVALID_VALUE, "mlops_embed: no output requested");
  if (xw && !norm_w) return fail(MLOPS_EMBED_INVALID_VALUE, "mlops_embed: xw needs norm_w");
  if (!aligned(table, 16) || !aligned(x ? (const void *)x : table, 16) || !aligned(xw ? (const void *)xw : table, 16) ||
      !aligned(norm_w ? (const void *)norm_w : table, 16))
    return fail(MLOPS_EMBED_INVALID_VALUE, "mlops_embed: table, norm_w, x and xw must be 16-byte aligned");
  op = EmbedOp{};
  op.table = table;
  op.ids = ids;
  op.w = norm_w;
  op.x = x;
  op.xw = xw;
  op.parts = parts;
  op.error = error;
  op.vocab = c->vocab;
  op.hidden = c->hidden;
  op.n = c->n;
  op.items = mlops_embed_item_count(c, v);
  return MLOPS_EMBED_OK;
}

}  // namespace

extern "C" mlops_embed_status mlops_embed(const mlops_embed_config *c, const mlops_embed_variant *v, const uint16_t *table, const int *ids,
                                          const uint16_t *norm_w, float *x, uint16_t *xw, float *parts, int *error,
                                          cudaStream_t stream) {
  const Instance *in;
  EmbedOp op;
  if (const mlops_embed_status st = prepare(c, v, table, ids, norm_w, x, xw, parts, error, in, op)) return st;
  in->kernel<<<op.items, 32 * in->warps, 0, stream>>>(op);
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? MLOPS_EMBED_OK : fail(MLOPS_EMBED_CUDA_ERROR, "mlops_embed: %s", cudaGetErrorString(e));
}

extern "C" int mlops_embed_items(const mlops_embed_config *c, const mlops_embed_variant *v, const uint16_t *table, const int *ids,
                                 const uint16_t *norm_w, float *x, uint16_t *xw, float *parts, int *error, mlops_vocab_items *out) {
  const Instance *in;
  EmbedOp op;
  if (!out || prepare(c, v, table, ids, norm_w, x, xw, parts, error, in, op)) return -1;
  cudaFuncAttributes a;
  if (cudaFuncGetAttributes(&a, reinterpret_cast<const void *>(in->kernel)) != cudaSuccess) return -1;
  *out = mlops_vocab_items{};
  out->kind = MLOPS_VOCAB_ITEM_EMBED;
  out->items = op.items;
  out->threads = 32 * in->warps;
  out->regs = a.numRegs;
  out->arch = a.ptxVersion;
  snprintf(out->entry, sizeof out->entry, "mlops_vocab_items::embed<%d>", in->warps);
  snprintf(out->args_type, sizeof out->args_type, "mlops_vocab_detail::EmbedOp");
  out->args_bytes = (int)sizeof op;
  memcpy(out->args.bytes, &op, sizeof op);
  out->geom[0] = op.vocab; out->geom[1] = op.hidden; out->geom[2] = op.n; out->geom[3] = in->warps;
  out->geom[4] = norm_w && xw; out->geom[5] = x != nullptr; out->geom[6] = parts != nullptr; out->geom[7] = error != nullptr;
  return 1;
}

int mlops_vocab_detail::embed_access(const mlops_vocab_items &it, int item, mlops_vocab_item_range *out, int max) {
  const long long vocab = it.geom[0], hidden = it.geom[1], n = it.geom[2], per_col = hidden / 256, total = n * per_col;
  const long long lo = item * total / it.items, hi = (item + 1) * total / it.items;
  int count = 0;
  if (hi <= lo) return 0;
  // The table rows are chosen by the ids at run time: the whole table is the envelope.
  put_range(out, max, count, MLOPS_VOCAB_ARG_TABLE, 0, 0, vocab * hidden * 2);
  put_range(out, max, count, MLOPS_VOCAB_ARG_IDS, 0, (lo / per_col) * 4, ((hi - 1) / per_col - lo / per_col + 1) * 4);
  if (it.geom[4]) put_range(out, max, count, MLOPS_VOCAB_ARG_NORM_W, 0, 0, hidden * 2);
  if (it.geom[5]) put_range(out, max, count, MLOPS_VOCAB_ARG_X, 1, lo * 256 * 4, (hi - lo) * 256 * 4);
  if (it.geom[4]) put_range(out, max, count, MLOPS_VOCAB_ARG_XW, 1, lo * 256 * 2, (hi - lo) * 256 * 2);
  if (it.geom[6]) put_range(out, max, count, MLOPS_VOCAB_ARG_PARTS, 1, lo * 4, (hi - lo) * 4);
  if (it.geom[7]) put_range(out, max, count, MLOPS_VOCAB_ARG_ERROR, 1, 0, 4);
  return count;
}
