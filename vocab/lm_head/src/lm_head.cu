// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_lm_head host side (include/mlops_lm_head.h): argument checks, the variants, the resolved element count.
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>

#include <cstring>

#include "items.h"
#include "lm_head.cuh"
#include "lm_head_mma.cuh"

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

template <int R, int S, int D, int NB> __global__ void __launch_bounds__(32 * S) head_kernel(HeadOp op) {
  __shared__ HeadSmem<S, NB> sm;
  head_item<R, S, D, NB>(op, blockIdx.x, threadIdx.x, sm);
}

// The instances. One table drives the kernels, the variants and their attributes. An instance exists when its
// units tile the phase; the variants of a device are those of them that use no local memory there.
template <int S, int J, int D, int KIND> __global__ void __launch_bounds__(32 * S) head_kernel_mma(HeadOp op) {
  __shared__ MmaSmem<S> sm;
  head_item_mma<S, J, D, KIND>(op, blockIdx.x, threadIdx.x, sm);
}

struct Instance {
  int r, s, d, nb, tc, j, smem;
  void (*kernel)(HeadOp);
};
constexpr int kMaxInstances = 4 * 3 * 3 * 4 + 3 * 3 * 3 * 3;
template <int R, int S, int D, int NB> void add(Instance *table, int &count) {
  if constexpr ((32 * S / NB) % R == 0)
    table[count++] = {R, S, D, NB, 0, 0, (int)sizeof(HeadSmem<S, NB>), head_kernel<R, S, D, NB>};
}
template <int S, int J, int D, int KIND> void add_mma(Instance *table, int &count) {
  table[count++] = {KIND == 2 ? 32 : 16, S, D, 8, KIND + 1, J, (int)sizeof(MmaSmem<S>), head_kernel_mma<S, J, D, KIND>};
}
template <int S, int J, int KIND> void add_mma_depths(Instance *t, int &n) { add_mma<S, J, 1, KIND>(t, n); add_mma<S, J, 2, KIND>(t, n); add_mma<S, J, 4, KIND>(t, n); }
template <int S, int KIND> void add_mma_chunks(Instance *t, int &n) { add_mma_depths<S, 1, KIND>(t, n); add_mma_depths<S, 2, KIND>(t, n); add_mma_depths<S, 4, KIND>(t, n); }
template <int NB, int R, int S> void add_depths(Instance *t, int &n) { add<R, S, 1, NB>(t, n); add<R, S, 2, NB>(t, n); add<R, S, 4, NB>(t, n); }
template <int NB, int R> void add_pieces(Instance *t, int &n) { add_depths<NB, R, 2>(t, n); add_depths<NB, R, 4>(t, n); add_depths<NB, R, 8>(t, n); }
template <int NB> void add_rows(Instance *t, int &n) { add_pieces<NB, 1>(t, n); add_pieces<NB, 2>(t, n); add_pieces<NB, 4>(t, n); add_pieces<NB, 8>(t, n); }
const Instance *instances(int &count) {
  static Instance table[kMaxInstances];
  static int n = 0;
  static const bool init = [] {
    add_rows<1>(table, n);
    add_rows<2>(table, n);
    add_rows<4>(table, n);
    add_rows<8>(table, n);
    add_mma_chunks<2, 0>(table, n);
    add_mma_chunks<4, 0>(table, n);
    add_mma_chunks<8, 0>(table, n);
    add_mma_chunks<2, 1>(table, n);
    add_mma_chunks<4, 1>(table, n);
    add_mma_chunks<8, 1>(table, n);
    add_mma_chunks<2, 2>(table, n);
    add_mma_chunks<4, 2>(table, n);
    add_mma_chunks<8, 2>(table, n);
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
        t[i].j == v.chunks)
      return t + i;
  return nullptr;
}
// Tensor-core instances need sm_80 (BF16 mma), sm_75 (FP16 m16n8k8) or sm_70 (FP16 m8n8k4) and a hidden size that
// their blocks (32 J elements; 8 J for m8n8k4) tile into at least S pieces.
bool structurally_fits(const Instance &in, const mlops_lm_head_config &c) {
  if (in.nb != columns_for(c.n)) return false;
  if (!in.tc) return c.hidden / 256 >= in.s;
  int dev = 0, major = 0, minor = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
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
    if (n < max) out[n] = {in.r, in.s, in.d, in.nb, in.tc, in.j, 32 * in.s, in.smem, a.numRegs, a.binaryVersion};
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
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, reinterpret_cast<const void *>(in->kernel), 32 * in->s, 0);
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
  in->kernel<<<op.items, 32 * in->s, 0, stream>>>(op);
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
  out->smem_bytes = in->smem;
  out->regs = a.numRegs;
  out->arch = a.ptxVersion;
  if (in->tc) snprintf(out->entry, sizeof out->entry, "mlops_vocab_items::head_tc<%d, %d, %d, %d>", in->s, in->j, in->d, in->tc - 1);
  else snprintf(out->entry, sizeof out->entry, "mlops_vocab_items::head<%d, %d, %d, %d>", in->r, in->s, in->d, in->nb);
  snprintf(out->args_type, sizeof out->args_type, "mlops_vocab_detail::HeadOp");
  out->args_bytes = (int)sizeof op;
  memcpy(out->args.bytes, &op, sizeof op);
  out->geom[0] = op.vocab; out->geom[1] = op.hidden; out->geom[2] = op.n; out->geom[3] = in->r; out->geom[4] = op.items;
  out->geom[5] = op.norm_parts_n; out->geom[6] = (logits != nullptr) | (stats_max != nullptr) << 1 | (op.norm_parts != nullptr) << 2 | (stats_hist != nullptr) << 3;
  return 1;
}

int mlops_vocab_detail::head_access(const mlops_vocab_items &it, int item, mlops_vocab_item_range *out, int max) {
  const long long vocab = it.geom[0], hidden = it.geom[1], n = it.geom[2], r = it.geom[3], items = it.geom[4];
  const long long units = (vocab + r - 1) / r;
  const long long lo = item * units / items * r, hi = std::min((item + 1) * units / items * r, vocab);
  int count = 0;
  if (hi <= lo) return 0;
  put_range(out, max, count, MLOPS_VOCAB_ARG_W, 0, lo * hidden * 2, (hi - lo) * hidden * 2);
  put_range(out, max, count, MLOPS_VOCAB_ARG_HEAD_X, 0, 0, n * hidden * 2);
  if (it.geom[6] & 4) put_range(out, max, count, MLOPS_VOCAB_ARG_NORM_PARTS, 0, 0, n * it.geom[5] * 4);
  if (it.geom[6] & 1) put_range(out, max, count, MLOPS_VOCAB_ARG_LOGITS, 1, lo * 4, (hi - lo) * 4, vocab * 4, (int)n);
  if (it.geom[6] & 2) {
    put_range(out, max, count, MLOPS_VOCAB_ARG_STATS_MAX, 1, item * 4, 4, items * 4, (int)n);
    put_range(out, max, count, MLOPS_VOCAB_ARG_STATS_ARG, 1, item * 4, 4, items * 4, (int)n);
  }
  if (it.geom[6] & 8) put_range(out, max, count, MLOPS_VOCAB_ARG_STATS_HIST, 1, item * 256 * 4, 256 * 4, items * 256 * 4, (int)n);
  return count;
}
