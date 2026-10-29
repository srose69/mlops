// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_norm host side (include/mlops_norm_layer.h): argument checks, the variants, the resolved element count, the
// description of a call as work items.
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "norm.cuh"

using namespace mlops_norm_detail;

namespace {

char *last_error_buf() {
  static thread_local char buf[256] = {0};
  return buf;
}
mlops_norm_status fail(mlops_norm_status st, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(last_error_buf(), 255, fmt, ap);
  va_end(ap);
  return st;
}

template <int KIND, int ORDER, int GATE, int XT, int YT, int OUT> __global__ void __launch_bounds__(256) norm_kernel(Params params) {
  using Kernel = NormKernel<Traits<KIND, ORDER, GATE, XT, YT, OUT>>;
  __shared__ typename Kernel::SharedStorage shared;
  Kernel(params, shared, threadIdx.x)(blockIdx.x);
}

// One instance per (flavor, dtypes, output): no dtype or format test in a kernel. The warps of the CTA are the launch's.
struct Instance {
  int kind, order, gate, xt, yt, out;
  void (*kernel)(Params);
};
constexpr int kFlavors = 7, kDtypePairs = 5, kInstances = kFlavors * kDtypePairs * 3;
template <int KIND, int ORDER, int GATE, int XT, int YT, int OUT> void add(Instance *t, int &n) {
  t[n++] = {KIND, ORDER, GATE, XT, YT, OUT, norm_kernel<KIND, ORDER, GATE, XT, YT, OUT>};
}
template <int KIND, int ORDER, int GATE, int XT, int YT> void add_outputs(Instance *t, int &n) {
  add<KIND, ORDER, GATE, XT, YT, MLOPS_NORM_OUT_DTYPE>(t, n);
  add<KIND, ORDER, GATE, XT, YT, MLOPS_NORM_OUT_INT8>(t, n);
  add<KIND, ORDER, GATE, XT, YT, MLOPS_NORM_OUT_NVFP4>(t, n);
}
// x in the dtype of weight and y, or an FP32 residual stream
template <int KIND, int ORDER, int GATE> void add_dtypes(Instance *t, int &n) {
  add_outputs<KIND, ORDER, GATE, kBf16, kBf16>(t, n);
  add_outputs<KIND, ORDER, GATE, kFp32, kBf16>(t, n);
  add_outputs<KIND, ORDER, GATE, kFp16, kFp16>(t, n);
  add_outputs<KIND, ORDER, GATE, kFp32, kFp16>(t, n);
  add_outputs<KIND, ORDER, GATE, kFp32, kFp32>(t, n);
}
const Instance *instances() {
  static Instance table[kInstances];
  static const bool init = [] {
    int n = 0;
    add_dtypes<MLOPS_NORM_RMS, MLOPS_NORM_ORDER_ROUND_THEN_SCALE, MLOPS_NORM_GATE_NONE>(table, n);
    add_dtypes<MLOPS_NORM_RMS, MLOPS_NORM_ORDER_FP32, MLOPS_NORM_GATE_NONE>(table, n);
    add_dtypes<MLOPS_NORM_RMS, MLOPS_NORM_ORDER_FP32, MLOPS_NORM_GATE_AFTER>(table, n);
    add_dtypes<MLOPS_NORM_RMS, MLOPS_NORM_ORDER_FP32, MLOPS_NORM_GATE_BEFORE>(table, n);
    add_dtypes<MLOPS_NORM_LAYER, MLOPS_NORM_ORDER_FP32, MLOPS_NORM_GATE_NONE>(table, n);
    add_dtypes<MLOPS_NORM_LAYER, MLOPS_NORM_ORDER_FP32, MLOPS_NORM_GATE_AFTER>(table, n);
    add_dtypes<MLOPS_NORM_LAYER, MLOPS_NORM_ORDER_FP32, MLOPS_NORM_GATE_BEFORE>(table, n);
    return true;
  }();
  (void)init;
  return table;
}

bool in_range(int v, int lo, int hi) { return v >= lo && v <= hi; }
// A configuration the instances can run (the rest of the checks, on the pointers, are in prepare)
bool valid(const mlops_norm_config *c) {
  if (!c || c->rows <= 0 || c->hidden <= 0 || c->eps < 0.f || c->items < 0 || c->ld_scale < 0 || c->groups < 0) return false;
  const int groups = c->groups ? c->groups : 1;
  if (c->hidden % (256 * groups) != 0 || c->hidden > 256 * kMaxUnits) return false;
  if (!in_range(c->kind, 0, 1) || !in_range(c->scale, 0, 2) || !in_range(c->order, 0, 1) || !in_range(c->gate, 0, 2) || !in_range(c->out, 0, 2)) return false;
  if (!in_range(c->x_dtype, 0, 2) || !in_range(c->y_dtype, 0, 2) || (c->x_dtype != c->y_dtype && c->x_dtype != MLOPS_NORM_FP32)) return false;
  if (c->bias != 0 && c->bias != 1) return false;
  if (c->ld_scale && c->ld_scale < c->hidden / 16) return false;
  // Hugging Face's order is RMSNorm's alone, with no bias and no gate; a gate and a bias are single-rounding things
  if (c->order == MLOPS_NORM_ORDER_ROUND_THEN_SCALE && (c->kind != MLOPS_NORM_RMS || c->bias || c->gate != MLOPS_NORM_GATE_NONE)) return false;
  return true;
}
bool aligned(const void *p, uintptr_t bytes) { return reinterpret_cast<uintptr_t>(p) % bytes == 0; }
const Instance *find(const mlops_norm_config &c) {
  const Instance *t = instances();
  for (int i = 0; i < kInstances; ++i)
    if (t[i].kind == c.kind && t[i].order == c.order && t[i].gate == c.gate && t[i].xt == c.x_dtype && t[i].yt == c.y_dtype && t[i].out == c.out) return t + i;
  return nullptr;
}
int elem_bytes(int dt) { return dt == MLOPS_NORM_FP32 ? 4 : 2; }
bool warps_ok(const mlops_norm_config &c, const mlops_norm_variant *v) {
  return v && (v->warps == 1 || v->warps == 2 || v->warps == 4 || v->warps == 8) && v->threads == 32 * v->warps && c.hidden / 256 >= v->warps;
}

}  // namespace

extern "C" int mlops_norm_variants(const mlops_norm_config *c, mlops_norm_variant *out, int max) {
  if (!valid(c) || max < 0 || (max > 0 && !out)) return 0;
  const Instance *in = find(*c);
  if (!in) return 0;
  cudaFuncAttributes a;
  if (cudaFuncGetAttributes(&a, reinterpret_cast<const void *>(in->kernel)) != cudaSuccess) return 0;
  int n = 0;
  for (int w : {1, 2, 4, 8}) {
    if (c->hidden / 256 < w) continue;
    if (n < max) out[n] = {w, 32 * w, (int)sizeof(NormKernel<Traits<0, 0, 0, 0, 0, 0>>::SharedStorage), a.numRegs, a.binaryVersion};
    ++n;
  }
  return n;
}

extern "C" int mlops_norm_item_count(const mlops_norm_config *c, const mlops_norm_variant *v) {
  const Instance *in = valid(c) ? find(*c) : nullptr;
  if (!in || !warps_ok(*c, v)) return 0;
  int items = c->items;
  if (!items) {
    int dev = 0, sms = 1, occ = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, reinterpret_cast<const void *>(in->kernel), v->threads, 0);
    items = sms * std::max(occ, 1);
  }
  return std::min(items, c->rows);
}

extern "C" const char *mlops_norm_last_error(void) { return last_error_buf(); }

namespace {

// Validates a call's arguments and builds its device arguments; shared by the call and its description.
mlops_norm_status prepare(const mlops_norm_config *c, const mlops_norm_variant *v, const mlops_norm_args *a, const Instance *&in, Params &op) {
  if (!valid(c))
    return fail(MLOPS_NORM_INVALID_VALUE,
                "mlops_norm: rows > 0, hidden a multiple of 256 * groups up to %d, eps >= 0, x_dtype y's or FP32, ld_scale 0 or >= hidden / 16, "
                "ORDER_ROUND_THEN_SCALE for RMS only with no bias and no gate", 256 * kMaxUnits);
  in = find(*c);
  if (!in || !warps_ok(*c, v)) return fail(MLOPS_NORM_INVALID_VALUE, "mlops_norm: variant not among mlops_norm_variants of this configuration");
  if (!a || !a->x || !a->y) return fail(MLOPS_NORM_INVALID_VALUE, "mlops_norm: null args, x or y");
  if ((a->residual == nullptr) != (a->residual_out == nullptr)) return fail(MLOPS_NORM_INVALID_VALUE, "mlops_norm: residual and residual_out go together");
  if ((c->scale != MLOPS_NORM_SCALE_NONE) != (a->weight != nullptr)) return fail(MLOPS_NORM_INVALID_VALUE, "mlops_norm: weight is given exactly when scale is not NONE");
  if ((c->bias != 0) != (a->bias != nullptr)) return fail(MLOPS_NORM_INVALID_VALUE, "mlops_norm: bias is given exactly when config.bias");
  if ((c->gate != MLOPS_NORM_GATE_NONE) != (a->gate != nullptr)) return fail(MLOPS_NORM_INVALID_VALUE, "mlops_norm: gate is given exactly when the gate is not NONE");
  if (c->out != MLOPS_NORM_OUT_DTYPE && !a->scales) return fail(MLOPS_NORM_INVALID_VALUE, "mlops_norm: a quantized output needs scales");
  const void *const ptrs[] = {a->x, a->residual, a->weight, a->bias, a->gate, a->residual_out, a->y};
  for (const void *p : ptrs)
    if (p && !aligned(p, 16)) return fail(MLOPS_NORM_INVALID_VALUE, "mlops_norm: x, residual, weight, bias, gate, residual_out and y must be 16-byte aligned");
  if (a->scales && !aligned(a->scales, 4)) return fail(MLOPS_NORM_INVALID_VALUE, "mlops_norm: scales must be 4-byte aligned");
  op = Params{};
  op.x = a->x;
  op.residual = a->residual;
  op.weight = a->weight;
  op.bias = a->bias;
  op.gate = a->gate;
  op.residual_out = a->residual_out;
  op.y = a->y;
  op.scales = a->scales;
  op.global = a->global;
  op.rows = c->rows;
  op.hidden = c->hidden;
  op.groups = c->groups ? c->groups : 1;
  op.items = mlops_norm_item_count(c, v);
  op.ld_scale = c->ld_scale ? c->ld_scale : c->hidden / 16;
  op.eps = c->eps;
  op.inv_n = 1.f / (float)(c->hidden / op.groups);
  op.w_alpha = c->scale == MLOPS_NORM_SCALE_NONE ? 0.f : 1.f;
  op.w_beta = c->scale == MLOPS_NORM_SCALE_WEIGHT ? 0.f : 1.f;
  op.has_weight = c->scale != MLOPS_NORM_SCALE_NONE;
  op.has_bias = c->bias;
  return MLOPS_NORM_OK;
}

}  // namespace

extern "C" mlops_norm_status mlops_norm(const mlops_norm_config *c, const mlops_norm_variant *v, const mlops_norm_args *a, cudaStream_t stream) {
  const Instance *in;
  Params op;
  if (const mlops_norm_status st = prepare(c, v, a, in, op)) return st;
  in->kernel<<<op.items, v->threads, 0, stream>>>(op);
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? MLOPS_NORM_OK : fail(MLOPS_NORM_CUDA_ERROR, "mlops_norm: %s", cudaGetErrorString(e));
}

extern "C" int mlops_norm_describe(const mlops_norm_config *c, const mlops_norm_variant *v, const mlops_norm_args *a, mlops_norm_items *out) {
  const Instance *in;
  Params op;
  if (!out || prepare(c, v, a, in, op)) return -1;
  static_assert(sizeof(Params) <= MLOPS_NORM_ITEM_ARGS_BYTES, "the arguments fit an item's blob");
  cudaFuncAttributes at;
  if (cudaFuncGetAttributes(&at, reinterpret_cast<const void *>(in->kernel)) != cudaSuccess) return -1;
  *out = mlops_norm_items{};
  out->kind = MLOPS_NORM_ITEM_NORM;
  out->items = op.items;
  out->threads = v->threads;
  out->smem_bytes = (int)sizeof(NormKernel<Traits<0, 0, 0, 0, 0, 0>>::SharedStorage);
  out->regs = at.numRegs;
  out->arch = at.ptxVersion;
  snprintf(out->entry, sizeof out->entry, "mlops_norm_entries::norm<%d, %d, %d, %d, %d, %d>", in->kind, in->order, in->gate, in->xt, in->yt, in->out);
  snprintf(out->args_type, sizeof out->args_type, "mlops_norm_detail::Params");
  out->args_bytes = (int)sizeof op;
  memcpy(out->args.bytes, &op, sizeof op);
  out->geom[0] = op.rows; out->geom[1] = op.hidden; out->geom[2] = op.items; out->geom[3] = elem_bytes(c->x_dtype);
  out->geom[4] = elem_bytes(c->y_dtype); out->geom[5] = c->out; out->geom[6] = op.ld_scale; out->geom[7] = a->residual != nullptr;
  out->geom[8] = a->global != nullptr; out->geom[9] = op.has_weight; out->geom[10] = op.has_bias; out->geom[11] = a->gate != nullptr;
  return 1;
}

extern "C" int mlops_norm_item_access(const mlops_norm_items *it, int item, mlops_norm_item_range *out, int max) {
  if (!it || item < 0 || item >= it->items || max < 0 || (max > 0 && !out) || it->kind != MLOPS_NORM_ITEM_NORM) return -1;
  const long long rows = it->geom[0], hidden = it->geom[1], items = it->geom[2], xb = it->geom[3], yb = it->geom[4], ld = it->geom[6];
  const long long lo = item * rows / items, hi = (item + 1) * rows / items, n = hi - lo;
  const int kind = (int)it->geom[5];
  int count = 0;
  auto put = [&](int arg, int write, long long offset, long long bytes, long long stride = 0, int nrows = 1) {
    if (count < max) out[count] = {arg, write, offset, bytes, stride, nrows};
    ++count;
  };
  if (n <= 0) return 0;
  put(MLOPS_NORM_ARG_X, 0, lo * hidden * xb, n * hidden * xb);
  if (it->geom[7]) {
    put(MLOPS_NORM_ARG_RESIDUAL, 0, lo * hidden * xb, n * hidden * xb);
    put(MLOPS_NORM_ARG_RESIDUAL_OUT, 1, lo * hidden * xb, n * hidden * xb);
  }
  if (it->geom[11]) put(MLOPS_NORM_ARG_GATE, 0, lo * hidden * xb, n * hidden * xb);
  if (it->geom[9]) put(MLOPS_NORM_ARG_WEIGHT, 0, 0, hidden * yb);
  if (it->geom[10]) put(MLOPS_NORM_ARG_BIAS, 0, 0, hidden * yb);
  const long long ybytes = kind == MLOPS_NORM_OUT_DTYPE ? hidden * yb : kind == MLOPS_NORM_OUT_INT8 ? hidden : hidden / 2;
  put(MLOPS_NORM_ARG_Y, 1, lo * ybytes, n * ybytes);
  if (kind != MLOPS_NORM_OUT_DTYPE) {
    const long long sz = kind == MLOPS_NORM_OUT_INT8 ? 4 : 1;
    put(MLOPS_NORM_ARG_SCALES, 1, lo * ld * sz, hidden / 16 * sz, ld * sz, (int)n);
  }
  if (it->geom[8]) put(MLOPS_NORM_ARG_GLOBAL, 0, 0, 4);
  return count;
}
