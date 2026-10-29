// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_sample host side (include/mlops_sample.h): argument checks, the workspace, one kernel per pass.
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>

#include <cstring>

#include "items.h"
#include "sample.cuh"

using namespace mlops_vocab_detail;

namespace {

char *last_error_buf() {
  static thread_local char buf[256] = {0};
  return buf;
}
mlops_sample_status fail(mlops_sample_status st, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(last_error_buf(), 255, fmt, ap);
  va_end(ap);
  return st;
}

bool valid(const mlops_sample_config *c) {
  return c && c->vocab > 0 && c->temperature >= 0.f && c->top_k >= 0 && c->top_p > 0.f && c->top_p <= 1.f && c->penalty > 0.f &&
         c->ctas >= 0;
}
size_t up256(size_t n) { return (n + 255) & ~(size_t)255; }

// The workspace's arrays, in order, 256-byte aligned; the counters come first (the only part that must start zero).
struct Layout {
  size_t counters, sampled, chunk_m, chunk_arg, chunk_mass, hist, cand_n, cand, max_logit, ctl, bytes;
  explicit Layout(int vocab) {
    const size_t ch = (vocab + kChunk - 1) / kChunk;
    size_t at = 0;
    auto take = [&](size_t n) { const size_t r = at; at += up256(n); return r; };
    counters = take(kSampleCounters * sizeof(unsigned));
    sampled = take((size_t)vocab * sizeof(float));
    chunk_m = take(ch * sizeof(float));
    chunk_arg = take(ch * sizeof(int));
    chunk_mass = take(ch * sizeof(unsigned long long));
    hist = take(kBuckets * sizeof(unsigned long long));
    cand_n = take(ch * sizeof(int));
    cand = take(kGather * sizeof(int2));
    max_logit = take(sizeof(float));
    ctl = take(4 * sizeof(unsigned long long));
    bytes = at;
  }
};

// One kernel per pass: the pass is a constant, so the switch of sample_pass_item folds away and each pass gets the
// registers it needs.
template <int PASS> __global__ void __launch_bounds__(kThreads) sample_kernel(SampleOp op, bool final) {
  __shared__ SampleSmem sm;
  __shared__ ItemSmem is;
  if (PASS == kPassPre || PASS == kPassRadix0Pre) sample_run_item(op, PASS, final, 0, sm, is);
  else
    for (int c = blockIdx.x; c < op.chunks(); c += gridDim.x) sample_run_item(op, PASS, final, c, sm, is);
}

using PassKernel = void (*)(SampleOp, bool);
PassKernel kernel_of(int pass) {
  static const PassKernel table[kPasses] = {sample_kernel<0>, sample_kernel<1>, sample_kernel<2>, sample_kernel<3>, sample_kernel<4>,
                                            sample_kernel<5>, sample_kernel<6>, sample_kernel<7>, sample_kernel<8>, sample_kernel<9>,
                                            sample_kernel<10>};
  return table[pass];
}

}  // namespace

extern "C" size_t mlops_sample_workspace_bytes(const mlops_sample_config *c) { return valid(c) ? Layout(c->vocab).bytes : 0; }

extern "C" const char *mlops_sample_last_error(void) { return last_error_buf(); }

namespace {

struct Plan {
  SampleOp op;
  Layout layout;
  int ctas, passes, pass[8];
  explicit Plan(int vocab) : layout(vocab) {}
};

// Validates a call's arguments and builds its device arguments and passes; shared by the call and its description.
mlops_sample_status prepare(const mlops_sample_config *c, const float *logits, const uint32_t *seen, const mlops_sample_stats *stats,
                            mlops_sample_state *state, void *workspace, size_t workspace_bytes, int *token, Plan *&plan) {
  if (!valid(c)) return fail(MLOPS_SAMPLE_INVALID_VALUE, "mlops_sample: vocab > 0, temperature >= 0, top_k >= 0, 0 < top_p <= 1, penalty > 0, ctas >= 0");
  if (!logits || !state || !workspace || !token) return fail(MLOPS_SAMPLE_INVALID_VALUE, "mlops_sample: null argument");
  if (reinterpret_cast<uintptr_t>(logits) % 16) return fail(MLOPS_SAMPLE_INVALID_VALUE, "mlops_sample: logits must be 16-byte aligned");
  if (c->penalty != 1.f && !seen) return fail(MLOPS_SAMPLE_INVALID_VALUE, "mlops_sample: penalty != 1 needs seen");
  if (stats && (c->penalty != 1.f || !stats->max || !stats->arg || stats->count < 1))
    return fail(MLOPS_SAMPLE_INVALID_VALUE, "mlops_sample: stats need penalty 1 and count >= 1 with max and arg");
  plan = new Plan(c->vocab);
  const Layout &L = plan->layout;
  if (workspace_bytes < L.bytes) {
    delete plan;
    plan = nullptr;
    return fail(MLOPS_SAMPLE_INVALID_VALUE, "mlops_sample: workspace of %zu bytes, need %zu", workspace_bytes, L.bytes);
  }
  if (reinterpret_cast<uintptr_t>(workspace) % 256) {
    delete plan;
    plan = nullptr;
    return fail(MLOPS_SAMPLE_INVALID_VALUE, "mlops_sample: workspace must be 256-byte aligned");
  }
  char *base = static_cast<char *>(workspace);
  SampleOp &op = plan->op;
  op = SampleOp{};
  op.logits = logits;
  op.vocab = c->vocab;
  op.seen = seen;
  op.temperature = c->temperature;
  op.top_k = c->top_k;
  op.top_p = c->top_p;
  op.penalty = c->penalty;
  op.inv_temperature = c->temperature > 0.f ? 1.f / c->temperature : 0.f;
  op.inv_penalty = 1.f / c->penalty;
  // With statistics the logits are the sampled ones: the later passes read them in place, nothing is copied.
  op.sampled = stats ? const_cast<float *>(logits) : reinterpret_cast<float *>(base + L.sampled);
  if (stats) {
    op.pre_max = stats->max;
    op.pre_arg = stats->arg;
    op.pre_count = stats->count;
    if (stats->hist && c->top_k > 0 && c->temperature > 0.f) op.pre_hist = stats->hist;
  }
  op.chunk_m = reinterpret_cast<float *>(base + L.chunk_m);
  op.chunk_arg = reinterpret_cast<int *>(base + L.chunk_arg);
  op.chunk_mass = reinterpret_cast<unsigned long long *>(base + L.chunk_mass);
  op.hist = reinterpret_cast<unsigned long long *>(base + L.hist);
  op.cand_n = reinterpret_cast<int *>(base + L.cand_n);
  op.cand = reinterpret_cast<int2 *>(base + L.cand);
  op.max_logit = reinterpret_cast<float *>(base + L.max_logit);
  op.ctl = reinterpret_cast<unsigned long long *>(base + L.ctl);
  op.counters = reinterpret_cast<unsigned *>(base + L.counters);
  op.token = token;
  op.rng = state;

  int ctas = c->ctas;
  if (!ctas) {
    int dev = 0, sms = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    ctas = 4 * sms;
  }
  plan->ctas = std::min(ctas, op.chunks());
  plan->passes = sample_passes(c->temperature, c->top_k, c->top_p, stats != nullptr, op.pre_hist != nullptr, plan->pass);
  return MLOPS_SAMPLE_OK;
}

}  // namespace

extern "C" mlops_sample_status mlops_sample(const mlops_sample_config *c, const float *logits, const uint32_t *seen,
                                            const mlops_sample_stats *stats, mlops_sample_state *state, void *workspace,
                                            size_t workspace_bytes, int *token, cudaStream_t stream) {
  Plan *plan;
  if (const mlops_sample_status st = prepare(c, logits, seen, stats, state, workspace, workspace_bytes, token, plan)) return st;
  for (int i = 0; i < plan->passes; ++i)
    kernel_of(plan->pass[i])<<<plan->pass[i] == kPassPre || plan->pass[i] == kPassRadix0Pre ? 1 : plan->ctas, kThreads, 0, stream>>>(plan->op, i == plan->passes - 1);
  delete plan;
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? MLOPS_SAMPLE_OK : fail(MLOPS_SAMPLE_CUDA_ERROR, "mlops_sample: %s", cudaGetErrorString(e));
}

extern "C" int mlops_sample_items(const mlops_sample_config *c, const float *logits, const uint32_t *seen, const mlops_sample_stats *stats,
                                  mlops_sample_state *state, void *workspace, size_t workspace_bytes, int *token, mlops_vocab_items *out,
                                  int max) {
  static_assert(sizeof(SampleOp) <= MLOPS_VOCAB_ITEM_ARGS_BYTES, "the sampling arguments fit an item's blob");
  Plan *plan;
  if (max < 0 || (max > 0 && !out)) return -1;
  if (prepare(c, logits, seen, stats, state, workspace, workspace_bytes, token, plan)) return -1;
  int written = 0;
  for (int i = 0; i < plan->passes && written < max; ++i, ++written) {
    const int pass = plan->pass[i];
    const bool final = i == plan->passes - 1;
    mlops_vocab_items &it = out[written];
    it = mlops_vocab_items{};
    it.kind = MLOPS_VOCAB_ITEM_SAMPLE;
    it.items = pass == kPassPre || pass == kPassRadix0Pre ? 1 : plan->op.chunks();
    it.threads = kThreads;
    it.smem_bytes = (int)sizeof(SampleItemSmem);
    cudaFuncAttributes a;
    cudaFuncGetAttributes(&a, reinterpret_cast<const void *>(kernel_of(pass)));
    it.regs = a.numRegs;
    it.arch = a.ptxVersion;
    snprintf(it.entry, sizeof it.entry, "mlops_vocab_items::sample<%d, %s>", pass, final ? "true" : "false");
    snprintf(it.args_type, sizeof it.args_type, "mlops_vocab_detail::SampleOp");
    it.args_bytes = (int)sizeof(SampleOp);
    memcpy(it.args.bytes, &plan->op, sizeof(SampleOp));
    it.geom[0] = pass; it.geom[1] = c->vocab; it.geom[2] = stats != nullptr; it.geom[3] = stats ? stats->count : 0;
    it.geom[4] = c->temperature > 0.f; it.geom[5] = c->penalty != 1.f; it.geom[6] = final;
  }
  const int total = plan->passes;
  delete plan;
  return written == total ? written : -1;
}

int mlops_vocab_detail::sample_access(const mlops_vocab_items &it, int item, mlops_vocab_item_range *out, int max) {
  const int pass = (int)it.geom[0], vocab = (int)it.geom[1];
  const bool pre = it.geom[2], scaled = it.geom[4], penalized = it.geom[5], final = it.geom[6];
  const Layout L(vocab);
  const long long chunks = (vocab + kChunk - 1) / kChunk, lo = (long long)item * kChunk, n = std::min<long long>(kChunk, vocab - lo);
  const int W = MLOPS_VOCAB_ARG_WORKSPACE, LG = MLOPS_VOCAB_ARG_LOGITS;
  int count = 0;
  auto sampled_read = [&] {  // the sampled logits of the chunk: the call's logits with statistics, else the workspace copy
    if (pre) put_range(out, max, count, LG, 0, lo * 4, n * 4);
    else put_range(out, max, count, W, 0, (long long)L.sampled + lo * 4, n * 4);
  };
  put_range(out, max, count, MLOPS_VOCAB_ARG_STATE, 0, 0, 16);
  if (pass != kPassPre && pass != kPassRadix0Pre) put_range(out, max, count, W, 2, L.counters + 4 * (pass == kPassStats ? kStats : pass == kPassMass ? kMass : pass == kPassCount ? kCount : pass == kPassWrite ? kWrite : pass == kPassDraw ? kDraw : kRadix0 + (pass - kPassRadix0)), 4);
  switch (pass) {
    case kPassPre:
      put_range(out, max, count, MLOPS_VOCAB_ARG_PRE_MAX, 0, 0, 4LL * it.geom[3]);
      put_range(out, max, count, MLOPS_VOCAB_ARG_PRE_ARG, 0, 0, 4LL * it.geom[3]);
      put_range(out, max, count, W, 1, L.max_logit, 4);
      put_range(out, max, count, W, 1, L.ctl, 16);
      put_range(out, max, count, MLOPS_VOCAB_ARG_TOKEN, 1, 0, 4);
      break;
    case kPassRadix0Pre:
      put_range(out, max, count, MLOPS_VOCAB_ARG_PRE_HIST, 0, 0, 1024LL * it.geom[3]);
      put_range(out, max, count, W, 0, L.ctl, 8);
      put_range(out, max, count, W, 1, L.ctl, 16);
      break;
    case kPassStats:
      put_range(out, max, count, LG, 0, lo * 4, n * 4);
      if (penalized) put_range(out, max, count, MLOPS_VOCAB_ARG_SEEN, 0, lo / 8, (n + 7) / 8);
      if (scaled && !pre) put_range(out, max, count, W, 1, (long long)L.sampled + lo * 4, n * 4);
      put_range(out, max, count, W, 1, L.chunk_m + item * 4LL, 4);
      put_range(out, max, count, W, 1, L.chunk_arg + item * 4LL, 4);
      put_range(out, max, count, W, 3, L.max_logit, 4);
      put_range(out, max, count, W, 3, L.ctl, 16);
      put_range(out, max, count, MLOPS_VOCAB_ARG_TOKEN, 3, 0, 4);
      break;
    case kPassMass:
      sampled_read();
      put_range(out, max, count, W, 0, L.max_logit, 4);
      put_range(out, max, count, W, 1, L.chunk_mass + item * 8LL, 8);
      put_range(out, max, count, W, 3, L.ctl, 24);
      break;
    case kPassCount:
      sampled_read();
      put_range(out, max, count, W, 0, L.ctl, 8);
      put_range(out, max, count, W, 1, L.cand_n + item * 4LL, 4);
      put_range(out, max, count, W, 3, L.cand_n, 4 * chunks);
      put_range(out, max, count, W, 3, L.ctl + 24, 8);
      break;
    case kPassWrite:
      sampled_read();
      put_range(out, max, count, W, 0, L.ctl, 8);
      put_range(out, max, count, W, 0, L.cand_n + item * 4LL, 4);
      put_range(out, max, count, W, 0, L.max_logit, 4);
      put_range(out, max, count, W, 1, L.cand, 8LL * kGather);
      put_range(out, max, count, MLOPS_VOCAB_ARG_TOKEN, 3, 0, 4);
      break;
    case kPassDraw:
      sampled_read();
      put_range(out, max, count, W, 0, L.ctl, 8);
      put_range(out, max, count, W, 0, L.max_logit, 4);
      put_range(out, max, count, W, 1, L.chunk_mass + item * 8LL, 8);
      put_range(out, max, count, W, 3, L.ctl + 24, 8);
      put_range(out, max, count, MLOPS_VOCAB_ARG_TOKEN, 3, 0, 4);
      break;
    default:  // radix r
      sampled_read();
      put_range(out, max, count, W, 0, L.ctl, 16);
      put_range(out, max, count, W, 0, L.max_logit, 4);
      put_range(out, max, count, W, 2, L.hist, 8LL * kBuckets);  // added to by every chunk, cleared by the closing item
      put_range(out, max, count, W, 3, L.hist, 8LL * kBuckets);
      put_range(out, max, count, W, 3, L.ctl, 16);
      break;
  }
  if (final) put_range(out, max, count, MLOPS_VOCAB_ARG_STATE, 3, 8, 8);
  return count;
}
