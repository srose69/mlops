// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_rope host side (include/mlops_rope.h): argument checks, the vector width, the frequency table. The
// kernel instances are in rope_inst.cu (rope_kernels.cuh).
#include "rope_kernels.cuh"
#include <cmath>
#include <cstdarg>
#include <cstdio>

using namespace mlops_rope_detail;

namespace {

char *last_error_buf() {
  static thread_local char buf[256] = {0};
  return buf;
}
mlops_rope_status fail(mlops_rope_status st, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(last_error_buf(), 255, fmt, ap);
  va_end(ap);
  return st;
}

bool aligned(const void *p, int64_t bytes) { return reinterpret_cast<uintptr_t>(p) % bytes == 0; }

// the widest V (pairs per vector access) every tensor allows: pairs % V == 0, pointers and the b / s / h strides
// aligned to the vector (V elements per half; 2V interleaved)
bool fits(const mlops_rope_tensor *t, int64_t vec_bytes, int64_t es) {
  return !t || (aligned(t->data, vec_bytes) && (t->stride_b * es) % vec_bytes == 0 && (t->stride_s * es) % vec_bytes == 0 &&
                (t->stride_h * es) % vec_bytes == 0);
}

}  // namespace

extern "C" mlops_rope_status mlops_rope(const mlops_rope_config *c, const mlops_rope_tensor *q_in,
                                        const mlops_rope_tensor *q_out, const mlops_rope_tensor *k_in,
                                        const mlops_rope_tensor *k_out, cudaStream_t stream) {
  if (!c || !q_in || !q_out || !q_in->data || !q_out->data)
    return fail(MLOPS_ROPE_INVALID_VALUE, "mlops_rope: cfg, q_in and q_out (with data) are required");
  if ((k_in == nullptr) != (k_out == nullptr) || (k_in && (!k_in->data || !k_out->data)))
    return fail(MLOPS_ROPE_INVALID_VALUE, "mlops_rope: k_in and k_out must both be given (with data), or neither");
  if (c->dtype != MLOPS_ROPE_BF16 && c->dtype != MLOPS_ROPE_FP16 && c->dtype != MLOPS_ROPE_FP32)
    return fail(MLOPS_ROPE_INVALID_VALUE, "mlops_rope: invalid dtype");
  if (c->batch < 0 || c->seq < 0 || c->head_dim <= 0)
    return fail(MLOPS_ROPE_INVALID_VALUE, "mlops_rope: batch, seq >= 0 and head_dim > 0 required");
  if (c->rotary_dim < 2 || c->rotary_dim % 2 || c->rotary_dim > c->head_dim || c->rotary_dim > 1024)
    return fail(MLOPS_ROPE_INVALID_VALUE, "mlops_rope: rotary_dim (%d) must be even, in [2, min(head_dim = %d, 1024)]",
                c->rotary_dim, c->head_dim);
  if (q_in->heads <= 0 || q_out->heads != q_in->heads || (k_in && (k_in->heads <= 0 || k_out->heads != k_in->heads)))
    return fail(MLOPS_ROPE_INVALID_VALUE, "mlops_rope: heads must be > 0 and equal for in and out");
  const bool tables = c->cos || c->sin;
  if (tables && (!c->cos || !c->sin || c->positions || c->freq))
    return fail(MLOPS_ROPE_INVALID_VALUE, "mlops_rope: give either cos and sin tables, or freq (and optionally positions)");
  if (!tables && !c->freq)
    return fail(MLOPS_ROPE_INVALID_VALUE, "mlops_rope: freq (mlops_rope_freq) or cos / sin tables are required");
  const int64_t tokens = (int64_t)c->batch * c->seq;
  if (tokens > 0x7FFFFFFF) return fail(MLOPS_ROPE_INVALID_VALUE, "mlops_rope: batch * seq must be < 2^31");
  if (tokens == 0) { last_error_buf()[0] = '\0'; return MLOPS_ROPE_OK; }

  const int64_t es = c->dtype == MLOPS_ROPE_FP32 ? 4 : 2;
  const int pairs = c->rotary_dim / 2;
  int v = 4;
  while (v > 1) {
    const int64_t vb = (c->interleaved ? 2 : 1) * v * es;
    if (pairs % v == 0 && fits(q_in, vb, es) && fits(q_out, vb, es) && fits(k_in, vb, es) && fits(k_out, vb, es)) break;
    v /= 2;
  }
  Args a{};
  a.q_in = q_in->data; a.q_out = q_out->data;
  a.q_sb = q_in->stride_b; a.q_ss = q_in->stride_s; a.q_sh = q_in->stride_h;
  a.qo_sb = q_out->stride_b; a.qo_ss = q_out->stride_s; a.qo_sh = q_out->stride_h;
  a.hq = q_in->heads;
  const auto moved = [](const mlops_rope_tensor *i, const mlops_rope_tensor *o) {
    return i->data != o->data || i->stride_b != o->stride_b || i->stride_s != o->stride_s || i->stride_h != o->stride_h;
  };
  a.copy_q = c->rotary_dim < c->head_dim && moved(q_in, q_out);
  if (k_in) {
    a.k_in = k_in->data; a.k_out = k_out->data;
    a.k_sb = k_in->stride_b; a.k_ss = k_in->stride_s; a.k_sh = k_in->stride_h;
    a.ko_sb = k_out->stride_b; a.ko_ss = k_out->stride_s; a.ko_sh = k_out->stride_h;
    a.hk = k_in->heads;
    a.copy_k = c->rotary_dim < c->head_dim && moved(k_in, k_out);
  }
  a.seq = c->seq; a.head_dim = c->head_dim; a.pairs = pairs;
  a.positions = c->positions; a.pos64 = c->positions_int64 != 0;
  a.pos_sb = c->pos_stride_b; a.pos_ss = c->pos_stride_s; a.offset = c->position_offset;
  a.freq = reinterpret_cast<const Freq *>(c->freq);
  a.att = c->attention_factor == 0.0f ? 1.0f : c->attention_factor;
  a.cos = c->cos; a.sin = c->sin; a.t_sb = c->table_stride_b; a.t_ss = c->table_stride_s;
  a.inverse = c->inverse != 0;
  switch (c->dtype) {
    case MLOPS_ROPE_BF16: launch<MLOPS_ROPE_BF16>(a, (int)tokens, v, c->interleaved != 0, tables, stream); break;
    case MLOPS_ROPE_FP16: launch<MLOPS_ROPE_FP16>(a, (int)tokens, v, c->interleaved != 0, tables, stream); break;
    default: launch<MLOPS_ROPE_FP32>(a, (int)tokens, v, c->interleaved != 0, tables, stream); break;
  }
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return fail(MLOPS_ROPE_CUDA_ERROR, "mlops_rope: %s", cudaGetErrorString(e));
  last_error_buf()[0] = '\0';
  return MLOPS_ROPE_OK;
}

extern "C" void mlops_rope_freq(const double *inv_freq, int n, float *out) {
  const double two_pi = 6.283185307179586476925286766559;
  for (int i = 0; i < n; ++i) {
    const double f = inv_freq[i] / two_pi, g = std::fmod(65536.0 * f, 1.0);  // 65536 f: exact in double
    const float fh = (float)f, gh = (float)g;
    out[4 * i] = fh; out[4 * i + 1] = (float)(f - fh);
    out[4 * i + 2] = gh; out[4 * i + 3] = (float)(g - gh);
  }
}

extern "C" const char *mlops_rope_status_string(mlops_rope_status status) {
  switch (status) {
    case MLOPS_ROPE_OK: return "ok";
    case MLOPS_ROPE_INVALID_VALUE: return "invalid value";
    case MLOPS_ROPE_CUDA_ERROR: return "cuda error";
    default: return "unknown";
  }
}

extern "C" const char *mlops_rope_last_error(void) { return last_error_buf(); }
