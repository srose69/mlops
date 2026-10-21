// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The mlops_rope kernel: one thread block per token. The block first puts its token's cos / sin factors of every
// pair in shared memory (computed from the position, or loaded from the tables), then rotates every head of q
// and k with them, V pairs per thread through one vector load / store per half. Template parameters: dtype,
// V (1, 2, 4), pair layout, arithmetic (exact, or Hugging Face's with tables); rope_inst.cu instantiates them.
#pragma once
#include "rope.cuh"

namespace mlops_rope_detail {

constexpr int kThreads = 128;

struct Args {
  const void *q_in, *k_in;
  void *q_out, *k_out;
  int64_t q_sb, q_ss, q_sh, qo_sb, qo_ss, qo_sh, k_sb, k_ss, k_sh, ko_sb, ko_ss, ko_sh;
  int seq, head_dim, pairs, hq, hk;  // pairs = rotary_dim / 2
  bool copy_q, copy_k;               // out of place with rotary_dim < head_dim: copy the rest
  const void *positions;
  bool pos64;
  int64_t pos_sb, pos_ss, offset;
  const Freq *freq;
  float att;
  const void *cos, *sin;
  int64_t t_sb, t_ss;
  bool inverse;
};

template <typename S, int N> struct alignas(N * sizeof(S)) Pack { S v[N]; };

// the factors of pair i of token (b, s) into cs: from the position (exact) or from the tables (HF)
template <int DT, bool INTER, bool HF>
__device__ __forceinline__ float4 pair_factors(const Args &a, int b, int s, int i) {
  using E = Elem<DT>;
  using S = typename E::S;
  if (HF) {
    const int64_t o = b * a.t_sb + s * a.t_ss;
    const int i1 = INTER ? 2 * i : i, i2 = INTER ? 2 * i + 1 : i + a.pairs;
    const S *ct = static_cast<const S *>(a.cos) + o, *st = static_cast<const S *>(a.sin) + o;
    const float c1 = E::load(ct[i1]), c2 = E::load(ct[i2]), s1 = E::load(st[i1]), s2 = E::load(st[i2]);
    // backward of y1 = x1 c1 - x2 s1, y2 = x2 c2 + x1 s2: g1' = g1 c1 + g2 s2, g2' = g2 c2 - g1 s1
    return a.inverse ? make_float4(c1, -s2, c2, -s1) : make_float4(c1, s1, c2, s2);
  } else {
    int32_t m;
    if (a.positions) {
      const int64_t o = b * a.pos_sb + s * a.pos_ss;
      m = a.pos64 ? (int32_t) static_cast<const int64_t *>(a.positions)[o] : static_cast<const int32_t *>(a.positions)[o];
    } else {
      m = (int32_t)(s + a.offset);
    }
    float c, sn;
    angle_cos_sin(m, a.freq[i], a.att, c, sn);
    if (a.inverse) sn = -sn;
    return make_float4(c, sn, c, sn);
  }
}

// rotate heads [0, heads) of one token: in / out point at head 0, sh = head stride
template <int DT, int V, bool INTER, bool HF>
__device__ __forceinline__ void rotate_heads(const float4 *cs, const void *in, void *out, int64_t sh, int64_t osh,
                                             int heads, int pairs) {
  using E = Elem<DT>;
  using S = typename E::S;
  const int nv = pairs / V, total = heads * nv;
  if (threadIdx.x >= total) return;
  // (head, vector) of this thread, advanced by kThreads items without a division per item
  int h = threadIdx.x / nv, j = threadIdx.x - h * nv;
  const int dh = kThreads / nv, dj = kThreads - dh * nv;
  for (int it = threadIdx.x; it < total; it += kThreads) {
    const S *x = static_cast<const S *>(in) + h * sh;
    S *y = static_cast<S *>(out) + h * osh;
    const int p0 = j * V;
    float x1[V], x2[V], y1[V], y2[V];
    if (INTER) {
      const Pack<S, 2 * V> v = *reinterpret_cast<const Pack<S, 2 * V> *>(x + 2 * p0);
#pragma unroll
      for (int k = 0; k < V; ++k) { x1[k] = E::load(v.v[2 * k]); x2[k] = E::load(v.v[2 * k + 1]); }
    } else {
      const Pack<S, V> v1 = *reinterpret_cast<const Pack<S, V> *>(x + p0);
      const Pack<S, V> v2 = *reinterpret_cast<const Pack<S, V> *>(x + p0 + pairs);
#pragma unroll
      for (int k = 0; k < V; ++k) { x1[k] = E::load(v1.v[k]); x2[k] = E::load(v2.v[k]); }
    }
#pragma unroll
    for (int k = 0; k < V; ++k) {
      if (HF) rotate_hf<DT>(x1[k], x2[k], cs[p0 + k], y1[k], y2[k]);
      else rotate_exact(x1[k], x2[k], cs[p0 + k], y1[k], y2[k]);
    }
    if (INTER) {
      Pack<S, 2 * V> v;
#pragma unroll
      for (int k = 0; k < V; ++k) { v.v[2 * k] = E::store(y1[k]); v.v[2 * k + 1] = E::store(y2[k]); }
      *reinterpret_cast<Pack<S, 2 * V> *>(y + 2 * p0) = v;
    } else {
      Pack<S, V> v1, v2;
#pragma unroll
      for (int k = 0; k < V; ++k) { v1.v[k] = E::store(y1[k]); v2.v[k] = E::store(y2[k]); }
      *reinterpret_cast<Pack<S, V> *>(y + p0) = v1;
      *reinterpret_cast<Pack<S, V> *>(y + p0 + pairs) = v2;
    }
    h += dh; j += dj;
    if (j >= nv) { j -= nv; ++h; }
  }
}

// elements [2 * pairs, head_dim) of every head, out of place: copied unchanged
template <int DT>
__device__ __forceinline__ void copy_rest(const void *in, void *out, int64_t sh, int64_t osh, int heads, int from, int dim) {
  using S = typename Elem<DT>::S;
  const int n = dim - from;
  for (int it = threadIdx.x; it < heads * n; it += kThreads) {
    const int h = it / n, e = from + it - h * n;
    static_cast<S *>(out)[h * osh + e] = static_cast<const S *>(in)[h * sh + e];
  }
}

template <int DT, int V, bool INTER, bool HF>
__global__ __launch_bounds__(kThreads) void rope_kernel(Args a) {
  extern __shared__ float4 cs[];
  const int s = blockIdx.x % a.seq, b = blockIdx.x / a.seq;
  for (int i = threadIdx.x; i < a.pairs; i += kThreads) cs[i] = pair_factors<DT, INTER, HF>(a, b, s, i);
  __syncthreads();
  using S = typename Elem<DT>::S;
  const S *q = static_cast<const S *>(a.q_in) + b * a.q_sb + s * a.q_ss;
  S *qo = static_cast<S *>(a.q_out) + b * a.qo_sb + s * a.qo_ss;
  rotate_heads<DT, V, INTER, HF>(cs, q, qo, a.q_sh, a.qo_sh, a.hq, a.pairs);
  if (a.copy_q) copy_rest<DT>(q, qo, a.q_sh, a.qo_sh, a.hq, 2 * a.pairs, a.head_dim);
  if (a.k_in) {
    const S *k = static_cast<const S *>(a.k_in) + b * a.k_sb + s * a.k_ss;
    S *ko = static_cast<S *>(a.k_out) + b * a.ko_sb + s * a.ko_ss;
    rotate_heads<DT, V, INTER, HF>(cs, k, ko, a.k_sh, a.ko_sh, a.hk, a.pairs);
    if (a.copy_k) copy_rest<DT>(k, ko, a.k_sh, a.ko_sh, a.hk, 2 * a.pairs, a.head_dim);
  }
}

// rope_kernel<DT, v, inter, hf> over `tokens` blocks (rope_inst.cu: one object per dtype)
template <int DT>
void launch(const Args &a, int tokens, int v, bool inter, bool hf, cudaStream_t stream);

}  // namespace mlops_rope_detail
