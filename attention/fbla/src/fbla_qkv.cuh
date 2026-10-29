// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The new token's attention inputs, one pass per KV head (mlops_fbla_qkv_at): q and k optionally RMS-normalized,
// then RoPE-rotated (rope.cuh); q is written for decode, k and v appended to the cache (close_group first).
#ifndef MLOPS_FBLA_QKV_CUH
#define MLOPS_FBLA_QKV_CUH

#include "fbla_prefill.cuh"
#include "../../../positional/rope/src/rope.cuh"

namespace mlops_fbla_detail {

struct QkvArgs {
  PrefillArgs p;  // the cache, the device length, n0 (the offset), count 1, hs / gs, batch and heads
  const uint16_t *q, *k, *v;
  int64_t batch_stride, q_group_stride, kv_group_stride;
  const uint16_t *q_norm, *k_norm;
  float eps, att;
  const mlops_rope_detail::Freq *freq;  // 64 pairs: the whole head, half split
  uint16_t *q_out;                      // [batch, q_heads, 128]
  FastDiv kv;                           // / kv_heads
};

struct QkvSmem {
  uint8_t code[2][G][D];  // close_group's
  float4 cs[D / 2];       // the token's (cos, sin, cos, sin) of every pair
};

// One head of 128: lane l holds elements 2 l, 2 l + 1, 64 + 2 l, 65 + 2 l.  RMSNorm (NORM): s = sum of squares
// (xor butterfly 16 .. 1), r = 1 / sqrt(s / 128 + eps), y = bf16(bf16(x r) w), then the rotation rounded once
// to BF16.  The four results go to y[].
template <bool NORM, class Pos = Grid>
__device__ __forceinline__ void norm_rope(const uint16_t *x, const uint16_t *w, float eps, const float4 *cs, int lane, uint16_t y[4]) {
  const uint32_t lo = InputLoad<Pos>::get(reinterpret_cast<const uint32_t *>(x) + lane),
                 hi = InputLoad<Pos>::get(reinterpret_cast<const uint32_t *>(x + D / 2) + lane);
  float v[4] = {bf16((uint16_t)lo), bf16((uint16_t)(lo >> 16)), bf16((uint16_t)hi), bf16((uint16_t)(hi >> 16))};
  if (NORM) {
    float s = __fmul_rn(v[0], v[0]);
    s = __fmaf_rn(v[1], v[1], s);
    s = __fmaf_rn(v[2], v[2], s);
    s = __fmaf_rn(v[3], v[3], s);
#pragma unroll
    for (int o = 16; o; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
    const float r = __frsqrt_rn(__fadd_rn(__fmul_rn(s, 1.f / D), eps));
    const uint32_t wl = reinterpret_cast<const uint32_t *>(w)[lane], wh = reinterpret_cast<const uint32_t *>(w + D / 2)[lane];
    const float wv[4] = {bf16((uint16_t)wl), bf16((uint16_t)(wl >> 16)), bf16((uint16_t)wh), bf16((uint16_t)(wh >> 16))};
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = bf16(to_bf16(__fmul_rn(bf16(to_bf16(__fmul_rn(v[i], r))), wv[i])));
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    float y1, y2;
    mlops_rope_detail::rotate_exact(v[i], v[2 + i], cs[2 * lane + i], y1, y2);
    y[i] = to_bf16(y1);
    y[2 + i] = to_bf16(y2);
  }
}

// KV head bh (batch row bh / kv_heads) of the token at position *length + n0.  Tasks: the q heads of the KV head, k,
// v -- task t on warp t mod (threads / 32).
template <int CB, int FB, bool NORM, int T, class Pos = Grid>
__device__ __forceinline__ void qkv_item(const QkvArgs &a, int bh, QkvSmem &sm) {
  const int pos = n0_of<Pos>(a.p), lane = Pos::tid() & 31, warp = Pos::tid() >> 5;
  const int b = a.kv.div(bh), kh = bh - b * a.p.kv_heads, qg = a.p.qg;
  for (int i = Pos::tid(); i < D / 2; i += T) {
    float c, s;
    mlops_rope_detail::angle_cos_sin(pos, a.freq[i], a.att, c, s);
    sm.cs[i] = make_float4(c, s, c, s);
  }
  if (pos > 0 && pos % G == 0) close_group<CB, FB, T, Pos>(a.p, pos, pos / G - 1, bh, sm.code);  // (CTA-uniform)
  __syncthreads();  // the factors; the closed group's rows read before the window is rewritten

  const size_t eh = (size_t)bh * G * D;
  const int row = pos % G;  // in the open window (the sink too below 128)
  for (int t = warp; t < qg + 2; t += T / 32) {
    if (t < qg) {
      const uint16_t *x = a.q + b * a.batch_stride + kh * a.q_group_stride + (size_t)t * D;
      uint16_t y[4];
      norm_rope<NORM, Pos>(x, a.q_norm, a.eps, sm.cs, lane, y);
      uint32_t *o = reinterpret_cast<uint32_t *>(a.q_out + ((size_t)b * a.p.q_heads + kh * qg + t) * D);
      o[lane] = y[0] | (uint32_t)y[1] << 16;
      o[D / 4 + lane] = y[2] | (uint32_t)y[3] << 16;
    } else {
      const bool isk = t == qg;
      uint32_t lo, hi;
      if (isk) {
        uint16_t y[4];
        norm_rope<NORM, Pos>(a.k + b * a.batch_stride + kh * a.kv_group_stride, a.k_norm, a.eps, sm.cs, lane, y);
        lo = y[0] | (uint32_t)y[1] << 16;
        hi = y[2] | (uint32_t)y[3] << 16;
      } else {
        const uint32_t *x = reinterpret_cast<const uint32_t *>(a.v + b * a.batch_stride + kh * a.kv_group_stride);
        lo = InputLoad<Pos>::get(x + lane);
        hi = InputLoad<Pos>::get(x + D / 4 + lane);
      }
      uint32_t *open = reinterpret_cast<uint32_t *>(const_cast<uint16_t *>(isk ? a.p.c.exact_open_k : a.p.c.exact_open_v) + eh +
                                                   (size_t)row * D);
      open[lane] = lo;
      open[D / 4 + lane] = hi;
      if (pos < G) {
        uint32_t *sink = reinterpret_cast<uint32_t *>(const_cast<uint16_t *>(isk ? a.p.c.exact_sink_k : a.p.c.exact_sink_v) + eh +
                                                     (size_t)row * D);
        sink[lane] = lo;
        sink[D / 4 + lane] = hi;
      }
    }
  }
  if (row == 0)  // a new group opened: its other rows cleared
    for (int e = Pos::tid(); e < 2 * (G - 1) * D / 8; e += T) {
      const int x = e >= (G - 1) * D / 8, r = e - x * (G - 1) * D / 8;
      reinterpret_cast<uint4 *>(const_cast<uint16_t *>(x ? a.p.c.exact_open_v : a.p.c.exact_open_k) + eh + D)[r] = make_uint4(0u, 0u, 0u, 0u);
    }
}

template <int CB, int FB, bool NORM>
__global__ void __launch_bounds__(2 * D) qkv_kernel(const QkvArgs a) {
  __shared__ QkvSmem sm;
  qkv_item<CB, FB, NORM, 2 * D>(a, blockIdx.x, sm);
}

#ifndef __CUDACC_RTC__
template <int CB, int FB> cudaError_t launch_qkv(const QkvArgs &a, cudaStream_t stream) {
  const int heads = a.p.batch * a.p.kv_heads;
  if (a.q_norm) qkv_kernel<CB, FB, true><<<heads, 2 * D, 0, stream>>>(a);
  else qkv_kernel<CB, FB, false><<<heads, 2 * D, 0, stream>>>(a);
  return cudaGetLastError();
}

// (host) mlops_fbla_qkv_at as the executor's items (mlops_fbla_items.h): one item a (batch, KV head)
template <int CB, int FB>
inline int qkv_passes(const QkvArgs &a, mlops_fbla_items *p, int max) {
  if (max < 1) return -1;
  char e[96];
  snprintf(e, sizeof e, "mlops_fbla_detail::qkv_item<%d, %d, %s, %d, mlops_fbla::Pos>", CB, FB, a.q_norm ? "true" : "false",
           2 * D);
  const bool ok = item_of(a.q_norm ? qkv_kernel<CB, FB, true> : qkv_kernel<CB, FB, false>, MLOPS_FBLA_ITEM_QKV,
                          a.p.batch * a.p.kv_heads, 1, 1, 2 * D, (int)sizeof(QkvSmem), e, a, p[0]);
  return ok ? 1 : -1;
}
#endif  // !__CUDACC_RTC__

}  // namespace mlops_fbla_detail
#endif
