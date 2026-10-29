// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The LM head element on CUDA cores (include/mlops_lm_head.h). A CTA of S warps takes a range of rows; a phase is
// PR = 32 S / NB rows in units of R, every warp summing its k-range of 256-column passes for NB columns of x, then
// the S partial sums of an (row, column) are added in warp order and scaled. The weights are read once (streaming
// loads); x is written by the producer in the same launch and is read through L2.
#pragma once
#include "mlops_lm_head.h"

namespace mlops_vocab_detail {

constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float bf16_to_f32(uint16_t v) { return __uint_as_float((unsigned)v << 16); }

struct HeadOp {
  const uint16_t *w;          // [vocab][hidden]
  const uint16_t *x;          // [n][hidden]
  const float *norm_parts;    // [n][norm_parts_n], or null
  float *logits;              // [n][vocab], or null
  float *stats_max;           // [n][items], or null
  int *stats_arg;             // [n][items]
  unsigned *hist;             // [n][items][256], or null
  int vocab, hidden, items, n, norm_parts_n;
  float inv_n, eps, logit_scale;
};

// The top byte of a float's order key (a larger float, a larger key): the bucket of the sampler's first radix pass.
__device__ __forceinline__ unsigned key_byte(float x) {
  const unsigned u = __float_as_uint(x);
  return (u & 0x80000000u ? ~u : u | 0x80000000u) >> 24;
}

template <int S, int NB> struct HeadSmem {
  float part[S][NB][32 * S / NB];  // the warps' partial sums of the phase's (column, row)
  unsigned hist[NB][256];          // counts of the key bytes of the element's rows
  float max[S][NB];
  int arg[S][NB];
  float scale[NB];
};

// 1 / sqrt(mean of a column's parts + eps): the lanes' strided sum, a butterfly, one correctly rounded rsqrt.
__device__ __forceinline__ float norm_scale(const float *parts, int count, float inv_n, float eps, int lane) {
  float s = 0.f;
  for (int i = lane; i < count; i += 32) s = __fadd_rn(s, __ldcg(parts + i));
#pragma unroll
  for (int o = 16; o; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(kFullMask, s, o));
  return __frsqrt_rn(__fadd_rn(__fmul_rn(s, inv_n), eps));
}

// One phase: the rows [r0, r0 + PR) in units of R rows over the warp's k-passes [p0, p1). The loads of a step (R
// rows, and NB columns of x, one 256-column pass) are issued D steps ahead in a register ring, across unit
// boundaries. Lane r < R ends a unit holding (row r, column c)'s sum (FMAs in (pass, column) order per lane, then
// the xor butterfly). Rows past the vocabulary read row 0 and columns past n column n - 1; neither is stored.
template <int R, int S, int D, int NB>
__device__ __forceinline__ void head_phase(const HeadOp &op, int r0, int hi, int p0, int p1, int warp, int lane, HeadSmem<S, NB> &sm) {
  static_assert(R >= 1 && D >= 1 && NB >= 1, "rows per unit, ring depth, columns");
  constexpr int PR = 32 * S / NB;
  static_assert(PR % R == 0, "units tile the phase");
  const int np = p1 - p0, units = min(PR / R, (hi - r0 + R - 1) / R), total = units * np;
  uint4 wr[D][R], xr[D][NB];
  int iu = 0, ip = 0;  // the next step to issue: unit, pass
  auto issue = [&](int d) {
    const int row0 = r0 + iu * R, p = p0 + ip;
#pragma unroll
    for (int c = 0; c < NB; ++c) {
      const int col = min(c, op.n - 1);
      xr[d][c] = __ldcg(reinterpret_cast<const uint4 *>(op.x + (size_t)col * op.hidden + 256 * p + 8 * lane));
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const size_t row = row0 + r < op.vocab ? row0 + r : 0;
      wr[d][r] = __ldcs(reinterpret_cast<const uint4 *>(op.w + row * op.hidden + 256 * p + 8 * lane));
    }
    if (++ip == np) { ip = 0; ++iu; }
  };
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < total) issue(d);

  float acc[R][NB];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int c = 0; c < NB; ++c) acc[r][c] = 0.f;
  int u = 0, pi = 0;
  for (int t = 0; t < total; t += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      if (t + d >= total) break;  // warp-uniform
      float xf[NB][8];
#pragma unroll
      for (int c = 0; c < NB; ++c) {
        const unsigned xs[4] = {xr[d][c].x, xr[d][c].y, xr[d][c].z, xr[d][c].w};
#pragma unroll
        for (int e = 0; e < 8; ++e) xf[c][e] = bf16_to_f32((uint16_t)(xs[e / 2] >> (16 * (e & 1))));
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const unsigned ws[4] = {wr[d][r].x, wr[d][r].y, wr[d][r].z, wr[d][r].w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float wf = bf16_to_f32((uint16_t)(ws[e / 2] >> (16 * (e & 1))));
#pragma unroll
          for (int c = 0; c < NB; ++c) acc[r][c] = __fmaf_rn(wf, xf[c][e], acc[r][c]);
        }
      }
      if (t + d + D < total) issue(d);
      if (++pi == np) {
        pi = 0;
#pragma unroll
        for (int c = 0; c < NB; ++c) {
          float mine = 0.f;
#pragma unroll
          for (int r = 0; r < R; ++r) {
            float v = acc[r][c];
#pragma unroll
            for (int o = 16; o; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(kFullMask, v, o));
            if (lane == r) mine = v;
            acc[r][c] = 0.f;
          }
          if (lane < R) sm.part[warp][c][u * R + lane] = mine;
        }
        ++u;
      }
    }
  }
}

// Element `item` of op.items: the rows of units [item U / items, (item + 1) U / items), U = ceil(vocab / R).
// All 32 S threads call it; the CTA is synchronized on return.
template <int R, int S, int D, int NB>
__device__ __forceinline__ void head_item(const HeadOp &op, int item, unsigned tid, HeadSmem<S, NB> &sm) {
  constexpr int PR = 32 * S / NB;  // rows of a phase; PR * NB = the CTA's threads: one (column, row) each
  const int warp = tid >> 5, lane = tid & 31;
  const long long units = (op.vocab + R - 1) / R;
  const int lo = (int)(item * units / op.items) * R, hi = min((int)(((item + 1) * units / op.items) * R), op.vocab);
  const int passes = op.hidden / 256, p0 = warp * passes / S, p1 = (warp + 1) * passes / S;

  if (op.hist)
    for (unsigned i = tid; i < NB * 256; i += 32 * S) (&sm.hist[0][0])[i] = 0u;
  if (warp == 0) {
#pragma unroll
    for (int c = 0; c < NB; ++c) {
      const int col = min(c, op.n - 1);
      const float s = op.norm_parts ? norm_scale(op.norm_parts + (size_t)col * op.norm_parts_n, op.norm_parts_n, op.inv_n, op.eps, lane) : 1.f;
      if (!lane) sm.scale[c] = s;
    }
  }
  __syncthreads();

  const int c_me = tid / PR, row_me = tid % PR;  // this thread's (column, row in phase) in the combine
  float best = __int_as_float(0xff800000);       // -inf
  int arg = 0x7fffffff;
  for (int r0 = lo; r0 < hi; r0 += PR) {
    head_phase<R, S, D, NB>(op, r0, hi, p0, p1, warp, lane, sm);
    __syncthreads();
    const int row = r0 + row_me;
    if (row < hi && c_me < op.n) {
      float sum = sm.part[0][c_me][row_me];
#pragma unroll
      for (int w = 1; w < S; ++w) sum = __fadd_rn(sum, sm.part[w][c_me][row_me]);
      const float l = __fmul_rn(__fmul_rn(sum, sm.scale[c_me]), op.logit_scale);
      if (op.logits) __stcg(op.logits + (size_t)c_me * op.vocab + row, l);  // read by other SMs: L2 only, not L1
      if (op.hist) atomicAdd(&sm.hist[c_me][key_byte(l)], 1u);
      if (l > best) { best = l; arg = row; }  // ascending rows: the first of equals
    }
    __syncthreads();
  }
  if (op.hist)  // after the last phase's barrier: this element's counts, every column
    for (unsigned i = tid; i < NB * 256; i += 32 * S)
      if (i / 256 < (unsigned)op.n) __stcg(op.hist + ((size_t)(i / 256) * op.items + item) * 256 + i % 256, (&sm.hist[0][0])[i]);
  if (op.stats_max) {
    // The warps' lanes may hold several columns: reduce each column over the lanes that carry it.
#pragma unroll
    for (int c = 0; c < NB; ++c) {
      float b = c_me == c ? best : __int_as_float(0xff800000);
      int a = c_me == c ? arg : 0x7fffffff;
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        const float b2 = __shfl_xor_sync(kFullMask, b, o);
        const int a2 = __shfl_xor_sync(kFullMask, a, o);
        if (b2 > b || (b2 == b && a2 < a)) { b = b2; a = a2; }
      }
      if (!lane) { sm.max[warp][c] = b; sm.arg[warp][c] = a; }
    }
    __syncthreads();
    if (tid < NB && tid < op.n) {
      float b = sm.max[0][tid];
      int a = sm.arg[0][tid];
      for (int w = 1; w < S; ++w)
        if (sm.max[w][tid] > b || (sm.max[w][tid] == b && sm.arg[w][tid] < a)) { b = sm.max[w][tid]; a = sm.arg[w][tid]; }
      __stcg(op.stats_max + (size_t)tid * op.items + item, b);
      __stcg(op.stats_arg + (size_t)tid * op.items + item, a);
    }
    __syncthreads();
  }
}

}  // namespace mlops_vocab_detail
