// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The LM head element on tensor cores, n <= 8 columns (include/mlops_lm_head.h). W is the A operand of
// mma.m16n8k16 (BF16 in, FP32 out, sm_80+) or, converted to FP16 in registers, of mma.m16n8k8 (sm_75+) or of
// mma.m8n8k4 (sm_70+); x the B operand, the columns the n dimension. KIND: 0, 1, 2 in that order. A warp takes a unit of 16 rows
// over its k-range; thread (g, c) = (lane / 4, lane % 4) loads rows g and g + 8, J 16-byte pieces each of one
// contiguous 8 J-element chunk of a 32 J-element block, and the matching pieces of x for column g. The sum over
// k does not depend on how k is numbered inside a mma, so the chunk order only has to be the same for W and x:
// every load is a whole 16-byte piece and a row's 4 lanes read 64 J contiguous bytes.
#pragma once
#include "lm_head.cuh"

namespace mlops_vocab_detail {

constexpr int kMmaPhaseUnits = 4;  // units of 16 rows per phase

template <int S> struct MmaSmem {
  float part[S][8][16 * kMmaPhaseUnits];
  unsigned hist[8][256];
  float max[S][8];
  int arg[S][8];
  float scale[8];
};

// Two BF16 in one word to two FP16 (round to nearest even): the mantissa is exact, the range is FP16's.
__device__ __forceinline__ uint32_t bf16x2_to_f16x2(uint32_t v) {
  uint32_t r = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  asm("{\n .reg .f16 l, h;\n cvt.rn.f16.f32 l, %1;\n cvt.rn.f16.f32 h, %2;\n mov.b32 %0, {l, h};\n}"
      : "=r"(r) : "f"(__uint_as_float(v << 16)), "f"(__uint_as_float(v & 0xffff0000u)));
#endif
  return r;
}

__device__ __forceinline__ void mma_f16_k8(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(b0));
#endif
}

// Volta's shape: a warp makes four independent 8 x 8 x 4 products, one per group of lanes {4q .. 4q + 3} and
// {16 + 4q .. 16 + 4q + 3}. A lane holds the 4 k of one row of A (and the 4 k of one column of B); of C, rows
// (lane & 1) + (i & 2) (+ 4 for the high lanes) at columns (i & 4) + (lane & 2) + (i & 1).
__device__ __forceinline__ void mma_f16_884(float (&c)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
               : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// One phase: the rows [r0, r0 + 16 kMmaPhaseUnits) over the warp's blocks [b0, b1) of 32 J elements, the loads of a
// step (rows g and g + 8, x of column g, J pieces each) issued D steps ahead in a register ring. KIND 1: the pieces
// are converted to FP16 and multiplied in four mma.m16n8k8 (a piece holds four pairs of k); KIND 0: two m16n8k16.
template <int S, int J, int D, int KIND>
__device__ __forceinline__ void head_phase_mma(const HeadOp &op, int r0, int hi, int b0, int b1, int warp, int lane, MmaSmem<S> &sm) {
  const int g = lane >> 2, c = lane & 3;
  const int nb = b1 - b0, units = min(kMmaPhaseUnits, (hi - r0 + 15) / 16), total = units * nb;
  uint4 wa[D][J], wb[D][J], xb[D][J];
  int iu = 0, ip = 0;
  const int col = min(g, op.n - 1);
  auto issue = [&](int d) {
    const int ra = r0 + iu * 16 + g;
    const size_t rowa = ra < op.vocab ? ra : 0, rowb = ra + 8 < op.vocab ? ra + 8 : 0;
    const size_t k = (size_t)(b0 + ip) * 32 * J + (size_t)c * 8 * J;
#pragma unroll
    for (int j = 0; j < J; ++j) {
      wa[d][j] = __ldcs(reinterpret_cast<const uint4 *>(op.w + rowa * op.hidden + k + 8 * j));
      wb[d][j] = __ldcs(reinterpret_cast<const uint4 *>(op.w + rowb * op.hidden + k + 8 * j));
      xb[d][j] = __ldcg(reinterpret_cast<const uint4 *>(op.x + (size_t)col * op.hidden + k + 8 * j));
    }
    if (++ip == nb) { ip = 0; ++iu; }
  };
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < total) issue(d);

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  int u = 0, pi = 0;
  for (int t = 0; t < total; t += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      if (t + d >= total) break;  // warp-uniform
#pragma unroll
      for (int j = 0; j < J; ++j) {
        if constexpr (KIND == 1) {
          mma_f16_k8(acc, bf16x2_to_f16x2(wa[d][j].x), bf16x2_to_f16x2(wb[d][j].x), bf16x2_to_f16x2(xb[d][j].x));
          mma_f16_k8(acc, bf16x2_to_f16x2(wa[d][j].y), bf16x2_to_f16x2(wb[d][j].y), bf16x2_to_f16x2(xb[d][j].y));
          mma_f16_k8(acc, bf16x2_to_f16x2(wa[d][j].z), bf16x2_to_f16x2(wb[d][j].z), bf16x2_to_f16x2(xb[d][j].z));
          mma_f16_k8(acc, bf16x2_to_f16x2(wa[d][j].w), bf16x2_to_f16x2(wb[d][j].w), bf16x2_to_f16x2(xb[d][j].w));
        } else {
          mma_bf16(acc, wa[d][j].x, wb[d][j].x, wa[d][j].y, wb[d][j].y, xb[d][j].x, xb[d][j].y);
          mma_bf16(acc, wa[d][j].z, wb[d][j].z, wa[d][j].w, wb[d][j].w, xb[d][j].z, xb[d][j].w);
        }
      }
      if (t + d + D < total) issue(d);
      if (++pi == nb) {
        pi = 0;
        sm.part[warp][2 * c][u * 16 + g] = acc[0];
        sm.part[warp][2 * c + 1][u * 16 + g] = acc[1];
        sm.part[warp][2 * c][u * 16 + g + 8] = acc[2];
        sm.part[warp][2 * c + 1][u * 16 + g + 8] = acc[3];
        acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
        ++u;
      }
    }
  }
}

// The Volta phase: units of 32 rows (four groups of 8), blocks of 8 J elements; lane (q, r) holds row 8 q + r of the
// unit and column r of x, and one 16-byte piece of each is two mma.m8n8k4.
template <int S, int J, int D>
__device__ __forceinline__ void head_phase_884(const HeadOp &op, int r0, int hi, int b0, int b1, int warp, int lane, MmaSmem<S> &sm) {
  constexpr int PU = 2;  // units of 32 rows per phase (the phase is 64 rows, as the others)
  const int q = (lane & 15) >> 2, half = lane >> 4, r = (lane & 3) + 4 * half;
  const int nb = b1 - b0, units = min(PU, (hi - r0 + 31) / 32), total = units * nb;
  uint4 wr[D][J], xr[D][J];
  int iu = 0, ip = 0;
  const int col = min(r, op.n - 1);
  auto issue = [&](int d) {
    const int row = r0 + iu * 32 + 8 * q + r;
    const size_t rowc = row < op.vocab ? row : 0, k = (size_t)(b0 + ip) * 8 * J;
#pragma unroll
    for (int j = 0; j < J; ++j) {
      wr[d][j] = __ldcs(reinterpret_cast<const uint4 *>(op.w + rowc * op.hidden + k + 8 * j));
      xr[d][j] = __ldcg(reinterpret_cast<const uint4 *>(op.x + (size_t)col * op.hidden + k + 8 * j));
    }
    if (++ip == nb) { ip = 0; ++iu; }
  };
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < total) issue(d);

  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.f;
  int u = 0, pi = 0;
  for (int t = 0; t < total; t += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      if (t + d >= total) break;  // warp-uniform
#pragma unroll
      for (int j = 0; j < J; ++j) {
        mma_f16_884(acc, bf16x2_to_f16x2(wr[d][j].x), bf16x2_to_f16x2(wr[d][j].y), bf16x2_to_f16x2(xr[d][j].x), bf16x2_to_f16x2(xr[d][j].y));
        mma_f16_884(acc, bf16x2_to_f16x2(wr[d][j].z), bf16x2_to_f16x2(wr[d][j].w), bf16x2_to_f16x2(xr[d][j].z), bf16x2_to_f16x2(xr[d][j].w));
      }
      if (t + d + D < total) issue(d);
      if (++pi == nb) {
        pi = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i)
          sm.part[warp][(i & 4) + (lane & 2) + (i & 1)][u * 32 + 8 * q + (lane & 1) + (i & 2) + 4 * half] = acc[i];
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.f;
        ++u;
      }
    }
  }
}

// Element `item` of op.items on a CTA of S warps: the rows of units [item U / items, (item + 1) U / items),
// U = ceil(vocab / UR), UR = 16 (32 on Volta's shape). The S partial sums of an (row, column) are added in warp order and scaled.
template <int S, int J, int D, int KIND>
__device__ __forceinline__ void head_item_mma(const HeadOp &op, int item, unsigned tid, MmaSmem<S> &sm) {
  constexpr int UR = KIND == 2 ? 32 : 16, BLK = KIND == 2 ? 8 * J : 32 * J;  // rows of a unit, elements of a block
  constexpr int PR = 16 * kMmaPhaseUnits, RS = 32 * S / 8;                    // rows of a phase; threads per column in the combine
  const int warp = tid >> 5, lane = tid & 31;
  const long long units = (op.vocab + UR - 1) / UR;
  const int lo = (int)(item * units / op.items) * UR, hi = min((int)(((item + 1) * units / op.items) * UR), op.vocab);
  const int blocks = op.hidden / BLK, b0 = warp * blocks / S, b1 = (warp + 1) * blocks / S;

  if (op.hist)
    for (unsigned i = tid; i < 8 * 256; i += 32 * S) (&sm.hist[0][0])[i] = 0u;
  if (warp == 0) {
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const int col = min(c, op.n - 1);
      const float s = op.norm_parts ? norm_scale(op.norm_parts + (size_t)col * op.norm_parts_n, op.norm_parts_n, op.inv_n, op.eps, lane) : 1.f;
      if (!lane) sm.scale[c] = s;
    }
  }
  __syncthreads();

  const int c_me = tid / RS, r_me = tid % RS;  // this thread's column and first row in the combine
  float best = __int_as_float(0xff800000);
  int arg = 0x7fffffff;
  for (int r0 = lo; r0 < hi; r0 += PR) {
    if constexpr (KIND == 2) head_phase_884<S, J, D>(op, r0, hi, b0, b1, warp, lane, sm);
    else head_phase_mma<S, J, D, KIND>(op, r0, hi, b0, b1, warp, lane, sm);
    __syncthreads();
    if (c_me < op.n)
      for (int rr = r_me; rr < PR; rr += RS) {
        const int row = r0 + rr;
        if (row >= hi) break;
        float sum = sm.part[0][c_me][rr];
#pragma unroll
        for (int w = 1; w < S; ++w) sum = __fadd_rn(sum, sm.part[w][c_me][rr]);
        const float l = __fmul_rn(__fmul_rn(sum, sm.scale[c_me]), op.logit_scale);
        if (op.logits) __stcg(op.logits + (size_t)c_me * op.vocab + row, l);  // read by other SMs: L2 only, not L1
        if (op.hist) atomicAdd(&sm.hist[c_me][key_byte(l)], 1u);
        if (l > best) { best = l; arg = row; }  // ascending rows: the first of equals
      }
    __syncthreads();
  }
  if (op.hist)
    for (unsigned i = tid; i < 8 * 256; i += 32 * S)
      if (i / 256 < (unsigned)op.n) __stcg(op.hist + ((size_t)(i / 256) * op.items + item) * 256 + i % 256, (&sm.hist[0][0])[i]);
  if (op.stats_max) {
#pragma unroll
    for (int c = 0; c < 8; ++c) {
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
    if (tid < 8 && tid < op.n) {
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
