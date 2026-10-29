/* SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
 * Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
 *
 * mlops_lm_head: logits[v] = s * sum_k W[v][k] x[k], W [vocab, hidden] BF16, x [hidden] BF16, FP32 accumulation.
 *
 * The matrix is cut along hidden: the S warps of a CTA take disjoint ranges of 256-column passes of the same
 * rows, each keeps its piece of x, and the S partial sums of a row are added in warp order in shared memory --
 * no atomics, no global partials, a fixed order. A unit is R rows of one warp; the loads of a step (R rows x
 * 16 bytes a lane, one 256-column pass) are issued D steps ahead, across units, in a register ring. The rows are
 * split over `items` elements in units, balanced to one unit.
 *
 * Epilogue (all optional):
 *   scale   s[c] = 1 / sqrt(sum(norm_parts[c]) * inv_n + eps), the parts added lane-strided, then a butterfly (the
 *           deferred RMSNorm scalar of the producer of x); NULL norm_parts: s = 1
 *   logits  [n][vocab] the scaled rows times logit_scale, FP32 (two roundings: s, then logit_scale); NULL: not written
 *   stats   [n][items] the maximum and its lowest index over each element's rows (the sampler's chunk statistics),
 *           and [n][items][256] the counts of the top byte of the rows' order keys (a float's bits, flipped so that a
 *           larger float is a larger key): the sampler's first radix pass for a top-k
 *
 * Variants are the instances the device accepts for the shape (mlops_lm_head_variants): CUDA cores (R, S, D, NB) on
 * every architecture, and tensor cores (S, D, J) from sm_70 (FP16 m8n8k4), sm_75 (FP16 m16n8k8) and sm_80 (BF16 m16n8k16); the call and the work
 * items run the one chosen by the caller. Tensor cores add products exactly and the products of a mma plus the
 * accumulator in an order of their own, so their logits are not bit-equal to the CUDA-core path's; each path is
 * deterministic and independent of the element count. The FP16 instances (Volta and Turing) convert W and x from BF16 in registers:
 * the mantissa is kept (FP16 has more bits), the range is FP16's -- |x| must be below 65504, and a weight below
 * 2^-14 loses at most 2^-25 of absolute precision per product factor of x. Stateless C call over raw device pointers, explicit stream.
 */
#ifndef MLOPS_LM_HEAD_H
#define MLOPS_LM_HEAD_H

#ifdef __CUDACC_RTC__
#include <cuda/std/cstdint>
using cuda::std::int64_t;
using cuda::std::uint16_t;
typedef struct CUstream_st *cudaStream_t;
#else
#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { MLOPS_LM_HEAD_OK = 0, MLOPS_LM_HEAD_INVALID_VALUE = 1, MLOPS_LM_HEAD_CUDA_ERROR = 2 } mlops_lm_head_status;

typedef struct {
  int vocab, hidden;     /* hidden a multiple of 256 */
  int n;                 /* columns of x, 1 .. 8 */
  int items;             /* elements (CTAs) the rows are split over; 0: the device's resident CTAs */
  int norm_parts_n;      /* entries of norm_parts */
  float inv_n, eps;      /* 1 / hidden of the norm, its epsilon */
  float logit_scale;     /* > 0: multiplies the scaled rows once more (the sampler's 1 / temperature); 1: none */
} mlops_lm_head_config;

/* One way to run the shape on the current device. */
typedef struct {
  int rows_per_unit;     /* R: rows a warp holds at once */
  int k_pieces;          /* S: warps of a CTA over hidden; threads = 32 S */
  int depth;             /* D: steps of loads in flight per lane (a ring of D x (R + NB) 16-byte values) */
  int columns;           /* NB: columns the instance carries, the smallest of 1, 2, 4, 8 that holds n */
  int tensor_cores;      /* 1: mma.m16n8k16 BF16 (sm_80+), 2: mma.m16n8k8 FP16 (sm_75+), 3: mma.m8n8k4 FP16 (sm_70+); rows_per_unit 16, 16 and 32, columns 8; 0: CUDA cores */
  int chunks;            /* tensor cores: J, 16-byte pieces a lane reads of each row per step; CUDA cores: 0 */
  int threads, smem_bytes, regs, arch;
} mlops_lm_head_variant;

/* Writes up to max variants, returns their number (0 for an invalid config). max = 0 queries the count. */
int mlops_lm_head_variants(const mlops_lm_head_config *c, mlops_lm_head_variant *out, int max);

/* x [n][hidden], norm_parts [n][norm_parts_n]; stats_max / stats_arg [n][items] (the resolved count:
 * mlops_lm_head_item_count), stats_hist [n][items][256] (needs stats_max). All pointers are device
 * pointers; w and x 128-byte aligned (a warp reads 512 contiguous bytes of a row per pass). */
mlops_lm_head_status mlops_lm_head(const mlops_lm_head_config *c, const mlops_lm_head_variant *variant, const uint16_t *w,
                                   const uint16_t *x, const float *norm_parts, float *logits, float *stats_max,
                                   int *stats_arg, unsigned *stats_hist, cudaStream_t stream);

/* The number of elements a call with this config and variant runs. */
int mlops_lm_head_item_count(const mlops_lm_head_config *c, const mlops_lm_head_variant *variant);

const char *mlops_lm_head_last_error(void);

#ifdef __cplusplus
}
#endif
#endif /* MLOPS_LM_HEAD_H */
