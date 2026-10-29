/* SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
 * Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
 *
 * mlops_embed: the rows of an embedding table for token ids that live on the device, with the inputs of the
 * first RMSNorm of the model.
 *
 * For each column c < n (a token): x[c] = table[ids[c]] widened to FP32; xw[c] = bf16(x[c] * norm_w), one FP32
 * product rounded to nearest even; parts[c][u] = the sum of squares of x[c] over its u-th 256 elements -- the
 * lanes' eight squares added in order, then an xor butterfly -- the partial sums that mlops_lm_head and the
 * consumers of a deferred RMSNorm scalar add in a fixed order. Every output is optional.
 *
 * The table is BF16 [vocab, hidden] in device memory or in mapped host memory. An id outside [0, vocab)
 * leaves its column zero and sets *error to 1 (error may be NULL). Stateless C call over raw pointers.
 */
#ifndef MLOPS_EMBED_H
#define MLOPS_EMBED_H

#ifdef __CUDACC_RTC__
#include <cuda/std/cstdint>
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

typedef enum { MLOPS_EMBED_OK = 0, MLOPS_EMBED_INVALID_VALUE = 1, MLOPS_EMBED_CUDA_ERROR = 2 } mlops_embed_status;

typedef struct {
  int vocab, hidden;  /* hidden a multiple of 256 */
  int n;              /* columns (tokens), >= 1 */
  int items;          /* CTAs the units are split over; 0: one per SM and resident block */
} mlops_embed_config;

/* One way to run the shape on the current device: the warps of a CTA. */
typedef struct {
  int warps, threads, smem_bytes, regs, arch;
} mlops_embed_variant;

/* Writes up to max variants, returns their number (0 for an invalid config). max = 0 queries the count. */
int mlops_embed_variants(const mlops_embed_config *c, mlops_embed_variant *out, int max);

/* Elements a call runs, and the parts a column has (hidden / 256). */
int mlops_embed_item_count(const mlops_embed_config *c, const mlops_embed_variant *variant);

/* ids [n]; norm_w [hidden] BF16; x [n][hidden] FP32; xw [n][hidden] BF16; parts [n][hidden / 256] FP32.
 * table, x, xw 16-byte aligned. */
mlops_embed_status mlops_embed(const mlops_embed_config *c, const mlops_embed_variant *variant, const uint16_t *table,
                               const int *ids, const uint16_t *norm_w, float *x, uint16_t *xw, float *parts, int *error,
                               cudaStream_t stream);

const char *mlops_embed_last_error(void);

#ifdef __cplusplus
}
#endif
#endif /* MLOPS_EMBED_H */
