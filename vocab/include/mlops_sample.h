/* SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
 * Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
 *
 * mlops_sample: the next token from a row of logits, on the device, without sorting the vocabulary.
 *
 * The logit as sampled is l' = l * fl(1 / penalty) (l > 0) or l * penalty (l <= 0) for a token whose bit in
 * `seen` is set, then l' * fl(1 / temperature). temperature = 0 is greedy: the argmax of l' (the lowest index
 * among equals). Otherwise the token is drawn from softmax(l') restricted to the top_k largest (ties at the
 * k-th kept) and then to the smallest prefix, in descending order, whose preceding mass reaches top_p of the
 * kept mass (the first token is always kept). top_k = 0 and top_p = 1 switch the filters off.
 *
 * Masses exp(l' - max) are counted in fixed point (units of 2^-40): sums are exact integers in any order, so a
 * token depends on (logits, config, seed, step) alone -- not on the grid, the schedule or the device. The
 * threshold of top_k / top_p is found by four radix passes over the order keys of the logits, the draw is one
 * Philox4x32-10 block of (seed, step).
 *
 * The call runs one kernel per pass (1 for greedy, up to 7; with stats, the first is one CTA, and with hist one pass less). `state` lives on the device: seed, and the step
 * the call consumes and advances, so a loop of calls (or a captured graph) needs no host value. The workspace
 * is zero before the first call and is left reusable by every call that completes.
 */
#ifndef MLOPS_SAMPLE_H
#define MLOPS_SAMPLE_H

#ifdef __CUDACC_RTC__
#include <cuda/std/cstdint>
using cuda::std::uint32_t;
using cuda::std::uint64_t;
typedef struct CUstream_st *cudaStream_t;
#else
#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { MLOPS_SAMPLE_OK = 0, MLOPS_SAMPLE_INVALID_VALUE = 1, MLOPS_SAMPLE_CUDA_ERROR = 2 } mlops_sample_status;

typedef struct {
  int vocab;
  float temperature;  /* 0: greedy */
  int top_k;          /* 0: off */
  float top_p;        /* (0, 1]; 1: off */
  float penalty;      /* repetition penalty > 0; 1: off (seen may then be NULL) */
  int ctas;           /* CTAs of a pass; 0: four per SM. The token does not depend on it */
} mlops_sample_config;

/* Statistics of the logits, from the head's epilogue (mlops_lm_head's stats_max / stats_arg of column 0): the
 * maximum and its lowest index over each of `count` consecutive ranges of the vocabulary. With them the sampler
 * does not scan the logits for the maximum: greedy needs no pass over them at all, and the others read them
 * as they are. That asks the logits to be the sampled logits already -- penalty 1 (nothing to adjust) and
 * multiplied by 1 / temperature (the head's logit_scale; greedy: 1). With hist and a top_k the first radix pass,
 * which counts the key bytes, does not scan the logits either; hist is ignored otherwise. */
typedef struct {
  const float *max;
  const int *arg;
  int count;
  const unsigned *hist; /* or NULL: [count][256] counts of the key byte of each range (mlops_lm_head's stats_hist) */
} mlops_sample_stats;

/* Device memory: the Philox key and the counter of the next draw. */
typedef struct {
  uint64_t seed, step;
} mlops_sample_state;

/* Bytes of the workspace; 0 for an invalid config. */
size_t mlops_sample_workspace_bytes(const mlops_sample_config *c);

/* logits: [vocab] FP32, 16-byte aligned; seen: [ceil(vocab / 32)] bits of the tokens to penalize; stats: or NULL; token: one
 * int. All device pointers; the logits are only read, the workspace is written by the passes of the call. */
mlops_sample_status mlops_sample(const mlops_sample_config *c, const float *logits, const uint32_t *seen,
                                 const mlops_sample_stats *stats, mlops_sample_state *state, void *workspace,
                                 size_t workspace_bytes, int *token, cudaStream_t stream);

/* Message of the calling thread's last failed call. */
const char *mlops_sample_last_error(void);

#ifdef __cplusplus
}
#endif
#endif /* MLOPS_SAMPLE_H */
