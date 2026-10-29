/* SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
 * Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
 *
 * mlops_rope: rotary position embedding (Su et al., RoFormer) of q and k in one pass, CUDA C++.
 *
 * Pair i of a head (i < rotary_dim / 2) is rotated by the angle a = m * inv_freq[i] of its token's position m:
 *     y1 = x1 cos a - x2 sin a,     y2 = x2 cos a + x1 sin a
 * with (x1, x2) = (x[i], x[i + rotary_dim / 2]) (half split: NeoX, Llama, Qwen, ...) or (x[2i], x[2i + 1])
 * (interleaved: GPT-J, ...). Elements from rotary_dim to head_dim pass through (partial rotary). inverse = 1
 * rotates by -a: the backward of the forward.
 *
 * The angles come from one of two sources:
 *   positions + freq      exact: the phase is computed in turns, a / 2pi = m * f, f = inv_freq / 2pi held as a
 *                         two-float sum (mlops_rope_freq), the position split m = 2^16 a + b, so the phase is
 *                         reduced to [-1/2, 1/2] turn with an error near 2^-24 turn for every |m| < 2^31 --
 *                         then sincospif, times attention_factor (YaRN's mscale, LongRoPE's factor). y is
 *                         computed in FP32 and rounded once to the tensor's dtype.
 *   cos, sin tables       [batch, seq, rotary_dim] in the tensor's dtype, as Hugging Face's rotary modules make
 *                         them (cos[i] and cos[i + rotary_dim / 2] both hold pair i's cosine, half split; cos[2i],
 *                         cos[2i + 1] interleaved). y is Hugging Face's apply_rotary_pos_emb bit for bit:
 *                         y1 = R(R(x1 cos1) + R(-x2 sin1)), y2 = R(R(x2 cos2) + R(x1 sin2)), R the rounding to
 *                         the dtype after each FP32 operation; with inverse = 1 its autograd backward bit for bit.
 *
 * The sines and cosines of a token are computed (or loaded) once and shared by all heads of q and k. Tensors
 * are [batch, seq, heads, head_dim] with any strides except a contiguous last dimension: [B, S, H, D] and
 * Hugging Face's [B, H, S, D] are both strided views of it. q and k may have different head counts (GQA);
 * out may equal in (in place). Stateless C call over raw device pointers, explicit stream, a status instead of
 * exceptions.
 */
#ifndef MLOPS_ROPE_H
#define MLOPS_ROPE_H

#ifdef __CUDACC_RTC__
#include <cuda/std/cstdint>
using cuda::std::int32_t;
using cuda::std::int64_t;
typedef struct CUstream_st *cudaStream_t;
#else
#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { MLOPS_ROPE_BF16 = 0, MLOPS_ROPE_FP16 = 1, MLOPS_ROPE_FP32 = 2 } mlops_rope_dtype;
typedef enum { MLOPS_ROPE_OK = 0, MLOPS_ROPE_INVALID_VALUE = 1, MLOPS_ROPE_CUDA_ERROR = 2 } mlops_rope_status;

/* A [batch, seq, heads, head_dim] tensor: strides in elements, the last dimension contiguous. */
typedef struct {
  void *data;
  int heads;
  int64_t stride_b, stride_s, stride_h;
} mlops_rope_tensor;

typedef struct {
  int batch, seq, head_dim, rotary_dim;  /* rotary_dim even, <= head_dim and <= 1024 */
  int interleaved;                       /* 0: pairs (i, i + rotary_dim / 2); 1: pairs (2i, 2i + 1) */
  int inverse;                           /* 1: rotate by -angle (the backward) */
  mlops_rope_dtype dtype;                /* of q, k and the cos / sin tables */

  /* angles from positions: positions [batch, seq] (int32, or int64 with |m| < 2^31), or NULL for
   * m = s + position_offset; freq: rotary_dim / 2 entries of 4 floats from mlops_rope_freq */
  const void *positions;
  int positions_int64;
  int64_t pos_stride_b, pos_stride_s;
  int64_t position_offset;
  const float *freq;            /* device */
  float attention_factor;       /* 0 = 1 */

  /* or angles from tables: cos, sin [batch, seq, rotary_dim] (strides may be 0: broadcast); then positions
   * and freq must be NULL */
  const void *cos, *sin;
  int64_t table_stride_b, table_stride_s;
} mlops_rope_config;

/* q_in -> q_out and, if k_in is not NULL, k_in -> k_out (out may equal in). */
mlops_rope_status mlops_rope(const mlops_rope_config *cfg, const mlops_rope_tensor *q_in, const mlops_rope_tensor *q_out,
                             const mlops_rope_tensor *k_in, const mlops_rope_tensor *k_out, cudaStream_t stream);

/* Host: the freq table of n = rotary_dim / 2 inverse frequencies (radians per position, e.g. a YaRN-scaled set):
 * out[4i .. 4i + 3] = f_hi, f_lo, g_hi, g_lo with f = inv_freq[i] / 2pi = f_hi + f_lo and
 * g = frac(2^16 f) = g_hi + g_lo. Copy it to the device for cfg->freq. */
void mlops_rope_freq(const double *inv_freq, int n, float *out);

/* RoPE frequency scalings, as transformers computes them (modeling_rope_utils), in float64. */
typedef enum {
  MLOPS_ROPE_SCALE_DEFAULT = 0, /* base^(-2i / rotary_dim) */
  MLOPS_ROPE_SCALE_LINEAR = 1,  /* factor: every frequency / factor */
  MLOPS_ROPE_SCALE_DYNAMIC = 2, /* factor: NTK-aware base growth with seq_len beyond max_position_embeddings */
  MLOPS_ROPE_SCALE_YARN = 3,    /* factor (0: max_position_embeddings / original), original_max_position_embeddings,
                                   beta_fast (0: 32), beta_slow (0: 1), attention_factor or mscale / mscale_all_dim,
                                   no_truncate; attention factor 0.1 ln(factor) + 1 */
  MLOPS_ROPE_SCALE_LLAMA3 = 4,  /* factor, low_freq_factor, high_freq_factor, original_max_position_embeddings */
  MLOPS_ROPE_SCALE_LONGROPE = 5 /* short_factor / long_factor (the long ones beyond the original length), original_max_position_
                                   embeddings, factor (0: max_position_embeddings / original), attention_factor */
} mlops_rope_scale_type;

typedef struct {
  mlops_rope_scale_type type;
  double factor;
  int original_max_position_embeddings;
  double low_freq_factor, high_freq_factor;
  double beta_fast, beta_slow;
  double attention_factor;   /* 0: from the factor */
  double mscale, mscale_all_dim;
  int no_truncate;           /* YaRN: 1 = do not round the correction range to whole dimensions */
  const double *short_factor, *long_factor;  /* LongRoPE: rotary_dim / 2 each */
} mlops_rope_scaling;

/* inv_freq[rotary_dim / 2] (radians per position) and the attention factor of RoPE with an optional scaling (NULL: none)
 * over `base` (rope_theta).  max_position_embeddings (0: unknown) and seq_len (0: none) as transformers uses them. */
mlops_rope_status mlops_rope_frequencies(int rotary_dim, double base, const mlops_rope_scaling *scaling,
                                         int max_position_embeddings, int seq_len, double *inv_freq, float *attention_factor);

const char *mlops_rope_status_string(mlops_rope_status status);
/* Description of the most recent failure of an mlops_rope call on the calling thread; "" after a success. */
const char *mlops_rope_last_error(void);

#ifdef __cplusplus
}
#endif
#endif /* MLOPS_ROPE_H */
