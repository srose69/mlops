/* SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
 * Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
 *
 * mlops_norm: RMSNorm and LayerNorm of rows of hidden values, in the flavors of current models, with an optional residual
 * add before and a quantization after.
 *
 * Per row (or per group of hidden / groups values, normalized independently), with c = v (RMS) or v - mean(v) (LayerNorm):
 *     v = round_x(x + residual)                          (v is also written to residual_out)
 *     v = v * silu(gate)                                 with gate BEFORE (Mamba-2 / Nemotron-H with norm_before_gate off)
 *     r = 1 / sqrt(mean(c^2) + eps)                      (the sums in FP32, in units of 256 elements: a lane's eight
 *                                                         values in order, an xor butterfly, the unit sums lane-strided)
 *     w' = weight, 1 + weight (zero-centered gamma: Gemma, Qwen3-Next, Nemotron's LayerNorm1P) or 1 (no weight)
 *   ORDER_ROUND_THEN_SCALE (Llama, Qwen, Mistral, DeepSeek; RMS only, no bias, no gate):
 *     y = round_y(round_x(c * r) * w')                 (Hugging Face's order of roundings)
 *   ORDER_FP32 (Gemma, Qwen3-Next, LayerNorm of PyTorch, bias, gates): one rounding at the end
 *     y = round_y(fma(c * r, w', bias));   with gate AFTER (Qwen3-Next, Mamba-2 with norm_before_gate): y *= silu(gate)
 * round_x / round_y round to nearest even to the dtype of x / of y (FP32: no rounding); silu(g) = g / (1 + exp(-g)).
 * The sums do not depend on the variant or the element count, so the result is the same on every GPU and every launch
 * shape (the gates' exp aside: libm's on the device).
 *
 * The output is y itself, or y quantized as NVFP4EffMA's nvfp4a_quantize quantizes it -- with its primitives, imported:
 *   MLOPS_NORM_OUT_INT8   int8 [rows][hidden] and FP32 scales [rows][ld_scale], one per block of 16: sb = amax / 127,
 *                         q = RNE(y / sb) (0 where sb = 0).
 *   MLOPS_NORM_OUT_NVFP4  E2M1 nibbles [rows][hidden / 2] (element 2i in the low nibble) and E4M3 scales [rows][ld_scale],
 *                         one per block of 16, the global a decode factor g (device float; NULL: 1).
 *
 * x may be BF16, FP16 or FP32 (an FP32 residual stream); weight, bias and y are in y_dtype; x_dtype is y_dtype or FP32; the
 * gate is in x_dtype. residual_out may be x or residual (every thread reads before it writes). A group is a multiple of
 * 256 values (a per-head norm of 64 or 128 is the attention library's: mlops_fbla_qkv_at). Stateless C call over raw device
 * pointers, explicit stream.
 */
#ifndef MLOPS_NORM_LAYER_H
#define MLOPS_NORM_LAYER_H

#ifdef __CUDACC_RTC__
#include <cuda/std/cstdint>
using cuda::std::int8_t;
using cuda::std::uint8_t;
typedef struct CUstream_st *cudaStream_t;
#else
#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { MLOPS_NORM_OK = 0, MLOPS_NORM_INVALID_VALUE = 1, MLOPS_NORM_CUDA_ERROR = 2 } mlops_norm_status;
typedef enum { MLOPS_NORM_BF16 = 0, MLOPS_NORM_FP16 = 1, MLOPS_NORM_FP32 = 2 } mlops_norm_dtype;
typedef enum { MLOPS_NORM_OUT_DTYPE = 0, MLOPS_NORM_OUT_INT8 = 1, MLOPS_NORM_OUT_NVFP4 = 2 } mlops_norm_out;
typedef enum { MLOPS_NORM_RMS = 0, MLOPS_NORM_LAYER = 1 } mlops_norm_kind;
typedef enum { MLOPS_NORM_SCALE_NONE = 0, MLOPS_NORM_SCALE_WEIGHT = 1, MLOPS_NORM_SCALE_ONE_PLUS_WEIGHT = 2 } mlops_norm_scale;
typedef enum { MLOPS_NORM_ORDER_ROUND_THEN_SCALE = 0, MLOPS_NORM_ORDER_FP32 = 1 } mlops_norm_order;
typedef enum { MLOPS_NORM_GATE_NONE = 0, MLOPS_NORM_GATE_AFTER = 1, MLOPS_NORM_GATE_BEFORE = 2 } mlops_norm_gate;

typedef struct {
  int rows, hidden;          /* hidden a multiple of 256 * groups, at most 32768 */
  int groups;                /* normalized independently, hidden / groups values each; 0: 1 */
  float eps;
  mlops_norm_kind kind;
  mlops_norm_scale scale;    /* the weight's role */
  int bias;                  /* 1: add the bias (y_dtype [hidden]) */
  mlops_norm_order order;
  mlops_norm_gate gate;
  mlops_norm_dtype x_dtype;  /* of x, residual, residual_out and gate */
  mlops_norm_dtype y_dtype;  /* of weight, bias and y (and the dtype y is rounded to before a quantization) */
  mlops_norm_out out;
  int ld_scale;              /* quantized outputs: scales between rows; 0: hidden / 16 */
  int items;                 /* CTAs the rows are split over; 0: the device's resident CTAs */
} mlops_norm_config;

/* The pointers of a call: x [rows][hidden]; residual and residual_out both or neither; weight [hidden] unless scale is NONE;
 * bias [hidden] when config.bias; gate [rows][hidden] unless the gate is NONE; y [rows][hidden] (y_dtype, or int8, or
 * [rows][hidden / 2] bytes); scales [rows][ld_scale] (FP32 for INT8, E4M3 bytes for NVFP4); global: NVFP4's g. All 16-byte
 * aligned (scales: 4-byte). */
typedef struct {
  const void *x, *residual, *weight, *bias, *gate;
  void *residual_out, *y, *scales;
  const float *global;
} mlops_norm_args;

/* One way to run the shape on the current device: the warps of a CTA (a row is cut in units of 256 over them). */
typedef struct {
  int warps, threads, smem_bytes, regs, arch;
} mlops_norm_variant;

/* Writes up to max variants, returns their number (0 for an invalid config). max = 0 queries the count. */
int mlops_norm_variants(const mlops_norm_config *c, mlops_norm_variant *out, int max);

/* The number of elements (CTAs) a call with this config and variant runs. */
int mlops_norm_item_count(const mlops_norm_config *c, const mlops_norm_variant *variant);

mlops_norm_status mlops_norm(const mlops_norm_config *c, const mlops_norm_variant *variant, const mlops_norm_args *args, cudaStream_t stream);

const char *mlops_norm_last_error(void);

#ifdef __cplusplus
}
#endif
#endif /* MLOPS_NORM_LAYER_H */
