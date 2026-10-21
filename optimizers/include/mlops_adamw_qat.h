/* SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
 * Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
 *
 * mlops_adamw_qat: AdamW that trains an NVFP4 weight in place, with no FP32 master and 5 bytes of
 * state per element + 8 per 1x16 block. Derivations: README.md. PyTorch reference of the same step
 * (bit-identical): python/mlops/optim/reference.py.
 *
 * Lattice. E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} are anchors at integer coordinates joined
 * by secants: g(u) is piecewise linear with slope g' = 0.5 / 1 / 2 (the 4-bit code read as an
 * integer, made continuous). Element i of block b has the master and the forward value
 *     w_i = sigma_b * g(u_i),   q_i = sigma_b * g(round(u_i)),   sigma_b = E4M3(k_b) * gdec
 * u_i a Q3.12 coordinate, k_b the block's scale byte, gdec the global decode factor (fixed).
 *
 * Element step (grad = dL/dq, exact). AdamW on q in weight units, mapped to u through the Jacobian
 * J = sigma~ * g'(u); 1 / sigma~ is interpolated linearly between the neighbouring block centers along
 * the row. The moments never see J.
 *     v    = beta2 * s^2 + (1 - beta2) * grad^2
 *     y    = rsqrt(v + eps^2)                     magic 0x5F1FFFF9 + one Newton step, |err| <= 6.5e-4
 *     s   <- SR_E5M2(v * y),  E[s^2] = v          per-block power-of-two exponent
 *     r   <- SR_E4M3((beta1 * r * s + (1 - beta1) * grad) * y)         r = m / sqrt(v), |r| <= 7.27
 *     u   <- SR_Q3.12(u - lr * (r * sqrt(1 - beta2^t) / (1 - beta1^t) + weight_decay * w) / J)
 *     qbar <- SR_Q3.4(beta1 * qbar + (1 - beta1) * q / sigma)          EMA of the forward value
 * SR = stochastic rounding, unbiased. Code nibbles = round(u), written straight into q.
 *
 * Block-scale step. Under QAT an element oscillates between the two codes around its optimum w*, so
 * neither u nor q locates w*; the mean of the oscillation does: m = H (qbar - w*), so
 *     w*_i = qbar_i - m_i / H_b
 * with H_b the block's curvature along the trajectory, the secant regression of the gradient on the
 * code moves d = q - qbar(t-1) (E[grad | past] = H (q - w*); grad is the only regressor independent of
 * the moves, which m(t-1) drove):
 *     H_b = EMA_beta2[sum grad d] / EMA_beta2[sum d^2]          (two BF16 accumulators, SR)
 * H_b is used only when significant: with H = 0 the numerator is a sum of independent terms with
 * E[grad^2] = v, so its null variance is (1 - b2)/(1 + b2) (1 + b2^t)/(1 - b2^t) mean(v) EMA[sum d^2];
 * the block votes when the numerator exceeds z of those standard deviations (z = scale_z).
 * The block's preferred byte minimizes the representation loss of w* over one E4M3 octave each way,
 *     k* = argmin_{|k - k_b| <= 8} sum_i huber(w*_i - Q_k(w*_i), delta = sigma_b)
 * and a two-sided CUSUM on the vote x = sign(k* - k_b) (0 without a significant H_b) decides the move:
 *     S <- sign(S + P x) * max(0, |S + P x| - P / 16);   |S| > h = 4 z^2  ->  k_b += sign(S), S = 0
 * The vote is evaluated every P = (1 + beta1) / (1 - beta1) steps (19 for beta1 = 0.9, the integrated
 * autocorrelation time of the beta1 EMAs it is made of) and held for P steps: S counts steps of
 * agreement, a move needs about 4 z^2 / (1 - 1/16) = 38 steps of a unanimous vote, and a vote that
 * stops recurring drains at 1/16 per step. A move keeps every master value: u and qbar are re-encoded
 * for the new sigma (SR), only the representation changes; at a fixed w* the argmin never revisits the
 * byte it left, so there is no oscillation between two scales.
 *
 * Layout and limits. Rowwise NVFP4, 1x16 blocks, E4M3 block scales (linear or swizzled), global per
 * tensor or per row; no columnwise copy. q is part of the state: initialized from the initial weight
 * (mlops_adamw_qat_init_quantized / mlops_adamw_qat_init), every step rewrites its codes and scales in
 * place. cols % 16 == 0.
 * Gradients E5M2 or E4M3 (decoded value times grad_scale: the per-tensor scale of an FP8 GEMM output)
 * or FP32, 16-byte aligned, finite (FP8 Inf / NaN codes decode as Inf / NaN from sm_89 on, as large
 * finite values below: outside the contract). Every architecture sm_50..sm_90 computes the same bits.
 */
#ifndef MLOPS_ADAMW_QAT_H
#define MLOPS_ADAMW_QAT_H

#include <stdint.h>
#include "mlops_adamw.h"
#include <nvfp4effma.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float lr, beta1, beta2, eps, weight_decay;  /* AdamW's, lr in weight units */
  int step;                                   /* 1-based; bias corrections are always on */
  float grad_scale;                           /* gradient multiplier (0 = 1) */
  float scale_z;                              /* curvature significance and CUSUM threshold, in null std devs (0 = 3) */
  uint32_t seed;                              /* stochastic-rounding stream */
} mlops_adamw_qat_config;

typedef struct {
  int16_t *u;       /* rows*cols  grid coordinate, Q3.12 */
  uint8_t *r;       /* rows*cols  m / sqrt(v), E4M3 */
  uint8_t *s;       /* rows*cols  sqrt(v) * 2^-s_exp, E5M2 */
  int8_t *qbar;     /* rows*cols  EMA of q / sigma, Q3.4 */
  int8_t *s_exp;    /* blocks     power-of-two exponent of s */
  int16_t *cusum;   /* blocks     scale-move CUSUM, Q7.8 */
  uint16_t *hnum;   /* blocks     secant numerator EMA, BF16 */
  uint16_t *hden;   /* blocks     secant denominator EMA, BF16 */
  uint8_t *next;    /* blocks     scratch: new scale bytes */
  float *gdec;      /* rows       scratch: decode factor of each row's global */
} mlops_adamw_qat_state;

/* Bytes of the single state buffer for a rows x cols weight, and the carve of `buffer` (16-byte
 * aligned device memory of that size) into `st`. */
size_t mlops_adamw_qat_state_bytes(int rows, int cols);
mlops_adamw_status mlops_adamw_qat_state_bind(void *buffer, int rows, int cols, mlops_adamw_qat_state *st);

/* Initial state from the initial weight w (rows x cols, BF16/FP16/FP32) and q, w already quantized into
 * it by NVFP4EffMA (1x16 blocks, E4M3 scales; e.g. nvfp4a_quantize_ex, nvfp4effma.quantize in Python):
 * sets u_i = g^-1(w_i / sigma_b) (RNE to Q3.12) and zeroes the rest of the state. The global is then
 * raised 2^3 and every block byte lowered 24 codes (exact for normals), so every block scale can grow 8x.
 * Without a global (global_scale NULL) blocks at the E4M3 maximum stay pinned there. w is not needed
 * afterwards. Needs no NVFP4EffMA code at link time. */
mlops_adamw_status mlops_adamw_qat_init_quantized(int rows, int cols, const void *w, mlops_adamw_dtype w_dtype,
                                                  const nvfp4a_tensor *q, const mlops_adamw_qat_state *st,
                                                  cudaStream_t stream);

/* nvfp4a_quantize_ex of w into q (qcfg: 1x16 tiles, any 4over6 / compute_amax setting), then
 * mlops_adamw_qat_init_quantized. Links NVFP4EffMA's library (-lnvfp4effma; pkg-config does it). */
mlops_adamw_status mlops_adamw_qat_init(int rows, int cols, const void *w, mlops_adamw_dtype w_dtype,
                                        const nvfp4a_tensor *q, const nvfp4a_quant_config *qcfg,
                                        const mlops_adamw_qat_state *st, cudaStream_t stream);

/* One step. grad = dL/dq for the dequantized weight the forward used (rows x cols; E5M2 / E4M3 / FP32). */
mlops_adamw_status mlops_adamw_qat(const mlops_adamw_qat_config *cfg, int rows, int cols, const void *grad,
                                   mlops_adamw_dtype grad_dtype, const nvfp4a_tensor *q,
                                   const mlops_adamw_qat_state *st, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif /* MLOPS_ADAMW_QAT_H */
