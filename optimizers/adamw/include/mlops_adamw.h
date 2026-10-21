/* mlops/optimizers/adamw: decoupled-weight-decay Adam (Loshchilov & Hutter), CUDA C++.
 *
 * param/grad: BF16, FP16 or FP32; exp_avg/exp_avg_sq: always FP32. Stateless C call over raw
 * device pointers, explicit cudaStream_t, returns a status instead of throwing/aborting.
 *
 * All four buffers must be 16-byte aligned (true for cudaMalloc'd buffers; a sub-view must start
 * on a 16-byte boundary) -- 4 elements/thread through one 128-bit load/store, bandwidth-bound.
 */
#ifndef MLOPS_ADAMW_H
#define MLOPS_ADAMW_H

#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MLOPS_ADAMW_VERSION "0.1.0"

/* E4M3 / E5M2: FP8 gradients of mlops_adamw_qat (decoded value * grad_scale); mlops_adamw rejects them */
typedef enum {
  MLOPS_ADAMW_BF16 = 0,
  MLOPS_ADAMW_FP16 = 1,
  MLOPS_ADAMW_FP32 = 2,
  MLOPS_ADAMW_E4M3 = 3,
  MLOPS_ADAMW_E5M2 = 4
} mlops_adamw_dtype;
typedef enum {
  MLOPS_ADAMW_OK = 0,
  MLOPS_ADAMW_INVALID_VALUE = 1,
  MLOPS_ADAMW_CUDA_ERROR = 2
} mlops_adamw_status;

/* Elementwise over `count` values, in place. Per element, in FP32 (bit-reproducible; host
 * emulation in tests/test_adamw.cu):
 *   g = grad * grad_scale
 *   m = beta1 * m + (1 - beta1) * g;        v = beta2 * v + (1 - beta2) * (g * g)   (one FMA each)
 *   u = (m * rbc1) / (sqrt(v * rbc2) + eps)  with rbc = 1 / (1 - beta^step) (1 if no bias corr.)
 *   p = p - (lr * weight_decay) * p   (one FMA, skipped if weight_decay == 0)
 *   p = p - lr * u                    (one FMA)
 * The scalar factors (1 - beta, rbc, lr * weight_decay) are formed on the host in double and
 * rounded to FP32 once.
 */
typedef struct {
  float lr, beta1, beta2, eps, weight_decay;
  int step;             /* 1-based step number */
  int bias_correction;  /* 1: Adam bias correction, 0: raw moments */
  float grad_scale;     /* gradient multiplier, e.g. 1 / loss_scale (0 is treated as 1) */
} mlops_adamw_config;

mlops_adamw_status mlops_adamw(const mlops_adamw_config *cfg, size_t count,
                                void *param, mlops_adamw_dtype param_dtype,
                                const void *grad, mlops_adamw_dtype grad_dtype,
                                float *exp_avg, float *exp_avg_sq, cudaStream_t stream);

const char *mlops_adamw_status_string(mlops_adamw_status status);
/* Description of the most recent failure of an mlops_adamw call on the calling thread; ""
 * after a successful call. Valid until the next mlops_adamw call. */
const char *mlops_adamw_last_error(void);

#ifdef __cplusplus
}
#endif
#endif /* MLOPS_ADAMW_H */
