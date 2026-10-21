// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_adamw: fused in-place AdamW step. Operation order: include/mlops_adamw.h. Host side and the tail
// kernel; the vectorized kernel instances are in adamw_inst.cu (adamw_kernels.cuh).
#include "adamw_kernels.cuh"

using namespace mlops_adamw_detail;

namespace {

// Remainder count % 4 (at most 3 elements): a runtime dtype switch is fine at this size.
__global__ void adamw_kernel_tail(AdamArgs a, size_t start, size_t count, void *__restrict__ param,
                                  mlops_adamw_dtype pdt, const void *__restrict__ grad, mlops_adamw_dtype gdt,
                                  float *__restrict__ exp_avg, float *__restrict__ exp_avg_sq) {
  const size_t i = start + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  float p = mlops_adamw_load_f32(param, pdt, i), m = exp_avg[i], v = exp_avg_sq[i];
  adam_elem(a, mlops_adamw_load_f32(grad, gdt, i), p, m, v);
  exp_avg[i] = m;
  exp_avg_sq[i] = v;
  mlops_adamw_store_f32(param, pdt, i, p);
}

template <mlops_adamw_dtype PDT, mlops_adamw_dtype GDT>
void launch(const AdamArgs &a, size_t count, void *param, const void *grad, float *exp_avg, float *exp_avg_sq,
           cudaStream_t stream) {
  const size_t count4 = count / 4, tail = count - count4 * 4;
  if (count4 > 0) launch_vec4<PDT, GDT>(a, count4, param, grad, exp_avg, exp_avg_sq, stream);
  if (tail > 0)
    adamw_kernel_tail<<<grid_for(tail), kThreads, 0, stream>>>(a, count4 * 4, count, param, PDT, grad, GDT, exp_avg, exp_avg_sq);
}

} // namespace

extern "C" mlops_adamw_status mlops_adamw(const mlops_adamw_config *cfg, size_t count, void *param,
                                          mlops_adamw_dtype param_dtype, const void *grad,
                                          mlops_adamw_dtype grad_dtype, float *exp_avg, float *exp_avg_sq,
                                          cudaStream_t stream) {
  if (!cfg || !param || !grad || !exp_avg || !exp_avg_sq)
    return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw: null pointer argument");
  if (cfg->step < 1) return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw: step must be >= 1");
  // vec4 path needs 16-byte-aligned param/grad/exp_avg/exp_avg_sq; cudaMalloc buffers always are,
  // a sliced sub-view might not be.
  const auto aligned16 = [](const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; };
  if (!aligned16(param) || !aligned16(grad) || !aligned16(exp_avg) || !aligned16(exp_avg_sq))
    return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw: param/grad/exp_avg/exp_avg_sq must be 16-byte aligned");
  if (count == 0) return MLOPS_ADAMW_OK_();

  const double beta1 = cfg->beta1, beta2 = cfg->beta2;
  AdamArgs a;
  a.lr = cfg->lr;
  a.omb1 = static_cast<float>(1.0 - beta1);
  a.omb2 = static_cast<float>(1.0 - beta2);
  a.rbc1 = cfg->bias_correction ? static_cast<float>(1.0 / (1.0 - pow(beta1, cfg->step))) : 1.0f;
  a.rbc2 = cfg->bias_correction ? static_cast<float>(1.0 / (1.0 - pow(beta2, cfg->step))) : 1.0f;
  a.eps = cfg->eps;
  a.lr_wd = static_cast<float>(static_cast<double>(cfg->lr) * cfg->weight_decay);
  a.gscale = cfg->grad_scale == 0.0f ? 1.0f : cfg->grad_scale;

#define MLOPS_ADAMW_CASE(PDT, GDT)                                                               \
  if (param_dtype == (PDT) && grad_dtype == (GDT)) launch<PDT, GDT>(a, count, param, grad, exp_avg, exp_avg_sq, stream); else
  MLOPS_ADAMW_CASE(MLOPS_ADAMW_FP32, MLOPS_ADAMW_FP32)
  MLOPS_ADAMW_CASE(MLOPS_ADAMW_FP32, MLOPS_ADAMW_FP16)
  MLOPS_ADAMW_CASE(MLOPS_ADAMW_FP32, MLOPS_ADAMW_BF16)
  MLOPS_ADAMW_CASE(MLOPS_ADAMW_FP16, MLOPS_ADAMW_FP32)
  MLOPS_ADAMW_CASE(MLOPS_ADAMW_FP16, MLOPS_ADAMW_FP16)
  MLOPS_ADAMW_CASE(MLOPS_ADAMW_FP16, MLOPS_ADAMW_BF16)
  MLOPS_ADAMW_CASE(MLOPS_ADAMW_BF16, MLOPS_ADAMW_FP32)
  MLOPS_ADAMW_CASE(MLOPS_ADAMW_BF16, MLOPS_ADAMW_FP16)
  MLOPS_ADAMW_CASE(MLOPS_ADAMW_BF16, MLOPS_ADAMW_BF16)
  { return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw: invalid dtype"); }
#undef MLOPS_ADAMW_CASE

  MLOPS_ADAMW_CUDA_CHECK(cudaGetLastError());
  return MLOPS_ADAMW_OK_();
}

extern "C" const char *mlops_adamw_status_string(mlops_adamw_status status) {
  switch (status) {
    case MLOPS_ADAMW_OK: return "ok";
    case MLOPS_ADAMW_INVALID_VALUE: return "invalid value";
    case MLOPS_ADAMW_CUDA_ERROR: return "cuda error";
    default: return "unknown";
  }
}

extern "C" const char *mlops_adamw_last_error(void) { return mlops_adamw_detail::last_error_buf(); }
