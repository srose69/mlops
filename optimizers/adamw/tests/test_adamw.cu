/* Correctness test: GPU mlops_adamw vs. a double-precision host reference, for every
 * {param_dtype, grad_dtype} combination over BF16/FP16/FP32. */
#include "mlops_adamw.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define CK(expr)                                                                                  \
  do {                                                                                             \
    cudaError_t _e = (expr);                                                                       \
    if (_e != cudaSuccess) {                                                                       \
      fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(_e)); \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

static float to_f32(mlops_adamw_dtype dt, const void *raw, size_t i) {
  switch (dt) {
    case MLOPS_ADAMW_BF16: return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(raw)[i]);
    case MLOPS_ADAMW_FP16: return __half2float(reinterpret_cast<const __half *>(raw)[i]);
    default: return reinterpret_cast<const float *>(raw)[i];
  }
}
static void from_f32(mlops_adamw_dtype dt, void *raw, size_t i, float v) {
  switch (dt) {
    case MLOPS_ADAMW_BF16: reinterpret_cast<__nv_bfloat16 *>(raw)[i] = __float2bfloat16(v); break;
    case MLOPS_ADAMW_FP16: reinterpret_cast<__half *>(raw)[i] = __float2half(v); break;
    default: reinterpret_cast<float *>(raw)[i] = v; break;
  }
}
static size_t elem_size(mlops_adamw_dtype dt) { return dt == MLOPS_ADAMW_FP32 ? 4 : 2; }

/* Bit-reproducible with mlops_adamw's kernel (same FMA order), in double for extra margin. */
static void reference(const mlops_adamw_config &cfg, size_t n, std::vector<float> &param_f32,
                       const std::vector<float> &grad_f32, std::vector<float> &m,
                       std::vector<float> &v) {
  double rbc1 = cfg.bias_correction ? 1.0 / (1.0 - pow((double)cfg.beta1, cfg.step)) : 1.0;
  double rbc2 = cfg.bias_correction ? 1.0 / (1.0 - pow((double)cfg.beta2, cfg.step)) : 1.0;
  double gs = cfg.grad_scale == 0.0f ? 1.0 : cfg.grad_scale;
  double lr_wd = (double)cfg.lr * cfg.weight_decay;
  for (size_t i = 0; i < n; i++) {
    double g = grad_f32[i] * gs;
    double mm = (double)cfg.beta1 * m[i] + (1.0 - cfg.beta1) * g;
    double vv = (double)cfg.beta2 * v[i] + (1.0 - cfg.beta2) * g * g;
    m[i] = (float)mm;
    v[i] = (float)vv;
    double u = (mm * rbc1) / (sqrt(vv * rbc2) + cfg.eps);
    double p = param_f32[i];
    if (lr_wd != 0.0) p -= lr_wd * p;
    p -= (double)cfg.lr * u;
    param_f32[i] = (float)p;
  }
}

static const char *dtype_name(mlops_adamw_dtype dt) {
  switch (dt) {
    case MLOPS_ADAMW_BF16: return "bf16";
    case MLOPS_ADAMW_FP16: return "fp16";
    default: return "fp32";
  }
}

static int run_case(mlops_adamw_dtype pdt, mlops_adamw_dtype gdt, bool bias_correction, int step) {
  const size_t n = 1 << 16;
  std::mt19937 rng(42 + step);
  std::uniform_real_distribution<float> pd(-1.0f, 1.0f), gd(-0.1f, 0.1f);

  std::vector<float> param_f32(n), grad_f32(n), m(n, 0.0f), v(n, 0.0f);
  std::vector<char> param_h(n * elem_size(pdt)), grad_h(n * elem_size(gdt));
  for (size_t i = 0; i < n; i++) {
    param_f32[i] = pd(rng);
    grad_f32[i] = gd(rng);
  }
  /* Quantize to the storage dtype first so param_f32/grad_f32 are the *exact* values the kernel
   * reads (bf16/fp16 round-trip is lossy; must match on both sides). */
  for (size_t i = 0; i < n; i++) from_f32(pdt, param_h.data(), i, param_f32[i]);
  for (size_t i = 0; i < n; i++) param_f32[i] = to_f32(pdt, param_h.data(), i);
  for (size_t i = 0; i < n; i++) from_f32(gdt, grad_h.data(), i, grad_f32[i]);
  for (size_t i = 0; i < n; i++) grad_f32[i] = to_f32(gdt, grad_h.data(), i);

  mlops_adamw_config cfg{};
  cfg.lr = 1e-3f; cfg.beta1 = 0.9f; cfg.beta2 = 0.999f; cfg.eps = 1e-8f; cfg.weight_decay = 0.01f;
  cfg.step = step; cfg.bias_correction = bias_correction; cfg.grad_scale = 1.0f;

  std::vector<float> ref_param = param_f32, ref_m = m, ref_v = v;
  reference(cfg, n, ref_param, grad_f32, ref_m, ref_v);

  void *d_param, *d_grad; float *d_m, *d_v;
  CK(cudaMalloc(&d_param, n * elem_size(pdt)));
  CK(cudaMalloc(&d_grad, n * elem_size(gdt)));
  CK(cudaMalloc(&d_m, n * sizeof(float)));
  CK(cudaMalloc(&d_v, n * sizeof(float)));
  CK(cudaMemcpy(d_param, param_h.data(), n * elem_size(pdt), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_grad, grad_h.data(), n * elem_size(gdt), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_m, m.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_v, v.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  mlops_adamw_status st = mlops_adamw(&cfg, n, d_param, pdt, d_grad, gdt, d_m, d_v, 0);
  if (st != MLOPS_ADAMW_OK) {
    fprintf(stderr, "mlops_adamw failed: %s (%s)\n", mlops_adamw_status_string(st), mlops_adamw_last_error());
    return 1;
  }
  CK(cudaDeviceSynchronize());

  std::vector<char> param_out_h(n * elem_size(pdt));
  CK(cudaMemcpy(param_out_h.data(), d_param, n * elem_size(pdt), cudaMemcpyDeviceToHost));

  double max_abs_err = 0.0, max_rel_err = 0.0;
  for (size_t i = 0; i < n; i++) {
    float got = to_f32(pdt, param_out_h.data(), i);
    double err = fabs((double)got - (double)ref_param[i]);
    max_abs_err = std::max(max_abs_err, err);
    max_rel_err = std::max(max_rel_err, err / (fabs((double)ref_param[i]) + 1e-8));
  }
  /* Tolerance: FP32 storage must match to FP32 ULPs; BF16/FP16 storage only needs to match its
   * own rounding grid, so allow one quantization step of slack. */
  double tol = pdt == MLOPS_ADAMW_FP32 ? 1e-5 : (pdt == MLOPS_ADAMW_FP16 ? 2e-3 : 1e-2);
  bool pass = max_abs_err < tol;
  printf("[%s] param=%-4s grad=%-4s bias_corr=%d step=%-3d  max_abs_err=%.3e max_rel_err=%.3e\n",
         pass ? "PASS" : "FAIL", dtype_name(pdt), dtype_name(gdt), bias_correction, step,
         max_abs_err, max_rel_err);

  cudaFree(d_param); cudaFree(d_grad); cudaFree(d_m); cudaFree(d_v);
  return pass ? 0 : 1;
}

int main() {
  mlops_adamw_dtype dtypes[] = {MLOPS_ADAMW_FP32, MLOPS_ADAMW_FP16, MLOPS_ADAMW_BF16};
  int failures = 0;
  for (auto pdt : dtypes)
    for (auto gdt : dtypes)
      for (int bc : {0, 1})
        for (int step : {1, 7})
          failures += run_case(pdt, gdt, bc, step);
  printf(failures == 0 ? "\nall cases passed\n" : "\n%d case(s) FAILED\n", failures);
  return failures == 0 ? 0 : 1;
}
