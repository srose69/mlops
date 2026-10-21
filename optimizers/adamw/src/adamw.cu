// mlops_adamw: fused in-place AdamW step. Operation order: include/mlops_adamw.h.
#include "common.cuh"
#include <algorithm>

namespace {

struct AdamArgs { float lr, omb1, omb2, rbc1, rbc2, eps, lr_wd, gscale; };

// One element, in the documented order (include/mlops_adamw.h); p, m, v updated in place.
// __f*_rn / __fmaf_rn / __fsqrt_rn pin IEEE round-to-nearest regardless of -use_fast_math, which
// would otherwise swap the division and sqrt for fast approximations and break the header's
// bit-reproducibility claim.
__device__ __forceinline__ void adam_elem(const AdamArgs &a, float g, float &p, float &m, float &v) {
  g = __fmul_rn(g, a.gscale);
  m = __fmaf_rn(a.omb1, g - m, m);
  v = __fmaf_rn(a.omb2, __fmul_rn(g, g) - v, v);
  const float u = __fdiv_rn(__fmul_rn(m, a.rbc1), __fadd_rn(__fsqrt_rn(__fmul_rn(v, a.rbc2)), a.eps));
  if (a.lr_wd != 0.0f) p = __fmaf_rn(-a.lr_wd, p, p);
  p = __fmaf_rn(-a.lr, u, p);
}

// load/store 4 contiguous elements of dtype DT as one 128-bit (FP32) or 64-bit (BF16/FP16) transaction.
template <mlops_adamw_dtype DT> struct quad4;

template <> struct quad4<MLOPS_ADAMW_FP32> {
  static __device__ __forceinline__ float4 load(const void *p, size_t i4) { return reinterpret_cast<const float4 *>(p)[i4]; }
  static __device__ __forceinline__ void store(void *p, size_t i4, float4 v) { reinterpret_cast<float4 *>(p)[i4] = v; }
};

// FP16 <-> FP32 is a genuine format conversion (different exponent bias/width): __half2float /
// __float2half compile to cvt.f32.f16 / cvt.rn.f16.f32 on every arch, so there's nothing to hand-roll.
template <> struct quad4<MLOPS_ADAMW_FP16> {
  static __device__ __forceinline__ float4 load(const void *p, size_t i4) {
    const uint2 w = reinterpret_cast<const uint2 *>(p)[i4];  // 4x fp16 = 8 bytes
    return make_float4(__half2float(__ushort_as_half((unsigned short)w.x)), __half2float(__ushort_as_half((unsigned short)(w.x >> 16))),
                        __half2float(__ushort_as_half((unsigned short)w.y)), __half2float(__ushort_as_half((unsigned short)(w.y >> 16))));
  }
  static __device__ __forceinline__ void store(void *p, size_t i4, float4 v) {
    const unsigned short h0 = __half_as_ushort(__float2half_rn(v.x)), h1 = __half_as_ushort(__float2half_rn(v.y)),
                          h2 = __half_as_ushort(__float2half_rn(v.z)), h3 = __half_as_ushort(__float2half_rn(v.w));
    reinterpret_cast<uint2 *>(p)[i4] = make_uint2((uint32_t)h0 | (uint32_t)h1 << 16, (uint32_t)h2 | (uint32_t)h3 << 16);
  }
};

// BF16 is the top 16 bits of FP32: decode is a free widen, encode is RNE truncation (mlops_bf16_rne_hi).
// Packing within a u32: even lane low, odd lane high (matches E4M3EffMA's load_any/ld8 convention).
template <> struct quad4<MLOPS_ADAMW_BF16> {
  static __device__ __forceinline__ float4 load(const void *p, size_t i4) {
    const uint2 w = reinterpret_cast<const uint2 *>(p)[i4];
    return make_float4(__uint_as_float((w.x & 0xFFFFu) << 16), __uint_as_float(w.x & 0xFFFF0000u),
                        __uint_as_float((w.y & 0xFFFFu) << 16), __uint_as_float(w.y & 0xFFFF0000u));
  }
  static __device__ __forceinline__ void store(void *p, size_t i4, float4 v) {
    const uint32_t x = (mlops_bf16_rne_hi(__float_as_uint(v.x)) >> 16) | (mlops_bf16_rne_hi(__float_as_uint(v.y)) & 0xFFFF0000u);
    const uint32_t y = (mlops_bf16_rne_hi(__float_as_uint(v.z)) >> 16) | (mlops_bf16_rne_hi(__float_as_uint(v.w)) & 0xFFFF0000u);
    reinterpret_cast<uint2 *>(p)[i4] = make_uint2(x, y);
  }
};

// PDT/GDT resolved at compile time (9 instantiations picked by the host switch in mlops_adamw):
// no per-lane dtype branch. Grid-stride over groups of 4, grid capped so huge `count` doesn't launch
// an unbounded number of blocks (same cap as E4M3EffMA/NVFP4EffMA's elementwise kernels).
template <mlops_adamw_dtype PDT, mlops_adamw_dtype GDT>
__global__ __launch_bounds__(256) void adamw_kernel_vec4(AdamArgs a, size_t count4, void *__restrict__ param,
                                                          const void *__restrict__ grad,
                                                          float4 *__restrict__ exp_avg,
                                                          float4 *__restrict__ exp_avg_sq) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count4; i += stride) {
    float4 g4 = quad4<GDT>::load(grad, i), p4 = quad4<PDT>::load(param, i), m4 = exp_avg[i], v4 = exp_avg_sq[i];
    float *g = reinterpret_cast<float *>(&g4), *p = reinterpret_cast<float *>(&p4);
    float *m = reinterpret_cast<float *>(&m4), *v = reinterpret_cast<float *>(&v4);
#pragma unroll
    for (int k = 0; k < 4; ++k) adam_elem(a, g[k], p[k], m[k], v[k]);
    exp_avg[i] = m4;
    exp_avg_sq[i] = v4;
    quad4<PDT>::store(param, i, p4);
  }
}

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

constexpr int kThreads = 256, kMaxBlocks = 8192;
inline int grid_for(size_t work) { return (int)std::min<size_t>((work + kThreads - 1) / kThreads, kMaxBlocks); }

template <mlops_adamw_dtype PDT, mlops_adamw_dtype GDT>
void launch(const AdamArgs &a, size_t count, void *param, const void *grad, float *exp_avg, float *exp_avg_sq,
           cudaStream_t stream) {
  const size_t count4 = count / 4, tail = count - count4 * 4;
  if (count4 > 0)
    adamw_kernel_vec4<PDT, GDT><<<grid_for(count4), kThreads, 0, stream>>>(
        a, count4, param, grad, reinterpret_cast<float4 *>(exp_avg), reinterpret_cast<float4 *>(exp_avg_sq));
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
