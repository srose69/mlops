// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
#ifndef MLOPS_ADAMW_COMMON_CUH
#define MLOPS_ADAMW_COMMON_CUH

#include "mlops_adamw.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdarg>
#include <cstdint>

namespace mlops_adamw_detail {
inline char *last_error_buf() {
  static thread_local char buf[256] = {0};
  return buf;
}
inline mlops_adamw_status fail(mlops_adamw_status st, const char *fmt, ...) {
  char *buf = last_error_buf();
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf) - 1, fmt, ap);
  va_end(ap);
  return st;
}
inline mlops_adamw_status ok() {
  last_error_buf()[0] = '\0';
  return MLOPS_ADAMW_OK;
}
} // namespace mlops_adamw_detail

#define MLOPS_ADAMW_FAIL(st, ...) mlops_adamw_detail::fail((st), __VA_ARGS__)
#define MLOPS_ADAMW_OK_() mlops_adamw_detail::ok()

#define MLOPS_ADAMW_CUDA_CHECK(expr)                                                             \
  do {                                                                                            \
    cudaError_t _e = (expr);                                                                      \
    if (_e != cudaSuccess)                                                                        \
      return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_CUDA_ERROR, "%s: %s", #expr, cudaGetErrorString(_e));   \
  } while (0)

/* FP32 bits -> BF16 bits in the high half, round-to-nearest-even; NaN -> canonical 0x7FC0.
 * u + 0x7FFF + lsb(bit 16), with the LSB moved into the carry flag (t + 0xFFFF0000 carries iff
 * t == 0x10000): LOP + IADD.CC + IADD.X, all full rate. Lifted from E4M3EffMA's rne_hi. */
__device__ __forceinline__ uint32_t mlops_bf16_rne_hi(uint32_t u) {
  u = (u & 0x7FFFFFFFu) > 0x7F800000u ? 0x7FC00000u : u;
  uint32_t t = u & 0x10000u, r;
  asm("add.cc.u32 %1, %1, 0xFFFF0000;\n\taddc.u32 %0, %2, 0x7FFF;" : "=r"(r), "+r"(t) : "r"(u));
  return r;
}

__device__ __forceinline__ float mlops_adamw_load_f32(const void *p, mlops_adamw_dtype dt, size_t i) {
  switch (dt) {
    case MLOPS_ADAMW_BF16: return __uint_as_float((uint32_t)reinterpret_cast<const uint16_t *>(p)[i] << 16);
    case MLOPS_ADAMW_FP16: return __half2float(reinterpret_cast<const __half *>(p)[i]);
    default:               return reinterpret_cast<const float *>(p)[i];
  }
}

__device__ __forceinline__ void mlops_adamw_store_f32(void *p, mlops_adamw_dtype dt, size_t i, float v) {
  switch (dt) {
    case MLOPS_ADAMW_BF16: reinterpret_cast<uint16_t *>(p)[i] = (uint16_t)(mlops_bf16_rne_hi(__float_as_uint(v)) >> 16); break;
    case MLOPS_ADAMW_FP16: reinterpret_cast<__half *>(p)[i] = __float2half(v); break;
    default:               reinterpret_cast<float *>(p)[i] = v; break;
  }
}

#endif /* MLOPS_ADAMW_COMMON_CUH */
