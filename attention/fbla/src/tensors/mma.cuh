// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// Warp-level instructions of the tensor-core prefill: ldmatrix, integer mma, barriers (sm_75+); BF16 mma, cp.async
// (sm_80+).  A wrapper an architecture lacks is not declared in its device pass.
//
// Fragments (PTX ISA, mma.m16n8k32 / m16n8k16), g = lane / 4, t = lane % 4:
//   s8 / u8 m16n8k32  A0 (row g, k 4t ..), A1 (row g + 8), A2 / A3 at k 16 + 4t; B0 (k 4t .., col g), B1 (k 16 + 4t);
//                     C c0, c1 (row g, cols 2t, 2t + 1), c2, c3 (row g + 8).  sm_75: four m8n8k16 on the same registers.
//   bf16 m16n8k16     A0 (row g, k 2t, 2t + 1), A1 (row g + 8), A2 / A3 at k 8 + 2t; B0 (k 2t, col g), B1 (k 8 + 2t)
// mma accumulate in place, or start from c in all four accumulators (_c).  ldmatrix carries a memory clobber, the
// barrier wrappers are compiler fences.
#ifndef MLOPS_FBLA_TENSORS_MMA_CUH
#define MLOPS_FBLA_TENSORS_MMA_CUH

#include <cstdint>
#include <utility>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
#define MLOPS_FBLA_TC75 1
#endif
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define MLOPS_FBLA_TC80 1
#endif

namespace mlops_fbla_detail {
namespace tc {

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }

// f(ic<I>) for I = 0 .. N - 1, straight-line
template <int I> struct ic {
  static constexpr int value = I;
  __host__ __device__ constexpr operator int() const { return I; }
};
template <class F, int... I> __device__ __forceinline__ void unroll_seq(F &&f, std::integer_sequence<int, I...>) {
  (f(ic<I>{}), ...);
}
template <int N, class F> __device__ __forceinline__ void unroll(F &&f) { unroll_seq(f, std::make_integer_sequence<int, N>{}); }

#ifdef MLOPS_FBLA_TC75
template <int OFF = 0> __device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4 + %5];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr), "n"(OFF) : "memory");
}
template <int OFF = 0> __device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
  asm("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4 + %5];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr), "n"(OFF) : "memory");
}

#ifdef MLOPS_FBLA_TC80
#define TC_IMMA(AT)                                                                                                  \
  asm("mma.sync.aligned.m16n8k32.row.col.s32." AT ".u8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "            \
      "{%0, %1, %2, %3};"                                                                                            \
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1))
#define TC_IMMA_C(AT)                                                                                                \
  asm("mma.sync.aligned.m16n8k32.row.col.s32." AT ".u8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "            \
      "{%10, %10, %10, %10};"                                                                                        \
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])                                                               \
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c))
#else  // sm_75: c0 c1 = A0 B0 + A2 B1, c2 c3 = A1 B0 + A3 B1 (integer sums: any order)
#define TC_IMMA_K16(AT) "mma.sync.aligned.m8n8k16.row.col.s32." AT ".u8.s32 "
#define TC_IMMA(AT)                                                                                                  \
  asm(TC_IMMA_K16(AT) "{%0, %1}, {%4}, {%8}, {%0, %1};\n\t" TC_IMMA_K16(AT) "{%0, %1}, {%6}, {%9}, {%0, %1};\n\t"      \
      TC_IMMA_K16(AT) "{%2, %3}, {%5}, {%8}, {%2, %3};\n\t" TC_IMMA_K16(AT) "{%2, %3}, {%7}, {%9}, {%2, %3};"           \
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1))
#define TC_IMMA_C(AT) { d[0] = d[1] = d[2] = d[3] = c; TC_IMMA(AT); }  // (no early-clobber output: nvcc 12.0)
#endif
__device__ __forceinline__ void mma_s8u8(uint32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) { TC_IMMA("s8"); }
__device__ __forceinline__ void mma_u8u8(uint32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) { TC_IMMA("u8"); }
__device__ __forceinline__ void mma_s8u8_c(uint32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t c) {
  TC_IMMA_C("s8");
}
__device__ __forceinline__ void mma_u8u8_c(uint32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t c) {
  TC_IMMA_C("u8");
}
#undef TC_IMMA
#undef TC_IMMA_C
#undef TC_IMMA_K16

// Named barriers (barrier.cta, not .aligned: the warps of a handoff use different forms and ids)
template <int N> __device__ __forceinline__ void bar_sync(uint32_t id) {
  asm volatile("" ::: "memory");
  asm volatile("barrier.cta.sync %0, %1;" :: "r"(id), "n"(N));
  asm volatile("" ::: "memory");
}
template <int N> __device__ __forceinline__ void bar_arrive(uint32_t id) {
  asm volatile("" ::: "memory");
  asm volatile("barrier.cta.arrive %0, %1;" :: "r"(id), "n"(N));
}
__device__ __forceinline__ void warp_sync() {  // memory order among the warp's lanes
  asm volatile("" ::: "memory");
  asm volatile("bar.warp.sync -1;");
  asm volatile("" ::: "memory");
}
__device__ __forceinline__ void prefetch_l1(const void *p) { asm volatile("prefetch.global.L1 [%0];" :: "l"(p)); }

// Warp w's READY barrier for group j (slot j & 1, ids 7 (j & 1) .. + 6; owner j & 7).  The other parity's warps keep
// ids w / 2; the slot parity's three non-owners rotate so that each keeps its id across the slot's generations and the
// warp leaving the owner role takes the id of the one entering it (v mod 3 by a multiply-high).  Not for the owner.
__host__ __device__ __forceinline__ uint32_t ready_id(uint32_t w, uint32_t j) {
  const uint32_t p = w >> 1, par = w & 1u;
  const uint32_t v = p - 1u + ((j + 8u - 2u * p - par) >> 3);
#ifdef __CUDA_ARCH__
  const uint32_t q3 = __umulhi(v, 0xAAAAAAABu) >> 1;
#else
  const uint32_t q3 = (uint32_t)(((uint64_t)v * 0xAAAAAAABull) >> 33);
#endif
  const uint32_t rot = 7u * par + 4u + v - 3u * q3;
  const uint32_t fix = 7u * (par ^ 1u) + p;
  return ((j ^ w) & 1u) ? fix : rot;
}
#endif  // MLOPS_FBLA_TC75

#ifdef MLOPS_FBLA_TC80
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// 16 bytes global -> shared (L2 only) at dst + DOFF, src + SOFF; visible to the issuing thread after its wait
template <int DOFF = 0, int SOFF = 0> __device__ __forceinline__ void cp_async16(uint32_t dst, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0 + %2], [%1 + %3], 16;" :: "r"(dst), "l"(src), "n"(DOFF), "n"(SOFF) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
#endif  // MLOPS_FBLA_TC80

}  // namespace tc
}  // namespace mlops_fbla_detail
#endif
