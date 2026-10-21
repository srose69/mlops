// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_adamw_qat kernels: math and layout in include/mlops_adamw_qat.h; formats in qat_codec.cuh, E2M1 / E4M3
// decode and RNE encode from NVFP4EffMA, FP8 decodes from qat_codec.cuh (E5M2EffMA below sm_89, native
// cvt from sm_89). Four threads per 1x16 block, four elements each: a warp moves 8 whole blocks with coalesced 2-8 byte accesses per array and a block's sums
// are two shuffles. (Two threads x 8 elements halves the access count but needs 72 registers: 3 thread
// blocks per SM instead of 4, and slower.)
//
// The step kernel has one instance per gradient type, each in its own translation unit (adamw_qat_inst.cu),
// so a multi-architecture build compiles them in parallel; the host side is adamw_qat.cu.
#pragma once
#include "common.cuh"
#include "qat_codec.cuh"
#include <mlops_adamw_qat.h>
#include <cmath>

namespace mlops_qat_detail {

using namespace qat;

constexpr int kB = 16, kQ = 4;  // block size, elements per thread (4 threads per block)

// n / d for n < 2^31 without an integer division (Granlund-Montgomery): t = umulhi(m, n),
// q = (t + ((n - t) >> 1)) >> (l - 1), l = ceil(log2 d), m = 2^32 (2^l - d) / d + 1; d = 1 passes through
struct Divisor {
  uint32_t d, m; int l;
  static Divisor of(uint32_t d) {
    int l = 0;
    while ((1ull << l) < d) ++l;
    return {d, (uint32_t)(((1ull << 32) * ((1ull << l) - d)) / d + 1), l};
  }
  __device__ __forceinline__ uint32_t div(uint32_t n) const {
    if (l == 0) return n;
    const uint32_t t = __umulhi(m, n);
    return (t + ((n - t) >> 1)) >> (l - 1);
  }
};

struct Lattice {  // the NVFP4 tensor's rowwise scale layout and global
  int rows, nb;
  Divisor by_nb;
  uint8_t *data, *scale;
  bool swizzled;
  int sc;  // swizzled: padded scale columns
  const float *global;
  bool amax_global, per_row;
  float scale_max;
  __device__ size_t sidx(int r, int b) const {
    return swizzled ? nvfp4a::swz_row(r, sc) + nvfp4a::swz_col(b) : (size_t)r * nb + b;
  }
  __device__ float gdec(int r) const {  // global value -> decode factor, as nvfp4a_quantize_ex encodes it
    const float g = global ? global[per_row ? r : 0] : 1.0f;
    if (!amax_global) return g;
    const float s_enc = fminf(__fdiv_rn(scale_max * 6.0f, g), 3.40282347e38f);
    return (g == 0.0f || s_enc == 0.0f) ? 1.0f : __fdiv_rn(1.0f, s_enc);
  }
};

mlops_adamw_status make_lattice(const nvfp4a_tensor *q, int rows, int cols, Lattice *L);  // adamw_qat.cu

struct Hyper {
  float lr, b1, omb1, b2, omb2, eps2, bc, wd, gscale;
  float rbc1, rbc1p;  // 1 / (1 - beta1^t), 1 / (1 - beta1^(t-1)) (0 at t = 1: no previous moments)
  int period, phase;  // scale votes every `period` steps; this step's phase t % period
  Divisor by_period;
  int hq;             // CUSUM threshold 4 z^2, Q7.8
  float sig;          // z^2 (1 - b2) / (1 + b2) * (1 + b2^t) / (1 - b2^t): null variance factor of the secant numerator
  uint32_t key;
};

// a block votes on its scale once per `period` steps, all blocks of a thread block (64) on the same step:
// the vote is the expensive branch, and a thread block holds its slot until its slowest warp is done
__device__ __forceinline__ bool votes(const Hyper &h, int blk) {
  const uint32_t c = (uint32_t)h.phase + ((uint32_t)blk >> 6);
  return c == h.by_period.div(c) * (uint32_t)h.period;
}

__device__ __forceinline__ uint32_t pack_code(float u) {  // round(u), +-7, sign-magnitude nibble
  const float k = fminf(fabsf(roundf(u)), 7.0f);
  return (k == 0.0f ? 0u : (u < 0.0f ? 8u : 0u)) | (uint32_t)k;
}

__device__ __forceinline__ float huber(float e, float d) {
  const float a = fabsf(e);
  return a <= d ? mul(mul(0.5f, a), a) : mul(d, fmar(-0.5f, d, a));
}

__device__ __forceinline__ float quad_sum(float v, unsigned m) {
  v = add(v, __shfl_xor_sync(m, v, 1));
  return add(v, __shfl_xor_sync(m, v, 2));
}

// 4 consecutive gradient values times grad_scale: FP8 (qat::dec_fp8x4) from one 32-bit load, FP32 from
// one 128-bit load
template <int GT> __device__ __forceinline__ void load_grad(const void *__restrict__ grad, size_t e, float gs, float g[kQ]) {
  if (GT == MLOPS_ADAMW_FP32) {
    const float4 f = __ldg(reinterpret_cast<const float4 *>(grad) + e / 4);
    g[0] = f.x; g[1] = f.y; g[2] = f.z; g[3] = f.w;
  } else {
    const uint32_t w = __ldg(reinterpret_cast<const unsigned *>(static_cast<const uint8_t *>(grad) + e));
    dec_fp8x4<GT == MLOPS_ADAMW_E5M2>(w, g[0], g[1], g[2], g[3]);
  }
#pragma unroll
  for (int i = 0; i < kQ; ++i) g[i] = mul(g[i], gs);
}

// the largest finite block-scale code: E4M3 0x7E (448), UE5M3 0xFE (114688); the smallest kept is 1
template <bool UE> constexpr int kMaxCode = UE ? 0xFE : 0x7E;

template <int GT, bool UE>
__global__ __launch_bounds__(256) void qat_step_kernel(Lattice L, Hyper h, const void *__restrict__ grad,
                                                       mlops_adamw_qat_state st) {
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int blk = (int)(tid >> 2), ql = (int)(tid & 3);  // ql: quarter of the block, elements 4 ql .. 4 ql + 3
  if (blk >= L.rows * L.nb) return;  // whole quads only: the grid covers 4 threads per block
  const unsigned qm = 0xFu << (threadIdx.x & 28u);
  const int row = (int)L.by_nb.div((uint32_t)blk), b = blk - row * L.nb;
  const size_t e0 = (size_t)blk * kB + kQ * ql;
  const float gd = st.gdec[L.per_row ? row : 0];
  const uint32_t kb = L.scale[L.sidx(row, b)];
  const float sm = nvfp4a::scale_val<UE>(kb), sigma = mul(sm, gd);
  // a quarter lies on one side of the block center: interpolate toward that side's neighbour
  const bool right = ql >= 2;
  const float sn = nvfp4a::scale_val<UE>(L.scale[L.sidx(row, right ? min(b + 1, L.nb - 1) : max(b - 1, 0))]);
  // 1 / sigma~: the reciprocal of the block scale interpolated between the two block centers (the step
  // multiplier is what the Jacobian enters as, so it is the quantity smoothed)
  const float im = __frcp_rn(sigma), in_ = __frcp_rn(mul(sn, gd));
  const float i_lo = right ? im : in_, i_hi = right ? in_ : im;

  float g[kQ];
  load_grad<GT>(grad, e0, h.gscale, g);
  const uint32_t cw = *reinterpret_cast<const uint16_t *>(L.data + e0 / 2);
  const uint2 uw = *reinterpret_cast<const uint2 *>(st.u + e0);
  const uint32_t rw = *reinterpret_cast<const uint32_t *>(st.r + e0), sw = *reinterpret_cast<const uint32_t *>(st.s + e0),
                 qw = *reinterpret_cast<const uint32_t *>(st.qbar + e0);
  const float p_old = pow2(st.s_exp[blk]);
  float r_dec[kQ], s_dec[kQ];
  dec_fp8x4<false>(rw, r_dec[0], r_dec[1], r_dec[2], r_dec[3]);
  dec_fp8x4<true>(sw, s_dec[0], s_dec[1], s_dec[2], s_dec[3]);
  const uint32_t u_words[2] = {uw.x, uw.y};

  float v[kQ], x[kQ], rn[kQ], uf[kQ], qb[kQ];
  float sum_ed = 0.0f, sum_dd = 0.0f, sum_v = 0.0f, xmax = 0.0f;
#pragma unroll
  for (int i = 0; i < kQ; ++i) {
    const float qn = nvfp4a::e2m1_value<0>(cw, i);  // q / sigma
    const float u = mul((float)(int16_t)(u_words[i >> 1] >> (16 * (i & 1))), 0x1p-12f);
    const float s_old = mul(s_dec[i], p_old), r_old = r_dec[i];
    const float qb_old = mul((float)(int8_t)((qw >> (8 * i)) & 255u), 0.0625f);
    const float m_old = mul(r_old, s_old);
    // secant regression of grad on d = q - qbar_hat(t-1): E[grad | past] = H (q - w*), and grad is the
    // only term independent of the moves (m(t-1) drove them: centering on it biases H upward)
    const float d = mul(sigma, fmar(-qb_old, h.rbc1p, qn));
    sum_ed = fmar(g[i], d, sum_ed);
    sum_dd = fmar(d, d, sum_dd);
    v[i] = fmar(h.b2, mul(s_old, s_old), mul(h.omb2, mul(g[i], g[i])));
    const float y = rsqrt_quake(add(v[i], h.eps2));
    x[i] = mul(v[i], y);
    rn[i] = mul(fmar(h.b1, m_old, mul(h.omb1, g[i])), y);
    xmax = fmaxf(xmax, x[i]);
    sum_v = add(sum_v, v[i]);
    qb[i] = fmar(h.b1, qb_old, mul(h.omb1, qn));
    // AdamW's step in weight units (decoupled decay on w = sigma * g(u)) over J = sigma~ * g'(u); 1 / g' exact
    const int j = kQ * ql + i;
    const float f = j < 8 ? (j + 8.5f) * 0.0625f : (j - 7.5f) * 0.0625f;
    const float inv_sig = fmar(f, sub(i_hi, i_lo), i_lo);
    const float au = fabsf(u), inv_slope = au < 4.0f ? 2.0f : au < 6.0f ? 1.0f : 0.5f;
    const float step = mul(h.lr, fmar(rn[i], h.bc, mul(h.wd, mul(sigma, g_of(u)))));
    uf[i] = sub(u, mul(mul(step, inv_slope), inv_sig));
  }
  sum_ed = quad_sum(sum_ed, qm);
  sum_dd = quad_sum(sum_dd, qm);
  sum_v = quad_sum(sum_v, qm);
  xmax = fmaxf(xmax, __shfl_xor_sync(qm, xmax, 1));
  xmax = fmaxf(xmax, __shfl_xor_sync(qm, xmax, 2));

  // s: per-block power-of-two exponent so the block's largest s sits at <= 2^12 (E5M2 max 57344)
  const int E = xmax > 0.0f ? max(-100, min(100, expo(xmax) + 1 - 12)) : 0;
  const float pi = pow2(-E);
  Draws dr[kQ];
  uint32_t r_out = 0, s_out = 0, rng = mix(h.key ^ (uint32_t)tid);
#pragma unroll
  for (int i = 0; i < kQ; ++i) {
    dr[i] = draws(rng);
    s_out |= sr_e5m2_sq(mul(x[i], pi), mul(mul(v[i], pi), pi), dr[i].s) << (8 * i);
    r_out |= sr_e4m3(rn[i], dr[i].r) << (8 * i);
  }
  *reinterpret_cast<uint32_t *>(st.s + e0) = s_out;
  *reinterpret_cast<uint32_t *>(st.r + e0) = r_out;

  // block scale: secant curvature, Newton estimate of each element's optimum, window argmin, CUSUM.
  // All four threads compute the same block values (symmetric shuffles); quarter 0 stores them.
  const float num = fmar(h.b2, dec_bf16(st.hnum[blk]), mul(h.omb2, sum_ed));
  const float den = fmar(h.b2, dec_bf16(st.hden[blk]), mul(h.omb2, sum_dd));
  uint32_t kn = kb;
  if (votes(h, blk)) {
    int vote = 0;
    // H = 0 makes the numerator a sum of independent e d terms with E[e^2] = v: vote only on a curvature
    // z null standard deviations above zero (else m / H is noise over noise)
    if (num > 0.0f && den > 0.0f && mul(num, num) > mul(mul(h.sig, mul(sum_v, 0.0625f)), den)) {
      const float inv_h = __fdiv_rn(den, num);
      float ws[kQ];
#pragma unroll
      for (int i = 0; i < kQ; ++i)  // |w*| = |qbar_hat * sigma - m_hat / H|
        ws[i] = fabsf(mul(fmar(qb[i], sigma, -mul(mul(rn[i], x[i]), inv_h)), h.rbc1));
      float best = 3.40282347e38f;
      int kbest = (int)kb;
#pragma unroll
      for (int o = -8; o <= 8; ++o) {  // clamped repeats at the code range's ends never win (strict <)
        const int c = min(kMaxCode<UE>, max(1, (int)kb + o));
        // huber(e * sc, sigma) = sc^2 * huber(e, sigma / sc): errors in grid units, one rescale per candidate
        const float sc = mul(nvfp4a::scale_val<UE>((uint32_t)c), gd), isc = __frcp_rn(sc), dl = mul(sigma, isc);
        float D = 0.0f;
#pragma unroll
        for (int i = 0; i < kQ; ++i) {
          const float a = mul(ws[i], isc);
          D = add(D, huber(sub(a, e2m1_round(a)), dl));
        }
        D = mul(quad_sum(D, qm), mul(sc, sc));
        if (D < best) { best = D; kbest = c; }
      }
      vote = (kbest > (int)kb) - (kbest < (int)kb);
    }
    // the vote holds for `period` steps: period CUSUM updates at once
    int S = (int)st.cusum[blk] + 256 * h.period * vote;
    const int mag = abs(S) - 16 * h.period;
    S = mag > 0 ? (S > 0 ? mag : -mag) : 0;
    if (abs(S) > h.hq) {
      kn = (uint32_t)min(kMaxCode<UE>, max(1, (int)kb + (S > 0 ? 1 : -1)));
      S = 0;
    }
    if (ql == 0) { st.cusum[blk] = (int16_t)S; st.next[blk] = (uint8_t)kn; }
  }
  if (ql == 0) {
    const uint32_t w = xorshift(rng ^ 0xB10C5CA1u);
    st.hnum[blk] = (uint16_t)sr_bf16(num, uniform24(w));
    st.hden[blk] = (uint16_t)sr_bf16(den, uniform24(xorshift(w)));
    st.s_exp[blk] = (int8_t)E;
  }

  // a move keeps the master values: re-express u and qbar against the new sigma
  if (kn != kb) {
    const float ratio = __fdiv_rn(sm, nvfp4a::scale_val<UE>(kn));
#pragma unroll
    for (int i = 0; i < kQ; ++i) {
      uf[i] = g_inv(mul(g_of(uf[i]), ratio));
      qb[i] = mul(qb[i], ratio);
    }
  }
  uint32_t codes = 0, qb_out = 0, u_out[2] = {0, 0};
#pragma unroll
  for (int i = 0; i < kQ; ++i) {
    const float un = sr_fixed(uf[i], 4096.0f, -32768.0f, 32767.0f, dr[i].u);
    const float qq = sr_fixed(qb[i], 16.0f, -128.0f, 127.0f, dr[i].qbar);
    codes |= pack_code(mul(un, 0x1p-12f)) << (4 * i);
    u_out[i >> 1] |= (uint32_t)(uint16_t)(int16_t)un << (16 * (i & 1));
    qb_out |= (uint32_t)(uint8_t)(int8_t)qq << (8 * i);
  }
  *reinterpret_cast<uint2 *>(st.u + e0) = make_uint2(u_out[0], u_out[1]);
  *reinterpret_cast<uint32_t *>(st.qbar + e0) = qb_out;
  *reinterpret_cast<uint16_t *>(L.data + e0 / 2) = (uint16_t)codes;
}

// qat_step_kernel<GT, UE> over `grid` blocks of 256 threads (adamw_qat_inst.cu: one (GT, UE) per object)
template <int GT, bool UE>
void launch_step(const Lattice &L, const Hyper &h, const void *grad, const mlops_adamw_qat_state &st, int grid,
                 cudaStream_t stream);

}  // namespace mlops_qat_detail
