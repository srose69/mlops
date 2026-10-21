// mlops_adamw_qat: math and layout in include/mlops_adamw_qat.h; formats in qat_codec.cuh, E2M1 / E4M3
// decode and RNE encode from NVFP4EffMA, FP8 decodes from qat_codec.cuh (E5M2EffMA below sm_89, native
// cvt from sm_89). Four threads per 1x16 block, four
// elements each: a warp moves 8 whole blocks with coalesced 2-8 byte accesses per array and a block's sums
// are two shuffles. (Two threads x 8 elements halves the access count but needs 72 registers: 3 thread
// blocks per SM instead of 4, and slower.)
#include "common.cuh"
#include "qat_codec.cuh"
#include <mlops_adamw_qat.h>
#include <algorithm>
#include <cmath>

using namespace qat;

namespace {

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

mlops_adamw_status make_lattice(const nvfp4a_tensor *q, int rows, int cols, Lattice *L) {
  if (!q || !q->data || !q->block_scale)
    return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw_qat: q (data, block_scale) is required");
  if (q->flags & NVFP4A_SCALES_UE5M3)
    return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw_qat: block scales must be E4M3 (the scale lattice is E4M3's)");
  if (rows <= 0 || cols <= 0 || cols % kB)
    return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw_qat: rows > 0 and cols a positive multiple of 16 required");
  L->rows = rows; L->nb = cols / kB; L->by_nb = Divisor::of((uint32_t)L->nb);
  L->data = static_cast<uint8_t *>(q->data);
  L->scale = static_cast<uint8_t *>(q->block_scale);
  L->swizzled = q->flags & NVFP4A_SCALES_SWIZZLED;
  L->sc = (L->nb + 3) / 4 * 4;
  if (!L->swizzled && q->ld_scale && q->ld_scale != L->nb)
    return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw_qat: linear block scales must be dense (ld_scale 0 or cols / 16)");
  L->global = q->global_scale;
  L->amax_global = q->flags & NVFP4A_GLOBAL_AMAX;
  L->per_row = q->flags & NVFP4A_GLOBAL_PER_ROW;
  L->scale_max = q->scale_max ? (float)q->scale_max : 448.0f;
  return MLOPS_ADAMW_OK_();
}

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

template <int GT>
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
  const float sm = nvfp4a::e4m3_value(kb), sigma = mul(sm, gd);
  // a quarter lies on one side of the block center: interpolate toward that side's neighbour
  const bool right = ql >= 2;
  const float sn = nvfp4a::e4m3_value(L.scale[L.sidx(row, right ? min(b + 1, L.nb - 1) : max(b - 1, 0))]);
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
      for (int o = -8; o <= 8; ++o) {  // clamped repeats at the E4M3 ends never win (strict <)
        const int c = min(126, max(1, (int)kb + o));
        // huber(e * sc, sigma) = sc^2 * huber(e, sigma / sc): errors in grid units, one rescale per candidate
        const float sc = mul(nvfp4a::e4m3_value((uint32_t)c), gd), isc = __frcp_rn(sc), dl = mul(sigma, isc);
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
      kn = (uint32_t)min(126, max(1, (int)kb + (S > 0 ? 1 : -1)));
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
    const float ratio = __fdiv_rn(sm, nvfp4a::e4m3_value(kn));
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

// scale bytes are written after every block has read its neighbours' old ones; blocks without a vote
// this step keep theirs
__global__ void qat_commit_scales(Lattice L, Hyper h, const uint8_t *__restrict__ next) {
  const int blk = blockIdx.x * blockDim.x + threadIdx.x;
  if (blk < L.rows * L.nb && votes(h, blk)) L.scale[L.sidx(blk / L.nb, blk % L.nb)] = next[blk];
}

// Headroom: amax scaling puts the tensor's largest block scale at the top of E4M3, where it could
// never grow. Init moves the global up by 2^kHead and every block byte down by 8 * kHead codes (one
// binade = 8 E4M3 codes, exact for normals), so values are unchanged and every block can grow 2^kHead.
constexpr int kHead = 3;

__global__ void qat_init_kernel(Lattice L, const void *__restrict__ w, mlops_adamw_dtype wdt, mlops_adamw_qat_state st) {
  const int blk = blockIdx.x * blockDim.x + threadIdx.x;
  if (blk >= L.rows * L.nb) return;
  const int r = blk / L.nb, b = blk % L.nb;
  const size_t e0 = (size_t)blk * kB;
  const float sv = mul(nvfp4a::e4m3_value(L.scale[L.sidx(r, b)]), pow2(-kHead));  // same sigma under the moved global
  const uint32_t byte = max(1u, nvfp4a::f32_to_e4m3(sv));
  const float sigma = mul(mul(nvfp4a::e4m3_value(byte), L.gdec(r)), pow2(kHead));
  uint32_t lo = 0, hi = 0;
  for (int j = 0; j < kB; ++j) {
    const float un = fminf(fmaxf(roundf(mul(g_inv(__fdiv_rn(mlops_adamw_load_f32(w, wdt, e0 + j), sigma)), 4096.0f)),
                                 -32768.0f), 32767.0f);
    st.u[e0 + j] = (int16_t)un;
    st.r[e0 + j] = 0;
    st.s[e0 + j] = 0;
    st.qbar[e0 + j] = 0;
    const uint32_t nib = pack_code(mul(un, 0x1p-12f));
    if (j < 8) lo |= nib << (4 * j); else hi |= nib << (4 * (j - 8));
  }
  *reinterpret_cast<uint2 *>(L.data + e0 / 2) = make_uint2(lo, hi);
  L.scale[L.sidx(r, b)] = (uint8_t)byte;
  st.s_exp[blk] = 0; st.cusum[blk] = 0; st.hnum[blk] = 0; st.hden[blk] = 0; st.next[blk] = (uint8_t)byte;
}

// the globals' decode factors, once per step (two IEEE divisions each)
__global__ void qat_gdec(Lattice L, float *__restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = L.gdec(i);
}

__global__ void qat_init_global(float *global, int n) {  // after qat_init_kernel has read it
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) global[i] = __fmul_rn(global[i], (float)(1 << kHead));
}

size_t align16(size_t n) { return (n + 15) & ~(size_t)15; }

}  // namespace

extern "C" size_t mlops_adamw_qat_state_bytes(int rows, int cols) {
  const size_t n = (size_t)rows * cols, nb = n / kB;
  return align16(2 * n) + 3 * align16(n) + align16(nb) + 3 * align16(2 * nb) + align16(nb) + align16(4 * (size_t)rows);
}

extern "C" mlops_adamw_status mlops_adamw_qat_state_bind(void *buffer, int rows, int cols, mlops_adamw_qat_state *st) {
  if (!buffer || !st || rows <= 0 || cols <= 0 || cols % kB)
    return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw_qat_state_bind: bad argument");
  if (reinterpret_cast<uintptr_t>(buffer) & 15u)
    return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw_qat_state_bind: buffer must be 16-byte aligned");
  const size_t n = (size_t)rows * cols, nb = n / kB;
  uint8_t *p = static_cast<uint8_t *>(buffer);
  st->u = reinterpret_cast<int16_t *>(p); p += align16(2 * n);
  st->r = p; p += align16(n);
  st->s = p; p += align16(n);
  st->qbar = reinterpret_cast<int8_t *>(p); p += align16(n);
  st->s_exp = reinterpret_cast<int8_t *>(p); p += align16(nb);
  st->cusum = reinterpret_cast<int16_t *>(p); p += align16(2 * nb);
  st->hnum = reinterpret_cast<uint16_t *>(p); p += align16(2 * nb);
  st->hden = reinterpret_cast<uint16_t *>(p); p += align16(2 * nb);
  st->next = p; p += align16(nb);
  st->gdec = reinterpret_cast<float *>(p);
  return MLOPS_ADAMW_OK_();
}

extern "C" mlops_adamw_status mlops_adamw_qat_init(int rows, int cols, const void *w, mlops_adamw_dtype w_dtype,
                                                    const nvfp4a_tensor *q, const nvfp4a_quant_config *qcfg,
                                                    const mlops_adamw_qat_state *st, cudaStream_t stream) {
  Lattice L;
  if (mlops_adamw_status s = make_lattice(q, rows, cols, &L); s != MLOPS_ADAMW_OK) return s;
  if (!w || !st) return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw_qat_init: null argument");
  if (qcfg && qcfg->tile_rows > 1)
    return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw_qat_init: 1x16 blocks only (tile_rows 1)");
  const nvfp4a_dtype dt = w_dtype == MLOPS_ADAMW_BF16 ? NVFP4A_BF16 : w_dtype == MLOPS_ADAMW_FP16 ? NVFP4A_FP16 : NVFP4A_FP32;
  const nvfp4a_tensor src = {const_cast<void *>(w), dt, nullptr, 0, nullptr, nullptr, 0, 0};
  const nvfp4a_status qs = nvfp4a_quantize_ex(&src, cols, rows, cols, q, cols, nullptr, 0, qcfg, stream);
  if (qs != NVFP4A_OK)
    return MLOPS_ADAMW_FAIL(qs == NVFP4A_CUDA_ERROR ? MLOPS_ADAMW_CUDA_ERROR : MLOPS_ADAMW_INVALID_VALUE,
                            "mlops_adamw_qat_init: nvfp4a_quantize_ex: %s (%s)", nvfp4a_status_string(qs), nvfp4a_last_error());
  const int nblk = rows * (cols / kB);
  qat_init_kernel<<<(nblk + 127) / 128, 128, 0, stream>>>(L, w, w_dtype, *st);
  if (q->global_scale) {
    const int ng = L.per_row ? rows : 1;
    qat_init_global<<<(ng + 255) / 256, 256, 0, stream>>>(const_cast<float *>(q->global_scale), ng);
  }
  MLOPS_ADAMW_CUDA_CHECK(cudaGetLastError());
  return MLOPS_ADAMW_OK_();
}

extern "C" mlops_adamw_status mlops_adamw_qat(const mlops_adamw_qat_config *cfg, int rows, int cols, const void *grad,
                                               mlops_adamw_dtype grad_dtype, const nvfp4a_tensor *q,
                                               const mlops_adamw_qat_state *st, cudaStream_t stream) {
  Lattice L;
  if (mlops_adamw_status s = make_lattice(q, rows, cols, &L); s != MLOPS_ADAMW_OK) return s;
  if (!cfg || !grad || !st) return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw_qat: null argument");
  if (cfg->step < 1) return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw_qat: step must be >= 1");
  if (grad_dtype != MLOPS_ADAMW_E5M2 && grad_dtype != MLOPS_ADAMW_E4M3 && grad_dtype != MLOPS_ADAMW_FP32)
    return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw_qat: grad must be E5M2, E4M3 or FP32");
  if (reinterpret_cast<uintptr_t>(grad) & 15u)
    return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw_qat: grad must be 16-byte aligned");
  const double b1 = cfg->beta1, b2 = cfg->beta2, t = cfg->step;
  const double z = cfg->scale_z > 0 ? cfg->scale_z : 3.0;
  Hyper h;
  h.period = std::max(1, (int)std::lround((1 + b1) / (1 - b1)));
  if (4 * z * z + h.period >= 127)
    return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw_qat: 4 scale_z^2 + (1 + beta1) / (1 - beta1) must stay below 127 (Q7.8 CUSUM)");
  h.lr = cfg->lr; h.b1 = (float)b1; h.omb1 = (float)(1 - b1); h.b2 = (float)b2; h.omb2 = (float)(1 - b2);
  h.eps2 = (float)((double)cfg->eps * cfg->eps);
  h.bc = (float)(sqrt(1 - pow(b2, t)) / (1 - pow(b1, t)));
  h.wd = cfg->weight_decay;
  h.gscale = cfg->grad_scale == 0.0f ? 1.0f : cfg->grad_scale;
  h.rbc1 = (float)(1 / (1 - pow(b1, t)));
  h.rbc1p = cfg->step > 1 ? (float)(1 / (1 - pow(b1, t - 1))) : 0.0f;
  h.phase = cfg->step % h.period;
  h.by_period = Divisor::of((uint32_t)h.period);
  h.hq = (int)std::lround(4 * z * z * 256);
  h.sig = (float)(z * z * (1 - b2) / (1 + b2) * (1 + pow(b2, t)) / (1 - pow(b2, t)));
  h.key = step_key(cfg->seed, cfg->step);
  const int nblk = rows * (cols / kB);
  const int grid = (int)(((size_t)nblk * 4 + 255) / 256), ng = L.per_row ? rows : 1;
  qat_gdec<<<(ng + 255) / 256, 256, 0, stream>>>(L, st->gdec, ng);
  switch (grad_dtype) {
    case MLOPS_ADAMW_E5M2: qat_step_kernel<MLOPS_ADAMW_E5M2><<<grid, 256, 0, stream>>>(L, h, grad, *st); break;
    case MLOPS_ADAMW_E4M3: qat_step_kernel<MLOPS_ADAMW_E4M3><<<grid, 256, 0, stream>>>(L, h, grad, *st); break;
    default: qat_step_kernel<MLOPS_ADAMW_FP32><<<grid, 256, 0, stream>>>(L, h, grad, *st); break;
  }
  qat_commit_scales<<<(nblk + 255) / 256, 256, 0, stream>>>(L, h, st->next);
  MLOPS_ADAMW_CUDA_CHECK(cudaGetLastError());
  return MLOPS_ADAMW_OK_();
}
