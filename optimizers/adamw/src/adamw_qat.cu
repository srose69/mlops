// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_adamw_qat host side (include/mlops_adamw_qat.h): argument checks, the per-step constants, state
// layout, initialization from a quantized weight and the scale commit. The step kernel instances are in
// adamw_qat_inst.cu (adamw_qat_kernels.cuh); mlops_adamw_qat_init, which quantizes with NVFP4EffMA, is in
// adamw_qat_init.cu so that only programs calling it link NVFP4EffMA.
#include "adamw_qat_kernels.cuh"
#include <algorithm>

namespace mlops_qat_detail {

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

}  // namespace mlops_qat_detail

using namespace mlops_qat_detail;

namespace {

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

extern "C" mlops_adamw_status mlops_adamw_qat_init_quantized(int rows, int cols, const void *w, mlops_adamw_dtype w_dtype,
                                                              const nvfp4a_tensor *q, const mlops_adamw_qat_state *st,
                                                              cudaStream_t stream) {
  Lattice L;
  if (mlops_adamw_status s = make_lattice(q, rows, cols, &L); s != MLOPS_ADAMW_OK) return s;
  if (!w || !st) return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw_qat_init_quantized: null argument");
  if (w_dtype != MLOPS_ADAMW_BF16 && w_dtype != MLOPS_ADAMW_FP16 && w_dtype != MLOPS_ADAMW_FP32)
    return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw_qat_init_quantized: w must be BF16, FP16 or FP32");
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
    case MLOPS_ADAMW_E5M2: launch_step<MLOPS_ADAMW_E5M2>(L, h, grad, *st, grid, stream); break;
    case MLOPS_ADAMW_E4M3: launch_step<MLOPS_ADAMW_E4M3>(L, h, grad, *st, grid, stream); break;
    default: launch_step<MLOPS_ADAMW_FP32>(L, h, grad, *st, grid, stream); break;
  }
  qat_commit_scales<<<(nblk + 255) / 256, 256, 0, stream>>>(L, h, st->next);
  MLOPS_ADAMW_CUDA_CHECK(cudaGetLastError());
  return MLOPS_ADAMW_OK_();
}
