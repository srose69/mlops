// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
/* mlops_rope tests.
 *   1. positions + freq (exact): every output within half an ulp of the dtype (+ the FP32 rounding of the
 *      computation) of a long-double reference, for every dtype, both pair layouts, partial rotary, GQA, the
 *      [B, S, H, D] and [B, H, S, D] layouts, int32 / int64 / implicit positions, positions up to 2^31 - 1 and
 *      negative ones, an attention factor, every vector width (pairs % 4, unaligned tensors); out of place and in
 *      place agree; inverse undoes the rotation.
 *   2. cos / sin tables (Hugging Face): bit for bit the host emulation of apply_rotary_pos_emb's arithmetic and
 *      of its autograd backward (inverse), tables broadcast over the batch.
 *   3. argument checks.
 */
#include "mlops_rope.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdarg>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(e) do { cudaError_t _e = (e); if (_e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #e, cudaGetErrorString(_e)); exit(1); } } while (0)

static int fails = 0;
static void report(bool ok, const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  printf("[%s] ", ok ? "PASS" : "FAIL"); vprintf(fmt, ap); printf("\n");
  va_end(ap);
  fails += !ok;
}

// ---- host dtype codecs -----------------------------------------------------------------------------------
static float bits_f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float round_dt(float v, int dt) {
  if (dt == MLOPS_ROPE_BF16) {
    uint32_t u = f_bits(v);
    u = (u & 0x7FFFFFFFu) > 0x7F800000u ? 0x7FC00000u : u;
    return bits_f((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
  }
  if (dt == MLOPS_ROPE_FP16) return __half2float(__float2half_rn(v));
  return v;
}
static int es_of(int dt) { return dt == MLOPS_ROPE_FP32 ? 4 : 2; }
static float load_dt(const std::vector<uint8_t> &buf, size_t i, int dt) {
  if (dt == MLOPS_ROPE_FP32) { float f; memcpy(&f, &buf[4 * i], 4); return f; }
  uint16_t h; memcpy(&h, &buf[2 * i], 2);
  if (dt == MLOPS_ROPE_BF16) return bits_f((uint32_t)h << 16);
  __half_raw r; r.x = h; return __half2float(__half(r));
}
static void store_dt(std::vector<uint8_t> &buf, size_t i, float v, int dt) {
  v = round_dt(v, dt);
  if (dt == MLOPS_ROPE_FP32) { memcpy(&buf[4 * i], &v, 4); return; }
  uint16_t h = dt == MLOPS_ROPE_BF16 ? (uint16_t)(f_bits(v) >> 16) : __half_raw(__float2half_rn(v)).x;
  memcpy(&buf[2 * i], &h, 2);
}
static double ulp_dt(double x, int dt) {
  const int mant = dt == MLOPS_ROPE_BF16 ? 8 : dt == MLOPS_ROPE_FP16 ? 11 : 24;
  const double a = std::max(fabs(x), dt == MLOPS_ROPE_FP16 ? 6.103515625e-05 : 1.17549435e-38);
  return ldexp(1.0, (int)floor(log2(a)) - (mant - 1));
}

// ---- a [B, S, H, D] tensor in device memory with a chosen layout and offset ---------------------------------
struct Tensor {
  int B, S, H, D, dt;
  bool bhsd;          // memory order [B, H, S, D] (Hugging Face) instead of [B, S, H, D]
  int64_t sb, ss, sh;
  size_t offset, n;   // element offset of [0, 0, 0, 0] (misalignment), elements allocated
  std::vector<uint8_t> host;
  void *dev = nullptr;
  Tensor(int B_, int S_, int H_, int D_, int dt_, bool bhsd_, size_t off) : B(B_), S(S_), H(H_), D(D_), dt(dt_), bhsd(bhsd_), offset(off) {
    if (bhsd) { sh = (int64_t)S * D; ss = D; sb = (int64_t)H * S * D; } else { sh = D; ss = (int64_t)H * D; sb = (int64_t)S * H * D; }
    n = offset + (size_t)B * S * H * D;
    host.assign(n * es_of(dt), 0);
    CK(cudaMalloc(&dev, n * es_of(dt)));
  }
  ~Tensor() { cudaFree(dev); }
  size_t idx(int b, int s, int h, int d) const { return offset + b * sb + s * ss + h * sh + d; }
  void fill(std::mt19937 &rng, float scale) {
    std::normal_distribution<float> N(0, scale);
    for (size_t i = 0; i < n; ++i) store_dt(host, i, N(rng), dt);
  }
  void up() { CK(cudaMemcpy(dev, host.data(), host.size(), cudaMemcpyHostToDevice)); }
  void down() { CK(cudaMemcpy(host.data(), dev, host.size(), cudaMemcpyDeviceToHost)); }
  mlops_rope_tensor desc() const { return {(char *)dev + offset * es_of(dt), H, sb, ss, sh}; }
};

struct Case {
  int dt, B, S, Hq, Hk, D, R;
  bool inter, bhsd, in_place;
  int pos_kind;          // 0: implicit (s + offset), 1: int32, 2: int64
  int64_t pos_base;      // positions around this value (+ random), may be large or negative
  float att;
  size_t misalign;       // element offset of the tensors
};

static std::vector<double> make_inv_freq(int R, double base) {
  std::vector<double> f(R / 2);
  for (int i = 0; i < R / 2; ++i) f[i] = pow(base, -2.0 * i / R);
  return f;
}

// ---- 1. exact mode ---------------------------------------------------------------------------------------
static void test_exact(const Case &c, std::mt19937 &rng) {
  const int P = c.R / 2;
  Tensor q(c.B, c.S, c.Hq, c.D, c.dt, c.bhsd, c.misalign), k(c.B, c.S, c.Hk, c.D, c.dt, c.bhsd, c.misalign);
  Tensor qo(c.B, c.S, c.Hq, c.D, c.dt, c.bhsd, c.misalign), ko(c.B, c.S, c.Hk, c.D, c.dt, c.bhsd, c.misalign);
  q.fill(rng, 1.0f); k.fill(rng, 3.0f); q.up(); k.up();
  std::vector<int64_t> pos((size_t)c.B * c.S);
  std::uniform_int_distribution<int64_t> J(0, 1000);
  for (int b = 0; b < c.B; ++b)
    for (int s = 0; s < c.S; ++s) {
      int64_t m = c.pos_kind == 0 ? s + c.pos_base : c.pos_base + J(rng);
      m = std::max<int64_t>(std::min<int64_t>(m, 0x7FFFFFFF), -0x7FFFFFFFLL);
      pos[(size_t)b * c.S + s] = m;
    }
  void *dpos = nullptr;
  if (c.pos_kind == 1) {
    std::vector<int32_t> p32(pos.begin(), pos.end());
    CK(cudaMalloc(&dpos, p32.size() * 4)); CK(cudaMemcpy(dpos, p32.data(), p32.size() * 4, cudaMemcpyHostToDevice));
  } else if (c.pos_kind == 2) {
    CK(cudaMalloc(&dpos, pos.size() * 8)); CK(cudaMemcpy(dpos, pos.data(), pos.size() * 8, cudaMemcpyHostToDevice));
  }
  const std::vector<double> inv = make_inv_freq(c.R, 10000.0);
  std::vector<float> ft(4 * P);
  mlops_rope_freq(inv.data(), P, ft.data());
  float *dfreq; CK(cudaMalloc(&dfreq, ft.size() * 4)); CK(cudaMemcpy(dfreq, ft.data(), ft.size() * 4, cudaMemcpyHostToDevice));

  mlops_rope_config cfg{};
  cfg.batch = c.B; cfg.seq = c.S; cfg.head_dim = c.D; cfg.rotary_dim = c.R; cfg.interleaved = c.inter;
  cfg.dtype = (mlops_rope_dtype)c.dt; cfg.positions = dpos; cfg.positions_int64 = c.pos_kind == 2;
  cfg.pos_stride_b = c.S; cfg.pos_stride_s = 1; cfg.position_offset = c.pos_base; cfg.freq = dfreq;
  cfg.attention_factor = c.att;
  const mlops_rope_tensor qi = q.desc(), ki = k.desc(), qod = c.in_place ? qi : qo.desc(), kod = c.in_place ? ki : ko.desc();
  const mlops_rope_status st = mlops_rope(&cfg, &qi, &qod, &ki, &kod, 0);
  CK(cudaDeviceSynchronize());
  Tensor &QO = c.in_place ? q : qo, &KO = c.in_place ? k : ko;
  std::vector<uint8_t> qin = q.host, kin = k.host;
  QO.down(); KO.down();

  double worst = 0;  // max error in units of the bound
  long bad = 0;
  const auto check = [&](const Tensor &in_t, const std::vector<uint8_t> &in, const Tensor &out, int H) {
    for (int b = 0; b < c.B; ++b)
      for (int s = 0; s < c.S; ++s) {
        const long double m = (long double)pos[(size_t)b * c.S + s];
        for (int h = 0; h < H; ++h) {
          for (int i = 0; i < P; ++i) {
            const int i1 = c.inter ? 2 * i : i, i2 = c.inter ? 2 * i + 1 : i + P;
            const long double a = m * (long double)inv[i], cs = cosl(a) * c.att, sn = sinl(a) * c.att;
            const double x1 = load_dt(in, in_t.idx(b, s, h, i1), c.dt), x2 = load_dt(in, in_t.idx(b, s, h, i2), c.dt);
            const double r1 = (double)(x1 * cs - x2 * sn), r2 = (double)(x2 * cs + x1 * sn);
            const double slack = 6e-7 * (fabs(x1) + fabs(x2)) * c.att;  // FP32 phase, sincospi and FMA rounding
            const double y1 = load_dt(out.host, out.idx(b, s, h, i1), c.dt), y2 = load_dt(out.host, out.idx(b, s, h, i2), c.dt);
            const double e1 = fabs(y1 - r1) / (0.5 * ulp_dt(r1, c.dt) + slack), e2 = fabs(y2 - r2) / (0.5 * ulp_dt(r2, c.dt) + slack);
            worst = std::max(worst, std::max(e1, e2));
            bad += (e1 > 1) + (e2 > 1);
          }
          for (int d = c.R; d < c.D; ++d)  // pass-through
            bad += load_dt(out.host, out.idx(b, s, h, d), c.dt) != load_dt(in, in_t.idx(b, s, h, d), c.dt);
        }
      }
  };
  check(q, qin, QO, c.Hq);
  check(k, kin, KO, c.Hk);

  // inverse: rotate the output back (out of place into the input buffers' copies)
  double back = 0;
  if (!c.in_place) {
    cfg.inverse = 1;
    const mlops_rope_tensor qb = qo.desc(), kb = ko.desc(), qbo = q.desc(), kbo = k.desc();
    mlops_rope(&cfg, &qb, &qbo, &kb, &kbo, 0);
    CK(cudaDeviceSynchronize());
    q.down();
    // two roundings to the dtype, each within half an ulp of the pair's magnitude: in units of that ulp
    for (int b = 0; b < c.B; ++b)
      for (int s = 0; s < c.S; ++s)
        for (int h = 0; h < c.Hq; ++h)
          for (int i = 0; i < P; ++i) {
            const int i1 = c.inter ? 2 * i : i, i2 = c.inter ? 2 * i + 1 : i + P;
            const double x1 = load_dt(qin, q.idx(b, s, h, i1), c.dt), x2 = load_dt(qin, q.idx(b, s, h, i2), c.dt);
            const double y1 = load_dt(q.host, q.idx(b, s, h, i1), c.dt), y2 = load_dt(q.host, q.idx(b, s, h, i2), c.dt);
            const double u = ulp_dt(hypot(x1, x2), c.dt);
            back = std::max(back, std::max(fabs(x1 - y1), fabs(x2 - y2)) / u);
          }
  }
  const char *dn = c.dt == MLOPS_ROPE_BF16 ? "bf16" : c.dt == MLOPS_ROPE_FP16 ? "fp16" : "fp32";
  // with a factor the inverse is not the identity; FP32 carries the sincospi / FMA error (a few FP32 ulp) itself
  const double back_tol = c.att != 1 ? 1e30 : c.dt == MLOPS_ROPE_FP32 ? 8.0 : 2.0;
  report(st == MLOPS_ROPE_OK && bad == 0 && (c.in_place || back <= back_tol),
         "exact %s B%d S%d Hq%d Hk%d D%d R%d %s %s %s pos%s%lld att %.3f +%zu: worst %.2f of the bound%s, inverse within %.2f ulp",
         dn, c.B, c.S, c.Hq, c.Hk, c.D, c.R, c.inter ? "inter" : "half ", c.bhsd ? "BHSD" : "BSHD",
         c.in_place ? "in place" : "out     ", c.pos_kind == 0 ? "=s+" : c.pos_kind == 1 ? "32~" : "64~", (long long)c.pos_base,
         c.att, c.misalign, worst, bad ? " (BAD)" : "", back);
  cudaFree(dpos); cudaFree(dfreq);
}

// ---- 2. tables (Hugging Face arithmetic) -----------------------------------------------------------------
static void test_tables(int dt, bool inter, bool inverse, bool bhsd, int R, int D, std::mt19937 &rng) {
  const int B = 2, S = 37, Hq = 6, Hk = 2, P = R / 2;
  Tensor q(B, S, Hq, D, dt, bhsd, 0), k(B, S, Hk, D, dt, bhsd, 0), qo(B, S, Hq, D, dt, bhsd, 0), ko(B, S, Hk, D, dt, bhsd, 0);
  q.fill(rng, 1.0f); k.fill(rng, 2.0f); q.up(); k.up();
  // tables as Hugging Face builds them: cosf of the FP32 angle, times a factor, rounded to the dtype; [1, S, R]
  // (broadcast over the batch: stride_b 0)
  const std::vector<double> inv = make_inv_freq(R, 500000.0);
  std::vector<uint8_t> ct((size_t)S * R * es_of(dt)), st((size_t)S * R * es_of(dt));
  for (int s = 0; s < S; ++s)
    for (int i = 0; i < P; ++i) {
      const float a = (float)(s * 1000 + 7) * (float)inv[i];
      for (int j : {inter ? 2 * i : i, inter ? 2 * i + 1 : i + P}) {
        store_dt(ct, (size_t)s * R + j, cosf(a) * 1.0625f, dt);
        store_dt(st, (size_t)s * R + j, sinf(a) * 1.0625f, dt);
      }
    }
  void *dc, *ds;
  CK(cudaMalloc(&dc, ct.size())); CK(cudaMalloc(&ds, st.size()));
  CK(cudaMemcpy(dc, ct.data(), ct.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(ds, st.data(), st.size(), cudaMemcpyHostToDevice));
  mlops_rope_config cfg{};
  cfg.batch = B; cfg.seq = S; cfg.head_dim = D; cfg.rotary_dim = R; cfg.interleaved = inter; cfg.inverse = inverse;
  cfg.dtype = (mlops_rope_dtype)dt; cfg.cos = dc; cfg.sin = ds; cfg.table_stride_b = 0; cfg.table_stride_s = R;
  const mlops_rope_tensor qi = q.desc(), ki = k.desc(), qod = qo.desc(), kod = ko.desc();
  const mlops_rope_status stt = mlops_rope(&cfg, &qi, &qod, &ki, &kod, 0);
  CK(cudaDeviceSynchronize());
  qo.down(); ko.down();
  long bad = 0;
  const auto R_ = [&](float v) { return round_dt(v, dt); };
  const auto check = [&](const Tensor &in, const Tensor &out, int H) {
    for (int b = 0; b < B; ++b)
      for (int s = 0; s < S; ++s)
        for (int h = 0; h < H; ++h)
          for (int i = 0; i < P; ++i) {
            const int i1 = inter ? 2 * i : i, i2 = inter ? 2 * i + 1 : i + P;
            const float x1 = load_dt(in.host, in.idx(b, s, h, i1), dt), x2 = load_dt(in.host, in.idx(b, s, h, i2), dt);
            const float c1 = load_dt(ct, (size_t)s * R + i1, dt), c2 = load_dt(ct, (size_t)s * R + i2, dt);
            const float s1 = load_dt(st, (size_t)s * R + i1, dt), s2 = load_dt(st, (size_t)s * R + i2, dt);
            float y1, y2;
            if (!inverse) {  // q * cos + rotate_half(q) * sin
              y1 = R_(R_(x1 * c1) + R_(-x2 * s1));
              y2 = R_(R_(x2 * c2) + R_(x1 * s2));
            } else {         // its autograd backward: g * cos + rotate_half^T(g * sin)
              y1 = R_(R_(x1 * c1) + R_(x2 * s2));
              y2 = R_(R_(x2 * c2) + R_(-(x1 * s1)));
            }
            const float o1 = load_dt(out.host, out.idx(b, s, h, i1), dt), o2 = load_dt(out.host, out.idx(b, s, h, i2), dt);
            bad += f_bits(o1) != f_bits(y1) || f_bits(o2) != f_bits(y2);
          }
  };
  check(q, qo, Hq);
  check(k, ko, Hk);
  const char *dn = dt == MLOPS_ROPE_BF16 ? "bf16" : dt == MLOPS_ROPE_FP16 ? "fp16" : "fp32";
  report(stt == MLOPS_ROPE_OK && bad == 0, "tables %s %s %s %s D%d R%d: Hugging Face's arithmetic bit for bit (%ld mismatches)",
         dn, inter ? "inter" : "half ", inverse ? "backward" : "forward ", bhsd ? "BHSD" : "BSHD", D, R, bad);
  cudaFree(dc); cudaFree(ds);
}

// ---- 3. argument checks ------------------------------------------------------------------------------------
static void test_rejects() {
  Tensor q(1, 4, 2, 64, MLOPS_ROPE_BF16, false, 0);
  float *fr; CK(cudaMalloc(&fr, 32 * 16));
  mlops_rope_config cfg{};
  cfg.batch = 1; cfg.seq = 4; cfg.head_dim = 64; cfg.rotary_dim = 64; cfg.dtype = MLOPS_ROPE_BF16; cfg.freq = fr;
  const mlops_rope_tensor t = q.desc();
  int ok = mlops_rope(&cfg, &t, &t, nullptr, nullptr, 0) == MLOPS_ROPE_OK;
  mlops_rope_config c2 = cfg; c2.rotary_dim = 66;
  ok &= mlops_rope(&c2, &t, &t, nullptr, nullptr, 0) == MLOPS_ROPE_INVALID_VALUE && strstr(mlops_rope_last_error(), "rotary_dim");
  c2 = cfg; c2.rotary_dim = 31;
  ok &= mlops_rope(&c2, &t, &t, nullptr, nullptr, 0) == MLOPS_ROPE_INVALID_VALUE;
  c2 = cfg; c2.freq = nullptr;
  ok &= mlops_rope(&c2, &t, &t, nullptr, nullptr, 0) == MLOPS_ROPE_INVALID_VALUE;
  c2 = cfg; c2.cos = fr;
  ok &= mlops_rope(&c2, &t, &t, nullptr, nullptr, 0) == MLOPS_ROPE_INVALID_VALUE;
  c2 = cfg; c2.dtype = (mlops_rope_dtype)7;
  ok &= mlops_rope(&c2, &t, &t, nullptr, nullptr, 0) == MLOPS_ROPE_INVALID_VALUE;
  mlops_rope_tensor t2 = t; t2.heads = 3;
  ok &= mlops_rope(&cfg, &t, &t2, nullptr, nullptr, 0) == MLOPS_ROPE_INVALID_VALUE;
  ok &= mlops_rope(&cfg, &t, &t, &t, nullptr, 0) == MLOPS_ROPE_INVALID_VALUE;
  c2 = cfg; c2.seq = 0;
  ok &= mlops_rope(&c2, &t, &t, nullptr, nullptr, 0) == MLOPS_ROPE_OK;
  report(ok, "argument checks: bad rotary_dim, missing / ambiguous angle source, dtype, head counts, k pairing; empty input");
  cudaFree(fr);
}

// ---- mlops_rope_frequencies against transformers' formulas (mlops.positional.rope_frequencies, float64) ------------
// generated with python: rope_frequencies(dim, base, scaling, max_position_embeddings=..., seq_len=...)
static const double kShort[8] = {1.943, 1.649, 1.901, 1.113, 1.469, 1.247, 1.544, 1.574}, kLong[8] = {5.36, 6.193, 6.566, 7.597, 6.179, 7.456, 1.203, 4.259};
struct FreqCase { const char *name; int dim; double base; mlops_rope_scaling s; int max_pos, seq_len; float att; std::vector<double> inv; };
static const FreqCase kFreqCases[] = {
  {"default theta 1e6", 128, 1000000.0, {MLOPS_ROPE_SCALE_DEFAULT, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, nullptr, nullptr}, 0, 0, 1.0f, {1, 0.80584218776148187, 0.64938163157621132, 0.5232991146814947, 0.42169650342858223, 0.33982083289425596, 0.27384196342643613, 0.22067340690845899, 0.17782794100389229, 0.14330125702369628, 0.11547819846894582, 0.0930572040929699, 0.074989420933245579, 0.060429639023813285, 0.048696752516586311, 0.039241897584845357, 0.031622776601683791, 0.025482967479793464, 0.02053525026457146, 0.016548170999431813, 0.013335214321633241, 0.010746078283213174, 0.0086596432336006543, 0.0069783058485986633, 0.005623413251903491, 0.0045315836376008181, 0.0036517412725483771, 0.002942727176209282, 0.0023713737056616554, 0.0019109529749704406, 0.001539926526059492, 0.0012409377607517195, 0.001, 0.00080584218776148179, 0.00064938163157621134, 0.00052329911468149473, 0.00042169650342858224, 0.00033982083289425596, 0.00027384196342643611, 0.00022067340690845897, 0.00017782794100389227, 0.00014330125702369627, 0.00011547819846894582, 9.3057204092969901e-05, 7.4989420933245586e-05, 6.0429639023813283e-05, 4.8696752516586312e-05, 3.9241897584845357e-05, 3.1622776601683795e-05, 2.5482967479793464e-05, 2.0535250264571459e-05, 1.6548170999431813e-05, 1.3335214321633241e-05, 1.0746078283213176e-05, 8.6596432336006536e-06, 6.9783058485986632e-06, 5.6234132519034912e-06, 4.531583637600818e-06, 3.6517412725483771e-06, 2.9427271762092816e-06, 2.3713737056616552e-06, 1.9109529749704404e-06, 1.5399265260594919e-06, 1.2409377607517195e-06}},
  {"linear 4", 128, 10000.0, {MLOPS_ROPE_SCALE_LINEAR, 4.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, nullptr, nullptr}, 0, 0, 1.0f, {0.25, 0.21649108084001634, 0.18747355233311397, 0.16234540789405283, 0.14058533129758727, 0.12174188129146578, 0.10542412585714556, 0.09129353181370943, 0.079056941504209485, 0.068460490856609033, 0.059284342641541379, 0.051338125661428653, 0.044456985250973073, 0.038498163151487297, 0.033338035804083101, 0.028869549617236454, 0.025000000000000001, 0.021649108084001632, 0.018747355233311395, 0.016234540789405283, 0.014058533129758728, 0.012174188129146578, 0.010542412585714556, 0.0091293531813709423, 0.0079056941504209478, 0.0068460490856609036, 0.0059284342641541385, 0.005133812566142865, 0.0044456985250973073, 0.0038498163151487299, 0.0033338035804083102, 0.0028869549617236453, 0.0025000000000000001, 0.0021649108084001636, 0.0018747355233311395, 0.0016234540789405282, 0.0014058533129758727, 0.0012174188129146577, 0.0010542412585714557, 0.00091293531813709428, 0.00079056941504209485, 0.00068460490856609032, 0.00059284342641541385, 0.0005133812566142865, 0.00044456985250973069, 0.00038498163151487301, 0.00033338035804083102, 0.00028869549617236453, 0.00025000000000000001, 0.00021649108084001635, 0.00018747355233311396, 0.00016234540789405284, 0.00014058533129758727, 0.00012174188129146577, 0.00010542412585714556, 9.1293531813709425e-05, 7.9056941504209485e-05, 6.8460490856609026e-05, 5.9284342641541385e-05, 5.1338125661428652e-05, 4.4456985250973067e-05, 3.8498163151487298e-05, 3.3338035804083099e-05, 2.8869549617236455e-05}},
  {"dynamic 2 at 10000", 64, 10000.0, {MLOPS_ROPE_SCALE_DYNAMIC, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, nullptr, nullptr}, 4096, 10000, 1.0f, {1, 0.71778648489792107, 0.5152174379021135, 0.369816113709871, 0.26544900831841822, 0.19053571060051644, 0.13676395795947224, 0.098167320644456643, 0.070463176017231663, 0.050577515428152212, 0.036303857014043747, 0.026058417914347203, 0.018704380196740294, 0.013425751313612502, 0.0096368228425115655, 0.0069171811937103686, 0.0049650591744353715, 0.0035638523721281392, 0.0025580850668849749, 0.0018361588882292296, 0.0013179700340961336, 0.00094602107797465691, 0.00067904114419877111, 0.0004874065559954983, 0.00034985383854421048, 0.00025112035699669368, 0.00018025079833496781, 0.00012938158693690061, 9.2868354497952673e-05, 6.6659649733339491e-05, 4.7847395666620396e-05, 3.4344213947063477e-05}},
  {"yarn 4 (Qwen2.5)", 128, 1000000.0, {MLOPS_ROPE_SCALE_YARN, 4.0, 32768, 0, 0, 0, 0, 0, 0, 0, 0, nullptr, nullptr}, 0, 0, 1.13862944f, {1, 0.80584218776148187, 0.64938163157621132, 0.5232991146814947, 0.42169650342858228, 0.33982083289425591, 0.27384196342643613, 0.22067340690845899, 0.17782794100389226, 0.14330125702369628, 0.1154781984689458, 0.0930572040929699, 0.074989420933245579, 0.060429639023813278, 0.048696752516586311, 0.039241897584845357, 0.031622776601683791, 0.025482967479793464, 0.02053525026457146, 0.016548170999431816, 0.013335214321633241, 0.010746078283213174, 0.0086596432336006526, 0.0069783058485986633, 0.0053753214907901019, 0.0041317380225183937, 0.0031684225747110918, 0.0024234223804076435, 0.0018482765647068781, 0.0014051124815959123, 0.0010643609812470017, 0.00080295972754523023, 0.00060294117647058821, 0.00045032357551376927, 0.00033424054566422634, 0.00024625840690893869, 0.00017984115587395423, 0.00012993149493015669, 9.2623017041294588e-05, 6.4903943208370279e-05, 4.4456985250973067e-05, 3.5825314255924068e-05, 2.8869549617236455e-05, 2.3264301023242475e-05, 1.8747355233311396e-05, 1.5107409755953319e-05, 1.2174188129146578e-05, 9.8104743962113409e-06, 7.9056941504209488e-06, 6.3707418699483669e-06, 5.1338125661428656e-06, 4.1370427498579532e-06, 3.3338035804083101e-06, 2.6865195708032935e-06, 2.1649108084001634e-06, 1.744576462149666e-06, 1.4058533129758728e-06, 1.1328959094002045e-06, 9.1293531813709427e-07, 7.3568179405232051e-07, 5.928434264154138e-07, 4.7773824374261011e-07, 3.8498163151487298e-07, 3.1023444018792988e-07}},
  {"yarn 40, mscale, no truncate", 64, 10000.0, {MLOPS_ROPE_SCALE_YARN, 40.0, 4096, 0, 0, 32, 1, 0, 0.707, 0.707, 1, nullptr, nullptr}, 0, 0, 1.0f, {1, 0.74989420933245587, 0.56234132519034907, 0.42169650342858228, 0.31622776601683794, 0.23713737056616555, 0.17782794100389226, 0.1333521432163324, 0.10000000000000001, 0.074989420933245579, 0.056234132519034905, 0.040367584494411413, 0.027710858475080421, 0.018860063652867329, 0.012703144164814531, 0.0084462353112918342, 0.0055240629774682649, 0.0035352585197611411, 0.0021957308897397687, 0.0013051097980435558, 0.00072263834189418056, 0.00034988744270538473, 0.00011838773159168898, 3.3338035804083099e-05, 2.5000000000000001e-05, 1.8747355233311396e-05, 1.4058533129758728e-05, 1.0542412585714557e-05, 7.9056941504209471e-06, 5.9284342641541385e-06, 4.4456985250973073e-06, 3.3338035804083101e-06}},
  {"llama3 8 (Llama 3.1)", 128, 500000.0, {MLOPS_ROPE_SCALE_LLAMA3, 8.0, 8192, 1.0, 4.0, 0, 0, 0, 0, 0, 0, nullptr, nullptr}, 0, 0, 1.0f, {1, 0.81461723385654472, 0.66360123769608848, 0.54058100463576697, 0.44036660267178046, 0.35873022375128988, 0.29222782257301511, 0.23805382048035073, 0.19392274474868576, 0.15797280990904319, 0.12868737343265052, 0.10483095217796996, 0.085397100285765609, 0.069565949614160322, 0.05666962144529105, 0.04616405026546052, 0.037606030930863933, 0.030634520893224042, 0.024955408670558694, 0.020329105980970152, 0.016560440080994446, 0.013490419890226748, 0.010989528534539826, 0.0089522593361943993, 0.0072926647372171093, 0.0059407303756749668, 0.0048394213457198928, 0.0039422760301166563, 0.0032114459947525909, 0.0021665707635033591, 0.0013718935677611379, 0.00085675141291963208, 0.00052484616099295468, 0.00031269375038406517, 0.00017850781276799641, 9.5562123539646833e-05, 7.7846552739324498e-05, 6.3415143457776151e-05, 5.1659068748189563e-05, 4.2082367687255257e-05, 3.4281021959525912e-05, 2.7925911282444468e-05, 2.2748928601828183e-05, 1.8531669290821308e-05, 1.509621717643313e-05, 1.2297638677963614e-05, 1.0017868402809974e-05, 8.1607282474359434e-06, 6.6478698711812354e-06, 5.4154693654999229e-06, 4.4115346745584042e-06, 3.5937121736509995e-06, 2.9274998701761677e-06, 2.3847918463583033e-06, 1.942692537204043e-06, 1.5825508208909101e-06, 1.2891731721515574e-06, 1.0501826834601687e-06, 8.5549691264436594e-07, 6.9690252855116742e-07, 5.6770881007598369e-07, 4.6246538050008837e-07, 3.767322690173964e-07, 3.0689259889145111e-07}},
  {"longrope long", 16, 10000.0, {MLOPS_ROPE_SCALE_LONGROPE, 0, 4096, 0, 0, 0, 0, 0, 0, 0, 0, kShort, kLong}, 131072, 5000, 1.19023807f, {0.18656716417910446, 0.051062129180823174, 0.015229972586049346, 0.0041625347639441613, 0.0016183848519177862, 0.00042412522266206801, 0.00083125519534497092, 7.4249299370001852e-05}},
};

static void test_frequencies() {
  for (const FreqCase &c : kFreqCases) {
    std::vector<double> inv(c.dim / 2);
    float att = 0.f;
    const mlops_rope_status st = mlops_rope_frequencies(c.dim, c.base, &c.s, c.max_pos, c.seq_len, inv.data(), &att);
    double worst = 0.0;
    for (int i = 0; i < c.dim / 2; ++i) worst = std::max(worst, std::fabs(inv[i] - c.inv[i]) / c.inv[i]);
    report(st == MLOPS_ROPE_OK && worst <= 1e-14 && att == c.att, "frequencies %-30s rel. error %.1e, attention factor %g (want %g)",
           c.name, worst, att, c.att);
  }
}

int main() {
  std::mt19937 rng(1);
  test_frequencies();
  const int dts[3] = {MLOPS_ROPE_BF16, MLOPS_ROPE_FP16, MLOPS_ROPE_FP32};
  for (int dt : dts) {
    //                dt  B  S   Hq Hk D    R    inter  bhsd   inpl   pos base          att   misalign
    test_exact({dt, 2, 33, 8, 2, 128, 128, false, false, false, 0, 0, 1.0f, 0}, rng);           // Llama, GQA
    test_exact({dt, 2, 33, 8, 2, 128, 128, false, true, false, 1, 4000, 1.0f, 0}, rng);         // HF layout
    test_exact({dt, 1, 17, 4, 4, 64, 64, true, false, false, 2, 100000, 1.0f, 0}, rng);         // GPT-J pairs
    test_exact({dt, 3, 9, 4, 1, 80, 20, false, false, false, 1, 77, 1.0f, 0}, rng);             // partial, pairs 10: V = 2
    test_exact({dt, 1, 9, 3, 3, 64, 18, true, false, false, 0, 5, 1.0f, 0}, rng);               // pairs 9: V = 1
    test_exact({dt, 2, 7, 4, 2, 64, 64, false, false, false, 1, 0, 1.0f, 1}, rng);              // unaligned: V = 1
    test_exact({dt, 2, 9, 4, 2, 128, 128, false, false, true, 2, 0x7FFFFC00LL, 1.0f, 0}, rng);  // positions near 2^31
    test_exact({dt, 1, 9, 4, 2, 64, 32, false, false, false, 2, -50000, 1.0f, 0}, rng);         // negative positions
    test_exact({dt, 2, 11, 4, 2, 64, 64, false, true, true, 0, 1 << 20, 1.2079f, 0}, rng);      // YaRN-like factor
    test_exact({dt, 64, 1, 32, 8, 128, 128, false, false, false, 1, 3000, 1.0f, 0}, rng);       // decode batch
  }
  for (int dt : dts)
    for (bool inter : {false, true})
      for (bool inv : {false, true}) test_tables(dt, inter, inv, inv, 64, 64, rng);
  test_tables(MLOPS_ROPE_BF16, false, false, false, 32, 96, rng);  // partial rotary with tables
  test_rejects();
  printf("\n%s\n", fails ? "FAILED" : "all passed");
  return fails ? 1 : 0;
}
