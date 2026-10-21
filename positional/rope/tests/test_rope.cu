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

int main() {
  std::mt19937 rng(1);
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
