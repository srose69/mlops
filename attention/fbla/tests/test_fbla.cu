// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_fbla_decode against a float64 host model of the format's definition (include/mlops_fbla.h): per
// group q-hat = rint(z * fl(127 / max|z|)) (one rounding), z = q * step_K exact, sq = max|z| * fl(sl / 127), sl = fl(scale * log2 e),
// per 16-token sub-block the coarse or the full code by
// W_j = 2^(lse_j - R) > tau (lse_j and R from the coarse scores), p = round(255 * 2^(s - m_block)), and
// o = sum_j 2^m_j sum_t p_t v_t / sum_j 2^m_j sum_t p_t.  Both layouts, GQA 1 / 2 / 4, contexts that exercise
// every work decomposition and a partial open group.  The kernel's 2^x and log2 are MUFU approximations,
// so a rounded p may differ by one: the bound is on the BF16 output.
#include "mlops_fbla.h"
#include "rope.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static float b2f(uint16_t x) { uint32_t u = (uint32_t)x << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2b(float x) {
  uint32_t u; memcpy(&u, &x, 4);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static uint16_t f2b_dir(float x, int dir) {  // toward -inf (dir < 0) or +inf
  uint16_t b = f2b(x);
  if (dir < 0 ? b2f(b) > x : b2f(b) < x) b = (b2f(b) > 0) == (dir > 0) ? b + 1 : b - 1;
  if (dir < 0 && b2f(b) > x) b = b2f(b) >= 0 ? b - 1 : b + 1;
  return b;
}

constexpr int D = 128, G = 128, S = 16;

// mlops_fbla.h, written from its text: the byte of a record holding word w of K (tensor 0) or V (1) of a lane.
static int word_at(int bits, int tensor, int lane, int w) {
  if (bits == 4) return 1024 * tensor + (w < 4 ? 16 * lane + 4 * w : 512 + 16 * lane + 4 * (w - 4));
  if (bits == 2) return 512 * tensor + 16 * lane + 4 * w;
  if (w < 4) return 512 * tensor + 16 * lane + 4 * w;
  return tensor ? 1280 + 8 * lane + 4 * (w - 4) : 1024 + 16 * (lane % 16) + 8 * (lane / 16) + 4 * (w - 4);
}
// V slot s -> (channel, token quad)
static void v_of(int bits, int s, int &ch, int &j) {
  if (bits == 4) { ch = s < 8 ? s % 2 : 2 + s % 2; j = s < 8 ? s / 2 : (s - 8) / 2; }
  else if (bits == 2) { ch = s / 4; j = s % 4; }
  else if (s < 4) { ch = 0; j = s; }
  else if (s >= 6 && s < 10) { ch = 1; j = s - 6; }
  else if (s >= 12) { ch = 3; j = s - 12; }
  else { ch = 2; j = s == 4 ? 0 : s == 5 ? 1 : s == 10 ? 2 : 3; }
}
// code of slot s, byte b of a lane into a record
static void put(uint8_t *rec, int bits, int tensor, int lane, int s, int b, int code) {
  auto set = [&](int w, int bit, int v, int n) {
    for (int k = 0; k < n; ++k) if (v >> k & 1) rec[word_at(bits, tensor, lane, w) + b] |= (uint8_t)(1u << (bit + k));
  };
  if (bits == 4) set(s % 8, 4 * (s / 8), code, 4);
  else if (bits == 2) set(s % 4, 2 * (s / 4), code, 2);
  else if (s < 12) set(s % 6, 3 * (s / 6), code, 3);
  else { const int r = s - 12; set(r, 6, code & 3, 2); set(4 + r / 2, 6 + r % 2, code >> 2, 1); }
}

struct Case {
  int cb, fb, B, QH, KH, N, groups, nb, cgroups;  // cgroups: groups of the capacity (layout stride)
  float tau;
  std::vector<float> q, k, v;                       // dense [.., N, 128], BF16-valued
  std::vector<uint16_t> qb, kbase, kstep, vbase, vstep, sk, sv, ok, ov, meta;
  std::vector<uint8_t> kcode, vcode;                // full 6-bit codes
  std::vector<uint8_t> coarse, fine;

  Case(int cb_, int fb_, int B_, int QH_, int KH_, int N_, float tau_, unsigned seed, int capacity)
      : cb(cb_), fb(fb_), B(B_), QH(QH_), KH(KH_), N(N_), groups((N_ + G - 1) / G), nb(groups * 8),
        cgroups((std::max(capacity, N_) + G - 1) / G), tau(tau_) {
    std::mt19937 r(seed);
    std::normal_distribution<float> n01(0.f, 1.f);
    const size_t heads = (size_t)B * KH, pad = (size_t)groups * G;
    q.resize((size_t)B * QH * D); qb.resize(q.size());
    for (size_t i = 0; i < q.size(); ++i) { qb[i] = f2b(1.5f * n01(r)); q[i] = b2f(qb[i]); }
    k.assign(heads * pad * D, 0.f); v.assign(k.size(), 0.f);
    std::vector<float> chan(D);
    for (int c = 0; c < D; ++c) chan[c] = .2f + 1.5f * fabsf(n01(r));  // uneven channel ranges, as in K
    for (size_t h = 0; h < heads; ++h)
      for (int t = 0; t < N; ++t)
        for (int c = 0; c < D; ++c) {
          k[(h * pad + t) * D + c] = b2f(f2b(chan[c] * n01(r) + (c % 7 == 0 ? 3.f : 0.f)));
          v[(h * pad + t) * D + c] = b2f(f2b(n01(r)));
        }
    // metadata and codes of every group (the exact groups' are unused)
    kbase.resize(heads * cgroups * D); kstep.resize(kbase.size()); vbase.resize(kbase.size()); vstep.resize(kbase.size());
    kcode.assign(heads * pad * D, 0); vcode.assign(kcode.size(), 0);
    for (int pass = 0; pass < 2; ++pass) {
      const std::vector<float> &x = pass ? v : k;
      std::vector<uint16_t> &base = pass ? vbase : kbase, &step = pass ? vstep : kstep;
      std::vector<uint8_t> &code = pass ? vcode : kcode;
      for (size_t h = 0; h < heads; ++h)
        for (int g = 0; g < groups; ++g)
          for (int c = 0; c < D; ++c) {
            const int n = std::min(G, N - g * G);
            float lo = INFINITY, hi = -INFINITY;
            for (int t = 0; t < n; ++t) { const float e = x[(h * pad + g * G + t) * D + c]; lo = std::min(lo, e); hi = std::max(hi, e); }
            const size_t m = (h * cgroups + g) * D + c;
            base[m] = f2b_dir(lo, -1);
            step[m] = f2b_dir((hi - b2f(base[m])) * (1.f / 63.f), +1);
            for (int t = 0; t < n; ++t) {
              const float e = x[(h * pad + g * G + t) * D + c];
              const float s = b2f(step[m]), inv = s > 0 ? 1.f / s : 0.f;
              code[(h * pad + g * G + t) * D + c] = (uint8_t)std::min(63.f, std::max(0.f, nearbyintf((e - b2f(base[m])) * inv)));
            }
          }
    }
    // exact sink (group 0) and open (last group) arrays, [heads][128][128]
    sk.assign(heads * G * D, 0); sv.assign(sk.size(), 0); ok.assign(sk.size(), 0); ov.assign(sk.size(), 0);
    for (size_t h = 0; h < heads; ++h)
      for (int t = 0; t < G; ++t)
        for (int c = 0; c < D; ++c) {
          const size_t e = (h * G + t) * D + c;
          sk[e] = f2b(k[(h * pad + t) * D + c]); sv[e] = f2b(v[(h * pad + t) * D + c]);
          const int to = (groups - 1) * G + t;
          if (to < N) { ok[e] = f2b(k[(h * pad + to) * D + c]); ov[e] = f2b(v[(h * pad + to) * D + c]); }
        }
    // records and metadata
    coarse.assign(heads * cgroups * 8 * 2 * cb * 256, 0);
    fine.assign(heads * cgroups * 8 * 2 * fb * 256, 0);
    for (size_t h = 0; h < heads; ++h)
      for (int t = 0; t < N; ++t)
        for (int c = 0; c < D; ++c) {
          const size_t sub = h * cgroups * 8 + t / S;
          const int tt = t % S, ck = kcode[(h * pad + t) * D + c], cv = vcode[(h * pad + t) * D + c];
          const int lk = tt + 16 * (c / 64), sk_ = (c % 64) / 4, bk = c % 4;
          int sv_ = 0;
          for (int x = 0; x < 16; ++x) { int ch, j; v_of(cb, x, ch, j); if (ch == c % 4 && j == tt / 4) sv_ = x; }
          int svf = 0;
          for (int x = 0; x < 16; ++x) { int ch, j; v_of(fb, x, ch, j); if (ch == c % 4 && j == tt / 4) svf = x; }
          uint8_t *rc = coarse.data() + sub * 2 * cb * 256, *rf = fine.data() + sub * 2 * fb * 256;
          put(rc, cb, 0, lk, sk_, bk, ck >> fb); put(rf, fb, 0, lk, sk_, bk, ck & ((1 << fb) - 1));
          put(rc, cb, 1, c / 4, sv_, tt % 4, cv >> fb); put(rf, fb, 1, c / 4, svf, tt % 4, cv & ((1 << fb) - 1));
        }
    meta.assign(heads * cgroups * 512, 0);
    for (size_t hg = 0; hg < heads * cgroups; ++hg)
      for (int c = 0; c < D; ++c) {
        uint16_t *m = meta.data() + hg * 512;
        const int l = c / 4, x = c % 4;
        m[8 * l + x] = kbase[hg * D + c]; m[8 * l + 4 + x] = kstep[hg * D + c];
        m[256 + 8 * l + x] = vbase[hg * D + c]; m[256 + 8 * l + 4 + x] = vstep[hg * D + c];
      }
  }

  // float64 model of the definition
  // decode semantics at position P - 1 over its prefix of P tokens, queries qv [B, QH, D]
  // pc / pe: tokens of a P block (its maximum and weight) in the closed / the BF16 groups: 16 in decode and the
  // DP4A prefill, 128 / 64 in the tensor-core prefill.  fpe: the BF16 groups' P = 255 2^(s - mb) rounded to BF16
  // (the tensor-core prefill), else rint (every block's of decode and DP4A)
  std::vector<double> reference_at(int P, const float *qv, int pc = S, int pe = S, bool fpe = false) const {
    const int groups = (P + G - 1) / G, nb = groups * 8, N = P;
    const size_t pad = (size_t)this->groups * G;
    const float sl = (1.f / sqrtf((float)D)) * 1.4426950408889634f;  // FP32, as the kernel scales q
    const double hoff = .5 * ((1 << fb) - 1);
    std::vector<double> out((size_t)B * QH * D);
    for (int b = 0; b < B; ++b)
      for (int hq = 0; hq < QH; ++hq) {
        const int hk = hq / (QH / KH);
        const size_t h = (size_t)b * KH + hk;
        float qs[D];
        for (int c = 0; c < D; ++c) qs[c] = qv[((size_t)b * QH + hq) * D + c] * sl;
        // per sub-block: coarse and full scores
        std::vector<double> sc((size_t)nb * S, -INFINITY), sf(sc.size(), -INFINITY);
        std::vector<char> exact(nb, 0);
        for (int g = 0; g < groups; ++g) {
          const bool ex = g == 0 || g == groups - 1;
          const int n = std::min(G, N - g * G);
          int8_t qh[D] = {};
          double sq = 0, qbs = 0, qsum = 0;
          if (!ex) {  // z = q step exact; q-hat = rint(z r) of the exact product (z r in double: exact, then one rounding)
            const float *qr = qv + ((size_t)b * QH + hq) * D;
            float amax = 0;
            for (int c = 0; c < D; ++c) amax = std::max(amax, fabsf(qr[c] * b2f(kstep[(h * cgroups + g) * D + c])));
            const float sqf = amax * (sl * (1.f / 127.f)), rq = amax > 0 ? 127.f / amax : 0.f;
            for (int c = 0; c < D; ++c) {
              const float z = qr[c] * b2f(kstep[(h * cgroups + g) * D + c]);
              qh[c] = (int8_t)nearbyint((double)z * rq);
              qbs += (double)qr[c] * b2f(kbase[(h * cgroups + g) * D + c]);
              qsum += qh[c];
            }
            qbs *= sl;
            sq = sqf;
          }
          for (int t = 0; t < n; ++t) {
            const int tok = g * G + t;
            if (ex) {
              double s = 0;
              for (int c = 0; c < D; ++c) s += (double)qs[c] * k[(h * pad + tok) * D + c];
              sc[tok] = sf[tok] = s;
            } else {
              double ic = 0, iff = 0;
              for (int c = 0; c < D; ++c) {
                const int code = kcode[(h * pad + tok) * D + c];
                ic += (double)qh[c] * (code >> fb); iff += (double)qh[c] * code;
              }
              sc[tok] = qbs + sq * ((1 << fb) * ic + hoff * qsum);
              sf[tok] = qbs + sq * iff;
            }
          }
          for (int j = 0; j < 8; ++j) exact[g * 8 + j] = ex;
        }
        // the P blocks: start token and length (pc or pe tokens, within a group)
        std::vector<int> bs, bl;
        for (int g = 0; g < groups; ++g) {
          const int len = exact[g * 8] ? pe : pc;
          for (int t0 = 0; t0 < G; t0 += len) { bs.push_back(g * G + t0); bl.push_back(len); }
        }
        const int nbk = (int)bs.size();
        std::vector<double> p(G);
        auto block = [&](const std::vector<double> &s, int j, double &mb, double &sp) {
          mb = -INFINITY;
          for (int t = 0; t < bl[j]; ++t) mb = std::max(mb, s[bs[j] + t]);
          sp = 0;
          const bool fp = fpe && exact[bs[j] / S];
          for (int t = 0; t < bl[j]; ++t) {
            const double x = std::isfinite(s[bs[j] + t]) ? 255.0 * exp2(s[bs[j] + t] - mb) : 0;
            p[t] = fp ? b2f(f2b((float)x)) : nearbyint(x);
            sp += p[t];
          }
        };
        double R = -INFINITY;
        std::vector<double> lse(nbk, -INFINITY);
        for (int j = 0; j < nbk; ++j) {
          double mb, sp;
          block(sc, j, mb, sp);
          if (!std::isfinite(mb)) continue;
          lse[j] = mb + log2(sp);
          R = std::max(R, lse[j]) + log2(1 + exp2(-fabs(R - lse[j])));
        }
        double L = 0, O[D] = {};
        for (int j = 0; j < nbk; ++j) {
          if (!std::isfinite(lse[j])) continue;
          const bool ex = exact[bs[j] / S];
          const bool fine = !ex && exp2(lse[j] - R) > tau;
          double mb, sp;
          block(fine ? sf : sc, j, mb, sp);
          const double w = exp2(mb - R);
          L += w * sp;
          const int g = bs[j] / G;
          for (int c = 0; c < D; ++c) {
            double acc = 0;
            for (int t = 0; t < bl[j]; ++t) {
              const int tok = bs[j] + t;
              if (!p[t]) continue;
              double val;
              if (ex) val = v[(h * pad + tok) * D + c];
              else {
                const int code = vcode[(h * pad + tok) * D + c];
                const double base = b2f(vbase[(h * cgroups + g) * D + c]), st = b2f(vstep[(h * cgroups + g) * D + c]);
                val = fine ? base + st * code : base + st * ((code >> fb) * (1 << fb) + hoff);
              }
              acc += p[t] * val;
            }
            O[c] += w * acc;
          }
        }
        for (int c = 0; c < D; ++c) out[((size_t)b * QH + hq) * D + c] = O[c] / L;
      }
    return out;
  }
  std::vector<double> reference() const { return reference_at(N, q.data()); }

};

template <class T> static T *up(const std::vector<T> &h) {
  T *p; CK(cudaMalloc(&p, h.size() * sizeof(T)));
  CK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}

// Strided runs (cfg head_stride / group_stride): the cache is layer 1 of 3 whose groups lie side by side -- unit (head
// h, group g) at (layer heads + h) + g (3 heads); the other layers' units hold random bytes.
static bool g_strided = false;
constexpr int kLayers = 3, kLayer = 1;
static void set_strides(mlops_fbla_config &cfg, size_t heads) {
  cfg.head_stride = g_strided ? 1 : 0;
  cfg.group_stride = g_strided ? (int)(kLayers * heads) : 0;
}
static size_t unit_at(size_t h, size_t g, size_t heads, size_t cgroups) {
  return g_strided ? (kLayer * heads + h) + g * kLayers * heads : h * cgroups + g;
}
// h: an array of [head][capacity group] units of `unit` elements -> the device array of the run's layout (*base: the
// allocation)
template <class T> static T *place(const std::vector<T> &h, size_t unit, size_t heads, size_t cgroups, T **base) {
  std::vector<T> s(g_strided ? kLayers * heads * cgroups * unit : h.size());
  std::mt19937 r(77);
  for (auto &x : s) x = (T)r();
  for (size_t hh = 0; hh < heads; ++hh)
    for (size_t g = 0; g < cgroups; ++g)
      std::copy_n(h.begin() + (hh * cgroups + g) * unit, unit, s.begin() + unit_at(hh, g, heads, cgroups) * unit);
  *base = up(s);
  return *base;
}
// the device array of the run's layout -> [head][capacity group] units
template <class T> static std::vector<T> gather(const T *base, size_t n, size_t unit, size_t heads, size_t cgroups) {
  std::vector<T> s(g_strided ? kLayers * heads * cgroups * unit : n), h(n);
  CK(cudaMemcpy(s.data(), base, s.size() * sizeof(T), cudaMemcpyDeviceToHost));
  for (size_t hh = 0; hh < heads; ++hh)
    for (size_t g = 0; g < cgroups; ++g)
      std::copy_n(s.begin() + unit_at(hh, g, heads, cgroups) * unit, unit, h.begin() + (hh * cgroups + g) * unit);
  return h;
}
// a run's cache pointer: unit 0 of its layer
template <class T> static T *own(T *base, size_t unit, size_t heads) { return base + (g_strided ? kLayer * heads * unit : 0); }

static int run(int cb, int fb, int B, int QH, int KH, int N, float tau, unsigned seed, int capacity = 0) {
  Case c(cb, fb, B, QH, KH, N, tau, seed, capacity);
  const std::vector<double> ref = c.reference();
  const size_t heads = (size_t)B * KH, uc = 8u * 2 * cb * 256, uf = 8u * 2 * fb * 256;
  uint16_t *dq = up(c.qb), *dmeta, *dsk = up(c.sk), *dsv = up(c.sv), *dok = up(c.ok), *dov = up(c.ov), *dout;
  uint8_t *dc, *df;
  place(c.meta, 512, heads, c.cgroups, &dmeta);
  place(c.coarse, uc, heads, c.cgroups, &dc);
  place(c.fine, uf, heads, c.cgroups, &df);
  CK(cudaMalloc(&dout, ref.size() * 2));
  mlops_fbla_config cfg{B, QH, KH, N, 0.f, tau, cb == 3 ? MLOPS_FBLA_3P3 : MLOPS_FBLA_4P2, capacity};
  set_strides(cfg, heads);
  const size_t sz = mlops_fbla_decode_scratch_bytes(&cfg);
  void *scratch; CK(cudaMalloc(&scratch, sz)); CK(cudaMemset(scratch, 0xFF, sz));  // no initial contents required
  mlops_fbla_cache cache{own(dc, uc, heads), own(df, uf, heads), own(dmeta, 512, heads), dsk, dsv, dok, dov};
  int f = 0;
  for (int rep = 0; rep < 2; ++rep) {  // twice: the result repeats bitwise on the used scratch
    if (mlops_fbla_decode(&cfg, dq, &cache, dout, scratch, sz, 0)) { fprintf(stderr, "%s\n", mlops_fbla_last_error()); exit(1); }
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> got(ref.size());
    CK(cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost));
    static std::vector<uint16_t> first;
    const bool same = rep == 0 || got == first;
    first = got;
    double worst = 0, rms = 0, nrm = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double e = b2f(got[i]) - ref[i];
      worst = std::max(worst, fabs(e)); rms += e * e; nrm += ref[i] * ref[i];
    }
    const double rel = sqrt(rms / nrm);
    const bool ok = rel < 4e-3 && worst < 2e-2 && same;
    printf("[%s] %d+%d B=%d heads %d/%d N=%-6d capacity %-6d tau %-6g run %d%s: rel. RMS error %.2e, max abs %.2e%s\n",
           ok ? "PASS" : "FAIL", cb, fb, B, QH, KH, N, capacity, tau, rep, g_strided ? " (strided)" : "", rel, worst,
           same ? "" : ", differs from run 0");
    f += !ok;
  }
  for (void *p : {(void *)dq, (void *)dmeta, (void *)dsk, (void *)dsv, (void *)dok, (void *)dov, (void *)dout,
                  (void *)dc, (void *)df, scratch}) cudaFree(p);
  return f;
}

// Prefill: an empty cache of capacity N filled by chunks; every chunk's queries (one per position and q head)
// against the float64 model at their positions (a sample of them: the first and last of a chunk, around group
// boundaries, and every 37th), then the records and metadata of every closed group against the host's
// packing, bit for bit.  append_only: the chunks are appended with q = null, and the cache alone is checked.
// the prefill runs on tensor cores (P blocks of 128 tokens in the closed groups, 64 in the BF16 ones) on sm_75 ..
// sm_89, unless MLOPS_FBLA_DP4A=1
static bool tc_prefill() {
  int dev = 0, major = 0, minor = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  const char *e = getenv("MLOPS_FBLA_DP4A");
  const int cc = 10 * major + minor;
  return cc >= 75 && cc <= 89 && !(e && e[0] == '1');
}

static bool g_images = false;  // the cache carries the tensor-core prefill's persistent images (tc_images)
static int run_prefill(int cb, int fb, int B, int QH, int KH, int N, unsigned seed, std::vector<int> chunks,
                       bool append_only = false) {
  const bool tc = tc_prefill();
  Case c(cb, fb, B, QH, KH, N, 1e-30f, seed, N);  // prefill refines every block
  const size_t pad = (size_t)c.groups * G, heads = (size_t)B * KH;
  std::mt19937 r(seed + 1000);
  std::normal_distribution<float> n01(0.f, 1.f);
  std::vector<float> qall((size_t)B * QH * N * D);  // [B, QH, N, D]
  for (auto &x : qall) x = b2f(f2b(1.5f * n01(r)));
  const size_t uc = 8u * 2 * cb * 256, uf = 8u * 2 * fb * 256;
  std::vector<uint8_t> zc(c.coarse.size(), 0), zf(c.fine.size(), 0);
  std::vector<uint16_t> zm(c.meta.size(), 0), ze(heads * G * D, 0);
  uint8_t *dc, *df;
  uint16_t *dm, *dsk = up(ze), *dsv = up(ze), *dok = up(ze), *dov = up(ze);
  place(zc, uc, heads, c.cgroups, &dc);
  place(zf, uf, heads, c.cgroups, &df);
  place(zm, 512, heads, c.cgroups, &dm);
  mlops_fbla_cache cache{own(dc, uc, heads), own(df, uf, heads), own(dm, 512, heads), dsk, dsv, dok, dov};
  uint8_t *dimg = nullptr;
  if (g_images) {  // zero-filled once, then kept over the chunks (strided: the other layers' slots random)
    const mlops_fbla_config one{1, 1, 1, 0, 0.f, 0.f, MLOPS_FBLA_3P3, G};
    const size_t slot = mlops_fbla_images_bytes(&one);
    place(std::vector<uint8_t>(heads * c.cgroups * slot, 0), slot, heads, c.cgroups, &dimg);
    cache.tc_images = own(dimg, slot, heads);
  }
  int f = 0, n0 = 0;
  double err2 = 0, nrm2 = 0, worst = 0;
  long checked = 0;
  for (int cnt : chunks) {
    std::vector<uint16_t> kh(heads * cnt * D), vh(kh.size()), qh((size_t)B * QH * cnt * D);
    for (size_t h = 0; h < heads; ++h)
      for (int t = 0; t < cnt; ++t)
        for (int x = 0; x < D; ++x) {
          kh[(h * cnt + t) * D + x] = f2b(c.k[(h * pad + n0 + t) * D + x]);
          vh[(h * cnt + t) * D + x] = f2b(c.v[(h * pad + n0 + t) * D + x]);
        }
    for (size_t hq = 0; hq < (size_t)B * QH; ++hq)
      for (int t = 0; t < cnt; ++t)
        for (int x = 0; x < D; ++x) qh[(hq * cnt + t) * D + x] = f2b(qall[(hq * N + n0 + t) * D + x]);
    uint16_t *dk = up(kh), *dv = up(vh), *dq = up(qh), *dout;
    CK(cudaMalloc(&dout, qh.size() * 2));
    mlops_fbla_config cfg{B, QH, KH, n0, 0.f, 0.f, cb == 3 ? MLOPS_FBLA_3P3 : MLOPS_FBLA_4P2, N};
    set_strides(cfg, heads);
    if (mlops_fbla_prefill(&cfg, cnt, append_only ? nullptr : dq, dk, dv, &cache, append_only ? nullptr : dout, 0)) {
      fprintf(stderr, "%s\n", mlops_fbla_last_error()); exit(1);
    }
    CK(cudaDeviceSynchronize());
    if (!append_only) {
      std::vector<uint16_t> got(qh.size());
      CK(cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost));
      for (int t = 0; t < cnt; ++t) {
        const int p = n0 + t;
        if (!(t == 0 || t == cnt - 1 || p % G <= 1 || p % G == G - 1 || p % 37 == 0)) continue;
        std::vector<float> qv((size_t)B * QH * D);
        for (size_t hq = 0; hq < (size_t)B * QH; ++hq)
          for (int x = 0; x < D; ++x) qv[hq * D + x] = qall[(hq * N + p) * D + x];
        const std::vector<double> ref = c.reference_at(p + 1, qv.data(), tc ? G : S, tc ? 64 : S, tc);
        for (size_t hq = 0; hq < (size_t)B * QH; ++hq)
          for (int x = 0; x < D; ++x) {
            const double e = b2f(got[(hq * cnt + t) * D + x]) - ref[hq * D + x];
            err2 += e * e; nrm2 += ref[hq * D + x] * ref[hq * D + x]; worst = std::max(worst, fabs(e));
          }
        ++checked;
      }
    }
    for (void *x : {(void *)dk, (void *)dv, (void *)dq, (void *)dout}) cudaFree(x);
    n0 += cnt;
  }
  // the closed groups' records and metadata, bit for bit
  const int closed = (N - 1) / G;
  const std::vector<uint8_t> gc = gather(dc, c.coarse.size(), uc, heads, c.cgroups), gf = gather(df, c.fine.size(), uf, heads, c.cgroups);
  const std::vector<uint16_t> gm = gather(dm, c.meta.size(), 512, heads, c.cgroups);
  long bad = 0;
  for (size_t h = 0; h < heads; ++h)
    for (int g = 0; g < closed; ++g) {
      const size_t hg = h * c.cgroups + g;
      for (size_t e = 0; e < 8u * 2 * cb * 256; ++e) bad += gc[hg * 8 * 2 * cb * 256 + e] != c.coarse[hg * 8 * 2 * cb * 256 + e];
      for (size_t e = 0; e < 8u * 2 * fb * 256; ++e) bad += gf[hg * 8 * 2 * fb * 256 + e] != c.fine[hg * 8 * 2 * fb * 256 + e];
      for (size_t e = 0; e < 512; ++e) bad += gm[hg * 512 + e] != c.meta[hg * 512 + e];
    }
  const double rel = nrm2 > 0 ? sqrt(err2 / nrm2) : 0;
  const bool ok = bad == 0 && (append_only || (rel < 4e-3 && worst < 2e-2));
  printf("[%s] prefill %d+%d B=%d heads %d/%d N=%-6d chunks %-3zu%s%s: ", ok ? "PASS" : "FAIL", cb, fb, B, QH, KH, N,
         chunks.size(), append_only ? " (append only)" : g_images ? " (images)" : "", g_strided ? " (strided)" : "");
  if (!append_only) printf("%ld positions, rel. RMS error %.2e, max abs %.2e; ", checked, rel, worst);
  printf("closed groups' bytes differing: %ld\n", bad);
  f += !ok;
  for (void *x : {(void *)dc, (void *)df, (void *)dm, (void *)dsk, (void *)dsv, (void *)dok, (void *)dov, (void *)dimg}) cudaFree(x);
  return f;
}

// The prefill's chunks at the device length and count: n0 tokens prefilled into both caches, then the same chunks two
// ways -- mlops_fbla_prefill per chunk on the host, and one CUDA graph of mlops_fbla_prefill_at calls with the length
// and each chunk's count in device memory (no launch of either call depending on them) -- bit for bit, every chunk's
// outputs and the caches.  The chunk buffers are the ceiling's size; the last chunk holds fewer tokens.
__global__ void step_chunk(int *length, int *count, const int *counts, int i) {
  if (threadIdx.x == 0) { *count = counts[i + 1]; *length += counts[i]; }
}
static int run_prefill_at(int cb, int fb, int B, int QH, int KH, int n0, std::vector<int> counts, unsigned seed,
                          bool append_only = false) {
  const int max_count = *std::max_element(counts.begin(), counts.end());
  int total = 0;
  for (int c : counts) total += c;
  const int cap = n0 + total + 100;
  const size_t heads = (size_t)B * KH, cg = (cap + G - 1) / G;
  const size_t uc = 8u * 2 * cb * 256, uf = 8u * 2 * fb * 256;
  const mlops_fbla_format fmt = cb == 3 ? MLOPS_FBLA_3P3 : MLOPS_FBLA_4P2;
  std::mt19937 r(seed);
  std::normal_distribution<float> n01(0.f, 1.f);
  auto randn = [&](size_t n, float sd) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2b(sd * n01(r)); return v; };
  struct Cache { uint8_t *c, *f; uint16_t *m, *e; mlops_fbla_cache x; uint8_t *cb_, *fb_; uint16_t *mb_; uint8_t *im = nullptr; } cache[2];
  {
    const mlops_fbla_config one{1, 1, 1, 0, 0.f, 0.f, fmt, G};
    const size_t slot = mlops_fbla_images_bytes(&one);  // (0 where the prefill runs on CUDA cores)
    for (Cache &k : cache) {
      place(std::vector<uint8_t>(heads * cg * uc, 0), uc, heads, cg, &k.cb_);
      place(std::vector<uint8_t>(heads * cg * uf, 0), uf, heads, cg, &k.fb_);
      place(std::vector<uint16_t>(heads * cg * 512, 0), 512, heads, cg, &k.mb_);
      k.e = up(std::vector<uint16_t>(4 * heads * G * D, 0));
      k.x = {own(k.cb_, uc, heads), own(k.fb_, uf, heads), own(k.mb_, 512, heads), k.e, k.e + heads * G * D,
             k.e + 2 * heads * G * D, k.e + 3 * heads * G * D, nullptr};
      if (slot) {
        place(std::vector<uint8_t>(heads * cg * slot, 0), slot, heads, cg, &k.im);
        k.x.tc_images = own(k.im, slot, heads);
      }
    }
  }
  const std::vector<uint16_t> kv = randn(2 * heads * (n0 + total) * D, 2.f), qall = randn((size_t)B * QH * total * D, 1.5f);
  mlops_fbla_config cfg{B, QH, KH, 0, 0.f, 0.f, fmt, cap};
  set_strides(cfg, heads);
  {  // the prompt, n0 tokens, into both, attending (the closed groups get their images: a prefill_at cannot
     // build images of groups older than its own chunk)
    const std::vector<uint16_t> p(kv.begin(), kv.begin() + 2 * heads * n0 * D), pq = randn((size_t)B * QH * n0 * D, 1.5f);
    uint16_t *d = up(p), *dq = up(pq), *dout;
    CK(cudaMalloc(&dout, pq.size() * 2));
    for (Cache &k : cache)
      if (mlops_fbla_prefill(&cfg, n0, dq, d, d + heads * n0 * D, &k.x, dout, 0)) {
        fprintf(stderr, "%s\n", mlops_fbla_last_error());
        exit(1);
      }
    cudaFree(d);
    cudaFree(dq);
    cudaFree(dout);
  }
  // the host way: every chunk compact, into cache[0]
  std::vector<std::vector<uint16_t>> outs;
  int pos = n0;
  for (int cnt : counts) {
    std::vector<uint16_t> kh(heads * cnt * D), vh(kh.size()), qh((size_t)B * QH * cnt * D);
    for (size_t h = 0; h < heads; ++h)
      for (int t = 0; t < cnt; ++t)
        for (int x = 0; x < D; ++x) {
          kh[(h * cnt + t) * D + x] = kv[(h * (n0 + total) + pos + t) * D + x];
          vh[(h * cnt + t) * D + x] = kv[(heads * (n0 + total) + h * (n0 + total) + pos + t) * D + x];
        }
    for (size_t hq = 0; hq < (size_t)B * QH; ++hq)
      for (int t = 0; t < cnt; ++t)
        for (int x = 0; x < D; ++x) qh[(hq * cnt + t) * D + x] = qall[(hq * total + (pos - n0) + t) * D + x];
    uint16_t *dk = up(kh), *dv = up(vh), *dq = up(qh), *dout;
    CK(cudaMalloc(&dout, qh.size() * 2));
    mlops_fbla_config c = cfg;
    c.tokens = pos;
    if (mlops_fbla_prefill(&c, cnt, append_only ? nullptr : dq, dk, dv, &cache[0].x, append_only ? nullptr : dout, 0)) {
      fprintf(stderr, "%s\n", mlops_fbla_last_error());
      exit(1);
    }
    CK(cudaDeviceSynchronize());
    if (!append_only) {
      outs.emplace_back(qh.size());
      CK(cudaMemcpy(outs.back().data(), dout, outs.back().size() * 2, cudaMemcpyDeviceToHost));
    }
    for (void *x : {(void *)dk, (void *)dv, (void *)dq, (void *)dout}) cudaFree(x);
    pos += cnt;
  }
  // the graph way: one capture of every chunk, the buffers the ceiling's size, into cache[1]
  const int n = (int)counts.size();
  const std::vector<int> cvec = [&] { std::vector<int> v(counts); v.push_back(0); return v; }();  // the last step's count
  int *dlen, *dcount, *dcounts;
  CK(cudaMalloc(&dlen, 4)); CK(cudaMalloc(&dcount, 4)); CK(cudaMalloc(&dcounts, (n + 1) * 4));
  CK(cudaMemcpy(dlen, &n0, 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcount, cvec.data(), 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcounts, cvec.data(), (n + 1) * 4, cudaMemcpyHostToDevice));
  std::vector<uint16_t *> dk(n), dv(n), dq(n), dout(n);
  pos = n0;
  for (int i = 0; i < n; ++i) {
    const int cnt = counts[i];
    std::vector<uint16_t> kh(heads * max_count * D, 0), vh(kh.size()), qh((size_t)B * QH * max_count * D, 0);
    for (size_t h = 0; h < heads; ++h)
      for (int t = 0; t < cnt; ++t)
        for (int x = 0; x < D; ++x) {
          kh[(h * max_count + t) * D + x] = kv[(h * (n0 + total) + pos + t) * D + x];
          vh[(h * max_count + t) * D + x] = kv[(heads * (n0 + total) + h * (n0 + total) + pos + t) * D + x];
        }
    for (size_t hq = 0; hq < (size_t)B * QH; ++hq)
      for (int t = 0; t < cnt; ++t)
        for (int x = 0; x < D; ++x) qh[(hq * max_count + t) * D + x] = qall[(hq * total + (pos - n0) + t) * D + x];
    dk[i] = up(kh); dv[i] = up(vh); dq[i] = up(qh);
    CK(cudaMalloc(&dout[i], qh.size() * 2));
    pos += cnt;
  }
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  int bad_call = 0;
  for (int i = 0; i < n; ++i) {
    bad_call |= mlops_fbla_prefill_at(&cfg, dlen, dcount, max_count, append_only ? nullptr : dq[i], dk[i], dv[i],
                                      &cache[1].x, append_only ? nullptr : dout[i], st);
    step_chunk<<<1, 32, 0, st>>>(dlen, dcount, dcounts, i);
  }
  CK(cudaStreamEndCapture(st, &graph));
  if (bad_call) { fprintf(stderr, "%s\n", mlops_fbla_last_error()); exit(1); }
  CK(cudaGraphInstantiate(&exec, graph, 0));
  CK(cudaGraphLaunch(exec, st));
  CK(cudaStreamSynchronize(st));
  long differ = 0;
  if (!append_only) {
    std::vector<uint16_t> all((size_t)B * QH * max_count * D);
    for (int i = 0; i < n; ++i) {
      CK(cudaMemcpy(all.data(), dout[i], all.size() * 2, cudaMemcpyDeviceToHost));
      for (size_t h = 0; h < (size_t)B * QH; ++h)  // the used rows within each head's ceiling-sized block
        differ += memcmp(all.data() + h * max_count * D, outs[i].data() + h * counts[i] * D, (size_t)counts[i] * D * 2) != 0;
    }
  }
  long bytes_differ = 0;  // the caches after the last chunk
  {
    const auto a = gather(cache[0].cb_, heads * cg * uc, uc, heads, cg), b = gather(cache[1].cb_, heads * cg * uc, uc, heads, cg);
    const auto af = gather(cache[0].fb_, heads * cg * uf, uf, heads, cg), bf = gather(cache[1].fb_, heads * cg * uf, uf, heads, cg);
    const auto am = gather(cache[0].mb_, heads * cg * 512, 512, heads, cg), bm = gather(cache[1].mb_, heads * cg * 512, 512, heads, cg);
    std::vector<uint16_t> ae(4 * heads * G * D), be(ae.size());
    CK(cudaMemcpy(ae.data(), cache[0].e, ae.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(be.data(), cache[1].e, be.size() * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < a.size(); ++i) bytes_differ += a[i] != b[i];
    for (size_t i = 0; i < af.size(); ++i) bytes_differ += af[i] != bf[i];
    for (size_t i = 0; i < am.size(); ++i) bytes_differ += am[i] != bm[i];
    for (size_t i = 0; i < ae.size(); ++i) bytes_differ += ae[i] != be[i];
    if (cache[0].im) {  // the images, as the attention reads them
      const size_t slot = 34048 + 128;
      const std::vector<uint8_t> ai = gather(cache[0].im, heads * cg * slot, slot, heads, cg);
      const std::vector<uint8_t> bi = gather(cache[1].im, heads * cg * slot, slot, heads, cg);
      for (size_t i = 0; i < ai.size(); ++i) bytes_differ += ai[i] != bi[i];
    }
  }
  const bool ok = differ == 0 && bytes_differ == 0;
  printf("[%s] prefill_at %d+%d B=%d heads %d/%d tokens %d..%d, %zu chunks (max %d)%s%s: chunk outputs differing %ld, cache elements differing %ld\n",
         ok ? "PASS" : "FAIL", cb, fb, B, QH, KH, n0, n0 + total, counts.size(), max_count,
         append_only ? ", append only" : "", g_strided ? " (strided)" : "", differ, bytes_differ);
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(st);
  for (Cache &k : cache)
    for (void *x : {(void *)k.cb_, (void *)k.fb_, (void *)k.mb_, (void *)k.e, (void *)k.im}) cudaFree(x);
  for (int i = 0; i < n; ++i)
    for (void *x : {(void *)dk[i], (void *)dv[i], (void *)dq[i], (void *)dout[i]}) cudaFree(x);
  for (void *x : {(void *)dlen, (void *)dcount, (void *)dcounts}) cudaFree(x);
  return !ok;
}

// Device length: the same token by token two ways from the same prefilled cache -- host appends and decodes with the
// length on the host, and one CUDA graph of append_at, the length's increment and decode_at replayed per token (the
// launches do not depend on the length) -- bit for bit, outputs and caches.
__global__ void advance(int *length) { ++*length; }
static int run_device_length(int cb, int fb, int B, int QH, int KH, int n0, int steps, float tau, unsigned seed) {
  const int cap = n0 + steps + 100;
  const size_t heads = (size_t)B * KH, cg = (cap + G - 1) / G;
  const size_t uc = 8u * 2 * cb * 256, uf = 8u * 2 * fb * 256;
  const mlops_fbla_format fmt = cb == 3 ? MLOPS_FBLA_3P3 : MLOPS_FBLA_4P2;
  std::mt19937 r(seed);
  std::normal_distribution<float> n01(0.f, 1.f);
  auto randn = [&](size_t n, float sd) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2b(sd * n01(r)); return v; };
  struct Cache { uint8_t *c, *f; uint16_t *m, *e; mlops_fbla_cache x; uint8_t *cb_, *fb_; uint16_t *mb_; } cache[2];
  for (Cache &k : cache) {
    place(std::vector<uint8_t>(heads * cg * uc, 0), uc, heads, cg, &k.cb_);
    place(std::vector<uint8_t>(heads * cg * uf, 0), uf, heads, cg, &k.fb_);
    place(std::vector<uint16_t>(heads * cg * 512, 0), 512, heads, cg, &k.mb_);
    k.e = up(std::vector<uint16_t>(4 * heads * G * D, 0));
    k.x = {own(k.cb_, uc, heads), own(k.fb_, uf, heads), own(k.mb_, 512, heads), k.e, k.e + heads * G * D, k.e + 2 * heads * G * D,
           k.e + 3 * heads * G * D, nullptr};
  }
  mlops_fbla_config cfg{B, QH, KH, 0, 0.f, tau, fmt, cap, 0, 0};
  set_strides(cfg, heads);
  {  // the prompt, n0 tokens, into both
    const std::vector<uint16_t> kv = randn(2 * heads * n0 * D, 2.f);
    uint16_t *d = up(kv);
    for (Cache &k : cache)
      if (mlops_fbla_prefill(&cfg, n0, nullptr, d, d + heads * n0 * D, &k.x, nullptr, 0)) { fprintf(stderr, "%s\n", mlops_fbla_last_error()); exit(1); }
    cudaFree(d);
  }
  // fixed buffers of the step: k, v [heads, D], q [B QH, D], the outputs, the device length, the scratches
  uint16_t *dk, *dv, *dq, *out[2];
  int *dlen;
  CK(cudaMalloc(&dk, heads * D * 2)); CK(cudaMalloc(&dv, heads * D * 2)); CK(cudaMalloc(&dq, (size_t)B * QH * D * 2));
  for (auto &o : out) CK(cudaMalloc(&o, (size_t)B * QH * D * 2));
  CK(cudaMalloc(&dlen, 4)); CK(cudaMemcpy(dlen, &n0, 4, cudaMemcpyHostToDevice));
  mlops_fbla_config lc = cfg;
  lc.tokens = cap;
  const size_t sz = mlops_fbla_decode_scratch_bytes(&lc);
  void *scratch[2];
  for (auto &x : scratch) CK(cudaMalloc(&x, sz));
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  int bad_call = mlops_fbla_append_at(&cfg, dlen, dk, dv, &cache[1].x, st);
  advance<<<1, 1, 0, st>>>(dlen);
  bad_call |= mlops_fbla_decode_at(&cfg, dlen, dq, &cache[1].x, out[1], scratch[1], sz, st);
  CK(cudaStreamEndCapture(st, &graph));
  if (bad_call) { fprintf(stderr, "%s\n", mlops_fbla_last_error()); exit(1); }
  CK(cudaGraphInstantiate(&exec, graph, 0));
  long differ = 0;
  for (int s = 0; s < steps; ++s) {
    const std::vector<uint16_t> kh = randn(heads * D, 2.f), vh = randn(heads * D, 2.f), qh = randn((size_t)B * QH * D, 1.5f);
    CK(cudaMemcpy(dk, kh.data(), kh.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, vh.data(), vh.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dq, qh.data(), qh.size() * 2, cudaMemcpyHostToDevice));
    mlops_fbla_config c = cfg;
    c.tokens = n0 + s;
    if (mlops_fbla_prefill(&c, 1, nullptr, dk, dv, &cache[0].x, nullptr, 0)) { fprintf(stderr, "%s\n", mlops_fbla_last_error()); exit(1); }
    c.tokens = n0 + s + 1;
    if (mlops_fbla_decode(&c, dq, &cache[0].x, out[0], scratch[0], sz, 0)) { fprintf(stderr, "%s\n", mlops_fbla_last_error()); exit(1); }
    CK(cudaDeviceSynchronize());
    CK(cudaGraphLaunch(exec, st));
    CK(cudaStreamSynchronize(st));
    std::vector<uint16_t> o0((size_t)B * QH * D), o1(o0.size());
    CK(cudaMemcpy(o0.data(), out[0], o0.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o1.data(), out[1], o1.size() * 2, cudaMemcpyDeviceToHost));
    differ += o0 != o1;
  }
  long bytes = 0;  // the caches after the last token
  {
    const auto a = gather(cache[0].cb_, heads * cg * uc, uc, heads, cg), b = gather(cache[1].cb_, heads * cg * uc, uc, heads, cg);
    const auto af = gather(cache[0].fb_, heads * cg * uf, uf, heads, cg), bf = gather(cache[1].fb_, heads * cg * uf, uf, heads, cg);
    const auto am = gather(cache[0].mb_, heads * cg * 512, 512, heads, cg), bm = gather(cache[1].mb_, heads * cg * 512, 512, heads, cg);
    std::vector<uint16_t> ae(4 * heads * G * D), be(ae.size());
    CK(cudaMemcpy(ae.data(), cache[0].e, ae.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(be.data(), cache[1].e, be.size() * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < a.size(); ++i) bytes += a[i] != b[i];
    for (size_t i = 0; i < af.size(); ++i) bytes += af[i] != bf[i];
    for (size_t i = 0; i < am.size(); ++i) bytes += am[i] != bm[i];
    for (size_t i = 0; i < ae.size(); ++i) bytes += ae[i] != be[i];
  }
  int length = 0;
  CK(cudaMemcpy(&length, dlen, 4, cudaMemcpyDeviceToHost));
  const bool ok = differ == 0 && bytes == 0 && length == n0 + steps;
  printf("[%s] device length %d+%d B=%d heads %d/%d tokens %d .. %d tau %-6g%s: outputs differing %ld, cache elements differing %ld\n",
         ok ? "PASS" : "FAIL", cb, fb, B, QH, KH, n0, n0 + steps, tau, g_strided ? " (strided)" : "", differ, bytes);
  cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(st);
  for (Cache &k : cache) for (void *x : {(void *)k.cb_, (void *)k.fb_, (void *)k.mb_, (void *)k.e}) cudaFree(x);
  for (void *x : {(void *)dk, (void *)dv, (void *)dq, (void *)out[0], (void *)out[1], (void *)dlen, scratch[0], scratch[1]}) cudaFree(x);
  return !ok;
}

// The fused step input (mlops_fbla_qkv_at) against its parts: q and k normalized on the host (the documented order:
// the lane's four squares, the xor butterfly, 1 / sqrt rounded to nearest), rotated by the RoPE device code itself
// (rope.cuh, as mlops_rope), k and v appended with append_at -- q and the cache bit for bit, token by token from n0,
// q / k / v in inferno's layout (per KV head its q heads, k, v).
__global__ void ref_rope(const uint16_t *x, uint16_t *y, const int *length, int off, const mlops_rope_detail::Freq *freq) {
  const int i = threadIdx.x, h = blockIdx.x;
  float c, sn;
  mlops_rope_detail::angle_cos_sin(*length + off, freq[i], 1.f, c, sn);
  float y1, y2;
  auto ld = [](uint16_t v) { return __uint_as_float((uint32_t)v << 16); };
  auto st = [](float v) { return (uint16_t)(mlops_rope_detail::bf16_rne_hi(__float_as_uint(v)) >> 16); };
  mlops_rope_detail::rotate_exact(ld(x[h * D + i]), ld(x[h * D + i + 64]), make_float4(c, sn, c, sn), y1, y2);
  y[h * D + i] = st(y1);
  y[h * D + i + 64] = st(y2);
}
static float host_bf16r(float x) { return b2f(f2b(x)); }
static std::vector<uint16_t> host_norm(const uint16_t *x, const std::vector<uint16_t> &w, float eps) {
  float lanes[32];
  for (int l = 0; l < 32; ++l) {
    const float v[4] = {b2f(x[2 * l]), b2f(x[2 * l + 1]), b2f(x[64 + 2 * l]), b2f(x[65 + 2 * l])};
    float s = v[0] * v[0];
    for (int i = 1; i < 4; ++i) s = fmaf(v[i], v[i], s);
    lanes[l] = s;
  }
  for (int o = 16; o; o >>= 1) {
    float nx[32];
    for (int l = 0; l < 32; ++l) nx[l] = lanes[l] + lanes[l ^ o];
    memcpy(lanes, nx, sizeof nx);
  }
  const float r = (float)(1.0 / sqrt((double)(lanes[0] * (1.f / D) + eps)));
  std::vector<uint16_t> y(D);
  for (int e = 0; e < D; ++e) y[e] = f2b(host_bf16r(b2f(x[e]) * r) * b2f(w[e]));
  return y;
}
static int run_qkv(int cb, int fb, int B, int QH, int KH, int n0, int steps, bool norm, unsigned seed) {
  const int cap = n0 + steps + 100, qg = QH / KH;
  const size_t heads = (size_t)B * KH, cg = (cap + G - 1) / G, row = (size_t)KH * (qg + 2) * D;
  const size_t uc = 8u * 2 * cb * 256, uf = 8u * 2 * fb * 256;
  std::mt19937 r(seed);
  std::normal_distribution<float> n01(0.f, 1.f);
  auto randn = [&](size_t n, float sd) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2b(sd * n01(r)); return v; };
  const float eps = 1e-6f;
  const std::vector<uint16_t> qw = randn(D, 0.5f), kw = randn(D, 0.5f);
  std::vector<float> freq(4 * 64);
  for (int i = 0; i < 64; ++i) {  // mlops_rope_freq of theta 1e6
    const double f = pow(1e6, -2.0 * i / D) / (2 * M_PI), g = fmod(65536.0 * f, 1.0);
    freq[4 * i] = (float)f; freq[4 * i + 1] = (float)(f - (float)f);
    freq[4 * i + 2] = (float)g; freq[4 * i + 3] = (float)(g - (float)g);
  }
  struct Cache { uint16_t *e; mlops_fbla_cache x; uint8_t *cb_, *fb_; uint16_t *mb_; } cache[2];
  for (Cache &k : cache) {
    place(std::vector<uint8_t>(heads * cg * uc, 0), uc, heads, cg, &k.cb_);
    place(std::vector<uint8_t>(heads * cg * uf, 0), uf, heads, cg, &k.fb_);
    place(std::vector<uint16_t>(heads * cg * 512, 0), 512, heads, cg, &k.mb_);
    k.e = up(std::vector<uint16_t>(4 * heads * G * D, 0));
    k.x = {own(k.cb_, uc, heads), own(k.fb_, uf, heads), own(k.mb_, 512, heads), k.e, k.e + heads * G * D, k.e + 2 * heads * G * D,
           k.e + 3 * heads * G * D, nullptr};
  }
  mlops_fbla_config cfg{B, QH, KH, 0, 0.f, 0.f, cb == 3 ? MLOPS_FBLA_3P3 : MLOPS_FBLA_4P2, cap, 0, 0};
  set_strides(cfg, heads);
  if (n0) {
    const std::vector<uint16_t> kv = randn(2 * heads * n0 * D, 2.f);
    uint16_t *d = up(kv);
    for (Cache &k : cache)
      if (mlops_fbla_prefill(&cfg, n0, nullptr, d, d + heads * n0 * D, &k.x, nullptr, 0)) { fprintf(stderr, "%s\n", mlops_fbla_last_error()); exit(1); }
    cudaFree(d);
  }
  uint16_t *dqkv, *dqn, *dkn, *dv, *dqr, *dkr, *dqo, *dqw = up(qw), *dkw = up(kw);
  float *dfreq = up(freq);
  int *len[2];
  CK(cudaMalloc(&dqkv, B * row * 2)); CK(cudaMalloc(&dqn, (size_t)B * QH * D * 2)); CK(cudaMalloc(&dkn, heads * D * 2));
  CK(cudaMalloc(&dv, heads * D * 2)); CK(cudaMalloc(&dqr, (size_t)B * QH * D * 2)); CK(cudaMalloc(&dkr, heads * D * 2));
  CK(cudaMalloc(&dqo, (size_t)B * QH * D * 2));
  for (auto &l : len) { CK(cudaMalloc(&l, 4)); CK(cudaMemcpy(l, &n0, 4, cudaMemcpyHostToDevice)); }
  const mlops_fbla_token tok{dqkv, dqkv + qg * D, dqkv + (qg + 1) * D, (int64_t)row, (int64_t)(qg + 2) * D, (int64_t)(qg + 2) * D,
                             norm ? dqw : nullptr, norm ? dkw : nullptr, eps, dfreq, 0.f, dqo};
  long qdiff = 0;
  for (int s = 0; s < steps; ++s) {
    const std::vector<uint16_t> h = randn(B * row, 2.f);
    std::vector<uint16_t> qn((size_t)B * QH * D), kn(heads * D), vv(heads * D);
    for (int b = 0; b < B; ++b)
      for (int g = 0; g < KH; ++g) {
        const uint16_t *grp = h.data() + b * row + (size_t)g * (qg + 2) * D;
        for (int j = 0; j < qg + 2; ++j) {
          std::vector<uint16_t> y(grp + j * D, grp + (j + 1) * D);
          if (norm && j <= qg) y = host_norm(grp + j * D, j < qg ? qw : kw, eps);
          uint16_t *to = j < qg ? &qn[(((size_t)b * QH) + g * qg + j) * D] : j == qg ? &kn[((size_t)b * KH + g) * D] : &vv[((size_t)b * KH + g) * D];
          std::copy(y.begin(), y.end(), to);
        }
      }
    CK(cudaMemcpy(dqkv, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqn, qn.data(), qn.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dkn, kn.data(), kn.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, vv.data(), vv.size() * 2, cudaMemcpyHostToDevice));
    // the parts: RoPE of the normalized q and k, k / v appended
    const auto *fr = reinterpret_cast<const mlops_rope_detail::Freq *>(dfreq);
    ref_rope<<<B * QH, 64>>>(dqn, dqr, len[0], 0, fr);
    ref_rope<<<(int)heads, 64>>>(dkn, dkr, len[0], 0, fr);
    if (mlops_fbla_append_at(&cfg, len[0], dkr, dv, &cache[0].x, 0)) { fprintf(stderr, "%s\n", mlops_fbla_last_error()); exit(1); }
    advance<<<1, 1>>>(len[0]);
    // fused
    if (mlops_fbla_qkv_at(&cfg, len[1], &tok, &cache[1].x, 0)) { fprintf(stderr, "%s\n", mlops_fbla_last_error()); exit(1); }
    advance<<<1, 1>>>(len[1]);
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> a(qn.size()), b(qn.size());
    CK(cudaMemcpy(a.data(), dqr, a.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), dqo, b.size() * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < a.size(); ++i) qdiff += a[i] != b[i];
  }
  long bytes = 0;
  {
    const auto a = gather(cache[0].cb_, heads * cg * uc, uc, heads, cg), b = gather(cache[1].cb_, heads * cg * uc, uc, heads, cg);
    const auto af = gather(cache[0].fb_, heads * cg * uf, uf, heads, cg), bf = gather(cache[1].fb_, heads * cg * uf, uf, heads, cg);
    const auto am = gather(cache[0].mb_, heads * cg * 512, 512, heads, cg), bm = gather(cache[1].mb_, heads * cg * 512, 512, heads, cg);
    std::vector<uint16_t> ae(4 * heads * G * D), be(ae.size());
    CK(cudaMemcpy(ae.data(), cache[0].e, ae.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(be.data(), cache[1].e, be.size() * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < a.size(); ++i) bytes += a[i] != b[i];
    for (size_t i = 0; i < af.size(); ++i) bytes += af[i] != bf[i];
    for (size_t i = 0; i < am.size(); ++i) bytes += am[i] != bm[i];
    for (size_t i = 0; i < ae.size(); ++i) bytes += ae[i] != be[i];
  }
  const bool ok = qdiff == 0 && bytes == 0;
  printf("[%s] qkv_at %d+%d B=%d heads %d/%d tokens %d .. %d %s%s: q differing %ld, cache elements differing %ld\n", ok ? "PASS" : "FAIL",
         cb, fb, B, QH, KH, n0, n0 + steps, norm ? "q/k norm" : "no norm", g_strided ? " (strided)" : "", qdiff, bytes);
  for (Cache &k : cache) for (void *x : {(void *)k.cb_, (void *)k.fb_, (void *)k.mb_, (void *)k.e}) cudaFree(x);
  for (void *x : {(void *)dqkv, (void *)dqn, (void *)dkn, (void *)dv, (void *)dqr, (void *)dkr, (void *)dqo, (void *)dqw, (void *)dkw,
                  (void *)dfreq, (void *)len[0], (void *)len[1]})
    cudaFree(x);
  return !ok;
}

int main() {
  int f = 0;
  for (int fmt = 0; fmt < 2; ++fmt) {
    const int cb = fmt ? 4 : 3, fb = fmt ? 2 : 3;
    f += run(cb, fb, 1, 2, 2, 384, 1e-2f, 1);      // MHA, one closed group
    f += run(cb, fb, 1, 4, 2, 1000, 3e-3f, 2);     // GQA 2, open group of 104 tokens
    f += run(cb, fb, 2, 8, 2, 4096, 1e-3f, 3);     // GQA 4, batch 2
    f += run(cb, fb, 1, 16, 8, 20000, 3e-4f, 4);   // Qwen3 geometry, long context
    f += run(cb, fb, 1, 4, 2, 2048, 0.f, 5);       // format default tau
    f += run(cb, fb, 1, 4, 2, 2048, 1e30f, 6);     // nothing refined
    f += run(cb, fb, 1, 4, 2, 2048, 1e-30f, 7);    // everything refined
    f += run(cb, fb, 1, 4, 2, 3000, 1e-3f, 8, 8192);  // a cache laid out for 8192 tokens, 3000 filled
    f += run(cb, fb, 1, 4, 2, 262144, 1e-4f, 9);   // 256K
    f += run(cb, fb, 1, 8, 2, 2048, 1e30f, 10);    // GQA 4, nothing refined
    f += run(cb, fb, 1, 8, 2, 2048, 1e-30f, 11);   // GQA 4, everything refined
    f += run(cb, fb, 1, 32, 8, 20000, 3e-4f, 12);  // GQA 4, Llama-3-8B geometry
    // ratios beyond 4: virtual KV heads of 4, 2 or 1 queries
    f += run(cb, fb, 1, 16, 2, 3000, 1e-3f, 13);   // GQA 8 (2 x 4), Qwen3-32B / Llama-3-70B
    f += run(cb, fb, 2, 32, 2, 2048, 1e-3f, 14);   // GQA 16 (4 x 4), batch 2
    f += run(cb, fb, 1, 10, 2, 3000, 1e-3f, 15);   // GQA 5 (5 x 1), Qwen3-14B
    f += run(cb, fb, 1, 14, 2, 3000, 1e-3f, 16);   // GQA 7 (7 x 1), Qwen2.5-7B
    f += run(cb, fb, 1, 12, 2, 2048, 1e-3f, 17);   // GQA 6 (3 x 2)
    f += run(cb, fb, 1, 64, 8, 20000, 3e-4f, 18);  // GQA 8, 8 KV heads, long context
  }
  for (int pass = 0; pass < (tc_prefill() ? 4 : 2); ++pass) {  // tensor cores: without and with persistent images
    const int fmt = pass % 2, cb = fmt ? 4 : 3, fb = fmt ? 2 : 3;
    g_images = pass >= 2;
    f += run_prefill(cb, fb, 1, 2, 2, 100, 21, {100});                 // inside the first group
    f += run_prefill(cb, fb, 1, 4, 2, 1000, 22, {1000});               // one chunk, open group of 104
    f += run_prefill(cb, fb, 2, 8, 2, 2048, 23, {300, 212, 128, 1, 1407});  // chunk edges on and off groups
    f += run_prefill(cb, fb, 1, 16, 8, 4096, 24, {1024, 1024, 2048});  // Qwen3 geometry
    f += run_prefill(cb, fb, 1, 2, 2, 3000, 25, {1500, 1500});         // MHA, two chunks
    f += run_prefill(cb, fb, 1, 8, 2, 3000, 26, {700, 2300});          // GQA 4
    f += run_prefill(cb, fb, 1, 4, 2, 8177, 28, {6000, 2100, 77});     // short chunks on a long stream (split-K)
    f += run_prefill(cb, fb, 1, 16, 2, 3000, 29, {1100, 1900});         // GQA 8
    f += run_prefill(cb, fb, 1, 6, 2, 2000, 30, {700, 1300});           // GQA 3 (the ratio at run time)
    f += run_prefill(cb, fb, 1, 32, 2, 1500, 31, {700, 800});           // GQA 16
    f += run_prefill(cb, fb, 1, 10, 2, 2000, 32, {900, 1100});          // GQA 5
    f += run_prefill(cb, fb, 1, 2, 2, 2000, 27, {1, 1, 254, 1744}, true);  // append only
    f += run_prefill_at(cb, fb, 1, 2, 2, 100, {100, 100}, 61);           // inside the first group
    f += run_prefill_at(cb, fb, 1, 4, 2, 1000, {300, 212, 128, 1, 1407}, 62);  // chunk edges on and off groups
    f += run_prefill_at(cb, fb, 2, 8, 2, 2048, {512, 512, 700, 330}, 63);     // GQA 4, batch 2, the last chunk short
    f += run_prefill_at(cb, fb, 1, 8, 2, 2048, {512, 512, 700, 330}, 68);    // GQA 4, batch 1
    f += run_prefill_at(cb, fb, 2, 8, 4, 2048, {512, 512, 700, 330}, 69);    // GQA 2, batch 2
    f += run_prefill_at(cb, fb, 1, 16, 8, 1024, {1024, 2048}, 64);       // Qwen3 geometry, the prompt over groups
    f += run_prefill_at(cb, fb, 1, 10, 2, 120, {100, 100}, 65);          // GQA 5
    f += run_prefill_at(cb, fb, 1, 2, 2, 2000, {1, 1, 254, 1744}, 66, true);  // append only
  }
  for (int fmt = 0; fmt < 2; ++fmt) {  // the length in device memory: across group edges, the sink's end, GQA 1 / 2 / 4;
    const int cb = fmt ? 4 : 3, fb = fmt ? 2 : 3;  // tau as the decode cases of the same geometry (4, 3, 1)
    f += run_device_length(cb, fb, 1, 16, 8, 120, 280, 3e-4f, 41);
    f += run_device_length(cb, fb, 2, 8, 2, 1, 300, 1e-3f, 42);
    f += run_device_length(cb, fb, 1, 2, 2, 1000, 150, 1e-2f, 43);
    f += run_device_length(cb, fb, 1, 16, 2, 120, 280, 1e-3f, 45);  // GQA 8: virtual heads at the device length
    f += run_device_length(cb, fb, 1, 10, 2, 120, 150, 1e-3f, 46);  // GQA 5
  }
  for (int fmt = 0; fmt < 2; ++fmt) {  // the fused step input: across group edges, from the sink, GQA 1 / 2 / 4, norm or not
    const int cb = fmt ? 4 : 3, fb = fmt ? 2 : 3;
    f += run_qkv(cb, fb, 1, 32, 8, 120, 280, true, 51);
    f += run_qkv(cb, fb, 2, 8, 2, 0, 300, true, 52);
    f += run_qkv(cb, fb, 1, 2, 2, 1000, 150, false, 53);
  }
  g_strided = true;  // the layers' groups side by side: the same results, the cache's units where the strides put them
  for (int fmt = 0; fmt < 2; ++fmt) {
    const int cb = fmt ? 4 : 3, fb = fmt ? 2 : 3;
    f += run(cb, fb, 2, 8, 2, 4096, 1e-3f, 3);
    f += run(cb, fb, 1, 16, 8, 20000, 3e-4f, 4);
    f += run(cb, fb, 1, 4, 2, 3000, 1e-3f, 8, 8192);
    for (int images = 0; images < (tc_prefill() ? 2 : 1); ++images) {
      g_images = images;
      f += run_prefill(cb, fb, 2, 8, 2, 2048, 23, {300, 212, 128, 1, 1407});
      f += run_prefill(cb, fb, 1, 16, 8, 4096, 24, {1024, 1024, 2048});
      f += run_prefill(cb, fb, 1, 4, 2, 8177, 28, {6000, 2100, 77});
    }
    f += run_prefill_at(cb, fb, 1, 4, 2, 1000, {300, 212, 128, 1, 1407}, 67);
    f += run_device_length(cb, fb, 1, 16, 8, 120, 280, 3e-4f, 44);
    f += run_qkv(cb, fb, 1, 32, 8, 120, 280, true, 54);
  }
  printf(f ? "\n%d FAILED\n" : "\nall passed\n", f);
  return f != 0;
}
