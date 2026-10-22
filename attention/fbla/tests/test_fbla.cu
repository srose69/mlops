// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_fbla_decode against a float64 host model of the format's definition (include/mlops_fbla.h): per
// group q-hat = rint(z * fl(127 / max|z|)), z = q * (scale * log2 e) * step_K in FP32, sq = max|z| * fl(1 / 127),
// per 16-token sub-block the coarse or the full code by
// W_j = 2^(lse_j - R) > tau (lse_j and R from the coarse scores), p = round(255 * 2^(s - m_block)), and
// o = sum_j 2^m_j sum_t p_t v_t / sum_j 2^m_j sum_t p_t.  Both layouts, GQA 1 / 2 / 4, contexts that exercise
// every work decomposition and a partial open group.  The kernel's 2^x and log2 are MUFU approximations,
// so a rounded p may differ by one: the bound is on the BF16 output.
#include "mlops_fbla.h"
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
            step[m] = f2b_dir((hi - b2f(base[m])) / 63.f, +1);
            for (int t = 0; t < n; ++t) {
              const float e = x[(h * pad + g * G + t) * D + c];
              const float s = b2f(step[m]);
              code[(h * pad + g * G + t) * D + c] = (uint8_t)std::min(63.f, std::max(0.f, s > 0 ? nearbyintf((e - b2f(base[m])) / s) : 0.f));
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
  std::vector<double> reference() const {
    const size_t pad = (size_t)groups * G;
    const float sl = (1.f / sqrtf((float)D)) * 1.4426950408889634f;  // FP32, as the kernel scales q
    const double hoff = .5 * ((1 << fb) - 1);
    std::vector<double> out((size_t)B * QH * D);
    for (int b = 0; b < B; ++b)
      for (int hq = 0; hq < QH; ++hq) {
        const int hk = hq / (QH / KH);
        const size_t h = (size_t)b * KH + hk;
        float qs[D];
        for (int c = 0; c < D; ++c) qs[c] = q[((size_t)b * QH + hq) * D + c] * sl;
        // per sub-block: coarse and full scores
        std::vector<double> sc((size_t)nb * S, -INFINITY), sf(sc.size(), -INFINITY);
        std::vector<char> exact(nb, 0);
        for (int g = 0; g < groups; ++g) {
          const bool ex = g == 0 || g == groups - 1;
          const int n = std::min(G, N - g * G);
          int8_t qh[D] = {};
          double sq = 0, qbs = 0, qsum = 0;
          if (!ex) {
            float amax = 0;
            for (int c = 0; c < D; ++c) amax = std::max(amax, fabsf(qs[c] * b2f(kstep[(h * cgroups + g) * D + c])));
            const float sqf = amax * (1.f / 127.f), rq = amax > 0 ? 127.f / amax : 0.f;
            for (int c = 0; c < D; ++c) {
              const float z = qs[c] * b2f(kstep[(h * cgroups + g) * D + c]);
              qh[c] = (int8_t)nearbyintf(z * rq);
              qbs += (double)qs[c] * b2f(kbase[(h * cgroups + g) * D + c]);
              qsum += qh[c];
            }
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
        auto block = [&](const std::vector<double> &s, int j, double &mb, double p[S], double &sp) {
          mb = -INFINITY;
          for (int t = 0; t < S; ++t) mb = std::max(mb, s[j * S + t]);
          sp = 0;
          for (int t = 0; t < S; ++t) { p[t] = std::isfinite(s[j * S + t]) ? nearbyint(255.0 * exp2(s[j * S + t] - mb)) : 0; sp += p[t]; }
        };
        double R = -INFINITY;
        std::vector<double> lse(nb, -INFINITY);
        for (int j = 0; j < nb; ++j) {
          double mb, p[S], sp;
          block(sc, j, mb, p, sp);
          if (!std::isfinite(mb)) continue;
          lse[j] = mb + log2(sp);
          R = std::max(R, lse[j]) + log2(1 + exp2(-fabs(R - lse[j])));
        }
        double L = 0, O[D] = {};
        for (int j = 0; j < nb; ++j) {
          if (!std::isfinite(lse[j])) continue;
          const bool fine = !exact[j] && exp2(lse[j] - R) > tau;
          double mb, p[S], sp;
          block(fine ? sf : sc, j, mb, p, sp);
          const double w = exp2(mb - R);
          L += w * sp;
          const int g = j / 8;
          for (int c = 0; c < D; ++c) {
            double acc = 0;
            for (int t = 0; t < S; ++t) {
              const int tok = j * S + t;
              if (!p[t]) continue;
              double val;
              if (exact[j]) val = v[(h * pad + tok) * D + c];
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
};

template <class T> static T *up(const std::vector<T> &h) {
  T *p; CK(cudaMalloc(&p, h.size() * sizeof(T)));
  CK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}

static int run(int cb, int fb, int B, int QH, int KH, int N, float tau, unsigned seed, int capacity = 0) {
  Case c(cb, fb, B, QH, KH, N, tau, seed, capacity);
  const std::vector<double> ref = c.reference();
  uint16_t *dq = up(c.qb), *dmeta = up(c.meta), *dsk = up(c.sk), *dsv = up(c.sv), *dok = up(c.ok), *dov = up(c.ov), *dout;
  uint8_t *dc = up(c.coarse), *df = up(c.fine);
  CK(cudaMalloc(&dout, ref.size() * 2));
  mlops_fbla_config cfg{B, QH, KH, N, 0.f, tau, cb == 3 ? MLOPS_FBLA_3P3 : MLOPS_FBLA_4P2, capacity};
  const size_t sz = mlops_fbla_decode_scratch_bytes(&cfg);
  void *scratch; CK(cudaMalloc(&scratch, sz)); CK(cudaMemset(scratch, 0, sz));
  mlops_fbla_cache cache{dc, df, dmeta, dsk, dsv, dok, dov};
  int f = 0;
  for (int rep = 0; rep < 2; ++rep) {  // twice: the CTA counter must reset itself, the result repeat bitwise
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
    printf("[%s] %d+%d B=%d heads %d/%d N=%-6d capacity %-6d tau %-6g run %d: rel. RMS error %.2e, max abs %.2e%s\n",
           ok ? "PASS" : "FAIL", cb, fb, B, QH, KH, N, capacity, tau, rep, rel, worst, same ? "" : ", differs from run 0");
    f += !ok;
  }
  for (void *p : {(void *)dq, (void *)dmeta, (void *)dsk, (void *)dsv, (void *)dok, (void *)dov, (void *)dout,
                  (void *)dc, (void *)df, scratch}) cudaFree(p);
  return f;
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
  }
  printf(f ? "\n%d FAILED\n" : "\nall passed\n", f);
  return f != 0;
}
