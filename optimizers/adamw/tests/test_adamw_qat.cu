/* mlops_adamw_qat tests.
 *   1. codecs: SR into E4M3 / E5M2 (E[s^2] = v) / BF16 / fixed point is unbiased, the Quake rsqrt stays
 *      within 6.5e-4, e2m1_round agrees with NVFP4EffMA's f32_to_e2m1 on every 7th float in [0, 8).
 *   2. one step vs an independent double-precision reference of include/mlops_adamw_qat.h, linear and
 *      swizzled scales: every stored s, r is one of the two SR neighbours of the exact value, u and qbar
 *      are within one quantum, code == round(u), the secant accumulators are BF16 neighbours, voting
 *      blocks update the CUSUM and move exactly as the reference (a move keeps the master values),
 *      silent blocks keep their CUSUM and scale.
 *   3. behaviour over many steps: pure-noise gradients leave the block scales put; blocks whose optimum
 *      is exactly representable 3 E4M3 codes up find that scale; a constant gradient walks one element
 *      down its cells at lr / cell width per step.
 */
#include "mlops_adamw_qat.h"
#include "../src/qat_codec.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <random>
#include <vector>

#define CK(e) do { cudaError_t _e = (e); if (_e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #e, cudaGetErrorString(_e)); exit(1); } } while (0)
#define ST(e) do { mlops_adamw_status _s = (e); if (_s != MLOPS_ADAMW_OK) { fprintf(stderr, "%s: %s\n", #e, mlops_adamw_last_error()); exit(1); } } while (0)

using namespace qat;

static int report(bool ok, const char *what) { printf("[%s] %s\n", ok ? "PASS" : "FAIL", what); return ok ? 0 : 1; }

// host decodes (the reference side; the kernel uses NVFP4EffMA's / E5M2EffMA's)
static double e4m3(uint32_t b) {
  const uint32_t e = (b >> 3) & 15u, m = b & 7u;
  const double v = e ? ldexp(1 + m / 8.0, (int)e - 7) : ldexp(m, -9);
  return (b & 0x80u) ? -v : v;
}
static double e5m2(uint32_t b) {
  const uint32_t e = (b >> 2) & 31u, m = b & 3u;
  const double v = e ? ldexp(1 + m / 4.0, (int)e - 15) : ldexp(m, -16);
  return (b & 0x80u) ? -v : v;
}
static double e2m1(uint32_t n) { static const double t[8] = {0, .5, 1, 1.5, 2, 3, 4, 6}; return (n & 8u) ? -t[n & 7u] : t[n & 7u]; }
static double gof(double u) { const double a = fabs(u); return copysign(a < 4 ? a / 2 : a < 6 ? a - 2 : 2 * a - 8, u); }
static double ginv(double y) { const double a = fabs(y); return copysign(a < 2 ? 2 * a : a < 4 ? a + 2 : (a - 4) / 2 + 6, y); }
static double huber_d(double e, double d) { const double a = fabs(e); return a <= d ? 0.5 * a * a : d * (a - 0.5 * d); }
static double e2m1_round_d(double a) {  // nearest E2M1 magnitude, ties to the even code, saturating
  static const double t[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
  int best = 0;
  for (int k = 1; k < 8; k++) {
    const double d = fabs(a - t[k]), db = fabs(a - t[best]);
    if (d < db || (d == db && !(k & 1))) best = k;
  }
  return t[best];
}

// ---- 1. codecs ------------------------------------------------------------------------------------
static int test_codecs() {
  std::mt19937 rng(3); std::uniform_real_distribution<float> U(0, 1);
  int f = 0;
  double worst = 0;
  for (int i = 0; i < 2000; i++) {
    const float x = std::exp(U(rng) * 30 - 20) * (U(rng) < .5 ? -1 : 1);
    if (fabsf(x) >= 400) continue;
    double sum = 0; const int n = 4000;
    for (int k = 0; k < n; k++) sum += e4m3(sr_e4m3(x, U(rng)));
    const double q = fabsf(x) < 0x1p-6f ? 0x1p-9 : ldexp(1, expo(fabsf(x)) - 3);
    worst = std::max(worst, fabs(sum / n - x) / q);
  }
  f += report(worst < 0.05, "SR E4M3 unbiased (mean error < 0.05 quantum over 2000 values)");
  worst = 0;  // exact expectation: the output switches lo -> hi at one U*, so E[s^2] = lo^2 + U* (hi^2 - lo^2)
  for (int i = 0; i < 20000; i++) {
    const float vt = std::exp(U(rng) * 60 - 40), x = std::sqrt(vt) * (1 + (U(rng) - .5f) * 2e-3f);  // seed off by up to 1e-3
    if (x >= 50000) continue;
    const double lo = e5m2(sr_e5m2_sq(x, vt, 0.9999999f)), hi = e5m2(sr_e5m2_sq(x, vt, 0.0f));
    float a = 0, b = 1;
    for (int k = 0; k < 300 && std::nextafter(a, 1.0f) < b; k++) { const float m = 0.5f * (a + b); (e5m2(sr_e5m2_sq(x, vt, m)) == hi ? a : b) = m; }
    const double e = lo * lo + (double)b * (hi * hi - lo * lo);
    worst = std::max(worst, fabs(e / vt - 1));
  }
  { char m[140]; snprintf(m, sizeof m, "SR E5M2: E[s^2] = v exactly (max rel. error %.2e over 20000 values, seed off by 1e-3)", worst);
    f += report(worst < 1e-6, m); }
  worst = 0;
  for (int i = 0; i < 1000; i++) {
    const float t = (U(rng) - .5f) * 60000; double sum = 0;
    for (int k = 0; k < 4000; k++) sum += sr_floor(t, U(rng));
    worst = std::max(worst, fabs(sum / 4000 - t));
  }
  f += report(worst < 0.03, "SR fixed point unbiased (mean error < 0.03 quantum)");
  worst = 0;
  for (int i = 0; i < 1000; i++) {
    const float x = std::exp(U(rng) * 40 - 20) * (U(rng) < .5 ? -1 : 1); double sum = 0;
    for (int k = 0; k < 4000; k++) sum += dec_bf16(sr_bf16(x, U(rng)));
    worst = std::max(worst, fabs(sum / 4000 - x) / ldexp(1, expo(fabsf(x)) - 7));
  }
  f += report(worst < 0.03, "SR BF16 unbiased (mean error < 0.03 quantum)");
  worst = 0;
  for (int i = 0; i < 100000; i++) { const float x = std::exp(U(rng) * 80 - 40); worst = std::max(worst, fabs(rsqrt_quake(x) * std::sqrt((double)x) - 1)); }
  f += report(worst < 6.6e-4, "Quake rsqrt |rel err| < 6.6e-4");
  int bad = 0;
  for (uint32_t b = 0; b < 0x41000000u; b += 7) {  // every 7th float in [0, 8)
    float a; memcpy(&a, &b, 4);
    if (e2m1_round(a) != (float)e2m1(nvfp4a::f32_to_e2m1(a))) bad++;
  }
  f += report(!bad, "e2m1_round == decode(nvfp4a::f32_to_e2m1) on every 7th float in [0, 8)");
  return f;
}

// ---- device helpers --------------------------------------------------------------------------------
struct Q {
  int rows, cols, nb; bool swz; size_t sbytes;
  uint8_t *data, *sc; float *g; void *state; mlops_adamw_qat_state st; nvfp4a_tensor t;
  Q(int r, int c, bool swizzled) : rows(r), cols(c), nb(c / 16), swz(swizzled) {
    sbytes = swz ? (size_t)((rows + 127) / 128) * (((nb + 3) / 4 * 4) / 4) * 512 : (size_t)rows * nb;
    CK(cudaMalloc(&data, (size_t)rows * cols / 2)); CK(cudaMalloc(&sc, sbytes)); CK(cudaMemset(sc, 0, sbytes));
    CK(cudaMalloc(&g, 4)); CK(cudaMemset(g, 0, 4));
    CK(cudaMalloc(&state, mlops_adamw_qat_state_bytes(rows, cols)));
    ST(mlops_adamw_qat_state_bind(state, rows, cols, &st));
    t = {data, NVFP4A_NVFP4, sc, 0, g, nullptr, NVFP4A_GLOBAL_AMAX | (swz ? NVFP4A_SCALES_SWIZZLED : 0u), 0};
  }
  ~Q() { cudaFree(data); cudaFree(sc); cudaFree(g); cudaFree(state); }
  size_t sidx(int r, int b) const {
    if (!swz) return (size_t)r * nb + b;
    const int pc = (nb + 3) / 4 * 4;
    return ((size_t)(r >> 7) * (size_t)(pc >> 2) << 9) + (size_t)((r & 31) << 4) + (size_t)(((r >> 5) & 3) << 2) + ((size_t)(b >> 2) << 9) + (b & 3);
  }
  void init(const std::vector<float> &w) {
    float *dw; CK(cudaMalloc(&dw, w.size() * 4)); CK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
    nvfp4a_quant_config qc{}; qc.tile_rows = 1; qc.compute_amax = 1;
    ST(mlops_adamw_qat_init(rows, cols, dw, MLOPS_ADAMW_FP32, &t, &qc, &st, 0)); CK(cudaDeviceSynchronize()); cudaFree(dw);
  }
  double gdec() const {
    float amax; CK(cudaMemcpy(&amax, g, 4, cudaMemcpyDeviceToHost));
    return 1.0 / (double)std::min(448.0f * 6.0f / amax, 3.40282347e38f);
  }
};
template <class T> static std::vector<T> down(const T *p, size_t n) { std::vector<T> h(n); CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
template <class T> static void up(T *p, const std::vector<T> &h) { CK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); }

// is `got` within the two codes of `dec` around x (x widened by a relative slack for the float/double gap)?
static bool sr_ok(double got, double x, double (*dec)(uint32_t), int ncodes, double slack) {
  double lo = -1e300, hi = 1e300;
  const double xl = x - slack * fabs(x), xh = x + slack * fabs(x);
  for (int b = 0; b < ncodes; b++) for (uint32_t s : {0u, 0x80u}) {
    const double v = dec(b | s);
    if (v <= xl) lo = std::max(lo, v);  // the code below the whole slack interval ...
    if (v >= xh) hi = std::min(hi, v);  // ... and the one above it
  }
  return got >= std::min(lo, hi) && got <= std::max(lo, hi);
}

// ---- 2. one step vs double reference ----------------------------------------------------------------
static int test_reference(bool swz) {
  const int rows = 8, cols = 1024, n = rows * cols, nb = cols / 16, nblk = rows * nb;
  Q q(rows, cols, swz);
  std::mt19937 rng(swz ? 21 : 20); std::normal_distribution<float> N(0, 1); std::uniform_real_distribution<float> U(0, 1);
  std::vector<float> w(n); for (auto &x : w) x = 0.02f * N(rng);
  q.init(w);
  // a nontrivial state: random moments and qbar, positive secant accumulators on most blocks (every 7th
  // negative: no vote), CUSUMs spread over the whole range so that some voting blocks cross the threshold
  const int P = 19, t = 19 * 30 + 17;  // beta1 0.9: period (1 + b1) / (1 - b1) = 19; phase 17 -> thread block 2 votes
  const int hq = 36 * 256;
  std::vector<uint8_t> r(n), s(n); std::vector<int8_t> qb(n), sexp(nblk); std::vector<int16_t> cus(nblk);
  std::vector<uint16_t> hn(nblk), hd(nblk);
  const auto codes0 = down(q.data, n / 2);
  for (int i = 0; i < n; i++) {
    r[i] = (uint8_t)(rng() % 0x50) | (rng() & 0x80); s[i] = (uint8_t)(20 + rng() % 40);
    const double qn = e2m1((codes0[i / 2] >> (4 * (i & 1))) & 15u);
    qb[i] = (int8_t)std::lround(std::max(-127.0, std::min(127.0, 16 * qn * (0.6 + 0.4 * U(rng)) + 3 * N(rng))));
  }
  for (int b = 0; b < nblk; b++) {
    sexp[b] = (int8_t)(-30 + (int)(rng() % 5));
    cus[b] = (int16_t)((U(rng) * 2 - 1) * (hq + 19 * 256 - 16 * 19));
    const float hnum = (b % 7 ? 1 : -1) * std::exp(N(rng)) * (b % 3 ? 1e-6f : 1e-9f), hden = std::exp(N(rng)) * 1e-9f;  // every 3rd insignificant
    hn[b] = (uint16_t)(bits(hnum) >> 16); hd[b] = (uint16_t)(bits(hden) >> 16);
  }
  up(q.st.r, r); up(q.st.s, s); up(q.st.qbar, qb); up(q.st.s_exp, sexp); up(q.st.cusum, cus); up(q.st.hnum, hn); up(q.st.hden, hd);
  std::vector<float> grad(n); for (auto &x : grad) x = 1e-3f * N(rng);
  float *dg; CK(cudaMalloc(&dg, n * 4)); up(dg, grad);

  const auto u0 = down(q.st.u, n); const auto sc0 = down(q.sc, q.sbytes); const auto d0 = down(q.data, n / 2);
  const double gdec = q.gdec();
  mlops_adamw_qat_config cfg{}; cfg.lr = 3e-4f; cfg.beta1 = 0.9f; cfg.beta2 = 0.999f; cfg.eps = 1e-12f; cfg.weight_decay = 0.1f;
  cfg.step = t; cfg.grad_scale = 1; cfg.scale_z = 3; cfg.seed = 7;
  ST(mlops_adamw_qat(&cfg, rows, cols, dg, MLOPS_ADAMW_FP32, &q.t, &q.st, 0)); CK(cudaDeviceSynchronize());
  const auto u1 = down(q.st.u, n); const auto cus1 = down(q.st.cusum, nblk);
  const auto r1 = down(q.st.r, n), s1 = down(q.st.s, n); const auto qb1 = down(q.st.qbar, n), e1 = down(q.st.s_exp, nblk);
  const auto hn1 = down(q.st.hnum, nblk), hd1 = down(q.st.hden, nblk);
  const auto sc1 = down(q.sc, q.sbytes), d1 = down(q.data, n / 2);

  const double b1 = 0.9f, b2 = 0.999f, bc = sqrt(1 - pow(b2, t)) / (1 - pow(b1, t));
  const double rbc1 = 1 / (1 - pow(b1, t)), rbc1p = 1 / (1 - pow(b1, t - 1));
  int bad_s = 0, bad_r = 0, bad_u = 0, bad_qb = 0, bad_code = 0, bad_h = 0, bad_scale = 0, voted = 0, moved = 0, ties = 0;
  for (int rr = 0; rr < rows; rr++)
    for (int b = 0; b < nb; b++) {
      const int blk = rr * nb + b;
      const bool votes = (17 + blk / 64) % P == 0;
      const uint32_t kb = sc0[q.sidx(rr, b)];
      const double sm = e4m3(kb), sigma = sm * gdec;
      const double sl = e4m3(sc0[q.sidx(rr, std::max(b - 1, 0))]), sr_ = e4m3(sc0[q.sidx(rr, std::min(b + 1, nb - 1))]);
      double sum_ed = 0, sum_dd = 0, sum_v = 0, xmax = 0;
      double v[16], x[16], rn[16], uf[16], qbn[16];
      for (int j = 0; j < 16; j++) {
        const int i = blk * 16 + j;
        const double qn = e2m1((d0[i / 2] >> (4 * (i & 1))) & 15u);
        const double u = u0[i] / 4096.0, so = e5m2(s[i]) * ldexp(1, sexp[blk]), ro = e4m3(r[i]), qo = qb[i] / 16.0;
        const double mo = ro * so, g = grad[i];
        const double e = g, d = sigma * (qn - qo * rbc1p);
        sum_ed += e * d; sum_dd += d * d;
        v[j] = b2 * so * so + (1 - b2) * g * g;
        const double y = 1 / sqrt(v[j] + 1e-24);
        x[j] = v[j] * y; rn[j] = (b1 * mo + (1 - b1) * g) * y;
        xmax = std::max(xmax, x[j]);
        sum_v += v[j];
        qbn[j] = b1 * qo + (1 - b1) * qn;
        const double f = j < 8 ? (j + 8.5) / 16 : (j - 7.5) / 16;
        const double ilo = j < 8 ? 1 / (sl * gdec) : 1 / sigma, ihi = j < 8 ? 1 / sigma : 1 / (sr_ * gdec);
        const double islope = fabs(u) < 4 ? 2 : fabs(u) < 6 ? 1 : 0.5;
        uf[j] = u - 3e-4 * (rn[j] * bc + 0.1 * sigma * gof(u)) * islope * (ilo + f * (ihi - ilo));
      }
      // s exponent: the kernel's xmax is within the Quake error of ours; skip blocks at a binade edge
      const double lx = log2(xmax);
      const bool e_edge = fabs(lx - std::round(lx)) < 1e-3;
      const int E = (int)floor(lx) + 1 - 12;
      if (!e_edge && e1[blk] != E) { if (bad_s++ < 3) printf("  blk %d: E %d, ref %d\n", blk, e1[blk], E); }
      for (int j = 0; j < 16; j++) {
        const int i = blk * 16 + j;
        if (!e_edge && !sr_ok(e5m2(s1[i]), x[j] * ldexp(1, -e1[blk]), e5m2, 124, 1e-3)) {
          if (bad_s++ < 3) printf("  el %d: s %g, ref %g (E %d)\n", i, e5m2(s1[i]), x[j] * ldexp(1, -e1[blk]), e1[blk]);
        }
        if (!sr_ok(e4m3(r1[i]), rn[j], e4m3, 127, 1e-3)) bad_r++;
      }
      const double num = b2 * dec_bf16(hn[blk]) + (1 - b2) * sum_ed, den = b2 * dec_bf16(hd[blk]) + (1 - b2) * sum_dd;
      auto bf_ok = [](uint16_t got, double xv) {
        const double g = dec_bf16(got);
        const double ulp = ldexp(1, expo(fabsf((float)xv)) - 7);
        return fabs(g - xv) <= ulp * (1 + 1e-3);
      };
      if (!bf_ok(hn1[blk], num) || !bf_ok(hd1[blk], den)) bad_h++;
      // vote, CUSUM, move
      uint32_t kn = kb; bool check_scale = true;
      if (votes) {
        int vote = 0;
        const double sig = 9 * (1 - b2) / (1 + b2) * (1 + pow(b2, t)) / (1 - pow(b2, t));
        const double margin = num * num - sig * sum_v / 16 * den;
        if (num > 0 && den > 0 && fabs(margin) < 1e-3 * num * num) { ties++; check_scale = false; }
        if (num > 0 && den > 0 && margin > 0) {
          voted++;
          double best = 1e300, second = 1e300; int kbest = (int)kb;
          for (int o = -8; o <= 8; o++) {
            const int c = std::min(126, std::max(1, (int)kb + o));
            if (o > -8 && c == std::min(126, std::max(1, (int)kb + o - 1))) continue;  // clamped repeat
            const double sc = e4m3(c) * gdec;
            double D = 0;
            for (int j = 0; j < 16; j++) {
              const double ws = fabs((qbn[j] * sigma - rn[j] * x[j] * den / num) * rbc1);
              D += huber_d(ws - sc * e2m1_round_d(ws / sc), sigma);
            }
            if (D < best) { second = best; best = D; kbest = c; }
            else if (D < second) second = D;
          }
          if (second - best < 1e-3 * best) { ties++; check_scale = false; }
          vote = (kbest > (int)kb) - (kbest < (int)kb);
        }
        int S = cus[blk] + 256 * P * vote;
        const int mag = abs(S) - 16 * P;
        S = mag > 0 ? (S > 0 ? mag : -mag) : 0;
        if (abs(S) > hq) { kn = (uint32_t)std::min(126, std::max(1, (int)kb + (S > 0 ? 1 : -1))); S = 0; }
        if (check_scale && cus1[blk] != S) { bad_scale++; printf("  blk %d: cusum %d, ref %d\n", blk, cus1[blk], S); }
      } else if (cus1[blk] != cus[blk]) { bad_scale++; printf("  blk %d: silent block's cusum changed\n", blk); }
      if (check_scale && sc1[q.sidx(rr, b)] != kn) { bad_scale++; printf("  blk %d: byte %u, ref %u\n", blk, sc1[q.sidx(rr, b)], kn); }
      const uint32_t kgot = sc1[q.sidx(rr, b)];
      if (kgot != kb) moved++;
      const double ratio = sm / e4m3(kgot);
      for (int j = 0; j < 16; j++) {
        const int i = blk * 16 + j;
        const double ue = kgot != kb ? ginv(gof(uf[j]) * ratio) : uf[j], qe = qbn[j] * ratio;
        // float vs double step: 1e-3 of it; a re-encode g^-1(g(u) ratio) can amplify that by the slope change (<= 4) times ratio
        const double amp = kgot != kb ? 4 * std::max(1.0, ratio) : 1.0;
        if (fabs(u1[i] / 4096.0 - std::max(-8.0, std::min(32767 / 4096.0, ue))) > 1 / 4096.0 + amp * (1e-3 * fabs(uf[j] - u0[i] / 4096.0) + 1e-6)) {
          if (bad_u++ < 3) printf("  el %d (blk %d, moved %d): u %.6f, ref %.6f (u0 %.6f, uf %.6f)\n", i, blk, kgot != kb, u1[i] / 4096.0, ue, u0[i] / 4096.0, uf[j]);
        }
        if (fabs(qb1[i] / 16.0 - std::max(-8.0, std::min(127 / 16.0, qe))) > 1 / 16.0 + 1e-6) bad_qb++;
        const uint32_t code = (d1[i / 2] >> (4 * (i & 1))) & 15u;
        const double k = std::min(fabs(std::round(u1[i] / 4096.0)), 7.0);
        if ((code & 7u) != (uint32_t)k || (k && ((code & 8u) != 0) != (u1[i] < 0))) bad_code++;
      }
    }
  char msg[300];
  snprintf(msg, sizeof msg, "%s scales: one step vs double reference -- bad s %d, r %d, u %d, qbar %d, code %d, secant %d, "
           "scale %d (%d el / %d blocks; %d voted, %d moved, %d near-ties skipped)",
           swz ? "swizzled" : "linear  ", bad_s, bad_r, bad_u, bad_qb, bad_code, bad_h, bad_scale, n, nblk, voted, moved, ties);
  cudaFree(dg);
  return report(!bad_s && !bad_r && !bad_u && !bad_qb && !bad_code && !bad_h && !bad_scale && voted > 16 && moved > 0, msg);
}

// ---- 3. behaviour over many steps ------------------------------------------------------------------
static int test_behaviour() {
  int f = 0;
  const int rows = 64, cols = 256, n = rows * cols, nblk = n / 16;
  std::mt19937 rng(5); std::normal_distribution<float> N(0, 1); std::uniform_int_distribution<int> C(0, 15);
  std::vector<float> w(n); for (auto &x : w) x = 0.02f * N(rng);
  float *dg; CK(cudaMalloc(&dg, n * 4));
  mlops_adamw_qat_config cfg{}; cfg.lr = 3e-4f; cfg.beta1 = 0.9f; cfg.beta2 = 0.999f; cfg.eps = 1e-12f; cfg.grad_scale = 1; cfg.seed = 1;
  {  // pure noise
    Q q(rows, cols, false); q.init(w);
    const auto s0 = down(q.sc, q.sbytes);
    std::vector<float> g(n);
    for (int t = 1; t <= 2000; t++) {
      for (auto &x : g) x = 1e-3f * N(rng);
      up(dg, g); cfg.step = t; ST(mlops_adamw_qat(&cfg, rows, cols, dg, MLOPS_ADAMW_FP32, &q.t, &q.st, 0));
    }
    const auto s1 = down(q.sc, q.sbytes); int changed = 0;
    for (size_t i = 0; i < s0.size(); i++) changed += s0[i] != s1[i];
    char m[160]; snprintf(m, sizeof m, "pure-noise gradients, 2000 steps: %d of %d block scales changed (want <= 1%%)", changed, nblk);
    f += report(changed <= nblk / 100, m);
  }
  {  // optimum exactly representable 3 codes up: grad = q - target (H = 1)
    Q q(rows, cols, false); q.init(w);
    const auto s0 = down(q.sc, q.sbytes);
    const double gdec = q.gdec();
    std::vector<float> target(n);
    std::vector<uint32_t> want(nblk);
    for (int b = 0; b < nblk; b++) {
      want[b] = s0[b] + 3;
      for (int j = 0; j < 16; j++) target[b * 16 + j] = (float)(e2m1((uint32_t)C(rng)) * e4m3(want[b]) * gdec);
    }
    std::vector<float> g(n);
    cfg.lr = (float)(1e-2 * e4m3(s0[0]) * gdec);  // a hundredth of a cell per step
    for (int t = 1; t <= 3000; t++) {
      const auto d = down(q.data, n / 2), sc = down(q.sc, q.sbytes);
      for (int i = 0; i < n; i++) g[i] = (float)(e2m1((d[i / 2] >> (4 * (i & 1))) & 15u) * e4m3(sc[i / 16]) * gdec) - target[i];
      up(dg, g); cfg.step = t; ST(mlops_adamw_qat(&cfg, rows, cols, dg, MLOPS_ADAMW_FP32, &q.t, &q.st, 0));
    }
    const auto s1 = down(q.sc, q.sbytes), d = down(q.data, n / 2);
    int found = 0; double err = 0, ref = 0;
    for (int b = 0; b < nblk; b++) found += s1[b] == want[b];
    for (int i = 0; i < n; i++) {
      const double qv = e2m1((d[i / 2] >> (4 * (i & 1))) & 15u) * e4m3(s1[i / 16]) * gdec;
      err += (qv - target[i]) * (qv - target[i]); ref += (double)target[i] * target[i];
    }
    char m[200]; snprintf(m, sizeof m, "optimum on the lattice 3 codes up, 3000 steps: %d of %d blocks found that scale, rel. error %.4f",
                          found, nblk, sqrt(err / ref));
    f += report(found > nblk * 9 / 10 && sqrt(err / ref) < 0.05, m);
  }
  {  // one element under a constant gradient walks down its cells: block amax 6 -> sigma 1, cell width 1 at
     // u = 5, so lr = 0.05 in weight units is 0.05 cells per step (12 steps, before the first scale vote)
    const int R = 16, Cc = 64, M = R * Cc;
    std::vector<float> w2(M, 0.0f);
    for (int rr = 0; rr < R; rr++) for (int b = 0; b < Cc / 16; b++) w2[rr * Cc + b * 16] = 6.0f;
    const int probe = 3 * Cc + 16 + 5; w2[probe] = 3.0f;  // code 5
    Q q(R, Cc, false); q.init(w2);
    const auto u0 = down(q.st.u, M);
    std::vector<float> g(M, 0.0f); g[probe] = 1e-3f; up(dg, g);
    cfg.lr = 0.05f; cfg.weight_decay = 0;
    int code_at[13] = {};
    for (int t = 1; t <= 12; t++) {
      cfg.step = t; ST(mlops_adamw_qat(&cfg, R, Cc, dg, MLOPS_ADAMW_FP32, &q.t, &q.st, 0));
      const auto d = down(q.data, M / 2); code_at[t] = (d[probe / 2] >> (4 * (probe & 1))) & 7u;
    }
    const auto u1 = down(q.st.u, M);
    char m[200]; snprintf(m, sizeof m, "constant gradient: u %.4f -> %.4f in 12 steps (want 5 -> 4.40 +- SR noise), code 5 at step 9: %d, code 4 at step 12: %d",
                          u0[probe] / 4096.0, u1[probe] / 4096.0, code_at[9], code_at[12]);
    f += report(fabs(u1[probe] / 4096.0 - 4.40) < 0.03 && code_at[9] == 5 && code_at[12] == 4, m);
  }
  cudaFree(dg);
  return f;
}

int main() {
  int f = test_codecs();
  f += test_reference(false);
  f += test_reference(true);
  f += test_behaviour();
  printf(f ? "\n%d FAILED\n" : "\nall passed\n", f);
  return f ? 1 : 0;
}
