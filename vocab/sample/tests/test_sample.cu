// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_sample against a host emulation: greedy against the sorted order; with temperature the kept set by
// top-k / top-p (double masses), the draw frequencies within 5 sigma, and the same token on any grid.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

#include "mlops_lm_head.h"
#include "mlops_sample.h"
#include "mlops_vocab_items.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static std::mt19937 rng(1234);

template <class T> static T *up(const std::vector<T> &h) {
  T *d;
  CK(cudaMalloc(&d, h.size() * sizeof(T) + 16));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

struct Sampler {
  mlops_sample_config cfg;
  const float *logits;
  const uint32_t *seen;
  mlops_sample_state *state;
  void *workspace;
  size_t bytes;
  int *token;

  Sampler(const mlops_sample_config &c, const std::vector<float> &l, const std::vector<uint32_t> &s, uint64_t seed) : cfg(c) {
    logits = up(l);
    seen = up(s);
    bytes = mlops_sample_workspace_bytes(&c);
    CK(cudaMalloc(&workspace, bytes));
    CK(cudaMemset(workspace, 0, bytes));
    CK(cudaMalloc(&state, sizeof(mlops_sample_state)));
    CK(cudaMalloc(&token, sizeof(int)));
    reset(seed, 0);
  }
  void reset(uint64_t seed, uint64_t step) {
    const mlops_sample_state s{seed, step};
    CK(cudaMemcpy(state, &s, sizeof s, cudaMemcpyHostToDevice));
  }
  int draw() {
    if (mlops_sample(&cfg, logits, seen, nullptr, state, workspace, bytes, token, 0)) {
      fprintf(stderr, "%s\n", mlops_sample_last_error());
      exit(1);
    }
    int t;
    CK(cudaMemcpy(&t, token, sizeof t, cudaMemcpyDeviceToHost));
    return t;
  }
  uint64_t step() {
    mlops_sample_state s;
    CK(cudaMemcpy(&s, state, sizeof s, cudaMemcpyDeviceToHost));
    return s.step;
  }
};

// The sampled logits on the host (the same IEEE operations), in descending order (index up on ties).
struct HostSample {
  std::vector<float> l;
  std::vector<int> order;
  HostSample(const std::vector<float> &logits, const std::vector<uint32_t> &seen, const mlops_sample_config &c) : l(logits) {
    for (size_t i = 0; i < l.size(); ++i) {
      if (c.penalty != 1.f && (seen[i >> 5] >> (i & 31) & 1u)) l[i] = l[i] > 0.f ? l[i] * (1.f / c.penalty) : l[i] * c.penalty;
      if (c.temperature > 0.f) l[i] = l[i] * (1.f / c.temperature);
    }
    order.resize(l.size());
    for (size_t i = 0; i < l.size(); ++i) order[i] = (int)i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return l[a] > l[b]; });
  }
};

static int report(const char *what, long bad, long n) {
  printf("[%s] %s (%ld of %ld differ)\n", bad ? "FAIL" : "PASS", what, bad, n);
  return bad != 0;
}

static int run_case(const char *name, mlops_sample_config c, int draws) {
  std::normal_distribution<float> d(0.f, 2.5f);
  std::vector<float> logits(c.vocab);
  for (auto &x : logits) x = d(rng);
  logits[c.vocab / 2] = logits[c.vocab / 3] = 11.f;  // a tie at the top (greedy: the lower index)
  std::vector<uint32_t> seen((c.vocab + 31) / 32, 0u);
  for (int i = 0; i < c.vocab; i += 7) seen[i >> 5] |= 1u << (i & 31);
  Sampler s(c, logits, seen, 1 + (uint64_t)c.top_k);
  const HostSample h(logits, seen, c);
  long bad = 0, n = 0;
  if (c.temperature <= 0.f) {
    bad += s.draw() != h.order[0];
    bad += s.step() != 1;  // the call advanced the step
    n += 2;
  } else {
    // The kept set by the parameters (top-p by double masses: the boundary token may go either way).
    const double M = h.l[h.order[0]];
    size_t cand = c.top_k > 0 ? std::min((size_t)c.top_k, h.order.size()) : h.order.size();
    while (cand < h.order.size() && h.l[h.order[cand]] == h.l[h.order[cand - 1]]) ++cand;  // ties at the k-th: kept
    double z = 0.;
    for (size_t i = 0; i < cand; ++i) z += exp((double)h.l[h.order[i]] - M);
    size_t keep = cand;
    if (c.top_p < 1.f) {
      double before = 0.;
      for (size_t i = 0; i < cand; ++i) {
        if (i && before >= c.top_p * z) { keep = i; break; }
        before += exp((double)h.l[h.order[i]] - M);
      }
    }
    std::vector<double> prob(c.vocab, 0.);
    double kz = 0.;
    for (size_t i = 0; i < keep; ++i) kz += exp((double)h.l[h.order[i]] - M);
    for (size_t i = 0; i < keep; ++i) prob[h.order[i]] = exp((double)h.l[h.order[i]] - M) / kz;
    const std::set<int> allowed(h.order.begin(), h.order.begin() + std::min(keep + 1, h.order.size()));
    std::vector<int> count(c.vocab, 0);
    for (int i = 0; i < draws; ++i) {
      const int t = s.draw();
      bad += !allowed.count(t);
      ++count[t];
      ++n;
    }
    bad += s.step() != (uint64_t)draws;
    ++n;
    for (int ctas : {1, 5, 40}) {  // the same draw on any grid
      s.reset(77, 7);
      const int ref = s.draw();
      s.cfg.ctas = ctas;
      s.reset(77, 7);
      bad += s.draw() != ref;
      s.cfg.ctas = c.ctas;
      ++n;
    }
    for (size_t i = 0; i < std::min(keep, (size_t)20); ++i) {  // frequencies within 5 sigma
      const int t = h.order[i];
      const double e = prob[t] * draws, sd = sqrt(draws * prob[t] * (1 - prob[t])) + 1;
      bad += fabs(count[t] - e) > 5 * sd;
      ++n;
    }
  }
  char what[160];
  snprintf(what, sizeof what, "sample %s vocab %d (T %.2f, top-k %d, top-p %.2f, penalty %.2f)", name, c.vocab, c.temperature, c.top_k,
           c.top_p, c.penalty);
  return report(what, bad, n);
}

static int run_invalid() {
  long bad = 0, n = 0;
  float *l;
  CK(cudaMalloc(&l, 64));
  mlops_sample_state *st;
  CK(cudaMalloc(&st, sizeof *st));
  int *tok;
  CK(cudaMalloc(&tok, 4));
  void *ws;
  CK(cudaMalloc(&ws, 1 << 20));
  const mlops_sample_config good{1000, 1.f, 0, 1.f, 1.f, 0};
  mlops_sample_config bad_cfg[] = {{0, 1.f, 0, 1.f, 1.f, 0}, {1000, -1.f, 0, 1.f, 1.f, 0}, {1000, 1.f, -1, 1.f, 1.f, 0},
                                   {1000, 1.f, 0, 0.f, 1.f, 0}, {1000, 1.f, 0, 1.f, 0.f, 0}, {1000, 1.f, 0, 1.f, 1.f, -1}};
  for (const auto &c : bad_cfg) {
    bad += mlops_sample(&c, l, nullptr, nullptr, st, ws, 1 << 20, tok, 0) != MLOPS_SAMPLE_INVALID_VALUE;
    bad += mlops_sample_workspace_bytes(&c) != 0;
    n += 2;
  }
  bad += mlops_sample(&good, l, nullptr, nullptr, st, ws, 16, tok, 0) != MLOPS_SAMPLE_INVALID_VALUE;       // workspace too small
  bad += mlops_sample(&good, nullptr, nullptr, nullptr, st, ws, 1 << 20, tok, 0) != MLOPS_SAMPLE_INVALID_VALUE;
  mlops_sample_config pen = good;
  pen.penalty = 1.3f;
  bad += mlops_sample(&pen, l, nullptr, nullptr, st, ws, 1 << 20, tok, 0) != MLOPS_SAMPLE_INVALID_VALUE;
  const mlops_sample_stats st_ok{(const float *)l, (const int *)l, 1};
  bad += mlops_sample(&pen, l, nullptr, &st_ok, st, ws, 1 << 20, tok, 0) != MLOPS_SAMPLE_INVALID_VALUE;        // stats with a penalty
  const mlops_sample_stats st_bad{nullptr, nullptr, 0};
  bad += mlops_sample(&good, l, nullptr, &st_bad, st, ws, 1 << 20, tok, 0) != MLOPS_SAMPLE_INVALID_VALUE;     // penalty without seen
  n += 5;
  return report("invalid arguments", bad, n);
}

// The head's statistics into the sampler against the sampler scanning the logits: the same tokens for the same draws.
static int run_fused(const char *name, mlops_sample_config sc, int hidden, int draws) {
  std::normal_distribution<float> d(0.f, 0.06f);
  std::vector<uint16_t> w((size_t)sc.vocab * hidden), x(hidden);
  auto bf = [](float v) { uint32_t u; memcpy(&u, &v, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); };
  for (auto &v : w) v = bf(d(rng));
  for (auto &v : x) v = bf(d(rng) * 12.f);
  uint16_t *dw = up(w), *dx = up(x);
  const float inv_t = sc.temperature > 0.f ? 1.f / sc.temperature : 1.f;
  const mlops_lm_head_config plain{sc.vocab, hidden, 1, 0, 0, 0.f, 0.f, 1.f}, scaled{sc.vocab, hidden, 1, 0, 0, 0.f, 0.f, inv_t};
  mlops_lm_head_variant v;
  if (mlops_lm_head_variants(&plain, &v, 1) < 1) return 1;
  const int items = mlops_lm_head_item_count(&scaled, &v);
  float *l_plain, *l_scaled, *smax;
  int *sarg;
  unsigned *shist;
  CK(cudaMalloc(&l_plain, (size_t)sc.vocab * 4));
  CK(cudaMalloc(&l_scaled, (size_t)sc.vocab * 4));
  CK(cudaMalloc(&smax, items * 4));
  CK(cudaMalloc(&sarg, items * 4));
  CK(cudaMalloc(&shist, (size_t)items * 256 * 4));
  if (mlops_lm_head(&plain, &v, dw, dx, nullptr, l_plain, nullptr, nullptr, nullptr, 0) || mlops_lm_head(&scaled, &v, dw, dx, nullptr, l_scaled, smax, sarg, shist, 0)) return 1;
  std::vector<float> host((size_t)sc.vocab);
  CK(cudaMemcpy(host.data(), l_plain, host.size() * 4, cudaMemcpyDeviceToHost));
  Sampler a(sc, host, std::vector<uint32_t>((sc.vocab + 31) / 32, 0u), 5), b = a;
  b.logits = l_scaled;
  const mlops_sample_stats stats{smax, sarg, items, shist};
  b.state = nullptr;
  CK(cudaMalloc(&b.state, sizeof(mlops_sample_state)));
  b.reset(5, 0);
  long bad = 0;
  for (int i = 0; i < draws; ++i) {
    const int t0 = a.draw();
    if (mlops_sample(&b.cfg, l_scaled, nullptr, &stats, b.state, b.workspace, b.bytes, b.token, 0)) { fprintf(stderr, "%s\n", mlops_sample_last_error()); return 1; }
    int t1;
    CK(cudaMemcpy(&t1, b.token, 4, cudaMemcpyDeviceToHost));
    bad += t0 != t1;
  }
  char what[160];
  snprintf(what, sizeof what, "head statistics into the sampler, %s (vocab %d, %d elements)", name, sc.vocab, items);
  return report(what, bad, draws);
}

// `test_sample bench`: the time of a call at Qwen3's vocabulary on the device, per configuration, with and without the
// head's statistics (the logits scanned for the maximum, or not).
static void bench(int vocab) {
  std::normal_distribution<float> d(0.f, 2.5f);
  std::vector<float> logits(vocab);
  for (auto &x : logits) x = d(rng);
  const std::vector<uint32_t> seen((vocab + 31) / 32, 0x01010101u);
  struct Row { const char *name; mlops_sample_config c; };
  const Row rows[] = {{"greedy", {vocab, 0.f, 0, 1.f, 1.f, 0}},
                      {"greedy, penalty 1.1", {vocab, 0.f, 0, 1.f, 1.1f, 0}},
                      {"temperature", {vocab, 0.7f, 0, 1.f, 1.f, 0}},
                      {"top-k 20, top-p 0.95", {vocab, 0.6f, 20, 0.95f, 1.f, 0}},
                      {"top-p 0.9", {vocab, 1.f, 0, 0.9f, 1.f, 0}},
                      {"top-k 2000", {vocab, 1.f, 2000, 1.f, 1.f, 0}}};
  {
    printf("%-24s %6s %12s %16s %20s\n", "configuration", "passes", "scan, us", "head stats, us", "+ counts (passes)");
  for (const Row &r : rows) {
    Sampler s(r.c, logits, seen, 1);
    s.cfg.penalty = r.c.penalty;
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    auto time = [&](const mlops_sample_stats *stats) {
      for (int i = 0; i < 5; ++i) mlops_sample(&s.cfg, s.logits, r.c.penalty != 1.f ? s.seen : nullptr, stats, s.state, s.workspace, s.bytes, s.token, 0);
      CK(cudaEventRecord(a));
      for (int i = 0; i < 200; ++i) mlops_sample(&s.cfg, s.logits, r.c.penalty != 1.f ? s.seen : nullptr, stats, s.state, s.workspace, s.bytes, s.token, 0);
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      float ms;
      CK(cudaEventElapsedTime(&ms, a, b));
      return ms / 200 * 1e3;
    };
    const double scan = time(nullptr);
    char with[32] = "-", counts[48] = "-";
    if (r.c.penalty == 1.f) {
      std::vector<float> smax(64, 0.f);
      std::vector<int> sarg(64, 0);
      std::vector<unsigned> shist(64 * 256, 0u);
      const int per = (vocab + 63) / 64;
      for (int i = 0; i < vocab; ++i) {
        const float v = r.c.temperature > 0.f ? logits[i] * (1.f / r.c.temperature) : logits[i];
        if (i % per == 0 || v > smax[i / per]) { smax[i / per] = v; sarg[i / per] = i; }
        uint32_t u;
        memcpy(&u, &v, 4);
        ++shist[(size_t)(i / per) * 256 + ((u & 0x80000000u ? ~u : u | 0x80000000u) >> 24)];
      }
      if (r.c.temperature > 0.f) {
        std::vector<float> sampled(logits);
        for (auto &x : sampled) x *= 1.f / r.c.temperature;
        CK(cudaMemcpy((void *)s.logits, sampled.data(), sampled.size() * 4, cudaMemcpyHostToDevice));
      }
      const mlops_sample_stats st{up(smax), up(sarg), 64, nullptr}, st_hist{st.max, st.arg, 64, up(shist)};
      snprintf(with, sizeof with, "%.1f", time(&st));
      if (r.c.top_k > 0) {
        mlops_vocab_items fewer[8];
        const int np_hist = mlops_sample_items(&s.cfg, s.logits, nullptr, &st_hist, s.state, s.workspace, s.bytes, s.token, fewer, 8);
        snprintf(counts, sizeof counts, "%.1f (%d)", time(&st_hist), np_hist);
      }
    }
    mlops_vocab_items passes[8];
    const int np = mlops_sample_items(&s.cfg, s.logits, r.c.penalty != 1.f ? s.seen : nullptr, nullptr, s.state, s.workspace, s.bytes, s.token, passes, 8);
    printf("%-24s %6d %12.1f %16s %20s\n", r.name, np, scan, with, counts);
  }
  }
}

int main(int argc, char **argv) {
  if (argc > 1 && !strcmp(argv[1], "bench")) {
    bench(151936);
    return 0;
  }
  int f = 0;
  f += run_case("greedy", {151936, 0.f, 0, 1.f, 1.3f, 0}, 1);
  f += run_case("top-k, top-p", {151936, 0.6f, 20, 0.95f, 1.1f, 0}, 3000);
  f += run_case("top-p", {50000, 1.f, 0, 0.9f, 1.f, 0}, 3000);
  f += run_case("temperature", {9000, 0.5f, 0, 1.f, 1.f, 0}, 3000);
  f += run_case("large top-k", {30000, 1.f, 2000, 1.f, 1.f, 0}, 3000);
  f += run_case("one chunk", {1500, 0.8f, 40, 0.9f, 1.f, 0}, 3000);
  f += run_fused("greedy", {151936, 0.f, 0, 1.f, 1.f, 0}, 1024, 20);
  f += run_fused("temperature, top-k", {151936, 0.7f, 30, 1.f, 1.f, 0}, 1024, 400);
  f += run_fused("top-p", {50000, 1.f, 0, 0.9f, 1.f, 0}, 1024, 400);
  f += run_fused("large top-k, with the head's counts", {151936, 1.f, 2000, 1.f, 1.f, 0}, 1024, 200);
  f += run_fused("top-k, top-p, with the head's counts", {151936, 0.6f, 20, 0.95f, 1.f, 0}, 1024, 200);
  f += run_invalid();
  printf(f ? "\n%d FAILED\n" : "\nALL PASS\n", f);
  return f != 0;
}
