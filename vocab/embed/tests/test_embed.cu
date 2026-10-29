// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_embed against a host emulation, bit for bit: the rows, xw = bf16(x * w), the parts; every variant and
// element count (the outputs do not depend on the split); a table in device and in mapped host memory; ids out
// of range.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "mlops_embed.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static std::mt19937 rng(99);
static uint16_t bf16(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) u = 0x7fc00000u;
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float f32(uint16_t v) {
  const uint32_t u = (uint32_t)v << 16;
  float x;
  memcpy(&x, &u, 4);
  return x;
}
template <class T> static T *up(const std::vector<T> &h) {
  T *d;
  CK(cudaMalloc(&d, h.size() * sizeof(T) + 16));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
template <class T> static std::vector<T> down(const T *d, size_t n) {
  std::vector<T> h(n);
  CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}
static int report(const char *what, long bad, long n) {
  printf("[%s] %s (%ld of %ld differ)\n", bad ? "FAIL" : "PASS", what, bad, n);
  return bad != 0;
}

static float unit_sq(const float *v) {  // 256 elements: 32 lanes of 8, in order, then the butterfly
  float lane[32];
  for (int l = 0; l < 32; ++l) {
    float s = 0.f;
    for (int e = 0; e < 8; ++e) s = s + v[8 * l + e] * v[8 * l + e];
    lane[l] = s;
  }
  for (int o = 16; o; o >>= 1) {
    float nxt[32];
    for (int l = 0; l < 32; ++l) nxt[l] = lane[l] + lane[l ^ o];
    memcpy(lane, nxt, sizeof lane);
  }
  return lane[0];
}

static int run_case(int vocab, int hidden, int n_tok, int items, bool mapped) {
  std::normal_distribution<float> d(0.f, 1.f);
  std::vector<uint16_t> table((size_t)vocab * hidden), w(hidden);
  for (auto &v : table) v = bf16(d(rng));
  for (auto &v : w) v = bf16(d(rng) * 0.5f + 1.f);
  std::vector<int> ids(n_tok);
  for (auto &i : ids) i = rng() % vocab;
  const int per = hidden / 256;
  const uint16_t *dtable;
  if (mapped) {
    uint16_t *h;
    CK(cudaHostAlloc((void **)&h, table.size() * 2, cudaHostAllocMapped));
    memcpy(h, table.data(), table.size() * 2);
    CK(cudaHostGetDevicePointer((void **)&dtable, h, 0));
  } else {
    dtable = up(table);
  }
  const mlops_embed_config c{vocab, hidden, n_tok, items};
  const int nv = mlops_embed_variants(&c, nullptr, 0);
  std::vector<mlops_embed_variant> vs(nv);
  mlops_embed_variants(&c, vs.data(), nv);
  int *dids = up(ids), *derr;
  const uint16_t *dw = up(w);
  float *dx, *dparts;
  uint16_t *dxw;
  CK(cudaMalloc(&dx, (size_t)n_tok * hidden * 4));
  CK(cudaMalloc(&dxw, (size_t)n_tok * hidden * 2));
  CK(cudaMalloc(&dparts, (size_t)n_tok * per * 4));
  CK(cudaMalloc(&derr, 4));
  long bad = 0, n = 0;
  for (const auto &v : vs) {
    CK(cudaMemset(derr, 0, 4));
    CK(cudaMemset(dx, 0xff, (size_t)n_tok * hidden * 4));
    CK(cudaMemset(dxw, 0xff, (size_t)n_tok * hidden * 2));
    if (mlops_embed(&c, &v, dtable, dids, dw, dx, dxw, dparts, derr, 0)) {
      fprintf(stderr, "%s\n", mlops_embed_last_error());
      return 1;
    }
    CK(cudaDeviceSynchronize());
    const auto x = down(dx, (size_t)n_tok * hidden);
    const auto xw = down(dxw, (size_t)n_tok * hidden);
    const auto parts = down(dparts, (size_t)n_tok * per);
    for (int col = 0; col < n_tok; ++col) {
      std::vector<float> row(hidden);
      for (int k = 0; k < hidden; ++k) row[k] = f32(table[(size_t)ids[col] * hidden + k]);
      for (int k = 0; k < hidden; ++k) {
        uint32_t u0, u1;
        memcpy(&u0, &x[(size_t)col * hidden + k], 4);
        memcpy(&u1, &row[k], 4);
        bad += u0 != u1 || xw[(size_t)col * hidden + k] != bf16(row[k] * f32(w[k]));
        n += 2;
      }
      for (int u = 0; u < per; ++u) {
        const float ref = unit_sq(row.data() + 256 * u);
        uint32_t u0, u1;
        memcpy(&u0, &parts[(size_t)col * per + u], 4);
        memcpy(&u1, &ref, 4);
        bad += u0 != u1;
        ++n;
      }
    }
    int err;
    CK(cudaMemcpy(&err, derr, 4, cudaMemcpyDeviceToHost));
    bad += err != 0;
    ++n;
  }
  char what[160];
  snprintf(what, sizeof what, "embed %d x %d, n %d, items %d, %s table: %d variants", vocab, hidden, n_tok, items, mapped ? "mapped host" : "device", nv);
  return report(what, bad, n);
}

static int run_ids_and_invalid() {
  const int vocab = 50, hidden = 512, n_tok = 4;
  std::vector<uint16_t> table((size_t)vocab * hidden, bf16(1.f));
  std::vector<int> ids = {3, -1, vocab, 7};
  uint16_t *dt = up(table);
  int *did = up(ids), *derr;
  float *dx;
  CK(cudaMalloc(&dx, (size_t)n_tok * hidden * 4));
  CK(cudaMalloc(&derr, 4));
  CK(cudaMemset(derr, 0, 4));
  const mlops_embed_config c{vocab, hidden, n_tok, 0};
  mlops_embed_variant v[4];
  const int nv = mlops_embed_variants(&c, v, 4);
  long bad = nv != 4, n = 1;
  bad += mlops_embed(&c, &v[0], dt, did, nullptr, dx, nullptr, nullptr, derr, 0) != MLOPS_EMBED_OK;
  CK(cudaDeviceSynchronize());
  const auto x = down(dx, (size_t)n_tok * hidden);
  int err;
  CK(cudaMemcpy(&err, derr, 4, cudaMemcpyDeviceToHost));
  bad += err != 1;
  for (int col = 0; col < n_tok; ++col)
    for (int k = 0; k < hidden; ++k) bad += x[(size_t)col * hidden + k] != ((col == 1 || col == 2) ? 0.f : 1.f), ++n;
  const mlops_embed_config broken[] = {{0, 512, 1, 0}, {50, 500, 1, 0}, {50, 512, 0, 0}, {50, 512, 1, -1}};
  for (const auto &b : broken) { bad += mlops_embed_variants(&b, nullptr, 0) != 0; ++n; }
  bad += mlops_embed(&c, &v[0], nullptr, did, nullptr, dx, nullptr, nullptr, nullptr, 0) != MLOPS_EMBED_INVALID_VALUE;
  bad += mlops_embed(&c, &v[0], dt, did, nullptr, nullptr, nullptr, nullptr, nullptr, 0) != MLOPS_EMBED_INVALID_VALUE;  // no output
  bad += mlops_embed(&c, &v[0], dt, did, nullptr, dx, dx ? (uint16_t *)dx : nullptr, nullptr, nullptr, 0) != MLOPS_EMBED_INVALID_VALUE;  // xw without norm_w
  bad += mlops_embed(&c, &v[0], dt + 1, did, nullptr, dx, nullptr, nullptr, nullptr, 0) != MLOPS_EMBED_INVALID_VALUE;     // misaligned
  mlops_embed_variant odd{3, 96, 0, 0, 0};
  bad += mlops_embed(&c, &odd, dt, did, nullptr, dx, nullptr, nullptr, nullptr, 0) != MLOPS_EMBED_INVALID_VALUE;
  n += 5;
  return report("ids out of range, invalid arguments", bad, n);
}

int main() {
  int f = 0;
  f += run_case(1000, 1024, 1, 0, false);
  f += run_case(151936, 1024, 1, 0, false);   // Qwen3-0.6B
  f += run_case(5000, 4096, 1, 0, false);     // Qwen3-8B's hidden size
  f += run_case(3000, 2560, 8, 0, false);     // eight columns, 10 units each
  f += run_case(3000, 2560, 8, 3, false);     // fewer elements than units
  f += run_case(3000, 2560, 3, 1000, false);  // more elements than units
  f += run_case(4000, 2048, 2, 0, true);      // mapped host memory
  f += run_ids_and_invalid();
  printf(f ? "\n%d FAILED\n" : "\nALL PASS\n", f);
  return f != 0;
}
