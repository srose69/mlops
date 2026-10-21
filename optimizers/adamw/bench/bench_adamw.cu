// mlops_adamw / mlops_adamw_qat bandwidth, n = 16Mi elements (a 4096 x 4096 weight).
//   mlops_adamw:     grad (read), param (rw), exp_avg + exp_avg_sq (rw, FP32)
//   mlops_adamw_qat: grad (read), u (rw, 2 B), r / s / qbar (rw, 1 B each), codes (rw, 0.5 B), plus per
//                    16-element block the scale byte, s_exp, cusum, hnum, hden and the next-byte scratch
// Bytes moved are counted from those formats and compared with the DRAM peak and a device-to-device
// copy of the same byte count. QAT rows run in the ALU-heaviest regime: every block's secant estimate
// is positive, so each block votes (evaluates its 17-candidate scale window) on its 1-in-1/(1-beta1)
// step. Median time per row after an idle pause (see bench_util.h).
#include "mlops_adamw.h"
#include "mlops_adamw_qat.h"
#include "bench_util.h"
#include "qat_codec.cuh"  // qat::enc_e5m2 (host)
#include <cstdlib>
#include <vector>

static size_t esz(mlops_adamw_dtype dt) { return dt == MLOPS_ADAMW_FP32 ? 4 : dt >= MLOPS_ADAMW_E4M3 ? 1 : 2; }
static const char *dtn(mlops_adamw_dtype dt) {
  return dt == MLOPS_ADAMW_FP32 ? "fp32" : dt == MLOPS_ADAMW_FP16 ? "fp16" : dt == MLOPS_ADAMW_BF16 ? "bf16" : dt == MLOPS_ADAMW_E4M3 ? "e4m3" : "e5m2";
}

__global__ void fill_normal_ish(float *x, size_t n, float scale, uint32_t seed) {  // sum of 4 uniforms, centered
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t h = (uint32_t)i * 0x9E3779B9u ^ seed;
  float acc = 0.0f;
  for (int k = 0; k < 4; ++k) { h ^= h >> 16; h *= 0x7feb352dU; h ^= h >> 15; h *= 0x846ca68bU; h ^= h >> 16; acc += (h >> 8) * 0x1p-24f; }
  x[i] = (acc - 2.0f) * scale;
}

int main() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int memory_clock_khz = 0;
  cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, 0);
  const double peak_bw = 2.0 * memory_clock_khz * 1e3 * (p.memoryBusWidth / 8) / 1e9;
  Bench bench;

  const int rows = 4096, cols = 4096;
  const size_t n = (size_t)rows * cols, nb = n / 16;
  const size_t bytes_ref = n * (4 * 2 + 4 + 4 * 2 * 2);  // the largest row: fp32 AdamW

  void *cp0, *cp1; cudaMalloc(&cp0, bytes_ref); cudaMalloc(&cp1, bytes_ref);
  const BenchRow copy = bench.run([&] { cudaMemcpyAsync(cp1, cp0, bytes_ref, cudaMemcpyDeviceToDevice); }, 200);
  const double copy_bw = 2.0 * bytes_ref / copy.ms / 1e6;
  cudaFree(cp0); cudaFree(cp1);
  printf("%s: n=%zu elements; DRAM peak %.0f GB/s (%d-bit bus), D2D copy %.0f GB/s\n\n", p.name, n, peak_bw,
         p.memoryBusWidth, copy_bw);
  printf("%-34s %9s %8s %11s %8s %8s %9s\n", "", "time", "B/elem", "bytes/s", "of DRAM", "of copy", "SM clock");
  auto row = [&](const char *name, const BenchRow &x, double bytes) {
    const double gbs = bytes / x.ms / 1e6;
    printf("%-34s %6.3f ms %8.2f %8.1f GB/s %7.0f%% %7.0f%% %5.0f MHz\n", name, x.ms, bytes / n, gbs, 100 * gbs / peak_bw,
           100 * gbs / copy_bw, x.mhz);
    fflush(stdout);
  };

  mlops_adamw_config cfg{};
  cfg.lr = 1e-3f; cfg.beta1 = 0.9f; cfg.beta2 = 0.999f; cfg.eps = 1e-8f; cfg.weight_decay = 0.01f;
  cfg.step = 1; cfg.bias_correction = 1; cfg.grad_scale = 1.0f;
  for (mlops_adamw_dtype pdt : {MLOPS_ADAMW_FP32, MLOPS_ADAMW_BF16, MLOPS_ADAMW_FP16})
    for (mlops_adamw_dtype gdt : {MLOPS_ADAMW_FP32, MLOPS_ADAMW_BF16, MLOPS_ADAMW_FP16}) {
      char name[48]; snprintf(name, sizeof name, "adamw param=%s grad=%s", dtn(pdt), dtn(gdt));
      if (!bench_want(name)) continue;
      void *param, *grad; float *m, *v;
      cudaMalloc(&param, n * esz(pdt)); cudaMalloc(&grad, n * esz(gdt));
      cudaMalloc(&m, n * 4); cudaMalloc(&v, n * 4);
      cudaMemset(param, 0, n * esz(pdt)); cudaMemset(grad, 0, n * esz(gdt)); cudaMemset(m, 0, n * 4); cudaMemset(v, 0, n * 4);
      row(name, bench.run([&] { mlops_adamw(&cfg, n, param, pdt, grad, gdt, m, v, 0); }, 200),
          (double)n * (esz(pdt) * 2 + esz(gdt) + 4 * 2 * 2));
      cudaFree(param); cudaFree(grad); cudaFree(m); cudaFree(v);
    }

  // ---- QAT ----------------------------------------------------------------------------------------
  float *w, *gf; cudaMalloc(&w, n * 4); cudaMalloc(&gf, n * 4);
  fill_normal_ish<<<(n + 255) / 256, 256>>>(w, n, 0.02f, 1);
  fill_normal_ish<<<(n + 255) / 256, 256>>>(gf, n, 1e-3f, 2);
  uint8_t *g8; cudaMalloc(&g8, n);  // FP8 gradient bytes: values ~1e-3 scaled by 2^10 (grad_scale undoes it)
  {
    std::vector<float> h(n); std::vector<uint8_t> hb(n);
    cudaMemcpy(h.data(), gf, n * 4, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n; ++i) hb[i] = (uint8_t)qat::enc_e5m2(h[i] * 1024.0f);
    cudaMemcpy(g8, hb.data(), n, cudaMemcpyHostToDevice);
  }
  uint8_t *qd, *qs; float *qg; void *sb;
  cudaMalloc(&qd, n / 2); cudaMalloc(&qs, nb); cudaMalloc(&qg, 4);
  const size_t sbytes = mlops_adamw_qat_state_bytes(rows, cols);
  cudaMalloc(&sb, sbytes);
  mlops_adamw_qat_state st; mlops_adamw_qat_state_bind(sb, rows, cols, &st);
  nvfp4a_tensor q{qd, NVFP4A_NVFP4, qs, 0, qg, nullptr, NVFP4A_GLOBAL_AMAX, 0};
  nvfp4a_quant_config qc{}; qc.tile_rows = 1; qc.compute_amax = 1;
  if (mlops_adamw_qat_init(rows, cols, w, MLOPS_ADAMW_FP32, &q, &qc, &st, 0) != MLOPS_ADAMW_OK) {
    fprintf(stderr, "qat_init: %s\n", mlops_adamw_last_error()); return 1;
  }
  {  // positive secant accumulators everywhere (BF16 1.0): every voting block runs its scale window
    std::vector<uint16_t> one(nb, 0x3F80);
    cudaMemcpy(st.hnum, one.data(), nb * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(st.hden, one.data(), nb * 2, cudaMemcpyHostToDevice);
  }
  mlops_adamw_qat_config qcfg{};
  qcfg.lr = 1e-3f; qcfg.beta1 = 0.9f; qcfg.beta2 = 0.999f; qcfg.eps = 1e-12f; qcfg.weight_decay = 0.1f;
  qcfg.step = 100; qcfg.grad_scale = 1; qcfg.seed = 1;
  const double per_block = 1 /* scale */ + 2 /* s_exp */ + 4 /* cusum */ + 8 /* hnum, hden */ + 3 /* next + commit */;
  for (mlops_adamw_dtype gdt : {MLOPS_ADAMW_FP32, MLOPS_ADAMW_E5M2}) {
    char name[48]; snprintf(name, sizeof name, "adamw_qat grad=%s", dtn(gdt));
    if (!bench_want(name)) continue;
    const void *g = gdt == MLOPS_ADAMW_FP32 ? (const void *)gf : (const void *)g8;
    qcfg.grad_scale = gdt == MLOPS_ADAMW_FP32 ? 1.0f : 1.0f / 1024;
    // one step per call; the step number advances so the vote phase rotates as in training
    row(name, bench.run([&] { ++qcfg.step; mlops_adamw_qat(&qcfg, rows, cols, g, gdt, &q, &st, 0); }, 100),
        (double)n * (esz(gdt) + 2 * (2 + 1 + 1 + 1 + 0.5)) + (double)nb * per_block);
  }
  return 0;
}
