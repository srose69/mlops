// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE, LICENSE-EXCEPTION (additional terms).
// Execute the public work-item plans through NVRTC, with fewer CTAs than items and shared-memory reuse.
// Compare with the stream API bit for bit; check query/truncation, range bounds, and each item's stores.
#include "mlops_fbla_items.h"
#include "fbla_qkv.cuh"
#include <cuda.h>
#include <nvrtc.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
#define CUDA(x) do { auto e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
#define DRIVER(x) do { auto e = (x); if (e != CUDA_SUCCESS) { const char *s; cuGetErrorString(e, &s); fprintf(stderr, "%s: %s\n", #x, s); exit(1); } } while (0)
#define RTC(x) CHECK((x) == NVRTC_SUCCESS)

namespace {
constexpr int D = MLOPS_FBLA_HEAD_DIM, G = MLOPS_FBLA_GROUP, ARGS = MLOPS_FBLA_ARG_PART_O + 1;
using Bytes = std::vector<unsigned char>;
struct Buffer {
  void *p = nullptr;
  size_t size;
  explicit Buffer(size_t n) : size(n) { CUDA(cudaMalloc(&p, n + 128)); CUDA(cudaMemset(p, 0, n + 128)); }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  template <class T> T *as() { return static_cast<T *>(p); }
  Bytes read() const { Bytes b(size + 128); CUDA(cudaMemcpy(b.data(), p, b.size(), cudaMemcpyDeviceToHost)); return b; }
  void set(const void *src, size_t n) { CHECK(n <= size); CUDA(cudaMemcpy(p, src, n, cudaMemcpyHostToDevice)); }
};
std::mt19937 rng(7301);
void random_bf16(Buffer &b) {
  std::normal_distribution<float> n(0.f, 1.5f);
  std::vector<uint16_t> v(b.size / 2);
  for (auto &x : v) { float f = n(rng); unsigned u; memcpy(&u, &f, 4); x = (u + 0x7fff + ((u >> 16) & 1)) >> 16; }
  b.set(v.data(), b.size);
}

struct Program { CUmodule module; CUfunction fn; };
struct Copy { void *dst; const void *src; unsigned long long bytes; };
bool publication = false;
bool bad_count = false;
bool benchmark = false;
bool readonly_control = false;
bool omit_count_range = false;
std::map<std::string, Program> programs;
Program executor(const mlops_fbla_items &p) {
  const std::string key = std::string(p.entry) + ":" + std::to_string(p.arch);
  auto old = programs.find(key);
  if (old != programs.end()) return old->second;
  std::string shared, call = std::string(p.entry) + "(a";
  using namespace mlops_fbla_detail;
  Args d{}; PrefillArgs a{};
  if (p.kind >= MLOPS_FBLA_ITEM_COARSE && p.kind <= MLOPS_FBLA_ITEM_MERGE) memcpy(&d, p.args.bytes, sizeof d);
  else if (p.kind != MLOPS_FBLA_ITEM_QKV) memcpy(&a, p.args.bytes, sizeof a);
  const int qg = d.kv_heads ? d.q_heads / d.kv_heads / d.vg : 0;
  switch (p.kind) {
    case MLOPS_FBLA_ITEM_CLOSE: call += ", *reinterpret_cast<unsigned char (*)[2][128][128]>(sm)"; break;
    case MLOPS_FBLA_ITEM_FINISH: break;
    case MLOPS_FBLA_ITEM_STREAM:
      call += std::string(", *reinterpret_cast<mlops_fbla_detail::StreamSmem<mlops_fbla_detail::Warps<") +
              (p.geom[23] == record_bytes<3>() ? "3" : "4") + ">::PROWS> *>(sm)"; break;
    case MLOPS_FBLA_ITEM_TRANSCODE:
      call += ", (unsigned char *)g[3], g[4], g[5], (int)g[6], (bool)g[7], sm"; break;
    case MLOPS_FBLA_ITEM_TC:
      call += ", (const unsigned char *)g[3], g[4], g[5], (int)g[6], (float2 *)g[7], (float *)g[8], sm"; break;
    case MLOPS_FBLA_ITEM_TC_MERGE: call += ", (const float2 *)g[3], (const float *)g[4], (int)g[5]"; break;
    case MLOPS_FBLA_ITEM_COARSE:
      call += ", *reinterpret_cast<mlops_fbla_detail::SmemCoarse" +
              (qg == 4 ? std::string("4") : "<" + std::to_string(qg) + ">") + " *>(sm)"; break;
    case MLOPS_FBLA_ITEM_REFINE: {
      const std::string slices = d.nslices == 1 ? "1" : "mlops_fbla_detail::MAX_SLICES";
      call += ", *reinterpret_cast<mlops_fbla_detail::SmemRefine" +
              (qg == 4 ? "4Storage<" + slices + ">" : "<" + std::to_string(qg) + ", " + slices + ">") + " *>(sm)";
      break;
    }
    case MLOPS_FBLA_ITEM_SELECT:
      call += ", *reinterpret_cast<float (*)[" + std::to_string(qg) + "]>(sm), *reinterpret_cast<int (*)[mlops_fbla_detail::W]>(sm + " +
              std::to_string(qg * 4) + ")"; break;
    case MLOPS_FBLA_ITEM_MERGE: call += ", *reinterpret_cast<mlops_fbla_detail::SmemMerge *>(sm)"; break;
    case MLOPS_FBLA_ITEM_QKV: call += ", mlops_fbla::Pos::x(), *reinterpret_cast<mlops_fbla_detail::QkvSmem *>(sm)"; break;
    default: CHECK(false);
  }
  call += ");\n";
  const std::string src = std::string(R"cuda(
#include "mlops_fbla_items.cuh"
namespace mlops_fbla {
__shared__ unsigned coordinates[6];
struct Pos {
  static constexpr bool kReread = true;
  template <class T> __device__ static __forceinline__ T ro(const T *p) {
#ifdef FBLA_TEST_READONLY
    return __ldg(p);
#else
    return __ldcg(p);
#endif
  }
  __device__ static unsigned tid() { return threadIdx.x; }
  __device__ static unsigned coordinate(int i) { return *reinterpret_cast<volatile unsigned *>(coordinates + i); }
  __device__ static unsigned x() { return coordinate(0); }
  __device__ static unsigned y() { return coordinate(1); }
  __device__ static unsigned z() { return coordinate(2); }
  __device__ static unsigned nx() { return coordinate(3); }
  __device__ static unsigned ny() { return coordinate(4); }
  __device__ static unsigned nz() { return coordinate(5); }
  __device__ static int x_v() { return x(); }
  __device__ static int y_v() { return y(); }
};
}
struct Copy { unsigned char *dst; const unsigned char *src; unsigned long long bytes; };
extern "C" __global__ void run(const void *args, const long long *g, int first, int end,
                             const Copy *copies, int ncopies, unsigned *flags) {
  extern __shared__ __align__(128) unsigned char sm[];
)cuda") + "  auto a = *static_cast<const " + p.args_type + " *>(args);\n" +
      (bad_count && p.kind <= MLOPS_FBLA_ITEM_FINISH ? "  a.dcount = nullptr;\n" : "") + R"cuda(
  if (flags) {
    if (blockIdx.x == 0) {
      if (!threadIdx.x) while (atomicAdd(flags, 0) == 0) {}
      __syncthreads();
      for (int c = 0; c < ncopies; ++c)
        for (unsigned long long b = threadIdx.x; b < copies[c].bytes; b += blockDim.x)
          copies[c].dst[b] = copies[c].src[b];
      __threadfence();
      __syncthreads();
      if (!threadIdx.x) atomicExch(flags + 1, 1);
      return;
    }
    // Deliberately prime the readonly cache with poison before the publisher restores the entry's inputs.
    unsigned checksum = 0;
    for (int c = 0; c < ncopies; ++c)
      for (unsigned long long b = 4 * threadIdx.x; b + 4 <= copies[c].bytes && b < 512; b += 4 * blockDim.x) {
        unsigned value;
        asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(value) : "l"(copies[c].dst + b) : "memory");
        checksum ^= value;
      }
    flags[2 + threadIdx.x] = checksum;
    __syncthreads();
    if (!threadIdx.x) atomicExch(flags, 1);
    if (!threadIdx.x) while (atomicAdd(flags + 1, 0) == 0) {}
    __syncthreads();
    __threadfence();
  }
  for (int i = first + (flags ? 0 : blockIdx.x); i < end; i += flags ? 1 : gridDim.x) {
    if (!threadIdx.x) {
      mlops_fbla::coordinates[0] = i % g[0];
      mlops_fbla::coordinates[1] = (i / g[0]) % g[1];
      mlops_fbla::coordinates[2] = i / (g[0] * g[1]);
      for (int k = 0; k < 3; ++k) mlops_fbla::coordinates[k + 3] = g[k];
    }
    __syncthreads();
)cuda" + call + "    __syncthreads();\n  }\n}\n";
  const std::string arch = "--gpu-architecture=compute_" + std::to_string(p.arch);
  const char *opts[] = {arch.c_str(), "--std=c++17", "--device-as-default-execution-space",
    "--include-path=" FBLA_ROOT "/include", "--include-path=" FBLA_ROOT "/fbla/src",
    "--include-path=" FBLA_ROOT "/../positional/include", "--include-path=" FBLA_ROOT "/../positional/rope/src",
    "--include-path=" FBLA_CUDA_INCLUDE, "--include-path=" FBLA_CUDA_INCLUDE "/cccl",
    readonly_control ? "--define-macro=FBLA_TEST_READONLY=1" : "--define-macro=FBLA_TEST_COHERENT=1"};
  nvrtcProgram prog;
  RTC(nvrtcCreateProgram(&prog, src.c_str(), "items_executor.cu", 0, nullptr, nullptr));
  const auto status = nvrtcCompileProgram(prog, sizeof opts / sizeof *opts, opts);
  size_t n; RTC(nvrtcGetProgramLogSize(prog, &n));
  std::string log(n, '\0'); RTC(nvrtcGetProgramLog(prog, log.data()));
  if (status != NVRTC_SUCCESS) { fprintf(stderr, "%s\n%s\n", p.entry, log.c_str()); exit(1); }
  RTC(nvrtcGetPTXSize(prog, &n)); std::string ptx(n, '\0'); RTC(nvrtcGetPTX(prog, ptx.data()));
  RTC(nvrtcDestroyProgram(&prog));
  Program result;
  DRIVER(cuModuleLoadData(&result.module, ptx.data())); DRIVER(cuModuleGetFunction(&result.fn, result.module, "run"));
  if (p.smem_bytes > 48 * 1024) DRIVER(cuFuncSetAttribute(result.fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, p.smem_bytes));
  programs.emplace(key, result);
  return result;
}

// Every argument has its own allocation, so a declared range cannot accidentally be justified by an adjacent tensor.
struct State {
  std::array<std::unique_ptr<Buffer>, ARGS> buffers;
  mlops_fbla_cache cache{};
  Buffer &get(int i) { CHECK(buffers[i]); return *buffers[i]; }
  void add(int i, size_t n) { buffers[i] = std::make_unique<Buffer>(n); }
  State(const mlops_fbla_config &c, int ceiling) {
    const size_t heads = c.batch * c.kv_heads, groups = (c.capacity + G - 1) / G;
    const size_t hs = c.head_stride ? c.head_stride : groups, gs = c.group_stride ? c.group_stride : 1;
    const size_t units = (heads - 1) * hs + (groups - 1) * gs + 1;
    const int cb = c.format == MLOPS_FBLA_3P3 ? 3 : 4, fb = 6 - cb;
    add(MLOPS_FBLA_ARG_COARSE, units * 8 * cb * 512); add(MLOPS_FBLA_ARG_FINE, units * 8 * fb * 512);
    add(MLOPS_FBLA_ARG_META, units * 1024);
    for (int i : {MLOPS_FBLA_ARG_SINK_K, MLOPS_FBLA_ARG_SINK_V, MLOPS_FBLA_ARG_OPEN_K, MLOPS_FBLA_ARG_OPEN_V}) add(i, heads * G * D * 2);
    if (mlops_fbla_images_bytes(&c)) add(MLOPS_FBLA_ARG_IMAGES, units * mlops_fbla_detail::IMAGE_SLOT);
    cache = {get(MLOPS_FBLA_ARG_COARSE).as<uint8_t>(), get(MLOPS_FBLA_ARG_FINE).as<uint8_t>(), get(MLOPS_FBLA_ARG_META).as<uint16_t>(),
      get(MLOPS_FBLA_ARG_SINK_K).as<uint16_t>(), get(MLOPS_FBLA_ARG_SINK_V).as<uint16_t>(),
      get(MLOPS_FBLA_ARG_OPEN_K).as<uint16_t>(), get(MLOPS_FBLA_ARG_OPEN_V).as<uint16_t>(),
      buffers[MLOPS_FBLA_ARG_IMAGES] ? get(MLOPS_FBLA_ARG_IMAGES).as<uint8_t>() : nullptr};
    for (int i : {MLOPS_FBLA_ARG_Q, MLOPS_FBLA_ARG_OUT}) add(i, (size_t)c.batch * c.q_heads * ceiling * D * 2);
    for (int i : {MLOPS_FBLA_ARG_K, MLOPS_FBLA_ARG_V}) add(i, heads * ceiling * D * 2);
    add(MLOPS_FBLA_ARG_LENGTH, 4); add(MLOPS_FBLA_ARG_COUNT, 4);
    auto cap = c; cap.tokens = c.capacity;
    add(MLOPS_FBLA_ARG_SCRATCH, mlops_fbla_decode_scratch_bytes(&cap));
  }
  void copy_from(State &s) {
    for (int i = 0; i < ARGS; ++i) if (buffers[i]) { CHECK(s.buffers[i]); auto b = s.get(i).read(); get(i).set(b.data(), get(i).size); }
  }
};

using Snapshot = std::array<Bytes, ARGS>;
Snapshot snapshot(State &s) { Snapshot b; for (int i = 0; i < ARGS; ++i) if (s.buffers[i]) b[i] = s.get(i).read(); return b; }
bool contains(const mlops_fbla_item_range &r, size_t byte) {
  if (byte < (size_t)r.offset) return false;
  const size_t delta = byte - r.offset;
  if (!r.stride) return delta < (size_t)r.bytes;
  const size_t last = std::min<size_t>(delta / r.stride, r.rows - 1);
  return delta - last * r.stride < (size_t)r.bytes;
}
std::vector<mlops_fbla_item_range> ranges(State &s, const std::vector<mlops_fbla_items> &passes, int pass, int item) {
  auto *p = passes.data(); int n = passes.size();
  const int nr = mlops_fbla_item_access(p, n, pass, item, nullptr, 0);
  CHECK(nr > 0);
  std::vector<mlops_fbla_item_range> r(nr);
  CHECK(mlops_fbla_item_access(p, n, pass, item, r.data(), nr) == nr);
  mlops_fbla_item_range truncated[2]; memset(truncated, 0xa5, sizeof truncated);
  std::array<unsigned char, sizeof truncated[1]> guard;
  memcpy(guard.data(), &truncated[1], guard.size());
  CHECK(mlops_fbla_item_access(p, n, pass, item, truncated, 1) == nr);
  CHECK(truncated[0].arg == r[0].arg && truncated[0].write == r[0].write && truncated[0].offset == r[0].offset &&
        truncated[0].bytes == r[0].bytes && truncated[0].stride == r[0].stride && truncated[0].rows == r[0].rows);
  CHECK(!memcmp(&truncated[1], guard.data(), guard.size()));
  for (const auto &q : r) {
    CHECK(q.arg >= 0 && q.arg < ARGS && s.buffers[q.arg]);
    CHECK(q.offset >= 0 && q.bytes > 0 && q.stride >= 0 && q.rows > 0 && q.write >= 0 && q.write <= 1);
    const long long end = q.offset + (q.rows - 1LL) * q.stride + q.bytes;
    if (end > (long long)s.get(q.arg).size) {
      fprintf(stderr, "kind %d item %d arg %d range ends %lld, allocation %zu\n", p[pass].kind, item, q.arg, end, s.get(q.arg).size);
      CHECK(false);
    }
  }
  // Device scalars are dependencies too. Store coverage alone cannot catch a silently dropped length/count read.
  if (omit_count_range)
    r.erase(std::remove_if(r.begin(), r.end(), [](const auto &q) { return q.arg == MLOPS_FBLA_ARG_COUNT; }), r.end());
  bool needs_length = false, needs_count = false;
  const auto &pass_desc = p[pass];
  if (pass_desc.kind <= MLOPS_FBLA_ITEM_FINISH) {
    mlops_fbla_detail::PrefillArgs a; memcpy(&a, pass_desc.args.bytes, sizeof a);
    needs_length = a.length && pass_desc.kind != MLOPS_FBLA_ITEM_TC_MERGE;
    needs_count = a.dcount;
  } else if (pass_desc.kind == MLOPS_FBLA_ITEM_QKV) {
    mlops_fbla_detail::QkvArgs a; memcpy(&a, pass_desc.args.bytes, sizeof a); needs_length = a.p.length;
  } else {
    mlops_fbla_detail::Args a; memcpy(&a, pass_desc.args.bytes, sizeof a);
    needs_length = a.length && pass_desc.kind != MLOPS_FBLA_ITEM_MERGE;
  }
  for (int arg : {MLOPS_FBLA_ARG_LENGTH, MLOPS_FBLA_ARG_COUNT}) {
    bool found = false;
    for (const auto &q : r) found |= q.arg == arg && !q.write && contains(q, 0) && contains(q, 3);
    CHECK(found == (arg == MLOPS_FBLA_ARG_LENGTH ? needs_length : needs_count));
  }
  return r;
}
void execute(State &s, std::vector<mlops_fbla_items> &p, bool individual) {
  CHECK(mlops_fbla_item_access(nullptr, p.size(), 0, 0, nullptr, 0) == -1);
  CHECK(mlops_fbla_item_access(p.data(), p.size(), -1, 0, nullptr, 0) == -1);
  CHECK(mlops_fbla_item_access(p.data(), p.size(), 0, p[0].items, nullptr, 0) == -1);
  CHECK(mlops_fbla_item_access(p.data(), p.size(), 0, 0, nullptr, 1) == -1);
  for (auto &pass : p) if (pass.kind == MLOPS_FBLA_ITEM_TC && pass.geom[11] > 1) {
    s.add(MLOPS_FBLA_ARG_PART_ML, pass.geom[9]); s.add(MLOPS_FBLA_ARG_PART_O, pass.geom[10]);
    pass.geom[7] = (long long)s.get(MLOPS_FBLA_ARG_PART_ML).p; pass.geom[8] = (long long)s.get(MLOPS_FBLA_ARG_PART_O).p;
    for (auto &merge : p) if (merge.kind == MLOPS_FBLA_ITEM_TC_MERGE) { merge.geom[3] = pass.geom[7]; merge.geom[4] = pass.geom[8]; }
  }
  for (int k = 0; k < (int)p.size(); ++k) {
    auto &pass = p[k]; CHECK(pass.items > 0 && pass.args_bytes > 0 && pass.args_bytes <= MLOPS_FBLA_ITEM_ARGS_BYTES);
    const Program prog = executor(pass);
    Buffer args(pass.args_bytes), geometry(sizeof pass.geom); args.set(pass.args.bytes, pass.args_bytes); geometry.set(pass.geom, sizeof pass.geom);
    std::vector<mlops_fbla_item_range> union_ranges;
    for (int item = 0; item < pass.items; ++item) {
      auto r = ranges(s, p, k, item); union_ranges.insert(union_ranges.end(), r.begin(), r.end());
    }
    const int steps = individual ? pass.items : 1;
    for (int step = 0; step < steps; ++step) {
      const auto before = snapshot(s);
      int first = individual ? step : 0, end = individual ? step + 1 : pass.items;
      const auto r = individual ? ranges(s, p, k, step) : union_ranges;
      std::vector<std::unique_ptr<Buffer>> sources;
      std::vector<Copy> copies;
      std::unique_ptr<Buffer> descriptors, flags;
      if (publication) {
        std::array<bool, ARGS> reads{};
        for (const auto &q : r) if (!q.write) reads[q.arg] = true;
        // Norm weights/frequencies remain immutable by contract.
        for (int arg = 0; arg < ARGS; ++arg) if (reads[arg] && arg != MLOPS_FBLA_ARG_TQN &&
            arg != MLOPS_FBLA_ARG_TKN && arg != MLOPS_FBLA_ARG_TFREQ) {
          auto src = std::make_unique<Buffer>(s.get(arg).size);
          CUDA(cudaMemcpy(src->p, s.get(arg).p, src->size, cudaMemcpyDeviceToDevice));
          CUDA(cudaMemset(s.get(arg).p, 0xa5, src->size));
          copies.push_back({s.get(arg).p, src->p, src->size}); sources.push_back(std::move(src));
        }
        descriptors = std::make_unique<Buffer>(copies.size() * sizeof(Copy)); descriptors->set(copies.data(), descriptors->size);
        flags = std::make_unique<Buffer>((2 + pass.threads) * 4);
      }
      void *dcopies = descriptors ? descriptors->p : nullptr, *dflags = flags ? flags->p : nullptr;
      int ncopies = copies.size();
      void *params[] = {&args.p, &geometry.p, &first, &end, &dcopies, &ncopies, &dflags};
      if (publication) {
        int blocks, sms, dev; DRIVER(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, prog.fn, pass.threads, pass.smem_bytes));
        CUDA(cudaGetDevice(&dev)); CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)); CHECK(blocks * sms >= 2);
        DRIVER(cuLaunchCooperativeKernel(prog.fn, 2, 1, 1, pass.threads, 1, 1, pass.smem_bytes, 0, params));
      } else {
        DRIVER(cuLaunchKernel(prog.fn, individual ? 1 : std::min(pass.items, 3), 1, 1, pass.threads, 1, 1, pass.smem_bytes, 0, params, nullptr));
      }
      CUDA(cudaDeviceSynchronize());
      const auto after = snapshot(s);
      for (int arg = 0; arg < ARGS; ++arg) for (size_t b = 0; b < before[arg].size(); ++b) if (before[arg][b] != after[arg][b]) {
        bool covered = false; for (const auto &q : r) covered |= q.arg == arg && q.write && contains(q, b);
        if (!covered) { fprintf(stderr, "undeclared store: kind %d item %d arg %d byte %zu\n", pass.kind, step, arg, b); CHECK(false); }
      }
    }
  }
}

// Already initialized decode passes: timing excludes compilation, allocation, copies, and correctness checks.
// Report the complete grid as well as persistent execution on three CTAs and one CTA.
void time_passes(const std::vector<mlops_fbla_items> &passes, int tokens) {
  for (const auto &p : passes) {
    const Program prog = executor(p);
    Buffer args(p.args_bytes), geom(sizeof p.geom);
    args.set(p.args.bytes, p.args_bytes); geom.set(p.geom, sizeof p.geom);
    int first = 0, end = p.items, ncopies = 0;
    void *copies = nullptr, *flags = nullptr;
    void *params[] = {&args.p, &geom.p, &first, &end, &copies, &ncopies, &flags};
    int regs, local;
    DRIVER(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, prog.fn));
    DRIVER(cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, prog.fn));
    for (int grid : {p.items, std::min(p.items, 3), 1}) {
      auto launch = [&] { DRIVER(cuLaunchKernel(prog.fn, grid, 1, 1, p.threads, 1, 1, p.smem_bytes, 0, params, nullptr)); };
      for (int k = 0; k < 10; ++k) launch();
      cudaEvent_t begin, finish; CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&finish));
      std::array<float, 7> us{};
      for (auto &x : us) {
        CUDA(cudaEventRecord(begin));
        for (int k = 0; k < 100; ++k) launch();
        CUDA(cudaEventRecord(finish)); CUDA(cudaEventSynchronize(finish));
        CUDA(cudaEventElapsedTime(&x, begin, finish)); x *= 10.f;
      }
      std::sort(us.begin(), us.end());
      printf("BENCH tokens %d kind %d items %d grid %d threads %d shared %d regs %d local %d: %.3f us\n",
             tokens, p.kind, p.items, grid, p.threads, p.smem_bytes, regs, local, us[3]); fflush(stdout);
      CUDA(cudaEventDestroy(begin)); CUDA(cudaEventDestroy(finish));
    }
  }
}

enum Operation { Prefill, Append, Decode, Qkv, AppendChunk };
void run(Operation op, int cb, int batch, int qg, int start, int count, int ceiling, bool strided, bool norm, int kh = 2,
         int per_head = 0) {
  const int capacity = start + ceiling + 173, heads = batch * kh;
  mlops_fbla_config cfg{batch, kh * qg, kh, 0, 0.f, 0.f, cb == 3 ? MLOPS_FBLA_3P3 : MLOPS_FBLA_4P2, capacity, 0, 0};
  if (strided) { cfg.head_stride = 1; cfg.group_stride = heads + 1; }
  State ref(cfg, ceiling), test(cfg, ceiling);
  if (per_head) for (State *s : {&ref, &test})
    s->add(MLOPS_FBLA_ARG_SCRATCH, mlops_fbla_detail::decode_scratch_bytes(&cfg, per_head));
  // The prompt attends so that a TC continuation has valid images of its older closed groups.
  if (start) {
    Buffer k((size_t)heads * start * D * 2), v(k.size), q((size_t)batch * cfg.q_heads * start * D * 2), out(q.size);
    random_bf16(k); random_bf16(v); random_bf16(q);
    CHECK(mlops_fbla_prefill(&cfg, start, q.as<uint16_t>(), k.as<uint16_t>(), v.as<uint16_t>(), &ref.cache, out.as<uint16_t>(), 0) == 0);
  }
  for (int i : {MLOPS_FBLA_ARG_Q, MLOPS_FBLA_ARG_K, MLOPS_FBLA_ARG_V}) random_bf16(ref.get(i));
  cfg.tokens = start >= 7 ? 7 : 0;
  const int length = start - cfg.tokens;
  ref.get(MLOPS_FBLA_ARG_LENGTH).set(&length, 4); ref.get(MLOPS_FBLA_ARG_COUNT).set(&count, 4);
  mlops_fbla_token tokens[2]{};
  if (op == Qkv) {
    for (State *s : {&ref, &test}) {
      const size_t stride = (size_t)kh * (qg + 2) * D + 16;
      for (int i : {MLOPS_FBLA_ARG_TQ, MLOPS_FBLA_ARG_TK, MLOPS_FBLA_ARG_TV}) s->add(i, batch * stride * 2);
      for (int i : {MLOPS_FBLA_ARG_TQN, MLOPS_FBLA_ARG_TKN}) if (norm) s->add(i, D * 2);
      s->add(MLOPS_FBLA_ARG_TFREQ, D / 2 * sizeof(mlops_rope_detail::Freq));
      s->add(MLOPS_FBLA_ARG_TQ_OUT, (size_t)batch * cfg.q_heads * D * 2);
      tokens[s == &test] = {s->get(MLOPS_FBLA_ARG_TQ).as<uint16_t>(), s->get(MLOPS_FBLA_ARG_TK).as<uint16_t>(),
        s->get(MLOPS_FBLA_ARG_TV).as<uint16_t>(), (int64_t)stride, (qg + 2LL) * D, (qg + 2LL) * D,
        norm ? s->get(MLOPS_FBLA_ARG_TQN).as<uint16_t>() : nullptr, norm ? s->get(MLOPS_FBLA_ARG_TKN).as<uint16_t>() : nullptr,
        1e-6f, s->get(MLOPS_FBLA_ARG_TFREQ).as<float>(), 1.f, s->get(MLOPS_FBLA_ARG_TQ_OUT).as<uint16_t>()};
    }
    for (int i : {MLOPS_FBLA_ARG_TQ, MLOPS_FBLA_ARG_TK, MLOPS_FBLA_ARG_TV, MLOPS_FBLA_ARG_TQN, MLOPS_FBLA_ARG_TKN})
      if (ref.buffers[i]) random_bf16(ref.get(i));
    std::vector<mlops_rope_detail::Freq> freq(D / 2);
    for (int i = 0; i < D / 2; ++i) freq[i] = {0.01f / (i + 1), 1e-10f, 0.03f / (i + 1), 1e-11f};
    ref.get(MLOPS_FBLA_ARG_TFREQ).set(freq.data(), ref.get(MLOPS_FBLA_ARG_TFREQ).size);
  }
  test.copy_from(ref);
  const auto initial = snapshot(test);
  auto plan = [&](mlops_fbla_items *p, int max) {
    auto *len = test.get(MLOPS_FBLA_ARG_LENGTH).as<int>(); auto *q = test.get(MLOPS_FBLA_ARG_Q).as<uint16_t>();
    auto *k = test.get(MLOPS_FBLA_ARG_K).as<uint16_t>(), *v = test.get(MLOPS_FBLA_ARG_V).as<uint16_t>();
    auto *out = test.get(MLOPS_FBLA_ARG_OUT).as<uint16_t>(); auto &ws = test.get(MLOPS_FBLA_ARG_SCRATCH);
    switch (op) {
      case Prefill:
      case AppendChunk: return mlops_fbla_prefill_at_items(&cfg, len, test.get(MLOPS_FBLA_ARG_COUNT).as<int>(), ceiling,
                                                         op == Prefill ? q : nullptr, k, v, &test.cache, out, p, max);
      case Append: return mlops_fbla_append_at_items(&cfg, len, k, v, &test.cache, p, max);
      case Decode: return mlops_fbla_decode_at_items(&cfg, len, q, &test.cache, out, ws.p, ws.size, per_head, p, max);
      default: return mlops_fbla_qkv_at_items(&cfg, len, &tokens[1], &test.cache, p, max);
    }
  };
  CHECK(plan(nullptr, -1) == -1); CHECK(plan(nullptr, 1) == -1);
  const int n = plan(nullptr, 0); if (n < 0) fprintf(stderr, "%s\n", mlops_fbla_last_error()); CHECK(n > 0 && n <= 5);
  std::vector<mlops_fbla_items> p(n); CHECK(plan(p.data(), n) == n);
  mlops_fbla_items short_plan[2]{}; memset(&short_plan[1], 0xa5, sizeof short_plan[1]);
  std::array<unsigned char, sizeof short_plan[1]> guard;
  memcpy(guard.data(), &short_plan[1], guard.size());
  CHECK(plan(short_plan, 1) == n); CHECK(!memcmp(&short_plan[1], guard.data(), guard.size())); CHECK(!memcmp(&short_plan[0], &p[0], sizeof p[0]));
  auto *len = ref.get(MLOPS_FBLA_ARG_LENGTH).as<int>(); auto *q = ref.get(MLOPS_FBLA_ARG_Q).as<uint16_t>();
  auto *k = ref.get(MLOPS_FBLA_ARG_K).as<uint16_t>(), *v = ref.get(MLOPS_FBLA_ARG_V).as<uint16_t>();
  auto *out = ref.get(MLOPS_FBLA_ARG_OUT).as<uint16_t>(); auto &ws = ref.get(MLOPS_FBLA_ARG_SCRATCH);
  mlops_fbla_status status;
  switch (op) {
    case Prefill:
    case AppendChunk: status = mlops_fbla_prefill_at(&cfg, len, ref.get(MLOPS_FBLA_ARG_COUNT).as<int>(), ceiling,
                                                    op == Prefill ? q : nullptr, k, v, &ref.cache, out, 0); break;
    case Append: status = mlops_fbla_append_at(&cfg, len, k, v, &ref.cache, 0); break;
    case Decode:
      if (per_head) {
        mlops_fbla_detail::Args a;
        status = mlops_fbla_detail::decode_args(&cfg, len, q, &ref.cache, out, ws.p, ws.size, per_head, a);
        CHECK(status == 0);
        CUDA((cb == 3 ? mlops_fbla_detail::launch<3, 3>(a, per_head, 0) : mlops_fbla_detail::launch<4, 2>(a, per_head, 0)));
      } else status = mlops_fbla_decode_at(&cfg, len, q, &ref.cache, out, ws.p, ws.size, 0);
      break;
    default: status = mlops_fbla_qkv_at(&cfg, len, &tokens[0], &ref.cache, 0); break;
  }
  CHECK(status == 0); CUDA(cudaDeviceSynchronize());
  execute(test, p, false);
  const auto expected = snapshot(ref), actual = snapshot(test);
  for (int arg : {MLOPS_FBLA_ARG_OUT, MLOPS_FBLA_ARG_COARSE, MLOPS_FBLA_ARG_FINE, MLOPS_FBLA_ARG_META,
                 MLOPS_FBLA_ARG_SINK_K, MLOPS_FBLA_ARG_SINK_V, MLOPS_FBLA_ARG_OPEN_K, MLOPS_FBLA_ARG_OPEN_V,
                 MLOPS_FBLA_ARG_IMAGES, MLOPS_FBLA_ARG_TQ_OUT}) {
    if (expected[arg] != actual[arg]) { fprintf(stderr, "output differs: op %d format %d batch %d qg %d start %d arg %d\n", op, cb, batch, qg, start, arg); CHECK(false); }
  }
  if (!publication && !benchmark && ceiling <= 17 && (qg <= 2 || per_head == 1)) {
    for (int arg = 0; arg < ARGS; ++arg) if (!initial[arg].empty()) test.get(arg).set(initial[arg].data(), test.get(arg).size);
    execute(test, p, true);
    const auto each = snapshot(test);
    for (int arg : {MLOPS_FBLA_ARG_OUT, MLOPS_FBLA_ARG_COARSE, MLOPS_FBLA_ARG_FINE, MLOPS_FBLA_ARG_META,
                   MLOPS_FBLA_ARG_SINK_K, MLOPS_FBLA_ARG_SINK_V, MLOPS_FBLA_ARG_OPEN_K, MLOPS_FBLA_ARG_OPEN_V,
                   MLOPS_FBLA_ARG_IMAGES, MLOPS_FBLA_ARG_TQ_OUT}) CHECK(each[arg] == expected[arg]);
  }
  if (benchmark) {
    printf("BENCH config format %d qg %d per_head %d\n", cb, qg, per_head);
    time_passes(p, start);
  }
  printf("PASS op %d format %d batch %d qg %d start %d count %d/%d%s%s (%d passes)\n", op, cb, batch, qg, start, count, ceiling,
         strided ? " strided" : "", norm ? " norm" : "", n); fflush(stdout);
}
}  // namespace

int main(int argc, char **argv) {
  if (argc == 2 && !strcmp(argv[1], "--publication")) publication = true;
  else if (argc == 2 && !strcmp(argv[1], "--readonly-control")) { publication = true; readonly_control = true; }
  else if (argc == 2 && !strcmp(argv[1], "--bad-count")) bad_count = true;
  else if (argc == 2 && !strcmp(argv[1], "--omit-count-range")) omit_count_range = true;
  else if (argc == 2 && !strcmp(argv[1], "--bench")) benchmark = true;
  else CHECK(argc == 1);
  DRIVER(cuInit(0));
  int device; CUDA(cudaGetDevice(&device));
  cudaDeviceProp prop{}; CUDA(cudaGetDeviceProperties(&prop, device));
  printf("device: %s; sm_%d%d; %d SMs; %zu MiB VRAM\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount, prop.totalGlobalMem >> 20); fflush(stdout);
  for (int cb : {3, 4}) {
    if (benchmark) {
      for (int tokens : {1, 16, 128, 300, 1024}) for (int qg : {1, 4}) run(Decode, cb, 1, qg, tokens, 1, 1, false, false);
      for (int qg : {1, 4}) run(Decode, cb, 1, qg, 300, 1, 1, false, false, 2, 1);
      continue;
    }
    for (int qg : {1, 2, 4, 5, 8}) {
      run(Prefill, cb, 2, qg, 127, 3, 17, qg % 2, false);
      run(Append, cb, 2, qg, 128, 1, 1, qg % 2, false);
      run(Decode, cb, 2, qg, 301, 1, 1, qg % 2, false);
      run(Qkv, cb, 2, qg, 128, 1, 1, qg % 2, false);
      run(Qkv, cb, 2, qg, 0, 1, 1, false, true);
    }
    run(Prefill, cb, 1, 1, 1000, 300, 512, true, false);
    run(AppendChunk, cb, 2, 2, 127, 3, 17, true, false);
    run(Prefill, cb, 1, 1, 256, 2049, 4096, false, false, 1); // TC split partials, including empty pieces
    run(Decode, cb, 1, 4, 128, 1, 1, false, false); // no closed groups: every refinement item is empty
    run(Decode, cb, 1, 4, 256, 1, 1, true, false); // no closed groups: sink and full open BF16 groups
    run(Decode, cb, 1, 2, 16210, 1, 1, false, false); // capacity 16384: last single-slice layout
    run(Decode, cb, 1, 2, 16211, 1, 1, false, false); // capacity 16385: first multi-slice layout
    run(Decode, cb, 1, 4, 20000, 1, 1, true, false); // multiple select slices
    run(Decode, cb, 2, 4, 300, 1, 1, true, false, 2, 1); // explicit executor grid and scratch layout
  }
  for (auto &p : programs) DRIVER(cuModuleUnload(p.second.module));
  if (readonly_control) {
    puts("READONLY CONTROL INCONCLUSIVE: no stale read observed on this run; readonly loads have no coherence guarantee.");
    return 77;
  }
  puts("ALL ITEMS PASS");
}
