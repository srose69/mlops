// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// Sampling passes over chunks of kChunk tokens (include/mlops_sample.h). Each pass is closed by its last item,
// found by an arrival counter that only grows (a multiple of the item count between runs).
//   stats         maximum M and argmax of the sampled logits (the lower index on ties); greedy ends here
//   mass          Z, the total mass: the target of top-p without a small top-k
//   radix 0 .. 3  8 bits of the order key per pass, a histogram (counts or masses) of the keys under the prefix
//   count, write  top-k <= kGather: candidates (key >= threshold) counted per chunk, written in index order,
//                 sorted, cut by top-p and drawn from by the last item
//   draw          otherwise: per-chunk mass of the kept tokens, then the chunk holding the draw and the token
// Runs (sample_passes): greedy: stats. top-k <= kGather: stats, radix, count, write. top-p alone: stats, mass,
// radix, draw. Larger top-k: stats, radix, draw. Temperature alone: stats, draw.
#pragma once
#include "mlops_sample.h"

namespace mlops_vocab_detail {

constexpr int kThreads = 256, kChunk = 2048, kBuckets = 256, kGather = 1024, kPer = kChunk / kThreads;

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

// Re-read where used: nothing derived from the thread index stays live across a pass's work.
__device__ __forceinline__ unsigned tid() {
  unsigned t;
  asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
  return t;
}

struct ItemSmem {
  int last;
};

// The CTA's arrival at the end of one of `parts` items: true, in every thread, in the item that completes the
// run, whose loads of the others' results then go through L2 (__ldcg).
__device__ __forceinline__ bool last_arrival(unsigned *counter, unsigned parts, ItemSmem &sm) {
  __threadfence();
  __syncthreads();
  if (tid() == 0) sm.last = (atomicAdd(counter, 1u) + 1u) % parts == 0u;
  __syncthreads();
  return sm.last;
}

enum SampleCounter { kStats, kMass, kRadix0, kCount = kRadix0 + 4, kWrite, kDraw, kSampleCounters };

struct SampleOp {
  const float *logits;
  float *sampled;                  // [vocab]: the logits as sampled, written by stats
  int vocab;
  const uint32_t *seen;            // [vocab / 32] bits
  const float *pre_max;            // the head's statistics, or null: stats scans the logits
  const int *pre_arg;
  const unsigned *pre_hist;        // [pre_count][256], or null
  int pre_count;
  float temperature;               // 0: greedy
  int top_k;                       // 0: off; above kGather only without top-p
  float top_p;                     // 1: off
  float penalty;                   // 1: off
  float inv_temperature, inv_penalty;
  mlops_sample_state *rng;         // the draw's Philox key and counter, in device memory
  uint64_t seed, step;             // read from rng at the start of an item
  float *chunk_m;                  // [chunks]
  int *chunk_arg;                  // [chunks]
  unsigned long long *chunk_mass;  // [chunks]
  unsigned long long *hist;        // [kBuckets]: counts or masses summed over the chunks; zero between passes
  int *cand_n;                     // [chunks]: then their offsets
  int2 *cand;                      // [kGather]: (index, logit bits)
  float *max_logit;                // M
  unsigned long long *ctl;         // prefix, remaining count or mass, Z, candidates / draw point
  unsigned *counters;              // [kSampleCounters]
  int *token;
  __host__ __device__ __forceinline__ int chunks() const { return (vocab + kChunk - 1) / kChunk; }
  __device__ __forceinline__ bool by_mass() const { return top_k == 0 && top_p < 1.f; }
};

// A thread's kPer consecutive values from p (16-byte aligned) in two 128-bit loads; past the end: -inf.
__device__ __forceinline__ void load_run(const float *p, int i0, int vocab, float (&v)[kPer]) {
  if (i0 + kPer <= vocab) {
    const float4 a = __ldcg(reinterpret_cast<const float4 *>(p + i0)), b = __ldcg(reinterpret_cast<const float4 *>(p + i0 + 4));
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
  } else {
#pragma unroll
    for (int j = 0; j < kPer; ++j) v[j] = i0 + j < vocab ? __ldcg(p + i0 + j) : neg_inf();
  }
}
__device__ __forceinline__ void store_run(float *p, int i0, int vocab, const float (&v)[kPer]) {
  if (i0 + kPer <= vocab) {
    __stcg(reinterpret_cast<float4 *>(p + i0), make_float4(v[0], v[1], v[2], v[3]));
    __stcg(reinterpret_cast<float4 *>(p + i0 + 4), make_float4(v[4], v[5], v[6], v[7]));
  } else {
#pragma unroll
    for (int j = 0; j < kPer; ++j)
      if (i0 + j < vocab) __stcg(p + i0 + j, v[j]);
  }
}

// The logit as sampled (stats; the later passes read it from op.sampled).
__device__ __forceinline__ float sample_logit(const SampleOp &op, float l, unsigned seen_word, int bit) {
  if (op.penalty != 1.f && (seen_word >> bit & 1u)) l = l > 0.f ? __fmul_rn(l, op.inv_penalty) : __fmul_rn(l, op.penalty);
  return op.temperature > 0.f ? __fmul_rn(l, op.inv_temperature) : l;
}
__device__ __forceinline__ unsigned order_key(float x) {  // larger float, larger key
  const unsigned u = __float_as_uint(x);
  return u & 0x80000000u ? ~u : u | 0x80000000u;
}
__device__ __forceinline__ unsigned long long mass_units(float l, float M) {  // exp(l - M) in units of 2^-40
  return (unsigned long long)__fmul_rn(__expf(__fsub_rn(l, M)), 0x1p40f);
}
__device__ __forceinline__ uint4 philox(uint64_t key, uint64_t ctr) {  // Philox4x32-10
  uint32_t c0 = (uint32_t)ctr, c1 = (uint32_t)(ctr >> 32), c2 = 0, c3 = 0, k0 = (uint32_t)key, k1 = (uint32_t)(key >> 32);
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    const uint32_t h0 = __umulhi(0xD2511F53u, c0), l0 = 0xD2511F53u * c0, h1 = __umulhi(0xCD9E8D57u, c2), l1 = 0xCD9E8D57u * c2;
    c0 = h1 ^ c1 ^ k0; c2 = h0 ^ c3 ^ k1; c1 = l1; c3 = l0;
    k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
  }
  return make_uint4(c0, c1, c2, c3);
}
// A point in [0, total) units from 53 random bits.
__device__ __forceinline__ unsigned long long draw_point(const SampleOp &op, unsigned long long total) {
  const uint4 r = philox(op.seed, op.step);
  const double u = (double)(((unsigned long long)r.x << 21) ^ (r.y >> 11)) * 0x1p-53;
  return (unsigned long long)(u * (double)total);
}

struct SampleCompact {  // the count, write and draw passes
  unsigned long long scan[kThreads];
  int2 cand[kGather];
};
struct SampleSmem {
  float m[kThreads / 32];
  int arg[kThreads / 32];
  unsigned long long buckets[kBuckets];
  unsigned bucket_lo[kBuckets], bucket_hi[kBuckets];  // the radix pass: 20-bit limbs of the buckets' sums
  SampleCompact compact;
  int n;
};

// The shared memory of a sampling item.
struct SampleItemSmem {
  SampleSmem pass;
  ItemSmem item;
};

// A thread's kPer consecutive tokens of chunk c: i = c kChunk + t kPer + j.
__device__ __forceinline__ int chunk_first(int c) { return c * kChunk + tid() * kPer; }

// Exclusive prefix sum of v over the CTA's threads in thread order (integers: exact): a scan in each warp, then the
// warps' totals. `total` is the sum over the CTA. Two barriers; the second protects the scratch for the next call.
__device__ __forceinline__ unsigned long long cta_prefix(unsigned long long v, SampleSmem &sm, unsigned long long &total) {
  const int lane = tid() & 31, warp = tid() >> 5;
  unsigned long long x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned long long y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) sm.compact.scan[warp] = x;
  __syncthreads();
  unsigned long long before = 0ull;
  total = 0ull;
#pragma unroll
  for (int w = 0; w < kThreads / 32; ++w) {
    const unsigned long long t = sm.compact.scan[w];
    before += w < warp ? t : 0ull;
    total += t;
  }
  __syncthreads();
  return before + x - v;
}

// The maximum and its lowest index over count (max, arg) pairs in L2: the CTA's threads each take a strided share, the
// warps combine by shuffles, and the result is in every thread (m, arg).
__device__ __forceinline__ void reduce_stats(const float *maxes, const int *args, int count, SampleSmem &sm, float &m, int &arg) {
  m = neg_inf();
  arg = 0x7fffffff;
  for (int k = tid(); k < count; k += kThreads) {
    const float mk = __ldcg(maxes + k);
    const int ak = __ldcg(args + k);
    if (mk > m || (mk == m && ak < arg)) { m = mk; arg = ak; }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    const float m2 = __shfl_xor_sync(0xffffffffu, m, o);
    const int a2 = __shfl_xor_sync(0xffffffffu, arg, o);
    if (m2 > m || (m2 == m && a2 < arg)) { m = m2; arg = a2; }
  }
  if ((tid() & 31) == 0) { sm.m[tid() >> 5] = m; sm.arg[tid() >> 5] = arg; }
  __syncthreads();
  m = sm.m[0];
  arg = sm.arg[0];
#pragma unroll
  for (int w = 1; w < kThreads / 32; ++w)
    if (sm.m[w] > m || (sm.m[w] == m && sm.arg[w] < arg)) { m = sm.m[w]; arg = sm.arg[w]; }
}

// ---- stats: M and the argmax -----------------------------------------------------------------------------------
__device__ __forceinline__ bool sample_stats_item(const SampleOp &op, int c, SampleSmem &sm, ItemSmem &is) {
  const int lane = tid() & 31, warp = tid() >> 5, i0 = chunk_first(c);
  float m = neg_inf(), lv[kPer];
  int arg = 0x7fffffff;
  load_run(op.logits, i0, op.vocab, lv);
  // The kPer tokens of a thread are 8 consecutive bits of one word of `seen`.
  const unsigned seen_word = op.penalty != 1.f && i0 < op.vocab ? __ldcg(op.seen + (i0 >> 5)) : 0u;
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    lv[j] = sample_logit(op, lv[j], seen_word, (i0 & 31) + j);
    if (i0 + j < op.vocab && lv[j] > m) { m = lv[j]; arg = i0 + j; }  // ascending: the first of equals
  }
  if (op.temperature > 0.f) store_run(op.sampled, i0, op.vocab, lv);  // greedy has no later pass
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    const float m2 = __shfl_xor_sync(0xffffffffu, m, o);
    const int a2 = __shfl_xor_sync(0xffffffffu, arg, o);
    if (m2 > m || (m2 == m && a2 < arg)) { m = m2; arg = a2; }
  }
  if (!lane) { sm.m[warp] = m; sm.arg[warp] = arg; }
  __syncthreads();
  if (tid() == 0) {
    for (int w = 1; w < kThreads / 32; ++w)
      if (sm.m[w] > sm.m[0] || (sm.m[w] == sm.m[0] && sm.arg[w] < sm.arg[0])) { sm.m[0] = sm.m[w]; sm.arg[0] = sm.arg[w]; }
    op.chunk_m[c] = sm.m[0];
    op.chunk_arg[c] = sm.arg[0];
  }
  if (!last_arrival(op.counters + kStats, op.chunks(), is)) return false;
  reduce_stats(op.chunk_m, op.chunk_arg, op.chunks(), sm, m, arg);
  if (tid() == 0) {
    *op.max_logit = m;
    op.ctl[0] = 0ull;                                                      // prefix
    op.ctl[1] = op.top_k > 0 ? (unsigned long long)op.top_k : 0ull;        // remaining count
    if (op.temperature <= 0.f) *op.token = arg;
  }
  return true;
}

// ---- pre: stats from the head's per-element maxima, one CTA; the logits are already the sampled ones ---------------
__device__ __forceinline__ bool sample_pre_item(const SampleOp &op, SampleSmem &sm) {
  float M;
  int A;
  reduce_stats(op.pre_max, op.pre_arg, op.pre_count, sm, M, A);
  if (tid() == 0) {
    *op.max_logit = M;
    op.ctl[0] = 0ull;
    op.ctl[1] = op.top_k > 0 ? (unsigned long long)op.top_k : 0ull;
    if (op.temperature <= 0.f) *op.token = A;
  }
  return true;
}

// ---- mass: Z, and the top-p target -----------------------------------------------------------------------------
__device__ __forceinline__ bool sample_mass_item(const SampleOp &op, int c, SampleSmem &sm, ItemSmem &is) {
  const int i0 = chunk_first(c);
  const float M = __ldcg(op.max_logit);
  unsigned long long z = 0ull;
  float sv[kPer];
  load_run(op.sampled, i0, op.vocab, sv);
#pragma unroll
  for (int j = 0; j < kPer; ++j)
    if (i0 + j < op.vocab) z += mass_units(sv[j], M);
  unsigned long long total;
  cta_prefix(z, sm, total);
  if (tid() == 0) op.chunk_mass[c] = total;
  if (!last_arrival(op.counters + kMass, op.chunks(), is)) return false;
  unsigned long long part = 0ull, Z;
  for (int k = tid(); k < op.chunks(); k += kThreads) part += __ldcg(op.chunk_mass + k);
  cta_prefix(part, sm, Z);
  if (tid() == 0) {
    op.ctl[2] = Z;
    op.ctl[1] = (unsigned long long)((double)op.top_p * (double)Z);  // remaining mass: the top-p target
  }
  return true;
}

// From the top bucket: the one where the remaining count (mass) runs out; it extends the prefix by 8 bits.
__device__ __forceinline__ void radix_choose(const SampleOp &op, SampleSmem &sm, unsigned prefix, int shift) {
  if (tid() == 0) {
    unsigned long long remain = __ldcg(op.ctl + 1);
    int chosen = 0;
    for (int b = kBuckets - 1; b >= 0; --b) {
      if (sm.buckets[b] >= remain) { chosen = b; break; }
      remain -= sm.buckets[b];
    }
    op.ctl[0] = (unsigned long long)(prefix | (unsigned)chosen << shift);
    op.ctl[1] = remain ? remain : 1ull;
  }
}

// ---- radix pass `pass`: bits 31 - 8 pass .. 24 - 8 pass of the keys under the prefix ----------------------------
__device__ __forceinline__ bool sample_radix_item(const SampleOp &op, int c, int pass, SampleSmem &sm, ItemSmem &is) {
  const int i0 = chunk_first(c), shift = 24 - 8 * pass;
  const unsigned prefix = (unsigned)__ldcg(op.ctl), mask = pass ? 0xffffffffu << (32 - 8 * pass) : 0u;
  const float M = __ldcg(op.max_logit);
  const bool by_mass = op.by_mass();
  // A bucket's sum is kept in two 20-bit limbs, each added with a 32-bit atomic of shared memory (a limb of 2048
  // tokens stays under 2^31): exact integers, so the order of the additions does not matter.
  for (int b = tid(); b < kBuckets; b += kThreads) sm.bucket_lo[b] = sm.bucket_hi[b] = 0u;
  float sv[kPer];
  load_run(op.sampled, i0, op.vocab, sv);
  __syncthreads();
#pragma unroll
  for (int j = 0; j < kPer; ++j)
    if (i0 + j < op.vocab) {
      const unsigned k = order_key(sv[j]);
      if ((k & mask) == prefix) {
        const int b = (int)((k >> shift) & 255u);
        if (by_mass) {
          const unsigned long long m = mass_units(sv[j], M);
          atomicAdd(&sm.bucket_lo[b], (unsigned)(m & 0xfffffull));
          atomicAdd(&sm.bucket_hi[b], (unsigned)(m >> 20));
        } else {
          atomicAdd(&sm.bucket_lo[b], 1u);
        }
      }
    }
  __syncthreads();
  // The chunk's sums join the call's in global memory (integers: any order); the closing item reads 256 values.
  for (int b = tid(); b < kBuckets; b += kThreads) {
    const unsigned long long add = ((unsigned long long)sm.bucket_hi[b] << 20) + sm.bucket_lo[b];
    if (add) atomicAdd(op.hist + b, add);
  }
  if (!last_arrival(op.counters + kRadix0 + pass, op.chunks(), is)) return false;
  for (int b = tid(); b < kBuckets; b += kThreads) {
    sm.buckets[b] = __ldcg(op.hist + b);
    __stcg(op.hist + b, 0ull);  // clean for the next pass
  }
  __syncthreads();
  radix_choose(op, sm, prefix, shift);
  return true;
}

// The first radix pass (counts) from the head's histograms: the buckets are their sums over its elements.
__device__ __forceinline__ bool sample_radix0_pre_item(const SampleOp &op, SampleSmem &sm) {
  for (int b = tid(); b < kBuckets; b += kThreads) {
    unsigned long long s = 0ull;
    for (int k = 0; k < op.pre_count; ++k) s += __ldcg(op.pre_hist + (size_t)k * kBuckets + b);
    sm.buckets[b] = s;
  }
  __syncthreads();
  radix_choose(op, sm, 0u, 24);
  return true;
}

// ---- count, write: the candidates in index order ---------------------------------------------------------------
__device__ __forceinline__ unsigned long long chunk_kept(const SampleOp &op, int i0, unsigned thr) {
  unsigned long long n = 0ull;
  float sv[kPer];
  load_run(op.sampled, i0, op.vocab, sv);
#pragma unroll
  for (int j = 0; j < kPer; ++j) n += i0 + j < op.vocab && order_key(sv[j]) >= thr;
  return n;
}
__device__ __forceinline__ bool sample_count_item(const SampleOp &op, int c, SampleSmem &sm, ItemSmem &is) {
  unsigned long long total;
  cta_prefix(chunk_kept(op, chunk_first(c), (unsigned)__ldcg(op.ctl)), sm, total);
  if (tid() == 0) op.cand_n[c] = (int)total;
  if (!last_arrival(op.counters + kCount, op.chunks(), is)) return false;
  unsigned long long carry = 0ull;  // counts -> offsets, a block of kThreads chunks at a time; the total
  for (int base = 0; base < op.chunks(); base += kThreads) {
    const int k = base + (int)tid();
    const unsigned long long nk = k < op.chunks() ? (unsigned long long)__ldcg(op.cand_n + k) : 0ull;
    unsigned long long block;
    const unsigned long long off = cta_prefix(nk, sm, block) + carry;
    if (k < op.chunks()) op.cand_n[k] = (int)off;
    carry += block;
  }
  if (tid() == 0) op.ctl[3] = carry;
  return true;
}

__device__ __forceinline__ bool sample_write_item(const SampleOp &op, int c, SampleSmem &sm, ItemSmem &is) {
  const int i0 = chunk_first(c);
  const unsigned thr = (unsigned)__ldcg(op.ctl);
  unsigned long long total;
  int at = __ldcg(op.cand_n + c) + (int)cta_prefix(chunk_kept(op, i0, thr), sm, total);
  float sv[kPer];
  load_run(op.sampled, i0, op.vocab, sv);
  for (int j = 0; j < kPer; ++j)
    if (i0 + j < op.vocab) {
      const float l = sv[j];
      if (order_key(l) >= thr && at < kGather) op.cand[at++] = make_int2(i0 + j, __float_as_int(l));
    }
  if (!last_arrival(op.counters + kWrite, op.chunks(), is)) return false;
  // The last item: bitonic sort over a power of two (logit down, index up), top-p over the candidates' mass, draw.
  const int n = (int)min(__ldcg(op.ctl + 3), (unsigned long long)kGather);
  int np = 1;
  while (np < n) np <<= 1;
  for (int i = tid(); i < np; i += kThreads)
    sm.compact.cand[i] = i < n ? __ldcg(op.cand + i) : make_int2(0x7fffffff, __float_as_int(neg_inf()));
  __syncthreads();
  for (int size = 2; size <= np; size <<= 1)
    for (int stride = size >> 1; stride; stride >>= 1) {
      for (int i = tid(); i < np; i += kThreads) {
        const int j = i ^ stride;
        if (j > i) {
          const int2 a = sm.compact.cand[i], b = sm.compact.cand[j];
          const float fa = __int_as_float(a.y), fb = __int_as_float(b.y);
          const bool a_first = fa > fb || (fa == fb && a.x < b.x);
          if ((i & size) == 0 ? !a_first : a_first) { sm.compact.cand[i] = b; sm.compact.cand[j] = a; }
        }
      }
      __syncthreads();
    }
  if (tid() == 0) {  // n <= kGather: one thread
    const float M = __ldcg(op.max_logit);
    unsigned long long z = 0ull;
    for (int i = 0; i < n; ++i) z += mass_units(__int_as_float(sm.compact.cand[i].y), M);
    int keep = n;
    if (op.top_p < 1.f) {  // keep a token while the mass before it is below top_p of the candidates'
      const unsigned long long lim = (unsigned long long)((double)op.top_p * (double)z);
      unsigned long long before = 0ull;
      for (int i = 0; i < n; ++i) {
        if (i && before >= lim) { keep = i; break; }
        before += mass_units(__int_as_float(sm.compact.cand[i].y), M);
      }
    }
    unsigned long long kept = 0ull;
    for (int i = 0; i < keep; ++i) kept += mass_units(__int_as_float(sm.compact.cand[i].y), M);
    const unsigned long long u = draw_point(op, kept);
    int pick = keep - 1;
    unsigned long long run = 0ull;
    for (int i = 0; i < keep; ++i) {
      run += mass_units(__int_as_float(sm.compact.cand[i].y), M);
      if (u < run) { pick = i; break; }
    }
    *op.token = sm.compact.cand[pick].x;
  }
  return true;
}

// ---- draw (no small top-k): the kept tokens' mass per chunk, the chunk holding the draw, the token in it --------
__device__ __forceinline__ unsigned long long chunk_kept_mass(const SampleOp &op, int i0, unsigned thr, float M) {
  unsigned long long z = 0ull;
  float sv[kPer];
  load_run(op.sampled, i0, op.vocab, sv);
#pragma unroll
  for (int j = 0; j < kPer; ++j)
    if (i0 + j < op.vocab && order_key(sv[j]) >= thr) z += mass_units(sv[j], M);
  return z;
}
__device__ __forceinline__ bool sample_draw_item(const SampleOp &op, int c, SampleSmem &sm, ItemSmem &is) {
  const unsigned thr = (unsigned)__ldcg(op.ctl);
  const float M = __ldcg(op.max_logit);
  unsigned long long total;
  cta_prefix(chunk_kept_mass(op, chunk_first(c), thr, M), sm, total);
  if (tid() == 0) op.chunk_mass[c] = total;
  if (!last_arrival(op.counters + kDraw, op.chunks(), is)) return false;
  unsigned long long part = 0ull, all;
  for (int k = tid(); k < op.chunks(); k += kThreads) part += __ldcg(op.chunk_mass + k);
  cta_prefix(part, sm, all);
  const unsigned long long point = draw_point(op, all);
  if (tid() == 0) { sm.n = op.chunks() - 1; op.ctl[3] = point; }  // (the last chunk holds a point past the others)
  __syncthreads();
  unsigned long long carry = 0ull;  // the chunk whose run of masses holds the point, and the point within it
  for (int base = 0; base < op.chunks(); base += kThreads) {
    const int k = base + (int)tid();
    const unsigned long long zk = k < op.chunks() ? __ldcg(op.chunk_mass + k) : 0ull;
    unsigned long long block;
    const unsigned long long off = cta_prefix(zk, sm, block) + carry;
    if (k < op.chunks() && off <= point && point < off + zk) { sm.n = k; op.ctl[3] = point - off; }
    carry += block;
  }
  __syncthreads();
  const int k = sm.n, i0 = chunk_first(k);
  const unsigned long long u = __ldcg(op.ctl + 3);
  const unsigned long long before = cta_prefix(chunk_kept_mass(op, i0, thr, M), sm, total);
  unsigned long long run = before;  // the thread whose run of kPer tokens holds u finds the token
  float sv[kPer];
  load_run(op.sampled, i0, op.vocab, sv);
  for (int j = 0; j < kPer; ++j)
    if (i0 + j < op.vocab) {
      const float l = sv[j];
      if (order_key(l) < thr) continue;
      const unsigned long long next = run + mass_units(l, M);
      if (u >= run && u < next) *op.token = i0 + j;
      run = next;
    }
  return true;
}

// ---- runs --------------------------------------------------------------------------------------------------------
enum SamplePass { kPassStats, kPassMass, kPassRadix0, kPassCount = kPassRadix0 + 4, kPassWrite, kPassDraw, kPassPre, kPassRadix0Pre, kPasses };

// The passes of the configuration's run, in order; their number.
__host__ __device__ inline int sample_passes(float temperature, int top_k, float top_p, bool pre, bool pre_hist, int pass[8]) {
  int n = 0;
  pass[n++] = pre ? kPassPre : kPassStats;
  if (temperature > 0.f) {
    const bool gather = top_k > 0 && top_k <= kGather, mass = top_k == 0 && top_p < 1.f;
    if (mass) pass[n++] = kPassMass;
    if (gather || mass || top_k > 0)
      for (int r = 0; r < 4; ++r) pass[n++] = r == 0 && pre_hist && top_k > 0 ? kPassRadix0Pre : kPassRadix0 + r;
    if (gather) { pass[n++] = kPassCount; pass[n++] = kPassWrite; }
    else pass[n++] = kPassDraw;
  }
  return n;
}

// Pass `pass` on chunk c of a call: the key and counter are read from the state, and the final pass's closing item
// advances the counter. True in the pass's last item.
__device__ __forceinline__ bool sample_pass_item(const SampleOp &op, int pass, int c, SampleSmem &sm, ItemSmem &is) {
  switch (pass) {
    case kPassStats: return sample_stats_item(op, c, sm, is);
    case kPassMass: return sample_mass_item(op, c, sm, is);
    case kPassCount: return sample_count_item(op, c, sm, is);
    case kPassWrite: return sample_write_item(op, c, sm, is);
    case kPassDraw: return sample_draw_item(op, c, sm, is);
    default: return sample_radix_item(op, c, pass - kPassRadix0, sm, is);
  }
}

__device__ __forceinline__ bool sample_run_item(SampleOp op, int pass, bool final, int c, SampleSmem &sm, ItemSmem &is) {
  op.seed = __ldcg(&op.rng->seed);
  op.step = __ldcg(&op.rng->step);
  const bool last = pass == kPassPre ? sample_pre_item(op, sm) : pass == kPassRadix0Pre ? sample_radix0_pre_item(op, sm) : sample_pass_item(op, pass, c, sm, is);
  if (final && last && tid() == 0) op.rng->step = op.step + 1;
  return last;
}

}  // namespace mlops_vocab_detail
