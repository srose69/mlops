// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The rope_kernel instances of one dtype (MLOPS_ROPE_DT: mlops_rope_dtype): V 1 / 2 / 4, both pair layouts,
// both arithmetics.
#include "rope_kernels.cuh"

#ifndef MLOPS_ROPE_DT
#error "build with -DMLOPS_ROPE_DT=<dtype> (positional/Makefile)"
#endif

namespace mlops_rope_detail {

template <int DT, int V, bool INTER, bool HF>
static void go(const Args &a, int tokens, cudaStream_t stream) {
  rope_kernel<DT, V, INTER, HF><<<tokens, kThreads, a.pairs * sizeof(float4), stream>>>(a);
}

template <int DT, bool INTER, bool HF>
static void by_v(const Args &a, int tokens, int v, cudaStream_t stream) {
  if (v == 4) go<DT, 4, INTER, HF>(a, tokens, stream);
  else if (v == 2) go<DT, 2, INTER, HF>(a, tokens, stream);
  else go<DT, 1, INTER, HF>(a, tokens, stream);
}

template <int DT>
void launch(const Args &a, int tokens, int v, bool inter, bool hf, cudaStream_t stream) {
  if (inter) hf ? by_v<DT, true, true>(a, tokens, v, stream) : by_v<DT, true, false>(a, tokens, v, stream);
  else hf ? by_v<DT, false, true>(a, tokens, v, stream) : by_v<DT, false, false>(a, tokens, v, stream);
}

template void launch<MLOPS_ROPE_DT>(const Args &, int, int, bool, bool, cudaStream_t);

}  // namespace mlops_rope_detail
