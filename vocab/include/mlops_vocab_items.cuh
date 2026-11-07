// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// Device entries of the work items (named in include/mlops_vocab_items.h), for an executor's kernels, compiled ahead of time with nvcc
// (-std=c++17 -gencode arch=compute_<arch>,code=sm_<the device's>) or at run time with NVRTC (--std=c++17
// --device-as-default-execution-space, compute_<arch>); include paths: vocab's include/ and the CUDA toolkit's include/ (CUDA 13 also
// include/cccl). Every thread of the described CTA calls the entry.
#pragma once
#include "../embed/src/embed.cuh"
#include "../lm_head/src/lm_head_mma.cuh"
#include "../sample/src/sample.cuh"

namespace mlops_vocab_items {

template <int W> __device__ __forceinline__ void embed(const mlops_vocab_detail::EmbedOp &op, int item, unsigned tid, void *) {
  mlops_vocab_detail::embed_item<W>(op, item, tid);
}

// The head's op with column 0 of x and norm_parts moved to the column the device index names (read through L2).
__device__ __forceinline__ mlops_vocab_detail::HeadOp head_column(mlops_vocab_detail::HeadOp op) {
  if (op.column) {
    const size_t c = static_cast<size_t>(__ldcg(op.column));
    op.x += c * op.hidden;
    if (op.norm_parts) op.norm_parts += c * op.norm_parts_n;
  }
  return op;
}

template <int R, int S, int D, int NB, bool XS>
__device__ __forceinline__ void head(const mlops_vocab_detail::HeadOp &op, int item, unsigned tid, void *smem) {
  mlops_vocab_detail::head_item<R, S, D, NB, XS>(head_column(op), item, tid, smem);
}

// KIND: 0 the BF16 mma.m16n8k16 instance (sm_80+), 1 the FP16 mma.m16n8k8 one (sm_75+), 2 the FP16 mma.m8n8k4 one (sm_70+).
template <int S, int J, int D, int KIND, bool XS>
__device__ __forceinline__ void head_tc(const mlops_vocab_detail::HeadOp &op, int item, unsigned tid, void *smem) {
  mlops_vocab_detail::head_item_mma<S, J, D, KIND, XS>(head_column(op), item, tid, smem);
}

// Pass PASS of a sampling run (mlops_vocab_detail::SamplePass); FINAL: the call's last pass.
template <int PASS, bool FINAL>
__device__ __forceinline__ void sample(const mlops_vocab_detail::SampleOp &op, int item, unsigned, void *smem) {
  auto *s = reinterpret_cast<mlops_vocab_detail::SampleItemSmem *>(smem);
  mlops_vocab_detail::sample_run_item(op, PASS, FINAL, item, s->pass, s->item);
}

}  // namespace mlops_vocab_items
