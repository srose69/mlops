// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The device entries of the work items (include/mlops_vocab_items.h names them and their argument types), for an
// executor's kernels. NVRTC: --std=c++17 --device-as-default-execution-space, include paths: vocab's include/ and the CUDA toolkit's include/ (CUDA 13
// also include/cccl). Compile for the plan's compute_<arch>; every thread of the described CTA calls the entry, and
// the CTA is synchronized before the entry's shared memory is reused.
#pragma once
#include "../embed/src/embed.cuh"
#include "../lm_head/src/lm_head_mma.cuh"
#include "../sample/src/sample.cuh"

namespace mlops_vocab_items {

template <int W> __device__ __forceinline__ void embed(const mlops_vocab_detail::EmbedOp &op, int item, unsigned tid, void *) {
  mlops_vocab_detail::embed_item<W>(op, item, tid);
}

template <int R, int S, int D, int NB>
__device__ __forceinline__ void head(const mlops_vocab_detail::HeadOp &op, int item, unsigned tid, void *smem) {
  mlops_vocab_detail::head_item<R, S, D, NB>(op, item, tid, *reinterpret_cast<mlops_vocab_detail::HeadSmem<S, NB> *>(smem));
}

// KIND: 0 the BF16 mma.m16n8k16 instance (sm_80+), 1 the FP16 mma.m16n8k8 one (sm_75+), 2 the FP16 mma.m8n8k4 one (sm_70+).
template <int S, int J, int D, int KIND>
__device__ __forceinline__ void head_tc(const mlops_vocab_detail::HeadOp &op, int item, unsigned tid, void *smem) {
  mlops_vocab_detail::head_item_mma<S, J, D, KIND>(op, item, tid, *reinterpret_cast<mlops_vocab_detail::MmaSmem<S> *>(smem));
}

// Pass PASS of a sampling run (mlops_vocab_detail::SamplePass); FINAL: the call's last pass.
template <int PASS, bool FINAL>
__device__ __forceinline__ void sample(const mlops_vocab_detail::SampleOp &op, int item, unsigned, void *smem) {
  auto *s = reinterpret_cast<mlops_vocab_detail::SampleItemSmem *>(smem);
  mlops_vocab_detail::sample_run_item(op, PASS, FINAL, item, s->pass, s->item);
}

}  // namespace mlops_vocab_items
