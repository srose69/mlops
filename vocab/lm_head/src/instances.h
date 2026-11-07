// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The head's instances (host): the kernel of each (R, S, D, NB, x staged) and (S, J, D, kind, x staged), and the table that
// describes them. The kernels are compiled in slices (lm_head_inst.cu, one object a slice) so that a build runs them side by
// side: slice 0 .. 3 the CUDA-core instances of NB = 1, 2, 4, 8, slice 4 .. 6 the tensor-core ones of mma kind 0 .. 2.
#pragma once
#include <cstddef>

#include "lm_head.cuh"
#include "lm_head_mma.cuh"

namespace mlops_vocab_detail {

template <int R, int S, int D, int NB, bool XS> __global__ void __launch_bounds__(32 * S) head_kernel(HeadOp op) {
  extern __shared__ __align__(16) unsigned char smem[];
  head_item<R, S, D, NB, XS>(op, blockIdx.x, threadIdx.x, smem);
}
template <int S, int J, int D, int KIND, bool XS> __global__ void __launch_bounds__(32 * S) head_kernel_mma(HeadOp op) {
  extern __shared__ __align__(16) unsigned char smem[];
  head_item_mma<S, J, D, KIND, XS>(op, blockIdx.x, threadIdx.x, smem);
}

// One table drives the kernels, the variants and their attributes. An instance exists when its units tile the phase; the
// variants of a device are those of them that use no local memory there. Each exists with x staged in shared memory (xs) and
// with x read from L2 at every step.
struct Instance {
  int r, s, d, nb, tc, j;
  bool xs;
  size_t tables;  // bytes of the kernel's tables in shared memory (HeadSmem, MmaSmem); x and the counts come after
  void (*kernel)(HeadOp);
};
constexpr int kSlices = 7;
constexpr int kMaxInstances = 2 * (4 * 3 * 3 * 4 + 3 * 3 * 3 * 3);

template <int R, int S, int D, int NB> void add(Instance *table, int &count) {
  if constexpr ((32 * S / NB) % R == 0) {
    table[count++] = {R, S, D, NB, 0, 0, false, sizeof(HeadSmem<S, NB>), head_kernel<R, S, D, NB, false>};
    table[count++] = {R, S, D, NB, 0, 0, true, sizeof(HeadSmem<S, NB>), head_kernel<R, S, D, NB, true>};
  }
}
template <int S, int J, int D, int KIND> void add_mma(Instance *table, int &count) {
  table[count++] = {KIND == 2 ? 32 : 16, S, D, 8, KIND + 1, J, false, sizeof(MmaSmem<S>), head_kernel_mma<S, J, D, KIND, false>};
  table[count++] = {KIND == 2 ? 32 : 16, S, D, 8, KIND + 1, J, true, sizeof(MmaSmem<S>), head_kernel_mma<S, J, D, KIND, true>};
}
template <int S, int J, int KIND> void add_mma_depths(Instance *t, int &n) { add_mma<S, J, 1, KIND>(t, n); add_mma<S, J, 2, KIND>(t, n); add_mma<S, J, 4, KIND>(t, n); }
template <int S, int KIND> void add_mma_chunks(Instance *t, int &n) { add_mma_depths<S, 1, KIND>(t, n); add_mma_depths<S, 2, KIND>(t, n); add_mma_depths<S, 4, KIND>(t, n); }
template <int NB, int R, int S> void add_depths(Instance *t, int &n) { add<R, S, 1, NB>(t, n); add<R, S, 2, NB>(t, n); add<R, S, 4, NB>(t, n); }
template <int NB, int R> void add_pieces(Instance *t, int &n) { add_depths<NB, R, 2>(t, n); add_depths<NB, R, 4>(t, n); add_depths<NB, R, 8>(t, n); }
template <int NB> void add_rows(Instance *t, int &n) { add_pieces<NB, 1>(t, n); add_pieces<NB, 2>(t, n); add_pieces<NB, 4>(t, n); add_pieces<NB, 8>(t, n); }

// The instances of slice k, appended to the table (defined in lm_head_inst.cu, one object per k).
void register_slice_0(Instance *table, int &count);
void register_slice_1(Instance *table, int &count);
void register_slice_2(Instance *table, int &count);
void register_slice_3(Instance *table, int &count);
void register_slice_4(Instance *table, int &count);
void register_slice_5(Instance *table, int &count);
void register_slice_6(Instance *table, int &count);

}  // namespace mlops_vocab_detail
