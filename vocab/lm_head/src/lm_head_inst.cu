// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// One slice of the head's instances (instances.h), chosen by -DMLOPS_LM_HEAD_SLICE=k: the kernels it names are compiled here
// and nowhere else.
#include "instances.h"

#ifndef MLOPS_LM_HEAD_SLICE
#error "compile with -DMLOPS_LM_HEAD_SLICE=k (0 .. 6)"
#endif
#define MLOPS_LM_HEAD_NAME2(k) register_slice_##k
#define MLOPS_LM_HEAD_NAME(k) MLOPS_LM_HEAD_NAME2(k)

namespace mlops_vocab_detail {

void MLOPS_LM_HEAD_NAME(MLOPS_LM_HEAD_SLICE)(Instance *t, int &n) {
#if MLOPS_LM_HEAD_SLICE < 4
  add_rows<(1 << MLOPS_LM_HEAD_SLICE)>(t, n);
#else
  add_mma_chunks<2, MLOPS_LM_HEAD_SLICE - 4>(t, n);
  add_mma_chunks<4, MLOPS_LM_HEAD_SLICE - 4>(t, n);
  add_mma_chunks<8, MLOPS_LM_HEAD_SLICE - 4>(t, n);
#endif
}

}  // namespace mlops_vocab_detail
