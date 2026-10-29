// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// What the modules of mlops-vocab give mlops_vocab_item_access: the ranges of an item of their kind.
#pragma once
#include "mlops_vocab_items.h"

namespace mlops_vocab_detail {

// Each returns the number of ranges of the item (writing at most max), as mlops_vocab_item_access.
int embed_access(const mlops_vocab_items &it, int item, mlops_vocab_item_range *out, int max);
int head_access(const mlops_vocab_items &it, int item, mlops_vocab_item_range *out, int max);
int sample_access(const mlops_vocab_items &it, int item, mlops_vocab_item_range *out, int max);

// Appends a range to out if there is room; always counts it.
inline void put_range(mlops_vocab_item_range *out, int max, int &count, int arg, int write, long long offset, long long bytes,
                      long long stride = 0, int rows = 1) {
  if (count < max) out[count] = {arg, write, offset, bytes, stride, rows};
  ++count;
}

}  // namespace mlops_vocab_detail
