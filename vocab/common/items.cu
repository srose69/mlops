// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_vocab_item_access (include/mlops_vocab_items.h): the dispatch to the modules by kind.
#include "items.h"

extern "C" int mlops_vocab_item_access(const mlops_vocab_items *items, int item, mlops_vocab_item_range *ranges, int max) {
  using namespace mlops_vocab_detail;
  if (!items || item < 0 || item >= items->items || max < 0 || (max > 0 && !ranges)) return -1;
  switch (items->kind) {
    case MLOPS_VOCAB_ITEM_EMBED: return embed_access(*items, item, ranges, max);
    case MLOPS_VOCAB_ITEM_HEAD:
    case MLOPS_VOCAB_ITEM_HEAD_TC: return head_access(*items, item, ranges, max);
    case MLOPS_VOCAB_ITEM_SAMPLE: return sample_access(*items, item, ranges, max);
    default: return -1;
  }
}
