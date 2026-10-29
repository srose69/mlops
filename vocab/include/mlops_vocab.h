/* SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
 * Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
 *
 * mlops-vocab: everything along the vocabulary axis (libmlops_vocab.a).
 *   mlops_embed.h    embedding rows for device-side ids, with the first norm's inputs
 *   mlops_sample.h   the next token from logits: greedy, temperature, top-k, top-p, repetition penalty
 *   mlops_lm_head.h  the BF16 head: logits from the final hidden state, cut along hidden
 *   mlops_vocab_items.h  the work of these calls as items for an executor, and what each item reads and writes
 */
#ifndef MLOPS_VOCAB_H
#define MLOPS_VOCAB_H

#define MLOPS_VOCAB_VERSION "0.1.0"

#include "mlops_embed.h"
#include "mlops_sample.h"
#include "mlops_lm_head.h"
#include "mlops_vocab_items.h"

#endif /* MLOPS_VOCAB_H */
