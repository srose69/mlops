/* SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
 * Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
 *
 * mlops-vocab work items: the work of a call as items that an executor runs from kernels of its own, together with
 * what each item reads and writes. The items of a call compute what the call does, bit for bit.
 *
 * mlops_embed_items / mlops_lm_head_items / mlops_sample_items take the arguments of the equally named calls
 * (validated the same way) and describe the call on the current device:
 *   kind        MLOPS_VOCAB_ITEM_EMBED      a range of 256-element units of the call's columns, on one CTA
 *               MLOPS_VOCAB_ITEM_HEAD       a range of rows of W, every column, over all of hidden, CUDA cores
 *               MLOPS_VOCAB_ITEM_HEAD_TC    the same on tensor cores (sm_80+, mma.m16n8k16, n <= 8)
 *               MLOPS_VOCAB_ITEM_SAMPLE     one chunk of one sampling pass (geom[0] = the pass), or the one item of the
 *                                           statistics pass
 *   items       their number; item i is the CTA i of the pass's grid
 *   threads     the CTA one item runs on, exactly (all of its threads call the entry; it synchronizes them)
 *   smem_bytes  shared memory the executor hands over (16-byte aligned); synchronize the CTA after an entry before
 *               reusing its shared memory or beginning the next item
 *   regs        registers a thread of the library's own kernel for this work uses (the executor's may differ)
 *   arch        virtual architecture of that instance (e.g. 80); compile the device entries for compute_<arch>, then
 *               JIT the PTX on the current device
 *   entry       the device entry (include/mlops_vocab_items.cuh), called as
 *                 entry(*(const args_type *)p, item, thread, smem)
 *               with p a 16-byte aligned device copy of args.bytes[0 .. args_bytes - 1], thread the thread's index in
 *               the CTA. The data the arguments point to must stay where they were at the time of the description.
 *
 * A sampling pass is a reduction: its items are unordered, and the one that arrives last (a counter in the
 * workspace, which the entry keeps; it must be zero before the first call and is left reusable) also performs the
 * pass's closing step. The writes of that step are declared by every item of the pass (write = 3), so they overlap
 * between items. The passes of a call run in order; the final one advances the draw counter in the state.
 *
 * The weight W, the table, norm_w and ids (when the producer is an earlier pass of the same launch: read through
 * L2) are as the call's. x, norm_parts and the logits that another item of the same launch writes are read through
 * L2; the executor must establish GPU-scope release/acquire ordering before calling an entry that reads them, and
 * publish an entry's outputs after all its writing threads finish (on sm_6x: a device fence in every writer, a CTA
 * barrier, then an atomic signal). The entries do not implement the executor's inter-CTA waits or signals.
 *
 * mlops_vocab_item_access lists what an item reads and writes as ranges of bytes relative to a call argument's
 * pointer: `rows` runs of `bytes` bytes, `stride` apart, from `offset`. write: 0 read, 1 store, 2 atomic,
 * 3 store by the pass's closing item.
 */
#ifndef MLOPS_VOCAB_ITEMS_H
#define MLOPS_VOCAB_ITEMS_H

#include "mlops_embed.h"
#include "mlops_lm_head.h"
#include "mlops_sample.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { MLOPS_VOCAB_ITEM_EMBED = 1, MLOPS_VOCAB_ITEM_HEAD, MLOPS_VOCAB_ITEM_HEAD_TC, MLOPS_VOCAB_ITEM_SAMPLE };

/* The argument a range is relative to. */
enum {
  MLOPS_VOCAB_ARG_TABLE = 0, MLOPS_VOCAB_ARG_IDS, MLOPS_VOCAB_ARG_NORM_W, MLOPS_VOCAB_ARG_X, MLOPS_VOCAB_ARG_XW,
  MLOPS_VOCAB_ARG_PARTS, MLOPS_VOCAB_ARG_ERROR,                        /* embed: x [n][hidden] is the output */
  MLOPS_VOCAB_ARG_W, MLOPS_VOCAB_ARG_HEAD_X, MLOPS_VOCAB_ARG_NORM_PARTS, MLOPS_VOCAB_ARG_LOGITS,
  MLOPS_VOCAB_ARG_STATS_MAX, MLOPS_VOCAB_ARG_STATS_ARG, MLOPS_VOCAB_ARG_STATS_HIST, /* head */
  MLOPS_VOCAB_ARG_SEEN, MLOPS_VOCAB_ARG_PRE_MAX, MLOPS_VOCAB_ARG_PRE_ARG, MLOPS_VOCAB_ARG_PRE_HIST, MLOPS_VOCAB_ARG_STATE,
  MLOPS_VOCAB_ARG_TOKEN, MLOPS_VOCAB_ARG_WORKSPACE                     /* sample (logits: MLOPS_VOCAB_ARG_LOGITS) */
};

typedef struct {
  int arg, write;
  long long offset, bytes, stride;
  int rows;
} mlops_vocab_item_range;

#define MLOPS_VOCAB_ITEM_ARGS_BYTES 256
typedef struct {
  int kind, items, threads, smem_bytes, regs, arch;
  char entry[160], args_type[48];
  int args_bytes;
  union { long long align; unsigned char bytes[MLOPS_VOCAB_ITEM_ARGS_BYTES]; } args;
  long long geom[16]; /* (mlops_vocab_item_access) */
} mlops_vocab_items;

/* Describe a call: one pass for embed and the head; the sampling passes in order (at most 8): max is the room in
 * `out`, the return value the number of passes written, or -1 (and the module's last_error) for bad arguments. */
int mlops_embed_items(const mlops_embed_config *c, const mlops_embed_variant *variant, const uint16_t *table, const int *ids,
                      const uint16_t *norm_w, float *x, uint16_t *xw, float *parts, int *error, mlops_vocab_items *out);
int mlops_lm_head_items(const mlops_lm_head_config *c, const mlops_lm_head_variant *variant, const uint16_t *w, const uint16_t *x,
                        const float *norm_parts, float *logits, float *stats_max, int *stats_arg, unsigned *stats_hist,
                        mlops_vocab_items *out);
int mlops_sample_items(const mlops_sample_config *c, const float *logits, const uint32_t *seen, const mlops_sample_stats *stats,
                       mlops_sample_state *state, void *workspace, size_t workspace_bytes, int *token, mlops_vocab_items *out, int max);

/* The ranges item `item` reads and writes, at most max of them to ranges; returns their total number.
 * max = 0, ranges = NULL queries the count. NULL items, an out-of-range item, max < 0 or NULL ranges with max > 0
 * returns -1. */
int mlops_vocab_item_access(const mlops_vocab_items *items, int item, mlops_vocab_item_range *ranges, int max);

#ifdef __cplusplus
}
#endif
#endif /* MLOPS_VOCAB_ITEMS_H */
