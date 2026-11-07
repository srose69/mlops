/* SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
 * Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
 */
/* mlops-norm work items: the work of a call as items that an executor runs from kernels of its own, with what each item reads and
 * writes. The items of a call compute what the call does, bit for bit. */
#ifndef MLOPS_NORM_ITEMS_H
#define MLOPS_NORM_ITEMS_H

#include "mlops_norm_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* kind: MLOPS_NORM_ITEM_NORM, a range of rows, each cut in units of 256 over the CTA's warps */
enum { MLOPS_NORM_ITEM_NORM = 1 };
enum { MLOPS_NORM_ARG_X = 0, MLOPS_NORM_ARG_RESIDUAL, MLOPS_NORM_ARG_WEIGHT, MLOPS_NORM_ARG_RESIDUAL_OUT, MLOPS_NORM_ARG_Y,
       MLOPS_NORM_ARG_SCALES, MLOPS_NORM_ARG_GLOBAL, MLOPS_NORM_ARG_BIAS, MLOPS_NORM_ARG_GATE };

/* mlops_norm_item_access lists what an item reads and writes as ranges of bytes relative to a call argument's pointer:
 * `rows` runs of `bytes` bytes, `stride` apart, from `offset`. write: 0 read, 1 store. */
typedef struct {
  int arg, write;
  long long offset, bytes, stride;
  int rows;
} mlops_norm_item_range;

/* The weight and bias are immutable for the executor launch. x, residual and gate are read through coherent L2 (another item of the same
 * launch may have written them); the executor orders GPU-scope release/acquire before calling the entry and publishes its outputs after all
 * writing threads finish (sm_6x: a device fence in every writer, a CTA barrier, an atomic signal). The entry does not implement inter-CTA waits or signals. */

/* items: their number; item i is the CTA i of the call's grid. threads: the CTA one item runs on, exactly (all of its threads call the
 * entry; it synchronizes them). smem_bytes: shared memory the executor hands over (16-byte aligned); synchronize the CTA after an entry
 * before reusing it or beginning the next item. */
/* regs: registers a thread of the library's own kernel for this work uses (the executor's may differ). arch: virtual architecture of that
 * instance; compile the device entries for compute_<arch> (mlops_norm_items.cuh: nvcc or NVRTC). */
/* entry: the device entry (include/mlops_norm_items.cuh), called entry(*(const args_type *)p, item, thread, smem), p a 16-byte aligned
 * device copy of args.bytes[0 .. args_bytes - 1]; the data the arguments point to must stay where it was at the time of the description. */
#define MLOPS_NORM_ITEM_ARGS_BYTES 256
typedef struct {
  int kind, items, threads, smem_bytes, regs, arch;
  char entry[160], args_type[48];
  int args_bytes;
  union { long long align; unsigned char bytes[MLOPS_NORM_ITEM_ARGS_BYTES]; } args;
  long long geom[16]; /* (mlops_norm_item_access) */
} mlops_norm_items;

/* Describe a call on the current device from the arguments of mlops_norm (validated the same way); returns 1, or -1 (and
 * mlops_norm_last_error()) for bad arguments. */
int mlops_norm_describe(const mlops_norm_config *c, const mlops_norm_variant *variant, const mlops_norm_args *args, mlops_norm_items *out);

/* The rows to run come from *rows (device int, at most config.rows; read at the item): the others are left alone. Items and
 * ranges stay those of config.rows. Returns 1, or -1 for a NULL argument. */
int mlops_norm_items_rows_device(mlops_norm_items *items, const int *rows);

/* The ranges item `item` reads and writes, at most max of them to ranges; returns their total number.
 * max = 0, ranges = NULL queries the count. NULL items, an out-of-range item, max < 0 or NULL ranges with max > 0
 * returns -1. */
int mlops_norm_item_access(const mlops_norm_items *items, int item, mlops_norm_item_range *ranges, int max);

#ifdef __cplusplus
}
#endif
#endif /* MLOPS_NORM_ITEMS_H */
