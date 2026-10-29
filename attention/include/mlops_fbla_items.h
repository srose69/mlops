// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE, LICENSE-EXCEPTION (additional terms).
/* FBLA work items: the work of a call as passes of items that an executor runs from kernels of its own, together
 * with what each item reads and writes.  The items of a call compute what the call does, bit for bit.
 *
 * mlops_fbla_prefill_at_items / mlops_fbla_append_at_items / mlops_fbla_decode_at_items / mlops_fbla_qkv_at_items
 * take the arguments of the equally named calls (validated the same way) and describe the call on the current
 * device as a sequence of passes, writing up to `max` of them and returning their number (-1 and
 * mlops_fbla_last_error() on bad arguments).  A pass describes:
 *   kind        MLOPS_FBLA_ITEM_CLOSE       one group the chunk closes: records and metadata of head y's unit
 *                                           (group (n0 ? (n0 - 1) / G : 0) + x; extra items return)
 *               MLOPS_FBLA_ITEM_TRANSCODE   one closed group's records into its image (the newest first:
 *                                           g = g_end - 1 - x)
 *               MLOPS_FBLA_ITEM_STREAM      PROWS query rows of one KV head on CUDA cores
 *               MLOPS_FBLA_ITEM_TC          one q head x one group's positions on tensor cores, split piece z
 *               MLOPS_FBLA_ITEM_TC_MERGE    16 output rows of one q head, the split pieces combined
 *               MLOPS_FBLA_ITEM_FINISH      one head's sink and open group take the chunk's tokens
 *               MLOPS_FBLA_ITEM_COARSE      one KV head's coarse pass (four queries a warp), x of a.ctas CTAs
 *               MLOPS_FBLA_ITEM_SELECT      one KV head's select pass, x of a.nslices slices
 *               MLOPS_FBLA_ITEM_REFINE      one KV head's refine pass, x of a.rctas CTAs
 *               MLOPS_FBLA_ITEM_MERGE       16 output channels of one decode row
 *               MLOPS_FBLA_ITEM_QKV         one (batch, KV head)'s q/k norm + RoPE + append at the device length
 *   items       their number; item i is the CTA (x, y, z) of the pass's grid, x = i % geom[0], y = (i / geom[0]) %
 *               geom[1], z = i / (geom[0] * geom[1])  (the grid of the equivalent launch)
 *   threads     the CTA one item runs on, exactly (all of its threads call the entry; it synchronizes them)
 *   smem_bytes  shared memory the executor hands the entry (16-byte aligned; MLOPS_FBLA_ITEM_TC: the plan's
 *               dynamic shared memory); synchronize the CTA after an entry before reusing its shared memory
 *   regs        registers a thread of the library's own kernel for this work uses (the executor's may differ)
 *   arch        virtual architecture of that instance (e.g. 75); compile the device entries for compute_<arch>,
 *               then JIT the PTX on the current device, so their CTA shape and shared layout match the plan
 *   entry       the device entry (include/mlops_fbla_items.cuh), a specialization of an item template of
 *               mlops_fbla_detail, called as
 *                 mlops_fbla_detail::<name><..., Pos>(args, ...) with Pos the executor's position policy
 *               (fbla_kernels.cuh: x, y, z, nx, ny, nz, tid and the reads of the length -- ro -- through it);
 *               the entry string names the specialization with mlops_fbla::Pos as the policy, to substitute.
 *               Entry arguments after args, by kind:
 *                 CLOSE: uint8_t (&code)[2][128][128]; STREAM: StreamSmem<Warps<CB>::PROWS>&;
 *                 FINISH: none; COARSE/REFINE: SmemCoarse<QG>& / SmemRefine<QG>& (QG=4: SmemCoarse4 /
 *                 SmemRefine4); SELECT: float (&R)[QG], int (&wc)[W]; MERGE: SmemMerge&;
 *                 QKV: int head (= x), QkvSmem&. These shared objects live in the executor's aligned storage.
 *                 TRANSCODE: uint8_t* tiles, size_t hstep, size_t gstep, int toff, bool flagged, uint8_t* smem
 *                 from geom[3..7]; TC: const uint8_t* tiles, size_t hstep, size_t gstep, int toff,
 *                 float2* part_ml, float* part_o, uint8_t* smem from geom[3..8]; TC_MERGE:
 *                 const float2* part_ml, const float* part_o, int nsplit from geom[3..5].
 *               TC's geom[9/10] are the partial allocations' byte sizes, geom[11] is nsplit. When nsplit > 1,
 *               the executor allocates both arrays, fills TC geom[7/8] and TC_MERGE geom[3/4] with their pointers,
 *               and keeps them alive through the merge. Plan creation allocates no device memory.
 *   args        a 16-byte aligned device copy of args.bytes[0 .. args_bytes - 1] is the entry's first argument
 *               (the call's device argument structure, args_type).
 * The pass sequence of a call is its launches' order (a stream's); an item of pass p depends on the items of the
 * passes before it exactly as mlops_fbla_item_access's ranges overlap (write after read, read after write) --
 * and on nothing else. Actual stores within a pass are disjoint. Device-dependent groups and rows are described
 * by conservative envelopes over all allowed positions/counts; these envelopes may overlap between items.
 * An executor may keep the resulting conservative dependencies, or narrow them using its known device state.
 *
 * The cache's records, metadata, sink, open groups and images are read and written through coherent L2; another
 * CTA may hold the producing pass's outputs in the same launch.  The executor must establish GPU-scope
 * release/acquire ordering between the passes' items before calling the entries. q, k, v and the position inputs
 * remain stable while consumed; scratch is written by its producing pass before a later pass reads it.
 * Token norm weights and RoPE frequencies are immutable during the executor launch.
 * The length and count (device memory) are read through the position policy's ro,
 * at most once per entry call site: the executor's policy may read them through L2.  The entries do not
 * implement the executor's inter-CTA waits or signals.
 *
 * mlops_fbla_item_access lists what an item reads and writes as ranges of bytes relative to a call argument's
 * pointer: `rows` runs of `bytes` bytes, `stride` apart, from `offset`.  The counts and lengths of a call with
 * its length and count in device memory are the ceiling's: the ranges cover every count up to it.  write: 0
 * read, 1 store. */
#ifndef MLOPS_FBLA_ITEMS_H
#define MLOPS_FBLA_ITEMS_H

#include "mlops_fbla.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { MLOPS_FBLA_ITEM_CLOSE = 1, MLOPS_FBLA_ITEM_TRANSCODE, MLOPS_FBLA_ITEM_STREAM, MLOPS_FBLA_ITEM_TC,
       MLOPS_FBLA_ITEM_TC_MERGE, MLOPS_FBLA_ITEM_FINISH, MLOPS_FBLA_ITEM_COARSE, MLOPS_FBLA_ITEM_SELECT,
       MLOPS_FBLA_ITEM_REFINE, MLOPS_FBLA_ITEM_MERGE, MLOPS_FBLA_ITEM_QKV };

/* the argument a range is relative to: q, k, v, output; the cache's coarse records, fine records, metadata,
 * sink K / V, open K / V, images; the decode's scratch; the device length and count */
enum {
  MLOPS_FBLA_ARG_Q = 0, MLOPS_FBLA_ARG_K, MLOPS_FBLA_ARG_V, MLOPS_FBLA_ARG_OUT,
  MLOPS_FBLA_ARG_COARSE, MLOPS_FBLA_ARG_FINE, MLOPS_FBLA_ARG_META, MLOPS_FBLA_ARG_SINK_K, MLOPS_FBLA_ARG_SINK_V,
  MLOPS_FBLA_ARG_OPEN_K, MLOPS_FBLA_ARG_OPEN_V, MLOPS_FBLA_ARG_IMAGES,
  MLOPS_FBLA_ARG_SCRATCH,  /* the scratch's first byte; its layout is the plan's geom */
  MLOPS_FBLA_ARG_LENGTH, MLOPS_FBLA_ARG_COUNT,  /* the device ints themselves */
  MLOPS_FBLA_ARG_TQ, MLOPS_FBLA_ARG_TK, MLOPS_FBLA_ARG_TV,  /* mlops_fbla_token's q, k, v */
  MLOPS_FBLA_ARG_TQN, MLOPS_FBLA_ARG_TKN, MLOPS_FBLA_ARG_TFREQ, MLOPS_FBLA_ARG_TQ_OUT,
  MLOPS_FBLA_ARG_PART_ML, MLOPS_FBLA_ARG_PART_O /* executor-owned TC split partials */
};

typedef struct {
  int arg, write;
  long long offset, bytes, stride;
  int rows;
} mlops_fbla_item_range;

/* Shared geometry slots. Entry-specific arguments occupy slots 3..11, as documented above. */
enum {
  MLOPS_FBLA_GEOM_GRID_X = 0, MLOPS_FBLA_GEOM_GRID_Y, MLOPS_FBLA_GEOM_GRID_Z,
  MLOPS_FBLA_GEOM_HEAD_STRIDE = 20, MLOPS_FBLA_GEOM_GROUP_STRIDE, MLOPS_FBLA_GEOM_CAPACITY_GROUPS,
  MLOPS_FBLA_GEOM_COARSE_BYTES, MLOPS_FBLA_GEOM_FINE_BYTES,
  MLOPS_FBLA_GEOM_LSE, MLOPS_FBLA_GEOM_PART, MLOPS_FBLA_GEOM_DPART, MLOPS_FBLA_GEOM_ROW_LSE,
  MLOPS_FBLA_GEOM_ITEMS, MLOPS_FBLA_GEOM_COUNTS
};

#define MLOPS_FBLA_ITEM_ARGS_BYTES 256
typedef struct {
  int kind, items, threads, smem_bytes, regs, arch;
  char entry[160], args_type[48];
  int args_bytes;
  union { long long align; unsigned char bytes[MLOPS_FBLA_ITEM_ARGS_BYTES]; } args;
  long long geom[32];  /* (mlops_fbla_item_access, and the entry's own arguments after the args blob) */
} mlops_fbla_items;

int mlops_fbla_prefill_at_items(const mlops_fbla_config *cfg, const int *length, const int *count, int max_count,
                                const uint16_t *q, const uint16_t *k, const uint16_t *v, const mlops_fbla_cache *cache,
                                uint16_t *output, mlops_fbla_items *passes, int max);
int mlops_fbla_append_at_items(const mlops_fbla_config *cfg, const int *length, const uint16_t *k, const uint16_t *v,
                               const mlops_fbla_cache *cache, mlops_fbla_items *passes, int max);
int mlops_fbla_decode_at_items(const mlops_fbla_config *cfg, const int *length, const uint16_t *q,
                               const mlops_fbla_cache *cache, uint16_t *output, void *scratch, size_t scratch_bytes,
                               int per_head, mlops_fbla_items *passes, int max);
/* decode: per_head = 0 uses the stream call's occupancy and scratch layout (size the scratch with a copy of cfg
 * whose tokens = capacity). Positive per_head fixes coarse/refine CTAs per virtual KV head; size that scratch
 * with mlops_fbla_detail::decode_scratch_bytes. In both cases all four passes use the same prepared layout. */
int mlops_fbla_qkv_at_items(const mlops_fbla_config *cfg, const int *length, const mlops_fbla_token *token,
                            const mlops_fbla_cache *cache, mlops_fbla_items *passes, int max);

/* The ranges item `item` of pass `pass` reads and writes, at most max of them to ranges; returns their total
 * number.  max = 0, ranges = NULL queries the count.  NULL passes, an out-of-range item, max < 0 or NULL ranges
 * with max > 0 returns -1. */
int mlops_fbla_item_access(const mlops_fbla_items *passes, int npasses, int pass, int item, mlops_fbla_item_range *ranges,
                           int max);

#ifdef __cplusplus
}
#endif
#endif /* MLOPS_FBLA_ITEMS_H */
