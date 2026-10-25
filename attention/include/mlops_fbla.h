/* SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
 * Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
 *
 * FBLA attention over a compressed KV cache, sm_61+ (DP4A; prefill on tensor cores on sm_75 .. sm_89): decode,
 * and prefill (mlops_fbla_prefill).  Head
 * dim 128; tokens form groups of 128 (per group and channel a BF16 base and step) and sub-blocks of 16 (the unit
 * of selective refinement).
 *
 * Format.  Every element of a closed group has a 6-bit code c, x = base + c * step (base = the group's
 * channel minimum rounded down, step = range / 63 rounded up, both BF16).  The code is stored as two planes,
 * the coarse plane (the top CB bits) and the fine plane (the low FB bits): 3+3 or 4+2.  Without its fine
 * bits an element decodes to the middle of its coarse cell.  The first group (attention sink) and the last
 * group (still being filled) are BF16.  Decode reads every coarse plane once and the fine planes only of the
 * sub-blocks that hold a share W_j > tau of a query's softmax mass (W_j from the coarse scores).  The query is
 * quantized per group to int8: z = q * step_K (BF16 x BF16, exact in FP32), q-hat = rint(z * fl(127 / max|z|))
 * (one rounding of the exact product, ties to even); the scale sl = fl(scale * log2 e) cancels there and enters
 * sq = max|z| * sl / 127 and the base term sl * (q . base_K) only: score = sq * (q-hat . code) + sl * (q . base_K).
 *
 * Planes.  coarse: [batch, kv_heads, capacity_groups * 8, 2 * CB * 256] bytes, fine: the same with FB: one
 * record per 16-token sub-block holding K and V.  A lane l (0..31) of the decoding warp owns 64 codes of K
 * and 64 of V, as 16 slots of four codes (one per byte):
 *   K: lane l = (token l % 16, half l / 16); slot s = channels 64 (l / 16) + 4 s .. + 3
 *   V: lane l = channels 4 l .. + 3;          slot s = (channel 4 l + c, tokens 4 j .. + 3) with (c, j) =
 *        4 bits: s < 8: (s % 2, s / 2), s >= 8: (2 + s % 2, (s - 8) / 2)
 *        3 bits: s < 4: (0, s), 6 <= s < 10: (1, s - 6), s = 4, 5, 10, 11: (2, 0 .. 3), s >= 12: (3, s - 12)
 *        2 bits: s = 4 m + k: (m, k)
 * and packs a tensor's slots into 2 * bits 32-bit words W, bitwise per byte:
 *   4 bits: slot s < 8 in bits 0..3 of W[s], slot 8 + k in bits 4..7 of W[k]
 *   3 bits: slot s < 6 in bits 0..2 of W[s], slot 6 + k in bits 3..5 of W[k], slot 12 + r: bits 0..1 in
 *           bits 6..7 of W[r], bit 2 in bit 6 + r % 2 of W[4 + r / 2]
 *   2 bits: slot 4 m + k in bits 2 m .. 2 m + 1 of W[k]
 * (so that a code word is one AND, and the channel or slot class it leaves scaled by 2^n is folded into the
 * epilogue).  Word bytes in the record (lane l), every load of a warp one contiguous 512 or 256 bytes:
 *   4 bits: K W0..3 at 16 l, K W4..7 at 512 + 16 l, V W0..3 at 1024 + 16 l, V W4..7 at 1536 + 16 l
 *   3 bits: K W0..3 at 16 l, V W0..3 at 512 + 16 l, K W4..5 at 1024 + 16 (l % 16) + 8 (l / 16) (a token's
 *           low words of both halves adjacent), V W4..5 at 1280 + 8 l
 *   2 bits: K W0..3 at 16 l, V W0..3 at 512 + 16 l
 * Metadata: [batch, kv_heads, capacity_groups, 512] BF16, 1 KiB per group: lane l's K base and step of
 * channels 4 l .. + 3 at byte 16 l (base first), V's at byte 512 + 16 l.  Exact groups: exact_sink_* and
 * exact_open_*: [batch, kv_heads, 128, 128] BF16, the open group's tokens at 0 .. n-1.  capacity (0 = tokens)
 * only sets the strides, so one allocation serves a growing context.
 */
#ifndef MLOPS_FBLA_H
#define MLOPS_FBLA_H

#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { MLOPS_FBLA_HEAD_DIM = 128, MLOPS_FBLA_GROUP = 128, MLOPS_FBLA_SUBBLOCK = 16 };

typedef enum {
  MLOPS_FBLA_3P3 = 0, /* 3-bit coarse + 3-bit fine plane, default tau 3e-4 */
  MLOPS_FBLA_4P2 = 1  /* 4-bit coarse + 2-bit fine plane, default tau 1e-3 */
} mlops_fbla_format;
typedef enum {
  MLOPS_FBLA_OK = 0,
  MLOPS_FBLA_INVALID_VALUE = 1,
  MLOPS_FBLA_CUDA_ERROR = 2
} mlops_fbla_status;

typedef struct {  /* decode only reads the cache; prefill writes it (records, metadata, sink, open group) */
  uint8_t *coarse, *fine;
  uint16_t *meta;
  uint16_t *exact_sink_k, *exact_sink_v, *exact_open_k, *exact_open_v;
  /* Optional (null: none): the tensor-core prefill's closed-group images, kept between calls
   * (mlops_fbla_images_bytes, zero-filled before first use).  With it a prefill call rebuilds only the groups closed
   * since, without it all of them.  Kept consistent with the records the library writes (records written otherwise:
   * zero-fill it again); used only with a nonzero cfg->capacity; not by decode. */
  uint8_t *tc_images;
} mlops_fbla_cache;

typedef struct {
  int batch, q_heads, kv_heads, tokens;
  float scale;  /* 0 means 1/sqrt(128) */
  float tau;    /* refinement threshold on W_j; 0 selects the format default */
  mlops_fbla_format format;
  int capacity; /* tokens the cache's planes and metadata are laid out for; 0 means tokens */
} mlops_fbla_config;

/* Scratch for one decode of this configuration (it depends on the current device's SM count).  Its initial
 * contents do not matter; it can be reused for the next call, also for another context length or configuration of
 * the same batch * kv_heads that fits. */
size_t mlops_fbla_decode_scratch_bytes(const mlops_fbla_config *cfg);

/* One decode query per q head: q / output [batch, q_heads, 128] BF16, q_heads / kv_heads = 1, 2 or 4 (the
 * GQA queries of a KV head share every load).  Three kernel launches on `stream`; tokens <= 2^26. */
mlops_fbla_status mlops_fbla_decode(const mlops_fbla_config *cfg, const uint16_t *q,
                                    const mlops_fbla_cache *cache, uint16_t *output,
                                    void *scratch, size_t scratch_bytes, cudaStream_t stream);

/* Bytes of mlops_fbla_cache.tc_images on the current device: 34052 per group of capacity and KV head; 0 where the
 * prefill does not run on tensor cores. */
size_t mlops_fbla_images_bytes(const mlops_fbla_config *cfg);

/* Prefill (and append): `count` new tokens.  k / v [batch, kv_heads, count, 128] BF16 are appended to the
 * cache, which holds cfg->tokens (>= 0) tokens before the call and must have cfg->capacity >= tokens + count
 * (capacity 0 means tokens + count): the groups they close are quantized into the records and metadata, the
 * sink and open groups take their BF16 tokens.  With q non-null, q / output [batch, q_heads, count, 128] BF16:
 * the query of position tokens + i attends to the tokens 0 .. tokens + i, the closed groups through their full
 * 6-bit codes (every block refined; cfg->tau is not used), the sink and its own group BF16.  P blocks (one maximum
 * and weight): DP4A 16 tokens, as decode; tensor cores 128 tokens a closed group (integer P) and 64 in the BF16
 * groups (P rounded to BF16).  q_heads / kv_heads: any integer ratio on tensor cores, 1, 2 or 4 on DP4A.  q null:
 * append only (output unused).  A zero-filled cache (tokens 0) of the capacity's size is an empty cache.  No
 * scratch. */
mlops_fbla_status mlops_fbla_prefill(const mlops_fbla_config *cfg, int count, const uint16_t *q,
                                     const uint16_t *k, const uint16_t *v, const mlops_fbla_cache *cache,
                                     uint16_t *output, cudaStream_t stream);

const char *mlops_fbla_status_string(mlops_fbla_status status);
const char *mlops_fbla_last_error(void);

#ifdef __cplusplus
}
#endif
#endif /* MLOPS_FBLA_H */
