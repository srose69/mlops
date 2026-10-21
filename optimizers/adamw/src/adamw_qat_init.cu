// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// mlops_adamw_qat_init: NVFP4EffMA's quantization of the initial weight, then mlops_adamw_qat_init_quantized.
// A separate object: a static link pulls it (and NVFP4EffMA) in only for programs that call it.
#include "adamw_qat_kernels.cuh"

extern "C" mlops_adamw_status mlops_adamw_qat_init(int rows, int cols, const void *w, mlops_adamw_dtype w_dtype,
                                                    const nvfp4a_tensor *q, const nvfp4a_quant_config *qcfg,
                                                    const mlops_adamw_qat_state *st, cudaStream_t stream) {
  mlops_qat_detail::Lattice L;  // checks q before anything is written to it
  if (mlops_adamw_status s = mlops_qat_detail::make_lattice(q, rows, cols, &L); s != MLOPS_ADAMW_OK) return s;
  if (!w || !st) return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw_qat_init: null argument");
  if (qcfg && qcfg->tile_rows > 1)
    return MLOPS_ADAMW_FAIL(MLOPS_ADAMW_INVALID_VALUE, "mlops_adamw_qat_init: 1x16 blocks only (tile_rows 1)");
  const nvfp4a_dtype dt = w_dtype == MLOPS_ADAMW_BF16 ? NVFP4A_BF16 : w_dtype == MLOPS_ADAMW_FP16 ? NVFP4A_FP16 : NVFP4A_FP32;
  const nvfp4a_tensor src = {const_cast<void *>(w), dt, nullptr, 0, nullptr, nullptr, 0, 0};
  const nvfp4a_status qs = nvfp4a_quantize_ex(&src, cols, rows, cols, q, cols, nullptr, 0, qcfg, stream);
  if (qs != NVFP4A_OK)
    return MLOPS_ADAMW_FAIL(qs == NVFP4A_CUDA_ERROR ? MLOPS_ADAMW_CUDA_ERROR : MLOPS_ADAMW_INVALID_VALUE,
                            "mlops_adamw_qat_init: nvfp4a_quantize_ex: %s (%s)", nvfp4a_status_string(qs), nvfp4a_last_error());
  return mlops_adamw_qat_init_quantized(rows, cols, w, w_dtype, q, st, stream);
}
