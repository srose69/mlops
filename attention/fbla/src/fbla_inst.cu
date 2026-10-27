// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// One object per format (MLOPS_FBLA_CB / FB, Makefile).  The kernels that do not depend on the format (the
// select and merge passes, the prefill's sink / open update, the tensor-core attention) are compiled into the 3+3
// object only, with the host functions that launch them (launch_select, launch_merge, launch_finish) for both.
#if MLOPS_FBLA_CB == 3
#define MLOPS_FBLA_DEFINE_SHARED
#endif
#include "fbla_qkv.cuh"

template cudaError_t mlops_fbla_detail::launch<MLOPS_FBLA_CB, MLOPS_FBLA_FB>(
    const mlops_fbla_detail::Args &, int, cudaStream_t);
template cudaError_t mlops_fbla_detail::launch_prefill<MLOPS_FBLA_CB, MLOPS_FBLA_FB>(
    const mlops_fbla_detail::PrefillArgs &, cudaStream_t);
template cudaError_t mlops_fbla_detail::launch_append<MLOPS_FBLA_CB, MLOPS_FBLA_FB>(
    const mlops_fbla_detail::PrefillArgs &, cudaStream_t);
template cudaError_t mlops_fbla_detail::launch_qkv<MLOPS_FBLA_CB, MLOPS_FBLA_FB>(
    const mlops_fbla_detail::QkvArgs &, cudaStream_t);
