// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// One object per format (MLOPS_FBLA_CB / FB).  Format-independent kernels (select, merge, finish, tensor-core
// attention) are compiled into the 3+3 object only, with their launch functions for both.
#if MLOPS_FBLA_CB == 3
#define MLOPS_FBLA_DEFINE_SHARED
#endif
#include "fbla_qkv.cuh"

template cudaError_t mlops_fbla_detail::launch<MLOPS_FBLA_CB, MLOPS_FBLA_FB>(
    const mlops_fbla_detail::Args &, int, cudaStream_t);
template cudaError_t mlops_fbla_detail::launch_prefill<MLOPS_FBLA_CB, MLOPS_FBLA_FB>(
    const mlops_fbla_detail::PrefillArgs &, cudaStream_t);
template cudaError_t mlops_fbla_detail::launch_prefill_at<MLOPS_FBLA_CB, MLOPS_FBLA_FB>(
    const mlops_fbla_detail::PrefillArgs &, cudaStream_t);
template int mlops_fbla_detail::decode_passes<MLOPS_FBLA_CB, MLOPS_FBLA_FB>(
    const mlops_fbla_detail::Args &, const void *, mlops_fbla_items *, int);
template int mlops_fbla_detail::prefill_at_passes<MLOPS_FBLA_CB, MLOPS_FBLA_FB>(
    const mlops_fbla_detail::PrefillArgs &, bool, mlops_fbla_items *, int);
template int mlops_fbla_detail::qkv_passes<MLOPS_FBLA_CB, MLOPS_FBLA_FB>(
    const mlops_fbla_detail::QkvArgs &, mlops_fbla_items *, int);
template cudaError_t mlops_fbla_detail::launch_append<MLOPS_FBLA_CB, MLOPS_FBLA_FB>(
    const mlops_fbla_detail::PrefillArgs &, cudaStream_t);
template cudaError_t mlops_fbla_detail::launch_qkv<MLOPS_FBLA_CB, MLOPS_FBLA_FB>(
    const mlops_fbla_detail::QkvArgs &, cudaStream_t);
