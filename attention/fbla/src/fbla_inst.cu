// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
#include "fbla_kernels.cuh"

template cudaError_t mlops_fbla_detail::launch<MLOPS_FBLA_CB, MLOPS_FBLA_FB>(
    const mlops_fbla_detail::Args &, int, cudaStream_t);
