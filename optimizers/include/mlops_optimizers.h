/* SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
 * Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
 *
 * mlops-optimizers: every optimizer of the package (libmlops_optimizers.a).
 *   mlops_adamw.h       AdamW on dense BF16 / FP16 / FP32 parameters, FP32 moments
 *   mlops_adamw_qat.h   AdamW that trains an NVFP4 weight in place (needs nvfp4effma.h)
 */
#ifndef MLOPS_OPTIMIZERS_H
#define MLOPS_OPTIMIZERS_H

#define MLOPS_OPTIMIZERS_VERSION "0.1.0"

#include "mlops_adamw.h"
#include "mlops_adamw_qat.h"

#endif /* MLOPS_OPTIMIZERS_H */
