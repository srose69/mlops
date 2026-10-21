// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// PyTorch bindings of mlops-positional (mlops.positional._C): mlops_rope on [B, S, H, D] views (the Python layer
// turns [B, H, S, D] and [T, H, D] into them) and the frequency table.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include "mlops_positional.h"

using at::Tensor;
using OptT = std::optional<Tensor>;

namespace {

mlops_rope_dtype dtype_of(at::ScalarType t) {
  switch (t) {
    case at::kBFloat16: return MLOPS_ROPE_BF16;
    case at::kHalf: return MLOPS_ROPE_FP16;
    case at::kFloat: return MLOPS_ROPE_FP32;
    default: TORCH_CHECK_TYPE(false, "unsupported dtype ", t, " (expected bfloat16, float16 or float32)");
  }
  return MLOPS_ROPE_FP32;
}

mlops_rope_tensor view(const Tensor &t, const at::Device &dev, at::ScalarType dt, const char *what) {
  TORCH_CHECK_VALUE(t.is_cuda() && t.device() == dev, what, " must be a CUDA tensor on ", dev, ", got ", t.device());
  TORCH_CHECK_TYPE(t.scalar_type() == dt, what, " must be ", dt, " like q, got ", t.scalar_type());
  TORCH_CHECK_VALUE(t.dim() == 4 && t.stride(3) == 1, what, " must be a [B, S, H, D] view with a contiguous last dimension");
  return mlops_rope_tensor{t.data_ptr(), (int)t.size(2), t.stride(0), t.stride(1), t.stride(2)};
}

void rope(Tensor q_in, Tensor q_out, OptT k_in, OptT k_out, int64_t rotary_dim, bool interleaved, bool inverse,
          OptT positions, int64_t position_offset, OptT freq, double attention_factor, OptT cos, OptT sin) {
  const at::Device dev = q_in.device();
  const at::ScalarType dt = q_in.scalar_type();
  dtype_of(dt);
  const mlops_rope_tensor qi = view(q_in, dev, dt, "q"), qo = view(q_out, dev, dt, "q out");
  TORCH_CHECK_VALUE(q_out.sizes() == q_in.sizes(), "q out must have q's shape ", q_in.sizes());
  const int64_t B = q_in.size(0), S = q_in.size(1), D = q_in.size(3);
  mlops_rope_tensor ki{}, ko{};
  if (k_in) {
    TORCH_CHECK_VALUE(k_out.has_value(), "k out is required with k");
    ki = view(*k_in, dev, dt, "k");
    ko = view(*k_out, dev, dt, "k out");
    TORCH_CHECK_VALUE(k_in->size(0) == B && k_in->size(1) == S && k_in->size(3) == D,
                      "k must have q's batch, sequence and head_dim [", B, ", ", S, ", *, ", D, "], got ", k_in->sizes());
    TORCH_CHECK_VALUE(k_out->sizes() == k_in->sizes(), "k out must have k's shape ", k_in->sizes());
  }
  mlops_rope_config c{};
  c.batch = (int)B; c.seq = (int)S; c.head_dim = (int)D; c.rotary_dim = (int)rotary_dim;
  c.interleaved = interleaved; c.inverse = inverse; c.dtype = dtype_of(dt);
  if (cos || sin) {
    TORCH_CHECK_VALUE(cos && sin, "cos and sin must both be given");
    for (const Tensor *t : {&*cos, &*sin}) {
      TORCH_CHECK_VALUE(t->is_cuda() && t->device() == dev, "cos / sin must be CUDA tensors on ", dev);
      TORCH_CHECK_TYPE(t->scalar_type() == dt, "cos / sin must have q's dtype ", dt, ", got ", t->scalar_type());
      TORCH_CHECK_VALUE(t->dim() == 3 && t->size(0) == B && t->size(1) == S && t->size(2) == rotary_dim && t->stride(2) == 1,
                        "cos / sin must be [", B, ", ", S, ", ", rotary_dim, "] views with a contiguous last dimension, got ",
                        t->sizes());
    }
    TORCH_CHECK_VALUE(cos->stride(0) == sin->stride(0) && cos->stride(1) == sin->stride(1), "cos and sin must have the same strides");
    c.cos = cos->data_ptr(); c.sin = sin->data_ptr();
    c.table_stride_b = cos->stride(0); c.table_stride_s = cos->stride(1);
  } else {
    TORCH_CHECK_VALUE(freq.has_value(), "freq (or cos and sin) is required");
    TORCH_CHECK_VALUE(freq->is_cuda() && freq->device() == dev && freq->scalar_type() == at::kFloat && freq->is_contiguous() &&
                          freq->numel() == 2 * rotary_dim,
                      "freq must be a contiguous float32 [rotary_dim / 2, 4] CUDA tensor on ", dev);
    c.freq = freq->data_ptr<float>();
    c.attention_factor = (float)attention_factor;
    c.position_offset = position_offset;
    if (positions) {
      TORCH_CHECK_VALUE(positions->is_cuda() && positions->device() == dev, "positions must be a CUDA tensor on ", dev);
      TORCH_CHECK_TYPE(positions->scalar_type() == at::kInt || positions->scalar_type() == at::kLong,
                       "positions must be int32 or int64, got ", positions->scalar_type());
      TORCH_CHECK_VALUE(positions->dim() == 2 && positions->size(0) == B && positions->size(1) == S,
                        "positions must be a [", B, ", ", S, "] view, got ", positions->sizes());
      c.positions = positions->data_ptr();
      c.positions_int64 = positions->scalar_type() == at::kLong;
      c.pos_stride_b = positions->stride(0); c.pos_stride_s = positions->stride(1);
    }
  }
  const c10::cuda::CUDAGuard guard(dev);
  const mlops_rope_status st = mlops_rope(&c, &qi, &qo, k_in ? &ki : nullptr, k_in ? &ko : nullptr,
                                          at::cuda::getCurrentCUDAStream().stream());
  if (st == MLOPS_ROPE_INVALID_VALUE) TORCH_CHECK_VALUE(false, mlops_rope_last_error());
  TORCH_CHECK(st == MLOPS_ROPE_OK, "mlops_rope failed (", mlops_rope_status_string(st), "): ", mlops_rope_last_error());
}

// inverse frequencies (float64, CPU) -> the float32 [n, 4] table mlops_rope reads (CPU)
Tensor freq_table(const Tensor &inv_freq) {
  TORCH_CHECK_VALUE(inv_freq.dim() == 1 && inv_freq.scalar_type() == at::kDouble && !inv_freq.is_cuda(),
                    "inv_freq must be a 1-D float64 CPU tensor");
  const Tensor f = inv_freq.contiguous();
  Tensor out = at::empty({f.size(0), 4}, f.options().dtype(at::kFloat));
  mlops_rope_freq(f.data_ptr<double>(), (int)f.size(0), out.data_ptr<float>());
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "mlops-positional: positional encodings (RoPE) on CUDA";
  m.attr("__version__") = MLOPS_POSITIONAL_VERSION;
  m.def("rope", &rope, py::arg("q_in"), py::arg("q_out"), py::arg("k_in") = py::none(), py::arg("k_out") = py::none(),
        py::arg("rotary_dim"), py::arg("interleaved") = false, py::arg("inverse") = false, py::arg("positions") = py::none(),
        py::arg("position_offset") = 0, py::arg("freq") = py::none(), py::arg("attention_factor") = 1.0,
        py::arg("cos") = py::none(), py::arg("sin") = py::none());
  m.def("freq_table", &freq_table, py::arg("inv_freq"));
}
