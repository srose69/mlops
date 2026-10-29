// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// PyTorch bindings of mlops-norm (mlops.norm._C): the variants and the call of mlops_norm on tensors. The Python layer
// chooses the flavor, the variant and the output.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <tuple>
#include <vector>
#include "mlops_norm.h"

using at::Tensor;
using OptT = std::optional<Tensor>;

namespace {

mlops_norm_dtype dtype_of(at::ScalarType t, const char *what) {
  switch (t) {
    case at::kBFloat16: return MLOPS_NORM_BF16;
    case at::kHalf: return MLOPS_NORM_FP16;
    case at::kFloat: return MLOPS_NORM_FP32;
    default: TORCH_CHECK_TYPE(false, what, " must be bfloat16, float16 or float32, got ", t);
  }
  return MLOPS_NORM_FP32;
}
at::ScalarType scalar_of(int64_t dt) { return dt == MLOPS_NORM_BF16 ? at::kBFloat16 : dt == MLOPS_NORM_FP16 ? at::kHalf : at::kFloat; }

struct Flavor {
  int64_t kind, scale, bias, order, gate, groups;
};

mlops_norm_config config_of(int64_t rows, int64_t hidden, double eps, const Flavor &f, int64_t x_dtype, int64_t y_dtype, int64_t out, int64_t items) {
  return mlops_norm_config{(int)rows, (int)hidden, (int)f.groups, (float)eps, (mlops_norm_kind)f.kind, (mlops_norm_scale)f.scale, (int)f.bias,
                           (mlops_norm_order)f.order, (mlops_norm_gate)f.gate, (mlops_norm_dtype)x_dtype, (mlops_norm_dtype)y_dtype,
                           (mlops_norm_out)out, 0, (int)items};
}

using Variant = std::tuple<int, int, int, int, int>;  // warps, threads, smem, regs, arch
std::vector<Variant> norm_variants(int64_t rows, int64_t hidden, int64_t groups, int64_t kind, int64_t scale, int64_t bias, int64_t order, int64_t gate,
                                   int64_t x_dtype, int64_t y_dtype, int64_t out) {
  const mlops_norm_config c = config_of(rows, hidden, 0.0, {kind, scale, bias, order, gate, groups}, x_dtype, y_dtype, out, 0);
  std::vector<mlops_norm_variant> v(mlops_norm_variants(&c, nullptr, 0));
  mlops_norm_variants(&c, v.data(), (int)v.size());
  std::vector<Variant> res;
  for (const auto &x : v) res.emplace_back(x.warps, x.threads, x.smem_bytes, x.regs, x.arch);
  return res;
}

void check_like(const OptT &t, const Tensor &x, const char *what, bool same_dtype) {
  if (!t) return;
  TORCH_CHECK_VALUE(t->is_cuda() && t->device() == x.device() && t->is_contiguous(), what, " must be a contiguous tensor on x's device");
  if (same_dtype) TORCH_CHECK_VALUE(t->scalar_type() == x.scalar_type() && t->sizes() == x.sizes(), what, " must be x's shape and dtype");
}

// y (y_dtype, or int8 / uint8 nibbles), the scales (or an empty tensor), the new residual (or an empty one)
std::tuple<Tensor, Tensor, Tensor> norm(Tensor x, OptT residual, OptT weight, OptT bias, OptT gate, double eps, int64_t groups, int64_t kind, int64_t scale,
                                        int64_t order, int64_t gate_mode, int64_t out, int64_t y_dtype, OptT global, int64_t variant, int64_t items) {
  TORCH_CHECK_VALUE(x.dim() == 2 && x.is_contiguous() && x.is_cuda(), "x must be a contiguous CUDA [rows, hidden] tensor");
  const at::Device dev = x.device();
  const int64_t rows = x.size(0), hidden = x.size(1);
  check_like(residual, x, "residual", true);
  check_like(gate, x, "gate", true);
  for (const OptT *p : {&weight, &bias}) {
    check_like(*p, x, "weight / bias", false);
    if (*p) TORCH_CHECK_VALUE((*p)->dim() == 1 && (*p)->numel() == hidden && (*p)->scalar_type() == scalar_of(y_dtype), "weight and bias must be [hidden] in y's dtype");
  }
  if (global) TORCH_CHECK_VALUE(global->is_cuda() && global->device() == dev && global->scalar_type() == at::kFloat && global->numel() == 1,
                                "global_scale must be a float32 CUDA tensor of one element");
  const mlops_norm_config c = config_of(rows, hidden, eps, {kind, scale, bias ? 1 : 0, order, gate_mode, groups}, (int64_t)dtype_of(x.scalar_type(), "x"), y_dtype, out, items);
  std::vector<mlops_norm_variant> v(mlops_norm_variants(&c, nullptr, 0));
  mlops_norm_variants(&c, v.data(), (int)v.size());
  TORCH_CHECK_VALUE(!v.empty(), mlops_norm_last_error()[0] ? mlops_norm_last_error() : "mlops_norm: no variant for this configuration");
  TORCH_CHECK_VALUE(variant >= 0 && variant < (int64_t)v.size(), "variant ", variant, " of ", v.size());
  Tensor y, scales = at::empty({0}, x.options().dtype(at::kByte)), new_residual = at::empty({0}, x.options());
  if (out == MLOPS_NORM_OUT_DTYPE) y = at::empty({rows, hidden}, x.options().dtype(scalar_of(y_dtype)));
  else if (out == MLOPS_NORM_OUT_INT8) {
    y = at::empty({rows, hidden}, x.options().dtype(at::kChar));
    scales = at::empty({rows, hidden / 16}, x.options().dtype(at::kFloat));
  } else {
    y = at::empty({rows, hidden / 2}, x.options().dtype(at::kByte));
    scales = at::empty({rows, hidden / 16}, x.options().dtype(at::kByte));
  }
  if (residual) new_residual = at::empty_like(x);
  mlops_norm_args a{};
  a.x = x.data_ptr();
  a.residual = residual ? residual->data_ptr() : nullptr;
  a.residual_out = residual ? new_residual.data_ptr() : nullptr;
  a.weight = weight ? weight->data_ptr() : nullptr;
  a.bias = bias ? bias->data_ptr() : nullptr;
  a.gate = gate ? gate->data_ptr() : nullptr;
  a.y = y.data_ptr();
  a.scales = out == MLOPS_NORM_OUT_DTYPE ? nullptr : scales.data_ptr();
  a.global = global ? global->data_ptr<float>() : nullptr;
  const c10::cuda::CUDAGuard guard(dev);
  const mlops_norm_status st = mlops_norm(&c, &v[variant], &a, at::cuda::getCurrentCUDAStream().stream());
  if (st == MLOPS_NORM_INVALID_VALUE) TORCH_CHECK_VALUE(false, mlops_norm_last_error());
  TORCH_CHECK(st == MLOPS_NORM_OK, "mlops_norm failed: ", mlops_norm_last_error());
  return {y, scales, new_residual};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "mlops-norm: RMSNorm and LayerNorm with a residual add, gates and INT8 / NVFP4 quantization on CUDA";
  m.attr("__version__") = MLOPS_NORM_VERSION;
  m.def("norm_variants", &norm_variants, py::arg("rows"), py::arg("hidden"), py::arg("groups"), py::arg("kind"), py::arg("scale"), py::arg("bias"),
        py::arg("order"), py::arg("gate"), py::arg("x_dtype"), py::arg("y_dtype"), py::arg("out"));
  m.def("norm", &norm, py::arg("x"), py::arg("residual"), py::arg("weight"), py::arg("bias"), py::arg("gate"), py::arg("eps"), py::arg("groups"),
        py::arg("kind"), py::arg("scale"), py::arg("order"), py::arg("gate_mode"), py::arg("out"), py::arg("y_dtype"), py::arg("global_scale"),
        py::arg("variant"), py::arg("items"));
}
