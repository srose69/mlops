// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// PyTorch bindings of mlops-optimizers (mlops.optim._C): mlops_adamw / mlops_adamw_qat (../../include). The NVFP4 tensor is taken as NVFP4EffMA's own
// bindings take operands: flat (data, scales, global, flags, scale_max), unpacked by the Python layer from an nvfp4effma.NVFP4Tensor; the QAT state is
// one uint8 tensor. The initial quantization is nvfp4effma.quantize's (Python), so nothing here links NVFP4EffMA.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <string>
#include "mlops_optimizers.h"

using at::Tensor;
using OptT = std::optional<Tensor>;

namespace {

mlops_adamw_dtype dense_dtype(at::ScalarType t, const char *what) {
  switch (t) {
    case at::kBFloat16: return MLOPS_ADAMW_BF16;
    case at::kHalf: return MLOPS_ADAMW_FP16;
    case at::kFloat: return MLOPS_ADAMW_FP32;
    default: TORCH_CHECK_TYPE(false, what, ": unsupported dtype ", t, " (expected bfloat16, float16 or float32)");
  }
  return MLOPS_ADAMW_FP32;
}

void check_state(const Tensor &param, const Tensor &t, const char *what) {
  TORCH_CHECK_VALUE(t.is_cuda() && t.device() == param.device(), what, " must be a CUDA tensor on ", param.device());
  TORCH_CHECK_TYPE(t.scalar_type() == at::kFloat, what, " must be float32, got ", t.scalar_type());
  TORCH_CHECK_VALUE(t.is_contiguous() && t.numel() == param.numel(), what, " must be contiguous with ", param.numel(),
                    " elements, got shape ", t.sizes());
}

void check_status(mlops_adamw_status s, const char *what) {
  if (s == MLOPS_ADAMW_OK) return;
  TORCH_CHECK(false, what, " failed (", mlops_adamw_status_string(s), "): ", mlops_adamw_last_error());
}

// gradient of a QAT step: FP8 (E5M2 / E4M3, decoded value times grad_scale) or FP32
mlops_adamw_dtype qat_grad_dtype(at::ScalarType t) {
  switch (t) {
    case at::kFloat8_e5m2: return MLOPS_ADAMW_E5M2;
    case at::kFloat8_e4m3fn: return MLOPS_ADAMW_E4M3;
    case at::kFloat: return MLOPS_ADAMW_FP32;
    default: TORCH_CHECK_TYPE(false, "grad: unsupported dtype ", t, " (expected float8_e5m2, float8_e4m3fn or float32)");
  }
  return MLOPS_ADAMW_FP32;
}

Tensor contig_dense(const Tensor &t, const at::Device &dev, const char *what) {
  TORCH_CHECK_VALUE(t.is_cuda() && t.device() == dev, what, " must be a CUDA tensor on ", dev);
  dense_dtype(t.scalar_type(), what);
  TORCH_CHECK_VALUE(t.is_contiguous(), what, " must be contiguous");
  return t;
}

// An NVFP4 operand's flat parts, as nvfp4effma.NVFP4Tensor._flat() gives them, -> nvfp4a_tensor.
nvfp4a_tensor as_nvfp4(const OptT &data, const OptT &scales, const OptT &global, int64_t flags, int64_t smax,
                      const at::Device &dev, const char *what) {
  TORCH_CHECK_VALUE(data.has_value() == scales.has_value(), what, ": data and scales must both be given, or neither");
  if (!data) return nvfp4a_tensor{};
  TORCH_CHECK_VALUE(data->is_cuda() && data->device() == dev, what, ".data must be a CUDA tensor on ", dev);
  TORCH_CHECK_TYPE(data->scalar_type() == at::kByte, what, ".data must be uint8, got ", data->scalar_type());
  TORCH_CHECK_VALUE(data->is_contiguous(), what, ".data must be contiguous");
  TORCH_CHECK_VALUE(scales->is_contiguous(), what, ".scales must be contiguous");
  const float *g = nullptr;
  if (global) {
    TORCH_CHECK_TYPE(global->scalar_type() == at::kFloat, what, ".global must be float32, got ", global->scalar_type());
    TORCH_CHECK_VALUE(global->is_contiguous(), what, ".global must be contiguous");
    g = global->data_ptr<float>();
  }
  return nvfp4a_tensor{data->data_ptr(), NVFP4A_NVFP4, scales->data_ptr(), 0, g, nullptr, (unsigned)flags, (int)smax};
}

void adamw(Tensor param, Tensor grad, Tensor exp_avg, Tensor exp_avg_sq, double lr, double beta1, double beta2,
          double eps, double weight_decay, int64_t step, bool bias_correction, double grad_scale) {
  const auto dev = param.device();
  dense_dtype(param.scalar_type(), "param");
  dense_dtype(grad.scalar_type(), "grad");
  check_state(param, exp_avg, "exp_avg");
  check_state(param, exp_avg_sq, "exp_avg_sq");
  TORCH_CHECK_VALUE(param.is_contiguous() && grad.is_contiguous(), "param and grad must be contiguous");
  TORCH_CHECK_VALUE(grad.numel() == param.numel() && grad.device() == dev, "grad must match param's shape and device");
  const mlops_adamw_config cfg{(float)lr, (float)beta1, (float)beta2, (float)eps, (float)weight_decay, (int)step,
                               bias_correction ? 1 : 0, (float)grad_scale};
  const c10::cuda::CUDAGuard guard(dev);
  check_status(mlops_adamw(&cfg, param.numel(), param.data_ptr(), dense_dtype(param.scalar_type(), "param"),
                           grad.data_ptr(), dense_dtype(grad.scalar_type(), "grad"), exp_avg.data_ptr<float>(),
                           exp_avg_sq.data_ptr<float>(), at::cuda::getCurrentCUDAStream().stream()),
               "mlops_adamw");
}

mlops_adamw_qat_state bind_state(const Tensor &state, int64_t rows, int64_t cols, const at::Device &dev) {
  TORCH_CHECK_VALUE(state.is_cuda() && state.device() == dev && state.scalar_type() == at::kByte && state.is_contiguous(),
                    "state must be a contiguous uint8 CUDA tensor on ", dev);
  TORCH_CHECK_VALUE((size_t)state.numel() >= mlops_adamw_qat_state_bytes((int)rows, (int)cols),
                    "state has ", state.numel(), " bytes, mlops_adamw_qat_state_bytes(rows, cols) = ",
                    mlops_adamw_qat_state_bytes((int)rows, (int)cols));
  mlops_adamw_qat_state st;
  check_status(mlops_adamw_qat_state_bind(state.data_ptr(), (int)rows, (int)cols, &st), "mlops_adamw_qat_state_bind");
  return st;
}

int64_t qat_state_bytes(int64_t rows, int64_t cols) { return (int64_t)mlops_adamw_qat_state_bytes((int)rows, (int)cols); }

// q: w already quantized by nvfp4effma.quantize (1x16 blocks, E4M3 or UE5M3 scales)
void qat_init_quantized(Tensor w, Tensor state, Tensor q_data, Tensor q_scales, OptT q_global, int64_t q_flags,
                        int64_t q_smax) {
  const auto dev = w.device();
  TORCH_CHECK_VALUE(w.dim() == 2 && w.is_contiguous() && w.is_cuda(), "w must be a contiguous 2-D CUDA tensor");
  const int64_t rows = w.size(0), cols = w.size(1);
  const mlops_adamw_qat_state st = bind_state(state, rows, cols, dev);
  const nvfp4a_tensor q = as_nvfp4(q_data, q_scales, q_global, q_flags, q_smax, dev, "q");
  const c10::cuda::CUDAGuard guard(dev);
  check_status(mlops_adamw_qat_init_quantized((int)rows, (int)cols, w.data_ptr(), dense_dtype(w.scalar_type(), "w"), &q,
                                              &st, at::cuda::getCurrentCUDAStream().stream()),
               "mlops_adamw_qat_init_quantized");
}

void qat_step(Tensor grad, Tensor state, Tensor q_data, Tensor q_scales, OptT q_global, int64_t q_flags, int64_t q_smax,
              double lr, double beta1, double beta2, double eps, double weight_decay, int64_t step, double grad_scale,
              double scale_z, int64_t seed) {
  const auto dev = grad.device();
  TORCH_CHECK_VALUE(grad.dim() == 2 && grad.is_contiguous() && grad.is_cuda(), "grad must be a contiguous 2-D CUDA tensor");
  const int64_t rows = grad.size(0), cols = grad.size(1);
  const mlops_adamw_qat_state st = bind_state(state, rows, cols, dev);
  const nvfp4a_tensor q = as_nvfp4(q_data, q_scales, q_global, q_flags, q_smax, dev, "q");
  const mlops_adamw_qat_config cfg{(float)lr, (float)beta1, (float)beta2, (float)eps, (float)weight_decay, (int)step,
                                   (float)grad_scale, (float)scale_z, (uint32_t)seed};
  const c10::cuda::CUDAGuard guard(dev);
  check_status(mlops_adamw_qat(&cfg, (int)rows, (int)cols, grad.data_ptr(), qat_grad_dtype(grad.scalar_type()), &q,
                               &st, at::cuda::getCurrentCUDAStream().stream()),
               "mlops_adamw_qat");
}

void check_seg(const Tensor &seg, int64_t groups, const at::Device &dev) {
  TORCH_CHECK_VALUE(seg.is_cuda() && seg.device() == dev, "seg must be a CUDA tensor on ", dev);
  TORCH_CHECK_TYPE(seg.scalar_type() == at::kInt, "seg must be int32, got ", seg.scalar_type());
  TORCH_CHECK_VALUE(seg.dim() == 1 && seg.size(0) == groups + 1 && seg.is_contiguous(), "seg must be a contiguous [", groups + 1, "] tensor");
}

// dx [rows, k] (gy's rows grouped by seg) = per group gy_g @ w_g, w the dequantized stacked weight [groups * n, k] float32; rows past seg[groups] are zero
Tensor grouped_dgrad(const Tensor &gy, const Tensor &w, const Tensor &seg, int64_t groups, at::ScalarType out_dtype) {
  const auto dev = gy.device();
  contig_dense(gy, dev, "gy");
  dense_dtype(out_dtype, "dx");
  TORCH_CHECK_VALUE(gy.dim() == 2 && w.dim() == 2 && groups >= 1 && w.size(0) % groups == 0, "gy must be [rows, n] and w [groups * n, k]");
  TORCH_CHECK_TYPE(w.scalar_type() == at::kFloat && w.is_cuda() && w.device() == dev && w.is_contiguous(), "w must be a contiguous float32 CUDA tensor on ", dev);
  const int64_t rows = gy.size(0), n = w.size(0) / groups, k = w.size(1);
  TORCH_CHECK_VALUE(gy.size(1) == n, "gy has ", gy.size(1), " columns, the weight's groups have n = ", n);
  check_seg(seg, groups, dev);
  Tensor dx = at::zeros({rows, k}, gy.options().dtype(out_dtype));
  const c10::cuda::CUDAGuard guard(dev);
  check_status(mlops_qat_grouped_dgrad((int)groups, (int)n, (int)k, (int)rows, gy.data_ptr(), dense_dtype(gy.scalar_type(), "gy"), (int)n,
                                       w.data_ptr<float>(), (int)k, dx.data_ptr(), dense_dtype(out_dtype, "dx"), (int)k, seg.data_ptr<int>(),
                                       at::cuda::getCurrentCUDAStream().stream()),
               "mlops_qat_grouped_dgrad");
  return dx;
}

// dw [groups * n, k] float32 = per group gy_g^T @ x_g (zeros for a group without rows)
Tensor grouped_wgrad(const Tensor &gy, const Tensor &x, const Tensor &seg, int64_t groups) {
  const auto dev = gy.device();
  contig_dense(gy, dev, "gy");
  contig_dense(x, dev, "x");
  TORCH_CHECK_VALUE(gy.dim() == 2 && x.dim() == 2 && gy.size(0) == x.size(0) && groups >= 1, "gy must be [rows, n] and x [rows, k]");
  const int64_t rows = gy.size(0), n = gy.size(1), k = x.size(1);
  check_seg(seg, groups, dev);
  Tensor dw = at::empty({groups * n, k}, gy.options().dtype(at::kFloat));
  const c10::cuda::CUDAGuard guard(dev);
  check_status(mlops_qat_grouped_wgrad((int)groups, (int)n, (int)k, (int)rows, gy.data_ptr(), dense_dtype(gy.scalar_type(), "gy"), (int)n, x.data_ptr(),
                                       dense_dtype(x.scalar_type(), "x"), (int)k, dw.data_ptr<float>(), (int)k, seg.data_ptr<int>(),
                                       at::cuda::getCurrentCUDAStream().stream()),
               "mlops_qat_grouped_wgrad");
  return dw;
}

Tensor grouped_dgrad_meta(const Tensor &gy, const Tensor &w, const Tensor &, int64_t groups, at::ScalarType out_dtype) {
  TORCH_CHECK_VALUE(gy.dim() == 2 && w.dim() == 2 && groups >= 1, "gy must be [rows, n] and w [groups * n, k]");
  return at::empty({gy.size(0), w.size(1)}, gy.options().dtype(out_dtype));
}

Tensor grouped_wgrad_meta(const Tensor &gy, const Tensor &x, const Tensor &, int64_t groups) {
  TORCH_CHECK_VALUE(gy.dim() == 2 && x.dim() == 2 && groups >= 1, "gy must be [rows, n] and x [rows, k]");
  return at::empty({groups * gy.size(1), x.size(1)}, gy.options().dtype(at::kFloat));
}

} // namespace

// The operations as PyTorch operators (torch.ops.mlops_optim.*), with their schemas (what each one mutates) and shape-only Meta kernels: torch.compile
// and CUDA graphs treat each call as one opaque node instead of breaking the graph at the extension's functions.  The calls enqueue on the current
// stream and read nothing back, except what the optimizers take as host numbers (lr, step, grad_scale).
TORCH_LIBRARY(mlops_optim, m) {
  m.def("grouped_dgrad(Tensor gy, Tensor w, Tensor seg, int groups, ScalarType out_dtype) -> Tensor");
  m.def("grouped_wgrad(Tensor gy, Tensor x, Tensor seg, int groups) -> Tensor");
  m.def("adamw(Tensor(a!) param, Tensor grad, Tensor(b!) exp_avg, Tensor(c!) exp_avg_sq, float lr, float beta1, float beta2, float eps, "
        "float weight_decay, int step, bool bias_correction, float grad_scale) -> ()");
  m.def("qat_step(Tensor grad, Tensor(a!) state, Tensor(b!) q_data, Tensor(c!) q_scales, Tensor(d!)? q_global, int q_flags, int q_smax, float lr, "
        "float beta1, float beta2, float eps, float weight_decay, int step, float grad_scale, float scale_z, int seed) -> ()");
}

TORCH_LIBRARY_IMPL(mlops_optim, CUDA, m) {
  m.impl("grouped_dgrad", &grouped_dgrad);
  m.impl("grouped_wgrad", &grouped_wgrad);
  m.impl("adamw", &adamw);
  m.impl("qat_step", &qat_step);
}

TORCH_LIBRARY_IMPL(mlops_optim, Meta, m) {
  m.impl("grouped_dgrad", &grouped_dgrad_meta);
  m.impl("grouped_wgrad", &grouped_wgrad_meta);
  m.impl("adamw", [](Tensor, Tensor, Tensor, Tensor, double, double, double, double, double, int64_t, bool, double) {});
  m.impl("qat_step", [](Tensor, Tensor, Tensor, Tensor, OptT, int64_t, int64_t, double, double, double, double, double, int64_t, double, double,
                        int64_t) {});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "mlops-optimizers: AdamW / NVFP4 AdamW-QAT CUDA kernels";
  m.attr("__version__") = MLOPS_OPTIMIZERS_VERSION;
  // the versions whose inline device code (headers) the library was compiled with; "" = not used
  m.attr("built_nvfp4effma") = MLOPS_BUILT_NVFP4EFFMA;
  m.attr("built_e5m2effma") = MLOPS_BUILT_E5M2EFFMA;
  m.def("adamw", &adamw, py::arg("param"), py::arg("grad"), py::arg("exp_avg"), py::arg("exp_avg_sq"), py::arg("lr"),
        py::arg("beta1"), py::arg("beta2"), py::arg("eps"), py::arg("weight_decay"), py::arg("step"),
        py::arg("bias_correction"), py::arg("grad_scale"));
  m.def("grouped_dgrad", &grouped_dgrad, py::arg("gy"), py::arg("w"), py::arg("seg"), py::arg("groups"), py::arg("out_dtype"));
  m.def("grouped_wgrad", &grouped_wgrad, py::arg("gy"), py::arg("x"), py::arg("seg"), py::arg("groups"));
  m.def("qat_state_bytes", &qat_state_bytes, py::arg("rows"), py::arg("cols"));
  m.def("qat_init_quantized", &qat_init_quantized, py::arg("w"), py::arg("state"), py::arg("q_data"),
        py::arg("q_scales"), py::arg("q_global") = py::none(), py::arg("q_flags") = 0, py::arg("q_smax") = 0);
  m.def("qat_step", &qat_step, py::arg("grad"), py::arg("state"), py::arg("q_data"), py::arg("q_scales"),
        py::arg("q_global") = py::none(), py::arg("q_flags") = 0, py::arg("q_smax") = 0, py::arg("lr"),
        py::arg("beta1"), py::arg("beta2"), py::arg("eps"), py::arg("weight_decay"), py::arg("step"),
        py::arg("grad_scale"), py::arg("scale_z"), py::arg("seed"));
}
