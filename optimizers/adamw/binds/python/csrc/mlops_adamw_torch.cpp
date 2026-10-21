// PyTorch bindings of mlops_adamw / mlops_adamw_qat (../../include). The NVFP4 tensor is taken the
// way NVFP4EffMA's own bindings take operands: flat (data, scales, global, flags, scale_max), which
// the Python layer unpacks from an nvfp4effma.NVFP4Tensor; the QAT state is one uint8 tensor.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <string>
#include "mlops_adamw.h"
#include "mlops_adamw_qat.h"

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

nvfp4a_quant_config make_qcfg(int64_t tile_rows, int64_t four_over_six, bool compute_amax) {
  nvfp4a_quant_config c{};
  c.tile_rows = (int)tile_rows;
  c.four_over_six = (nvfp4a_4over6)four_over_six;
  c.compute_amax = compute_amax ? 1 : 0;
  return c;
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

void qat_init(Tensor w, Tensor state, Tensor q_data, Tensor q_scales, OptT q_global, int64_t q_flags, int64_t q_smax,
              int64_t four_over_six) {
  const auto dev = w.device();
  TORCH_CHECK_VALUE(w.dim() == 2 && w.is_contiguous(), "w must be a contiguous 2-D tensor");
  const int64_t rows = w.size(0), cols = w.size(1);
  const mlops_adamw_qat_state st = bind_state(state, rows, cols, dev);
  const nvfp4a_tensor q = as_nvfp4(q_data, q_scales, q_global, q_flags, q_smax, dev, "q");
  nvfp4a_quant_config qc = make_qcfg(1, four_over_six, (q_flags & NVFP4A_GLOBAL_AMAX) != 0);
  const c10::cuda::CUDAGuard guard(dev);
  check_status(mlops_adamw_qat_init((int)rows, (int)cols, w.data_ptr(), dense_dtype(w.scalar_type(), "w"), &q, &qc, &st,
                                    at::cuda::getCurrentCUDAStream().stream()),
               "mlops_adamw_qat_init");
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

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "mlops_adamw: AdamW / QAT-AdamW CUDA kernels (NVFP4 QAT via NVFP4EffMA)";
  m.attr("__version__") = MLOPS_ADAMW_VERSION;
  m.def("adamw", &adamw, py::arg("param"), py::arg("grad"), py::arg("exp_avg"), py::arg("exp_avg_sq"), py::arg("lr"),
        py::arg("beta1"), py::arg("beta2"), py::arg("eps"), py::arg("weight_decay"), py::arg("step"),
        py::arg("bias_correction"), py::arg("grad_scale"));
  m.def("qat_state_bytes", &qat_state_bytes, py::arg("rows"), py::arg("cols"));
  m.def("qat_init", &qat_init, py::arg("w"), py::arg("state"), py::arg("q_data"), py::arg("q_scales"),
        py::arg("q_global") = py::none(), py::arg("q_flags") = 0, py::arg("q_smax") = 0, py::arg("four_over_six") = 0);
  m.def("qat_step", &qat_step, py::arg("grad"), py::arg("state"), py::arg("q_data"), py::arg("q_scales"),
        py::arg("q_global") = py::none(), py::arg("q_flags") = 0, py::arg("q_smax") = 0, py::arg("lr"),
        py::arg("beta1"), py::arg("beta2"), py::arg("eps"), py::arg("weight_decay"), py::arg("step"),
        py::arg("grad_scale"), py::arg("scale_z"), py::arg("seed"));
}
