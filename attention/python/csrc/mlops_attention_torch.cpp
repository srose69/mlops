// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "mlops_attention.h"

using at::Tensor;

namespace {
void tensor(const Tensor &x, const at::Device &dev, at::ScalarType dt, const char *name) {
  TORCH_CHECK_VALUE(x.is_cuda() && x.device() == dev, name, " must be a CUDA tensor on ", dev);
  TORCH_CHECK_TYPE(x.scalar_type() == dt, name, " has dtype ", x.scalar_type(), ", expected ", dt);
  TORCH_CHECK_VALUE(x.is_contiguous(), name, " must be contiguous");
}

mlops_fbla_config config(int64_t batch, int64_t q_heads, int64_t kv_heads, int64_t tokens, int64_t format,
                         int64_t capacity, double scale, double tau) {
  TORCH_CHECK_VALUE(format == MLOPS_FBLA_3P3 || format == MLOPS_FBLA_4P2, "format must be 0 (3+3) or 1 (4+2)");
  return mlops_fbla_config{(int)batch, (int)q_heads, (int)kv_heads, (int)tokens, (float)scale, (float)tau,
                           (mlops_fbla_format)format, (int)capacity};
}

int64_t scratch_bytes(int64_t batch, int64_t q_heads, int64_t kv_heads, int64_t tokens, int64_t format, int64_t capacity) {
  const mlops_fbla_config cfg = config(batch, q_heads, kv_heads, tokens, format, capacity, 0.0, 0.0);
  const size_t n = mlops_fbla_decode_scratch_bytes(&cfg);
  TORCH_CHECK_VALUE(n > 0, "invalid FBLA configuration: batch ", batch, ", heads ", q_heads, " / ", kv_heads,
                    ", tokens ", tokens, ", capacity ", capacity);
  return (int64_t)n;
}

Tensor decode(Tensor q, Tensor coarse, Tensor fine, Tensor meta, Tensor sk, Tensor sv, Tensor ok, Tensor ov,
              Tensor scratch, int64_t tokens, int64_t capacity, int64_t kv_heads, int64_t format, double scale,
              double tau) {
  TORCH_CHECK_VALUE(q.dim() == 3 && q.size(2) == MLOPS_FBLA_HEAD_DIM, "q must be [batch, q_heads, 128], got ", q.sizes());
  const at::Device dev = q.device();
  tensor(q, dev, at::kBFloat16, "q");
  for (const auto &x : {coarse, fine}) tensor(x, dev, at::kByte, "packed cache records");
  for (const auto &x : {meta, sk, sv, ok, ov}) tensor(x, dev, at::kBFloat16, "cache metadata / exact group");
  tensor(scratch, dev, at::kByte, "scratch");
  const mlops_fbla_config cfg = config(q.size(0), q.size(1), kv_heads, tokens, format, capacity, scale, tau);
  mlops_fbla_cache cache{coarse.data_ptr<uint8_t>(), fine.data_ptr<uint8_t>(), (const uint16_t *)meta.data_ptr(),
                         (const uint16_t *)sk.data_ptr(), (const uint16_t *)sv.data_ptr(),
                         (const uint16_t *)ok.data_ptr(), (const uint16_t *)ov.data_ptr()};
  Tensor out = at::empty_like(q);
  const c10::cuda::CUDAGuard guard(dev);
  const mlops_fbla_status st = mlops_fbla_decode(&cfg, (const uint16_t *)q.data_ptr(), &cache, (uint16_t *)out.data_ptr(),
                                                 scratch.data_ptr(), scratch.numel(), at::cuda::getCurrentCUDAStream().stream());
  if (st == MLOPS_FBLA_INVALID_VALUE) TORCH_CHECK_VALUE(false, mlops_fbla_last_error());
  TORCH_CHECK(st == MLOPS_FBLA_OK, "mlops_fbla_decode failed: ", mlops_fbla_last_error());
  return out;
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "mlops-attention: FBLA FlashAttention decode";
  m.attr("__version__") = MLOPS_ATTENTION_VERSION;
  m.def("scratch_bytes", &scratch_bytes, py::arg("batch"), py::arg("q_heads"), py::arg("kv_heads"),
        py::arg("tokens"), py::arg("format"), py::arg("capacity"));
  m.def("decode", &decode, py::arg("q"), py::arg("coarse"), py::arg("fine"), py::arg("meta"), py::arg("sink_k"),
        py::arg("sink_v"), py::arg("open_k"), py::arg("open_v"), py::arg("scratch"), py::arg("tokens"),
        py::arg("capacity"), py::arg("kv_heads"), py::arg("format"), py::arg("scale") = 0.0, py::arg("tau") = 0.0);
}
