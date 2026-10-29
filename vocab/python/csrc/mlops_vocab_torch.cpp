// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// PyTorch bindings of mlops-vocab (mlops.vocab._C): the variants and the calls of embed, the LM head and the sampler
// on tensors. The Python layer chooses the variant and owns the sampler's state and workspace.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <tuple>
#include <vector>
#include "mlops_vocab.h"

using at::Tensor;
using OptT = std::optional<Tensor>;

namespace {

void check_cuda(const Tensor &t, at::ScalarType dt, const char *what, const at::Device &dev) {
  TORCH_CHECK_VALUE(t.is_cuda() && t.device() == dev, what, " must be a CUDA tensor on ", dev, ", got ", t.device());
  TORCH_CHECK_TYPE(t.scalar_type() == dt, what, " must be ", dt, ", got ", t.scalar_type());
  TORCH_CHECK_VALUE(t.is_contiguous(), what, " must be contiguous");
}
cudaStream_t stream() { return at::cuda::getCurrentCUDAStream().stream(); }

// ---- embed ------------------------------------------------------------------------------------------------
using EmbedVariant = std::tuple<int, int, int, int, int>;  // warps, threads, smem, regs, arch
std::vector<EmbedVariant> embed_variants(int64_t vocab, int64_t hidden, int64_t n) {
  const mlops_embed_config c{(int)vocab, (int)hidden, (int)n, 0};
  std::vector<mlops_embed_variant> v(mlops_embed_variants(&c, nullptr, 0));
  mlops_embed_variants(&c, v.data(), (int)v.size());
  std::vector<EmbedVariant> out;
  for (const auto &x : v) out.emplace_back(x.warps, x.threads, x.smem_bytes, x.regs, x.arch);
  return out;
}

std::tuple<Tensor, OptT, Tensor, Tensor> embed(Tensor table, Tensor ids, OptT norm_w, int64_t variant, int64_t items) {
  const at::Device dev = ids.device();
  check_cuda(ids, at::kInt, "ids", dev);
  TORCH_CHECK_VALUE(table.dim() == 2 && table.is_contiguous() && table.scalar_type() == at::kBFloat16 && table.device() == dev,
                    "table must be a contiguous bfloat16 [vocab, hidden] tensor on ", dev);
  if (norm_w) check_cuda(*norm_w, at::kBFloat16, "norm_w", dev);
  const int64_t vocab = table.size(0), hidden = table.size(1), n = ids.numel();
  TORCH_CHECK_VALUE(!norm_w || norm_w->numel() == hidden, "norm_w must have hidden = ", hidden, " elements");
  const mlops_embed_config c{(int)vocab, (int)hidden, (int)n, (int)items};
  std::vector<mlops_embed_variant> v(mlops_embed_variants(&c, nullptr, 0));
  mlops_embed_variants(&c, v.data(), (int)v.size());
  TORCH_CHECK_VALUE(variant >= 0 && variant < (int64_t)v.size(), "variant ", variant, " of ", v.size());
  Tensor x = at::empty({n, hidden}, table.options().dtype(at::kFloat));
  OptT xw;
  if (norm_w) xw = at::empty({n, hidden}, table.options());
  Tensor parts = at::empty({n, hidden / 256}, table.options().dtype(at::kFloat));
  Tensor error = at::zeros({1}, table.options().dtype(at::kInt));
  const c10::cuda::CUDAGuard guard(dev);
  const mlops_embed_status st = mlops_embed(&c, &v[variant], (const uint16_t *)table.data_ptr(), ids.data_ptr<int>(),
                                            norm_w ? (const uint16_t *)norm_w->data_ptr() : nullptr, x.data_ptr<float>(),
                                            xw ? (uint16_t *)xw->data_ptr() : nullptr, parts.data_ptr<float>(), error.data_ptr<int>(), stream());
  if (st == MLOPS_EMBED_INVALID_VALUE) TORCH_CHECK_VALUE(false, mlops_embed_last_error());
  TORCH_CHECK(st == MLOPS_EMBED_OK, "mlops_embed failed: ", mlops_embed_last_error());
  return {x, xw, parts, error};
}

// ---- lm_head ----------------------------------------------------------------------------------------------
using HeadVariant = std::tuple<int, int, int, int, int, int, int, int, int, int>;  // R, S, D, NB, tc, J, threads, smem, regs, arch
std::vector<HeadVariant> lm_head_variants(int64_t vocab, int64_t hidden, int64_t n) {
  const mlops_lm_head_config c{(int)vocab, (int)hidden, (int)n, 0, 0, 0.f, 0.f, 1.f};
  std::vector<mlops_lm_head_variant> v(mlops_lm_head_variants(&c, nullptr, 0));
  mlops_lm_head_variants(&c, v.data(), (int)v.size());
  std::vector<HeadVariant> out;
  for (const auto &x : v)
    out.emplace_back(x.rows_per_unit, x.k_pieces, x.depth, x.columns, x.tensor_cores, x.chunks, x.threads, x.smem_bytes, x.regs, x.arch);
  return out;
}

std::tuple<OptT, OptT, OptT, OptT> lm_head(Tensor w, Tensor x, OptT norm_parts, double inv_n, double eps, double logit_scale, int64_t variant,
                                     int64_t items, bool want_logits, bool want_stats) {
  const at::Device dev = x.device();
  TORCH_CHECK_VALUE(w.dim() == 2 && w.is_contiguous() && w.scalar_type() == at::kBFloat16 && w.device() == dev,
                    "w must be a contiguous bfloat16 [vocab, hidden] tensor on ", dev);
  TORCH_CHECK_VALUE(x.dim() == 2 && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.size(1) == w.size(1),
                    "x must be a contiguous bfloat16 [n, hidden] tensor with w's hidden size");
  const int64_t vocab = w.size(0), hidden = w.size(1), n = x.size(0);
  int64_t np = 0;
  if (norm_parts) {
    check_cuda(*norm_parts, at::kFloat, "norm_parts", dev);
    TORCH_CHECK_VALUE(norm_parts->dim() == 2 && norm_parts->size(0) == n, "norm_parts must be [n, parts]");
    np = norm_parts->size(1);
  }
  const mlops_lm_head_config c{(int)vocab, (int)hidden, (int)n, (int)items, (int)np, (float)inv_n, (float)eps, (float)logit_scale};
  std::vector<mlops_lm_head_variant> v(mlops_lm_head_variants(&c, nullptr, 0));
  mlops_lm_head_variants(&c, v.data(), (int)v.size());
  TORCH_CHECK_VALUE(variant >= 0 && variant < (int64_t)v.size(), "variant ", variant, " of ", v.size());
  const int ni = mlops_lm_head_item_count(&c, &v[variant]);
  OptT logits, smax, sarg, shist;
  if (want_logits) logits = at::empty({n, vocab}, x.options().dtype(at::kFloat));
  if (want_stats) {
    smax = at::empty({n, ni}, x.options().dtype(at::kFloat));
    sarg = at::empty({n, ni}, x.options().dtype(at::kInt));
    shist = at::empty({n, ni, 256}, x.options().dtype(at::kInt));
  }
  const c10::cuda::CUDAGuard guard(dev);
  const mlops_lm_head_status st =
      mlops_lm_head(&c, &v[variant], (const uint16_t *)w.data_ptr(), (const uint16_t *)x.data_ptr(), norm_parts ? norm_parts->data_ptr<float>() : nullptr,
                    logits ? logits->data_ptr<float>() : nullptr, smax ? smax->data_ptr<float>() : nullptr, sarg ? sarg->data_ptr<int>() : nullptr,
                    shist ? (unsigned *)shist->data_ptr<int>() : nullptr, stream());
  if (st == MLOPS_LM_HEAD_INVALID_VALUE) TORCH_CHECK_VALUE(false, mlops_lm_head_last_error());
  TORCH_CHECK(st == MLOPS_LM_HEAD_OK, "mlops_lm_head failed: ", mlops_lm_head_last_error());
  return {logits, smax, sarg, shist};
}

// ---- sample -----------------------------------------------------------------------------------------------
int64_t sample_workspace_bytes(int64_t vocab) {
  const mlops_sample_config c{(int)vocab, 0.f, 0, 1.f, 1.f, 0};
  return (int64_t)mlops_sample_workspace_bytes(&c);
}

void sample(Tensor logits, OptT seen, OptT stats_max, OptT stats_arg, OptT stats_hist, Tensor state, Tensor workspace, Tensor token, double temperature,
            int64_t top_k, double top_p, double penalty, int64_t ctas) {
  const at::Device dev = logits.device();
  check_cuda(logits, at::kFloat, "logits", dev);
  check_cuda(state, at::kLong, "state", dev);
  check_cuda(workspace, at::kByte, "workspace", dev);
  check_cuda(token, at::kInt, "token", dev);
  TORCH_CHECK_VALUE(state.numel() == 2 && token.numel() == 1, "state is [seed, step] and token one int");
  if (seen) check_cuda(*seen, at::kInt, "seen", dev);
  const mlops_sample_config c{(int)logits.numel(), (float)temperature, (int)top_k, (float)top_p, (float)penalty, (int)ctas};
  mlops_sample_stats stats{};
  if (stats_max) {
    TORCH_CHECK_VALUE(stats_arg.has_value(), "stats_max needs stats_arg");
    check_cuda(*stats_max, at::kFloat, "stats_max", dev);
    check_cuda(*stats_arg, at::kInt, "stats_arg", dev);
    stats = {stats_max->data_ptr<float>(), stats_arg->data_ptr<int>(), (int)stats_max->numel(), nullptr};
    if (stats_hist) {
      check_cuda(*stats_hist, at::kInt, "stats_hist", dev);
      TORCH_CHECK_VALUE(stats_hist->numel() == 256 * stats_max->numel(), "stats_hist must be [count, 256]");
      stats.hist = (const unsigned *)stats_hist->data_ptr<int>();
    }
  }
  const c10::cuda::CUDAGuard guard(dev);
  const mlops_sample_status st =
      mlops_sample(&c, logits.data_ptr<float>(), seen ? (const uint32_t *)seen->data_ptr<int>() : nullptr, stats_max ? &stats : nullptr,
                   (mlops_sample_state *)state.data_ptr<int64_t>(), workspace.data_ptr(), (size_t)workspace.numel(), token.data_ptr<int>(), stream());
  if (st == MLOPS_SAMPLE_INVALID_VALUE) TORCH_CHECK_VALUE(false, mlops_sample_last_error());
  TORCH_CHECK(st == MLOPS_SAMPLE_OK, "mlops_sample failed: ", mlops_sample_last_error());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "mlops-vocab: embedding rows, the LM head and sampling on CUDA";
  m.attr("__version__") = MLOPS_VOCAB_VERSION;
  m.def("embed_variants", &embed_variants, py::arg("vocab"), py::arg("hidden"), py::arg("n"));
  m.def("embed", &embed, py::arg("table"), py::arg("ids"), py::arg("norm_w") = py::none(), py::arg("variant") = 0, py::arg("items") = 0);
  m.def("lm_head_variants", &lm_head_variants, py::arg("vocab"), py::arg("hidden"), py::arg("n"));
  m.def("lm_head", &lm_head, py::arg("w"), py::arg("x"), py::arg("norm_parts") = py::none(), py::arg("inv_n") = 0.0, py::arg("eps") = 0.0,
        py::arg("logit_scale") = 1.0, py::arg("variant") = 0, py::arg("items") = 0, py::arg("want_logits") = true, py::arg("want_stats") = false);
  m.def("sample_workspace_bytes", &sample_workspace_bytes, py::arg("vocab"));
  m.def("sample", &sample, py::arg("logits"), py::arg("seen") = py::none(), py::arg("stats_max") = py::none(), py::arg("stats_arg") = py::none(), py::arg("stats_hist") = py::none(),
        py::arg("state"), py::arg("workspace"), py::arg("token"), py::arg("temperature") = 0.0, py::arg("top_k") = 0, py::arg("top_p") = 1.0,
        py::arg("penalty") = 1.0, py::arg("ctas") = 0);
}
