#pragma once

#ifdef FLASH_MLA_SKIP_FALLBACK

#include <stdexcept>

namespace flash {
namespace detail {

template <bool kIsVarlen, bool kIsMla, class Mask, class... Args>
void run_fmha_fwd_sm120_fallback(Args&&...) {
  throw std::runtime_error("SM120 fallback disabled in probe build.");
}

template <bool kIsVarlen, bool kIsMla, class Mask, class... Args>
void run_fmha_bwd_sm120_fallback(Args&&...) {
  throw std::runtime_error("SM120 fallback disabled in probe build.");
}

}  // namespace detail
}  // namespace flash

#else

#include <limits>
#include <vector>

#include <ATen/Functions.h>
#include <ATen/Operators.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "collective/fmha_fusion.hpp"
#include "common/mask.cuh"

namespace flash {
namespace detail {

template <bool kIsVarlen, bool kIsMla, class Mask>
void run_fmha_fwd_sm120_fallback(const c10::cuda::CUDAStream& stream,
                                 const c10::ScalarType dtype_in,
                                 const c10::ScalarType dtype_out,
                                 at::Tensor q,
                                 at::Tensor k,
                                 at::Tensor v,
                                 at::Tensor o,
                                 at::Tensor lse,
                                 float scale_softmax,
                                 at::Tensor cumulative_seqlen_q,
                                 at::Tensor cumulative_seqlen_kv,
                                 int max_seqlen_q,
                                 int max_seqlen_kv) {
  constexpr bool kIsCausal =
      std::is_same_v<Mask, cutlass::fmha::collective::CausalMask<false>> ||
      std::is_same_v<Mask, cutlass::fmha::collective::CausalMask<true>>;
  TORCH_CHECK(dtype_in == at::kBFloat16,
              "SM120 fallback currently supports bfloat16 inputs only.");

  c10::cuda::OptionalCUDAGuard device_guard(q.device());
  c10::cuda::CUDAStreamGuard stream_guard(stream);

  at::Tensor q_float = q.contiguous().to(at::kFloat);
  at::Tensor k_float = k.contiguous().to(at::kFloat);
  at::Tensor v_float = v.contiguous().to(at::kFloat);

  at::Tensor o_float = o.to(at::kFloat).clone();
  at::Tensor lse_float = lse.to(at::kFloat).clone();
  o_float.zero_();
  lse_float.zero_();

  const auto total_q = q_float.size(0);
  const auto heads_q = q_float.size(1);
  const auto heads_k = k_float.size(1);
  TORCH_CHECK(heads_k > 0, "SM120 fallback requires positive kv head dimension.");
  if (o_float.dim() == 2 || lse_float.dim() == 1) {
    TORCH_CHECK(heads_q == 1,
                "SM120 fallback expects a single head when output tensors are rank-2.");
  }

  auto make_lengths = [](at::Tensor cumulative) -> std::vector<int> {
    at::Tensor cpu_tensor = cumulative.device().is_cuda() ? cumulative.to(at::kCPU) : cumulative;
    const auto* data = cpu_tensor.data_ptr<int>();
    std::vector<int> result(cpu_tensor.numel());
    for (int i = 0; i < cpu_tensor.numel(); ++i) {
      result[i] = data[i];
    }
    return result;
  };

  std::vector<int> cum_q, cum_kv;
  int batches = 1;
  if constexpr (kIsVarlen) {
    cum_q = make_lengths(cumulative_seqlen_q);
    cum_kv = make_lengths(cumulative_seqlen_kv);
    batches = static_cast<int>(cum_q.size()) - 1;
  } else {
    batches = (max_seqlen_q > 0)
                  ? static_cast<int>(total_q / max_seqlen_q)
                  : 1;
  }

  auto start_q = [&](int b) -> int {
    if constexpr (kIsVarlen) {
      return cum_q[b];
    } else {
      const int rows_per_batch = (max_seqlen_q > 0) ? max_seqlen_q : (total_q / batches);
      return b * rows_per_batch;
    }
  };
  auto end_q = [&](int b) -> int {
    if constexpr (kIsVarlen) {
      return cum_q[b + 1];
    } else {
      const int rows_per_batch = (max_seqlen_q > 0) ? max_seqlen_q : (total_q / batches);
      return (b + 1) * rows_per_batch;
    }
  };
  auto start_k = [&](int b) -> int {
    if constexpr (kIsVarlen) {
      return cum_kv[b];
    } else {
      const int rows_per_batch =
          (max_seqlen_kv > 0) ? max_seqlen_kv : (k_float.size(0) / batches);
      return b * rows_per_batch;
    }
  };
  auto end_k = [&](int b) -> int {
    if constexpr (kIsVarlen) {
      return cum_kv[b + 1];
    } else {
      const int rows_per_batch =
          (max_seqlen_kv > 0) ? max_seqlen_kv : (k_float.size(0) / batches);
      return (b + 1) * rows_per_batch;
    }
  };

  at::Scalar negate_inf = -std::numeric_limits<float>::infinity();

  for (int b = 0; b < batches; ++b) {
    const int q_start = start_q(b);
    const int q_end = end_q(b);
    const int kv_start = start_k(b);
    const int kv_end = end_k(b);

    const int rows_q = q_end - q_start;
    const int rows_kv = kv_end - kv_start;

    if (rows_q == 0 || rows_kv == 0) {
      continue;
    }

    at::Tensor q_batch = q_float.slice(0, q_start, q_end);
    at::Tensor k_batch = k_float.slice(0, kv_start, kv_end);
    at::Tensor v_batch = v_float.slice(0, kv_start, kv_end);
    at::Tensor o_batch = o_float.slice(0, q_start, q_end);
    at::Tensor lse_batch = lse_float.slice(0, q_start, q_end);

    for (int h = 0; h < heads_q; ++h) {
      const int k_head_index = h % heads_k;
      at::Tensor q_head = q_batch.select(1, h);
      at::Tensor k_head = k_batch.select(1, k_head_index);
      at::Tensor v_head = v_batch.select(1, k_head_index);

      at::Tensor scores = at::matmul(q_head, k_head.transpose(0, 1));
      scores.mul_(scale_softmax);

      if constexpr (kIsCausal) {
        at::Tensor row_idx =
            at::arange(0, rows_q, scores.options().dtype(at::kLong)).unsqueeze(1);
        at::Tensor col_idx =
            at::arange(0, rows_kv, scores.options().dtype(at::kLong)).unsqueeze(0);
        at::Tensor causal_mask = col_idx > row_idx;
        scores.masked_fill_(causal_mask, negate_inf);
      }

      at::Tensor lse_head = at::logsumexp(scores, /*dim=*/1, /*keepdim=*/true);
      at::Tensor log_probs = scores - lse_head;
      at::Tensor probs = log_probs.exp();

      at::Tensor out = at::matmul(probs, v_head);

      if (o_batch.dim() == 3) {
        o_batch.select(1, h).copy_(out);
      } else if (o_batch.dim() == 2) {
        o_batch.copy_(out);
      } else {
        TORCH_CHECK(false, "SM120 fallback encountered unsupported output tensor rank.");
      }

      at::Tensor lse_target = lse_head.squeeze(1);
      if (lse_batch.dim() == 2) {
        lse_batch.select(1, h).copy_(lse_target);
      } else if (lse_batch.dim() == 1) {
        lse_batch.copy_(lse_target);
      } else {
        TORCH_CHECK(false, "SM120 fallback encountered unsupported LSE tensor rank.");
      }
    }
  }

  o.copy_(o_float.to(dtype_out));
  lse.copy_(lse_float.to(lse.scalar_type()));
}

template <bool kIsVarlen, bool kIsMla, class Mask>
void run_fmha_bwd_sm120_fallback(const c10::cuda::CUDAStream& stream,
                                 at::Tensor d_o,
                                 at::Tensor q,
                                 at::Tensor k,
                                 at::Tensor v,
                                 at::Tensor o,
                                 at::Tensor lse,
                                 at::Tensor dq,
                                 at::Tensor dk,
                                 at::Tensor dv,
                                 at::Tensor cumulative_seqlen_q,
                                 at::Tensor cumulative_seqlen_kv,
                                 float scale_softmax,
                                 int max_seqlen_q,
                                 int max_seqlen_kv) {
  constexpr bool kIsCausal =
      std::is_same_v<Mask, cutlass::fmha::collective::CausalMask<false>> ||
      std::is_same_v<Mask, cutlass::fmha::collective::CausalMask<true>>;

  c10::cuda::OptionalCUDAGuard device_guard(q.device());
  c10::cuda::CUDAStreamGuard stream_guard(stream);

  at::Tensor q_float = q.contiguous().to(at::kFloat);
  at::Tensor k_float = k.contiguous().to(at::kFloat);
  at::Tensor v_float = v.contiguous().to(at::kFloat);
  at::Tensor d_o_float = d_o.contiguous().to(at::kFloat);
  at::Tensor lse_float = lse.contiguous().to(at::kFloat);

  at::Tensor dq_float = dq.to(at::kFloat).clone();
  at::Tensor dk_float = dk.to(at::kFloat).clone();
  at::Tensor dv_float = dv.to(at::kFloat).clone();
  dq_float.zero_();
  dk_float.zero_();
  dv_float.zero_();

  const auto total_q = q_float.size(0);
  const auto heads_q = q_float.size(1);
  const auto heads_k = k_float.size(1);
  TORCH_CHECK(heads_k > 0, "SM120 backward fallback requires positive kv head dimension.");

  std::vector<int> cum_q, cum_kv;
  int batches = 1;
  if constexpr (kIsVarlen) {
    auto make_lengths = [](at::Tensor cumulative) -> std::vector<int> {
      at::Tensor cpu_tensor =
          cumulative.device().is_cuda() ? cumulative.to(at::kCPU) : cumulative;
      const auto* data = cpu_tensor.data_ptr<int>();
      std::vector<int> result(cpu_tensor.numel());
      for (int i = 0; i < cpu_tensor.numel(); ++i) {
        result[i] = data[i];
      }
      return result;
    };
    cum_q = make_lengths(cumulative_seqlen_q);
    cum_kv = make_lengths(cumulative_seqlen_kv);
    batches = static_cast<int>(cum_q.size()) - 1;
  } else {
    batches = (max_seqlen_q > 0)
                  ? static_cast<int>(total_q / max_seqlen_q)
                  : 1;
  }

  auto start_q = [&](int b) -> int {
    if constexpr (kIsVarlen) {
      return cum_q[b];
    } else {
      const int rows_per_batch = (max_seqlen_q > 0) ? max_seqlen_q : (total_q / batches);
      return b * rows_per_batch;
    }
  };
  auto end_q = [&](int b) -> int {
    if constexpr (kIsVarlen) {
      return cum_q[b + 1];
    } else {
      const int rows_per_batch = (max_seqlen_q > 0) ? max_seqlen_q : (total_q / batches);
      return (b + 1) * rows_per_batch;
    }
  };
  auto start_k = [&](int b) -> int {
    if constexpr (kIsVarlen) {
      return cum_kv[b];
    } else {
      const int rows_per_batch =
          (max_seqlen_kv > 0) ? max_seqlen_kv : (k_float.size(0) / batches);
      return b * rows_per_batch;
    }
  };
  auto end_k = [&](int b) -> int {
    if constexpr (kIsVarlen) {
      return cum_kv[b + 1];
    } else {
      const int rows_per_batch =
          (max_seqlen_kv > 0) ? max_seqlen_kv : (k_float.size(0) / batches);
      return (b + 1) * rows_per_batch;
    }
  };

  for (int b = 0; b < batches; ++b) {
    const int q_start = start_q(b);
    const int q_end = end_q(b);
    const int kv_start = start_k(b);
    const int kv_end = end_k(b);

    const int rows_q = q_end - q_start;
    const int rows_kv = kv_end - kv_start;
    if (rows_q == 0 || rows_kv == 0) {
      continue;
    }

    at::Tensor q_batch = q_float.slice(0, q_start, q_end);
    at::Tensor k_batch = k_float.slice(0, kv_start, kv_end);
    at::Tensor v_batch = v_float.slice(0, kv_start, kv_end);
    at::Tensor d_o_batch = d_o_float.slice(0, q_start, q_end);
    at::Tensor lse_batch = lse_float.slice(0, q_start, q_end);
    at::Tensor dq_batch = dq_float.slice(0, q_start, q_end);
    at::Tensor dk_batch = dk_float.slice(0, kv_start, kv_end);
    at::Tensor dv_batch = dv_float.slice(0, kv_start, kv_end);

    for (int h = 0; h < heads_q; ++h) {
      const int k_head_index = h % heads_k;
      at::Tensor q_head = q_batch.select(1, h);
      at::Tensor k_head = k_batch.select(1, k_head_index);
      at::Tensor v_head = v_batch.select(1, k_head_index);
      at::Tensor d_o_head = d_o_batch.select(1, h);

      at::Tensor scores = at::matmul(q_head, k_head.transpose(0, 1));
      scores.mul_(scale_softmax);

      if constexpr (kIsCausal) {
        at::Tensor row_idx =
            at::arange(0, rows_q, scores.options().dtype(at::kLong)).unsqueeze(1);
        at::Tensor col_idx =
            at::arange(0, rows_kv, scores.options().dtype(at::kLong)).unsqueeze(0);
        at::Tensor causal_mask = col_idx > row_idx;
        scores.masked_fill_(causal_mask,
                            -std::numeric_limits<float>::infinity());
      }

      at::Tensor lse_head = (lse_batch.dim() == 2)
                                ? lse_batch.select(1, h)
                                : lse_batch;
      at::Tensor log_probs = scores - lse_head.unsqueeze(1);
      at::Tensor probs = log_probs.exp();

      at::Tensor dV_head = at::matmul(probs.transpose(0, 1), d_o_head);
      at::Tensor dP = at::matmul(d_o_head, v_head.transpose(0, 1));
      at::Tensor sum_dp = (dP * probs).sum(-1, true);
      at::Tensor dScores = (dP - sum_dp) * probs;

      if constexpr (kIsCausal) {
        at::Tensor row_idx =
            at::arange(0, rows_q, scores.options().dtype(at::kLong)).unsqueeze(1);
        at::Tensor col_idx =
            at::arange(0, rows_kv, scores.options().dtype(at::kLong)).unsqueeze(0);
        at::Tensor causal_mask = col_idx > row_idx;
        dScores.masked_fill_(causal_mask, 0.0f);
      }

      dScores.mul_(scale_softmax);

      at::Tensor dQ_head = at::matmul(dScores, k_head);
      at::Tensor dK_head = at::matmul(dScores.transpose(0, 1), q_head);

      dq_batch.select(1, h).add_(dQ_head);
      dk_batch.select(1, k_head_index).add_(dK_head);
      dv_batch.select(1, k_head_index).add_(dV_head);
    }
  }

  dq.copy_(dq_float.to(dq.scalar_type()));
  dk.copy_(dk_float.to(dk.scalar_type()));
  dv.copy_(dv_float.to(dv.scalar_type()));
}

}  // namespace detail
}  // namespace flash

#endif  // FLASH_MLA_SKIP_FALLBACK
