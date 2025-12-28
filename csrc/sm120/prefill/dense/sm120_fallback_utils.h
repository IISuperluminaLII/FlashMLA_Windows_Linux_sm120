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

  // Convert to float for numerical stability
  at::Tensor q_float = q.contiguous().to(at::kFloat);
  at::Tensor k_float = k.contiguous().to(at::kFloat);
  at::Tensor v_float = v.contiguous().to(at::kFloat);
  at::Tensor d_o_float = d_o.contiguous().to(at::kFloat);
  at::Tensor lse_float = lse.contiguous().to(at::kFloat);

  const auto total_q = q_float.size(0);
  const auto heads_q = q_float.size(1);
  const auto heads_k = k_float.size(1);
  const auto head_dim = q_float.size(2);
  const auto head_dim_v = v_float.size(2);
  TORCH_CHECK(heads_k > 0, "SM120 backward fallback requires positive kv head dimension.");

  // For non-varlen case with uniform batch sizes, use efficient batched operations
  // This avoids the O(batches * heads) loop overhead
  if constexpr (!kIsVarlen) {
    const int seqlen_q = max_seqlen_q > 0 ? max_seqlen_q : static_cast<int>(total_q);
    const int seqlen_kv = max_seqlen_kv > 0 ? max_seqlen_kv : static_cast<int>(k_float.size(0));
    const int batches = static_cast<int>(total_q / seqlen_q);

    // Initialize output gradients
    at::Tensor dq_float = at::zeros_like(q_float);
    at::Tensor dk_float = at::zeros_like(k_float);
    at::Tensor dv_float = at::zeros_like(v_float);

    // Reshape from [total, heads, dim] to [batch, seqlen, heads, dim]
    at::Tensor q_4d = q_float.view({batches, seqlen_q, heads_q, head_dim});
    at::Tensor k_4d = k_float.view({batches, seqlen_kv, heads_k, head_dim});
    at::Tensor v_4d = v_float.view({batches, seqlen_kv, heads_k, head_dim_v});
    at::Tensor d_o_4d = d_o_float.view({batches, seqlen_q, heads_q, head_dim_v});
    at::Tensor lse_4d = lse_float.view({batches, seqlen_q, heads_q});
    at::Tensor dq_4d = dq_float.view({batches, seqlen_q, heads_q, head_dim});
    at::Tensor dk_4d = dk_float.view({batches, seqlen_kv, heads_k, head_dim});
    at::Tensor dv_4d = dv_float.view({batches, seqlen_kv, heads_k, head_dim_v});

    // Transpose to [batch, heads, seqlen, dim] for efficient bmm
    // q_t: [batch, heads_q, seqlen_q, head_dim]
    at::Tensor q_t = q_4d.permute({0, 2, 1, 3}).contiguous();
    at::Tensor k_t = k_4d.permute({0, 2, 1, 3}).contiguous();
    at::Tensor v_t = v_4d.permute({0, 2, 1, 3}).contiguous();
    at::Tensor d_o_t = d_o_4d.permute({0, 2, 1, 3}).contiguous();
    at::Tensor lse_t = lse_4d.permute({0, 2, 1}).contiguous();

    // For MHA (heads_q == heads_k), use batched operations directly
    // Merge batch and head dims for efficient bmm: [batch*heads, seqlen, dim]
    const int bh = batches * heads_q;
    at::Tensor q_bh = q_t.view({bh, seqlen_q, head_dim});
    at::Tensor d_o_bh = d_o_t.view({bh, seqlen_q, head_dim_v});
    at::Tensor lse_bh = lse_t.view({bh, seqlen_q});

    // Handle GQA/MQA: expand k/v to match heads_q
    at::Tensor k_bh, v_bh;
    if (heads_q == heads_k) {
      k_bh = k_t.view({bh, seqlen_kv, head_dim});
      v_bh = v_t.view({bh, seqlen_kv, head_dim_v});
    } else {
      // GQA: repeat k/v heads to match q heads
      int repeat = heads_q / heads_k;
      // k_t: [batch, heads_k, seqlen, dim] -> [batch, heads_q, seqlen, dim]
      at::Tensor k_expanded = k_t.unsqueeze(2).expand({batches, heads_k, repeat, seqlen_kv, head_dim})
                                  .reshape({batches, heads_q, seqlen_kv, head_dim});
      at::Tensor v_expanded = v_t.unsqueeze(2).expand({batches, heads_k, repeat, seqlen_kv, head_dim_v})
                                  .reshape({batches, heads_q, seqlen_kv, head_dim_v});
      k_bh = k_expanded.view({bh, seqlen_kv, head_dim}).contiguous();
      v_bh = v_expanded.view({bh, seqlen_kv, head_dim_v}).contiguous();
    }

    // Compute attention scores: [bh, seqlen_q, seqlen_kv]
    at::Tensor scores = at::bmm(q_bh, k_bh.transpose(1, 2));
    scores.mul_(scale_softmax);

    // Apply causal mask if needed
    if constexpr (kIsCausal) {
      at::Tensor row_idx = at::arange(0, seqlen_q, scores.options().dtype(at::kLong)).unsqueeze(1);
      at::Tensor col_idx = at::arange(0, seqlen_kv, scores.options().dtype(at::kLong)).unsqueeze(0);
      at::Tensor causal_mask = col_idx > row_idx;  // [seqlen_q, seqlen_kv]
      scores.masked_fill_(causal_mask.unsqueeze(0), -std::numeric_limits<float>::infinity());
    }

    // Compute softmax probs using saved LSE: [bh, seqlen_q, seqlen_kv]
    at::Tensor log_probs = scores - lse_bh.unsqueeze(2);
    at::Tensor probs = log_probs.exp();

    // dV = P^T @ dO: [bh, seqlen_kv, head_dim_v]
    at::Tensor dV_bh = at::bmm(probs.transpose(1, 2), d_o_bh);

    // dP = dO @ V^T: [bh, seqlen_q, seqlen_kv]
    at::Tensor dP = at::bmm(d_o_bh, v_bh.transpose(1, 2));

    // dScores = (dP - sum(dP * P, dim=-1, keepdim=True)) * P
    at::Tensor sum_dp = (dP * probs).sum(-1, true);
    at::Tensor dScores = (dP - sum_dp) * probs;

    // Apply causal mask to dScores
    if constexpr (kIsCausal) {
      at::Tensor row_idx = at::arange(0, seqlen_q, scores.options().dtype(at::kLong)).unsqueeze(1);
      at::Tensor col_idx = at::arange(0, seqlen_kv, scores.options().dtype(at::kLong)).unsqueeze(0);
      at::Tensor causal_mask = col_idx > row_idx;
      dScores.masked_fill_(causal_mask.unsqueeze(0), 0.0f);
    }

    dScores.mul_(scale_softmax);

    // dQ = dScores @ K: [bh, seqlen_q, head_dim]
    at::Tensor dQ_bh = at::bmm(dScores, k_bh);

    // dK = dScores^T @ Q: [bh, seqlen_kv, head_dim]
    at::Tensor dK_bh = at::bmm(dScores.transpose(1, 2), q_bh);

    // Reshape back: [bh, seqlen, dim] -> [batch, heads, seqlen, dim] -> [batch, seqlen, heads, dim]
    at::Tensor dQ_t = dQ_bh.view({batches, heads_q, seqlen_q, head_dim});
    dq_4d.copy_(dQ_t.permute({0, 2, 1, 3}));

    // For dK and dV, handle GQA reduction
    if (heads_q == heads_k) {
      at::Tensor dK_t = dK_bh.view({batches, heads_k, seqlen_kv, head_dim});
      at::Tensor dV_t = dV_bh.view({batches, heads_k, seqlen_kv, head_dim_v});
      dk_4d.copy_(dK_t.permute({0, 2, 1, 3}));
      dv_4d.copy_(dV_t.permute({0, 2, 1, 3}));
    } else {
      // GQA: sum gradients across repeated heads
      int repeat = heads_q / heads_k;
      at::Tensor dK_expanded = dK_bh.view({batches, heads_k, repeat, seqlen_kv, head_dim});
      at::Tensor dV_expanded = dV_bh.view({batches, heads_k, repeat, seqlen_kv, head_dim_v});
      at::Tensor dK_summed = dK_expanded.sum(2);  // [batch, heads_k, seqlen_kv, head_dim]
      at::Tensor dV_summed = dV_expanded.sum(2);  // [batch, heads_k, seqlen_kv, head_dim_v]
      dk_4d.copy_(dK_summed.permute({0, 2, 1, 3}));
      dv_4d.copy_(dV_summed.permute({0, 2, 1, 3}));
    }

    // Copy results back
    dq.copy_(dq_float.to(dq.scalar_type()));
    dk.copy_(dk_float.to(dk.scalar_type()));
    dv.copy_(dv_float.to(dv.scalar_type()));
    return;
  }

  // Varlen path: must iterate over variable-length sequences
  // But still use batched head operations within each sequence
  at::Tensor dq_float = dq.to(at::kFloat).clone();
  at::Tensor dk_float = dk.to(at::kFloat).clone();
  at::Tensor dv_float = dv.to(at::kFloat).clone();
  dq_float.zero_();
  dk_float.zero_();
  dv_float.zero_();

  std::vector<int> cum_q, cum_kv;
  int batches = 1;
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

  for (int b = 0; b < batches; ++b) {
    const int q_start = cum_q[b];
    const int q_end = cum_q[b + 1];
    const int kv_start = cum_kv[b];
    const int kv_end = cum_kv[b + 1];

    const int rows_q = q_end - q_start;
    const int rows_kv = kv_end - kv_start;
    if (rows_q == 0 || rows_kv == 0) {
      continue;
    }

    // Slice batch data: [seqlen, heads, dim]
    at::Tensor q_batch = q_float.slice(0, q_start, q_end);
    at::Tensor k_batch = k_float.slice(0, kv_start, kv_end);
    at::Tensor v_batch = v_float.slice(0, kv_start, kv_end);
    at::Tensor d_o_batch = d_o_float.slice(0, q_start, q_end);
    at::Tensor lse_batch = lse_float.slice(0, q_start, q_end);
    at::Tensor dq_batch = dq_float.slice(0, q_start, q_end);
    at::Tensor dk_batch = dk_float.slice(0, kv_start, kv_end);
    at::Tensor dv_batch = dv_float.slice(0, kv_start, kv_end);

    // Transpose to [heads, seqlen, dim] for batched operations
    at::Tensor q_ht = q_batch.permute({1, 0, 2}).contiguous();
    at::Tensor d_o_ht = d_o_batch.permute({1, 0, 2}).contiguous();
    at::Tensor lse_ht = lse_batch.permute({1, 0}).contiguous();

    // Handle GQA: expand k/v if needed
    at::Tensor k_ht, v_ht;
    if (heads_q == heads_k) {
      k_ht = k_batch.permute({1, 0, 2}).contiguous();
      v_ht = v_batch.permute({1, 0, 2}).contiguous();
    } else {
      int repeat = heads_q / heads_k;
      at::Tensor k_t = k_batch.permute({1, 0, 2});  // [heads_k, seqlen, dim]
      at::Tensor v_t = v_batch.permute({1, 0, 2});
      k_ht = k_t.unsqueeze(1).expand({heads_k, repeat, rows_kv, head_dim})
                .reshape({heads_q, rows_kv, head_dim}).contiguous();
      v_ht = v_t.unsqueeze(1).expand({heads_k, repeat, rows_kv, head_dim_v})
                .reshape({heads_q, rows_kv, head_dim_v}).contiguous();
    }

    // Batched attention backward over all heads at once
    // scores: [heads_q, rows_q, rows_kv]
    at::Tensor scores = at::bmm(q_ht, k_ht.transpose(1, 2));
    scores.mul_(scale_softmax);

    if constexpr (kIsCausal) {
      at::Tensor row_idx = at::arange(0, rows_q, scores.options().dtype(at::kLong)).unsqueeze(1);
      at::Tensor col_idx = at::arange(0, rows_kv, scores.options().dtype(at::kLong)).unsqueeze(0);
      at::Tensor causal_mask = col_idx > row_idx;
      scores.masked_fill_(causal_mask.unsqueeze(0), -std::numeric_limits<float>::infinity());
    }

    at::Tensor log_probs = scores - lse_ht.unsqueeze(2);
    at::Tensor probs = log_probs.exp();

    at::Tensor dV_ht = at::bmm(probs.transpose(1, 2), d_o_ht);
    at::Tensor dP = at::bmm(d_o_ht, v_ht.transpose(1, 2));
    at::Tensor sum_dp = (dP * probs).sum(-1, true);
    at::Tensor dScores = (dP - sum_dp) * probs;

    if constexpr (kIsCausal) {
      at::Tensor row_idx = at::arange(0, rows_q, scores.options().dtype(at::kLong)).unsqueeze(1);
      at::Tensor col_idx = at::arange(0, rows_kv, scores.options().dtype(at::kLong)).unsqueeze(0);
      at::Tensor causal_mask = col_idx > row_idx;
      dScores.masked_fill_(causal_mask.unsqueeze(0), 0.0f);
    }

    dScores.mul_(scale_softmax);

    at::Tensor dQ_ht = at::bmm(dScores, k_ht);
    at::Tensor dK_ht = at::bmm(dScores.transpose(1, 2), q_ht);

    // Transpose back and accumulate: [heads, seqlen, dim] -> [seqlen, heads, dim]
    dq_batch.add_(dQ_ht.permute({1, 0, 2}));

    if (heads_q == heads_k) {
      dk_batch.add_(dK_ht.permute({1, 0, 2}));
      dv_batch.add_(dV_ht.permute({1, 0, 2}));
    } else {
      // GQA: reduce over repeated heads
      int repeat = heads_q / heads_k;
      at::Tensor dK_expanded = dK_ht.view({heads_k, repeat, rows_kv, head_dim});
      at::Tensor dV_expanded = dV_ht.view({heads_k, repeat, rows_kv, head_dim_v});
      at::Tensor dK_summed = dK_expanded.sum(1).permute({1, 0, 2});
      at::Tensor dV_summed = dV_expanded.sum(1).permute({1, 0, 2});
      dk_batch.add_(dK_summed);
      dv_batch.add_(dV_summed);
    }
  }

  dq.copy_(dq_float.to(dq.scalar_type()));
  dk.copy_(dk_float.to(dk.scalar_type()));
  dv.copy_(dv_float.to(dv.scalar_type()));
}

}  // namespace detail
}  // namespace flash

#endif  // FLASH_MLA_SKIP_FALLBACK
