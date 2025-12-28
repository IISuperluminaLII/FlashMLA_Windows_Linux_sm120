#include "interface.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include "sm120/prefill/dense/common/mask.cuh"
#include "sm120/prefill/dense/common/utils.hpp"

#include "sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "sm120/prefill/dense/fmha_cutlass_bwd_sm120.cuh"
#include "sm120/prefill/dense/fmha_bwd_kernel_sm120.cuh"

template<class Mask, class Varlen, class Element, class ElementOut, class Mla>
void call_run_fmha_bwd([[maybe_unused]] Mask mask, [[maybe_unused]] Varlen is_varlen,
                      [[maybe_unused]] Element in, [[maybe_unused]] ElementOut out, [[maybe_unused]] Mla mla,
                  at::Tensor workspace_buffer, at::Tensor d_o, at::Tensor q, at::Tensor k,
                  at::Tensor v, at::Tensor o, at::Tensor lse,
                  at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                  at::Tensor dq, at::Tensor dk, at::Tensor dv,
  float softmax_scale, int max_seqlen_q, int total_seqlen_kv) {
  static constexpr bool IsVarlen = std::is_same_v<Varlen, true_type>;
  static constexpr bool IsMla = std::is_same_v<Mla, true_type>;
  static constexpr bool IsCausal =
      std::is_same_v<Mask, CausalForBackwardMask<false>> ||
      std::is_same_v<Mask, CausalForBackwardMask<true>>;

  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  const int sm_version = prop.major * 10 + prop.minor;

  TORCH_CHECK(
      sm_version >= 120 && sm_version < 130,
      "flash_mla_sm120 build only supports SM120-class GPUs. Detected sm_",
      prop.major,
      prop.minor,
      ". Please install the SM100 build for server parts.");

  const auto stream = c10::cuda::getCurrentCUDAStream();

  // Check head dimensions for WMMA kernel compatibility
  const int head_dim = q.size(-1);
  const int num_heads = q.size(1);
  const int batch_size = cumulative_seqlen_q.size(0) - 1;

  // WMMA kernel supports both varlen and non-varlen modes
  // Requirements: head_dim == 128, bf16, non-MLA
  bool can_use_wmma = !IsMla &&
                      head_dim == 128 &&
                      max_seqlen_q > 0 &&
                      total_seqlen_kv > 0 &&
                      q.scalar_type() == at::ScalarType::BFloat16;

  if (can_use_wmma) {
    // Create float32 tensors for atomic accumulation
    // K-major kernel: dQ needs atomics (float), dK/dV don't (accumulated in smem)
    auto dq_float = at::zeros_like(dq, dq.options().dtype(at::kFloat));
    auto dk_float = at::zeros_like(dk, dk.options().dtype(at::kFloat));
    auto dv_float = at::zeros_like(dv, dv.options().dtype(at::kFloat));

    // Compute max_seqlen_kv for grid sizing
    int max_seqlen_kv = 0;
    if (IsVarlen) {
      // For varlen, find max from cu_seqlens
      auto cu_kv_cpu = cumulative_seqlen_kv.to(at::kCPU);
      auto cu_kv_data = cu_kv_cpu.data_ptr<int>();
      for (int b = 0; b < batch_size; ++b) {
        int seq_len = cu_kv_data[b + 1] - cu_kv_data[b];
        if (seq_len > max_seqlen_kv) max_seqlen_kv = seq_len;
      }
    } else {
      // For non-varlen, uniform sequence length
      max_seqlen_kv = total_seqlen_kv / batch_size;
    }

    // DUAL KERNEL ARCHITECTURE for optimal performance:
    // 1. Q-major dQ kernel: Grid over Q-blocks, NO ATOMICS for dQ
    // 2. K-major dK/dV kernel: Grid over K-blocks, NO ATOMICS for dK/dV
    // Both kernels use cp.async double-buffered pipelining
    if (IsCausal) {
      flash::detail::launch_fmha_bwd_sm120_dual<true>(
          stream,
          d_o,
          q,
          k,
          v,
          o,
          lse,
          cumulative_seqlen_q,
          cumulative_seqlen_kv,
          dq_float,
          dk_float,
          dv_float,
          softmax_scale,
          max_seqlen_q,
          max_seqlen_kv);
    } else {
      flash::detail::launch_fmha_bwd_sm120_dual<false>(
          stream,
          d_o,
          q,
          k,
          v,
          o,
          lse,
          cumulative_seqlen_q,
          cumulative_seqlen_kv,
          dq_float,
          dk_float,
          dv_float,
          softmax_scale,
          max_seqlen_q,
          max_seqlen_kv);
    }

    // Convert float outputs back to bf16
    dq.copy_(dq_float.to(dq.scalar_type()));
    dk.copy_(dk_float.to(dk.scalar_type()));
    dv.copy_(dv_float.to(dv.scalar_type()));
    return;
  }

  // Fall back to batched ATen implementation for MLA or other unsupported cases
  flash::detail::run_fmha_bwd_sm120_fallback<IsVarlen, IsMla, Mask>(
      stream,
      d_o,
      q,
      k,
      v,
      o,
      lse,
      dq,
      dk,
      dv,
      cumulative_seqlen_q,
      cumulative_seqlen_kv,
      softmax_scale,
      max_seqlen_q,
      total_seqlen_kv);
}


void FMHACutlassSM120BwdRun(at::Tensor workspace_buffer, at::Tensor d_o, at::Tensor q, at::Tensor k,
                            at::Tensor v, at::Tensor o, at::Tensor lse,
                            at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                            at::Tensor dq, at::Tensor dk, at::Tensor dv,
                            int mask_mode_code, float softmax_scale, int max_seqlen_q, int max_seqlen_kv, bool is_varlen) {

  const c10::cuda::OptionalCUDAGuard device_guard(q.device());

  int head_dim_qk = q.size(-1);
  int head_dim_vo = v.size(-1);
  MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  auto scalar_type_in = q.scalar_type();
  auto scalar_type_out = o.scalar_type();

  if(scalar_type_in == at::ScalarType::BFloat16 && scalar_type_out == at::ScalarType::BFloat16) {
    using Element = cutlass::bfloat16_t;
    using ElementOut = cutlass::bfloat16_t;

    auto apply_config = [&](auto fn) {
      if (mask_mode == MaskMode::kCausal) {
#if !defined(FLASH_MLA_SM120_DISABLE_VARLEN_BWD)
        if (is_varlen) {
          fn(CausalForBackwardMask<false>{}, cute::true_type{}, Element{}, ElementOut{});
        } else
#endif
        {
          fn(CausalForBackwardMask<false>{}, cute::false_type{}, Element{}, ElementOut{});
        }
      } else {
#if !defined(FLASH_MLA_SM120_DISABLE_VARLEN_BWD)
        if (is_varlen) {
          fn(ResidualMaskForBackward{}, cute::true_type{}, Element{}, ElementOut{});
        } else
#endif
        {
          fn(ResidualMaskForBackward{}, cute::false_type{}, Element{}, ElementOut{});
        }
      }
    };

    apply_config([&](auto mask, auto varlen, auto in, auto out) {
#if !defined(FLASH_MLA_SM120_DISABLE_MLA_BWD)
      if (head_dim_qk == 192 && head_dim_vo == 128) {
        call_run_fmha_bwd(mask, varlen, in, out, true_type{}, workspace_buffer, d_o, q, k, v, o, lse,
                          cumulative_seqlen_q, cumulative_seqlen_kv,
                          dq, dk, dv,
                          softmax_scale, max_seqlen_q, max_seqlen_kv);
      } else
#endif
      if (head_dim_qk == 128 && head_dim_vo == 128) {
        call_run_fmha_bwd(mask, varlen, in, out, false_type{}, workspace_buffer, d_o, q, k, v, o, lse,
                          cumulative_seqlen_q, cumulative_seqlen_kv,
                          dq, dk, dv,
                          softmax_scale, max_seqlen_q, max_seqlen_kv);      }
      else {
        std::cout << "No kernel instantiated for head_dim_qk=" << head_dim_qk << " head_dim_vo=" << head_dim_vo << std::endl;
      }
    });

  } else {
    FLASH_MLA_ASSERT(false);
  }
}
