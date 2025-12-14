#include "interface.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include "sm120/prefill/dense/common/mask.cuh"
#include "sm120/prefill/dense/common/utils.hpp"

#include "sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "sm120/prefill/dense/fmha_cutlass_bwd_sm120.cuh"

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

  // SM120 uses fallback implementation due to TMEM/TMA constraints
  // SM120 (Blackwell workstation GPUs) does NOT have TMEM or TCGEN05/UMMA
  // which are datacenter-only features (SM100). Use ATen fallback instead.
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

  // Use fallback for all SM120 configurations - the CUTLASS backward mainloop
  // has TMA/TMEM dependencies that SM120 doesn't support
  const auto stream = c10::cuda::getCurrentCUDAStream();
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
