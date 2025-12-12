#include "interface.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include "sm120/prefill/dense/common/mask.cuh"
#include "sm120/prefill/dense/common/utils.hpp"

#include "sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "sm120/prefill/dense/fmha_cutlass_fwd_sm120.cuh"

template <class Mask, class Varlen, class Element, class ElementOut, class Mla>
void call_run_fmha_fwd([[maybe_unused]] Mask mask, [[maybe_unused]] Varlen is_varlen,
                       [[maybe_unused]] Element in, [[maybe_unused]] ElementOut out,
                       [[maybe_unused]] Mla mla, at::Tensor workspace_buffer, at::Tensor q,
                       at::Tensor k, at::Tensor v, at::Tensor cumulative_seqlen_q,
                       at::Tensor cumulative_seqlen_kv, at::Tensor o, at::Tensor lse,
                       float softmax_scale, int max_seqlen_q, int max_seqlen_kv) {
  static constexpr bool IsVarlen = std::is_same_v<Varlen, true_type>;
  static constexpr bool IsMla = std::is_same_v<Mla, true_type>;
  static constexpr bool IsCausalMask = std::is_same_v<Mask, CausalMask<false>>;
  using Option =
      std::conditional_t<IsCausalMask || (IsVarlen), Option<Tag::kIsPersistent, false_type>,
                         Option<Tag::kIsPersistent, true_type>>;

  // SM120 uses fallback implementation due to TMEM layout constraints
  // The SM120 small-tile config (64x16) has TMEM copy atom size mismatches
  // that require extensive layout engineering to resolve.
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

  // Use fallback for all SM120 configurations - the CUTLASS mainloop
  // has TMEM atom layout incompatibilities with the small-tile config
  const auto stream = c10::cuda::getCurrentCUDAStream();
  flash::detail::run_fmha_fwd_sm120_fallback<IsVarlen, IsMla, Mask>(
      stream,
      q.scalar_type(),
      o.scalar_type(),
      q,
      k,
      v,
      o,
      lse,
      softmax_scale,
      cumulative_seqlen_q,
      cumulative_seqlen_kv,
      max_seqlen_q,
      max_seqlen_kv);
}

void FMHACutlassSM120FwdRun(at::Tensor workspace_buffer, at::Tensor q, at::Tensor k,
                            at::Tensor v, at::Tensor cumulative_seqlen_q,
                            at::Tensor cumulative_seqlen_kv, at::Tensor o, at::Tensor lse,
                            int mask_mode_code, float sm_scale, int max_seqlen_q,
                            int max_seqlen_kv, bool is_varlen) {
  const c10::cuda::OptionalCUDAGuard device_guard(q.device());
  CHECK(q.scalar_type() == k.scalar_type());
  auto scalar_type_in = q.scalar_type();
  auto scalar_type_out = o.scalar_type();
  int head_dim_qk = q.size(-1);
  int head_dim_vo = v.size(-1);
  MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  if (scalar_type_in == at::ScalarType::BFloat16 &&
      scalar_type_out == at::ScalarType::BFloat16) {
    using Element = cutlass::bfloat16_t;
    using ElementOut = cutlass::bfloat16_t;

    auto apply_config = [&](auto fn) {
      if (mask_mode == MaskMode::kCausal) {
        if (is_varlen) {
          fn(CausalMask<false>{}, cute::true_type{}, Element{}, ElementOut{});
        } else {
          fn(CausalMask<false>{}, cute::false_type{}, Element{}, ElementOut{});
        }
      } else {
        if (is_varlen) {
          fn(ResidualMask{}, cute::true_type{}, Element{}, ElementOut{});
        } else {
          fn(ResidualMask{}, cute::false_type{}, Element{}, ElementOut{});
        }
      }
    };

    apply_config([&](auto mask, auto varlen, auto in, auto out) {
      if (head_dim_qk == 192 && head_dim_vo == 128) {
        const auto stream = c10::cuda::getCurrentCUDAStream();
        flash::detail::run_fmha_fwd_sm120_fallback<decltype(varlen)::value, true, decltype(mask)>(
            stream,
            q.scalar_type(),
            o.scalar_type(),
            q,
            k,
            v,
            o,
            lse,
            sm_scale,
            cumulative_seqlen_q,
            cumulative_seqlen_kv,
            max_seqlen_q,
            max_seqlen_kv);
      } else if (head_dim_qk == 128 && head_dim_vo == 128) {
        call_run_fmha_fwd(mask, varlen, in, out, false_type{}, workspace_buffer, q, k, v,
                          cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale,
                          max_seqlen_q, max_seqlen_kv);
      } else {
        std::cout << "No kernel instantiated for head_dim_qk=" << head_dim_qk
                  << " head_dim_vo=" << head_dim_vo << std::endl;
      }
    });

  } else {
    FLASH_MLA_ASSERT(false);
  }
}
