#include "interface.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include "sm120/prefill/dense/common/mask.cuh"
#include "sm120/prefill/dense/common/utils.hpp"

#include "sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "sm120/prefill/dense/fmha_cutlass_fwd_sm120.cuh"
#include "sm120/prefill/dense/fmha_fwd_mla_kernel_sm120.cuh"
#include "sm120/prefill/dense/fmha_fwd_mma_sm120.cuh"

#include <cstdlib>

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

    // EXPERIMENTAL raw mma.sync + ldmatrix forward (OPT-IN, default OFF) for A/B vs WMMA.
    // 128/128 MHA only for now; gated by FLASH_MLA_SM120_FWD_MMA=1.
    {
      static const bool kUseMmaFwd = []() {
        const char* e = std::getenv("FLASH_MLA_SM120_FWD_MMA");
        return (e == nullptr) || (e[0] != '0');  // DEFAULT ON; opt out with =0
      }();
      const bool mma_mha = (q.size(1) == k.size(1));
      const bool mma_128 = (head_dim_qk == 128 && head_dim_vo == 128);
      const bool mma_192 = (head_dim_qk == 192 && head_dim_vo == 128);
      if (kUseMmaFwd && mma_mha && (mma_128 || mma_192)) {
        const auto stream = c10::cuda::getCurrentCUDAStream();
        const bool causal = (mask_mode == MaskMode::kCausal);
        // FLASH_MLA_SM120_FWD_MMA_CFG ladder (0 = legacy byte-identical; 1 smem alias,
        // 2 ldmatrix.x4, 3 exp2-folded softmax, 4 conditional rescale, 5 reversed causal
        // order, 6 ILP, 7 BKV=64 staggered buffers, 8 warp-specialized 128-row CTA with the
        // full/empty mbarrier pipeline, 9 = 8 with the cp.async.bulk row loader, 10 = 8 with
        // the TMA tensor-map loader, 11 = 10 with FROST's kv_tile 128 single-stage split
        // consumption). Read once.
        static const int kFwdCfg = []() {
          const char* e = std::getenv("FLASH_MLA_SM120_FWD_MMA_CFG");
          int v = (e == nullptr) ? 0 : std::atoi(e);
          return v < 0 ? 0 : (v > 12 ? 12 : v);
        }();
        // Diagnostic rungs (softmax skipped, WRONG numerics): a separate env so the CFG ladder
        // can never select them; used only to measure a loop's tensor-only rate.
        //   1 -> kCfg 99 (rung-7 loop), 2 -> kCfg 98 (rung-8 loop), 3 -> kCfg 97 (rung-11 loop),
        //   4 -> kCfg 96 (rung-12 loop).
        static const int kFwdDiag = []() {
          const char* e = std::getenv("FLASH_MLA_SM120_FWD_MMA_DIAG_NOSOFTMAX");
          int v = (e == nullptr) ? 0 : std::atoi(e);
          return v < 0 ? 0 : (v > 4 ? 4 : v);
        }();
        auto run = [&](auto dqk_tag, auto causal_tag) {
          constexpr int kDqk = decltype(dqk_tag)::value;
          constexpr bool kC = decltype(causal_tag)::value;
          // Timing diag (correct results, perturbed timing): rung 11 with per-phase clock64 attribution.
          static const bool kFwdDiagTiming = []() {
            const char* e = std::getenv("FLASH_MLA_SM120_FWD_MMA_DIAG_TIMING");
            return e != nullptr && e[0] == '1';
          }();
          if (kFwdDiagTiming) {
            flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 95>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv);
            return;
          }
          if (kFwdDiag == 1) {
            flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 99>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv);
            return;
          }
          if (kFwdDiag == 2) {
            flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 98>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv);
            return;
          }
          if (kFwdDiag == 3) {
            flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 97>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv);
            return;
          }
          if (kFwdDiag == 4) {
            flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 96>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv);
            return;
          }
          switch (kFwdCfg) {
            case 1: flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 1>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv); break;
            case 2: flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 2>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv); break;
            case 3: flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 3>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv); break;
            case 4: flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 4>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv); break;
            case 5: flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 5>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv); break;
            case 6: flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 6>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv); break;
            case 7: flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 7>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv); break;
            case 8: flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 8>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv); break;
            case 9: flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 9>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv); break;
            case 10: flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 10>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv); break;
            case 11: flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 11>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv); break;
            case 12: flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 12>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv); break;
            default: flash::detail::mma_fwd::launch_fmha_fwd_mma<kDqk, 128, kC, 0>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv); break;
          }
        };
        if (mma_192) {
          if (causal) run(std::integral_constant<int, 192>{}, std::true_type{});
          else        run(std::integral_constant<int, 192>{}, std::false_type{});
        } else {
          if (causal) run(std::integral_constant<int, 128>{}, std::true_type{});
          else        run(std::integral_constant<int, 128>{}, std::false_type{});
        }
        return;
      }
    }

    // Fused WMMA forward (DEFAULT ON; opt-OUT with FLASH_MLA_SM120_FUSED_FWD=0). Replaces
    // the ATen fp32 forward fallback for the DENSE path -- ports the sparse-prefill WMMA
    // approach to dense full attention (FlashAttention-2 online softmax). Covers 192/128
    // (MLA) and 128/128, MHA only; emits O (bf16) + LSE (fp32, head-major .T view).
    {
      static const bool kUseFusedFwd = []() {
        const char* e = std::getenv("FLASH_MLA_SM120_FUSED_FWD");
        return (e == nullptr) || (e[0] != '0');
      }();
      const bool causal = (mask_mode == MaskMode::kCausal);
      const bool mha = (q.size(1) == k.size(1));
      const bool dims_ok = (head_dim_vo == 128) && (head_dim_qk == 192 || head_dim_qk == 128);
      if (kUseFusedFwd && dims_ok && mha) {
        const auto stream = c10::cuda::getCurrentCUDAStream();
        if (head_dim_qk == 192) {
          if (causal) flash::detail::launch_fmha_fwd_sm120_mla<192, 128, true>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv);
          else        flash::detail::launch_fmha_fwd_sm120_mla<192, 128, false>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv);
        } else {
          if (causal) flash::detail::launch_fmha_fwd_sm120_mla<128, 128, true>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv);
          else        flash::detail::launch_fmha_fwd_sm120_mla<128, 128, false>(stream, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale, max_seqlen_q, max_seqlen_kv);
        }
        return;
      }
    }

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
