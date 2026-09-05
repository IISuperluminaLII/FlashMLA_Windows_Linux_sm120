#include "interface.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include "sm120/prefill/dense/common/mask.cuh"
#include "sm120/prefill/dense/common/utils.hpp"

#include "sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "sm120/prefill/dense/fmha_cutlass_bwd_sm120.cuh"
#include "sm120/prefill/dense/fmha_bwd_kernel_sm120.cuh"
#include "sm120/prefill/dense/fmha_bwd_mla_kernel_sm120.cuh"
#include "sm120/prefill/dense/fmha_bwd_mma_sm120.cuh"

#include <cstdlib>
#include <algorithm>

// FLASH_MLA_SM120_BWD_MMA_CFG for the raw mma.sync backward (read once, clamped to [0, 8]):
//   8 = CFG=6 + (a) Q(n+1) issued right after B1(n) (skew-free stage WAR), (b) red.v4 fragment order (one
//       16-B reduction per n-frag pair), (c) all fragment loads of a k-step issued before its mma group in
//       dV/dK/dQ. From the diag-94 per-phase attribution (issue 8-14%, reds 10%, GEMM phases at 37-50%).
//   7 = CFG=6 with the S and dP GEMMs interleaved over their shared d range (4 accumulator chains per
//       warp in the S/dP phase); same per-accumulator order -> bitwise the same values as CFG=6.
//   6 = CFG=5 with FROST's FRAGMENT-ORDER dQ workspace: every warp red.v2 writes 256 contiguous
//       bytes (the old layout touched 8 separate 32-B row segments per instruction; diag 95 showed
//       the reductions cost 22-31% of the tile), per-batch tile-padded blocks, un-scrambled by
//       fmha_bwd_mma_dq_unscramble_kernel; dK/dV bitwise identical to CFG=2/4/5.
//   5 = CFG=4 with hoisted swizzled addressing (per-lane base registers + compile-time immediates,
//       Q-tile loop unrolled x2 so the stage is a constant): -34% instructions per tile, same math
//       order (dK/dV bitwise identical to CFG=2/4); measured a wash (the backward is not issue-bound).
//       ALL backward arms apply the BOTTOM-RIGHT causal alignment (query r sees key g iff
//       g <= r + skv - sq), the forward's / FlashAttention's convention (fixed 2026-09-05; the old
//       top-left mask produced NaN gradients for causal sq > skv in every tier).
//   4 = FROST-structured single kernel (audit/design-bwd-cfg4-frost.md, audit/impl-bwd-cfg4.md):
//       8 math warps (256 threads, 255-register cap), tiles 32 q x 64 kv @192 / 64 q x 64 kv
//       @128, per-GEMM warp partitions, V register-resident, P/dS through 2-stage swizzled
//       bf16 slabs (ldmatrix.x4.trans), Q/dO/lse/delta streamed through mbarrier-tracked
//       cp.async stages issued one tile ahead by the math warps, ONE blocking barrier per
//       tile. DETERMINISM: every dK/dV element is accumulated by exactly one warp in CFG=2's
//       fp32 summation order (ascending 16-query k-tiles, ascending Q tiles) -> dK/dV are
//       bitwise IDENTICAL to CFG=2 and bitwise reproducible run to run (the discriminating
//       gate, tests/test_bwd_cfg4_dkdv_identity.py); dQ is reduced with red.v2 into the fp32
//       accumulator -> nondeterministic at the fp32-ULP level across runs, exactly as CFG=2/3.
//   3 = CFG=2's math and dQ path with an 8-WARP CTA (256 threads) at the same smem: warp =
//       key group (w & 3) x query half (w >> 2) -> per-warp PARTIAL dK/dV over 16 queries,
//       merged ONCE at kernel end through the dead Q/dO stage buffers in a fixed fp32 order
//       (dK/dV DETERMINISTIC, bitwise reproducible; the fp32 summation order differs from
//       CFG 0-2's, so values differ from them at the fp32-ULP level); dQ stays red.v2 into
//       the fp32 accumulator -> nondeterministic at the fp32-ULP level across runs, exactly
//       as CFG=2. Design: audit/design-bwd-cfg3-warp-pairs.md (2 resident warps per SM
//       sub-partition hide the single dependent mma chain's latency).
//   2 = single-kernel like 1, but dQ is reduced register-direct with red.global.add.v2.f32
//       (no smem staging / proxy fence / extra barriers; the FROST sm120 backward's scheme).
//       Measured: CFG=1's staging protocol exposed the bulk engine's latency twice per Q block
//       and LOST to the two-kernel path (158 -> 131 TF); CFG=2 removes that serialization.
//   0 = two-kernel split (dKdV kernel + dQ kernel). Legacy path, byte-identical.
//   1 = single-kernel: the dKdV kernel also forms dQ per Q block and reduces it into an fp32
//       accumulator in workspace_buffer with cp.reduce.async.bulk (the sm100 authors' TMA
//       reduce-add scheme on sm120 silicon); a convert kernel casts it to bf16; the dQ kernel
//       (2 of 7 GEMM units of S/dP recompute + a second K/V stream) is not launched.
//       DETERMINISM: the KV-block CTAs' fp32 adds land in completion order, so dQ is
//       nondeterministic at the fp32-ULP level across runs (exactly like the authors' sm100
//       TMA reduce-add and FlashAttention-2's dQ atomics); dK/dV are unchanged and bitwise
//       reproducible. CFG=0 keeps the fully deterministic two-kernel dQ.
static int fmha_bwd_mma_cfg() {
  static const int kCfg = []() {
    const char* e = std::getenv("FLASH_MLA_SM120_BWD_MMA_CFG");
    int v = (e == nullptr) ? 0 : std::atoi(e);
    return v < 0 ? 0 : (v > 8 ? 8 : v);
  }();
  return kCfg;
}

template <int DQK, bool kCausal>
static void run_fmha_bwd_mma_cfg(const c10::cuda::CUDAStream& stream, at::Tensor workspace_buffer,
                                 at::Tensor d_o, at::Tensor q, at::Tensor k, at::Tensor v,
                                 at::Tensor o, at::Tensor lse, at::Tensor cu_q, at::Tensor cu_kv,
                                 at::Tensor dq, at::Tensor dk, at::Tensor dv,
                                 float softmax_scale, int max_seqlen_q, int max_seqlen_kv) {
  // Diagnostic (WRONG dQ, timing only): FLASH_MLA_SM120_BWD_MMA_DIAG_NORED=1 runs the CFG=5 tile with
  // the dQ red.v2 reductions removed, to measure the L2 reduction traffic's share of the tile time.
  static const bool kDiagNoRed = []() {
    const char* e = std::getenv("FLASH_MLA_SM120_BWD_MMA_DIAG_NORED");
    return e != nullptr && e[0] == '1';
  }();
  // Diagnostic (correct results, perturbed timing): FLASH_MLA_SM120_BWD_MMA_DIAG_TIMING=1 runs the CFG=6
  // tile with per-phase clock64 attribution and prints the breakdown to stderr after every launch.
  static const bool kDiagTiming = []() {
    const char* e = std::getenv("FLASH_MLA_SM120_BWD_MMA_DIAG_TIMING");
    return e != nullptr && e[0] == '1';
  }();
  if (kDiagTiming) {
    flash::detail::mma_bwd::launch_fmha_bwd_mma<DQK, 128, kCausal, 94>(
        stream, d_o, q, k, v, o, lse, cu_q, cu_kv, dq, dk, dv, softmax_scale, max_seqlen_q,
        max_seqlen_kv, workspace_buffer);
    return;
  }
  if (kDiagNoRed) {
    flash::detail::mma_bwd::launch_fmha_bwd_mma<DQK, 128, kCausal, 95>(
        stream, d_o, q, k, v, o, lse, cu_q, cu_kv, dq, dk, dv, softmax_scale, max_seqlen_q,
        max_seqlen_kv, workspace_buffer);
    return;
  }
  switch (fmha_bwd_mma_cfg()) {
    case 1:
      flash::detail::mma_bwd::launch_fmha_bwd_mma<DQK, 128, kCausal, 1>(
          stream, d_o, q, k, v, o, lse, cu_q, cu_kv, dq, dk, dv, softmax_scale, max_seqlen_q,
          max_seqlen_kv, workspace_buffer);
      break;
    case 2:   // single kernel, dQ via register-direct red.global.add.v2.f32 (FROST sm120 scheme)
      flash::detail::mma_bwd::launch_fmha_bwd_mma<DQK, 128, kCausal, 2>(
          stream, d_o, q, k, v, o, lse, cu_q, cu_kv, dq, dk, dv, softmax_scale, max_seqlen_q,
          max_seqlen_kv, workspace_buffer);
      break;
    case 3:   // CFG=2 with the 8-warp warp-pair split; dK/dV merged deterministically at kernel
              // end; dQ still red.v2 (fp32-ULP nondeterministic across runs, like CFG=2)
      flash::detail::mma_bwd::launch_fmha_bwd_mma<DQK, 128, kCausal, 3>(
          stream, d_o, q, k, v, o, lse, cu_q, cu_kv, dq, dk, dv, softmax_scale, max_seqlen_q,
          max_seqlen_kv, workspace_buffer);
      break;
    case 4:   // FROST-structured 8-math-warp single kernel; dK/dV bitwise identical to CFG=2
              // (one warp per element, CFG=2's summation order); dQ red.v2 as CFG=2/3
      flash::detail::mma_bwd::launch_fmha_bwd_mma<DQK, 128, kCausal, 4>(
          stream, d_o, q, k, v, o, lse, cu_q, cu_kv, dq, dk, dv, softmax_scale, max_seqlen_q,
          max_seqlen_kv, workspace_buffer);
      break;
    case 5:   // CFG=4 with HOISTED swizzled addressing (per-lane base registers + immediates, stage
              // unrolled x2): same math order -> dK/dV bitwise identical to CFG=2/4; dQ red.v2
      flash::detail::mma_bwd::launch_fmha_bwd_mma<DQK, 128, kCausal, 5>(
          stream, d_o, q, k, v, o, lse, cu_q, cu_kv, dq, dk, dv, softmax_scale, max_seqlen_q,
          max_seqlen_kv, workspace_buffer);
      break;
    case 6:   // CFG=5 + fragment-order (coalesced) dQ reductions + un-scramble convert kernel
      flash::detail::mma_bwd::launch_fmha_bwd_mma<DQK, 128, kCausal, 6>(
          stream, d_o, q, k, v, o, lse, cu_q, cu_kv, dq, dk, dv, softmax_scale, max_seqlen_q,
          max_seqlen_kv, workspace_buffer);
      break;
    case 7:   // CFG=6 + S/dP GEMM interleave (4 accumulator chains in the S/dP phase)
      flash::detail::mma_bwd::launch_fmha_bwd_mma<DQK, 128, kCausal, 7>(
          stream, d_o, q, k, v, o, lse, cu_q, cu_kv, dq, dk, dv, softmax_scale, max_seqlen_q,
          max_seqlen_kv, workspace_buffer);
      break;
    case 8:   // CFG=6 + issue-after-B1 + red.v4 fragment order + batched fragment loads
      flash::detail::mma_bwd::launch_fmha_bwd_mma<DQK, 128, kCausal, 8>(
          stream, d_o, q, k, v, o, lse, cu_q, cu_kv, dq, dk, dv, softmax_scale, max_seqlen_q,
          max_seqlen_kv, workspace_buffer);
      break;
    default:
      flash::detail::mma_bwd::launch_fmha_bwd_mma<DQK, 128, kCausal, 0>(
          stream, d_o, q, k, v, o, lse, cu_q, cu_kv, dq, dk, dv, softmax_scale, max_seqlen_q,
          max_seqlen_kv, workspace_buffer);
      break;
  }
}

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
    // CUDA-graph-safe max_seqlen_kv: total_seqlen_kv already carries Python's max_seqlen_kv (the
    // per-seq max). Reading cu_seqlens on the host (device->host copy + sync) is ILLEGAL during
    // CUDA-graph capture (max-autotune uses CUDA graphs), so use the passed value directly.
    int max_seqlen_kv = IsVarlen ? total_seqlen_kv : (total_seqlen_kv / batch_size);

    // raw mma.sync two-kernel-split backward (DEFAULT ON; opt out FLASH_MLA_SM120_BWD_MMA=0).
    // Atomic-free -> writes bf16 dq/dk/dv DIRECTLY (no fp32 scratch, no .to() copy).
    {
      static const bool kUseMmaBwd = []() {
        const char* e = std::getenv("FLASH_MLA_SM120_BWD_MMA");
        return (e == nullptr) || (e[0] != '0');  // DEFAULT ON; opt out with =0
      }();
      if (kUseMmaBwd) {
        // FLASH_MLA_SM120_BWD_MMA_CFG: 0 = two-kernel split, 1 = single-kernel dQ via
        // cp.reduce.async.bulk, 2 = single-kernel dQ via red.v2, 3 = CFG=2 with the 8-warp
        // CTA, 4 = FROST-structured 8-math-warp kernel (see fmha_bwd_mma_cfg() above for the
        // determinism note).
        run_fmha_bwd_mma_cfg<128, IsCausal>(
            stream, workspace_buffer, d_o, q, k, v, o, lse, cumulative_seqlen_q, cumulative_seqlen_kv,
            dq, dk, dv, softmax_scale, max_seqlen_q, max_seqlen_kv);
        return;
      }
    }

    // WMMA fallback path uses fp32 atomic accumulation (dq) + fp32->bf16 cast.
    auto dq_float = at::zeros_like(dq, dq.options().dtype(at::kFloat));
    auto dk_float = at::zeros_like(dk, dk.options().dtype(at::kFloat));
    auto dv_float = at::zeros_like(dv, dv.options().dtype(at::kFloat));

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

  // Fused 192/128 MLA WMMA backward. DEFAULT ON for the MLA head dims (this repo's
  // 192/128 model is the primary workload); OPT-OUT with FLASH_MLA_SM120_FUSED_MLA_BWD=0
  // to fall back to the ATen path for ablation/debugging. Only valid for the MLA head
  // dims (head_dim_qk=192, head_dim_vo=128); produces fp32 dQ/dK/dV (dQ via atomics
  // across KV-blocks) then casts back to the caller dtype, mirroring the 128 WMMA path.
  if constexpr (IsMla) {
    // EXPERIMENTAL raw mma.sync + two-kernel-split backward for MLA 192/128 (OPT-IN,
    // default OFF), gated by FLASH_MLA_SM120_BWD_MMA=1. Atomic-free; ~5x the WMMA MLA bwd.
    {
      static const bool kUseMmaBwd = []() {
        const char* e = std::getenv("FLASH_MLA_SM120_BWD_MMA");
        return (e == nullptr) || (e[0] != '0');  // DEFAULT ON; opt out with =0
      }();
      if (kUseMmaBwd && head_dim == 192 && v.size(-1) == 128 &&
          q.scalar_type() == at::ScalarType::BFloat16) {
        // CUDA-graph-safe: total_seqlen_kv == Python's max_seqlen_kv; a device->host copy of
        // cu_seqlens is illegal during CUDA-graph capture (max-autotune).
        int max_seqlen_kv = IsVarlen ? total_seqlen_kv : (total_seqlen_kv / batch_size);
        // FLASH_MLA_SM120_BWD_MMA_CFG: 0 = two-kernel split, 1 = single-kernel dQ via
        // cp.reduce.async.bulk, 2 = single-kernel dQ via red.v2, 3 = CFG=2 with the 8-warp
        // CTA, 4 = FROST-structured 8-math-warp kernel (see fmha_bwd_mma_cfg() above for the
        // determinism note).
        run_fmha_bwd_mma_cfg<192, IsCausal>(
            stream, workspace_buffer, d_o, q, k, v, o, lse, cumulative_seqlen_q, cumulative_seqlen_kv,
            dq, dk, dv, softmax_scale, max_seqlen_q, max_seqlen_kv);
        return;
      }
    }

    static const bool kUseFusedMlaBwd = []() {
      const char* e = std::getenv("FLASH_MLA_SM120_FUSED_MLA_BWD");
      return (e == nullptr) || (e[0] != '0');  // unset/anything-but-0 => ON; "0" => OFF
    }();
    if (kUseFusedMlaBwd && head_dim == 192 && v.size(-1) == 128 &&
        q.scalar_type() == at::ScalarType::BFloat16) {
      // CUDA-graph-safe: total_seqlen_kv == Python's max_seqlen_kv; a device->host copy of
      // cu_seqlens is illegal during CUDA-graph capture (max-autotune).
      int max_seqlen_kv = IsVarlen ? total_seqlen_kv : (total_seqlen_kv / batch_size);
      auto dq_float = at::zeros_like(dq, dq.options().dtype(at::kFloat));
      auto dk_float = at::empty_like(dk, dk.options().dtype(at::kFloat));
      auto dv_float = at::empty_like(dv, dv.options().dtype(at::kFloat));
      flash::detail::launch_fmha_bwd_sm120_mla<IsCausal>(
          stream, d_o, q, k, v, o, lse, cumulative_seqlen_q, cumulative_seqlen_kv,
          dq_float, dk_float, dv_float, softmax_scale, max_seqlen_q, max_seqlen_kv);
      dq.copy_(dq_float.to(dq.scalar_type()));
      dk.copy_(dk_float.to(dk.scalar_type()));
      dv.copy_(dv_float.to(dv.scalar_type()));
      return;
    }
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
