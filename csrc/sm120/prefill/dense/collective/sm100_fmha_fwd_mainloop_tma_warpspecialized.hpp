/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/arch/simd_sm100.hpp"
#include "cute/tensor.hpp"
#include "cute/layout.hpp"

#include "../collective/fmha_common.hpp"
#include "../collective/fmha_fusion.hpp"
#include "../collective/sm100_fmha_load_tma_warpspecialized.hpp"
#include "cute/atom/copy_traits_sm100.hpp"

//==============================================================================
// SM120 Architecture Compatibility Notes
//==============================================================================
// SM120 (RTX 50 series / Blackwell workstation) does NOT have:
//   - TCGEN05/UMMA (SM100_MMA_F16BF16_SS atoms) - datacenter only
//   - TMEM (Tensor Memory) - datacenter only
//
// SM120 DOES have:
//   - TMA (Tensor Memory Accelerator) via sm_120a
//   - Legacy mma.sync.aligned for bf16 (SM80-style)
//   - Native F8F6F4 MMA support
//
// Current Status:
//   - bf16 path uses SM100 CollectiveBuilder which selects UMMA atoms
//   - This FAILS on SM120 with "SM100_MMA_F16BF16_SS without CUTE_ARCH_MMA_SM100A_ENABLED"
//
// Options for SM120:
//   1. [FUTURE] SM80-style bf16: Requires new mainloop with mma.sync.aligned
//   2. [AVAILABLE] FP8 fallback: Set FLASH_MLA_SM120_USE_FP8=1 to use native SM120 F8F6F4
//
// To enable FP8 fallback: FLASH_MLA_SM120_USE_FP8=1 pip install .
//==============================================================================

#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
// SM120 bf16 path - currently uses SM100 UMMA which is NOT available on SM120
// This will compile but fail at runtime with UMMA errors
// TODO: Implement SM80-style bf16 mainloop for SM120
#define FLASH_MLA_SM120_BF16_PATH 1
#endif

#if defined(FLASH_MLA_BUILD_SM120) && defined(FLASH_MLA_SM120_USE_FP8)
// SM120 FP8 path - use native SM120 F8F6F4 MMA
// TODO: Full FP8 implementation requires:
//   1. Change Element type to cutlass::float_e4m3_t
//   2. Change arch from Sm100 to Sm120
//   3. Use SM120-compatible kernel schedule (KernelTmaWarpSpecializedCooperativeSm120)
//   4. Remove/stub TMEM operations (not available on SM120)
#define FLASH_MLA_SM120_FP8_PATH 1
#endif

namespace flash::detail {
// Create a contiguous register tensor with the same shape as the partitioned source view.
// Uses shape (not layout) to ensure contiguous strides for register storage.
template <class Element, class Tensor>
CUTE_HOST_DEVICE auto make_softmax_store_register(Tensor const& tmem_store_cS) {
  return cute::make_tensor<Element>(cute::shape(tmem_store_cS));
}
}  // namespace flash::detail

namespace cutlass::fmha::collective {

using namespace cute;

template<
  class Element_,
  class ElementQK_,
  class ElementPV_,
  class TileShape_,
  class StrideQ_,
  class StrideK_,
  class StrideV_,
  class Mask_,
  // shape here is QG K H
  // and referes to the two softmax warps
  // (2, 1, 1) means that they are stacked (best for large Q since it loads the least K/V)
  // (1, 2, 1) means they sit side by side (best for small Q / large K)
  class ThreadShape = Shape<_2, _1, _1>,
  // Since shared memory is sufficient for FMHA, there is no need to reuse shared memory.
  class OrderLoadEpilogue = cute::false_type
>
struct Sm100FmhaFwdMainloopTmaWarpspecialized {

  using Element = Element_;
  using ElementQK = ElementQK_;
  using ElementPV = ElementPV_;
  using TileShape = TileShape_;
  using StrideQ = StrideQ_;
  using StrideK = StrideK_;
  using StrideV = StrideV_;
  using Mask = Mask_;

  // SM120: aggressively bound staging to fit 99KB SMEM
  // Keep Q double-buffered, but cap KV to 2 stages to reduce smem
  static constexpr int StageCountQ = 2;
  static constexpr int StageCountKV = 2;

  using StagesQ = cutlass::gemm::collective::StageCount<StageCountQ>;
  using StagesKV = cutlass::gemm::collective::StageCount<StageCountKV>;
  
  using ClusterShape = Shape<_1, _1, _1>;

  static const int Alignment = 128 / sizeof_bits_v<Element>;

  using TileShapeQK = decltype(shape_div(TileShape{}, ThreadShape{}));

  using TileShapePV = decltype(select<0,2,1>(TileShapeQK{}));

  static constexpr bool kIsSm120SmallTile =
      (decltype(size<0>(TileShape{}))::value == 64) &&
      (decltype(size<1>(TileShape{}))::value == 16);
  // SM120 small-tile (64 rows): use 16dp atoms (16*4=64 matches row count)
  // SM100 (128+ rows): use 32dp atoms
  static constexpr bool kUseSm120Tmem16dp = kIsSm120SmallTile;

  // V stats buffer: 64 rows × 4 stats = 256 float elements
  // ValID after upcast<32>: 4x atom = 128 elements (too small), 8x atom = 256 elements (correct)
  using TMEM_LOAD_V = std::conditional_t<
      kIsSm120SmallTile,
      SM100_TMEM_LOAD_16dp32b8x,   // 16*4=64 rows, 8x -> 256 float elements for SM120
      SM100_TMEM_LOAD_32dp32b2x>;

  using TMEM_STORE_V = std::conditional_t<
      kIsSm120SmallTile,
      SM100_TMEM_STORE_16dp32b8x,  // 16*4=64 rows, 8x -> 256 float elements for SM120
      SM100_TMEM_STORE_32dp32b2x>;

  // Use same atoms for stats operations (V buffer load/store)
  using TMEM_LOAD_V_OP = TMEM_LOAD_V;
  using TMEM_STORE_V_OP = TMEM_STORE_V;

  // P buffer store atoms: 64×16 = 1024 elements for SM120, larger for SM100
  // ValID after upcast<32>: 16x = 512 elements (too small), 32x = 1024 elements (correct)
  using TMEM_STORE_P = std::conditional_t<
      kIsSm120SmallTile,
      SM100_TMEM_STORE_16dp32b32x,  // 16*4=64 rows, 32x -> 1024 float elements for SM120
      SM100_TMEM_STORE_32dp32b32x>; // 32*4=128 rows for SM100

  struct StoreVTraits {
    using CopyTraits = Copy_Traits<TMEM_STORE_V_OP>;
    using ThrID = typename CopyTraits::ThrID;
    using ValID = typename CopyTraits::ValID;
    using SrcLayout = typename CopyTraits::SrcLayout;
    using DstLayout = typename CopyTraits::DstLayout;
    using RefLayout = typename CopyTraits::RefLayout;
  };
  using StorePTraits = Copy_Traits<TMEM_STORE_P>;
  // Stats load for S matrix: must match TileShapeQK element count (64×16 = 1024 for SM120)
  // ValID after upcast<32>: 16x = 512 elements (too small), 32x = 1024 elements (correct)
  // SM100: (256, 128) = 32768 elements per stage (128 rows per thread with ThreadShape=(2,1,1))
  using StatsLoadOp = std::conditional_t<
      kIsSm120SmallTile,
      SM100_TMEM_LOAD_16dp32b32x,   // 16*4=64 rows, 32x -> 1024 float elements for SM120
      SM100_TMEM_LOAD_32dp32b16x>;
  using StatsLoadTraits = Copy_Traits<StatsLoadOp>;
  // TMEM layouts: coalesce(upcast(ValID)) - this matches make_tmem_copy's atom_v_layout pattern
  // coalesce is critical! make_tmem_copy uses coalesce(upcast<sizeof_bits<T>>(ValID))
  using StatsLayoutVStore = decltype(coalesce(upcast<sizeof_bits_v<ElementQK>>(typename Copy_Traits<TMEM_STORE_V_OP>::ValID{})));
  using StatsLayoutVLoad = decltype(coalesce(upcast<sizeof_bits_v<ElementQK>>(typename Copy_Traits<StatsLoadOp>::ValID{})));
  // Register layouts: same shape as TMEM layouts, but with contiguous strides for register storage
  // Using make_layout(shape(...)) creates row-major contiguous layout matching the coalesced shape
  using StatsRegLayoutStore = decltype(make_layout(shape(StatsLayoutVStore{})));
  using StatsRegLayoutLoad = decltype(make_layout(shape(StatsLayoutVLoad{})));

  // P-buffer: same pattern - coalesce(upcast(ValID)) for TMEM, contiguous for registers
  using PLayoutTmem = decltype(coalesce(upcast<sizeof_bits_v<ElementQK>>(typename StorePTraits::ValID{})));
  using PLayoutReg = decltype(make_layout(shape(PLayoutTmem{})));

  // Correction rescale: 16dp32b32x atoms for 64x16 tile (kTileM=64, kCorrectionTileSize=16)
  // Uses 16dp (16 rows/warp * 4 warps = 64 rows), 32x (32 elements/thread * 32 threads = 1024 elements)
  using CorrectionLoadOp = SM100_TMEM_LOAD_16dp32b32x;
  using CorrectionStoreOp = SM100_TMEM_STORE_16dp32b32x;
  using CorrectionLayoutTmem = decltype(coalesce(upcast<sizeof_bits_v<ElementPV>>(typename Copy_Traits<CorrectionLoadOp>::ValID{})));
  using CorrectionLayoutReg = decltype(make_layout(shape(CorrectionLayoutTmem{})));

  //============================================================================
  // CollectiveBuilder Configuration
  //============================================================================
  // WARNING: SM120 does NOT support SM100 UMMA (TCGEN05) atoms!
  // The CollectiveBuilder below uses cutlass::arch::Sm100 which selects
  // SM100_MMA_F16BF16_SS atoms. These will FAIL at runtime on SM120 with:
  //   "Attempting to use SM100_MMA_F16BF16_SS without CUTE_ARCH_MMA_SM100A_ENABLED"
  //
  // For SM120 native support, two options:
  //   1. [FUTURE] SM80-style bf16: New mainloop with mma.sync.aligned.m16n8k16
  //   2. [FP8] Use FLASH_MLA_SM120_USE_FP8=1 and switch to:
  //      - cutlass::arch::Sm120
  //      - cutlass::float_e4m3_t Element type
  //      - KernelTmaWarpSpecializedCooperativeSm120 schedule
  //============================================================================

#if defined(FLASH_MLA_SM120_FP8_PATH)
  // TODO: SM120 FP8 CollectiveBuilder - requires full Element type refactoring
  // For now, this path is a placeholder. Full implementation requires:
  //   1. Change Element template parameter to cutlass::float_e4m3_t at call sites
  //   2. Use cutlass::arch::Sm120 below
  //   3. Remove TMEM dependencies (SM120 has no TMEM)
  #error "SM120 FP8 path not yet fully implemented. Use bf16 path with SM100 hardware, or contribute the FP8 implementation."
#endif

  //==============================================================================
  // SM120 BF16 Path - Use SM80 architecture for MMA atom selection
  //==============================================================================
  // SM120 does NOT have TCGEN05/UMMA hardware (datacenter-only).
  // Instead, SM120 supports legacy mma.sync.aligned instructions from SM80.
  // By using cutlass::arch::Sm80, the CollectiveBuilder selects:
  //   - SM80_16x8x16_F32BF16BF16F32_TN via mma.sync.aligned.m16n8k16
  // These atoms are backward compatible and work on all SM >= 80.
  //
  // WARNING: Even with SM80 CollectiveBuilder, this mainloop still uses TMEM
  // operations which are NOT available on SM120. A complete SM120 solution
  // requires replacing all TMEM usage with shared memory operations.
  // This is tracked as a future enhancement.
  //==============================================================================

#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
  // SM120 BF16: Use SM90 architecture to get TMA support while avoiding UMMA
  // SM90 provides TMA types (TMA_A, TMA_B) needed by Load collective
  // SM120 has TMA hardware (sm_120a) but no TCGEN05/UMMA (datacenter only)
  // KernelTmaWarpSpecialized provides TMA descriptors in Params
  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Element, StrideQ, Alignment,
      Element, StrideK, Alignment,
      ElementQK,
      TileShapeQK, ClusterShape, cutlass::gemm::collective::StageCount<3>,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;

  using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Element, StrideK, Alignment,
      Element, decltype(select<1,0,2>(StrideV{})), Alignment,
      ElementPV,
      TileShapePV, ClusterShape, cutlass::gemm::collective::StageCount<3>,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;
#else
  // SM100 (datacenter): Use SM100 architecture with UMMA atoms
  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Element, StrideQ, Alignment,
      Element, StrideK, Alignment,
      ElementQK,
      TileShapeQK, ClusterShape, cutlass::gemm::collective::StageCount<3> /* we change it later anyways*/,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;

  using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      // the stride for A does not matter since we do not load from smem at all
      Element, StrideK, Alignment,
      Element, decltype(select<1,0,2>(StrideV{})), Alignment,
      ElementPV,
      TileShapePV, ClusterShape, cutlass::gemm::collective::StageCount<3> /* we change it later anyways*/,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;
#endif

  using SmemLayoutQ = decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutA{}, Int<StageCountQ>{}));
  using SmemLayoutK = decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutB{}, Int<StageCountKV>{}));
  using SmemLayoutV = decltype(unstageSmemLayout(typename CollectiveMmaPV::SmemLayoutB{}, Int<StageCountKV>{}));

  // Reuse shared memory for V and O.
  static constexpr bool IsOrderLoadEpilogue = std::is_same_v<OrderLoadEpilogue, cute::true_type>;
  struct TensorStorage {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    };
  };

  enum class TmemAllocation : uint32_t {
    kSizeS = 128,
    kSizeO = 128,
    kSizeP = 32,
    S0 = 0,
    S1 = S0 + kSizeS,
    V0 = S0,  // stats storage from softmax to correction
    V1 = S1,
    P0 = S0 + kSizeP,
    P1 = S1 + kSizeP,
    O0 = S1 + kSizeS,
    O1 = O0 + kSizeO,
    kEnd = O1 + kSizeO
  };

  // indices for V0 / V1
  enum : int {
    kIdxOldRowMax = 0,
    kIdxNewRowMax = 1,
    kIdxFinalRowSum = 0,
    kIdxFinalRowMax = 1
  };

  //==============================================================================
  // Pipeline Type Selection for SM120 vs SM100
  //==============================================================================
  // SM100: Uses PipelineTmaUmmaAsync with AtomThrShapeMNK (requires UMMA)
  // SM120: Uses standard PipelineAsync (no UMMA, no AtomThrShapeMNK)
  //==============================================================================

#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
  // SM120 BF16 path: Use standard async pipelines (no UMMA)
  // Note: SM80 CpAsyncWarpSpecialized CollectiveOp does NOT have AtomThrShapeMNK

  // from load to mma warp, protects q in smem
  using PipelineQ = cutlass::PipelineAsync<StageCountQ>;

  // from load to mma warp, protects k/v in smem
  using PipelineKV = cutlass::PipelineAsync<StageCountKV>;

  // from mma to softmax0/1 warp, protects S in registers (not TMEM on SM120)
  using PipelineS = cutlass::PipelineAsync<1>;

  // from softmax0/1/ to correction wg
  using PipelineC = cutlass::PipelineAsync<1>;

  // from mma to correction
  using PipelineO = cutlass::PipelineAsync<1>;

#else
  // SM100 path: Use UMMA-specific async pipelines

  // from load to mma warp, protects q in smem
  using PipelineQ = cutlass::PipelineTmaUmmaAsync<
    StageCountQ,
    typename CollectiveMmaQK::AtomThrShapeMNK
  >;

  // from load to mma warp, protects k/v in smem
  using PipelineKV = cutlass::PipelineTmaUmmaAsync<
    StageCountKV,
    typename CollectiveMmaQK::AtomThrShapeMNK
  >;

  // from mma to softmax0/1 warp, protects S in tmem
  // (not sure yet about the reverse direction)
  // there is one pipe per softmax warp, and the mma warp alternates between them
  using PipelineS = cutlass::PipelineUmmaAsync<1>;

  // from softmax0/1/ to correction wg
  using PipelineC = cutlass::PipelineAsync<1>;

  // from mma to correction
  // SM120: reduce O pipeline stages to minimize barrier storage
  using PipelineO = cutlass::PipelineUmmaAsync<1>;

#endif

  // from corr to epilogue
  // SM120: reduce Epilogue pipeline stages to minimize barrier storage
  using PipelineE = cutlass::PipelineAsync<1>;

  using OrderBarrierSoftmax = cutlass::OrderedSequenceBarrier<
    /*stages*/ 1, /*groups*/ 2>;

  static const int TransactionBytesLoadQ = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutQ{})) * cute::sizeof_bits_v<Element>);

  static const int TransactionBytesLoadK = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutK{})) * cute::sizeof_bits_v<Element>);
  static const int TransactionBytesLoadV = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutV{})) * cute::sizeof_bits_v<Element>);

  static_assert(TransactionBytesLoadK == TransactionBytesLoadV, "K and V smem layouts must be of equal size");

  using Load = Sm100FmhaLoadTmaWarpspecialized<
    Element, StrideQ, StrideK, StrideV,
    CollectiveMmaQK, CollectiveMmaPV,
    SmemLayoutQ, SmemLayoutK, SmemLayoutV,
    TensorStorage, PipelineQ, PipelineKV, Mask, TileShape
  >;

  struct Arguments {
    typename Load::Arguments load;

    // if zero, defaults to 1/sqrt(D)
    float scale_softmax = 0.0f;

    // scaling factors to dequantize QKV
    float scale_q = 1.0f;
    float scale_k = 1.0f;
    float scale_v = 1.0f;

    // scaling factor to quantize O
    float inv_scale_o = 1.0f;
  };

  struct Params {
    typename Load::Params load;

    float scale_softmax;
    float scale_softmax_log2;

    float scale_output;
  };

  template<class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template<class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace) {

    float scale_softmax = args.scale_softmax;
    if (scale_softmax == 0.0f) {
      scale_softmax = 1.0f / (float) std::sqrt(get<2>(problem_shape));
    }
    float log2_e = static_cast<float>(std::log2(std::exp(1.0)));

    return Params{
        Load::to_underlying_arguments(problem_shape, args.load, workspace),
        args.scale_q * args.scale_k * scale_softmax,
        args.scale_q * args.scale_k * log2_e * scale_softmax,
        args.scale_v * args.inv_scale_o
    };
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
      Load::prefetch_tma_descriptors(params.load);
  }

  template<class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void
  load(
      BlkCoord const& blk_coord, ProblemShape const& problem_shape,
      Params const& params, ParamsProblemShape const& params_problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_producer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_producer_state) {

    Load load;
    load.load(blk_coord, problem_shape, params.load, params_problem_shape,
        storage,
        pipeline_q, pipeline_q_producer_state,
        pipeline_kv, pipeline_kv_producer_state);
  }

  template<class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE auto
  mma(
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_consumer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_consumer_state,
      PipelineS& pipeline_s0, typename PipelineS::PipelineState& pipeline_s0_producer_state,
      PipelineS& pipeline_s1, typename PipelineS::PipelineState& pipeline_s1_producer_state,
      PipelineO& pipeline_corr, typename PipelineO::PipelineState& pipeline_corr_producer_state) {

    auto pipeline_q_release_state = pipeline_q_consumer_state;
    auto pipeline_kv_release_state = pipeline_kv_consumer_state;

    int mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);

    typename CollectiveMmaQK::TiledMma mma_qk;
    ThrMMA thr_mma_qk = mma_qk.get_slice(0);

    typename CollectiveMmaPV::TiledMma mma_pv;
    TiledMMA mma_pv_ts = to_tiled_mma_sm100_ts(mma_pv);
    ThrMMA thr_mma_pv = mma_pv_ts.get_slice(0);

    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

    Tensor tSrQ = thr_mma_qk.make_fragment_A(sQ);
    Tensor tSrK = thr_mma_qk.make_fragment_B(sK);
    Tensor tOrV = thr_mma_pv.make_fragment_B(sV);

    // tmem layout is
    // S0 S1`O0 O1
    // sequential in memory, where S overlaps with P and V

    Tensor tStS = partition_fragment_C(mma_qk, select<0,1>(TileShapeQK{}));
    Tensor tOtO = partition_fragment_C(mma_pv_ts, select<0,1>(TileShapePV{}));

#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
    // SM120: Use register-based accumulators (no TMEM)
    // Keep tStS and tOtO as register tensors without TMEM pointer arithmetic
    Tensor tStS0 = tStS;
    Tensor tStS1 = tStS;
    Tensor tOtO0 = tOtO;
    Tensor tOtO1 = tOtO;
#else
    // SM100: Use TMEM for accumulators
    Tensor tStS0 = tStS;
    tStS0.data() = tStS.data().get() + uint32_t(TmemAllocation::S0);
    Tensor tStS1 = tStS;
    tStS1.data() = tStS.data().get() + uint32_t(TmemAllocation::S1);

    Tensor tOtO0 = tOtO;
    tOtO0.data() = tOtO.data().get() + uint32_t(TmemAllocation::O0);
    Tensor tOtO1 = tOtO;
    tOtO1.data() = tOtO.data().get() + uint32_t(TmemAllocation::O1);
#endif

    Tensor sP = make_tensor(make_smem_ptr((Element*)nullptr), typename CollectiveMmaPV::SmemLayoutA{});
    Tensor tOrP = thr_mma_pv.make_fragment_A(sP)(_, _, _, _0{});  // slice out staging

#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
    // SM120: No TMEM pointer arithmetic for P buffer
    Tensor tOrP0 = tOrP;
    Tensor tOrP1 = tOrP;
#else
    // SM100: Use TMEM for P buffer
    Tensor tOrP0 = tOrP;
    tOrP0.data() = tOrP0.data().get() + uint32_t(TmemAllocation::P0);
    Tensor tOrP1 = tOrP;
    tOrP1.data() = tOrP1.data().get() + uint32_t(TmemAllocation::P1);
#endif

    int k_index = 0;
    int v_index = 0;
    int q_index = 0;

    // wait for Q1
    q_index = pipeline_q_consumer_state.index();
    pipeline_q.consumer_wait(pipeline_q_consumer_state);
    ++pipeline_q_consumer_state;

    Tensor tSrQ0 = tSrQ(_,_,_,q_index);


    // wait for K1
    k_index = pipeline_kv_consumer_state.index();
    pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
    ++pipeline_kv_consumer_state;

    // gemm Q1 * K1 -> S1
    pipeline_s0.producer_acquire(pipeline_s0_producer_state);

    gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index), tStS0);

    pipeline_s0.producer_commit(pipeline_s0_producer_state);
    ++pipeline_s0_producer_state;

    // release K1
    if constexpr (get<1>(ThreadShape{}) > 1) {
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;
    }

    // wait for Q2
    if constexpr (get<0>(ThreadShape{}) > 1 || get<2>(ThreadShape{}) > 1) {
      q_index = pipeline_q_consumer_state.index();
      pipeline_q.consumer_wait(pipeline_q_consumer_state);
      ++pipeline_q_consumer_state;
    }

    Tensor tSrQ1 = tSrQ(_,_,_,q_index);

    if constexpr (get<1>(ThreadShape{}) > 1) {
      k_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;
    }

    pipeline_s1.producer_acquire(pipeline_s1_producer_state);

    // gemm Q2 * K1 -> S2
    gemm_zero_acc(mma_qk, tSrQ1, tSrK(_,_,_,k_index), tStS1);

    pipeline_s1.producer_commit(pipeline_s1_producer_state);
    ++pipeline_s1_producer_state;

    // release K1
    pipeline_kv.consumer_release(pipeline_kv_release_state);
    ++pipeline_kv_release_state;

    // wait for V1
    v_index = pipeline_kv_consumer_state.index();
    pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
    ++pipeline_kv_consumer_state;

    // this acquire returns the ownership of all of S0 to the mma warp
    // including the P0 part
    // acquire corr first to take it out of the critical
    // path since softmax takes longer
    pipeline_corr.producer_acquire(pipeline_corr_producer_state);
    pipeline_s0.producer_acquire(pipeline_s0_producer_state);

    // gemm P1 * V1 -> O1
    gemm_zero_acc(mma_pv_ts, tOrP0, tOrV(_,_,_,v_index), tOtO0);

    pipeline_corr.producer_commit(pipeline_corr_producer_state);
    ++pipeline_corr_producer_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;
    }

    mma_pv_ts.accumulate_ = UMMA::ScaleOut::Zero;

    // loop:
    mask_tile_count -= 1;
    for (; mask_tile_count > 0; mask_tile_count -= 1) {

      // wait for Ki
      k_index = (pipeline_kv_consumer_state.index());
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;

      // gemm Q1 * Ki -> S1
      gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index), tStS0);

      pipeline_s0.producer_commit(pipeline_s0_producer_state);
      ++pipeline_s0_producer_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;
      }

      // gemm P2 * V(i-1) -> O2
      if constexpr (get<1>(ThreadShape{}) > 1) {
        v_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;
      }

      pipeline_corr.producer_acquire(pipeline_corr_producer_state);
      pipeline_s1.producer_acquire(pipeline_s1_producer_state);

      gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_,_,_,v_index), tOtO1);

      pipeline_corr.producer_commit(pipeline_corr_producer_state);
      ++pipeline_corr_producer_state;

      // release V(i-1)
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
        k_index = (pipeline_kv_consumer_state.index());
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;
      }

      // gemm Q2 * Ki -> S2
      gemm_zero_acc(mma_qk, tSrQ1, tSrK(_,_,_,k_index), tStS1);

      pipeline_s1.producer_commit(pipeline_s1_producer_state);
      ++pipeline_s1_producer_state;

      // release Ki
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;

      // wait for Vi
      v_index = (pipeline_kv_consumer_state.index());
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;

      // gemm P1 * Vi -> O1
      pipeline_corr.producer_acquire(pipeline_corr_producer_state);

      pipeline_s0.producer_acquire(pipeline_s0_producer_state);

      gemm_reset_zero_acc(mma_pv_ts, tOrP0, tOrV(_,_,_,v_index), tOtO0);

      pipeline_corr.producer_commit(pipeline_corr_producer_state);
      ++pipeline_corr_producer_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;
      }
    }

    // release Q1
    pipeline_q.consumer_release(pipeline_q_release_state);
    ++pipeline_q_release_state;

    // release Q2
    if constexpr (get<0>(ThreadShape{}) > 1) {
      pipeline_q.consumer_release(pipeline_q_release_state);
      ++pipeline_q_release_state;
    }

    // wait for Vi
    if constexpr (get<1>(ThreadShape{}) > 1) {
      v_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;
    }

    // gemm P2 * Vi -> O2
    pipeline_corr.producer_acquire(pipeline_corr_producer_state);
    pipeline_s1.producer_acquire(pipeline_s1_producer_state);

    gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_,_,_,v_index), tOtO1);

    pipeline_corr.producer_commit(pipeline_corr_producer_state);
    ++pipeline_corr_producer_state;

    // release Vi
    pipeline_kv.consumer_release(pipeline_kv_release_state);
    ++pipeline_kv_release_state;

    pipeline_s0.producer_commit(pipeline_s0_producer_state);
    ++pipeline_s0_producer_state;

    pipeline_s1.producer_commit(pipeline_s1_producer_state);
    ++pipeline_s1_producer_state;

    // T0 S00 B1, T0 S10 B1, T0 S00 B2, T0 S01 B1, T0 S10 B2, T0 S11 B1, T0 S01 B2, T1 S00 B1, T0 S11 B2, ...
    // Q1 * K1  , Q2 * K1  , S11 * V1 , Q1 * K2  , S21 * V1  , Q2 * K2 , S12 * V2 , Q1 * K3  , S22 * K2 , ...
  }

  // H1 approach: Create tensors with EXPLICIT ValID layouts for TMEM copy compatibility.
  // The key insight from hypothesis testing: TMEM Copy_Traits require specific layouts
  // that match coalesce(upcast<32>(ValID)). MMA-produced layouts don't match directly.
  template<class Stage, class TensorS, class CoordTensor>
  CUTLASS_DEVICE static auto
  make_softmax_stats_views(Stage stage, TensorS const& /*tStS*/, CoordTensor const& /*tScS*/) {
#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
    // SM120: Use register-based storage (no TMEM)
    auto tStS_v = make_tensor<ElementQK>(StatsRegLayoutStore{});
    auto tScS_v = make_tensor<ElementQK>(StatsRegLayoutStore{});
    auto tStS_P = make_tensor<ElementQK>(PLayoutReg{});
    auto tScS_P = make_tensor<ElementQK>(PLayoutReg{});
    auto tStS_load = make_tensor<ElementQK>(StatsRegLayoutLoad{});
    auto tScS_load = make_tensor<ElementQK>(StatsRegLayoutLoad{});

    return cute::make_tuple(tStS_v, tScS_v, tStS_P, tScS_P, tStS_load, tScS_load);
#else
    // SM100: V stats: 64 rows x 4 stats = 256 float elements with StatsLayoutVStore
    uint32_t v_ptr = uint32_t(stage == _0{} ? TmemAllocation::V0 : TmemAllocation::V1);
    auto tStS_v = make_tensor(make_tmem_ptr<uint32_t>(v_ptr), StatsLayoutVStore{});
    auto tScS_v = make_tensor<ElementQK>(StatsRegLayoutStore{});

    // P buffer: 64x16 = 1024 elements with PLayoutTmem
    uint32_t p_ptr = uint32_t(stage == _0{} ? TmemAllocation::P0 : TmemAllocation::P1);
    auto tStS_P = make_tensor(make_tmem_ptr<uint32_t>(p_ptr), PLayoutTmem{});
    auto tScS_P = make_tensor<ElementQK>(PLayoutReg{});

    // S load: 64x16 = 1024 elements with StatsLayoutVLoad
    uint32_t s_ptr = uint32_t(stage == _0{} ? TmemAllocation::S0 : TmemAllocation::S1);
    auto tStS_load = make_tensor(make_tmem_ptr<uint32_t>(s_ptr), StatsLayoutVLoad{});
    auto tScS_load = make_tensor<ElementQK>(StatsRegLayoutLoad{});

    return cute::make_tuple(tStS_v, tScS_v, tStS_P, tScS_P, tStS_load, tScS_load);
#endif
  }

  // Legacy 4-tuple interface for backward compatibility with mainloop code
  template<class Stage, class TensorS, class CoordTensor>
  CUTLASS_DEVICE static auto
  make_softmax_tmem_views(Stage stage, TensorS const& tStS, CoordTensor const& tScS) {
    auto [tStS_v, tScS_v, tStS_P, tScS_P, tStS_load, tScS_load] =
        make_softmax_stats_views(stage, tStS, tScS);
    (void)tScS_v;  // Not used in 4-tuple interface
    (void)tScS_P;  // Not used in 4-tuple interface
    return cute::make_tuple(tStS_v, tStS_P, tStS_load, tScS_load);
  }

  template<bool need_apply_mask, class Stage, class BlkCoord, class CoordTensor, class ProblemShape>
  CUTLASS_DEVICE auto
  softmax_step(
      float& row_max, float& row_sum,
      Stage stage, bool final_call,
      BlkCoord const& blk_coord, CoordTensor const& cS,
      Params const& params, ProblemShape const& problem_shape,
      PipelineS& pipeline_s, typename PipelineS::PipelineState& pipeline_s_consumer_state,
      PipelineC& pipeline_c, typename PipelineC::PipelineState& pipeline_c_producer_state,
      OrderBarrierSoftmax& order_s) {

    Tensor tScS = typename CollectiveMmaQK::TiledMma{}.get_slice(0).partition_C(cS);

    Tensor tStS = partition_fragment_C(typename CollectiveMmaQK::TiledMma{}, select<0,1>(TileShapeQK{}));
#if !defined(FLASH_MLA_BUILD_SM120) || defined(FLASH_MLA_SM120_USE_FP8)
    // SM100: Set TMEM pointer for S buffer
    tStS.data() = uint32_t(stage == _0{} ? TmemAllocation::S0 : TmemAllocation::S1);
#endif

    // Get TMEM tensor views for V stats, P buffer, and S load
    // tScS_load is the original coordinate tensor (used for masking)
    auto [tStS_v, tStS_P, tStS_load, tScS_load] = make_softmax_tmem_views(stage, tStS, tScS);

    // Each thread owns a single row
    static constexpr int kTileN = decltype(size<1>(TileShapeQK{}))::value;
    static constexpr int kTilePCols = kTileN * int(sizeof(Element)) / int(sizeof(float));

    using TMEM_LOAD = StatsLoadOp;  // Stats load atom for 128 float elements
    using TMEM_STORE = TMEM_STORE_P;  // P buffer store atom selection
    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
    // SM120: Use direct register access (no TMEM copy)
    // Since make_softmax_tmem_views returns register tensors on SM120,
    // we can use them directly without TMEM copy operations
    Tensor tTMEM_LOADtS = tStS_load;
    Tensor tTMEM_LOADcS = tScS_load;
    Tensor tTMEM_STOREVtS = tStS_v;
    Tensor tTMEM_STOREVrS = tStS_v;  // Same tensor, no partition needed
    Tensor tTMEM_STOREtS_x4 = tStS_P;
#else
    // SM100: TMEM copy approach: Use partition_fragment_C tensors directly without coalescing.
    // The MMA-produced tensors have layouts matching what TMEM atoms expect.
    // Key insight: partition_fragment_C creates TMEM tensors, partition_S/D preserve layout structure.

    // S matrix load: Use tStS_load directly (from partition_fragment_C via make_softmax_tmem_views)
    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tStS_load);
    auto thr_tmem_load   = tiled_tmem_load.get_slice(thread_idx);

    // Partition preserves MMA-compatible layout structure
    Tensor tTMEM_LOADtS = thr_tmem_load.partition_S(tStS_load);
    Tensor tTMEM_LOADcS = thr_tmem_load.partition_D(tScS_load);

    // V stats store: Use tStS_v directly (has StatsLayoutVStore from atom's ValID)
    auto tiled_tmem_storev = make_tmem_copy(TMEM_STORE_V_OP{}, tStS_v);
    auto thr_tmem_storev  = tiled_tmem_storev.get_slice(thread_idx);

    // Partition preserves layout structure
    Tensor tTMEM_STOREVtS = thr_tmem_storev.partition_D(tStS_v);
    // Create register tensor with explicit layout, then partition to get per-thread view
    auto rS_v_reg = make_tensor<ElementQK>(StatsRegLayoutStore{});
    Tensor tTMEM_STOREVrS = thr_tmem_storev.partition_S(rS_v_reg);

    // P-buffer store: Use tStS_P directly (from partition_fragment_C)
    auto tiled_tmem_store = make_tmem_copy(TMEM_STORE{}, tStS_P);
    auto thr_tmem_store  = tiled_tmem_store.get_slice(thread_idx);

    // Partition preserves layout structure for size<2> check and slicing
    Tensor tTMEM_STOREtS_x4 = thr_tmem_store.partition_D(tStS_P);
    tTMEM_STOREtS_x4.data() = warp_uniform(tTMEM_STOREtS_x4.data().get());
#endif

    // wait on tensor core pipe
    pipeline_s.consumer_wait(pipeline_s_consumer_state);

#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
    // SM120: Direct register access, no copy needed (tTMEM_LOADtS already is register)
    Tensor tTMEM_LOADrS = tTMEM_LOADtS;
#else
    // SM100: Create register tensor with explicit layout, then partition to get per-thread view
    auto rS_load_reg = make_tensor<ElementQK>(StatsRegLayoutLoad{});
    Tensor tTMEM_LOADrS = thr_tmem_load.partition_D(rS_load_reg);
    // Copy from partitioned TMEM source to partitioned register destination
    copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS);
#endif

    if constexpr (need_apply_mask) {
      Mask{}.apply_mask(tTMEM_LOADrS, tTMEM_LOADcS, problem_shape);
    }

    ElementQK old_row_max = row_max;
    {
      // compute rowmax
      float row_max_0 = row_max;
      float row_max_1 = row_max;
      float row_max_2 = row_max;
      float row_max_3 = row_max;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTMEM_LOADrS); i += 4) {
        row_max_0  = ::fmax(row_max_0, tTMEM_LOADrS(i));
        row_max_1 = ::fmax(row_max_1, tTMEM_LOADrS(i+1));
        row_max_2 = ::fmax(row_max_2, tTMEM_LOADrS(i+2));
        row_max_3 = ::fmax(row_max_3, tTMEM_LOADrS(i+3));
      }
      row_max = ::fmax(row_max_0, row_max_1);
      row_max = ::fmax(row_max, row_max_2);
      row_max = ::fmax(row_max, row_max_3);
    }

    ElementQK row_max_safe = row_max;
    if (!(row_max_safe == row_max_safe)) {
      row_max_safe = ElementQK(0);
    }

    tTMEM_STOREVrS(kIdxOldRowMax) = old_row_max;
    tTMEM_STOREVrS(kIdxNewRowMax) = row_max_safe;
#if !defined(FLASH_MLA_BUILD_SM120) || defined(FLASH_MLA_SM120_USE_FP8)
    // SM100: Copy without flatten - partitioned tensors have compatible layouts
    copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);
#endif
    // SM120: tTMEM_STOREVrS and tTMEM_STOREVtS are same register tensor, no copy needed

    pipeline_c.producer_commit(pipeline_c_producer_state);
    ++pipeline_c_producer_state;

    // notify correction wg that they are ready (might need addtl ordering between S0 and S1 WG's)

    ElementQK scale = params.scale_softmax_log2;
    ElementQK row_max_scale = row_max_safe * scale;

    float2 scale_fp32x2 = make_float2(scale, scale);
    float2 minus_row_max_scale_fp32x2 = make_float2(-row_max_scale, -row_max_scale);

#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
    // SM120: Use tTMEM_STOREtS_x4 directly as register tensor
    auto tTMEM_STORErS_x4 = tTMEM_STOREtS_x4;
#else
    // SM100: Create register tensor with explicit layout, then partition to get per-thread view
    auto rS_store_reg = make_tensor<uint32_t>(PLayoutReg{});
    auto tTMEM_STORErS_x4 = thr_tmem_store.partition_S(rS_store_reg);
#endif

    constexpr int kConversionsPerStep = 2;

    auto tTMEM_STORErS_x4_e = recast<Array<Element, kConversionsPerStep>>(tTMEM_STORErS_x4);

    NumericArrayConverter<Element, ElementQK, kConversionsPerStep> convert;

    const int kReleasePipeCount = 10;  // must be multiple of 2

    order_s.wait();

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTMEM_LOADrS); i += 2) {
      float2 in = make_float2(
        tTMEM_LOADrS(i + 0),
        tTMEM_LOADrS(i + 1)
      );
      float2 out;
      cute::fma(out, scale_fp32x2, in, minus_row_max_scale_fp32x2);
      tTMEM_LOADrS(i + 0) = out.x;
      tTMEM_LOADrS(i + 1) = out.y;

      tTMEM_LOADrS(i+0) = ::exp2f(tTMEM_LOADrS(i+0));
      tTMEM_LOADrS(i+1) = ::exp2f(tTMEM_LOADrS(i+1));

      Array<ElementQK, kConversionsPerStep> in_conv;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < kConversionsPerStep; j++) {
        in_conv[j] = tTMEM_LOADrS(i + j);
      }
      tTMEM_STORErS_x4_e[i / kConversionsPerStep] = convert(in_conv);


      if (i == size(tTMEM_LOADrS) - kReleasePipeCount) {
        order_s.arrive();
      }

      // this prevents register spills in fp16
#if !defined(FLASH_MLA_BUILD_SM120) || defined(FLASH_MLA_SM120_USE_FP8)
      // SM100 only: TMEM store optimization
      if constexpr (size<2>(tTMEM_STORErS_x4) == _2{}) {
        if (i == size(tTMEM_LOADrS) - 6) {
          // STORE: use partitioned tensors directly
          copy(tiled_tmem_store, tTMEM_STORErS_x4(_, _, 0), tTMEM_STOREtS_x4(_, _, 0));
        }
      }
#endif
    }

#if !defined(FLASH_MLA_BUILD_SM120) || defined(FLASH_MLA_SM120_USE_FP8)
    // SM100: tmem_store(reg_S8) -> op_P - use partitioned tensors directly
    copy(tiled_tmem_store, tTMEM_STORErS_x4, tTMEM_STOREtS_x4);

    cutlass::arch::fence_view_async_tmem_store();
#endif
    // SM120: No TMEM store needed, P buffer is already in registers

    // notify tensor core warp that P is ready
    pipeline_s.consumer_release(pipeline_s_consumer_state);
    ++pipeline_s_consumer_state;

    pipeline_c.producer_acquire(pipeline_c_producer_state);

    ElementQK acc_scale = 0.5f * ::exp2f(scale * (old_row_max - row_max_safe));
    row_sum *= acc_scale;
    // row_sum = sum(reg_S)
    float2 local_row_sum_f32x2 = make_float2(row_sum, row_sum);
    float2 local_row_sum_1 = make_float2(0, 0);
    float2 local_row_sum_2 = make_float2(0, 0);
    float2 local_row_sum_3 = make_float2(0, 0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTMEM_LOADrS); i += 8) {
      // row_sum += tTMEM_LOADrS(i);
      float2 in = make_float2(tTMEM_LOADrS(i), tTMEM_LOADrS(i+1));
      cute::add(local_row_sum_f32x2, local_row_sum_f32x2, in);

      in = make_float2(tTMEM_LOADrS(i+2), tTMEM_LOADrS(i+2+1));
      cute::add(local_row_sum_1, local_row_sum_1, in);

      in = make_float2(tTMEM_LOADrS(i+4), tTMEM_LOADrS(i+4+1));
      cute::add(local_row_sum_2, local_row_sum_2, in);

      in = make_float2(tTMEM_LOADrS(i+6), tTMEM_LOADrS(i+6+1));
      cute::add(local_row_sum_3, local_row_sum_3, in);
    }

    cute::add(local_row_sum_f32x2, local_row_sum_f32x2, local_row_sum_1);
    cute::add(local_row_sum_2, local_row_sum_2, local_row_sum_3);
    cute::add(local_row_sum_f32x2, local_row_sum_f32x2, local_row_sum_2);
    float local_row_sum = local_row_sum_f32x2.x + local_row_sum_f32x2.y;

    row_sum = local_row_sum;

    if (final_call) {
      // re-acquire the S part in the final step
      pipeline_s.consumer_wait(pipeline_s_consumer_state);

      tTMEM_STOREVrS(kIdxFinalRowMax) = row_max;
      tTMEM_STOREVrS(kIdxFinalRowSum) = row_sum;
#if !defined(FLASH_MLA_BUILD_SM120) || defined(FLASH_MLA_SM120_USE_FP8)
      // SM100: Copy without flatten - partitioned tensors have compatible layouts
      copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);
#endif
      // SM120: tTMEM_STOREVrS and tTMEM_STOREVtS are same register tensor, no copy needed
    }
  }

  template<class Stage, class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE auto
  softmax(
      Stage stage,
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      PipelineS& pipeline_s, typename PipelineS::PipelineState& pipeline_s_consumer_state,
      PipelineC& pipeline_c, typename PipelineC::PipelineState& pipeline_c_producer_state,
      OrderBarrierSoftmax& order_s) {

    int mask_tile_count = Mask{}.get_unmasked_trip_count(blk_coord, TileShape{}, problem_shape);

    ElementQK row_max = -INFINITY;
    ElementQK row_sum = 0;

    Tensor cS_base = make_identity_tensor(select<0,1>(TileShapeQK{}));
    auto logical_offset = make_coord(
        get<0>(blk_coord) * get<0>(TileShape{}) + (stage % get<0>(ThreadShape{})) * get<0>(TileShapeQK{}),
        0 + (stage % get<1>(ThreadShape{})) * get<1>(TileShapeQK{})
    );
    Tensor cS = domain_offset(logical_offset, cS_base);

    pipeline_c.producer_acquire(pipeline_c_producer_state);

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1) {
      softmax_step<false /* need_apply_mask */>(
          row_max, row_sum, stage,
          (mask_tile_count == 1) &&
              (Mask{}.get_masked_trip_count(blk_coord, TileShape{}, problem_shape) == 0),
          blk_coord, cS, params, problem_shape,
          pipeline_s, pipeline_s_consumer_state,
          pipeline_c, pipeline_c_producer_state,
          order_s
      );

      cS.data() = cS.data() + E<1>{} * get<1>(ThreadShape{}) * get<1>(TileShapeQK{});
    }

    // Masked iterations
    mask_tile_count = Mask{}.get_masked_trip_count(blk_coord, TileShape{}, problem_shape);

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1) {
      softmax_step<true /* need_apply_mask */>(
          row_max, row_sum, stage, mask_tile_count == 1,
          blk_coord, cS, params, problem_shape,
          pipeline_s, pipeline_s_consumer_state,
          pipeline_c, pipeline_c_producer_state,
          order_s
      );

      cS.data() = cS.data() + E<1>{} * get<1>(ThreadShape{}) * get<1>(TileShapeQK{});
    }

    pipeline_c.producer_commit(pipeline_c_producer_state);
    ++pipeline_c_producer_state;

    pipeline_c.producer_acquire(pipeline_c_producer_state);
    // empty step to sync against pipe s
    pipeline_s.consumer_release(pipeline_s_consumer_state);
    ++pipeline_s_consumer_state;
  }

  // H1 approach for correction_epilogue: Create explicit ValID layouts for TMEM copy compatibility
  // Key insight: partition_fragment_C creates MMA layouts that don't match TMEM Copy_Traits ValID
  // This function loads from TMEM O buffer and copies to shared memory with scaling
  // Solution: Use TMEM copy's partition_D for both register AND SMEM tensors
  template<class Stage, class TensorO>
  CUTLASS_DEVICE auto
  correction_epilogue(
      float scale,
      Stage stage,
      TensorO const& sO_01) {

    using ElementOut = typename TensorO::value_type;

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    Tensor sO = sO_01(_,_,stage);

    // Tile size for register capacity management
    // For bf16: 32/2 = 16 cols per iteration
    const int kCorrectionTileSize = 32 / sizeof(ElementOut);

    // TileShapePV.M = TileShape.M / ThreadShape.M = 64 rows (both SM100 and SM120)
    constexpr int kTileM = decltype(size<0>(TileShapePV{}))::value;
    constexpr int kTileK = decltype(size<2>(TileShape{}))::value;
    constexpr int kLoopCount = kTileK / kCorrectionTileSize;

    // Epilogue load operation: 64 rows × kCorrectionTileSize cols
    // For bf16: 64 × 16 = 1024 elements, needs 32x atom (16dp32b32x)
    // For fp32: 64 × 8 = 512 elements, needs 16x atom (16dp32b16x)
    using EpilogueLoadOp = std::conditional_t<kCorrectionTileSize >= 16,
        SM100_TMEM_LOAD_16dp32b32x, SM100_TMEM_LOAD_16dp32b16x>;
    using EpilogueLayoutTmem = decltype(coalesce(upcast<sizeof_bits_v<ElementPV>>(typename Copy_Traits<EpilogueLoadOp>::ValID{})));
    using EpilogueLayoutReg = decltype(make_layout(shape(EpilogueLayoutTmem{})));

#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
    // SM120: Use register-based O buffer (no TMEM)
    // Create register tensor directly without TMEM
    auto tOrO_reg = make_tensor<ElementPV>(EpilogueLayoutReg{});
    Tensor tTMEM_LOADtO = tOrO_reg;  // Dummy for unified interface
    Tensor tTMEM_LOADrO = tOrO_reg;
#else
    // SM100: Determine TMEM base address based on stage
    uint32_t tmem_O_base = uint32_t(stage == _0{} ? TmemAllocation::O0 : TmemAllocation::O1);

    // Create TMEM tensor with explicit ValID layout (H1 approach)
    auto tOtO_tmem = make_tensor(make_tmem_ptr<uint32_t>(tmem_O_base), EpilogueLayoutTmem{});

    // Create register tensor with contiguous layout
    auto tOrO_reg = make_tensor<ElementPV>(EpilogueLayoutReg{});

    // Create TMEM copy with explicit layout tensor
    auto tiled_tmem_load = make_tmem_copy(EpilogueLoadOp{}, tOtO_tmem);
    auto thr_tmem_load = tiled_tmem_load.get_slice(thread_idx);

    // Partition the explicit layout tensors
    Tensor tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_tmem);
    Tensor tTMEM_LOADrO = thr_tmem_load.partition_D(tOrO_reg);
#endif

    // H2 approach: Create 2D SMEM view for each iteration inside the loop
    // The EpilogueLayoutTmem shape is (64, 16) - we create SMEM tiles matching this
    using SmemTileLayout = decltype(make_layout(make_shape(Int<kTileM>{}, Int<kCorrectionTileSize>{})));

    float2 scale_f32x2 = make_float2(scale, scale);

    // loop: TMEM_LOAD, FMUL2 scale, copy to SMEM
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kLoopCount; i++) {
#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
      // SM120: Use register tensor directly (no TMEM load)
      Tensor tTMrO = tOrO_reg;  // Already in registers
#else
      // SM100: Update TMEM pointer for this iteration slice
      Tensor tTMEM_LOADtO_i = tTMEM_LOADtO;
      tTMEM_LOADtO_i.data() = tTMEM_LOADtO_i.data().get() + uint32_t(i * kCorrectionTileSize);

      // Create register tensor for this iteration with proper shape
      Tensor tTMrO = make_tensor<ElementPV>(shape(tTMEM_LOADrO));

      // Load from TMEM to registers using H1 explicit layouts
      copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO);
#endif

#ifndef ONLY_SOFTMAX
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tTMrO); j += 2) {
        float2 in = make_float2(tTMrO(j), tTMrO(j+1));
        float2 out;
        cute::mul(out, scale_f32x2, in);
        tTMrO(j) = out.x;
        tTMrO(j+1) = out.y;
      }
#endif

      // Convert float to output type
      constexpr int N = 4 / sizeof(ElementOut);
      NumericArrayConverter<ElementOut, ElementPV, N> convert;

      // Create output register tensor with same element count
      Tensor tSMrO = make_tensor<ElementOut>(shape(tTMrO));

      auto tCs = recast<typename decltype(convert)::source_type>(tTMrO);
      auto tCd = recast<typename decltype(convert)::result_type>(tSMrO);

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tCs); j++) {
        tCd(j) = convert.convert(tCs(j));
      }

      // H2: Create 2D SMEM view for this iteration with shape (kTileM, kCorrectionTileSize)
      // This matches the EpilogueLayoutTmem shape of (64, 16)
      auto sO_tile = make_tensor(sO.data() + i * kTileM * kCorrectionTileSize, SmemTileLayout{});
#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
      // SM120: Direct copy without TMEM partitioning
      auto tSMsO_i = recast<uint32_t>(sO_tile);
#else
      // SM100: Use TMEM load partitioning
      Tensor tTMEM_LOADsO_i = thr_tmem_load.partition_D(sO_tile);
      auto tSMsO_i = recast<uint32_t>(tTMEM_LOADsO_i);
#endif
      auto tSMrO_i = recast<uint32_t>(tSMrO);

      // Copy to shared memory using vectorized copy
      copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, tSMrO_i, tSMsO_i);
    }

    cutlass::arch::fence_view_async_shared();
  }

  // H1 approach for correction_rescale: Create explicit ValID layouts for TMEM copy compatibility
  // Key insight: partition_fragment_C creates MMA layouts that don't match TMEM Copy_Traits ValID
  CUTLASS_DEVICE auto
  correction_rescale(
      float scale,
      uint32_t tmem_O) {

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    // Loop tile size for register capacity management
    const int kCorrectionTileSize = 16;

    // TileShapePV.M = TileShape.M / ThreadShape.M = 64 rows (both SM100 and SM120)
    constexpr int kTileM = decltype(size<0>(TileShapePV{}))::value;
    constexpr int kLoopCount = kTileM / kCorrectionTileSize;

    // Use explicit ValID layouts from CorrectionLoadOp/CorrectionStoreOp (16dp32b32x)
    // These are defined at class level using coalesce(upcast<32>(ValID)) pattern

#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
    // SM120: Use register-based O buffer (no TMEM)
    auto tOrO_reg = make_tensor<ElementPV>(CorrectionLayoutReg{});
    Tensor tTMEM_LOADtO = tOrO_reg;  // Dummy for unified interface
    Tensor tTMEM_LOADrO = tOrO_reg;
    Tensor tTMEM_STOREtO = tOrO_reg;  // Dummy for unified interface
    Tensor tTMEM_STORErO = tOrO_reg;
#else
    // SM100: Create TMEM tensor with explicit CorrectionLayoutTmem (matches atom's ValID)
    auto tOtO_tmem = make_tensor(make_tmem_ptr<uint32_t>(tmem_O), CorrectionLayoutTmem{});

    // Create register tensor with contiguous CorrectionLayoutReg
    auto tOrO_reg = make_tensor<ElementPV>(CorrectionLayoutReg{});

    // Create TMEM copy with explicit layout tensor
    auto tiled_tmem_load = make_tmem_copy(CorrectionLoadOp{}, tOtO_tmem);
    auto thr_tmem_load = tiled_tmem_load.get_slice(thread_idx);

    auto tiled_tmem_store = make_tmem_copy(CorrectionStoreOp{}, tOtO_tmem);
    auto thr_tmem_store = tiled_tmem_store.get_slice(thread_idx);

    // Partition the explicit layout tensors
    Tensor tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_tmem);
    Tensor tTMEM_LOADrO = thr_tmem_load.partition_D(tOrO_reg);
    Tensor tTMEM_STOREtO = thr_tmem_store.partition_D(tOtO_tmem);
    Tensor tTMEM_STORErO = thr_tmem_store.partition_S(tOrO_reg);
#endif

    float2 scale_f32x2 = make_float2(scale, scale);

    // Number of loop iterations based on head dimension
    int count = get<2>(TileShape{}) / kCorrectionTileSize;

    // Register tensor sized for loop tiling
    Tensor tTMrO = make_tensor<ElementPV>(make_shape(shape(tTMEM_LOADrO), Int<kLoopCount>{}));

    auto copy_in = [&](int i) {
#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
      // SM120: No TMEM copy, data already in registers
      (void)i;  // Suppress unused warning
#else
      // SM100: Load from TMEM
      Tensor tTMEM_LOADtO_i = tTMEM_LOADtO;
      tTMEM_LOADtO_i.data() = tTMEM_LOADtO_i.data().get() + uint32_t(i * kCorrectionTileSize);
      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
      copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO_i);
#endif
    };

    auto copy_out = [&](int i) {
#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
      // SM120: No TMEM copy, data stays in registers
      (void)i;  // Suppress unused warning
#else
      // SM100: Store to TMEM
      Tensor tTMEM_STOREtO_i = tTMEM_STOREtO;
      tTMEM_STOREtO_i.data() = tTMEM_STOREtO_i.data().get() + uint32_t(i * kCorrectionTileSize);
      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
      copy(tiled_tmem_store, tTMrO_i, tTMEM_STOREtO_i);
#endif
    };

    // sequence: LLMSLMSLMSS
    // loop: TMEM_LOAD, FMUL2 scale, TMEM_STORE
    copy_in(0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < count; i++) {
      if (i != count - 1) {
        copy_in(i+1);
      }

      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tTMrO_i); j += 2) {
        float2 in = make_float2(tTMrO_i(j), tTMrO_i(j+1));
        float2 out;
        cute::mul(out, scale_f32x2, in);
        tTMrO_i(j) = out.x;
        tTMrO_i(j+1) = out.y;
      }

      copy_out(i);
    }
  }

  template<
    class BlkCoord, class ProblemShape, class ParamsProblemShape,
    class TensorStorageEpi, class CollectiveEpilogue
  >
  CUTLASS_DEVICE auto
  correction(
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      ParamsProblemShape const& params_problem_shape,
      TensorStorageEpi& shared_storage_epi,
      PipelineC& pipeline_s0_c, typename PipelineC::PipelineState& pipeline_s0_c_consumer_state,
      PipelineC& pipeline_s1_c, typename PipelineC::PipelineState& pipeline_s1_c_consumer_state,
      PipelineO& pipeline_o, typename PipelineO::PipelineState& pipeline_o_consumer_state,
      PipelineE& pipeline_epi, typename PipelineE::PipelineState& pipeline_epi_producer_state,
      CollectiveEpilogue& epilogue) {

    int mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    Tensor tStS = partition_fragment_C(typename CollectiveMmaQK::TiledMma{}, select<0,1>(TileShapeQK{}));

    Tensor cS = make_identity_tensor(select<0,1>(TileShapeQK{}));
    Tensor tScS = typename CollectiveMmaQK::TiledMma{}.get_slice(0).partition_C(cS);

    auto tileN = size<1>(TileShapeQK{});

    auto [tStS_v_unused, tStS_P_unused, tStS_load, tScS_load] =
        make_softmax_tmem_views(_0{}, tStS, tScS);
    (void)tStS_v_unused;
    (void)tStS_P_unused;

#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
    // SM120: Use register-based tensors (no TMEM)
    // make_softmax_tmem_views already returns register tensors for SM120
    auto tTMEM_LOADVtS = tStS_load;
    auto tTMEM_LOADVcS = tScS_load;
    auto tTMEM_LOADVtS0 = tStS_load;
    auto tTMEM_LOADVtS1 = tStS_load;  // For SM120, both stages use same register
#else
    // SM100: TMEM_LOAD_V_OP = TMEM_LOAD_V = StatsLoadOp (all use 16dp atoms for SM120 small-tile).
    // Use partition_fragment_C tensors directly without coalescing - MMA-produced layouts match TMEM atoms.
    auto tiled_tmem_loadv = make_tmem_copy(TMEM_LOAD_V_OP{}, tStS_load);
    auto thr_tmem_loadv  = tiled_tmem_loadv.get_slice(thread_idx);

    // Partition TMEM tensor for source - preserves layout structure
    auto tTMEM_LOADVtS = thr_tmem_loadv.partition_S(tStS_load);
    auto tTMEM_LOADVcS = thr_tmem_loadv.partition_D(tScS_load);

    auto tTMEM_LOADVtS0 = tTMEM_LOADVtS;
    tTMEM_LOADVtS0.data() = tTMEM_LOADVtS0.data().get();
    auto tTMEM_LOADVtS1 = tTMEM_LOADVtS;
    tTMEM_LOADVtS1.data() = tTMEM_LOADVtS1.data().get() + uint32_t(TmemAllocation::V1) - uint32_t(TmemAllocation::V0);
#endif

    // ignore first signal from softmax as no correction is required
    pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);
    pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
    ++pipeline_s0_c_consumer_state;

    pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

    // handle the last iteration differently (i.e. tmem_load/stsm for epi)
    mask_tile_count -= 1;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1) {

      pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);

#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
      // SM120: Direct register access (no TMEM copy)
      Tensor tTMEM_LOADVrS = tTMEM_LOADVtS0;
#else
      // SM100: Destination tensor must match partition_D shape (register coords) for copy atom
      Tensor tTMEM_LOADVrS = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));

      // read row_wise new global max
      // Copy without flatten - partitioned tensors have compatible layouts
      copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS);
#endif

      // e^(scale * (old_max - new_max)
      float scale = ::exp2f(params.scale_softmax_log2 * (tTMEM_LOADVrS(kIdxOldRowMax) - tTMEM_LOADVrS(kIdxNewRowMax)));

      pipeline_o.consumer_wait(pipeline_o_consumer_state);

      correction_rescale(scale, uint32_t(TmemAllocation::O0));

      pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
      ++pipeline_s1_c_consumer_state;

#if !defined(FLASH_MLA_BUILD_SM120) || defined(FLASH_MLA_SM120_USE_FP8)
      // SM100: TMEM fence
      cutlass::arch::fence_view_async_tmem_store();
#endif

      pipeline_o.consumer_release(pipeline_o_consumer_state);
      ++pipeline_o_consumer_state;

      pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
      // SM120: Direct register access (no TMEM copy)
      tTMEM_LOADVrS = tTMEM_LOADVtS1;
#else
      // SM100: Copy without flatten - partitioned tensors have compatible layouts
      copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS);
#endif

      scale = ::exp2f(params.scale_softmax_log2 * (tTMEM_LOADVrS(kIdxOldRowMax) - tTMEM_LOADVrS(kIdxNewRowMax)));

      pipeline_o.consumer_wait(pipeline_o_consumer_state);

      correction_rescale(scale, uint32_t(TmemAllocation::O1));

      pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
      ++pipeline_s0_c_consumer_state;

#if !defined(FLASH_MLA_BUILD_SM120) || defined(FLASH_MLA_SM120_USE_FP8)
      // SM100: TMEM fence
      cutlass::arch::fence_view_async_tmem_store();
#endif

      pipeline_o.consumer_release(pipeline_o_consumer_state);
      ++pipeline_o_consumer_state;
    }

    pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
    ++pipeline_s1_c_consumer_state;

    // do the final correction to O1
    // better to somehow special-case it in the loop above
    // doesn't matter for non-persistent code, but if it were
    // persistent we do not want to release O too early

    pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);

    // read from V0
    // read row_sum and final row_max here
#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
    // SM120: Direct register access (no TMEM copy)
    Tensor tTMEM_LOADVrS = tTMEM_LOADVtS0;
#else
    // SM100: Destination tensor must match partition_D shape (register coords) for copy atom
    Tensor tTMEM_LOADVrS = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));
    // Copy without flatten - partitioned tensors have compatible layouts
    copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS);
#endif

    pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
    ++pipeline_s0_c_consumer_state;

    pipeline_o.consumer_wait(pipeline_o_consumer_state);
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);
    // store to epi smem

    // loop:
    //    TMEM_LOAD
    //    FMUL2 scale = 1 / global_sum * out_quant_scale
    //    F2FP
    //    store to smem
    Tensor sO = make_tensor(make_smem_ptr(shared_storage_epi.smem_o.data()), typename TensorStorageEpi::SmemLayoutO{});
    Tensor gLSE = make_tensor(make_gmem_ptr(epilogue.params.ptr_LSE), select<0,3>(problem_shape), epilogue.params.dLSE);
    
    correction_epilogue(params.scale_output / tTMEM_LOADVrS(kIdxFinalRowSum), _0{}, sO);

    if (epilogue.params.ptr_LSE != nullptr) {
      int row_idx = get<0>(tTMEM_LOADVcS(_0{})) + get<0>(TileShape{}) * get<0>(blk_coord);

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      ElementPV lse = cutlass::fast_log(tTMEM_LOADVrS(kIdxFinalRowSum)) + params.scale_softmax * tTMEM_LOADVrS(kIdxFinalRowMax);

      if (row_idx < get<0>(problem_shape)) {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

#if !defined(FLASH_MLA_BUILD_SM120) || defined(FLASH_MLA_SM120_USE_FP8)
    // SM100: TMEM fence
    cutlass::arch::fence_view_async_tmem_load();
#endif

    pipeline_o.consumer_release(pipeline_o_consumer_state);
    ++pipeline_o_consumer_state;

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;

    pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

#if defined(FLASH_MLA_BUILD_SM120) && !defined(FLASH_MLA_SM120_USE_FP8)
    // SM120: Direct register access (no TMEM copy)
    tTMEM_LOADVrS = tTMEM_LOADVtS1;
#else
    // SM100: load from V1 - Copy without flatten - partitioned tensors have compatible layouts
    copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS);
#endif

    pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
    ++pipeline_s1_c_consumer_state;

    pipeline_o.consumer_wait(pipeline_o_consumer_state);
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    correction_epilogue(params.scale_output / tTMEM_LOADVrS(kIdxFinalRowSum), _1{}, sO);

    if (epilogue.params.ptr_LSE != nullptr) {
      int row_idx = get<0>(tTMEM_LOADVcS(_0{})) + get<0>(TileShape{}) * get<0>(blk_coord) + get<0>(TileShapeQK{});

      ElementPV lse = cutlass::fast_log(tTMEM_LOADVrS(kIdxFinalRowSum)) + params.scale_softmax * tTMEM_LOADVrS(kIdxFinalRowMax);

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      if (row_idx < get<0>(problem_shape)) {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

#if !defined(FLASH_MLA_BUILD_SM120) || defined(FLASH_MLA_SM120_USE_FP8)
    // SM100: TMEM fence
    cutlass::arch::fence_view_async_tmem_load();
#endif

    pipeline_o.consumer_release(pipeline_o_consumer_state);
    ++pipeline_o_consumer_state;

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;
  }


  template<
    class BlkCoord, class ProblemShape, class ParamsProblemShape,
    class TensorStorageEpi, class CollectiveEpilogue
  >
  CUTLASS_DEVICE auto
  correction_empty(
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      ParamsProblemShape const& params_problem_shape,
      TensorStorageEpi& shared_storage_epi,
      PipelineE& pipeline_epi, typename PipelineE::PipelineState& pipeline_epi_producer_state,
      CollectiveEpilogue& epilogue) {

    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    Tensor sO = make_tensor(make_smem_ptr(shared_storage_epi.smem_o.data()), typename TensorStorageEpi::SmemLayoutO{});
    Tensor gLSE = make_tensor(make_gmem_ptr(epilogue.params.ptr_LSE), select<0,3>(problem_shape), epilogue.params.dLSE);
    float lse = -INFINITY;
    int thread_idx = threadIdx.x % (4 * NumThreadsPerWarp);

#if 1

    using ElementOut = typename CollectiveEpilogue::ElementOut;
    auto tiled_copy = make_cotiled_copy(
        Copy_Atom<UniversalCopy<uint32_t>, ElementOut>{},
        make_ordered_layout(make_shape(_128{}, Int<sizeof(uint32_t) / sizeof(ElementOut)>{}), Step<_1, _0>{}),
        sO.layout());

    auto thr_copy = tiled_copy.get_slice(thread_idx);
    auto tOgO = thr_copy.partition_D(sO);
    // Create rank-1 register tensor (copy atom expects rank-1)
    auto tOgO_slice = tOgO(_,_,_,_0{});
    // Register tensor must match destination shape for copy atom
    auto tOrO = make_tensor<ElementOut>(shape(tOgO_slice));
    clear(tOrO);

    // Shared memory copy with matching shapes
    copy(tiled_copy, tOrO, tOgO_slice);
#endif
    
    if (epilogue.params.ptr_LSE != nullptr) {
      int row_idx = thread_idx + get<0>(TileShape{}) * get<0>(blk_coord);

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      if (row_idx < get<0>(problem_shape)) {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;

    // Shared memory copy: use tensors as-is
    copy(tiled_copy, tOrO, tOgO(_,_,_,_1{}));
    cutlass::arch::fence_view_async_shared();
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    if (epilogue.params.ptr_LSE != nullptr) {
      int row_idx = thread_idx + get<0>(TileShape{}) * get<0>(blk_coord) + get<0>(TileShapeQK{});

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      if (row_idx < get<0>(problem_shape)) {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    cutlass::arch::fence_view_async_shared();
    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;
  }

};

}  // namespace cutlass::fmha::collective
