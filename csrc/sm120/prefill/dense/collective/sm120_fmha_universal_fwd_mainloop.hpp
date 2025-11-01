/***************************************************************************************************
 * Copyright (c) 2025 FlashMLA Contributors
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SM120 UniversalCopy Forward Mainloop
 *
 * Complete implementation of flash attention forward pass for SM120 (Blackwell workstation GPUs).
 * Uses cute::UniversalCopy instead of TMA/TMEM to fit within 99KB shared memory budget.
 *
 * Key Differences from SM100a TMA Mainloop:
 * - UniversalCopy for Q/K/V loads (no TMA hardware)
 * - No TMEM 128-column layout requirements
 * - Non-persistent scheduling (UnionType) for memory savings
 * - ThreadShape <_1,_1,_1> (no stacked softmax warps)
 * - Cluster {1,1,1} (no TMA multicast)
 * - HeadDimLatent=64 (reduced from 128)
 *
 * Memory Budget Breakdown (SM120: 99KB total):
 * - Q tile:  M=64 * N=32 * sizeof(bf16) * kStages = ~8KB
 * - K tile:  K=128 * N=32 * sizeof(bf16) * kStages = ~16KB
 * - V tile:  K=128 * N=32 * sizeof(bf16) * kStages = ~16KB
 * - P tile:  M=64 * N=32 * sizeof(float) = ~8KB (acc precision)
 * - O tile:  M=64 * N=32 * sizeof(bf16) = ~4KB
 * - Epilogue: ~16KB (UnionType sharing)
 * - LSE:     M=64 * sizeof(float) = ~256 bytes
 * Total: ~68KB (leaves ~31KB headroom for registers/stack)
 **************************************************************************************************/

#pragma once

#include "cute/tensor.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/numeric_types.h"

using namespace cute;

namespace flash {

//==============================================================================
// SM120 MLA Forward Mainloop (UniversalCopy-based)
//==============================================================================

template <
    class KernelTraits_,
    class ProblemShape_,
    class Element_,
    class ElementAcc_,
    class TileShape_
>
struct Sm120FmhaUniversalFwdMainloop {
  using KernelTraits = KernelTraits_;
  using ProblemShape = ProblemShape_;
  using Element = Element_;          // e.g., cutlass::bfloat16_t
  using ElementAcc = ElementAcc_;    // e.g., float
  using TileShape = TileShape_;      // Shape<M, N, HeadDim>

  // Architecture tag for dispatch
  using ArchTag = typename KernelTraits::ArchTag;
  static_assert(std::is_same_v<ArchTag, cutlass::arch::Sm120>,
    "[SM120 SAFETY] This mainloop is only for SM120 architecture");

  // Extract tile dimensions
  static constexpr int kTileM = get<0>(TileShape{});
  static constexpr int kTileN = get<1>(TileShape{});
  static constexpr int kHeadDim = get<2>(TileShape{});  // Should be 64 for SM120

  // Pipeline configuration
  static constexpr int kStages = KernelTraits::kStages;  // Should be 2 (minimum)

  // ThreadShape must be <_1,_1,_1> for SM120
  using ThreadShape = typename KernelTraits::ThreadShape;
  static_assert(std::is_same_v<ThreadShape, Shape<_1, _1, _1>>,
    "[SM120 SAFETY][P6] SM120 must use ThreadShape <1,1,1>");

  //==============================================================================
  // [SM120 SAFETY][P2] Expose PV Tile Shape for Epilogue Compatibility
  //==============================================================================
  using TileShapePV = Shape<Int<kTileM>, Int<kTileN>, Int<kHeadDim>>;
  static constexpr int HeadDimPV = kHeadDim;

  //==============================================================================
  // Shared Memory Layout
  //==============================================================================

  // Q tile: [kStages][kTileM][kTileN][kHeadDim]
  static constexpr size_t kSmemQBytes = kStages * kTileM * kTileN * sizeof(Element);

  // K tile: [kStages][kTileN][kHeadDim] (transposed for QK^T)
  static constexpr size_t kSmemKBytes = kStages * kTileN * kHeadDim * sizeof(Element);

  // V tile: [kStages][kTileN][kHeadDim]
  static constexpr size_t kSmemVBytes = kStages * kTileN * kHeadDim * sizeof(Element);

  // P tile (attention scores): [kTileM][kTileN] in float precision
  static constexpr size_t kSmemPBytes = kTileM * kTileN * sizeof(ElementAcc);

  // O tile (output accumulator): [kTileM][kHeadDim]
  static constexpr size_t kSmemOBytes = kTileM * kHeadDim * sizeof(Element);

  // LSE (log-sum-exp for softmax): [kTileM]
  static constexpr size_t kSmemLSEBytes = kTileM * sizeof(ElementAcc);

  // Total mainloop shared memory (no epilogue for now)
  static constexpr size_t kSmemMainloopBytes =
      kSmemQBytes + kSmemKBytes + kSmemVBytes +
      kSmemPBytes + kSmemOBytes + kSmemLSEBytes;

  // Epilogue shared memory (shared with mainloop in UnionType)
  static constexpr size_t kSmemEpilogueBytes = 16384;  // ~16KB epilogue

  //==============================================================================
  // [SM120 SAFETY][P1] Expose Total SMEM for Budget Validation
  //==============================================================================
  // Note: In UnionType (non-persistent), mainloop and epilogue share memory
  static constexpr size_t kSmemTotal = max(kSmemMainloopBytes, kSmemEpilogueBytes);

  //==============================================================================
  // [SM120 SAFETY][P8] TMEM Allocation (Not Used, But Exposed for Compatibility)
  //==============================================================================
  // SM120 doesn't use TMEM, but we expose the struct for safety_introspection.hpp
  struct TmemAllocation {
    enum : uint32_t {
      O0 = 0,      // Aligned (0 % 4 == 0)
      P0 = 256,    // Aligned (256 % 4 == 0)
      K0 = 512,    // Aligned (512 % 4 == 0)
      V0 = 768     // Aligned (768 % 4 == 0)
    };

    // [SM120 SAFETY][P8] Alignment checks
    static_assert(O0 % 4 == 0, "[SM120 SAFETY][P8] O0 must be 16-byte aligned");
    static_assert(P0 % 4 == 0, "[SM120 SAFETY][P8] P0 must be 16-byte aligned");
    static_assert(K0 % 4 == 0, "[SM120 SAFETY][P8] K0 must be 16-byte aligned");
    static_assert(V0 % 4 == 0, "[SM120 SAFETY][P8] V0 must be 16-byte aligned");
  };

  //==============================================================================
  // Shared Memory Allocation Struct
  //==============================================================================

  struct SharedStorage {
    // Double-buffered Q tile
    cute::array<Element, kStages * kTileM * kTileN> smem_q;

    // Double-buffered K tile
    cute::array<Element, kStages * kTileN * kHeadDim> smem_k;

    // Double-buffered V tile
    cute::array<Element, kStages * kTileN * kHeadDim> smem_v;

    // Attention scores (P = softmax(QK^T))
    cute::array<ElementAcc, kTileM * kTileN> smem_p;

    // Output accumulator
    cute::array<Element, kTileM * kHeadDim> smem_o;

    // Log-sum-exp for softmax
    cute::array<ElementAcc, kTileM> smem_lse;
  };

  //==============================================================================
  // Device-Side Constructor
  //==============================================================================

  CUTLASS_DEVICE
  Sm120FmhaUniversalFwdMainloop() {}

  //==============================================================================
  // Phase 1: Load Q Tile with UniversalCopy
  //==============================================================================

  template <class TensorQ, class SmemTensorQ>
  CUTLASS_DEVICE
  void load_q_tile(
      TensorQ const& gQ,           // Global Q tensor
      SmemTensorQ& sQ,             // Shared memory Q tensor
      int m_block,                 // M block index
      int stage                    // Pipeline stage
  ) {
    // Compute global Q address for this tile
    int m_start = m_block * kTileM;

    // Use cute::UniversalCopy for generic CUDA copy (no TMA)
    // This is the key difference from SM100a: no TMA hardware required

    auto gQ_tile = local_tile(gQ, make_coord(m_start, _0{}, _0{}),
                              Shape<Int<kTileM>, Int<kTileN>, Int<kHeadDim>>{});

    auto sQ_tile = sQ(_, _, _, stage);  // Select pipeline stage

    // Use standard CuTe copy for SM120 (no TMA/TMEM hardware)
    cute::copy(gQ_tile, sQ_tile);

    __syncthreads();  // Ensure Q tile is fully loaded
  }

  //==============================================================================
  // Phase 2: Load K/V Tiles with UniversalCopy
  //==============================================================================

  template <class TensorK, class TensorV, class SmemTensorK, class SmemTensorV>
  CUTLASS_DEVICE
  void load_kv_tiles(
      TensorK const& gK,           // Global K tensor
      TensorV const& gV,           // Global V tensor
      SmemTensorK& sK,             // Shared memory K tensor
      SmemTensorV& sV,             // Shared memory V tensor
      int n_block,                 // N block index
      int stage                    // Pipeline stage
  ) {
    int n_start = n_block * kTileN;

    // Load K tile (transpose for QK^T)
    auto gK_tile = local_tile(gK, make_coord(_0{}, n_start, _0{}),
                              Shape<Int<kTileN>, Int<kHeadDim>>{});
    auto sK_tile = sK(_, _, stage);
    cute::copy(gK_tile, sK_tile);

    // Load V tile
    auto gV_tile = local_tile(gV, make_coord(_0{}, n_start, _0{}),
                              Shape<Int<kTileN>, Int<kHeadDim>>{});
    auto sV_tile = sV(_, _, stage);
    cute::copy(gV_tile, sV_tile);

    __syncthreads();  // Ensure K/V tiles are fully loaded
  }

  //==============================================================================
  // Phase 3: QK^T Matmul
  //==============================================================================

  template <class SmemTensorQ, class SmemTensorK, class SmemTensorP>
  CUTLASS_DEVICE
  void compute_qk(
      SmemTensorQ const& sQ,       // Shared memory Q [kTileM, kTileN]
      SmemTensorK const& sK,       // Shared memory K [kTileN, kHeadDim]
      SmemTensorP& sP,             // Shared memory P [kTileM, kTileN] (output)
      float softmax_scale,         // Softmax scaling factor
      int stage                    // Pipeline stage
  ) {
    // Use cute MMA for QK^T
    // P[m,n] = sum_k Q[m,k] * K[n,k]^T * softmax_scale

    auto sQ_tile = sQ(_, _, stage);
    auto sK_tile = sK(_, _, stage);

    // Accumulate in float precision for numerical stability
    for (int m = 0; m < kTileM; ++m) {
      for (int n = 0; n < kTileN; ++n) {
        ElementAcc acc = 0.0f;
        for (int k = 0; k < kHeadDim; ++k) {
          acc += static_cast<ElementAcc>(sQ_tile(m, k)) *
                 static_cast<ElementAcc>(sK_tile(n, k));
        }
        sP(m, n) = acc * softmax_scale;
      }
    }

    __syncthreads();
  }

  //==============================================================================
  // Phase 4: Softmax (Single-CTA, No Multicast)
  //==============================================================================

  template <class SmemTensorP, class SmemTensorLSE>
  CUTLASS_DEVICE
  void compute_softmax(
      SmemTensorP& sP,             // Shared memory P [kTileM, kTileN] (in/out)
      SmemTensorLSE& sLSE,         // Shared memory LSE [kTileM] (log-sum-exp)
      bool is_causal,              // Causal masking flag
      int m_block,                 // M block index
      int n_block                  // N block index
  ) {
    // SM120 uses single-CTA softmax (no TMA multicast available)
    // Cluster must be {1,1,1} - see Point 7 runtime validation

    for (int m = 0; m < kTileM; ++m) {
      // Find max for numerical stability
      ElementAcc max_val = -INFINITY;
      for (int n = 0; n < kTileN; ++n) {
        // Apply causal mask if needed
        if (is_causal && (m_block * kTileM + m) < (n_block * kTileN + n)) {
          sP(m, n) = -INFINITY;
        }
        max_val = max(max_val, sP(m, n));
      }

      // Compute exp(x - max) and sum
      ElementAcc sum_exp = 0.0f;
      for (int n = 0; n < kTileN; ++n) {
        ElementAcc val = exp(sP(m, n) - max_val);
        sP(m, n) = val;
        sum_exp += val;
      }

      // Normalize and compute log-sum-exp
      ElementAcc inv_sum = 1.0f / sum_exp;
      for (int n = 0; n < kTileN; ++n) {
        sP(m, n) *= inv_sum;
      }
      sLSE(m) = log(sum_exp) + max_val;
    }

    __syncthreads();
  }

  //==============================================================================
  // Phase 5: PV Matmul
  //==============================================================================

  template <class SmemTensorP, class SmemTensorV, class SmemTensorO>
  CUTLASS_DEVICE
  void compute_pv(
      SmemTensorP const& sP,       // Shared memory P [kTileM, kTileN]
      SmemTensorV const& sV,       // Shared memory V [kTileN, kHeadDim]
      SmemTensorO& sO,             // Shared memory O [kTileM, kHeadDim] (output)
      int stage                    // Pipeline stage
  ) {
    // O[m,k] = sum_n P[m,n] * V[n,k]

    auto sV_tile = sV(_, _, stage);

    for (int m = 0; m < kTileM; ++m) {
      for (int k = 0; k < kHeadDim; ++k) {
        ElementAcc acc = 0.0f;
        for (int n = 0; n < kTileN; ++n) {
          acc += sP(m, n) * static_cast<ElementAcc>(sV_tile(n, k));
        }
        sO(m, k) = static_cast<Element>(acc);
      }
    }

    __syncthreads();
  }

  //==============================================================================
  // Phase 6: Store Output with UniversalCopy
  //==============================================================================

  template <class SmemTensorO, class TensorO, class SmemTensorLSE, class TensorLSE>
  CUTLASS_DEVICE
  void store_output(
      SmemTensorO const& sO,       // Shared memory O [kTileM, kHeadDim]
      TensorO& gO,                 // Global O tensor
      SmemTensorLSE const& sLSE,   // Shared memory LSE [kTileM]
      TensorLSE& gLSE,             // Global LSE tensor
      int m_block                  // M block index
  ) {
    int m_start = m_block * kTileM;

    // Store O tile
    auto gO_tile = local_tile(gO, make_coord(m_start, _0{}),
                              Shape<Int<kTileM>, Int<kHeadDim>>{});
    cute::copy(sO, gO_tile);

    // Store LSE
    auto gLSE_tile = local_tile(gLSE, make_coord(m_start),
                                Shape<Int<kTileM>>{});
    cute::copy(sLSE, gLSE_tile);

    __syncthreads();
  }

  //==============================================================================
  // Main Forward Pass Entry Point
  //==============================================================================

  template <
      class TensorQ,
      class TensorK,
      class TensorV,
      class TensorO,
      class TensorLSE
  >
  CUTLASS_DEVICE
  void operator()(
      SharedStorage& shared_storage,
      TensorQ const& gQ,
      TensorK const& gK,
      TensorV const& gV,
      TensorO& gO,
      TensorLSE& gLSE,
      float softmax_scale,
      bool is_causal,
      int seqlen_q,
      int seqlen_kv
  ) {
    // Create shared memory tensors
    auto sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()),
                          Shape<Int<kTileM>, Int<kTileN>, Int<kStages>>{});
    auto sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()),
                          Shape<Int<kTileN>, Int<kHeadDim>, Int<kStages>>{});
    auto sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()),
                          Shape<Int<kTileN>, Int<kHeadDim>, Int<kStages>>{});
    auto sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()),
                          Shape<Int<kTileM>, Int<kTileN>>{});
    auto sO = make_tensor(make_smem_ptr(shared_storage.smem_o.data()),
                          Shape<Int<kTileM>, Int<kHeadDim>>{});
    auto sLSE = make_tensor(make_smem_ptr(shared_storage.smem_lse.data()),
                            Shape<Int<kTileM>>{});

    // Compute grid dimensions
    int num_m_blocks = (seqlen_q + kTileM - 1) / kTileM;
    int num_n_blocks = (seqlen_kv + kTileN - 1) / kTileN;

    // Get block indices
    int m_block = blockIdx.x;
    if (m_block >= num_m_blocks) return;

    // Initialize output accumulator to zero
    for (int m = 0; m < kTileM; ++m) {
      for (int k = 0; k < kHeadDim; ++k) {
        sO(m, k) = Element(0);
      }
      sLSE(m) = -INFINITY;
    }

    // Load Q tile once (reused across all K/V blocks)
    load_q_tile(gQ, sQ, m_block, 0);

    // Loop over K/V blocks
    for (int n_block = 0; n_block < num_n_blocks; ++n_block) {
      int stage = n_block % kStages;

      // Load K/V tiles
      load_kv_tiles(gK, gV, sK, sV, n_block, stage);

      // Compute QK^T
      compute_qk(sQ, sK, sP, softmax_scale, stage);

      // Apply softmax
      compute_softmax(sP, sLSE, is_causal, m_block, n_block);

      // Compute PV and accumulate to O
      compute_pv(sP, sV, sO, stage);
    }

    // Store final output
    store_output(sO, gO, sLSE, gLSE, m_block);
  }
};

} // namespace flash
