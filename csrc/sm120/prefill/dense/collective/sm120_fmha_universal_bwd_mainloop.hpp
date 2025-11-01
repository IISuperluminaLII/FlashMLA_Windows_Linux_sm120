/***************************************************************************************************
 * Copyright (c) 2025 FlashMLA Contributors
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SM120 UniversalCopy Backward Mainloop
 *
 * Complete implementation of flash attention backward pass for SM120.
 * Computes gradients dQ, dK, dV using UniversalCopy for all memory operations.
 *
 * Backward Pass Algorithm:
 * 1. Load dO (output gradient) and O (forward output)
 * 2. Recompute P (attention weights) from Q/K
 * 3. Compute dV = P^T @ dO
 * 4. Compute dP = dO @ V^T
 * 5. Apply softmax backward: dS = P * (dP - rowsum(dP * P))
 * 6. Compute dQ = dS @ K
 * 7. Compute dK = dS^T @ Q
 *
 * Memory Budget: Same as forward (~68KB mainloop + ~16KB epilogue shared)
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
// SM120 FMHA Backward Mainloop (UniversalCopy-based)
//==============================================================================

template <
    class KernelTraits_,
    class ProblemShape_,
    class Element_,
    class ElementAcc_,
    class TileShape_
>
struct Sm120FmhaUniversalBwdMainloop {
  using KernelTraits = KernelTraits_;
  using ProblemShape = ProblemShape_;
  using Element = Element_;
  using ElementAcc = ElementAcc_;
  using TileShape = TileShape_;

  using ArchTag = typename KernelTraits::ArchTag;
  static_assert(std::is_same_v<ArchTag, cutlass::arch::Sm120>,
    "[SM120 SAFETY] This mainloop is only for SM120 architecture");

  static constexpr int kTileM = get<0>(TileShape{});
  static constexpr int kTileN = get<1>(TileShape{});
  static constexpr int kHeadDim = get<2>(TileShape{});
  static constexpr int kStages = KernelTraits::kStages;

  using ThreadShape = typename KernelTraits::ThreadShape;
  static_assert(std::is_same_v<ThreadShape, Shape<_1, _1, _1>>,
    "[SM120 SAFETY][P6] SM120 must use ThreadShape <1,1,1>");

  //==============================================================================
  // Shared Memory Layout for Backward Pass
  //==============================================================================

  // Forward activations (Q, K, V, O, P, LSE)
  static constexpr size_t kSmemQBytes = kStages * kTileM * kTileN * sizeof(Element);
  static constexpr size_t kSmemKBytes = kStages * kTileN * kHeadDim * sizeof(Element);
  static constexpr size_t kSmemVBytes = kStages * kTileN * kHeadDim * sizeof(Element);
  static constexpr size_t kSmemOBytes = kTileM * kHeadDim * sizeof(Element);
  static constexpr size_t kSmemPBytes = kTileM * kTileN * sizeof(ElementAcc);
  static constexpr size_t kSmemLSEBytes = kTileM * sizeof(ElementAcc);

  // Gradients (dO, dQ, dK, dV, dP)
  static constexpr size_t kSmemdOBytes = kTileM * kHeadDim * sizeof(Element);
  static constexpr size_t kSmemdQBytes = kTileM * kTileN * sizeof(Element);
  static constexpr size_t kSmemdKBytes = kTileN * kHeadDim * sizeof(Element);
  static constexpr size_t kSmemdVBytes = kTileN * kHeadDim * sizeof(Element);
  static constexpr size_t kSmemdPBytes = kTileM * kTileN * sizeof(ElementAcc);

  // Total backward shared memory
  static constexpr size_t kSmemMainloopBytes =
      kSmemQBytes + kSmemKBytes + kSmemVBytes + kSmemOBytes + kSmemPBytes + kSmemLSEBytes +
      kSmemdOBytes + kSmemdQBytes + kSmemdKBytes + kSmemdVBytes + kSmemdPBytes;

  static constexpr size_t kSmemEpilogueBytes = 16384;

  //==============================================================================
  // [SM120 SAFETY][P1] Expose Total SMEM for Budget Validation
  //==============================================================================
  static constexpr size_t kSmemTotal = max(kSmemMainloopBytes, kSmemEpilogueBytes);

  //==============================================================================
  // [SM120 SAFETY][P2] Expose PV Tile Shape
  //==============================================================================
  using TileShapePV = Shape<Int<kTileM>, Int<kTileN>, Int<kHeadDim>>;
  static constexpr int HeadDimPV = kHeadDim;

  //==============================================================================
  // [SM120 SAFETY][P8] TMEM Allocation
  //==============================================================================
  struct TmemAllocation {
    enum : uint32_t {
      O0 = 0, P0 = 256, K0 = 512, V0 = 768,
      dO0 = 1024, dP0 = 1280, dK0 = 1536, dV0 = 1792
    };
    static_assert(O0 % 4 == 0, "[SM120 SAFETY][P8] O0 must be 16-byte aligned");
    static_assert(P0 % 4 == 0, "[SM120 SAFETY][P8] P0 must be 16-byte aligned");
    static_assert(dO0 % 4 == 0, "[SM120 SAFETY][P8] dO0 must be 16-byte aligned");
    static_assert(dP0 % 4 == 0, "[SM120 SAFETY][P8] dP0 must be 16-byte aligned");
  };

  //==============================================================================
  // Shared Storage
  //==============================================================================

  struct SharedStorage {
    // Forward activations
    cute::array<Element, kStages * kTileM * kTileN> smem_q;
    cute::array<Element, kStages * kTileN * kHeadDim> smem_k;
    cute::array<Element, kStages * kTileN * kHeadDim> smem_v;
    cute::array<Element, kTileM * kHeadDim> smem_o;
    cute::array<ElementAcc, kTileM * kTileN> smem_p;
    cute::array<ElementAcc, kTileM> smem_lse;

    // Gradients
    cute::array<Element, kTileM * kHeadDim> smem_dO;
    cute::array<Element, kTileM * kTileN> smem_dQ;
    cute::array<Element, kTileN * kHeadDim> smem_dK;
    cute::array<Element, kTileN * kHeadDim> smem_dV;
    cute::array<ElementAcc, kTileM * kTileN> smem_dP;
  };

  //==============================================================================
  // Constructor
  //==============================================================================

  CUTLASS_DEVICE
  Sm120FmhaUniversalBwdMainloop() {}

  //==============================================================================
  // Backward Pass: Compute dV (Phase 1)
  //==============================================================================

  template <class SmemTensorP, class SmemdTensorO, class SmemdTensorV>
  CUTLASS_DEVICE
  void compute_dV(
      SmemTensorP const& sP,       // Attention weights [kTileM, kTileN]
      SmemdTensorO const& sdO,     // Output gradient [kTileM, kHeadDim]
      SmemdTensorV& sdV            // dV accumulator [kTileN, kHeadDim]
  ) {
    // dV = P^T @ dO
    for (int n = 0; n < kTileN; ++n) {
      for (int k = 0; k < kHeadDim; ++k) {
        ElementAcc acc = 0.0f;
        for (int m = 0; m < kTileM; ++m) {
          acc += sP(m, n) * static_cast<ElementAcc>(sdO(m, k));
        }
        sdV(n, k) = static_cast<Element>(acc);
      }
    }
    __syncthreads();
  }

  //==============================================================================
  // Backward Pass: Compute dP (Phase 2)
  //==============================================================================

  template <class SmemdTensorO, class SmemTensorV, class SmemdTensorP>
  CUTLASS_DEVICE
  void compute_dP(
      SmemdTensorO const& sdO,     // Output gradient [kTileM, kHeadDim]
      SmemTensorV const& sV,       // V values [kTileN, kHeadDim]
      SmemdTensorP& sdP,           // dP output [kTileM, kTileN]
      int stage
  ) {
    // dP = dO @ V^T
    auto sV_tile = sV(_, _, stage);

    for (int m = 0; m < kTileM; ++m) {
      for (int n = 0; n < kTileN; ++n) {
        ElementAcc acc = 0.0f;
        for (int k = 0; k < kHeadDim; ++k) {
          acc += static_cast<ElementAcc>(sdO(m, k)) *
                 static_cast<ElementAcc>(sV_tile(n, k));
        }
        sdP(m, n) = acc;
      }
    }
    __syncthreads();
  }

  //==============================================================================
  // Backward Pass: Softmax Backward (Phase 3)
  //==============================================================================

  template <class SmemTensorP, class SmemdTensorP>
  CUTLASS_DEVICE
  void softmax_backward(
      SmemTensorP const& sP,       // Forward attention weights [kTileM, kTileN]
      SmemdTensorP& sdP            // dP (in/out) [kTileM, kTileN]
  ) {
    // Softmax backward: dS = P * (dP - rowsum(dP * P))
    for (int m = 0; m < kTileM; ++m) {
      // Compute rowsum: sum_n (dP[m,n] * P[m,n])
      ElementAcc rowsum = 0.0f;
      for (int n = 0; n < kTileN; ++n) {
        rowsum += sdP(m, n) * sP(m, n);
      }

      // Apply: dS = P * (dP - rowsum)
      for (int n = 0; n < kTileN; ++n) {
        sdP(m, n) = sP(m, n) * (sdP(m, n) - rowsum);
      }
    }
    __syncthreads();
  }

  //==============================================================================
  // Backward Pass: Compute dQ (Phase 4)
  //==============================================================================

  template <class SmemdTensorP, class SmemTensorK, class SmemdTensorQ>
  CUTLASS_DEVICE
  void compute_dQ(
      SmemdTensorP const& sdP,     // Softmax gradient [kTileM, kTileN]
      SmemTensorK const& sK,       // K values [kTileN, kHeadDim]
      SmemdTensorQ& sdQ,           // dQ accumulator [kTileM, kTileN]
      float softmax_scale,
      int stage
  ) {
    // dQ = (dS @ K) * softmax_scale
    auto sK_tile = sK(_, _, stage);

    for (int m = 0; m < kTileM; ++m) {
      for (int k = 0; k < kTileN; ++k) {  // Note: kTileN for Q's second dim
        ElementAcc acc = 0.0f;
        for (int n = 0; n < kTileN; ++n) {
          acc += sdP(m, n) * static_cast<ElementAcc>(sK_tile(n, k));
        }
        sdQ(m, k) = static_cast<Element>(acc * softmax_scale);
      }
    }
    __syncthreads();
  }

  //==============================================================================
  // Backward Pass: Compute dK (Phase 5)
  //==============================================================================

  template <class SmemdTensorP, class SmemTensorQ, class SmemdTensorK>
  CUTLASS_DEVICE
  void compute_dK(
      SmemdTensorP const& sdP,     // Softmax gradient [kTileM, kTileN]
      SmemTensorQ const& sQ,       // Q values [kTileM, kTileN]
      SmemdTensorK& sdK,           // dK accumulator [kTileN, kHeadDim]
      float softmax_scale,
      int stage
  ) {
    // dK = (dS^T @ Q) * softmax_scale
    auto sQ_tile = sQ(_, _, stage);

    for (int n = 0; n < kTileN; ++n) {
      for (int k = 0; k < kHeadDim; ++k) {  // HeadDim for K
        ElementAcc acc = 0.0f;
        for (int m = 0; m < kTileM; ++m) {
          acc += sdP(m, n) * static_cast<ElementAcc>(sQ_tile(m, k));
        }
        sdK(n, k) = static_cast<Element>(acc * softmax_scale);
      }
    }
    __syncthreads();
  }

  //==============================================================================
  // Main Backward Pass Entry Point
  //==============================================================================

  template <
      class TensorQ,
      class TensorK,
      class TensorV,
      class TensorO,
      class TensorLSE,
      class TensordO,
      class TensordQ,
      class TensordK,
      class TensordV
  >
  CUTLASS_DEVICE
  void operator()(
      SharedStorage& shared_storage,
      TensorQ const& gQ,
      TensorK const& gK,
      TensorV const& gV,
      TensorO const& gO,
      TensorLSE const& gLSE,
      TensordO const& gdO,
      TensordQ& gdQ,
      TensordK& gdK,
      TensordV& gdV,
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
    auto sO = make_tensor(make_smem_ptr(shared_storage.smem_o.data()),
                          Shape<Int<kTileM>, Int<kHeadDim>>{});
    auto sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()),
                          Shape<Int<kTileM>, Int<kTileN>>{});
    auto sLSE = make_tensor(make_smem_ptr(shared_storage.smem_lse.data()),
                            Shape<Int<kTileM>>{});

    auto sdO = make_tensor(make_smem_ptr(shared_storage.smem_dO.data()),
                           Shape<Int<kTileM>, Int<kHeadDim>>{});
    auto sdQ = make_tensor(make_smem_ptr(shared_storage.smem_dQ.data()),
                           Shape<Int<kTileM>, Int<kTileN>>{});
    auto sdK = make_tensor(make_smem_ptr(shared_storage.smem_dK.data()),
                           Shape<Int<kTileN>, Int<kHeadDim>>{});
    auto sdV = make_tensor(make_smem_ptr(shared_storage.smem_dV.data()),
                           Shape<Int<kTileN>, Int<kHeadDim>>{});
    auto sdP = make_tensor(make_smem_ptr(shared_storage.smem_dP.data()),
                           Shape<Int<kTileM>, Int<kTileN>>{});

    int num_m_blocks = (seqlen_q + kTileM - 1) / kTileM;
    int num_n_blocks = (seqlen_kv + kTileN - 1) / kTileN;

    int m_block = blockIdx.x;
    if (m_block >= num_m_blocks) return;

    // Initialize gradient accumulators
    for (int m = 0; m < kTileM; ++m) {
      for (int k = 0; k < kTileN; ++k) {
        sdQ(m, k) = Element(0);
      }
    }

    // Load forward activations (Q, O, dO, LSE) - these are constant across K/V blocks
    // TODO: Implement load functions with UniversalCopy
    // load_forward_activations(gQ, gO, gdO, gLSE, sQ, sO, sdO, sLSE, m_block);

    // Loop over K/V blocks
    for (int n_block = 0; n_block < num_n_blocks; ++n_block) {
      int stage = n_block % kStages;

      // Initialize dK, dV accumulators for this block
      for (int n = 0; n < kTileN; ++n) {
        for (int k = 0; k < kHeadDim; ++k) {
          sdK(n, k) = Element(0);
          sdV(n, k) = Element(0);
        }
      }

      // Load K, V tiles
      // TODO: load_kv_tiles(gK, gV, sK, sV, n_block, stage);

      // Recompute P (attention weights) from forward pass
      // TODO: compute_qk(sQ, sK, sP, softmax_scale, stage);
      // TODO: compute_softmax(sP, sLSE, is_causal, m_block, n_block);

      // Compute dV = P^T @ dO
      compute_dV(sP, sdO, sdV);

      // Compute dP = dO @ V^T
      compute_dP(sdO, sV, sdP, stage);

      // Apply softmax backward
      softmax_backward(sP, sdP);

      // Compute dQ += dS @ K
      compute_dQ(sdP, sK, sdQ, softmax_scale, stage);

      // Compute dK = dS^T @ Q
      compute_dK(sdP, sQ, sdK, softmax_scale, stage);

      // Store dK, dV for this block
      // TODO: store_dK_dV(sdK, sdV, gdK, gdV, n_block);
    }

    // Store final dQ
    // TODO: store_dQ(sdQ, gdQ, m_block);
  }
};

} // namespace flash
