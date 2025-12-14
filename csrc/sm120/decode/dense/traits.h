#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/barrier.h>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include "config.h"

using namespace cute;

namespace sm120 {

// TMA barrier type for async memory transfers
using TMABarrier = cutlass::arch::ClusterTransactionBarrier;

//==============================================================================
// SM120 Decode Traits - Memory-Efficient Tiled Design
//
// SM120 has only 99KB shared memory. We tile Q along the K dimension:
// - Instead of storing full Q [64, 576], we store Q tiles [64, 64]
// - Iterate over 9 K-dimension tiles, accumulating scores
// - This reduces Q memory from 73KB to 8KB
//
// Uses optimized scalar operations with loop unrolling and vectorized access
// patterns. This approach is simpler than CuTe MMA and still achieves good
// performance through aggressive compiler optimization.
//==============================================================================

template<typename InputT_>
struct Traits {
    using InputT = InputT_;

    static constexpr int BLOCK_SIZE_M = Config::BLOCK_SIZE_M;      // 64
    static constexpr int PAGE_BLOCK_SIZE = Config::PAGE_BLOCK_SIZE; // 64
    static constexpr int HEAD_DIM_K = Config::HEAD_DIM_K;          // 576
    static constexpr int HEAD_DIM_V = Config::HEAD_DIM_V;          // 512
    static constexpr int NUM_THREADS = Config::NUM_THREADS;        // 256
    static constexpr int NUM_WARPS = Config::NUM_WARPS;            // 8
    static constexpr int NUM_WARPGROUPS = Config::NUM_WARPGROUPS;  // 2
    static constexpr int THREADS_PER_WARPGROUP = Config::THREADS_PER_WARPGROUP;  // 128
    static constexpr int WARPS_PER_WARPGROUP = Config::WARPS_PER_WARPGROUP;      // 4
    static constexpr int V_COLS_PER_WARPGROUP = Config::V_COLS_PER_WARPGROUP;    // 256
    static constexpr int K_TILE_DIM = Config::K_TILE_DIM;          // 64
    static constexpr int NUM_K_TILES = Config::NUM_K_TILES;        // 9

    static_assert(std::is_same_v<InputT, cutlass::bfloat16_t> || std::is_same_v<InputT, cutlass::half_t>);

    //==========================================================================
    // Shared Memory Layouts - TILED for 99KB budget
    //
    // Using swizzled layouts to avoid bank conflicts
    //==========================================================================

    static constexpr int Alignment = 128 / sizeof_bits_v<InputT>;  // 8 for bf16

    // Swizzle pattern for 128-byte alignment
    using SmemSwizzle = Swizzle<3, 3, 3>;

    // Base layout atom
    using SmemLayoutAtom = decltype(
        composition(SmemSwizzle{},
                    Layout<Shape<_8, Int<Alignment>>,
                           Stride<Int<Alignment>, _1>>{}));

    // Q_tile layout: [BLOCK_SIZE_M, K_TILE_DIM] = [64, 64] - TILED!
    // We load Q in 64-element K-dim tiles instead of full 576
    using SmemLayoutQ_tile = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<BLOCK_SIZE_M>, Int<K_TILE_DIM>>{}));

    // K_tile layout: [PAGE_BLOCK_SIZE, K_TILE_DIM] = [64, 64]
    using SmemLayoutK_tile = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<PAGE_BLOCK_SIZE>, Int<K_TILE_DIM>>{}));

    // V layout: [PAGE_BLOCK_SIZE, HEAD_DIM_V] = [64, 512]
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<PAGE_BLOCK_SIZE>, Int<HEAD_DIM_V>>{}));

    // P layout for attention scores: [BLOCK_SIZE_M, PAGE_BLOCK_SIZE] = [64, 64]
    using SmemLayoutP = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<BLOCK_SIZE_M>, Int<PAGE_BLOCK_SIZE>>{}));

    //==========================================================================
    // Shared Memory Plan - Fits in 99KB with TMA double-buffering!
    //
    // Memory budget: 99KB (101,376 bytes)
    //
    // QK Phase (with double-buffered K for TMA overlap):
    // - Q_tile: 64 * 64 * 2 = 8KB
    // - K_tile0: 64 * 64 * 2 = 8KB (buffer 0 for TMA)
    // - K_tile1: 64 * 64 * 2 = 8KB (buffer 1 for TMA)
    // - QK subtotal: 24KB
    //
    // PV Phase (sequential, overlaps with QK):
    // - V: 64 * 512 * 2 = 65KB
    //
    // Non-overlapping:
    // - P (bf16): 64 * 64 * 2 = 8KB
    // - scores (fp32): 64 * 64 * 4 = 16KB
    // - stats: 64 * 4 * 3 = 768 bytes
    // - TMA barriers: NUM_K_TILES * 2 * 64 = ~1KB
    //
    // Total: max(24KB, 65KB) + 8KB + 16KB + 1KB + 1KB = ~91KB - fits!
    //==========================================================================

    // Sizes for memory calculation
    static constexpr size_t Q_tile_size = BLOCK_SIZE_M * K_TILE_DIM;      // 4096 elements = 8KB
    static constexpr size_t K_tile_size = PAGE_BLOCK_SIZE * K_TILE_DIM;   // 4096 elements = 8KB
    static constexpr size_t V_size = PAGE_BLOCK_SIZE * HEAD_DIM_V;        // 32768 elements = 65KB
    static constexpr size_t P_size = BLOCK_SIZE_M * PAGE_BLOCK_SIZE;      // 4096 elements = 8KB

    struct SharedMemoryPlan {
        // Phase 1: QK computation with double-buffered K for TMA overlap
        // Phase 2: PV computation - V reuses QK space
        union {
            struct {
                cute::array_aligned<InputT, Q_tile_size> smem_Q_tile;     // 8KB
                cute::array_aligned<InputT, K_tile_size> smem_K_tile0;    // 8KB (TMA buffer 0)
                cute::array_aligned<InputT, K_tile_size> smem_K_tile1;    // 8KB (TMA buffer 1)
            } qk_phase;

            cute::array_aligned<InputT, V_size> smem_V;                   // 65KB (overlaps QK)
        };

        // P (attention scores) - needed across phases
        cute::array_aligned<InputT, P_size> smem_P;                       // 8KB

        // Score accumulator in fp32 for numerical stability
        cute::array_aligned<float, P_size> smem_scores;                   // 16KB

        // Softmax statistics
        cute::array_aligned<float, BLOCK_SIZE_M> smem_M;                  // 256 bytes (row-wise max)
        cute::array_aligned<float, BLOCK_SIZE_M> smem_L;                  // 256 bytes (row-wise sum)
        cute::array_aligned<float, BLOCK_SIZE_M> smem_scale;              // 256 bytes (rescaling)

        // TMA barriers for async K tile loading (double-buffered)
        TMABarrier barriers_K0[NUM_K_TILES];                              // Barriers for K buffer 0
        TMABarrier barriers_K1[NUM_K_TILES];                              // Barriers for K buffer 1
    };

    static constexpr size_t SharedMemSize = sizeof(SharedMemoryPlan);
    // 99KB = 101,376 bytes
    static_assert(SharedMemSize <= 101376, "Shared memory exceeds SM120 limit (99KB)");
};

}  // namespace sm120

// Include params.h for DecodingParams (shared with MSVC-safe header)
#include "params.h"
