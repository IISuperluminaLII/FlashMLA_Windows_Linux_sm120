#pragma once

/***************************************************************************************************
 * SM120 Sparse Prefill Traits - Memory Layout and Configuration
 *
 * Memory Budget: 99KB (101,376 bytes) for SM120
 *
 * This follows the same patterns as SM120 dense decode but adapted for sparse prefill:
 * - Uses WMMA 16x16x16 operations (no GMMA/UMMA on SM120)
 * - Double-buffered K tiles for compute/memory overlap
 * - Online softmax with rescaling
 * - Sparse index handling for non-contiguous KV access
 *
 * Key dimensions (MLA - Multi-head Latent Attention):
 * - D_QK = 576 (query/key dimension after RoPE)
 * - D_V = 512 (value dimension)
 * - h_kv = 1 (single KV head, shared across h_q query heads)
 **************************************************************************************************/

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

namespace sm120 {
namespace sparse_prefill {

using namespace cute;

// Constants for MLA architecture
static constexpr int D_QK = 576;           // Query/Key dimension
static constexpr int D_V = 512;            // Value dimension

// Block sizes
static constexpr int B_H = 64;             // Query rows per block (h_q must be multiple)
static constexpr int B_TOPK = 64;          // TopK tokens per block (topk must be multiple)
static constexpr int K_TILE_DIM = 64;      // K-dimension tile (576 / 64 = 9 tiles)
static constexpr int NUM_K_TILES = (D_QK + K_TILE_DIM - 1) / K_TILE_DIM;  // 9

// Thread configuration
static constexpr int NUM_THREADS = 256;    // 8 warps
static constexpr int NUM_WARPS = NUM_THREADS / 32;
static constexpr int WARP_SIZE = 32;

// WMMA configuration
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// Output elements per thread for PV GEMM accumulator
// O shape: [B_H, D_V] = [64, 512]
// With 256 threads: 64 * 512 / 256 = 128 floats per thread
static constexpr int O_ELEMS_PER_THREAD = (B_H * D_V + NUM_THREADS - 1) / NUM_THREADS;

// V tiles for PV GEMM: D_V / K_TILE_DIM = 512 / 64 = 8 tiles
static constexpr int NUM_V_TILES = D_V / K_TILE_DIM;

//==============================================================================
// Swizzle pattern for bank-conflict-free shared memory access
// Uses 128-byte swizzle (8 bf16 elements alignment)
//==============================================================================
template<typename InputT>
struct SmemLayoutConfig {
    static constexpr int Alignment = 128 / sizeof_bits_v<InputT>;  // 8 for bf16
    using SmemSwizzle = Swizzle<3, 3, 3>;

    using SmemLayoutAtom = decltype(
        composition(SmemSwizzle{},
                    Layout<Shape<_8, Int<Alignment>>,
                           Stride<Int<Alignment>, _1>>{}));
};

//==============================================================================
// Traits template parameterized by input type
//==============================================================================
template<typename InputT>
struct Traits {
    using Input = InputT;
    using SmemConfig = SmemLayoutConfig<InputT>;

    // Sizes
    static constexpr int BLOCK_SIZE_M = B_H;
    static constexpr int BLOCK_SIZE_N = B_TOPK;
    static constexpr int HEAD_DIM_K = D_QK;
    static constexpr int HEAD_DIM_V = D_V;
    static constexpr int K_TILE_SIZE = K_TILE_DIM;
    static constexpr int NUM_K_ITERATIONS = NUM_K_TILES;
    static constexpr int NUM_V_ITERATIONS = NUM_V_TILES;

    // Thread counts
    static constexpr int NUM_THREADS_TOTAL = NUM_THREADS;
    static constexpr int NUM_WARPS_TOTAL = NUM_WARPS;

    // Output elements per thread
    static constexpr int O_ELEMS_PER_THREAD =
        (BLOCK_SIZE_M * HEAD_DIM_V + NUM_THREADS_TOTAL - 1) / NUM_THREADS_TOTAL;

    //==========================================================================
    // Shared Memory Layouts
    //==========================================================================

    // Q_tile: [B_H, K_TILE_DIM] = [64, 64]
    using SmemLayoutQ_tile = decltype(tile_to_shape(
        typename SmemConfig::SmemLayoutAtom{},
        Shape<Int<B_H>, Int<K_TILE_DIM>>{}));

    // K_tile: [B_TOPK, K_TILE_DIM] = [64, 64] (double-buffered)
    using SmemLayoutK_tile = decltype(tile_to_shape(
        typename SmemConfig::SmemLayoutAtom{},
        Shape<Int<B_TOPK>, Int<K_TILE_DIM>>{}));

    // V: [B_TOPK, D_V] = [64, 512]
    using SmemLayoutV = decltype(tile_to_shape(
        typename SmemConfig::SmemLayoutAtom{},
        Shape<Int<B_TOPK>, Int<D_V>>{}));

    // P (attention weights): [B_H, B_TOPK] = [64, 64]
    using SmemLayoutP = decltype(tile_to_shape(
        typename SmemConfig::SmemLayoutAtom{},
        Shape<Int<B_H>, Int<B_TOPK>>{}));

    // O_tile: [B_H, K_TILE_DIM] = [64, 64] for tiled PV output
    using SmemLayoutO_tile = decltype(tile_to_shape(
        typename SmemConfig::SmemLayoutAtom{},
        Shape<Int<B_H>, Int<K_TILE_DIM>>{}));

    //==========================================================================
    // Shared Memory Sizes
    //==========================================================================
    static constexpr size_t Q_tile_size = cosize_v<SmemLayoutQ_tile>;   // 4096
    static constexpr size_t K_tile_size = cosize_v<SmemLayoutK_tile>;   // 4096
    static constexpr size_t V_size = cosize_v<SmemLayoutV>;             // 32768
    static constexpr size_t P_size = cosize_v<SmemLayoutP>;             // 4096
    static constexpr size_t O_tile_size = cosize_v<SmemLayoutO_tile>;   // 4096

    //==========================================================================
    // Shared Memory Plan
    //
    // Uses union to overlap QK phase and PV phase memory:
    // - QK phase: Q_tile + K_tile0 + K_tile1 = 8 + 8 + 8 = 24 KB
    // - PV phase: V = 64 KB
    //
    // Total layout:
    // - Union of (QK phase | PV phase): max(24, 64) = 64 KB
    // - Scores (fp32): 64*64*4 = 16 KB
    // - P (bf16): 64*64*2 = 8 KB
    // - O_tile (fp32): 64*64*4 = 16 KB (for WMMA output staging)
    // - Softmax stats: 64*3*4 = 768 bytes
    // - Indices: 64*4 = 256 bytes
    // - Is_valid mask: 64 bytes
    //
    // Grand total: ~105 KB (too big!)
    //
    // Optimization: Overlap scores with O_tile (both 16KB, different phases)
    // New total: ~89 KB (fits in 99KB)
    //==========================================================================

    struct SharedMemoryPlan {
        // Phase 1 & 2 union: QK phase (Q + K0 + K1) overlaps with PV phase (V)
        union {
            struct {
                cute::array_aligned<InputT, Q_tile_size> smem_Q_tile;     // 8 KB
                cute::array_aligned<InputT, K_tile_size> smem_K_tile0;    // 8 KB
                cute::array_aligned<InputT, K_tile_size> smem_K_tile1;    // 8 KB
            } qk_phase;
            cute::array_aligned<InputT, V_size> smem_V;                   // 64 KB
        };

        // P attention weights (bf16) - needed for PV GEMM
        cute::array_aligned<InputT, P_size> smem_P;                       // 8 KB

        // Score accumulator (fp32) / O_tile buffer (fp32) - union (different phases)
        union {
            cute::array_aligned<float, P_size> smem_scores;               // 16 KB
            cute::array_aligned<float, O_tile_size> smem_O_tile;          // 16 KB
        };

        // Softmax statistics
        cute::array_aligned<float, B_H> smem_M;                           // 256 bytes (row max)
        cute::array_aligned<float, B_H> smem_L;                           // 256 bytes (row sum)
        cute::array_aligned<float, B_H> smem_scale;                       // 256 bytes (rescale)

        // Sparse index handling
        cute::array_aligned<int, B_TOPK> smem_indices;                    // 256 bytes
        cute::array_aligned<bool, B_TOPK> smem_is_valid;                  // 64 bytes
    };

    static constexpr size_t SMEM_SIZE = sizeof(SharedMemoryPlan);

    // Verify fits in SM120's 99KB shared memory
    static_assert(SMEM_SIZE <= 101376, "Shared memory exceeds SM120 limit of 99KB");
};

// Type aliases for common input types
using TraitsBF16 = Traits<cutlass::bfloat16_t>;
using TraitsFP16 = Traits<cutlass::half_t>;

} // namespace sparse_prefill
} // namespace sm120
