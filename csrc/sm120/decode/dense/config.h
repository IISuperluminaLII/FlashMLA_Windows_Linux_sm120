#pragma once

//==============================================================================
// SM120 Decode Kernel Configuration
//
// SM120 (Blackwell workstation GPUs: RTX 6000 Pro, RTX 50 series)
// - 99 KB shared memory (vs 227 KB on SM100, 232 KB on SM90)
// - Uses SM80-compatible mma.sync.aligned for BF16/FP16 (translates to HMMA)
// - Does NOT have GMMA (SM90) or tcgen05/UMMA for BF16 (only FP4/FP6/FP8)
// - Supports TMA (Tensor Memory Accelerator) for async data loading
// - Cluster shape fixed to 1x1x1 (no multicast)
// - Only TN layout supported for SM120-specific MMA (A row-major, B col-major)
//
// References:
// - https://deepwiki.com/NVIDIA/cutlass/7.3-sm120-blackwell-geforce-architecture
// - https://arxiv.org/html/2507.10789v1 (Blackwell microbenchmarks)
//==============================================================================

namespace sm120 {

struct Config {
    // Tile sizes - reduced to fit 99KB shared memory budget
    static constexpr int BLOCK_SIZE_M = 64;      // Query tile rows (same as SM90)
    static constexpr int PAGE_BLOCK_SIZE = 64;   // KV cache block size (same as SM90)
    static constexpr int HEAD_DIM_K = 576;       // MLA Q/K head dimension
    static constexpr int HEAD_DIM_V = 512;       // MLA V head dimension

    // Thread configuration - 256 threads (8 warps)
    // Only 4 warps do WMMA, other 4 help with memory ops and output
    static constexpr int NUM_THREADS = 256;      // 8 warps
    static constexpr int NUM_WARPS = NUM_THREADS / 32;
    static constexpr int NUM_WARPGROUPS = 2;
    static constexpr int THREADS_PER_WARPGROUP = 128;
    static constexpr int WARPS_PER_WARPGROUP = 4;

    // V split configuration (each warpgroup handles half)
    static constexpr int V_COLS_PER_WARPGROUP = HEAD_DIM_V / NUM_WARPGROUPS;  // 256

    // MMA tile sizes for SM80_16x8x16
    static constexpr int MMA_M = 16;
    static constexpr int MMA_N = 8;
    static constexpr int MMA_K = 16;

    // Memory budget (99KB = 101376 bytes)
    // Q:  64 * 576 * 2 = 73,728 bytes
    // K:  64 * 576 * 2 = 73,728 bytes (double buffered = 147,456 - too big!)
    //
    // Solution: Single-buffered K with smaller tiles
    // Q:  64 * 576 * 2 = 73,728 bytes
    // K:  64 * 64 * 2 * 9 tiles = 73,728 bytes (loaded tile-by-tile)
    // Total: ~74KB + overhead = fits in 99KB
    static constexpr int K_TILE_DIM = 64;        // K is loaded in 64-dim tiles
    static constexpr int NUM_K_TILES = HEAD_DIM_K / K_TILE_DIM;  // 9 tiles for 576
};

}  // namespace sm120