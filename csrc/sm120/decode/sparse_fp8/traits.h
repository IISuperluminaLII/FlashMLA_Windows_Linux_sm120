#pragma once
// SM120 Sparse Decode Traits - WMMA-based configuration
// Fits within 99KB shared memory constraint

#include <cutlass/numeric_types.h>
#include <cuda_bf16.h>
#include <mma.h>

namespace sm120 {
namespace sparse_decode {

using bf16 = cutlass::bfloat16_t;
using fp8 = cutlass::float_e4m3_t;

// ============================================================================
// Block dimensions
// ============================================================================
static constexpr int BLOCK_M = 64;              // Query heads per block
static constexpr int TOPK_BLOCK_SIZE = 64;      // Top-K tokens per iteration
static constexpr int PAGE_BLOCK_SIZE = 64;      // KV cache page size

// MLA dimensions (from DeepSeek-V3)
static constexpr int HEAD_DIM_K = 576;          // d_qk
static constexpr int HEAD_DIM_V = 512;          // d_v
static constexpr int HEAD_DIM_NOPE = HEAD_DIM_V;  // Non-RoPE part (stored in FP8)
static constexpr int HEAD_DIM_ROPE = HEAD_DIM_K - HEAD_DIM_V;  // RoPE part (stored in BF16)

// FP8 quantization
static constexpr int QUANT_TILE_SIZE = 128;     // Elements per scale
static constexpr int NUM_SCALES = HEAD_DIM_NOPE / QUANT_TILE_SIZE;  // 4 scales per token

// Bytes per token in FP8 format:
// - HEAD_DIM_NOPE bytes FP8 (512)
// - NUM_SCALES * 4 bytes float scales (16)
// - HEAD_DIM_ROPE * 2 bytes BF16 for RoPE (128)
static constexpr int NUM_BYTES_PER_TOKEN = HEAD_DIM_NOPE + NUM_SCALES * sizeof(float) + HEAD_DIM_ROPE * sizeof(bf16);  // 656 bytes

// ============================================================================
// Thread configuration
// ============================================================================
static constexpr int NUM_WARPS = 8;
static constexpr int NUM_THREADS = NUM_WARPS * 32;  // 256 threads
static constexpr int NUM_K_BUFS = 2;  // Double buffer for K/V

// WMMA tile dimensions
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// ============================================================================
// Shared memory layout
// ============================================================================

// Q: [BLOCK_M, HEAD_DIM_K] = [64, 576] in BF16 = 73,728 bytes
// But we process Q in tiles of 64 columns at a time

// K (after dequant): [TOPK_BLOCK_SIZE, HEAD_DIM_K] = [64, 576] in BF16 = 73,728 bytes
// V: [TOPK_BLOCK_SIZE, HEAD_DIM_V] = [64, 512] in BF16 = 65,536 bytes
// K and V share same buffer since we process K first then V

// S (softmax output): [BLOCK_M, TOPK_BLOCK_SIZE] = [64, 64] in BF16 = 8,192 bytes

// Output accumulator: [BLOCK_M, HEAD_DIM_V] in FP32 = 131,072 bytes (too large for smem)
// -> Keep output in registers, process V in tiles

struct SharedMemoryLayout {
    // Phase 1: Load Q once, process all K blocks
    // Phase 2: Compute P @ V for each block

    // Q buffer: process in 64-column tiles (9 tiles for 576 dims)
    // Per-tile: [64, 64] * 2 bytes = 8,192 bytes
    static constexpr int Q_TILE_COLS = 64;
    static constexpr int Q_TILES = (HEAD_DIM_K + Q_TILE_COLS - 1) / Q_TILE_COLS;  // 9 tiles
    static constexpr int Q_TILE_SIZE = BLOCK_M * Q_TILE_COLS * sizeof(bf16);  // 8,192 bytes

    // K buffer (after FP8->BF16 dequant): [TOPK_BLOCK_SIZE, HEAD_DIM_K] double-buffered
    // Per-buffer: [64, 576] * 2 bytes = 73,728 bytes
    // Total K: 73,728 * 2 = 147,456 bytes -> TOO LARGE!

    // Solution: Process K in 64-column tiles like Q, store only current tile
    // K tile: [64, 64] * 2 bytes = 8,192 bytes per buffer
    static constexpr int K_TILE_COLS = 64;
    static constexpr int K_TILE_SIZE = TOPK_BLOCK_SIZE * K_TILE_COLS * sizeof(bf16);  // 8,192 bytes

    // V buffer (after FP8->BF16 dequant): [TOPK_BLOCK_SIZE, HEAD_DIM_V]
    // Process in 64-column tiles: [64, 64] * 2 bytes = 8,192 bytes
    static constexpr int V_TILE_COLS = 64;
    static constexpr int V_TILES = HEAD_DIM_V / V_TILE_COLS;  // 8 tiles
    static constexpr int V_TILE_SIZE = TOPK_BLOCK_SIZE * V_TILE_COLS * sizeof(bf16);  // 8,192 bytes

    // S matrix: [BLOCK_M, TOPK_BLOCK_SIZE] = [64, 64] in BF16
    static constexpr int S_SIZE = BLOCK_M * TOPK_BLOCK_SIZE * sizeof(bf16);  // 8,192 bytes

    // Validity flags for sparse indices
    static constexpr int VALIDITY_SIZE = NUM_K_BUFS * TOPK_BLOCK_SIZE;  // 128 bytes

    // Running max and sum for online softmax
    static constexpr int MAX_SUM_SIZE = BLOCK_M * sizeof(float) * 2;  // 512 bytes

    // Total layout using union for overlapping phases
    // Phase 1 (QK): Q_tile + K_tile[2] + S
    // Phase 2 (PV): V_tile + S + O_tile

    // QK phase: Q_tile (8K) + K_tile*2 (16K) + S (8K) + validity (128) + max_sum (512) = ~33KB
    // PV phase: V_tile (8K) + S (8K) + validity (128) + max_sum (512) = ~17KB

    // We can fit everything in 99KB!
};

// Shared memory structure
struct SharedMemoryPlan {
    // Q tile buffer (load Q in tiles)
    __align__(128) bf16 q_tile[BLOCK_M][SharedMemoryLayout::Q_TILE_COLS];  // 8,192 bytes

    // K/V buffers (double-buffered, share same storage since processed sequentially)
    union {
        __align__(128) bf16 k_tile[NUM_K_BUFS][TOPK_BLOCK_SIZE][SharedMemoryLayout::K_TILE_COLS];  // 16,384 bytes
        __align__(128) bf16 v_tile[TOPK_BLOCK_SIZE][SharedMemoryLayout::V_TILE_COLS];  // 8,192 bytes
    } kv;

    // Softmax output S: [BLOCK_M, TOPK_BLOCK_SIZE]
    __align__(128) bf16 s[BLOCK_M][TOPK_BLOCK_SIZE];  // 8,192 bytes

    // Validity flags for sparse indices (-1 = invalid)
    bool is_valid[NUM_K_BUFS][TOPK_BLOCK_SIZE];  // 128 bytes

    // Online softmax state
    float row_max[BLOCK_M];   // 256 bytes
    float row_sum[BLOCK_M];   // 256 bytes

    // Synchronization
    int producer_ready[NUM_K_BUFS];  // Producer signals K tile ready
    int consumer_done[NUM_K_BUFS];   // Consumer signals K tile consumed
};

// Verify shared memory fits in 99KB
static_assert(sizeof(SharedMemoryPlan) <= 99 * 1024,
    "SharedMemoryPlan exceeds SM120's 99KB shared memory limit");

// ============================================================================
// Decode parameters
// ============================================================================
struct SparseFP8DecodeParams {
    // Dimensions
    int batch_size;           // Number of sequences
    int s_q;                  // Query sequence positions (usually 1 for decode)
    int h_q;                  // Number of query heads
    int h_kv;                 // Number of KV heads (h_q / h_kv = GQA ratio)
    int topk;                 // Number of top-K tokens to attend to
    int page_size;            // KV cache page size

    // Softmax scale
    float sm_scale;
    float sm_scale_log2;

    // Input pointers
    bf16* q_ptr;              // [batch, s_q, h_q, HEAD_DIM_K]
    fp8* kv_ptr;              // [num_pages, page_size, NUM_BYTES_PER_TOKEN]
    int* indices_ptr;         // [batch, s_q, h_kv, topk] - sparse attention indices
    int* block_table_ptr;     // [batch, max_pages] - page table
    int* seq_lens_ptr;        // [batch] - sequence lengths

    // Strides
    int q_batch_stride;
    int q_seq_stride;
    int q_head_stride;
    int kv_page_stride;
    int kv_token_stride;
    int indices_batch_stride;
    int indices_seq_stride;
    int indices_head_stride;

    // Output pointers
    bf16* o_ptr;              // [batch, s_q, h_q, HEAD_DIM_V]
    float* softmax_lse_ptr;   // [batch, s_q, h_q] - log-sum-exp for split-kv merging

    int o_batch_stride;
    int o_seq_stride;
    int o_head_stride;

    // Split-KV parameters
    int num_splits;
    float* oaccum_ptr;        // [num_splits, batch, s_q, h_q, HEAD_DIM_V]
    float* lse_accum_ptr;     // [num_splits, batch, s_q, h_q]

    // CUDA stream
    cudaStream_t stream;
};

} // namespace sparse_decode
} // namespace sm120
