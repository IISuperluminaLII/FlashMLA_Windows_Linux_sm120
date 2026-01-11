/***************************************************************************************************
 * SM120 Dense FMHA Backward Kernel using WMMA with Dynamic Shared Memory
 *
 * This kernel implements the backward pass for Flash Multi-head Attention on SM120
 * (Blackwell workstation GPUs: RTX PRO 6000, RTX 50 series) using WMMA tensor core operations.
 *
 * SM120 constraints:
 * - Has WMMA mma.sync.aligned for bf16 (m16n16k16)
 * - Does NOT have TMEM (datacenter only)
 * - Does NOT have TCGEN05/UMMA (datacenter only)
 * - Has up to 99KB dynamic shared memory
 * - Supports cp.async for async global-to-shared memory copies (SM80+)
 *
 * Optimization: Double-buffered K/V tiles with cp.async for compute-memory overlap
 * - Uses ping-pong buffers for K and V tiles
 * - Prefetches next KV block while computing on current
 * - cp.async provides non-blocking memory transfers
 *
 * Backward pass computes:
 * - dV = P^T @ dO
 * - dP = dO @ V^T
 * - dScores = (dP - sum(dP * P, dim=-1, keepdim=True)) * P
 * - dQ = dScores @ K
 * - dK = dScores^T @ Q
 *
 * Where P = softmax(scores), scores = Q @ K^T * scale
 *
 **************************************************************************************************/
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

// For cp.async support detection
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define SM120_HAS_CP_ASYNC 1
#else
#define SM120_HAS_CP_ASYNC 0
#endif

namespace flash {
namespace detail {

// WMMA tile dimensions
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Kernel configuration - tile sizes for SM120's dynamic shared memory
// SM120 supports up to 99KB dynamic shared memory via cudaFuncSetAttribute.
// Optimized for larger tiles to increase parallelism and reduce launch overhead
// Calculation with M=32, N=32, D=128, 8 warps:
//   q_tile: 32*128*2 = 8KB
//   k_tile (double buffered): 2*32*128*2 = 16KB
//   v_tile (double buffered): 2*32*128*2 = 16KB
//   do_tile: 8KB, o_tile: 8KB
//   scores/probs/dscores: 32*32*4 = 4KB each (12KB total)
//   lse/delta: ~1KB, dq_acc: 32*128*4 = 16KB, wmma_staging: 8*16*16*4 = 8KB
//   temp_bf16: 32*32*2 = 2KB
//   Total: ~95KB < 99KB limit
constexpr int BWD_BLOCK_M = 32;   // Queries per block (2x larger)
constexpr int BWD_BLOCK_N = 32;   // Keys per block
constexpr int BWD_BLOCK_D = 128;  // Head dimension (must match kernel interface)
constexpr int BWD_NUM_WARPS = 8;  // Warps for tile parallelism (2x more)
constexpr int BWD_NUM_THREADS = BWD_NUM_WARPS * 32;

// K-major specific configuration - reduced N-tile for smem efficiency
// K-major loads K/V once (no double-buffering needed), uses smaller N-tiles
// Calculation with M=32, N=16, D=128, 8 warps:
//   q_tile: 32*128*2 = 8KB, k_tile: 16*128*2 = 4KB, v_tile: 4KB
//   do_tile: 8KB, o_tile: 8KB, scores/probs/dscores: 32*16*4 = 2KB each (6KB)
//   lse/delta: ~0.5KB, dq_acc: 16KB, wmma_staging: 8KB, temp_bf16: 1KB
//   dk_acc: 16*128*4 = 8KB, dv_acc: 8KB
//   Total: ~79KB < 99KB limit
constexpr int KMAJOR_BLOCK_N = 16;   // Smaller N-tile for K-major smem efficiency

// Aligned shared memory layout for WMMA (256-byte alignment for optimal performance)
// Double-buffered K and V tiles for async prefetching with cp.async
struct alignas(256) BwdSmemLayout {
    // Input tiles - bf16, need 256-byte alignment for WMMA
    static constexpr size_t q_tile_offset = 0;
    static constexpr size_t q_tile_size = BWD_BLOCK_M * BWD_BLOCK_D * sizeof(__nv_bfloat16);

    // Double-buffered K tiles (ping-pong buffers for async prefetch)
    static constexpr size_t k_tile_0_offset = q_tile_offset + ((q_tile_size + 255) / 256) * 256;
    static constexpr size_t k_tile_size = BWD_BLOCK_N * BWD_BLOCK_D * sizeof(__nv_bfloat16);
    static constexpr size_t k_tile_1_offset = k_tile_0_offset + ((k_tile_size + 255) / 256) * 256;

    // Double-buffered V tiles (ping-pong buffers for async prefetch)
    static constexpr size_t v_tile_0_offset = k_tile_1_offset + ((k_tile_size + 255) / 256) * 256;
    static constexpr size_t v_tile_size = BWD_BLOCK_N * BWD_BLOCK_D * sizeof(__nv_bfloat16);
    static constexpr size_t v_tile_1_offset = v_tile_0_offset + ((v_tile_size + 255) / 256) * 256;

    static constexpr size_t do_tile_offset = v_tile_1_offset + ((v_tile_size + 255) / 256) * 256;
    static constexpr size_t do_tile_size = BWD_BLOCK_M * BWD_BLOCK_D * sizeof(__nv_bfloat16);

    // Intermediate float tiles - for WMMA accumulator stores
    static constexpr size_t scores_offset = do_tile_offset + ((do_tile_size + 255) / 256) * 256;
    static constexpr size_t scores_size = BWD_BLOCK_M * BWD_BLOCK_N * sizeof(float);

    static constexpr size_t probs_offset = scores_offset + ((scores_size + 255) / 256) * 256;
    static constexpr size_t probs_size = BWD_BLOCK_M * BWD_BLOCK_N * sizeof(float);

    static constexpr size_t dscores_offset = probs_offset + ((probs_size + 255) / 256) * 256;
    static constexpr size_t dscores_size = BWD_BLOCK_M * BWD_BLOCK_N * sizeof(float);

    // LSE and delta (precomputed O dot dO for correct multi-block rowsum)
    static constexpr size_t lse_offset = dscores_offset + ((dscores_size + 255) / 256) * 256;
    static constexpr size_t lse_size = BWD_BLOCK_M * sizeof(float);

    // Delta = O dot dO, precomputed once per M-block (replaces per-KV-block rowsum)
    static constexpr size_t delta_offset = lse_offset + ((lse_size + 255) / 256) * 256;
    static constexpr size_t delta_size = BWD_BLOCK_M * sizeof(float);

    // O tile for computing delta (same size as dO tile)
    static constexpr size_t o_tile_offset = delta_offset + ((delta_size + 255) / 256) * 256;
    static constexpr size_t o_tile_size = BWD_BLOCK_M * BWD_BLOCK_D * sizeof(__nv_bfloat16);

    // dQ accumulator - float for precision
    static constexpr size_t dq_acc_offset = o_tile_offset + ((o_tile_size + 255) / 256) * 256;
    static constexpr size_t dq_acc_size = BWD_BLOCK_M * BWD_BLOCK_D * sizeof(float);

    // WMMA staging buffer for dQ computation - one 16x16 tile per warp
    static constexpr size_t wmma_staging_offset = dq_acc_offset + ((dq_acc_size + 255) / 256) * 256;
    static constexpr size_t wmma_staging_size = BWD_NUM_WARPS * WMMA_M * WMMA_N * sizeof(float);

    // Temporary bf16 buffer for WMMA dScores conversion
    static constexpr size_t temp_bf16_offset = wmma_staging_offset + ((wmma_staging_size + 255) / 256) * 256;
    static constexpr size_t temp_bf16_size = BWD_BLOCK_M * BWD_BLOCK_N * sizeof(__nv_bfloat16);

    // dK/dV accumulators for K-major loop (eliminates atomics!)
    // Only used by K-major kernel, Q-major kernel doesn't use these
    static constexpr size_t dk_acc_offset = temp_bf16_offset + ((temp_bf16_size + 255) / 256) * 256;
    static constexpr size_t dk_acc_size = BWD_BLOCK_N * BWD_BLOCK_D * sizeof(float);

    static constexpr size_t dv_acc_offset = dk_acc_offset + ((dk_acc_size + 255) / 256) * 256;
    static constexpr size_t dv_acc_size = BWD_BLOCK_N * BWD_BLOCK_D * sizeof(float);

    // Total size for K-major kernel (includes dk_acc, dv_acc)
    static constexpr size_t total_size_kmajor = dv_acc_offset + ((dv_acc_size + 255) / 256) * 256;

    // Total size for Q-major kernel (doesn't include dk_acc, dv_acc)
    static constexpr size_t total_size = temp_bf16_offset + ((temp_bf16_size + 255) / 256) * 256;

    // Legacy single-buffer offsets for backward compatibility (point to buffer 0)
    static constexpr size_t k_tile_offset = k_tile_0_offset;
    static constexpr size_t v_tile_offset = v_tile_0_offset;
};

// K-major specific shared memory layout - optimized for K-major loop ordering
// Uses KMAJOR_BLOCK_N=16 and single-buffered K/V to fit within 99KB smem limit
// Memory-efficient layout for K-major algorithm matching SM100/SM90 logic
// OPTIMIZED: Double-buffered Q tiles for cp.async prefetching
struct alignas(256) KMajorSmemLayout {
    // Q tile - DOUBLE BUFFERED for cp.async prefetch during compute
    static constexpr size_t q_tile_0_offset = 0;
    static constexpr size_t q_tile_size = BWD_BLOCK_M * BWD_BLOCK_D * sizeof(__nv_bfloat16);
    static constexpr size_t q_tile_1_offset = q_tile_0_offset + ((q_tile_size + 255) / 256) * 256;

    // K tile - SINGLE buffer (loaded once at start), smaller N-tile
    static constexpr size_t k_tile_offset = q_tile_1_offset + ((q_tile_size + 255) / 256) * 256;
    static constexpr size_t k_tile_size = KMAJOR_BLOCK_N * BWD_BLOCK_D * sizeof(__nv_bfloat16);

    // V tile - SINGLE buffer (loaded once at start), smaller N-tile
    static constexpr size_t v_tile_offset = k_tile_offset + ((k_tile_size + 255) / 256) * 256;
    static constexpr size_t v_tile_size = KMAJOR_BLOCK_N * BWD_BLOCK_D * sizeof(__nv_bfloat16);

    // dO tile - DOUBLE BUFFERED for cp.async prefetch during compute
    static constexpr size_t do_tile_0_offset = v_tile_offset + ((v_tile_size + 255) / 256) * 256;
    static constexpr size_t do_tile_size = BWD_BLOCK_M * BWD_BLOCK_D * sizeof(__nv_bfloat16);
    static constexpr size_t do_tile_1_offset = do_tile_0_offset + ((do_tile_size + 255) / 256) * 256;

    // O tile - for computing delta (single buffer, only used briefly)
    static constexpr size_t o_tile_offset = do_tile_1_offset + ((do_tile_size + 255) / 256) * 256;
    static constexpr size_t o_tile_size = BWD_BLOCK_M * BWD_BLOCK_D * sizeof(__nv_bfloat16);

    // Legacy single-buffer offsets for backward compatibility
    static constexpr size_t q_tile_offset = q_tile_0_offset;
    static constexpr size_t do_tile_offset = do_tile_0_offset;

    // Scores/probs/dscores - float, smaller with N=16
    static constexpr size_t scores_offset = o_tile_offset + ((o_tile_size + 255) / 256) * 256;
    static constexpr size_t scores_size = BWD_BLOCK_M * KMAJOR_BLOCK_N * sizeof(float);

    static constexpr size_t probs_offset = scores_offset + ((scores_size + 255) / 256) * 256;
    static constexpr size_t probs_size = BWD_BLOCK_M * KMAJOR_BLOCK_N * sizeof(float);

    static constexpr size_t dscores_offset = probs_offset + ((probs_size + 255) / 256) * 256;
    static constexpr size_t dscores_size = BWD_BLOCK_M * KMAJOR_BLOCK_N * sizeof(float);

    // LSE and delta
    static constexpr size_t lse_offset = dscores_offset + ((dscores_size + 255) / 256) * 256;
    static constexpr size_t lse_size = BWD_BLOCK_M * sizeof(float);

    static constexpr size_t delta_offset = lse_offset + ((lse_size + 255) / 256) * 256;
    static constexpr size_t delta_size = BWD_BLOCK_M * sizeof(float);

    // dQ accumulator - float, full M*D size
    static constexpr size_t dq_acc_offset = delta_offset + ((delta_size + 255) / 256) * 256;
    static constexpr size_t dq_acc_size = BWD_BLOCK_M * BWD_BLOCK_D * sizeof(float);

    // WMMA staging buffer
    static constexpr size_t wmma_staging_offset = dq_acc_offset + ((dq_acc_size + 255) / 256) * 256;
    static constexpr size_t wmma_staging_size = BWD_NUM_WARPS * WMMA_M * WMMA_N * sizeof(float);

    // Temporary bf16 buffer for dScores conversion (smaller with N=16)
    static constexpr size_t temp_bf16_offset = wmma_staging_offset + ((wmma_staging_size + 255) / 256) * 256;
    static constexpr size_t temp_bf16_size = BWD_BLOCK_M * KMAJOR_BLOCK_N * sizeof(__nv_bfloat16);

    // dK/dV accumulators - float, smaller with N=16 (KEY SAVINGS!)
    static constexpr size_t dk_acc_offset = temp_bf16_offset + ((temp_bf16_size + 255) / 256) * 256;
    static constexpr size_t dk_acc_size = KMAJOR_BLOCK_N * BWD_BLOCK_D * sizeof(float);

    static constexpr size_t dv_acc_offset = dk_acc_offset + ((dk_acc_size + 255) / 256) * 256;
    static constexpr size_t dv_acc_size = KMAJOR_BLOCK_N * BWD_BLOCK_D * sizeof(float);

    // Total size for K-major kernel
    static constexpr size_t total_size = dv_acc_offset + ((dv_acc_size + 255) / 256) * 256;
};

// Accessor for K-major dynamic shared memory
// OPTIMIZED: Supports double-buffered Q/dO tiles for cp.async pipelining
struct KMajorSmemAccessor {
    char* base;
    int cur_buf;  // Current buffer index for double-buffering (0 or 1)

    // Initialize accessor with base pointer and buffer index
    __device__ __forceinline__ void init(char* base_ptr, int buffer_idx = 0) {
        base = base_ptr;
        cur_buf = buffer_idx;
    }

    // Set current buffer index
    __device__ __forceinline__ void set_buffer(int buf_idx) {
        cur_buf = buf_idx;
    }

    // Q tile access - uses current buffer
    __device__ __forceinline__ __nv_bfloat16* q_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base +
            (cur_buf == 0 ? KMajorSmemLayout::q_tile_0_offset : KMajorSmemLayout::q_tile_1_offset));
    }
    // Explicit buffer-indexed Q tile access for async loading
    __device__ __forceinline__ __nv_bfloat16* q_tile_buf(int buf_idx) {
        return reinterpret_cast<__nv_bfloat16*>(base +
            (buf_idx == 0 ? KMajorSmemLayout::q_tile_0_offset : KMajorSmemLayout::q_tile_1_offset));
    }
    __device__ __forceinline__ __nv_bfloat16* k_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base + KMajorSmemLayout::k_tile_offset);
    }
    __device__ __forceinline__ __nv_bfloat16* v_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base + KMajorSmemLayout::v_tile_offset);
    }
    // dO tile access - uses current buffer
    __device__ __forceinline__ __nv_bfloat16* do_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base +
            (cur_buf == 0 ? KMajorSmemLayout::do_tile_0_offset : KMajorSmemLayout::do_tile_1_offset));
    }
    // Explicit buffer-indexed dO tile access for async loading
    __device__ __forceinline__ __nv_bfloat16* do_tile_buf(int buf_idx) {
        return reinterpret_cast<__nv_bfloat16*>(base +
            (buf_idx == 0 ? KMajorSmemLayout::do_tile_0_offset : KMajorSmemLayout::do_tile_1_offset));
    }
    __device__ __forceinline__ __nv_bfloat16* o_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base + KMajorSmemLayout::o_tile_offset);
    }
    __device__ __forceinline__ float* scores() {
        return reinterpret_cast<float*>(base + KMajorSmemLayout::scores_offset);
    }
    __device__ __forceinline__ float* probs() {
        return reinterpret_cast<float*>(base + KMajorSmemLayout::probs_offset);
    }
    __device__ __forceinline__ float* dscores() {
        return reinterpret_cast<float*>(base + KMajorSmemLayout::dscores_offset);
    }
    __device__ __forceinline__ float* lse() {
        return reinterpret_cast<float*>(base + KMajorSmemLayout::lse_offset);
    }
    __device__ __forceinline__ float* delta() {
        return reinterpret_cast<float*>(base + KMajorSmemLayout::delta_offset);
    }
    __device__ __forceinline__ float* dq_acc() {
        return reinterpret_cast<float*>(base + KMajorSmemLayout::dq_acc_offset);
    }
    __device__ __forceinline__ float* wmma_staging() {
        return reinterpret_cast<float*>(base + KMajorSmemLayout::wmma_staging_offset);
    }
    __device__ __forceinline__ float* wmma_staging_warp(int warp_id) {
        return wmma_staging() + warp_id * WMMA_M * WMMA_N;
    }
    __device__ __forceinline__ __nv_bfloat16* temp_bf16() {
        return reinterpret_cast<__nv_bfloat16*>(base + KMajorSmemLayout::temp_bf16_offset);
    }
    __device__ __forceinline__ float* dk_acc() {
        return reinterpret_cast<float*>(base + KMajorSmemLayout::dk_acc_offset);
    }
    __device__ __forceinline__ float* dv_acc() {
        return reinterpret_cast<float*>(base + KMajorSmemLayout::dv_acc_offset);
    }

    // Row accessors - use current buffer for Q/dO
    __device__ __forceinline__ __nv_bfloat16* q_row(int m) {
        return q_tile() + m * BWD_BLOCK_D;  // q_tile() uses cur_buf
    }
    // Buffer-indexed Q row accessor for async loading
    __device__ __forceinline__ __nv_bfloat16* q_row_buf(int m, int buf_idx) {
        return q_tile_buf(buf_idx) + m * BWD_BLOCK_D;
    }
    __device__ __forceinline__ __nv_bfloat16* k_row(int n) {
        return k_tile() + n * BWD_BLOCK_D;
    }
    __device__ __forceinline__ __nv_bfloat16* v_row(int n) {
        return v_tile() + n * BWD_BLOCK_D;
    }
    __device__ __forceinline__ __nv_bfloat16* do_row(int m) {
        return do_tile() + m * BWD_BLOCK_D;  // do_tile() uses cur_buf
    }
    // Buffer-indexed dO row accessor for async loading
    __device__ __forceinline__ __nv_bfloat16* do_row_buf(int m, int buf_idx) {
        return do_tile_buf(buf_idx) + m * BWD_BLOCK_D;
    }
    __device__ __forceinline__ __nv_bfloat16* o_row(int m) {
        return o_tile() + m * BWD_BLOCK_D;
    }
    __device__ __forceinline__ float* scores_row(int m) {
        return scores() + m * KMAJOR_BLOCK_N;
    }
    __device__ __forceinline__ float* probs_row(int m) {
        return probs() + m * KMAJOR_BLOCK_N;
    }
    __device__ __forceinline__ float* dscores_row(int m) {
        return dscores() + m * KMAJOR_BLOCK_N;
    }
};

// ============================================================================
// Q-MAJOR SMEM LAYOUT AND ACCESSOR FOR DQ-ONLY KERNEL
// ============================================================================

// Q-major block sizes for dQ kernel (smaller M to fit smem)
constexpr int QMAJOR_BLOCK_M = 32;  // Q-block size
constexpr int QMAJOR_BLOCK_N = 32;  // K-block size per iteration (increased from 16 to halve loop count)

// Q-major smem layout for dQ kernel
// Uses M=32 to fit within 99KB smem limit
// Double-buffered K/V tiles for async prefetch during K-block loop
struct alignas(256) QMajorSmemLayout {
    // Q tile - loaded once per block
    static constexpr size_t q_tile_offset = 0;
    static constexpr size_t q_tile_size = QMAJOR_BLOCK_M * BWD_BLOCK_D * sizeof(__nv_bfloat16);

    // K tile - DOUBLE BUFFERED for async prefetch
    static constexpr size_t k_tile_0_offset = q_tile_offset + ((q_tile_size + 255) / 256) * 256;
    static constexpr size_t k_tile_size = QMAJOR_BLOCK_N * BWD_BLOCK_D * sizeof(__nv_bfloat16);
    static constexpr size_t k_tile_1_offset = k_tile_0_offset + ((k_tile_size + 255) / 256) * 256;

    // V tile - DOUBLE BUFFERED for async prefetch
    static constexpr size_t v_tile_0_offset = k_tile_1_offset + ((k_tile_size + 255) / 256) * 256;
    static constexpr size_t v_tile_size = QMAJOR_BLOCK_N * BWD_BLOCK_D * sizeof(__nv_bfloat16);
    static constexpr size_t v_tile_1_offset = v_tile_0_offset + ((v_tile_size + 255) / 256) * 256;

    // dO tile - loaded once per block
    static constexpr size_t do_tile_offset = v_tile_1_offset + ((v_tile_size + 255) / 256) * 256;
    static constexpr size_t do_tile_size = QMAJOR_BLOCK_M * BWD_BLOCK_D * sizeof(__nv_bfloat16);

    // O tile - loaded once per block (for delta computation)
    static constexpr size_t o_tile_offset = do_tile_offset + ((do_tile_size + 255) / 256) * 256;
    static constexpr size_t o_tile_size = QMAJOR_BLOCK_M * BWD_BLOCK_D * sizeof(__nv_bfloat16);

    // Scores/probs/dscores - float
    static constexpr size_t scores_offset = o_tile_offset + ((o_tile_size + 255) / 256) * 256;
    static constexpr size_t scores_size = QMAJOR_BLOCK_M * QMAJOR_BLOCK_N * sizeof(float);

    static constexpr size_t probs_offset = scores_offset + ((scores_size + 255) / 256) * 256;
    static constexpr size_t probs_size = QMAJOR_BLOCK_M * QMAJOR_BLOCK_N * sizeof(float);

    static constexpr size_t dscores_offset = probs_offset + ((probs_size + 255) / 256) * 256;
    static constexpr size_t dscores_size = QMAJOR_BLOCK_M * QMAJOR_BLOCK_N * sizeof(float);

    // LSE and delta - loaded once per block
    static constexpr size_t lse_offset = dscores_offset + ((dscores_size + 255) / 256) * 256;
    static constexpr size_t lse_size = QMAJOR_BLOCK_M * sizeof(float);

    static constexpr size_t delta_offset = lse_offset + ((lse_size + 255) / 256) * 256;
    static constexpr size_t delta_size = QMAJOR_BLOCK_M * sizeof(float);

    // dQ accumulator - accumulated across all K-blocks (KEY DIFFERENCE from K-major)
    static constexpr size_t dq_acc_offset = delta_offset + ((delta_size + 255) / 256) * 256;
    static constexpr size_t dq_acc_size = QMAJOR_BLOCK_M * BWD_BLOCK_D * sizeof(float);

    // WMMA staging buffer
    static constexpr size_t wmma_staging_offset = dq_acc_offset + ((dq_acc_size + 255) / 256) * 256;
    static constexpr size_t wmma_staging_size = BWD_NUM_WARPS * WMMA_M * WMMA_N * sizeof(float);

    // Temporary bf16 buffer for dScores conversion
    static constexpr size_t temp_bf16_offset = wmma_staging_offset + ((wmma_staging_size + 255) / 256) * 256;
    static constexpr size_t temp_bf16_size = QMAJOR_BLOCK_M * QMAJOR_BLOCK_N * sizeof(__nv_bfloat16);

    // Total size for Q-major kernel
    static constexpr size_t total_size = temp_bf16_offset + ((temp_bf16_size + 255) / 256) * 256;
};

// Accessor for Q-major dynamic shared memory
// Double-buffered K/V tiles for cp.async pipelining
struct QMajorSmemAccessor {
    char* base;
    int cur_buf;  // Current buffer index for K/V double-buffering

    __device__ __forceinline__ void init(char* base_ptr, int buffer_idx = 0) {
        base = base_ptr;
        cur_buf = buffer_idx;
    }

    __device__ __forceinline__ void set_buffer(int buf_idx) {
        cur_buf = buf_idx;
    }

    __device__ __forceinline__ __nv_bfloat16* q_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base + QMajorSmemLayout::q_tile_offset);
    }
    __device__ __forceinline__ __nv_bfloat16* k_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base +
            (cur_buf == 0 ? QMajorSmemLayout::k_tile_0_offset : QMajorSmemLayout::k_tile_1_offset));
    }
    __device__ __forceinline__ __nv_bfloat16* k_tile_buf(int buf_idx) {
        return reinterpret_cast<__nv_bfloat16*>(base +
            (buf_idx == 0 ? QMajorSmemLayout::k_tile_0_offset : QMajorSmemLayout::k_tile_1_offset));
    }
    __device__ __forceinline__ __nv_bfloat16* v_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base +
            (cur_buf == 0 ? QMajorSmemLayout::v_tile_0_offset : QMajorSmemLayout::v_tile_1_offset));
    }
    __device__ __forceinline__ __nv_bfloat16* v_tile_buf(int buf_idx) {
        return reinterpret_cast<__nv_bfloat16*>(base +
            (buf_idx == 0 ? QMajorSmemLayout::v_tile_0_offset : QMajorSmemLayout::v_tile_1_offset));
    }
    __device__ __forceinline__ __nv_bfloat16* do_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base + QMajorSmemLayout::do_tile_offset);
    }
    __device__ __forceinline__ __nv_bfloat16* o_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base + QMajorSmemLayout::o_tile_offset);
    }
    __device__ __forceinline__ float* scores() {
        return reinterpret_cast<float*>(base + QMajorSmemLayout::scores_offset);
    }
    __device__ __forceinline__ float* probs() {
        return reinterpret_cast<float*>(base + QMajorSmemLayout::probs_offset);
    }
    __device__ __forceinline__ float* dscores() {
        return reinterpret_cast<float*>(base + QMajorSmemLayout::dscores_offset);
    }
    __device__ __forceinline__ float* lse() {
        return reinterpret_cast<float*>(base + QMajorSmemLayout::lse_offset);
    }
    __device__ __forceinline__ float* delta() {
        return reinterpret_cast<float*>(base + QMajorSmemLayout::delta_offset);
    }
    __device__ __forceinline__ float* dq_acc() {
        return reinterpret_cast<float*>(base + QMajorSmemLayout::dq_acc_offset);
    }
    __device__ __forceinline__ float* wmma_staging() {
        return reinterpret_cast<float*>(base + QMajorSmemLayout::wmma_staging_offset);
    }
    __device__ __forceinline__ float* wmma_staging_warp(int warp_id) {
        return wmma_staging() + warp_id * WMMA_M * WMMA_N;
    }
    __device__ __forceinline__ __nv_bfloat16* temp_bf16() {
        return reinterpret_cast<__nv_bfloat16*>(base + QMajorSmemLayout::temp_bf16_offset);
    }

    // Row accessors
    __device__ __forceinline__ __nv_bfloat16* q_row(int m) {
        return q_tile() + m * BWD_BLOCK_D;
    }
    __device__ __forceinline__ __nv_bfloat16* k_row(int n) {
        return k_tile() + n * BWD_BLOCK_D;
    }
    __device__ __forceinline__ __nv_bfloat16* v_row(int n) {
        return v_tile() + n * BWD_BLOCK_D;
    }
    __device__ __forceinline__ __nv_bfloat16* do_row(int m) {
        return do_tile() + m * BWD_BLOCK_D;
    }
    __device__ __forceinline__ __nv_bfloat16* o_row(int m) {
        return o_tile() + m * BWD_BLOCK_D;
    }
    __device__ __forceinline__ float* scores_row(int m) {
        return scores() + m * QMAJOR_BLOCK_N;
    }
    __device__ __forceinline__ float* probs_row(int m) {
        return probs() + m * QMAJOR_BLOCK_N;
    }
    __device__ __forceinline__ float* dscores_row(int m) {
        return dscores() + m * QMAJOR_BLOCK_N;
    }
};

// Accessor for dynamic shared memory with proper alignment
// Supports double-buffered K/V tiles via buffer index parameter
struct BwdSmemAccessor {
    char* base;

    // Get aligned pointers to each array
    __device__ __forceinline__ __nv_bfloat16* q_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base + BwdSmemLayout::q_tile_offset);
    }

    // Single-buffer k_tile (buffer 0) for backward compatibility
    __device__ __forceinline__ __nv_bfloat16* k_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base + BwdSmemLayout::k_tile_0_offset);
    }

    // Double-buffered K tile access
    __device__ __forceinline__ __nv_bfloat16* k_tile_buf(int buf_idx) {
        return reinterpret_cast<__nv_bfloat16*>(base +
            (buf_idx == 0 ? BwdSmemLayout::k_tile_0_offset : BwdSmemLayout::k_tile_1_offset));
    }

    // Single-buffer v_tile (buffer 0) for backward compatibility
    __device__ __forceinline__ __nv_bfloat16* v_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base + BwdSmemLayout::v_tile_0_offset);
    }

    // Double-buffered V tile access
    __device__ __forceinline__ __nv_bfloat16* v_tile_buf(int buf_idx) {
        return reinterpret_cast<__nv_bfloat16*>(base +
            (buf_idx == 0 ? BwdSmemLayout::v_tile_0_offset : BwdSmemLayout::v_tile_1_offset));
    }

    __device__ __forceinline__ __nv_bfloat16* do_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base + BwdSmemLayout::do_tile_offset);
    }
    __device__ __forceinline__ float* scores() {
        return reinterpret_cast<float*>(base + BwdSmemLayout::scores_offset);
    }
    __device__ __forceinline__ float* probs() {
        return reinterpret_cast<float*>(base + BwdSmemLayout::probs_offset);
    }
    __device__ __forceinline__ float* dscores() {
        return reinterpret_cast<float*>(base + BwdSmemLayout::dscores_offset);
    }
    __device__ __forceinline__ float* lse() {
        return reinterpret_cast<float*>(base + BwdSmemLayout::lse_offset);
    }
    __device__ __forceinline__ float* delta() {
        return reinterpret_cast<float*>(base + BwdSmemLayout::delta_offset);
    }
    __device__ __forceinline__ __nv_bfloat16* o_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base + BwdSmemLayout::o_tile_offset);
    }
    __device__ __forceinline__ float* dq_acc() {
        return reinterpret_cast<float*>(base + BwdSmemLayout::dq_acc_offset);
    }
    __device__ __forceinline__ float* wmma_staging() {
        return reinterpret_cast<float*>(base + BwdSmemLayout::wmma_staging_offset);
    }
    __device__ __forceinline__ float* wmma_staging_warp(int warp_id) {
        return wmma_staging() + warp_id * WMMA_M * WMMA_N;
    }
    __device__ __forceinline__ __nv_bfloat16* temp_bf16() {
        return reinterpret_cast<__nv_bfloat16*>(base + BwdSmemLayout::temp_bf16_offset);
    }

    // dK/dV accumulators for K-major kernel (eliminates atomics)
    __device__ __forceinline__ float* dk_acc() {
        return reinterpret_cast<float*>(base + BwdSmemLayout::dk_acc_offset);
    }
    __device__ __forceinline__ float* dv_acc() {
        return reinterpret_cast<float*>(base + BwdSmemLayout::dv_acc_offset);
    }
    __device__ __forceinline__ float* dk_acc_row(int n) {
        return dk_acc() + n * BWD_BLOCK_D;
    }
    __device__ __forceinline__ float* dv_acc_row(int n) {
        return dv_acc() + n * BWD_BLOCK_D;
    }

    // Row-major 2D accessors (for WMMA: stride is the leading dimension)
    __device__ __forceinline__ __nv_bfloat16* q_row(int m) {
        return q_tile() + m * BWD_BLOCK_D;
    }
    __device__ __forceinline__ __nv_bfloat16* k_row(int n) {
        return k_tile() + n * BWD_BLOCK_D;
    }
    __device__ __forceinline__ __nv_bfloat16* k_row_buf(int n, int buf_idx) {
        return k_tile_buf(buf_idx) + n * BWD_BLOCK_D;
    }
    __device__ __forceinline__ __nv_bfloat16* v_row(int n) {
        return v_tile() + n * BWD_BLOCK_D;
    }
    __device__ __forceinline__ __nv_bfloat16* v_row_buf(int n, int buf_idx) {
        return v_tile_buf(buf_idx) + n * BWD_BLOCK_D;
    }
    __device__ __forceinline__ __nv_bfloat16* do_row(int m) {
        return do_tile() + m * BWD_BLOCK_D;
    }
    __device__ __forceinline__ __nv_bfloat16* o_row(int m) {
        return o_tile() + m * BWD_BLOCK_D;
    }
    __device__ __forceinline__ float* scores_row(int m) {
        return scores() + m * BWD_BLOCK_N;
    }
    __device__ __forceinline__ float* probs_row(int m) {
        return probs() + m * BWD_BLOCK_N;
    }
    __device__ __forceinline__ float* dscores_row(int m) {
        return dscores() + m * BWD_BLOCK_N;
    }
    __device__ __forceinline__ float* dq_row(int m) {
        return dq_acc() + m * BWD_BLOCK_D;
    }
};

// ============================================================================
// CP.ASYNC HELPER FUNCTIONS FOR DOUBLE-BUFFERED PREFETCHING
// ============================================================================
// SM120 supports cp.async for async global-to-shared memory copies.
// These functions provide non-blocking memory transfers that can overlap
// with compute operations.
// ============================================================================

// Issue a cp.async for 16-byte (128-bit) aligned copy
// Uses PTX inline assembly for SM80+ cp.async instruction
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
#if SM120_HAS_CP_ASYNC
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
    );
#else
    // Fallback for older architectures: synchronous copy
    *reinterpret_cast<float4*>(smem_ptr) = *reinterpret_cast<const float4*>(gmem_ptr);
#endif
}

// Issue a cp.async for 8-byte (64-bit) copy
__device__ __forceinline__ void cp_async_cg_8(void* smem_ptr, const void* gmem_ptr) {
#if SM120_HAS_CP_ASYNC
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 8;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
    );
#else
    // Fallback: synchronous copy
    *reinterpret_cast<float2*>(smem_ptr) = *reinterpret_cast<const float2*>(gmem_ptr);
#endif
}

// Issue a cp.async for 4-byte (32-bit) copy
__device__ __forceinline__ void cp_async_cg_4(void* smem_ptr, const void* gmem_ptr) {
#if SM120_HAS_CP_ASYNC
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
    );
#else
    // Fallback: synchronous copy
    *reinterpret_cast<float*>(smem_ptr) = *reinterpret_cast<const float*>(gmem_ptr);
#endif
}

// Commit all outstanding cp.async operations
__device__ __forceinline__ void cp_async_commit_group() {
#if SM120_HAS_CP_ASYNC
    asm volatile("cp.async.commit_group;\n");
#endif
}

// Wait for all cp.async groups to complete
__device__ __forceinline__ void cp_async_wait_all() {
#if SM120_HAS_CP_ASYNC
    asm volatile("cp.async.wait_all;\n");
#endif
}

// ============================================================================
// WARP-LEVEL REDUCTION HELPER FUNCTIONS
// ============================================================================
// These functions enable efficient parallel reductions within a warp using
// shuffle instructions, avoiding shared memory bank conflicts.
// ============================================================================

// Warp-level sum reduction using shuffle instructions
// Reduces 32 values (one per lane) to a single sum in lane 0
__device__ __forceinline__ float warp_reduce_sum(float val) {
    const unsigned int FULL_MASK = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// Warp-level max reduction using shuffle instructions
// Reduces 32 values (one per lane) to the maximum across all lanes
__device__ __forceinline__ float warp_reduce_max(float val) {
    const unsigned int FULL_MASK = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, offset));
    }
    return val;
}

// ============================================================================
// BANK CONFLICT ELIMINATION VIA XOR SWIZZLING
// ============================================================================
// XOR-based swizzle pattern for eliminating shared memory bank conflicts.
// Based on gau-nernst Flash Attention 5090 research achieving 94.39% SOL.
//
// Formula: swizzled_index = index ^ ((row_idx / divisor) << 4)
// where divisor = max(64 / STRIDE, 1)
//
// For bf16 with D=128 (STRIDE=128 bytes per row = 64 bf16 elements):
// - divisor = max(64/128, 1) = 1 (but we use 64 bf16 = 128 bytes)
// - XOR bits 4-6 of address with bits 0-2 of row index
//
// Reference: https://gau-nernst.github.io/fa-5090/
// ============================================================================

// Swizzle index for bf16 tiles with ROW_STRIDE elements per row
// For D=128 head dim, ROW_STRIDE = 128 (bf16 elements)
template <int ROW_STRIDE>
__device__ __forceinline__ int swizzle_bf16_index(int row, int col) {
    // For 128-byte swizzle pattern:
    // - 8 bf16 elements = 16 bytes (one bank)
    // - XOR bits based on row index within each 8-row group
    constexpr int ELEMENTS_PER_BANK = 8;  // 8 bf16 = 16 bytes

    // Get row index within 8-row swizzle group
    int row_in_group = row & 0x7;  // row % 8

    // Calculate which 16-byte bank the column falls into
    int bank = col / ELEMENTS_PER_BANK;
    int offset_in_bank = col & (ELEMENTS_PER_BANK - 1);  // col % 8

    // XOR the bank with row_in_group to eliminate conflicts
    int swizzled_bank = bank ^ row_in_group;

    // Reconstruct the swizzled column index
    int swizzled_col = swizzled_bank * ELEMENTS_PER_BANK + offset_in_bank;

    return row * ROW_STRIDE + swizzled_col;
}

// Swizzle index for float tiles (scores, accumulators)
// For M=32, N=32 tiles, ROW_STRIDE = 32 floats = 128 bytes per row
template <int ROW_STRIDE>
__device__ __forceinline__ int swizzle_float_index(int row, int col) {
    // For float (4 bytes), 4 elements = 16 bytes (one bank)
    constexpr int ELEMENTS_PER_BANK = 4;

    int row_in_group = row & 0x7;
    int bank = col / ELEMENTS_PER_BANK;
    int offset_in_bank = col & (ELEMENTS_PER_BANK - 1);

    int swizzled_bank = bank ^ row_in_group;
    int swizzled_col = swizzled_bank * ELEMENTS_PER_BANK + offset_in_bank;

    return row * ROW_STRIDE + swizzled_col;
}

// Inverse swizzle to read back in original order
template <int ROW_STRIDE>
__device__ __forceinline__ int unswizzle_bf16_index(int row, int col) {
    // XOR is its own inverse, so same operation
    return swizzle_bf16_index<ROW_STRIDE>(row, col);
}

template <int ROW_STRIDE>
__device__ __forceinline__ int unswizzle_float_index(int row, int col) {
    return swizzle_float_index<ROW_STRIDE>(row, col);
}

// ============================================================================
// SWIZZLED MEMORY ACCESS HELPERS
// ============================================================================

// Store bf16 value to swizzled location
template <int ROW_STRIDE>
__device__ __forceinline__ void store_bf16_swizzled(
    __nv_bfloat16* base, int row, int col, __nv_bfloat16 val
) {
    base[swizzle_bf16_index<ROW_STRIDE>(row, col)] = val;
}

// Load bf16 value from swizzled location
template <int ROW_STRIDE>
__device__ __forceinline__ __nv_bfloat16 load_bf16_swizzled(
    const __nv_bfloat16* base, int row, int col
) {
    return base[swizzle_bf16_index<ROW_STRIDE>(row, col)];
}

// Store float value to swizzled location
template <int ROW_STRIDE>
__device__ __forceinline__ void store_float_swizzled(
    float* base, int row, int col, float val
) {
    base[swizzle_float_index<ROW_STRIDE>(row, col)] = val;
}

// Load float value from swizzled location
template <int ROW_STRIDE>
__device__ __forceinline__ float load_float_swizzled(
    const float* base, int row, int col
) {
    return base[swizzle_float_index<ROW_STRIDE>(row, col)];
}

// Vectorized dot product for bf16x2 vectors - reduces 2 elements at once
// Uses explicit low/high extraction for better compatibility
__device__ __forceinline__ float dot_bf16x2(const __nv_bfloat162& a, const __nv_bfloat162& b) {
    float ax = __bfloat162float(__low2bfloat16(a));
    float ay = __bfloat162float(__high2bfloat16(a));
    float bx = __bfloat162float(__low2bfloat16(b));
    float by = __bfloat162float(__high2bfloat16(b));
    return ax * bx + ay * by;
}

// ============================================================================
// ASYNC LOAD FUNCTIONS FOR DOUBLE-BUFFERED K/V TILES
// ============================================================================

// Async load K tile to specified buffer using cp.async
__device__ __forceinline__ void async_load_k_tile_varlen(
    BwdSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ k,
    int kv_seq_start,
    int n_start,
    int n_end,
    int num_heads,
    int head_idx,
    int head_dim,
    int buf_idx
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int cols = n_end - n_start;
    const int stride_token = num_heads * head_dim;

    // Each warp processes one or more rows (keys), threads in warp access consecutive d values
    for (int n = warp_id; n < cols; n += num_warps) {
        const int global_token = kv_seq_start + n_start + n;
        const __nv_bfloat16* src_row = k + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.k_tile_buf(buf_idx) + n * BWD_BLOCK_D;

        // Use 16-byte (8 bf16) async copies for best efficiency
        // Each thread handles 8 elements, 32 threads * 8 = 256 elements per iteration
        // For head_dim=128, need only 1 iteration (128/2 = 64 bf16x2 = 32 float4s)
        for (int d = lane_id * 8; d < head_dim; d += 256) {
            if (d + 8 <= head_dim) {
                cp_async_cg_16(dst_row + d, src_row + d);
            } else {
                // Handle tail with smaller copies or scalar
                for (int dd = d; dd < head_dim && dd < d + 8; dd++) {
                    dst_row[dd] = src_row[dd];
                }
            }
        }
    }
}

// Async load V tile to specified buffer using cp.async
__device__ __forceinline__ void async_load_v_tile_varlen(
    BwdSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ v,
    int kv_seq_start,
    int n_start,
    int n_end,
    int num_heads,
    int head_idx,
    int head_dim,
    int buf_idx
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int cols = n_end - n_start;
    const int stride_token = num_heads * head_dim;

    // Each warp processes one or more rows (values)
    for (int n = warp_id; n < cols; n += num_warps) {
        const int global_token = kv_seq_start + n_start + n;
        const __nv_bfloat16* src_row = v + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.v_tile_buf(buf_idx) + n * BWD_BLOCK_D;

        // Use 16-byte async copies
        for (int d = lane_id * 8; d < head_dim; d += 256) {
            if (d + 8 <= head_dim) {
                cp_async_cg_16(dst_row + d, src_row + d);
            } else {
                for (int dd = d; dd < head_dim && dd < d + 8; dd++) {
                    dst_row[dd] = src_row[dd];
                }
            }
        }
    }
}

// Zero a K/V buffer using vectorized stores
__device__ __forceinline__ void zero_kv_buffer(
    BwdSmemAccessor& smem,
    int buf_idx
) {
    const int tid = threadIdx.x;
    const __nv_bfloat162 zero_bf16x2 = __floats2bfloat162_rn(0.0f, 0.0f);

    __nv_bfloat162* k_ptr2 = reinterpret_cast<__nv_bfloat162*>(smem.k_tile_buf(buf_idx));
    __nv_bfloat162* v_ptr2 = reinterpret_cast<__nv_bfloat162*>(smem.v_tile_buf(buf_idx));

    for (int i = tid; i < (BWD_BLOCK_N * BWD_BLOCK_D) / 2; i += BWD_NUM_THREADS) {
        k_ptr2[i] = zero_bf16x2;
        v_ptr2[i] = zero_bf16x2;
    }
}

// ============================================================================
// ASYNC LOAD FUNCTIONS FOR K-MAJOR KERNEL DOUBLE-BUFFERED Q/dO TILES
// ============================================================================

// Async load Q tile to specified buffer using cp.async (K-major kernel)
__device__ __forceinline__ void kmajor_async_load_q_tile(
    KMajorSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ q,
    int q_seq_start,
    int m_start,
    int m_end,
    int num_heads,
    int head_idx,
    int head_dim,
    int buf_idx
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int rows = m_end - m_start;
    const int stride_token = num_heads * head_dim;

    // Each warp processes one or more rows (queries)
    for (int m = warp_id; m < rows; m += num_warps) {
        const int global_token = q_seq_start + m_start + m;
        const __nv_bfloat16* src_row = q + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.q_tile_buf(buf_idx) + m * BWD_BLOCK_D;

        // Use 16-byte (8 bf16) async copies for best efficiency
        // Each thread handles 8 elements, 32 threads * 8 = 256 elements per iteration
        // For head_dim=128, need only 1 iteration
        for (int d = lane_id * 8; d < head_dim; d += 256) {
            if (d + 8 <= head_dim) {
                cp_async_cg_16(dst_row + d, src_row + d);
            } else {
                // Handle tail with smaller copies
                for (int dd = d; dd < head_dim && dd < d + 8; dd++) {
                    dst_row[dd] = src_row[dd];
                }
            }
        }
    }
}

// Async load dO tile to specified buffer using cp.async (K-major kernel)
__device__ __forceinline__ void kmajor_async_load_do_tile(
    KMajorSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ d_o,
    int q_seq_start,
    int m_start,
    int m_end,
    int num_heads,
    int head_idx,
    int head_dim,
    int buf_idx
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int rows = m_end - m_start;
    const int stride_token = num_heads * head_dim;

    // Each warp processes one or more rows
    for (int m = warp_id; m < rows; m += num_warps) {
        const int global_token = q_seq_start + m_start + m;
        const __nv_bfloat16* src_row = d_o + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.do_tile_buf(buf_idx) + m * BWD_BLOCK_D;

        // Use 16-byte async copies
        for (int d = lane_id * 8; d < head_dim; d += 256) {
            if (d + 8 <= head_dim) {
                cp_async_cg_16(dst_row + d, src_row + d);
            } else {
                for (int dd = d; dd < head_dim && dd < d + 8; dd++) {
                    dst_row[dd] = src_row[dd];
                }
            }
        }
    }
}

// Synchronous load Q tile from buffer (for use after cp.async.wait)
__device__ __forceinline__ void kmajor_load_q_tile_sync(
    KMajorSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ q,
    int q_seq_start,
    int m_start,
    int m_end,
    int num_heads,
    int head_idx,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int rows = m_end - m_start;
    const int stride_token = num_heads * head_dim;

    for (int m = warp_id; m < rows; m += num_warps) {
        const int global_token = q_seq_start + m_start + m;
        const __nv_bfloat16* src_row = q + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.q_tile() + m * BWD_BLOCK_D;

        // Vectorized synchronous load
        for (int d = lane_id * 8; d < head_dim; d += 256) {
            if (d + 8 <= head_dim) {
                float4 val = *reinterpret_cast<const float4*>(src_row + d);
                *reinterpret_cast<float4*>(dst_row + d) = val;
            } else {
                for (int dd = d; dd < head_dim && dd < d + 8; dd++) {
                    dst_row[dd] = src_row[dd];
                }
            }
        }
    }
}

// Synchronous load dO tile (for use when not pipelining)
__device__ __forceinline__ void kmajor_load_do_tile_sync(
    KMajorSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ d_o,
    int q_seq_start,
    int m_start,
    int m_end,
    int num_heads,
    int head_idx,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int rows = m_end - m_start;
    const int stride_token = num_heads * head_dim;

    for (int m = warp_id; m < rows; m += num_warps) {
        const int global_token = q_seq_start + m_start + m;
        const __nv_bfloat16* src_row = d_o + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.do_tile() + m * BWD_BLOCK_D;

        // Vectorized synchronous load
        for (int d = lane_id * 8; d < head_dim; d += 256) {
            if (d + 8 <= head_dim) {
                float4 val = *reinterpret_cast<const float4*>(src_row + d);
                *reinterpret_cast<float4*>(dst_row + d) = val;
            } else {
                for (int dd = d; dd < head_dim && dd < d + 8; dd++) {
                    dst_row[dd] = src_row[dd];
                }
            }
        }
    }
}

// ============================================================================
// OPTIMIZED COALESCED MEMORY ACCESS FUNCTIONS
// ============================================================================

// Load Q tile from global to shared memory (varlen format: [total_tokens, heads, dim])
__device__ __forceinline__ void load_q_tile_varlen(
    BwdSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ q,
    int q_seq_start,
    int m_start,
    int m_end,
    int num_heads,
    int head_idx,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int rows = m_end - m_start;
    const int stride_token = num_heads * head_dim;

    for (int m = warp_id; m < rows; m += num_warps) {
        const int global_token = q_seq_start + m_start + m;
        const __nv_bfloat16* src_row = q + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.q_tile() + m * BWD_BLOCK_D;

        for (int d = lane_id * 2; d < head_dim; d += 64) {
            if (d + 1 < head_dim) {
                __nv_bfloat162 val = *reinterpret_cast<const __nv_bfloat162*>(src_row + d);
                *reinterpret_cast<__nv_bfloat162*>(dst_row + d) = val;
            } else if (d < head_dim) {
                dst_row[d] = src_row[d];
            }
        }
    }
}

// Load K tile from global to shared memory (synchronous version)
__device__ __forceinline__ void load_k_tile_varlen(
    BwdSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ k,
    int kv_seq_start,
    int n_start,
    int n_end,
    int num_heads,
    int head_idx,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int cols = n_end - n_start;
    const int stride_token = num_heads * head_dim;

    for (int n = warp_id; n < cols; n += num_warps) {
        const int global_token = kv_seq_start + n_start + n;
        const __nv_bfloat16* src_row = k + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.k_tile() + n * BWD_BLOCK_D;

        for (int d = lane_id * 2; d < head_dim; d += 64) {
            if (d + 1 < head_dim) {
                __nv_bfloat162 val = *reinterpret_cast<const __nv_bfloat162*>(src_row + d);
                *reinterpret_cast<__nv_bfloat162*>(dst_row + d) = val;
            } else if (d < head_dim) {
                dst_row[d] = src_row[d];
            }
        }
    }
}

// Load V tile from global to shared memory (synchronous version)
__device__ __forceinline__ void load_v_tile_varlen(
    BwdSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ v,
    int kv_seq_start,
    int n_start,
    int n_end,
    int num_heads,
    int head_idx,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int cols = n_end - n_start;
    const int stride_token = num_heads * head_dim;

    for (int n = warp_id; n < cols; n += num_warps) {
        const int global_token = kv_seq_start + n_start + n;
        const __nv_bfloat16* src_row = v + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.v_tile() + n * BWD_BLOCK_D;

        for (int d = lane_id * 2; d < head_dim; d += 64) {
            if (d + 1 < head_dim) {
                __nv_bfloat162 val = *reinterpret_cast<const __nv_bfloat162*>(src_row + d);
                *reinterpret_cast<__nv_bfloat162*>(dst_row + d) = val;
            } else if (d < head_dim) {
                dst_row[d] = src_row[d];
            }
        }
    }
}

// Load dO tile from global to shared memory
__device__ __forceinline__ void load_do_tile_varlen(
    BwdSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ d_o,
    int q_seq_start,
    int m_start,
    int m_end,
    int num_heads,
    int head_idx,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int rows = m_end - m_start;
    const int stride_token = num_heads * head_dim;

    for (int m = warp_id; m < rows; m += num_warps) {
        const int global_token = q_seq_start + m_start + m;
        const __nv_bfloat16* src_row = d_o + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.do_tile() + m * BWD_BLOCK_D;

        for (int d = lane_id * 2; d < head_dim; d += 64) {
            if (d + 1 < head_dim) {
                __nv_bfloat162 val = *reinterpret_cast<const __nv_bfloat162*>(src_row + d);
                *reinterpret_cast<__nv_bfloat162*>(dst_row + d) = val;
            } else if (d < head_dim) {
                dst_row[d] = src_row[d];
            }
        }
    }
}

// Load LSE values for softmax recomputation
__device__ __forceinline__ void load_lse_varlen(
    BwdSmemAccessor& smem,
    const float* __restrict__ lse,
    int q_seq_start,
    int m_start,
    int m_end,
    int num_heads,
    int head_idx
) {
    const int tid = threadIdx.x;
    const int rows = m_end - m_start;
    const int stride_token = num_heads;

    for (int m = tid; m < rows; m += BWD_NUM_THREADS) {
        int global_token = q_seq_start + m_start + m;
        smem.lse()[m] = lse[global_token * stride_token + head_idx];
    }
}

// Load O tile from global to shared memory
__device__ __forceinline__ void load_o_tile_varlen(
    BwdSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ o,
    int q_seq_start,
    int m_start,
    int m_end,
    int num_heads,
    int head_idx,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int rows = m_end - m_start;
    const int stride_token = num_heads * head_dim;

    for (int m = warp_id; m < rows; m += num_warps) {
        const int global_token = q_seq_start + m_start + m;
        const __nv_bfloat16* src_row = o + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.o_tile() + m * BWD_BLOCK_D;

        for (int d = lane_id * 2; d < head_dim; d += 64) {
            if (d + 1 < head_dim) {
                __nv_bfloat162 val = *reinterpret_cast<const __nv_bfloat162*>(src_row + d);
                *reinterpret_cast<__nv_bfloat162*>(dst_row + d) = val;
            } else if (d < head_dim) {
                dst_row[d] = src_row[d];
            }
        }
    }
}

// Compute delta[m] = O[m] dot dO[m] for each row
// Optimized with warp-level parallel reduction over D dimension
__device__ __forceinline__ void compute_delta(
    BwdSmemAccessor& smem,
    int m_size,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Each warp processes one row m
    // Lanes within warp parallel reduce over D dimension
    for (int m = warp_id; m < m_size; m += BWD_NUM_WARPS) {
        float sum = 0.0f;

        // Each lane accumulates D/32 elements (for D=128, each lane handles 4 elements)
        // Use vectorized loads for better throughput
        const __nv_bfloat16* o_row = smem.o_tile() + m * BWD_BLOCK_D;
        const __nv_bfloat16* do_row = smem.do_tile() + m * BWD_BLOCK_D;

        // Process 4 elements per lane for D=128 (128/32 = 4)
        // Use bf16x2 vectorized loads when possible
        #pragma unroll
        for (int d_base = lane_id * 2; d_base < head_dim; d_base += 64) {
            // Load bf16x2 pairs
            __nv_bfloat162 o_val2 = *reinterpret_cast<const __nv_bfloat162*>(o_row + d_base);
            __nv_bfloat162 do_val2 = *reinterpret_cast<const __nv_bfloat162*>(do_row + d_base);

            // Dot product of 2 elements
            sum += dot_bf16x2(o_val2, do_val2);
        }

        // Warp-level reduction using shuffle
        sum = warp_reduce_sum(sum);

        // Lane 0 writes the result
        if (lane_id == 0) {
            smem.delta()[m] = sum;
        }
    }
    __syncthreads();
}

// Compute QK^T scores using WMMA with double-buffered K tile
__device__ void compute_qk_scores_wmma_buf(
    BwdSmemAccessor& smem,
    int m_size,
    int n_size,
    int head_dim,
    float scale,
    int buf_idx
) {
    using namespace nvcuda::wmma;

    const int warp_id = threadIdx.x / 32;

    const int m_tiles = (m_size + WMMA_M - 1) / WMMA_M;
    const int n_tiles = (n_size + WMMA_N - 1) / WMMA_N;
    const int d_tiles = (head_dim + WMMA_K - 1) / WMMA_K;

    for (int mn = warp_id; mn < m_tiles * n_tiles; mn += BWD_NUM_WARPS) {
        int m_tile = mn / n_tiles;
        int n_tile = mn % n_tiles;

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        fill_fragment(acc, 0.0f);

        for (int k_tile = 0; k_tile < d_tiles; ++k_tile) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;

            load_matrix_sync(a_frag, smem.q_row(m_tile * WMMA_M) + k_tile * WMMA_K, BWD_BLOCK_D);
            load_matrix_sync(b_frag, smem.k_row_buf(n_tile * WMMA_N, buf_idx) + k_tile * WMMA_K, BWD_BLOCK_D);

            mma_sync(acc, a_frag, b_frag, acc);
        }

        float* scores_ptr = smem.scores_row(m_tile * WMMA_M) + n_tile * WMMA_N;
        store_matrix_sync(scores_ptr, acc, BWD_BLOCK_N, mem_row_major);
    }
    __syncthreads();

    const int tid = threadIdx.x;
    // Unroll for ILP: 32x32=1024 elements, 256 threads, ~4 iterations per thread
    #pragma unroll 4
    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        int m = idx / n_size;
        int n = idx % n_size;
        if (m < m_size && n < n_size) {
            smem.scores()[m * BWD_BLOCK_N + n] *= scale;
        }
    }
    __syncthreads();
}

// Original compute_qk_scores_wmma for compatibility
__device__ void compute_qk_scores_wmma(
    BwdSmemAccessor& smem,
    int m_size,
    int n_size,
    int head_dim,
    float scale
) {
    compute_qk_scores_wmma_buf(smem, m_size, n_size, head_dim, scale, 0);
}

// Recompute softmax probabilities from scores and LSE
__device__ void recompute_softmax(
    BwdSmemAccessor& smem,
    int m_size,
    int n_size,
    bool is_causal,
    int m_global_start,
    int n_global_start
) {
    const int tid = threadIdx.x;

    // Unroll for ILP: 32x32=1024 elements, 256 threads, ~4 iterations per thread
    #pragma unroll 4
    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        int m = idx / n_size;
        int n = idx % n_size;

        if (m < m_size && n < n_size) {
            float score = smem.scores()[m * BWD_BLOCK_N + n];
            float lse_val = smem.lse()[m];

            int m_global = m_global_start + m;
            int n_global = n_global_start + n;
            if (is_causal && m_global < n_global) {
                smem.probs()[m * BWD_BLOCK_N + n] = 0.0f;
            } else {
                smem.probs()[m * BWD_BLOCK_N + n] = expf(score - lse_val);
            }
        }
    }
    __syncthreads();
}

// Compute dP = dO @ V^T using WMMA with double-buffered V tile
__device__ void compute_dp_wmma_buf(
    BwdSmemAccessor& smem,
    int m_size,
    int n_size,
    int head_dim,
    int buf_idx
) {
    using namespace nvcuda::wmma;

    const int warp_id = threadIdx.x / 32;

    const int m_tiles = (m_size + WMMA_M - 1) / WMMA_M;
    const int n_tiles = (n_size + WMMA_N - 1) / WMMA_N;
    const int d_tiles = (head_dim + WMMA_K - 1) / WMMA_K;

    for (int mn = warp_id; mn < m_tiles * n_tiles; mn += BWD_NUM_WARPS) {
        int m_tile = mn / n_tiles;
        int n_tile = mn % n_tiles;

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        fill_fragment(acc, 0.0f);

        for (int k_tile = 0; k_tile < d_tiles; ++k_tile) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> do_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> v_frag;

            load_matrix_sync(do_frag, smem.do_row(m_tile * WMMA_M) + k_tile * WMMA_K, BWD_BLOCK_D);
            load_matrix_sync(v_frag, smem.v_row_buf(n_tile * WMMA_N, buf_idx) + k_tile * WMMA_K, BWD_BLOCK_D);

            mma_sync(acc, do_frag, v_frag, acc);
        }

        float* dscores_ptr = smem.dscores_row(m_tile * WMMA_M) + n_tile * WMMA_N;
        store_matrix_sync(dscores_ptr, acc, BWD_BLOCK_N, mem_row_major);
    }
    __syncthreads();
}

// Original compute_dp_wmma for compatibility
__device__ void compute_dp_wmma(
    BwdSmemAccessor& smem,
    int m_size,
    int n_size,
    int head_dim
) {
    compute_dp_wmma_buf(smem, m_size, n_size, head_dim, 0);
}

// Compute dScores = (dP - delta) * P * scale
__device__ void compute_dscores(
    BwdSmemAccessor& smem,
    int m_size,
    int n_size,
    float scale
) {
    const int tid = threadIdx.x;

    // Unroll for ILP: 32x32=1024 elements, 256 threads, ~4 iterations per thread
    #pragma unroll 4
    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        int m = idx / n_size;
        int n = idx % n_size;

        if (m < m_size && n < n_size) {
            float dp = smem.dscores()[m * BWD_BLOCK_N + n];
            float p = smem.probs()[m * BWD_BLOCK_N + n];
            float delta_m = smem.delta()[m];
            smem.dscores()[m * BWD_BLOCK_N + n] = (dp - delta_m) * p * scale;
        }
    }
    __syncthreads();
}

// Compute dQ = dScores @ K using WMMA with double-buffered K tile
__device__ void compute_dq_wmma_buf(
    BwdSmemAccessor& smem,
    int m_size,
    int n_size,
    int head_dim,
    int buf_idx
) {
    using namespace nvcuda::wmma;

    const int warp_id = threadIdx.x / 32;
    const int tid = threadIdx.x;

    __nv_bfloat16* dscores_bf16 = smem.temp_bf16();

    for (int idx = tid; idx < BWD_BLOCK_M * BWD_BLOCK_N; idx += BWD_NUM_THREADS) {
        dscores_bf16[idx] = __float2bfloat16(0.0f);
    }
    __syncthreads();

    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        int m = idx / n_size;
        int n = idx % n_size;
        if (m < m_size && n < n_size) {
            dscores_bf16[m * BWD_BLOCK_N + n] = __float2bfloat16(smem.dscores()[m * BWD_BLOCK_N + n]);
        }
    }
    __syncthreads();

    const int m_tiles = (m_size + WMMA_M - 1) / WMMA_M;
    const int d_tiles = (head_dim + WMMA_N - 1) / WMMA_N;
    const int n_tiles = (n_size + WMMA_K - 1) / WMMA_K;

    for (int md = warp_id; md < m_tiles * d_tiles; md += BWD_NUM_WARPS) {
        int m_tile = md / d_tiles;
        int d_tile = md % d_tiles;

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        fill_fragment(acc, 0.0f);

        for (int k_tile = 0; k_tile < n_tiles; ++k_tile) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> ds_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> k_frag;

            load_matrix_sync(ds_frag, dscores_bf16 + m_tile * WMMA_M * BWD_BLOCK_N + k_tile * WMMA_K, BWD_BLOCK_N);
            load_matrix_sync(k_frag, smem.k_row_buf(k_tile * WMMA_K, buf_idx) + d_tile * WMMA_N, BWD_BLOCK_D);

            mma_sync(acc, ds_frag, k_frag, acc);
        }

        float* staging = smem.wmma_staging_warp(warp_id);
        store_matrix_sync(staging, acc, WMMA_N, mem_row_major);
        __syncwarp();

        const int lane_id = threadIdx.x % 32;
        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int row = i / WMMA_N;
            int col = i % WMMA_N;
            int global_row = m_tile * WMMA_M + row;
            int global_col = d_tile * WMMA_N + col;
            if (global_row < m_size && global_col < head_dim) {
                smem.dq_acc()[global_row * BWD_BLOCK_D + global_col] += staging[i];
            }
        }
        __syncwarp();
    }
    __syncthreads();
}

// Original compute_dq_wmma for compatibility
__device__ void compute_dq_wmma(
    BwdSmemAccessor& smem,
    int m_size,
    int n_size,
    int head_dim
) {
    compute_dq_wmma_buf(smem, m_size, n_size, head_dim, 0);
}

// Write dQ from shared memory accumulator to global memory
__device__ void write_dq_varlen(
    BwdSmemAccessor& smem,
    __nv_bfloat16* __restrict__ dq,
    int q_seq_start,
    int m_start,
    int m_end,
    int num_heads,
    int head_idx,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int rows = m_end - m_start;
    const int stride_token = num_heads * head_dim;

    for (int m = warp_id; m < rows; m += num_warps) {
        const int global_token = q_seq_start + m_start + m;
        __nv_bfloat16* dst_row = dq + global_token * stride_token + head_idx * head_dim;
        float* src_row = smem.dq_acc() + m * BWD_BLOCK_D;

        for (int d = lane_id * 2; d < head_dim; d += 64) {
            if (d + 1 < head_dim) {
                __nv_bfloat162 val = __floats2bfloat162_rn(src_row[d], src_row[d + 1]);
                *reinterpret_cast<__nv_bfloat162*>(dst_row + d) = val;
            } else if (d < head_dim) {
                dst_row[d] = __float2bfloat16(src_row[d]);
            }
        }
    }
}

// ============================================================================
// WMMA-BASED dK/dV COMPUTATION
// ============================================================================
// Computes dK = dScores^T @ Q and dV = P^T @ dO using tensor cores
// This replaces the scalar loop + atomic implementation for better performance
// ============================================================================

// Compute dK = dScores^T @ Q using WMMA with tiled output for better cache locality
// dScores: [M, N] row-major float in smem
// Q: [M, D] row-major bf16 in smem
// dK: [N, D] output, atomically added to global memory
//
// Optimization: Process output in N-major order (complete all D for each N position)
// This improves L2 cache hit rate by writing to contiguous memory locations
__device__ void compute_dk_wmma_varlen(
    BwdSmemAccessor& smem,
    float* __restrict__ dk,
    int kv_seq_start,
    int n_start,
    int n_end,
    int m_size,
    int num_heads,
    int head_idx,
    int head_dim
) {
    using namespace nvcuda::wmma;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid = threadIdx.x;
    const int n_size = n_end - n_start;
    const int stride_token = num_heads * head_dim;

    // Step 1: Transpose dScores [M, N] -> temp_bf16 as [N, M] in bf16
    __nv_bfloat16* dscores_t = smem.temp_bf16();

    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        int m = idx / n_size;
        int n = idx % n_size;
        dscores_t[n * BWD_BLOCK_M + m] = __float2bfloat16(smem.dscores()[m * BWD_BLOCK_N + n]);
    }
    __syncthreads();

    // Step 2: WMMA computation with N-major tile ordering for cache locality
    // Process output tiles in order: for each N tile, complete all D tiles
    // This groups atomic writes to contiguous memory (same token's head)
    const int n_tiles = (n_size + WMMA_M - 1) / WMMA_M;
    const int d_tiles = (head_dim + WMMA_N - 1) / WMMA_N;
    const int m_tiles = (m_size + WMMA_K - 1) / WMMA_K;

    // Each warp processes tiles in N-major order for better memory locality
    for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
        // Process all D tiles for this N tile
        for (int d_tile = warp_id; d_tile < d_tiles; d_tile += BWD_NUM_WARPS) {
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            fill_fragment(acc, 0.0f);

            for (int k_tile = 0; k_tile < m_tiles; ++k_tile) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag;

                load_matrix_sync(a_frag, dscores_t + n_tile * WMMA_M * BWD_BLOCK_M + k_tile * WMMA_K, BWD_BLOCK_M);
                load_matrix_sync(b_frag, smem.q_tile() + k_tile * WMMA_K * BWD_BLOCK_D + d_tile * WMMA_N, BWD_BLOCK_D);

                mma_sync(acc, a_frag, b_frag, acc);
            }

            float* staging = smem.wmma_staging_warp(warp_id);
            store_matrix_sync(staging, acc, WMMA_N, mem_row_major);
            __syncwarp();

            // Write output in row-major order (contiguous D for each N)
            // This improves memory coalescing and L2 cache utilization
            for (int local_n = 0; local_n < WMMA_M; ++local_n) {
                int global_n = n_tile * WMMA_M + local_n;
                if (global_n >= n_size) continue;

                int global_token = kv_seq_start + n_start + global_n;
                float* dk_row = dk + global_token * stride_token + head_idx * head_dim;

                // Each lane writes contiguous D elements for better coalescing
                for (int local_d = lane_id; local_d < WMMA_N; local_d += 32) {
                    int global_d = d_tile * WMMA_N + local_d;
                    if (global_d < head_dim) {
                        float val = staging[local_n * WMMA_N + local_d];
                        if (val != 0.0f) {
                            atomicAdd(&dk_row[global_d], val);
                        }
                    }
                }
            }
            __syncwarp();
        }
        // REMOVED: __syncthreads() - redundant after atomic writes to global memory
        // Each warp uses its own staging buffer, no cross-warp smem dependency
    }
}

// Compute dV = P^T @ dO using WMMA with tiled output for better cache locality
// P: [M, N] row-major float in smem
// dO: [M, D] row-major bf16 in smem
// dV: [N, D] output, atomically added to global memory
//
// Optimization: Process output in N-major order (complete all D for each N position)
// This improves L2 cache hit rate by writing to contiguous memory locations
__device__ void compute_dv_wmma_varlen(
    BwdSmemAccessor& smem,
    float* __restrict__ dv,
    int kv_seq_start,
    int n_start,
    int n_end,
    int m_size,
    int num_heads,
    int head_idx,
    int head_dim
) {
    using namespace nvcuda::wmma;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid = threadIdx.x;
    const int n_size = n_end - n_start;
    const int stride_token = num_heads * head_dim;

    // Step 1: Transpose P [M, N] -> temp_bf16 as [N, M] in bf16
    __nv_bfloat16* p_t = smem.temp_bf16();

    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        int m = idx / n_size;
        int n = idx % n_size;
        p_t[n * BWD_BLOCK_M + m] = __float2bfloat16(smem.probs()[m * BWD_BLOCK_N + n]);
    }
    __syncthreads();

    // Step 2: WMMA computation with N-major tile ordering for cache locality
    const int n_tiles = (n_size + WMMA_M - 1) / WMMA_M;
    const int d_tiles = (head_dim + WMMA_N - 1) / WMMA_N;
    const int m_tiles = (m_size + WMMA_K - 1) / WMMA_K;

    // Process tiles in N-major order for better memory locality
    for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
        for (int d_tile = warp_id; d_tile < d_tiles; d_tile += BWD_NUM_WARPS) {
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            fill_fragment(acc, 0.0f);

            for (int k_tile = 0; k_tile < m_tiles; ++k_tile) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag;

                load_matrix_sync(a_frag, p_t + n_tile * WMMA_M * BWD_BLOCK_M + k_tile * WMMA_K, BWD_BLOCK_M);
                load_matrix_sync(b_frag, smem.do_tile() + k_tile * WMMA_K * BWD_BLOCK_D + d_tile * WMMA_N, BWD_BLOCK_D);

                mma_sync(acc, a_frag, b_frag, acc);
            }

            float* staging = smem.wmma_staging_warp(warp_id);
            store_matrix_sync(staging, acc, WMMA_N, mem_row_major);
            __syncwarp();

            // Write output in row-major order (contiguous D for each N)
            for (int local_n = 0; local_n < WMMA_M; ++local_n) {
                int global_n = n_tile * WMMA_M + local_n;
                if (global_n >= n_size) continue;

                int global_token = kv_seq_start + n_start + global_n;
                float* dv_row = dv + global_token * stride_token + head_idx * head_dim;

                // Each lane writes contiguous D elements for better coalescing
                for (int local_d = lane_id; local_d < WMMA_N; local_d += 32) {
                    int global_d = d_tile * WMMA_N + local_d;
                    if (global_d < head_dim) {
                        float val = staging[local_n * WMMA_N + local_d];
                        if (val != 0.0f) {
                            atomicAdd(&dv_row[global_d], val);
                        }
                    }
                }
            }
            __syncwarp();
        }
        // REMOVED: __syncthreads() - redundant after atomic writes to global memory
    }
}

// Legacy scalar implementations kept for fallback/comparison
// Atomic add dK to global memory (scalar version)
// dK[n,d] = sum_m(dScores[m,n] * Q[m,d])
__device__ void atomic_add_dk_varlen_scalar(
    BwdSmemAccessor& smem,
    float* __restrict__ dk,
    int kv_seq_start,
    int n_start,
    int n_end,
    int m_size,
    int num_heads,
    int head_idx,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int cols = n_end - n_start;
    const int stride_token = num_heads * head_dim;

    for (int n = 0; n < cols; ++n) {
        for (int d = tid; d < head_dim; d += BWD_NUM_THREADS) {
            float sum = 0.0f;
            for (int m = 0; m < m_size; ++m) {
                float ds = smem.dscores()[m * BWD_BLOCK_N + n];
                float q_val = __bfloat162float(smem.q_tile()[m * BWD_BLOCK_D + d]);
                sum += ds * q_val;
            }
            if (sum != 0.0f) {
                int global_token = kv_seq_start + n_start + n;
                atomicAdd(&dk[global_token * stride_token + head_idx * head_dim + d], sum);
            }
        }
    }
}

// Atomic add dV to global memory (scalar version)
// dV[n,d] = sum_m(P[m,n] * dO[m,d])
__device__ void atomic_add_dv_varlen_scalar(
    BwdSmemAccessor& smem,
    float* __restrict__ dv,
    int kv_seq_start,
    int n_start,
    int n_end,
    int m_size,
    int num_heads,
    int head_idx,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int cols = n_end - n_start;
    const int stride_token = num_heads * head_dim;

    for (int n = 0; n < cols; ++n) {
        for (int d = tid; d < head_dim; d += BWD_NUM_THREADS) {
            float sum = 0.0f;
            for (int m = 0; m < m_size; ++m) {
                float p = smem.probs()[m * BWD_BLOCK_N + n];
                float do_val = __bfloat162float(smem.do_tile()[m * BWD_BLOCK_D + d]);
                sum += p * do_val;
            }
            if (sum != 0.0f) {
                int global_token = kv_seq_start + n_start + n;
                atomicAdd(&dv[global_token * stride_token + head_idx * head_dim + d], sum);
            }
        }
    }
}

// Wrapper functions that use WMMA by default
__device__ __forceinline__ void atomic_add_dk_varlen(
    BwdSmemAccessor& smem,
    float* __restrict__ dk,
    int kv_seq_start,
    int n_start,
    int n_end,
    int m_size,
    int num_heads,
    int head_idx,
    int head_dim
) {
    compute_dk_wmma_varlen(smem, dk, kv_seq_start, n_start, n_end, m_size, num_heads, head_idx, head_dim);
}

__device__ __forceinline__ void atomic_add_dv_varlen(
    BwdSmemAccessor& smem,
    float* __restrict__ dv,
    int kv_seq_start,
    int n_start,
    int n_end,
    int m_size,
    int num_heads,
    int head_idx,
    int head_dim
) {
    compute_dv_wmma_varlen(smem, dv, kv_seq_start, n_start, n_end, m_size, num_heads, head_idx, head_dim);
}

// Zero-initialize all shared memory buffers for WMMA boundary handling
__device__ __forceinline__ void zero_init_smem(BwdSmemAccessor& smem) {
    const int tid = threadIdx.x;

    const __nv_bfloat162 zero_bf16x2 = __floats2bfloat162_rn(0.0f, 0.0f);
    const float4 zero_f4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Zero Q tile (32*128/2 = 2048 elements, 256 threads, 8 iters)
    __nv_bfloat162* q_ptr2 = reinterpret_cast<__nv_bfloat162*>(smem.q_tile());
    #pragma unroll 8
    for (int i = tid; i < (BWD_BLOCK_M * BWD_BLOCK_D) / 2; i += BWD_NUM_THREADS) {
        q_ptr2[i] = zero_bf16x2;
    }

    // Zero both K tile buffers (32*128/2 = 2048 elements, 8 iters)
    __nv_bfloat162* k_ptr2_0 = reinterpret_cast<__nv_bfloat162*>(smem.k_tile_buf(0));
    __nv_bfloat162* k_ptr2_1 = reinterpret_cast<__nv_bfloat162*>(smem.k_tile_buf(1));
    #pragma unroll 8
    for (int i = tid; i < (BWD_BLOCK_N * BWD_BLOCK_D) / 2; i += BWD_NUM_THREADS) {
        k_ptr2_0[i] = zero_bf16x2;
        k_ptr2_1[i] = zero_bf16x2;
    }

    // Zero both V tile buffers
    __nv_bfloat162* v_ptr2_0 = reinterpret_cast<__nv_bfloat162*>(smem.v_tile_buf(0));
    __nv_bfloat162* v_ptr2_1 = reinterpret_cast<__nv_bfloat162*>(smem.v_tile_buf(1));
    #pragma unroll 8
    for (int i = tid; i < (BWD_BLOCK_N * BWD_BLOCK_D) / 2; i += BWD_NUM_THREADS) {
        v_ptr2_0[i] = zero_bf16x2;
        v_ptr2_1[i] = zero_bf16x2;
    }

    // Zero dO tile
    __nv_bfloat162* do_ptr2 = reinterpret_cast<__nv_bfloat162*>(smem.do_tile());
    #pragma unroll 8
    for (int i = tid; i < (BWD_BLOCK_M * BWD_BLOCK_D) / 2; i += BWD_NUM_THREADS) {
        do_ptr2[i] = zero_bf16x2;
    }

    // Zero scores/probs/dscores (32*32/4 = 256 elements, 1 iter)
    float4* scores_ptr4 = reinterpret_cast<float4*>(smem.scores());
    float4* probs_ptr4 = reinterpret_cast<float4*>(smem.probs());
    float4* dscores_ptr4 = reinterpret_cast<float4*>(smem.dscores());
    #pragma unroll 1
    for (int i = tid; i < (BWD_BLOCK_M * BWD_BLOCK_N) / 4; i += BWD_NUM_THREADS) {
        scores_ptr4[i] = zero_f4;
        probs_ptr4[i] = zero_f4;
        dscores_ptr4[i] = zero_f4;
    }

    // Zero LSE and delta (32 elements, 1 iter for most threads)
    float* lse_ptr = smem.lse();
    float* delta_ptr = smem.delta();
    for (int i = tid; i < BWD_BLOCK_M; i += BWD_NUM_THREADS) {
        lse_ptr[i] = 0.0f;
        delta_ptr[i] = 0.0f;
    }

    // Zero O tile
    __nv_bfloat162* o_ptr2 = reinterpret_cast<__nv_bfloat162*>(smem.o_tile());
    #pragma unroll 8
    for (int i = tid; i < (BWD_BLOCK_M * BWD_BLOCK_D) / 2; i += BWD_NUM_THREADS) {
        o_ptr2[i] = zero_bf16x2;
    }

    // Zero dQ accumulator (32*128/4 = 1024 elements, 4 iters)
    float4* dq_ptr4 = reinterpret_cast<float4*>(smem.dq_acc());
    #pragma unroll 4
    for (int i = tid; i < (BWD_BLOCK_M * BWD_BLOCK_D) / 4; i += BWD_NUM_THREADS) {
        dq_ptr4[i] = zero_f4;
    }

    __syncthreads();
}

// Main backward kernel with double-buffered K/V tiles and cp.async prefetching
template<bool kIsCausal>
__global__ void __launch_bounds__(BWD_NUM_THREADS)
fmha_bwd_sm120_varlen_kernel(
    const __nv_bfloat16* __restrict__ d_o,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ o,
    const float* __restrict__ lse,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ cu_seqlens_kv,
    __nv_bfloat16* __restrict__ dq,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int num_heads,
    int head_dim,
    float scale,
    int max_seqlen_q,
    int max_seqlen_kv
) {
    extern __shared__ char smem_base[];
    BwdSmemAccessor smem;
    smem.base = smem_base;

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int m_block_idx = blockIdx.x;

    const int q_start = cu_seqlens_q[batch_idx];
    const int q_end = cu_seqlens_q[batch_idx + 1];
    const int kv_start = cu_seqlens_kv[batch_idx];
    const int kv_end = cu_seqlens_kv[batch_idx + 1];

    const int seq_len_q = q_end - q_start;
    const int seq_len_kv = kv_end - kv_start;

    if (head_idx >= num_heads) return;

    const int m_start = m_block_idx * BWD_BLOCK_M;
    const int m_end = min(m_start + BWD_BLOCK_M, seq_len_q);
    if (m_start >= seq_len_q) return;

    const int m_size = m_end - m_start;

    // Zero-initialize all shared memory
    zero_init_smem(smem);

    // Load Q, dO, O tiles
    load_q_tile_varlen(smem, q, q_start, m_start, m_end, num_heads, head_idx, head_dim);
    load_do_tile_varlen(smem, d_o, q_start, m_start, m_end, num_heads, head_idx, head_dim);
    load_o_tile_varlen(smem, o, q_start, m_start, m_end, num_heads, head_idx, head_dim);
    load_lse_varlen(smem, lse, q_start, m_start, m_end, num_heads, head_idx);
    __syncthreads();

    // Precompute delta = O dot dO
    compute_delta(smem, m_size, head_dim);

    const int num_kv_blocks = (seq_len_kv + BWD_BLOCK_N - 1) / BWD_BLOCK_N;

    if (num_kv_blocks == 0) {
        write_dq_varlen(smem, dq, q_start, m_start, m_end, num_heads, head_idx, head_dim);
        return;
    }

    // ========================================================================
    // DOUBLE-BUFFERED KV BLOCK LOOP WITH CP.ASYNC PREFETCHING
    // ========================================================================
    // Strategy:
    // 1. Prefetch first KV block into buffer 0
    // 2. For each block: wait for current buffer, prefetch next into other buffer, compute
    // 3. This overlaps memory transfers with compute for better latency hiding
    // ========================================================================

    int curr_buf = 0;
    int next_buf = 1;

    // Find first valid KV block (respecting causal mask)
    int first_valid_kv_block = 0;
    if (kIsCausal) {
        // Find first block that has at least one valid key position
        for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
            const int n_start_check = kv_block * BWD_BLOCK_N;
            if (n_start_check <= (m_start + m_size - 1)) {
                first_valid_kv_block = kv_block;
                break;
            }
        }
    }

    // Prefetch first valid KV block into buffer 0
    {
        const int n_start_0 = first_valid_kv_block * BWD_BLOCK_N;
        const int n_end_0 = min(n_start_0 + BWD_BLOCK_N, seq_len_kv);

        async_load_k_tile_varlen(smem, k, kv_start, n_start_0, n_end_0, num_heads, head_idx, head_dim, curr_buf);
        async_load_v_tile_varlen(smem, v, kv_start, n_start_0, n_end_0, num_heads, head_idx, head_dim, curr_buf);
        cp_async_commit_group();
    }

    for (int kv_block = first_valid_kv_block; kv_block < num_kv_blocks; ++kv_block) {
        const int n_start = kv_block * BWD_BLOCK_N;
        const int n_end = min(n_start + BWD_BLOCK_N, seq_len_kv);
        const int n_size = n_end - n_start;

        // Skip if causal and this KV block is entirely in the future
        if (kIsCausal && n_start > (m_start + m_size - 1)) {
            break;  // No more valid blocks
        }

        // Wait for current buffer's data to be ready
        cp_async_wait_all();
        __syncthreads();

        // Prefetch next block into other buffer (if exists and valid)
        bool prefetch_next = false;
        int next_kv_block = kv_block + 1;
        if (next_kv_block < num_kv_blocks) {
            const int next_n_start = next_kv_block * BWD_BLOCK_N;
            if (!kIsCausal || next_n_start <= (m_start + m_size - 1)) {
                const int next_n_end = min(next_n_start + BWD_BLOCK_N, seq_len_kv);

                // Zero next buffer before async load
                zero_kv_buffer(smem, next_buf);
                __syncthreads();

                async_load_k_tile_varlen(smem, k, kv_start, next_n_start, next_n_end, num_heads, head_idx, head_dim, next_buf);
                async_load_v_tile_varlen(smem, v, kv_start, next_n_start, next_n_end, num_heads, head_idx, head_dim, next_buf);
                cp_async_commit_group();
                prefetch_next = true;
            }
        }

        // Zero scores/probs/dscores for this block
        const int tid = threadIdx.x;
        const float4 zero_f4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4* scores_ptr4 = reinterpret_cast<float4*>(smem.scores());
        float4* probs_ptr4 = reinterpret_cast<float4*>(smem.probs());
        float4* dscores_ptr4 = reinterpret_cast<float4*>(smem.dscores());
        for (int i = tid; i < (BWD_BLOCK_M * BWD_BLOCK_N) / 4; i += BWD_NUM_THREADS) {
            scores_ptr4[i] = zero_f4;
            probs_ptr4[i] = zero_f4;
            dscores_ptr4[i] = zero_f4;
        }
        __syncthreads();

        // Compute on current buffer (overlaps with prefetch of next)
        compute_qk_scores_wmma_buf(smem, m_size, n_size, head_dim, scale, curr_buf);
        recompute_softmax(smem, m_size, n_size, kIsCausal, m_start, n_start);
        compute_dp_wmma_buf(smem, m_size, n_size, head_dim, curr_buf);
        compute_dscores(smem, m_size, n_size, scale);
        compute_dq_wmma_buf(smem, m_size, n_size, head_dim, curr_buf);

        // Atomic add dK and dV
        atomic_add_dk_varlen(smem, dk, kv_start, n_start, n_end, m_size, num_heads, head_idx, head_dim);
        atomic_add_dv_varlen(smem, dv, kv_start, n_start, n_end, m_size, num_heads, head_idx, head_dim);

        __syncthreads();

        // Swap buffers for next iteration
        if (prefetch_next) {
            int tmp = curr_buf;
            curr_buf = next_buf;
            next_buf = tmp;
        }
    }

    // Write final dQ
    write_dq_varlen(smem, dq, q_start, m_start, m_end, num_heads, head_idx, head_dim);
}

// ============================================================================
// K-MAJOR BACKWARD KERNEL (ELIMINATES dK/dV ATOMICS!)
// ============================================================================
// This kernel uses K-major loop ordering where each block owns one K-block
// and loops over all Q-blocks, accumulating dK/dV in shared memory.
// This eliminates atomic operations for dK/dV (the main bottleneck).
//
// Grid: (n_blocks, num_heads, batch_size) - one block per K-block
// Each block:
//   1. Loads K, V once (fixed for this block)
//   2. Loops over all Q-blocks
//   3. Accumulates dK, dV in shared memory (NO ATOMICS!)
//   4. Uses atomicAdd for dQ (still needed but much faster than dK/dV atomics)
//   5. Writes dK, dV to global memory at the end (single non-atomic write)
// ============================================================================

// Accumulate dK to shared memory accumulator (no atomics!)
// dK_acc[n, d] += dScores[m, n]^T @ Q[m, d]
__device__ void accumulate_dk_to_smem(
    BwdSmemAccessor& smem,
    int m_size,
    int n_size,
    int head_dim
) {
    using namespace nvcuda::wmma;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid = threadIdx.x;

    // Transpose dScores [M, N] -> temp_bf16 as [N, M] in bf16
    __nv_bfloat16* dscores_t = smem.temp_bf16();

    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        int m = idx / n_size;
        int n = idx % n_size;
        dscores_t[n * BWD_BLOCK_M + m] = __float2bfloat16(smem.dscores()[m * BWD_BLOCK_N + n]);
    }
    __syncthreads();

    // WMMA: dK[N, D] += dScores_T[N, M] @ Q[M, D]
    const int n_tiles = (n_size + WMMA_M - 1) / WMMA_M;
    const int d_tiles = (head_dim + WMMA_N - 1) / WMMA_N;
    const int m_tiles = (m_size + WMMA_K - 1) / WMMA_K;

    for (int nd = warp_id; nd < n_tiles * d_tiles; nd += BWD_NUM_WARPS) {
        int n_tile = nd / d_tiles;
        int d_tile = nd % d_tiles;

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        fill_fragment(acc, 0.0f);

        for (int k_tile = 0; k_tile < m_tiles; ++k_tile) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag;

            load_matrix_sync(a_frag, dscores_t + n_tile * WMMA_M * BWD_BLOCK_M + k_tile * WMMA_K, BWD_BLOCK_M);
            load_matrix_sync(b_frag, smem.q_tile() + k_tile * WMMA_K * BWD_BLOCK_D + d_tile * WMMA_N, BWD_BLOCK_D);

            mma_sync(acc, a_frag, b_frag, acc);
        }

        // Store to staging and add to dk_acc (non-atomic add to smem)
        float* staging = smem.wmma_staging_warp(warp_id);
        store_matrix_sync(staging, acc, WMMA_N, mem_row_major);
        __syncwarp();

        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int local_n = i / WMMA_N;
            int local_d = i % WMMA_N;
            int global_n = n_tile * WMMA_M + local_n;
            int global_d = d_tile * WMMA_N + local_d;

            if (global_n < n_size && global_d < head_dim) {
                float val = staging[i];
                if (val != 0.0f) {
                    smem.dk_acc()[global_n * BWD_BLOCK_D + global_d] += val;
                }
            }
        }
        __syncwarp();
    }
    __syncthreads();
}

// Accumulate dV to shared memory accumulator (no atomics!)
// dV_acc[n, d] += P[m, n]^T @ dO[m, d]
__device__ void accumulate_dv_to_smem(
    BwdSmemAccessor& smem,
    int m_size,
    int n_size,
    int head_dim
) {
    using namespace nvcuda::wmma;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid = threadIdx.x;

    // Transpose P [M, N] -> temp_bf16 as [N, M] in bf16
    __nv_bfloat16* p_t = smem.temp_bf16();

    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        int m = idx / n_size;
        int n = idx % n_size;
        p_t[n * BWD_BLOCK_M + m] = __float2bfloat16(smem.probs()[m * BWD_BLOCK_N + n]);
    }
    __syncthreads();

    // WMMA: dV[N, D] += P_T[N, M] @ dO[M, D]
    const int n_tiles = (n_size + WMMA_M - 1) / WMMA_M;
    const int d_tiles = (head_dim + WMMA_N - 1) / WMMA_N;
    const int m_tiles = (m_size + WMMA_K - 1) / WMMA_K;

    for (int nd = warp_id; nd < n_tiles * d_tiles; nd += BWD_NUM_WARPS) {
        int n_tile = nd / d_tiles;
        int d_tile = nd % d_tiles;

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        fill_fragment(acc, 0.0f);

        for (int k_tile = 0; k_tile < m_tiles; ++k_tile) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag;

            load_matrix_sync(a_frag, p_t + n_tile * WMMA_M * BWD_BLOCK_M + k_tile * WMMA_K, BWD_BLOCK_M);
            load_matrix_sync(b_frag, smem.do_tile() + k_tile * WMMA_K * BWD_BLOCK_D + d_tile * WMMA_N, BWD_BLOCK_D);

            mma_sync(acc, a_frag, b_frag, acc);
        }

        // Store to staging and add to dv_acc (non-atomic add to smem)
        float* staging = smem.wmma_staging_warp(warp_id);
        store_matrix_sync(staging, acc, WMMA_N, mem_row_major);
        __syncwarp();

        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int local_n = i / WMMA_N;
            int local_d = i % WMMA_N;
            int global_n = n_tile * WMMA_M + local_n;
            int global_d = d_tile * WMMA_N + local_d;

            if (global_n < n_size && global_d < head_dim) {
                float val = staging[i];
                if (val != 0.0f) {
                    smem.dv_acc()[global_n * BWD_BLOCK_D + global_d] += val;
                }
            }
        }
        __syncwarp();
    }
    __syncthreads();
}

// atomicAdd dQ to global memory (for K-major kernel)
__device__ void atomic_add_dq_to_global(
    BwdSmemAccessor& smem,
    __nv_bfloat16* __restrict__ dq,
    int q_seq_start,
    int m_start,
    int m_end,
    int num_heads,
    int head_idx,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int m_size = m_end - m_start;
    const int stride_token = num_heads * head_dim;

    // Convert and atomicAdd dq_acc to global dq
    for (int idx = tid; idx < m_size * head_dim; idx += BWD_NUM_THREADS) {
        int m = idx / head_dim;
        int d = idx % head_dim;

        float val = smem.dq_acc()[m * BWD_BLOCK_D + d];
        if (val != 0.0f) {
            int global_token = q_seq_start + m_start + m;
            // atomicAdd to bf16 via convert-atomic-convert
            // Note: This is less efficient than float atomics, but dQ atomics
            // are much less frequent than dK/dV atomics in the Q-major kernel
            float* dq_float = reinterpret_cast<float*>(&dq[(global_token * stride_token + head_idx * head_dim) / 2]);
            // Use bf16 atomics via half2
            __nv_bfloat162* dq_bf16x2 = reinterpret_cast<__nv_bfloat162*>(
                &dq[global_token * stride_token + head_idx * head_dim + (d & ~1)]);
            if (d % 2 == 0) {
                __nv_bfloat162 old_val = *dq_bf16x2;
                __nv_bfloat162 add_val = __floats2bfloat162_rn(val, 0.0f);
                // Simple atomic via CAS loop would be complex, so use float accumulator approach
            }
            // Simpler: just use the float dq buffer that gets converted later
            // The caller should provide float dq for atomics
        }
    }
}

// Write dK/dV accumulators from smem to global memory (no atomics!)
__device__ void write_dk_dv_from_smem(
    BwdSmemAccessor& smem,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int kv_seq_start,
    int n_start,
    int n_end,
    int num_heads,
    int head_idx,
    int head_dim,
    float scale
) {
    const int tid = threadIdx.x;
    const int n_size = n_end - n_start;
    const int stride_token = num_heads * head_dim;

    // Write dk_acc to global dk (with scale for dK)
    for (int idx = tid; idx < n_size * head_dim; idx += BWD_NUM_THREADS) {
        int n = idx / head_dim;
        int d = idx % head_dim;

        int global_token = kv_seq_start + n_start + n;
        int global_idx = global_token * stride_token + head_idx * head_dim + d;

        // dK needs scale factor applied
        dk[global_idx] = smem.dk_acc()[n * BWD_BLOCK_D + d] * scale;
        dv[global_idx] = smem.dv_acc()[n * BWD_BLOCK_D + d];
    }
}

// ============================================================================
// K-MAJOR SMEM-EFFICIENT BACKWARD KERNEL (N=16 tiles to fit 99KB limit)
// ============================================================================
// This kernel uses K-major loop ordering with KMAJOR_BLOCK_N=16 to fit within
// SM120's 99KB shared memory limit while still eliminating dK/dV atomics.
//
// Grid: (n_blocks, num_heads, batch_size) - one block per K-tile (16 KV positions)
// Each block:
//   1. Loads K, V once (N=16 tile, fixed for this block)
//   2. Loops over all Q-blocks (M=32)
//   3. Accumulates dK, dV in shared memory (NO ATOMICS!)
//   4. Uses atomicAdd for dQ (still needed but much faster than dK/dV atomics)
//   5. Writes dK, dV to global memory at the end (single non-atomic write)
// ============================================================================

// K-major specific load helpers using KMajorSmemAccessor
__device__ __forceinline__ void kmajor_load_k_tile(
    KMajorSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ k,
    int kv_seq_start,
    int n_start,
    int n_end,
    int num_heads,
    int head_idx,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int cols = n_end - n_start;
    const int stride_token = num_heads * head_dim;

    for (int n = warp_id; n < cols; n += num_warps) {
        const int global_token = kv_seq_start + n_start + n;
        const __nv_bfloat16* src_row = k + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.k_tile() + n * BWD_BLOCK_D;

        for (int d = lane_id * 2; d < head_dim; d += 64) {
            if (d + 1 < head_dim) {
                __nv_bfloat162 val = *reinterpret_cast<const __nv_bfloat162*>(src_row + d);
                *reinterpret_cast<__nv_bfloat162*>(dst_row + d) = val;
            } else if (d < head_dim) {
                dst_row[d] = src_row[d];
            }
        }
    }
}

__device__ __forceinline__ void kmajor_load_v_tile(
    KMajorSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ v,
    int kv_seq_start,
    int n_start,
    int n_end,
    int num_heads,
    int head_idx,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int cols = n_end - n_start;
    const int stride_token = num_heads * head_dim;

    for (int n = warp_id; n < cols; n += num_warps) {
        const int global_token = kv_seq_start + n_start + n;
        const __nv_bfloat16* src_row = v + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.v_tile() + n * BWD_BLOCK_D;

        for (int d = lane_id * 2; d < head_dim; d += 64) {
            if (d + 1 < head_dim) {
                __nv_bfloat162 val = *reinterpret_cast<const __nv_bfloat162*>(src_row + d);
                *reinterpret_cast<__nv_bfloat162*>(dst_row + d) = val;
            } else if (d < head_dim) {
                dst_row[d] = src_row[d];
            }
        }
    }
}

__device__ __forceinline__ void kmajor_load_q_tile(
    KMajorSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ q,
    int q_seq_start,
    int m_start,
    int m_end,
    int num_heads,
    int head_idx,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int rows = m_end - m_start;
    const int stride_token = num_heads * head_dim;

    for (int m = warp_id; m < rows; m += num_warps) {
        const int global_token = q_seq_start + m_start + m;
        const __nv_bfloat16* src_row = q + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.q_tile() + m * BWD_BLOCK_D;

        for (int d = lane_id * 2; d < head_dim; d += 64) {
            if (d + 1 < head_dim) {
                __nv_bfloat162 val = *reinterpret_cast<const __nv_bfloat162*>(src_row + d);
                *reinterpret_cast<__nv_bfloat162*>(dst_row + d) = val;
            } else if (d < head_dim) {
                dst_row[d] = src_row[d];
            }
        }
    }
}

__device__ __forceinline__ void kmajor_load_do_tile(
    KMajorSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ d_o,
    int q_seq_start,
    int m_start,
    int m_end,
    int num_heads,
    int head_idx,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int rows = m_end - m_start;
    const int stride_token = num_heads * head_dim;

    for (int m = warp_id; m < rows; m += num_warps) {
        const int global_token = q_seq_start + m_start + m;
        const __nv_bfloat16* src_row = d_o + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.do_tile() + m * BWD_BLOCK_D;

        for (int d = lane_id * 2; d < head_dim; d += 64) {
            if (d + 1 < head_dim) {
                __nv_bfloat162 val = *reinterpret_cast<const __nv_bfloat162*>(src_row + d);
                *reinterpret_cast<__nv_bfloat162*>(dst_row + d) = val;
            } else if (d < head_dim) {
                dst_row[d] = src_row[d];
            }
        }
    }
}

__device__ __forceinline__ void kmajor_load_o_tile(
    KMajorSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ o,
    int q_seq_start,
    int m_start,
    int m_end,
    int num_heads,
    int head_idx,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int rows = m_end - m_start;
    const int stride_token = num_heads * head_dim;

    for (int m = warp_id; m < rows; m += num_warps) {
        const int global_token = q_seq_start + m_start + m;
        const __nv_bfloat16* src_row = o + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.o_tile() + m * BWD_BLOCK_D;

        for (int d = lane_id * 2; d < head_dim; d += 64) {
            if (d + 1 < head_dim) {
                __nv_bfloat162 val = *reinterpret_cast<const __nv_bfloat162*>(src_row + d);
                *reinterpret_cast<__nv_bfloat162*>(dst_row + d) = val;
            } else if (d < head_dim) {
                dst_row[d] = src_row[d];
            }
        }
    }
}

__device__ __forceinline__ void kmajor_load_lse(
    KMajorSmemAccessor& smem,
    const float* __restrict__ lse,
    int q_seq_start,
    int m_start,
    int m_end,
    int num_heads,
    int head_idx
) {
    const int tid = threadIdx.x;
    const int rows = m_end - m_start;

    for (int m = tid; m < rows; m += BWD_NUM_THREADS) {
        int global_token = q_seq_start + m_start + m;
        smem.lse()[m] = lse[global_token * num_heads + head_idx];
    }
}

// K-major specific compute functions using KMajorSmemAccessor
__device__ __forceinline__ void kmajor_compute_delta(
    KMajorSmemAccessor& smem,
    int m_size,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Each warp handles one or more rows
    for (int m = warp_id; m < m_size; m += BWD_NUM_WARPS) {
        __nv_bfloat16* o_row = smem.o_row(m);
        __nv_bfloat16* do_row = smem.do_row(m);

        float sum = 0.0f;
        for (int d = lane_id * 2; d < head_dim; d += 64) {
            if (d + 1 < head_dim) {
                __nv_bfloat162 o_val = *reinterpret_cast<__nv_bfloat162*>(o_row + d);
                __nv_bfloat162 do_val = *reinterpret_cast<__nv_bfloat162*>(do_row + d);
                float o_lo = __bfloat162float(__low2bfloat16(o_val));
                float o_hi = __bfloat162float(__high2bfloat16(o_val));
                float do_lo = __bfloat162float(__low2bfloat16(do_val));
                float do_hi = __bfloat162float(__high2bfloat16(do_val));
                sum += o_lo * do_lo + o_hi * do_hi;
            } else if (d < head_dim) {
                sum += __bfloat162float(o_row[d]) * __bfloat162float(do_row[d]);
            }
        }

        // Warp reduce
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            smem.delta()[m] = sum;
        }
    }
    __syncthreads();
}

// K-major QK scores computation with N=KMAJOR_BLOCK_N
// OPTIMIZED: Fuse scale into accumulator, use power-of-2 index math
__device__ void kmajor_compute_qk_scores_wmma(
    KMajorSmemAccessor& smem,
    int m_size,
    int n_size,
    int head_dim,
    float scale
) {
    using namespace nvcuda::wmma;

    const int warp_id = threadIdx.x / 32;
    const int tid = threadIdx.x;

    const int m_tiles = (m_size + WMMA_M - 1) / WMMA_M;
    const int n_tiles = (n_size + WMMA_N - 1) / WMMA_N;
    const int d_tiles = (head_dim + WMMA_K - 1) / WMMA_K;

    for (int mn = warp_id; mn < m_tiles * n_tiles; mn += BWD_NUM_WARPS) {
        int m_tile = mn / n_tiles;
        int n_tile = mn % n_tiles;

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        fill_fragment(acc, 0.0f);

        #pragma unroll 4
        for (int k_tile = 0; k_tile < d_tiles; ++k_tile) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;

            load_matrix_sync(a_frag, smem.q_row(m_tile * WMMA_M) + k_tile * WMMA_K, BWD_BLOCK_D);
            load_matrix_sync(b_frag, smem.k_row(n_tile * WMMA_N) + k_tile * WMMA_K, BWD_BLOCK_D);

            mma_sync(acc, a_frag, b_frag, acc);
        }

        // OPTIMIZED: Apply scale directly to accumulator fragments before store
        #pragma unroll
        for (int i = 0; i < acc.num_elements; ++i) {
            acc.x[i] *= scale;
        }

        float* scores_ptr = smem.scores_row(m_tile * WMMA_M) + n_tile * WMMA_N;
        store_matrix_sync(scores_ptr, acc, KMAJOR_BLOCK_N, mem_row_major);
    }
    __syncthreads();
}

// K-major softmax recomputation
// OPTIMIZED: Use power-of-2 index math for N=16
__device__ void kmajor_recompute_softmax(
    KMajorSmemAccessor& smem,
    int m_size,
    int n_size,
    bool is_causal,
    int m_start_global,
    int n_start_global
) {
    const int tid = threadIdx.x;
    // KMAJOR_BLOCK_N=16 is power of 2
    constexpr int N_SHIFT = 4;  // log2(16)
    constexpr int N_MASK = KMAJOR_BLOCK_N - 1;

    #pragma unroll 4
    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        int m = idx >> N_SHIFT;
        int n = idx & N_MASK;

        if (m < m_size && n < n_size) {
            float score = smem.scores()[idx];  // Already in row-major with KMAJOR_BLOCK_N stride
            float lse_m = smem.lse()[m];

            bool masked = is_causal && ((m_start_global + m) < (n_start_global + n));

            float prob = masked ? 0.0f : expf(score - lse_m);
            smem.probs()[idx] = prob;
        }
    }
    __syncthreads();
}

// K-major dP = dO @ V^T computation
// OPTIMIZED: Added unroll pragma for k_tile loop
__device__ void kmajor_compute_dp_wmma(
    KMajorSmemAccessor& smem,
    int m_size,
    int n_size,
    int head_dim
) {
    using namespace nvcuda::wmma;

    const int warp_id = threadIdx.x / 32;

    const int m_tiles = (m_size + WMMA_M - 1) / WMMA_M;
    const int n_tiles = (n_size + WMMA_N - 1) / WMMA_N;
    const int d_tiles = (head_dim + WMMA_K - 1) / WMMA_K;

    for (int mn = warp_id; mn < m_tiles * n_tiles; mn += BWD_NUM_WARPS) {
        int m_tile = mn / n_tiles;
        int n_tile = mn % n_tiles;

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        fill_fragment(acc, 0.0f);

        #pragma unroll 4
        for (int k_tile = 0; k_tile < d_tiles; ++k_tile) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> do_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> v_frag;

            load_matrix_sync(do_frag, smem.do_row(m_tile * WMMA_M) + k_tile * WMMA_K, BWD_BLOCK_D);
            load_matrix_sync(v_frag, smem.v_row(n_tile * WMMA_N) + k_tile * WMMA_K, BWD_BLOCK_D);

            mma_sync(acc, do_frag, v_frag, acc);
        }

        float* dscores_ptr = smem.dscores_row(m_tile * WMMA_M) + n_tile * WMMA_N;
        store_matrix_sync(dscores_ptr, acc, KMAJOR_BLOCK_N, mem_row_major);
    }
    __syncthreads();
}

// K-major dScores = (dP - delta) * P * scale
// OPTIMIZED: Use power-of-2 index math for N=16
__device__ void kmajor_compute_dscores(
    KMajorSmemAccessor& smem,
    int m_size,
    int n_size,
    float scale
) {
    const int tid = threadIdx.x;
    constexpr int N_SHIFT = 4;  // log2(16)
    constexpr int N_MASK = KMAJOR_BLOCK_N - 1;

    #pragma unroll 4
    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        int m = idx >> N_SHIFT;
        int n = idx & N_MASK;

        if (m < m_size && n < n_size) {
            float dp = smem.dscores()[idx];
            float p = smem.probs()[idx];
            float delta_m = smem.delta()[m];
            smem.dscores()[idx] = (dp - delta_m) * p * scale;
        }
    }
    __syncthreads();
}

// K-major dQ = dScores @ K computation
// OPTIMIZED: Power-of-2 index math and unroll pragmas
__device__ void kmajor_compute_dq_wmma(
    KMajorSmemAccessor& smem,
    int m_size,
    int n_size,
    int head_dim
) {
    using namespace nvcuda::wmma;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid = threadIdx.x;
    constexpr int N_SHIFT = 4;  // log2(16)
    constexpr int N_MASK = KMAJOR_BLOCK_N - 1;

    __nv_bfloat16* dscores_bf16 = smem.temp_bf16();

    // Convert dscores to bf16 - use power-of-2 math
    #pragma unroll 4
    for (int idx = tid; idx < BWD_BLOCK_M * KMAJOR_BLOCK_N; idx += BWD_NUM_THREADS) {
        dscores_bf16[idx] = __float2bfloat16(0.0f);
    }
    __syncthreads();

    #pragma unroll 4
    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        int m = idx >> N_SHIFT;
        int n = idx & N_MASK;
        if (m < m_size && n < n_size) {
            dscores_bf16[idx] = __float2bfloat16(smem.dscores()[idx]);
        }
    }
    __syncthreads();

    const int m_tiles = (m_size + WMMA_M - 1) / WMMA_M;
    const int d_tiles = (head_dim + WMMA_N - 1) / WMMA_N;
    const int n_tiles = (n_size + WMMA_K - 1) / WMMA_K;

    for (int md = warp_id; md < m_tiles * d_tiles; md += BWD_NUM_WARPS) {
        int m_tile = md / d_tiles;
        int d_tile = md % d_tiles;

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        fill_fragment(acc, 0.0f);

        #pragma unroll
        for (int k_tile = 0; k_tile < n_tiles; ++k_tile) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> ds_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> k_frag;

            load_matrix_sync(ds_frag, dscores_bf16 + m_tile * WMMA_M * KMAJOR_BLOCK_N + k_tile * WMMA_K, KMAJOR_BLOCK_N);
            load_matrix_sync(k_frag, smem.k_row(k_tile * WMMA_K) + d_tile * WMMA_N, BWD_BLOCK_D);

            mma_sync(acc, ds_frag, k_frag, acc);
        }

        float* staging = smem.wmma_staging_warp(warp_id);
        store_matrix_sync(staging, acc, WMMA_N, mem_row_major);
        __syncwarp();

        // WMMA_N=16 is power of 2
        constexpr int WMMA_N_SHIFT = 4;
        constexpr int WMMA_N_MASK = WMMA_N - 1;
        #pragma unroll 8
        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int row = i >> WMMA_N_SHIFT;
            int col = i & WMMA_N_MASK;
            int global_row = m_tile * WMMA_M + row;
            int global_col = d_tile * WMMA_N + col;
            if (global_row < m_size && global_col < head_dim) {
                smem.dq_acc()[global_row * BWD_BLOCK_D + global_col] += staging[i];
            }
        }
        __syncwarp();
    }
    __syncthreads();
}

// K-major accumulate dK to smem
__device__ void kmajor_accumulate_dk_to_smem(
    KMajorSmemAccessor& smem,
    int m_size,
    int n_size,
    int head_dim
) {
    using namespace nvcuda::wmma;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid = threadIdx.x;

    // Transpose dScores [M, N] -> temp_bf16 as [N, M]
    // OPTIMIZED: No pre-zeroing, direct transpose with fast index math
    __nv_bfloat16* dscores_t = smem.temp_bf16();

    // Fast transpose: use bit operations for power-of-2 N=16
    // idx = m * N + n, so m = idx >> 4, n = idx & 15
    constexpr int N_SHIFT = 4;  // log2(KMAJOR_BLOCK_N=16)
    constexpr int N_MASK = KMAJOR_BLOCK_N - 1;

    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        int m = idx >> N_SHIFT;
        int n = idx & N_MASK;
        // Store transposed: dscores_t[n][m] = dscores[m][n]
        dscores_t[n * BWD_BLOCK_M + m] = __float2bfloat16(smem.dscores()[idx]);
    }
    // Zero remaining entries only if m_size < BWD_BLOCK_M
    if (m_size < BWD_BLOCK_M) {
        for (int idx = tid + m_size * n_size; idx < BWD_BLOCK_M * KMAJOR_BLOCK_N; idx += BWD_NUM_THREADS) {
            int m = idx >> N_SHIFT;
            int n = idx & N_MASK;
            dscores_t[n * BWD_BLOCK_M + m] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // dK = dScores^T @ Q
    const int n_tiles = (n_size + WMMA_M - 1) / WMMA_M;
    const int d_tiles = (head_dim + WMMA_N - 1) / WMMA_N;
    const int m_tiles = (m_size + WMMA_K - 1) / WMMA_K;

    for (int nd = warp_id; nd < n_tiles * d_tiles; nd += BWD_NUM_WARPS) {
        int n_tile = nd / d_tiles;
        int d_tile = nd % d_tiles;

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        fill_fragment(acc, 0.0f);

        for (int k_tile = 0; k_tile < m_tiles; ++k_tile) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag;

            load_matrix_sync(a_frag, dscores_t + n_tile * WMMA_M * BWD_BLOCK_M + k_tile * WMMA_K, BWD_BLOCK_M);
            load_matrix_sync(b_frag, smem.q_row(k_tile * WMMA_K) + d_tile * WMMA_N, BWD_BLOCK_D);

            mma_sync(acc, a_frag, b_frag, acc);
        }

        float* staging = smem.wmma_staging_warp(warp_id);
        store_matrix_sync(staging, acc, WMMA_N, mem_row_major);
        __syncwarp();

        // OPTIMIZED: Unconditional accumulation (removes branch divergence)
        // Write with coalesced pattern: threads in warp write consecutive d values
        const int global_n_base = n_tile * WMMA_M;
        const int global_d_base = d_tile * WMMA_N;
        #pragma unroll
        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int local_n = i >> 4;  // i / 16
            int local_d = i & 15;  // i % 16
            int global_n = global_n_base + local_n;
            int global_d = global_d_base + local_d;

            if (global_n < n_size && global_d < head_dim) {
                smem.dk_acc()[global_n * BWD_BLOCK_D + global_d] += staging[i];
            }
        }
        __syncwarp();
    }
    __syncthreads();
}

// K-major accumulate dV to smem
__device__ void kmajor_accumulate_dv_to_smem(
    KMajorSmemAccessor& smem,
    int m_size,
    int n_size,
    int head_dim
) {
    using namespace nvcuda::wmma;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid = threadIdx.x;

    // Transpose P [M, N] -> temp_bf16 as [N, M]
    // OPTIMIZED: No pre-zeroing, direct transpose with fast index math
    __nv_bfloat16* p_t = smem.temp_bf16();

    // Fast transpose: use bit operations for power-of-2 N=16
    constexpr int N_SHIFT = 4;  // log2(KMAJOR_BLOCK_N=16)
    constexpr int N_MASK = KMAJOR_BLOCK_N - 1;

    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        int m = idx >> N_SHIFT;
        int n = idx & N_MASK;
        p_t[n * BWD_BLOCK_M + m] = __float2bfloat16(smem.probs()[idx]);
    }
    // Zero remaining entries only if m_size < BWD_BLOCK_M
    if (m_size < BWD_BLOCK_M) {
        for (int idx = tid + m_size * n_size; idx < BWD_BLOCK_M * KMAJOR_BLOCK_N; idx += BWD_NUM_THREADS) {
            int m = idx >> N_SHIFT;
            int n = idx & N_MASK;
            p_t[n * BWD_BLOCK_M + m] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // dV = P^T @ dO
    const int n_tiles = (n_size + WMMA_M - 1) / WMMA_M;
    const int d_tiles = (head_dim + WMMA_N - 1) / WMMA_N;
    const int m_tiles = (m_size + WMMA_K - 1) / WMMA_K;

    for (int nd = warp_id; nd < n_tiles * d_tiles; nd += BWD_NUM_WARPS) {
        int n_tile = nd / d_tiles;
        int d_tile = nd % d_tiles;

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        fill_fragment(acc, 0.0f);

        for (int k_tile = 0; k_tile < m_tiles; ++k_tile) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag;

            load_matrix_sync(a_frag, p_t + n_tile * WMMA_M * BWD_BLOCK_M + k_tile * WMMA_K, BWD_BLOCK_M);
            load_matrix_sync(b_frag, smem.do_row(k_tile * WMMA_K) + d_tile * WMMA_N, BWD_BLOCK_D);

            mma_sync(acc, a_frag, b_frag, acc);
        }

        float* staging = smem.wmma_staging_warp(warp_id);
        store_matrix_sync(staging, acc, WMMA_N, mem_row_major);
        __syncwarp();

        // OPTIMIZED: Unconditional accumulation (removes branch divergence)
        const int global_n_base = n_tile * WMMA_M;
        const int global_d_base = d_tile * WMMA_N;
        #pragma unroll
        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int local_n = i >> 4;  // i / 16
            int local_d = i & 15;  // i % 16
            int global_n = global_n_base + local_n;
            int global_d = global_d_base + local_d;

            if (global_n < n_size && global_d < head_dim) {
                smem.dv_acc()[global_n * BWD_BLOCK_D + global_d] += staging[i];
            }
        }
        __syncwarp();
    }
    __syncthreads();
}

// K-major write dK/dV from smem to global
__device__ void kmajor_write_dk_dv_from_smem(
    KMajorSmemAccessor& smem,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int kv_seq_start,
    int n_start,
    int n_end,
    int num_heads,
    int head_idx,
    int head_dim,
    float scale
) {
    const int tid = threadIdx.x;
    const int n_size = n_end - n_start;
    const int stride_token = num_heads * head_dim;

    for (int idx = tid; idx < n_size * head_dim; idx += BWD_NUM_THREADS) {
        int n = idx / head_dim;
        int d = idx % head_dim;

        int global_token = kv_seq_start + n_start + n;
        int global_idx = global_token * stride_token + head_idx * head_dim + d;

        dk[global_idx] = smem.dk_acc()[n * BWD_BLOCK_D + d] * scale;
        dv[global_idx] = smem.dv_acc()[n * BWD_BLOCK_D + d];
    }
}

// K-major backward kernel (smem-efficient with N=16)
// kComputeDQ=false skips dQ computation for dual-kernel mode (Q-major kernel handles dQ)
template<bool kIsCausal, bool kComputeDQ = true>
__global__ void __launch_bounds__(BWD_NUM_THREADS)
fmha_bwd_sm120_kmajor_kernel(
    const __nv_bfloat16* __restrict__ d_o,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ o,
    const float* __restrict__ lse,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ cu_seqlens_kv,
    float* __restrict__ dq,  // Float for atomic accumulation (unused if kComputeDQ=false)
    float* __restrict__ dk,
    float* __restrict__ dv,
    int num_heads,
    int head_dim,
    float scale,
    int max_seqlen_q,
    int max_seqlen_kv
) {
    extern __shared__ char smem_base[];
    KMajorSmemAccessor smem;
    smem.init(smem_base, 0);  // Initialize with buffer 0

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int n_block_idx = blockIdx.x;  // K-major: grid over K-blocks (N=16 per block)

    const int q_start = cu_seqlens_q[batch_idx];
    const int q_end = cu_seqlens_q[batch_idx + 1];
    const int kv_start = cu_seqlens_kv[batch_idx];
    const int kv_end = cu_seqlens_kv[batch_idx + 1];

    const int seq_len_q = q_end - q_start;
    const int seq_len_kv = kv_end - kv_start;

    if (head_idx >= num_heads) return;

    const int n_start = n_block_idx * KMAJOR_BLOCK_N;
    const int n_end = min(n_start + KMAJOR_BLOCK_N, seq_len_kv);
    if (n_start >= seq_len_kv) return;

    const int n_size = n_end - n_start;
    const int tid = threadIdx.x;

    // OPTIMIZED: Only zero dk_acc and dv_acc (critical for accumulation across Q-blocks)
    // scores/probs/dscores are computed fresh each iteration via WMMA fill_fragment(0)
    const float4 zero_f4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float4* dk_acc_ptr4 = reinterpret_cast<float4*>(smem.dk_acc());
    float4* dv_acc_ptr4 = reinterpret_cast<float4*>(smem.dv_acc());
    [[maybe_unused]] float4* dq_ptr4 = nullptr;
    if constexpr (kComputeDQ) {
        dq_ptr4 = reinterpret_cast<float4*>(smem.dq_acc());
    }

    // Zero dk_acc and dv_acc ONCE at kernel start (accumulated across all Q-blocks)
    for (int i = tid; i < (KMAJOR_BLOCK_N * BWD_BLOCK_D) / 4; i += BWD_NUM_THREADS) {
        dk_acc_ptr4[i] = zero_f4;
        dv_acc_ptr4[i] = zero_f4;
    }

    __syncthreads();

    // Load K, V tiles ONCE (they stay fixed for all Q iterations)
    kmajor_load_k_tile(smem, k, kv_start, n_start, n_end, num_heads, head_idx, head_dim);
    kmajor_load_v_tile(smem, v, kv_start, n_start, n_end, num_heads, head_idx, head_dim);
    __syncthreads();

    const int num_q_blocks = (seq_len_q + BWD_BLOCK_M - 1) / BWD_BLOCK_M;

    // OPTIMIZED: Double-buffered Q/dO with cp.async pipelining
    // Pre-load first non-skipped Q/dO block into buffer 0
    int cur_buf = 0;
    int first_m_block = 0;

    // Find first non-skipped block for causal case
    if (kIsCausal) {
        for (int m = 0; m < num_q_blocks; ++m) {
            int m_s = m * BWD_BLOCK_M;
            int m_e = min(m_s + BWD_BLOCK_M, seq_len_q);
            if ((m_s + (m_e - m_s) - 1) >= n_start) {
                first_m_block = m;
                break;
            }
        }
    }

    // Pre-load first block if exists
    if (first_m_block < num_q_blocks) {
        int m_s0 = first_m_block * BWD_BLOCK_M;
        int m_e0 = min(m_s0 + BWD_BLOCK_M, seq_len_q);
        kmajor_async_load_q_tile(smem, q, q_start, m_s0, m_e0, num_heads, head_idx, head_dim, 0);
        kmajor_async_load_do_tile(smem, d_o, q_start, m_s0, m_e0, num_heads, head_idx, head_dim, 0);
        cp_async_commit_group();
    }

    // Loop over all Q-blocks with double-buffered pipelining
    for (int m_block = first_m_block; m_block < num_q_blocks; ++m_block) {
        const int m_start = m_block * BWD_BLOCK_M;
        const int m_end = min(m_start + BWD_BLOCK_M, seq_len_q);
        const int m_size = m_end - m_start;

        // Skip if causal and this Q-block is entirely before this K-block
        if (kIsCausal && (m_start + m_size - 1) < n_start) {
            continue;
        }

        // Find next non-skipped block for prefetch
        int next_m_block = -1;
        for (int nm = m_block + 1; nm < num_q_blocks; ++nm) {
            int nm_s = nm * BWD_BLOCK_M;
            int nm_e = min(nm_s + BWD_BLOCK_M, seq_len_q);
            if (!kIsCausal || (nm_s + (nm_e - nm_s) - 1) >= n_start) {
                next_m_block = nm;
                break;
            }
        }

        // Prefetch next Q/dO block to alternate buffer while waiting for current
        if (next_m_block >= 0) {
            int next_buf = 1 - cur_buf;
            int nm_s = next_m_block * BWD_BLOCK_M;
            int nm_e = min(nm_s + BWD_BLOCK_M, seq_len_q);
            kmajor_async_load_q_tile(smem, q, q_start, nm_s, nm_e, num_heads, head_idx, head_dim, next_buf);
            kmajor_async_load_do_tile(smem, d_o, q_start, nm_s, nm_e, num_heads, head_idx, head_dim, next_buf);
            cp_async_commit_group();
        }

        // Wait for current buffer's async loads to complete
        cp_async_wait_all();
        __syncthreads();

        // Set accessor to use current buffer for Q/dO access
        smem.set_buffer(cur_buf);

        // OPTIMIZED: Only zero dq_acc per iteration (needed for atomicAdd accumulation)
        // scores/probs/dscores are computed fresh via WMMA fill_fragment(0)
        if constexpr (kComputeDQ) {
            for (int i = tid; i < (BWD_BLOCK_M * BWD_BLOCK_D) / 4; i += BWD_NUM_THREADS) {
                dq_ptr4[i] = zero_f4;
            }
        }

        // Load O (not double-buffered) and LSE for this Q-block (sync load - small)
        kmajor_load_o_tile(smem, o, q_start, m_start, m_end, num_heads, head_idx, head_dim);
        kmajor_load_lse(smem, lse, q_start, m_start, m_end, num_heads, head_idx);
        __syncthreads();

        // Compute delta = O dot dO for this Q-block
        kmajor_compute_delta(smem, m_size, head_dim);

        // Compute S = Q @ K^T
        kmajor_compute_qk_scores_wmma(smem, m_size, n_size, head_dim, scale);

        // Recompute P = softmax(S) with causal mask
        kmajor_recompute_softmax(smem, m_size, n_size, kIsCausal, m_start, n_start);

        // Compute dP = dO @ V^T
        kmajor_compute_dp_wmma(smem, m_size, n_size, head_dim);

        // Compute dScores = (dP - delta) * P * scale
        kmajor_compute_dscores(smem, m_size, n_size, scale);

        // Compute and atomicAdd dQ (only if kComputeDQ=true)
        if constexpr (kComputeDQ) {
            // Compute dQ
            kmajor_compute_dq_wmma(smem, m_size, n_size, head_dim);

            // OPTIMIZED: atomicAdd dQ to global with fast index math (head_dim=128 is power of 2)
            constexpr int D_SHIFT = 7;  // log2(128)
            constexpr int D_MASK = BWD_BLOCK_D - 1;
            const int stride_token = num_heads * head_dim;
            #pragma unroll 4
            for (int idx = tid; idx < m_size * head_dim; idx += BWD_NUM_THREADS) {
                int m = idx >> D_SHIFT;
                int d = idx & D_MASK;
                float val = smem.dq_acc()[idx];  // dq_acc is row-major [M, D]
                if (val != 0.0f) {
                    int global_token = q_start + m_start + m;
                    atomicAdd(&dq[global_token * stride_token + head_idx * head_dim + d], val);
                }
            }
            __syncthreads();
        }

        // Accumulate dK to shared memory (NO ATOMICS!)
        kmajor_accumulate_dk_to_smem(smem, m_size, n_size, head_dim);

        // Accumulate dV to shared memory (NO ATOMICS!)
        kmajor_accumulate_dv_to_smem(smem, m_size, n_size, head_dim);

        // Swap buffers for next iteration - next iteration's data is in alternate buffer
        if (next_m_block >= 0) {
            cur_buf = 1 - cur_buf;
        }
    }

    // Write final dK, dV from smem to global memory (single non-atomic write!)
    kmajor_write_dk_dv_from_smem(smem, dk, dv, kv_start, n_start, n_end, num_heads, head_idx, head_dim, scale);
}

// ============================================================================
// Q-MAJOR DQ-ONLY KERNEL (DUAL KERNEL ARCHITECTURE)
// ============================================================================
// This kernel computes ONLY dQ, parallelized over Q-blocks
// Each block owns one Q-block and loops over all K-blocks
// No atomics needed - dQ is accumulated in smem and written once

// Async load K tile for Q-major kernel
__device__ __forceinline__ void qmajor_async_load_k_tile(
    QMajorSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ k,
    int kv_seq_start,
    int n_start,
    int n_end,
    int num_heads,
    int head_idx,
    int head_dim,
    int buf_idx
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int cols = n_end - n_start;
    const int stride_token = num_heads * head_dim;

    for (int n = warp_id; n < cols; n += num_warps) {
        const int global_token = kv_seq_start + n_start + n;
        const __nv_bfloat16* src_row = k + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.k_tile_buf(buf_idx) + n * BWD_BLOCK_D;

        for (int d = lane_id * 8; d < head_dim; d += 256) {
            if (d + 8 <= head_dim) {
                cp_async_cg_16(dst_row + d, src_row + d);
            } else {
                for (int dd = d; dd < head_dim && dd < d + 8; dd++) {
                    dst_row[dd] = src_row[dd];
                }
            }
        }
    }
}

// Async load V tile for Q-major kernel
__device__ __forceinline__ void qmajor_async_load_v_tile(
    QMajorSmemAccessor& smem,
    const __nv_bfloat16* __restrict__ v,
    int kv_seq_start,
    int n_start,
    int n_end,
    int num_heads,
    int head_idx,
    int head_dim,
    int buf_idx
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int cols = n_end - n_start;
    const int stride_token = num_heads * head_dim;

    for (int n = warp_id; n < cols; n += num_warps) {
        const int global_token = kv_seq_start + n_start + n;
        const __nv_bfloat16* src_row = v + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* dst_row = smem.v_tile_buf(buf_idx) + n * BWD_BLOCK_D;

        for (int d = lane_id * 8; d < head_dim; d += 256) {
            if (d + 8 <= head_dim) {
                cp_async_cg_16(dst_row + d, src_row + d);
            } else {
                for (int dd = d; dd < head_dim && dd < d + 8; dd++) {
                    dst_row[dd] = src_row[dd];
                }
            }
        }
    }
}

// Q-major dQ-only kernel
template<bool kIsCausal>
__global__ void __launch_bounds__(BWD_NUM_THREADS)
fmha_bwd_sm120_dq_kernel(
    const __nv_bfloat16* __restrict__ d_o,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ o,
    const float* __restrict__ lse,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ cu_seqlens_kv,
    float* __restrict__ dq,
    int num_heads,
    int head_dim,
    float scale,
    int max_seqlen_q,
    int max_seqlen_kv
) {
    using namespace nvcuda::wmma;

    extern __shared__ char smem_base[];
    QMajorSmemAccessor smem;
    smem.init(smem_base, 0);

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int m_block_idx = blockIdx.x;  // Q-major: grid over Q-blocks

    const int q_start = cu_seqlens_q[batch_idx];
    const int q_end = cu_seqlens_q[batch_idx + 1];
    const int kv_start = cu_seqlens_kv[batch_idx];
    const int kv_end = cu_seqlens_kv[batch_idx + 1];

    const int seq_len_q = q_end - q_start;
    const int seq_len_kv = kv_end - kv_start;

    if (head_idx >= num_heads) return;

    const int m_start = m_block_idx * QMAJOR_BLOCK_M;
    const int m_end = min(m_start + QMAJOR_BLOCK_M, seq_len_q);
    if (m_start >= seq_len_q) return;

    const int m_size = m_end - m_start;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int stride_token = num_heads * head_dim;

    // Zero dQ accumulator ONCE at kernel start
    const float4 zero_f4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4* dq_acc_ptr4 = reinterpret_cast<float4*>(smem.dq_acc());
    for (int i = tid; i < (QMAJOR_BLOCK_M * BWD_BLOCK_D) / 4; i += BWD_NUM_THREADS) {
        dq_acc_ptr4[i] = zero_f4;
    }
    __syncthreads();

    // Load Q, dO, O, LSE ONCE (they stay fixed for all K iterations)
    // Synchronous load since these are used immediately
    for (int m = warp_id; m < m_size; m += BWD_NUM_WARPS) {
        const int global_token = q_start + m_start + m;
        const __nv_bfloat16* q_src = q + global_token * stride_token + head_idx * head_dim;
        const __nv_bfloat16* do_src = d_o + global_token * stride_token + head_idx * head_dim;
        const __nv_bfloat16* o_src = o + global_token * stride_token + head_idx * head_dim;
        __nv_bfloat16* q_dst = smem.q_row(m);
        __nv_bfloat16* do_dst = smem.do_row(m);
        __nv_bfloat16* o_dst = smem.o_row(m);

        for (int d = lane_id * 8; d < head_dim; d += 256) {
            if (d + 8 <= head_dim) {
                float4 qval = *reinterpret_cast<const float4*>(q_src + d);
                float4 doval = *reinterpret_cast<const float4*>(do_src + d);
                float4 oval = *reinterpret_cast<const float4*>(o_src + d);
                *reinterpret_cast<float4*>(q_dst + d) = qval;
                *reinterpret_cast<float4*>(do_dst + d) = doval;
                *reinterpret_cast<float4*>(o_dst + d) = oval;
            }
        }
    }

    // Load LSE
    for (int m = tid; m < m_size; m += BWD_NUM_THREADS) {
        const int global_token = q_start + m_start + m;
        smem.lse()[m] = lse[global_token * num_heads + head_idx];
    }
    __syncthreads();

    // Compute delta = O dot dO (only once per Q-block)
    for (int m = tid; m < m_size; m += BWD_NUM_THREADS) {
        float delta_m = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < head_dim; d++) {
            float o_val = __bfloat162float(smem.o_row(m)[d]);
            float do_val = __bfloat162float(smem.do_row(m)[d]);
            delta_m += o_val * do_val;
        }
        smem.delta()[m] = delta_m;
    }
    __syncthreads();

    const int num_k_blocks = (seq_len_kv + QMAJOR_BLOCK_N - 1) / QMAJOR_BLOCK_N;

    // Pre-load first K/V block into buffer 0
    int cur_buf = 0;
    int first_n_block = 0;

    // For causal, find first valid K-block
    if (kIsCausal) {
        // Last Q position must be >= first K position in block
        for (int nb = 0; nb < num_k_blocks; ++nb) {
            int n_s = nb * QMAJOR_BLOCK_N;
            if ((m_start + m_size - 1) >= n_s) {
                first_n_block = nb;
                break;
            }
        }
    }

    if (first_n_block < num_k_blocks) {
        int n_s0 = first_n_block * QMAJOR_BLOCK_N;
        int n_e0 = min(n_s0 + QMAJOR_BLOCK_N, seq_len_kv);
        qmajor_async_load_k_tile(smem, k, kv_start, n_s0, n_e0, num_heads, head_idx, head_dim, 0);
        qmajor_async_load_v_tile(smem, v, kv_start, n_s0, n_e0, num_heads, head_idx, head_dim, 0);
        cp_async_commit_group();
    }

    // Loop over all K-blocks with double-buffered pipelining
    for (int n_block = first_n_block; n_block < num_k_blocks; ++n_block) {
        const int n_start = n_block * QMAJOR_BLOCK_N;
        const int n_end = min(n_start + QMAJOR_BLOCK_N, seq_len_kv);
        const int n_size = n_end - n_start;

        // Skip if causal and all Q positions < all K positions
        if (kIsCausal && (m_start + m_size - 1) < n_start) {
            continue;
        }

        // Find next valid K-block for prefetch
        int next_n_block = -1;
        for (int nb = n_block + 1; nb < num_k_blocks; ++nb) {
            int nb_s = nb * QMAJOR_BLOCK_N;
            if (!kIsCausal || (m_start + m_size - 1) >= nb_s) {
                next_n_block = nb;
                break;
            }
        }

        // Prefetch next K/V block while waiting for current
        if (next_n_block >= 0) {
            int next_buf = 1 - cur_buf;
            int nb_s = next_n_block * QMAJOR_BLOCK_N;
            int nb_e = min(nb_s + QMAJOR_BLOCK_N, seq_len_kv);
            qmajor_async_load_k_tile(smem, k, kv_start, nb_s, nb_e, num_heads, head_idx, head_dim, next_buf);
            qmajor_async_load_v_tile(smem, v, kv_start, nb_s, nb_e, num_heads, head_idx, head_dim, next_buf);
            cp_async_commit_group();
        }

        // Wait for current buffer
        cp_async_wait_all();
        __syncthreads();

        smem.set_buffer(cur_buf);

        // Compute S = Q @ K^T using WMMA
        const int m_tiles = (m_size + WMMA_M - 1) / WMMA_M;
        const int n_tiles = (n_size + WMMA_N - 1) / WMMA_N;
        const int d_tiles = (head_dim + WMMA_K - 1) / WMMA_K;

        for (int mn = warp_id; mn < m_tiles * n_tiles; mn += BWD_NUM_WARPS) {
            int m_tile = mn / n_tiles;
            int n_tile = mn % n_tiles;

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            fill_fragment(acc, 0.0f);

            #pragma unroll 4
            for (int k_tile = 0; k_tile < d_tiles; ++k_tile) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> q_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> k_frag;

                load_matrix_sync(q_frag, smem.q_row(m_tile * WMMA_M) + k_tile * WMMA_K, BWD_BLOCK_D);
                load_matrix_sync(k_frag, smem.k_row(n_tile * WMMA_N) + k_tile * WMMA_K, BWD_BLOCK_D);

                mma_sync(acc, q_frag, k_frag, acc);
            }

            // Apply scale directly
            #pragma unroll
            for (int i = 0; i < acc.num_elements; ++i) {
                acc.x[i] *= scale;
            }

            float* scores_ptr = smem.scores_row(m_tile * WMMA_M) + n_tile * WMMA_N;
            store_matrix_sync(scores_ptr, acc, QMAJOR_BLOCK_N, mem_row_major);
        }
        __syncthreads();

        // Recompute softmax P
        constexpr int N_SHIFT = 5;  // log2(QMAJOR_BLOCK_N=32)
        constexpr int N_MASK = QMAJOR_BLOCK_N - 1;
        #pragma unroll 4
        for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
            int m = idx >> N_SHIFT;
            int n = idx & N_MASK;
            if (m < m_size && n < n_size) {
                float score = smem.scores()[idx];
                float lse_m = smem.lse()[m];
                bool masked = kIsCausal && ((m_start + m) < (n_start + n));
                float prob = masked ? 0.0f : expf(score - lse_m);
                smem.probs()[idx] = prob;
            }
        }
        __syncthreads();

        // Compute dP = dO @ V^T using WMMA
        for (int mn = warp_id; mn < m_tiles * n_tiles; mn += BWD_NUM_WARPS) {
            int m_tile = mn / n_tiles;
            int n_tile = mn % n_tiles;

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            fill_fragment(acc, 0.0f);

            #pragma unroll 4
            for (int k_tile = 0; k_tile < d_tiles; ++k_tile) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> do_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> v_frag;

                load_matrix_sync(do_frag, smem.do_row(m_tile * WMMA_M) + k_tile * WMMA_K, BWD_BLOCK_D);
                load_matrix_sync(v_frag, smem.v_row(n_tile * WMMA_N) + k_tile * WMMA_K, BWD_BLOCK_D);

                mma_sync(acc, do_frag, v_frag, acc);
            }

            float* dscores_ptr = smem.dscores_row(m_tile * WMMA_M) + n_tile * WMMA_N;
            store_matrix_sync(dscores_ptr, acc, QMAJOR_BLOCK_N, mem_row_major);
        }
        __syncthreads();

        // Compute dScores = (dP - delta) * P * scale
        #pragma unroll 4
        for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
            int m = idx >> N_SHIFT;
            int n = idx & N_MASK;
            if (m < m_size && n < n_size) {
                float dp = smem.dscores()[idx];
                float p = smem.probs()[idx];
                float delta_m = smem.delta()[m];
                smem.dscores()[idx] = (dp - delta_m) * p * scale;
            }
        }
        __syncthreads();

        // Convert dscores to bf16
        __nv_bfloat16* dscores_bf16 = smem.temp_bf16();
        #pragma unroll 4
        for (int idx = tid; idx < QMAJOR_BLOCK_M * QMAJOR_BLOCK_N; idx += BWD_NUM_THREADS) {
            dscores_bf16[idx] = __float2bfloat16(0.0f);
        }
        __syncthreads();
        #pragma unroll 4
        for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
            dscores_bf16[idx] = __float2bfloat16(smem.dscores()[idx]);
        }
        __syncthreads();

        // Accumulate dQ += dScores @ K using WMMA
        const int dq_m_tiles = (m_size + WMMA_M - 1) / WMMA_M;
        const int dq_d_tiles = (head_dim + WMMA_N - 1) / WMMA_N;
        const int dq_k_tiles = (n_size + WMMA_K - 1) / WMMA_K;

        for (int md = warp_id; md < dq_m_tiles * dq_d_tiles; md += BWD_NUM_WARPS) {
            int m_tile = md / dq_d_tiles;
            int d_tile = md % dq_d_tiles;

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            fill_fragment(acc, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < dq_k_tiles; ++k_tile) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> ds_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> k_frag;

                load_matrix_sync(ds_frag, dscores_bf16 + m_tile * WMMA_M * QMAJOR_BLOCK_N + k_tile * WMMA_K, QMAJOR_BLOCK_N);
                load_matrix_sync(k_frag, smem.k_row(k_tile * WMMA_K) + d_tile * WMMA_N, BWD_BLOCK_D);

                mma_sync(acc, ds_frag, k_frag, acc);
            }

            // Accumulate into dQ_acc
            float* staging = smem.wmma_staging_warp(warp_id);
            store_matrix_sync(staging, acc, WMMA_N, mem_row_major);
            __syncwarp();

            constexpr int WMMA_N_SHIFT = 4;
            constexpr int WMMA_N_MASK = WMMA_N - 1;
            #pragma unroll 8
            for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
                int row = i >> WMMA_N_SHIFT;
                int col = i & WMMA_N_MASK;
                int global_row = m_tile * WMMA_M + row;
                int global_col = d_tile * WMMA_N + col;
                if (global_row < m_size && global_col < head_dim) {
                    smem.dq_acc()[global_row * BWD_BLOCK_D + global_col] += staging[i];
                }
            }
            __syncwarp();
        }
        __syncthreads();

        // Swap buffers
        if (next_n_block >= 0) {
            cur_buf = 1 - cur_buf;
        }
    }

    // Write final dQ from smem to global memory (NO ATOMICS!)
    constexpr int D_SHIFT = 7;  // log2(128)
    constexpr int D_MASK = BWD_BLOCK_D - 1;
    #pragma unroll 4
    for (int idx = tid; idx < m_size * head_dim; idx += BWD_NUM_THREADS) {
        int m = idx >> D_SHIFT;
        int d = idx & D_MASK;
        int global_token = q_start + m_start + m;
        dq[global_token * stride_token + head_idx * head_dim + d] = smem.dq_acc()[idx];
    }
}

// Launch wrapper for dual-kernel backward pass
template<bool kIsCausal>
void launch_fmha_bwd_sm120_dual(
    const c10::cuda::CUDAStream& stream,
    at::Tensor d_o,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor o,
    at::Tensor lse,
    at::Tensor cu_seqlens_q,
    at::Tensor cu_seqlens_kv,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    float scale,
    int max_seqlen_q,
    int max_seqlen_kv
) {
    const int batch_size = cu_seqlens_q.size(0) - 1;
    const int num_heads = q.size(1);
    const int head_dim = q.size(2);

    // Launch Q-major dQ kernel
    const int m_blocks = (max_seqlen_q + QMAJOR_BLOCK_M - 1) / QMAJOR_BLOCK_M;
    dim3 dq_grid(m_blocks, num_heads, batch_size);
    dim3 dq_block(BWD_NUM_THREADS);

    size_t dq_smem_size = QMajorSmemLayout::total_size;
    cudaFuncSetAttribute(
        fmha_bwd_sm120_dq_kernel<kIsCausal>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dq_smem_size
    );

    fmha_bwd_sm120_dq_kernel<kIsCausal><<<dq_grid, dq_block, dq_smem_size, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(d_o.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()),
        lse.data_ptr<float>(),
        cu_seqlens_q.data_ptr<int>(),
        cu_seqlens_kv.data_ptr<int>(),
        dq.data_ptr<float>(),
        num_heads,
        head_dim,
        scale,
        max_seqlen_q,
        max_seqlen_kv
    );

    // Launch K-major dK/dV kernel with kComputeDQ=false (skips dQ computation)
    const int n_blocks = (max_seqlen_kv + KMAJOR_BLOCK_N - 1) / KMAJOR_BLOCK_N;
    dim3 dkdv_grid(n_blocks, num_heads, batch_size);
    dim3 dkdv_block(BWD_NUM_THREADS);

    size_t dkdv_smem_size = KMajorSmemLayout::total_size;
    cudaFuncSetAttribute(
        fmha_bwd_sm120_kmajor_kernel<kIsCausal, false>,  // kComputeDQ=false
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dkdv_smem_size
    );

    // Launch with kComputeDQ=false - dQ is handled by the Q-major kernel above
    fmha_bwd_sm120_kmajor_kernel<kIsCausal, false><<<dkdv_grid, dkdv_block, dkdv_smem_size, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(d_o.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()),
        lse.data_ptr<float>(),
        cu_seqlens_q.data_ptr<int>(),
        cu_seqlens_kv.data_ptr<int>(),
        nullptr,  // dq unused when kComputeDQ=false
        dk.data_ptr<float>(),
        dv.data_ptr<float>(),
        num_heads,
        head_dim,
        scale,
        max_seqlen_q,
        max_seqlen_kv
    );
}

// Launch wrapper for K-major varlen kernel
template<bool kIsCausal>
void launch_fmha_bwd_sm120_kmajor(
    const c10::cuda::CUDAStream& stream,
    at::Tensor d_o,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor o,
    at::Tensor lse,
    at::Tensor cu_seqlens_q,
    at::Tensor cu_seqlens_kv,
    at::Tensor dq,  // Float tensor for atomic accumulation
    at::Tensor dk,
    at::Tensor dv,
    float scale,
    int max_seqlen_q,
    int max_seqlen_kv
) {
    const int batch_size = cu_seqlens_q.size(0) - 1;
    const int num_heads = q.size(1);
    const int head_dim = q.size(2);

    // K-major: grid over K-blocks with N=KMAJOR_BLOCK_N (16) per block
    const int n_blocks = (max_seqlen_kv + KMAJOR_BLOCK_N - 1) / KMAJOR_BLOCK_N;
    dim3 grid(n_blocks, num_heads, batch_size);
    dim3 block(BWD_NUM_THREADS);

    // Use K-major specific smem layout (~79KB, fits within 99KB limit)
    size_t smem_size = KMajorSmemLayout::total_size;

    cudaFuncSetAttribute(
        fmha_bwd_sm120_kmajor_kernel<kIsCausal>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );

    fmha_bwd_sm120_kmajor_kernel<kIsCausal><<<grid, block, smem_size, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(d_o.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()),
        lse.data_ptr<float>(),
        cu_seqlens_q.data_ptr<int>(),
        cu_seqlens_kv.data_ptr<int>(),
        dq.data_ptr<float>(),  // Float for atomic accumulation
        dk.data_ptr<float>(),
        dv.data_ptr<float>(),
        num_heads,
        head_dim,
        scale,
        max_seqlen_q,
        max_seqlen_kv
    );
}

// Launch wrapper for varlen (original Q-major)
template<bool kIsCausal>
void launch_fmha_bwd_sm120_varlen(
    const c10::cuda::CUDAStream& stream,
    at::Tensor d_o,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor o,
    at::Tensor lse,
    at::Tensor cu_seqlens_q,
    at::Tensor cu_seqlens_kv,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    float scale,
    int max_seqlen_q,
    int max_seqlen_kv
) {
    const int batch_size = cu_seqlens_q.size(0) - 1;
    const int num_heads = q.size(1);
    const int head_dim = q.size(2);

    const int m_blocks = (max_seqlen_q + BWD_BLOCK_M - 1) / BWD_BLOCK_M;
    dim3 grid(m_blocks, num_heads, batch_size);
    dim3 block(BWD_NUM_THREADS);

    size_t smem_size = BwdSmemLayout::total_size;

    cudaFuncSetAttribute(
        fmha_bwd_sm120_varlen_kernel<kIsCausal>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );

    fmha_bwd_sm120_varlen_kernel<kIsCausal><<<grid, block, smem_size, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(d_o.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()),
        lse.data_ptr<float>(),
        cu_seqlens_q.data_ptr<int>(),
        cu_seqlens_kv.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()),
        dk.data_ptr<float>(),
        dv.data_ptr<float>(),
        num_heads,
        head_dim,
        scale,
        max_seqlen_q,
        max_seqlen_kv
    );
}

}  // namespace detail
}  // namespace flash
