/***************************************************************************************************
 * SM120 Decode Kernel (MLA Split-KV Attention) - SM90/SM100 Style Implementation
 *
 * This kernel implements Flash Attention decode for SM120 (Blackwell workstation GPUs).
 * Follows the architectural patterns of SM90/SM100 implementations:
 * - Double-buffered K tiles (prepared for TMA async loading)
 * - Pipeline structure for memory/compute overlap
 * - WMMA (mma.sync.aligned) for BF16/FP16 tensor core operations
 *
 * Architecture Notes (SM120 vs SM90/SM100):
 * - SM120 does NOT have GMMA (SM90) or tcgen05/UMMA for BF16 (only FP4/FP6/FP8)
 * - SM120 uses SM80-compatible mma.sync.aligned which translates to HMMA
 * - TMA IS supported on SM120 (cluster shape fixed to 1x1x1)
 * - 99KB shared memory (vs 227KB SM100, 232KB SM90)
 *
 * Memory Layout:
 * - Q tile: [64, 64] (loaded per K-tile)
 * - K tiles: [64, 64] x 2 (double-buffered)
 * - V: [64, 512] (reuses QK space)
 * - Total: ~91KB (fits in 99KB)
 *
 * WMMA Configuration:
 * - Uses nvcuda::wmma with 16x16x16 tiles for BF16/FP16
 * - 4 warps process 64x64 output tiles cooperatively
 * - Each warp handles a 16x64 strip of the output
 *
 * References:
 * - https://deepwiki.com/NVIDIA/cutlass/7.3-sm120-blackwell-geforce-architecture
 * - https://arxiv.org/html/2507.10789v1 (Blackwell microbenchmarks)
 **************************************************************************************************/

#include <cutlass/cutlass.h>
#include <cutlass/arch/memory_sm80.h>
#include <cute/tensor.hpp>
#include <mma.h>

#include "traits.h"
#include "params.h"

using namespace cute;
using namespace nvcuda;

namespace sm120 {

// Constants
static constexpr float NEGATIVE_INFINITY = -1e30f;
static constexpr float LOG2E = 1.4426950408889634f;

// WMMA configuration for 16x16x16 tiles
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// Type trait to map CUTLASS types to WMMA types
template<typename T> struct WmmaType;
template<> struct WmmaType<cutlass::bfloat16_t> { using type = __nv_bfloat16; };
template<> struct WmmaType<cutlass::half_t> { using type = __half; };

//==============================================================================
// Warp-level reduction primitives
//==============================================================================
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

//==============================================================================
// WMMA-based GEMM for QK computation
// Computes: Scores[64, 64] += Q_tile[64, K_TILE] @ K_tile[64, K_TILE]^T
//
// K_TILE_DIM = 64, WMMA_K = 16, so 4 inner iterations per outer K-tile
// HEAD_DIM_K = 576 requires 9 outer K-tiles, accumulating in sScores
//==============================================================================
template<typename InputT>
__device__ void wmma_gemm_qk(
    const InputT* __restrict__ sQ_tile,  // [64, 64] in smem (row-major)
    const InputT* __restrict__ sK_tile,  // [64, 64] in smem (row-major)
    float* __restrict__ sScores,          // [64, 64] accumulator in smem
    int warp_idx,
    int lane_idx,
    bool is_first_k_tile  // True if this is the first K-tile (initialize), false to accumulate
) {
    using WmmaInputT = typename WmmaType<InputT>::type;

    // Each warp handles a 16x64 strip of the output (4 16x16 tiles)
    // 4 warps cover the full 64x64 output
    const int warp_m = warp_idx * WMMA_M;  // 0, 16, 32, 48

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, WmmaInputT, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, WmmaInputT, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Iterate over N dimension (4 x 16 = 64 columns)
    #pragma unroll
    for (int n_tile = 0; n_tile < 4; ++n_tile) {
        const int out_n = n_tile * WMMA_N;

        // Initialize or load existing accumulator
        if (is_first_k_tile) {
            wmma::fill_fragment(c_frag, 0.0f);
        } else {
            // Load existing accumulator from sScores to continue accumulating
            wmma::load_matrix_sync(c_frag, sScores + warp_m * 64 + out_n, 64, wmma::mem_row_major);
        }

        // Iterate over K dimension (4 x 16 = 64) within this K_TILE_DIM chunk
        #pragma unroll
        for (int k_tile = 0; k_tile < 4; ++k_tile) {
            const int k_offset = k_tile * WMMA_K;

            // Load Q fragment: Q[warp_m:warp_m+16, k_offset:k_offset+16]
            wmma::load_matrix_sync(a_frag,
                                   reinterpret_cast<const WmmaInputT*>(sQ_tile) + warp_m * 64 + k_offset,
                                   64);  // stride = K_TILE_DIM

            // Load K fragment (transposed): K[out_n:out_n+16, k_offset:k_offset+16]
            // Since we want K^T, and K is row-major, we load K as col_major
            wmma::load_matrix_sync(b_frag,
                                   reinterpret_cast<const WmmaInputT*>(sK_tile) + out_n * 64 + k_offset,
                                   64);  // stride = K_TILE_DIM

            // MMA: C += A * B^T
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Store accumulator back to shared memory
        wmma::store_matrix_sync(sScores + warp_m * 64 + out_n, c_frag, 64, wmma::mem_row_major);
    }
}

//==============================================================================
// WMMA-based GEMM for PV computation (tiled)
// Computes: O[64, 512] += P[64, 64] @ V[64, 512]
//
// Processes output in 64-column tiles to fit output buffer in shared memory.
// sO_tile is [64, 64] = 16KB (reuses smem_scores buffer)
//==============================================================================
template<typename InputT>
__device__ void wmma_gemm_pv_tiled(
    const InputT* __restrict__ sP,     // [64, 64] attention weights in smem
    const InputT* __restrict__ sV,     // [64, 512] values in smem (row-major)
    float* __restrict__ sO_tile,        // [64, 64] output tile buffer in smem
    float* __restrict__ rO,             // Register output accumulator
    int warp_idx,
    int lane_idx,
    int thread_idx,
    int v_col_tile_idx                  // Which 64-column tile of V (0-7)
) {
    using WmmaInputT = typename WmmaType<InputT>::type;

    // Each warp handles a 16-row strip of output (rows warp_m:warp_m+16)
    const int warp_m = warp_idx * WMMA_M;  // 0, 16, 32, 48

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, WmmaInputT, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, WmmaInputT, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    constexpr int PAGE_SIZE = 64;
    constexpr int HEAD_V = 512;
    constexpr int V_TILE_DIM = 64;

    // Output tile offset in V dimension
    const int v_col_offset = v_col_tile_idx * V_TILE_DIM;

    // Process 64 output columns in 4 WMMA tiles (each 16 cols)
    #pragma unroll
    for (int n_tile = 0; n_tile < 4; ++n_tile) {
        const int out_n = n_tile * WMMA_N;  // 0, 16, 32, 48 within this tile
        wmma::fill_fragment(c_frag, 0.0f);

        // K dimension iteration (64 / 16 = 4)
        #pragma unroll
        for (int k_tile = 0; k_tile < 4; ++k_tile) {
            const int k_offset = k_tile * WMMA_K;

            // Load P fragment: P[warp_m:warp_m+16, k_offset:k_offset+16]
            wmma::load_matrix_sync(a_frag,
                                   reinterpret_cast<const WmmaInputT*>(sP) + warp_m * PAGE_SIZE + k_offset,
                                   PAGE_SIZE);

            // Load V fragment: V[k_offset:k_offset+16, v_col_offset+out_n:v_col_offset+out_n+16]
            wmma::load_matrix_sync(b_frag,
                                   reinterpret_cast<const WmmaInputT*>(sV) + k_offset * HEAD_V + v_col_offset + out_n,
                                   HEAD_V);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Store to output tile buffer [64, 64] row-major
        wmma::store_matrix_sync(sO_tile + warp_m * V_TILE_DIM + out_n, c_frag, V_TILE_DIM, wmma::mem_row_major);
    }
}

//==============================================================================
// Main Kernel - WMMA-accelerated implementation with batch-parallel scheduling
//
// Key optimization: 256 threads = 2 warpgroups, each handling half of V (256 cols)
// - Warpgroup 0 (threads 0-127): V columns 0-255
// - Warpgroup 1 (threads 128-255): V columns 256-511
// - QK GEMM and softmax are cooperative (all threads)
// - PV GEMM is split (each warpgroup handles its V columns)
// - Output accumulator reduced from 256 to 128 floats per thread
//
// Parallelization (batch-parallel mode):
// - Grid: (num_m_blocks, h_k, batch_size) - each thread block handles ONE batch
// - No split-K combining needed - each batch is independent
// - Fully utilizes all SMs for large batch sizes (e.g., 128 batches = 256 blocks)
//==============================================================================
template<typename T>
__global__ void __launch_bounds__(T::NUM_THREADS, 1)
flash_fwd_splitkv_mla_sm120_kernel(const DecodingParams params) {
    using InputT = typename T::InputT;
    using SmemLayoutQ_tile = typename T::SmemLayoutQ_tile;
    using SmemLayoutK_tile = typename T::SmemLayoutK_tile;
    using SmemLayoutV = typename T::SmemLayoutV;
    using SmemLayoutP = typename T::SmemLayoutP;
    using SharedMemoryPlan = typename T::SharedMemoryPlan;

    // Grid and thread indices
    // Batch-parallel mode: batch_idx = blockIdx.z directly
    const int m_block_idx = blockIdx.x;
    const int k_head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;  // Direct batch index - no tile scheduler
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / 32;
    const int lane_idx = thread_idx % 32;

    // Early exit for out-of-bounds batch
    if (batch_idx >= params.b) return;

    // Shared memory
    extern __shared__ char smem_buf[];
    SharedMemoryPlan& smem = *reinterpret_cast<SharedMemoryPlan*>(smem_buf);

    // Create shared memory tensors
    Tensor sQ_tile = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_tile.data()), SmemLayoutQ_tile{});
    // Double-buffered K tiles for TMA overlap
    Tensor sK_tile0 = make_tensor(make_smem_ptr(smem.qk_phase.smem_K_tile0.data()), SmemLayoutK_tile{});
    Tensor sK_tile1 = make_tensor(make_smem_ptr(smem.qk_phase.smem_K_tile1.data()), SmemLayoutK_tile{});
    InputT* sK_tiles[2] = {smem.qk_phase.smem_K_tile0.data(), smem.qk_phase.smem_K_tile1.data()};
    Tensor sP = make_tensor(make_smem_ptr(smem.smem_P.data()), SmemLayoutP{});
    Tensor sV = make_tensor(make_smem_ptr(smem.smem_V.data()), SmemLayoutV{});

    // Score accumulator and softmax stats in shared memory
    float* sScores = smem.smem_scores.data();
    float* sM = smem.smem_M.data();
    float* sL = smem.smem_L.data();
    float* sScale = smem.smem_scale.data();

    // Output accumulator in registers (per thread)
    // With 256 threads: (64 * 512) / 256 = 128 floats per thread (same as SM90!)
    // Simple flat layout: thread i owns output indices {i, i+256, i+512, ...}
    constexpr int O_ELEMS_PER_THREAD = (T::BLOCK_SIZE_M * T::HEAD_DIM_V + T::NUM_THREADS - 1) / T::NUM_THREADS;
    float rO[O_ELEMS_PER_THREAD];  // 128 floats (same as SM90!)

    // Process single batch (batch-parallel mode)
    // Each thread block handles ALL KV blocks for its assigned batch
    {
        const int seqlen_k = __ldg(params.seqlens_k_ptr + batch_idx);
        const int start_block_idx = 0;  // Process all blocks from start
        const int end_block_idx = (seqlen_k + T::PAGE_BLOCK_SIZE - 1) / T::PAGE_BLOCK_SIZE;

        // Reset softmax stats
        if (thread_idx < T::BLOCK_SIZE_M) {
            sM[thread_idx] = NEGATIVE_INFINITY;
            sL[thread_idx] = 0.0f;
            sScale[thread_idx] = 1.0f;
        }

        // Zero output accumulator
        #pragma unroll
        for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) {
            rO[i] = 0.0f;
        }
        __syncthreads();

        // Get pointers
        // After reshape in pybind, Q is [batch, q_seq_per_hk, num_heads_k, head_dim_k]
        // - q_batch_stride: stride for batch dimension
        // - q_row_stride: stride for q_seq_per_hk dimension (Q entries within KV group)
        // - q_head_stride: stride for num_heads_k dimension (KV head selection)
        const InputT* q_ptr = reinterpret_cast<const InputT*>(params.q_ptr) +
                              batch_idx * params.q_batch_stride +
                              m_block_idx * T::BLOCK_SIZE_M * params.q_row_stride +
                              k_head_idx * params.q_head_stride;

        const InputT* k_ptr = reinterpret_cast<const InputT*>(params.k_ptr) +
                              k_head_idx * params.k_head_stride;

        // Output pointer: same layout as Q after reshape
        // O is [batch, q_seq_per_hk, num_heads_k, head_dim_v]
        InputT* o_ptr = reinterpret_cast<InputT*>(params.o_ptr) +
                        batch_idx * params.o_batch_stride +
                        m_block_idx * T::BLOCK_SIZE_M * params.o_row_stride +
                        k_head_idx * params.o_head_stride;

        float* lse_ptr = reinterpret_cast<float*>(params.softmax_lse_ptr) +
                         (batch_idx * params.h_k + k_head_idx) * params.q_seq_per_hk +
                         m_block_idx * T::BLOCK_SIZE_M;

        int* block_table_ptr = params.block_table + batch_idx * params.block_table_batch_stride;

        // Process KV blocks
        for (int block_idx = start_block_idx; block_idx < end_block_idx; ++block_idx) {
            const int phys_block_idx = __ldg(block_table_ptr + block_idx);
            const int start_token_idx = block_idx * T::PAGE_BLOCK_SIZE;
            const int valid_tokens = min(seqlen_k - start_token_idx, T::PAGE_BLOCK_SIZE);

            const InputT* kv_block_ptr = k_ptr + phys_block_idx * params.k_batch_stride;

            // Clear score accumulator
            constexpr int score_elems = T::BLOCK_SIZE_M * T::PAGE_BLOCK_SIZE;
            constexpr int score_per_thread = (score_elems + T::NUM_THREADS - 1) / T::NUM_THREADS;

            #pragma unroll
            for (int i = 0; i < score_per_thread; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS;
                if (idx < score_elems) {
                    sScores[idx] = 0.0f;
                }
            }
            __syncthreads();

            //==================================================================
            // Phase 1: QK GEMM (tiled along K dimension) with double-buffered K
            // Uses WMMA for tensor core acceleration
            //
            // Pipeline structure (for future TMA):
            // - Load K_tile into buffer[k%2] while computing with buffer[(k-1)%2]
            // - Uses wmma::load_matrix_sync for accumulator to properly accumulate
            //==================================================================
            constexpr int q_tile_elems = T::BLOCK_SIZE_M * T::K_TILE_DIM;
            constexpr int q_per_thread = (q_tile_elems + T::NUM_THREADS - 1) / T::NUM_THREADS;
            constexpr int k_tile_elems = T::PAGE_BLOCK_SIZE * T::K_TILE_DIM;
            constexpr int k_per_thread = (k_tile_elems + T::NUM_THREADS - 1) / T::NUM_THREADS;

            // Load first K tile into buffer 0
            int k_tile = 0;
            int k_offset = k_tile * T::K_TILE_DIM;
            InputT* sK_cur = sK_tiles[0];

            #pragma unroll
            for (int i = 0; i < k_per_thread; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS;
                if (idx < k_tile_elems) {
                    int row = idx / T::K_TILE_DIM;
                    int col = idx % T::K_TILE_DIM;
                    sK_cur[row * T::K_TILE_DIM + col] = (row < valid_tokens) ?
                                        kv_block_ptr[row * params.k_row_stride + k_offset + col] :
                                        InputT(0);
                }
            }

            // Main QK GEMM loop with double-buffered K
            for (k_tile = 0; k_tile < T::NUM_K_TILES; ++k_tile) {
                k_offset = k_tile * T::K_TILE_DIM;
                int cur_buf = k_tile % 2;
                int next_buf = (k_tile + 1) % 2;
                sK_cur = sK_tiles[cur_buf];
                InputT* sK_next = sK_tiles[next_buf];

                // Load Q_tile [BLOCK_SIZE_M, K_TILE_DIM] = [64, 64]
                // Use raw linear storage (row-major) for WMMA compatibility
                InputT* sQ_raw_store = smem.qk_phase.smem_Q_tile.data();

                #pragma unroll
                for (int i = 0; i < q_per_thread; ++i) {
                    int idx = thread_idx + i * T::NUM_THREADS;
                    if (idx < q_tile_elems) {
                        int row = idx / T::K_TILE_DIM;
                        int col = idx % T::K_TILE_DIM;
                        int q_row = row + m_block_idx * T::BLOCK_SIZE_M;
                        if (q_row < params.q_seq_per_hk) {
                            sQ_raw_store[row * T::K_TILE_DIM + col] = q_ptr[row * params.q_row_stride + k_offset + col];
                        } else {
                            sQ_raw_store[row * T::K_TILE_DIM + col] = InputT(0);
                        }
                    }
                }

                // Prefetch next K tile into the alternate buffer (for future TMA overlap)
                if (k_tile + 1 < T::NUM_K_TILES) {
                    int next_k_offset = (k_tile + 1) * T::K_TILE_DIM;
                    #pragma unroll
                    for (int i = 0; i < k_per_thread; ++i) {
                        int idx = thread_idx + i * T::NUM_THREADS;
                        if (idx < k_tile_elems) {
                            int row = idx / T::K_TILE_DIM;
                            int col = idx % T::K_TILE_DIM;
                            sK_next[row * T::K_TILE_DIM + col] = (row < valid_tokens) ?
                                                kv_block_ptr[row * params.k_row_stride + next_k_offset + col] :
                                                InputT(0);
                        }
                    }
                }
                __syncthreads();

                // WMMA QK GEMM using tensor cores - only 4 warps (warp_idx 0-3) do WMMA
                // Warps 4-7 help with memory ops but skip WMMA to avoid OOB access
                // Computes: Scores[64, 64] += Q_tile[64, 64] @ K_tile[64, 64]^T
                if (warp_idx < 4) {
                    wmma_gemm_qk<InputT>(
                        smem.qk_phase.smem_Q_tile.data(),
                        sK_cur,
                        sScores,
                        warp_idx,
                        lane_idx,
                        (k_tile == 0)  // is_first_k_tile: init on first, accumulate on rest
                    );
                }
                __syncthreads();
            }

            //==================================================================
            // Phase 2: Online Softmax
            //==================================================================

            // Apply softmax scale
            const float scale = params.scale_softmax;

            #pragma unroll
            for (int i = 0; i < score_per_thread; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS;
                if (idx < score_elems) {
                    sScores[idx] *= scale;
                }
            }
            __syncthreads();

            // Row-wise max and online softmax update
            if (thread_idx < T::BLOCK_SIZE_M) {
                float row_max = NEGATIVE_INFINITY;
                #pragma unroll
                for (int j = 0; j < T::PAGE_BLOCK_SIZE; ++j) {
                    if (j < valid_tokens) {
                        row_max = fmaxf(row_max, sScores[thread_idx * T::PAGE_BLOCK_SIZE + j]);
                    }
                }

                // Update global max with rescaling
                float old_max = sM[thread_idx];
                float new_max = fmaxf(old_max, row_max);
                float rescale = (old_max == NEGATIVE_INFINITY) ? 1.0f :
                               exp2f((old_max - new_max) * LOG2E);

                sL[thread_idx] *= rescale;
                sM[thread_idx] = new_max;
                sScale[thread_idx] = rescale;
            }
            __syncthreads();

            // Compute exp(score - max) and accumulate row sum
            if (thread_idx < T::BLOCK_SIZE_M) {
                float row_max = sM[thread_idx];
                float row_sum = 0.0f;

                #pragma unroll
                for (int j = 0; j < T::PAGE_BLOCK_SIZE; ++j) {
                    int idx = thread_idx * T::PAGE_BLOCK_SIZE + j;
                    if (j < valid_tokens) {
                        float exp_val = exp2f((sScores[idx] - row_max) * LOG2E);
                        sScores[idx] = exp_val;
                        row_sum += exp_val;
                    } else {
                        sScores[idx] = 0.0f;
                    }
                }
                sL[thread_idx] += row_sum;
            }
            __syncthreads();

            // Rescale previous output accumulator (flat layout)
            // Thread i owns output indices {i, i+256, i+512, ...}
            #pragma unroll
            for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS;
                int row = idx / T::HEAD_DIM_V;
                if (row < T::BLOCK_SIZE_M) {
                    rO[i] *= sScale[row];
                }
            }

            //==================================================================
            // Phase 3: PV GEMM (P @ V) - Warpgroup-split approach
            // Warps 0-3: process V columns 0-255 (tiles 0-3)
            // Warps 4-7: process V columns 256-511 (tiles 4-7)
            // Each warpgroup handles its 256 columns sequentially (4 tiles)
            // Output stored directly to registers with warpgroup-local indexing
            //==================================================================

            // Convert scores to bf16/fp16 and store in P
            #pragma unroll
            for (int i = 0; i < score_per_thread; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS;
                if (idx < score_elems) {
                    sP.data()[idx] = InputT(sScores[idx]);
                }
            }
            __syncthreads();

            // Load V [PAGE_BLOCK_SIZE, HEAD_DIM_V] = [64, 512]
            const InputT* v_block_ptr = kv_block_ptr;
            constexpr int v_elems = T::PAGE_BLOCK_SIZE * T::HEAD_DIM_V;
            constexpr int v_per_thread = (v_elems + T::NUM_THREADS - 1) / T::NUM_THREADS;

            #pragma unroll
            for (int i = 0; i < v_per_thread; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS;
                if (idx < v_elems) {
                    int row = idx / T::HEAD_DIM_V;
                    int col = idx % T::HEAD_DIM_V;
                    sV.data()[idx] = (row < valid_tokens) ?
                                     v_block_ptr[row * params.k_row_stride + col] :
                                     InputT(0);
                }
            }
            __syncthreads();

            // PV GEMM - All 8 V tiles processed sequentially
            // Only 4 warps (0-3) do WMMA, covering 64 rows (16 rows each)
            // All 256 threads read results and accumulate to their registers
            constexpr int V_TILE_DIM = 64;
            constexpr int NUM_V_TILES = T::HEAD_DIM_V / V_TILE_DIM;  // 8

            #pragma unroll 2
            for (int v_tile = 0; v_tile < NUM_V_TILES; ++v_tile) {
                // Only first 4 warps compute WMMA (warp_idx 0-3)
                if (warp_idx < 4) {
                    wmma_gemm_pv_tiled<InputT>(
                        smem.smem_P.data(),
                        smem.smem_V.data(),
                        sScores,           // Output tile buffer [64, 64]
                        rO,
                        warp_idx,          // 0-3 for WMMA
                        lane_idx,
                        thread_idx,
                        v_tile
                    );
                }
                __syncthreads();

                // All 256 threads read from sScores to accumulate their owned elements
                const int tile_col_start = v_tile * V_TILE_DIM;

                #pragma unroll
                for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) {
                    // Flat layout: thread i owns indices {i, i+256, i+512, ...}
                    int global_out_idx = thread_idx + i * T::NUM_THREADS;
                    int row = global_out_idx / T::HEAD_DIM_V;
                    int global_col = global_out_idx % T::HEAD_DIM_V;

                    // Check if this column is in the current tile
                    if (global_col >= tile_col_start && global_col < tile_col_start + V_TILE_DIM) {
                        int col_in_tile = global_col - tile_col_start;
                        int tile_idx = row * V_TILE_DIM + col_in_tile;

                        if (row < T::BLOCK_SIZE_M && tile_idx < score_elems) {
                            rO[i] += sScores[tile_idx];
                        }
                    }
                }
                __syncthreads();
            }
        }

        //==================================================================
        // Final: Normalize and store output (flat layout)
        // Thread i writes indices {i, i+256, i+512, ...}
        //==================================================================
        const int num_valid_seq_q = min(params.q_seq_per_hk - m_block_idx * T::BLOCK_SIZE_M, T::BLOCK_SIZE_M);

        #pragma unroll
        for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) {
            int idx = thread_idx + i * T::NUM_THREADS;
            int row = idx / T::HEAD_DIM_V;
            int col = idx % T::HEAD_DIM_V;

            if (row < num_valid_seq_q) {
                float inv_sum = 1.0f / (sL[row] + 1e-6f);
                int o_global_idx = row * params.o_row_stride + col;
                o_ptr[o_global_idx] = InputT(rO[i] * inv_sum);
            }
        }

        // Store LSE: log-sum-exp = max + log(sum_exp)
        // cur_M is the max score value (already scaled by 1/sqrt(d))
        // cur_L is sum of exp(score - cur_M), so LSE = cur_M + log(cur_L)
        if (thread_idx < num_valid_seq_q) {
            float cur_L = sL[thread_idx];
            float cur_M = sM[thread_idx];
            lse_ptr[thread_idx] = (cur_L <= 0.0f || cur_L != cur_L) ?
                                  float(1e30) :
                                  (cur_M + logf(cur_L));
        }

        __syncthreads();
    }
}

//==============================================================================
// Kernel Launch Functions
//==============================================================================
template<typename InputT>
void run_flash_splitkv_mla_kernel_impl(DecodingParams& params, cudaStream_t stream) {
    using T = Traits<InputT>;

    const int num_m_blocks = (params.q_seq_per_hk + T::BLOCK_SIZE_M - 1) / T::BLOCK_SIZE_M;

    // Batch-parallel mode: grid.z = batch_size instead of num_sm_parts
    // This launches one thread block per (m_block, head, batch) tuple
    // Total thread blocks = num_m_blocks * h_k * batch_size
    // For typical config (128 batches, 2 m_blocks, 1 head): 256 thread blocks
    dim3 grid(num_m_blocks, params.h_k, params.b);
    dim3 block(T::NUM_THREADS);

    constexpr size_t smem_size = sizeof(typename T::SharedMemoryPlan);

    cudaFuncSetAttribute(
        flash_fwd_splitkv_mla_sm120_kernel<T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );

    flash_fwd_splitkv_mla_sm120_kernel<T><<<grid, block, smem_size, stream>>>(params);
}

void run_flash_splitkv_mla_kernel_bf16(DecodingParams& params, cudaStream_t stream) {
    run_flash_splitkv_mla_kernel_impl<cutlass::bfloat16_t>(params, stream);
}

void run_flash_splitkv_mla_kernel_fp16(DecodingParams& params, cudaStream_t stream) {
    run_flash_splitkv_mla_kernel_impl<cutlass::half_t>(params, stream);
}

}  // namespace sm120
