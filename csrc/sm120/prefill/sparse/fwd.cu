/***************************************************************************************************
 * SM120 Sparse Prefill Kernel - WMMA-based Implementation
 *
 * This kernel implements sparse attention prefill for SM120 (Blackwell workstation GPUs).
 * Uses WMMA 16x16x16 tensor core operations (no GMMA/UMMA on consumer Blackwell).
 *
 * Algorithm: O = softmax(Q @ K[indices]^T) @ V[indices]
 *
 * Key features:
 * - Sparse index-based KV access (top-k selection)
 * - Online softmax with rescaling for numerical stability
 * - Double-buffered K tiles for compute/memory overlap
 * - WMMA tensor cores for matrix operations
 *
 * Memory constraints: 99KB shared memory on SM120
 *
 * Thread organization (256 threads = 8 warps):
 * - Warps 0-3: WMMA computation (QK and PV GEMMs)
 * - Warps 4-7: Memory operations (Q, K, V, indices loading)
 *
 * References:
 * - SM90 sparse prefill: csrc/sm90/prefill/sparse/fwd.cu (GMMA-based, logic reference)
 * - SM120 dense decode: csrc/sm120/decode/dense/splitkv_mla.cu (WMMA patterns)
 **************************************************************************************************/

#include <cutlass/cutlass.h>
#include <cutlass/arch/memory_sm80.h>
#include <cute/tensor.hpp>
#include <mma.h>

#include "fwd.h"
#include "traits.h"
#include "params.h"
#include "fwd_mma.cuh"   // CFG>=4: raw mma.sync + ldmatrix + cp.async sparse kernel (F1 tier)

using namespace cute;
using namespace nvcuda;

namespace sm120 {

// Constants
static constexpr float NEGATIVE_INFINITY = -1e30f;
static constexpr float LOG2E = 1.4426950408889634f;

// WMMA configuration
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
// WMMA-based GEMM for QK computation (sparse)
// Computes: Scores[B_H, B_TOPK] += Q_tile[B_H, K_TILE] @ K_tile[B_TOPK, K_TILE]^T
//
// K_TILE_DIM = 64, WMMA_K = 16, so 4 inner iterations
// HEAD_DIM_K = 576 requires 9 outer iterations
//==============================================================================
template<typename InputT>
__device__ void wmma_gemm_qk_sparse(
    const InputT* __restrict__ sQ_tile,   // [64, 64] in smem
    const InputT* __restrict__ sK_tile,   // [64, 64] in smem
    float* __restrict__ sScores,          // [64, 64] accumulator in smem
    int warp_idx,
    int lane_idx,
    bool is_first_k_tile
) {
    using WmmaInputT = typename WmmaType<InputT>::type;

    // Only first 4 warps do WMMA computation
    if (warp_idx >= 4) return;

    // Each warp handles 16 rows (warp 0: rows 0-15, warp 1: rows 16-31, etc.)
    const int warp_m = warp_idx * WMMA_M;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, WmmaInputT, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, WmmaInputT, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Process 4 output column tiles (4 x 16 = 64)
    #pragma unroll
    for (int n_tile = 0; n_tile < 4; ++n_tile) {
        const int out_n = n_tile * WMMA_N;

        if (is_first_k_tile) {
            wmma::fill_fragment(c_frag, 0.0f);
        } else {
            wmma::load_matrix_sync(c_frag, sScores + warp_m * 64 + out_n, 64, wmma::mem_row_major);
        }

        // Process K dimension (4 x 16 = 64)
        #pragma unroll
        for (int k_tile = 0; k_tile < 4; ++k_tile) {
            const int k_offset = k_tile * WMMA_K;

            // Q[warp_m:warp_m+16, k_offset:k_offset+16]
            wmma::load_matrix_sync(a_frag,
                                   reinterpret_cast<const WmmaInputT*>(sQ_tile) + warp_m * 64 + k_offset,
                                   64);

            // K[out_n:out_n+16, k_offset:k_offset+16] loaded as col_major for K^T
            wmma::load_matrix_sync(b_frag,
                                   reinterpret_cast<const WmmaInputT*>(sK_tile) + out_n * 64 + k_offset,
                                   64);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        wmma::store_matrix_sync(sScores + warp_m * 64 + out_n, c_frag, 64, wmma::mem_row_major);
    }
}

//==============================================================================
// WMMA-based GEMM for PV computation (tiled output)
// Computes: O_tile[B_H, 64] = P[B_H, B_TOPK] @ V_slice[B_TOPK, 64]
//==============================================================================
template<typename InputT>
__device__ void wmma_gemm_pv_tiled(
    const InputT* __restrict__ sP,        // [64, 64] attention weights
    const InputT* __restrict__ sV,        // [64, 512] values
    float* __restrict__ sO_tile,          // [64, 64] output tile buffer
    int warp_idx,
    int lane_idx,
    int v_col_offset                      // Which 64-column tile of V (0, 64, 128, ...)
) {
    using WmmaInputT = typename WmmaType<InputT>::type;

    if (warp_idx >= 4) return;

    const int warp_m = warp_idx * WMMA_M;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, WmmaInputT, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, WmmaInputT, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    constexpr int B_TOPK = 64;
    constexpr int D_V = 512;
    constexpr int V_TILE_DIM = 64;

    // Process 4 output column tiles within the 64-column window
    #pragma unroll
    for (int n_tile = 0; n_tile < 4; ++n_tile) {
        const int out_n = n_tile * WMMA_N;  // 0, 16, 32, 48 within tile

        wmma::fill_fragment(c_frag, 0.0f);

        // K dimension: B_TOPK = 64, process in 4 tiles
        #pragma unroll
        for (int k_tile = 0; k_tile < 4; ++k_tile) {
            const int k_offset = k_tile * WMMA_K;

            // P[warp_m:warp_m+16, k_offset:k_offset+16]
            wmma::load_matrix_sync(a_frag,
                                   reinterpret_cast<const WmmaInputT*>(sP) + warp_m * B_TOPK + k_offset,
                                   B_TOPK);

            // V[k_offset:k_offset+16, v_col_offset+out_n:v_col_offset+out_n+16]
            wmma::load_matrix_sync(b_frag,
                                   reinterpret_cast<const WmmaInputT*>(sV) + k_offset * D_V + v_col_offset + out_n,
                                   D_V);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        wmma::store_matrix_sync(sO_tile + warp_m * V_TILE_DIM + out_n, c_frag, V_TILE_DIM, wmma::mem_row_major);
    }
}

//==============================================================================
// Main sparse prefill kernel
//
// CFG ladder (runtime-selected via FLASH_MLA_SM120_SPARSE_FWD_CFG, default 0; each level
// stacks on the previous per audit/design-prefill-configs.md 2.0 combination rule):
//   0  legacy: byte-identical original codegen.
//   1  A0 (design 2.1): FULL unroll of every rO[] loop -> compile-time indices -> rO in
//      registers (legacy `#pragma unroll 8` left runtime indices: measured 528 B local
//      frame/thread). Measured 1.84x (9.7 -> 17.9 TFlops on authors' perf cases).
//   2  A0+A1 (design 2.2): QK accumulator fragments PERSIST in registers across all 9
//      d-tiles (one store_matrix_sync at the end instead of a 16 KB smem round-trip per
//      d-tile) and ALL 8 warps issue WMMA (2 tiles each; legacy idles warps 4-7) in both
//      the QK and PV GEMMs. Same warp->tile map as the proven sparse-decode kernel.
//==============================================================================
template<typename T, int CFG = 0>
__global__ void __launch_bounds__(T::NUM_THREADS_TOTAL, 1)
sparse_prefill_fwd_kernel(const SparsePrefillParams params) {
    constexpr bool FULL_UNROLL_O = CFG >= 1;
    constexpr bool REG_ACC       = CFG >= 2;
    // CFG>=3 (A2+A3+A4, design 2.3): warp-parallel BASE-2 softmax (4 threads/row +
    // shuffle reductions, sm_scale*log2e folded once -- sM then holds the base-2-scaled
    // max and the epilogue emits max_logits = sM, lse = log2f(sL) + sM), and the
    // closed-form PV write-back (each thread's flat rO layout hits exactly one v_tile
    // per parity: v_tile == tid/64 -> even rO indices, v_tile == 4 + tid/64 -> odd).
    constexpr bool SCALAR_OPT    = CFG >= 3;
    using InputT = typename T::Input;
    using Plan = typename T::SharedMemoryPlan;

    // Shared memory
    extern __shared__ char smem_raw[];
    Plan& smem = *reinterpret_cast<Plan*>(smem_raw);

    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / 32;
    const int lane_idx = thread_idx % 32;

    // Grid indexing: (h_q / B_H) * s_q blocks
    // Each block processes B_H query heads for one query position
    const int h_q_block_idx = blockIdx.x % (params.h_q / T::BLOCK_SIZE_M);
    const int s_q_idx = blockIdx.x / (params.h_q / T::BLOCK_SIZE_M);

    // Pointers
    const InputT* q_ptr = reinterpret_cast<const InputT*>(params.q) +
                          s_q_idx * params.stride_q_s_q + h_q_block_idx * T::BLOCK_SIZE_M * params.stride_q_h_q;
    const InputT* kv_ptr = reinterpret_cast<const InputT*>(params.kv);
    const int* indices_ptr = params.indices + s_q_idx * params.stride_indices_s_q;
    InputT* out_ptr = reinterpret_cast<InputT*>(params.out) +
                      s_q_idx * (params.h_q * params.d_v) + h_q_block_idx * T::BLOCK_SIZE_M * params.d_v;

    // Output accumulator in registers
    float rO[T::O_ELEMS_PER_THREAD];
    if constexpr (FULL_UNROLL_O) {
        #pragma unroll
        for (int i = 0; i < T::O_ELEMS_PER_THREAD; ++i) {
            rO[i] = 0.0f;
        }
    } else {
        #pragma unroll 8
        for (int i = 0; i < T::O_ELEMS_PER_THREAD; ++i) {
            rO[i] = 0.0f;
        }
    }

    // Softmax statistics (initialized per block)
    float* sM = smem.smem_M.data();
    float* sL = smem.smem_L.data();
    float* sScale = smem.smem_scale.data();
    float* sScores = smem.smem_scores.data();

    // Initialize softmax stats
    if (thread_idx < T::BLOCK_SIZE_M) {
        sM[thread_idx] = NEGATIVE_INFINITY;
        sL[thread_idx] = 0.0f;
    }
    __syncthreads();

    // Number of topk blocks to process
    const int num_topk_blocks = params.topk / T::BLOCK_SIZE_N;

    // Process each topk block
    for (int topk_block = 0; topk_block < num_topk_blocks; ++topk_block) {
        // Load sparse indices for this block
        int* sIndices = smem.smem_indices.data();
        bool* sIsValid = smem.smem_is_valid.data();

        // Cooperative loading of indices and validity mask
        if (thread_idx < T::BLOCK_SIZE_N) {
            int global_idx = topk_block * T::BLOCK_SIZE_N + thread_idx;
            int token_idx = indices_ptr[global_idx];
            sIndices[thread_idx] = token_idx;
            sIsValid[thread_idx] = (token_idx >= 0 && token_idx < params.s_kv);
        }
        __syncthreads();

        //======================================================================
        // Phase 1: QK^T computation
        // Accumulate scores across all K tiles (9 for D_QK=576)
        //======================================================================
        InputT* sQ_tile = smem.qk_phase.smem_Q_tile.data();
        InputT* sK_tiles[2] = {smem.qk_phase.smem_K_tile0.data(), smem.qk_phase.smem_K_tile1.data()};

        // CFG>=2 (A1): accumulator fragments persist in registers across ALL d-tiles and
        // ALL 8 warps issue WMMA (t = warp + i*8 -> (m,n) = (t>>2, t&3), 2 tiles/warp --
        // the proven sparse-decode map). One store_matrix_sync at the end replaces the
        // legacy per-d-tile 16 KB load/store round-trip of sScores. Unused (elided) at CFG<2.
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> qk_acc[2];
        if constexpr (REG_ACC) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) wmma::fill_fragment(qk_acc[i], 0.0f);
        }

        for (int k_tile = 0; k_tile < T::NUM_K_ITERATIONS; ++k_tile) {
            const int k_offset = k_tile * T::K_TILE_SIZE;
            const int actual_k_cols = min(T::K_TILE_SIZE, params.d_qk - k_offset);

            // Load Q tile [B_H, K_TILE] cooperatively
            constexpr int q_tile_elems = T::BLOCK_SIZE_M * T::K_TILE_SIZE;
            constexpr int q_per_thread = (q_tile_elems + T::NUM_THREADS_TOTAL - 1) / T::NUM_THREADS_TOTAL;

            #pragma unroll
            for (int i = 0; i < q_per_thread; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS_TOTAL;
                if (idx < q_tile_elems) {
                    int row = idx / T::K_TILE_SIZE;
                    int col = idx % T::K_TILE_SIZE;
                    if (col < actual_k_cols) {
                        sQ_tile[idx] = q_ptr[row * params.stride_q_h_q + k_offset + col];
                    } else {
                        sQ_tile[idx] = InputT(0);
                    }
                }
            }

            // Load K tile [B_TOPK, K_TILE] using sparse indices
            InputT* sK_cur = sK_tiles[k_tile % 2];
            constexpr int k_tile_elems = T::BLOCK_SIZE_N * T::K_TILE_SIZE;
            constexpr int k_per_thread = (k_tile_elems + T::NUM_THREADS_TOTAL - 1) / T::NUM_THREADS_TOTAL;

            #pragma unroll
            for (int i = 0; i < k_per_thread; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS_TOTAL;
                if (idx < k_tile_elems) {
                    int row = idx / T::K_TILE_SIZE;
                    int col = idx % T::K_TILE_SIZE;
                    int token_idx = sIndices[row];
                    bool is_valid = sIsValid[row];

                    if (is_valid && col < actual_k_cols) {
                        sK_cur[idx] = kv_ptr[token_idx * params.stride_kv_s_kv + k_offset + col];
                    } else {
                        sK_cur[idx] = InputT(0);
                    }
                }
            }

            __syncthreads();

            if constexpr (REG_ACC) {
                // All 8 warps, 2 WMMA tiles each, accumulating into persistent qk_acc.
                using WmmaInputT = typename WmmaType<InputT>::type;
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int t = warp_idx + i * 8;
                    const int m = t >> 2, n = t & 3;
                    #pragma unroll
                    for (int k4 = 0; k4 < 4; ++k4) {
                        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, WmmaInputT, wmma::row_major> a_frag;
                        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, WmmaInputT, wmma::col_major> b_frag;
                        wmma::load_matrix_sync(a_frag,
                            reinterpret_cast<const WmmaInputT*>(sQ_tile) + m * 16 * 64 + k4 * 16, 64);
                        wmma::load_matrix_sync(b_frag,
                            reinterpret_cast<const WmmaInputT*>(sK_cur) + n * 16 * 64 + k4 * 16, 64);
                        wmma::mma_sync(qk_acc[i], a_frag, b_frag, qk_acc[i]);
                    }
                }
            } else {
                // WMMA QK computation (first 4 warps; helper self-guards warp_idx >= 4)
                wmma_gemm_qk_sparse<InputT>(sQ_tile, sK_cur, sScores, warp_idx, lane_idx, k_tile == 0);
            }

            __syncthreads();
        }

        if constexpr (REG_ACC) {
            // Single materialization of the full 64x64 fp32 score tile.
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int t = warp_idx + i * 8;
                const int m = t >> 2, n = t & 3;
                wmma::store_matrix_sync(sScores + m * 16 * T::BLOCK_SIZE_N + n * 16, qk_acc[i],
                                        T::BLOCK_SIZE_N, wmma::mem_row_major);
            }
            __syncthreads();
        }

        //======================================================================
        // Phase 2: Online softmax
        // Apply scale, compute max, rescale previous O, compute exp
        //======================================================================
        if constexpr (SCALAR_OPT) {
            // A2+A4: all 256 threads, 4 per row, BASE-2 domain (scale2 folded once).
            // sM holds the base-2-scaled max; formulas mirror the proven decode kernel.
            const float scale2 = params.sm_scale * LOG2E;
            const int r   = thread_idx >> 2;
            const int sub = thread_idx & 3;

            float s2[16];
            float cm = NEGATIVE_INFINITY;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int col = sub + 4 * j;
                s2[j] = sIsValid[col] ? (sScores[r * T::BLOCK_SIZE_N + col] * scale2)
                                      : NEGATIVE_INFINITY;
                cm = fmaxf(cm, s2[j]);
            }
            cm = fmaxf(cm, __shfl_xor_sync(0xffffffff, cm, 1));
            cm = fmaxf(cm, __shfl_xor_sync(0xffffffff, cm, 2));

            const float old_max = sM[r];
            const float new_max = fmaxf(old_max, cm);
            const float rescale = exp2f(old_max - new_max);   // -1e30 arithmetic stays finite

            float rs = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int col = sub + 4 * j;
                const int idx = r * T::BLOCK_SIZE_N + col;
                // Branch mandatory: all-invalid block has new_max == -1e30 and a branchless
                // exp2f(s2 - new_max) would emit 1.0 for every masked column.
                const float p = sIsValid[col] ? exp2f(s2[j] - new_max) : 0.0f;
                sScores[idx] = p;
                rs += p;
            }
            rs += __shfl_xor_sync(0xffffffff, rs, 1);
            rs += __shfl_xor_sync(0xffffffff, rs, 2);
            if (sub == 0) {
                sM[r]     = new_max;
                sL[r]     = sL[r] * rescale + rs;
                sScale[r] = rescale;
            }
        } else
        if (thread_idx < T::BLOCK_SIZE_M) {
            // Find row max
            float row_max = NEGATIVE_INFINITY;
            #pragma unroll
            for (int j = 0; j < T::BLOCK_SIZE_N; ++j) {
                if (sIsValid[j]) {
                    float val = sScores[thread_idx * T::BLOCK_SIZE_N + j] * params.sm_scale;
                    row_max = fmaxf(row_max, val);
                }
            }

            // Update global max and compute rescale factor
            float old_max = sM[thread_idx];
            float new_max = fmaxf(old_max, row_max);
            float rescale = (old_max == NEGATIVE_INFINITY) ? 1.0f : exp2f((old_max - new_max) * LOG2E);

            sL[thread_idx] *= rescale;
            sM[thread_idx] = new_max;
            sScale[thread_idx] = rescale;

            // Compute exp(score - new_max) and accumulate sum
            float row_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < T::BLOCK_SIZE_N; ++j) {
                int idx = thread_idx * T::BLOCK_SIZE_N + j;
                if (sIsValid[j]) {
                    float exp_val = exp2f((sScores[idx] * params.sm_scale - new_max) * LOG2E);
                    sScores[idx] = exp_val;
                    row_sum += exp_val;
                } else {
                    sScores[idx] = 0.0f;
                }
            }
            sL[thread_idx] += row_sum;
        }
        __syncthreads();

        // Rescale previous output accumulator
        if constexpr (FULL_UNROLL_O) {
            #pragma unroll
            for (int i = 0; i < T::O_ELEMS_PER_THREAD; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS_TOTAL;
                int row = idx / T::HEAD_DIM_V;
                if (row < T::BLOCK_SIZE_M) {
                    rO[i] *= sScale[row];
                }
            }
        } else {
            #pragma unroll 8
            for (int i = 0; i < T::O_ELEMS_PER_THREAD; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS_TOTAL;
                int row = idx / T::HEAD_DIM_V;
                if (row < T::BLOCK_SIZE_M) {
                    rO[i] *= sScale[row];
                }
            }
        }
        __syncthreads();

        // Convert scores to InputT for P matrix
        InputT* sP = smem.smem_P.data();
        constexpr int p_elems = T::BLOCK_SIZE_M * T::BLOCK_SIZE_N;
        constexpr int p_per_thread = (p_elems + T::NUM_THREADS_TOTAL - 1) / T::NUM_THREADS_TOTAL;

        #pragma unroll
        for (int i = 0; i < p_per_thread; ++i) {
            int idx = thread_idx + i * T::NUM_THREADS_TOTAL;
            if (idx < p_elems) {
                sP[idx] = InputT(sScores[idx]);
            }
        }
        __syncthreads();

        //======================================================================
        // Phase 3: Load V and compute PV
        //======================================================================
        InputT* sV = smem.smem_V.data();
        constexpr int v_elems = T::BLOCK_SIZE_N * T::HEAD_DIM_V;
        constexpr int v_per_thread = (v_elems + T::NUM_THREADS_TOTAL - 1) / T::NUM_THREADS_TOTAL;

        // Load V using sparse indices
        #pragma unroll
        for (int i = 0; i < v_per_thread; ++i) {
            int idx = thread_idx + i * T::NUM_THREADS_TOTAL;
            if (idx < v_elems) {
                int row = idx / T::HEAD_DIM_V;
                int col = idx % T::HEAD_DIM_V;
                int token_idx = sIndices[row];
                bool is_valid = sIsValid[row];

                if (is_valid) {
                    // MLA latent KV: V is the FIRST d_v elements of the 576-wide KV row
                    // (the latent part, aliased with K's nope portion). Authors' reference:
                    // result = attn_score @ kvs[:, :, :d_v] (test_flash_mla_prefill.py), and
                    // sm90 fwd.cu SmemLayoutHalfV = transposed view of K tiles 0..7.
                    // (Was `+ params.d_qk + col`: an OOB read into the NEXT token's K data.)
                    sV[idx] = kv_ptr[token_idx * params.stride_kv_s_kv + col];
                } else {
                    sV[idx] = InputT(0);
                }
            }
        }
        __syncthreads();

        // PV GEMM - process in 64-column tiles
        float* sO_tile = smem.smem_O_tile.data();

        for (int v_tile = 0; v_tile < T::NUM_V_ITERATIONS; ++v_tile) {
            const int v_col_offset = v_tile * T::K_TILE_SIZE;

            if constexpr (REG_ACC) {
                // All 8 warps, 2 WMMA tiles each (legacy helper idles warps 4-7).
                using WmmaInputT = typename WmmaType<InputT>::type;
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int t = warp_idx + i * 8;
                    const int m = t >> 2, n = t & 3;
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, WmmaInputT, wmma::row_major> pa;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, WmmaInputT, wmma::row_major> vb;
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pc;
                    wmma::fill_fragment(pc, 0.0f);
                    #pragma unroll
                    for (int k4 = 0; k4 < 4; ++k4) {
                        wmma::load_matrix_sync(pa,
                            reinterpret_cast<const WmmaInputT*>(sP) + m * 16 * T::BLOCK_SIZE_N + k4 * 16,
                            T::BLOCK_SIZE_N);
                        wmma::load_matrix_sync(vb,
                            reinterpret_cast<const WmmaInputT*>(sV) + (k4 * 16) * T::HEAD_DIM_V + v_col_offset + n * 16,
                            T::HEAD_DIM_V);
                        wmma::mma_sync(pc, pa, vb, pc);
                    }
                    wmma::store_matrix_sync(sO_tile + m * 16 * T::K_TILE_SIZE + n * 16, pc,
                                            T::K_TILE_SIZE, wmma::mem_row_major);
                }
            } else {
                // WMMA PV computation
                wmma_gemm_pv_tiled<InputT>(sP, sV, sO_tile, warp_idx, lane_idx, v_col_offset);
            }

            __syncthreads();

            // Accumulate to register output
            constexpr int tile_elems = T::BLOCK_SIZE_M * T::K_TILE_SIZE;

            if constexpr (FULL_UNROLL_O) {
                if constexpr (SCALAR_OPT) {
                    // A3 closed-form: thread's flat rO holds (row, col) pairs with only TWO
                    // col values -- c0 = tid (< 512 for 256 threads, even i) and c1 = tid+256
                    // (odd i). A 64-wide v_tile window therefore matches exactly one parity:
                    //   v_tile == tid/64      -> even indices rO[2r], col-in-tile = tid & 63
                    //   v_tile == 4 + tid/64  -> odd  indices rO[2r+1], same col-in-tile
                    // 64 direct adds when active, 0 otherwise (legacy scans all 128 with
                    // guards, 7/8 wasted). Branch is warp-uniform (tid/64 constant per 2 warps).
                    const int home = thread_idx / 64;
                    const int cit  = thread_idx & 63;
                    if (v_tile == home) {
                        #pragma unroll
                        for (int rrow = 0; rrow < T::BLOCK_SIZE_M; ++rrow) {
                            rO[2 * rrow] += sO_tile[rrow * T::K_TILE_SIZE + cit];
                        }
                    } else if (v_tile == 4 + home) {
                        #pragma unroll
                        for (int rrow = 0; rrow < T::BLOCK_SIZE_M; ++rrow) {
                            rO[2 * rrow + 1] += sO_tile[rrow * T::K_TILE_SIZE + cit];
                        }
                    }
                } else {
                #pragma unroll
                for (int i = 0; i < T::O_ELEMS_PER_THREAD; ++i) {
                    int global_idx = thread_idx + i * T::NUM_THREADS_TOTAL;
                    int row = global_idx / T::HEAD_DIM_V;
                    int global_col = global_idx % T::HEAD_DIM_V;

                    if (global_col >= v_col_offset && global_col < v_col_offset + T::K_TILE_SIZE) {
                        int col_in_tile = global_col - v_col_offset;
                        int tile_idx = row * T::K_TILE_SIZE + col_in_tile;

                        if (row < T::BLOCK_SIZE_M && tile_idx < tile_elems) {
                            rO[i] += sO_tile[tile_idx];
                        }
                    }
                }
                }
            } else {
                #pragma unroll 8
                for (int i = 0; i < T::O_ELEMS_PER_THREAD; ++i) {
                    int global_idx = thread_idx + i * T::NUM_THREADS_TOTAL;
                    int row = global_idx / T::HEAD_DIM_V;
                    int global_col = global_idx % T::HEAD_DIM_V;

                    // Check if this column is in current tile
                    if (global_col >= v_col_offset && global_col < v_col_offset + T::K_TILE_SIZE) {
                        int col_in_tile = global_col - v_col_offset;
                        int tile_idx = row * T::K_TILE_SIZE + col_in_tile;

                        if (row < T::BLOCK_SIZE_M && tile_idx < tile_elems) {
                            rO[i] += sO_tile[tile_idx];
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    //==========================================================================
    // Epilogue: Scale output by 1/L and store
    //==========================================================================
    __syncthreads();

    if constexpr (FULL_UNROLL_O) {
        #pragma unroll
        for (int i = 0; i < T::O_ELEMS_PER_THREAD; ++i) {
            int idx = thread_idx + i * T::NUM_THREADS_TOTAL;
            int row = idx / T::HEAD_DIM_V;
            int col = idx % T::HEAD_DIM_V;

            if (row < T::BLOCK_SIZE_M && col < params.d_v) {
                float inv_sum = 1.0f / (sL[row] + 1e-6f);
                out_ptr[row * params.d_v + col] = InputT(rO[i] * inv_sum);
            }
        }
    } else {
        #pragma unroll 8
        for (int i = 0; i < T::O_ELEMS_PER_THREAD; ++i) {
            int idx = thread_idx + i * T::NUM_THREADS_TOTAL;
            int row = idx / T::HEAD_DIM_V;
            int col = idx % T::HEAD_DIM_V;

            if (row < T::BLOCK_SIZE_M && col < params.d_v) {
                float inv_sum = 1.0f / (sL[row] + 1e-6f);
                out_ptr[row * params.d_v + col] = InputT(rO[i] * inv_sum);
            }
        }
    }

    // Store max_logits and LSE
    if (thread_idx < T::BLOCK_SIZE_M) {
        int global_h_idx = h_q_block_idx * T::BLOCK_SIZE_M + thread_idx;
        int out_idx = s_q_idx * params.h_q + global_h_idx;

        if (global_h_idx < params.h_q) {
            if constexpr (SCALAR_OPT) {
                // Base-2 softmax (A4): sM ALREADY holds max(S * sm_scale * log2e).
                params.max_logits[out_idx] = sM[thread_idx];
                params.lse[out_idx] = log2f(sL[thread_idx]) + sM[thread_idx];  // 2-based LSE
            } else {
                // Authors' convention (test_flash_mla_prefill.py reference_torch): max_logits is
                // the max of scores AFTER `attn_score *= sm_scale * log2(e)`. sM holds
                // max(S*sm_scale) in natural units, so max_logits = sM * LOG2E. (Was
                // sM / sm_scale = raw unscaled max — the 15.635 rel-err fingerprint.)
                params.max_logits[out_idx] = sM[thread_idx] * LOG2E;
                params.lse[out_idx] = log2f(sL[thread_idx]) + sM[thread_idx] * LOG2E;  // 2-based LSE
            }
        }
    }
}

//==============================================================================
// Kernel launcher
//==============================================================================
void run_sparse_fwd_kernel(const SparsePrefillParams& params) {
    // Validation
    CUTLASS_ASSERT(params.h_kv == 1);  // MLA requires single KV head
    CUTLASS_ASSERT(params.topk % sparse_prefill::B_TOPK == 0);  // topk must be multiple of block size
    CUTLASS_ASSERT(params.h_q % sparse_prefill::B_H == 0);  // h_q must be multiple of block size
    CUTLASS_ASSERT(params.d_qk == sparse_prefill::D_QK);  // Fixed MLA dimensions
    CUTLASS_ASSERT(params.d_v == sparse_prefill::D_V);

    // Grid: one block per (h_q_block, s_q) pair
    const int num_h_blocks = params.h_q / sparse_prefill::B_H;
    const dim3 grid(num_h_blocks * params.s_q, 1, 1);
    const dim3 block(sparse_prefill::NUM_THREADS, 1, 1);

    // Launch based on input type (assume bfloat16 for MLA)
    using Traits = sparse_prefill::TraitsBF16;
    const size_t smem_size = Traits::SMEM_SIZE;

    // Runtime config selection (build once, A/B at runtime; default 0 = byte-identical
    // legacy codegen). FLASH_MLA_SM120_SPARSE_FWD_CFG:
    //   0 (default) legacy; 1 = A0 register-rO; 2 = +A1 persistent QK frags + 8-warp WMMA;
    //   3 = +A2/A3/A4 parallel base-2 softmax + closed-form PV write-back;
    //   4 = F1: raw mma.sync + ldmatrix + XOR swizzle + cp.async gather pipeline
    //       (fwd_mma.cuh -- resident Q, streamed gathered KV, V-share semantics).
    static const int fwd_cfg = []() {
        const char* e = getenv("FLASH_MLA_SM120_SPARSE_FWD_CFG");
        int v = e ? atoi(e) : 0;
        return (v < 0 || v > 4) ? 0 : v;
    }();

    if (fwd_cfg == 4) {
        sparse_mma::launch_sparse_fwd_mma(params);
        return;
    }

    auto launch = [&](auto kernel_fn) {
        if (smem_size > 48 * 1024) {
            cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }
        kernel_fn<<<grid, block, smem_size, params.stream>>>(params);
    };

    switch (fwd_cfg) {
        case 3:  launch(sparse_prefill_fwd_kernel<Traits, 3>); break;
        case 2:  launch(sparse_prefill_fwd_kernel<Traits, 2>); break;
        case 1:  launch(sparse_prefill_fwd_kernel<Traits, 1>); break;
        default: launch(sparse_prefill_fwd_kernel<Traits, 0>); break;
    }
}

} // namespace sm120
