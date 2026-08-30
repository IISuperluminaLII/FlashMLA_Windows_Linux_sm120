/***************************************************************************************************
 * WMMA Tensor Core Microbenchmark for SM120 (Blackwell Workstation)
 *
 * Benchmarks isolated WMMA mma.sync.aligned bf16 m16n16k16 operations that match
 * the FlashMLA backward pass kernel (fmha_bwd_kernel_sm120.cuh).
 *
 * Tests:
 * 1. QK^T matmul: row_major A x col_major B (attention scores)
 * 2. dO @ V^T matmul: row_major A x col_major B (dP computation)
 * 3. dScores @ K matmul: row_major A x row_major B (dQ computation)
 *
 * Tile configuration (matches BWD kernel):
 * - BWD_BLOCK_M = 16 (queries per block)
 * - BWD_BLOCK_N = 32 (keys per block)
 * - BWD_BLOCK_D = 128 (head dimension)
 * - NUM_WARPS = 4 (128 threads total)
 *
 * WMMA tile dimensions: 16x16x16 (bf16)
 *
 * Compile:
 *   nvcc -arch=sm_120 -O3 -o microbench_wmma microbench_wmma.cu
 *
 * Run:
 *   ./microbench_wmma [num_iterations]
 *
 **************************************************************************************************/

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ==============================================================================
// CONFIGURATION CONSTANTS (matching fmha_bwd_kernel_sm120.cuh)
// ==============================================================================

// WMMA tile dimensions for bf16 m16n16k16
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Backward kernel block configuration
constexpr int BWD_BLOCK_M = 16;   // Queries per block (matches Q tile rows)
constexpr int BWD_BLOCK_N = 32;   // Keys per block (matches K/V tile rows)
constexpr int BWD_BLOCK_D = 128;  // Head dimension

// Thread configuration
constexpr int NUM_WARPS = 4;
constexpr int NUM_THREADS = NUM_WARPS * 32;  // 128 threads

// Benchmark parameters
constexpr int DEFAULT_ITERATIONS = 1000;
constexpr int WARMUP_ITERATIONS = 100;

// ==============================================================================
// CUDA ERROR CHECKING
// ==============================================================================

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            printf("[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__,               \
                   cudaGetErrorString(err));                                     \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

// ==============================================================================
// SHARED MEMORY LAYOUT (simplified version for benchmarking)
// ==============================================================================

struct alignas(256) BenchSmemLayout {
    // Q tile: BWD_BLOCK_M x BWD_BLOCK_D (16 x 128) bf16
    static constexpr size_t q_tile_offset = 0;
    static constexpr size_t q_tile_size = BWD_BLOCK_M * BWD_BLOCK_D * sizeof(__nv_bfloat16);

    // K tile: BWD_BLOCK_N x BWD_BLOCK_D (32 x 128) bf16
    static constexpr size_t k_tile_offset = ((q_tile_size + 255) / 256) * 256;
    static constexpr size_t k_tile_size = BWD_BLOCK_N * BWD_BLOCK_D * sizeof(__nv_bfloat16);

    // V tile: BWD_BLOCK_N x BWD_BLOCK_D (32 x 128) bf16
    static constexpr size_t v_tile_offset = k_tile_offset + ((k_tile_size + 255) / 256) * 256;
    static constexpr size_t v_tile_size = BWD_BLOCK_N * BWD_BLOCK_D * sizeof(__nv_bfloat16);

    // dO tile: BWD_BLOCK_M x BWD_BLOCK_D (16 x 128) bf16
    static constexpr size_t do_tile_offset = v_tile_offset + ((v_tile_size + 255) / 256) * 256;
    static constexpr size_t do_tile_size = BWD_BLOCK_M * BWD_BLOCK_D * sizeof(__nv_bfloat16);

    // Scores tile: BWD_BLOCK_M x BWD_BLOCK_N (16 x 32) float
    static constexpr size_t scores_offset = do_tile_offset + ((do_tile_size + 255) / 256) * 256;
    static constexpr size_t scores_size = BWD_BLOCK_M * BWD_BLOCK_N * sizeof(float);

    // dScores bf16: BWD_BLOCK_M x BWD_BLOCK_N (16 x 32) bf16
    static constexpr size_t dscores_bf16_offset = scores_offset + ((scores_size + 255) / 256) * 256;
    static constexpr size_t dscores_bf16_size = BWD_BLOCK_M * BWD_BLOCK_N * sizeof(__nv_bfloat16);

    // dQ accumulator: BWD_BLOCK_M x BWD_BLOCK_D (16 x 128) float
    static constexpr size_t dq_acc_offset = dscores_bf16_offset + ((dscores_bf16_size + 255) / 256) * 256;
    static constexpr size_t dq_acc_size = BWD_BLOCK_M * BWD_BLOCK_D * sizeof(float);

    static constexpr size_t total_size = dq_acc_offset + ((dq_acc_size + 255) / 256) * 256;
};

// ==============================================================================
// KERNEL 1: QK^T MATMUL BENCHMARK
// Computes: scores[M,N] = Q[M,D] @ K^T[D,N] = Q[M,D] @ K[N,D]^T
// Matrix A: row_major, Matrix B: col_major (transposed K)
// This matches compute_qk_scores_wmma in the backward kernel.
// ==============================================================================

__global__ void __launch_bounds__(NUM_THREADS)
benchmark_qkt_wmma(int iterations) {
    using namespace nvcuda::wmma;

    extern __shared__ char smem_base[];
    __nv_bfloat16* q_tile = reinterpret_cast<__nv_bfloat16*>(smem_base + BenchSmemLayout::q_tile_offset);
    __nv_bfloat16* k_tile = reinterpret_cast<__nv_bfloat16*>(smem_base + BenchSmemLayout::k_tile_offset);
    float* scores = reinterpret_cast<float*>(smem_base + BenchSmemLayout::scores_offset);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    // Initialize shared memory with some values (prevents compiler optimization)
    for (int i = tid; i < BWD_BLOCK_M * BWD_BLOCK_D; i += NUM_THREADS) {
        q_tile[i] = __float2bfloat16(0.01f * (i % 16));
    }
    for (int i = tid; i < BWD_BLOCK_N * BWD_BLOCK_D; i += NUM_THREADS) {
        k_tile[i] = __float2bfloat16(0.01f * (i % 16));
    }
    __syncthreads();

    // Tile counts
    const int m_tiles = BWD_BLOCK_M / WMMA_M;  // 16/16 = 1
    const int n_tiles = BWD_BLOCK_N / WMMA_N;  // 32/16 = 2
    const int d_tiles = BWD_BLOCK_D / WMMA_K;  // 128/16 = 8

    // Main benchmark loop
    for (int iter = 0; iter < iterations; ++iter) {
        // Each warp handles one (m_tile, n_tile) pair
        for (int mn = warp_id; mn < m_tiles * n_tiles; mn += NUM_WARPS) {
            int m_tile = mn / n_tiles;
            int n_tile = mn % n_tiles;

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            fill_fragment(acc, 0.0f);

            // Accumulate over D dimension (8 WMMA tiles for D=128)
            #pragma unroll
            for (int k_tile = 0; k_tile < d_tiles; ++k_tile) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;

                // Load Q tile (row-major): Q[m_tile*16:(m_tile+1)*16, k_tile*16:(k_tile+1)*16]
                // Stride = BWD_BLOCK_D = 128
                load_matrix_sync(a_frag,
                    q_tile + m_tile * WMMA_M * BWD_BLOCK_D + k_tile * WMMA_K,
                    BWD_BLOCK_D);

                // Load K tile (col-major for transposed access): K[n_tile*16:(n_tile+1)*16, k_tile*16:(k_tile+1)*16]
                // Stride = BWD_BLOCK_D = 128
                load_matrix_sync(b_frag,
                    k_tile + n_tile * WMMA_N * BWD_BLOCK_D + k_tile * WMMA_K,
                    BWD_BLOCK_D);

                mma_sync(acc, a_frag, b_frag, acc);
            }

            // Store result to scores
            store_matrix_sync(scores + m_tile * WMMA_M * BWD_BLOCK_N + n_tile * WMMA_N,
                              acc, BWD_BLOCK_N, mem_row_major);
        }
        __syncthreads();
    }
}

// ==============================================================================
// KERNEL 2: dO @ V^T MATMUL BENCHMARK
// Computes: dP[M,N] = dO[M,D] @ V^T[D,N] = dO[M,D] @ V[N,D]^T
// Matrix A: row_major, Matrix B: col_major (transposed V)
// This matches compute_dp_wmma in the backward kernel.
// ==============================================================================

__global__ void __launch_bounds__(NUM_THREADS)
benchmark_dovt_wmma(int iterations) {
    using namespace nvcuda::wmma;

    extern __shared__ char smem_base[];
    __nv_bfloat16* do_tile = reinterpret_cast<__nv_bfloat16*>(smem_base + BenchSmemLayout::do_tile_offset);
    __nv_bfloat16* v_tile = reinterpret_cast<__nv_bfloat16*>(smem_base + BenchSmemLayout::v_tile_offset);
    float* scores = reinterpret_cast<float*>(smem_base + BenchSmemLayout::scores_offset);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    // Initialize shared memory
    for (int i = tid; i < BWD_BLOCK_M * BWD_BLOCK_D; i += NUM_THREADS) {
        do_tile[i] = __float2bfloat16(0.01f * (i % 16));
    }
    for (int i = tid; i < BWD_BLOCK_N * BWD_BLOCK_D; i += NUM_THREADS) {
        v_tile[i] = __float2bfloat16(0.01f * (i % 16));
    }
    __syncthreads();

    const int m_tiles = BWD_BLOCK_M / WMMA_M;  // 1
    const int n_tiles = BWD_BLOCK_N / WMMA_N;  // 2
    const int d_tiles = BWD_BLOCK_D / WMMA_K;  // 8

    for (int iter = 0; iter < iterations; ++iter) {
        for (int mn = warp_id; mn < m_tiles * n_tiles; mn += NUM_WARPS) {
            int m_tile = mn / n_tiles;
            int n_tile = mn % n_tiles;

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            fill_fragment(acc, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < d_tiles; ++k_tile) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> do_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> v_frag;

                load_matrix_sync(do_frag,
                    do_tile + m_tile * WMMA_M * BWD_BLOCK_D + k_tile * WMMA_K,
                    BWD_BLOCK_D);
                load_matrix_sync(v_frag,
                    v_tile + n_tile * WMMA_N * BWD_BLOCK_D + k_tile * WMMA_K,
                    BWD_BLOCK_D);

                mma_sync(acc, do_frag, v_frag, acc);
            }

            store_matrix_sync(scores + m_tile * WMMA_M * BWD_BLOCK_N + n_tile * WMMA_N,
                              acc, BWD_BLOCK_N, mem_row_major);
        }
        __syncthreads();
    }
}

// ==============================================================================
// KERNEL 3: dScores @ K MATMUL BENCHMARK
// Computes: dQ[M,D] = dScores[M,N] @ K[N,D]
// Matrix A: row_major, Matrix B: row_major
// This matches compute_dq_wmma in the backward kernel.
// ==============================================================================

__global__ void __launch_bounds__(NUM_THREADS)
benchmark_dsk_wmma(int iterations) {
    using namespace nvcuda::wmma;

    extern __shared__ char smem_base[];
    __nv_bfloat16* dscores_bf16 = reinterpret_cast<__nv_bfloat16*>(smem_base + BenchSmemLayout::dscores_bf16_offset);
    __nv_bfloat16* k_tile = reinterpret_cast<__nv_bfloat16*>(smem_base + BenchSmemLayout::k_tile_offset);
    float* dq_acc = reinterpret_cast<float*>(smem_base + BenchSmemLayout::dq_acc_offset);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    // Initialize shared memory
    for (int i = tid; i < BWD_BLOCK_M * BWD_BLOCK_N; i += NUM_THREADS) {
        dscores_bf16[i] = __float2bfloat16(0.01f * (i % 16));
    }
    for (int i = tid; i < BWD_BLOCK_N * BWD_BLOCK_D; i += NUM_THREADS) {
        k_tile[i] = __float2bfloat16(0.01f * (i % 16));
    }
    for (int i = tid; i < BWD_BLOCK_M * BWD_BLOCK_D; i += NUM_THREADS) {
        dq_acc[i] = 0.0f;
    }
    __syncthreads();

    // For dQ = dScores @ K:
    // dScores: [M, N] = [16, 32] -> m_tiles=1, k_tiles=2 (k here is N)
    // K: [N, D] = [32, 128] -> n_tiles=8 (n here is D)
    const int m_tiles = BWD_BLOCK_M / WMMA_M;     // 1
    const int d_tiles = BWD_BLOCK_D / WMMA_N;     // 8 (output columns)
    const int n_tiles = BWD_BLOCK_N / WMMA_K;     // 2 (reduction dimension)

    for (int iter = 0; iter < iterations; ++iter) {
        // Reset accumulator
        for (int i = tid; i < BWD_BLOCK_M * BWD_BLOCK_D; i += NUM_THREADS) {
            dq_acc[i] = 0.0f;
        }
        __syncthreads();

        for (int md = warp_id; md < m_tiles * d_tiles; md += NUM_WARPS) {
            int m_tile = md / d_tiles;
            int d_tile = md % d_tiles;

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            fill_fragment(acc, 0.0f);

            #pragma unroll
            for (int k_tile_idx = 0; k_tile_idx < n_tiles; ++k_tile_idx) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> ds_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> k_frag;

                // dScores[m_tile*16:(m_tile+1)*16, k_tile*16:(k_tile+1)*16], stride=BWD_BLOCK_N
                load_matrix_sync(ds_frag,
                    dscores_bf16 + m_tile * WMMA_M * BWD_BLOCK_N + k_tile_idx * WMMA_K,
                    BWD_BLOCK_N);

                // K[k_tile*16:(k_tile+1)*16, d_tile*16:(d_tile+1)*16], stride=BWD_BLOCK_D
                load_matrix_sync(k_frag,
                    k_tile + k_tile_idx * WMMA_K * BWD_BLOCK_D + d_tile * WMMA_N,
                    BWD_BLOCK_D);

                mma_sync(acc, ds_frag, k_frag, acc);
            }

            // Store to dQ accumulator
            store_matrix_sync(dq_acc + m_tile * WMMA_M * BWD_BLOCK_D + d_tile * WMMA_N,
                              acc, BWD_BLOCK_D, mem_row_major);
        }
        __syncthreads();
    }
}

// ==============================================================================
// COMBINED KERNEL: Full backward pass WMMA operations
// Tests all three operations in sequence (as in actual backward kernel)
// ==============================================================================

__global__ void __launch_bounds__(NUM_THREADS)
benchmark_combined_wmma(int iterations) {
    using namespace nvcuda::wmma;

    extern __shared__ char smem_base[];
    __nv_bfloat16* q_tile = reinterpret_cast<__nv_bfloat16*>(smem_base + BenchSmemLayout::q_tile_offset);
    __nv_bfloat16* k_tile = reinterpret_cast<__nv_bfloat16*>(smem_base + BenchSmemLayout::k_tile_offset);
    __nv_bfloat16* v_tile = reinterpret_cast<__nv_bfloat16*>(smem_base + BenchSmemLayout::v_tile_offset);
    __nv_bfloat16* do_tile = reinterpret_cast<__nv_bfloat16*>(smem_base + BenchSmemLayout::do_tile_offset);
    float* scores = reinterpret_cast<float*>(smem_base + BenchSmemLayout::scores_offset);
    __nv_bfloat16* dscores_bf16 = reinterpret_cast<__nv_bfloat16*>(smem_base + BenchSmemLayout::dscores_bf16_offset);
    float* dq_acc = reinterpret_cast<float*>(smem_base + BenchSmemLayout::dq_acc_offset);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    // Initialize all tiles
    for (int i = tid; i < BWD_BLOCK_M * BWD_BLOCK_D; i += NUM_THREADS) {
        q_tile[i] = __float2bfloat16(0.01f * (i % 16));
        do_tile[i] = __float2bfloat16(0.01f * (i % 16));
    }
    for (int i = tid; i < BWD_BLOCK_N * BWD_BLOCK_D; i += NUM_THREADS) {
        k_tile[i] = __float2bfloat16(0.01f * (i % 16));
        v_tile[i] = __float2bfloat16(0.01f * (i % 16));
    }
    __syncthreads();

    const int m_tiles = BWD_BLOCK_M / WMMA_M;
    const int n_tiles = BWD_BLOCK_N / WMMA_N;
    const int d_tiles = BWD_BLOCK_D / WMMA_K;

    for (int iter = 0; iter < iterations; ++iter) {
        // Step 1: QK^T (scores = Q @ K^T)
        for (int mn = warp_id; mn < m_tiles * n_tiles; mn += NUM_WARPS) {
            int m_tile = mn / n_tiles;
            int n_tile = mn % n_tiles;

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            fill_fragment(acc, 0.0f);

            #pragma unroll
            for (int k_tile_idx = 0; k_tile_idx < d_tiles; ++k_tile_idx) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;

                load_matrix_sync(a_frag, q_tile + m_tile * WMMA_M * BWD_BLOCK_D + k_tile_idx * WMMA_K, BWD_BLOCK_D);
                load_matrix_sync(b_frag, k_tile + n_tile * WMMA_N * BWD_BLOCK_D + k_tile_idx * WMMA_K, BWD_BLOCK_D);
                mma_sync(acc, a_frag, b_frag, acc);
            }

            store_matrix_sync(scores + m_tile * WMMA_M * BWD_BLOCK_N + n_tile * WMMA_N, acc, BWD_BLOCK_N, mem_row_major);
        }
        __syncthreads();

        // Step 2: dO @ V^T (dP = dO @ V^T, reusing scores buffer)
        for (int mn = warp_id; mn < m_tiles * n_tiles; mn += NUM_WARPS) {
            int m_tile = mn / n_tiles;
            int n_tile = mn % n_tiles;

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            fill_fragment(acc, 0.0f);

            #pragma unroll
            for (int k_tile_idx = 0; k_tile_idx < d_tiles; ++k_tile_idx) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> do_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> v_frag;

                load_matrix_sync(do_frag, do_tile + m_tile * WMMA_M * BWD_BLOCK_D + k_tile_idx * WMMA_K, BWD_BLOCK_D);
                load_matrix_sync(v_frag, v_tile + n_tile * WMMA_N * BWD_BLOCK_D + k_tile_idx * WMMA_K, BWD_BLOCK_D);
                mma_sync(acc, do_frag, v_frag, acc);
            }

            store_matrix_sync(scores + m_tile * WMMA_M * BWD_BLOCK_N + n_tile * WMMA_N, acc, BWD_BLOCK_N, mem_row_major);
        }
        __syncthreads();

        // Convert scores to bf16 for dScores (simulate dScores computation)
        for (int i = tid; i < BWD_BLOCK_M * BWD_BLOCK_N; i += NUM_THREADS) {
            dscores_bf16[i] = __float2bfloat16(scores[i] * 0.1f);
        }
        __syncthreads();

        // Step 3: dScores @ K (dQ = dScores @ K)
        const int d_out_tiles = BWD_BLOCK_D / WMMA_N;  // 8
        const int k_tiles = BWD_BLOCK_N / WMMA_K;      // 2

        for (int i = tid; i < BWD_BLOCK_M * BWD_BLOCK_D; i += NUM_THREADS) {
            dq_acc[i] = 0.0f;
        }
        __syncthreads();

        for (int md = warp_id; md < m_tiles * d_out_tiles; md += NUM_WARPS) {
            int m_tile = md / d_out_tiles;
            int d_tile = md % d_out_tiles;

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            fill_fragment(acc, 0.0f);

            #pragma unroll
            for (int k_tile_idx = 0; k_tile_idx < k_tiles; ++k_tile_idx) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> ds_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> k_frag;

                load_matrix_sync(ds_frag, dscores_bf16 + m_tile * WMMA_M * BWD_BLOCK_N + k_tile_idx * WMMA_K, BWD_BLOCK_N);
                load_matrix_sync(k_frag, k_tile + k_tile_idx * WMMA_K * BWD_BLOCK_D + d_tile * WMMA_N, BWD_BLOCK_D);
                mma_sync(acc, ds_frag, k_frag, acc);
            }

            store_matrix_sync(dq_acc + m_tile * WMMA_M * BWD_BLOCK_D + d_tile * WMMA_N, acc, BWD_BLOCK_D, mem_row_major);
        }
        __syncthreads();
    }
}

// ==============================================================================
// HELPER FUNCTIONS
// ==============================================================================

double compute_tflops_qkt(double time_ms, int iterations) {
    // QK^T: M=16, N=32, K=128 (head_dim)
    // FLOPs per matmul = 2 * M * N * K = 2 * 16 * 32 * 128 = 131072
    double flops_per_iter = 2.0 * BWD_BLOCK_M * BWD_BLOCK_N * BWD_BLOCK_D;
    double total_flops = flops_per_iter * iterations;
    double time_sec = time_ms / 1000.0;
    return (total_flops / time_sec) / 1e12;
}

double compute_tflops_dovt(double time_ms, int iterations) {
    // dO @ V^T: same dimensions as QK^T
    double flops_per_iter = 2.0 * BWD_BLOCK_M * BWD_BLOCK_N * BWD_BLOCK_D;
    double total_flops = flops_per_iter * iterations;
    double time_sec = time_ms / 1000.0;
    return (total_flops / time_sec) / 1e12;
}

double compute_tflops_dsk(double time_ms, int iterations) {
    // dScores @ K: M=16, N=128 (head_dim), K=32 (block_n)
    double flops_per_iter = 2.0 * BWD_BLOCK_M * BWD_BLOCK_D * BWD_BLOCK_N;
    double total_flops = flops_per_iter * iterations;
    double time_sec = time_ms / 1000.0;
    return (total_flops / time_sec) / 1e12;
}

double compute_tflops_combined(double time_ms, int iterations) {
    // All three operations combined
    double flops_qkt = 2.0 * BWD_BLOCK_M * BWD_BLOCK_N * BWD_BLOCK_D;
    double flops_dovt = 2.0 * BWD_BLOCK_M * BWD_BLOCK_N * BWD_BLOCK_D;
    double flops_dsk = 2.0 * BWD_BLOCK_M * BWD_BLOCK_D * BWD_BLOCK_N;
    double flops_per_iter = flops_qkt + flops_dovt + flops_dsk;
    double total_flops = flops_per_iter * iterations;
    double time_sec = time_ms / 1000.0;
    return (total_flops / time_sec) / 1e12;
}

void print_config() {
    printf("================================================================================\n");
    printf("WMMA Tensor Core Microbenchmark for SM120 Backward Kernel\n");
    printf("================================================================================\n");
    printf("\n");
    printf("Configuration:\n");
    printf("  WMMA tile size:     %d x %d x %d (bf16)\n", WMMA_M, WMMA_N, WMMA_K);
    printf("  BWD_BLOCK_M:        %d (queries per block)\n", BWD_BLOCK_M);
    printf("  BWD_BLOCK_N:        %d (keys per block)\n", BWD_BLOCK_N);
    printf("  BWD_BLOCK_D:        %d (head dimension)\n", BWD_BLOCK_D);
    printf("  NUM_WARPS:          %d\n", NUM_WARPS);
    printf("  NUM_THREADS:        %d\n", NUM_THREADS);
    printf("  Shared memory:      %zu bytes\n", BenchSmemLayout::total_size);
    printf("\n");
    printf("WMMA tiles per operation:\n");
    printf("  QK^T:      M_tiles=%d, N_tiles=%d, K_tiles=%d\n",
           BWD_BLOCK_M/WMMA_M, BWD_BLOCK_N/WMMA_N, BWD_BLOCK_D/WMMA_K);
    printf("  dO@V^T:    M_tiles=%d, N_tiles=%d, K_tiles=%d\n",
           BWD_BLOCK_M/WMMA_M, BWD_BLOCK_N/WMMA_N, BWD_BLOCK_D/WMMA_K);
    printf("  dS@K:      M_tiles=%d, N_tiles=%d, K_tiles=%d\n",
           BWD_BLOCK_M/WMMA_M, BWD_BLOCK_D/WMMA_N, BWD_BLOCK_N/WMMA_K);
    printf("\n");
    printf("FLOPs per iteration:\n");
    printf("  QK^T:      %.0f FLOPs (2*%d*%d*%d)\n",
           2.0*BWD_BLOCK_M*BWD_BLOCK_N*BWD_BLOCK_D, BWD_BLOCK_M, BWD_BLOCK_N, BWD_BLOCK_D);
    printf("  dO@V^T:    %.0f FLOPs (2*%d*%d*%d)\n",
           2.0*BWD_BLOCK_M*BWD_BLOCK_N*BWD_BLOCK_D, BWD_BLOCK_M, BWD_BLOCK_N, BWD_BLOCK_D);
    printf("  dS@K:      %.0f FLOPs (2*%d*%d*%d)\n",
           2.0*BWD_BLOCK_M*BWD_BLOCK_D*BWD_BLOCK_N, BWD_BLOCK_M, BWD_BLOCK_D, BWD_BLOCK_N);
    printf("  Combined:  %.0f FLOPs\n",
           3.0*2.0*BWD_BLOCK_M*BWD_BLOCK_N*BWD_BLOCK_D);
    printf("\n");
}

int main(int argc, char** argv) {
    int iterations = DEFAULT_ITERATIONS;
    if (argc > 1) {
        iterations = atoi(argv[1]);
        if (iterations < 1) iterations = DEFAULT_ITERATIONS;
    }

    print_config();

    // Query device
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\n", prop.name);
    printf("SM version: %d.%d\n", prop.major, prop.minor);
    printf("Max shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Max shared memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("\n");

    // Check shared memory requirement
    if (BenchSmemLayout::total_size > prop.sharedMemPerBlock) {
        printf("[ERROR] Required shared memory (%zu) exceeds device limit (%zu)\n",
               BenchSmemLayout::total_size, prop.sharedMemPerBlock);
        return 1;
    }

    printf("Running benchmarks with %d iterations (+ %d warmup)...\n", iterations, WARMUP_ITERATIONS);
    printf("\n");

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 grid(1);
    dim3 block(NUM_THREADS);
    size_t smem_size = BenchSmemLayout::total_size;

    // Set max dynamic shared memory
    CUDA_CHECK(cudaFuncSetAttribute(benchmark_qkt_wmma,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(benchmark_dovt_wmma,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(benchmark_dsk_wmma,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(benchmark_combined_wmma,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    float time_ms;

    // ===========================================================================
    // Benchmark 1: QK^T (attention scores)
    // ===========================================================================
    printf("--------------------------------------------------------------------------------\n");
    printf("Benchmark 1: QK^T matmul (Q @ K^T, row_major x col_major)\n");
    printf("  Matrix A: Q [%d x %d] bf16 row_major\n", BWD_BLOCK_M, BWD_BLOCK_D);
    printf("  Matrix B: K [%d x %d] bf16 col_major (transposed)\n", BWD_BLOCK_N, BWD_BLOCK_D);
    printf("  Output C: scores [%d x %d] fp32\n", BWD_BLOCK_M, BWD_BLOCK_N);
    printf("--------------------------------------------------------------------------------\n");

    // Warmup
    benchmark_qkt_wmma<<<grid, block, smem_size>>>(WARMUP_ITERATIONS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    benchmark_qkt_wmma<<<grid, block, smem_size>>>(iterations);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    printf("  Time:     %.3f ms\n", time_ms);
    printf("  TFLOPS:   %.3f\n", compute_tflops_qkt(time_ms, iterations));
    printf("\n");

    // ===========================================================================
    // Benchmark 2: dO @ V^T (dP computation)
    // ===========================================================================
    printf("--------------------------------------------------------------------------------\n");
    printf("Benchmark 2: dO @ V^T matmul (row_major x col_major)\n");
    printf("  Matrix A: dO [%d x %d] bf16 row_major\n", BWD_BLOCK_M, BWD_BLOCK_D);
    printf("  Matrix B: V [%d x %d] bf16 col_major (transposed)\n", BWD_BLOCK_N, BWD_BLOCK_D);
    printf("  Output C: dP [%d x %d] fp32\n", BWD_BLOCK_M, BWD_BLOCK_N);
    printf("--------------------------------------------------------------------------------\n");

    // Warmup
    benchmark_dovt_wmma<<<grid, block, smem_size>>>(WARMUP_ITERATIONS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    benchmark_dovt_wmma<<<grid, block, smem_size>>>(iterations);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    printf("  Time:     %.3f ms\n", time_ms);
    printf("  TFLOPS:   %.3f\n", compute_tflops_dovt(time_ms, iterations));
    printf("\n");

    // ===========================================================================
    // Benchmark 3: dScores @ K (dQ computation)
    // ===========================================================================
    printf("--------------------------------------------------------------------------------\n");
    printf("Benchmark 3: dScores @ K matmul (row_major x row_major)\n");
    printf("  Matrix A: dScores [%d x %d] bf16 row_major\n", BWD_BLOCK_M, BWD_BLOCK_N);
    printf("  Matrix B: K [%d x %d] bf16 row_major\n", BWD_BLOCK_N, BWD_BLOCK_D);
    printf("  Output C: dQ [%d x %d] fp32\n", BWD_BLOCK_M, BWD_BLOCK_D);
    printf("--------------------------------------------------------------------------------\n");

    // Warmup
    benchmark_dsk_wmma<<<grid, block, smem_size>>>(WARMUP_ITERATIONS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    benchmark_dsk_wmma<<<grid, block, smem_size>>>(iterations);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    printf("  Time:     %.3f ms\n", time_ms);
    printf("  TFLOPS:   %.3f\n", compute_tflops_dsk(time_ms, iterations));
    printf("\n");

    // ===========================================================================
    // Benchmark 4: Combined (all three operations)
    // ===========================================================================
    printf("--------------------------------------------------------------------------------\n");
    printf("Benchmark 4: Combined backward WMMA operations\n");
    printf("  Sequence: QK^T -> dO@V^T -> dScores@K\n");
    printf("--------------------------------------------------------------------------------\n");

    // Warmup
    benchmark_combined_wmma<<<grid, block, smem_size>>>(WARMUP_ITERATIONS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    benchmark_combined_wmma<<<grid, block, smem_size>>>(iterations);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    printf("  Time:     %.3f ms\n", time_ms);
    printf("  TFLOPS:   %.3f\n", compute_tflops_combined(time_ms, iterations));
    printf("\n");

    // ===========================================================================
    // Summary
    // ===========================================================================
    printf("================================================================================\n");
    printf("SUMMARY\n");
    printf("================================================================================\n");
    printf("All benchmarks completed successfully.\n");
    printf("\n");
    printf("Note: TFLOPS values represent isolated WMMA tensor core throughput.\n");
    printf("Actual kernel performance will be lower due to:\n");
    printf("  - Memory load/store operations\n");
    printf("  - Softmax and other non-WMMA computations\n");
    printf("  - Synchronization overhead\n");
    printf("  - Global memory atomics for dK/dV\n");
    printf("\n");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("[DONE] Microbenchmark completed.\n");
    return 0;
}
