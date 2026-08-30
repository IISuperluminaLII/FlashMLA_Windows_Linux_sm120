/******************************************************************************
 * Microbenchmark: dK/dV Accumulation Strategies for FlashMLA Backward Pass
 *
 * This file benchmarks different approaches for the dK/dV accumulation which
 * is the main bottleneck in the backward pass:
 *   dK[n,d] = sum_m(dScores[m,n] * Q[m,d])
 *   dV[n,d] = sum_m(P[m,n] * dO[m,d])
 *
 * Implementations compared:
 * 1. Current (baseline): Threads parallel over D, serial over M, serial over N
 * 2. Warp-cooperative: Warp handles (n,d) pair, lanes reduce over M
 * 3. WMMA-based: Use tensor cores for dK = dScores^T @ Q
 * 4. Thread-block cooperative: All threads reduce over M using shared memory
 *
 * Compile: nvcc -arch=sm_120 -O3 -std=c++17 microbench_dkdv.cu -o microbench_dkdv
 *
 * Author: Generated for FlashMLA optimization analysis
 *****************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>

// ============================================================================
// Configuration
// ============================================================================
constexpr int M_SIZE = 16;        // Number of queries (rows)
constexpr int N_SIZE = 32;        // Number of keys (columns)
constexpr int HEAD_DIM = 128;     // Head dimension
constexpr int NUM_THREADS = 128;  // 4 warps
constexpr int NUM_WARPS = NUM_THREADS / 32;
constexpr int WARMUP_ITERS = 10;
constexpr int BENCH_ITERS = 100;

// WMMA tile dimensions (m16n16k16 for bf16)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// ============================================================================
// CUDA Error Checking
// ============================================================================
#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t _e = (stmt);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "[CUDA ERROR] %s failed: %s (%d) at %s:%d\n", #stmt,     \
              cudaGetErrorString(_e), static_cast<int>(_e), __FILE__,          \
              __LINE__);                                                       \
      fflush(stderr);                                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ============================================================================
// GPU Timer using CUDA events
// ============================================================================
struct GpuTimer {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_event));
    }

    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
        return ms;
    }
};

// ============================================================================
// Host utility: LCG pseudo-random number generator
// ============================================================================
inline uint32_t lcg(uint32_t& state) {
    state = 1664525u * state + 1013904223u;
    return state;
}

// Fill array with random bf16 values in range [-0.5, 0.5]
void fill_random_bf16(float* host_data, size_t count, uint32_t seed = 42) {
    for (size_t i = 0; i < count; ++i) {
        seed = lcg(seed);
        host_data[i] = static_cast<float>(seed & 0xFFFF) / 65535.0f - 0.5f;
    }
}

// ============================================================================
// Kernel 1: Current Implementation (Baseline)
// Threads parallel over D, serial loop over M, outer serial loop over N
// This matches the pattern in fmha_bwd_kernel_sm120.cuh
// ============================================================================
__global__ void __launch_bounds__(NUM_THREADS)
kernel_current_baseline(
    const float* __restrict__ dscores,  // [M, N]
    const float* __restrict__ q,        // [M, D]
    const float* __restrict__ p,        // [M, N] softmax probabilities
    const float* __restrict__ d_o,      // [M, D] gradient of output
    float* __restrict__ dk,             // [N, D]
    float* __restrict__ dv,             // [N, D]
    int m_size, int n_size, int head_dim
) {
    const int tid = threadIdx.x;

    // dK computation: dK[n,d] = sum_m(dScores[m,n] * Q[m,d])
    for (int n = 0; n < n_size; ++n) {
        for (int d = tid; d < head_dim; d += NUM_THREADS) {
            float sum = 0.0f;
            for (int m = 0; m < m_size; ++m) {
                float ds = dscores[m * n_size + n];
                float q_val = q[m * head_dim + d];
                sum += ds * q_val;
            }
            atomicAdd(&dk[n * head_dim + d], sum);
        }
    }

    // dV computation: dV[n,d] = sum_m(P[m,n] * dO[m,d])
    for (int n = 0; n < n_size; ++n) {
        for (int d = tid; d < head_dim; d += NUM_THREADS) {
            float sum = 0.0f;
            for (int m = 0; m < m_size; ++m) {
                float prob = p[m * n_size + n];
                float do_val = d_o[m * head_dim + d];
                sum += prob * do_val;
            }
            atomicAdd(&dv[n * head_dim + d], sum);
        }
    }
}

// ============================================================================
// Kernel 2: Warp-Cooperative Reduction
// Each warp handles one (n, d) output element
// Lanes within warp reduce over M dimension using shuffle instructions
// ============================================================================
__device__ __forceinline__ float warp_reduce_sum(float val) {
    const unsigned int FULL_MASK = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__global__ void __launch_bounds__(NUM_THREADS)
kernel_warp_cooperative(
    const float* __restrict__ dscores,  // [M, N]
    const float* __restrict__ q,        // [M, D]
    const float* __restrict__ p,        // [M, N]
    const float* __restrict__ d_o,      // [M, D]
    float* __restrict__ dk,             // [N, D]
    float* __restrict__ dv,             // [N, D]
    int m_size, int n_size, int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Total output elements: N * D
    const int total_outputs = n_size * head_dim;

    // Each warp processes multiple (n, d) pairs
    for (int out_idx = warp_id; out_idx < total_outputs; out_idx += NUM_WARPS) {
        int n = out_idx / head_dim;
        int d = out_idx % head_dim;

        // dK: Each lane accumulates partial sum over M
        float dk_sum = 0.0f;
        float dv_sum = 0.0f;

        // Lanes cooperatively reduce over M (each lane handles M/32 elements if M >= 32)
        // For M=16, lanes 0-15 each handle one M element
        for (int m = lane_id; m < m_size; m += 32) {
            float ds = dscores[m * n_size + n];
            float q_val = q[m * head_dim + d];
            dk_sum += ds * q_val;

            float prob = p[m * n_size + n];
            float do_val = d_o[m * head_dim + d];
            dv_sum += prob * do_val;
        }

        // Warp-level reduction
        dk_sum = warp_reduce_sum(dk_sum);
        dv_sum = warp_reduce_sum(dv_sum);

        // Lane 0 writes result
        if (lane_id == 0) {
            atomicAdd(&dk[n * head_dim + d], dk_sum);
            atomicAdd(&dv[n * head_dim + d], dv_sum);
        }
    }
}

// ============================================================================
// Kernel 3: WMMA-Based (Tensor Core)
// Compute dK = dScores^T @ Q using WMMA mma.sync
// dScores: [M, N], Q: [M, D] => dK: [N, D] = dScores^T @ Q
// ============================================================================
__global__ void __launch_bounds__(NUM_THREADS)
kernel_wmma_based(
    const __nv_bfloat16* __restrict__ dscores_bf16,  // [M, N] in bf16
    const __nv_bfloat16* __restrict__ q_bf16,        // [M, D] in bf16
    const __nv_bfloat16* __restrict__ p_bf16,        // [M, N] in bf16
    const __nv_bfloat16* __restrict__ d_o_bf16,      // [M, D] in bf16
    float* __restrict__ dk,                           // [N, D]
    float* __restrict__ dv,                           // [N, D]
    int m_size, int n_size, int head_dim
) {
    using namespace nvcuda::wmma;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Tile dimensions for WMMA
    const int n_tiles = (n_size + WMMA_M - 1) / WMMA_M;  // N dimension tiles
    const int d_tiles = (head_dim + WMMA_N - 1) / WMMA_N;  // D dimension tiles
    const int m_tiles = (m_size + WMMA_K - 1) / WMMA_K;    // M dimension tiles (reduction)

    // Each warp handles one (n_tile, d_tile) output tile
    for (int nd = warp_id; nd < n_tiles * d_tiles; nd += NUM_WARPS) {
        int n_tile = nd / d_tiles;
        int d_tile = nd % d_tiles;

        // dK accumulator
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> dk_acc;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> dv_acc;
        fill_fragment(dk_acc, 0.0f);
        fill_fragment(dv_acc, 0.0f);

        // Reduce over M dimension
        for (int k_tile = 0; k_tile < m_tiles; ++k_tile) {
            // Load dScores^T fragment: need col_major since we're transposing
            // dScores: [M, N], we want dScores^T: [N, M]
            // A matrix: dScores^T [N, M] stored as col_major = dScores row_major
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> ds_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> q_frag;

            // dScores is [M, N] row-major, we load col_major to get transpose
            // Ptr to dScores[k_tile*K, n_tile*M]
            const __nv_bfloat16* ds_ptr = dscores_bf16 + k_tile * WMMA_K * n_size + n_tile * WMMA_M;
            const __nv_bfloat16* q_ptr = q_bf16 + k_tile * WMMA_K * head_dim + d_tile * WMMA_N;

            load_matrix_sync(ds_frag, ds_ptr, n_size);  // stride = N for col_major view of row_major data
            load_matrix_sync(q_frag, q_ptr, head_dim);

            mma_sync(dk_acc, ds_frag, q_frag, dk_acc);

            // Similarly for dV = P^T @ dO
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> p_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> do_frag;

            const __nv_bfloat16* p_ptr = p_bf16 + k_tile * WMMA_K * n_size + n_tile * WMMA_M;
            const __nv_bfloat16* do_ptr = d_o_bf16 + k_tile * WMMA_K * head_dim + d_tile * WMMA_N;

            load_matrix_sync(p_frag, p_ptr, n_size);
            load_matrix_sync(do_frag, do_ptr, head_dim);

            mma_sync(dv_acc, p_frag, do_frag, dv_acc);
        }

        // Store accumulated results using atomicAdd
        // WMMA stores 16x16 tile, need to atomically add to global
        __shared__ float dk_staging[NUM_WARPS * WMMA_M * WMMA_N];
        __shared__ float dv_staging[NUM_WARPS * WMMA_M * WMMA_N];

        float* dk_stage = &dk_staging[warp_id * WMMA_M * WMMA_N];
        float* dv_stage = &dv_staging[warp_id * WMMA_M * WMMA_N];

        store_matrix_sync(dk_stage, dk_acc, WMMA_N, mem_row_major);
        store_matrix_sync(dv_stage, dv_acc, WMMA_N, mem_row_major);
        __syncwarp();

        // Each lane writes part of the 16x16 tile
        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int local_n = i / WMMA_N;
            int local_d = i % WMMA_N;
            int global_n = n_tile * WMMA_M + local_n;
            int global_d = d_tile * WMMA_N + local_d;

            if (global_n < n_size && global_d < head_dim) {
                atomicAdd(&dk[global_n * head_dim + global_d], dk_stage[i]);
                atomicAdd(&dv[global_n * head_dim + global_d], dv_stage[i]);
            }
        }
    }
}

// ============================================================================
// Kernel 4: Thread-Block Cooperative Reduction
// All threads cooperate to reduce over M using shared memory
// ============================================================================
__global__ void __launch_bounds__(NUM_THREADS)
kernel_block_cooperative(
    const float* __restrict__ dscores,  // [M, N]
    const float* __restrict__ q,        // [M, D]
    const float* __restrict__ p,        // [M, N]
    const float* __restrict__ d_o,      // [M, D]
    float* __restrict__ dk,             // [N, D]
    float* __restrict__ dv,             // [N, D]
    int m_size, int n_size, int head_dim
) {
    const int tid = threadIdx.x;

    // Shared memory for partial sums
    // We process D elements in chunks, using shared memory for M-reduction
    __shared__ float dk_partial[NUM_THREADS];
    __shared__ float dv_partial[NUM_THREADS];

    // Process each (n, d) output element
    for (int n = 0; n < n_size; ++n) {
        // Process D in chunks to fit in shared memory
        for (int d_base = 0; d_base < head_dim; d_base += NUM_THREADS) {
            int d = d_base + tid;

            if (d < head_dim) {
                // Each thread computes partial sum over a subset of M
                float dk_sum = 0.0f;
                float dv_sum = 0.0f;

                // Divide M across threads (when M < NUM_THREADS, some threads idle)
                int m_per_thread = (m_size + NUM_THREADS - 1) / NUM_THREADS;
                int m_start = tid * m_per_thread;
                int m_end = min(m_start + m_per_thread, m_size);

                // However, for small M, we want all threads to help
                // Alternative: each thread processes the full M (but for different d)
                for (int m = 0; m < m_size; ++m) {
                    float ds = dscores[m * n_size + n];
                    float q_val = q[m * head_dim + d];
                    dk_sum += ds * q_val;

                    float prob = p[m * n_size + n];
                    float do_val = d_o[m * head_dim + d];
                    dv_sum += prob * do_val;
                }

                // Write directly (no reduction needed across threads for same d)
                atomicAdd(&dk[n * head_dim + d], dk_sum);
                atomicAdd(&dv[n * head_dim + d], dv_sum);
            }
        }
    }
}

// ============================================================================
// Kernel 5: Optimized Thread-Parallel with Register Blocking
// Each thread handles multiple D elements, maximizing ILP
// ============================================================================
constexpr int BLOCK_D = 4;  // Each thread handles 4 D elements

__global__ void __launch_bounds__(NUM_THREADS)
kernel_register_blocked(
    const float* __restrict__ dscores,  // [M, N]
    const float* __restrict__ q,        // [M, D]
    const float* __restrict__ p,        // [M, N]
    const float* __restrict__ d_o,      // [M, D]
    float* __restrict__ dk,             // [N, D]
    float* __restrict__ dv,             // [N, D]
    int m_size, int n_size, int head_dim
) {
    const int tid = threadIdx.x;

    // Each thread handles BLOCK_D consecutive D elements
    // Total D coverage per iteration: NUM_THREADS * BLOCK_D = 128 * 4 = 512
    // For HEAD_DIM=128, one iteration covers all D

    for (int n = 0; n < n_size; ++n) {
        // Process D in chunks
        for (int d_base = tid * BLOCK_D; d_base < head_dim; d_base += NUM_THREADS * BLOCK_D) {
            // Register accumulator for BLOCK_D elements
            float dk_acc[BLOCK_D] = {0.0f};
            float dv_acc[BLOCK_D] = {0.0f};

            // Reduce over M
            for (int m = 0; m < m_size; ++m) {
                float ds = dscores[m * n_size + n];
                float prob = p[m * n_size + n];

                #pragma unroll
                for (int bd = 0; bd < BLOCK_D; ++bd) {
                    int d = d_base + bd;
                    if (d < head_dim) {
                        float q_val = q[m * head_dim + d];
                        float do_val = d_o[m * head_dim + d];

                        dk_acc[bd] += ds * q_val;
                        dv_acc[bd] += prob * do_val;
                    }
                }
            }

            // Write results
            #pragma unroll
            for (int bd = 0; bd < BLOCK_D; ++bd) {
                int d = d_base + bd;
                if (d < head_dim && dk_acc[bd] != 0.0f) {
                    atomicAdd(&dk[n * head_dim + d], dk_acc[bd]);
                }
                if (d < head_dim && dv_acc[bd] != 0.0f) {
                    atomicAdd(&dv[n * head_dim + d], dv_acc[bd]);
                }
            }
        }
    }
}

// ============================================================================
// Main Benchmark Function
// ============================================================================
int main() {
    printf("========================================================\n");
    printf("dK/dV Accumulation Microbenchmark\n");
    printf("========================================================\n");
    printf("Configuration:\n");
    printf("  M (queries)     = %d\n", M_SIZE);
    printf("  N (keys)        = %d\n", N_SIZE);
    printf("  D (head_dim)    = %d\n", HEAD_DIM);
    printf("  Threads/block   = %d (%d warps)\n", NUM_THREADS, NUM_WARPS);
    printf("  Warmup iters    = %d\n", WARMUP_ITERS);
    printf("  Benchmark iters = %d\n", BENCH_ITERS);
    printf("--------------------------------------------------------\n");

    // Check device
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    int sm_version = prop.major * 100 + prop.minor * 10;
    if (sm_version < 800) {
        printf("[WARNING] WMMA bf16 requires SM80+. Some tests may fail.\n");
    }
    printf("--------------------------------------------------------\n");

    // Allocate host memory
    size_t dscores_size = M_SIZE * N_SIZE;
    size_t q_size = M_SIZE * HEAD_DIM;
    size_t p_size = M_SIZE * N_SIZE;
    size_t do_size = M_SIZE * HEAD_DIM;
    size_t dk_size = N_SIZE * HEAD_DIM;
    size_t dv_size = N_SIZE * HEAD_DIM;

    float* h_dscores = new float[dscores_size];
    float* h_q = new float[q_size];
    float* h_p = new float[p_size];
    float* h_do = new float[do_size];
    float* h_dk_ref = new float[dk_size];
    float* h_dv_ref = new float[dv_size];

    // Initialize with random data
    fill_random_bf16(h_dscores, dscores_size, 1);
    fill_random_bf16(h_q, q_size, 2);
    fill_random_bf16(h_p, p_size, 3);
    fill_random_bf16(h_do, do_size, 4);

    // Compute reference on CPU
    memset(h_dk_ref, 0, dk_size * sizeof(float));
    memset(h_dv_ref, 0, dv_size * sizeof(float));

    for (int n = 0; n < N_SIZE; ++n) {
        for (int d = 0; d < HEAD_DIM; ++d) {
            float dk_sum = 0.0f;
            float dv_sum = 0.0f;
            for (int m = 0; m < M_SIZE; ++m) {
                dk_sum += h_dscores[m * N_SIZE + n] * h_q[m * HEAD_DIM + d];
                dv_sum += h_p[m * N_SIZE + n] * h_do[m * HEAD_DIM + d];
            }
            h_dk_ref[n * HEAD_DIM + d] = dk_sum;
            h_dv_ref[n * HEAD_DIM + d] = dv_sum;
        }
    }

    // Allocate device memory (float versions)
    float *d_dscores, *d_q, *d_p, *d_do, *d_dk, *d_dv;
    CUDA_CHECK(cudaMalloc(&d_dscores, dscores_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q, q_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p, p_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_do, do_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dk, dk_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dv, dv_size * sizeof(float)));

    // Allocate device memory (bf16 versions for WMMA)
    __nv_bfloat16 *d_dscores_bf16, *d_q_bf16, *d_p_bf16, *d_do_bf16;
    CUDA_CHECK(cudaMalloc(&d_dscores_bf16, dscores_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q_bf16, q_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_p_bf16, p_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_do_bf16, do_size * sizeof(__nv_bfloat16)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_dscores, h_dscores, dscores_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, h_q, q_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p, h_p, p_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_do, h_do, do_size * sizeof(float), cudaMemcpyHostToDevice));

    // Convert to bf16 on host and copy
    __nv_bfloat16* h_dscores_bf16 = new __nv_bfloat16[dscores_size];
    __nv_bfloat16* h_q_bf16 = new __nv_bfloat16[q_size];
    __nv_bfloat16* h_p_bf16 = new __nv_bfloat16[p_size];
    __nv_bfloat16* h_do_bf16 = new __nv_bfloat16[do_size];

    for (size_t i = 0; i < dscores_size; ++i) h_dscores_bf16[i] = __float2bfloat16(h_dscores[i]);
    for (size_t i = 0; i < q_size; ++i) h_q_bf16[i] = __float2bfloat16(h_q[i]);
    for (size_t i = 0; i < p_size; ++i) h_p_bf16[i] = __float2bfloat16(h_p[i]);
    for (size_t i = 0; i < do_size; ++i) h_do_bf16[i] = __float2bfloat16(h_do[i]);

    CUDA_CHECK(cudaMemcpy(d_dscores_bf16, h_dscores_bf16, dscores_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_bf16, h_q_bf16, q_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_bf16, h_p_bf16, p_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_do_bf16, h_do_bf16, do_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // Host buffer for results
    float* h_dk = new float[dk_size];
    float* h_dv = new float[dv_size];

    GpuTimer timer;

    // Calculate FLOPS
    // dK: 2 * M * N * D (multiply-add)
    // dV: 2 * M * N * D (multiply-add)
    // Total: 4 * M * N * D
    double total_flops = 4.0 * M_SIZE * N_SIZE * HEAD_DIM;
    double total_bytes = (dscores_size + q_size + p_size + do_size + 2 * dk_size) * sizeof(float);

    printf("\nBenchmark Results:\n");
    printf("%-30s %12s %12s %12s %12s\n", "Kernel", "Time (us)", "TFLOPS", "GB/s", "Status");
    printf("--------------------------------------------------------\n");

    // Lambda to verify results
    auto verify = [&](const char* name) -> bool {
        CUDA_CHECK(cudaMemcpy(h_dk, d_dk, dk_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_dv, d_dv, dv_size * sizeof(float), cudaMemcpyDeviceToHost));

        float max_dk_err = 0.0f, max_dv_err = 0.0f;
        for (size_t i = 0; i < dk_size; ++i) {
            float err_dk = fabsf(h_dk[i] - h_dk_ref[i]);
            float err_dv = fabsf(h_dv[i] - h_dv_ref[i]);
            max_dk_err = fmaxf(max_dk_err, err_dk);
            max_dv_err = fmaxf(max_dv_err, err_dv);
        }

        // Allow some tolerance for atomicAdd non-determinism
        float tolerance = 1e-3f;
        bool pass = (max_dk_err < tolerance) && (max_dv_err < tolerance);
        if (!pass) {
            printf("  [%s] max_dk_err=%.6f, max_dv_err=%.6f\n", name, max_dk_err, max_dv_err);
        }
        return pass;
    };

    // Lambda to benchmark a kernel
    auto benchmark = [&](const char* name, auto kernel_fn) {
        // Clear output
        CUDA_CHECK(cudaMemset(d_dk, 0, dk_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_dv, 0, dv_size * sizeof(float)));

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            CUDA_CHECK(cudaMemset(d_dk, 0, dk_size * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_dv, 0, dv_size * sizeof(float)));
            kernel_fn();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Verify correctness
        CUDA_CHECK(cudaMemset(d_dk, 0, dk_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_dv, 0, dv_size * sizeof(float)));
        kernel_fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        bool correct = verify(name);

        // Benchmark
        timer.start();
        for (int i = 0; i < BENCH_ITERS; ++i) {
            CUDA_CHECK(cudaMemset(d_dk, 0, dk_size * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_dv, 0, dv_size * sizeof(float)));
            kernel_fn();
        }
        float total_ms = timer.stop_ms();
        float avg_us = (total_ms * 1000.0f) / BENCH_ITERS;

        double tflops = (total_flops / (avg_us * 1e-6)) / 1e12;
        double gbps = (total_bytes / (avg_us * 1e-6)) / 1e9;

        printf("%-30s %12.2f %12.4f %12.2f %12s\n",
               name, avg_us, tflops, gbps, correct ? "[PASS]" : "[FAIL]");
    };

    // Benchmark 1: Current Baseline
    benchmark("1. Current (baseline)", [&]() {
        kernel_current_baseline<<<1, NUM_THREADS>>>(
            d_dscores, d_q, d_p, d_do, d_dk, d_dv,
            M_SIZE, N_SIZE, HEAD_DIM
        );
    });

    // Benchmark 2: Warp-Cooperative
    benchmark("2. Warp-cooperative", [&]() {
        kernel_warp_cooperative<<<1, NUM_THREADS>>>(
            d_dscores, d_q, d_p, d_do, d_dk, d_dv,
            M_SIZE, N_SIZE, HEAD_DIM
        );
    });

    // Benchmark 3: WMMA-based (only if SM >= 80)
    if (sm_version >= 800) {
        // WMMA version needs different verification due to bf16 precision
        auto verify_wmma = [&]() -> bool {
            CUDA_CHECK(cudaMemcpy(h_dk, d_dk, dk_size * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_dv, d_dv, dv_size * sizeof(float), cudaMemcpyDeviceToHost));

            float max_dk_err = 0.0f, max_dv_err = 0.0f;
            for (size_t i = 0; i < dk_size; ++i) {
                float err_dk = fabsf(h_dk[i] - h_dk_ref[i]) / (fabsf(h_dk_ref[i]) + 1e-6f);
                float err_dv = fabsf(h_dv[i] - h_dv_ref[i]) / (fabsf(h_dv_ref[i]) + 1e-6f);
                max_dk_err = fmaxf(max_dk_err, err_dk);
                max_dv_err = fmaxf(max_dv_err, err_dv);
            }

            // BF16 has lower precision, allow 1% relative error
            float tolerance = 0.05f;
            return (max_dk_err < tolerance) && (max_dv_err < tolerance);
        };

        // Clear and run WMMA kernel
        CUDA_CHECK(cudaMemset(d_dk, 0, dk_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_dv, 0, dv_size * sizeof(float)));

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            CUDA_CHECK(cudaMemset(d_dk, 0, dk_size * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_dv, 0, dv_size * sizeof(float)));
            kernel_wmma_based<<<1, NUM_THREADS>>>(
                d_dscores_bf16, d_q_bf16, d_p_bf16, d_do_bf16, d_dk, d_dv,
                M_SIZE, N_SIZE, HEAD_DIM
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Verify
        CUDA_CHECK(cudaMemset(d_dk, 0, dk_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_dv, 0, dv_size * sizeof(float)));
        kernel_wmma_based<<<1, NUM_THREADS>>>(
            d_dscores_bf16, d_q_bf16, d_p_bf16, d_do_bf16, d_dk, d_dv,
            M_SIZE, N_SIZE, HEAD_DIM
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        bool correct = verify_wmma();

        // Benchmark
        timer.start();
        for (int i = 0; i < BENCH_ITERS; ++i) {
            CUDA_CHECK(cudaMemset(d_dk, 0, dk_size * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_dv, 0, dv_size * sizeof(float)));
            kernel_wmma_based<<<1, NUM_THREADS>>>(
                d_dscores_bf16, d_q_bf16, d_p_bf16, d_do_bf16, d_dk, d_dv,
                M_SIZE, N_SIZE, HEAD_DIM
            );
        }
        float total_ms = timer.stop_ms();
        float avg_us = (total_ms * 1000.0f) / BENCH_ITERS;

        double tflops = (total_flops / (avg_us * 1e-6)) / 1e12;
        double gbps = (total_bytes / (avg_us * 1e-6)) / 1e9;

        printf("%-30s %12.2f %12.4f %12.2f %12s\n",
               "3. WMMA-based (tensor core)", avg_us, tflops, gbps, correct ? "[PASS]" : "[FAIL]");
    } else {
        printf("%-30s %12s %12s %12s %12s\n",
               "3. WMMA-based (tensor core)", "N/A", "N/A", "N/A", "[SKIP]");
    }

    // Benchmark 4: Thread-Block Cooperative
    benchmark("4. Block-cooperative", [&]() {
        kernel_block_cooperative<<<1, NUM_THREADS>>>(
            d_dscores, d_q, d_p, d_do, d_dk, d_dv,
            M_SIZE, N_SIZE, HEAD_DIM
        );
    });

    // Benchmark 5: Register-Blocked
    benchmark("5. Register-blocked", [&]() {
        kernel_register_blocked<<<1, NUM_THREADS>>>(
            d_dscores, d_q, d_p, d_do, d_dk, d_dv,
            M_SIZE, N_SIZE, HEAD_DIM
        );
    });

    printf("--------------------------------------------------------\n");

    // Summary
    printf("\nAnalysis:\n");
    printf("- dK/dV accumulation is O(M*N*D) FLOPs\n");
    printf("- For M=%d, N=%d, D=%d: %.2f KFLOPS per output element\n",
           M_SIZE, N_SIZE, HEAD_DIM, (double)M_SIZE / 1000.0);
    printf("- Total FLOPs: %.2f MFLOPS\n", total_flops / 1e6);
    printf("- Memory footprint: %.2f KB\n", total_bytes / 1024.0);
    printf("\nRecommendation:\n");
    printf("- WMMA-based approach leverages tensor cores for matrix multiply\n");
    printf("- Warp-cooperative reduces atomicAdd contention\n");
    printf("- Register-blocking improves ILP and reduces memory traffic\n");

    // Cleanup
    delete[] h_dscores;
    delete[] h_q;
    delete[] h_p;
    delete[] h_do;
    delete[] h_dk_ref;
    delete[] h_dv_ref;
    delete[] h_dk;
    delete[] h_dv;
    delete[] h_dscores_bf16;
    delete[] h_q_bf16;
    delete[] h_p_bf16;
    delete[] h_do_bf16;

    CUDA_CHECK(cudaFree(d_dscores));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_do));
    CUDA_CHECK(cudaFree(d_dk));
    CUDA_CHECK(cudaFree(d_dv));
    CUDA_CHECK(cudaFree(d_dscores_bf16));
    CUDA_CHECK(cudaFree(d_q_bf16));
    CUDA_CHECK(cudaFree(d_p_bf16));
    CUDA_CHECK(cudaFree(d_do_bf16));

    printf("\n[DONE] Microbenchmark complete.\n");
    return 0;
}
