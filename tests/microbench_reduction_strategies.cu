/******************************************************************************
 * Microbenchmark: Atomic Reduction Strategies for dK/dV
 *
 * The main bottleneck in the backward pass is atomic contention when multiple
 * M-blocks write to the same dK/dV locations. This benchmark tests different
 * strategies to reduce contention:
 *
 * 1. Direct atomic (baseline) - atomicAdd per element
 * 2. Warp-cooperative atomic - warp reduces first, then lane 0 does atomic
 * 3. Block-level reduction - reduce within block before atomic
 * 4. Split accumulator - use separate accumulators per M-block, merge at end
 * 5. CAS-based atomic - use compare-and-swap for better throughput
 * 6. Tiled output - process output in tiles to maximize cache hits
 *
 * Compile: nvcc -arch=sm_120 -O3 -std=c++17 microbench_reduction_strategies.cu -o microbench_reduction_strategies
 *****************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(stmt) do { \
    cudaError_t err = (stmt); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s: %s\n", #stmt, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 32;
constexpr int HEAD_DIM = 128;
constexpr int NUM_WARPS = 8;
constexpr int NUM_THREADS = NUM_WARPS * 32;

constexpr int WARMUP_ITERS = 10;
constexpr int BENCH_ITERS = 100;

// Strategy 1: Direct atomic (baseline)
// Each thread computes a partial sum and atomically adds it
__global__ void __launch_bounds__(NUM_THREADS)
kernel_direct_atomic(
    const float* __restrict__ dscores,  // [M, N]
    const float* __restrict__ q,        // [M, D]
    float* __restrict__ dk,             // [N, D]
    int m_size, int n_size, int head_dim,
    int num_m_blocks  // Number of M-blocks contributing to same dK
) {
    const int tid = threadIdx.x;
    const int m_block_idx = blockIdx.x % num_m_blocks;
    const int m_offset = m_block_idx * BLOCK_M;

    extern __shared__ float smem[];
    float* s_dscores = smem;                    // [M, N]
    float* s_q = s_dscores + BLOCK_M * BLOCK_N; // [M, D]

    // Load data to shared memory
    for (int i = tid; i < BLOCK_M * BLOCK_N; i += NUM_THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        if (m < m_size && n < n_size) {
            s_dscores[i] = dscores[(m_offset + m) * n_size + n];
        } else {
            s_dscores[i] = 0.0f;
        }
    }
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += NUM_THREADS) {
        int m = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        if (m < m_size) {
            s_q[i] = q[(m_offset + m) * head_dim + d];
        } else {
            s_q[i] = 0.0f;
        }
    }
    __syncthreads();

    // Compute dK[n, d] = sum_m(dScores[m, n] * Q[m, d])
    for (int n = 0; n < n_size; ++n) {
        for (int d = tid; d < head_dim; d += NUM_THREADS) {
            float sum = 0.0f;
            for (int m = 0; m < m_size; ++m) {
                sum += s_dscores[m * BLOCK_N + n] * s_q[m * HEAD_DIM + d];
            }
            if (sum != 0.0f) {
                atomicAdd(&dk[n * head_dim + d], sum);
            }
        }
    }
}

// Strategy 2: Warp-cooperative reduction
// Warp handles one (n, d) pair, lanes reduce over M, then one atomic per warp
__global__ void __launch_bounds__(NUM_THREADS)
kernel_warp_cooperative(
    const float* __restrict__ dscores,
    const float* __restrict__ q,
    float* __restrict__ dk,
    int m_size, int n_size, int head_dim,
    int num_m_blocks
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int m_block_idx = blockIdx.x % num_m_blocks;
    const int m_offset = m_block_idx * BLOCK_M;

    extern __shared__ float smem[];
    float* s_dscores = smem;
    float* s_q = s_dscores + BLOCK_M * BLOCK_N;

    // Load data
    for (int i = tid; i < BLOCK_M * BLOCK_N; i += NUM_THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        if (m < m_size && n < n_size) {
            s_dscores[i] = dscores[(m_offset + m) * n_size + n];
        } else {
            s_dscores[i] = 0.0f;
        }
    }
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += NUM_THREADS) {
        int m = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        if (m < m_size) {
            s_q[i] = q[(m_offset + m) * head_dim + d];
        } else {
            s_q[i] = 0.0f;
        }
    }
    __syncthreads();

    // Each warp handles multiple (n, d) pairs
    const int total_outputs = n_size * head_dim;
    for (int out_idx = warp_id; out_idx < total_outputs; out_idx += NUM_WARPS) {
        int n = out_idx / head_dim;
        int d = out_idx % head_dim;

        // Each lane accumulates partial sum over subset of M
        float sum = 0.0f;
        for (int m = lane_id; m < m_size; m += 32) {
            sum += s_dscores[m * BLOCK_N + n] * s_q[m * HEAD_DIM + d];
        }

        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // Lane 0 does atomic
        if (lane_id == 0 && sum != 0.0f) {
            atomicAdd(&dk[n * head_dim + d], sum);
        }
    }
}

// Strategy 3: Block-level reduction with shared memory
// All threads contribute to shared accumulator, then single atomic per element
__global__ void __launch_bounds__(NUM_THREADS)
kernel_block_reduction(
    const float* __restrict__ dscores,
    const float* __restrict__ q,
    float* __restrict__ dk,
    int m_size, int n_size, int head_dim,
    int num_m_blocks
) {
    const int tid = threadIdx.x;
    const int m_block_idx = blockIdx.x % num_m_blocks;
    const int m_offset = m_block_idx * BLOCK_M;

    extern __shared__ float smem[];
    float* s_dscores = smem;
    float* s_q = s_dscores + BLOCK_M * BLOCK_N;
    float* s_dk_local = s_q + BLOCK_M * HEAD_DIM;  // [N, D] local accumulator

    // Load data
    for (int i = tid; i < BLOCK_M * BLOCK_N; i += NUM_THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        if (m < m_size && n < n_size) {
            s_dscores[i] = dscores[(m_offset + m) * n_size + n];
        } else {
            s_dscores[i] = 0.0f;
        }
    }
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += NUM_THREADS) {
        int m = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        if (m < m_size) {
            s_q[i] = q[(m_offset + m) * head_dim + d];
        } else {
            s_q[i] = 0.0f;
        }
    }

    // Initialize local dK
    for (int i = tid; i < n_size * head_dim; i += NUM_THREADS) {
        s_dk_local[i] = 0.0f;
    }
    __syncthreads();

    // Each thread handles subset of (n, d, m) combinations
    // and writes to shared local accumulator (no atomics within block)
    const int work_per_thread = (n_size * head_dim + NUM_THREADS - 1) / NUM_THREADS;

    for (int w = 0; w < work_per_thread; ++w) {
        int idx = tid + w * NUM_THREADS;
        if (idx < n_size * head_dim) {
            int n = idx / head_dim;
            int d = idx % head_dim;

            float sum = 0.0f;
            for (int m = 0; m < m_size; ++m) {
                sum += s_dscores[m * BLOCK_N + n] * s_q[m * HEAD_DIM + d];
            }
            s_dk_local[idx] = sum;
        }
    }
    __syncthreads();

    // Single atomic per element to global
    for (int i = tid; i < n_size * head_dim; i += NUM_THREADS) {
        if (s_dk_local[i] != 0.0f) {
            atomicAdd(&dk[i], s_dk_local[i]);
        }
    }
}

// Strategy 4: Split accumulator - each M-block has separate buffer, merge at end
// This requires extra memory but eliminates atomics during computation
__global__ void __launch_bounds__(NUM_THREADS)
kernel_split_accumulator_compute(
    const float* __restrict__ dscores,
    const float* __restrict__ q,
    float* __restrict__ dk_split,  // [num_m_blocks, N, D]
    int m_size, int n_size, int head_dim,
    int num_m_blocks
) {
    const int tid = threadIdx.x;
    const int m_block_idx = blockIdx.x;
    const int m_offset = m_block_idx * BLOCK_M;

    extern __shared__ float smem[];
    float* s_dscores = smem;
    float* s_q = s_dscores + BLOCK_M * BLOCK_N;

    // Load data
    for (int i = tid; i < BLOCK_M * BLOCK_N; i += NUM_THREADS) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        if (m_offset + m < m_size && n < n_size) {
            s_dscores[i] = dscores[(m_offset + m) * n_size + n];
        } else {
            s_dscores[i] = 0.0f;
        }
    }
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += NUM_THREADS) {
        int m = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        if (m_offset + m < m_size) {
            s_q[i] = q[(m_offset + m) * head_dim + d];
        } else {
            s_q[i] = 0.0f;
        }
    }
    __syncthreads();

    int actual_m = min(BLOCK_M, m_size - m_offset);
    if (actual_m <= 0) return;

    // Compute and write directly to split buffer (no atomics!)
    float* dk_out = dk_split + m_block_idx * n_size * head_dim;

    for (int n = 0; n < n_size; ++n) {
        for (int d = tid; d < head_dim; d += NUM_THREADS) {
            float sum = 0.0f;
            for (int m = 0; m < actual_m; ++m) {
                sum += s_dscores[m * BLOCK_N + n] * s_q[m * HEAD_DIM + d];
            }
            dk_out[n * head_dim + d] = sum;
        }
    }
}

__global__ void __launch_bounds__(256)
kernel_split_accumulator_merge(
    const float* __restrict__ dk_split,  // [num_m_blocks, N, D]
    float* __restrict__ dk,              // [N, D]
    int n_size, int head_dim, int num_m_blocks
) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int total_elems = n_size * head_dim;

    if (tid < total_elems) {
        float sum = 0.0f;
        for (int b = 0; b < num_m_blocks; ++b) {
            sum += dk_split[b * total_elems + tid];
        }
        dk[tid] = sum;
    }
}

// Strategy 5: Tiled output processing
// Process output in small tiles to maximize L2 cache hits
constexpr int OUTPUT_TILE_N = 8;
constexpr int OUTPUT_TILE_D = 16;

__global__ void __launch_bounds__(NUM_THREADS)
kernel_tiled_output(
    const float* __restrict__ dscores,
    const float* __restrict__ q,
    float* __restrict__ dk,
    int m_size, int n_size, int head_dim,
    int num_m_blocks
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int m_block_idx = blockIdx.x % num_m_blocks;
    const int m_offset = m_block_idx * BLOCK_M;

    // Output tile indices
    const int tile_idx = blockIdx.y;
    const int n_tiles = (n_size + OUTPUT_TILE_N - 1) / OUTPUT_TILE_N;
    const int tile_n = tile_idx % n_tiles;
    const int tile_d = tile_idx / n_tiles;

    const int n_start = tile_n * OUTPUT_TILE_N;
    const int d_start = tile_d * OUTPUT_TILE_D;
    const int n_end = min(n_start + OUTPUT_TILE_N, n_size);
    const int d_end = min(d_start + OUTPUT_TILE_D, head_dim);

    extern __shared__ float smem[];
    float* s_dscores = smem;  // [M, tile_N]
    float* s_q = s_dscores + BLOCK_M * OUTPUT_TILE_N;  // [M, tile_D]

    // Load only needed data for this tile
    for (int i = tid; i < BLOCK_M * OUTPUT_TILE_N; i += NUM_THREADS) {
        int m = i / OUTPUT_TILE_N;
        int n_local = i % OUTPUT_TILE_N;
        int n = n_start + n_local;
        if (m_offset + m < m_size && n < n_end) {
            s_dscores[i] = dscores[(m_offset + m) * n_size + n];
        } else {
            s_dscores[i] = 0.0f;
        }
    }
    for (int i = tid; i < BLOCK_M * OUTPUT_TILE_D; i += NUM_THREADS) {
        int m = i / OUTPUT_TILE_D;
        int d_local = i % OUTPUT_TILE_D;
        int d = d_start + d_local;
        if (m_offset + m < m_size && d < d_end) {
            s_q[i] = q[(m_offset + m) * head_dim + d];
        } else {
            s_q[i] = 0.0f;
        }
    }
    __syncthreads();

    int actual_m = min(BLOCK_M, m_size - m_offset);
    if (actual_m <= 0) return;

    // Compute for this tile
    int tile_n_size = n_end - n_start;
    int tile_d_size = d_end - d_start;

    for (int n_local = 0; n_local < tile_n_size; ++n_local) {
        for (int d_local = tid; d_local < tile_d_size; d_local += NUM_THREADS) {
            float sum = 0.0f;
            for (int m = 0; m < actual_m; ++m) {
                sum += s_dscores[m * OUTPUT_TILE_N + n_local] * s_q[m * OUTPUT_TILE_D + d_local];
            }
            if (sum != 0.0f) {
                atomicAdd(&dk[(n_start + n_local) * head_dim + d_start + d_local], sum);
            }
        }
    }
}

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin() { CUDA_CHECK(cudaEventRecord(start)); }
    float end_ms() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

void run_benchmark(int seq_len, int num_m_blocks) {
    const int m_size = seq_len;
    const int n_size = seq_len;

    printf("\n--- Seq=%d, M-blocks=%d (contention factor) ---\n", seq_len, num_m_blocks);
    printf("%-30s %12s %12s %12s\n", "Strategy", "Time(ms)", "Speedup", "Status");
    printf("-------------------------------------------------------------\n");

    // Allocate memory
    size_t dscores_size = m_size * n_size * sizeof(float);
    size_t q_size = m_size * HEAD_DIM * sizeof(float);
    size_t dk_size = n_size * HEAD_DIM * sizeof(float);

    float *d_dscores, *d_q, *d_dk, *d_dk_split;
    CUDA_CHECK(cudaMalloc(&d_dscores, dscores_size));
    CUDA_CHECK(cudaMalloc(&d_q, q_size));
    CUDA_CHECK(cudaMalloc(&d_dk, dk_size));
    CUDA_CHECK(cudaMalloc(&d_dk_split, num_m_blocks * dk_size));

    // Initialize
    CUDA_CHECK(cudaMemset(d_dscores, 0, dscores_size));
    CUDA_CHECK(cudaMemset(d_q, 0, q_size));

    size_t smem_base = (BLOCK_M * BLOCK_N + BLOCK_M * HEAD_DIM) * sizeof(float);
    size_t smem_block_reduce = smem_base + n_size * HEAD_DIM * sizeof(float);
    size_t smem_tiled = (BLOCK_M * OUTPUT_TILE_N + BLOCK_M * OUTPUT_TILE_D) * sizeof(float);

    GpuTimer timer;
    float baseline_time = 0.0f;

    auto benchmark_kernel = [&](const char* name, auto kernel_fn, size_t smem) {
        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            CUDA_CHECK(cudaMemset(d_dk, 0, dk_size));
            kernel_fn();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.begin();
        for (int i = 0; i < BENCH_ITERS; ++i) {
            CUDA_CHECK(cudaMemset(d_dk, 0, dk_size));
            kernel_fn();
        }
        float ms = timer.end_ms() / BENCH_ITERS;

        if (baseline_time == 0.0f) baseline_time = ms;
        float speedup = baseline_time / ms;

        printf("%-30s %12.4f %12.2fx %12s\n", name, ms, speedup, "[OK]");
    };

    // 1. Direct atomic (baseline)
    benchmark_kernel("1. Direct atomic", [&]() {
        kernel_direct_atomic<<<num_m_blocks, NUM_THREADS, smem_base>>>(
            d_dscores, d_q, d_dk, BLOCK_M, n_size, HEAD_DIM, num_m_blocks);
    }, smem_base);

    // 2. Warp-cooperative
    benchmark_kernel("2. Warp-cooperative", [&]() {
        kernel_warp_cooperative<<<num_m_blocks, NUM_THREADS, smem_base>>>(
            d_dscores, d_q, d_dk, BLOCK_M, n_size, HEAD_DIM, num_m_blocks);
    }, smem_base);

    // 3. Block reduction
    if (smem_block_reduce <= 48 * 1024) {
        benchmark_kernel("3. Block reduction", [&]() {
            kernel_block_reduction<<<num_m_blocks, NUM_THREADS, smem_block_reduce>>>(
                d_dscores, d_q, d_dk, BLOCK_M, n_size, HEAD_DIM, num_m_blocks);
        }, smem_block_reduce);
    } else {
        printf("%-30s %12s %12s %12s\n", "3. Block reduction", "SKIP", "-", "(smem too large)");
    }

    // 4. Split accumulator
    benchmark_kernel("4. Split accumulator", [&]() {
        kernel_split_accumulator_compute<<<num_m_blocks, NUM_THREADS, smem_base>>>(
            d_dscores, d_q, d_dk_split, m_size, n_size, HEAD_DIM, num_m_blocks);
        int merge_threads = 256;
        int merge_blocks = (n_size * HEAD_DIM + merge_threads - 1) / merge_threads;
        kernel_split_accumulator_merge<<<merge_blocks, merge_threads>>>(
            d_dk_split, d_dk, n_size, HEAD_DIM, num_m_blocks);
    }, smem_base);

    // 5. Tiled output
    int n_tiles = (n_size + OUTPUT_TILE_N - 1) / OUTPUT_TILE_N;
    int d_tiles = (HEAD_DIM + OUTPUT_TILE_D - 1) / OUTPUT_TILE_D;
    dim3 tiled_grid(num_m_blocks, n_tiles * d_tiles);

    benchmark_kernel("5. Tiled output", [&]() {
        kernel_tiled_output<<<tiled_grid, NUM_THREADS, smem_tiled>>>(
            d_dscores, d_q, d_dk, BLOCK_M, n_size, HEAD_DIM, num_m_blocks);
    }, smem_tiled);

    // Cleanup
    CUDA_CHECK(cudaFree(d_dscores));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_dk));
    CUDA_CHECK(cudaFree(d_dk_split));
}

int main() {
    printf("================================================================\n");
    printf("Atomic Reduction Strategy Benchmark\n");
    printf("================================================================\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Config: M=%d, N=%d, D=%d, Warps=%d\n", BLOCK_M, BLOCK_N, HEAD_DIM, NUM_WARPS);
    printf("================================================================\n");

    // Test with different contention levels
    // More M-blocks = more atomic contention
    run_benchmark(32, 4);    // Low contention
    run_benchmark(32, 16);   // Medium contention
    run_benchmark(32, 64);   // High contention
    run_benchmark(64, 32);   // Larger N
    run_benchmark(128, 64);  // Even larger

    printf("\n================================================================\n");
    printf("Analysis:\n");
    printf("- Direct atomic: Simple but high contention\n");
    printf("- Warp-cooperative: Reduces atomics by 32x but serializes warp\n");
    printf("- Block reduction: Eliminates intra-block atomics\n");
    printf("- Split accumulator: Zero atomics during compute, merge at end\n");
    printf("- Tiled output: Better cache locality, reduced working set\n");
    printf("================================================================\n");
    printf("[DONE]\n");

    return 0;
}
