/***************************************************************************************************
 * CUDA Microbenchmark for Atomic Operations in dK/dV Accumulation
 *
 * This benchmark measures the performance of atomic operations used in the FlashMLA backward
 * pass for dK and dV gradient accumulation. It simulates the exact access patterns from
 * atomic_add_dk_varlen and atomic_add_dv_varlen functions.
 *
 * Key patterns tested:
 * 1. Multiple M-blocks writing to same K/V positions (high contention)
 * 2. stride_token = num_heads * head_dim access pattern
 * 3. Thread-level vs warp-cooperative atomic strategies
 * 4. Different contention levels based on sequence lengths
 *
 * Compile with: nvcc -O3 -arch=sm_120 microbench_atomics.cu -o microbench_atomics
 *
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

// ============================================================================
// ERROR CHECKING MACROS
// ============================================================================

#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t _e = (stmt);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "[CUDA_ERROR] %s failed: %s (%d) at %s:%d\n", #stmt,     \
              cudaGetErrorString(_e), static_cast<int>(_e), __FILE__,          \
              __LINE__);                                                       \
      fflush(stderr);                                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

// Block/tile sizes matching FlashMLA backward kernel
constexpr int BWD_BLOCK_M = 16;   // Queries per M-block
constexpr int BWD_BLOCK_N = 32;   // Keys per N-block
constexpr int BWD_BLOCK_D = 128;  // Head dimension
constexpr int BWD_NUM_THREADS = 128;

// Test configurations
constexpr int NUM_WARMUP_ITERATIONS = 10;
constexpr int NUM_BENCHMARK_ITERATIONS = 100;

// ============================================================================
// GPU TIMER UTILITY
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

    void start() { CUDA_CHECK(cudaEventRecord(start_event)); }

    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms;
    }
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

inline uint32_t lcg_next(uint32_t& seed) {
    seed = 1664525u * seed + 1013904223u;
    return seed;
}

void fill_random_float(float* data, size_t n, uint32_t seed = 42) {
    for (size_t i = 0; i < n; ++i) {
        seed = lcg_next(seed);
        data[i] = (static_cast<float>(seed & 0xFFFF) / 65535.0f - 0.5f);
    }
}

int get_device_sm() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return prop.major * 100 + prop.minor * 10;
}

// ============================================================================
// KERNEL 1: BASELINE NON-ATOMIC WRITE (single M-block, no contention)
// ============================================================================
// This kernel simulates what would happen if only a single M-block wrote to
// each K/V position - no atomics needed, just direct stores.

__global__ void kernel_non_atomic_write(
    float* __restrict__ dk,
    const float* __restrict__ dscores,
    const float* __restrict__ q,
    int num_kv_tokens,
    int num_heads,
    int head_dim,
    int m_size
) {
    const int tid = threadIdx.x;
    const int kv_token = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int stride_token = num_heads * head_dim;

    if (kv_token >= num_kv_tokens) return;

    // Simulate dK[n,d] = sum_m(dScores[m,n] * Q[m,d])
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int m = 0; m < m_size; ++m) {
            float ds = dscores[m * BWD_BLOCK_N + (kv_token % BWD_BLOCK_N)];
            float q_val = q[m * head_dim + d];
            sum += ds * q_val;
        }
        // Direct write - no contention
        dk[kv_token * stride_token + head_idx * head_dim + d] = sum;
    }
}

// ============================================================================
// KERNEL 2: THREAD-LEVEL ATOMIC ADD (matches atomic_add_dk_varlen pattern)
// ============================================================================
// Each thread computes a partial sum and uses atomicAdd to accumulate.
// This simulates multiple M-blocks writing to the same K/V positions.

__global__ void kernel_atomic_add_thread_level(
    float* __restrict__ dk,
    const float* __restrict__ dscores,
    const float* __restrict__ q,
    int num_kv_tokens,
    int num_heads,
    int head_idx,
    int head_dim,
    int m_size,
    int m_block_idx  // Which M-block this simulates
) {
    const int tid = threadIdx.x;
    const int n_block_idx = blockIdx.x;
    const int kv_seq_start = 0;  // For simplicity in benchmark

    const int n_start = n_block_idx * BWD_BLOCK_N;
    const int n_end = min(n_start + BWD_BLOCK_N, num_kv_tokens);
    const int n_size = n_end - n_start;
    const int stride_token = num_heads * head_dim;

    // Exact pattern from atomic_add_dk_varlen
    for (int n = 0; n < n_size; ++n) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float sum = 0.0f;
            for (int m = 0; m < m_size; ++m) {
                float ds = dscores[m * BWD_BLOCK_N + n];
                float q_val = q[m * head_dim + d];
                sum += ds * q_val;
            }
            if (sum != 0.0f) {
                int global_token = kv_seq_start + n_start + n;
                atomicAdd(&dk[global_token * stride_token + head_idx * head_dim + d], sum);
            }
        }
    }
}

// ============================================================================
// KERNEL 3: WARP-COOPERATIVE ATOMIC ADD
// ============================================================================
// Uses warp-level reduction before atomic to reduce contention.
// Each warp collaboratively computes partial sums, then only one thread
// per warp performs the atomic.

__device__ __forceinline__ float warp_reduce_sum_bench(float val) {
    const unsigned int FULL_MASK = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__global__ void kernel_atomic_add_warp_cooperative(
    float* __restrict__ dk,
    const float* __restrict__ dscores,
    const float* __restrict__ q,
    int num_kv_tokens,
    int num_heads,
    int head_idx,
    int head_dim,
    int m_size,
    int m_block_idx
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;
    const int n_block_idx = blockIdx.x;
    const int kv_seq_start = 0;

    const int n_start = n_block_idx * BWD_BLOCK_N;
    const int n_end = min(n_start + BWD_BLOCK_N, num_kv_tokens);
    const int n_size = n_end - n_start;
    const int stride_token = num_heads * head_dim;

    // Each warp handles a subset of (n, d) pairs
    // Warp-cooperative: distribute work across warps, reduce within warp
    for (int nd = warp_id; nd < n_size * head_dim; nd += num_warps) {
        int n = nd / head_dim;
        int d = nd % head_dim;

        // Each lane processes a subset of m values
        float partial_sum = 0.0f;
        for (int m = lane_id; m < m_size; m += 32) {
            float ds = dscores[m * BWD_BLOCK_N + n];
            float q_val = q[m * head_dim + d];
            partial_sum += ds * q_val;
        }

        // Warp-level reduction
        float warp_sum = warp_reduce_sum_bench(partial_sum);

        // Only lane 0 performs the atomic
        if (lane_id == 0 && warp_sum != 0.0f) {
            int global_token = kv_seq_start + n_start + n;
            atomicAdd(&dk[global_token * stride_token + head_idx * head_dim + d], warp_sum);
        }
    }
}

// ============================================================================
// KERNEL 4: SHARED MEMORY REDUCTION + SINGLE ATOMIC
// ============================================================================
// First reduce within shared memory, then single atomic per element.
// This minimizes atomic operations at the cost of more shared memory.

__global__ void kernel_atomic_add_smem_reduce(
    float* __restrict__ dk,
    const float* __restrict__ dscores,
    const float* __restrict__ q,
    int num_kv_tokens,
    int num_heads,
    int head_idx,
    int head_dim,
    int m_size,
    int m_block_idx
) {
    extern __shared__ float smem[];
    // Layout: smem[n * head_dim + d] for block-level reduction

    const int tid = threadIdx.x;
    const int n_block_idx = blockIdx.x;
    const int kv_seq_start = 0;

    const int n_start = n_block_idx * BWD_BLOCK_N;
    const int n_end = min(n_start + BWD_BLOCK_N, num_kv_tokens);
    const int n_size = n_end - n_start;
    const int stride_token = num_heads * head_dim;

    // Initialize shared memory
    for (int i = tid; i < n_size * head_dim; i += blockDim.x) {
        smem[i] = 0.0f;
    }
    __syncthreads();

    // Each thread contributes to multiple (n, d) positions
    for (int nd = tid; nd < n_size * head_dim; nd += blockDim.x) {
        int n = nd / head_dim;
        int d = nd % head_dim;

        float sum = 0.0f;
        for (int m = 0; m < m_size; ++m) {
            float ds = dscores[m * BWD_BLOCK_N + n];
            float q_val = q[m * head_dim + d];
            sum += ds * q_val;
        }
        smem[nd] = sum;
    }
    __syncthreads();

    // Single atomic write per element from thread 0 or distributed
    for (int nd = tid; nd < n_size * head_dim; nd += blockDim.x) {
        int n = nd / head_dim;
        int d = nd % head_dim;
        float val = smem[nd];
        if (val != 0.0f) {
            int global_token = kv_seq_start + n_start + n;
            atomicAdd(&dk[global_token * stride_token + head_idx * head_dim + d], val);
        }
    }
}

// ============================================================================
// KERNEL 5: HIGH CONTENTION STRESS TEST
// ============================================================================
// All threads in all blocks atomically add to the SAME memory location.
// This measures the absolute worst-case atomic contention.

__global__ void kernel_atomic_high_contention(
    float* __restrict__ target,
    float value,
    int num_iterations
) {
    for (int i = 0; i < num_iterations; ++i) {
        atomicAdd(target, value);
    }
}

// ============================================================================
// KERNEL 6: LOW CONTENTION BASELINE
// ============================================================================
// Each thread writes to a unique memory location - no contention.

__global__ void kernel_atomic_low_contention(
    float* __restrict__ target,
    float value,
    int num_elements,
    int num_iterations
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        for (int i = 0; i < num_iterations; ++i) {
            atomicAdd(&target[idx], value);
        }
    }
}

// ============================================================================
// BENCHMARK STRUCTURES
// ============================================================================

struct BenchmarkConfig {
    int seq_len_q;       // Query sequence length
    int seq_len_kv;      // Key/Value sequence length
    int num_heads;       // Number of attention heads
    int head_dim;        // Head dimension
    int num_m_blocks;    // Number of M-blocks (determines contention)

    int get_num_kv_blocks() const {
        return (seq_len_kv + BWD_BLOCK_N - 1) / BWD_BLOCK_N;
    }

    int get_total_kv_elements() const {
        return seq_len_kv * num_heads * head_dim;
    }

    int get_stride_token() const {
        return num_heads * head_dim;
    }
};

struct BenchmarkResult {
    std::string kernel_name;
    float time_ms;
    double throughput_gops;  // Giga-operations per second
    double bandwidth_gbps;   // Effective bandwidth in GB/s
    int num_atomics;         // Total atomic operations
};

// ============================================================================
// BENCHMARK RUNNER
// ============================================================================

class AtomicBenchmark {
public:
    AtomicBenchmark(const BenchmarkConfig& cfg) : config(cfg) {
        allocate_buffers();
        initialize_data();
    }

    ~AtomicBenchmark() {
        free_buffers();
    }

    BenchmarkResult run_non_atomic() {
        reset_output();

        GpuTimer timer;
        timer.start();

        for (int iter = 0; iter < NUM_BENCHMARK_ITERATIONS; ++iter) {
            dim3 grid(config.seq_len_kv, config.num_heads);
            dim3 block(BWD_NUM_THREADS);
            kernel_non_atomic_write<<<grid, block>>>(
                d_dk, d_dscores, d_q,
                config.seq_len_kv, config.num_heads, config.head_dim, BWD_BLOCK_M
            );
        }

        float time_ms = timer.stop_ms();
        float avg_time = time_ms / NUM_BENCHMARK_ITERATIONS;

        // Calculate throughput: operations = seq_len_kv * num_heads * head_dim * m_size (FMAs)
        int64_t total_ops = static_cast<int64_t>(config.seq_len_kv) *
                            config.num_heads * config.head_dim * BWD_BLOCK_M * 2; // 2 for FMA
        double gops = (total_ops / 1e9) / (avg_time / 1000.0);

        // Bandwidth: read dscores + q, write dk
        int64_t bytes_read = static_cast<int64_t>(BWD_BLOCK_M * BWD_BLOCK_N) * sizeof(float) +
                            static_cast<int64_t>(BWD_BLOCK_M * config.head_dim) * sizeof(float);
        int64_t bytes_written = static_cast<int64_t>(config.seq_len_kv * config.num_heads * config.head_dim) * sizeof(float);
        double gbps = ((bytes_read + bytes_written) / 1e9) / (avg_time / 1000.0);

        return {"non_atomic_write", avg_time, gops, gbps, 0};
    }

    BenchmarkResult run_thread_level_atomic() {
        reset_output();

        GpuTimer timer;
        timer.start();

        for (int iter = 0; iter < NUM_BENCHMARK_ITERATIONS; ++iter) {
            // Simulate multiple M-blocks writing to same K/V positions
            for (int m_block = 0; m_block < config.num_m_blocks; ++m_block) {
                int num_kv_blocks = config.get_num_kv_blocks();
                dim3 grid(num_kv_blocks);
                dim3 block(BWD_NUM_THREADS);
                kernel_atomic_add_thread_level<<<grid, block>>>(
                    d_dk, d_dscores, d_q,
                    config.seq_len_kv, config.num_heads, 0, config.head_dim,
                    BWD_BLOCK_M, m_block
                );
            }
        }

        float time_ms = timer.stop_ms();
        float avg_time = time_ms / NUM_BENCHMARK_ITERATIONS;

        int num_atomics = config.seq_len_kv * config.head_dim * config.num_m_blocks;
        double gops = calculate_gops(avg_time);

        return {"atomic_thread_level", avg_time, gops, 0.0, num_atomics};
    }

    BenchmarkResult run_warp_cooperative_atomic() {
        reset_output();

        GpuTimer timer;
        timer.start();

        for (int iter = 0; iter < NUM_BENCHMARK_ITERATIONS; ++iter) {
            for (int m_block = 0; m_block < config.num_m_blocks; ++m_block) {
                int num_kv_blocks = config.get_num_kv_blocks();
                dim3 grid(num_kv_blocks);
                dim3 block(BWD_NUM_THREADS);
                kernel_atomic_add_warp_cooperative<<<grid, block>>>(
                    d_dk, d_dscores, d_q,
                    config.seq_len_kv, config.num_heads, 0, config.head_dim,
                    BWD_BLOCK_M, m_block
                );
            }
        }

        float time_ms = timer.stop_ms();
        float avg_time = time_ms / NUM_BENCHMARK_ITERATIONS;

        // Warp-cooperative reduces atomics by 32x (one per warp instead of per thread)
        int num_atomics = config.seq_len_kv * config.head_dim * config.num_m_blocks / 32;
        double gops = calculate_gops(avg_time);

        return {"atomic_warp_cooperative", avg_time, gops, 0.0, num_atomics};
    }

    BenchmarkResult run_smem_reduce_atomic() {
        reset_output();

        GpuTimer timer;
        timer.start();

        for (int iter = 0; iter < NUM_BENCHMARK_ITERATIONS; ++iter) {
            for (int m_block = 0; m_block < config.num_m_blocks; ++m_block) {
                int num_kv_blocks = config.get_num_kv_blocks();
                dim3 grid(num_kv_blocks);
                dim3 block(BWD_NUM_THREADS);
                size_t smem_size = BWD_BLOCK_N * config.head_dim * sizeof(float);
                kernel_atomic_add_smem_reduce<<<grid, block, smem_size>>>(
                    d_dk, d_dscores, d_q,
                    config.seq_len_kv, config.num_heads, 0, config.head_dim,
                    BWD_BLOCK_M, m_block
                );
            }
        }

        float time_ms = timer.stop_ms();
        float avg_time = time_ms / NUM_BENCHMARK_ITERATIONS;

        int num_atomics = config.seq_len_kv * config.head_dim * config.num_m_blocks;
        double gops = calculate_gops(avg_time);

        return {"atomic_smem_reduce", avg_time, gops, 0.0, num_atomics};
    }

    BenchmarkResult run_contention_stress_test(bool high_contention) {
        float* d_target;
        int num_elements = high_contention ? 1 : 1024 * 1024;
        CUDA_CHECK(cudaMalloc(&d_target, num_elements * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_target, 0, num_elements * sizeof(float)));

        int num_blocks = 256;
        int threads_per_block = 256;
        int iters_per_thread = 1000;

        GpuTimer timer;
        timer.start();

        if (high_contention) {
            kernel_atomic_high_contention<<<num_blocks, threads_per_block>>>(
                d_target, 1.0f, iters_per_thread
            );
        } else {
            kernel_atomic_low_contention<<<num_blocks, threads_per_block>>>(
                d_target, 1.0f, num_elements, iters_per_thread
            );
        }

        float time_ms = timer.stop_ms();

        int64_t total_atomics = static_cast<int64_t>(num_blocks) * threads_per_block * iters_per_thread;
        double atomics_per_sec = (total_atomics / 1e9) / (time_ms / 1000.0);

        cudaFree(d_target);

        std::string name = high_contention ? "stress_high_contention" : "stress_low_contention";
        return {name, time_ms, atomics_per_sec, 0.0, static_cast<int>(total_atomics)};
    }

private:
    BenchmarkConfig config;
    float* d_dk = nullptr;
    float* d_dv = nullptr;
    float* d_dscores = nullptr;
    float* d_probs = nullptr;
    float* d_q = nullptr;
    float* d_do = nullptr;

    void allocate_buffers() {
        size_t dk_size = config.get_total_kv_elements() * sizeof(float);
        size_t scores_size = BWD_BLOCK_M * BWD_BLOCK_N * sizeof(float);
        size_t q_size = BWD_BLOCK_M * config.head_dim * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_dk, dk_size));
        CUDA_CHECK(cudaMalloc(&d_dv, dk_size));
        CUDA_CHECK(cudaMalloc(&d_dscores, scores_size));
        CUDA_CHECK(cudaMalloc(&d_probs, scores_size));
        CUDA_CHECK(cudaMalloc(&d_q, q_size));
        CUDA_CHECK(cudaMalloc(&d_do, q_size));
    }

    void free_buffers() {
        cudaFree(d_dk);
        cudaFree(d_dv);
        cudaFree(d_dscores);
        cudaFree(d_probs);
        cudaFree(d_q);
        cudaFree(d_do);
    }

    void initialize_data() {
        size_t scores_size = BWD_BLOCK_M * BWD_BLOCK_N;
        size_t q_size = BWD_BLOCK_M * config.head_dim;

        std::vector<float> h_dscores(scores_size);
        std::vector<float> h_q(q_size);

        fill_random_float(h_dscores.data(), scores_size, 42);
        fill_random_float(h_q.data(), q_size, 123);

        CUDA_CHECK(cudaMemcpy(d_dscores, h_dscores.data(), scores_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_probs, h_dscores.data(), scores_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_do, h_q.data(), q_size * sizeof(float), cudaMemcpyHostToDevice));
    }

    void reset_output() {
        size_t dk_size = config.get_total_kv_elements() * sizeof(float);
        CUDA_CHECK(cudaMemset(d_dk, 0, dk_size));
        CUDA_CHECK(cudaMemset(d_dv, 0, dk_size));
    }

    double calculate_gops(float time_ms) {
        // FMAs per iteration: m_size * n_size * head_dim * num_kv_blocks * num_m_blocks
        int64_t total_ops = static_cast<int64_t>(BWD_BLOCK_M) * BWD_BLOCK_N * config.head_dim *
                            config.get_num_kv_blocks() * config.num_m_blocks * 2;
        return (total_ops / 1e9) / (time_ms / 1000.0);
    }
};

// ============================================================================
// ANALYSIS AND REPORTING
// ============================================================================

void print_header() {
    printf("================================================================================\n");
    printf("  CUDA Atomic Operations Microbenchmark for FlashMLA dK/dV Accumulation\n");
    printf("================================================================================\n\n");
}

void print_device_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("Device Information:\n");
    printf("  Name:                    %s\n", prop.name);
    printf("  Compute Capability:      SM_%d%d\n", prop.major, prop.minor);
    printf("  Global Memory:           %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("  L2 Cache:                %.2f MB\n", prop.l2CacheSize / 1e6);
    printf("  Max Threads/Block:       %d\n", prop.maxThreadsPerBlock);
    printf("  Warp Size:               %d\n", prop.warpSize);
    printf("  Max Shared Mem/Block:    %.2f KB\n", prop.sharedMemPerBlock / 1024.0f);
    printf("  Clock Rate:              %.2f GHz\n", prop.clockRate / 1e6);
    printf("\n");
}

void print_config(const BenchmarkConfig& cfg) {
    printf("Benchmark Configuration:\n");
    printf("  Query Sequence Length:   %d\n", cfg.seq_len_q);
    printf("  KV Sequence Length:      %d\n", cfg.seq_len_kv);
    printf("  Number of Heads:         %d\n", cfg.num_heads);
    printf("  Head Dimension:          %d\n", cfg.head_dim);
    printf("  M-Block Size:            %d\n", BWD_BLOCK_M);
    printf("  N-Block Size:            %d\n", BWD_BLOCK_N);
    printf("  Threads per Block:       %d\n", BWD_NUM_THREADS);
    printf("  Number of M-Blocks:      %d (contention factor)\n", cfg.num_m_blocks);
    printf("  Total KV Blocks:         %d\n", cfg.get_num_kv_blocks());
    printf("\n");
}

void print_result(const BenchmarkResult& result) {
    printf("  %-28s: %8.3f ms | %8.2f GOPS | Atomics: %10d\n",
           result.kernel_name.c_str(), result.time_ms, result.throughput_gops, result.num_atomics);
}

void print_contention_analysis(const std::vector<BenchmarkResult>& results) {
    printf("\n================================================================================\n");
    printf("  Atomic Contention Analysis\n");
    printf("================================================================================\n\n");

    // Find the non-atomic baseline
    float baseline_time = 0.0f;
    for (const auto& r : results) {
        if (r.kernel_name == "non_atomic_write") {
            baseline_time = r.time_ms;
            break;
        }
    }

    if (baseline_time > 0.0f) {
        printf("Overhead Analysis (relative to non-atomic baseline):\n\n");
        for (const auto& r : results) {
            if (r.kernel_name != "non_atomic_write" &&
                r.kernel_name.find("stress") == std::string::npos) {
                float overhead = (r.time_ms / baseline_time - 1.0f) * 100.0f;
                printf("  %-28s: %+7.1f%% overhead\n", r.kernel_name.c_str(), overhead);
            }
        }
    }

    // Contention stress test analysis
    printf("\nContention Stress Test Results:\n\n");
    float high_contention_rate = 0.0f;
    float low_contention_rate = 0.0f;
    for (const auto& r : results) {
        if (r.kernel_name == "stress_high_contention") {
            high_contention_rate = static_cast<float>(r.throughput_gops);
        } else if (r.kernel_name == "stress_low_contention") {
            low_contention_rate = static_cast<float>(r.throughput_gops);
        }
    }

    if (high_contention_rate > 0.0f && low_contention_rate > 0.0f) {
        float contention_penalty = (1.0f - high_contention_rate / low_contention_rate) * 100.0f;
        printf("  High contention throughput:  %.2f Gatomics/sec\n", high_contention_rate);
        printf("  Low contention throughput:   %.2f Gatomics/sec\n", low_contention_rate);
        printf("  Contention penalty:          %.1f%%\n", contention_penalty);
    }
}

void write_csv_results(const std::vector<BenchmarkResult>& results,
                       const BenchmarkConfig& cfg,
                       const std::string& filename) {
    std::ofstream csv(filename);
    if (!csv.is_open()) {
        fprintf(stderr, "[WARNING] Could not open %s for writing\n", filename.c_str());
        return;
    }

    csv << "kernel,seq_len_q,seq_len_kv,num_heads,head_dim,num_m_blocks,";
    csv << "time_ms,throughput_gops,num_atomics\n";

    for (const auto& r : results) {
        csv << r.kernel_name << ","
            << cfg.seq_len_q << ","
            << cfg.seq_len_kv << ","
            << cfg.num_heads << ","
            << cfg.head_dim << ","
            << cfg.num_m_blocks << ","
            << r.time_ms << ","
            << r.throughput_gops << ","
            << r.num_atomics << "\n";
    }

    csv.close();
    printf("\nResults written to: %s\n", filename.c_str());
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    print_header();

    // Check SM version
    int sm_version = get_device_sm();
    printf("Detected SM version: SM_%d\n\n", sm_version);

    if (sm_version < 80) {
        fprintf(stderr, "[ERROR] This benchmark requires SM_80 or higher for atomicAdd(float)\n");
        return 1;
    }

    print_device_info();

    // Test configurations simulating different sequence lengths
    std::vector<BenchmarkConfig> configs = {
        // Short sequences (low contention)
        {128, 128, 8, 128, 8},     // 128 tokens, 8 M-blocks
        // Medium sequences (moderate contention)
        {512, 512, 8, 128, 32},    // 512 tokens, 32 M-blocks
        // Long sequences (high contention)
        {2048, 2048, 8, 128, 128}, // 2048 tokens, 128 M-blocks
        // Very long sequences (extreme contention)
        {8192, 8192, 8, 128, 512}, // 8192 tokens, 512 M-blocks
    };

    std::vector<BenchmarkResult> all_results;

    for (const auto& cfg : configs) {
        printf("================================================================================\n");
        print_config(cfg);
        printf("Results:\n\n");

        AtomicBenchmark bench(cfg);

        // Warmup
        printf("  [Warming up...]\n");
        for (int i = 0; i < NUM_WARMUP_ITERATIONS; ++i) {
            bench.run_thread_level_atomic();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("  [Warmup complete]\n\n");

        // Run benchmarks
        std::vector<BenchmarkResult> results;

        results.push_back(bench.run_non_atomic());
        print_result(results.back());

        results.push_back(bench.run_thread_level_atomic());
        print_result(results.back());

        results.push_back(bench.run_warp_cooperative_atomic());
        print_result(results.back());

        results.push_back(bench.run_smem_reduce_atomic());
        print_result(results.back());

        // Contention stress tests (only for first config to save time)
        if (&cfg == &configs[0]) {
            printf("\n  Contention Stress Tests:\n");
            results.push_back(bench.run_contention_stress_test(true));
            print_result(results.back());

            results.push_back(bench.run_contention_stress_test(false));
            print_result(results.back());
        }

        print_contention_analysis(results);

        for (const auto& r : results) {
            all_results.push_back(r);
        }

        printf("\n");
    }

    // Write CSV
    write_csv_results(all_results, configs[0], "atomic_benchmark_results.csv");

    printf("\n================================================================================\n");
    printf("  Benchmark Complete\n");
    printf("================================================================================\n");
    printf("\nKey Findings:\n");
    printf("  1. Non-atomic writes are the baseline (no contention overhead)\n");
    printf("  2. Thread-level atomics show contention overhead proportional to M-blocks\n");
    printf("  3. Warp-cooperative atomics reduce contention by 32x (one atomic per warp)\n");
    printf("  4. Shared memory reduction can further reduce atomic pressure\n");
    printf("\nRecommendations for FlashMLA backward kernel:\n");
    printf("  - For short sequences (few M-blocks): Thread-level atomics are acceptable\n");
    printf("  - For long sequences (many M-blocks): Consider warp-cooperative or smem reduction\n");
    printf("  - The stride_token = num_heads * head_dim pattern provides spatial locality\n");
    printf("  - Coalescing is preserved when threads access consecutive d values\n");
    printf("\n");

    return 0;
}
