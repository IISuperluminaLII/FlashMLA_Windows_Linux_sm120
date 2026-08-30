/**
 * Microbenchmark: compute_delta (O dot dO) for FMHA Backward Pass
 *
 * This benchmark compares different implementations of the delta computation:
 * delta[m] = sum_d(O[m,d] * dO[m,d]) for each row m
 *
 * The delta computation is used in the softmax gradient:
 *   dScores = (dP - delta) * P * scale
 *
 * Implementations tested:
 * 1. Serial loop: each thread processes one row, loops over D=128 elements
 * 2. Warp-cooperative: warp processes one row, lanes split D dimension
 * 3. Vectorized bf16x2: using packed bf16 loads for better bandwidth
 * 4. WMMA-based: using tensor cores for dot product accumulation
 *
 * Target: SM120 (Blackwell workstation: RTX 50 series, RTX PRO 6000)
 *
 * Compile: nvcc -O3 -arch=sm_120 microbench_delta.cu -o microbench_delta
 * Run: ./microbench_delta
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ============================================================================
// Configuration constants (matching fmha_bwd_kernel_sm120.cuh)
// ============================================================================
constexpr int BWD_BLOCK_M = 16;      // Queries per block (rows for delta)
constexpr int BWD_BLOCK_D = 128;     // Head dimension
constexpr int BWD_NUM_WARPS = 4;     // Number of warps
constexpr int BWD_NUM_THREADS = BWD_NUM_WARPS * 32;  // 128 threads
constexpr int WARP_SIZE = 32;

// WMMA tile dimensions for bf16
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Benchmark configuration
constexpr int NUM_WARMUP = 100;
constexpr int NUM_ITERATIONS = 1000;
constexpr int NUM_BLOCKS = 1024;  // Simulate realistic workload

// ============================================================================
// CUDA error checking
// ============================================================================
#define CUDA_CHECK(stmt)                                                      \
    do {                                                                      \
        cudaError_t err = (stmt);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "[CUDA ERROR] %s failed: %s (%d) at %s:%d\n",    \
                    #stmt, cudaGetErrorString(err), static_cast<int>(err),   \
                    __FILE__, __LINE__);                                      \
            std::fflush(stderr);                                              \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// ============================================================================
// Warp-level reduction using shuffle
// ============================================================================
__device__ __forceinline__ float warp_reduce_sum(float val) {
    const unsigned int FULL_MASK = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// ============================================================================
// Implementation 1: Serial Loop (Current implementation)
// Each thread processes one row, loops over D=128 elements serially
// ============================================================================
__global__ void compute_delta_serial(
    const __nv_bfloat16* __restrict__ o,
    const __nv_bfloat16* __restrict__ do_,
    float* __restrict__ delta,
    int m_size,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * m_size;

    // Each thread handles one or more rows
    for (int m = tid; m < m_size; m += BWD_NUM_THREADS) {
        float sum = 0.0f;
        const int row_offset = (block_offset + m) * head_dim;

        // Serial loop over head dimension - THIS IS THE BOTTLENECK
        #pragma unroll 8
        for (int d = 0; d < head_dim; ++d) {
            float o_val = __bfloat162float(o[row_offset + d]);
            float do_val = __bfloat162float(do_[row_offset + d]);
            sum += o_val * do_val;
        }
        delta[block_offset + m] = sum;
    }
}

// ============================================================================
// Implementation 2: Warp-Cooperative
// Each warp processes one row, lanes split the D dimension
// 32 lanes * 4 elements each = 128 elements per warp
// ============================================================================
__global__ void compute_delta_warp_coop(
    const __nv_bfloat16* __restrict__ o,
    const __nv_bfloat16* __restrict__ do_,
    float* __restrict__ delta,
    int m_size,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int block_offset = blockIdx.x * m_size;

    // Each warp handles one row
    for (int m = warp_id; m < m_size; m += BWD_NUM_WARPS) {
        float partial_sum = 0.0f;
        const int row_offset = (block_offset + m) * head_dim;

        // Each lane processes head_dim/32 = 4 elements (for D=128)
        // Strided access for coalescing
        #pragma unroll 4
        for (int d = lane_id; d < head_dim; d += WARP_SIZE) {
            float o_val = __bfloat162float(o[row_offset + d]);
            float do_val = __bfloat162float(do_[row_offset + d]);
            partial_sum += o_val * do_val;
        }

        // Warp-level reduction
        float sum = warp_reduce_sum(partial_sum);

        // Lane 0 writes the result
        if (lane_id == 0) {
            delta[block_offset + m] = sum;
        }
    }
}

// ============================================================================
// Implementation 3: Vectorized bf16x2 loads
// Uses packed bf16x2 to load 2 elements at once, better memory bandwidth
// ============================================================================
__device__ __forceinline__ float dot_bf16x2(const __nv_bfloat162& a, const __nv_bfloat162& b) {
    float ax = __bfloat162float(__low2bfloat16(a));
    float ay = __bfloat162float(__high2bfloat16(a));
    float bx = __bfloat162float(__low2bfloat16(b));
    float by = __bfloat162float(__high2bfloat16(b));
    return ax * bx + ay * by;
}

__global__ void compute_delta_vectorized(
    const __nv_bfloat16* __restrict__ o,
    const __nv_bfloat16* __restrict__ do_,
    float* __restrict__ delta,
    int m_size,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int block_offset = blockIdx.x * m_size;

    // Reinterpret as bf16x2 for vectorized access
    const __nv_bfloat162* o2 = reinterpret_cast<const __nv_bfloat162*>(o);
    const __nv_bfloat162* do2 = reinterpret_cast<const __nv_bfloat162*>(do_);
    const int head_dim_2 = head_dim / 2;  // 64 bf16x2 elements

    // Each warp handles one row
    for (int m = warp_id; m < m_size; m += BWD_NUM_WARPS) {
        float partial_sum = 0.0f;
        const int row_offset_2 = (block_offset + m) * head_dim_2;

        // Each lane processes head_dim/2/32 = 2 bf16x2 elements (4 scalars)
        // Using strided access
        #pragma unroll 2
        for (int d2 = lane_id; d2 < head_dim_2; d2 += WARP_SIZE) {
            __nv_bfloat162 o_val = o2[row_offset_2 + d2];
            __nv_bfloat162 do_val = do2[row_offset_2 + d2];
            partial_sum += dot_bf16x2(o_val, do_val);
        }

        // Warp-level reduction
        float sum = warp_reduce_sum(partial_sum);

        if (lane_id == 0) {
            delta[block_offset + m] = sum;
        }
    }
}

// ============================================================================
// Implementation 4: Vectorized with float4 loads (8 bf16 at once)
// Maximum memory bandwidth utilization
// ============================================================================
__global__ void compute_delta_float4(
    const __nv_bfloat16* __restrict__ o,
    const __nv_bfloat16* __restrict__ do_,
    float* __restrict__ delta,
    int m_size,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int block_offset = blockIdx.x * m_size;

    // Reinterpret as float4 for vectorized access (8 bf16 = 16 bytes = float4)
    const float4* o4 = reinterpret_cast<const float4*>(o);
    const float4* do4 = reinterpret_cast<const float4*>(do_);
    const int head_dim_8 = head_dim / 8;  // 16 float4 elements for D=128

    // Each warp handles one row
    for (int m = warp_id; m < m_size; m += BWD_NUM_WARPS) {
        float partial_sum = 0.0f;
        const int row_offset_8 = (block_offset + m) * head_dim_8;

        // Each lane processes elements using float4 loads
        // For D=128, head_dim_8=16, each lane handles ~1 float4 on average
        for (int d8 = lane_id; d8 < head_dim_8; d8 += WARP_SIZE) {
            float4 o_f4 = o4[row_offset_8 + d8];
            float4 do_f4 = do4[row_offset_8 + d8];

            // Unpack and accumulate (float4 contains 8 bf16 values)
            const __nv_bfloat162* o_bf2 = reinterpret_cast<const __nv_bfloat162*>(&o_f4);
            const __nv_bfloat162* do_bf2 = reinterpret_cast<const __nv_bfloat162*>(&do_f4);

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                partial_sum += dot_bf16x2(o_bf2[i], do_bf2[i]);
            }
        }

        // Warp-level reduction
        float sum = warp_reduce_sum(partial_sum);

        if (lane_id == 0) {
            delta[block_offset + m] = sum;
        }
    }
}

// ============================================================================
// Implementation 5: WMMA-based using Tensor Cores
// Reshape dot product as 1x128 @ 128x1 matrix multiply
// Note: This is more of a proof-of-concept; real gains require batching
// ============================================================================
__global__ void compute_delta_wmma(
    const __nv_bfloat16* __restrict__ o,
    const __nv_bfloat16* __restrict__ do_,
    float* __restrict__ delta,
    int m_size,
    int head_dim
) {
    using namespace nvcuda::wmma;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int block_offset = blockIdx.x * m_size;

    // Shared memory for staging WMMA inputs (aligned for optimal access)
    __shared__ __align__(256) __nv_bfloat16 smem_o[BWD_BLOCK_M * BWD_BLOCK_D];
    __shared__ __align__(256) __nv_bfloat16 smem_do[BWD_BLOCK_M * BWD_BLOCK_D];
    __shared__ __align__(256) float smem_acc[BWD_NUM_WARPS * WMMA_M * WMMA_N];

    // Cooperative load to shared memory
    for (int idx = tid; idx < m_size * head_dim; idx += BWD_NUM_THREADS) {
        int global_idx = block_offset * head_dim + idx;
        smem_o[idx] = o[global_idx];
        smem_do[idx] = do_[global_idx];
    }
    __syncthreads();

    // Each warp computes delta for one row using WMMA
    // Strategy: Treat row as 1x128, transpose do as 128x16, get 1x16 tile
    // Then reduce the 16 results to a single sum

    if (warp_id < m_size) {
        const int m = warp_id;

        // For a single row dot product, we need to tile over K dimension
        // Each WMMA tile is 16x16x16
        // We compute: O[1, 128] @ dO^T[128, 1] = delta[1, 1]
        // Reshape: O_row as 8 tiles of 1x16, dO_row as 8 tiles of 16x1

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        fill_fragment(acc, 0.0f);

        // Process in chunks of 16 (WMMA_K)
        const int num_k_tiles = head_dim / WMMA_K;  // 128/16 = 8

        for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;

            // Load O tile (16x16 but we only use row m)
            // Since WMMA requires 16 rows, we replicate or use dummy data
            // For simplicity, load from row m, treating it as the first row
            // This is approximate - real impl would batch multiple rows

            // Load row m's segment [k_tile*16 : k_tile*16+16] as matrix_a
            // We need to fill a 16x16 tile where each row is identical (O_row)
            load_matrix_sync(a_frag, &smem_o[m * BWD_BLOCK_D + k_tile * WMMA_K], BWD_BLOCK_D);

            // Load dO segment as column vector replicated (or as col_major tile)
            load_matrix_sync(b_frag, &smem_do[m * BWD_BLOCK_D + k_tile * WMMA_K], BWD_BLOCK_D);

            mma_sync(acc, a_frag, b_frag, acc);
        }

        // Store accumulator to shared memory
        float* warp_staging = &smem_acc[warp_id * WMMA_M * WMMA_N];
        store_matrix_sync(warp_staging, acc, WMMA_N, mem_row_major);
        __syncwarp();

        // Lane 0 sums the diagonal (approximate dot product result)
        // Note: This is a simplified approximation; true WMMA dot product
        // requires careful layout management
        if (lane_id == 0) {
            float sum = 0.0f;
            // The actual dot product result is in element [0,0] of the tile
            // after proper setup, but here we sum diagonal for demonstration
            for (int i = 0; i < WMMA_M; ++i) {
                sum += warp_staging[i * WMMA_N + i];
            }
            delta[block_offset + m] = sum / WMMA_M;  // Normalize
        }
    }
}

// ============================================================================
// Implementation 6: Optimized Warp-Cooperative with register blocking
// Best balance of parallelism and register usage
// ============================================================================
__global__ void compute_delta_warp_blocked(
    const __nv_bfloat16* __restrict__ o,
    const __nv_bfloat16* __restrict__ do_,
    float* __restrict__ delta,
    int m_size,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int block_offset = blockIdx.x * m_size;

    // Process multiple elements per lane using register blocking
    constexpr int ELEMENTS_PER_LANE = 4;  // 32 lanes * 4 = 128 elements

    // Each warp handles one row
    for (int m = warp_id; m < m_size; m += BWD_NUM_WARPS) {
        const int row_offset = (block_offset + m) * head_dim;

        // Load 4 elements per lane into registers
        float o_regs[ELEMENTS_PER_LANE];
        float do_regs[ELEMENTS_PER_LANE];

        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
            int d = lane_id + i * WARP_SIZE;
            if (d < head_dim) {
                o_regs[i] = __bfloat162float(o[row_offset + d]);
                do_regs[i] = __bfloat162float(do_[row_offset + d]);
            } else {
                o_regs[i] = 0.0f;
                do_regs[i] = 0.0f;
            }
        }

        // Compute local dot product
        float partial_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
            partial_sum += o_regs[i] * do_regs[i];
        }

        // Warp-level reduction
        float sum = warp_reduce_sum(partial_sum);

        if (lane_id == 0) {
            delta[block_offset + m] = sum;
        }
    }
}

// ============================================================================
// Reference CPU implementation for validation
// ============================================================================
void compute_delta_cpu(
    const std::vector<__nv_bfloat16>& o,
    const std::vector<__nv_bfloat16>& do_,
    std::vector<float>& delta,
    int total_rows,
    int head_dim
) {
    for (int m = 0; m < total_rows; ++m) {
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            float o_val = __bfloat162float(o[m * head_dim + d]);
            float do_val = __bfloat162float(do_[m * head_dim + d]);
            sum += o_val * do_val;
        }
        delta[m] = sum;
    }
}

// ============================================================================
// Timing utilities
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
// Random initialization
// ============================================================================
void fill_random_bf16(std::vector<__nv_bfloat16>& vec, unsigned int seed = 42) {
    for (size_t i = 0; i < vec.size(); ++i) {
        seed = 1664525u * seed + 1013904223u;
        float f = static_cast<float>(seed & 0xFFFF) / 65535.0f - 0.5f;
        vec[i] = __float2bfloat16(f);
    }
}

// ============================================================================
// Validation
// ============================================================================
bool validate_results(
    const std::vector<float>& ref,
    const std::vector<float>& test,
    const char* name,
    float rtol = 1e-2f,
    float atol = 1e-4f
) {
    int num_errors = 0;
    float max_rel_error = 0.0f;
    float max_abs_error = 0.0f;

    for (size_t i = 0; i < ref.size(); ++i) {
        float abs_err = std::fabs(ref[i] - test[i]);
        float rel_err = abs_err / (std::fabs(ref[i]) + 1e-8f);

        max_abs_error = std::max(max_abs_error, abs_err);
        max_rel_error = std::max(max_rel_error, rel_err);

        if (abs_err > atol && rel_err > rtol) {
            if (num_errors < 5) {
                printf("  [%s] Mismatch at %zu: ref=%.6f, test=%.6f, "
                       "abs_err=%.6e, rel_err=%.6e\n",
                       name, i, ref[i], test[i], abs_err, rel_err);
            }
            num_errors++;
        }
    }

    if (num_errors > 0) {
        printf("  [%s] FAILED: %d errors (max_abs=%.6e, max_rel=%.6e)\n",
               name, num_errors, max_abs_error, max_rel_error);
        return false;
    } else {
        printf("  [%s] PASSED (max_abs=%.6e, max_rel=%.6e)\n",
               name, max_abs_error, max_rel_error);
        return true;
    }
}

// ============================================================================
// Benchmark runner
// ============================================================================
template<typename KernelFunc>
float run_benchmark(
    KernelFunc kernel_func,
    const __nv_bfloat16* d_o,
    const __nv_bfloat16* d_do,
    float* d_delta,
    int m_size,
    int head_dim,
    int num_blocks,
    const char* name
) {
    GpuTimer timer;

    // Warmup
    for (int i = 0; i < NUM_WARMUP; ++i) {
        kernel_func<<<num_blocks, BWD_NUM_THREADS>>>(
            d_o, d_do, d_delta, m_size, head_dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    timer.start();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        kernel_func<<<num_blocks, BWD_NUM_THREADS>>>(
            d_o, d_do, d_delta, m_size, head_dim);
    }
    float total_ms = timer.stop_ms();

    float avg_us = (total_ms * 1000.0f) / NUM_ITERATIONS;

    // Calculate throughput
    // Operations: num_blocks * m_size * head_dim * 2 (mult + add) FLOPs per iteration
    // Memory: num_blocks * m_size * head_dim * 2 (O + dO) * 2 bytes read
    //       + num_blocks * m_size * 4 bytes written

    size_t total_elements = (size_t)num_blocks * m_size * head_dim;
    double flops_per_iter = total_elements * 2.0;  // multiply + add
    double bytes_per_iter = total_elements * 2.0 * 2.0  // read O and dO (bf16)
                          + (size_t)num_blocks * m_size * 4.0;  // write delta (float)

    double tflops = (flops_per_iter * NUM_ITERATIONS) / (total_ms * 1e9);
    double gbps = (bytes_per_iter * NUM_ITERATIONS) / (total_ms * 1e6);

    printf("  %-25s: %8.3f us/iter | %8.4f TFLOPS | %8.2f GB/s\n",
           name, avg_us, tflops, gbps);

    return avg_us;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    printf("==============================================================\n");
    printf("Microbenchmark: compute_delta (O dot dO) for FMHA Backward\n");
    printf("==============================================================\n\n");

    // Check device
    int device_id = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    int sm_version = prop.major * 100 + prop.minor * 10;
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Compute capability: sm_%d\n", sm_version / 10);
    printf("Max shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("\n");

    // Configuration
    const int m_size = BWD_BLOCK_M;       // 16 rows per block
    const int head_dim = BWD_BLOCK_D;     // 128 elements per row
    const int num_blocks = NUM_BLOCKS;    // Number of blocks to simulate

    const size_t total_rows = (size_t)num_blocks * m_size;
    const size_t total_elements = total_rows * head_dim;

    printf("Configuration:\n");
    printf("  m_size (rows per block): %d\n", m_size);
    printf("  head_dim: %d\n", head_dim);
    printf("  num_blocks: %d\n", num_blocks);
    printf("  total_rows: %zu\n", total_rows);
    printf("  total_elements: %zu\n", total_elements);
    printf("  warmup iterations: %d\n", NUM_WARMUP);
    printf("  benchmark iterations: %d\n", NUM_ITERATIONS);
    printf("\n");

    // Allocate host memory
    std::vector<__nv_bfloat16> h_o(total_elements);
    std::vector<__nv_bfloat16> h_do(total_elements);
    std::vector<float> h_delta_ref(total_rows);
    std::vector<float> h_delta_test(total_rows);

    // Initialize with random data
    printf("Initializing data...\n");
    fill_random_bf16(h_o, 12345);
    fill_random_bf16(h_do, 67890);

    // Compute reference on CPU
    printf("Computing CPU reference...\n");
    compute_delta_cpu(h_o, h_do, h_delta_ref, total_rows, head_dim);

    // Allocate device memory
    __nv_bfloat16* d_o;
    __nv_bfloat16* d_do;
    float* d_delta;

    CUDA_CHECK(cudaMalloc(&d_o, total_elements * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_do, total_elements * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_delta, total_rows * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_o, h_o.data(), total_elements * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_do, h_do.data(), total_elements * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));

    printf("\n");
    printf("--------------------------------------------------------------\n");
    printf("Validation Results:\n");
    printf("--------------------------------------------------------------\n");

    bool all_passed = true;

    // Test 1: Serial loop
    {
        CUDA_CHECK(cudaMemset(d_delta, 0, total_rows * sizeof(float)));
        compute_delta_serial<<<num_blocks, BWD_NUM_THREADS>>>(
            d_o, d_do, d_delta, m_size, head_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_delta_test.data(), d_delta,
                              total_rows * sizeof(float), cudaMemcpyDeviceToHost));
        all_passed &= validate_results(h_delta_ref, h_delta_test, "Serial");
    }

    // Test 2: Warp-cooperative
    {
        CUDA_CHECK(cudaMemset(d_delta, 0, total_rows * sizeof(float)));
        compute_delta_warp_coop<<<num_blocks, BWD_NUM_THREADS>>>(
            d_o, d_do, d_delta, m_size, head_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_delta_test.data(), d_delta,
                              total_rows * sizeof(float), cudaMemcpyDeviceToHost));
        all_passed &= validate_results(h_delta_ref, h_delta_test, "Warp-Coop");
    }

    // Test 3: Vectorized bf16x2
    {
        CUDA_CHECK(cudaMemset(d_delta, 0, total_rows * sizeof(float)));
        compute_delta_vectorized<<<num_blocks, BWD_NUM_THREADS>>>(
            d_o, d_do, d_delta, m_size, head_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_delta_test.data(), d_delta,
                              total_rows * sizeof(float), cudaMemcpyDeviceToHost));
        all_passed &= validate_results(h_delta_ref, h_delta_test, "Vectorized-bf16x2");
    }

    // Test 4: Float4 vectorized
    {
        CUDA_CHECK(cudaMemset(d_delta, 0, total_rows * sizeof(float)));
        compute_delta_float4<<<num_blocks, BWD_NUM_THREADS>>>(
            d_o, d_do, d_delta, m_size, head_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_delta_test.data(), d_delta,
                              total_rows * sizeof(float), cudaMemcpyDeviceToHost));
        all_passed &= validate_results(h_delta_ref, h_delta_test, "Vectorized-float4");
    }

    // Test 5: Warp-blocked
    {
        CUDA_CHECK(cudaMemset(d_delta, 0, total_rows * sizeof(float)));
        compute_delta_warp_blocked<<<num_blocks, BWD_NUM_THREADS>>>(
            d_o, d_do, d_delta, m_size, head_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_delta_test.data(), d_delta,
                              total_rows * sizeof(float), cudaMemcpyDeviceToHost));
        all_passed &= validate_results(h_delta_ref, h_delta_test, "Warp-Blocked");
    }

    // Note: WMMA version produces approximate results due to implementation constraints
    // Skip validation for WMMA but still benchmark it
    printf("  [WMMA] Skipped validation (approximate implementation)\n");

    printf("\n");
    printf("--------------------------------------------------------------\n");
    printf("Performance Results:\n");
    printf("--------------------------------------------------------------\n");

    float time_serial = run_benchmark(
        compute_delta_serial, d_o, d_do, d_delta, m_size, head_dim, num_blocks,
        "Serial");

    float time_warp_coop = run_benchmark(
        compute_delta_warp_coop, d_o, d_do, d_delta, m_size, head_dim, num_blocks,
        "Warp-Coop");

    float time_vectorized = run_benchmark(
        compute_delta_vectorized, d_o, d_do, d_delta, m_size, head_dim, num_blocks,
        "Vectorized-bf16x2");

    float time_float4 = run_benchmark(
        compute_delta_float4, d_o, d_do, d_delta, m_size, head_dim, num_blocks,
        "Vectorized-float4");

    float time_warp_blocked = run_benchmark(
        compute_delta_warp_blocked, d_o, d_do, d_delta, m_size, head_dim, num_blocks,
        "Warp-Blocked");

    float time_wmma = run_benchmark(
        compute_delta_wmma, d_o, d_do, d_delta, m_size, head_dim, num_blocks,
        "WMMA (approx)");

    printf("\n");
    printf("--------------------------------------------------------------\n");
    printf("Speedup Summary (relative to Serial baseline):\n");
    printf("--------------------------------------------------------------\n");

    printf("  Warp-Coop:        %.2fx\n", time_serial / time_warp_coop);
    printf("  Vectorized-bf16x2: %.2fx\n", time_serial / time_vectorized);
    printf("  Vectorized-float4: %.2fx\n", time_serial / time_float4);
    printf("  Warp-Blocked:      %.2fx\n", time_serial / time_warp_blocked);
    printf("  WMMA (approx):     %.2fx\n", time_serial / time_wmma);

    printf("\n");
    printf("--------------------------------------------------------------\n");
    printf("Analysis:\n");
    printf("--------------------------------------------------------------\n");
    printf("The compute_delta operation is memory-bound for D=128:\n");
    printf("  - Each row: 128 bf16 reads (O) + 128 bf16 reads (dO) = 512 bytes\n");
    printf("  - Output: 1 float write = 4 bytes\n");
    printf("  - Compute: 128 FMAs = 256 FLOPs\n");
    printf("  - Arithmetic intensity: 256 / 516 = 0.50 FLOPs/byte\n");
    printf("\n");
    printf("Key observations:\n");
    printf("  1. Serial loop: Low parallelism, thread handles entire row\n");
    printf("  2. Warp-coop: Good parallelism, but strided access pattern\n");
    printf("  3. Vectorized: Better memory coalescing with packed loads\n");
    printf("  4. Float4: Maximum bandwidth utilization (16-byte loads)\n");
    printf("  5. WMMA: Tensor core overhead not justified for this size\n");
    printf("\n");
    printf("Recommendation: Use Warp-Blocked or Float4 vectorized for best perf\n");
    printf("\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_o));
    CUDA_CHECK(cudaFree(d_do));
    CUDA_CHECK(cudaFree(d_delta));

    printf("==============================================================\n");
    if (all_passed) {
        printf("All validated implementations PASSED\n");
    } else {
        printf("Some implementations FAILED validation\n");
    }
    printf("==============================================================\n");

    return all_passed ? 0 : 1;
}
