/******************************************************************************
 * microbench_memory.cu - CUDA Memory Microbenchmark for FlashMLA Backward Kernel
 *
 * Benchmarks memory operations used in the SM120 backward kernel:
 * - Global to shared memory loads (bf16 tiles)
 * - Shared to global memory stores (bf16 and float)
 * - cp.async operations for async prefetching
 * - Coalesced vs non-coalesced access patterns
 * - Vectorized bf16x2 loads
 * - Atomic float adds to global memory
 *
 * Target: SM 120 (Blackwell workstation GPUs)
 * Compile: nvcc -arch=sm_120 -O3 microbench_memory.cu -o microbench_memory
 *
 *****************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// --------------------------------------------------------------------------
// Macros and Configuration
// --------------------------------------------------------------------------

#define CUDA_CHECK(stmt)                                                       \
    do {                                                                       \
        cudaError_t _e = (stmt);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA ERROR] %s failed: %s (%d) at %s:%d\n",      \
                    #stmt, cudaGetErrorString(_e), static_cast<int>(_e),       \
                    __FILE__, __LINE__);                                       \
            std::fflush(stderr);                                               \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Tile dimensions matching backward kernel
constexpr int Q_TILE_ROWS = 16;
constexpr int Q_TILE_COLS = 128;    // Head dimension D
constexpr int KV_TILE_ROWS = 32;
constexpr int KV_TILE_COLS = 128;   // Head dimension D

// Thread configuration
constexpr int THREADS_PER_BLOCK = 128;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;

// Benchmark parameters
constexpr int WARMUP_ITERS = 10;
constexpr int BENCH_ITERS = 100;

// cp.async support (SM80+)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define HAS_CP_ASYNC 1
#else
#define HAS_CP_ASYNC 0
#endif

// --------------------------------------------------------------------------
// Utility Structures
// --------------------------------------------------------------------------

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
        CUDA_CHECK(cudaEventRecord(start_event, 0));
    }

    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
        return ms;
    }
};

struct BenchResult {
    const char* name;
    float time_ms;
    double bytes_total;
    double bandwidth_gbs;
    double efficiency_pct;
};

// --------------------------------------------------------------------------
// Device Helper Functions
// --------------------------------------------------------------------------

// Inline PTX for cp.async 16-byte copy (4 bf16x2 = 16 bytes)
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
#if HAS_CP_ASYNC
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr)
    );
#else
    // Fallback for older architectures
    __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(smem_ptr);
    const __nv_bfloat162* src = reinterpret_cast<const __nv_bfloat162*>(gmem_ptr);
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if HAS_CP_ASYNC
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if HAS_CP_ASYNC
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_group(int n) {
#if HAS_CP_ASYNC
    // Wait for all but n groups
    if (n == 0) {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    } else if (n == 1) {
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    }
#endif
}

// --------------------------------------------------------------------------
// Benchmark Kernels
// --------------------------------------------------------------------------

// Kernel 1: Coalesced global-to-shared bf16 load (Q tile: 16x128)
__global__ void bench_coalesced_g2s_q_tile(
    const __nv_bfloat16* __restrict__ g_src,
    __nv_bfloat16* __restrict__ g_dst,  // dummy output for verification
    int iters
) {
    extern __shared__ __nv_bfloat16 smem[];

    constexpr int TILE_ELEMS = Q_TILE_ROWS * Q_TILE_COLS;  // 16*128 = 2048
    constexpr int ELEMS_PER_THREAD = TILE_ELEMS / THREADS_PER_BLOCK;  // 16

    int tid = threadIdx.x;

    for (int iter = 0; iter < iters; ++iter) {
        // Coalesced load: threads read consecutive elements
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            int idx = tid + i * THREADS_PER_BLOCK;
            smem[idx] = g_src[idx];
        }
        __syncthreads();

        // Write back to verify correctness
        if (iter == iters - 1) {
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                int idx = tid + i * THREADS_PER_BLOCK;
                g_dst[idx] = smem[idx];
            }
        }
        __syncthreads();
    }
}

// Kernel 2: Non-coalesced (strided) global-to-shared bf16 load
__global__ void bench_strided_g2s_q_tile(
    const __nv_bfloat16* __restrict__ g_src,
    __nv_bfloat16* __restrict__ g_dst,
    int iters
) {
    extern __shared__ __nv_bfloat16 smem[];

    constexpr int TILE_ELEMS = Q_TILE_ROWS * Q_TILE_COLS;
    constexpr int STRIDE = Q_TILE_COLS;  // 128-element stride (bad for coalescing)

    int tid = threadIdx.x;
    int row = tid / 8;    // 16 rows
    int col = tid % 8;    // 8 threads per row (not covering full row)

    for (int iter = 0; iter < iters; ++iter) {
        // Strided access: each thread accesses row-major with gaps
        for (int c = col; c < Q_TILE_COLS; c += 8) {
            if (row < Q_TILE_ROWS) {
                int src_idx = row * STRIDE + c;
                smem[src_idx] = g_src[src_idx];
            }
        }
        __syncthreads();

        if (iter == iters - 1 && tid < TILE_ELEMS) {
            g_dst[tid] = smem[tid];
        }
        __syncthreads();
    }
}

// Kernel 3: Coalesced K/V tile load (32x128 bf16)
__global__ void bench_coalesced_g2s_kv_tile(
    const __nv_bfloat16* __restrict__ g_src,
    __nv_bfloat16* __restrict__ g_dst,
    int iters
) {
    extern __shared__ __nv_bfloat16 smem[];

    constexpr int TILE_ELEMS = KV_TILE_ROWS * KV_TILE_COLS;  // 32*128 = 4096
    constexpr int ELEMS_PER_THREAD = TILE_ELEMS / THREADS_PER_BLOCK;  // 32

    int tid = threadIdx.x;

    for (int iter = 0; iter < iters; ++iter) {
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            int idx = tid + i * THREADS_PER_BLOCK;
            smem[idx] = g_src[idx];
        }
        __syncthreads();

        if (iter == iters - 1) {
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                int idx = tid + i * THREADS_PER_BLOCK;
                g_dst[idx] = smem[idx];
            }
        }
        __syncthreads();
    }
}

// Kernel 4: Vectorized bf16x2 loads (2 bf16 per load = 4 bytes)
__global__ void bench_vectorized_bf16x2_load(
    const __nv_bfloat16* __restrict__ g_src,
    __nv_bfloat16* __restrict__ g_dst,
    int iters
) {
    extern __shared__ __nv_bfloat162 smem_v2[];

    constexpr int TILE_ELEMS = KV_TILE_ROWS * KV_TILE_COLS;
    constexpr int VEC2_ELEMS = TILE_ELEMS / 2;  // 2048 bf16x2
    constexpr int VECS_PER_THREAD = VEC2_ELEMS / THREADS_PER_BLOCK;  // 16

    const __nv_bfloat162* g_src_v2 = reinterpret_cast<const __nv_bfloat162*>(g_src);
    __nv_bfloat162* g_dst_v2 = reinterpret_cast<__nv_bfloat162*>(g_dst);

    int tid = threadIdx.x;

    for (int iter = 0; iter < iters; ++iter) {
        #pragma unroll
        for (int i = 0; i < VECS_PER_THREAD; ++i) {
            int idx = tid + i * THREADS_PER_BLOCK;
            smem_v2[idx] = g_src_v2[idx];
        }
        __syncthreads();

        if (iter == iters - 1) {
            #pragma unroll
            for (int i = 0; i < VECS_PER_THREAD; ++i) {
                int idx = tid + i * THREADS_PER_BLOCK;
                g_dst_v2[idx] = smem_v2[idx];
            }
        }
        __syncthreads();
    }
}

// Kernel 5: cp.async.cg 16-byte copies for async prefetch
__global__ void bench_cp_async_16byte(
    const __nv_bfloat16* __restrict__ g_src,
    __nv_bfloat16* __restrict__ g_dst,
    int iters
) {
    extern __shared__ __nv_bfloat16 smem[];

    constexpr int TILE_ELEMS = KV_TILE_ROWS * KV_TILE_COLS;  // 4096 bf16
    constexpr int BYTES_PER_COPY = 16;
    constexpr int BF16_PER_COPY = BYTES_PER_COPY / sizeof(__nv_bfloat16);  // 8
    constexpr int COPIES_TOTAL = TILE_ELEMS / BF16_PER_COPY;  // 512
    constexpr int COPIES_PER_THREAD = COPIES_TOTAL / THREADS_PER_BLOCK;  // 4

    int tid = threadIdx.x;

    for (int iter = 0; iter < iters; ++iter) {
        #pragma unroll
        for (int i = 0; i < COPIES_PER_THREAD; ++i) {
            int copy_idx = tid + i * THREADS_PER_BLOCK;
            int elem_idx = copy_idx * BF16_PER_COPY;
            cp_async_cg_16(&smem[elem_idx], &g_src[elem_idx]);
        }
        cp_async_commit_group();
        cp_async_wait_all();
        __syncthreads();

        if (iter == iters - 1) {
            constexpr int ELEMS_PER_THREAD = TILE_ELEMS / THREADS_PER_BLOCK;
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                int idx = tid + i * THREADS_PER_BLOCK;
                g_dst[idx] = smem[idx];
            }
        }
        __syncthreads();
    }
}

// Kernel 6: Shared-to-global float store (dQ accumulator writeback)
__global__ void bench_s2g_float_store(
    const float* __restrict__ g_src,
    float* __restrict__ g_dst,
    int iters
) {
    extern __shared__ float smem_f[];

    constexpr int TILE_ELEMS = Q_TILE_ROWS * Q_TILE_COLS;  // 16*128 = 2048 floats
    constexpr int ELEMS_PER_THREAD = TILE_ELEMS / THREADS_PER_BLOCK;

    int tid = threadIdx.x;

    for (int iter = 0; iter < iters; ++iter) {
        // Load to shared
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            int idx = tid + i * THREADS_PER_BLOCK;
            smem_f[idx] = g_src[idx];
        }
        __syncthreads();

        // Coalesced store to global
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            int idx = tid + i * THREADS_PER_BLOCK;
            g_dst[idx] = smem_f[idx];
        }
        __syncthreads();
    }
}

// Kernel 7: Shared-to-global bf16 store with conversion
__global__ void bench_s2g_bf16_store(
    const float* __restrict__ g_src,
    __nv_bfloat16* __restrict__ g_dst,
    int iters
) {
    extern __shared__ float smem_f[];

    constexpr int TILE_ELEMS = Q_TILE_ROWS * Q_TILE_COLS;
    constexpr int ELEMS_PER_THREAD = TILE_ELEMS / THREADS_PER_BLOCK;

    int tid = threadIdx.x;

    for (int iter = 0; iter < iters; ++iter) {
        // Load float to shared
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            int idx = tid + i * THREADS_PER_BLOCK;
            smem_f[idx] = g_src[idx];
        }
        __syncthreads();

        // Convert and store as bf16
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            int idx = tid + i * THREADS_PER_BLOCK;
            g_dst[idx] = __float2bfloat16(smem_f[idx]);
        }
        __syncthreads();
    }
}

// Kernel 8: Atomic float adds (for gradient accumulation)
__global__ void bench_atomic_float_add(
    float* __restrict__ g_dst,
    int num_elements,
    int iters
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float val = 1.0f;
    for (int iter = 0; iter < iters; ++iter) {
        for (int i = tid; i < num_elements; i += stride) {
            atomicAdd(&g_dst[i], val);
        }
    }
}

// Kernel 9: Double-buffered cp.async (ping-pong pattern)
__global__ void bench_double_buffer_cp_async(
    const __nv_bfloat16* __restrict__ g_src,
    __nv_bfloat16* __restrict__ g_dst,
    int num_tiles,
    int iters
) {
    extern __shared__ __nv_bfloat16 smem_db[];

    constexpr int TILE_ELEMS = KV_TILE_ROWS * KV_TILE_COLS;
    constexpr int BYTES_PER_COPY = 16;
    constexpr int BF16_PER_COPY = BYTES_PER_COPY / sizeof(__nv_bfloat16);
    constexpr int COPIES_TOTAL = TILE_ELEMS / BF16_PER_COPY;
    constexpr int COPIES_PER_THREAD = COPIES_TOTAL / THREADS_PER_BLOCK;
    constexpr int ELEMS_PER_THREAD = TILE_ELEMS / THREADS_PER_BLOCK;

    __nv_bfloat16* smem_buf0 = smem_db;
    __nv_bfloat16* smem_buf1 = smem_db + TILE_ELEMS;

    int tid = threadIdx.x;

    for (int iter = 0; iter < iters; ++iter) {
        int current_buf = 0;

        // Prefetch first tile
        #pragma unroll
        for (int i = 0; i < COPIES_PER_THREAD; ++i) {
            int copy_idx = tid + i * THREADS_PER_BLOCK;
            int elem_idx = copy_idx * BF16_PER_COPY;
            cp_async_cg_16(&smem_buf0[elem_idx], &g_src[elem_idx]);
        }
        cp_async_commit_group();

        for (int tile = 0; tile < num_tiles; ++tile) {
            __nv_bfloat16* curr_buf = (current_buf == 0) ? smem_buf0 : smem_buf1;
            __nv_bfloat16* next_buf = (current_buf == 0) ? smem_buf1 : smem_buf0;

            // Prefetch next tile (if not last)
            if (tile < num_tiles - 1) {
                int next_tile_offset = ((tile + 1) % num_tiles) * TILE_ELEMS;
                #pragma unroll
                for (int i = 0; i < COPIES_PER_THREAD; ++i) {
                    int copy_idx = tid + i * THREADS_PER_BLOCK;
                    int elem_idx = copy_idx * BF16_PER_COPY;
                    cp_async_cg_16(&next_buf[elem_idx], &g_src[next_tile_offset + elem_idx]);
                }
                cp_async_commit_group();
            }

            // Wait for current tile
            cp_async_wait_group(tile < num_tiles - 1 ? 1 : 0);
            __syncthreads();

            // Process current tile (dummy operation - just sum for verification)
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                int idx = tid + i * THREADS_PER_BLOCK;
                sum += __bfloat162float(curr_buf[idx]);
            }

            // Write result for last iteration
            if (iter == iters - 1 && tile == num_tiles - 1) {
                #pragma unroll
                for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                    int idx = tid + i * THREADS_PER_BLOCK;
                    g_dst[idx] = curr_buf[idx];
                }
            }

            current_buf = 1 - current_buf;
            __syncthreads();
        }
    }
}

// --------------------------------------------------------------------------
// Host Benchmark Runner
// --------------------------------------------------------------------------

class MemoryBenchmark {
public:
    MemoryBenchmark() {
        CUDA_CHECK(cudaGetDeviceProperties(&device_props_, 0));
        sm_version_ = device_props_.major * 100 + device_props_.minor * 10;

        // Calculate theoretical peak bandwidth
        // memory_clock_rate is in KHz, memory_bus_width in bits
        // Bandwidth = clock * bus_width / 8 * 2 (DDR) / 1e9 GB/s
        double clock_khz = static_cast<double>(device_props_.memoryClockRate);
        double bus_width_bytes = static_cast<double>(device_props_.memoryBusWidth) / 8.0;
        theoretical_bandwidth_gbs_ = clock_khz * 1e3 * bus_width_bytes * 2.0 / 1e9;

        printf("================================================================================\n");
        printf("FlashMLA Memory Microbenchmark\n");
        printf("================================================================================\n");
        printf("Device: %s\n", device_props_.name);
        printf("SM Version: sm_%d\n", sm_version_);
        printf("Memory Clock: %.2f MHz\n", clock_khz / 1e3);
        printf("Memory Bus Width: %d bits\n", device_props_.memoryBusWidth);
        printf("Theoretical Peak Bandwidth: %.2f GB/s\n", theoretical_bandwidth_gbs_);
        printf("Shared Memory per Block: %zu KB\n", device_props_.sharedMemPerBlock / 1024);
        printf("================================================================================\n\n");
    }

    void run_all() {
        std::vector<BenchResult> results;

        results.push_back(bench_q_tile_coalesced());
        results.push_back(bench_q_tile_strided());
        results.push_back(bench_kv_tile_coalesced());
        results.push_back(bench_kv_tile_vectorized());
        results.push_back(bench_kv_tile_cp_async());
        results.push_back(bench_float_store());
        results.push_back(bench_bf16_store());
        results.push_back(bench_atomic_add());
        results.push_back(bench_double_buffer());

        print_results(results);
    }

private:
    cudaDeviceProp device_props_;
    int sm_version_;
    double theoretical_bandwidth_gbs_;

    void print_results(const std::vector<BenchResult>& results) {
        printf("\n================================================================================\n");
        printf("BENCHMARK RESULTS\n");
        printf("================================================================================\n");
        printf("%-40s %10s %12s %12s %10s\n",
               "Benchmark", "Time(ms)", "Bytes", "BW(GB/s)", "Eff(%)");
        printf("--------------------------------------------------------------------------------\n");

        for (const auto& r : results) {
            printf("%-40s %10.4f %12.2e %12.2f %10.2f\n",
                   r.name, r.time_ms, r.bytes_total, r.bandwidth_gbs, r.efficiency_pct);
        }

        printf("================================================================================\n");
        printf("Theoretical Peak: %.2f GB/s\n", theoretical_bandwidth_gbs_);
        printf("================================================================================\n");
    }

    BenchResult bench_q_tile_coalesced() {
        const char* name = "Q tile coalesced G2S (16x128 bf16)";
        constexpr int TILE_ELEMS = Q_TILE_ROWS * Q_TILE_COLS;
        size_t bytes_per_iter = TILE_ELEMS * sizeof(__nv_bfloat16);

        __nv_bfloat16 *d_src, *d_dst;
        CUDA_CHECK(cudaMalloc(&d_src, bytes_per_iter));
        CUDA_CHECK(cudaMalloc(&d_dst, bytes_per_iter));
        CUDA_CHECK(cudaMemset(d_src, 0x3f, bytes_per_iter));  // ~1.0 in bf16

        size_t smem_size = bytes_per_iter;
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(1);

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            bench_coalesced_g2s_q_tile<<<grid, block, smem_size>>>(d_src, d_dst, 1);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        GpuTimer timer;
        timer.start();
        bench_coalesced_g2s_q_tile<<<grid, block, smem_size>>>(d_src, d_dst, BENCH_ITERS);
        float ms = timer.stop_ms();

        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_dst));

        double bytes_total = static_cast<double>(bytes_per_iter) * BENCH_ITERS;
        double bandwidth = bytes_total / (ms * 1e-3) / 1e9;
        double efficiency = (bandwidth / theoretical_bandwidth_gbs_) * 100.0;

        return {name, ms, bytes_total, bandwidth, efficiency};
    }

    BenchResult bench_q_tile_strided() {
        const char* name = "Q tile strided G2S (16x128 bf16)";
        constexpr int TILE_ELEMS = Q_TILE_ROWS * Q_TILE_COLS;
        size_t bytes_per_iter = TILE_ELEMS * sizeof(__nv_bfloat16);

        __nv_bfloat16 *d_src, *d_dst;
        CUDA_CHECK(cudaMalloc(&d_src, bytes_per_iter));
        CUDA_CHECK(cudaMalloc(&d_dst, bytes_per_iter));
        CUDA_CHECK(cudaMemset(d_src, 0x3f, bytes_per_iter));

        size_t smem_size = bytes_per_iter;
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(1);

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            bench_strided_g2s_q_tile<<<grid, block, smem_size>>>(d_src, d_dst, 1);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        GpuTimer timer;
        timer.start();
        bench_strided_g2s_q_tile<<<grid, block, smem_size>>>(d_src, d_dst, BENCH_ITERS);
        float ms = timer.stop_ms();

        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_dst));

        double bytes_total = static_cast<double>(bytes_per_iter) * BENCH_ITERS;
        double bandwidth = bytes_total / (ms * 1e-3) / 1e9;
        double efficiency = (bandwidth / theoretical_bandwidth_gbs_) * 100.0;

        return {name, ms, bytes_total, bandwidth, efficiency};
    }

    BenchResult bench_kv_tile_coalesced() {
        const char* name = "KV tile coalesced G2S (32x128 bf16)";
        constexpr int TILE_ELEMS = KV_TILE_ROWS * KV_TILE_COLS;
        size_t bytes_per_iter = TILE_ELEMS * sizeof(__nv_bfloat16);

        __nv_bfloat16 *d_src, *d_dst;
        CUDA_CHECK(cudaMalloc(&d_src, bytes_per_iter));
        CUDA_CHECK(cudaMalloc(&d_dst, bytes_per_iter));
        CUDA_CHECK(cudaMemset(d_src, 0x3f, bytes_per_iter));

        size_t smem_size = bytes_per_iter;
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(1);

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            bench_coalesced_g2s_kv_tile<<<grid, block, smem_size>>>(d_src, d_dst, 1);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        GpuTimer timer;
        timer.start();
        bench_coalesced_g2s_kv_tile<<<grid, block, smem_size>>>(d_src, d_dst, BENCH_ITERS);
        float ms = timer.stop_ms();

        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_dst));

        double bytes_total = static_cast<double>(bytes_per_iter) * BENCH_ITERS;
        double bandwidth = bytes_total / (ms * 1e-3) / 1e9;
        double efficiency = (bandwidth / theoretical_bandwidth_gbs_) * 100.0;

        return {name, ms, bytes_total, bandwidth, efficiency};
    }

    BenchResult bench_kv_tile_vectorized() {
        const char* name = "KV tile vectorized bf16x2 (32x128)";
        constexpr int TILE_ELEMS = KV_TILE_ROWS * KV_TILE_COLS;
        size_t bytes_per_iter = TILE_ELEMS * sizeof(__nv_bfloat16);

        __nv_bfloat16 *d_src, *d_dst;
        CUDA_CHECK(cudaMalloc(&d_src, bytes_per_iter));
        CUDA_CHECK(cudaMalloc(&d_dst, bytes_per_iter));
        CUDA_CHECK(cudaMemset(d_src, 0x3f, bytes_per_iter));

        size_t smem_size = bytes_per_iter;
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(1);

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            bench_vectorized_bf16x2_load<<<grid, block, smem_size>>>(d_src, d_dst, 1);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        GpuTimer timer;
        timer.start();
        bench_vectorized_bf16x2_load<<<grid, block, smem_size>>>(d_src, d_dst, BENCH_ITERS);
        float ms = timer.stop_ms();

        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_dst));

        double bytes_total = static_cast<double>(bytes_per_iter) * BENCH_ITERS;
        double bandwidth = bytes_total / (ms * 1e-3) / 1e9;
        double efficiency = (bandwidth / theoretical_bandwidth_gbs_) * 100.0;

        return {name, ms, bytes_total, bandwidth, efficiency};
    }

    BenchResult bench_kv_tile_cp_async() {
        const char* name = "KV tile cp.async 16B (32x128 bf16)";
        constexpr int TILE_ELEMS = KV_TILE_ROWS * KV_TILE_COLS;
        size_t bytes_per_iter = TILE_ELEMS * sizeof(__nv_bfloat16);

        __nv_bfloat16 *d_src, *d_dst;
        CUDA_CHECK(cudaMalloc(&d_src, bytes_per_iter));
        CUDA_CHECK(cudaMalloc(&d_dst, bytes_per_iter));
        CUDA_CHECK(cudaMemset(d_src, 0x3f, bytes_per_iter));

        size_t smem_size = bytes_per_iter;
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(1);

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            bench_cp_async_16byte<<<grid, block, smem_size>>>(d_src, d_dst, 1);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        GpuTimer timer;
        timer.start();
        bench_cp_async_16byte<<<grid, block, smem_size>>>(d_src, d_dst, BENCH_ITERS);
        float ms = timer.stop_ms();

        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_dst));

        double bytes_total = static_cast<double>(bytes_per_iter) * BENCH_ITERS;
        double bandwidth = bytes_total / (ms * 1e-3) / 1e9;
        double efficiency = (bandwidth / theoretical_bandwidth_gbs_) * 100.0;

        return {name, ms, bytes_total, bandwidth, efficiency};
    }

    BenchResult bench_float_store() {
        const char* name = "Float S2G store (16x128 f32)";
        constexpr int TILE_ELEMS = Q_TILE_ROWS * Q_TILE_COLS;
        size_t bytes_per_iter = TILE_ELEMS * sizeof(float);

        float *d_src, *d_dst;
        CUDA_CHECK(cudaMalloc(&d_src, bytes_per_iter));
        CUDA_CHECK(cudaMalloc(&d_dst, bytes_per_iter));
        CUDA_CHECK(cudaMemset(d_src, 0, bytes_per_iter));

        // Initialize with some values
        std::vector<float> h_src(TILE_ELEMS);
        for (int i = 0; i < TILE_ELEMS; ++i) {
            h_src[i] = static_cast<float>(i % 100) / 100.0f;
        }
        CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), bytes_per_iter, cudaMemcpyHostToDevice));

        size_t smem_size = bytes_per_iter;
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(1);

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            bench_s2g_float_store<<<grid, block, smem_size>>>(d_src, d_dst, 1);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        GpuTimer timer;
        timer.start();
        bench_s2g_float_store<<<grid, block, smem_size>>>(d_src, d_dst, BENCH_ITERS);
        float ms = timer.stop_ms();

        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_dst));

        // Count both load and store
        double bytes_total = static_cast<double>(bytes_per_iter) * 2 * BENCH_ITERS;
        double bandwidth = bytes_total / (ms * 1e-3) / 1e9;
        double efficiency = (bandwidth / theoretical_bandwidth_gbs_) * 100.0;

        return {name, ms, bytes_total, bandwidth, efficiency};
    }

    BenchResult bench_bf16_store() {
        const char* name = "BF16 S2G store w/ conv (16x128)";
        constexpr int TILE_ELEMS = Q_TILE_ROWS * Q_TILE_COLS;
        size_t float_bytes = TILE_ELEMS * sizeof(float);
        size_t bf16_bytes = TILE_ELEMS * sizeof(__nv_bfloat16);

        float* d_src;
        __nv_bfloat16* d_dst;
        CUDA_CHECK(cudaMalloc(&d_src, float_bytes));
        CUDA_CHECK(cudaMalloc(&d_dst, bf16_bytes));

        std::vector<float> h_src(TILE_ELEMS);
        for (int i = 0; i < TILE_ELEMS; ++i) {
            h_src[i] = static_cast<float>(i % 100) / 100.0f;
        }
        CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), float_bytes, cudaMemcpyHostToDevice));

        size_t smem_size = float_bytes;  // smem holds floats
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(1);

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            bench_s2g_bf16_store<<<grid, block, smem_size>>>(d_src, d_dst, 1);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        GpuTimer timer;
        timer.start();
        bench_s2g_bf16_store<<<grid, block, smem_size>>>(d_src, d_dst, BENCH_ITERS);
        float ms = timer.stop_ms();

        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_dst));

        // Load float, store bf16
        double bytes_total = static_cast<double>(float_bytes + bf16_bytes) * BENCH_ITERS;
        double bandwidth = bytes_total / (ms * 1e-3) / 1e9;
        double efficiency = (bandwidth / theoretical_bandwidth_gbs_) * 100.0;

        return {name, ms, bytes_total, bandwidth, efficiency};
    }

    BenchResult bench_atomic_add() {
        const char* name = "Atomic float add (gradient accum)";
        constexpr int NUM_ELEMENTS = Q_TILE_ROWS * Q_TILE_COLS;  // 2048
        size_t bytes = NUM_ELEMENTS * sizeof(float);

        float* d_dst;
        CUDA_CHECK(cudaMalloc(&d_dst, bytes));
        CUDA_CHECK(cudaMemset(d_dst, 0, bytes));

        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(4);  // Multiple blocks to stress atomics

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            bench_atomic_float_add<<<grid, block>>>(d_dst, NUM_ELEMENTS, 1);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(d_dst, 0, bytes));

        // Benchmark
        GpuTimer timer;
        timer.start();
        bench_atomic_float_add<<<grid, block>>>(d_dst, NUM_ELEMENTS, BENCH_ITERS);
        float ms = timer.stop_ms();

        CUDA_CHECK(cudaFree(d_dst));

        // Each thread does NUM_ELEMENTS / (grid * block) atomic adds per iter
        int total_threads = grid.x * block.x;
        int ops_per_thread = (NUM_ELEMENTS + total_threads - 1) / total_threads;
        double total_ops = static_cast<double>(total_threads) * ops_per_thread * BENCH_ITERS;
        double bytes_total = total_ops * sizeof(float);  // RMW
        double bandwidth = bytes_total / (ms * 1e-3) / 1e9;
        double efficiency = (bandwidth / theoretical_bandwidth_gbs_) * 100.0;

        return {name, ms, bytes_total, bandwidth, efficiency};
    }

    BenchResult bench_double_buffer() {
        const char* name = "Double-buffer cp.async (ping-pong)";
        constexpr int TILE_ELEMS = KV_TILE_ROWS * KV_TILE_COLS;
        constexpr int NUM_TILES = 4;
        size_t bytes_per_tile = TILE_ELEMS * sizeof(__nv_bfloat16);

        __nv_bfloat16 *d_src, *d_dst;
        CUDA_CHECK(cudaMalloc(&d_src, bytes_per_tile * NUM_TILES));
        CUDA_CHECK(cudaMalloc(&d_dst, bytes_per_tile));
        CUDA_CHECK(cudaMemset(d_src, 0x3f, bytes_per_tile * NUM_TILES));

        // Double buffer needs 2x tile size
        size_t smem_size = bytes_per_tile * 2;
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(1);

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            bench_double_buffer_cp_async<<<grid, block, smem_size>>>(d_src, d_dst, NUM_TILES, 1);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        GpuTimer timer;
        timer.start();
        bench_double_buffer_cp_async<<<grid, block, smem_size>>>(d_src, d_dst, NUM_TILES, BENCH_ITERS);
        float ms = timer.stop_ms();

        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_dst));

        // Total bytes loaded across all tiles and iterations
        double bytes_total = static_cast<double>(bytes_per_tile) * NUM_TILES * BENCH_ITERS;
        double bandwidth = bytes_total / (ms * 1e-3) / 1e9;
        double efficiency = (bandwidth / theoretical_bandwidth_gbs_) * 100.0;

        return {name, ms, bytes_total, bandwidth, efficiency};
    }
};

// --------------------------------------------------------------------------
// Comparison utilities
// --------------------------------------------------------------------------

template <typename T>
bool compare_host(const std::vector<T>& a, const std::vector<T>& b, float atol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        float fa = static_cast<float>(a[i]);
        float fb = static_cast<float>(b[i]);
        if (std::fabs(fa - fb) > atol) {
            return false;
        }
    }
    return true;
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------

int main(int argc, char** argv) {
    // Initialize CUDA
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "[ERROR] No CUDA devices found\n");
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));

    // Check SM version
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    int sm_version = props.major * 100 + props.minor * 10;

    printf("[INFO] Running on %s (SM %d)\n\n", props.name, sm_version);

    if (sm_version < 80) {
        printf("[WARN] SM < 80 detected. cp.async tests will use fallback path.\n\n");
    }

    // Run benchmarks
    MemoryBenchmark bench;
    bench.run_all();

    printf("\n[DONE] Microbenchmark complete.\n");
    return 0;
}
