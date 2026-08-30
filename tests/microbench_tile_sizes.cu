/******************************************************************************
 * Microbenchmark: Tile Size Exploration
 *
 * Tests different M/N tile sizes to find optimal configuration for SM120.
 * Constraints:
 * - Total smem < 99KB
 * - WMMA requires 16x16 tiles
 * - M, N must be multiples of 16
 *
 * Configurations tested:
 * - M=16, N=32, 4 warps (original)
 * - M=32, N=32, 8 warps (current)
 * - M=32, N=64, 8 warps (if fits)
 * - M=64, N=32, 8 warps
 * - M=64, N=64, 8 warps (if fits)
 *
 * Compile: nvcc -arch=sm_120 -O3 -std=c++17 microbench_tile_sizes.cu -o microbench_tile_sizes
 *****************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
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

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int HEAD_DIM = 128;

constexpr int WARMUP_ITERS = 5;
constexpr int BENCH_ITERS = 50;

// Calculate shared memory size for a given configuration
// Double-buffered K/V tiles
__host__ size_t calc_smem_size(int M, int N, int D, int num_warps) {
    size_t q_tile = M * D * sizeof(__nv_bfloat16);
    size_t k_tile = 2 * N * D * sizeof(__nv_bfloat16);  // double buffered
    size_t v_tile = 2 * N * D * sizeof(__nv_bfloat16);  // double buffered
    size_t do_tile = M * D * sizeof(__nv_bfloat16);
    size_t o_tile = M * D * sizeof(__nv_bfloat16);
    size_t scores = M * N * sizeof(float);
    size_t probs = M * N * sizeof(float);
    size_t dscores = M * N * sizeof(float);
    size_t lse = M * sizeof(float);
    size_t delta = M * sizeof(float);
    size_t dq_acc = M * D * sizeof(float);
    size_t wmma_staging = num_warps * 16 * 16 * sizeof(float);
    size_t temp_bf16 = M * N * sizeof(__nv_bfloat16);

    return q_tile + k_tile + v_tile + do_tile + o_tile +
           scores + probs + dscores + lse + delta +
           dq_acc + wmma_staging + temp_bf16;
}

// Template kernel for different tile sizes
template<int BLOCK_M, int BLOCK_N, int BLOCK_D, int NUM_WARPS>
__global__ void __launch_bounds__(NUM_WARPS * 32)
kernel_full_backward_tile(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ d_o,
    const __nv_bfloat16* __restrict__ o,
    const float* __restrict__ lse,
    __nv_bfloat16* __restrict__ dq,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int seq_len_q,
    int seq_len_kv,
    float scale
) {
    using namespace nvcuda::wmma;

    constexpr int NUM_THREADS = NUM_WARPS * 32;

    extern __shared__ char smem_raw[];

    // Simplified layout - just enough to benchmark compute patterns
    __nv_bfloat16* s_q = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* s_k = s_q + BLOCK_M * BLOCK_D;
    __nv_bfloat16* s_v = s_k + BLOCK_N * BLOCK_D;
    __nv_bfloat16* s_do = s_v + BLOCK_N * BLOCK_D;
    __nv_bfloat16* s_o = s_do + BLOCK_M * BLOCK_D;
    float* s_scores = reinterpret_cast<float*>(s_o + BLOCK_M * BLOCK_D);
    float* s_probs = s_scores + BLOCK_M * BLOCK_N;
    float* s_dscores = s_probs + BLOCK_M * BLOCK_N;
    float* s_lse = s_dscores + BLOCK_M * BLOCK_N;
    float* s_delta = s_lse + BLOCK_M;
    float* s_dq_acc = s_delta + BLOCK_M;
    float* s_staging = s_dq_acc + BLOCK_M * BLOCK_D;
    __nv_bfloat16* s_temp = reinterpret_cast<__nv_bfloat16*>(s_staging + NUM_WARPS * 16 * 16);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int m_block = blockIdx.x;
    const int m_start = m_block * BLOCK_M;
    const int m_size = min(BLOCK_M, seq_len_q - m_start);

    if (m_start >= seq_len_q) return;

    // Load Q, dO, O tiles
    for (int m = warp_id; m < m_size; m += NUM_WARPS) {
        for (int d = lane_id * 2; d < BLOCK_D; d += 64) {
            int idx = (m_start + m) * BLOCK_D + d;
            *reinterpret_cast<__nv_bfloat162*>(&s_q[m * BLOCK_D + d]) =
                *reinterpret_cast<const __nv_bfloat162*>(&q[idx]);
            *reinterpret_cast<__nv_bfloat162*>(&s_do[m * BLOCK_D + d]) =
                *reinterpret_cast<const __nv_bfloat162*>(&d_o[idx]);
            *reinterpret_cast<__nv_bfloat162*>(&s_o[m * BLOCK_D + d]) =
                *reinterpret_cast<const __nv_bfloat162*>(&o[idx]);
        }
    }

    // Load LSE
    for (int m = tid; m < m_size; m += NUM_THREADS) {
        s_lse[m] = lse[m_start + m];
    }

    // Initialize dq_acc
    for (int i = tid; i < BLOCK_M * BLOCK_D; i += NUM_THREADS) {
        s_dq_acc[i] = 0.0f;
    }
    __syncthreads();

    // Compute delta = O dot dO
    for (int m = warp_id; m < m_size; m += NUM_WARPS) {
        float sum = 0.0f;
        for (int d = lane_id * 2; d < BLOCK_D; d += 64) {
            __nv_bfloat162 o_val = *reinterpret_cast<const __nv_bfloat162*>(&s_o[m * BLOCK_D + d]);
            __nv_bfloat162 do_val = *reinterpret_cast<const __nv_bfloat162*>(&s_do[m * BLOCK_D + d]);
            sum += __bfloat162float(__low2bfloat16(o_val)) * __bfloat162float(__low2bfloat16(do_val));
            sum += __bfloat162float(__high2bfloat16(o_val)) * __bfloat162float(__high2bfloat16(do_val));
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane_id == 0) s_delta[m] = sum;
    }
    __syncthreads();

    // Loop over KV blocks
    const int num_kv_blocks = (seq_len_kv + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int n_start = kv_block * BLOCK_N;
        const int n_size = min(BLOCK_N, seq_len_kv - n_start);

        // Load K, V tiles
        for (int n = warp_id; n < n_size; n += NUM_WARPS) {
            for (int d = lane_id * 2; d < BLOCK_D; d += 64) {
                int idx = (n_start + n) * BLOCK_D + d;
                *reinterpret_cast<__nv_bfloat162*>(&s_k[n * BLOCK_D + d]) =
                    *reinterpret_cast<const __nv_bfloat162*>(&k[idx]);
                *reinterpret_cast<__nv_bfloat162*>(&s_v[n * BLOCK_D + d]) =
                    *reinterpret_cast<const __nv_bfloat162*>(&v[idx]);
            }
        }
        __syncthreads();

        // QK scores via WMMA
        const int m_tiles = (m_size + WMMA_M - 1) / WMMA_M;
        const int n_tiles = (n_size + WMMA_N - 1) / WMMA_N;
        const int d_tiles = (BLOCK_D + WMMA_K - 1) / WMMA_K;

        for (int mn = warp_id; mn < m_tiles * n_tiles; mn += NUM_WARPS) {
            int m_tile = mn / n_tiles;
            int n_tile = mn % n_tiles;

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            fill_fragment(acc, 0.0f);

            for (int k_tile = 0; k_tile < d_tiles; ++k_tile) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;

                load_matrix_sync(a_frag, &s_q[m_tile * WMMA_M * BLOCK_D + k_tile * WMMA_K], BLOCK_D);
                load_matrix_sync(b_frag, &s_k[n_tile * WMMA_N * BLOCK_D + k_tile * WMMA_K], BLOCK_D);
                mma_sync(acc, a_frag, b_frag, acc);
            }

            store_matrix_sync(&s_scores[m_tile * WMMA_M * BLOCK_N + n_tile * WMMA_N], acc, BLOCK_N, mem_row_major);
        }
        __syncthreads();

        // Softmax and compute P
        for (int idx = tid; idx < m_size * n_size; idx += NUM_THREADS) {
            int m = idx / n_size;
            int n = idx % n_size;
            float score = s_scores[m * BLOCK_N + n] * scale;
            s_probs[m * BLOCK_N + n] = expf(score - s_lse[m]);
        }
        __syncthreads();

        // dP = dO @ V^T via WMMA
        for (int mn = warp_id; mn < m_tiles * n_tiles; mn += NUM_WARPS) {
            int m_tile = mn / n_tiles;
            int n_tile = mn % n_tiles;

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            fill_fragment(acc, 0.0f);

            for (int k_tile = 0; k_tile < d_tiles; ++k_tile) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;

                load_matrix_sync(a_frag, &s_do[m_tile * WMMA_M * BLOCK_D + k_tile * WMMA_K], BLOCK_D);
                load_matrix_sync(b_frag, &s_v[n_tile * WMMA_N * BLOCK_D + k_tile * WMMA_K], BLOCK_D);
                mma_sync(acc, a_frag, b_frag, acc);
            }

            store_matrix_sync(&s_dscores[m_tile * WMMA_M * BLOCK_N + n_tile * WMMA_N], acc, BLOCK_N, mem_row_major);
        }
        __syncthreads();

        // dScores = (dP - delta) * P * scale
        for (int idx = tid; idx < m_size * n_size; idx += NUM_THREADS) {
            int m = idx / n_size;
            float dp = s_dscores[m * BLOCK_N + idx % n_size];
            float p = s_probs[m * BLOCK_N + idx % n_size];
            s_dscores[m * BLOCK_N + idx % n_size] = (dp - s_delta[m]) * p * scale;
        }
        __syncthreads();

        // dQ = dScores @ K via WMMA
        // Convert dScores to bf16
        for (int i = tid; i < m_size * n_size; i += NUM_THREADS) {
            s_temp[i] = __float2bfloat16(s_dscores[i]);
        }
        __syncthreads();

        const int dq_d_tiles = (BLOCK_D + WMMA_N - 1) / WMMA_N;
        const int dq_n_tiles = (n_size + WMMA_K - 1) / WMMA_K;

        for (int md = warp_id; md < m_tiles * dq_d_tiles; md += NUM_WARPS) {
            int m_tile = md / dq_d_tiles;
            int d_tile = md % dq_d_tiles;

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            fill_fragment(acc, 0.0f);

            for (int k_tile = 0; k_tile < dq_n_tiles; ++k_tile) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag;

                load_matrix_sync(a_frag, &s_temp[m_tile * WMMA_M * BLOCK_N + k_tile * WMMA_K], BLOCK_N);
                load_matrix_sync(b_frag, &s_k[k_tile * WMMA_K * BLOCK_D + d_tile * WMMA_N], BLOCK_D);
                mma_sync(acc, a_frag, b_frag, acc);
            }

            float* staging = &s_staging[warp_id * 16 * 16];
            store_matrix_sync(staging, acc, WMMA_N, mem_row_major);
            __syncwarp();

            for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
                int row = i / WMMA_N;
                int col = i % WMMA_N;
                int global_row = m_tile * WMMA_M + row;
                int global_col = d_tile * WMMA_N + col;
                if (global_row < m_size && global_col < BLOCK_D) {
                    s_dq_acc[global_row * BLOCK_D + global_col] += staging[i];
                }
            }
            __syncwarp();
        }
        __syncthreads();

        // dK = dScores^T @ Q - transpose first
        for (int idx = tid; idx < m_size * n_size; idx += NUM_THREADS) {
            int m = idx / n_size;
            int n = idx % n_size;
            s_temp[n * BLOCK_M + m] = __float2bfloat16(s_dscores[m * BLOCK_N + n]);
        }
        __syncthreads();

        const int dk_n_tiles = (n_size + WMMA_M - 1) / WMMA_M;
        const int dk_d_tiles = (BLOCK_D + WMMA_N - 1) / WMMA_N;
        const int dk_m_tiles = (m_size + WMMA_K - 1) / WMMA_K;

        for (int nd = warp_id; nd < dk_n_tiles * dk_d_tiles; nd += NUM_WARPS) {
            int n_tile = nd / dk_d_tiles;
            int d_tile = nd % dk_d_tiles;

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            fill_fragment(acc, 0.0f);

            for (int k_tile = 0; k_tile < dk_m_tiles; ++k_tile) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag;

                load_matrix_sync(a_frag, &s_temp[n_tile * WMMA_M * BLOCK_M + k_tile * WMMA_K], BLOCK_M);
                load_matrix_sync(b_frag, &s_q[k_tile * WMMA_K * BLOCK_D + d_tile * WMMA_N], BLOCK_D);
                mma_sync(acc, a_frag, b_frag, acc);
            }

            float* staging = &s_staging[warp_id * 16 * 16];
            store_matrix_sync(staging, acc, WMMA_N, mem_row_major);
            __syncwarp();

            for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
                int local_n = i / WMMA_N;
                int local_d = i % WMMA_N;
                int global_n = n_tile * WMMA_M + local_n;
                int global_d = d_tile * WMMA_N + local_d;
                if (global_n < n_size && global_d < BLOCK_D) {
                    atomicAdd(&dk[(n_start + global_n) * BLOCK_D + global_d], staging[i]);
                }
            }
            __syncwarp();
        }
        __syncthreads();

        // dV = P^T @ dO - transpose P first
        for (int idx = tid; idx < m_size * n_size; idx += NUM_THREADS) {
            int m = idx / n_size;
            int n = idx % n_size;
            s_temp[n * BLOCK_M + m] = __float2bfloat16(s_probs[m * BLOCK_N + n]);
        }
        __syncthreads();

        for (int nd = warp_id; nd < dk_n_tiles * dk_d_tiles; nd += NUM_WARPS) {
            int n_tile = nd / dk_d_tiles;
            int d_tile = nd % dk_d_tiles;

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            fill_fragment(acc, 0.0f);

            for (int k_tile = 0; k_tile < dk_m_tiles; ++k_tile) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag;

                load_matrix_sync(a_frag, &s_temp[n_tile * WMMA_M * BLOCK_M + k_tile * WMMA_K], BLOCK_M);
                load_matrix_sync(b_frag, &s_do[k_tile * WMMA_K * BLOCK_D + d_tile * WMMA_N], BLOCK_D);
                mma_sync(acc, a_frag, b_frag, acc);
            }

            float* staging = &s_staging[warp_id * 16 * 16];
            store_matrix_sync(staging, acc, WMMA_N, mem_row_major);
            __syncwarp();

            for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
                int local_n = i / WMMA_N;
                int local_d = i % WMMA_N;
                int global_n = n_tile * WMMA_M + local_n;
                int global_d = d_tile * WMMA_N + local_d;
                if (global_n < n_size && global_d < BLOCK_D) {
                    atomicAdd(&dv[(n_start + global_n) * BLOCK_D + global_d], staging[i]);
                }
            }
            __syncwarp();
        }
        __syncthreads();
    }

    // Write dQ
    for (int m = warp_id; m < m_size; m += NUM_WARPS) {
        for (int d = lane_id * 2; d < BLOCK_D; d += 64) {
            __nv_bfloat162 val = __floats2bfloat162_rn(
                s_dq_acc[m * BLOCK_D + d],
                s_dq_acc[m * BLOCK_D + d + 1]);
            *reinterpret_cast<__nv_bfloat162*>(&dq[(m_start + m) * BLOCK_D + d]) = val;
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

template<int M, int N, int D, int WARPS>
void benchmark_config(const char* name, int seq_len,
                      __nv_bfloat16* d_q, __nv_bfloat16* d_k, __nv_bfloat16* d_v,
                      __nv_bfloat16* d_do, __nv_bfloat16* d_o, float* d_lse,
                      __nv_bfloat16* d_dq, float* d_dk, float* d_dv) {
    size_t smem_size = calc_smem_size(M, N, D, WARPS);

    if (smem_size > 99 * 1024) {
        printf("%-20s %5d %5d %5d %5d %10s %10s (smem %.1fKB > 99KB)\n",
               name, M, N, D, WARPS, "SKIP", "SKIP", smem_size / 1024.0f);
        return;
    }

    // Set shared memory size
    CUDA_CHECK(cudaFuncSetAttribute(
        kernel_full_backward_tile<M, N, D, WARPS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    const int m_blocks = (seq_len + M - 1) / M;
    dim3 grid(m_blocks);
    dim3 block(WARPS * 32);

    float scale = 1.0f / sqrtf((float)D);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_dk, 0, seq_len * D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_dv, 0, seq_len * D * sizeof(float)));
        kernel_full_backward_tile<M, N, D, WARPS><<<grid, block, smem_size>>>(
            d_q, d_k, d_v, d_do, d_o, d_lse, d_dq, d_dk, d_dv, seq_len, seq_len, scale);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.begin();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_dk, 0, seq_len * D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_dv, 0, seq_len * D * sizeof(float)));
        kernel_full_backward_tile<M, N, D, WARPS><<<grid, block, smem_size>>>(
            d_q, d_k, d_v, d_do, d_o, d_lse, d_dq, d_dk, d_dv, seq_len, seq_len, scale);
    }
    float total_ms = timer.end_ms();
    float avg_ms = total_ms / BENCH_ITERS;

    // Calculate TFLOPS
    // Forward: 4 * seq^2 * D (2 for QK, 2 for PV)
    // Backward: ~4x forward
    double flops = 16.0 * seq_len * seq_len * D;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    printf("%-20s %5d %5d %5d %5d %10.3f %10.2f (smem %.1fKB, grid %d)\n",
           name, M, N, D, WARPS, avg_ms, tflops, smem_size / 1024.0f, m_blocks);
}

int main() {
    printf("================================================================\n");
    printf("Tile Size Exploration Benchmark\n");
    printf("================================================================\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Max shared memory per block: %zu KB\n", prop.sharedMemPerBlockOptin / 1024);
    printf("================================================================\n\n");

    // Test multiple sequence lengths
    int seq_lens[] = {512, 1024, 2048};

    for (int seq_len : seq_lens) {
        printf("\n--- Sequence Length: %d ---\n", seq_len);
        printf("%-20s %5s %5s %5s %5s %10s %10s\n",
               "Config", "M", "N", "D", "Warps", "Time(ms)", "TFLOPS");
        printf("---------------------------------------------------------------\n");

        // Allocate memory
        size_t total_tokens = seq_len;
        __nv_bfloat16 *d_q, *d_k, *d_v, *d_do, *d_o, *d_dq;
        float *d_lse, *d_dk, *d_dv;

        CUDA_CHECK(cudaMalloc(&d_q, total_tokens * HEAD_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_k, total_tokens * HEAD_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_v, total_tokens * HEAD_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_do, total_tokens * HEAD_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_o, total_tokens * HEAD_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_lse, total_tokens * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dq, total_tokens * HEAD_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_dk, total_tokens * HEAD_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dv, total_tokens * HEAD_DIM * sizeof(float)));

        // Initialize with dummy data
        CUDA_CHECK(cudaMemset(d_q, 0, total_tokens * HEAD_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(d_k, 0, total_tokens * HEAD_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(d_v, 0, total_tokens * HEAD_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(d_do, 0, total_tokens * HEAD_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(d_o, 0, total_tokens * HEAD_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(d_lse, 0, total_tokens * sizeof(float)));

        // Test configurations
        benchmark_config<16, 32, 128, 4>("M16_N32_W4", seq_len, d_q, d_k, d_v, d_do, d_o, d_lse, d_dq, d_dk, d_dv);
        benchmark_config<32, 32, 128, 4>("M32_N32_W4", seq_len, d_q, d_k, d_v, d_do, d_o, d_lse, d_dq, d_dk, d_dv);
        benchmark_config<32, 32, 128, 8>("M32_N32_W8", seq_len, d_q, d_k, d_v, d_do, d_o, d_lse, d_dq, d_dk, d_dv);
        benchmark_config<64, 32, 128, 8>("M64_N32_W8", seq_len, d_q, d_k, d_v, d_do, d_o, d_lse, d_dq, d_dk, d_dv);
        benchmark_config<32, 64, 128, 8>("M32_N64_W8", seq_len, d_q, d_k, d_v, d_do, d_o, d_lse, d_dq, d_dk, d_dv);
        benchmark_config<64, 64, 128, 8>("M64_N64_W8", seq_len, d_q, d_k, d_v, d_do, d_o, d_lse, d_dq, d_dk, d_dv);
        benchmark_config<48, 48, 128, 6>("M48_N48_W6", seq_len, d_q, d_k, d_v, d_do, d_o, d_lse, d_dq, d_dk, d_dv);

        // Cleanup
        CUDA_CHECK(cudaFree(d_q));
        CUDA_CHECK(cudaFree(d_k));
        CUDA_CHECK(cudaFree(d_v));
        CUDA_CHECK(cudaFree(d_do));
        CUDA_CHECK(cudaFree(d_o));
        CUDA_CHECK(cudaFree(d_lse));
        CUDA_CHECK(cudaFree(d_dq));
        CUDA_CHECK(cudaFree(d_dk));
        CUDA_CHECK(cudaFree(d_dv));
    }

    printf("\n================================================================\n");
    printf("[DONE]\n");
    return 0;
}
