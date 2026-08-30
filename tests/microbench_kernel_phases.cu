/******************************************************************************
 * Microbenchmark: Kernel Phase Timing Breakdown
 *
 * This benchmark isolates each phase of the backward pass to identify bottlenecks:
 * 1. Memory loads (Q, K, V, dO, O, LSE)
 * 2. compute_delta (O dot dO)
 * 3. compute_qk_scores (Q @ K^T via WMMA)
 * 4. recompute_softmax (exp(scores - LSE))
 * 5. compute_dp (dO @ V^T via WMMA)
 * 6. compute_dscores ((dP - delta) * P * scale)
 * 7. compute_dq (dScores @ K via WMMA)
 * 8. compute_dk (dScores^T @ Q via WMMA)
 * 9. compute_dv (P^T @ dO via WMMA)
 * 10. Memory stores (dQ, atomic dK/dV)
 *
 * Compile: nvcc -arch=sm_120 -O3 -std=c++17 microbench_kernel_phases.cu -o microbench_kernel_phases
 *****************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Configuration matching the optimized kernel
constexpr int BWD_BLOCK_M = 32;
constexpr int BWD_BLOCK_N = 32;
constexpr int BWD_BLOCK_D = 128;
constexpr int BWD_NUM_WARPS = 8;
constexpr int BWD_NUM_THREADS = BWD_NUM_WARPS * 32;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int WARMUP_ITERS = 5;
constexpr int BENCH_ITERS = 50;

#define CUDA_CHECK(stmt) do { \
    cudaError_t err = (stmt); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s: %s\n", #stmt, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Shared memory layout (simplified)
struct SmemLayout {
    __nv_bfloat16 q_tile[BWD_BLOCK_M * BWD_BLOCK_D];
    __nv_bfloat16 k_tile[BWD_BLOCK_N * BWD_BLOCK_D];
    __nv_bfloat16 v_tile[BWD_BLOCK_N * BWD_BLOCK_D];
    __nv_bfloat16 do_tile[BWD_BLOCK_M * BWD_BLOCK_D];
    __nv_bfloat16 o_tile[BWD_BLOCK_M * BWD_BLOCK_D];
    float scores[BWD_BLOCK_M * BWD_BLOCK_N];
    float probs[BWD_BLOCK_M * BWD_BLOCK_N];
    float dscores[BWD_BLOCK_M * BWD_BLOCK_N];
    float lse[BWD_BLOCK_M];
    float delta[BWD_BLOCK_M];
    float dq_acc[BWD_BLOCK_M * BWD_BLOCK_D];
    float wmma_staging[BWD_NUM_WARPS * WMMA_M * WMMA_N];
    __nv_bfloat16 temp_bf16[BWD_BLOCK_M * BWD_BLOCK_N];
};

// Warp reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Phase 1: Memory load simulation (coalesced loads)
__global__ void __launch_bounds__(BWD_NUM_THREADS)
kernel_phase_memory_load(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ d_o,
    const __nv_bfloat16* __restrict__ o,
    const float* __restrict__ lse,
    int num_heads, int head_dim
) {
    extern __shared__ char smem_raw[];
    SmemLayout* smem = reinterpret_cast<SmemLayout*>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Load Q tile [M, D]
    for (int m = warp_id; m < BWD_BLOCK_M; m += BWD_NUM_WARPS) {
        for (int d = lane_id * 2; d < head_dim; d += 64) {
            __nv_bfloat162 val = *reinterpret_cast<const __nv_bfloat162*>(q + m * head_dim + d);
            *reinterpret_cast<__nv_bfloat162*>(&smem->q_tile[m * BWD_BLOCK_D + d]) = val;
        }
    }

    // Load K tile [N, D]
    for (int n = warp_id; n < BWD_BLOCK_N; n += BWD_NUM_WARPS) {
        for (int d = lane_id * 2; d < head_dim; d += 64) {
            __nv_bfloat162 val = *reinterpret_cast<const __nv_bfloat162*>(k + n * head_dim + d);
            *reinterpret_cast<__nv_bfloat162*>(&smem->k_tile[n * BWD_BLOCK_D + d]) = val;
        }
    }

    // Load V tile [N, D]
    for (int n = warp_id; n < BWD_BLOCK_N; n += BWD_NUM_WARPS) {
        for (int d = lane_id * 2; d < head_dim; d += 64) {
            __nv_bfloat162 val = *reinterpret_cast<const __nv_bfloat162*>(v + n * head_dim + d);
            *reinterpret_cast<__nv_bfloat162*>(&smem->v_tile[n * BWD_BLOCK_D + d]) = val;
        }
    }

    // Load dO and O tiles [M, D]
    for (int m = warp_id; m < BWD_BLOCK_M; m += BWD_NUM_WARPS) {
        for (int d = lane_id * 2; d < head_dim; d += 64) {
            __nv_bfloat162 do_val = *reinterpret_cast<const __nv_bfloat162*>(d_o + m * head_dim + d);
            __nv_bfloat162 o_val = *reinterpret_cast<const __nv_bfloat162*>(o + m * head_dim + d);
            *reinterpret_cast<__nv_bfloat162*>(&smem->do_tile[m * BWD_BLOCK_D + d]) = do_val;
            *reinterpret_cast<__nv_bfloat162*>(&smem->o_tile[m * BWD_BLOCK_D + d]) = o_val;
        }
    }

    // Load LSE [M]
    for (int m = tid; m < BWD_BLOCK_M; m += BWD_NUM_THREADS) {
        smem->lse[m] = lse[m];
    }

    __syncthreads();

    // Dummy read to prevent optimization
    if (tid == 0) {
        volatile float dummy = __bfloat162float(smem->q_tile[0]) + __bfloat162float(smem->k_tile[0]);
        (void)dummy;
    }
}

// Phase 2: compute_delta (O dot dO)
__global__ void __launch_bounds__(BWD_NUM_THREADS)
kernel_phase_compute_delta(int m_size, int head_dim) {
    extern __shared__ char smem_raw[];
    SmemLayout* smem = reinterpret_cast<SmemLayout*>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Initialize O and dO with dummy data
    for (int i = tid; i < BWD_BLOCK_M * BWD_BLOCK_D; i += BWD_NUM_THREADS) {
        smem->o_tile[i] = __float2bfloat16(0.1f);
        smem->do_tile[i] = __float2bfloat16(0.1f);
    }
    __syncthreads();

    // Compute delta with warp-level reduction
    for (int m = warp_id; m < m_size; m += BWD_NUM_WARPS) {
        float sum = 0.0f;
        const __nv_bfloat16* o_row = &smem->o_tile[m * BWD_BLOCK_D];
        const __nv_bfloat16* do_row = &smem->do_tile[m * BWD_BLOCK_D];

        #pragma unroll
        for (int d_base = lane_id * 2; d_base < head_dim; d_base += 64) {
            __nv_bfloat162 o_val2 = *reinterpret_cast<const __nv_bfloat162*>(o_row + d_base);
            __nv_bfloat162 do_val2 = *reinterpret_cast<const __nv_bfloat162*>(do_row + d_base);
            float o_lo = __bfloat162float(__low2bfloat16(o_val2));
            float o_hi = __bfloat162float(__high2bfloat16(o_val2));
            float do_lo = __bfloat162float(__low2bfloat16(do_val2));
            float do_hi = __bfloat162float(__high2bfloat16(do_val2));
            sum += o_lo * do_lo + o_hi * do_hi;
        }

        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            smem->delta[m] = sum;
        }
    }
    __syncthreads();
}

// Phase 3: QK scores via WMMA
__global__ void __launch_bounds__(BWD_NUM_THREADS)
kernel_phase_qk_scores(int m_size, int n_size, int head_dim, float scale) {
    using namespace nvcuda::wmma;
    extern __shared__ char smem_raw[];
    SmemLayout* smem = reinterpret_cast<SmemLayout*>(smem_raw);

    const int warp_id = threadIdx.x / 32;
    const int tid = threadIdx.x;

    // Initialize Q and K
    for (int i = tid; i < BWD_BLOCK_M * BWD_BLOCK_D; i += BWD_NUM_THREADS) {
        smem->q_tile[i] = __float2bfloat16(0.1f);
    }
    for (int i = tid; i < BWD_BLOCK_N * BWD_BLOCK_D; i += BWD_NUM_THREADS) {
        smem->k_tile[i] = __float2bfloat16(0.1f);
    }
    __syncthreads();

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

            load_matrix_sync(a_frag, &smem->q_tile[m_tile * WMMA_M * BWD_BLOCK_D + k_tile * WMMA_K], BWD_BLOCK_D);
            load_matrix_sync(b_frag, &smem->k_tile[n_tile * WMMA_N * BWD_BLOCK_D + k_tile * WMMA_K], BWD_BLOCK_D);

            mma_sync(acc, a_frag, b_frag, acc);
        }

        store_matrix_sync(&smem->scores[m_tile * WMMA_M * BWD_BLOCK_N + n_tile * WMMA_N], acc, BWD_BLOCK_N, mem_row_major);
    }
    __syncthreads();

    // Apply scale
    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        smem->scores[idx] *= scale;
    }
    __syncthreads();
}

// Phase 4: Softmax recomputation
__global__ void __launch_bounds__(BWD_NUM_THREADS)
kernel_phase_softmax(int m_size, int n_size) {
    extern __shared__ char smem_raw[];
    SmemLayout* smem = reinterpret_cast<SmemLayout*>(smem_raw);

    const int tid = threadIdx.x;

    // Initialize scores and LSE
    for (int i = tid; i < m_size * n_size; i += BWD_NUM_THREADS) {
        smem->scores[i] = 1.0f;
    }
    for (int i = tid; i < m_size; i += BWD_NUM_THREADS) {
        smem->lse[i] = 2.0f;
    }
    __syncthreads();

    // Recompute softmax
    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        int m = idx / n_size;
        float score = smem->scores[idx];
        float lse_val = smem->lse[m];
        smem->probs[idx] = expf(score - lse_val);
    }
    __syncthreads();
}

// Phase 5: dP computation (dO @ V^T via WMMA)
__global__ void __launch_bounds__(BWD_NUM_THREADS)
kernel_phase_dp(int m_size, int n_size, int head_dim) {
    using namespace nvcuda::wmma;
    extern __shared__ char smem_raw[];
    SmemLayout* smem = reinterpret_cast<SmemLayout*>(smem_raw);

    const int warp_id = threadIdx.x / 32;
    const int tid = threadIdx.x;

    // Initialize dO and V
    for (int i = tid; i < BWD_BLOCK_M * BWD_BLOCK_D; i += BWD_NUM_THREADS) {
        smem->do_tile[i] = __float2bfloat16(0.1f);
    }
    for (int i = tid; i < BWD_BLOCK_N * BWD_BLOCK_D; i += BWD_NUM_THREADS) {
        smem->v_tile[i] = __float2bfloat16(0.1f);
    }
    __syncthreads();

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

            load_matrix_sync(do_frag, &smem->do_tile[m_tile * WMMA_M * BWD_BLOCK_D + k_tile * WMMA_K], BWD_BLOCK_D);
            load_matrix_sync(v_frag, &smem->v_tile[n_tile * WMMA_N * BWD_BLOCK_D + k_tile * WMMA_K], BWD_BLOCK_D);

            mma_sync(acc, do_frag, v_frag, acc);
        }

        store_matrix_sync(&smem->dscores[m_tile * WMMA_M * BWD_BLOCK_N + n_tile * WMMA_N], acc, BWD_BLOCK_N, mem_row_major);
    }
    __syncthreads();
}

// Phase 6: dScores computation ((dP - delta) * P * scale)
__global__ void __launch_bounds__(BWD_NUM_THREADS)
kernel_phase_dscores(int m_size, int n_size, float scale) {
    extern __shared__ char smem_raw[];
    SmemLayout* smem = reinterpret_cast<SmemLayout*>(smem_raw);

    const int tid = threadIdx.x;

    // Initialize dP (in dscores), probs, delta
    for (int i = tid; i < m_size * n_size; i += BWD_NUM_THREADS) {
        smem->dscores[i] = 0.5f;  // dP
        smem->probs[i] = 0.3f;
    }
    for (int i = tid; i < m_size; i += BWD_NUM_THREADS) {
        smem->delta[i] = 0.1f;
    }
    __syncthreads();

    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        int m = idx / n_size;
        float dp = smem->dscores[idx];
        float p = smem->probs[idx];
        float delta_m = smem->delta[m];
        smem->dscores[idx] = (dp - delta_m) * p * scale;
    }
    __syncthreads();
}

// Phase 7: dQ computation (dScores @ K via WMMA)
__global__ void __launch_bounds__(BWD_NUM_THREADS)
kernel_phase_dq(int m_size, int n_size, int head_dim) {
    using namespace nvcuda::wmma;
    extern __shared__ char smem_raw[];
    SmemLayout* smem = reinterpret_cast<SmemLayout*>(smem_raw);

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid = threadIdx.x;

    // Initialize dScores and K
    for (int i = tid; i < m_size * n_size; i += BWD_NUM_THREADS) {
        smem->temp_bf16[i] = __float2bfloat16(0.1f);
    }
    for (int i = tid; i < BWD_BLOCK_N * BWD_BLOCK_D; i += BWD_NUM_THREADS) {
        smem->k_tile[i] = __float2bfloat16(0.1f);
    }
    for (int i = tid; i < BWD_BLOCK_M * BWD_BLOCK_D; i += BWD_NUM_THREADS) {
        smem->dq_acc[i] = 0.0f;
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

            load_matrix_sync(ds_frag, &smem->temp_bf16[m_tile * WMMA_M * BWD_BLOCK_N + k_tile * WMMA_K], BWD_BLOCK_N);
            load_matrix_sync(k_frag, &smem->k_tile[k_tile * WMMA_K * BWD_BLOCK_D + d_tile * WMMA_N], BWD_BLOCK_D);

            mma_sync(acc, ds_frag, k_frag, acc);
        }

        float* staging = &smem->wmma_staging[warp_id * WMMA_M * WMMA_N];
        store_matrix_sync(staging, acc, WMMA_N, mem_row_major);
        __syncwarp();

        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int row = i / WMMA_N;
            int col = i % WMMA_N;
            int global_row = m_tile * WMMA_M + row;
            int global_col = d_tile * WMMA_N + col;
            if (global_row < m_size && global_col < head_dim) {
                smem->dq_acc[global_row * BWD_BLOCK_D + global_col] += staging[i];
            }
        }
        __syncwarp();
    }
    __syncthreads();
}

// Phase 8: dK computation (dScores^T @ Q via WMMA with transpose)
__global__ void __launch_bounds__(BWD_NUM_THREADS)
kernel_phase_dk(int m_size, int n_size, int head_dim, float* dk_out) {
    using namespace nvcuda::wmma;
    extern __shared__ char smem_raw[];
    SmemLayout* smem = reinterpret_cast<SmemLayout*>(smem_raw);

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid = threadIdx.x;

    // Initialize dScores [M, N] and Q [M, D]
    for (int i = tid; i < m_size * n_size; i += BWD_NUM_THREADS) {
        smem->dscores[i] = 0.1f;
    }
    for (int i = tid; i < BWD_BLOCK_M * BWD_BLOCK_D; i += BWD_NUM_THREADS) {
        smem->q_tile[i] = __float2bfloat16(0.1f);
    }
    __syncthreads();

    // Transpose dScores [M, N] -> temp_bf16 [N, M]
    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        int m = idx / n_size;
        int n = idx % n_size;
        smem->temp_bf16[n * BWD_BLOCK_M + m] = __float2bfloat16(smem->dscores[m * BWD_BLOCK_N + n]);
    }
    __syncthreads();

    // WMMA: dK[N, D] = dScores_T[N, M] @ Q[M, D]
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

            load_matrix_sync(a_frag, &smem->temp_bf16[n_tile * WMMA_M * BWD_BLOCK_M + k_tile * WMMA_K], BWD_BLOCK_M);
            load_matrix_sync(b_frag, &smem->q_tile[k_tile * WMMA_K * BWD_BLOCK_D + d_tile * WMMA_N], BWD_BLOCK_D);

            mma_sync(acc, a_frag, b_frag, acc);
        }

        float* staging = &smem->wmma_staging[warp_id * WMMA_M * WMMA_N];
        store_matrix_sync(staging, acc, WMMA_N, mem_row_major);
        __syncwarp();

        // Atomic add to global
        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int local_n = i / WMMA_N;
            int local_d = i % WMMA_N;
            int global_n = n_tile * WMMA_M + local_n;
            int global_d = d_tile * WMMA_N + local_d;
            if (global_n < n_size && global_d < head_dim) {
                float val = staging[i];
                if (val != 0.0f) {
                    atomicAdd(&dk_out[global_n * head_dim + global_d], val);
                }
            }
        }
        __syncwarp();
    }
    __syncthreads();
}

// Phase 9: dV computation (P^T @ dO via WMMA with transpose)
__global__ void __launch_bounds__(BWD_NUM_THREADS)
kernel_phase_dv(int m_size, int n_size, int head_dim, float* dv_out) {
    using namespace nvcuda::wmma;
    extern __shared__ char smem_raw[];
    SmemLayout* smem = reinterpret_cast<SmemLayout*>(smem_raw);

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid = threadIdx.x;

    // Initialize P [M, N] and dO [M, D]
    for (int i = tid; i < m_size * n_size; i += BWD_NUM_THREADS) {
        smem->probs[i] = 0.1f;
    }
    for (int i = tid; i < BWD_BLOCK_M * BWD_BLOCK_D; i += BWD_NUM_THREADS) {
        smem->do_tile[i] = __float2bfloat16(0.1f);
    }
    __syncthreads();

    // Transpose P [M, N] -> temp_bf16 [N, M]
    for (int idx = tid; idx < m_size * n_size; idx += BWD_NUM_THREADS) {
        int m = idx / n_size;
        int n = idx % n_size;
        smem->temp_bf16[n * BWD_BLOCK_M + m] = __float2bfloat16(smem->probs[m * BWD_BLOCK_N + n]);
    }
    __syncthreads();

    // WMMA: dV[N, D] = P_T[N, M] @ dO[M, D]
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

            load_matrix_sync(a_frag, &smem->temp_bf16[n_tile * WMMA_M * BWD_BLOCK_M + k_tile * WMMA_K], BWD_BLOCK_M);
            load_matrix_sync(b_frag, &smem->do_tile[k_tile * WMMA_K * BWD_BLOCK_D + d_tile * WMMA_N], BWD_BLOCK_D);

            mma_sync(acc, a_frag, b_frag, acc);
        }

        float* staging = &smem->wmma_staging[warp_id * WMMA_M * WMMA_N];
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
                    atomicAdd(&dv_out[global_n * head_dim + global_d], val);
                }
            }
        }
        __syncwarp();
    }
    __syncthreads();
}

// Phase 10: Memory store (dQ write, simulated)
__global__ void __launch_bounds__(BWD_NUM_THREADS)
kernel_phase_memory_store(
    __nv_bfloat16* __restrict__ dq_out,
    int m_size, int head_dim
) {
    extern __shared__ char smem_raw[];
    SmemLayout* smem = reinterpret_cast<SmemLayout*>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Initialize dq_acc
    for (int i = tid; i < BWD_BLOCK_M * BWD_BLOCK_D; i += BWD_NUM_THREADS) {
        smem->dq_acc[i] = 0.5f;
    }
    __syncthreads();

    // Write dQ to global
    for (int m = warp_id; m < m_size; m += BWD_NUM_WARPS) {
        for (int d = lane_id * 2; d < head_dim; d += 64) {
            __nv_bfloat162 val = __floats2bfloat162_rn(
                smem->dq_acc[m * BWD_BLOCK_D + d],
                smem->dq_acc[m * BWD_BLOCK_D + d + 1]
            );
            *reinterpret_cast<__nv_bfloat162*>(dq_out + m * head_dim + d) = val;
        }
    }
    __syncthreads();
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

int main() {
    printf("================================================================\n");
    printf("Kernel Phase Timing Breakdown Benchmark\n");
    printf("================================================================\n");
    printf("Config: M=%d, N=%d, D=%d, Warps=%d, Threads=%d\n",
           BWD_BLOCK_M, BWD_BLOCK_N, BWD_BLOCK_D, BWD_NUM_WARPS, BWD_NUM_THREADS);
    printf("================================================================\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    const int m_size = BWD_BLOCK_M;
    const int n_size = BWD_BLOCK_N;
    const int head_dim = BWD_BLOCK_D;
    const float scale = 1.0f / sqrtf((float)head_dim);

    // Allocate memory
    size_t smem_size = sizeof(SmemLayout);
    printf("Shared memory size: %.2f KB\n\n", smem_size / 1024.0f);

    __nv_bfloat16 *d_q, *d_k, *d_v, *d_do, *d_o, *d_dq;
    float *d_lse, *d_dk, *d_dv;

    CUDA_CHECK(cudaMalloc(&d_q, m_size * head_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, n_size * head_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, n_size * head_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_do, m_size * head_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_o, m_size * head_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_lse, m_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dq, m_size * head_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_dk, n_size * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dv, n_size * head_dim * sizeof(float)));

    GpuTimer timer;

    printf("%-35s %12s %12s %12s\n", "Phase", "Time(us)", "Pct(%)", "Notes");
    printf("----------------------------------------------------------------\n");

    float total_time = 0.0f;
    float phase_times[10];
    const char* phase_names[10] = {
        "1. Memory Load (Q,K,V,dO,O,LSE)",
        "2. compute_delta (O dot dO)",
        "3. QK scores (Q @ K^T WMMA)",
        "4. Softmax recompute (exp)",
        "5. dP compute (dO @ V^T WMMA)",
        "6. dScores ((dP-delta)*P*s)",
        "7. dQ compute (dS @ K WMMA)",
        "8. dK compute (dS^T @ Q WMMA)",
        "9. dV compute (P^T @ dO WMMA)",
        "10. Memory Store (dQ write)"
    };

    // Benchmark each phase
    auto bench_phase = [&](int phase_idx, auto kernel_fn) {
        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            kernel_fn();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.begin();
        for (int i = 0; i < BENCH_ITERS; ++i) {
            kernel_fn();
        }
        float ms = timer.end_ms();
        phase_times[phase_idx] = ms / BENCH_ITERS * 1000.0f;  // us
        total_time += phase_times[phase_idx];
    };

    // Phase 1: Memory Load
    bench_phase(0, [&]() {
        kernel_phase_memory_load<<<1, BWD_NUM_THREADS, smem_size>>>(
            d_q, d_k, d_v, d_do, d_o, d_lse, 1, head_dim);
    });

    // Phase 2: compute_delta
    bench_phase(1, [&]() {
        kernel_phase_compute_delta<<<1, BWD_NUM_THREADS, smem_size>>>(m_size, head_dim);
    });

    // Phase 3: QK scores
    bench_phase(2, [&]() {
        kernel_phase_qk_scores<<<1, BWD_NUM_THREADS, smem_size>>>(m_size, n_size, head_dim, scale);
    });

    // Phase 4: Softmax
    bench_phase(3, [&]() {
        kernel_phase_softmax<<<1, BWD_NUM_THREADS, smem_size>>>(m_size, n_size);
    });

    // Phase 5: dP
    bench_phase(4, [&]() {
        kernel_phase_dp<<<1, BWD_NUM_THREADS, smem_size>>>(m_size, n_size, head_dim);
    });

    // Phase 6: dScores
    bench_phase(5, [&]() {
        kernel_phase_dscores<<<1, BWD_NUM_THREADS, smem_size>>>(m_size, n_size, scale);
    });

    // Phase 7: dQ
    bench_phase(6, [&]() {
        kernel_phase_dq<<<1, BWD_NUM_THREADS, smem_size>>>(m_size, n_size, head_dim);
    });

    // Phase 8: dK
    bench_phase(7, [&]() {
        CUDA_CHECK(cudaMemset(d_dk, 0, n_size * head_dim * sizeof(float)));
        kernel_phase_dk<<<1, BWD_NUM_THREADS, smem_size>>>(m_size, n_size, head_dim, d_dk);
    });

    // Phase 9: dV
    bench_phase(8, [&]() {
        CUDA_CHECK(cudaMemset(d_dv, 0, n_size * head_dim * sizeof(float)));
        kernel_phase_dv<<<1, BWD_NUM_THREADS, smem_size>>>(m_size, n_size, head_dim, d_dv);
    });

    // Phase 10: Memory Store
    bench_phase(9, [&]() {
        kernel_phase_memory_store<<<1, BWD_NUM_THREADS, smem_size>>>(d_dq, m_size, head_dim);
    });

    // Print results
    for (int i = 0; i < 10; ++i) {
        float pct = (phase_times[i] / total_time) * 100.0f;
        const char* note = "";
        if (pct > 20.0f) note = "** BOTTLENECK **";
        else if (pct > 10.0f) note = "* significant *";
        printf("%-35s %12.2f %12.1f %s\n", phase_names[i], phase_times[i], pct, note);
    }

    printf("----------------------------------------------------------------\n");
    printf("%-35s %12.2f %12.1f\n", "TOTAL (single tile)", total_time, 100.0f);
    printf("================================================================\n");

    // Analysis
    printf("\nAnalysis:\n");
    float wmma_time = phase_times[2] + phase_times[4] + phase_times[6] + phase_times[7] + phase_times[8];
    float memory_time = phase_times[0] + phase_times[9];
    float compute_time = phase_times[1] + phase_times[3] + phase_times[5];
    float atomic_time = phase_times[7] + phase_times[8];  // dK and dV include atomics

    printf("  WMMA operations:    %.2f us (%.1f%%)\n", wmma_time, wmma_time/total_time*100);
    printf("  Memory ops:         %.2f us (%.1f%%)\n", memory_time, memory_time/total_time*100);
    printf("  Scalar compute:     %.2f us (%.1f%%)\n", compute_time, compute_time/total_time*100);
    printf("  dK/dV (with atomic):%.2f us (%.1f%%)\n", atomic_time, atomic_time/total_time*100);

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

    printf("\n[DONE]\n");
    return 0;
}
