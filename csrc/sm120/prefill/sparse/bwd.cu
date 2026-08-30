/***************************************************************************************************
 * SM120 Sparse Prefill Backward Kernel - WMMA implementation (full rewrite)
 *
 * Faithful to the FlashAttention-2 backward formulation (arXiv:2307.08691) and to this repo's
 * sm120 sparse FORWARD structure (fwd.cu): one CTA owns ONE query position and a block of
 * B_H=64 query heads; the attention-matrix rows are HEADS, and the sparse indices are loaded
 * PER QUERY POSITION (indices differ per position -- the old kernel wrongly reused the first
 * query's indices for a whole 16-query block).
 *
 * Math (per query position, heads h, selected tokens t of this position's topk):
 *   S[h,t]  = Q[h,:] . K[t,:] * sm_scale                 (K = full 576-wide KV row)
 *   P[h,t]  = exp2(S[h,t]*log2e*... ) via stored LSE:    P = exp2(S2 - lse2),
 *             S2 = S_raw * sm_scale * LOG2E, lse2 = 2-based LSE from forward (GLOBAL over
 *             the full topk -- the old kernel recomputed a per-64-block LOCAL softmax,
 *             which is wrong for topk > 64)
 *   D[h]    = sum_v dO[h,v] * O[h,v]                     (FA-2 correction term; the old
 *             kernel never read params.o at all)
 *   dP[h,t] = dO[h,:] . V[t,:]                           (V = first 512 of the KV row)
 *   dS[h,t] = P[h,t] * (dP[h,t] - D[h]) * sm_scale
 *   dQ[h,:] += dS[h,:] @ K       (exclusive per (position, head) -> plain fp32 RMW, no atomics)
 *   dK[t,:] += dS[:,t]^T @ Q     (scattered by index -> fp32 atomicAdd)
 *   dV[t,:] += P[:,t]^T  @ dO    (scattered by index -> fp32 atomicAdd)
 *
 * All five GEMMs run on WMMA 16x16x16 bf16 tensor cores through one templated 64x64x64 tile
 * helper that only ever addresses shared memory through load/store_matrix_sync (NO manual
 * fragment-layout indexing -- the old kernel's lane_id==0 "simplification" wrote garbage).
 *
 * Shared memory: ~57KB < 99KB sm120 limit. Launch: grid = (h_q/64) * s_q, 256 threads.
 **************************************************************************************************/

#include "bwd.h"
#include "traits.h"
#include <cuda_bf16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <type_traits>
#include <cstdlib>   // getenv/atoi for the CFG ladder gate
#include <cstdio>    // fprintf/stderr used by CHECK_CUDA
#include "../../../utils.h"   // CHECK_CUDA / CHECK_CUDA_KERNEL_LAUNCH

// mma.sync tier (CFG=1): raw mma.sync + ldmatrix + cp.async backward.
#include "bwd_mma.cuh"

namespace sm120 {
namespace sparse_bwd {

using namespace nvcuda;
using bf16 = cutlass::bfloat16_t;

static constexpr int B_H = 64;         // heads per CTA (rows of the attention matrix)
static constexpr int B_TOPK = 64;      // topk tokens per iteration (cols)
static constexpr int TILE = 64;        // feature-dim tile (d_qk = 9 tiles, d_v = 8 tiles)
static constexpr int D_QK = 576;
static constexpr int D_V = 512;
static constexpr int NUM_THREADS = 256;
static constexpr int NUM_D_TILES = D_QK / TILE;   // 9
static constexpr int NUM_V_TILES = D_V / TILE;    // 8
static constexpr float LOG2E = 1.4426950408889634f;

struct SharedMemory {
    __align__(128) bf16  sQ[B_H * TILE];        // 8KB  Q d-tile (QK, dK phases)
    __align__(128) bf16  sKV[B_TOPK * TILE];    // 8KB  gathered K d-tile / V v-tile
    __align__(128) bf16  sDO[B_H * TILE];       // 8KB  dO v-tile
    __align__(128) float sAcc[B_H * B_TOPK];    // 16KB S accum -> dP accum -> WMMA staging
    __align__(128) bf16  sP[B_H * B_TOPK];      // 8KB  probabilities
    __align__(128) bf16  sDS[B_H * B_TOPK];     // 8KB  dS (bf16 operand for WMMA)
    __align__(128) float sD[B_H];               // D-term rowsum(dO*O)
    __align__(128) float sLse[B_H];             // 2-based LSE from forward
    __align__(128) int   sIdx[B_TOPK];
    __align__(128) bool  sValid[B_TOPK];
};
static_assert(sizeof(SharedMemory) <= 99 * 1024, "exceeds SM120 99KB smem limit");

//==============================================================================
// One 64x64x64 GEMM tile on WMMA: C[64,64] (+)= A op B with A/B layout selected
// by template (row_major = as stored, col_major = transposed view). All operands
// live in smem with leading dimension 64; C is fp32 in smem (ld 64).
// Warps 0-3 each own 16 output rows; 4 output col-tiles x 4 inner k-tiles.
// Addressing follows the PROVEN pattern of fwd.cu's wmma_gemm_qk_sparse.
//==============================================================================
template <typename LayoutA, typename LayoutB>
__device__ __forceinline__ void wmma_tile_gemm(
    const bf16* __restrict__ aPtr,   // [64,64] smem, ld 64
    const bf16* __restrict__ bPtr,   // [64,64] smem, ld 64
    float* __restrict__ cPtr,        // [64,64] smem fp32, ld 64
    int warp_idx,
    bool accumulate
) {
    if (warp_idx >= 4) return;
    const int warp_m = warp_idx * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, LayoutA> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, LayoutB> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    #pragma unroll
    for (int n_tile = 0; n_tile < 4; ++n_tile) {
        const int out_n = n_tile * 16;

        if (accumulate) {
            wmma::load_matrix_sync(c_frag, cPtr + warp_m * 64 + out_n, 64, wmma::mem_row_major);
        } else {
            wmma::fill_fragment(c_frag, 0.0f);
        }

        #pragma unroll
        for (int k_tile = 0; k_tile < 4; ++k_tile) {
            const int k_off = k_tile * 16;

            // A element (i=out_row, j=inner): row_major -> a[warp_m+i][k_off+j] at base+warp_m*64+k_off
            //                                 col_major -> a[k_off+j][warp_m+i] at base+warp_m+k_off*64
            const bf16* aBase = std::is_same<LayoutA, wmma::row_major>::value
                                    ? (aPtr + warp_m * 64 + k_off)
                                    : (aPtr + warp_m + k_off * 64);
            // B element (i=inner, j=out_col): row_major -> b[k_off+i][out_n+j] at base+k_off*64+out_n
            //                                 col_major -> b[out_n+j][k_off+i] at base+out_n*64+k_off
            const bf16* bBase = std::is_same<LayoutB, wmma::row_major>::value
                                    ? (bPtr + k_off * 64 + out_n)
                                    : (bPtr + out_n * 64 + k_off);

            wmma::load_matrix_sync(a_frag, reinterpret_cast<const __nv_bfloat16*>(aBase), 64);
            wmma::load_matrix_sync(b_frag, reinterpret_cast<const __nv_bfloat16*>(bBase), 64);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        wmma::store_matrix_sync(cPtr + warp_m * 64 + out_n, c_frag, 64, wmma::mem_row_major);
    }
}

//==============================================================================
// Cooperative smem tile loaders (256 threads, 4096 elems -> 16 per thread)
//==============================================================================

// Q or dO tile: rows are heads of this CTA's head-block at this query position.
__device__ __forceinline__ void load_rows_tile(
    const bf16* __restrict__ gptr,   // pre-offset to (s_q_idx, h_block*64, 0)
    bf16* __restrict__ smem,         // [64, TILE]
    int stride_h,                    // stride between heads
    int col_start,                   // feature-dim tile offset
    int col_limit                    // d_qk or d_v (zero-fill past it)
) {
    const int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < (B_H * TILE) / NUM_THREADS; ++i) {
        int idx = tid + i * NUM_THREADS;
        int row = idx / TILE;
        int col = idx % TILE;
        int gcol = col_start + col;
        smem[idx] = (gcol < col_limit) ? gptr[row * stride_h + gcol] : bf16(0.0f);
    }
}

// Gathered K/V tile: rows are the topk tokens of this block (invalid rows -> 0).
__device__ __forceinline__ void load_gather_tile(
    const bf16* __restrict__ kv_ptr,
    bf16* __restrict__ smem,          // [64, TILE]
    const int* __restrict__ sIdx,
    const bool* __restrict__ sValid,
    int stride_s_kv,
    int col_start,
    int col_limit                     // D_QK for K, D_V for V (V = FIRST d_v of the KV row)
) {
    const int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < (B_TOPK * TILE) / NUM_THREADS; ++i) {
        int idx = tid + i * NUM_THREADS;
        int row = idx / TILE;
        int col = idx % TILE;
        int gcol = col_start + col;
        bf16 val = bf16(0.0f);
        if (sValid[row] && gcol < col_limit) {
            val = kv_ptr[sIdx[row] * (long)stride_s_kv + gcol];
        }
        smem[idx] = val;
    }
}

//==============================================================================
// Main kernel: grid.x = (h_q / B_H) * s_q, one CTA per (head-block, query position)
//==============================================================================
__global__ void __launch_bounds__(NUM_THREADS, 1)
sparse_prefill_bwd_kernel(const SparsePrefillBwdParams params) {
    extern __shared__ char smem_raw[];
    SharedMemory& smem = *reinterpret_cast<SharedMemory*>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_idx = tid / 32;
    const int num_h_blocks = params.h_q / B_H;
    const int h_block = blockIdx.x % num_h_blocks;
    const int s_q_idx = blockIdx.x / num_h_blocks;

    // Global pointers pre-offset to (this position, this head block)
    const bf16* q_ptr  = params.q   + (long)s_q_idx * params.stride_q_s_q  + (long)h_block * B_H * params.stride_q_h_q;
    const bf16* do_ptr = params.d_o + (long)s_q_idx * params.stride_do_s_q + (long)h_block * B_H * params.stride_do_h_q;
    const bf16* o_ptr  = params.o   + (long)s_q_idx * params.stride_o_s_q  + (long)h_block * B_H * params.stride_o_h_q;
    const int* idx_ptr = params.indices + (long)s_q_idx * params.stride_indices_s_q;   // h_kv == 1
    float* dq_ptr = params.dq + (long)s_q_idx * params.stride_dq_s_q + (long)h_block * B_H * params.stride_dq_h_q;

    //--------------------------------------------------------------------------
    // Once per CTA: LSE (2-based, global over the full topk) and the FA-2
    // D-term D[h] = sum_v dO[h,v] * O[h,v], accumulated in fp32.
    //--------------------------------------------------------------------------
    if (tid < B_H) {
        smem.sLse[tid] = params.lse[(long)s_q_idx * params.h_q + h_block * B_H + tid];
        smem.sD[tid] = 0.0f;
    }
    __syncthreads();

    {
        // 4 threads per head row, each strides the 512-wide V dim by 4.
        const int row = tid / 4;
        const int sub = tid % 4;
        float part = 0.0f;
        for (int v = sub; v < params.d_v; v += 4) {
            part += float(do_ptr[row * params.stride_do_h_q + v]) * float(o_ptr[row * params.stride_o_h_q + v]);
        }
        atomicAdd(&smem.sD[row], part);
    }
    __syncthreads();

    //--------------------------------------------------------------------------
    // Loop over topk blocks of 64 tokens (each with its own gathered K/V)
    //--------------------------------------------------------------------------
    const int num_topk_blocks = (params.topk + B_TOPK - 1) / B_TOPK;

    for (int tb = 0; tb < num_topk_blocks; ++tb) {
        // Load this block's indices + validity (ragged tail -> invalid)
        if (tid < B_TOPK) {
            int k_idx = tb * B_TOPK + tid;
            int token = (k_idx < params.topk) ? idx_ptr[k_idx] : -1;
            smem.sIdx[tid] = token;
            smem.sValid[tid] = (token >= 0 && token < params.s_kv);
        }
        __syncthreads();

        //===== Phase 1: S_raw[64h, 64t] = Q @ K^T (9 d-tiles, WMMA-accumulated) =====
        for (int dt = 0; dt < NUM_D_TILES; ++dt) {
            load_rows_tile(q_ptr, smem.sQ, params.stride_q_h_q, dt * TILE, params.d_qk);
            load_gather_tile(params.kv, smem.sKV, smem.sIdx, smem.sValid,
                             params.stride_kv_s_kv, dt * TILE, params.d_qk);
            __syncthreads();
            wmma_tile_gemm<wmma::row_major, wmma::col_major>(smem.sQ, smem.sKV, smem.sAcc,
                                                             warp_idx, /*accumulate=*/dt != 0);
            __syncthreads();
        }

        //===== Phase 2: P = exp2(S2 - lse2) via stored GLOBAL LSE (FA-2 style) =====
        {
            const float scale2 = params.sm_scale * LOG2E;
            #pragma unroll
            for (int i = 0; i < (B_H * B_TOPK) / NUM_THREADS; ++i) {
                int idx = tid + i * NUM_THREADS;
                int row = idx / B_TOPK;
                int col = idx % B_TOPK;
                float lse2 = smem.sLse[row];
                float p = 0.0f;
                // !isfinite(lse2): all-invalid row (lse = -inf) -> P = 0 everywhere
                if (smem.sValid[col] && isfinite(lse2)) {
                    p = exp2f(smem.sAcc[idx] * scale2 - lse2);
                }
                smem.sP[idx] = bf16(p);
            }
        }
        __syncthreads();

        //===== Phase 3: dP[64h, 64t] = dO @ V^T (8 v-tiles; V = KV[:, :512]) =====
        for (int vt = 0; vt < NUM_V_TILES; ++vt) {
            load_rows_tile(do_ptr, smem.sDO, params.stride_do_h_q, vt * TILE, params.d_v);
            load_gather_tile(params.kv, smem.sKV, smem.sIdx, smem.sValid,
                             params.stride_kv_s_kv, vt * TILE, params.d_v);
            __syncthreads();
            wmma_tile_gemm<wmma::row_major, wmma::col_major>(smem.sDO, smem.sKV, smem.sAcc,
                                                             warp_idx, /*accumulate=*/vt != 0);
            __syncthreads();
        }

        //===== Phase 4: dS = P * (dP - D[h]) * sm_scale (bf16 operand for WMMA) =====
        {
            #pragma unroll
            for (int i = 0; i < (B_H * B_TOPK) / NUM_THREADS; ++i) {
                int idx = tid + i * NUM_THREADS;
                int row = idx / B_TOPK;
                float ds = float(smem.sP[idx]) * (smem.sAcc[idx] - smem.sD[row]) * params.sm_scale;
                smem.sDS[idx] = bf16(ds);   // invalid cols already have P == 0 -> ds == 0
            }
        }
        __syncthreads();

        //===== Phase 5: dV[t,:] += P^T @ dO  (8 v-tiles, atomic scatter by index) =====
        for (int vt = 0; vt < NUM_V_TILES; ++vt) {
            load_rows_tile(do_ptr, smem.sDO, params.stride_do_h_q, vt * TILE, params.d_v);
            __syncthreads();
            wmma_tile_gemm<wmma::col_major, wmma::row_major>(smem.sP, smem.sDO, smem.sAcc,
                                                             warp_idx, /*accumulate=*/false);
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < (B_TOPK * TILE) / NUM_THREADS; ++i) {
                int idx = tid + i * NUM_THREADS;
                int krow = idx / TILE;
                int col = idx % TILE;
                if (smem.sValid[krow]) {
                    float val = smem.sAcc[idx];
                    if (val != 0.0f) {
                        atomicAdd(params.dv + (long)smem.sIdx[krow] * params.stride_dv_s_kv + vt * TILE + col, val);
                    }
                }
            }
            __syncthreads();
        }

        //===== Phase 6: dQ[h,:] += dS @ K (fp32 RMW, exclusive) and dK[t,:] += dS^T @ Q =====
        for (int dt = 0; dt < NUM_D_TILES; ++dt) {
            load_rows_tile(q_ptr, smem.sQ, params.stride_q_h_q, dt * TILE, params.d_qk);
            load_gather_tile(params.kv, smem.sKV, smem.sIdx, smem.sValid,
                             params.stride_kv_s_kv, dt * TILE, params.d_qk);
            __syncthreads();

            // (a) dQ d-tile = dS @ K_tile -> RMW into fp32 dq (this CTA owns these rows)
            wmma_tile_gemm<wmma::row_major, wmma::row_major>(smem.sDS, smem.sKV, smem.sAcc,
                                                             warp_idx, /*accumulate=*/false);
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < (B_H * TILE) / NUM_THREADS; ++i) {
                int idx = tid + i * NUM_THREADS;
                int row = idx / TILE;
                int col = idx % TILE;
                float* addr = dq_ptr + row * (long)params.stride_dq_h_q + dt * TILE + col;
                *addr += smem.sAcc[idx];
            }
            __syncthreads();

            // (b) dK d-tile = dS^T @ Q_tile -> atomic scatter by index
            wmma_tile_gemm<wmma::col_major, wmma::row_major>(smem.sDS, smem.sQ, smem.sAcc,
                                                             warp_idx, /*accumulate=*/false);
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < (B_TOPK * TILE) / NUM_THREADS; ++i) {
                int idx = tid + i * NUM_THREADS;
                int krow = idx / TILE;
                int col = idx % TILE;
                if (smem.sValid[krow]) {
                    float val = smem.sAcc[idx];
                    if (val != 0.0f) {
                        atomicAdd(params.dk + (long)smem.sIdx[krow] * params.stride_dk_s_kv + dt * TILE + col, val);
                    }
                }
            }
            __syncthreads();
        }
    }
}

}  // namespace sparse_bwd

void run_sparse_bwd_kernel(const SparsePrefillBwdParams& params) {
    using namespace sparse_bwd;

    // CFG ladder (FLASH_MLA_SM120_SPARSE_BWD_CFG): 0 = legacy WMMA (default,
    // byte-identical), 1 = raw mma.sync + ldmatrix + cp.async (bwd_mma.cuh),
    // 2 = +vectorized dK/dV scatter via red.global.add.v2.f32 (Blackwell-class
    //     reduction, ptxas-verified on plain sm_120 -- _probe_blackwell_reducers.sh),
    // 3 = +lane-paired red.global.add.v4.f32 (shfl-fused 16B payloads, half the
    //     reduction transactions of CFG=2 again).
    static const int bwd_cfg = [] {
        const char* e = getenv("FLASH_MLA_SM120_SPARSE_BWD_CFG");
        const int v = e ? atoi(e) : 0;
        return v < 0 ? 0 : (v > 3 ? 3 : v);
    }();
    if (bwd_cfg >= 1) {
        sparse_bwd_mma::launch_sparse_bwd_mma(params, /*red_tier=*/bwd_cfg - 1);
        CHECK_CUDA_KERNEL_LAUNCH();
        return;
    }

    const int num_h_blocks = params.h_q / B_H;
    const dim3 grid(num_h_blocks * params.s_q);
    const dim3 block(NUM_THREADS);
    const size_t smem_size = sizeof(SharedMemory);

    if (smem_size > 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(sparse_prefill_bwd_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    sparse_prefill_bwd_kernel<<<grid, block, smem_size, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace sm120
