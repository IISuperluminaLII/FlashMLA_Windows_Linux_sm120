/***************************************************************************************************
 * SM120 Sparse-FP8 Decode Kernel - full rewrite (WMMA, batch-parallel, no split-KV phase 1).
 * Design: audit/design-sparse-decode.md (implementation-ready; decisions D-1..D-12).
 *
 * Semantics (faithful to the authors' sm90 sparse-fp8 decode):
 *   For request b, query position s, over its topk selected tokens t (indices[b][s][:]):
 *     S[h,t] = Q[h,:] . K[t,:]                 (full 576: dequantized nope + bf16 rope)
 *     P      = softmax over VALID t (global online softmax, base-2 domain, -1e30 init)
 *     O[h,:] = sum_t P[h,t] * V[t,:]           (V = dequantized FIRST 512 of the row)
 *     lse    = (L==0) ? +inf : ln(L) + M2/log2(e)     [NATURAL log final sink]
 *
 *   - NO block_table: indices address the paged cache directly, page = idx/64, offset = idx%64
 *     (README.md: "the kernel does not require the block_table parameter").
 *   - token == -1 is the ONLY invalid form (decode convention); no address is ever formed for
 *     it and its K/V smem lanes are HARD-ZEROED (the oracle NaN-poisons every unreferenced
 *     token, and 0 * NaN = NaN would otherwise poison the GEMMs).
 *   - All KV address arithmetic in int64_t: page * kv_page_stride(41,984) exceeds INT32_MAX
 *     at the authors' own perf shapes (pool > ~51K pages).
 *   - WMMA fragments are touched ONLY via fill/load/store_matrix_sync/mma_sync -- zero
 *     hand-indexing of frag.x[] (the root cause of the previous placeholder's garbage).
 *
 * Grid: dim3(ceil(q_head_per_hk/64), s_q, b), 256 threads, 1 CTA/SM (98.4 KiB smem).
 **************************************************************************************************/

#include "params.h"
#include "traits.h"
#include "dequant.h"
#include "sparse_decode_cfg.h"   // shared CFG latch (also read by pybind.cpp)
#include "../../../utils.h"

#include <mma.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>   // fprintf/stderr used by CHECK_CUDA from csrc/utils.h
#include <cstdlib>  // exit used by CHECK_CUDA (getenv/atoi live in sparse_decode_cfg.h)

namespace sm120 {
namespace sparse_decode {

using namespace nvcuda;

}  // namespace sparse_decode
}  // namespace sm120

// mma.sync tier (CFG=1): raw mma.sync + ldmatrix + cp.async port of the WMMA kernel
// below. Needs traits.h/dequant.h/params.h symbols from sm120::sparse_decode.
#include "splitkv_mla_mma.cuh"

namespace sm120 {
namespace sparse_decode {

//==============================================================================
// Cooperative tile loaders (all 256 threads: token = tid/4, 16-elem chunk = tid%4)
//==============================================================================

// Gather + dequant one 64-wide column tile of the fp8 nope region into sKV[64][64].
// Serves the K d-tiles 0..7 of QK AND all 8 V column tiles of PV (V aliases nope).
// One fp32 scale per tile: a 64-aligned 64-wide range never crosses a 128-elem quant tile.
__device__ __forceinline__ void load_nope_tile(SharedMemoryPlan& sm, int col_start) {
    const int token = threadIdx.x >> 2;
    const int chunk = threadIdx.x & 3;
    __nv_bfloat16* dst = &sm.sKV[token * 64 + chunk * 16];
    const uint8_t* tp = sm.sTokPtr[token];
    if (tp == nullptr) {                        // invalid token -> HARD ZERO (NaN-poison trap)
        *(uint4*)(dst)     = make_uint4(0u, 0u, 0u, 0u);
        *(uint4*)(dst + 8) = make_uint4(0u, 0u, 0u, 0u);
        return;
    }
    const float scale = *(const float*)(tp + HEAD_DIM_NOPE + 4 * (col_start >> 7));
    const fp8x16 raw  = load_128b<fp8x16>(tp + col_start + chunk * 16);
    const bf16x8 lo = cvt_fp8x8_bf16x8_fp32(raw.lo, scale);
    const bf16x8 hi = cvt_fp8x8_bf16x8_fp32(raw.hi, scale);
    store_128b(dst,     lo);
    store_128b(dst + 8, hi);
}

// Gather the 64 unquantized bf16 rope elements (bytes [528,656) of the row) into sKV[64][64].
__device__ __forceinline__ void load_rope_tile(SharedMemoryPlan& sm) {
    const int token = threadIdx.x >> 2;
    const int chunk = threadIdx.x & 3;
    __nv_bfloat16* dst = &sm.sKV[token * 64 + chunk * 16];
    const uint8_t* tp = sm.sTokPtr[token];
    if (tp == nullptr) {
        *(uint4*)(dst)     = make_uint4(0u, 0u, 0u, 0u);
        *(uint4*)(dst + 8) = make_uint4(0u, 0u, 0u, 0u);
        return;
    }
    const uint8_t* rp = tp + HEAD_DIM_NOPE + NUM_SCALES * (int)sizeof(float);   // byte +528
    *(uint4*)(dst)     = *(const uint4*)(rp + chunk * 32);        // 8 bf16
    *(uint4*)(dst + 8) = *(const uint4*)(rp + chunk * 32 + 16);   // 8 bf16
}

// Load the Q-rope tile [64 heads][64] (elements 512..575 of each Q row) into sPQ.
// sPQ is UNIONED with P; phase separation proven in the design (section 3.5).
__device__ __forceinline__ void load_qrope_tile(SharedMemoryPlan& sm,
                                                const __nv_bfloat16* q_row_base,
                                                int q_row_stride, int num_valid_rows) {
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vec  = threadIdx.x + i * NUM_THREADS;    // 0..511 (8 x 16B vectors per row)
        const int row  = vec >> 3;
        const int col8 = vec & 7;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (row < num_valid_rows)
            v = *(const uint4*)(q_row_base + (int64_t)row * q_row_stride + 512 + col8 * 8);
        *(uint4*)(&sm.sPQ[row * 64 + col8 * 8]) = v;
    }
}

//==============================================================================
// Main kernel
//==============================================================================
__global__ void __launch_bounds__(NUM_THREADS, 1)
sparse_fp8_decode_kernel(const SparseFP8DecodeParams params) {
    extern __shared__ char smem_raw[];
    SharedMemoryPlan& sm = *reinterpret_cast<SharedMemoryPlan*>(smem_raw);

    const int tid         = threadIdx.x;
    const int warp_idx    = tid / 32;
    const int m_block_idx = blockIdx.x;
    const int s_q_idx     = blockIdx.y;
    const int batch_idx   = blockIdx.z;

    const int head_start     = m_block_idx * BLOCK_M;                              // in [0, q_head_per_hk)
    const int num_valid_rows = min(params.q_head_per_hk - head_start, BLOCK_M);    // ragged head tail
    // Folded (s_q, head) axis is position-major, head-minor: row = s*q_head_per_hk + h_local.
    const int row0           = s_q_idx * params.q_head_per_hk + head_start;

    const __nv_bfloat16* q_row_base = (const __nv_bfloat16*)params.q_ptr
        + (int64_t)batch_idx * params.q_batch_stride
        + (int64_t)row0      * params.q_row_stride;
    __nv_bfloat16* o_row_base = (__nv_bfloat16*)params.o_ptr
        + (int64_t)batch_idx * params.o_batch_stride
        + (int64_t)row0      * params.o_row_stride;
    float* lse_base = params.softmax_lse_ptr
        + (int64_t)batch_idx * params.q_seq_per_hk
        + (int64_t)row0;                                            // h_kv == 1 (host-enforced)
    const int* idx_row = params.indices_ptr
        + (int64_t)batch_idx * params.indices_batch_stride
        + (int64_t)s_q_idx   * params.indices_seq_stride;
    const uint8_t* kv_base = (const uint8_t*)params.kv_ptr;

    //--------------------------------------------------------------------------
    // Prologue: resident Q nope half [64][512], softmax state, O accumulator
    //--------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int vec  = tid + i * NUM_THREADS;    // 0..4095 (64 x 16B vectors per row)
        const int row  = vec >> 6;
        const int col8 = vec & 63;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (row < num_valid_rows)
            v = *(const uint4*)(q_row_base + (int64_t)row * params.q_row_stride + col8 * 8);
        *(uint4*)(&sm.sQ[row * 512 + col8 * 8]) = v;   // rows >= num_valid_rows -> hard zero
    }
    if (tid < BLOCK_M) { sm.sM[tid] = MAX_INIT_VAL; sm.sL[tid] = 0.0f; }

    // rO[v*16+j] <-> O[row = tid/64 + 4j][col = v*64 + tid%64]; every index compile-time
    // constant under FULL unrolling (a runtime index would spill the array to .local).
    float rO[O_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < O_PER_THREAD; ++i) rO[i] = 0.0f;
    __syncthreads();                                                            // S0

    const int num_topk_blocks = (params.topk + TOPK_BLOCK_SIZE - 1) / TOPK_BLOCK_SIZE;

    for (int kb = 0; kb < num_topk_blocks; ++kb) {
        //===== Phase A: indices -> validity + token base pointers =====
        if (tid < TOPK_BLOCK_SIZE) {
            const int k     = kb * TOPK_BLOCK_SIZE + tid;
            const int token = (k < params.topk) ? idx_row[k] : -1;   // ragged topk tail -> invalid
            const bool ok   = (token >= 0);                          // decode: ONLY -1 is invalid
            sm.sValid[tid]  = ok ? 1 : 0;
            sm.sTokPtr[tid] = ok ? (kv_base
                                    + (int64_t)(token >> 6) * params.kv_page_stride
                                    + (int64_t)(token & 63) * params.kv_token_stride)
                                 : nullptr;
        }
        __syncthreads();                                                        // S1

        //===== Phase B: S_raw[64h, 64t] = Q . K^T over 9 d-tiles, accumulators persistent =====
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[TILES_PER_WARP];
            #pragma unroll
            for (int i = 0; i < TILES_PER_WARP; ++i) wmma::fill_fragment(c_frag[i], 0.0f);

            for (int dt = 0; dt < 9; ++dt) {
                if (dt < 8) {
                    load_nope_tile(sm, dt * 64);
                } else {
                    load_rope_tile(sm);
                    load_qrope_tile(sm, q_row_base, params.q_row_stride, num_valid_rows);
                }
                __syncthreads();                                                // S2
                #pragma unroll
                for (int i = 0; i < TILES_PER_WARP; ++i) {
                    const int t = warp_idx + i * NUM_WARPS;    // warp w owns tiles w, w+8
                    const int m = t >> 2, n = t & 3;
                    #pragma unroll
                    for (int k4 = 0; k4 < 4; ++k4) {
                        const __nv_bfloat16* aP = (dt < 8)
                            ? (sm.sQ  + m * 16 * 512 + dt * 64 + k4 * 16)
                            : (sm.sPQ + m * 16 * 64  +           k4 * 16);
                        const int aLd = (dt < 8) ? 512 : 64;
                        wmma::load_matrix_sync(a_frag, aP, aLd);
                        // Row-major K tile viewed as col_major B == K^T (no smem transpose).
                        wmma::load_matrix_sync(b_frag, sm.sKV + n * 16 * 64 + k4 * 16, 64);
                        wmma::mma_sync(c_frag[i], a_frag, b_frag, c_frag[i]);
                    }
                }
                __syncthreads();                                                // S3
            }
            #pragma unroll
            for (int i = 0; i < TILES_PER_WARP; ++i) {
                const int t = warp_idx + i * NUM_WARPS;
                const int m = t >> 2, n = t & 3;
                wmma::store_matrix_sync(sm.sAcc + m * 16 * ACC_LD + n * 16, c_frag[i],
                                        ACC_LD, wmma::mem_row_major);
            }
        }
        __syncthreads();                                                        // S4

        //===== Phase C: online softmax, base-2 domain, 4 threads per row =====
        {
            const float scale2 = params.sm_scale_log2;     // sm_scale * log2(e)
            const int r   = tid >> 2;
            const int sub = tid & 3;

            float s[16];
            float cm = MAX_INIT_VAL;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int col = sub + 4 * j;               // (4r+sub+4j) mod 32 covers all banks
                s[j] = sm.sValid[col] ? (sm.sAcc[r * ACC_LD + col] * scale2) : MAX_INIT_VAL;
                cm   = fmaxf(cm, s[j]);
            }
            cm = fmaxf(cm, __shfl_xor_sync(0xffffffff, cm, 1));
            cm = fmaxf(cm, __shfl_xor_sync(0xffffffff, cm, 2));   // 4 lanes now share the row max

            const float m_old   = sm.sM[r];
            const float m_new   = fmaxf(m_old, cm);
            const float rescale = exp2f(m_old - m_new);    // finite - finite, never NaN

            float rs = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int col = sub + 4 * j;
                // Branch is MANDATORY: in an all-invalid block m_new == MAX_INIT_VAL and a
                // branchless exp2f(s - m_new) would be exp2f(0) = 1 for every masked column.
                const float p = sm.sValid[col] ? exp2f(s[j] - m_new) : 0.0f;
                sm.sPQ[r * 64 + col] = __float2bfloat16_rn(p);
                rs += p;
            }
            rs += __shfl_xor_sync(0xffffffff, rs, 1);
            rs += __shfl_xor_sync(0xffffffff, rs, 2);
            if (sub == 0) {
                sm.sM[r]     = m_new;
                sm.sL[r]     = sm.sL[r] * rescale + rs;
                sm.sScale[r] = rescale;
            }
        }
        __syncthreads();                                                        // S5

        //===== Phase D: rescale the O accumulator (register-private; sScale published at S5) =====
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float sc = sm.sScale[tid / 64 + 4 * j];   // warp-uniform -> smem broadcast
            #pragma unroll
            for (int v = 0; v < 8; ++v) rO[v * 16 + j] *= sc;
        }

        //===== Phase E: O += P . V over 8 V column tiles (V = first 512 of the same rows) =====
        // FULL unroll is REQUIRED: rO[v*16+j] must be a compile-time index in every loop that
        // touches it, or ptxas allocates rO[128] on the stack (measured: 512 B local frame
        // before this pragma). Design KEY FACT 7. Syncs inside the unrolled bodies are legal.
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            load_nope_tile(sm, v * 64);
            __syncthreads();                                                    // S6
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vb;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> pc;
                #pragma unroll
                for (int i = 0; i < TILES_PER_WARP; ++i) {
                    const int t = warp_idx + i * NUM_WARPS;
                    const int m = t >> 2, n = t & 3;
                    wmma::fill_fragment(pc, 0.0f);
                    #pragma unroll
                    for (int k4 = 0; k4 < 4; ++k4) {
                        wmma::load_matrix_sync(pa, sm.sPQ + m * 16 * 64 + k4 * 16, 64);
                        wmma::load_matrix_sync(vb, sm.sKV + k4 * 16 * 64 + n * 16, 64);
                        wmma::mma_sync(pc, pa, vb, pc);
                    }
                    wmma::store_matrix_sync(sm.sAcc + m * 16 * ACC_LD + n * 16, pc,
                                            ACC_LD, wmma::mem_row_major);
                }
            }
            __syncthreads();                                                    // S7
            #pragma unroll
            for (int j = 0; j < 16; ++j)
                rO[v * 16 + j] += sm.sAcc[(tid / 64 + 4 * j) * ACC_LD + (tid & 63)];
            __syncthreads();                                                    // S8 (WAR on sKV/sAcc)
        }
    }

    //--------------------------------------------------------------------------
    // Epilogue: O = rO / L (lonely row -> exact 0), LSE natural log with +inf sentinel.
    // Explicit row guard: sm90 gets tail clipping free from TMA OOB; sm120 must guard.
    //--------------------------------------------------------------------------
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int row = tid / 64 + 4 * j;
        if (row < num_valid_rows) {
            const float L   = sm.sL[row];
            const float inv = (L == 0.0f) ? 0.0f : (1.0f / L);   // L is exactly 0 or >= 1
            __nv_bfloat16* dst = o_row_base + (int64_t)row * params.o_row_stride + (tid & 63);
            #pragma unroll
            for (int v = 0; v < 8; ++v)
                dst[v * 64] = __float2bfloat16_rn(rO[v * 16 + j] * inv);
        }
    }
    if (tid < num_valid_rows) {
        const float L = sm.sL[tid];
        // NATURAL-log final sink with +INFINITY sentinel (oracle compares inf masks exactly).
        lse_base[tid] = (L == 0.0f) ? INFINITY : (logf(L) + sm.sM[tid] / LOG2E);
    }
}

//==============================================================================
// Launcher
//==============================================================================
void run_sparse_fp8_decode_kernel(const SparseFP8DecodeParams& params) {
    // CFG ladder (FLASH_MLA_SM120_SPARSE_DECODE_CFG): 0 = legacy WMMA (default,
    // byte-identical), 1 = raw mma.sync batch-parallel (splitkv_mla_mma.cuh),
    // 2 = mma.sync + authors' split-KV (metadata walk + partials + combine),
    // 3 = 2 + the 2-CTA cluster DSM crossover (half-topk dequant sharing;
    //     documented NEGATIVE on this part, opt-in for the record),
    // 4 = the CFG=2 splitkv kernel with the split cap raised 64 -> 192
    //     (metadata arm emits up to 188 parts; combine carries 128/192 tiers).
    // sparse_decode_cfg() is the SAME single-instance latch pybind's metadata arm
    // reads (sparse_decode_cfg.h): CFG>=2 makes get_attn_impl_meta emit real
    // multi-split schedules that ONLY the splitkv tiers consume (CFG=3 with the
    // head-block count padded to full cluster pairs); CFG<2 keeps
    // num_sm_parts=1 so the combine stays inert and both batch-parallel tiers
    // ignore the metadata entirely.
    const int decode_cfg = sparse_decode_cfg();
    if (decode_cfg == 3) {
        mma::launch_sparse_fp8_decode_mma_splitkv_x(params);
        CHECK_CUDA_KERNEL_LAUNCH();
        return;
    }
    if (decode_cfg == 2 || decode_cfg == 4) {
        mma::launch_sparse_fp8_decode_mma_splitkv(params);
        CHECK_CUDA_KERNEL_LAUNCH();
        return;
    }
    if (decode_cfg == 1) {
        mma::launch_sparse_fp8_decode_mma(params);
        CHECK_CUDA_KERNEL_LAUNCH();
        return;
    }
    const int num_m_blocks = (params.q_head_per_hk + BLOCK_M - 1) / BLOCK_M;
    const dim3 grid(num_m_blocks, params.s_q, params.b);
    const dim3 block(NUM_THREADS);
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    static_assert(smem_size <= 99 * 1024, "exceeds SM120 99KB opt-in smem");
    if (smem_size > 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(sparse_fp8_decode_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)smem_size));
    }
    sparse_fp8_decode_kernel<<<grid, block, smem_size, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace sparse_decode
}  // namespace sm120
