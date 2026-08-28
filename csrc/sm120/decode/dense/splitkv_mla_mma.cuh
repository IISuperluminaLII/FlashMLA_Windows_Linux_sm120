#pragma once
// ============================================================================
// DENSE DECODE via raw mma.sync + ldmatrix + cp.async + SPLIT-KV (SM120) [CFG>=1]
// ============================================================================
// The dense-decode analog of the sparse mma tiers, plus the authors' FULL
// split-KV machinery consumed 1:1:
//   - grid (m_blocks, 1, num_sm_parts): each CTA reads its tile-scheduler
//     metadata row (begin_batch, begin_block, end_batch, end_block(excl),
//     begin_n_split_idx) produced by csrc/smxx/get_mla_metadata.cu and walks
//     its batch/block range -- the authors' persistent load balancer.
//   - Unsplit batches: finals written directly (bf16/fp16 out + NATURAL-log lse
//     with the legacy 1e30 sentinel for L == 0 -- matches the WMMA kernel and
//     the oracle bit-for-bit conventions).
//   - Split batches: NORMALIZED fp32 partial O + 2-BASED partial lse into the
//     authors' accum layout ([split_row, h_k*q_seq_per_hk(, 512)], row =
//     num_splits_ptr[batch] + n_split_idx; empty split -> lse2 = -inf), merged
//     by the authors' own flash_fwd_mla_combine_kernel (already launched
//     unconditionally after this kernel; it skips 1-split batches).
//   - Bottom-right causal per ROW (rows fold s_q x q_head_per_hk; a 64-row
//     block may straddle two s_q positions).
//   - KV pages streamed with the proven 2-stage cp.async ping-pong; V re-reads
//     the page's first 512 columns (L2-hot; the 128MB L2 absorbs the re-read).
//   - CUDA 13 builds: the Q prologue uses 256-bit ld.global.nc.v8 (ptxas-
//     verified on plain sm_120 under PTX 9.0; 12.9 builds fall back to 16B).
//
// Smem: sQ 73,728 (swz 576) + sKV 2 x 8,192 (swz 64) + sP 8,192 (swz 64)
//       + stats 512 -> 98,816 B <= 101,376 (99KB opt-in), 1 CTA/SM.
// ============================================================================

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <utility>

namespace sm120 {
namespace dense_decode_mma {

constexpr int DD_MMA_M = 16, DD_MMA_N = 8, DD_MMA_K = 16;
constexpr int DD_NW = 8;
constexpr int DD_THREADS = DD_NW * 32;      // 256
constexpr int DD_BM = 64;                   // folded (s_q x head) rows per CTA
constexpr int DD_PAGE = 64;                 // tokens per KV page == token tile
constexpr int DD_DQK = 576, DD_DV = 512, DD_DTILE = 64;
constexpr int DD_NKTILES = DD_DQK / DD_DTILE;   // 9
constexpr int DD_NVTILES = DD_DV / DD_DTILE;    // 8
constexpr float DD_LOG2E = 1.4426950408889634f;

// ---- dtype traits: the ONLY type-specific pieces (mma mnemonic, packing) ----
template <typename T> struct DDT;
template <> struct DDT<__nv_bfloat16> {
    static __device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4],
                                               const uint32_t (&b)[2], const float (&c)[4]) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    }
    static __device__ __forceinline__ uint32_t pack2(float x, float y) {
        const __nv_bfloat162 v = __floats2bfloat162_rn(x, y);
        return *reinterpret_cast<const uint32_t*>(&v);
    }
};
template <> struct DDT<__half> {
    static __device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4],
                                               const uint32_t (&b)[2], const float (&c)[4]) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    }
    static __device__ __forceinline__ uint32_t pack2(float x, float y) {
        const __half2 v = __floats2half2_rn(x, y);
        return *reinterpret_cast<const uint32_t*>(&v);
    }
};

// ---- PTX primitives (identical to the proven mma kernels) -------------------
__device__ __forceinline__ uint32_t dd_cvta(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void dd_ldm_x4(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void dd_ldm_x2(uint32_t (&r)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void dd_ldm_x2_trans(uint32_t (&r)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void dd_cp16(uint32_t sa, const void* g, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 ::"r"(sa), "l"(g), "r"(src_bytes));
}
__device__ __forceinline__ void dd_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void dd_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ int dd_swz(int r, int c, int dim) {
    return r * dim + (c ^ ((r & 7) << 3));
}

// ---- smem plan --------------------------------------------------------------
template <typename InputT>
struct SmemPlanDenseMma {
    __align__(128) InputT sQ[DD_BM * DD_DQK];            // 73,728 B, swz(576)
    __align__(128) InputT sKV[2][DD_PAGE * DD_DTILE];    // 2 x 8,192 B, swz(64)
    __align__(128) InputT sP[DD_BM * DD_PAGE];           // 8,192 B, swz(64)
    __align__(128) float sStatM[DD_BM];
    __align__(128) float sStatS[DD_BM];
};

// Stream one 64-token x 64-col tile of a KV page into stage buffer `dst`,
// swizzled; rows beyond valid_tokens zero-fill via src-size 0 (no OOB read).
template <typename InputT>
__device__ __forceinline__ void dd_load_kv_tile(SmemPlanDenseMma<InputT>& sm, InputT* dst,
                                                const InputT* page_ptr, int row_stride,
                                                int col_start, int valid_tokens) {
    const int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int g = tid + i * DD_THREADS;
        const int r = g >> 3;
        const int c = (g & 7) * 8;
        dd_cp16(dd_cvta(&dst[dd_swz(r, c, DD_DTILE)]),
                page_ptr + (int64_t)r * row_stride + col_start + c,
                (r < valid_tokens) ? 16 : 0);
    }
}

// ---- kernel -----------------------------------------------------------------
template <typename InputT>
__global__ void __launch_bounds__(DD_THREADS, 1)
dense_decode_mma_kernel(const DecodingParams params) {
    extern __shared__ char smem_raw[];
    SmemPlanDenseMma<InputT>& sm = *reinterpret_cast<SmemPlanDenseMma<InputT>*>(smem_raw);

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int m_block_idx = blockIdx.x;
    const int part_idx = blockIdx.z;

    // ---- tile-scheduler metadata row (authors' format, 5 ints used;
    //      TileSchedulerMetaDataSize from params.h, via the including TU) ----
    const int* meta = params.tile_scheduler_metadata_ptr + part_idx * TileSchedulerMetaDataSize;
    const int begin_idx = meta[0];
    const int begin_block = meta[1];
    const int end_idx = meta[2];
    const int end_block_last = meta[3];          // exclusive, for batch == end_idx
    const int begin_n_split_idx = meta[4];

    const int ms = warp & 3, nh = warp >> 2;
    const int wrow = ms * DD_MMA_M;
    const int row0 = m_block_idx * DD_BM;        // folded (s_q x head) row base
    const int num_valid_rows = min(params.q_seq_per_hk - row0, DD_BM);
    const int row_a = wrow + lane / 4, row_b = row_a + 8;

    for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx) {
        const int seqlen_k = __ldg(params.seqlens_k_ptr + batch_idx);
        const int total_blocks = (max(seqlen_k, 1) + DD_PAGE - 1) / DD_PAGE;  // seqlen 0 -> 1 block
        const int blk_lo = (batch_idx == begin_idx) ? begin_block : 0;
        const int blk_hi = (batch_idx == end_idx) ? end_block_last : total_blocks;
        // Zero-seqlen batches own one scheduler block (get_mla_metadata's
        // max(seqlen, 1) convention) but have nothing to read: collapse the
        // walk to zero iterations (exact legacy semantics). Without this, a
        // begin/mid zero-seqlen batch would read block_table[b][0], which is
        // an OOB read when EVERY batch is zero-seqlen (zero-width block_table).
        const int blk_walk_hi = (seqlen_k > 0) ? blk_hi : blk_lo;
        const int n_split_idx = (batch_idx == begin_idx) ? begin_n_split_idx : 0;
        // Authoritative split count from the scheduler's prefix sum -- the EXACT
        // condition the combine kernel uses to skip a batch. (Deriving it from
        // block ranges mishandles zero-seqlen end-batches: metadata end_block is
        // 0 there, yet the batch has 1 split and combine will skip it, so THIS
        // kernel must write its finals.)
        const int split_base = __ldg(params.num_splits_ptr + batch_idx);
        const bool no_split = (__ldg(params.num_splits_ptr + batch_idx + 1) - split_base) == 1;

        // ---- resident swizzled Q for this batch (zero ragged rows) ---------
        const InputT* q_ptr = reinterpret_cast<const InputT*>(params.q_ptr)
            + (int64_t)batch_idx * params.q_batch_stride
            + (int64_t)row0 * params.q_row_stride;
        {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13)
            // 256-bit loads (Blackwell; PTX 9.0+, ptxas-verified on plain sm_120):
            // 9 x 32B per thread. Requires 32B-aligned sources -- guaranteed when
            // the row stride is a multiple of 16 elements (576 in practice) and
            // the row base is 32B-aligned; otherwise fall back to 16B loads.
            // One v8 covers granules c and c+8; each 16B half is stored to its
            // own swizzled slot (the XOR may swap their order, both land right).
            const bool use_v8 = (params.q_row_stride % 16 == 0) &&
                                ((reinterpret_cast<uintptr_t>(q_ptr) & 31) == 0);
            if (use_v8) {
                #pragma unroll
                for (int i = 0; i < 9; ++i) {
                    const int g = tid + i * DD_THREADS;     // 32B granule 0..2303
                    const int r = g / (DD_DQK / 16);
                    const int c = (g % (DD_DQK / 16)) * 16;
                    float4 lo = make_float4(0.f, 0.f, 0.f, 0.f), hi = lo;
                    if (r < num_valid_rows) {
                        const float* src = reinterpret_cast<const float*>(
                            q_ptr + (int64_t)r * params.q_row_stride + c);
                        asm volatile("ld.global.nc.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
                            : "=f"(lo.x), "=f"(lo.y), "=f"(lo.z), "=f"(lo.w),
                              "=f"(hi.x), "=f"(hi.y), "=f"(hi.z), "=f"(hi.w) : "l"(src));
                    }
                    *reinterpret_cast<float4*>(&sm.sQ[dd_swz(r, c, DD_DQK)]) = lo;
                    *reinterpret_cast<float4*>(&sm.sQ[dd_swz(r, c + 8, DD_DQK)]) = hi;
                }
            } else
#endif
            {
                #pragma unroll
                for (int i = 0; i < 18; ++i) {
                    const int g = tid + i * DD_THREADS;
                    const int r = g / (DD_DQK / 8);
                    const int c = (g % (DD_DQK / 8)) * 8;
                    int4 v = make_int4(0, 0, 0, 0);
                    if (r < num_valid_rows)
                        v = *reinterpret_cast<const int4*>(
                                q_ptr + (int64_t)r * params.q_row_stride + c);
                    *reinterpret_cast<int4*>(&sm.sQ[dd_swz(r, c, DD_DQK)]) = v;
                }
            }
        }

        // Per-lane bottom-right causal limits (rows may straddle s_q positions).
        int lim_a = seqlen_k, lim_b = seqlen_k;
        if (params.is_causal) {
            const int qa = min(row0 + row_a, params.q_seq_per_hk - 1) / params.q_head_per_hk;
            const int qb = min(row0 + row_b, params.q_seq_per_hk - 1) / params.q_head_per_hk;
            lim_a = min(seqlen_k, seqlen_k - params.s_q + 1 + qa);
            lim_b = min(seqlen_k, seqlen_k - params.s_q + 1 + qb);
        }

        float rmax[2] = {-INFINITY, -INFINITY}, rsum[2] = {0.f, 0.f};
        float Or[DD_NVTILES][4][4];
        #pragma unroll
        for (int t = 0; t < DD_NVTILES; ++t)
            #pragma unroll
            for (int n = 0; n < 4; ++n) { Or[t][n][0] = Or[t][n][1] = Or[t][n][2] = Or[t][n][3] = 0.f; }

        const int* block_table_row = params.block_table
            + (int64_t)batch_idx * params.block_table_batch_stride;
        const InputT* k_base = reinterpret_cast<const InputT*>(params.k_ptr);

        __syncthreads();   // sQ visible (and previous batch's smem fully consumed)

        for (int blk = blk_lo; blk < blk_walk_hi; ++blk) {
            const int phys = __ldg(block_table_row + blk);
            const InputT* page_ptr = k_base + (int64_t)phys * params.k_batch_stride;
            const int tok0 = blk * DD_PAGE;
            const int valid_tokens = min(seqlen_k - tok0, DD_PAGE);   // may be <= 0 only if seqlen 0

            // ---- QK over 9 pipelined K d-tiles -----------------------------
            float Sr[4][4];
            #pragma unroll
            for (int n = 0; n < 4; ++n) { Sr[n][0] = Sr[n][1] = Sr[n][2] = Sr[n][3] = 0.f; }

            dd_load_kv_tile(sm, sm.sKV[0], page_ptr, params.k_row_stride, 0,
                            max(valid_tokens, 0));
            dd_cp_commit();
            for (int dt = 0; dt < DD_NKTILES; ++dt) {
                const int cur = dt & 1;
                if (dt + 1 < DD_NKTILES) {
                    dd_load_kv_tile(sm, sm.sKV[cur ^ 1], page_ptr, params.k_row_stride,
                                    (dt + 1) * DD_DTILE, max(valid_tokens, 0));
                    dd_cp_commit();
                    dd_cp_wait<1>();
                } else {
                    dd_cp_wait<0>();
                }
                __syncthreads();
                #pragma unroll
                for (int dl = 0; dl < 4; ++dl) {
                    uint32_t Qr[4];
                    {
                        const int row = wrow + (lane % 16);
                        const int col = dt * DD_DTILE + dl * DD_MMA_K + (lane / 16) * 8;
                        dd_ldm_x4(Qr, dd_cvta(&sm.sQ[dd_swz(row, col, DD_DQK)]));
                    }
                    #pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        uint32_t Kr[2];
                        const int krow = (nh * 4 + n) * DD_MMA_N + (lane % 8);
                        const int kcol = dl * DD_MMA_K + ((lane / 8) % 2) * 8;
                        dd_ldm_x2(Kr, dd_cvta(&sm.sKV[cur][dd_swz(krow, kcol, DD_DTILE)]));
                        DDT<InputT>::mma(Sr[n], Qr, Kr, Sr[n]);
                    }
                }
                __syncthreads();   // stage consumed before dt+2 overwrites it
            }

            // ---- mask (seqlen + per-row causal), scale, partial max --------
            const float scale = params.scale_softmax;
            float tmax[2] = {-INFINITY, -INFINITY};
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int col0 = (nh * 4 + n) * DD_MMA_N + (lane % 4) * 2;
                const int t0 = tok0 + col0, t1 = t0 + 1;
                Sr[n][0] = (t0 < lim_a) ? Sr[n][0] * scale : -INFINITY;
                Sr[n][1] = (t1 < lim_a) ? Sr[n][1] * scale : -INFINITY;
                Sr[n][2] = (t0 < lim_b) ? Sr[n][2] * scale : -INFINITY;
                Sr[n][3] = (t1 < lim_b) ? Sr[n][3] * scale : -INFINITY;
                tmax[0] = fmaxf(tmax[0], fmaxf(Sr[n][0], Sr[n][1]));
                tmax[1] = fmaxf(tmax[1], fmaxf(Sr[n][2], Sr[n][3]));
            }
            tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffff, tmax[0], 1));
            tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffff, tmax[0], 2));
            tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffff, tmax[1], 1));
            tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffff, tmax[1], 2));

            if (nh == 0 && (lane % 4) == 0) {
                sm.sStatM[wrow + lane / 4] = tmax[0];
                sm.sStatM[wrow + lane / 4 + 8] = tmax[1];
            }
            __syncthreads();
            if (nh == 1 && (lane % 4) == 0) {
                sm.sStatM[wrow + lane / 4] = fmaxf(sm.sStatM[wrow + lane / 4], tmax[0]);
                sm.sStatM[wrow + lane / 4 + 8] = fmaxf(sm.sStatM[wrow + lane / 4 + 8], tmax[1]);
            }
            __syncthreads();
            const float bmax0 = sm.sStatM[row_a];
            const float bmax1 = sm.sStatM[row_b];

            // ---- online rescale + P = exp(S - max) -> sP -------------------
            const float nmax0 = fmaxf(rmax[0], bmax0), nmax1 = fmaxf(rmax[1], bmax1);
            const float rs0 = (rmax[0] == -INFINITY) ? ((nmax0 == -INFINITY) ? 1.f : 0.f)
                                                     : __expf(rmax[0] - nmax0);
            const float rs1 = (rmax[1] == -INFINITY) ? ((nmax1 == -INFINITY) ? 1.f : 0.f)
                                                     : __expf(rmax[1] - nmax1);
            rmax[0] = nmax0; rmax[1] = nmax1;
            #pragma unroll
            for (int t = 0; t < DD_NVTILES; ++t)
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    Or[t][n][0] *= rs0; Or[t][n][1] *= rs0; Or[t][n][2] *= rs1; Or[t][n][3] *= rs1;
                }

            float tsum[2] = {0.f, 0.f};
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                const float p0 = (Sr[n][0] == -INFINITY) ? 0.f : __expf(Sr[n][0] - nmax0);
                const float p1 = (Sr[n][1] == -INFINITY) ? 0.f : __expf(Sr[n][1] - nmax0);
                const float p2 = (Sr[n][2] == -INFINITY) ? 0.f : __expf(Sr[n][2] - nmax1);
                const float p3 = (Sr[n][3] == -INFINITY) ? 0.f : __expf(Sr[n][3] - nmax1);
                tsum[0] += p0 + p1; tsum[1] += p2 + p3;
                const int pcol = (nh * 4 + n) * DD_MMA_N + (lane % 4) * 2;
                *reinterpret_cast<uint32_t*>(&sm.sP[dd_swz(row_a, pcol, DD_PAGE)]) =
                    DDT<InputT>::pack2(p0, p1);
                *reinterpret_cast<uint32_t*>(&sm.sP[dd_swz(row_b, pcol, DD_PAGE)]) =
                    DDT<InputT>::pack2(p2, p3);
            }
            rsum[0] = rsum[0] * rs0 + tsum[0];
            rsum[1] = rsum[1] * rs1 + tsum[1];

            // ---- PV: 8 V tiles = the page's first 512 columns (re-read) ----
            dd_load_kv_tile(sm, sm.sKV[0], page_ptr, params.k_row_stride, 0,
                            max(valid_tokens, 0));
            dd_cp_commit();
            for (int vt = 0; vt < DD_NVTILES; ++vt) {
                const int cur = vt & 1;
                if (vt + 1 < DD_NVTILES) {
                    dd_load_kv_tile(sm, sm.sKV[cur ^ 1], page_ptr, params.k_row_stride,
                                    (vt + 1) * DD_DTILE, max(valid_tokens, 0));
                    dd_cp_commit();
                    dd_cp_wait<1>();
                } else {
                    dd_cp_wait<0>();
                }
                __syncthreads();   // V tile + sP visible
                #pragma unroll
                for (int pk = 0; pk < 4; ++pk) {
                    uint32_t Pr[4];
                    {
                        const int prow = wrow + (lane % 16);
                        const int pcol = pk * DD_MMA_K + (lane / 16) * 8;
                        dd_ldm_x4(Pr, dd_cvta(&sm.sP[dd_swz(prow, pcol, DD_PAGE)]));
                    }
                    #pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        uint32_t Vr[2];
                        const int vrow = pk * DD_MMA_K + (lane % 16);
                        const int vcol = nh * 32 + n * DD_MMA_N + (lane / 16) * 8;
                        dd_ldm_x2_trans(Vr, dd_cvta(&sm.sKV[cur][dd_swz(vrow, vcol, DD_DTILE)]));
                        DDT<InputT>::mma(Or[vt][n], Pr, Vr, Or[vt][n]);
                    }
                }
                __syncthreads();   // stage + sP consumed before reuse
            }
        }

        // ---- epilogue for this batch ---------------------------------------
        rsum[0] += __shfl_xor_sync(0xffffffff, rsum[0], 1);
        rsum[0] += __shfl_xor_sync(0xffffffff, rsum[0], 2);
        rsum[1] += __shfl_xor_sync(0xffffffff, rsum[1], 1);
        rsum[1] += __shfl_xor_sync(0xffffffff, rsum[1], 2);
        if (nh == 0 && (lane % 4) == 0) {
            sm.sStatS[wrow + lane / 4] = rsum[0];
            sm.sStatS[wrow + lane / 4 + 8] = rsum[1];
        }
        __syncthreads();
        if (nh == 1 && (lane % 4) == 0) {
            atomicAdd(&sm.sStatS[row_a], rsum[0]);
            atomicAdd(&sm.sStatS[row_b], rsum[1]);
        }
        __syncthreads();
        const float sum0 = sm.sStatS[row_a], sum1 = sm.sStatS[row_b];
        const float inv0 = (sum0 > 0.f) ? 1.f / sum0 : 0.f;
        const float inv1 = (sum1 > 0.f) ? 1.f / sum1 : 0.f;

        if (no_split) {
            // Finals: legacy conventions (natural-log lse, 1e30 sentinel).
            InputT* o_ptr = reinterpret_cast<InputT*>(params.o_ptr)
                + (int64_t)batch_idx * params.o_batch_stride
                + (int64_t)row0 * params.o_row_stride;
            #pragma unroll
            for (int t = 0; t < DD_NVTILES; ++t)
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const int ocol = t * DD_DTILE + nh * 32 + n * DD_MMA_N + (lane % 4) * 2;
                    if (row_a < num_valid_rows)
                        *reinterpret_cast<uint32_t*>(
                            o_ptr + (int64_t)row_a * params.o_row_stride + ocol) =
                            DDT<InputT>::pack2(Or[t][n][0] * inv0, Or[t][n][1] * inv0);
                    if (row_b < num_valid_rows)
                        *reinterpret_cast<uint32_t*>(
                            o_ptr + (int64_t)row_b * params.o_row_stride + ocol) =
                            DDT<InputT>::pack2(Or[t][n][2] * inv1, Or[t][n][3] * inv1);
                }
            if (nh == 0 && (lane % 4) == 0) {
                float* lse_ptr = reinterpret_cast<float*>(params.softmax_lse_ptr)
                    + (int64_t)batch_idx * params.q_seq_per_hk + row0;   // h_k == 1
                if (row_a < num_valid_rows)
                    lse_ptr[row_a] = (sum0 > 0.f) ? (rmax[0] + __logf(sum0)) : 1e30f;
                if (row_b < num_valid_rows)
                    lse_ptr[row_b] = (sum1 > 0.f) ? (rmax[1] + __logf(sum1)) : 1e30f;
            }
        } else {
            // Partials in the authors' accum layout: NORMALIZED fp32 O + 2-based
            // lse (-inf for an empty split); the authors' combine kernel merges.
            const int split_row = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
            float* oaccum = reinterpret_cast<float*>(params.oaccum_ptr)
                + ((int64_t)split_row * params.q_seq_per_hk + row0) * DD_DV;
            #pragma unroll
            for (int t = 0; t < DD_NVTILES; ++t)
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const int ocol = t * DD_DTILE + nh * 32 + n * DD_MMA_N + (lane % 4) * 2;
                    if (row_a < num_valid_rows) {
                        float* pa = oaccum + (int64_t)row_a * DD_DV + ocol;
                        pa[0] = Or[t][n][0] * inv0; pa[1] = Or[t][n][1] * inv0;
                    }
                    if (row_b < num_valid_rows) {
                        float* pb = oaccum + (int64_t)row_b * DD_DV + ocol;
                        pb[0] = Or[t][n][2] * inv1; pb[1] = Or[t][n][3] * inv1;
                    }
                }
            if (nh == 0 && (lane % 4) == 0) {
                float* lse_accum = reinterpret_cast<float*>(params.softmax_lseaccum_ptr)
                    + (int64_t)split_row * params.q_seq_per_hk + row0;
                if (row_a < num_valid_rows)
                    lse_accum[row_a] = (sum0 > 0.f) ? (rmax[0] + __logf(sum0)) * DD_LOG2E
                                                    : -INFINITY;
                if (row_b < num_valid_rows)
                    lse_accum[row_b] = (sum1 > 0.f) ? (rmax[1] + __logf(sum1)) * DD_LOG2E
                                                    : -INFINITY;
            }
        }
        __syncthreads();   // stats/smem free before the next batch reuses them
        // (no stat re-init needed: the next batch's nh==0 writers OVERWRITE
        //  sStatM/sStatS before any read)
    }
}

template <typename InputT>
inline void launch_dense_decode_mma(const DecodingParams& params, cudaStream_t stream) {
    const int num_m_blocks = (params.q_seq_per_hk + DD_BM - 1) / DD_BM;
    const dim3 grid(num_m_blocks, 1, params.num_sm_parts);
    constexpr size_t smem = sizeof(SmemPlanDenseMma<InputT>);
    static_assert(smem <= 99 * 1024, "dense decode mma smem exceeds 99KB");
    CHECK_CUDA(cudaFuncSetAttribute(dense_decode_mma_kernel<InputT>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    dense_decode_mma_kernel<InputT><<<grid, dim3(DD_THREADS), smem, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

// ============================================================================
// SMALL-M SINGLE-PASS tier [CFG>=2, q_seq_per_hk <= 16]: the bandwidth-row
// specialization (audit/design-dense-m16.md rationale in Benchmarks MD). On
// small-head decode shapes (h_q <= 16, s_q = 1) the BM=64 kernel wastes 4x mma
// on padded rows, re-reads every page's first 512 columns for PV (1.89x
// logical traffic), and exposes gather latency 17 times per block through a
// 2-stage pipeline. Shrinking the Q tile to 16 rows frees the smem to RETAIN
// all 8 V d-tiles + the rope tile: each page is streamed EXACTLY ONCE through
// a 9-deep cp.async pipeline (prologue issues all 9 tile gathers back-to-back)
// and PV reads the retained tiles -- no re-gather, and each warp's PV output
// slice (64 of the 512 columns) lives entirely in ITS OWN retained tile (warp
// w <-> sVret[w]): zero cross-warp contention.
//
// Warp layout (8 warps, M = 16 rows = the whole CTA):
//   QK: warp w owns the 8-token slice [w*8, w*8+8) -> one m16n8k16 mma per
//       16-col step (4 per tile); Sr[4] frag, token = w*8 + (lane%4)*2 (+1).
//   softmax: per-warp 4-lane shfl partials -> sM8/sS8[8][16] -> cross-warp
//       reduce by every thread (rows lane/4 and lane/4+8).
//   PV: A = sP [16 x 64] (x4 ldmatrix per 16-token chunk), B = sVret[w] only;
//       Or[8][4] = 32 regs for the warp's [16 x 64] output slice.
// Split-KV walk, causal limits, finals/partials conventions: identical to the
// BM=64 tier (scheduler consumption already wave-proven).
//
// Smem: sQ 18,432 + sVret 8 x 8,192 + sRope 8,192 + sP 2,048 + stats 1,024
//       -> 95,232 B <= 101,376, 1 CTA/SM.
// ============================================================================

// Stream one 64x64 tile into an arbitrary swizzled smem destination (the
// retained-tile variant of dd_load_kv_tile; same mapping, no plan coupling).
template <typename InputT>
__device__ __forceinline__ void dd_load_tile_to(InputT* dst, const InputT* page_ptr,
                                                int row_stride, int col_start,
                                                int valid_tokens) {
    const int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int g = tid + i * DD_THREADS;
        const int r = g >> 3;
        const int c = (g & 7) * 8;
        dd_cp16(dd_cvta(&dst[dd_swz(r, c, DD_DTILE)]),
                page_ptr + (int64_t)r * row_stride + col_start + c,
                (r < valid_tokens) ? 16 : 0);
    }
}

constexpr int DD_M16 = 16;   // CTA rows of the small-M tier

// Compile-time for: the 9-deep pipeline's cp.async waits are template
// immediates (dd_cp_wait<N>), so the tile index must be a constant expression
// -- #pragma unroll does NOT constify a runtime loop variable.
template <typename F, int... Is>
__device__ __forceinline__ void dd_static_for_impl(F&& f, std::integer_sequence<int, Is...>) {
    (f(std::integral_constant<int, Is>{}), ...);
}
template <int N, typename F>
__device__ __forceinline__ void dd_static_for(F&& f) {
    dd_static_for_impl(static_cast<F&&>(f), std::make_integer_sequence<int, N>{});
}

template <typename InputT>
struct SmemPlanDenseM16 {
    __align__(128) InputT sQ[DD_M16 * DD_DQK];            // 18,432 B, swz(576)
    __align__(128) InputT sVret[DD_NVTILES][DD_PAGE * DD_DTILE];  // 8 x 8,192 B, swz(64)
    __align__(128) InputT sRope[DD_PAGE * DD_DTILE];      // 8,192 B, swz(64)
    __align__(128) InputT sP[DD_M16 * DD_PAGE];           // 2,048 B, swz(64)
    __align__(128) float sM8[DD_NW][DD_M16];              // per-warp row-max partials
    __align__(128) float sS8[DD_NW][DD_M16];              // per-warp row-sum partials
};

template <typename InputT>
__global__ void __launch_bounds__(DD_THREADS, 1)
dense_decode_mma_m16_kernel(const DecodingParams params) {
    extern __shared__ char smem_raw[];
    SmemPlanDenseM16<InputT>& sm = *reinterpret_cast<SmemPlanDenseM16<InputT>*>(smem_raw);

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int part_idx = blockIdx.z;

    const int* meta = params.tile_scheduler_metadata_ptr + part_idx * TileSchedulerMetaDataSize;
    const int begin_idx = meta[0];
    const int begin_block = meta[1];
    const int end_idx = meta[2];
    const int end_block_last = meta[3];
    const int begin_n_split_idx = meta[4];

    const int num_valid_rows = min(params.q_seq_per_hk, DD_M16);   // grid.x == 1, row0 == 0
    const int row_a = lane / 4, row_b = row_a + 8;

    for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx) {
        const int seqlen_k = __ldg(params.seqlens_k_ptr + batch_idx);
        const int total_blocks = (max(seqlen_k, 1) + DD_PAGE - 1) / DD_PAGE;
        const int blk_lo = (batch_idx == begin_idx) ? begin_block : 0;
        const int blk_hi = (batch_idx == end_idx) ? end_block_last : total_blocks;
        const int blk_walk_hi = (seqlen_k > 0) ? blk_hi : blk_lo;   // zero-seqlen collapse
        const int n_split_idx = (batch_idx == begin_idx) ? begin_n_split_idx : 0;
        const int split_base = __ldg(params.num_splits_ptr + batch_idx);
        const bool no_split = (__ldg(params.num_splits_ptr + batch_idx + 1) - split_base) == 1;

        const InputT* q_ptr = reinterpret_cast<const InputT*>(params.q_ptr)
            + (int64_t)batch_idx * params.q_batch_stride;
        {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13)
            // 256-bit Q prologue (same guard/mapping as the BM=64 tier).
            const bool use_v8 = (params.q_row_stride % 16 == 0) &&
                                ((reinterpret_cast<uintptr_t>(q_ptr) & 31) == 0);
            if (use_v8) {
                #pragma unroll
                for (int i = 0; i < 3; ++i) {                  // 16*36 granules
                    const int g = tid + i * DD_THREADS;
                    if (g < DD_M16 * (DD_DQK / 16)) {
                        const int r = g / (DD_DQK / 16);
                        const int c = (g % (DD_DQK / 16)) * 16;
                        float4 lo = make_float4(0.f, 0.f, 0.f, 0.f), hi = lo;
                        if (r < num_valid_rows) {
                            const float* src = reinterpret_cast<const float*>(
                                q_ptr + (int64_t)r * params.q_row_stride + c);
                            asm volatile("ld.global.nc.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
                                : "=f"(lo.x), "=f"(lo.y), "=f"(lo.z), "=f"(lo.w),
                                  "=f"(hi.x), "=f"(hi.y), "=f"(hi.z), "=f"(hi.w) : "l"(src));
                        }
                        *reinterpret_cast<float4*>(&sm.sQ[dd_swz(r, c, DD_DQK)]) = lo;
                        *reinterpret_cast<float4*>(&sm.sQ[dd_swz(r, c + 8, DD_DQK)]) = hi;
                    }
                }
            } else
#endif
            {
                #pragma unroll
                for (int i = 0; i < 5; ++i) {                  // 16*72 = 1152 granules
                    const int g = tid + i * DD_THREADS;
                    if (g < DD_M16 * (DD_DQK / 8)) {
                        const int r = g / (DD_DQK / 8);
                        const int c = (g % (DD_DQK / 8)) * 8;
                        int4 v = make_int4(0, 0, 0, 0);
                        if (r < num_valid_rows)
                            v = *reinterpret_cast<const int4*>(
                                    q_ptr + (int64_t)r * params.q_row_stride + c);
                        *reinterpret_cast<int4*>(&sm.sQ[dd_swz(r, c, DD_DQK)]) = v;
                    }
                }
            }
        }

        int lim_a = seqlen_k, lim_b = seqlen_k;
        if (params.is_causal) {
            const int qa = min(row_a, params.q_seq_per_hk - 1) / params.q_head_per_hk;
            const int qb = min(row_b, params.q_seq_per_hk - 1) / params.q_head_per_hk;
            lim_a = min(seqlen_k, seqlen_k - params.s_q + 1 + qa);
            lim_b = min(seqlen_k, seqlen_k - params.s_q + 1 + qb);
        }

        float rmax[2] = {-INFINITY, -INFINITY}, rsum[2] = {0.f, 0.f};
        float Or[8][4];
        #pragma unroll
        for (int n = 0; n < 8; ++n) { Or[n][0] = Or[n][1] = Or[n][2] = Or[n][3] = 0.f; }

        const int* block_table_row = params.block_table
            + (int64_t)batch_idx * params.block_table_batch_stride;
        const InputT* k_base = reinterpret_cast<const InputT*>(params.k_ptr);

        __syncthreads();   // sQ visible; previous batch's smem fully consumed

        for (int blk = blk_lo; blk < blk_walk_hi; ++blk) {
            const int phys = __ldg(block_table_row + blk);
            const InputT* page_ptr = k_base + (int64_t)phys * params.k_batch_stride;
            const int tok0 = blk * DD_PAGE;
            const int valid_tokens = min(seqlen_k - tok0, DD_PAGE);

            // ---- SINGLE PASS over the page: 9 gathers issued back-to-back --
            // INVARIANT (audit verify-m16-2): the QK waits below are ABSOLUTE
            // constants (dd_cp_wait<8-dt>) premised on (a) an EMPTY cp.async
            // ledger at page entry and (b) EXACTLY these 9 uniform commits.
            // Any added prefetch/commit -- or making issuance thread-divergent
            // -- breaks every wait constant AND the loop-bottom WAR barrier at
            // once; a restructure must switch to relative waits.
            #pragma unroll
            for (int t = 0; t < DD_NVTILES; ++t) {
                dd_load_tile_to<InputT>(sm.sVret[t], page_ptr, params.k_row_stride,
                                        t * DD_DTILE, max(valid_tokens, 0));
                dd_cp_commit();
            }
            dd_load_tile_to<InputT>(sm.sRope, page_ptr, params.k_row_stride,
                                    DD_DV, max(valid_tokens, 0));
            dd_cp_commit();

            // ---- QK: warp w owns tokens [w*8, w*8+8); one mma per 16-col step
            float Sr[4] = {0.f, 0.f, 0.f, 0.f};
            dd_static_for<DD_NKTILES>([&](auto dtc) {
                constexpr int dt = decltype(dtc)::value;
                dd_cp_wait<DD_NKTILES - 1 - dt>();   // tile dt landed (deep pipeline)
                __syncthreads();
                const InputT* ktile = (dt < DD_NVTILES) ? sm.sVret[dt] : sm.sRope;
                #pragma unroll
                for (int dl = 0; dl < 4; ++dl) {
                    uint32_t Qr[4];
                    {
                        const int row = lane % 16;
                        const int col = dt * DD_DTILE + dl * DD_MMA_K + (lane / 16) * 8;
                        dd_ldm_x4(Qr, dd_cvta(&sm.sQ[dd_swz(row, col, DD_DQK)]));
                    }
                    uint32_t Kr[2];
                    const int krow = warp * 8 + (lane % 8);
                    const int kcol = dl * DD_MMA_K + ((lane / 8) % 2) * 8;
                    dd_ldm_x2(Kr, dd_cvta(&ktile[dd_swz(krow, kcol, DD_DTILE)]));
                    DDT<InputT>::mma(Sr, Qr, Kr, Sr);
                }
            });

            // ---- mask + scale + per-warp partial max ------------------------
            const float scale = params.scale_softmax;
            const int t0 = tok0 + warp * 8 + (lane % 4) * 2, t1 = t0 + 1;
            Sr[0] = (t0 < lim_a) ? Sr[0] * scale : -INFINITY;
            Sr[1] = (t1 < lim_a) ? Sr[1] * scale : -INFINITY;
            Sr[2] = (t0 < lim_b) ? Sr[2] * scale : -INFINITY;
            Sr[3] = (t1 < lim_b) ? Sr[3] * scale : -INFINITY;
            float tmax[2] = { fmaxf(Sr[0], Sr[1]), fmaxf(Sr[2], Sr[3]) };
            tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffff, tmax[0], 1));
            tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffff, tmax[0], 2));
            tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffff, tmax[1], 1));
            tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffff, tmax[1], 2));
            // one writer per row pair: lanes 0,4,...,28 hold rows lane/4 and lane/4+8
            if ((lane % 4) == 0) {
                sm.sM8[warp][row_a] = tmax[0];
                sm.sM8[warp][row_b] = tmax[1];
            }
            __syncthreads();
            float bmax0 = -INFINITY, bmax1 = -INFINITY;
            #pragma unroll
            for (int w = 0; w < DD_NW; ++w) {
                bmax0 = fmaxf(bmax0, sm.sM8[w][row_a]);
                bmax1 = fmaxf(bmax1, sm.sM8[w][row_b]);
            }

            // ---- online rescale + P -> sP -----------------------------------
            const float nmax0 = fmaxf(rmax[0], bmax0), nmax1 = fmaxf(rmax[1], bmax1);
            const float rs0 = (rmax[0] == -INFINITY) ? ((nmax0 == -INFINITY) ? 1.f : 0.f)
                                                     : __expf(rmax[0] - nmax0);
            const float rs1 = (rmax[1] == -INFINITY) ? ((nmax1 == -INFINITY) ? 1.f : 0.f)
                                                     : __expf(rmax[1] - nmax1);
            rmax[0] = nmax0; rmax[1] = nmax1;
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
                Or[n][0] *= rs0; Or[n][1] *= rs0; Or[n][2] *= rs1; Or[n][3] *= rs1;
            }

            const float p0 = (Sr[0] == -INFINITY) ? 0.f : __expf(Sr[0] - nmax0);
            const float p1 = (Sr[1] == -INFINITY) ? 0.f : __expf(Sr[1] - nmax0);
            const float p2 = (Sr[2] == -INFINITY) ? 0.f : __expf(Sr[2] - nmax1);
            const float p3 = (Sr[3] == -INFINITY) ? 0.f : __expf(Sr[3] - nmax1);
            float tsum[2] = { p0 + p1, p2 + p3 };
            tsum[0] += __shfl_xor_sync(0xffffffff, tsum[0], 1);
            tsum[0] += __shfl_xor_sync(0xffffffff, tsum[0], 2);
            tsum[1] += __shfl_xor_sync(0xffffffff, tsum[1], 1);
            tsum[1] += __shfl_xor_sync(0xffffffff, tsum[1], 2);
            if ((lane % 4) == 0) {
                sm.sS8[warp][row_a] = tsum[0];
                sm.sS8[warp][row_b] = tsum[1];
            }
            {
                const int pcol = warp * 8 + (lane % 4) * 2;
                *reinterpret_cast<uint32_t*>(&sm.sP[dd_swz(row_a, pcol, DD_PAGE)]) =
                    DDT<InputT>::pack2(p0, p1);
                *reinterpret_cast<uint32_t*>(&sm.sP[dd_swz(row_b, pcol, DD_PAGE)]) =
                    DDT<InputT>::pack2(p2, p3);
            }
            __syncthreads();   // sP + sS8 complete
            float bsum0 = 0.f, bsum1 = 0.f;
            #pragma unroll
            for (int w = 0; w < DD_NW; ++w) { bsum0 += sm.sS8[w][row_a]; bsum1 += sm.sS8[w][row_b]; }
            rsum[0] = rsum[0] * rs0 + bsum0;
            rsum[1] = rsum[1] * rs1 + bsum1;

            // ---- PV from the RETAINED tiles: warp w -> its own sVret[w] -----
            #pragma unroll
            for (int pk = 0; pk < 4; ++pk) {
                uint32_t Pr[4];
                {
                    const int prow = lane % 16;
                    const int pcol = pk * DD_MMA_K + (lane / 16) * 8;
                    dd_ldm_x4(Pr, dd_cvta(&sm.sP[dd_swz(prow, pcol, DD_PAGE)]));
                }
                // NOTE: the (lane/16)*8 term in vcol is DEAD for ldm_x2 (only
                // lanes 0-15 supply addresses) -- kept to mirror the proven
                // BM=64 pattern; do not "fix" it into a live offset.
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    uint32_t Vr[2];
                    const int vrow = pk * DD_MMA_K + (lane % 16);
                    const int vcol = n * DD_MMA_N + (lane / 16) * 8;   // warp's own tile
                    dd_ldm_x2_trans(Vr, dd_cvta(&sm.sVret[warp][dd_swz(vrow, vcol, DD_DTILE)]));
                    DDT<InputT>::mma(Or[n], Pr, Vr, Or[n]);
                }
                #pragma unroll
                for (int n = 4; n < 8; ++n) {
                    uint32_t Vr[2];
                    const int vrow = pk * DD_MMA_K + (lane % 16);
                    const int vcol = n * DD_MMA_N + (lane / 16) * 8;
                    dd_ldm_x2_trans(Vr, dd_cvta(&sm.sVret[warp][dd_swz(vrow, vcol, DD_DTILE)]));
                    DDT<InputT>::mma(Or[n], Pr, Vr, Or[n]);
                }
            }
            // Load-bearing barrier: retires every PV/sP reader AND (with the
            // wait<0> drain above) closes the empty-ledger page boundary the
            // next block's 9 gathers are premised on.
            __syncthreads();   // retained tiles + sP consumed before next block's gathers
        }

        // ---- epilogue (BM=64 conventions; warp w owns out cols [w*64, +64)) -
        const float sum0 = rsum[0], sum1 = rsum[1];
        const float inv0 = (sum0 > 0.f) ? 1.f / sum0 : 0.f;
        const float inv1 = (sum1 > 0.f) ? 1.f / sum1 : 0.f;

        if (no_split) {
            InputT* o_ptr = reinterpret_cast<InputT*>(params.o_ptr)
                + (int64_t)batch_idx * params.o_batch_stride;
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int ocol = warp * DD_DTILE + n * DD_MMA_N + (lane % 4) * 2;
                if (row_a < num_valid_rows)
                    *reinterpret_cast<uint32_t*>(
                        o_ptr + (int64_t)row_a * params.o_row_stride + ocol) =
                        DDT<InputT>::pack2(Or[n][0] * inv0, Or[n][1] * inv0);
                if (row_b < num_valid_rows)
                    *reinterpret_cast<uint32_t*>(
                        o_ptr + (int64_t)row_b * params.o_row_stride + ocol) =
                        DDT<InputT>::pack2(Or[n][2] * inv1, Or[n][3] * inv1);
            }
            if (warp == 0 && (lane % 4) == 0) {
                float* lse_ptr = reinterpret_cast<float*>(params.softmax_lse_ptr)
                    + (int64_t)batch_idx * params.q_seq_per_hk;      // h_k == 1
                if (row_a < num_valid_rows)
                    lse_ptr[row_a] = (sum0 > 0.f) ? (rmax[0] + __logf(sum0)) : 1e30f;
                if (row_b < num_valid_rows)
                    lse_ptr[row_b] = (sum1 > 0.f) ? (rmax[1] + __logf(sum1)) : 1e30f;
            }
        } else {
            const int split_row = split_base + n_split_idx;
            float* oaccum = reinterpret_cast<float*>(params.oaccum_ptr)
                + (int64_t)split_row * params.q_seq_per_hk * DD_DV;
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int ocol = warp * DD_DTILE + n * DD_MMA_N + (lane % 4) * 2;
                if (row_a < num_valid_rows) {
                    float* pa = oaccum + (int64_t)row_a * DD_DV + ocol;
                    pa[0] = Or[n][0] * inv0; pa[1] = Or[n][1] * inv0;
                }
                if (row_b < num_valid_rows) {
                    float* pb = oaccum + (int64_t)row_b * DD_DV + ocol;
                    pb[0] = Or[n][2] * inv1; pb[1] = Or[n][3] * inv1;
                }
            }
            if (warp == 0 && (lane % 4) == 0) {
                float* lse_accum = reinterpret_cast<float*>(params.softmax_lseaccum_ptr)
                    + (int64_t)split_row * params.q_seq_per_hk;
                if (row_a < num_valid_rows)
                    lse_accum[row_a] = (sum0 > 0.f) ? (rmax[0] + __logf(sum0)) * DD_LOG2E
                                                    : -INFINITY;
                if (row_b < num_valid_rows)
                    lse_accum[row_b] = (sum1 > 0.f) ? (rmax[1] + __logf(sum1)) * DD_LOG2E
                                                    : -INFINITY;
            }
        }
        __syncthreads();   // smem free before the next batch reuses it
    }
}

template <typename InputT>
inline void launch_dense_decode_mma_m16(const DecodingParams& params, cudaStream_t stream) {
    const dim3 grid(1, 1, params.num_sm_parts);
    constexpr size_t smem = sizeof(SmemPlanDenseM16<InputT>);
    static_assert(smem <= 99 * 1024, "dense decode m16 smem exceeds 99KB");
    CHECK_CUDA(cudaFuncSetAttribute(dense_decode_mma_m16_kernel<InputT>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    dense_decode_mma_m16_kernel<InputT><<<grid, dim3(DD_THREADS), smem, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

// ============================================================================
// M32 SINGLE-PASS TIER (16 < q_seq_per_hk <= 32; covers the 7b model's h=22).
// Same single-pass retained-V design as M16, extended to 32 Q rows by halving
// the token tile: each 64-token page is processed as TWO 32-token halves, each
// streamed from gmem exactly once through the same 9-deep cp.async pipeline
// (8 retained V half-tiles + rope half-tile; a byte is still read only once).
//
// Warp map (8 warps): rb = warp/4 owns row block [rb*16, rb*16+16);
//                     qd = warp%4 owns QK token slice [qd*8, qd*8+8) of the
//                     half AND the PV output column slice [qd*128, qd*128+128)
//                     (= retained tiles qd*2 and qd*2+1).
// QK rows == PV rows for every thread (block rb both phases), so the per-
// thread online-softmax state (rmax/rsum/Or) carries over from the M16 design
// unchanged; cross-warp stats combine within a row block over the 4 qd-quads
// (sM4/sS4 indexed [qd][absolute row] -- warps sharing qd write disjoint rows).
//
// Smem: sQ 36,864 + sVret 8 x 4,096 + sRope 4,096 + sP 4,096 + stats 1,024
//       -> 78,848 B <= 101,376, 1 CTA/SM.
// sP keeps stride 64 (tokens live in cols 0..31): dd_swz flips col bits 3-5,
// so a 32-wide row would swizzle OUT of the row -- the pad keeps the
// bijection in-row (same reason the bwd staging tiles pad to 64).
// ============================================================================

constexpr int DD_M32 = 32;        // CTA rows of this tier
constexpr int DD_HALF_PAGE = 32;  // token tile = half a 64-token page

// One 32x64 half-tile gather: exactly one 16B granule per thread (32 rows x 8
// granules = 256 = DD_THREADS). row0 selects which half of the page.
template <typename InputT>
__device__ __forceinline__ void dd_load_half_tile_to(InputT* dst, const InputT* page_ptr,
                                                     int row_stride, int col_start,
                                                     int row0, int valid_tokens) {
    const int tid = threadIdx.x;
    const int r = tid >> 3;
    const int c = (tid & 7) * 8;
    dd_cp16(dd_cvta(&dst[dd_swz(r, c, DD_DTILE)]),
            page_ptr + (int64_t)(row0 + r) * row_stride + col_start + c,
            (r < valid_tokens) ? 16 : 0);
}

template <typename InputT>
struct SmemPlanDenseM32 {
    __align__(128) InputT sQ[DD_M32 * DD_DQK];                        // 36,864 B, swz(576)
    __align__(128) InputT sVret[DD_NVTILES][DD_HALF_PAGE * DD_DTILE]; // 8 x 4,096 B, swz(64)
    __align__(128) InputT sRope[DD_HALF_PAGE * DD_DTILE];             // 4,096 B, swz(64)
    __align__(128) InputT sP[DD_M32 * DD_DTILE];                      // 4,096 B, swz(64); cols 0..31 live
    __align__(128) float sM4[4][DD_M32];                              // per-quad row-max partials
    __align__(128) float sS4[4][DD_M32];                              // per-quad row-sum partials
};

template <typename InputT>
__global__ void __launch_bounds__(DD_THREADS, 1)
dense_decode_mma_m32_kernel(const DecodingParams params) {
    extern __shared__ char smem_raw[];
    SmemPlanDenseM32<InputT>& sm = *reinterpret_cast<SmemPlanDenseM32<InputT>*>(smem_raw);

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int rb = warp / 4;          // row block (rows rb*16 .. rb*16+16)
    const int qd = warp % 4;          // QK token slice / PV column slice
    const int part_idx = blockIdx.z;
    // Multi-block generalization (CFG>=5): grid.x = ceil(q_seq_per_hk / 32),
    // each CTA single-passes the KV for its own 32-row band. Concurrent bands
    // of the same (batch, split) walk pages in lockstep, so L2 dedups their
    // streams exactly like the BM=64 tier's m-blocks -- but each band avoids
    // that tier's V re-read. row0 == 0 when grid.x == 1 (CFG=4 semantics).
    const int row0 = blockIdx.x * DD_M32;

    const int* meta = params.tile_scheduler_metadata_ptr + part_idx * TileSchedulerMetaDataSize;
    const int begin_idx = meta[0];
    const int begin_block = meta[1];
    const int end_idx = meta[2];
    const int end_block_last = meta[3];
    const int begin_n_split_idx = meta[4];

    const int num_valid_rows = min(params.q_seq_per_hk - row0, DD_M32);  // rows of THIS band
    const int row_a = rb * 16 + lane / 4, row_b = row_a + 8;       // BAND-LOCAL rows 0..31
    // Band-local rows index every smem structure (sQ/sP/sM4/sS4); row0 is
    // added ONLY at the gmem edges (Q source, causal fold, output rows).

    for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx) {
        const int seqlen_k = __ldg(params.seqlens_k_ptr + batch_idx);
        const int total_blocks = (max(seqlen_k, 1) + DD_PAGE - 1) / DD_PAGE;
        const int blk_lo = (batch_idx == begin_idx) ? begin_block : 0;
        const int blk_hi = (batch_idx == end_idx) ? end_block_last : total_blocks;
        const int blk_walk_hi = (seqlen_k > 0) ? blk_hi : blk_lo;   // zero-seqlen collapse
        const int n_split_idx = (batch_idx == begin_idx) ? begin_n_split_idx : 0;
        const int split_base = __ldg(params.num_splits_ptr + batch_idx);
        const bool no_split = (__ldg(params.num_splits_ptr + batch_idx + 1) - split_base) == 1;

        const InputT* q_ptr = reinterpret_cast<const InputT*>(params.q_ptr)
            + (int64_t)batch_idx * params.q_batch_stride;
        {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13)
            // 256-bit Q prologue (same guard/mapping as the BM=64/M16 tiers).
            const bool use_v8 = (params.q_row_stride % 16 == 0) &&
                                ((reinterpret_cast<uintptr_t>(q_ptr) & 31) == 0);
            if (use_v8) {
                #pragma unroll
                for (int i = 0; i < 5; ++i) {                  // 32*36 = 1152 granules
                    const int g = tid + i * DD_THREADS;
                    if (g < DD_M32 * (DD_DQK / 16)) {
                        const int r = g / (DD_DQK / 16);
                        const int c = (g % (DD_DQK / 16)) * 16;
                        float4 lo = make_float4(0.f, 0.f, 0.f, 0.f), hi = lo;
                        if (r < num_valid_rows) {
                            const float* src = reinterpret_cast<const float*>(
                                q_ptr + (int64_t)(row0 + r) * params.q_row_stride + c);
                            asm volatile("ld.global.nc.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
                                : "=f"(lo.x), "=f"(lo.y), "=f"(lo.z), "=f"(lo.w),
                                  "=f"(hi.x), "=f"(hi.y), "=f"(hi.z), "=f"(hi.w) : "l"(src));
                        }
                        *reinterpret_cast<float4*>(&sm.sQ[dd_swz(r, c, DD_DQK)]) = lo;
                        *reinterpret_cast<float4*>(&sm.sQ[dd_swz(r, c + 8, DD_DQK)]) = hi;
                    }
                }
            } else
#endif
            {
                #pragma unroll
                for (int i = 0; i < 9; ++i) {                  // 32*72 = 2304 granules
                    const int g = tid + i * DD_THREADS;
                    if (g < DD_M32 * (DD_DQK / 8)) {
                        const int r = g / (DD_DQK / 8);
                        const int c = (g % (DD_DQK / 8)) * 8;
                        int4 v = make_int4(0, 0, 0, 0);
                        if (r < num_valid_rows)
                            v = *reinterpret_cast<const int4*>(
                                    q_ptr + (int64_t)(row0 + r) * params.q_row_stride + c);
                        *reinterpret_cast<int4*>(&sm.sQ[dd_swz(r, c, DD_DQK)]) = v;
                    }
                }
            }
        }

        int lim_a = seqlen_k, lim_b = seqlen_k;
        if (params.is_causal) {
            const int qa = min(row0 + row_a, params.q_seq_per_hk - 1) / params.q_head_per_hk;
            const int qb = min(row0 + row_b, params.q_seq_per_hk - 1) / params.q_head_per_hk;
            lim_a = min(seqlen_k, seqlen_k - params.s_q + 1 + qa);
            lim_b = min(seqlen_k, seqlen_k - params.s_q + 1 + qb);
        }

        float rmax[2] = {-INFINITY, -INFINITY}, rsum[2] = {0.f, 0.f};
        float Or[2][8][4];
        #pragma unroll
        for (int t = 0; t < 2; ++t)
            #pragma unroll
            for (int n = 0; n < 8; ++n) { Or[t][n][0] = Or[t][n][1] = Or[t][n][2] = Or[t][n][3] = 0.f; }

        const int* block_table_row = params.block_table
            + (int64_t)batch_idx * params.block_table_batch_stride;
        const InputT* k_base = reinterpret_cast<const InputT*>(params.k_ptr);

        __syncthreads();   // sQ visible; previous batch's smem fully consumed

        for (int blk = blk_lo; blk < blk_walk_hi; ++blk) {
            const int phys = __ldg(block_table_row + blk);
            const InputT* page_ptr = k_base + (int64_t)phys * params.k_batch_stride;

            // Both halves ALWAYS run (CTA-uniform barrier count); an empty
            // second half zero-fills with src-size-0 and fully masks to -inf.
            #pragma unroll
            for (int ht = 0; ht < 2; ++ht) {
                const int half_tok0 = blk * DD_PAGE + ht * DD_HALF_PAGE;
                const int valid_half = min(max(seqlen_k - half_tok0, 0), DD_HALF_PAGE);

                // ---- SINGLE PASS over the half-page: 9 gathers back-to-back.
                // INVARIANT (mirrors the M16 tier, audit verify-m16-2): the QK
                // waits below are ABSOLUTE constants premised on an empty
                // cp.async ledger at half entry and EXACTLY these 9 uniform
                // commits; restructure requires relative waits.
                #pragma unroll
                for (int t = 0; t < DD_NVTILES; ++t) {
                    dd_load_half_tile_to<InputT>(sm.sVret[t], page_ptr, params.k_row_stride,
                                                 t * DD_DTILE, ht * DD_HALF_PAGE, valid_half);
                    dd_cp_commit();
                }
                dd_load_half_tile_to<InputT>(sm.sRope, page_ptr, params.k_row_stride,
                                             DD_DV, ht * DD_HALF_PAGE, valid_half);
                dd_cp_commit();

                // ---- QK: quad qd owns tokens [qd*8, qd*8+8) of the half;
                //      rows = block rb; one mma per 16-col step.
                float Sr[4] = {0.f, 0.f, 0.f, 0.f};
                dd_static_for<DD_NKTILES>([&](auto dtc) {
                    constexpr int dt = decltype(dtc)::value;
                    dd_cp_wait<DD_NKTILES - 1 - dt>();   // tile dt landed
                    __syncthreads();
                    const InputT* ktile = (dt < DD_NVTILES) ? sm.sVret[dt] : sm.sRope;
                    #pragma unroll
                    for (int dl = 0; dl < 4; ++dl) {
                        uint32_t Qr[4];
                        {
                            const int row = rb * 16 + (lane % 16);
                            const int col = dt * DD_DTILE + dl * DD_MMA_K + (lane / 16) * 8;
                            dd_ldm_x4(Qr, dd_cvta(&sm.sQ[dd_swz(row, col, DD_DQK)]));
                        }
                        uint32_t Kr[2];
                        const int krow = qd * 8 + (lane % 8);
                        const int kcol = dl * DD_MMA_K + ((lane / 8) % 2) * 8;
                        dd_ldm_x2(Kr, dd_cvta(&ktile[dd_swz(krow, kcol, DD_DTILE)]));
                        DDT<InputT>::mma(Sr, Qr, Kr, Sr);
                    }
                });

                // ---- mask + scale + per-quad partial max --------------------
                const float scale = params.scale_softmax;
                const int t0 = half_tok0 + qd * 8 + (lane % 4) * 2, t1 = t0 + 1;
                Sr[0] = (t0 < lim_a) ? Sr[0] * scale : -INFINITY;
                Sr[1] = (t1 < lim_a) ? Sr[1] * scale : -INFINITY;
                Sr[2] = (t0 < lim_b) ? Sr[2] * scale : -INFINITY;
                Sr[3] = (t1 < lim_b) ? Sr[3] * scale : -INFINITY;
                float tmax[2] = { fmaxf(Sr[0], Sr[1]), fmaxf(Sr[2], Sr[3]) };
                tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffff, tmax[0], 1));
                tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffff, tmax[0], 2));
                tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffff, tmax[1], 1));
                tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffff, tmax[1], 2));
                // one writer per row pair: lanes 0,4,...,28 hold rows lane/4 (+8);
                // warps sharing qd write DISJOINT absolute rows (rb differs)
                if ((lane % 4) == 0) {
                    sm.sM4[qd][row_a] = tmax[0];
                    sm.sM4[qd][row_b] = tmax[1];
                }
                __syncthreads();
                float bmax0 = -INFINITY, bmax1 = -INFINITY;
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    bmax0 = fmaxf(bmax0, sm.sM4[q][row_a]);
                    bmax1 = fmaxf(bmax1, sm.sM4[q][row_b]);
                }

                // ---- online rescale + P -> sP -------------------------------
                const float nmax0 = fmaxf(rmax[0], bmax0), nmax1 = fmaxf(rmax[1], bmax1);
                const float rs0 = (rmax[0] == -INFINITY) ? ((nmax0 == -INFINITY) ? 1.f : 0.f)
                                                         : __expf(rmax[0] - nmax0);
                const float rs1 = (rmax[1] == -INFINITY) ? ((nmax1 == -INFINITY) ? 1.f : 0.f)
                                                         : __expf(rmax[1] - nmax1);
                rmax[0] = nmax0; rmax[1] = nmax1;
                #pragma unroll
                for (int t = 0; t < 2; ++t)
                    #pragma unroll
                    for (int n = 0; n < 8; ++n) {
                        Or[t][n][0] *= rs0; Or[t][n][1] *= rs0;
                        Or[t][n][2] *= rs1; Or[t][n][3] *= rs1;
                    }

                const float p0 = (Sr[0] == -INFINITY) ? 0.f : __expf(Sr[0] - nmax0);
                const float p1 = (Sr[1] == -INFINITY) ? 0.f : __expf(Sr[1] - nmax0);
                const float p2 = (Sr[2] == -INFINITY) ? 0.f : __expf(Sr[2] - nmax1);
                const float p3 = (Sr[3] == -INFINITY) ? 0.f : __expf(Sr[3] - nmax1);
                float tsum[2] = { p0 + p1, p2 + p3 };
                tsum[0] += __shfl_xor_sync(0xffffffff, tsum[0], 1);
                tsum[0] += __shfl_xor_sync(0xffffffff, tsum[0], 2);
                tsum[1] += __shfl_xor_sync(0xffffffff, tsum[1], 1);
                tsum[1] += __shfl_xor_sync(0xffffffff, tsum[1], 2);
                if ((lane % 4) == 0) {
                    sm.sS4[qd][row_a] = tsum[0];
                    sm.sS4[qd][row_b] = tsum[1];
                }
                {
                    const int pcol = qd * 8 + (lane % 4) * 2;
                    *reinterpret_cast<uint32_t*>(&sm.sP[dd_swz(row_a, pcol, DD_DTILE)]) =
                        DDT<InputT>::pack2(p0, p1);
                    *reinterpret_cast<uint32_t*>(&sm.sP[dd_swz(row_b, pcol, DD_DTILE)]) =
                        DDT<InputT>::pack2(p2, p3);
                }
                __syncthreads();   // sP + sS4 complete
                float bsum0 = 0.f, bsum1 = 0.f;
                #pragma unroll
                for (int q = 0; q < 4; ++q) { bsum0 += sm.sS4[q][row_a]; bsum1 += sm.sS4[q][row_b]; }
                rsum[0] = rsum[0] * rs0 + bsum0;
                rsum[1] = rsum[1] * rs1 + bsum1;

                // ---- PV from the RETAINED half-tiles: warp -> rows of block
                //      rb x cols [qd*128, qd*128+128) = tiles qd*2, qd*2+1
                #pragma unroll
                for (int pk = 0; pk < 2; ++pk) {
                    uint32_t Pr[4];
                    {
                        const int prow = rb * 16 + (lane % 16);
                        const int pcol = pk * DD_MMA_K + (lane / 16) * 8;
                        dd_ldm_x4(Pr, dd_cvta(&sm.sP[dd_swz(prow, pcol, DD_DTILE)]));
                    }
                    #pragma unroll
                    for (int t = 0; t < 2; ++t) {
                        const InputT* vtile = sm.sVret[qd * 2 + t];
                        #pragma unroll
                        for (int n = 0; n < 8; ++n) {
                            uint32_t Vr[2];
                            const int vrow = pk * DD_MMA_K + (lane % 16);
                            const int vcol = n * DD_MMA_N;   // x2: lanes 0-15 address
                            dd_ldm_x2_trans(Vr, dd_cvta(&vtile[dd_swz(vrow, vcol, DD_DTILE)]));
                            DDT<InputT>::mma(Or[t][n], Pr, Vr, Or[t][n]);
                        }
                    }
                }
                // Load-bearing barrier: retires every PV/sP reader AND (with
                // the wait<0> drain above) closes the empty-ledger boundary the
                // next half's 9 gathers are premised on.
                __syncthreads();
            }
        }

        // ---- epilogue (BM=64 conventions; warp -> rows of block rb x cols
        //      [qd*128, qd*128+128)) --------------------------------------
        const float sum0 = rsum[0], sum1 = rsum[1];
        const float inv0 = (sum0 > 0.f) ? 1.f / sum0 : 0.f;
        const float inv1 = (sum1 > 0.f) ? 1.f / sum1 : 0.f;

        if (no_split) {
            InputT* o_ptr = reinterpret_cast<InputT*>(params.o_ptr)
                + (int64_t)batch_idx * params.o_batch_stride;
            #pragma unroll
            for (int t = 0; t < 2; ++t)
                #pragma unroll
                for (int n = 0; n < 8; ++n) {
                    const int ocol = qd * 128 + t * DD_DTILE + n * DD_MMA_N + (lane % 4) * 2;
                    if (row_a < num_valid_rows)
                        *reinterpret_cast<uint32_t*>(
                            o_ptr + (int64_t)(row0 + row_a) * params.o_row_stride + ocol) =
                            DDT<InputT>::pack2(Or[t][n][0] * inv0, Or[t][n][1] * inv0);
                    if (row_b < num_valid_rows)
                        *reinterpret_cast<uint32_t*>(
                            o_ptr + (int64_t)(row0 + row_b) * params.o_row_stride + ocol) =
                            DDT<InputT>::pack2(Or[t][n][2] * inv1, Or[t][n][3] * inv1);
                }
            // lse writers: warp 0 covers rows 0-15, warp 4 rows 16-31 (their
            // rmax/rsum are the block stats, identical across the 4 quads)
            if ((warp == 0 || warp == 4) && (lane % 4) == 0) {
                float* lse_ptr = reinterpret_cast<float*>(params.softmax_lse_ptr)
                    + (int64_t)batch_idx * params.q_seq_per_hk;      // h_k == 1
                if (row_a < num_valid_rows)
                    lse_ptr[row0 + row_a] = (sum0 > 0.f) ? (rmax[0] + __logf(sum0)) : 1e30f;
                if (row_b < num_valid_rows)
                    lse_ptr[row0 + row_b] = (sum1 > 0.f) ? (rmax[1] + __logf(sum1)) : 1e30f;
            }
        } else {
            const int split_row = split_base + n_split_idx;
            float* oaccum = reinterpret_cast<float*>(params.oaccum_ptr)
                + (int64_t)split_row * params.q_seq_per_hk * DD_DV;
            #pragma unroll
            for (int t = 0; t < 2; ++t)
                #pragma unroll
                for (int n = 0; n < 8; ++n) {
                    const int ocol = qd * 128 + t * DD_DTILE + n * DD_MMA_N + (lane % 4) * 2;
                    if (row_a < num_valid_rows) {
                        float* pa = oaccum + (int64_t)(row0 + row_a) * DD_DV + ocol;
                        pa[0] = Or[t][n][0] * inv0; pa[1] = Or[t][n][1] * inv0;
                    }
                    if (row_b < num_valid_rows) {
                        float* pb = oaccum + (int64_t)(row0 + row_b) * DD_DV + ocol;
                        pb[0] = Or[t][n][2] * inv1; pb[1] = Or[t][n][3] * inv1;
                    }
                }
            if ((warp == 0 || warp == 4) && (lane % 4) == 0) {
                float* lse_accum = reinterpret_cast<float*>(params.softmax_lseaccum_ptr)
                    + (int64_t)split_row * params.q_seq_per_hk;
                if (row_a < num_valid_rows)
                    lse_accum[row0 + row_a] = (sum0 > 0.f) ? (rmax[0] + __logf(sum0)) * DD_LOG2E
                                                           : -INFINITY;
                if (row_b < num_valid_rows)
                    lse_accum[row0 + row_b] = (sum1 > 0.f) ? (rmax[1] + __logf(sum1)) * DD_LOG2E
                                                           : -INFINITY;
            }
        }
        __syncthreads();   // smem free before the next batch reuses it
    }
}

template <typename InputT>
inline void launch_dense_decode_mma_m32(const DecodingParams& params, cudaStream_t stream) {
    // grid.x = one CTA per 32-row band. Rung 4 only dispatches q_seq_per_hk
    // <= 32 (grid.x == 1, the certified single-band form); rung 5 lifts the
    // gate and runs every dense shape as single-pass bands.
    const int m_blocks = (params.q_seq_per_hk + DD_M32 - 1) / DD_M32;
    const dim3 grid(m_blocks, 1, params.num_sm_parts);
    constexpr size_t smem = sizeof(SmemPlanDenseM32<InputT>);
    static_assert(smem <= 99 * 1024, "dense decode m32 smem exceeds 99KB");
    CHECK_CUDA(cudaFuncSetAttribute(dense_decode_mma_m32_kernel<InputT>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    dense_decode_mma_m32_kernel<InputT><<<grid, dim3(DD_THREADS), smem, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace dense_decode_mma
}  // namespace sm120
