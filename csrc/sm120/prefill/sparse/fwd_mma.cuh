// ============================================================================
// SPARSE PREFILL FORWARD via raw mma.sync + ldmatrix + cp.async (SM120)  [CFG>=4]
// ============================================================================
// The F1 tier of audit/design-prefill-configs.md: the sparse gather grafted onto the
// PROVEN dense mma.sync machinery of csrc/sm120/prefill/dense/fmha_fwd_mma_sm120.cuh
// (80%-of-peak precedent). Faithful to the authors' sm90 sparse fwd:
//   - in-kernel index gather (NO materialized workspace, O(1) extra memory)
//   - V IS K's first 512 columns (SmemLayoutHalfV innovation -> V tiles re-gather the
//     same rows, cols 0..511)
//   - online softmax, invalid indices masked to -inf, max_logits + 2-based LSE outputs
//
// Structure (256 threads / 8 warps, grid = (h_q/64) * s_q like the WMMA kernel):
//   sQ  [64 heads x 576]  swizzled, loaded ONCE per CTA (reused topk/64 * 17 times)
//   sKV [64 tok x 64]     2-stage cp.async pipeline, gathered by sparse indices;
//                         serves the 9 K d-tiles of QK then the 8 V d-tiles of PV
//   sP  [64 x 64]         bf16 P tile (swizzled) so all 8 warps run PV
//   S-phase: warp (ms, nh) = (w&3, w>>2): rows ms*16..+16, n8-tiles nh*4..+4 -> 16 S regs
//   PV-phase: warp (ms, sub): every V tile split into two 32-col halves -> all warps busy
//             O regs: 8 tiles x 4 n8 x 4 = 128 fp32/thread (full unroll, register-resident)
// ============================================================================
#pragma once

#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

namespace sm120 {
namespace sparse_mma {

constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int SM_NW = 8;                  // warps
constexpr int SM_THREADS = SM_NW * 32;    // 256
constexpr int SM_BH = 64;                 // head rows per CTA
constexpr int SM_BTOPK = 64;              // gathered tokens per block
constexpr int SM_DQK = 576, SM_DV = 512, SM_DTILE = 64;
constexpr int SM_NKTILES = SM_DQK / SM_DTILE;   // 9
constexpr int SM_NVTILES = SM_DV / SM_DTILE;    // 8
constexpr float SM_LOG2E = 1.4426950408889634f;

// ---- PTX primitives (identical to the proven dense mma kernel) --------------
__device__ __forceinline__ uint32_t s_cvta(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void s_ldm_x4(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void s_ldm_x2(uint32_t (&r)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void s_ldm_x2_trans(uint32_t (&r)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void s_mma(float (&d)[4], const uint32_t (&a)[4],
                                      const uint32_t (&b)[2], const float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
// Predicated gather copy: src-size 0 zero-fills the destination (authors' sm90 trick for
// invalid topk lanes -- no branch, no OOB dereference of a poisoned row).
__device__ __forceinline__ void s_cp16(uint32_t sa, const void* g, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 ::"r"(sa), "l"(g), "r"(src_bytes));
}
__device__ __forceinline__ void s_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void s_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// XOR swizzle (identical math to the dense kernel; valid for any dim multiple of 64).
__device__ __forceinline__ int s_swz(int r, int c, int dim) {
    return r * dim + (c ^ ((r & 7) << 3));
}

// ---- smem plan --------------------------------------------------------------
struct SmemPlanMma {
    __align__(128) __nv_bfloat16 sQ[SM_BH * SM_DQK];            // 73,728 B, swizzled(576)
    __align__(128) __nv_bfloat16 sKV[2][SM_BTOPK * SM_DTILE];   // 2 x 8,192 B, swizzled(64)
    __align__(128) __nv_bfloat16 sP[SM_BH * SM_BTOPK];          // 8,192 B, swizzled(64)
    __align__(128) float sStatM[SM_BH];                          // cross-warp row max
    __align__(128) float sStatS[SM_BH];                          // cross-warp row sum
    __align__(128) const __nv_bfloat16* sTokPtr[SM_BTOPK];       // gather base per token
    __align__(128) int8_t sValid[SM_BTOPK];
};
static_assert(sizeof(SmemPlanMma) <= 99 * 1024, "sparse mma smem exceeds 99KB");

// Gather one 64-token x 64-col tile (bf16 rows addressed by sTokPtr) into stage `st`,
// swizzled, via predicated cp.async. 64*64 elems = 512 x 16B granules / 256 thr = 2 each.
__device__ __forceinline__ void gather_tile_cp(SmemPlanMma& sm, int st, int col_start) {
    const int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int g = tid + i * SM_THREADS;         // granule 0..511
        const int r = g >> 3;                       // token row 0..63
        const int c = (g & 7) * 8;                  // col 0..56 step 8
        const __nv_bfloat16* tp = sm.sTokPtr[r];
        // Invalid token: tp points at row 0 of the pool (a legal address) but src-size 0
        // zero-fills, so poisoned rows are never read (NaN trap).
        s_cp16(s_cvta(&sm.sKV[st][s_swz(r, c, SM_DTILE)]),
               tp + col_start + c, sm.sValid[r] ? 16 : 0);
    }
}

// ---- kernel -----------------------------------------------------------------
__global__ void __launch_bounds__(SM_THREADS, 1)
sparse_prefill_fwd_mma_kernel(const SparsePrefillParams params) {
    extern __shared__ char smem_raw[];
    SmemPlanMma& sm = *reinterpret_cast<SmemPlanMma*>(smem_raw);

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int num_h_blocks = params.h_q / SM_BH;
    const int h_block = blockIdx.x % num_h_blocks;
    const int s_q_idx = blockIdx.x / num_h_blocks;

    const __nv_bfloat16* q_ptr = reinterpret_cast<const __nv_bfloat16*>(params.q)
        + (int64_t)s_q_idx * params.stride_q_s_q
        + (int64_t)h_block * SM_BH * params.stride_q_h_q;
    const __nv_bfloat16* kv_base = reinterpret_cast<const __nv_bfloat16*>(params.kv);
    const int* idx_row = params.indices + (int64_t)s_q_idx * params.stride_indices_s_q;
    __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(params.out)
        + (int64_t)s_q_idx * params.h_q * SM_DV + (int64_t)h_block * SM_BH * SM_DV;

    // ---- prologue: resident swizzled Q [64 x 576], 16B vectors -------------
    {
        // 64*576/8 = 4608 granules / 256 threads = 18 per thread
        #pragma unroll
        for (int i = 0; i < 18; ++i) {
            const int g = tid + i * SM_THREADS;
            const int r = g / (SM_DQK / 8);
            const int c = (g % (SM_DQK / 8)) * 8;
            *reinterpret_cast<int4*>(&sm.sQ[s_swz(r, c, SM_DQK)]) =
                *reinterpret_cast<const int4*>(q_ptr + (int64_t)r * params.stride_q_h_q + c);
        }
    }

    // S-phase warp map: ms = warp & 3 (row strip), nh = warp >> 2 (n-half of 8 n8 tiles)
    const int ms = warp & 3, nh = warp >> 2;
    const int wrow = ms * MMA_M;

    // Online-softmax state per lane: rows wrow + lane/4 and +8 (mma C layout)
    float rmax[2] = {-INFINITY, -INFINITY}, rsum[2] = {0.f, 0.f};
    // O accumulator: 8 v-tiles x 4 n8 (32-col half per tile) x 4 = 128 fp32
    float Or[SM_NVTILES][4][4];
    #pragma unroll
    for (int t = 0; t < SM_NVTILES; ++t)
        #pragma unroll
        for (int n = 0; n < 4; ++n) { Or[t][n][0] = Or[t][n][1] = Or[t][n][2] = Or[t][n][3] = 0.f; }

    const int num_topk_blocks = (params.topk + SM_BTOPK - 1) / SM_BTOPK;

    for (int kb = 0; kb < num_topk_blocks; ++kb) {
        // ---- token pointers + validity (once per block) --------------------
        if (tid < SM_BTOPK) {
            const int k = kb * SM_BTOPK + tid;
            const int token = (k < params.topk) ? idx_row[k] : -1;
            const bool ok = (token >= 0 && token < params.s_kv);
            sm.sValid[tid] = ok ? 1 : 0;
            sm.sTokPtr[tid] = ok ? (kv_base + (int64_t)token * params.stride_kv_s_kv)
                                 : kv_base;   // legal address; src-size 0 makes it unread
        }
        __syncthreads();

        // ---- QK: S[64h x 64t] accumulated in registers over 9 pipelined d-tiles
        float Sr[4][4];   // this warp's n-half: 4 n8 tiles
        #pragma unroll
        for (int n = 0; n < 4; ++n) { Sr[n][0] = Sr[n][1] = Sr[n][2] = Sr[n][3] = 0.f; }

        gather_tile_cp(sm, 0, 0);
        s_cp_commit();
        for (int dt = 0; dt < SM_NKTILES; ++dt) {
            const int cur = dt & 1;
            if (dt + 1 < SM_NKTILES) {
                gather_tile_cp(sm, cur ^ 1, (dt + 1) * SM_DTILE);
                s_cp_commit();
                s_cp_wait<1>();
            } else {
                s_cp_wait<0>();
            }
            __syncthreads();
            #pragma unroll
            for (int dl = 0; dl < 4; ++dl) {                       // 4 k16 per 64-col tile
                uint32_t Qr[4];
                {
                    const int row = wrow + (lane % 16);
                    const int col = dt * SM_DTILE + dl * MMA_K + (lane / 16) * 8;
                    s_ldm_x4(Qr, s_cvta(&sm.sQ[s_swz(row, col, SM_DQK)]));
                }
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    uint32_t Kr[2];
                    const int krow = (nh * 4 + n) * MMA_N + (lane % 8);
                    const int kcol = dl * MMA_K + ((lane / 8) % 2) * 8;
                    s_ldm_x2(Kr, s_cvta(&sm.sKV[cur][s_swz(krow, kcol, SM_DTILE)]));
                    s_mma(Sr[n], Qr, Kr, Sr[n]);
                }
            }
            __syncthreads();   // stage `cur` consumed before dt+2 overwrites it
        }

        // ---- mask invalid tokens, scale, per-lane partial max ---------------
        const float scale = params.sm_scale;
        float tmax[2] = {-INFINITY, -INFINITY};
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int col0 = (nh * 4 + n) * MMA_N + (lane % 4) * 2;
            const bool v0 = sm.sValid[col0] != 0, v1 = sm.sValid[col0 + 1] != 0;
            Sr[n][0] = v0 ? Sr[n][0] * scale : -INFINITY;
            Sr[n][1] = v1 ? Sr[n][1] * scale : -INFINITY;
            Sr[n][2] = v0 ? Sr[n][2] * scale : -INFINITY;
            Sr[n][3] = v1 ? Sr[n][3] * scale : -INFINITY;
            tmax[0] = fmaxf(tmax[0], fmaxf(Sr[n][0], Sr[n][1]));
            tmax[1] = fmaxf(tmax[1], fmaxf(Sr[n][2], Sr[n][3]));
        }
        tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffff, tmax[0], 1));
        tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffff, tmax[0], 2));
        tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffff, tmax[1], 1));
        tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffff, tmax[1], 2));

        // ---- cross-warp (n-half) row-max combine via smem -------------------
        // Writer: lane%4==0 holds the strip's per-half max for rows lane/4 and lane/4+8.
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
        const float bmax0 = sm.sStatM[wrow + lane / 4];
        const float bmax1 = sm.sStatM[wrow + lane / 4 + 8];

        // ---- online rescale + P = exp(S - max), pack to smem ---------------
        const float nmax0 = fmaxf(rmax[0], bmax0), nmax1 = fmaxf(rmax[1], bmax1);
        const float rs0 = (rmax[0] == -INFINITY) ? ((nmax0 == -INFINITY) ? 1.f : 0.f)
                                                 : __expf(rmax[0] - nmax0);
        const float rs1 = (rmax[1] == -INFINITY) ? ((nmax1 == -INFINITY) ? 1.f : 0.f)
                                                 : __expf(rmax[1] - nmax1);
        rmax[0] = nmax0; rmax[1] = nmax1;
        #pragma unroll
        for (int t = 0; t < SM_NVTILES; ++t)
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
            // C-layout element (row, col): rows wrow+lane/4 (+8), cols (nh*4+n)*8+(lane%4)*2 (+1)
            const int prow_a = wrow + lane / 4, prow_b = prow_a + 8;
            const int pcol = (nh * 4 + n) * MMA_N + (lane % 4) * 2;
            *reinterpret_cast<__nv_bfloat162*>(&sm.sP[s_swz(prow_a, pcol, SM_BTOPK)]) =
                __floats2bfloat162_rn(p0, p1);
            *reinterpret_cast<__nv_bfloat162*>(&sm.sP[s_swz(prow_b, pcol, SM_BTOPK)]) =
                __floats2bfloat162_rn(p2, p3);
        }
        rsum[0] = rsum[0] * rs0 + tsum[0];
        rsum[1] = rsum[1] * rs1 + tsum[1];
        // NOTE: rsum is per-lane over this warp's n-half; the epilogue reduces across the
        // 4 sharing lanes AND the two n-halves (via sStatS) before normalizing.

        // ---- PV: stream 8 V tiles (same gathered rows, cols 0..511) --------
        // sP swizzle write/read pair: 2-elem (4B) granularity stays inside the 16B granule
        // because the swizzle only permutes 16B granules (bits 3-5 of the element index).
        gather_tile_cp(sm, 0, 0 * SM_DTILE);   // V tile 0 = cols [0,64)
        s_cp_commit();
        for (int vt = 0; vt < SM_NVTILES; ++vt) {
            const int cur = vt & 1;
            if (vt + 1 < SM_NVTILES) {
                gather_tile_cp(sm, cur ^ 1, (vt + 1) * SM_DTILE);
                s_cp_commit();
                s_cp_wait<1>();
            } else {
                s_cp_wait<0>();
            }
            __syncthreads();
            // warp (ms, sub=nh): this tile's 32-col half [sub*32, sub*32+32)
            #pragma unroll
            for (int pk = 0; pk < 4; ++pk) {                 // contract 64 tokens = 4 k16
                uint32_t Pr[4];
                {
                    const int prow = wrow + (lane % 16);
                    const int pcol = pk * MMA_K + (lane / 16) * 8;
                    s_ldm_x4(Pr, s_cvta(&sm.sP[s_swz(prow, pcol, SM_BTOPK)]));
                }
                #pragma unroll
                for (int n = 0; n < 4; ++n) {                // 4 n8 = this warp's 32 cols
                    uint32_t Vr[2];
                    const int vrow = pk * MMA_K + (lane % 16);
                    const int vcol = nh * 32 + n * MMA_N + (lane / 16) * 8;
                    s_ldm_x2_trans(Vr, s_cvta(&sm.sKV[cur][s_swz(vrow, vcol, SM_DTILE)]));
                    s_mma(Or[vt][n], Pr, Vr, Or[vt][n]);
                }
            }
            __syncthreads();
        }
    }

    // ---- epilogue -----------------------------------------------------------
    // rsum: reduce across the 4 lanes sharing each row, then across the two n-halves.
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
        atomicAdd(&sm.sStatS[wrow + lane / 4], rsum[0]);
        atomicAdd(&sm.sStatS[wrow + lane / 4 + 8], rsum[1]);
    }
    __syncthreads();
    const int row_a = wrow + lane / 4, row_b = row_a + 8;
    const float sum0 = sm.sStatS[row_a], sum1 = sm.sStatS[row_b];
    const float inv0 = (sum0 > 0.f) ? 1.f / sum0 : 0.f;
    const float inv1 = (sum1 > 0.f) ? 1.f / sum1 : 0.f;

    #pragma unroll
    for (int t = 0; t < SM_NVTILES; ++t)
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int ocol = t * SM_DTILE + nh * 32 + n * MMA_N + (lane % 4) * 2;
            *reinterpret_cast<__nv_bfloat162*>(out_ptr + (int64_t)row_a * SM_DV + ocol) =
                __floats2bfloat162_rn(Or[t][n][0] * inv0, Or[t][n][1] * inv0);
            *reinterpret_cast<__nv_bfloat162*>(out_ptr + (int64_t)row_b * SM_DV + ocol) =
                __floats2bfloat162_rn(Or[t][n][2] * inv1, Or[t][n][3] * inv1);
        }

    // max_logits + 2-based LSE (authors' prefill conventions; rmax is max(S*sm_scale),
    // identical across n-halves after the block-max combine). One writer per row.
    if (nh == 0 && (lane % 4) == 0) {
        const int gh = h_block * SM_BH;
        const int64_t o0 = (int64_t)s_q_idx * params.h_q + gh;
        params.max_logits[o0 + row_a] = rmax[0] * SM_LOG2E;
        params.max_logits[o0 + row_b] = rmax[1] * SM_LOG2E;
        params.lse[o0 + row_a] = (sum0 > 0.f) ? (rmax[0] + __logf(sum0)) * SM_LOG2E : -INFINITY;
        params.lse[o0 + row_b] = (sum1 > 0.f) ? (rmax[1] + __logf(sum1)) * SM_LOG2E : -INFINITY;
    }
}

inline void launch_sparse_fwd_mma(const SparsePrefillParams& params) {
    const int num_h_blocks = params.h_q / SM_BH;
    const dim3 grid(num_h_blocks * params.s_q);
    constexpr size_t smem = sizeof(SmemPlanMma);
    static_assert(smem <= 99 * 1024, "sparse mma smem exceeds 99KB");
    cudaFuncSetAttribute(sparse_prefill_fwd_mma_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    sparse_prefill_fwd_mma_kernel<<<grid, dim3(SM_THREADS), smem, params.stream>>>(params);
}

}  // namespace sparse_mma
}  // namespace sm120
