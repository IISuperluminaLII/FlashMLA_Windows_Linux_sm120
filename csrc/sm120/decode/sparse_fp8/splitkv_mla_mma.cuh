#pragma once
// ============================================================================
// SPARSE-FP8 DECODE via raw mma.sync + ldmatrix + cp.async (SM120)   [CFG>=1]
// ============================================================================
// The decode analog of csrc/sm120/prefill/sparse/fwd_mma.cuh (the CFG=4 tier that
// took sparse prefill 21.4 -> ~264 TFlops): identical mma/ldmatrix/swizzle core,
// with the fp8 cache handled by a cp.async raw-byte staging + in-smem dequant
// (fp32-domain, bit-identical to tests/quant.py -- cvt_fp8x8_bf16x8_fp32).
//
// Faithful decode semantics (== the WMMA kernel in splitkv_mla.cu):
//   - NO block_table: page = idx/64, offset = idx%64, all address math int64.
//   - token == -1 is the only invalid form; invalid lanes are never read
//     (cp.async src-size-0 zero-fill) so the oracle's NaN-poison never enters.
//   - KV row = 656 B: [0,512) fp8 e4m3 nope, [512,528) 4 x fp32 scales
//     (one per 128-elem quant tile), [528,656) 64 bf16 rope.
//   - V = dequantized FIRST 512 of the same rows (re-gather cols 0..511).
//   - lse = NATURAL log with +INFINITY sentinel for all-invalid rows.
//   - ragged head tail (q_head_per_hk % 64 != 0): zero-fill Q, guard all writes.
//
// Smem (byte-exact, <= 101,376 B opt-in):
//   sQ    bf16 [64][576] swz(576)  73,728   resident Q nope+rope (WMMA reloads rope
//                                           every kb block; here it is resident)
//   sFp8  u8   [2][64*64] linear    8,192   cp.async staging for fp8 nope tiles;
//                                           doubles as the ROPE bf16 tile (swz(64))
//   sKV   bf16 [64][64]  swz(64)    8,192   dequantized bf16 tile (K d-tile / V tile)
//   sP    bf16 [64][64]  swz(64)    8,192   P tile for the PV ldmatrix
//   stats/scales/ptrs                2,112
//   TOTAL 100,416 B -> 1 CTA/SM
// ============================================================================

#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

namespace sm120 {
namespace sparse_decode {
namespace mma {

constexpr int DM_MMA_M = 16, DM_MMA_N = 8, DM_MMA_K = 16;
constexpr int DM_NW = 8;                    // warps
constexpr int DM_THREADS = DM_NW * 32;      // 256
constexpr int DM_BH = 64;                   // head rows per CTA
constexpr int DM_BTOPK = 64;                // gathered tokens per block
constexpr int DM_DQK = 576, DM_DV = 512, DM_DTILE = 64;
constexpr int DM_NFP8TILES = DM_DV / DM_DTILE;    // 8 fp8 d-tiles (nope)
constexpr int DM_NVTILES = DM_DV / DM_DTILE;      // 8 V column tiles
constexpr int DM_ROPE_BYTE_OFF = 528;             // 512 fp8 + 16 scale bytes
constexpr float DM_LOG2E = 1.4426950408889634f;

// ---- PTX primitives (identical to the proven prefill mma kernel) ------------
__device__ __forceinline__ uint32_t d_cvta(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void d_ldm_x4(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void d_ldm_x2(uint32_t (&r)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void d_ldm_x2_trans(uint32_t (&r)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void d_mma(float (&d)[4], const uint32_t (&a)[4],
                                      const uint32_t (&b)[2], const float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ void d_cp16(uint32_t sa, const void* g, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 ::"r"(sa), "l"(g), "r"(src_bytes));
}
__device__ __forceinline__ void d_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void d_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ int d_swz(int r, int c, int dim) {
    return r * dim + (c ^ ((r & 7) << 3));
}

// ---- smem plan --------------------------------------------------------------
struct SmemPlanDecodeMma {
    __align__(128) __nv_bfloat16 sQ[DM_BH * DM_DQK];           // 73,728 B, swz(576)
    __align__(128) uint8_t sFp8[2][DM_BTOPK * DM_DTILE];       // 2 x 4,096 B, linear
                                                               // (rope tile: bf16 swz(64)
                                                               //  spanning both stages)
    __align__(128) __nv_bfloat16 sKV[DM_BTOPK * DM_DTILE];     // 8,192 B, swz(64)
    __align__(128) __nv_bfloat16 sP[DM_BH * DM_BTOPK];         // 8,192 B, swz(64)
    __align__(128) float sStatM[DM_BH];
    __align__(128) float sStatS[DM_BH];
    __align__(128) float sScl[DM_BTOPK][4];                    // 4 fp32 scales per token
    __align__(128) const uint8_t* sTokPtr[DM_BTOPK];           // row base (bytes)
    __align__(128) int8_t sValid[DM_BTOPK];
};
static_assert(sizeof(SmemPlanDecodeMma) <= 99 * 1024, "decode mma smem exceeds 99KB");

// Gather one 64-token x 64-col FP8 tile (64 B/row = 4 x 16B granules) into stage `st`.
// 256 granules / 256 threads = 1 each. Linear dst; the dequant step swizzles.
__device__ __forceinline__ void gather_fp8_tile(SmemPlanDecodeMma& sm, int st, int col_start) {
    const int tid = threadIdx.x;
    const int r = tid >> 2;                    // token row 0..63
    const int c = (tid & 3) * 16;              // byte col 0..48 step 16
    const uint8_t* tp = sm.sTokPtr[r];
    d_cp16(d_cvta(&sm.sFp8[st][r * DM_DTILE + c]), tp + col_start + c,
           sm.sValid[r] ? 16 : 0);
}

// Gather the 64 bf16 rope elements per token (bytes [528,656)) into the sFp8 union
// region, SWIZZLED bf16 layout so the K-tile ldmatrix can read it directly.
// 64 rows x 8 granules = 512 granules / 256 threads = 2 each.
__device__ __forceinline__ void gather_rope_tile(SmemPlanDecodeMma& sm) {
    __nv_bfloat16* stage = reinterpret_cast<__nv_bfloat16*>(&sm.sFp8[0][0]);
    const int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int g = tid + i * DM_THREADS;
        const int r = g >> 3;
        const int c = (g & 7) * 8;
        const uint8_t* tp = sm.sTokPtr[r];
        d_cp16(d_cvta(&stage[d_swz(r, c, DM_DTILE)]),
               tp + DM_ROPE_BYTE_OFF + c * 2, sm.sValid[r] ? 16 : 0);
    }
}

// Gather the 4 fp32 scales of every token ([512,528) of the row) -- one 16B granule
// per token, threads 0..63. Invalid rows zero-fill -> scale 0.0f (times fp8 zeros).
__device__ __forceinline__ void gather_scales(SmemPlanDecodeMma& sm) {
    const int tid = threadIdx.x;
    if (tid < DM_BTOPK) {
        const uint8_t* tp = sm.sTokPtr[tid];
        d_cp16(d_cvta(&sm.sScl[tid][0]), tp + HEAD_DIM_NOPE, sm.sValid[tid] ? 16 : 0);
    }
}

// Dequantize stage `st` (fp8, linear) into sKV (bf16, swizzled). dt selects the scale
// (a 64-aligned 64-wide range never crosses a 128-elem quant tile: scale idx = dt/2).
// 4096 elems / 256 threads = 16 each (one fp8x16 -> two 16B bf16x8 granule stores).
__device__ __forceinline__ void dequant_tile(SmemPlanDecodeMma& sm, int st, int dt) {
    const int tid = threadIdx.x;
    const int r = tid >> 2;                    // token row 0..63
    const int c = (tid & 3) * 16;              // element col 0..48 step 16
    const float scale = sm.sScl[r][dt >> 1];
    const fp8x16 raw = *reinterpret_cast<const fp8x16*>(&sm.sFp8[st][r * DM_DTILE + c]);
    const bf16x8 lo = cvt_fp8x8_bf16x8_fp32(raw.lo, scale);
    const bf16x8 hi = cvt_fp8x8_bf16x8_fp32(raw.hi, scale);
    store_128b(&sm.sKV[d_swz(r, c, DM_DTILE)], lo);
    store_128b(&sm.sKV[d_swz(r, c + 8, DM_DTILE)], hi);
}

// ---- kernel -----------------------------------------------------------------
__global__ void __launch_bounds__(DM_THREADS, 1)
sparse_fp8_decode_mma_kernel(const SparseFP8DecodeParams params) {
    extern __shared__ char smem_raw[];
    SmemPlanDecodeMma& sm = *reinterpret_cast<SmemPlanDecodeMma*>(smem_raw);

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int m_block_idx = blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int head_start = m_block_idx * DM_BH;
    const int num_valid_rows = min(params.q_head_per_hk - head_start, DM_BH);
    const int row0 = s_q_idx * params.q_head_per_hk + head_start;

    const __nv_bfloat16* q_row_base = (const __nv_bfloat16*)params.q_ptr
        + (int64_t)batch_idx * params.q_batch_stride
        + (int64_t)row0 * params.q_row_stride;
    __nv_bfloat16* o_row_base = (__nv_bfloat16*)params.o_ptr
        + (int64_t)batch_idx * params.o_batch_stride
        + (int64_t)row0 * params.o_row_stride;
    float* lse_base = params.softmax_lse_ptr
        + (int64_t)batch_idx * params.q_seq_per_hk + (int64_t)row0;
    const int* idx_row = params.indices_ptr
        + (int64_t)batch_idx * params.indices_batch_stride
        + (int64_t)s_q_idx * params.indices_seq_stride;
    const uint8_t* kv_base = (const uint8_t*)params.kv_ptr;

    // ---- prologue: resident swizzled Q [64 x 576] (nope + rope), zero ragged rows
    {
        #pragma unroll
        for (int i = 0; i < 18; ++i) {                 // 64*576/8 granules / 256 thr
            const int g = tid + i * DM_THREADS;
            const int r = g / (DM_DQK / 8);
            const int c = (g % (DM_DQK / 8)) * 8;
            int4 v = make_int4(0, 0, 0, 0);
            if (r < num_valid_rows)
                v = *reinterpret_cast<const int4*>(
                        q_row_base + (int64_t)r * params.q_row_stride + c);
            *reinterpret_cast<int4*>(&sm.sQ[d_swz(r, c, DM_DQK)]) = v;
        }
    }

    const int ms = warp & 3, nh = warp >> 2;
    const int wrow = ms * DM_MMA_M;

    float rmax[2] = {-INFINITY, -INFINITY}, rsum[2] = {0.f, 0.f};
    float Or[DM_NVTILES][4][4];
    #pragma unroll
    for (int t = 0; t < DM_NVTILES; ++t)
        #pragma unroll
        for (int n = 0; n < 4; ++n) { Or[t][n][0] = Or[t][n][1] = Or[t][n][2] = Or[t][n][3] = 0.f; }

    const int num_topk_blocks = (params.topk + DM_BTOPK - 1) / DM_BTOPK;

    for (int kb = 0; kb < num_topk_blocks; ++kb) {
        // ---- token pointers + validity (decode: ONLY -1 is invalid) --------
        if (tid < DM_BTOPK) {
            const int k = kb * DM_BTOPK + tid;
            const int token = (k < params.topk) ? idx_row[k] : -1;
            const bool ok = (token >= 0);
            sm.sValid[tid] = ok ? 1 : 0;
            sm.sTokPtr[tid] = ok ? (kv_base
                                    + (int64_t)(token >> 6) * params.kv_page_stride
                                    + (int64_t)(token & 63) * params.kv_token_stride)
                                 : kv_base;    // legal address; src-size 0 keeps it unread
        }
        __syncthreads();

        // ---- QK: S[64h x 64t] over 9 d-tiles (8 fp8 + 1 rope), registers ---
        float Sr[4][4];
        #pragma unroll
        for (int n = 0; n < 4; ++n) { Sr[n][0] = Sr[n][1] = Sr[n][2] = Sr[n][3] = 0.f; }

        gather_scales(sm);
        gather_fp8_tile(sm, 0, 0);
        d_cp_commit();
        for (int dt = 0; dt < DM_NFP8TILES; ++dt) {          // fp8 d-tiles 0..7
            const int cur = dt & 1;
            if (dt + 1 < DM_NFP8TILES) {
                gather_fp8_tile(sm, cur ^ 1, (dt + 1) * DM_DTILE);
                d_cp_commit();
                d_cp_wait<1>();
            } else {
                d_cp_wait<0>();
            }
            __syncthreads();          // stage landed for all; prior mma done (sKV free)
            dequant_tile(sm, cur, dt);
            __syncthreads();          // sKV visible to all warps
            if (dt == DM_NFP8TILES - 1) {
                gather_rope_tile(sm); // into the sFp8 union; overlaps this mma
                d_cp_commit();
            }
            #pragma unroll
            for (int dl = 0; dl < 4; ++dl) {
                uint32_t Qr[4];
                {
                    const int row = wrow + (lane % 16);
                    const int col = dt * DM_DTILE + dl * DM_MMA_K + (lane / 16) * 8;
                    d_ldm_x4(Qr, d_cvta(&sm.sQ[d_swz(row, col, DM_DQK)]));
                }
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    uint32_t Kr[2];
                    const int krow = (nh * 4 + n) * DM_MMA_N + (lane % 8);
                    const int kcol = dl * DM_MMA_K + ((lane / 8) % 2) * 8;
                    d_ldm_x2(Kr, d_cvta(&sm.sKV[d_swz(krow, kcol, DM_DTILE)]));
                    d_mma(Sr[n], Qr, Kr, Sr[n]);
                }
            }
        }
        {   // rope d-tile (dt == 8): bf16, read straight from the sFp8 union region
            const __nv_bfloat16* stage = reinterpret_cast<const __nv_bfloat16*>(&sm.sFp8[0][0]);
            d_cp_wait<0>();
            __syncthreads();          // rope landed + every thread past its last fp8 mma
            #pragma unroll
            for (int dl = 0; dl < 4; ++dl) {
                uint32_t Qr[4];
                {
                    const int row = wrow + (lane % 16);
                    const int col = DM_DV + dl * DM_MMA_K + (lane / 16) * 8;
                    d_ldm_x4(Qr, d_cvta(&sm.sQ[d_swz(row, col, DM_DQK)]));
                }
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    uint32_t Kr[2];
                    const int krow = (nh * 4 + n) * DM_MMA_N + (lane % 8);
                    const int kcol = dl * DM_MMA_K + ((lane / 8) % 2) * 8;
                    d_ldm_x2(Kr, d_cvta(&stage[d_swz(krow, kcol, DM_DTILE)]));
                    d_mma(Sr[n], Qr, Kr, Sr[n]);
                }
            }
        }

        // ---- mask invalid tokens, scale, per-lane partial max ---------------
        const float scale = params.sm_scale;
        float tmax[2] = {-INFINITY, -INFINITY};
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int col0 = (nh * 4 + n) * DM_MMA_N + (lane % 4) * 2;
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
        for (int t = 0; t < DM_NVTILES; ++t)
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
            const int prow_a = wrow + lane / 4, prow_b = prow_a + 8;
            const int pcol = (nh * 4 + n) * DM_MMA_N + (lane % 4) * 2;
            *reinterpret_cast<__nv_bfloat162*>(&sm.sP[d_swz(prow_a, pcol, DM_BTOPK)]) =
                __floats2bfloat162_rn(p0, p1);
            *reinterpret_cast<__nv_bfloat162*>(&sm.sP[d_swz(prow_b, pcol, DM_BTOPK)]) =
                __floats2bfloat162_rn(p2, p3);
        }
        rsum[0] = rsum[0] * rs0 + tsum[0];
        rsum[1] = rsum[1] * rs1 + tsum[1];

        // ---- PV: 8 V tiles = the fp8 nope tiles re-gathered + re-dequantized
        // (sFp8's rope contents are dead now; the softmax combine syncs above
        //  separate the rope ldmatrix reads from this overwrite.)
        gather_fp8_tile(sm, 0, 0);
        d_cp_commit();
        #pragma unroll
        for (int vt = 0; vt < DM_NVTILES; ++vt) {
            const int cur = vt & 1;
            if (vt + 1 < DM_NVTILES) {
                gather_fp8_tile(sm, cur ^ 1, (vt + 1) * DM_DTILE);
                d_cp_commit();
                d_cp_wait<1>();
            } else {
                d_cp_wait<0>();
            }
            __syncthreads();          // stage landed; prior mma done (sKV free)
            dequant_tile(sm, cur, vt);
            __syncthreads();          // sKV (V tile) + sP visible
            #pragma unroll
            for (int pk = 0; pk < 4; ++pk) {
                uint32_t Pr[4];
                {
                    const int prow = wrow + (lane % 16);
                    const int pcol = pk * DM_MMA_K + (lane / 16) * 8;
                    d_ldm_x4(Pr, d_cvta(&sm.sP[d_swz(prow, pcol, DM_BTOPK)]));
                }
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    uint32_t Vr[2];
                    const int vrow = pk * DM_MMA_K + (lane % 16);
                    const int vcol = nh * 32 + n * DM_MMA_N + (lane / 16) * 8;
                    d_ldm_x2_trans(Vr, d_cvta(&sm.sKV[d_swz(vrow, vcol, DM_DTILE)]));
                    d_mma(Or[vt][n], Pr, Vr, Or[vt][n]);
                }
            }
        }
        __syncthreads();   // last V mma done before next kb rewrites sTokPtr/sFp8/sP
    }

    // ---- epilogue -----------------------------------------------------------
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
    for (int t = 0; t < DM_NVTILES; ++t)
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int ocol = t * DM_DTILE + nh * 32 + n * DM_MMA_N + (lane % 4) * 2;
            if (row_a < num_valid_rows)
                *reinterpret_cast<__nv_bfloat162*>(
                    o_row_base + (int64_t)row_a * params.o_row_stride + ocol) =
                    __floats2bfloat162_rn(Or[t][n][0] * inv0, Or[t][n][1] * inv0);
            if (row_b < num_valid_rows)
                *reinterpret_cast<__nv_bfloat162*>(
                    o_row_base + (int64_t)row_b * params.o_row_stride + ocol) =
                    __floats2bfloat162_rn(Or[t][n][2] * inv1, Or[t][n][3] * inv1);
        }

    // NATURAL-log lse with +INFINITY sentinel (decode convention; rmax is the
    // natural-domain max of S*sm_scale). One writer per row, ragged-guarded.
    if (nh == 0 && (lane % 4) == 0) {
        if (row_a < num_valid_rows)
            lse_base[row_a] = (sum0 > 0.f) ? (rmax[0] + __logf(sum0)) : INFINITY;
        if (row_b < num_valid_rows)
            lse_base[row_b] = (sum1 > 0.f) ? (rmax[1] + __logf(sum1)) : INFINITY;
    }
}

inline void launch_sparse_fp8_decode_mma(const SparseFP8DecodeParams& params) {
    const int num_m_blocks = (params.q_head_per_hk + DM_BH - 1) / DM_BH;
    const dim3 grid(num_m_blocks, params.s_q, params.b);
    constexpr size_t smem = sizeof(SmemPlanDecodeMma);
    CHECK_CUDA(cudaFuncSetAttribute(sparse_fp8_decode_mma_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    sparse_fp8_decode_mma_kernel<<<grid, dim3(DM_THREADS), smem, params.stream>>>(params);
}

}  // namespace mma
}  // namespace sparse_decode
}  // namespace sm120
