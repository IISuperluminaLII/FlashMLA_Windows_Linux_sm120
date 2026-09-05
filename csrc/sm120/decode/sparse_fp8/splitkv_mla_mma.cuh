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
//   stats/scales/ptrs                2,112   (+64 B struct tail padding to 128)
//   TOTAL sizeof = 100,480 B -> 1 CTA/SM
//
// OPT ladder (env FLASH_MLA_SM120_SPARSE_DECODE_OPT, sparse_decode_opt.h; template int kOpt
// on the CFG=1 and CFG=2/4 kernels; 0 = every statement below verbatim, byte-identical SASS;
// audit/design-sparse-decode-opt.md + audit/impl-sparse-decode-opt.md):
//   1  K-path register-fragment dequant: the QK B fragment is read as two 2 B pairs of ONE
//      token row straight from the fp8 stage (LDS.U16 x2 -> fp8x2 -> f32 -> *scale -> bf16x2,
//      dequant.h cvt_fp8x2_bf16x2_fp32 = bit-identical values), no sKV round trip, ONE
//      barrier per K tile (issue-after-barrier 2-stage ring). The stage rows are XOR-swizzled
//      at 16 B granularity (d_fp8_swz) so the 8-row fragment reads are conflict-free at 0 B;
//      the rope tile moves into sKV (dead during QK). V path unchanged (x2.trans + smem dequant
//      into sKV), so sKV STAYS (the V tile needs it): smem unchanged at 100,480 B.
//   2  never-drain copy stream: V tile 0 gathered at dt=7, the NEXT block's K tile 0 / scales
//      at vt=6 / vt=7 from a parity buffer of token pointers resolved one block ahead
//      (SmemPlanDecodeMmaOpt2, +640 B = 101,120 B). 26 barriers per block instead of 37.
// ============================================================================

#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <type_traits>
#include "sparse_decode_opt.h"

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

// Rung 2 (OPT>=2) plan: + a parity buffer holding the NEXT block's token pointers/validity
// for the one-block-ahead K0/scales prefetch. Inherits the base layout so every OPT=0 member
// access stays verbatim (sTokPtr/sValid keep their names and their "current block" role).
// 100,480 + 512 + 128 = 101,120 B <= 101,376. The 256 B left cannot hold a second sScl
// (1,024 B): the next block's scales therefore go into the single sScl at vt=7, after its
// last reader of the block (dequant vt=7) has been fenced by SYNC_P2(7).
struct SmemPlanDecodeMmaOpt2 : SmemPlanDecodeMma {
    __align__(128) const uint8_t* sTokPtrN[DM_BTOPK];          // next block's row bases
    __align__(128) int8_t sValidN[DM_BTOPK];
};
static_assert(sizeof(SmemPlanDecodeMmaOpt2) <= 99 * 1024, "decode mma OPT2 smem exceeds 99KB");
static_assert(sizeof(SmemPlanDecodeMmaOpt2) == sizeof(SmemPlanDecodeMma) + 640,
              "OPT2 parity buffers must append to the base plan (+512 +128 B)");

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

// ============================================================================
// OPT ladder helpers (kOpt >= 1). The OPT=0 helpers above are untouched.
// ============================================================================

// Swizzled fp8 stage address of 16 B granule g (0..3) of token row r (0..63): the granule
// index is XORed with (r>>1)&3. WHY: the K-path register dequant reads 2 B pairs at a 64 B
// row stride; a 64 B row is 16 banks, so rows {r, r+2, r+4, r+6} of an 8-row n-tile fall on
// the SAME banks -> a 4-way conflict per LDS.U16 on the linear layout. XORing the granule
// with (r>>1)&3 spreads those four rows over four distinct granules (bank quads); odd rows
// sit 16 banks away. Result: the 8 rows x 2 banks of one fragment read hit 16 distinct banks
// (the 4 lanes sharing a row read the same 4 B word = broadcast) -> conflict-free at 0 B.
// Chosen over the 80 B row pad: the pad (+2,048 B) would push the rung-2 plan to 103,168 B
// > 101,376 (sKV cannot be deleted -- the V dequant still needs it). Every 16 B granule
// still lands whole and 16 B-aligned (cp.async 16 needs that); the gather and the V dequant
// address granules through this map, and the 8 consecutive threads of a gather/dequant
// phase still cover two whole rows = 128 contiguous bytes -> conflict-free as before.
__device__ __forceinline__ int d_fp8_swz(int r, int g) {
    return r * DM_DTILE + ((g ^ ((r >> 1) & 3)) << 4);
}

// gather_fp8_tile on the swizzled stage layout from EXPLICIT pointer/validity buffers (rung 2
// feeds the next block's parity buffer through the same helper). Same thread map.
__device__ __forceinline__ void gather_fp8_tile_swz(SmemPlanDecodeMma& sm, int st, int col_start,
                                                    const uint8_t* const* tokptr,
                                                    const int8_t* valid) {
    const int tid = threadIdx.x;
    const int r = tid >> 2;                    // token row 0..63
    const int g = tid & 3;                     // 16 B granule 0..3
    const uint8_t* tp = tokptr[r];
    d_cp16(d_cvta(&sm.sFp8[st][d_fp8_swz(r, g)]), tp + col_start + g * 16,
           valid[r] ? 16 : 0);
}

// Rope tile (bf16, swz(64)) into sKV instead of the sFp8 union: under OPT>=1 the K-path mma
// reads the stages DIRECTLY until the end of dt=7, so the union is never free for the rope,
// while sKV is dead for the whole QK phase (the V dequant is its only other writer). Same
// layout the rope ldmatrix expects (== the OPT=0 sKV K-tile layout); same thread map as
// gather_rope_tile.
__device__ __forceinline__ void gather_rope_tile_skv(SmemPlanDecodeMma& sm,
                                                     const uint8_t* const* tokptr,
                                                     const int8_t* valid) {
    const int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int g = tid + i * DM_THREADS;
        const int r = g >> 3;
        const int c = (g & 7) * 8;
        const uint8_t* tp = tokptr[r];
        d_cp16(d_cvta(&sm.sKV[d_swz(r, c, DM_DTILE)]),
               tp + DM_ROPE_BYTE_OFF + c * 2, valid[r] ? 16 : 0);
    }
}

// gather_scales from explicit buffers (rung 2 prefetches the next block's scales).
__device__ __forceinline__ void gather_scales_at(SmemPlanDecodeMma& sm,
                                                 const uint8_t* const* tokptr,
                                                 const int8_t* valid) {
    const int tid = threadIdx.x;
    if (tid < DM_BTOPK) {
        const uint8_t* tp = tokptr[tid];
        d_cp16(d_cvta(&sm.sScl[tid][0]), tp + HEAD_DIM_NOPE, valid[tid] ? 16 : 0);
    }
}

// dequant_tile on the swizzled stage layout (V path only under OPT>=1). Same math, same
// destination layout; only the source granule address goes through d_fp8_swz.
__device__ __forceinline__ void dequant_tile_swz(SmemPlanDecodeMma& sm, int st, int dt) {
    const int tid = threadIdx.x;
    const int r = tid >> 2;                    // token row 0..63
    const int g = tid & 3;                     // 16 B granule 0..3 == element col g*16
    const float scale = sm.sScl[r][dt >> 1];
    const fp8x16 raw = *reinterpret_cast<const fp8x16*>(&sm.sFp8[st][d_fp8_swz(r, g)]);
    const bf16x8 lo = cvt_fp8x8_bf16x8_fp32(raw.lo, scale);
    const bf16x8 hi = cvt_fp8x8_bf16x8_fp32(raw.hi, scale);
    store_128b(&sm.sKV[d_swz(r, g * 16, DM_DTILE)], lo);
    store_128b(&sm.sKV[d_swz(r, g * 16 + 8, DM_DTILE)], hi);
}

// K-path register-fragment dequant (rung 1). The m16n8k16 B fragment of lane l is
// {K[n0 + l/4][k0 + (l%4)*2, +1], K[n0 + l/4][k0 + 8 + (l%4)*2, +1]}: two 2 B pairs of ONE
// token row. kp points at the first pair inside the swizzled stage row; the second pair sits
// 8 B further in the same granule. fp32-domain math == dequant.h (bit-identical values).
__device__ __forceinline__ void d_kfrag_regdq(uint32_t (&Kr)[2], const uint8_t* kp, float scale) {
    const unsigned short lo = *reinterpret_cast<const unsigned short*>(kp);
    const unsigned short hi = *reinterpret_cast<const unsigned short*>(kp + 8);
    Kr[0] = cvt_fp8x2_bf16x2_fp32(lo, scale);
    Kr[1] = cvt_fp8x2_bf16x2_fp32(hi, scale);
}

// Token pointers + validity of block kb into EXPLICIT buffers (tid < 64, own index only):
// the OPT=0 phase-A statements plus a walk-end guard -- blocks at or beyond blk_hi (the
// exclusive end of this CTA's walk) resolve ALL-INVALID, so the rung-2 one-block-ahead
// prefetch issued during the walk's last block degenerates to src-size-0 zero-fills that
// read no memory (and idx_row is never read past the walk). ONLY token == -1 is invalid.
__device__ __forceinline__ void d_phase_a(const uint8_t** tokptr, int8_t* valid,
                                          const int* idx_row, const uint8_t* kv_base,
                                          const SparseFP8DecodeParams& params,
                                          int kb, int blk_hi) {
    const int tid = threadIdx.x;
    if (tid < DM_BTOPK) {
        const int k = kb * DM_BTOPK + tid;
        const int token = (kb < blk_hi && k < params.topk) ? idx_row[k] : -1;
        const bool ok = (token >= 0);
        valid[tid] = ok ? 1 : 0;
        tokptr[tid] = ok ? (kv_base
                            + (int64_t)(token >> 6) * params.kv_page_stride
                            + (int64_t)(token & 63) * params.kv_token_stride)
                         : kv_base;    // legal address; src-size 0 keeps it unread
    }
}

// Implemented rung for the latched OPT value: rung 3 (V-path half-tile dequant/mma software
// pipelining) is reserved but NOT implemented -> route to rung 2 and say so once (warn, never
// error out).
inline int sparse_decode_opt_rung() {
    static const int rung = [] {
        const int opt = sparse_decode_opt();
        if (opt > 2) {
            std::fprintf(stderr, "[WARN] FLASH_MLA_SM120_SPARSE_DECODE_OPT=%d: rung 3 is not "
                                 "implemented; running rung 2\n", opt);
            return 2;
        }
        return opt;
    }();
    return rung;
}

// ---- kernel -----------------------------------------------------------------
// kOpt = FLASH_MLA_SM120_SPARSE_DECODE_OPT rung (sparse_decode_opt.h); 0 = byte-identical.
template <int kOpt>
__global__ void __launch_bounds__(DM_THREADS, 1)
sparse_fp8_decode_mma_kernel(const SparseFP8DecodeParams params) {
    using Plan = std::conditional_t<(kOpt >= 2), SmemPlanDecodeMmaOpt2, SmemPlanDecodeMma>;
    extern __shared__ char smem_raw[];
    Plan& sm = *reinterpret_cast<Plan*>(smem_raw);

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

    if constexpr (kOpt >= 2) {
        // ---- rung 2 prologue: block 0's pointers straight into the working buffer, then
        // its scales + K tile 0 in flight BEFORE the loop; every later block's K0/scales come
        // from the previous block's vt=6/vt=7 prefetch. (The barrier also publishes sQ.)
        // topk == 0 (no blocks): all-invalid zero-fills, drained after the loop.
        d_phase_a(sm.sTokPtr, sm.sValid, idx_row, kv_base, params, 0, num_topk_blocks);
        __syncthreads();          // sTokPtr/sValid (+ sQ) visible to the 256 gather threads
        gather_scales_at(sm, sm.sTokPtr, sm.sValid);
        gather_fp8_tile_swz(sm, 0, 0, sm.sTokPtr, sm.sValid);
        d_cp_commit();            // group {Scl(0), K0(0)}
    }

    for (int kb = 0; kb < num_topk_blocks; ++kb) {
        if constexpr (kOpt >= 2) {
            // ---- rung 2: token pointers ONE BLOCK AHEAD, no SYNC_A (tid<64, own index).
            // Promote block kb's pointers (resolved during block kb-1 into the parity buffer,
            // or by the prologue for kb=0) to the working buffer, then resolve block kb+1 into
            // the parity buffer for this block's vt=6/vt=7 prefetch.
            //   working buffer WAR: block kb-1's last readers (its V7 gather at the vt=6 top,
            //   its masking before SYNC_M1) precede SYNC_P1(6) of block kb-1, which these
            //   writers passed. RAW: published by SYNC_Q(0) below (first readers: this
            //   block's K1 gather and masking).
            //   parity buffer WAR: block kb-1's readers were its vt=6 K0 gather (all threads,
            //   before SYNC_P1(7)) and its vt=7 scales gather (these same tid<64 threads,
            //   program order). RAW: SYNC_Q(0) (first reader: the vt=6 K0 gather).
            if (tid < DM_BTOPK && kb > 0) {
                sm.sTokPtr[tid] = sm.sTokPtrN[tid];
                sm.sValid[tid] = sm.sValidN[tid];
            }
            d_phase_a(sm.sTokPtrN, sm.sValidN, idx_row, kv_base, params, kb + 1, num_topk_blocks);
        } else {
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
        }

        // ---- QK: S[64h x 64t] over 9 d-tiles (8 fp8 + 1 rope), registers ---
        float Sr[4][4];
        #pragma unroll
        for (int n = 0; n < 4; ++n) { Sr[n][0] = Sr[n][1] = Sr[n][2] = Sr[n][3] = 0.f; }

        if constexpr (kOpt >= 1) {
            // ---- OPT>=1 QK: the K-tile B fragments are built by the mma warps STRAIGHT
            // from the swizzled fp8 stage (register-fragment dequant, no sKV round trip),
            // ONE barrier per K tile on a 2-stage issue-after-barrier ring:
            //   wait -> SYNC_Q(dt) -> gather K(dt+1) into the stage mma(dt-1) drained -> mma(dt)
            // Per-lane constants: fragment row = n0 + lane/4, byte pairs (lane%4)*2 and +8;
            // koff = row base (bits 6+) + pair offset (bits 1-2) + swizzle key (bits 4-5:
            // (row>>1)&3 == (lane>>3)&3 for every n-tile because n0 % 8 == 0), so the
            // swizzled granule of d-step dl is koff ^ (dl*16) and n-tile n adds n*512.
            const int koff = (nh * 32 + (lane >> 2)) * DM_DTILE + (lane & 3) * 2
                           + (((lane >> 3) & 3) << 4);
            if constexpr (kOpt < 2) {
                gather_scales_at(sm, sm.sTokPtr, sm.sValid);          // rung 1: this
                gather_fp8_tile_swz(sm, 0, 0, sm.sTokPtr, sm.sValid);  // block's scales +
                d_cp_commit();    // group {Scl(kb), K0(kb)}            // K0 issued here
            }
            for (int dt = 0; dt < DM_NFP8TILES; ++dt) {
                const int cur = dt & 1;
                // Ledger (per thread): dt=0 pending {Scl+K0} (rung 1) / {K0(kb)},{Scl(kb)}
                // (rung 2: from the previous block's vt=6/7 or the prologue) -> wait<0>;
                // dt=1 pending {K1},{rope} -> wait<1> retires K1 (the rope may still fly);
                // dt>=2 pending {rope} (dt=2 only),{K(dt)} -> wait<0>.
                if (dt == 1) d_cp_wait<1>(); else d_cp_wait<0>();
                __syncthreads();  // SYNC_Q(dt). RAW: stage cur (at dt=0 also sScl + the
                                  // pointer buffers, at dt>=2 the rope in sKV) visible to
                                  // every warp. WAR: every warp is past its mma(dt-1) reads
                                  // of stage cur^1, which the gather below overwrites -- and
                                  // at dt=0 past the previous block's V mma(7) reads of sKV,
                                  // which the rope gather overwrites.
                if (dt + 1 < DM_NFP8TILES) {
                    gather_fp8_tile_swz(sm, cur ^ 1, (dt + 1) * DM_DTILE, sm.sTokPtr, sm.sValid);
                    d_cp_commit();    // group {K(dt+1)}: lands under mma(dt)
                    if (dt == 0) {
                        gather_rope_tile_skv(sm, sm.sTokPtr, sm.sValid);
                        d_cp_commit(); // group {rope}: retired by dt=2's wait<0>, read after dt=7
                    }
                } else if constexpr (kOpt >= 2) {
                    // Rung 2(a): V tile 0 into stage 0 NOW -- its last readers, mma(6), are
                    // behind SYNC_Q(7) -- so it lands under mma(7) + rope mma + softmax.
                    gather_fp8_tile_swz(sm, 0, 0, sm.sTokPtr, sm.sValid);
                    d_cp_commit();    // group {V0}
                }
                float sc[4];          // this lane's 4 fragment rows' scales for this d-tile
                #pragma unroll
                for (int n = 0; n < 4; ++n)
                    sc[n] = sm.sScl[(nh * 4 + n) * DM_MMA_N + (lane >> 2)][dt >> 1];
                #pragma unroll
                for (int dl = 0; dl < 4; ++dl) {
                    uint32_t Qr[4];
                    {
                        const int row = wrow + (lane % 16);
                        const int col = dt * DM_DTILE + dl * DM_MMA_K + (lane / 16) * 8;
                        d_ldm_x4(Qr, d_cvta(&sm.sQ[d_swz(row, col, DM_DQK)]));
                    }
                    const int kd = koff ^ (dl * 16);   // swizzled 16 B granule of d-step dl
                    #pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        uint32_t Kr[2];
                        d_kfrag_regdq(Kr, &sm.sFp8[cur][kd + n * (DM_MMA_N * DM_DTILE)], sc[n]);
                        d_mma(Sr[n], Qr, Kr, Sr[n]);
                    }
                }
            }
            {   // rope d-tile (dt == 8): bf16 from sKV. Landed by dt=2's wait<0> on every
                // thread and published by SYNC_Q(2..7): no wait, no barrier here.
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
                        d_ldm_x2(Kr, d_cvta(&sm.sKV[d_swz(krow, kcol, DM_DTILE)]));
                        d_mma(Sr[n], Qr, Kr, Sr[n]);
                    }
                }
            }
        } else {
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
        if constexpr (kOpt >= 1) {
            // ---- OPT>=1 PV: the V path is structurally OPT=0 (2-stage gather, smem dequant
            // into sKV, x2.trans) on the swizzled stages. Rung 2: V0 is already in flight
            // (issued at dt=7) and the NEXT block's K tile 0 / scales are issued at vt=6 /
            // vt=7, so the vt=7 wait leaves exactly ONE (newer) group pending. No block-
            // bottom barrier (note after the loop).
            if constexpr (kOpt < 2) {
                gather_fp8_tile_swz(sm, 0, 0, sm.sTokPtr, sm.sValid);  // V0 -> stage 0: its last
                d_cp_commit();    // group {V0}                        // readers, mma(6), are behind SYNC_Q(7)
            }
            #pragma unroll
            for (int vt = 0; vt < DM_NVTILES; ++vt) {
                const int cur = vt & 1;
                if (vt + 1 < DM_NVTILES) {
                    // stage cur^1: last readers dequant(vt-1), behind SYNC_P2(vt-1); for vt=0
                    // the K-path mma(7) reads of stage 1, behind SYNC_M1/SYNC_M2.
                    gather_fp8_tile_swz(sm, cur ^ 1, (vt + 1) * DM_DTILE, sm.sTokPtr, sm.sValid);
                    d_cp_commit();    // group {V(vt+1)}
                    d_cp_wait<1>();   // pending {V(vt)},{V(vt+1)}: the older, V(vt), landed
                } else if constexpr (kOpt >= 2) {
                    d_cp_wait<1>();   // pending {V7},{K0(kb+1)}: V7 (older) landed, K0 may fly
                } else {
                    d_cp_wait<0>();   // pending {V7}
                }
                __syncthreads();      // SYNC_P1(vt): stage cur landed for all; prior V mma done (sKV free)
                dequant_tile_swz(sm, cur, vt);
                __syncthreads();      // SYNC_P2(vt): sKV (V tile) + sP visible; stage cur consumed by all
                if constexpr (kOpt >= 2) {
                    if (vt == 6) {
                        // Rung 2(b): the NEXT block's K tile 0 into stage 0 (tile 6 consumed by
                        // every thread at SYNC_P2(6)); pointers from the parity buffer, resolved
                        // at this block's start and published by SYNC_Q(0). At the last block
                        // this is an all-invalid zero-fill (d_phase_a), drained after the loop.
                        gather_fp8_tile_swz(sm, 0, 0, sm.sTokPtrN, sm.sValidN);
                        d_cp_commit();    // group {K0(kb+1)}
                    }
                    if (vt == 7) {
                        // The NEXT block's scales into the single sScl: this block's last
                        // readers of it (dequant vt=7; the K path long before) are behind
                        // SYNC_P2(7). Both prefetch groups retire at the next dt=0 wait<0>.
                        gather_scales_at(sm, sm.sTokPtrN, sm.sValidN);
                        d_cp_commit();    // group {Scl(kb+1)}
                    }
                }
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
            // No block-bottom barrier under OPT>=1. The next block's first writes are: the
            // pointer buffers (tid<64, after every reader of them passed SYNC_P1(6) at the
            // latest -- phase-A notes), sScl / stage 0 (rung 1: issued after SYNC_A; rung 2:
            // issued above at vt=7 / vt=6, behind SYNC_P2(7) / SYNC_P2(6), their last
            // readers), stage 1 and sKV (after SYNC_Q(0), which every warp reaches only after
            // its own mma(7)), and sP (after SYNC_M2 of the next block).
        } else {
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
    }
    if constexpr (kOpt >= 2) d_cp_wait<0>();   // retire the last block's (all-invalid, zero-
                                               // fill) K0/scales prefetch: nothing may outlive
                                               // the kernel or leak into the epilogue

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

template <int kOpt>
inline void launch_sparse_fp8_decode_mma_opt(const SparseFP8DecodeParams& params, const dim3& grid) {
    using Plan = std::conditional_t<(kOpt >= 2), SmemPlanDecodeMmaOpt2, SmemPlanDecodeMma>;
    constexpr size_t smem = sizeof(Plan);
    CHECK_CUDA(cudaFuncSetAttribute(sparse_fp8_decode_mma_kernel<kOpt>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    sparse_fp8_decode_mma_kernel<kOpt><<<grid, dim3(DM_THREADS), smem, params.stream>>>(params);
}

inline void launch_sparse_fp8_decode_mma(const SparseFP8DecodeParams& params) {
    const int num_m_blocks = (params.q_head_per_hk + DM_BH - 1) / DM_BH;
    const dim3 grid(num_m_blocks, params.s_q, params.b);
    // OPT ladder (FLASH_MLA_SM120_SPARSE_DECODE_OPT, read once): 0 = legacy byte-identical.
    switch (sparse_decode_opt_rung()) {
        case 1:  launch_sparse_fp8_decode_mma_opt<1>(params, grid); break;
        case 2:  launch_sparse_fp8_decode_mma_opt<2>(params, grid); break;
        default: launch_sparse_fp8_decode_mma_opt<0>(params, grid); break;
    }
}

// ============================================================================
// SPLIT-KV tier [CFG>=2]: the SAME mma core, driven by the authors' tile
// scheduler (audit/design-sparse-splitkv.md). Grid (m_blocks, s_q, num_sm_parts);
// each CTA walks its metadata row's batch/topk-block range. 1-split batches
// write finals with the certified CFG=1 conventions (natural lse, +INFINITY
// sentinel -- which equals the combine's own all-empty output, so the oracle
// contract is preserved across tiers); split batches write NORMALIZED fp32
// partial O + 2-BASED partial lse (-INFINITY empty) into the authors' accum
// layout at split_row = num_splits_ptr[batch] + n_split_idx, merged by the
// (already unconditionally launched) flash_fwd_mla_combine_kernel, which skips
// 1-split batches -- so no_split MUST be the num_splits prefix difference, the
// combine's own predicate.
// ============================================================================

constexpr float DM_LOG2E = 1.4426950408889634f;

// kOpt = FLASH_MLA_SM120_SPARSE_DECODE_OPT rung (sparse_decode_opt.h); 0 = byte-identical.
template <int kOpt>
__global__ void __launch_bounds__(DM_THREADS, 1)
sparse_fp8_decode_mma_splitkv_kernel(const SparseFP8DecodeParams params) {
    using Plan = std::conditional_t<(kOpt >= 2), SmemPlanDecodeMmaOpt2, SmemPlanDecodeMma>;
    extern __shared__ char smem_raw[];
    Plan& sm = *reinterpret_cast<Plan*>(smem_raw);

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int m_block_idx = blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int part_idx = blockIdx.z;

    const int head_start = m_block_idx * DM_BH;
    const int num_valid_rows = min(params.q_head_per_hk - head_start, DM_BH);
    const int row0 = s_q_idx * params.q_head_per_hk + head_start;

    // ---- tile-scheduler metadata row (authors' format, 5 ints used) --------
    const int* meta = params.tile_scheduler_metadata_ptr
        + part_idx * TileSchedulerMetaDataSize;
    const int begin_idx = meta[0];
    const int begin_block = meta[1];
    const int end_idx = meta[2];
    const int end_block_last = meta[3];          // exclusive, for batch == end_idx
    const int begin_n_split_idx = meta[4];

    // Sparse schedule geometry: the scheduler substitutes topk for seqlen
    // (get_mla_metadata.cu cur_s_k), uniformly for every batch. ceil form --
    // ragged topk blocks are real work (k < topk guards the tail).
    const int num_topk_blocks = (params.topk + DM_BTOPK - 1) / DM_BTOPK;

    const uint8_t* kv_base = (const uint8_t*)params.kv_ptr;
    const int ms = warp & 3, nh = warp >> 2;
    const int wrow = ms * DM_MMA_M;
    const int row_a = wrow + lane / 4, row_b = row_a + 8;

    for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx) {
        const int blk_lo = (batch_idx == begin_idx) ? begin_block : 0;
        const int blk_hi_sched = (batch_idx == end_idx) ? end_block_last : num_topk_blocks;
        // topk == 0 gives every batch ONE phantom scheduler block (the producer's
        // max(seqlen,1) convention); clamp the walk to the real block count so it
        // collapses to zero iterations (same class as the dense zero-seqlen skip).
        const int blk_hi = min(blk_hi_sched, num_topk_blocks);
        const int n_split_idx = (batch_idx == begin_idx) ? begin_n_split_idx : 0;
        // The combine kernel's OWN skip predicate (prefix difference == 1) --
        // never derive split-ness from block ranges.
        const int split_base = __ldg(params.num_splits_ptr + batch_idx);
        const bool no_split = (__ldg(params.num_splits_ptr + batch_idx + 1) - split_base) == 1;

        const __nv_bfloat16* q_row_base = (const __nv_bfloat16*)params.q_ptr
            + (int64_t)batch_idx * params.q_batch_stride
            + (int64_t)row0 * params.q_row_stride;
        const int* idx_row = params.indices_ptr
            + (int64_t)batch_idx * params.indices_batch_stride
            + (int64_t)s_q_idx * params.indices_seq_stride;

        // ---- resident swizzled Q [64 x 576] for this batch (zero ragged rows)
        {
            #pragma unroll
            for (int i = 0; i < 18; ++i) {
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

        float rmax[2] = {-INFINITY, -INFINITY}, rsum[2] = {0.f, 0.f};
        float Or[DM_NVTILES][4][4];
        #pragma unroll
        for (int t = 0; t < DM_NVTILES; ++t)
            #pragma unroll
            for (int n = 0; n < 4; ++n) { Or[t][n][0] = Or[t][n][1] = Or[t][n][2] = Or[t][n][3] = 0.f; }

        if constexpr (kOpt >= 2)   // rung 2: the walk's first block's pointers -> working buffer
            d_phase_a(sm.sTokPtr, sm.sValid, idx_row, kv_base, params, blk_lo, blk_hi);
        __syncthreads();   // sQ visible; previous batch's smem fully consumed
                           // (also closes the empty-walk sStatS overwrite race)
        if constexpr (kOpt >= 2) {
            // ---- rung 2 prologue: the first block's scales + K tile 0 in flight BEFORE the
            // walk; every later block's K0/scales come from the previous block's vt=6/vt=7
            // prefetch. An empty walk issues all-invalid zero-fills, drained after the loop.
            gather_scales_at(sm, sm.sTokPtr, sm.sValid);
            gather_fp8_tile_swz(sm, 0, 0, sm.sTokPtr, sm.sValid);
            d_cp_commit();            // group {Scl(blk_lo), K0(blk_lo)}
        }

        for (int kb = blk_lo; kb < blk_hi; ++kb) {
            if constexpr (kOpt >= 2) {
                // ---- rung 2: token pointers ONE BLOCK AHEAD, no SYNC_A (tid<64, own index).
                // Promote block kb's pointers (resolved during block kb-1 into the parity
                // buffer, or by the prologue for the first block) to the working buffer, then
                // resolve block kb+1 into the parity buffer for this block's vt=6/7 prefetch.
                //   working buffer WAR: block kb-1's last readers (its V7 gather at the vt=6
                //   top, its masking before SYNC_M1) precede SYNC_P1(6) of block kb-1, which
                //   these writers passed. RAW: published by SYNC_Q(0) below (first readers:
                //   this block's K1 gather and masking).
                //   parity buffer WAR: block kb-1's readers were its vt=6 K0 gather (all
                //   threads, before SYNC_P1(7)) and its vt=7 scales gather (these same tid<64
                //   threads, program order). RAW: SYNC_Q(0) (first reader: the vt=6 K0 gather).
                if (tid < DM_BTOPK && kb > blk_lo) {
                    sm.sTokPtr[tid] = sm.sTokPtrN[tid];
                    sm.sValid[tid] = sm.sValidN[tid];
                }
                d_phase_a(sm.sTokPtrN, sm.sValidN, idx_row, kv_base, params, kb + 1, blk_hi);
            } else {
            // ---- token pointers + validity (decode: ONLY -1 is invalid) ----
            if (tid < DM_BTOPK) {
                const int k = kb * DM_BTOPK + tid;
                const int token = (k < params.topk) ? idx_row[k] : -1;
                const bool ok = (token >= 0);
                sm.sValid[tid] = ok ? 1 : 0;
                sm.sTokPtr[tid] = ok ? (kv_base
                                        + (int64_t)(token >> 6) * params.kv_page_stride
                                        + (int64_t)(token & 63) * params.kv_token_stride)
                                     : kv_base;   // legal address; src-size 0 keeps it unread
            }
            __syncthreads();
            }

            // ---- QK: S[64h x 64t] over 9 d-tiles (8 fp8 + 1 rope), registers
            float Sr[4][4];
            #pragma unroll
            for (int n = 0; n < 4; ++n) { Sr[n][0] = Sr[n][1] = Sr[n][2] = Sr[n][3] = 0.f; }

            if constexpr (kOpt >= 1) {
                // ---- OPT>=1 QK: the K-tile B fragments are built by the mma warps STRAIGHT
                // from the swizzled fp8 stage (register-fragment dequant, no sKV round trip),
                // ONE barrier per K tile on a 2-stage issue-after-barrier ring:
                //   wait -> SYNC_Q(dt) -> gather K(dt+1) into the stage mma(dt-1) drained -> mma(dt)
                // Per-lane constants: fragment row = n0 + lane/4, byte pairs (lane%4)*2 and +8;
                // koff = row base (bits 6+) + pair offset (bits 1-2) + swizzle key (bits 4-5:
                // (row>>1)&3 == (lane>>3)&3 for every n-tile because n0 % 8 == 0), so the
                // swizzled granule of d-step dl is koff ^ (dl*16) and n-tile n adds n*512.
                const int koff = (nh * 32 + (lane >> 2)) * DM_DTILE + (lane & 3) * 2
                               + (((lane >> 3) & 3) << 4);
                if constexpr (kOpt < 2) {
                    gather_scales_at(sm, sm.sTokPtr, sm.sValid);          // rung 1: this
                    gather_fp8_tile_swz(sm, 0, 0, sm.sTokPtr, sm.sValid);  // block's scales +
                    d_cp_commit();    // group {Scl(kb), K0(kb)}            // K0 issued here
                }
                for (int dt = 0; dt < DM_NFP8TILES; ++dt) {
                    const int cur = dt & 1;
                    // Ledger (per thread): dt=0 pending {Scl+K0} (rung 1) / {K0(kb)},{Scl(kb)}
                    // (rung 2: from the previous block's vt=6/7 or the prologue) -> wait<0>;
                    // dt=1 pending {K1},{rope} -> wait<1> retires K1 (the rope may still fly);
                    // dt>=2 pending {rope} (dt=2 only),{K(dt)} -> wait<0>.
                    if (dt == 1) d_cp_wait<1>(); else d_cp_wait<0>();
                    __syncthreads();  // SYNC_Q(dt). RAW: stage cur (at dt=0 also sScl + the
                                      // pointer buffers, at dt>=2 the rope in sKV) visible to
                                      // every warp. WAR: every warp is past its mma(dt-1) reads
                                      // of stage cur^1, which the gather below overwrites -- and
                                      // at dt=0 past the previous block's V mma(7) reads of sKV,
                                      // which the rope gather overwrites.
                    if (dt + 1 < DM_NFP8TILES) {
                        gather_fp8_tile_swz(sm, cur ^ 1, (dt + 1) * DM_DTILE, sm.sTokPtr, sm.sValid);
                        d_cp_commit();    // group {K(dt+1)}: lands under mma(dt)
                        if (dt == 0) {
                            gather_rope_tile_skv(sm, sm.sTokPtr, sm.sValid);
                            d_cp_commit(); // group {rope}: retired by dt=2's wait<0>, read after dt=7
                        }
                    } else if constexpr (kOpt >= 2) {
                        // Rung 2(a): V tile 0 into stage 0 NOW -- its last readers, mma(6), are
                        // behind SYNC_Q(7) -- so it lands under mma(7) + rope mma + softmax.
                        gather_fp8_tile_swz(sm, 0, 0, sm.sTokPtr, sm.sValid);
                        d_cp_commit();    // group {V0}
                    }
                    float sc[4];          // this lane's 4 fragment rows' scales for this d-tile
                    #pragma unroll
                    for (int n = 0; n < 4; ++n)
                        sc[n] = sm.sScl[(nh * 4 + n) * DM_MMA_N + (lane >> 2)][dt >> 1];
                    #pragma unroll
                    for (int dl = 0; dl < 4; ++dl) {
                        uint32_t Qr[4];
                        {
                            const int row = wrow + (lane % 16);
                            const int col = dt * DM_DTILE + dl * DM_MMA_K + (lane / 16) * 8;
                            d_ldm_x4(Qr, d_cvta(&sm.sQ[d_swz(row, col, DM_DQK)]));
                        }
                        const int kd = koff ^ (dl * 16);   // swizzled 16 B granule of d-step dl
                        #pragma unroll
                        for (int n = 0; n < 4; ++n) {
                            uint32_t Kr[2];
                            d_kfrag_regdq(Kr, &sm.sFp8[cur][kd + n * (DM_MMA_N * DM_DTILE)], sc[n]);
                            d_mma(Sr[n], Qr, Kr, Sr[n]);
                        }
                    }
                }
                {   // rope d-tile (dt == 8): bf16 from sKV. Landed by dt=2's wait<0> on every
                    // thread and published by SYNC_Q(2..7): no wait, no barrier here.
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
                            d_ldm_x2(Kr, d_cvta(&sm.sKV[d_swz(krow, kcol, DM_DTILE)]));
                            d_mma(Sr[n], Qr, Kr, Sr[n]);
                        }
                    }
                }
            } else {
            gather_scales(sm);
            gather_fp8_tile(sm, 0, 0);
            d_cp_commit();
            for (int dt = 0; dt < DM_NFP8TILES; ++dt) {
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
            {   // rope d-tile (dt == 8): bf16, straight from the sFp8 union region
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
            }

            // ---- mask invalid tokens, scale, per-lane partial max -----------
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

            // ---- cross-warp (n-half) row-max combine via smem ---------------
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

            // ---- online rescale + P = exp(S - max), pack to smem ------------
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
            if constexpr (kOpt >= 1) {
                // ---- OPT>=1 PV: the V path is structurally OPT=0 (2-stage gather, smem
                // dequant into sKV, x2.trans) on the swizzled stages. Rung 2: V0 is already in
                // flight (issued at dt=7) and the NEXT block's K tile 0 / scales are issued at
                // vt=6 / vt=7, so the vt=7 wait leaves exactly ONE (newer) group pending. No
                // block-bottom barrier (note after the loop).
                if constexpr (kOpt < 2) {
                    gather_fp8_tile_swz(sm, 0, 0, sm.sTokPtr, sm.sValid);  // V0 -> stage 0: its last
                    d_cp_commit();    // group {V0}                        // readers, mma(6), are behind SYNC_Q(7)
                }
                #pragma unroll
                for (int vt = 0; vt < DM_NVTILES; ++vt) {
                    const int cur = vt & 1;
                    if (vt + 1 < DM_NVTILES) {
                        // stage cur^1: last readers dequant(vt-1), behind SYNC_P2(vt-1); for
                        // vt=0 the K-path mma(7) reads of stage 1, behind SYNC_M1/SYNC_M2.
                        gather_fp8_tile_swz(sm, cur ^ 1, (vt + 1) * DM_DTILE, sm.sTokPtr, sm.sValid);
                        d_cp_commit();    // group {V(vt+1)}
                        d_cp_wait<1>();   // pending {V(vt)},{V(vt+1)}: the older, V(vt), landed
                    } else if constexpr (kOpt >= 2) {
                        d_cp_wait<1>();   // pending {V7},{K0(kb+1)}: V7 (older) landed, K0 may fly
                    } else {
                        d_cp_wait<0>();   // pending {V7}
                    }
                    __syncthreads();      // SYNC_P1(vt): stage cur landed for all; prior V mma done (sKV free)
                    dequant_tile_swz(sm, cur, vt);
                    __syncthreads();      // SYNC_P2(vt): sKV (V tile) + sP visible; stage cur consumed by all
                    if constexpr (kOpt >= 2) {
                        if (vt == 6) {
                            // Rung 2(b): the NEXT block's K tile 0 into stage 0 (tile 6 consumed
                            // by every thread at SYNC_P2(6)); pointers from the parity buffer,
                            // resolved at this block's start and published by SYNC_Q(0). At the
                            // walk's last block this is an all-invalid zero-fill (d_phase_a).
                            gather_fp8_tile_swz(sm, 0, 0, sm.sTokPtrN, sm.sValidN);
                            d_cp_commit();    // group {K0(kb+1)}
                        }
                        if (vt == 7) {
                            // The NEXT block's scales into the single sScl: this block's last
                            // readers of it (dequant vt=7; the K path long before) are behind
                            // SYNC_P2(7). Both prefetch groups retire at the next dt=0 wait<0>.
                            gather_scales_at(sm, sm.sTokPtrN, sm.sValidN);
                            d_cp_commit();    // group {Scl(kb+1)}
                        }
                    }
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
                // No block-bottom barrier under OPT>=1. The next block's first writes are:
                // the pointer buffers (tid<64, after every reader of them passed SYNC_P1(6)
                // at the latest -- phase-A notes), sScl / stage 0 (rung 1: issued after
                // SYNC_A; rung 2: issued above at vt=7 / vt=6, behind SYNC_P2(7) / SYNC_P2(6),
                // their last readers), stage 1 and sKV (after SYNC_Q(0), which every warp
                // reaches only after its own mma(7)), and sP (after SYNC_M2 of the next block).
            } else {
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
        }
        if constexpr (kOpt >= 2) d_cp_wait<0>();   // retire the walk's last (all-invalid, zero-
                                                   // fill) K0/scales prefetch before the next
                                                   // batch re-targets stage 0 / sScl

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
            // Finals: the certified CFG=1 conventions verbatim (natural lse,
            // +INFINITY sentinel == the combine's own all-empty output).
            __nv_bfloat16* o_row_base = (__nv_bfloat16*)params.o_ptr
                + (int64_t)batch_idx * params.o_batch_stride
                + (int64_t)row0 * params.o_row_stride;
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
            if (nh == 0 && (lane % 4) == 0) {
                float* lse_base = params.softmax_lse_ptr
                    + (int64_t)batch_idx * params.q_seq_per_hk + (int64_t)row0;
                if (row_a < num_valid_rows)
                    lse_base[row_a] = (sum0 > 0.f) ? (rmax[0] + __logf(sum0)) : INFINITY;
                if (row_b < num_valid_rows)
                    lse_base[row_b] = (sum1 > 0.f) ? (rmax[1] + __logf(sum1)) : INFINITY;
            }
        } else {
            // Partials in the authors' accum layout: NORMALIZED fp32 O + 2-based
            // lse (-INFINITY for an empty split); the combine kernel merges.
            const int split_row = split_base + n_split_idx;
            float* oaccum = params.oaccum_ptr
                + ((int64_t)split_row * params.q_seq_per_hk + row0) * DM_DV;
            #pragma unroll
            for (int t = 0; t < DM_NVTILES; ++t)
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const int ocol = t * DM_DTILE + nh * 32 + n * DM_MMA_N + (lane % 4) * 2;
                    if (row_a < num_valid_rows) {
                        float* pa = oaccum + (int64_t)row_a * DM_DV + ocol;
                        pa[0] = Or[t][n][0] * inv0; pa[1] = Or[t][n][1] * inv0;
                    }
                    if (row_b < num_valid_rows) {
                        float* pb = oaccum + (int64_t)row_b * DM_DV + ocol;
                        pb[0] = Or[t][n][2] * inv1; pb[1] = Or[t][n][3] * inv1;
                    }
                }
            if (nh == 0 && (lane % 4) == 0) {
                float* lse_accum = params.softmax_lseaccum_ptr
                    + (int64_t)split_row * params.q_seq_per_hk + row0;
                if (row_a < num_valid_rows)
                    lse_accum[row_a] = (sum0 > 0.f) ? (rmax[0] + __logf(sum0)) * DM_LOG2E
                                                    : -INFINITY;
                if (row_b < num_valid_rows)
                    lse_accum[row_b] = (sum1 > 0.f) ? (rmax[1] + __logf(sum1)) * DM_LOG2E
                                                    : -INFINITY;
            }
        }
        __syncthreads();   // stats/smem free before the next batch reuses them
    }
}

template <int kOpt>
inline void launch_sparse_fp8_decode_mma_splitkv_opt(const SparseFP8DecodeParams& params,
                                                     const dim3& grid) {
    using Plan = std::conditional_t<(kOpt >= 2), SmemPlanDecodeMmaOpt2, SmemPlanDecodeMma>;
    constexpr size_t smem = sizeof(Plan);
    CHECK_CUDA(cudaFuncSetAttribute(sparse_fp8_decode_mma_splitkv_kernel<kOpt>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    sparse_fp8_decode_mma_splitkv_kernel<kOpt><<<grid, dim3(DM_THREADS), smem, params.stream>>>(params);
}

inline void launch_sparse_fp8_decode_mma_splitkv(const SparseFP8DecodeParams& params) {
    const int num_m_blocks = (params.q_head_per_hk + DM_BH - 1) / DM_BH;
    const dim3 grid(num_m_blocks, params.s_q, params.num_sm_parts);
    // OPT ladder (FLASH_MLA_SM120_SPARSE_DECODE_OPT, read once): 0 = legacy byte-identical.
    switch (sparse_decode_opt_rung()) {
        case 1:  launch_sparse_fp8_decode_mma_splitkv_opt<1>(params, grid); break;
        case 2:  launch_sparse_fp8_decode_mma_splitkv_opt<2>(params, grid); break;
        default: launch_sparse_fp8_decode_mma_splitkv_opt<0>(params, grid); break;
    }
}

// ============================================================================
// CLUSTER CROSSOVER tier [CFG>=3]: the split-KV tier + the authors' sm90 H800
// "crossover" transformed to plain sm_120 (audit/design-sparse-splitkv.md
// Phase 2; every primitive ptxas- AND runtime-proven on this silicon by
// tests/_probe_cluster_dsm_*). 2-CTA cluster along x: the pair owns DIFFERENT
// 64-head blocks but shares the (batch, s_q) topk index list, so each CTA
// gathers + dequants only ITS 32-token half per tile and st.asyncs the
// dequantized bf16 into its peer's exchange buffer -- halving fp8 gmem
// traffic, dequant ALU, and staging smem writes.
//
// Race-free by construction (design doc "Phase 2"):
//   - sL = my dequanted half (local WAR via the existing syncthreads),
//     sX[2] = peer's half, DOUBLE-buffered: peer's writes for tile T+2 need my
//     T+1 credits, which follow my completed mma(T) -- distance-2 safety, no
//     reverse avail-barrier needed (sm90's doesn't fit in 99KB).
//   - ARM-BEFORE-CREDIT proven two ways: the first two tiles' expect_tx are
//     armed BEFORE the initial cluster fence (peer credits cannot predate its
//     fence exit, which follows my arrival, which follows my arm); steady
//     state re-arms bar[T&1] for tile T+2 immediately after the T-wait, and
//     peer's T+2 credits need my T+1 st.async which comes later in my program
//     order. No reliance on signed tx-count semantics.
//   - Tile chain alternates bars strictly (QK dt 0..7, PV vt 0..7; 7 odd ->
//     next even), continuing seamlessly across blocks and batches.
//   - Rope tile is NOT crossed (bf16 already, L2-absorbed): full-gathered by
//     both CTAs into sP, which is dead during QK; the sStatM softmax syncs
//     already separate the rope mma reads from the P stores (same structure
//     the certified tiers rely on).
//   - The padding CTA of an odd head-block count (num_valid_rows <= 0) writes
//     NO output but runs FULL producer duties -- sm90's free load balancing.
//   - Final cluster fence: no CTA exits while its peer might still push.
// ============================================================================

constexpr int DM_XH = 32;   // tokens per CTA half in the crossover

// ---- cluster/DSM primitives (runtime-proven on plain sm_120) ---------------
__device__ __forceinline__ uint32_t d_cluster_rank() {
    uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
}
__device__ __forceinline__ uint32_t d_mapa(uint32_t sa, uint32_t rank) {
    uint32_t peer;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(peer) : "r"(sa), "r"(rank));
    return peer;
}
__device__ __forceinline__ void d_bar_init(uint32_t bar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count));
}
__device__ __forceinline__ void d_bar_arrive_expect_tx(uint32_t bar, int bytes) {
    uint64_t st;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;"
                 : "=l"(st) : "r"(bar), "r"(bytes));
}
__device__ __forceinline__ void d_bar_wait_parity(uint32_t bar, uint32_t phase) {
    uint32_t done = 0;
    while (!done)
        asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; "
                     "selp.u32 %0, 1, 0, p;}" : "=r"(done) : "r"(bar), "r"(phase));
}
__device__ __forceinline__ void d_st_async16(uint32_t peer_sa, const bf16x8& v, uint32_t peer_bar) {
    const int64_t* p = reinterpret_cast<const int64_t*>(&v);
    asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64 "
                 "[%0], {%1, %2}, [%3];" :: "r"(peer_sa), "l"(p[0]), "l"(p[1]), "r"(peer_bar));
}
__device__ __forceinline__ void d_cluster_arrive_wait() {
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

// ---- crossover smem plan (100,608 B <= 101,376) -----------------------------
struct SmemPlanDecodeMmaX {
    __align__(128) __nv_bfloat16 sQ[DM_BH * DM_DQK];        // 73,728 B, swz(576)
    __align__(128) uint8_t sFp8[2][DM_XH * DM_DTILE];       // 2 x 2,048 B: MY half only
    __align__(128) __nv_bfloat16 sL[DM_XH * DM_DTILE];      // 4,096 B: my dequanted half
    __align__(128) __nv_bfloat16 sX[2][DM_XH * DM_DTILE];   // 2 x 4,096 B: peer halves
    __align__(128) __nv_bfloat16 sP[DM_BH * DM_BTOPK];      // 8,192 B (+ rope home in QK)
    __align__(128) float sStatM[DM_BH];
    __align__(128) float sStatS[DM_BH];
    __align__(128) float sScl[DM_BTOPK][4];
    __align__(128) const uint8_t* sTokPtr[DM_BTOPK];
    __align__(128) int8_t sValid[DM_BTOPK];
    __align__(128) uint64_t sBarX[2];                       // exchange tx mbarriers
};
static_assert(sizeof(SmemPlanDecodeMmaX) <= 99 * 1024, "crossover smem exceeds 99KB");

// Gather MY 32-token half of one 64-col fp8 tile: 128 granules, threads 0..127.
__device__ __forceinline__ void gather_fp8_half(SmemPlanDecodeMmaX& sm, int st, int col_start,
                                                uint32_t my_rank) {
    const int tid = threadIdx.x;
    if (tid < 128) {
        const int rl = tid >> 2;                    // local row 0..31
        const int c = (tid & 3) * 16;
        const int tok = (int)my_rank * DM_XH + rl;  // row in the 64-token block
        const uint8_t* tp = sm.sTokPtr[tok];
        d_cp16(d_cvta(&sm.sFp8[st][rl * DM_DTILE + c]), tp + col_start + c,
               sm.sValid[tok] ? 16 : 0);
    }
}

// Dequant MY half of stage `st`: bf16 into sL (local) AND the peer's sX[st]
// via st.async (128 thr x 2 x 16 B = 4,096 B credited to the peer's bar).
__device__ __forceinline__ void dequant_half_x(SmemPlanDecodeMmaX& sm, int st, int dt,
                                               uint32_t my_rank, uint32_t peer_bar_sa) {
    const int tid = threadIdx.x;
    if (tid < 128) {
        const int rl = tid >> 2;
        const int c = (tid & 3) * 16;
        const int tok = (int)my_rank * DM_XH + rl;
        const float scale = sm.sScl[tok][dt >> 1];
        const fp8x16 raw = *reinterpret_cast<const fp8x16*>(&sm.sFp8[st][rl * DM_DTILE + c]);
        const bf16x8 lo = cvt_fp8x8_bf16x8_fp32(raw.lo, scale);
        const bf16x8 hi = cvt_fp8x8_bf16x8_fp32(raw.hi, scale);
        store_128b(&sm.sL[d_swz(rl, c, DM_DTILE)], lo);
        store_128b(&sm.sL[d_swz(rl, c + 8, DM_DTILE)], hi);
        const uint32_t peer = my_rank ^ 1u;
        d_st_async16(d_mapa(d_cvta(&sm.sX[st][d_swz(rl, c, DM_DTILE)]), peer), lo, peer_bar_sa);
        d_st_async16(d_mapa(d_cvta(&sm.sX[st][d_swz(rl, c + 8, DM_DTILE)]), peer), hi, peer_bar_sa);
    }
}

// Scales full-gather (both CTAs, redundant -- cheaper than a second DSM
// channel; sm90 does the same for validity). Same mapping as gather_scales.
__device__ __forceinline__ void gather_scales_x(SmemPlanDecodeMmaX& sm) {
    const int tid = threadIdx.x;
    if (tid < DM_BTOPK) {
        const uint8_t* tp = sm.sTokPtr[tid];
        d_cp16(d_cvta(&sm.sScl[tid][0]), tp + HEAD_DIM_NOPE, sm.sValid[tid] ? 16 : 0);
    }
}

// Rope tile full-gather into sP (both CTAs; NOT crossed). Same mapping as
// gather_rope_tile, destination sP.
__device__ __forceinline__ void gather_rope_to_sp(SmemPlanDecodeMmaX& sm) {
    const int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int g = tid + i * DM_THREADS;
        const int r = g >> 3;
        const int c = (g & 7) * 8;
        const uint8_t* tp = sm.sTokPtr[r];
        d_cp16(d_cvta(&sm.sP[d_swz(r, c, DM_DTILE)]),
               tp + DM_ROPE_BYTE_OFF + c * 2, sm.sValid[r] ? 16 : 0);
    }
}

__global__ void __launch_bounds__(DM_THREADS, 1) __cluster_dims__(2, 1, 1)
sparse_fp8_decode_mma_splitkv_x_kernel(const SparseFP8DecodeParams params) {
    extern __shared__ char smem_raw[];
    SmemPlanDecodeMmaX& sm = *reinterpret_cast<SmemPlanDecodeMmaX*>(smem_raw);

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int m_block_idx = blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int part_idx = blockIdx.z;
    const uint32_t my_rank = d_cluster_rank();          // == blockIdx.x & 1

    // ---- exchange barriers: init -> CTA sync -> ARM first two tiles -> fence.
    // The arm precedes my cluster arrival; peer credits follow ITS fence exit,
    // which follows my arrival: arm-before-credit PROVEN for tiles 0 and 1.
    const uint32_t bar0_sa = d_cvta(&sm.sBarX[0]);
    const uint32_t bar1_sa = d_cvta(&sm.sBarX[1]);
    if (tid == 0) { d_bar_init(bar0_sa, 1); d_bar_init(bar1_sa, 1); }
    __syncthreads();
    if (tid == 0) {
        d_bar_arrive_expect_tx(bar0_sa, 4096);
        d_bar_arrive_expect_tx(bar1_sa, 4096);
    }
    d_cluster_arrive_wait();
    const uint32_t peer_bar0 = d_mapa(bar0_sa, my_rank ^ 1u);
    const uint32_t peer_bar1 = d_mapa(bar1_sa, my_rank ^ 1u);
    uint32_t xphase0 = 0, xphase1 = 0;

    // A padding CTA (odd head-block count rounded up to a full cluster pair)
    // clamps to zero valid rows: no output writes, FULL producer duties.
    const int head_start = m_block_idx * DM_BH;
    const int num_valid_rows = min(params.q_head_per_hk - head_start, DM_BH);
    const int row0 = s_q_idx * params.q_head_per_hk + head_start;

    const int* meta = params.tile_scheduler_metadata_ptr
        + part_idx * TileSchedulerMetaDataSize;
    const int begin_idx = meta[0];
    const int begin_block = meta[1];
    const int end_idx = meta[2];
    const int end_block_last = meta[3];
    const int begin_n_split_idx = meta[4];

    const int num_topk_blocks = (params.topk + DM_BTOPK - 1) / DM_BTOPK;

    const uint8_t* kv_base = (const uint8_t*)params.kv_ptr;
    const int ms = warp & 3, nh = warp >> 2;
    const int wrow = ms * DM_MMA_M;
    const int row_a = wrow + lane / 4, row_b = row_a + 8;

    for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx) {
        const int blk_lo = (batch_idx == begin_idx) ? begin_block : 0;
        const int blk_hi_sched = (batch_idx == end_idx) ? end_block_last : num_topk_blocks;
        const int blk_hi = min(blk_hi_sched, num_topk_blocks);
        const int n_split_idx = (batch_idx == begin_idx) ? begin_n_split_idx : 0;
        const int split_base = __ldg(params.num_splits_ptr + batch_idx);
        const bool no_split = (__ldg(params.num_splits_ptr + batch_idx + 1) - split_base) == 1;

        const __nv_bfloat16* q_row_base = (const __nv_bfloat16*)params.q_ptr
            + (int64_t)batch_idx * params.q_batch_stride
            + (int64_t)row0 * params.q_row_stride;
        const int* idx_row = params.indices_ptr
            + (int64_t)batch_idx * params.indices_batch_stride
            + (int64_t)s_q_idx * params.indices_seq_stride;

        {
            #pragma unroll
            for (int i = 0; i < 18; ++i) {
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

        float rmax[2] = {-INFINITY, -INFINITY}, rsum[2] = {0.f, 0.f};
        float Or[DM_NVTILES][4][4];
        #pragma unroll
        for (int t = 0; t < DM_NVTILES; ++t)
            #pragma unroll
            for (int n = 0; n < 4; ++n) { Or[t][n][0] = Or[t][n][1] = Or[t][n][2] = Or[t][n][3] = 0.f; }

        __syncthreads();   // sQ visible; previous batch's smem fully consumed

        for (int kb = blk_lo; kb < blk_hi; ++kb) {
            if (tid < DM_BTOPK) {
                const int k = kb * DM_BTOPK + tid;
                const int token = (k < params.topk) ? idx_row[k] : -1;
                const bool ok = (token >= 0);
                sm.sValid[tid] = ok ? 1 : 0;
                sm.sTokPtr[tid] = ok ? (kv_base
                                        + (int64_t)(token >> 6) * params.kv_page_stride
                                        + (int64_t)(token & 63) * params.kv_token_stride)
                                     : kv_base;
            }
            __syncthreads();

            // Mma row bases: rows [0,32) are rank-0's tokens, [32,64) rank-1's.
            // My sL holds MY tokens; sX[st] receives the peer's.
            const __nv_bfloat16* base_lo_l = (my_rank == 0) ? sm.sL : nullptr;
            const __nv_bfloat16* base_hi_l = (my_rank == 0) ? nullptr : sm.sL;

            // ---- QK: 8 crossed fp8 d-tiles + 1 uncrossed rope tile ----------
            float Sr[4][4];
            #pragma unroll
            for (int n = 0; n < 4; ++n) { Sr[n][0] = Sr[n][1] = Sr[n][2] = Sr[n][3] = 0.f; }

            gather_scales_x(sm);
            gather_fp8_half(sm, 0, 0, my_rank);
            d_cp_commit();
            for (int dt = 0; dt < DM_NFP8TILES; ++dt) {
                const int cur = dt & 1;
                if (dt + 1 < DM_NFP8TILES) {
                    gather_fp8_half(sm, cur ^ 1, (dt + 1) * DM_DTILE, my_rank);
                    d_cp_commit();
                    d_cp_wait<1>();
                } else {
                    d_cp_wait<0>();
                }
                __syncthreads();          // my fp8 half landed; sL free; sX[cur] free (distance-2)
                dequant_half_x(sm, cur, dt, my_rank, cur ? peer_bar1 : peer_bar0);
                // Single-waiter: only t0 polls (255 spinners would contend the
                // MIO/smem pipe the peer's st.asyncs are landing in); the
                // following __syncthreads hands the completed phase to all
                // threads (t0's observation happens-before their sX reads).
                if (tid == 0) {
                    if (cur == 0) d_bar_wait_parity(bar0_sa, xphase0);
                    else          d_bar_wait_parity(bar1_sa, xphase1);
                    d_bar_arrive_expect_tx(cur ? bar1_sa : bar0_sa, 4096);  // re-arm for tile+2
                }
                if (cur == 0) xphase0 ^= 1; else xphase1 ^= 1;
                __syncthreads();          // sL (all writers) + sX[cur] visible to all
                if (dt == DM_NFP8TILES - 1) {
                    gather_rope_to_sp(sm);   // sP is dead until softmax
                    d_cp_commit();
                }
                #pragma unroll
                for (int dl = 0; dl < 4; ++dl) {
                    uint32_t Qr[4];
                    {
                        const int rowq = wrow + (lane % 16);
                        const int col = dt * DM_DTILE + dl * DM_MMA_K + (lane / 16) * 8;
                        d_ldm_x4(Qr, d_cvta(&sm.sQ[d_swz(rowq, col, DM_DQK)]));
                    }
                    #pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        uint32_t Kr[2];
                        const int krow = (nh * 4 + n) * DM_MMA_N + (lane % 8);
                        const int kcol = dl * DM_MMA_K + ((lane / 8) % 2) * 8;
                        const __nv_bfloat16* kb8 = (krow < DM_XH)
                            ? (base_lo_l ? base_lo_l : sm.sX[cur])
                            : (base_hi_l ? base_hi_l : sm.sX[cur]);
                        d_ldm_x2(Kr, d_cvta(&kb8[d_swz(krow & (DM_XH - 1), kcol, DM_DTILE)]));
                        d_mma(Sr[n], Qr, Kr, Sr[n]);
                    }
                }
            }
            {   // rope d-tile: bf16 from sP (uncrossed full gather)
                d_cp_wait<0>();
                __syncthreads();          // rope landed + every thread past its last fp8 mma
                #pragma unroll
                for (int dl = 0; dl < 4; ++dl) {
                    uint32_t Qr[4];
                    {
                        const int rowq = wrow + (lane % 16);
                        const int col = DM_DV + dl * DM_MMA_K + (lane / 16) * 8;
                        d_ldm_x4(Qr, d_cvta(&sm.sQ[d_swz(rowq, col, DM_DQK)]));
                    }
                    #pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        uint32_t Kr[2];
                        const int krow = (nh * 4 + n) * DM_MMA_N + (lane % 8);
                        const int kcol = dl * DM_MMA_K + ((lane / 8) % 2) * 8;
                        d_ldm_x2(Kr, d_cvta(&sm.sP[d_swz(krow, kcol, DM_DTILE)]));
                        d_mma(Sr[n], Qr, Kr, Sr[n]);
                    }
                }
            }

            // ---- mask, scale, online softmax (identical to the certified core)
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

            if (nh == 0 && (lane % 4) == 0) {
                sm.sStatM[wrow + lane / 4] = tmax[0];
                sm.sStatM[wrow + lane / 4 + 8] = tmax[1];
            }
            __syncthreads();              // (also: rope mma reads done before P stores)
            if (nh == 1 && (lane % 4) == 0) {
                sm.sStatM[wrow + lane / 4] = fmaxf(sm.sStatM[wrow + lane / 4], tmax[0]);
                sm.sStatM[wrow + lane / 4 + 8] = fmaxf(sm.sStatM[wrow + lane / 4 + 8], tmax[1]);
            }
            __syncthreads();
            const float bmax0 = sm.sStatM[wrow + lane / 4];
            const float bmax1 = sm.sStatM[wrow + lane / 4 + 8];

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

            // ---- PV: 8 crossed V tiles (re-gathered + re-dequantized halves)
            gather_fp8_half(sm, 0, 0, my_rank);
            d_cp_commit();
            #pragma unroll
            for (int vt = 0; vt < DM_NVTILES; ++vt) {
                const int cur = vt & 1;
                if (vt + 1 < DM_NVTILES) {
                    gather_fp8_half(sm, cur ^ 1, (vt + 1) * DM_DTILE, my_rank);
                    d_cp_commit();
                    d_cp_wait<1>();
                } else {
                    d_cp_wait<0>();
                }
                __syncthreads();          // my half landed; sL free; sX[cur] free (distance-2)
                dequant_half_x(sm, cur, vt, my_rank, cur ? peer_bar1 : peer_bar0);
                if (tid == 0) {           // single-waiter (see QK loop note)
                    if (cur == 0) d_bar_wait_parity(bar0_sa, xphase0);
                    else          d_bar_wait_parity(bar1_sa, xphase1);
                    d_bar_arrive_expect_tx(cur ? bar1_sa : bar0_sa, 4096);
                }
                if (cur == 0) xphase0 ^= 1; else xphase1 ^= 1;
                __syncthreads();          // sL + sX[cur] (V tile) + sP visible
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
                        const __nv_bfloat16* vb8 = (vrow < DM_XH)
                            ? (base_lo_l ? base_lo_l : sm.sX[cur])
                            : (base_hi_l ? base_hi_l : sm.sX[cur]);
                        d_ldm_x2_trans(Vr, d_cvta(&vb8[d_swz(vrow & (DM_XH - 1), vcol, DM_DTILE)]));
                        d_mma(Or[vt][n], Pr, Vr, Or[vt][n]);
                    }
                }
            }
            __syncthreads();   // last V mma done before next kb rewrites sTokPtr/sFp8/sP
        }

        // ---- epilogue (identical to the split-KV tier) ----------------------
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
            __nv_bfloat16* o_row_base = (__nv_bfloat16*)params.o_ptr
                + (int64_t)batch_idx * params.o_batch_stride
                + (int64_t)row0 * params.o_row_stride;
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
            if (nh == 0 && (lane % 4) == 0) {
                float* lse_base = params.softmax_lse_ptr
                    + (int64_t)batch_idx * params.q_seq_per_hk + (int64_t)row0;
                if (row_a < num_valid_rows)
                    lse_base[row_a] = (sum0 > 0.f) ? (rmax[0] + __logf(sum0)) : INFINITY;
                if (row_b < num_valid_rows)
                    lse_base[row_b] = (sum1 > 0.f) ? (rmax[1] + __logf(sum1)) : INFINITY;
            }
        } else {
            const int split_row = split_base + n_split_idx;
            float* oaccum = params.oaccum_ptr
                + ((int64_t)split_row * params.q_seq_per_hk + row0) * DM_DV;
            #pragma unroll
            for (int t = 0; t < DM_NVTILES; ++t)
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const int ocol = t * DM_DTILE + nh * 32 + n * DM_MMA_N + (lane % 4) * 2;
                    if (row_a < num_valid_rows) {
                        float* pa = oaccum + (int64_t)row_a * DM_DV + ocol;
                        pa[0] = Or[t][n][0] * inv0; pa[1] = Or[t][n][1] * inv0;
                    }
                    if (row_b < num_valid_rows) {
                        float* pb = oaccum + (int64_t)row_b * DM_DV + ocol;
                        pb[0] = Or[t][n][2] * inv1; pb[1] = Or[t][n][3] * inv1;
                    }
                }
            if (nh == 0 && (lane % 4) == 0) {
                float* lse_accum = params.softmax_lseaccum_ptr
                    + (int64_t)split_row * params.q_seq_per_hk + row0;
                if (row_a < num_valid_rows)
                    lse_accum[row_a] = (sum0 > 0.f) ? (rmax[0] + __logf(sum0)) * DM_LOG2E
                                                    : -INFINITY;
                if (row_b < num_valid_rows)
                    lse_accum[row_b] = (sum1 > 0.f) ? (rmax[1] + __logf(sum1)) * DM_LOG2E
                                                    : -INFINITY;
            }
        }
        __syncthreads();   // stats/smem free before the next batch reuses them
    }

    // No CTA exits while its peer might still push into its smem.
    d_cluster_arrive_wait();
}

inline void launch_sparse_fp8_decode_mma_splitkv_x(const SparseFP8DecodeParams& params) {
    const int m_real = (params.q_head_per_hk + DM_BH - 1) / DM_BH;
    const int m_pad = (m_real + 1) & ~1;    // full cluster pairs (pad CTA = producer only)
    const dim3 grid(m_pad, params.s_q, params.num_sm_parts);
    constexpr size_t smem = sizeof(SmemPlanDecodeMmaX);
    CHECK_CUDA(cudaFuncSetAttribute(sparse_fp8_decode_mma_splitkv_x_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    sparse_fp8_decode_mma_splitkv_x_kernel<<<grid, dim3(DM_THREADS), smem, params.stream>>>(params);
}

}  // namespace mma
}  // namespace sparse_decode
}  // namespace sm120
