#pragma once
// ============================================================================
// SPARSE-FP8 DECODE with FP8 TENSOR MATH (SM120)   [CFG=5 batch-parallel, CFG=6 split-KV]
// ============================================================================
// Build opt-in: compiled only under -DFLASH_MLA_SM120_FP8_MMA (setup.py env FLASH_MLA_SM120_FP8_MMA=1).
// Design + numerics contract: audit/design-sparse-decode-fp8mma.md. Verified primitives:
// tests/_probe_fp8_mma.cu (m16n8k32 e4m3 fragment map, token-permuted V construction, 1078 TF).
//
// Semantics (== the certified bf16-math tiers): NO block_table (page = idx/64, offset = idx%64,
// int64 math); token == -1 is the only invalid form, invalid rows never read (src-size-0
// zero-fill); KV row = 656 B [0,512) e4m3 nope | [512,528) 4 fp32 scales | [528,656) 64 bf16
// rope; V = the FIRST 512 bytes of the same rows; lse natural log, +INFINITY sentinel; ragged
// head tail zero-filled and write-guarded. Split-KV tier: authors' scheduler walk + partials.
//
// What is DIFFERENT (operator-approved fp8-math contract, not the authors' bf16-Q design):
//   Q nope quantized per head row to e4m3 (qs = amax/448); S = qs * sum_g sk[t][g] * (Q8.K8^T)_g
//   + rope (bf16); the PV operand is P'_g = 256 exp(S + ln sk[t][g] - Mg) in e4m3 against the RAW
//   V bytes, with Mg = ONLINE per-(row, group) reference (softmax-rescale pattern); epilogue
//   O_g = Og * exp(Mg - m) / (256 L). LSE is computed from fp32 P and is unchanged in kind.
// ============================================================================

#include <cuda_fp8.h>

namespace sm120 {
namespace sparse_decode {
namespace fp8mma {

using mma::d_cvta; using mma::d_ldm_x4; using mma::d_ldm_x2; using mma::d_mma;
using mma::d_cp16; using mma::d_cp_commit; using mma::d_cp_wait; using mma::d_swz;
using mma::d_phase_a;
using mma::DM_THREADS; using mma::DM_BH; using mma::DM_BTOPK; using mma::DM_DQK; using mma::DM_DV;
using mma::DM_MMA_M; using mma::DM_MMA_N; using mma::DM_MMA_K; using mma::DM_ROPE_BYTE_OFF;
using mma::DM_LOG2E;

constexpr int F_NG = 4;                 // 128-d quant groups per token (== fp8 tiles per block)
constexpr int F_GB = 128;               // bytes per group per token
constexpr int F_ROPE = 64;              // bf16 rope elements
constexpr float F_PSCALE = 256.f;       // P' headroom (< 448 satfinite): argmax token -> exactly 256
constexpr float F_PINV = 1.f / 256.f;

// ---- PTX primitives ---------------------------------------------------------
__device__ __forceinline__ void f_mma_e4m3(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void f_ldm_x4_trans(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ uint32_t f_prmt(uint32_t lo, uint32_t hi, uint32_t sel) {
    uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(sel)); return r;
}
// Two fp32 -> packed e4m3x2 (RN, satfinite): x0 in the LOW byte, x1 in the HIGH byte (PTX
// cvt.rn.satfinite.e4m3x2.f32 d, a, b packs a high / b low). Verified by the probe.
__device__ __forceinline__ uint32_t f_pack2(float x0, float x1) {
    uint16_t v; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(v) : "f"(x1), "f"(x0)); return v;
}
__device__ __forceinline__ uint32_t f_pack4(float x0, float x1, float x2, float x3) {
    return f_pack2(x0, x1) | (f_pack2(x2, x3) << 16);
}
// Named barrier over the 128 threads of one row-block GROUP (warps {0,1,4,5} = rows 0..31,
// {2,3,6,7} = rows 32..63): the softmax combine and the P' publish only involve a group's own
// rows, so the two groups may drift within the CTA-wide tile barriers (the tensor pipe of one
// group's GEMM phase overlaps the other group's softmax -- the FA-3 ping-pong, self-organized).
// ids 1 / 2 (0 is __syncthreads).
__device__ __forceinline__ void f_bar_group(int grp) {
    asm volatile("bar.sync %0, 128;\n" :: "r"(1 + grp) : "memory");
}

// ---- smem plan (97,152 B <= 101,376) ------------------------------------------
// sQ8  [64][512] B  row-granule swizzle g ^ (r & 7)         32,768
// sQr  bf16 [64][64] swz(64)  Q rope (resident)               8,192
// sKr  bf16 [64][64] swz(64)  K rope tile                     8,192
// sKV  [4][64][128] B group tiles, granule ^ (t & 7)         32,768
// sP8  [2][64][64] B  P'_0, P'_1 then P'_2, P'_3 (after PV(1)) 8,192
// sScl [64][4] raw cp.async scales; sSclT/sLsk [4][64]        3,072
// stats / Q scales / pointer parity buffers                   ~4,000
struct SmemPlanFp8Mma {
    __align__(1024) uint8_t sQ8[DM_BH * DM_DV];
    __align__(128) __nv_bfloat16 sQr[DM_BH * F_ROPE];
    __align__(128) __nv_bfloat16 sKr[DM_BTOPK * F_ROPE];
    __align__(1024) uint8_t sKV[F_NG][DM_BTOPK * F_GB];
    __align__(128) uint8_t sP8[2][DM_BH * DM_BTOPK];
    __align__(128) float sScl[DM_BTOPK][4];
    __align__(128) float sSclT[F_NG][DM_BTOPK];
    __align__(128) float sLsk[F_NG][DM_BTOPK];
    __align__(128) float sQs[DM_BH];
    __align__(128) float sStatM[DM_BH];
    __align__(128) float sStatMg[F_NG][DM_BH];
    __align__(128) float sStatS[DM_BH];
    __align__(128) const uint8_t* sTokPtr[2][DM_BTOPK];
    __align__(128) int8_t sValid[2][DM_BTOPK];
};
static_assert(sizeof(SmemPlanFp8Mma) <= 99 * 1024, "fp8mma smem exceeds 99KB");

__device__ __forceinline__ int f_q8_off(int r, int gi) { return r * DM_DV + ((gi ^ (r & 7)) << 4); }
__device__ __forceinline__ int f_kv_off(int t, int gi) { return t * F_GB + ((gi ^ (t & 7)) << 4); }

// Gather group tile g (64 tokens x 128 B) of the block described by (tokptr, valid) into sKV[g].
// 512 granules / 256 threads = 2 each; 8 consecutive threads cover one row (coalesced).
__device__ __forceinline__ void f_gather_tile(SmemPlanFp8Mma& sm, int g,
                                              const uint8_t* const* tokptr, const int8_t* valid) {
    const int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int idx = tid + i * DM_THREADS;
        const int t = idx >> 3, gi = idx & 7;
        d_cp16(d_cvta(&sm.sKV[g][f_kv_off(t, gi)]), tokptr[t] + g * F_GB + gi * 16, valid[t] ? 16 : 0);
    }
}
__device__ __forceinline__ void f_gather_rope(SmemPlanFp8Mma& sm,
                                              const uint8_t* const* tokptr, const int8_t* valid) {
    const int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int g = tid + i * DM_THREADS;
        const int r = g >> 3, c = (g & 7) * 8;
        d_cp16(d_cvta(&sm.sKr[d_swz(r, c, F_ROPE)]), tokptr[r] + DM_ROPE_BYTE_OFF + c * 2, valid[r] ? 16 : 0);
    }
}
__device__ __forceinline__ void f_gather_scales(SmemPlanFp8Mma& sm,
                                                const uint8_t* const* tokptr, const int8_t* valid) {
    const int tid = threadIdx.x;
    if (tid < DM_BTOPK)
        d_cp16(d_cvta(&sm.sScl[tid][0]), tokptr[tid] + HEAD_DIM_NOPE, valid[tid] ? 16 : 0);
}
// Issue the whole next-block prefetch set in the uniform ledger order {Scl rope}, {K0 K1},
// {K2 K3} (3 commit groups). Used by the walk prologue only; the steady state issues the same
// groups at their WAR-safe points inside the block (Scl+rope after SYNC_Q(2) -- the rope GEMM
// runs right after SYNC_Q(0), so every warp's rope reads precede SYNC_Q(2); K0K1 after PV(1);
// K2K3 after PV(3)).
__device__ __forceinline__ void f_issue_block(SmemPlanFp8Mma& sm,
                                              const uint8_t* const* tokptr, const int8_t* valid) {
    f_gather_scales(sm, tokptr, valid); f_gather_rope(sm, tokptr, valid); d_cp_commit();
    f_gather_tile(sm, 0, tokptr, valid); f_gather_tile(sm, 1, tokptr, valid); d_cp_commit();
    f_gather_tile(sm, 2, tokptr, valid); f_gather_tile(sm, 3, tokptr, valid); d_cp_commit();
}

// Prologue: resident Q for this (batch, head block): nope quantized per row to e4m3 (row scale
// qs = amax/448, 0 for an all-zero / ragged row), rope bf16 swizzled. 4 threads per row, each
// owning one 128-element group (two passes over gmem: amax, then convert -- L2-resident).
__device__ __forceinline__ void f_load_q(SmemPlanFp8Mma& sm, const __nv_bfloat16* q_row_base,
                                         int q_row_stride, int num_valid_rows) {
    const int tid = threadIdx.x;
    const int r = tid >> 2, q4 = tid & 3;
    const bool ok = r < num_valid_rows;
    const __nv_bfloat16* src = q_row_base + (int64_t)r * q_row_stride + q4 * 128;
    // ONE pass over gmem: the 128 bf16 (16 x 16 B) stay in registers (Or is not live yet) --
    // the two-pass form was latency-bound (measured 9-12% of the kernel on 25-32-block walks).
    int4 v[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = ok ? *reinterpret_cast<const int4*>(src + i * 8) : make_int4(0, 0, 0, 0);
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v[i]);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(h[j]);
            amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    const float qs = amax / 448.f;
    const float inv = (amax > 0.f) ? 448.f / amax : 0.f;
    if (q4 == 0) sm.sQs[r] = qs;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {              // 8 granules of 16 e4m3 = 2 int4 of bf16 each
        uint32_t w[4];
        const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&v[2 * i]);
        const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&v[2 * i + 1]);
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float2 a = __bfloat1622float2(h0[2 * j]), b = __bfloat1622float2(h0[2 * j + 1]);
            w[j] = f_pack4(a.x * inv, a.y * inv, b.x * inv, b.y * inv);
            const float2 c = __bfloat1622float2(h1[2 * j]), d = __bfloat1622float2(h1[2 * j + 1]);
            w[2 + j] = f_pack4(c.x * inv, c.y * inv, d.x * inv, d.y * inv);
        }
        *reinterpret_cast<uint4*>(&sm.sQ8[f_q8_off(r, q4 * 8 + i)]) = make_uint4(w[0], w[1], w[2], w[3]);
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {              // rope: 512 granules / 256 threads
        const int g = tid + i * DM_THREADS;
        const int rr = g >> 3, c = (g & 7) * 8;
        int4 v = make_int4(0, 0, 0, 0);
        if (rr < num_valid_rows)
            v = *reinterpret_cast<const int4*>(q_row_base + (int64_t)rr * q_row_stride + DM_DV + c);
        *reinterpret_cast<int4*>(&sm.sQr[d_swz(rr, c, F_ROPE)]) = v;
    }
}

// ---- timing diag (env FLASH_MLA_SM120_SPARSE_DECODE_DIAG_TIMING=1): lane 0 of every warp
// accumulates clock64 deltas per phase; summed into f_diag_cycles at kernel end; the launcher
// prints the breakdown. Correct results, perturbed timing.
__device__ unsigned long long f_diag_cycles[16];
enum FDiagPhase { FD_T_WAITQ = 0, FD_T_QK, FD_T_ROPE, FD_T_SOFTMAX, FD_T_SYNCP, FD_T_PV, FD_T_SYNCV, FD_T_PROLOG, FD_T_NPHASE, FD_T_BLOCKS = 12, FD_T_WARPS = 13 };

// ---- kernel -----------------------------------------------------------------
// kSplit = false: grid (m_blocks, s_q, b), one batch per CTA, full topk walk, finals only.
// kSplit = true : grid (m_blocks, s_q, num_sm_parts), authors' metadata walk, finals/partials.
template <bool kSplit, bool kDiagTiming = false>
__global__ void __launch_bounds__(DM_THREADS, 1)
sparse_fp8_decode_fp8mma_kernel(const SparseFP8DecodeParams params) {
    [[maybe_unused]] unsigned long long tdiag[FD_T_NPHASE] = {};
    [[maybe_unused]] long long tprev = 0;
    [[maybe_unused]] unsigned long long nblk = 0;
    auto tstamp = [&](int phase) {
        if constexpr (kDiagTiming) { const long long t = clock64(); tdiag[phase] += static_cast<unsigned long long>(t - tprev); tprev = t; }
    };
    if constexpr (kDiagTiming) tprev = clock64();
    extern __shared__ char smem_raw[];
    SmemPlanFp8Mma& sm = *reinterpret_cast<SmemPlanFp8Mma*>(smem_raw);

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int m_block_idx = blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int head_start = m_block_idx * DM_BH;
    const int num_valid_rows = min(params.q_head_per_hk - head_start, DM_BH);
    const int row0 = s_q_idx * params.q_head_per_hk + head_start;
    const int num_topk_blocks = (params.topk + DM_BTOPK - 1) / DM_BTOPK;
    const uint8_t* kv_base = (const uint8_t*)params.kv_ptr;

    const int ms = warp & 3, nh = warp >> 2;
    const int wrow = ms * DM_MMA_M;
    const int gl = lane >> 2, tl = lane & 3;
    const int row_a = wrow + gl, row_b = row_a + 8;
    const int grp = ms >> 1;          // row-block group (rows 0..31 | 32..63), see f_bar_group

    // walk bounds
    int begin_idx, end_idx, begin_block = 0, end_block_last = 0, begin_n_split_idx = 0;
    if constexpr (kSplit) {
        const int* meta = params.tile_scheduler_metadata_ptr + blockIdx.z * TileSchedulerMetaDataSize;
        begin_idx = meta[0]; begin_block = meta[1]; end_idx = meta[2];
        end_block_last = meta[3]; begin_n_split_idx = meta[4];
    } else {
        begin_idx = end_idx = blockIdx.z;
    }

    for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx) {
        int blk_lo = 0, blk_hi = num_topk_blocks, n_split_idx = 0;
        bool no_split = true;
        int split_base = 0;
        if constexpr (kSplit) {
            blk_lo = (batch_idx == begin_idx) ? begin_block : 0;
            const int blk_hi_sched = (batch_idx == end_idx) ? end_block_last : num_topk_blocks;
            blk_hi = min(blk_hi_sched, num_topk_blocks);     // topk == 0 phantom block -> empty walk
            n_split_idx = (batch_idx == begin_idx) ? begin_n_split_idx : 0;
            split_base = __ldg(params.num_splits_ptr + batch_idx);
            no_split = (__ldg(params.num_splits_ptr + batch_idx + 1) - split_base) == 1;
        }
        const __nv_bfloat16* q_row_base = (const __nv_bfloat16*)params.q_ptr
            + (int64_t)batch_idx * params.q_batch_stride + (int64_t)row0 * params.q_row_stride;
        const int* idx_row = params.indices_ptr
            + (int64_t)batch_idx * params.indices_batch_stride
            + (int64_t)s_q_idx * params.indices_seq_stride;

        // ---- prologue: Q (quantized) + first block's pointers, then the uniform 5-group issue
        d_phase_a(sm.sTokPtr[blk_lo & 1], sm.sValid[blk_lo & 1], idx_row, kv_base, params, blk_lo, blk_hi);
        __syncthreads();      // pointer buffer visible; previous batch's smem consumed
        const bool nonempty = blk_lo < blk_hi;
        if (nonempty) f_issue_block(sm, sm.sTokPtr[blk_lo & 1], sm.sValid[blk_lo & 1]);   // flies under the Q load
        f_load_q(sm, q_row_base, params.q_row_stride, num_valid_rows);
        __syncthreads();      // sQ8/sQr/sQs visible

        float rmax[2] = {-INFINITY, -INFINITY}, rsum[2] = {0.f, 0.f};
        float Mg[F_NG][2];
        float Or[F_NG][8][4];
        #pragma unroll
        for (int g = 0; g < F_NG; ++g) {
            Mg[g][0] = Mg[g][1] = -INFINITY;
            #pragma unroll
            for (int j = 0; j < 8; ++j) { Or[g][j][0] = Or[g][j][1] = Or[g][j][2] = Or[g][j][3] = 0.f; }
        }
        const float qs0 = sm.sQs[row_a], qs1 = sm.sQs[row_b];
        tstamp(FD_T_PROLOG);

        for (int kb = blk_lo; kb < blk_hi; ++kb) {
            if constexpr (kDiagTiming) ++nblk;
            const int par = kb & 1;
            const uint8_t* const* tokp = sm.sTokPtr[par];
            const int8_t* valp = sm.sValid[par];
            const uint8_t* const* tokn = sm.sTokPtr[par ^ 1];
            const int8_t* valn = sm.sValid[par ^ 1];
            // next block's pointers -> parity buffer (published by SYNC_Q(0); WAR: its last
            // readers were block kb-1's masking (before its SYNC_P) and block kb's own gathers,
            // issued during block kb-1's PV -- all behind barriers this writer has passed).
            d_phase_a(sm.sTokPtr[par ^ 1], sm.sValid[par ^ 1], idx_row, kv_base, params, kb + 1, blk_hi);

            // ---- QK: S[64h x 64t] = qs * sum_g sk_g (Q8 K8^T)_g + rope, in registers -------
            float Sr[4][4];
            #pragma unroll
            for (int n = 0; n < 4; ++n) { Sr[n][0] = Sr[n][1] = Sr[n][2] = Sr[n][3] = 0.f; }

            #pragma unroll
            for (int g = 0; g < F_NG; ++g) {
                // Ledger: pending {Scl rope, K0K1, K2K3} at g=0 -> wait<1> retires {Scl rope} +
                // {K0K1} (tiles 0 AND 1, so g=1 needs no wait/barrier); g=2 -> wait<0> retires
                // K2K3 (g=3 needs none); after SYNC_Q(2) {Scl rope}(kb+1) is issued.
                if (g == 0 || g == 2) {
                    tstamp(FD_T_QK);
                    if (g == 0) d_cp_wait<1>(); else d_cp_wait<0>();
                    __syncthreads();  // SYNC_Q(0): tiles 0,1 + scales + pointers; SYNC_Q(2): tiles 2,3 + rope + sSclT/sLsk
                    tstamp(FD_T_WAITQ);
                }
                if (g == 0) {
                    // transpose raw scales -> sSclT (float2-able by token) + sLsk = ln(scale).
                    // Published by SYNC_Q(2) before its first readers (the g>=2 folds read sSclT;
                    // the g=0/1 folds read sScl raw). Zero scale -> -inf: an all-zero group
                    // contributes nothing to Mg / P' (exp(-inf) = 0).
                    const int gg = tid >> 6, t = tid & 63;
                    const float s = sm.sScl[t][gg];
                    sm.sSclT[gg][t] = s;
                    // ln(scale), clamped at -80 for positive denormal-class scales so that
                    // E = 256 exp(nmax - Mg) <= 256 e^80 stays finite (P' = p E sk < 256 then);
                    // exactly-zero scales keep -inf (the group contributes nothing).
                    sm.sLsk[gg][t] = (s > 0.f) ? fmaxf(__logf(s), -80.f) : -INFINITY;
                    // rope (bf16 m16n8k16, true units) FIRST: landed in the {Scl rope} group, so
                    // every warp's sKr reads precede SYNC_Q(2), after which the next block's rope
                    // gather re-targets sKr (WAR by the CTA barrier, independent of group drift).
                    #pragma unroll
                    for (int dl = 0; dl < 4; ++dl) {
                        uint32_t Qr[4];
                        {
                            const int row = wrow + (lane & 15);
                            const int col = dl * DM_MMA_K + (lane >> 4) * 8;
                            d_ldm_x4(Qr, d_cvta(&sm.sQr[d_swz(row, col, F_ROPE)]));
                        }
                        #pragma unroll
                        for (int n = 0; n < 4; ++n) {
                            uint32_t Kr[2];
                            const int krow = nh * 32 + n * 8 + (lane & 7);
                            const int kcol = dl * DM_MMA_K + ((lane >> 3) & 1) * 8;
                            d_ldm_x2(Kr, d_cvta(&sm.sKr[d_swz(krow, kcol, F_ROPE)]));
                            d_mma(Sr[n], Qr, Kr, Sr[n]);
                        }
                    }
                    tstamp(FD_T_ROPE);
                }
                if (g == 2) {         // raw sScl dead (transposed + folded at g=0,1) and sKr dead
                                      // (rope GEMM at g=0), both behind SYNC_Q(2)
                    f_gather_scales(sm, tokn, valn); f_gather_rope(sm, tokn, valn); d_cp_commit();  // {Scl rope (kb+1)}
                }
                // 4 independent accumulator chains (n-tiles); an 8-chain even/odd-k variant was
                // measured NEUTRAL (QK 36 -> 38%) at +16 registers -> not worth the spills.
                float Sg[4][4];
                #pragma unroll
                for (int n = 0; n < 4; ++n) { Sg[n][0] = Sg[n][1] = Sg[n][2] = Sg[n][3] = 0.f; }
                #pragma unroll
                for (int kp = 0; kp < 2; ++kp) {          // two k32 steps per x4 K load
                    uint32_t Qa[2][4];
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int r = wrow + (lane & 15);
                        const int gi = g * 8 + (kp * 2 + h) * 2 + (lane >> 4);
                        d_ldm_x4(Qa[h], d_cvta(&sm.sQ8[f_q8_off(r, gi)]));
                    }
                    #pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        uint32_t Kb[4];
                        const int t = nh * 32 + n * 8 + (lane & 7);
                        const int gi = kp * 4 + (lane >> 3);
                        d_ldm_x4(Kb, d_cvta(&sm.sKV[g][f_kv_off(t, gi)]));
                        const uint32_t b0[2] = {Kb[0], Kb[1]}, b1[2] = {Kb[2], Kb[3]};
                        f_mma_e4m3(Sg[n], Qa[0], b0);
                        f_mma_e4m3(Sg[n], Qa[1], b1);
                    }
                }
                // fold: S += Sg * sk[t][g] * qs[row]
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const int col0 = nh * 32 + n * 8 + tl * 2;
                    float sx, sy;
                    if (g < 2) { sx = sm.sScl[col0][g]; sy = sm.sScl[col0 + 1][g]; }
                    else { const float2 s2 = *reinterpret_cast<const float2*>(&sm.sSclT[g][col0]); sx = s2.x; sy = s2.y; }
                    Sr[n][0] = fmaf(Sg[n][0], sx * qs0, Sr[n][0]);
                    Sr[n][1] = fmaf(Sg[n][1], sy * qs0, Sr[n][1]);
                    Sr[n][2] = fmaf(Sg[n][2], sx * qs1, Sr[n][2]);
                    Sr[n][3] = fmaf(Sg[n][3], sy * qs1, Sr[n][3]);
                }
            }
            tstamp(FD_T_QK);
            // ---- mask + scale, row max, per-group (S + ln sk) max -------------------
            const float scale = params.sm_scale;
            float tmax[2] = {-INFINITY, -INFINITY};
            float tmg[F_NG][2];
            #pragma unroll
            for (int g = 0; g < F_NG; ++g) { tmg[g][0] = tmg[g][1] = -INFINITY; }
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int col0 = nh * 32 + n * 8 + tl * 2;
                const bool v0 = valp[col0] != 0, v1 = valp[col0 + 1] != 0;
                Sr[n][0] = v0 ? Sr[n][0] * scale : -INFINITY;
                Sr[n][1] = v1 ? Sr[n][1] * scale : -INFINITY;
                Sr[n][2] = v0 ? Sr[n][2] * scale : -INFINITY;
                Sr[n][3] = v1 ? Sr[n][3] * scale : -INFINITY;
                tmax[0] = fmaxf(tmax[0], fmaxf(Sr[n][0], Sr[n][1]));
                tmax[1] = fmaxf(tmax[1], fmaxf(Sr[n][2], Sr[n][3]));
                #pragma unroll
                for (int g = 0; g < F_NG; ++g) {
                    const float2 l2 = *reinterpret_cast<const float2*>(&sm.sLsk[g][col0]);
                    // (-inf) + (-inf) = -inf; a valid token with a zero group: S + (-inf) = -inf
                    tmg[g][0] = fmaxf(tmg[g][0], fmaxf(Sr[n][0] + l2.x, Sr[n][1] + l2.y));
                    tmg[g][1] = fmaxf(tmg[g][1], fmaxf(Sr[n][2] + l2.x, Sr[n][3] + l2.y));
                }
            }
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                tmax[h] = fmaxf(tmax[h], __shfl_xor_sync(0xffffffff, tmax[h], 1));
                tmax[h] = fmaxf(tmax[h], __shfl_xor_sync(0xffffffff, tmax[h], 2));
                #pragma unroll
                for (int g = 0; g < F_NG; ++g) {
                    tmg[g][h] = fmaxf(tmg[g][h], __shfl_xor_sync(0xffffffff, tmg[g][h], 1));
                    tmg[g][h] = fmaxf(tmg[g][h], __shfl_xor_sync(0xffffffff, tmg[g][h], 2));
                }
            }
            // cross-half combine via smem (also orders the rope ldmatrix reads before the P'
            // stores into sKr below)
            if (nh == 0 && tl == 0) {
                sm.sStatM[row_a] = tmax[0]; sm.sStatM[row_b] = tmax[1];
                #pragma unroll
                for (int g = 0; g < F_NG; ++g) { sm.sStatMg[g][row_a] = tmg[g][0]; sm.sStatMg[g][row_b] = tmg[g][1]; }
            }
            f_bar_group(grp);     // SYNC_M1 (group-private rows)
            if (nh == 1 && tl == 0) {
                sm.sStatM[row_a] = fmaxf(sm.sStatM[row_a], tmax[0]);
                sm.sStatM[row_b] = fmaxf(sm.sStatM[row_b], tmax[1]);
                #pragma unroll
                for (int g = 0; g < F_NG; ++g) {
                    sm.sStatMg[g][row_a] = fmaxf(sm.sStatMg[g][row_a], tmg[g][0]);
                    sm.sStatMg[g][row_b] = fmaxf(sm.sStatMg[g][row_b], tmg[g][1]);
                }
            }
            f_bar_group(grp);     // SYNC_M2
            const float bmax0 = sm.sStatM[row_a], bmax1 = sm.sStatM[row_b];

            // ---- online softmax (fp32 P for the normalizer) + per-group references ------
            const float nmax0 = fmaxf(rmax[0], bmax0), nmax1 = fmaxf(rmax[1], bmax1);
            const float rs0 = (rmax[0] == -INFINITY) ? ((nmax0 == -INFINITY) ? 1.f : 0.f) : __expf(rmax[0] - nmax0);
            const float rs1 = (rmax[1] == -INFINITY) ? ((nmax1 == -INFINITY) ? 1.f : 0.f) : __expf(rmax[1] - nmax1);
            rmax[0] = nmax0; rmax[1] = nmax1;
            float E[F_NG][2];     // 256 * exp(nmax - nMg): P'_g = p * sk * E
            #pragma unroll
            for (int g = 0; g < F_NG; ++g) {
                const float bm0 = sm.sStatMg[g][row_a], bm1 = sm.sStatMg[g][row_b];
                const float nm0 = fmaxf(Mg[g][0], bm0), nm1 = fmaxf(Mg[g][1], bm1);
                const float rg0 = (Mg[g][0] == -INFINITY) ? ((nm0 == -INFINITY) ? 1.f : 0.f) : __expf(Mg[g][0] - nm0);
                const float rg1 = (Mg[g][1] == -INFINITY) ? ((nm1 == -INFINITY) ? 1.f : 0.f) : __expf(Mg[g][1] - nm1);
                Mg[g][0] = nm0; Mg[g][1] = nm1;
                // conditional rescale (FA-3 pattern, warp-uniform): the per-group reference rises
                // rarely after the first blocks; rg == 1.0f exactly when it did not (expf(0)).
                if (__any_sync(0xffffffff, (rg0 != 1.f) || (rg1 != 1.f))) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) { Or[g][j][0] *= rg0; Or[g][j][1] *= rg0; Or[g][j][2] *= rg1; Or[g][j][3] *= rg1; }
                }
                E[g][0] = (nm0 == -INFINITY) ? 0.f : F_PSCALE * __expf(nmax0 - nm0);
                E[g][1] = (nm1 == -INFINITY) ? 0.f : F_PSCALE * __expf(nmax1 - nm1);
            }
            // P (fp32, in place over S) for the normalizer
            float (&p)[4][4] = Sr;
            float tsum[2] = {0.f, 0.f};
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                p[n][0] = (Sr[n][0] == -INFINITY) ? 0.f : __expf(Sr[n][0] - nmax0);
                p[n][1] = (Sr[n][1] == -INFINITY) ? 0.f : __expf(Sr[n][1] - nmax0);
                p[n][2] = (Sr[n][2] == -INFINITY) ? 0.f : __expf(Sr[n][2] - nmax1);
                p[n][3] = (Sr[n][3] == -INFINITY) ? 0.f : __expf(Sr[n][3] - nmax1);
                tsum[0] += p[n][0] + p[n][1]; tsum[1] += p[n][2] + p[n][3];
            }
            rsum[0] = rsum[0] * rs0 + tsum[0];
            rsum[1] = rsum[1] * rs1 + tsum[1];
            // P'_g -> e4m3 words in the k-permuted layout the PV A-fragment expects: within the
            // 32-token half, n-tiles (0,1) pack into the word at byte 4*tl, (2,3) at 16 + 4*tl.
            // Groups 0,1 go to sP8[0..1] now; groups 2,3 are packed into registers and stored
            // into the SAME two tiles after PV(1) (their last readers are behind SYNC_V(1)).
            uint32_t pw[F_NG][4];     // [g] = {wa0, wa1, wb0, wb1}
            #pragma unroll
            for (int g = 0; g < F_NG; ++g) {
                pw[g][0] = pw[g][1] = pw[g][2] = pw[g][3] = 0u;
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const int col0 = nh * 32 + n * 8 + tl * 2;
                    const float2 s2 = *reinterpret_cast<const float2*>(&sm.sSclT[g][col0]);
                    const uint32_t ha = f_pack2(p[n][0] * E[g][0] * s2.x, p[n][1] * E[g][0] * s2.y);
                    const uint32_t hb = f_pack2(p[n][2] * E[g][1] * s2.x, p[n][3] * E[g][1] * s2.y);
                    pw[g][n >> 1]     |= ha << ((n & 1) * 16);
                    pw[g][2 + (n >> 1)] |= hb << ((n & 1) * 16);
                }
            }
            auto store_p8 = [&](int g) {
                uint8_t* P8 = sm.sP8[g & 1];
                *reinterpret_cast<uint32_t*>(P8 + row_a * DM_BTOPK + nh * 32 + tl * 4)      = pw[g][0];
                *reinterpret_cast<uint32_t*>(P8 + row_a * DM_BTOPK + nh * 32 + 16 + tl * 4) = pw[g][1];
                *reinterpret_cast<uint32_t*>(P8 + row_b * DM_BTOPK + nh * 32 + tl * 4)      = pw[g][2];
                *reinterpret_cast<uint32_t*>(P8 + row_b * DM_BTOPK + nh * 32 + 16 + tl * 4) = pw[g][3];
            };
            store_p8(0); store_p8(1);
            tstamp(FD_T_SOFTMAX);
            f_bar_group(grp);     // SYNC_P0: this group's P'_0/1 rows visible (PV A-fragments read own rows only)
            tstamp(FD_T_SYNCP);

            // ---- PV: Og += P'_g . V_g over the RESIDENT tiles (e4m3 x e4m3), then hand tiles
            // to the next block's gather -------------------------------------------------
            #pragma unroll
            for (int g = 0; g < F_NG; ++g) {
                if (g == 2) {         // P'_2/3 into the tiles PV(0)/PV(1) just consumed (behind SYNC_V(1))
                    store_p8(2); store_p8(3);
                    f_bar_group(grp); // SYNC_P1
                    tstamp(FD_T_SYNCP);
                }
                const uint8_t* P8 = sm.sP8[g & 1];
                #pragma unroll
                for (int pk = 0; pk < 2; ++pk) {
                    uint32_t Pa[4];
                    {
                        const int r = wrow + (lane & 15);
                        d_ldm_x4(Pa, d_cvta(P8 + r * DM_BTOPK + pk * 32 + (lane >> 4) * 16));
                    }
                    #pragma unroll
                    for (int sl = 0; sl < 4; ++sl) {
                        uint32_t R[4];
                        const int t = pk * 32 + lane;
                        f_ldm_x4_trans(R, d_cvta(&sm.sKV[g][f_kv_off(t, nh * 4 + sl)]));
                        const uint32_t be[2] = {f_prmt(R[0], R[1], 0x6420u), f_prmt(R[2], R[3], 0x6420u)};
                        const uint32_t bo[2] = {f_prmt(R[0], R[1], 0x7531u), f_prmt(R[2], R[3], 0x7531u)};
                        f_mma_e4m3(Or[g][sl * 2], Pa, be);
                        f_mma_e4m3(Or[g][sl * 2 + 1], Pa, bo);
                    }
                }
                if (g == 1 || g == 3) {
                    tstamp(FD_T_PV);
                    __syncthreads();  // SYNC_V(1): every warp done with tiles 0,1 (+ P'_0/1); SYNC_V(3): tiles 2,3
                    if (g == 1) { f_gather_tile(sm, 0, tokn, valn); f_gather_tile(sm, 1, tokn, valn); d_cp_commit(); }   // {K0K1(kb+1)}
                    else { f_gather_tile(sm, 2, tokn, valn); f_gather_tile(sm, 3, tokn, valn); d_cp_commit(); }          // {K2K3(kb+1)}
                    tstamp(FD_T_SYNCV);
                }
            }
        }
        if (nonempty) d_cp_wait<0>();   // retire the beyond-walk (all-invalid zero-fill) prefetch
        if constexpr (kDiagTiming) {
            __syncwarp();
            if (lane == 0) {
                #pragma unroll
                for (int i = 0; i < FD_T_NPHASE; ++i) atomicAdd(&f_diag_cycles[i], tdiag[i]);
                atomicAdd(&f_diag_cycles[FD_T_BLOCKS], nblk);
                atomicAdd(&f_diag_cycles[FD_T_WARPS], 1ull);
            }
            #pragma unroll
            for (int i = 0; i < FD_T_NPHASE; ++i) tdiag[i] = 0;
            nblk = 0;
            tprev = clock64();
        }

        // ---- epilogue --------------------------------------------------------------
        rsum[0] += __shfl_xor_sync(0xffffffff, rsum[0], 1);
        rsum[0] += __shfl_xor_sync(0xffffffff, rsum[0], 2);
        rsum[1] += __shfl_xor_sync(0xffffffff, rsum[1], 1);
        rsum[1] += __shfl_xor_sync(0xffffffff, rsum[1], 2);
        __syncthreads();      // (empty walk: prologue readers of sStatS from a previous batch)
        if (nh == 0 && tl == 0) { sm.sStatS[row_a] = rsum[0]; sm.sStatS[row_b] = rsum[1]; }
        __syncthreads();
        if (nh == 1 && tl == 0) { atomicAdd(&sm.sStatS[row_a], rsum[0]); atomicAdd(&sm.sStatS[row_b], rsum[1]); }
        __syncthreads();
        const float sum0 = sm.sStatS[row_a], sum1 = sm.sStatS[row_b];
        const float inv0 = (sum0 > 0.f) ? 1.f / sum0 : 0.f;
        const float inv1 = (sum1 > 0.f) ? 1.f / sum1 : 0.f;
        float F[F_NG][2];     // O_g = Og * exp(Mg - m) / (256 L)
        #pragma unroll
        for (int g = 0; g < F_NG; ++g) {
            F[g][0] = (sum0 > 0.f && Mg[g][0] != -INFINITY) ? __expf(Mg[g][0] - rmax[0]) * inv0 * F_PINV : 0.f;
            F[g][1] = (sum1 > 0.f && Mg[g][1] != -INFINITY) ? __expf(Mg[g][1] - rmax[1]) * inv1 * F_PINV : 0.f;
        }
        // Or[g][2s + par][c]: d = g*128 + nh*64 + s*16 + 2*(tl*2 + (c&1)) + par -> (even, odd)
        // pairs are adjacent d -> one bf162 / float2 per (s, c&1).
        if (no_split) {
            __nv_bfloat16* o_row_base = (__nv_bfloat16*)params.o_ptr
                + (int64_t)batch_idx * params.o_batch_stride + (int64_t)row0 * params.o_row_stride;
            #pragma unroll
            for (int g = 0; g < F_NG; ++g)
                #pragma unroll
                for (int s = 0; s < 4; ++s)
                    #pragma unroll
                    for (int cc = 0; cc < 2; ++cc) {
                        const int d = g * 128 + nh * 64 + s * 16 + tl * 4 + cc * 2;
                        if (row_a < num_valid_rows)
                            *reinterpret_cast<__nv_bfloat162*>(o_row_base + (int64_t)row_a * params.o_row_stride + d) =
                                __floats2bfloat162_rn(Or[g][2 * s][cc] * F[g][0], Or[g][2 * s + 1][cc] * F[g][0]);
                        if (row_b < num_valid_rows)
                            *reinterpret_cast<__nv_bfloat162*>(o_row_base + (int64_t)row_b * params.o_row_stride + d) =
                                __floats2bfloat162_rn(Or[g][2 * s][2 + cc] * F[g][1], Or[g][2 * s + 1][2 + cc] * F[g][1]);
                    }
            if (nh == 0 && tl == 0) {
                float* lse_base = params.softmax_lse_ptr + (int64_t)batch_idx * params.q_seq_per_hk + (int64_t)row0;
                if (row_a < num_valid_rows) lse_base[row_a] = (sum0 > 0.f) ? (rmax[0] + __logf(sum0)) : INFINITY;
                if (row_b < num_valid_rows) lse_base[row_b] = (sum1 > 0.f) ? (rmax[1] + __logf(sum1)) : INFINITY;
            }
        } else {
            const int split_row = split_base + n_split_idx;
            float* oaccum = params.oaccum_ptr + ((int64_t)split_row * params.q_seq_per_hk + row0) * DM_DV;
            #pragma unroll
            for (int g = 0; g < F_NG; ++g)
                #pragma unroll
                for (int s = 0; s < 4; ++s)
                    #pragma unroll
                    for (int cc = 0; cc < 2; ++cc) {
                        const int d = g * 128 + nh * 64 + s * 16 + tl * 4 + cc * 2;
                        if (row_a < num_valid_rows)
                            *reinterpret_cast<float2*>(oaccum + (int64_t)row_a * DM_DV + d) =
                                make_float2(Or[g][2 * s][cc] * F[g][0], Or[g][2 * s + 1][cc] * F[g][0]);
                        if (row_b < num_valid_rows)
                            *reinterpret_cast<float2*>(oaccum + (int64_t)row_b * DM_DV + d) =
                                make_float2(Or[g][2 * s][2 + cc] * F[g][1], Or[g][2 * s + 1][2 + cc] * F[g][1]);
                    }
            if (nh == 0 && tl == 0) {
                float* lse_accum = params.softmax_lseaccum_ptr + (int64_t)split_row * params.q_seq_per_hk + row0;
                if (row_a < num_valid_rows) lse_accum[row_a] = (sum0 > 0.f) ? (rmax[0] + __logf(sum0)) * DM_LOG2E : -INFINITY;
                if (row_b < num_valid_rows) lse_accum[row_b] = (sum1 > 0.f) ? (rmax[1] + __logf(sum1)) * DM_LOG2E : -INFINITY;
            }
        }
        __syncthreads();      // stats / smem free before the next batch's prologue
    }
}

inline bool fp8mma_diag_timing() {
    static const bool v = [] { const char* e = std::getenv("FLASH_MLA_SM120_SPARSE_DECODE_DIAG_TIMING"); return e && e[0] == '1'; }();
    return v;
}

template <bool kSplit, bool kDiag>
inline void launch_fp8mma_inst(const SparseFP8DecodeParams& params, const dim3& grid) {
    constexpr size_t smem = sizeof(SmemPlanFp8Mma);
    CHECK_CUDA(cudaFuncSetAttribute(sparse_fp8_decode_fp8mma_kernel<kSplit, kDiag>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    sparse_fp8_decode_fp8mma_kernel<kSplit, kDiag><<<grid, dim3(DM_THREADS), smem, params.stream>>>(params);
    if constexpr (kDiag) {
        CHECK_CUDA(cudaStreamSynchronize(params.stream));
        unsigned long long c[16] = {};
        CHECK_CUDA(cudaMemcpyFromSymbol(c, f_diag_cycles, sizeof(c)));
        static const char* names[FD_T_NPHASE] = {"wait+syncQ", "QK", "rope", "softmax", "syncP", "PV", "syncV+issue", "prolog"};
        unsigned long long tot = 0;
        for (int i = 0; i < FD_T_NPHASE; ++i) tot += c[i];
        const double per_blk = c[FD_T_BLOCKS] ? static_cast<double>(tot - c[FD_T_PROLOG]) / static_cast<double>(c[FD_T_BLOCKS]) : 0.0;
        std::fprintf(stderr, "[FP8MMA-TIMING split=%d] warps=%llu blocks/warp=%.1f cycles/block/warp=%.0f |", static_cast<int>(kSplit), c[FD_T_WARPS],
                     c[FD_T_WARPS] ? static_cast<double>(c[FD_T_BLOCKS]) / static_cast<double>(c[FD_T_WARPS]) : 0.0, per_blk);
        for (int i = 0; i < FD_T_NPHASE; ++i) std::fprintf(stderr, " %s %.1f%%", names[i], tot ? 100.0 * static_cast<double>(c[i]) / static_cast<double>(tot) : 0.0);
        std::fprintf(stderr, "%s", "\n");
        unsigned long long z[16] = {};
        CHECK_CUDA(cudaMemcpyToSymbol(f_diag_cycles, z, sizeof(z)));
    }
}

template <bool kSplit>
inline void launch_fp8mma(const SparseFP8DecodeParams& params) {
    const int num_m_blocks = (params.q_head_per_hk + DM_BH - 1) / DM_BH;
    const dim3 grid(num_m_blocks, params.s_q, kSplit ? params.num_sm_parts : params.b);
    if (fp8mma_diag_timing()) launch_fp8mma_inst<kSplit, true>(params, grid);
    else launch_fp8mma_inst<kSplit, false>(params, grid);
}

}  // namespace fp8mma
}  // namespace sparse_decode
}  // namespace sm120
