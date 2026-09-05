// ============================================================================
// SPARSE PREFILL FORWARD, CFG=5: pipelined mma.sync kernel on the CFG=4 math (SM120)
// ============================================================================
// Design: audit/design-sparse-fwd-structural.md (rung 5). Implementation note, including the
// forced deviation from the design's 288-thread producer-warp shape (a 9-warp CTA is charged as
// 12 warps by the cc 12.0 register-launch check -> 168-register cap, measured; setmaxnreg is
// rejected on plain sm_120): audit/impl-sparse-fwd-cfg5.md.
//
// Kept VERBATIM from fwd_mma.cuh (CFG=4): the resident swizzled sQ, the per-warp fragment math
// (rows [16ms,+16); S columns [32nh,+32) -> Sr[4][4]; PV output columns [32nh,+32) of every V
// d-tile -> Or[8][4][4]), the mask / scale / online-softmax / epilogue arithmetic with its
// reduction orders, and the zero-fill semantics of invalid or ragged-tail tokens (src-size-0
// cp.async from a clamped legal address + -inf mask). The output is bitwise identical to CFG=4:
// the V-tile order 7,6,..,0 only changes which independent accumulator is touched first, and the
// pair max / rsum combines evaluate the same fmaxf / add with the same operand order.
//
// What changes (the structure):
//   - 256 threads = 8 MMA warps, __launch_bounds__(256, 1); warp map re-indexed to ms = warp>>1,
//     nh = warp&1 so the pair sharing a 16-row strip sits on two different SM sub-partitions
//     (w%4 assignment) and de-phases: one warp's softmax overlaps the other's mma;
//   - the 2-stage 64x64 KV ring is driven by a STATIC 17-slot schedule per 64-token block
//     (K0..K8, V7* served from K7's stage without a gather, V6..V0) through a CUTLASS-style
//     FULL/EMPTY mbarrier pipeline: every thread issues its 2 granules of slot g+1 at the START
//     of slot g (one slot of lead), gated on the EMPTY phase of slot g-1 (the stage's previous
//     occupant, consumed by all 8 warps), completion tracked by cp.async.mbarrier.arrive.noinc
//     (full count 256). No cp.async.commit_group / wait_group, no __syncthreads after the
//     prologue, no smem pointer table: token pointers live in registers, the next block's
//     indices are loaded 8 slots ahead and its rows are L2-prefetched (cp.async.bulk.prefetch);
//   - the cross-warp max / P / rsum exchange is pair-local on named barriers 3..6 (count 64);
//   - x4 ldmatrix K / V pairs replace the x2 loads (same bytes, same fragment bits, half the
//     LDSM instructions).
//
// Env: FLASH_MLA_SM120_SPARSE_FWD_CFG=5 (6/7 route here with a one-time [WARN]);
//      FLASH_MLA_SM120_SPARSE_FWD_DIAG = 0 production | 1 RED: EMPTY (WAR) waits removed + an
//      adversarial 8192-clk lag injected in warp 7 before every slot's mma (numerics wrong by
//      design -> the suites MUST fail) | 2 softmax skipped, P = packed raw S (GEMM-pipeline
//      ceiling, bench only) | 3 the same lag with the protocol intact (must stay green).
// ============================================================================
#pragma once

#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

// Requires (included by fwd.cu before this header): fwd_mma.cuh (sparse_mma:: primitives and
// tile constants, never modified) and utils.h (CHECK_CUDA).

namespace sm120 {
namespace sparse_ws {

using sparse_mma::MMA_M;
using sparse_mma::MMA_N;
using sparse_mma::MMA_K;
using sparse_mma::SM_BH;
using sparse_mma::SM_BTOPK;
using sparse_mma::SM_DQK;
using sparse_mma::SM_DV;
using sparse_mma::SM_DTILE;
using sparse_mma::SM_NKTILES;
using sparse_mma::SM_NVTILES;
using sparse_mma::SM_LOG2E;
using sparse_mma::s_cvta;
using sparse_mma::s_ldm_x4;
using sparse_mma::s_mma;
using sparse_mma::s_swz;

constexpr int WS_NW = 8;                                   // MMA warps (every warp computes)
constexpr int WS_THREADS = WS_NW * 32;                     // 256
constexpr int WS_SLOTS = 17;                               // per 64-token block: K0..K8, V7*, V6..V0
constexpr uint32_t WS_STAGE_BYTES = SM_BTOPK * SM_DTILE * 2;   // 8,192
constexpr uint32_t WS_FULL_COUNT = WS_THREADS;             // one arrive per thread per slot
constexpr uint32_t WS_EMPTY_COUNT = WS_NW;                 // one elected lane-0 arrive per warp per slot
constexpr long long WS_SKEW_CLK = 8192;                    // kDiag 1/3: lag injected in warp 7 per slot

// ---- PTX primitives (spellings copied verbatim from the certified dense kernel; every one is
// accepted by ptxas on PLAIN sm_120: tests/_probe_warpspec_ptx.cu, tests/_probe_sm120_extras.sh).
// Semantics: PTX ISA "Parallel Synchronization and Communication Instructions". mbarrier.arrive
// defaults to .release.cta, mbarrier.try_wait to .acquire.cta. ----------------------------------
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(a), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" :: "r"(a) : "memory");
}
// Block until the phase with the given parity has completed (try_wait suspends the thread with a
// hardware time limit between polls; no smem spinning).
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t parity) {
    uint32_t done;
    do {
        asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                     : "=r"(done) : "r"(a), "r"(parity) : "memory");
    } while (!done);
}
// 16-B cp.async with an explicit source size: src_bytes == 0 reads nothing and ZERO-FILLS the 16 B
// (the LDGSTS zfill predicate); the source address must still be valid (callers clamp it).
__device__ __forceinline__ void cp_cg16_zfill(uint32_t sa, const void* g, uint32_t src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sa), "l"(g), "r"(src_bytes) : "memory");
}
// Asynchronous arrive-on: fires once ALL prior cp.async of this thread have landed. .noinc: the
// arrival counts against the mbarrier's init count (one per issuing thread), no pending-count bump.
__device__ __forceinline__ void cp_async_mbar_arrive_noinc(uint32_t a) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(a) : "memory");
}
__device__ __forceinline__ void ldm_x4_trans(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
// Named barrier: arrive + wait of CNT threads on hardware barrier ID (bar.sync = release/acquire).
template <int ID, int CNT>
__device__ __forceinline__ void nbar_sync() {
    asm volatile("bar.sync %0, %1;\n" :: "n"(ID), "n"(CNT) : "memory");
}
// Bulk L2 prefetch hint (sm_90+ ISA, ptxas-accepted on plain sm_120; inert-at-worst).
__device__ __forceinline__ void l2_prefetch_bulk(const void* g, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(g), "r"(bytes) : "memory");
}
// Pair-local rendezvous of the two warps (2ms, 2ms+1) sharing row strip ms: ids 3..6, count 64.
// Warp-uniform switch so the barrier id is an immediate (ptxas then reports the exact id set).
__device__ __forceinline__ void pair_sync(int ms) {
    switch (ms) {
        case 0:  nbar_sync<3, 64>(); break;
        case 1:  nbar_sync<4, 64>(); break;
        case 2:  nbar_sync<5, 64>(); break;
        default: nbar_sync<6, 64>(); break;
    }
}

// ---- smem plan (99,584 B; every array 128-B aligned) ------------------------
struct SmemPlanWs {
    __align__(128) __nv_bfloat16 sQ[SM_BH * SM_DQK];            // 73,728 B @0       swizzled(576), resident
    __align__(128) __nv_bfloat16 sKV[2][SM_BTOPK * SM_DTILE];   // 16,384 B @73,728  2-stage ring, swizzled(64)
    __align__(128) __nv_bfloat16 sP[SM_BH * SM_BTOPK];          //  8,192 B @90,112  P staging, swizzled(64), pair-local rows
    __align__(128) float sStatM[2][SM_BH];                       //    512 B @98,304  pair max exchange [nh][row]
    __align__(128) float sStatS[2][SM_BH];                       //    512 B @98,816  epilogue rsum exchange [nh][row]
    __align__(128) uint8_t sValid[2][SM_BTOPK];                  //    128 B @99,328  per-block validity, parity kb&1
    __align__(128) uint64_t mbar[4];                             //     32 B @99,456  full[0..1] @+0/+8, empty[0..1] @+16/+24
};
static_assert(sizeof(SmemPlanWs) == 99584, "sparse ws smem plan changed size");
static_assert(sizeof(SmemPlanWs) <= 101376, "sparse ws smem exceeds the sm_120 per-block opt-in limit");

// ---- kernel -----------------------------------------------------------------
// Slot protocol (global slot g = 17*kb + i; stage s = g & 1; phase index k = g >> 1 on BOTH ring
// mbarriers of that stage; parity k & 1), every thread of every warp:
//   (1) if slot g+1 exists: [kDiag != 1] mbar_wait(empty[s^1], parity((g-1)>>1))  -- WAR: slot g-1,
//       the previous occupant of stage s^1, consumed by all 8 warps; then issue slot g+1 into
//       stage s^1 (2 x cp.async 16 B + cp.async.mbarrier.arrive.noinc on full[s^1]); the virtual
//       slot V7* (i+1 == 9) needs no copy: one plain mbarrier.arrive per thread completes its phase.
//   (2) mbar_wait(full[s], parity(g>>1)) -- slot g landed (acquire orders the ldmatrix reads).
//   (3) compute(g): QK(dt) for i <= 8, PV(vt) for i >= 9 (CFG=4 math).
//   (4) __syncwarp; lane 0: mbarrier.arrive(empty[s]) -- this warp's reads of slot g are done.
// Proofs (audit/impl-sparse-fwd-cfg5.md sec. 2): RAW by (2); WAR by (1); full-phase parity is
// exact because a thread's arrive for phase k on full[s] is program-ordered after its own wait on
// phase k-1 of the same barrier, so no phase can complete ahead of a thread still waiting on the
// previous one; empty-phase parity is exact because empty phase k+1 needs every warp's compute
// of slot g+2, which needs full[s] phase k+1, which needs every thread past its empty wait for
// phase k. Deadlock-free: every wait depends only on slots issued earlier.
template <int kRung, int kDiag>
__global__ void __launch_bounds__(WS_THREADS, 1)
sparse_prefill_fwd_ws_kernel(const SparsePrefillParams params) {
    static_assert(kRung == 5, "fwd_ws.cuh implements rung 5 (the dispatcher routes 6/7 here)");
    static_assert(kDiag >= 0 && kDiag <= 3, "kDiag: 0 production, 1 RED, 2 no-softmax, 3 skew-only");
    extern __shared__ char smem_raw[];
    SmemPlanWs& sm = *reinterpret_cast<SmemPlanWs*>(smem_raw);

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

    // ring mbarriers: full[s] = data of the slot in stage s landed; empty[s] = stage s consumed
    const uint32_t mb = s_cvta(&sm.mbar[0]);
    auto full_bar  = [&](int s) -> uint32_t { return mb + 8u * static_cast<uint32_t>(s); };
    auto empty_bar = [&](int s) -> uint32_t { return mb + 16u + 8u * static_cast<uint32_t>(s); };

    // Gather ownership: thread -> token row gr = tid>>2 and the two 16-B granules gc, gc+8 of every
    // d-tile (4 threads per 128-B row line; 512 granules / 256 threads = 2 each, as CFG=4).
    const int gr = tid >> 2, gc = (tid & 3) * 16;
    const uint32_t dst_a = s_cvta(&sm.sKV[0][s_swz(gr, gc, SM_DTILE)]);       // stage 0; stage 1 = +8192 B
    const uint32_t dst_b = s_cvta(&sm.sKV[0][s_swz(gr, gc + 8, SM_DTILE)]);

    const int num_topk_blocks = (params.topk + SM_BTOPK - 1) / SM_BTOPK;

    // Per-block token metadata for row gr, in registers: the row base (clamped to the pool base
    // when invalid -- a legal address that src-size 0 never reads) and the cp.async source size
    // (16 valid / 0 invalid = hardware zero-fill). Exactly CFG=4's sTokPtr / sValid semantics
    // (ragged tail k >= topk -> -1 -> invalid; never dereferences indices past topk). One of the
    // row's 4 threads publishes the validity bit for the softmax mask and L2-prefetches the row.
    auto load_block_meta = [&](int kb, const __nv_bfloat16*& ptr, uint32_t& nbytes) {
        const int k = kb * SM_BTOPK + gr;
        const int token = (k < params.topk) ? idx_row[k] : -1;
        const bool ok = (token >= 0 && token < params.s_kv);
        ptr = ok ? (kv_base + (int64_t)token * params.stride_kv_s_kv) : kv_base;
        nbytes = ok ? 16u : 0u;
        if ((tid & 3) == 0) {
            sm.sValid[kb & 1][gr] = ok ? 1 : 0;
            if (ok) l2_prefetch_bulk(ptr, SM_DQK * 2);   // the whole 1152-B row, once per block
        }
    };
    // This thread's 2 granules of one 64x64 d-tile (cols [col, col+64) of the gathered rows) into
    // stage `st`; the stage's FULL mbarrier is arrived-on once both copies landed.
    auto issue_gather = [&](int st, int col, const __nv_bfloat16* ptr, uint32_t nbytes) {
        const uint32_t so = static_cast<uint32_t>(st) * WS_STAGE_BYTES;
        const __nv_bfloat16* src = ptr + col + gc;
        cp_cg16_zfill(dst_a + so, src, nbytes);
        cp_cg16_zfill(dst_b + so, src + 8, nbytes);
        cp_async_mbar_arrive_noinc(full_bar(st));
    };
    // kDiag 1/3: adversarial inter-warp skew -- warp 7 lags every slot by WS_SKEW_CLK before its
    // mma, far more than any gather latency, so a missing WAR wait (kDiag 1) deterministically
    // lets the other warps' refills land on the stage warp 7 is about to read.
    auto skew_lag = [&]() {
        if constexpr (kDiag == 1 || kDiag == 3) {
            if (warp == WS_NW - 1) {
                const long long t0 = clock64();
                while (clock64() - t0 < WS_SKEW_CLK) { }
            }
        }
    };

    // ---- prologue -----------------------------------------------------------
    const __nv_bfloat16* cur_ptr = kv_base;
    uint32_t cur_nb = 0;
    if (num_topk_blocks > 0) load_block_meta(0, cur_ptr, cur_nb);   // block 0 indices + L2 prefetch overlap the Q g2s
    {
        // resident swizzled Q [64 x 576], 16B vectors: 4608 granules / 256 threads = 18 each (CFG=4)
        #pragma unroll
        for (int i = 0; i < 18; ++i) {
            const int g = tid + i * WS_THREADS;
            const int r = g / (SM_DQK / 8);
            const int c = (g % (SM_DQK / 8)) * 8;
            *reinterpret_cast<int4*>(&sm.sQ[s_swz(r, c, SM_DQK)]) =
                *reinterpret_cast<const int4*>(q_ptr + (int64_t)r * params.stride_q_h_q + c);
        }
    }
    if (tid == 0) {
        mbar_init(full_bar(0), WS_FULL_COUNT);
        mbar_init(full_bar(1), WS_FULL_COUNT);
        mbar_init(empty_bar(0), WS_EMPTY_COUNT);
        mbar_init(empty_bar(1), WS_EMPTY_COUNT);
        mbar_fence_init();
    }
    __syncthreads();   // the LAST CTA-wide barrier: sQ, sValid[0] and the mbarrier inits are visible
    if (num_topk_blocks > 0) issue_gather(0, 0, cur_ptr, cur_nb);   // slot 0 = K0 of block 0 -> stage 0

    // S-phase warp map: ms = warp >> 1 (row strip), nh = warp & 1 (n-half of 8 n8 tiles)
    const int ms = warp >> 1, nh = warp & 1;
    const int wrow = ms * MMA_M;

    // Online-softmax state per lane: rows wrow + lane/4 and +8 (mma C layout)
    float rmax[2] = {-INFINITY, -INFINITY}, rsum[2] = {0.f, 0.f};
    // O accumulator: 8 v-tiles x 4 n8 (32-col half per tile) x 4 = 128 fp32
    float Or[SM_NVTILES][4][4];
    #pragma unroll
    for (int t = 0; t < SM_NVTILES; ++t)
        #pragma unroll
        for (int n = 0; n < 4; ++n) { Or[t][n][0] = Or[t][n][1] = Or[t][n][2] = Or[t][n][3] = 0.f; }

    int g = 0;   // global slot counter: stage = g & 1, phase index = g >> 1 on both ring mbarriers

    for (int kb = 0; kb < num_topk_blocks; ++kb) {
        // ---- QK: S[64h x 64t] accumulated in registers over the 9 K d-tiles (slots 0..8) ----
        float Sr[4][4];   // this warp's n-half: 4 n8 tiles
        #pragma unroll
        for (int n = 0; n < 4; ++n) { Sr[n][0] = Sr[n][1] = Sr[n][2] = Sr[n][3] = 0.f; }

        #pragma unroll 1
        for (int dt = 0; dt < SM_NKTILES; ++dt, ++g) {
            const int s = g & 1;
            if (dt + 1 < SM_NKTILES) {
                // slot g+1 = K d-tile dt+1 -> stage s^1: its previous occupant (slot g-1) must be
                // consumed by all 8 warps (EMPTY phase (g-1)>>1). kDiag 1 removes this WAR wait.
                if constexpr (kDiag != 1) {
                    if (g >= 1) mbar_wait(empty_bar(s ^ 1), static_cast<uint32_t>((g - 1) >> 1) & 1u);
                }
                issue_gather(s ^ 1, (dt + 1) * SM_DTILE, cur_ptr, cur_nb);
            } else {
                // slot g+1 = V7*: V tile 7 IS K tile 7's data, still resident in stage s^1 (slot
                // g-1, cols 448..511 of the gathered rows). No copy: one plain (release) arrive
                // per thread completes the phase; consumers wait it like any other slot.
                mbar_arrive(full_bar(s ^ 1));
            }
            mbar_wait(full_bar(s), static_cast<uint32_t>(g >> 1) & 1u);   // K(dt) landed (acquire)
            skew_lag();
            // QK(dt): CFG=4 fragments; each x4 K pair load == the two x2 loads of n-tiles 2np, 2np+1
            #pragma unroll
            for (int dl = 0; dl < 4; ++dl) {                       // 4 k16 per 64-col tile
                uint32_t Qr[4];
                {
                    const int row = wrow + (lane % 16);
                    const int col = dt * SM_DTILE + dl * MMA_K + (lane / 16) * 8;
                    s_ldm_x4(Qr, s_cvta(&sm.sQ[s_swz(row, col, SM_DQK)]));
                }
                #pragma unroll
                for (int np = 0; np < 2; ++np) {
                    uint32_t Kf[4];
                    const int krow = (nh * 4 + 2 * np + (lane / 16)) * MMA_N + (lane % 8);
                    const int kcol = dl * MMA_K + ((lane / 8) % 2) * 8;
                    s_ldm_x4(Kf, s_cvta(&sm.sKV[s][s_swz(krow, kcol, SM_DTILE)]));
                    const uint32_t Kr0[2] = {Kf[0], Kf[1]};
                    const uint32_t Kr1[2] = {Kf[2], Kf[3]};
                    s_mma(Sr[2 * np], Qr, Kr0, Sr[2 * np]);
                    s_mma(Sr[2 * np + 1], Qr, Kr1, Sr[2 * np + 1]);
                }
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(empty_bar(s));   // K stage s consumed by this warp (release)
        }
        // here g == 17*kb + 9: this warp is past K0..K8 of block kb

        if constexpr (kDiag == 2) {
            // DIAGNOSTIC ONLY: softmax skipped, P = packed raw S (WRONG numerics by construction);
            // keeps the pair rendezvous that publishes P (GEMM-pipeline ceiling measurement).
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int prow_a = wrow + lane / 4, prow_b = prow_a + 8;
                const int pcol = (nh * 4 + n) * MMA_N + (lane % 4) * 2;
                *reinterpret_cast<__nv_bfloat162*>(&sm.sP[s_swz(prow_a, pcol, SM_BTOPK)]) =
                    __floats2bfloat162_rn(Sr[n][0], Sr[n][1]);
                *reinterpret_cast<__nv_bfloat162*>(&sm.sP[s_swz(prow_b, pcol, SM_BTOPK)]) =
                    __floats2bfloat162_rn(Sr[n][2], Sr[n][3]);
            }
            rsum[0] += 1.f; rsum[1] += 1.f;
            pair_sync(ms);
        } else {
            // ---- mask invalid tokens, scale, per-lane partial max (CFG=4 verbatim) ----
            const float scale = params.sm_scale;
            const uint8_t* valid = sm.sValid[kb & 1];
            float tmax[2] = {-INFINITY, -INFINITY};
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int col0 = (nh * 4 + n) * MMA_N + (lane % 4) * 2;
                const bool v0 = valid[col0] != 0, v1 = valid[col0 + 1] != 0;
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

            // ---- pair (n-half) row-max combine: one slot per writer, ONE rendezvous ----
            // Writer: lane%4==0 holds the strip's per-half max for rows lane/4 and lane/4+8.
            // bmax = fmaxf(half 0, half 1) in both warps == CFG=4's two-writer sStatM result.
            if ((lane % 4) == 0) {
                sm.sStatM[nh][wrow + lane / 4] = tmax[0];
                sm.sStatM[nh][wrow + lane / 4 + 8] = tmax[1];
            }
            pair_sync(ms);   // pair sync #1
            const float bmax0 = fmaxf(sm.sStatM[0][wrow + lane / 4], sm.sStatM[1][wrow + lane / 4]);
            const float bmax1 = fmaxf(sm.sStatM[0][wrow + lane / 4 + 8], sm.sStatM[1][wrow + lane / 4 + 8]);

            // ---- online rescale + P = exp(S - max), pack to smem (CFG=4 verbatim) ----
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
            // rsum is per-lane over this warp's n-half; the epilogue reduces across the 4 sharing
            // lanes AND the two n-halves (via sStatS) before normalizing.
            pair_sync(ms);   // pair sync #2: P rows [wrow, wrow+16) complete for this pair
        }

        // ---- PV: slots 9..16 = V7 (K7's resident stage, no gather), V6, ..., V0 ----
        // sP rows [wrow, +16) are written and read only by this pair; the V tile order 7..0 only
        // changes which independent accumulator Or[vt] is touched first (bitwise-identical O).
        const __nv_bfloat16* nxt_ptr = kv_base;
        uint32_t nxt_nb = 0;
        #pragma unroll
        for (int i = 9; i < WS_SLOTS; ++i, ++g) {
            const int s = g & 1;
            const int vt = (i == 9) ? 7 : (WS_SLOTS - 1 - i);
            const bool more_blocks = (kb + 1 < num_topk_blocks);
            // next block's indices, validity and L2 row prefetch, 8 slots before its K0 is issued.
            // WAR on sValid[(kb+1)&1]: its last readers were block kb-1's masks, which every warp
            // finished before consuming K7 of block kb, hence before this thread passed full(K8).
            if (i == 9 && more_blocks) load_block_meta(kb + 1, nxt_ptr, nxt_nb);
            if (i + 1 < WS_SLOTS || more_blocks) {
                if constexpr (kDiag != 1) {
                    mbar_wait(empty_bar(s ^ 1), static_cast<uint32_t>((g - 1) >> 1) & 1u);
                }
                if (i + 1 < WS_SLOTS) {
                    issue_gather(s ^ 1, (WS_SLOTS - 2 - i) * SM_DTILE, cur_ptr, cur_nb);   // V tile 15-i
                } else {
                    cur_ptr = nxt_ptr; cur_nb = nxt_nb;                                   // block kb+1
                    issue_gather(s ^ 1, 0, cur_ptr, cur_nb);                              // its K0
                }
            }
            mbar_wait(full_bar(s), static_cast<uint32_t>(g >> 1) & 1u);   // V(vt) landed / V7* phase
            skew_lag();
            // warp (ms, sub=nh): this tile's 32-col half [sub*32, sub*32+32); each x4.trans V pair
            // load == the two x2.trans loads of n-tiles 2np, 2np+1
            #pragma unroll
            for (int pk = 0; pk < 4; ++pk) {                 // contract 64 tokens = 4 k16
                uint32_t Pr[4];
                {
                    const int prow = wrow + (lane % 16);
                    const int pcol = pk * MMA_K + (lane / 16) * 8;
                    s_ldm_x4(Pr, s_cvta(&sm.sP[s_swz(prow, pcol, SM_BTOPK)]));
                }
                #pragma unroll
                for (int np = 0; np < 2; ++np) {
                    uint32_t Vf[4];
                    const int vrow = pk * MMA_K + (lane % 16);
                    const int vcol = nh * 32 + (2 * np) * MMA_N + (lane / 16) * 8;
                    ldm_x4_trans(Vf, s_cvta(&sm.sKV[s][s_swz(vrow, vcol, SM_DTILE)]));
                    const uint32_t Vr0[2] = {Vf[0], Vf[1]};
                    const uint32_t Vr1[2] = {Vf[2], Vf[3]};
                    s_mma(Or[vt][2 * np], Pr, Vr0, Or[vt][2 * np]);
                    s_mma(Or[vt][2 * np + 1], Pr, Vr1, Or[vt][2 * np + 1]);
                }
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(empty_bar(s));   // V stage s consumed by this warp (release)
        }
    }
    // Every slot this thread issued was waited on (full) before its compute, so none of its
    // cp.async is in flight here; nothing to drain.

    // ---- epilogue (CFG=4 verbatim; the n-half sum via one pair-local slot each) ----
    // rsum: reduce across the 4 lanes sharing each row, then across the two n-halves.
    rsum[0] += __shfl_xor_sync(0xffffffff, rsum[0], 1);
    rsum[0] += __shfl_xor_sync(0xffffffff, rsum[0], 2);
    rsum[1] += __shfl_xor_sync(0xffffffff, rsum[1], 1);
    rsum[1] += __shfl_xor_sync(0xffffffff, rsum[1], 2);
    if ((lane % 4) == 0) {
        sm.sStatS[nh][wrow + lane / 4] = rsum[0];
        sm.sStatS[nh][wrow + lane / 4 + 8] = rsum[1];
    }
    pair_sync(ms);
    const int row_a = wrow + lane / 4, row_b = row_a + 8;
    // half 0 + half 1: the same fp32 add CFG=4's atomicAdd performed, same operand order
    const float sum0 = sm.sStatS[0][row_a] + sm.sStatS[1][row_a];
    const float sum1 = sm.sStatS[0][row_b] + sm.sStatS[1][row_b];
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
    // identical across n-halves after the pair-max combine). One writer per row.
    if (nh == 0 && (lane % 4) == 0) {
        const int gh = h_block * SM_BH;
        const int64_t o0 = (int64_t)s_q_idx * params.h_q + gh;
        params.max_logits[o0 + row_a] = rmax[0] * SM_LOG2E;
        params.max_logits[o0 + row_b] = rmax[1] * SM_LOG2E;
        params.lse[o0 + row_a] = (sum0 > 0.f) ? (rmax[0] + __logf(sum0)) * SM_LOG2E : -INFINITY;
        params.lse[o0 + row_b] = (sum1 > 0.f) ? (rmax[1] + __logf(sum1)) * SM_LOG2E : -INFINITY;
    }
}

// diag: 0 production, 1 RED (empty waits removed + warp-7 skew), 2 no softmax, 3 skew only
template <int kRung>
inline void launch_sparse_fwd_ws(const SparsePrefillParams& params, int diag) {
    static_assert(kRung == 5, "launch_sparse_fwd_ws: rung 5 only");
    const int num_h_blocks = params.h_q / SM_BH;
    const dim3 grid(num_h_blocks * params.s_q);
    constexpr size_t smem = sizeof(SmemPlanWs);
    auto launch = [&](auto kernel_fn) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        kernel_fn<<<grid, dim3(WS_THREADS), smem, params.stream>>>(params);
    };
    switch (diag) {
        case 1:  launch(sparse_prefill_fwd_ws_kernel<kRung, 1>); break;
        case 2:  launch(sparse_prefill_fwd_ws_kernel<kRung, 2>); break;
        case 3:  launch(sparse_prefill_fwd_ws_kernel<kRung, 3>); break;
        default: launch(sparse_prefill_fwd_ws_kernel<kRung, 0>); break;
    }
}

}  // namespace sparse_ws
}  // namespace sm120
