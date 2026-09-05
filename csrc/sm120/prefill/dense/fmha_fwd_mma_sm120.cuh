// ============================================================================
// FUSED PREFILL FORWARD via raw mma.sync + ldmatrix (SM120, register-tiled FA-2)
// ============================================================================
// Hand-issued mma.sync.m16n8k16 + ldmatrix path -- the sm_120 speed-of-light route
// that nvcuda::wmma structurally cannot reach (opaque fragment layout blocks
// ldmatrix.x4 and the zero-shuffle S->P relayout). Clean-room implementation from
// the NVIDIA PTX ISA (mma/ldmatrix fragment+accumulator layouts) and the public
// FlashAttention-2 online-softmax algorithm. Correctness is checked against the
// known-good nvcuda::wmma kernel and torch SDPA.
//
// Config: BLOCK_Q=64 (4 warps x WARP_Q=16 = 1 M-tile/warp), BLOCK_KV=32, register-O.
// mma.m16n8k16 accumulator C layout (per lane, 4 fp32): c0,c1 = row (lane/4),
// cols (lane%4)*2 + {0,1}; c2,c3 = row (lane/4)+8, same cols.
#pragma once

#include <cstdio>
#include <cuda_bf16.h>
#include <cuda.h>              // CUtensorMap + cuTensorMapEncodeTiled types (fwd rung 10, no -lcuda)
#include <cuda_runtime_api.h>  // cudaGetDriverEntryPointByVersion
#include <c10/cuda/CUDAStream.h>

namespace flash {
namespace detail {
namespace mma_fwd {

constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int FM_NW = 4;                  // warps/block
constexpr int FM_BQ = 64;                 // Q rows/block (WARP_Q = 16 -> 1 M-tile/warp)
constexpr int FM_WQ = FM_BQ / FM_NW;      // 16
constexpr int FM_BKV = 32;                // KV/block
constexpr int FM_THREADS = FM_NW * 32;    // 128

// ---- PTX primitives (NVIDIA PTX ISA) ---------------------------------------
__device__ __forceinline__ uint32_t cvta_shared(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldm_x2(uint32_t (&r)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void ldm_x2_trans(uint32_t (&r)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void ldm_x4_trans(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
// ex2.approx.ftz.f32 -- the MUFU path __expf uses internally (CFG ladder rung 3+).
// Non-volatile: a pure function, so ptxas may schedule/CSE it freely.
__device__ __forceinline__ float ex2_approx(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ void mma_16x8x16(float (&d)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2], const float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ void cp_cg16(uint32_t sa, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }
// Wait until at most N cp.async groups remain outstanding (the most recent N stay in
// flight, everything older is visible in smem). N is a compile-time immediate.
template <int N>
__device__ __forceinline__ void cp_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ---- Rung 8/9 primitives: mbarrier, mbarrier-tracked async copies, named barriers -------------
// Every spelling below is accepted by ptxas on PLAIN sm_120 (tests/_probe_warpspec_ptx.cu) and
// lowers to SYNCS.* / LDGSTS(+ARRIVES.LDGSTSBAR) / UBLKCP / BAR.SYNC|BAR.ARV. Semantics: PTX ISA
// "Parallel Synchronization and Communication Instructions" (mbarrier, cp.async, cp.async.bulk,
// bar). mbarrier.try_wait defaults to .acquire.cta, mbarrier.arrive to .release.cta.
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(a), "r"(count) : "memory");
}
// Orders the mbarrier.init before any async-proxy (cp.async.bulk complete_tx) access to it.
__device__ __forceinline__ void mbar_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" :: "r"(a) : "memory");
}
// One arrival + registers tx_bytes of expected async-proxy (complete_tx) traffic for this phase.
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t a, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(a), "r"(tx_bytes) : "memory");
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
// arrival counts against the mbarrier's init count (one per issuing lane), no pending-count bump.
__device__ __forceinline__ void cp_async_mbar_arrive_noinc(uint32_t a) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(a) : "memory");
}
// 1-D bulk copy global -> shared (async proxy), completion signalled as complete_tx bytes on mbar.
// Requires 16-B aligned src/dst and a 16-B multiple size. shared::cluster with a CTA-local
// address is the executing CTA's own smem (implicit 1-CTA cluster).
__device__ __forceinline__ void cp_bulk_g2s(uint32_t sa, const void* g, uint32_t bytes, uint32_t mbar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 :: "r"(sa), "l"(g), "r"(bytes), "r"(mbar) : "memory");
}
// TMA tensor-map load of one box of a rank-3 map ({col, head, row} coordinates) into smem; the
// full box byte count (OOB elements zero-filled and counted) completes as complete_tx on mbar.
// tmap is the generic address of a __grid_constant__ kernel parameter (param space).
// Runtime-verified on this GPU: tests/_probe_tma_tensormap.cu (SWIZZLE_128B image == swz on a
// 64-wide row; rows past the tensor extent arrive as zeros).
__device__ __forceinline__ void tma_load_3d(uint32_t sa, const void* tmap, int c0, int c1, int c2, uint32_t mbar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
                 :: "r"(sa), "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(mbar) : "memory");
}

// XOR swizzle for bank-conflict-free ldmatrix. Permute the 8-elem (16B) granule
// within each 64-elem span by (row mod 8): swz_col = col ^ ((row&7)<<3). It is a
// per-row bijection on the column and only touches column bits 3-5, so every
// 8-aligned int4/ldmatrix granule stays contiguous and 16B-aligned, while the 8
// rows of each ldmatrix tile (row&7 = 0..7) land in 8 distinct granules -> the
// 8-way smem bank conflict (row*256 mod 128 == 0 for DIM=128) collapses to zero.
// Writers (g2s) and readers (ldmatrix) MUST both go through swz for data integrity.
__device__ __forceinline__ int swz(int r, int c, int dim) {
    return r * dim + (c ^ ((r & 7) << 3));
}

// vectorized synchronous global->shared load of [rows, dim] bf16 (dim mult of 8)
__device__ __forceinline__ void g2s(__nv_bfloat16* dst, const __nv_bfloat16* src,
                                    int seq_start, int start, int rows, int num_heads,
                                    int head_idx, int dim) {
    const int tid = threadIdx.x;
    const int stride_token = num_heads * dim;
    const int vec = dim / 8;                       // 8 bf16 per 16B
    for (int i = tid; i < rows * vec; i += FM_THREADS) {
        int r = i / vec, c = (i % vec) * 8;
        const __nv_bfloat16* s = src + (seq_start + start + r) * stride_token + head_idx * dim + c;
        *reinterpret_cast<int4*>(dst + swz(r, c, dim)) = *reinterpret_cast<const int4*>(s);
    }
}

// async (cp.async.cg) variant of g2s: same swizzled 16B-granule mapping, but issues
// non-blocking global->shared copies so the load latency of KV block N+1 overlaps the
// tensor-core compute of block N. Caller commits the group and waits via cp_wait_group.
__device__ __forceinline__ void g2s_cp(__nv_bfloat16* dst, const __nv_bfloat16* src,
                                       int seq_start, int start, int rows, int num_heads,
                                       int head_idx, int dim) {
    const int tid = threadIdx.x;
    const int stride_token = num_heads * dim;
    const int vec = dim / 8;
    for (int i = tid; i < rows * vec; i += FM_THREADS) {
        int r = i / vec, c = (i % vec) * 8;
        const __nv_bfloat16* s = src + (seq_start + start + r) * stride_token + head_idx * dim + c;
        cp_cg16(cvta_shared(dst + swz(r, c, dim)), s);
    }
}

// ---- Rungs 8/9/10/11: FROST-structured 128-row CTA on the rung-7 math -----------------------
// Design: audit/design-fwd-rung8-warpspec.md (+ RUNG 11 ADDENDUM); implementation note
// (including the forced deviation from the design's 288-thread layout): audit/impl-fwd-rung8.md.
// CTA = 8 MMA warps x 32 = 256 threads; warp w owns Q rows [16w, 16w+16) of a 128-row tile and
// the FULL key strip (exactly rung 7's per-warp math: Qr, Sr, Or, rmax/rsum). K/V stream
// through a smem ring driven by a CUTLASS-style full/empty mbarrier pipeline:
//   full[s]  completed by the loads themselves (rung 8: cp.async + cp.async.mbarrier.arrive.noinc
//            from every issuing thread; rung 9: 1-D cp.async.bulk rows, expect_tx/complete_tx;
//            rungs 10/11: one TMA tensor-map box per 64-column chunk, expect_tx/complete_tx),
//   empty[s] completed by one elected arrive per MMA warp once its QK (K) / PV (V) on the stage
//            is done (non-blocking for the arriving warp, like FROST's bar.arrive).
// Rungs 8-10 (2 stages, BKV=64): loads are issued ONE TILE AHEAD by the MMA warps themselves
// (rung 8: every thread issues 1/256 of the tile, rungs 9/10: warp 0), each issue gated on the
// empty event of the tile BEFORE the one being computed -- so a warp blocks only on an event
// that is a full tile old, and the 8 warps (2 per SMSP) drift freely within that slack.
// Rung 11 (1 stage, BKV=128, FROST's kv_tile): thread 0 waits for every warp's QK(n) and issues
// K(n+1) to land under softmax(n)+PV(n), then for every warp's PV(n) and issues V(n+1) to land
// under QK(n+1) (split consumption). No CTA-wide barrier after the prologue, no
// cp.async.commit_group / wait_group anywhere.
// WHY NOT the design's dedicated producer warp (288 threads): the hardware launch check charges
// a CTA's warps rounded UP to the 4 SM sub-partitions (cuda_occupancy.h,
// cudaOccMaxBlocksPerSMRegsLimit: regsAssumedPerCTA = regsPerWarp * roundUp(warps, 4)), so 9
// warps count as 12 and ptxas caps them at 168 registers/thread (measured: every 288-thread
// instantiation spilled at exactly 168); the rung-7 per-warp math needs ~200-216. FROST escapes
// this with setmaxnreg register donation, which ptxas rejects on plain sm_120. 8 warps = 2 per
// sub-partition -> 255-register cap, 1 CTA/SM.
template <int DQK, int DVO, int kRung>
struct WsCfg {
    static constexpr int BQ = 128;                      // Q rows per CTA (8 MMA warps x 16)
    static constexpr int BKV = (kRung >= 11) ? 128 : 64; // keys per tile (rungs 11/12: FROST's kv_tile 128)
    static constexpr int NW = BQ / MMA_M;               // 8 warps, all MMA
    static constexpr int kThreads = NW * 32;            // 256
    static constexpr bool kBulk = (kRung == 9);
    static constexpr bool kTma = (kRung >= 10);
    // K/V ring depth. Rungs 8-10: 2 stages, loads issued one tile ahead. Rung 11: ONE stage each
    // with FROST's split consumption (K(n+1) issued once every warp finished QK(n), landing under
    // softmax(n)+PV(n); V(n+1) once every warp finished PV(n), landing under QK(n+1)).
    static constexpr int kStages = (kRung >= 11) ? 1 : 2;
    static constexpr int kStageShift = (kRung >= 11) ? 0 : 1;   // log2(kStages): tile -> phase index
    // K/V smem row pitch in ELEMENTS. Rung 8: dense rows + the XOR granule swizzle (swz), so the
    // QK/PV ldmatrix code is rung 7's verbatim. Rung 9: padded pitch (+16 B per row, no XOR):
    // 400 B @192 and 272 B @128 are == 1 (mod 8) 16-B granules, so the 8 rows of every ldmatrix
    // 8x8 land on 8 distinct bank groups (conflict-free) and each row is one contiguous
    // cp.async.bulk destination. Rung 10: dense size, but CHUNK-MAJOR [dim/64][64 rows][64 cols]
    // = the TMA SWIZZLE_128B box image (64-wide rows XOR-swizzled exactly like swz); every
    // stage/chunk base must be 1024-B aligned (SWIZZLE_128B), hence the ring alignment slack.
    static constexpr int KP = kBulk ? (DQK + 8) : DQK;
    static constexpr int VP = kBulk ? (DVO + 8) : DVO;
    static constexpr int kStageK = BKV * KP;            // elements
    static constexpr int kStageV = BKV * VP;
    static constexpr int kStage = kStageK + kStageV;
    static constexpr int kQBytes = BQ * DQK * 2;        // prologue Q staging (dead after Qr)
    static constexpr int kRingBytes = kStages * kStage * 2;   // stages x (K + V), aliases Q
    static constexpr int kRingSlack = kTma ? 1024 : 0;  // runtime round-up of the ring base
    static constexpr int kMbarOff = (kQBytes > kRingBytes + kRingSlack) ? kQBytes : (kRingBytes + kRingSlack);
    // mbarrier block (8 x 8 B): fullK[2] @0, fullV[2] @16, emptyK[2] @32, emptyV[2] @48
    static constexpr int kSmemBytes = kMbarOff + 64;
    // arrivals completing a full phase: rung 8 one noinc arrive per issuing thread, rung 9 one
    // per warp-0 lane, rung 10 the single expect_tx arrive (tx bytes complete the phase)
    static constexpr int kFullCount = kTma ? 1 : (kBulk ? 32 : kThreads);
    static constexpr int kEmptyCount = NW;              // one elected arrival per MMA warp
    static constexpr int kChunksK = DQK / 64, kChunksV = DVO / 64;   // TMA boxes per tile
    static constexpr int kChunkElems = BKV * 64;        // one chunk = [BKV rows][64 cols] (chunk-major)
    static constexpr int kBoxBytes = kChunkElems * 2;   // one {64 cols, 1 head, BKV rows} bf16 box
    static_assert(kStages == 1 || kStages == 2, "ring depth");
    static_assert(!kTma || (DQK % 64 == 0 && DVO % 64 == 0), "TMA boxes are 64 columns wide");
    static_assert(!kTma || BKV <= 256, "TMA box rows");
    static_assert(!kTma || ((kStageK * 2) % 1024 == 0 && (kStage * 2) % 1024 == 0),
                  "SWIZZLE_128B: every stage/chunk base must stay 1024-B aligned");
    static_assert(kMbarOff % 8 == 0, "mbarrier block must be 8-B aligned");
    static_assert((KP * 2) % 16 == 0 && (VP * 2) % 16 == 0, "row pitch must keep 16-B granules");
    static_assert((BKV * (DQK / 8)) % kThreads == 0 && (BKV * (DVO / 8)) % kThreads == 0,
                  "rung-8 loader: every thread issues the same number of 16-B granules");
};

// Diag kCfg 95 (env FLASH_MLA_SM120_FWD_MMA_DIAG_TIMING=1): rung 11 with per-phase clock64 attribution
// (lane 0 of every warp accumulates its cycles per phase; atomically summed into fw_diag_cycles at
// kernel end; the launcher prints the breakdown). Correct results, perturbed timing.
__device__ unsigned long long fw_diag_cycles[16];
enum FwDiagPhase { FWD_T_WAITK = 0, FWD_T_QK, FWD_T_ISSUEK, FWD_T_SOFTMAX, FWD_T_WAITV, FWD_T_PV, FWD_T_ISSUEV, FWD_T_NPHASE, FWD_T_TILES = 12, FWD_T_WARPS = 13 };
template <int DQK, int DVO, bool kCausal, int kRung, bool kDiag, bool kDiagTiming = false>
__device__ __forceinline__ void fmha_fwd_ws_body(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int* __restrict__ cu_q,
    const int* __restrict__ cu_kv,
    __nv_bfloat16* __restrict__ o,
    float* __restrict__ lse,
    int num_heads, float scale, int lse_head_stride,
    const CUtensorMap* __restrict__ tmk,   // rung 10 only (K tensor map, kernel param space)
    const CUtensorMap* __restrict__ tmv) { // rung 10 only (V tensor map); nullptr otherwise
    using C = WsCfg<DQK, DVO, kRung>;
    constexpr int BQ = C::BQ, BKV = C::BKV, kThreads = C::kThreads;
    constexpr int KP = C::KP, VP = C::VP;
    constexpr int kStages = C::kStages, kStageShift = C::kStageShift;
    constexpr int DK = DQK / MMA_K;      // d-tiles of the QK contraction (12 @192, 8 @128)
    constexpr int NKV = BKV / MMA_N;     // n8 key tiles per tile (8; 16 at rung 11)
    constexpr int DN = DVO / MMA_N;      // n8 output tiles (16)
    constexpr int PKT = BKV / MMA_K;     // k16 P tiles (4; 8 at rung 11)

    extern __shared__ char smem_raw[];
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem_raw);   // prologue only
    // The K/V ring aliases the dead Q staging. Rung 10 rounds its base up to 1024 B at runtime
    // (SWIZZLE_128B); the slack is part of kMbarOff so the mbarriers never overlap the ring.
    const int ring_off = C::kTma ? static_cast<int>((1024u - (cvta_shared(smem_raw) & 1023u)) & 1023u) : 0;
    __nv_bfloat16* ring = reinterpret_cast<__nv_bfloat16*>(smem_raw + ring_off);
    const uint32_t mbar = cvta_shared(smem_raw + C::kMbarOff);
    auto fullK  = [&](int s) { return mbar + 8 * s; };
    auto fullV  = [&](int s) { return mbar + 16 + 8 * s; };
    auto emptyK = [&](int s) { return mbar + 32 + 8 * s; };
    auto emptyV = [&](int s) { return mbar + 48 + 8 * s; };

    const int b = blockIdx.z, h = blockIdx.y;
    // rung-5 reversed q-block order under a causal mask (LPT: the longest CTAs launch first)
    const int qb = kCausal ? (static_cast<int>(gridDim.x) - 1 - static_cast<int>(blockIdx.x))
                           : static_cast<int>(blockIdx.x);
    if (h >= num_heads) return;
    const int q0 = cu_q[b], q1 = cu_q[b + 1], kv0 = cu_kv[b], kv1 = cu_kv[b + 1];
    const int sq = q1 - q0, skv = kv1 - kv0;
    const int qtile = qb * BQ;
    if (qtile >= sq) return;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wrow = warp * MMA_M;                 // this warp's first Q row in the tile
    const int qrows = min(BQ, sq - qtile);

    // ---- prologue (all 256 threads): Q g2s -> mbarrier init -> sync -> Qr ldmatrix -> sync
    {
        constexpr int vec = DQK / 8;
        const int stride_token = num_heads * DQK;
        for (int i = tid; i < qrows * vec; i += kThreads) {
            const int r = i / vec, c = (i % vec) * 8;
            const __nv_bfloat16* s = q + (q0 + qtile + r) * stride_token + h * DQK + c;
            *reinterpret_cast<int4*>(Qs + swz(r, c, DQK)) = *reinterpret_cast<const int4*>(s);
        }
    }
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            mbar_init(fullK(s), C::kFullCount);   mbar_init(fullV(s), C::kFullCount);
            mbar_init(emptyK(s), C::kEmptyCount); mbar_init(emptyV(s), C::kEmptyCount);
        }
        mbar_fence_init();
    }
    __syncthreads();   // Q staged; mbarriers initialised and visible to every thread
    uint32_t Qr[DK][4];
    #pragma unroll
    for (int d = 0; d < DK; ++d) {
        const int row = wrow + (lane % 16);
        const int col = d * MMA_K + (lane / 16) * 8;
        ldm_x4(Qr[d], cvta_shared(Qs + swz(row, col, DQK)));
    }
    __syncthreads();   // Qs is dead from here on: the K/V ring overwrites it (rung-1 aliasing)

    int nb_max = (skv + BKV - 1) / BKV;
    if (kCausal) {
        // bottom-right causal alignment (KV-cache convention), exactly as rung 7 with BQ=128
        const int last_q = qtile + qrows - 1;
        const int last_key = last_q + (skv - sq);
        nb_max = min(nb_max, (last_key + 1 + BKV - 1) / BKV);
    }
    // smem element index of K/V (row, col) inside a stage (rung 10: chunk-major TMA image)
    auto kidx = [](int r, int c) {
        return C::kTma  ? ((c >> 6) * C::kChunkElems + r * 64 + ((c & 63) ^ ((r & 7) << 3)))
             : C::kBulk ? (r * KP + c) : swz(r, c, DQK);
    };
    auto vidx = [](int r, int c) {
        return C::kTma  ? ((c >> 6) * C::kChunkElems + r * 64 + ((c & 63) ^ ((r & 7) << 3)))
             : C::kBulk ? (r * VP + c) : swz(r, c, DVO);
    };

    // ---- loaders: tile nb -> stage nb & 1. Executed by the ISSUING threads only.
    //   rung 8:  every thread (g2s_cp's lane->granule map over 256 threads), 16-B cp.async with
    //            hardware zero-fill (src-size 0) for tail rows r >= kvsz (source clamped to a
    //            valid row); completion = one noinc arrive per thread, fired by the hardware once
    //            all of that thread's copies landed (256 arrivals). The 4-6 iteration loop is
    //            kept ROLLED: fully unrolled, ptxas hoists per-iteration address invariants
    //            across the mainloop and pins the kernel at the 255-register cap.
    //   rung 9:  warp 0; one 1-D cp.async.bulk per valid row (lane r, r+32), tail rows zero-
    //            filled with 16-B st.shared; each lane arrives AFTER its issues / stores (the
    //            release orders the zero-fill), lane 0's arrival carrying expect_tx for the
    //            phase's bulk bytes (a complete_tx landing before it drives tx-count negative
    //            transiently: legal, the phase cannot complete before lane 0's arrival).
    //   rung 10: warp 0 lane 0; expect_tx for the FULL box bytes, then one 3-D tensor-map load
    //            per 64-column chunk (coordinates {chunk*64, h, kv0 + kvtile}). Rows past the
    //            tensor's row extent arrive as zeros (and still count); rows past kvsz that
    //            belong to the next batch are finite real data, masked by the S mask (K) and
    //            multiplied by P == 0 (V) -- no zero-fill needed, no NaN hazard.
    const int stride_k = num_heads * DQK, stride_v = num_heads * DVO;
    // (tensor-map pointers captured BY VALUE: a by-reference capture next to the "memory"-
    // clobbered TMA asm made ptxas keep them in an 8-byte stack slot)
    auto issue_k = [&, tmk](int nb) {
        const int s = nb & (kStages - 1), kvtile = nb * BKV;
        __nv_bfloat16* Ks = ring + s * C::kStage;
        if constexpr (C::kTma) {
            if (lane == 0) {
                // rung 11: kv0 re-read by the single issuing thread (L1/L2-hot) instead of being
                // held in every thread's registers across the mainloop (register lever)
                const int kvb = (kRung >= 11) ? cu_kv[b] : kv0;
                mbar_arrive_expect_tx(fullK(s), C::kChunksK * C::kBoxBytes);
                #pragma unroll
                for (int ch = 0; ch < C::kChunksK; ++ch)
                    tma_load_3d(cvta_shared(Ks + ch * C::kChunkElems), tmk, ch * 64, h, kvb + kvtile, fullK(s));
            }
        } else if constexpr (!C::kBulk) {
            const int kvsz = min(BKV, skv - kvtile);
            constexpr int vec = DQK / 8;
            #pragma unroll 1
            for (int j = 0; j < (BKV * vec) / kThreads; ++j) {
                const int i = tid + j * kThreads;
                const int r = i / vec, c = (i % vec) * 8;
                const int rs = (r < kvsz) ? r : (kvsz - 1);
                const __nv_bfloat16* src = k + (kv0 + kvtile + rs) * stride_k + h * DQK + c;
                cp_cg16_zfill(cvta_shared(Ks + kidx(r, c)), src, (r < kvsz) ? 16u : 0u);
            }
            cp_async_mbar_arrive_noinc(fullK(s));
        } else {
            const int kvsz = min(BKV, skv - kvtile);
            constexpr uint32_t ROW = DQK * 2;
            #pragma unroll
            for (int r = lane; r < BKV; r += 32) {
                if (r < kvsz) {
                    cp_bulk_g2s(cvta_shared(Ks + r * KP), k + (kv0 + kvtile + r) * stride_k + h * DQK, ROW, fullK(s));
                } else {
                    int4* row = reinterpret_cast<int4*>(Ks + r * KP);
                    #pragma unroll
                    for (int c = 0; c < DQK / 8; ++c) row[c] = make_int4(0, 0, 0, 0);
                }
            }
            if (lane == 0) mbar_arrive_expect_tx(fullK(s), static_cast<uint32_t>(kvsz) * ROW);
            else           mbar_arrive(fullK(s));
        }
    };
    auto issue_v = [&, tmv](int nb) {
        const int s = nb & (kStages - 1), kvtile = nb * BKV;
        __nv_bfloat16* Vs = ring + s * C::kStage + C::kStageK;
        if constexpr (C::kTma) {
            if (lane == 0) {
                const int kvb = (kRung >= 11) ? cu_kv[b] : kv0;
                mbar_arrive_expect_tx(fullV(s), C::kChunksV * C::kBoxBytes);
                #pragma unroll
                for (int ch = 0; ch < C::kChunksV; ++ch)
                    tma_load_3d(cvta_shared(Vs + ch * C::kChunkElems), tmv, ch * 64, h, kvb + kvtile, fullV(s));
            }
        } else if constexpr (!C::kBulk) {
            const int kvsz = min(BKV, skv - kvtile);
            constexpr int vec = DVO / 8;
            #pragma unroll 1
            for (int j = 0; j < (BKV * vec) / kThreads; ++j) {
                const int i = tid + j * kThreads;
                const int r = i / vec, c = (i % vec) * 8;
                const int rs = (r < kvsz) ? r : (kvsz - 1);
                const __nv_bfloat16* src = v + (kv0 + kvtile + rs) * stride_v + h * DVO + c;
                cp_cg16_zfill(cvta_shared(Vs + vidx(r, c)), src, (r < kvsz) ? 16u : 0u);
            }
            cp_async_mbar_arrive_noinc(fullV(s));
        } else {
            const int kvsz = min(BKV, skv - kvtile);
            constexpr uint32_t ROW = DVO * 2;
            #pragma unroll
            for (int r = lane; r < BKV; r += 32) {
                if (r < kvsz) {
                    cp_bulk_g2s(cvta_shared(Vs + r * VP), v + (kv0 + kvtile + r) * stride_v + h * DVO, ROW, fullV(s));
                } else {
                    int4* row = reinterpret_cast<int4*>(Vs + r * VP);
                    #pragma unroll
                    for (int c = 0; c < DVO / 8; ++c) row[c] = make_int4(0, 0, 0, 0);
                }
            }
            if (lane == 0) mbar_arrive_expect_tx(fullV(s), static_cast<uint32_t>(kvsz) * ROW);
            else           mbar_arrive(fullV(s));
        }
    };
    // rung 11: only thread 0 waits on the empty events and issues, so lanes 1..31 of warp 0 run
    // ahead into the mask / per-lane max and rejoin at the softmax's first warp shuffle.
    const bool issuer = (kRung >= 11) ? (tid == 0) : ((C::kBulk || C::kTma) ? (warp == 0) : true);

    // pipeline prologue: tile 0 into stage 0 (both stages start empty: no empty wait)
    if (issuer && nb_max > 0) { issue_k(0); issue_v(0); }

    float Or[DN][4];
    #pragma unroll
    for (int i = 0; i < DN; ++i) { Or[i][0] = Or[i][1] = Or[i][2] = Or[i][3] = 0.f; }
    float rmax[2] = {-INFINITY, -INFINITY}, rsum[2] = {0.f, 0.f};
    const float c2 = scale * 1.4426950408889634f;   // scale * log2(e)
    const int causal_off = skv - sq;
    const int row_a = qtile + wrow + (lane / 4) + causal_off;
    const int row_b = row_a + 8;
    [[maybe_unused]] unsigned long long tdiag[FWD_T_NPHASE] = {};
    [[maybe_unused]] long long tprev = 0;
    auto tstamp = [&](int phase) {
        if constexpr (kDiagTiming) { const long long t = clock64(); tdiag[phase] += static_cast<unsigned long long>(t - tprev); tprev = t; }
    };
    for (int nb = 0; nb < nb_max; ++nb) {
        if constexpr (kDiagTiming) tprev = clock64();
        const int s = nb & (kStages - 1);
        const uint32_t ph = static_cast<uint32_t>(nb >> kStageShift) & 1u;   // k-th use of stage s -> parity k & 1
        const int kvtile = nb * BKV;
        const int kvsz = min(BKV, skv - kvtile);
        const __nv_bfloat16* Ks = ring + s * C::kStage;
        const __nv_bfloat16* Vs = Ks + C::kStageK;

        mbar_wait(fullK(s), ph);                    // K(nb) landed in stage s (acquire)
        tstamp(FWD_T_WAITK);
        // S = Q K^T: d-outer / n-inner with K-fragment prefetch (rung-7 structure, 8 n-tiles)
        float Sr[NKV][4];
        #pragma unroll
        for (int n = 0; n < NKV; ++n) { Sr[n][0] = Sr[n][1] = Sr[n][2] = Sr[n][3] = 0.f; }
        if constexpr (kRung == 12) {
            // Rung 12: rung 11's nesting over TWO n-pairs at once -> FOUR independent accumulator
            // chains per warp (S[2np..2np+3]) instead of two. The ILP map (tests/_probe_mma_ilp_map.cu)
            // gives NACC 2 at 8 warps/SM = 473 TF, NACC 4 = 540: rung 11's SASS shows the back-to-back
            // dependent HMMA spacing as NOPs. Two x4 K fragments live (+4 regs). Per accumulator the
            // d order is unchanged (ascending) -> identical values to rungs 10/11.
            #pragma unroll
            for (int np = 0; np < NKV / 2; np += 2) {
                #pragma unroll
                for (int d = 0; d < DK; ++d) {
                    uint32_t Ka[4], Kb[4];
                    const int krow_a = (2 * np + (lane / 16)) * MMA_N + (lane % 8);
                    const int krow_b = (2 * (np + 1) + (lane / 16)) * MMA_N + (lane % 8);
                    const int kcol = d * MMA_K + ((lane / 8) % 2) * 8;
                    ldm_x4(Ka, cvta_shared(Ks + kidx(krow_a, kcol)));
                    ldm_x4(Kb, cvta_shared(Ks + kidx(krow_b, kcol)));
                    const uint32_t Ka0[2] = {Ka[0], Ka[1]}, Ka1[2] = {Ka[2], Ka[3]};
                    const uint32_t Kb0[2] = {Kb[0], Kb[1]}, Kb1[2] = {Kb[2], Kb[3]};
                    mma_16x8x16(Sr[2 * np],     Qr[d], Ka0, Sr[2 * np]);
                    mma_16x8x16(Sr[2 * np + 1], Qr[d], Ka1, Sr[2 * np + 1]);
                    mma_16x8x16(Sr[2 * np + 2], Qr[d], Kb0, Sr[2 * np + 2]);
                    mma_16x8x16(Sr[2 * np + 3], Qr[d], Kb1, Sr[2 * np + 3]);
                }
            }
        } else if constexpr (kRung == 11) {
            // Rung 11 (FROST F5 nesting): n-pair OUTER / d INNER, one ldmatrix.x4 fragment live
            // (4 regs) instead of Kf[2][NKV/2][4]; Sr[16][4] holds the 128-key strip. Per
            // accumulator the d order is unchanged (ascending) -> same values as the other nesting.
            #pragma unroll
            for (int np = 0; np < NKV / 2; ++np) {
                #pragma unroll
                for (int d = 0; d < DK; ++d) {
                    uint32_t Kf4[4];
                    const int krow = (2 * np + (lane / 16)) * MMA_N + (lane % 8);
                    const int kcol = d * MMA_K + ((lane / 8) % 2) * 8;
                    ldm_x4(Kf4, cvta_shared(Ks + kidx(krow, kcol)));
                    const uint32_t Kr0[2] = {Kf4[0], Kf4[1]};
                    const uint32_t Kr1[2] = {Kf4[2], Kf4[3]};
                    mma_16x8x16(Sr[2 * np], Qr[d], Kr0, Sr[2 * np]);
                    mma_16x8x16(Sr[2 * np + 1], Qr[d], Kr1, Sr[2 * np + 1]);
                }
            }
        } else {
            uint32_t Kf[2][NKV / 2][4];
            auto load_k = [&](int d, uint32_t (&dst)[NKV / 2][4]) {
                #pragma unroll
                for (int np = 0; np < NKV / 2; ++np) {
                    const int krow = (2 * np + (lane / 16)) * MMA_N + (lane % 8);
                    const int kcol = d * MMA_K + ((lane / 8) % 2) * 8;
                    ldm_x4(dst[np], cvta_shared(Ks + kidx(krow, kcol)));
                }
            };
            load_k(0, Kf[0]);
            #pragma unroll
            for (int d = 0; d < DK; ++d) {
                if (d + 1 < DK) load_k(d + 1, Kf[(d + 1) & 1]);
                #pragma unroll
                for (int np = 0; np < NKV / 2; ++np) {
                    const uint32_t Kr0[2] = {Kf[d & 1][np][0], Kf[d & 1][np][1]};
                    const uint32_t Kr1[2] = {Kf[d & 1][np][2], Kf[d & 1][np][3]};
                    mma_16x8x16(Sr[2 * np], Qr[d], Kr0, Sr[2 * np]);
                    mma_16x8x16(Sr[2 * np + 1], Qr[d], Kr1, Sr[2 * np + 1]);
                }
            }
        }
        tstamp(FWD_T_QK);
        // K stage s consumed by this warp: one elected release-arrive after the warp's reads
        __syncwarp();
        if (lane == 0) mbar_arrive(emptyK(s));
        if (issuer && nb + 1 < nb_max) {
            // K(nb+1) -> the stage that held K(nb+1-kStages): every warp must have finished its
            // QK on it (that tile's empty phase). 2 stages: tile nb-1, a full tile old -> normally
            // already done. 1 stage (rung 11): tile nb itself -> thread 0 waits for all 8 warps'
            // QK(nb), then K(nb+1) lands under softmax(nb)+PV(nb) (FROST's split consumption).
            if (nb >= kStages - 1)
                mbar_wait(emptyK(s ^ (kStages - 1)), static_cast<uint32_t>((nb + 1 - kStages) >> kStageShift) & 1u);
            issue_k(nb + 1);
        }
        tstamp(FWD_T_ISSUEK);

        // masks on RAW scores (bottom-right causal + partial-block tail). Rung 11: evaluated only
        // on tiles that intersect the causal diagonal (some row < some key of the tile, i.e. the
        // tile's last key > the q tile's first row) or the seqlen tail; warp-uniform branch. On
        // every other tile the mask code sets nothing by construction -> identical results.
        const bool need_mask = (kRung < 11) || (kvsz < BKV) ||
                               (kCausal && (kvtile + BKV - 1 > qtile + causal_off));
        if (need_mask) {
            if (kCausal) {
                #pragma unroll
                for (int n = 0; n < NKV; ++n) {
                    const int col0 = kvtile + n * MMA_N + (lane % 4) * 2;
                    if (row_a < col0)     Sr[n][0] = -INFINITY;
                    if (row_a < col0 + 1) Sr[n][1] = -INFINITY;
                    if (row_b < col0)     Sr[n][2] = -INFINITY;
                    if (row_b < col0 + 1) Sr[n][3] = -INFINITY;
                }
            }
            #pragma unroll
            for (int n = 0; n < NKV; ++n) {
                const int col0 = n * MMA_N + (lane % 4) * 2;
                if (col0 >= kvsz)     Sr[n][0] = -INFINITY;
                if (col0 + 1 >= kvsz) Sr[n][1] = -INFINITY;
                if (col0 >= kvsz)     Sr[n][2] = -INFINITY;
                if (col0 + 1 >= kvsz) Sr[n][3] = -INFINITY;
            }
        }
        uint32_t Pr[PKT][4];
        if constexpr (kDiag) {
            // DIAGNOSTIC ONLY (rung 98 = env FLASH_MLA_SM120_FWD_MMA_DIAG_NOSOFTMAX=2): softmax
            // skipped, P = packed raw S. WRONG numerics by construction (rung-99 packing).
            #pragma unroll
            for (int n = 0; n < NKV; ++n) {
                __nv_bfloat162* P = reinterpret_cast<__nv_bfloat162*>(Pr[n / 2]);
                P[(n % 2) * 2]     = __floats2bfloat162_rn(Sr[n][0], Sr[n][1]);
                P[(n % 2) * 2 + 1] = __floats2bfloat162_rn(Sr[n][2], Sr[n][3]);
            }
            rsum[0] += 1.f; rsum[1] += 1.f;
        } else {
            // online softmax: base-2 folded scale + conditional rescale (rungs 3-4 semantics)
            float tmax[2] = {-INFINITY, -INFINITY};
            #pragma unroll
            for (int n = 0; n < NKV; ++n) {
                tmax[0] = fmaxf(tmax[0], fmaxf(Sr[n][0], Sr[n][1]));
                tmax[1] = fmaxf(tmax[1], fmaxf(Sr[n][2], Sr[n][3]));
            }
            tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffff, tmax[0], 1));
            tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffff, tmax[0], 2));
            tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffff, tmax[1], 1));
            tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffff, tmax[1], 2));
            const float nmax0 = fmaxf(rmax[0], tmax[0]), nmax1 = fmaxf(rmax[1], tmax[1]);
            const bool up0 = (rmax[0] == -INFINITY) || ((nmax0 - rmax[0]) * c2 > 8.f);
            const bool up1 = (rmax[1] == -INFINITY) || ((nmax1 - rmax[1]) * c2 > 8.f);
            const float ref0 = up0 ? nmax0 : rmax[0];
            const float ref1 = up1 ? nmax1 : rmax[1];
            const float rs0 = up0 ? ((rmax[0] == -INFINITY) ? 0.f : ex2_approx((rmax[0] - nmax0) * c2)) : 1.f;
            const float rs1 = up1 ? ((rmax[1] == -INFINITY) ? 0.f : ex2_approx((rmax[1] - nmax1) * c2)) : 1.f;
            if (__any_sync(0xffffffff, up0 || up1)) {
                #pragma unroll
                for (int i = 0; i < DN; ++i) {
                    Or[i][0] *= rs0; Or[i][1] *= rs0; Or[i][2] *= rs1; Or[i][3] *= rs1;
                }
            }
            rmax[0] = ref0; rmax[1] = ref1;
            const float m0 = (ref0 == -INFINITY) ? 0.f : ref0 * c2;
            const float m1 = (ref1 == -INFINITY) ? 0.f : ref1 * c2;
            float tsum[2] = {0.f, 0.f};
            #pragma unroll
            for (int n = 0; n < NKV; ++n) {
                const float p0 = ex2_approx(fmaf(Sr[n][0], c2, -m0));
                const float p1 = ex2_approx(fmaf(Sr[n][1], c2, -m0));
                const float p2 = ex2_approx(fmaf(Sr[n][2], c2, -m1));
                const float p3 = ex2_approx(fmaf(Sr[n][3], c2, -m1));
                tsum[0] += p0 + p1; tsum[1] += p2 + p3;
                __nv_bfloat162* P = reinterpret_cast<__nv_bfloat162*>(Pr[n / 2]);
                P[(n % 2) * 2]     = __floats2bfloat162_rn(p0, p1);
                P[(n % 2) * 2 + 1] = __floats2bfloat162_rn(p2, p3);
            }
            rsum[0] = rsum[0] * rs0 + tsum[0];
            rsum[1] = rsum[1] * rs1 + tsum[1];
        }

        tstamp(FWD_T_SOFTMAX);
        mbar_wait(fullV(s), ph);                    // V(nb) landed in stage s (acquire)
        tstamp(FWD_T_WAITV);
        // O += P V: pk-outer / dd-inner with V-fragment prefetch (rung-7 structure)
        if constexpr (kRung >= 11) {
            // Register lever (design step 2, "then the V one"): a single live V fragment instead
            // of the Vf[2] double buffer -- at BKV=128 the 255-register cap was hit with a 20-byte
            // spill of loop-invariant addresses at 128/128. Same mma order -> same values.
            #pragma unroll
            for (int pk = 0; pk < PKT; ++pk) {
                #pragma unroll
                for (int ddp = 0; ddp < DN / 2; ++ddp) {
                    uint32_t Vf4[4];
                    const int vrow = pk * MMA_K + (lane % 16);
                    const int vcol = 2 * ddp * MMA_N + (lane / 16) * 8;
                    ldm_x4_trans(Vf4, cvta_shared(Vs + vidx(vrow, vcol)));
                    const uint32_t Vr0[2] = {Vf4[0], Vf4[1]};
                    const uint32_t Vr1[2] = {Vf4[2], Vf4[3]};
                    mma_16x8x16(Or[2 * ddp], Pr[pk], Vr0, Or[2 * ddp]);
                    mma_16x8x16(Or[2 * ddp + 1], Pr[pk], Vr1, Or[2 * ddp + 1]);
                }
            }
        } else {
        #pragma unroll
        for (int pk = 0; pk < PKT; ++pk) {
            uint32_t Vf[2][4];
            auto load_v = [&](int dd, uint32_t (&dst)[4]) {
                const int vrow = pk * MMA_K + (lane % 16);
                const int vcol = dd * MMA_N + (lane / 16) * 8;
                ldm_x4_trans(dst, cvta_shared(Vs + vidx(vrow, vcol)));
            };
            load_v(0, Vf[0]);
            #pragma unroll
            for (int ddp = 0; ddp < DN / 2; ++ddp) {
                if (ddp + 1 < DN / 2) load_v(2 * (ddp + 1), Vf[(ddp + 1) & 1]);
                const uint32_t Vr0[2] = {Vf[ddp & 1][0], Vf[ddp & 1][1]};
                const uint32_t Vr1[2] = {Vf[ddp & 1][2], Vf[ddp & 1][3]};
                mma_16x8x16(Or[2 * ddp], Pr[pk], Vr0, Or[2 * ddp]);
                mma_16x8x16(Or[2 * ddp + 1], Pr[pk], Vr1, Or[2 * ddp + 1]);
            }
        }
        }   // end PV (rungs 8-10 double-buffered V fragments)
        tstamp(FWD_T_PV);
        // V stage s consumed by this warp
        __syncwarp();
        if (lane == 0) mbar_arrive(emptyV(s));
        if (issuer && nb + 1 < nb_max) {
            // V(nb+1) -> the stage that held V(nb+1-kStages): every warp must have finished its
            // PV on it (2 stages: tile nb-1; rung 11: tile nb, then V(nb+1) lands under QK(nb+1))
            if (nb >= kStages - 1)
                mbar_wait(emptyV(s ^ (kStages - 1)), static_cast<uint32_t>((nb + 1 - kStages) >> kStageShift) & 1u);
            issue_v(nb + 1);
        }
        tstamp(FWD_T_ISSUEV);
    }
    if constexpr (kDiagTiming) {
        __syncwarp();
        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < FWD_T_NPHASE; ++i) atomicAdd(&fw_diag_cycles[i], tdiag[i]);
            atomicAdd(&fw_diag_cycles[FWD_T_TILES], static_cast<unsigned long long>(nb_max > 0 ? nb_max : 0));
            atomicAdd(&fw_diag_cycles[FWD_T_WARPS], 1ull);
        }
    }
    // Every issued tile was consumed above (each full phase waited on), so nothing this thread
    // issued is still in flight at exit.

    // ---- epilogue (identical to the shared one with BQ=128 rows) ----
    // Rung 11 re-reads q0 / sq from cu_q here (two L2-hot loads once per CTA) instead of holding
    // them in registers across the whole mainloop (register lever; identical values).
    int q0e = q0, sqe = sq;
    if constexpr (kRung >= 11) { q0e = cu_q[b]; sqe = cu_q[b + 1] - q0e; }
    rsum[0] += __shfl_xor_sync(0xffffffff, rsum[0], 1);
    rsum[0] += __shfl_xor_sync(0xffffffff, rsum[0], 2);
    rsum[1] += __shfl_xor_sync(0xffffffff, rsum[1], 1);
    rsum[1] += __shfl_xor_sync(0xffffffff, rsum[1], 2);
    const int orow_a = qtile + wrow + (lane / 4);
    const int orow_b = orow_a + 8;
    const float inv0 = (rsum[0] > 0.f) ? 1.f / rsum[0] : 0.f;
    const float inv1 = (rsum[1] > 0.f) ? 1.f / rsum[1] : 0.f;
    const int stride_o = num_heads * DVO;
    #pragma unroll
    for (int dd = 0; dd < DN; ++dd) {
        const int ocol = dd * MMA_N + (lane % 4) * 2;
        if (orow_a < sqe) {
            __nv_bfloat162 r = __floats2bfloat162_rn(Or[dd][0] * inv0, Or[dd][1] * inv0);
            *reinterpret_cast<__nv_bfloat162*>(o + (q0e + orow_a) * stride_o + h * DVO + ocol) = r;
        }
        if (orow_b < sqe) {
            __nv_bfloat162 r = __floats2bfloat162_rn(Or[dd][2] * inv1, Or[dd][3] * inv1);
            *reinterpret_cast<__nv_bfloat162*>(o + (q0e + orow_b) * stride_o + h * DVO + ocol) = r;
        }
    }
    if ((lane % 4) == 0) {
        // rmax is in the RAW score domain -> natural-log LSE = rmax*scale + log(rsum)
        const float mref0 = rmax[0] * scale, mref1 = rmax[1] * scale;
        if (orow_a < sqe) lse[(q0e + orow_a) + h * lse_head_stride] = (rsum[0] > 0.f) ? (mref0 + __logf(rsum[0])) : -INFINITY;
        if (orow_b < sqe) lse[(q0e + orow_b) + h * lse_head_stride] = (rsum[1] > 0.f) ? (mref1 + __logf(rsum[1])) : -INFINITY;
    }
}

// ---- Kernel ----------------------------------------------------------------
// Split head dims: Q,K (and S contraction) use DQK; V,O use DVO. DQK==DVO==128 is the MHA
// case; DQK=192/DVO=128 is MLA. Smem: Q[BQ,DQK], K[BKV,DQK] (x2 staged), V[BKV,DVO] (x2).
// kCfg = FLASH_MLA_SM120_FWD_MMA_CFG ladder (cumulative; 0 = legacy, byte-identical):
//   1  K/V stages alias the dead Q smem region -> 40KB @192 (2 CTAs/SM), 32KB @128 (3)
//   2  ldmatrix.x4 for K and V (halves ldmatrix count; identical values/order)
//   3  base-2 softmax, scale folded into one FFMA/element, no per-element -inf selects
//   4  FA-3 conditional rescaling (skip the O/rsum rescale unless max grows by > 2^8)
//   5  reversed q-block launch order for causal (authors' tile scheduler; tail balance)
//   6  ILP restructure: independent accumulator chains adjacent + register double-buffered
//      ldmatrix prefetch (measured NEUTRAL: ptxas already interleaves; kept for structure)
//   7  BKV=64 with FA2-style staggered single K/V buffers (2 barriers per 64 keys; loads in
//      the shadow of compute); same 40KB smem / occupancy as rungs 1-6
//   8  FROST-structured 128-row CTA: 8 MMA warps (256 threads), 2-stage K/V ring with a
//      full/empty mbarrier pipeline (loads issued one tile ahead by the MMA warps, cp.async
//      tracked by cp.async.mbarrier.arrive.noinc); no CTA barrier in the mainloop; 1 CTA/SM
//   9  rung 8 with the 1-D cp.async.bulk row loader into a padded-pitch (no-XOR) layout
//   10 rung 8 with the TMA tensor-map loader (cp.async.bulk.tensor.3d, SWIZZLE_128B image,
//      one elected issuing lane per tile; separate kernel entry fmha_fwd_mma_tma_kernel)
//   11 rung 10 with FROST's kv_tile 128: BKV=128, ONE K and ONE V stage with split consumption
//      (K(n+1) under softmax(n)+PV(n), V(n+1) under QK(n+1)), n-pair-outer QK nesting, masks
//      only on diagonal / tail tiles
//   97 = rung 11, 98 = rung 8, 99 = rung 7 with the softmax skipped (diag, env
//      FLASH_MLA_SM120_FWD_MMA_DIAG_NOSOFTMAX=3/2/1). None is reachable from the CFG ladder.
// __launch_bounds__: 256 threads / 1 CTA per SM (ptxas cap 255 regs) for the 128-row rungs;
// the others keep exactly `(FM_THREADS)` -- a min-blocks argument of 0 emits no .minnctapersm
// directive (verified: identical PTX), so their codegen is unchanged.
template <int DQK, int DVO, bool kCausal, int kCfg>
__global__ void __launch_bounds__(((kCfg >= 8 && kCfg != 99) ? 256 : FM_THREADS),
                                  ((kCfg >= 8 && kCfg != 99) ? 1 : 0))
fmha_fwd_mma_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int* __restrict__ cu_q,
    const int* __restrict__ cu_kv,
    __nv_bfloat16* __restrict__ o,
    float* __restrict__ lse,
    int num_heads, float scale, int max_sq, int max_skv, int lse_head_stride) {
    // kRung = the mainloop generation; the diagnostic values map onto the generation they
    // measure (99 -> 7, 98 -> 8, 97 -> 11, 96 -> 12). Identical values to the former literal
    // checks for every rung <= 7 instantiation.
    constexpr int kRung = (kCfg == 99) ? 7 : (kCfg == 98) ? 8 : (kCfg == 97) ? 11 : (kCfg == 96) ? 12 : (kCfg == 95) ? 11 : kCfg;
    constexpr bool kDiagNoSoftmax = (kCfg == 99 || kCfg == 98 || kCfg == 97 || kCfg == 96);
    static_assert((kRung >= 8) == (kCfg >= 8 && kCfg != 99), "launch bounds / rung mismatch");
    if constexpr (kRung >= 8) {
        // rungs 8/9 (+98): self-contained 128-row pipelined body (prologue, mainloop, epilogue)
        fmha_fwd_ws_body<DQK, DVO, kCausal, kRung, kDiagNoSoftmax>(
            q, k, v, cu_q, cu_kv, o, lse, num_heads, scale, lse_head_stride, nullptr, nullptr);
        return;
    }
    constexpr int DK = DQK / MMA_K;     // d-tiles for QK contraction (DQK=192 -> 12)
    constexpr int NKV = FM_BKV / MMA_N; // n8 tiles in a KV block (4)
    constexpr int DN = DVO / MMA_N;     // n8 output tiles for O (DVO=128 -> 16)
    constexpr int PK = FM_BKV / MMA_K;  // k16 P tiles (2)

    // Double-buffered K/V for the cp.async software pipeline: K stage = FM_BKV*DQK,
    // V stage = FM_BKV*DVO. Q is single-buffered.
    extern __shared__ char smem_raw[];
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    // CFG>=1: the K/V stages ALIAS the Q region. Q is consumed into registers (Qr) in the
    // prologue and Qs is dead for the whole mainloop -- the first cp.async prefetch is
    // issued only after the post-ldmatrix __syncthreads, so no warp can still be reading
    // Qs when stage data lands. 64KB -> 40KB @192/128 (1 -> 2 CTAs/SM), 48 -> 32KB @128.
    __nv_bfloat16* Kbuf = (kCfg >= 1) ? Qs : (Qs + FM_BQ * DQK);
    __nv_bfloat16* Vbuf = Kbuf + 2 * FM_BKV * DQK;

    // Rung 5: reversed q-block order (the authors' causal tile scheduler: "last q tile
    // launches first"). Under a causal mask the LAST q blocks walk the MOST KV blocks, so
    // issuing them first lets the short early blocks fill the tail (LPT balance). Pure
    // block re-ordering: every CTA computes exactly what it did before.
    const int b = blockIdx.z, h = blockIdx.y;
    const int qb = (kCfg >= 5 && kCausal) ? (static_cast<int>(gridDim.x) - 1 - static_cast<int>(blockIdx.x))
                                           : static_cast<int>(blockIdx.x);
    if (h >= num_heads) return;
    const int q0 = cu_q[b], q1 = cu_q[b + 1], kv0 = cu_kv[b], kv1 = cu_kv[b + 1];
    const int sq = q1 - q0, skv = kv1 - kv0;
    const int qtile = qb * FM_BQ;
    if (qtile >= sq) return;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wrow = warp * FM_WQ;                 // this warp's first Q row in the tile

    // load Q tile, ldmatrix -> registers (kept for the whole kernel)
    g2s(Qs, q, q0, qtile, min(FM_BQ, sq - qtile), num_heads, h, DQK);
    __syncthreads();
    uint32_t Qr[DK][4];
    #pragma unroll
    for (int d = 0; d < DK; ++d) {
        int row = wrow + (lane % 16);
        int col = d * MMA_K + (lane / 16) * 8;
        ldm_x4(Qr[d], cvta_shared(Qs + swz(row, col, DQK)));
    }
    __syncthreads();

    float Or[DN][4];
    #pragma unroll
    for (int i = 0; i < DN; ++i) { Or[i][0] = Or[i][1] = Or[i][2] = Or[i][3] = 0.f; }
    float rmax[2] = {-INFINITY, -INFINITY}, rsum[2] = {0.f, 0.f};

    int nb_max = (skv + FM_BKV - 1) / FM_BKV;
    if (kCausal) {
        // Bottom-right causal alignment (FlashAttention / KV-cache convention): a query at
        // sequence index iq sees keys 0..iq+(skv-sq). Reduces to iq when sq==skv (prefill/
        // training); lets a single decode query (sq=1) reach all skv cached keys.
        int last_q = qtile + min(FM_BQ, sq - qtile) - 1;
        int last_key = last_q + (skv - sq);
        nb_max = min(nb_max, (last_key + 1 + FM_BKV - 1) / FM_BKV);
    }

    if constexpr (kRung >= 7) {
        // Rung 7: BKV=64 with FlashAttention-2-style SINGLE K and V buffers, staggered so the
        // loads hide behind compute: V(nb) streams during QK(nb)+softmax(nb), K(nb+1) during
        // PV(nb). Two barriers per 64 keys instead of per 32 -- the per-block serial overhead
        // (barriers, max-shuffle/MUFU chain, bookkeeping) is what pins the 32-key loop at
        // ~270 TF regardless of occupancy (FA2 sm80 reaches 351 TF on this board with this
        // structure). Same 40KB smem as rungs 1-6 (one 64-key K+V stage == two 32-key stages)
        // -> same CTAs/SM. Online-softmax block size changes -> rounding-level differences
        // vs rung 6 (the accuracy suites certify).
        constexpr int BKV7 = 64, NKV7 = BKV7 / MMA_N, PK7 = BKV7 / MMA_K;
        __nv_bfloat16* Ks = Kbuf;                   // single K stage [BKV7 x DQK]
        __nv_bfloat16* Vs = Kbuf + BKV7 * DQK;      // single V stage [BKV7 x DVO]
        int nb_max7 = (skv + BKV7 - 1) / BKV7;
        if (kCausal) {
            int last_q = qtile + min(FM_BQ, sq - qtile) - 1;
            int last_key = last_q + (skv - sq);
            nb_max7 = min(nb_max7, (last_key + 1 + BKV7 - 1) / BKV7);
        }
        { int kvsz0 = min(BKV7, skv); g2s_cp(Ks, k, kv0, 0, kvsz0, num_heads, h, DQK); cp_commit(); }
        const float c2 = scale * 1.4426950408889634f;   // scale * log2(e)
        const int causal_off = skv - sq;
        const int row_a = qtile + wrow + (lane / 4) + causal_off;
        const int row_b = row_a + 8;
        for (int nb = 0; nb < nb_max7; ++nb) {
            const int kvtile = nb * BKV7;
            const int kvsz = min(BKV7, skv - kvtile);
            cp_wait_group<0>();                         // this thread's K(nb) copies landed
            for (int i = tid; i < (BKV7 - kvsz) * DQK; i += FM_THREADS) {   // tail rows never
                int r = kvsz + i / DQK, c = i % DQK;                            // targeted by cp.async
                Ks[swz(r, c, DQK)] = __float2bfloat16(0.f);
            }
            __syncthreads();   // K(nb) visible to every warp; every warp finished PV(nb-1) on Vs
            g2s_cp(Vs, v, kv0, kvtile, kvsz, num_heads, h, DVO);   // V(nb): shadow of QK+softmax
            cp_commit();

            // S = Q K^T: d-outer / n-inner with K-fragment prefetch (rung-6 structure, 8 n-tiles)
            float Sr[NKV7][4];
            #pragma unroll
            for (int n = 0; n < NKV7; ++n) { Sr[n][0] = Sr[n][1] = Sr[n][2] = Sr[n][3] = 0.f; }
            {
                uint32_t Kf[2][NKV7 / 2][4];
                auto load_k = [&](int d, uint32_t (&dst)[NKV7 / 2][4]) {
                    #pragma unroll
                    for (int np = 0; np < NKV7 / 2; ++np) {
                        int krow = (2 * np + (lane / 16)) * MMA_N + (lane % 8);
                        int kcol = d * MMA_K + ((lane / 8) % 2) * 8;
                        ldm_x4(dst[np], cvta_shared(Ks + swz(krow, kcol, DQK)));
                    }
                };
                load_k(0, Kf[0]);
                #pragma unroll
                for (int d = 0; d < DK; ++d) {
                    if (d + 1 < DK) load_k(d + 1, Kf[(d + 1) & 1]);
                    #pragma unroll
                    for (int np = 0; np < NKV7 / 2; ++np) {
                        const uint32_t Kr0[2] = {Kf[d & 1][np][0], Kf[d & 1][np][1]};
                        const uint32_t Kr1[2] = {Kf[d & 1][np][2], Kf[d & 1][np][3]};
                        mma_16x8x16(Sr[2 * np], Qr[d], Kr0, Sr[2 * np]);
                        mma_16x8x16(Sr[2 * np + 1], Qr[d], Kr1, Sr[2 * np + 1]);
                    }
                }
            }
            // masks on RAW scores (bottom-right causal + partial-block tail)
            if (kCausal) {
                #pragma unroll
                for (int n = 0; n < NKV7; ++n) {
                    int col0 = kvtile + n * MMA_N + (lane % 4) * 2;
                    if (row_a < col0)     Sr[n][0] = -INFINITY;
                    if (row_a < col0 + 1) Sr[n][1] = -INFINITY;
                    if (row_b < col0)     Sr[n][2] = -INFINITY;
                    if (row_b < col0 + 1) Sr[n][3] = -INFINITY;
                }
            }
            #pragma unroll
            for (int n = 0; n < NKV7; ++n) {
                int col0 = n * MMA_N + (lane % 4) * 2;
                if (col0 >= kvsz)     Sr[n][0] = -INFINITY;
                if (col0 + 1 >= kvsz) Sr[n][1] = -INFINITY;
                if (col0 >= kvsz)     Sr[n][2] = -INFINITY;
                if (col0 + 1 >= kvsz) Sr[n][3] = -INFINITY;
            }
            uint32_t Pr[PK7][4];
            if constexpr (kDiagNoSoftmax) {
                // DIAGNOSTIC ONLY (see rung-99 note above): softmax skipped, P = packed raw S.
                #pragma unroll
                for (int n = 0; n < NKV7; ++n) {
                    __nv_bfloat162* P = reinterpret_cast<__nv_bfloat162*>(Pr[n / 2]);
                    P[(n % 2) * 2]     = __floats2bfloat162_rn(Sr[n][0], Sr[n][1]);
                    P[(n % 2) * 2 + 1] = __floats2bfloat162_rn(Sr[n][2], Sr[n][3]);
                }
                rsum[0] += 1.f; rsum[1] += 1.f;
            } else {
            // online softmax: base-2 folded scale + conditional rescale (rungs 3-4 semantics)
            float tmax[2] = {-INFINITY, -INFINITY};
            #pragma unroll
            for (int n = 0; n < NKV7; ++n) {
                tmax[0] = fmaxf(tmax[0], fmaxf(Sr[n][0], Sr[n][1]));
                tmax[1] = fmaxf(tmax[1], fmaxf(Sr[n][2], Sr[n][3]));
            }
            tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffff, tmax[0], 1));
            tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffff, tmax[0], 2));
            tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffff, tmax[1], 1));
            tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffff, tmax[1], 2));
            const float nmax0 = fmaxf(rmax[0], tmax[0]), nmax1 = fmaxf(rmax[1], tmax[1]);
            const bool up0 = (rmax[0] == -INFINITY) || ((nmax0 - rmax[0]) * c2 > 8.f);
            const bool up1 = (rmax[1] == -INFINITY) || ((nmax1 - rmax[1]) * c2 > 8.f);
            const float ref0 = up0 ? nmax0 : rmax[0];
            const float ref1 = up1 ? nmax1 : rmax[1];
            const float rs0 = up0 ? ((rmax[0] == -INFINITY) ? 0.f : ex2_approx((rmax[0] - nmax0) * c2)) : 1.f;
            const float rs1 = up1 ? ((rmax[1] == -INFINITY) ? 0.f : ex2_approx((rmax[1] - nmax1) * c2)) : 1.f;
            if (__any_sync(0xffffffff, up0 || up1)) {
                #pragma unroll
                for (int i = 0; i < DN; ++i) {
                    Or[i][0] *= rs0; Or[i][1] *= rs0; Or[i][2] *= rs1; Or[i][3] *= rs1;
                }
            }
            rmax[0] = ref0; rmax[1] = ref1;
            const float m0 = (ref0 == -INFINITY) ? 0.f : ref0 * c2;
            const float m1 = (ref1 == -INFINITY) ? 0.f : ref1 * c2;
            float tsum[2] = {0.f, 0.f};
            #pragma unroll
            for (int n = 0; n < NKV7; ++n) {
                float p0 = ex2_approx(fmaf(Sr[n][0], c2, -m0));
                float p1 = ex2_approx(fmaf(Sr[n][1], c2, -m0));
                float p2 = ex2_approx(fmaf(Sr[n][2], c2, -m1));
                float p3 = ex2_approx(fmaf(Sr[n][3], c2, -m1));
                tsum[0] += p0 + p1; tsum[1] += p2 + p3;
                __nv_bfloat162* P = reinterpret_cast<__nv_bfloat162*>(Pr[n / 2]);
                P[(n % 2) * 2]     = __floats2bfloat162_rn(p0, p1);
                P[(n % 2) * 2 + 1] = __floats2bfloat162_rn(p2, p3);
            }
            rsum[0] = rsum[0] * rs0 + tsum[0];
            rsum[1] = rsum[1] * rs1 + tsum[1];
            }   // end softmax (rung-7 loop)

            cp_wait_group<0>();                         // this thread's V(nb) copies landed
            for (int i = tid; i < (BKV7 - kvsz) * DVO; i += FM_THREADS) {
                int r = kvsz + i / DVO, c = i % DVO;
                Vs[swz(r, c, DVO)] = __float2bfloat16(0.f);
            }
            __syncthreads();   // V(nb) visible to every warp; every warp finished QK(nb) on Ks
            if (nb + 1 < nb_max7) {                     // K(nb+1): shadow of PV(nb)
                int kvsz2 = min(BKV7, skv - (nb + 1) * BKV7);
                g2s_cp(Ks, k, kv0, (nb + 1) * BKV7, kvsz2, num_heads, h, DQK);
                cp_commit();
            }
            // O += P V: pk-outer / dd-inner with V-fragment prefetch (rung-6 structure)
            #pragma unroll
            for (int pk = 0; pk < PK7; ++pk) {
                uint32_t Vf[2][4];
                auto load_v = [&](int dd, uint32_t (&dst)[4]) {
                    int vrow = pk * MMA_K + (lane % 16);
                    int vcol = dd * MMA_N + (lane / 16) * 8;
                    ldm_x4_trans(dst, cvta_shared(Vs + swz(vrow, vcol, DVO)));
                };
                load_v(0, Vf[0]);
                #pragma unroll
                for (int ddp = 0; ddp < DN / 2; ++ddp) {
                    if (ddp + 1 < DN / 2) load_v(2 * (ddp + 1), Vf[(ddp + 1) & 1]);
                    const uint32_t Vr0[2] = {Vf[ddp & 1][0], Vf[ddp & 1][1]};
                    const uint32_t Vr1[2] = {Vf[ddp & 1][2], Vf[ddp & 1][3]};
                    mma_16x8x16(Or[2 * ddp], Pr[pk], Vr0, Or[2 * ddp]);
                    mma_16x8x16(Or[2 * ddp + 1], Pr[pk], Vr1, Or[2 * ddp + 1]);
                }
            }
        }
    } else {
    // cp.async software pipeline: prefetch KV block 0 into stage 0, then each iteration
    // prefetches block nb+1 into the alternate stage while computing block nb -- the
    // global-load latency of nb+1 overlaps the tensor-core compute of nb.
    {
        int kvsz0 = min(FM_BKV, skv);
        g2s_cp(Kbuf, k, kv0, 0, kvsz0, num_heads, h, DQK);
        g2s_cp(Vbuf, v, kv0, 0, kvsz0, num_heads, h, DVO);
        cp_commit();
    }

    for (int nb = 0; nb < nb_max; ++nb) {
        const int cur = nb & 1;
        __nv_bfloat16* Ks = Kbuf + cur * (FM_BKV * DQK);
        __nv_bfloat16* Vs = Vbuf + cur * (FM_BKV * DVO);
        int kvtile = nb * FM_BKV;
        int kvsz = min(FM_BKV, skv - kvtile);

        // Prefetch next block into the alternate stage, then wait for the current block
        // (the older outstanding group) to land. The trailing __syncthreads() at the end
        // of the body guarantees nb's reads of `cur` finish before nb+1 writes into it.
        if (nb + 1 < nb_max) {
            const int nxt = (nb + 1) & 1;
            int kvtile2 = (nb + 1) * FM_BKV;
            int kvsz2 = min(FM_BKV, skv - kvtile2);
            g2s_cp(Kbuf + nxt * (FM_BKV * DQK), k, kv0, kvtile2, kvsz2, num_heads, h, DQK);
            g2s_cp(Vbuf + nxt * (FM_BKV * DVO), v, kv0, kvtile2, kvsz2, num_heads, h, DVO);
            cp_commit();
            cp_wait_group<1>();   // current block ready (1 newer group still in flight)
        } else {
            cp_wait_group<0>();   // last block: drain all outstanding groups
        }
        // zero unused tail rows of a partial last block: masked keys give P=0, but the
        // P@V mma still computes 0*V, and uninitialized smem can be NaN (0*NaN=NaN). K/V
        // have different head dims -> separate loops.
        for (int i = tid; i < (FM_BKV - kvsz) * DQK; i += FM_THREADS) {
            int r = kvsz + i / DQK, c = i % DQK;
            Ks[swz(r, c, DQK)] = __float2bfloat16(0.f);
        }
        for (int i = tid; i < (FM_BKV - kvsz) * DVO; i += FM_THREADS) {
            int r = kvsz + i / DVO, c = i % DVO;
            Vs[swz(r, c, DVO)] = __float2bfloat16(0.f);
        }
        __syncthreads();

        // S = scale * Q @ K^T  -> Sr[NKV][4] (m16n8 accumulator per n8 tile)
        float Sr[NKV][4];
        if constexpr (kCfg >= 6) {
            // Rung 6: ILP restructure. mma.sync bf16->f32 on sm_120 is chain-latency bound: one
            // dependent accumulator chain per warp at 1 warp/SMSP caps at 275 TF (measured,
            // tests/_probe_mma_ilp_map.cu) while >= 2 independent chains reach 470-540 TF.
            // d-OUTER / n-INNER puts the NKV independent Sr chains adjacent in the stream, and
            // the K fragments for d+1 are ldmatrix'd (register double buffer) BEFORE the mmas
            // of d issue, so no mma waits on its own ldmatrix. Per-accumulator order is still
            // d ascending -> values identical to rung 2.
            #pragma unroll
            for (int n = 0; n < NKV; ++n) { Sr[n][0] = Sr[n][1] = Sr[n][2] = Sr[n][3] = 0.f; }
            uint32_t Kf[2][NKV / 2][4];
            auto load_k = [&](int d, uint32_t (&dst)[NKV / 2][4]) {
                #pragma unroll
                for (int np = 0; np < NKV / 2; ++np) {
                    int krow = (2 * np + (lane / 16)) * MMA_N + (lane % 8);
                    int kcol = d * MMA_K + ((lane / 8) % 2) * 8;
                    ldm_x4(dst[np], cvta_shared(Ks + swz(krow, kcol, DQK)));
                }
            };
            load_k(0, Kf[0]);
            #pragma unroll
            for (int d = 0; d < DK; ++d) {
                if (d + 1 < DK) load_k(d + 1, Kf[(d + 1) & 1]);
                #pragma unroll
                for (int np = 0; np < NKV / 2; ++np) {
                    const uint32_t Kr0[2] = {Kf[d & 1][np][0], Kf[d & 1][np][1]};
                    const uint32_t Kr1[2] = {Kf[d & 1][np][2], Kf[d & 1][np][3]};
                    mma_16x8x16(Sr[2 * np], Qr[d], Kr0, Sr[2 * np]);
                    mma_16x8x16(Sr[2 * np + 1], Qr[d], Kr1, Sr[2 * np + 1]);
                }
            }
            if constexpr (kCfg < 3) {
                #pragma unroll
                for (int n = 0; n < NKV; ++n) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) Sr[n][i] *= scale;
                }
            }
        } else if constexpr (kCfg >= 2) {
            // ldmatrix.x4 over the n-PAIR (n, n+1): lanes 16-31 address the second n8 key
            // tile, so one x4 yields both B fragments. Per-tile accumulation order (d
            // ascending) is unchanged -> identical values to the x2 path.
            #pragma unroll
            for (int n = 0; n < NKV; ++n) { Sr[n][0] = Sr[n][1] = Sr[n][2] = Sr[n][3] = 0.f; }
            #pragma unroll
            for (int n = 0; n < NKV; n += 2) {
                #pragma unroll
                for (int d = 0; d < DK; ++d) {
                    uint32_t Kr4[4];
                    int krow = (n + (lane / 16)) * MMA_N + (lane % 8);
                    int kcol = d * MMA_K + ((lane / 8) % 2) * 8;
                    ldm_x4(Kr4, cvta_shared(Ks + swz(krow, kcol, DQK)));
                    const uint32_t Kr0[2] = {Kr4[0], Kr4[1]};
                    const uint32_t Kr1[2] = {Kr4[2], Kr4[3]};
                    mma_16x8x16(Sr[n], Qr[d], Kr0, Sr[n]);
                    mma_16x8x16(Sr[n + 1], Qr[d], Kr1, Sr[n + 1]);
                }
            }
            if constexpr (kCfg < 3) {   // rung 3+ folds the scale into the softmax FFMA
                #pragma unroll
                for (int n = 0; n < NKV; ++n) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) Sr[n][i] *= scale;
                }
            }
        } else {
        #pragma unroll
        for (int n = 0; n < NKV; ++n) {
            Sr[n][0] = Sr[n][1] = Sr[n][2] = Sr[n][3] = 0.f;
            #pragma unroll
            for (int d = 0; d < DK; ++d) {
                uint32_t Kr[2];
                int krow = n * MMA_N + (lane % 8);          // key within block (n8)
                int kcol = d * MMA_K + (lane / 8 % 2) * 8;  // d (lanes 0-15 used: 0 or 8)
                ldm_x2(Kr, cvta_shared(Ks + swz(krow, kcol, DQK)));  // NON-trans (d contiguous)
                mma_16x8x16(Sr[n], Qr[d], Kr, Sr[n]);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) Sr[n][i] *= scale;
        }
        }

#ifdef FMA_DEBUG_QK
        if (nb == 0) {                              // dump raw scores S[:, :BKV] to O, then bail
            int rA = qtile + wrow + (lane / 4), rB = rA + 8, so = num_heads * DVO;
            #pragma unroll
            for (int n = 0; n < NKV; ++n) {
                int c = n * MMA_N + (lane % 4) * 2;
                if (rA < sq) { o[(q0 + rA) * so + h * DVO + c] = __float2bfloat16(Sr[n][0]); o[(q0 + rA) * so + h * DVO + c + 1] = __float2bfloat16(Sr[n][1]); }
                if (rB < sq) { o[(q0 + rB) * so + h * DVO + c] = __float2bfloat16(Sr[n][2]); o[(q0 + rB) * so + h * DVO + c + 1] = __float2bfloat16(Sr[n][3]); }
            }
            return;
        }
#endif

        // causal mask + online softmax (C layout: c0,c1 row=lane/4; c2,c3 row=lane/4+8).
        // Bottom-right alignment: shift query rows by (skv - sq) so "key_pos > query_pos"
        // masks correctly when sq != skv (KV-cache decode). sq==skv -> offset 0 (identical).
        const int causal_off = skv - sq;
        const int row_a = qtile + wrow + (lane / 4) + causal_off;
        const int row_b = row_a + 8;
        if (kCausal) {
            #pragma unroll
            for (int n = 0; n < NKV; ++n) {
                int col0 = kvtile + n * MMA_N + (lane % 4) * 2;
                if (row_a < col0)     Sr[n][0] = -INFINITY;
                if (row_a < col0 + 1) Sr[n][1] = -INFINITY;
                if (row_b < col0)     Sr[n][2] = -INFINITY;
                if (row_b < col0 + 1) Sr[n][3] = -INFINITY;
            }
        }
        // also mask keys past kvsz (partial last block)
        #pragma unroll
        for (int n = 0; n < NKV; ++n) {
            int col0 = n * MMA_N + (lane % 4) * 2;
            if (col0 >= kvsz)     Sr[n][0] = -INFINITY;
            if (col0 + 1 >= kvsz) Sr[n][1] = -INFINITY;
            if (col0 >= kvsz)     Sr[n][2] = -INFINITY;
            if (col0 + 1 >= kvsz) Sr[n][3] = -INFINITY;
        }

        float tmax[2] = {-INFINITY, -INFINITY};
        #pragma unroll
        for (int n = 0; n < NKV; ++n) {
            tmax[0] = fmaxf(tmax[0], fmaxf(Sr[n][0], Sr[n][1]));
            tmax[1] = fmaxf(tmax[1], fmaxf(Sr[n][2], Sr[n][3]));
        }
        tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffff, tmax[0], 1));
        tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffff, tmax[0], 2));
        tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffff, tmax[1], 1));
        tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffff, tmax[1], 2));
        uint32_t Pr[PK][4];
        float tsum[2] = {0.f, 0.f};
        if constexpr (kCfg == 99) {
            // DIAGNOSTIC ONLY (env FLASH_MLA_SM120_FWD_MMA_DIAG_NOSOFTMAX=1, never reachable from
            // the CFG ladder): skip the softmax entirely (P = packed raw S, no max/exp/rescale)
            // to measure the tensor+ldmatrix-only rate of this exact loop structure. WRONG
            // NUMERICS by construction; the accuracy suites fail loudly if it is ever enabled.
            #pragma unroll
            for (int n = 0; n < NKV; ++n) {
                __nv_bfloat162* P = reinterpret_cast<__nv_bfloat162*>(Pr[n / 2]);
                P[(n % 2) * 2]     = __floats2bfloat162_rn(Sr[n][0], Sr[n][1]);
                P[(n % 2) * 2 + 1] = __floats2bfloat162_rn(Sr[n][2], Sr[n][3]);
            }
            rsum[0] += 1.f; rsum[1] += 1.f;
        } else if constexpr (kCfg >= 3) {
            // Rung 3: base-2 softmax on RAW scores with the scale folded into one FFMA per
            // element: p = ex2(S*c2 - ref*c2). Masked scores (-inf) flow through ex2 to an
            // exact 0 with no select, because the row reference is forced finite.
            // Rung 4: FA-3 conditional rescaling -- keep the previous reference max unless
            // the block max exceeds it by > 2^8 (so P <= 256; fp32 O and bf16 P keep their
            // relative precision, and O/rsum stay relative to the SAME reference). The
            // O/rsum rescale (DN*4 FMULs) then runs only when some lane of the warp needs it.
            const float c2 = scale * 1.4426950408889634f;   // scale * log2(e)
            const float nmax0 = fmaxf(rmax[0], tmax[0]), nmax1 = fmaxf(rmax[1], tmax[1]);
            bool up0 = true, up1 = true;
            if constexpr (kCfg >= 4) {
                up0 = (rmax[0] == -INFINITY) || ((nmax0 - rmax[0]) * c2 > 8.f);
                up1 = (rmax[1] == -INFINITY) || ((nmax1 - rmax[1]) * c2 > 8.f);
            }
            const float ref0 = up0 ? nmax0 : rmax[0];
            const float ref1 = up1 ? nmax1 : rmax[1];
            const float rs0 = up0 ? ((rmax[0] == -INFINITY) ? 0.f : ex2_approx((rmax[0] - nmax0) * c2)) : 1.f;
            const float rs1 = up1 ? ((rmax[1] == -INFINITY) ? 0.f : ex2_approx((rmax[1] - nmax1) * c2)) : 1.f;
            if (__any_sync(0xffffffff, up0 || up1)) {   // warp-uniform: skipped only when
                #pragma unroll                             // every lane's rows are stable
                for (int i = 0; i < DN; ++i) {
                    Or[i][0] *= rs0; Or[i][1] *= rs0; Or[i][2] *= rs1; Or[i][3] *= rs1;
                }
            }
            rmax[0] = ref0; rmax[1] = ref1;
            const float m0 = (ref0 == -INFINITY) ? 0.f : ref0 * c2;
            const float m1 = (ref1 == -INFINITY) ? 0.f : ref1 * c2;
            #pragma unroll
            for (int n = 0; n < NKV; ++n) {
                float p0 = ex2_approx(fmaf(Sr[n][0], c2, -m0));
                float p1 = ex2_approx(fmaf(Sr[n][1], c2, -m0));
                float p2 = ex2_approx(fmaf(Sr[n][2], c2, -m1));
                float p3 = ex2_approx(fmaf(Sr[n][3], c2, -m1));
                tsum[0] += p0 + p1; tsum[1] += p2 + p3;
                __nv_bfloat162* P = reinterpret_cast<__nv_bfloat162*>(Pr[n / 2]);
                P[(n % 2) * 2]     = __floats2bfloat162_rn(p0, p1);  // row a (top)
                P[(n % 2) * 2 + 1] = __floats2bfloat162_rn(p2, p3);  // row b (bottom)
            }
            rsum[0] = rsum[0] * rs0 + tsum[0];
            rsum[1] = rsum[1] * rs1 + tsum[1];
        } else {
        float nmax0 = fmaxf(rmax[0], tmax[0]), nmax1 = fmaxf(rmax[1], tmax[1]);
        float rs0 = (rmax[0] == -INFINITY) ? 0.f : __expf(rmax[0] - nmax0);
        float rs1 = (rmax[1] == -INFINITY) ? 0.f : __expf(rmax[1] - nmax1);
        rmax[0] = nmax0; rmax[1] = nmax1;
        #pragma unroll
        for (int i = 0; i < DN; ++i) {
            Or[i][0] *= rs0; Or[i][1] *= rs0; Or[i][2] *= rs1; Or[i][3] *= rs1;
        }

        // P = exp(S - max); pack to bf16x2 A-fragments for P@V
        #pragma unroll
        for (int n = 0; n < NKV; ++n) {
            float p0 = (Sr[n][0] == -INFINITY) ? 0.f : __expf(Sr[n][0] - nmax0);
            float p1 = (Sr[n][1] == -INFINITY) ? 0.f : __expf(Sr[n][1] - nmax0);
            float p2 = (Sr[n][2] == -INFINITY) ? 0.f : __expf(Sr[n][2] - nmax1);
            float p3 = (Sr[n][3] == -INFINITY) ? 0.f : __expf(Sr[n][3] - nmax1);
            tsum[0] += p0 + p1; tsum[1] += p2 + p3;
            __nv_bfloat162* P = reinterpret_cast<__nv_bfloat162*>(Pr[n / 2]);
            P[(n % 2) * 2]     = __floats2bfloat162_rn(p0, p1);  // row a (top)
            P[(n % 2) * 2 + 1] = __floats2bfloat162_rn(p2, p3);  // row b (bottom)
        }
        rsum[0] = rsum[0] * rs0 + tsum[0];
        rsum[1] = rsum[1] * rs1 + tsum[1];
        }

        // O += P @ V  (contract over kv; V via ldmatrix.x2.trans)
        if constexpr (kCfg >= 6) {
            // Rung 6 (PV): pk-OUTER / dd-INNER exposes the DN independent Or chains, and the V
            // fragments of the next dd-pair are ldmatrix'd before the current pair's mmas.
            // Per-accumulator order stays pk ascending -> identical values.
            #pragma unroll
            for (int pk = 0; pk < PK; ++pk) {
                uint32_t Vf[2][4];
                auto load_v = [&](int dd, uint32_t (&dst)[4]) {
                    int vrow = pk * MMA_K + (lane % 16);
                    int vcol = dd * MMA_N + (lane / 16) * 8;
                    ldm_x4_trans(dst, cvta_shared(Vs + swz(vrow, vcol, DVO)));
                };
                load_v(0, Vf[0]);
                #pragma unroll
                for (int ddp = 0; ddp < DN / 2; ++ddp) {
                    if (ddp + 1 < DN / 2) load_v(2 * (ddp + 1), Vf[(ddp + 1) & 1]);
                    const uint32_t Vr0[2] = {Vf[ddp & 1][0], Vf[ddp & 1][1]};
                    const uint32_t Vr1[2] = {Vf[ddp & 1][2], Vf[ddp & 1][3]};
                    mma_16x8x16(Or[2 * ddp], Pr[pk], Vr0, Or[2 * ddp]);
                    mma_16x8x16(Or[2 * ddp + 1], Pr[pk], Vr1, Or[2 * ddp + 1]);
                }
            }
        } else if constexpr (kCfg >= 2) {
            // x4.trans: the same address formula makes lanes 16-31 supply the next n8
            // output tile (col + 8), so one ldmatrix feeds Or[dd] and Or[dd+1].
            #pragma unroll
            for (int dd = 0; dd < DN; dd += 2) {
                #pragma unroll
                for (int pk = 0; pk < PK; ++pk) {
                    uint32_t Vr4[4];
                    int vrow = pk * MMA_K + (lane % 16);
                    int vcol = dd * MMA_N + (lane / 16) * 8;
                    ldm_x4_trans(Vr4, cvta_shared(Vs + swz(vrow, vcol, DVO)));
                    const uint32_t Vr0[2] = {Vr4[0], Vr4[1]};
                    const uint32_t Vr1[2] = {Vr4[2], Vr4[3]};
                    mma_16x8x16(Or[dd], Pr[pk], Vr0, Or[dd]);
                    mma_16x8x16(Or[dd + 1], Pr[pk], Vr1, Or[dd + 1]);
                }
            }
        } else {
        #pragma unroll
        for (int dd = 0; dd < DN; ++dd) {
            #pragma unroll
            for (int pk = 0; pk < PK; ++pk) {
                uint32_t Vr[2];
                int vrow = pk * MMA_K + (lane % 16);
                int vcol = dd * MMA_N + (lane / 16) * 8;
                ldm_x2_trans(Vr, cvta_shared(Vs + swz(vrow, vcol, DVO)));
                mma_16x8x16(Or[dd], Pr[pk], Vr, Or[dd]);
            }
        }
        }
        __syncthreads();
    }
    }  // end of the kCfg < 7 (32-key double-buffered) mainloop

    // FULL row-sum: O (from the P@V mma) sums over ALL keys across lanes, but rsum was
    // only accumulated per-lane (each lane's own columns) -- reduce across the 4 lanes
    // that share each row so the normalizer matches O. (rmax is already butterfly-reduced.)
    rsum[0] += __shfl_xor_sync(0xffffffff, rsum[0], 1);
    rsum[0] += __shfl_xor_sync(0xffffffff, rsum[0], 2);
    rsum[1] += __shfl_xor_sync(0xffffffff, rsum[1], 1);
    rsum[1] += __shfl_xor_sync(0xffffffff, rsum[1], 2);

    // epilogue: O /= rowsum; write O[row], O[row+8]; LSE = max + log(sum)
    const int orow_a = qtile + wrow + (lane / 4);
    const int orow_b = orow_a + 8;
    float inv0 = (rsum[0] > 0.f) ? 1.f / rsum[0] : 0.f;
    float inv1 = (rsum[1] > 0.f) ? 1.f / rsum[1] : 0.f;
    const int stride_o = num_heads * DVO;
    #pragma unroll
    for (int dd = 0; dd < DN; ++dd) {
        int ocol = dd * MMA_N + (lane % 4) * 2;
        if (orow_a < sq) {
            __nv_bfloat162 r = __floats2bfloat162_rn(Or[dd][0] * inv0, Or[dd][1] * inv0);
            *reinterpret_cast<__nv_bfloat162*>(o + (q0 + orow_a) * stride_o + h * DVO + ocol) = r;
        }
        if (orow_b < sq) {
            __nv_bfloat162 r = __floats2bfloat162_rn(Or[dd][2] * inv1, Or[dd][3] * inv1);
            *reinterpret_cast<__nv_bfloat162*>(o + (q0 + orow_b) * stride_o + h * DVO + ocol) = r;
        }
    }
    if ((lane % 4) == 0) {
        // Rung 3+ keeps rmax in the RAW score domain -> the natural-log LSE is
        // rmax*scale + log(rsum) (rsum is relative to the same reference max).
        const float mref0 = (kCfg >= 3) ? rmax[0] * scale : rmax[0];
        const float mref1 = (kCfg >= 3) ? rmax[1] * scale : rmax[1];
        if (orow_a < sq) lse[(q0 + orow_a) + h * lse_head_stride] = (rsum[0] > 0.f) ? (mref0 + __logf(rsum[0])) : -INFINITY;
        if (orow_b < sq) lse[(q0 + orow_b) + h * lse_head_stride] = (rsum[1] > 0.f) ? (mref1 + __logf(rsum[1])) : -INFINITY;
    }
}

// Rung 10 entry: the same 128-row pipelined body with the K/V TMA tensor maps as
// __grid_constant__ parameters (cp.async.bulk.tensor reads the descriptor from param space).
// A separate entry so the existing kernel's parameter layout (and SASS) is untouched.
template <int DQK, int DVO, bool kCausal, int kCfg>
__global__ void __launch_bounds__(256, 1)
fmha_fwd_mma_tma_kernel(
    const __grid_constant__ CUtensorMap tmk,
    const __grid_constant__ CUtensorMap tmv,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int* __restrict__ cu_q,
    const int* __restrict__ cu_kv,
    __nv_bfloat16* __restrict__ o,
    float* __restrict__ lse,
    int num_heads, float scale, int lse_head_stride) {
    static_assert(kCfg == 10 || kCfg == 11 || kCfg == 12 || kCfg == 97 || kCfg == 96 || kCfg == 95,
                  "fmha_fwd_mma_tma_kernel is the TMA-loader entry (rungs 10-12 and diags 97/96/95) only");
    constexpr int kRung = (kCfg == 97) ? 11 : (kCfg == 96) ? 12 : (kCfg == 95) ? 11 : kCfg;
    constexpr bool kDiagNoSoftmax = (kCfg == 97 || kCfg == 96);
    fmha_fwd_ws_body<DQK, DVO, kCausal, kRung, kDiagNoSoftmax, (kCfg == 95)>(
        q, k, v, cu_q, cu_kv, o, lse, num_heads, scale, lse_head_stride, &tmk, &tmv);
}

// Host side of rung 10: cuTensorMapEncodeTiled resolved through the runtime's driver entry
// point (no -lcuda link), cached once. Rank-3 bf16 map over a packed [tokens][heads][dim]
// tensor: dims {dim, heads, tokens}, byte strides {dim*2, heads*dim*2}, box {64, 1, 64},
// SWIZZLE_128B, L2 promotion 128 B, OOB fill NONE (zeros) -- exactly the encode that
// tests/_probe_tma_tensormap.cu verified on this GPU (image == swz on 64-wide rows).
using PFN_fwdTensorMapEncodeTiled = CUresult (*)(
    CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
    const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
    CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
inline PFN_fwdTensorMapEncodeTiled fwd_tensormap_encoder() {
    static const PFN_fwdTensorMapEncodeTiled fn = []() -> PFN_fwdTensorMapEncodeTiled {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult qr = cudaDriverEntryPointSymbolNotFound;
        const cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000,
                                                               cudaEnableDefault, &qr);
        return (e == cudaSuccess && qr == cudaDriverEntryPointSuccess && p != nullptr)
                   ? reinterpret_cast<PFN_fwdTensorMapEncodeTiled>(p) : nullptr;
    }();
    return fn;
}
inline CUtensorMap fwd_make_kv_tensormap(const at::Tensor& t, int dim, int num_heads, int box_rows) {
    alignas(64) CUtensorMap m;
    const cuuint64_t dims[3]    = {static_cast<cuuint64_t>(dim), static_cast<cuuint64_t>(num_heads),
                                   static_cast<cuuint64_t>(t.size(0))};
    const cuuint64_t strides[2] = {static_cast<cuuint64_t>(dim) * 2,
                                   static_cast<cuuint64_t>(num_heads) * static_cast<cuuint64_t>(dim) * 2};
    const cuuint32_t box[3]     = {64, 1, static_cast<cuuint32_t>(box_rows)};   // rows = BKV (64; 128 at rung 11)
    const cuuint32_t estr[3]    = {1, 1, 1};
    const PFN_fwdTensorMapEncodeTiled encode = fwd_tensormap_encoder();
    TORCH_CHECK(encode != nullptr,
                "flash_mla_sm120 fwd rung 10: cuTensorMapEncodeTiled driver entry point unavailable");
    const CUresult cr = encode(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, t.data_ptr(), dims, strides, box, estr,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(cr == CUDA_SUCCESS, "flash_mla_sm120 fwd rung 10: cuTensorMapEncodeTiled failed rc=",
                static_cast<int>(cr));
    return m;
}

template <int DQK, int DVO, bool kCausal, int kCfg>
void launch_fmha_fwd_mma(const c10::cuda::CUDAStream& stream,
                         at::Tensor q, at::Tensor k, at::Tensor v,
                         at::Tensor cu_q, at::Tensor cu_kv, at::Tensor o, at::Tensor lse,
                         float scale, int max_sq, int max_skv) {
    constexpr int kRung = (kCfg == 99) ? 7 : (kCfg == 98) ? 8 : (kCfg == 97) ? 11 : (kCfg == 96) ? 12 : (kCfg == 95) ? 11 : kCfg;
    constexpr bool kWs = (kRung >= 8);                 // 128-row pipelined rungs 8..12 (+98/97/96/95)
    constexpr int kBQ = kWs ? WsCfg<DQK, DVO, kRung>::BQ : FM_BQ;
    constexpr int kThreads = kWs ? WsCfg<DQK, DVO, kRung>::kThreads : FM_THREADS;
    const int batch = cu_q.size(0) - 1, num_heads = q.size(1);
    const int qblocks = (max_sq + kBQ - 1) / kBQ;
    dim3 grid(qblocks, num_heads, batch);
    // Q[BQ,DQK] + 2x K[BKV,DQK] + 2x V[BKV,DVO]; rung 1+ overlays the stages on Q.
    const size_t q_bytes  = static_cast<size_t>(FM_BQ * DQK) * sizeof(__nv_bfloat16);
    const size_t kv_bytes = static_cast<size_t>(2 * FM_BKV * DQK + 2 * FM_BKV * DVO) * sizeof(__nv_bfloat16);
    // rungs 8/9: max(Q staging, 2-stage ring) + 64 B of mbarriers (WsCfg::kSmemBytes)
    const size_t smem = kWs ? static_cast<size_t>(WsCfg<DQK, DVO, kRung>::kSmemBytes)
                            : ((kCfg >= 1) ? (q_bytes > kv_bytes ? q_bytes : kv_bytes) : (q_bytes + kv_bytes));
    if constexpr (kRung >= 9) {
        // cp.async.bulk / TMA: 16-B aligned global row starts (base pointer + every row stride)
        // and a 16-B multiple row size (DQK*2 / DVO*2 always are). The kernel and the tensor
        // maps address K/V as packed [tokens][heads][dim], so the strides must be that layout.
        const auto aligned16 = [](const at::Tensor& t) {
            return (reinterpret_cast<uintptr_t>(t.data_ptr()) % 16) == 0;
        };
        TORCH_CHECK(aligned16(k) && aligned16(v),
                    "flash_mla_sm120 fwd rung ", kRung, " (bulk/TMA loader) needs 16-B aligned k/v data pointers");
        TORCH_CHECK(k.stride(-1) == 1 && v.stride(-1) == 1 &&
                    k.stride(-2) == DQK && v.stride(-2) == DVO &&
                    k.stride(0) == static_cast<int64_t>(num_heads) * DQK &&
                    v.stride(0) == static_cast<int64_t>(num_heads) * DVO,
                    "flash_mla_sm120 fwd rung ", kRung, " needs packed [tokens][heads][dim] k/v (16-B row starts)");
    }
    if constexpr (kRung >= 10) {   // TMA loader rungs 10 / 11 (+ diag 97)
        constexpr int kBoxRows = WsCfg<DQK, DVO, kRung>::BKV;
        const CUtensorMap tmk = fwd_make_kv_tensormap(k, DQK, num_heads, kBoxRows);
        const CUtensorMap tmv = fwd_make_kv_tensormap(v, DVO, num_heads, kBoxRows);
        auto kern10 = fmha_fwd_mma_tma_kernel<DQK, DVO, kCausal, kCfg>;
        cudaFuncSetAttribute(kern10, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kern10<<<grid, dim3(kThreads), smem, stream.stream()>>>(
            tmk, tmv,
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
            cu_q.data_ptr<int>(), cu_kv.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(o.data_ptr()), lse.data_ptr<float>(),
            num_heads, scale, static_cast<int>(lse.stride(1)));
        if constexpr (kCfg == 95) {   // timing diag: sync, read the per-phase cycle sums, print, reset
            C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            unsigned long long c[16] = {};
            C10_CUDA_CHECK(cudaMemcpyFromSymbol(c, fw_diag_cycles, sizeof(c)));
            static const char* names[FWD_T_NPHASE] = {"wait_K", "QK", "issue_K", "mask+softmax", "wait_V", "PV", "issue_V"};
            unsigned long long tot = 0;
            for (int i = 0; i < FWD_T_NPHASE; ++i) tot += c[i];
            const double per_tile_warp = (c[FWD_T_TILES] > 0) ? static_cast<double>(tot) / static_cast<double>(c[FWD_T_TILES]) : 0.0;
            fprintf(stderr, "[FWD-TIMING DQK=%d causal=%d] warps=%llu tiles/warp=%.1f cycles/tile/warp=%.0f |", DQK, static_cast<int>(kCausal), c[FWD_T_WARPS],
                    c[FWD_T_WARPS] ? static_cast<double>(c[FWD_T_TILES]) / static_cast<double>(c[FWD_T_WARPS]) : 0.0, per_tile_warp);
            for (int i = 0; i < FWD_T_NPHASE; ++i) fprintf(stderr, " %s %.1f%%", names[i], tot ? 100.0 * static_cast<double>(c[i]) / static_cast<double>(tot) : 0.0);
            fprintf(stderr, "%s", "\n");
            unsigned long long z[16] = {};
            C10_CUDA_CHECK(cudaMemcpyToSymbol(fw_diag_cycles, z, sizeof(z)));
        }
    } else {   // (else-branch so the generic kernel is not instantiated for the TMA rung)
    auto kern = fmha_fwd_mma_kernel<DQK, DVO, kCausal, kCfg>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    const int lse_head_stride = static_cast<int>(lse.stride(1));
    kern<<<grid, dim3(kThreads), smem, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        cu_q.data_ptr<int>(), cu_kv.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(o.data_ptr()), lse.data_ptr<float>(),
        num_heads, scale, max_sq, max_skv, lse_head_stride);
    }
}

}  // namespace mma_fwd
}  // namespace detail
}  // namespace flash
