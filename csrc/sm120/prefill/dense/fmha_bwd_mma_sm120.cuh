// ============================================================================
// FUSED PREFILL BACKWARD via raw mma.sync + ldmatrix (SM120, register-tiled FA-2)
// ============================================================================
// The mma.sync analog of fmha_fwd_mma_sm120.cuh for the BACKWARD pass. Replaces the
// nvcuda::wmma backward (16x16 tiles, intermediates round-tripped through smem, ~5% of
// peak) with register-tiled accumulation + swizzled ldmatrix loads. Split head dims:
// Q,K,dQ,dK use DQK; V,O,dO,dV use DVO. DQK==DVO==128 is MHA; DQK=192/DVO=128 is MLA.
//
// TWO-KERNEL SPLIT (the FA-2 / SM100-author design): a fused single kernel forces dQ to
// be atomicAdd-ed to global (every KV-block CTA touches every query's dQ); those contended
// fp32 atomics measured as ~1.9x the entire backward runtime. Splitting kills them:
//   * dKdV kernel: CTA per KV block (BN keys); NW=4 warps partition the keys; loop Q;
//     dK/dV resident in registers -> single non-atomic store.
//   * dQ kernel: CTA per Q block (BM queries); warps partition the queries; loop KV; dQ
//     resident in registers -> single non-atomic store.
//
// The 5 matmuls (operand -> ldmatrix variant), all m16n8k16 bf16->fp32:
//   S  = scale*Q @ K^T   A=Q  x4 nt , B=K  x2 nt    contract DQK
//   dP = dO @ V^T        A=dO x4 nt , B=V  x2 nt    contract DVO
//   dV+= P^T @ dO        A=P^T(smem) x4 nt , B=dO x2.T  contract queries; out DVO
//   dK+= dS^T @ Q        A=dS^T(smem) x4 nt, B=Q  x2.T  contract queries; out DQK
//   dQ+= dS @ K          A=dS(smem)  x4 nt , B=K  x2.T  contract keys;    out DQK
// S and dP have DIFFERENT contraction dims (DQK vs DVO) -> separate accumulation loops.
// P/dS produced in the mma C-layout; transposed operands materialized to swizzled smem.
// delta=rowsum(dO*O) (over DVO) and S/P recomputed from the forward LSE (token-major).
//
// MLA (DQK=192) halves the query tile to BM=32 so the larger Q/K smem still fits 99KB.
//
// SINGLE-KERNEL VARIANT (FLASH_MLA_SM120_BWD_MMA_CFG=1; 0 = the two-kernel split above,
// byte-identical): the dKdV kernel also forms dQ for each Q block from the dS it already
// holds (dQ = dS @ K over its 64 resident keys) and reduces the fp32 [32 x DQK] tile into an
// fp32 accumulator in the caller's workspace with cp.reduce.async.bulk -- the sm120
// substitute for the sm100 authors' TMA reduce-add (measured 3506 GB/s into L2-resident
// accumulators, 7.2x scalar atomics, faster than plain stores; tests/_probe_bulk_reduce.cu).
// A convert kernel casts the accumulator to bf16; the dQ kernel (which RE-COMPUTES S and dP
// = 2 of the 7 GEMM units and re-streams K/V) is not launched. dQ accumulation order is then
// fp32-ULP nondeterministic across runs (like the authors' sm100 path and FlashAttention-2's
// dQ atomics); dK/dV are unchanged.
//
// CFG=2: single kernel like CFG=1, but dQ is reduced register-direct from the dQ GEMM
// C-fragments with red.global.add.v2.f32 (no staging, no proxy fence, no extra barriers).
// CFG=3 (audit/design-bwd-cfg3-warp-pairs.md): CFG=2's math and dQ path with an 8-WARP CTA
// (256 threads) at the SAME smem -> still 1 CTA/SM, but 2 resident warps per SM sub-partition
// (measured: one dependent mma chain 275 TF at 4 warps/SM vs 473 TF at 8). Warp w = key group
// (w & 3, CFG<=2's key map) x query half (w >> 2, m-tile of the BM=32 block): each warp owns a
// 16x16 slab of S/dP/P/dS and accumulates PARTIAL dK/dV over its 16 queries; the two halves of
// a key group are merged once at kernel end through the dead Q/dO stage buffers in a fixed
// fp32 order (dK/dV deterministic); dQ stays red.v2 (fp32-ULP nondeterministic, as CFG=2).
// CFG=4 (audit/design-bwd-cfg4-frost.md, audit/impl-bwd-cfg4.md): FROST's B-structure on the
// CFG=2 math -- 8 math warps (256 threads, 255-register cap, 1 CTA/SM), tiles 32 q x 64 kv @192 /
// 64 q x 64 kv @128, per-GEMM warp partitions S/dP (2,4), dK/dV (2,4), dQ (1,8) @192 / (2,4)
// @128, V register-resident as dP B fragments (loaded once), P and dS written ONCE to 2-stage
// swizzled bf16 slabs and read back with ldmatrix.x4.trans / .x4 (no movmatrix), Q/dO/lse/delta
// streamed through mbarrier-tracked cp.async stages issued ONE TILE AHEAD by the math warps
// themselves (stage WAR = an EMPTY mbarrier per stage), ONE blocking barrier per tile, dQ via
// red.v2 as CFG=2. dK/dV are accumulated by exactly one warp per element in CFG=2's summation
// order (ascending 16-q k-tiles, ascending Q tiles) -> bitwise IDENTICAL to CFG=2 and run to
// run; dQ stays fp32-ULP nondeterministic across runs (red order), exactly as CFG=2/3.
#pragma once

#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <cstdio>

#include "sm120/prefill/dense/fmha_fwd_mma_sm120.cuh"  // reuse swz/ldm_*/mma primitives

namespace flash {
namespace detail {
namespace mma_bwd {

using mma_fwd::swz;
using mma_fwd::cvta_shared;
using mma_fwd::ldm_x4;
using mma_fwd::ldm_x2;
using mma_fwd::ldm_x2_trans;
using mma_fwd::mma_16x8x16;
using mma_fwd::g2s_cp;
using mma_fwd::cp_commit;
using mma_fwd::cp_wait_group;

// pack two fp32 -> bf16x2 register (low=a, high=b), matching the mma A-fragment packing.
__device__ __forceinline__ uint32_t packbf(float a, float b) {
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&v);
}
// warp 8x8 in-register transpose (sm_75+). Transposes the mma C-accumulator 8x8 fragment
// {row=lane/4, cols (lane%4)*2,+1} into the mma A-operand 8x8 fragment -- eliminates the
// shared-memory scatter+reload round-trip for the P^T / dS^T transposed operands.
__device__ __forceinline__ uint32_t movmatrix_t(uint32_t a) {
    uint32_t d;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(d) : "r"(a));
    return d;
}

// ---- Bulk async-group primitives (PTX ISA 8.0+, sm_90+; proven on plain sm_120 by ----
// tests/_probe_bulk_reduce.cu: 3506 GB/s, bit-exact sums). Used by the CFG>=1 dQ phase.
// Every one carries a "memory" clobber so the compiler cannot move the generic st.shared
// of the staging tile across the fence / issue / wait points.
//
// cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [gdst], [ssrc], kBytes:
// the bulk engine (ASYNC proxy) reads kBytes of fp32 from a shared::cta address and adds
// them element-wise into global memory (performed in L2). Both addresses must be
// 16B-aligned and kBytes a multiple of 16. Completion is tracked per ISSUING THREAD via
// bulk async-groups (commit_group / wait_group[.read]).
template <int kBytes>
__device__ __forceinline__ void bulk_reduce_add_f32(float* gdst, uint32_t ssrc) {
    static_assert(kBytes > 0 && kBytes % 16 == 0, "cp.reduce.async.bulk size must be a 16B multiple");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n"
                 :: "l"(gdst), "r"(ssrc), "n"(kBytes) : "memory");
}
// Close this thread's current bulk async-group (all bulk ops issued since the last commit).
__device__ __forceinline__ void bulk_commit_group() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}
// Wait until at most N of this thread's bulk async-groups are still READING their smem
// source (the global-side adds may still be in flight). Compile-time immediate N.
template <int N>
__device__ __forceinline__ void bulk_wait_group_read() {
    asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(N) : "memory");
}
// Wait until at most N of this thread's bulk async-groups are pending at all (smem reads
// AND global adds complete).
template <int N>
__device__ __forceinline__ void bulk_wait_group() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}
// Generic-proxy -> async-proxy ordering for shared::cta: makes the EXECUTING thread's prior
// st.shared visible to later async-proxy reads (cp.reduce.async.bulk) of the same smem.
// Required by the PTX memory model; CUTLASS issues the same fence_view_async_shared()
// before every TMA / bulk store out of smem.
// red.global.add.v2.f32: fire-and-forget 8B vector reduction (ptxas-proven on plain sm_120,
// tests/_probe_blackwell_reducers.sh; the v4 form measured 2.0 TB/s into L2-resident fp32 in
// tests/_probe_bulk_reduce.cu). NVIDIA's FROST sm120 backward reduces dQ exactly this way
// (red.add.f32x2 into an fp32 dq_accum): no smem staging, no proxy fence, no barriers.
__device__ __forceinline__ void red_add_v4f32(float* g, float a, float b, float c, float d) {   // 16-B aligned g
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(g), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
__device__ __forceinline__ void red_add_v2f32(float* g, float a, float b) {
    asm volatile("red.global.add.v2.f32 [%0], {%1, %2};\n" :: "l"(g), "f"(a), "f"(b) : "memory");
}
__device__ __forceinline__ void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// ---- Preprocess: delta[token,head] = sum_d O*dO (over DVO) ------------------
// FA-2 computes this ONCE in a separate pass (Algo 2 line 4); doing it inline in the main
// kernels forced the O tile to stay resident in smem -> blocked cp.async. One warp/token.
__global__ void fmha_bwd_mma_delta_kernel(
    const __nv_bfloat16* __restrict__ o, const __nv_bfloat16* __restrict__ d_o,
    float* __restrict__ delta, int n_th, int dvo) {
    const int warps = blockDim.x / 32;
    const int fid = blockIdx.x * warps + threadIdx.x / 32;   // flat (token*num_heads + head)
    const int lane = threadIdx.x % 32;
    if (fid >= n_th) return;
    const __nv_bfloat16* op  = o   + (size_t)fid * dvo;
    const __nv_bfloat16* dop = d_o + (size_t)fid * dvo;
    float s = 0.f;
    for (int d = lane * 2; d < dvo; d += 64) {
        __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(op + d);
        __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(dop + d);
        s += __bfloat162float(__low2bfloat16(a)) * __bfloat162float(__low2bfloat16(b))
           + __bfloat162float(__high2bfloat16(a)) * __bfloat162float(__high2bfloat16(b));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    if (lane == 0) delta[fid] = s;
}

constexpr int BM_M = 16, BM_N = 8, BM_K = 16;     // mma shape
constexpr int BW_NW = 4;                          // warps/block
constexpr int BW_BN = 64;                         // keys per CTA (dKdV) / per KV-iter (dQ)
constexpr int BW_WN = BW_BN / BW_NW;              // 16 keys per warp (dKdV)
constexpr int BW_THREADS = BW_NW * 32;            // 128
// Query tile: 64 for MHA, 32 for MLA (the 192-wide Q/K smem needs the smaller tile to
// stay under the 99KB dynamic-smem wall).
template <int DQK> __host__ __device__ constexpr int bm_for() { return DQK > 128 ? 32 : 64; }
// CTA thread count of the dKdV kernel per backward config: CFG 0-2 = 4 warps (BW_THREADS);
// CFG=3 = 8 warps (warp-pair occupancy split, same smem); CFG=4 = 8 math warps (BW4_THREADS,
// FROST structure). Static member so it is usable in __launch_bounds__, the launcher's block
// dim and the kernel's strided loops alike.
// WHY CFG=4 is 256 and not the design's 288 (8 + 1 producer warp): on cc 12.0 the hardware
// register launch check charges a CTA's warps rounded UP to the 4 SM sub-partitions
// (cuda_occupancy.h, cudaOccMaxBlocksPerSMRegsLimit: regsAssumedPerCTA = regsAllocatedPerWarp *
// roundUp(warps, numSubPartitions)), so 9 warps are charged as 12 -> 12 x 32 x R <= 65536 ->
// R <= 168: __launch_bounds__(288) caps ptxas at 168 registers (the dense fwd rung-8 attempt
// measured exactly that, audit/impl-fwd-rung8.md sect. 0) and setmaxnreg is rejected on plain
// sm_120 (no donors). 8 warps = 2 per sub-partition at the 255 cap; the loads are striped over
// the math warps instead (see fmha_bwd_mma_dkdv_cfg4_body).
constexpr int BW4_THREADS = 256;
template <int kCfg> struct BwCta { static constexpr int kThreads = (kCfg >= 4) ? BW4_THREADS : ((kCfg >= 3) ? 2 * BW_THREADS : BW_THREADS); };

// ---- CFG>=1 (single-kernel) dQ-phase smem geometry ---------------------------------
// Dedicated regions appended after lseS/deltaS in the dKdV kernel (nothing aliases them):
//   sDS  bf16 [BM=32][BW_BN=64], XOR-swizzled swz(r,c,64)               =  4,096 B
//   sDQh fp32 [BM=32][DQK/2 + BW_DQH_PAD] row-major, padded row stride   = 13,312 B @192 / 9,216 B @128
// Why BW_DQH_PAD = 8 floats: the mma C-fragment float2 stores write row = lane/4, i.e. a
// 64-bit warp store is serviced as two 16-lane phases of 4 rows x 8 consecutive fp32. With a
// row stride == 0 (mod 32 floats) every row starts at bank 0 -> 4-way conflict per phase;
// with stride == 8 (mod 32) rows r..r+3 occupy banks 8r..8r+7 -> disjoint -> conflict-free
// (2 wavefronts, the minimum for a 256B warp store). A +4 pad (stride == 4 mod 32) leaves a
// 2-way conflict (adjacent rows overlap on 4 banks); it is the value that reproduces the
// design note's byte totals (99,072 @192 / 78,592 @128). Any pad must keep the row stride a
// 16B multiple because each row is a cp.reduce.async.bulk source.
constexpr int BW_DQH_PAD = 8;
template <int DQK> __host__ __device__ constexpr int bw_dqh_stride() { return DQK / 2 + BW_DQH_PAD; }
template <int DQK, int DVO, int BM> __host__ __device__ constexpr size_t bw_dkdv_base_smem() {
    return (size_t)(BW_BN * DQK + BW_BN * DVO + 2 * BM * DQK + 2 * BM * DVO) * sizeof(__nv_bfloat16)
         + (size_t)(2 * BM) * sizeof(float);
}
// CFG=1 needs sDS + the fp32 staging half-tile; CFG=2 and CFG=3 (register-direct red.v2) need
// sDS only (the CFG=3 end-of-kernel dK/dV merge stages through the DEAD Q/dO buffers: no extra smem).
template <int DQK, int BM, int kCfg> __host__ __device__ constexpr size_t bw_dq_phase_smem() {
    return (size_t)BM * BW_BN * sizeof(__nv_bfloat16)
         + (kCfg == 1 ? (size_t)BM * bw_dqh_stride<DQK>() * sizeof(float) : (size_t)0);
}

// vectorized synchronous global->shared load of [rows, dim] bf16, swizzled (dim mult 8).
// kThr = the calling CTA's thread count (default BW_THREADS = the 4-warp kernels; the CFG=3
// dKdV kernel passes its 256).
template <int kThr = BW_THREADS>
__device__ __forceinline__ void bg2s(__nv_bfloat16* dst, const __nv_bfloat16* src,
                                     int seq_start, int start, int rows, int num_heads,
                                     int head_idx, int dim) {
    const int tid = threadIdx.x;
    const int stride_token = num_heads * dim;
    const int vec = dim / 8;
    for (int i = tid; i < rows * vec; i += kThr) {
        int r = i / vec, c = (i % vec) * 8;
        const __nv_bfloat16* s = src + (seq_start + start + r) * stride_token + head_idx * dim + c;
        *reinterpret_cast<int4*>(dst + swz(r, c, dim)) = *reinterpret_cast<const int4*>(s);
    }
}
// cp.async.cg (16B granules, swizzled) global->shared of [rows, dim] bf16 strided over the
// calling CTA's kThr threads. kThr == mma_fwd::FM_THREADS (128) forwards to mma_fwd::g2s_cp
// itself (the CFG 0-2 call is unchanged after inlining); the 256-thread CFG=3 CTA runs the
// same loop with its own stride so every granule is issued exactly once.
template <int kThr>
__device__ __forceinline__ void g2s_cp_t(__nv_bfloat16* dst, const __nv_bfloat16* src,
                                         int seq_start, int start, int rows, int num_heads,
                                         int head_idx, int dim) {
    if constexpr (kThr == mma_fwd::FM_THREADS) {
        g2s_cp(dst, src, seq_start, start, rows, num_heads, head_idx, dim);
    } else {
        const int tid = threadIdx.x;
        const int stride_token = num_heads * dim;
        const int vec = dim / 8;
        for (int i = tid; i < rows * vec; i += kThr) {
            int r = i / vec, c = (i % vec) * 8;
            const __nv_bfloat16* s = src + (seq_start + start + r) * stride_token + head_idx * dim + c;
            mma_fwd::cp_cg16(cvta_shared(dst + swz(r, c, dim)), s);
        }
    }
}

// ============================================================================
// CFG=4 primitives (audit/design-bwd-cfg4-frost.md sect. 7/9; implementation note
// audit/impl-bwd-cfg4.md). Every spelling is ptxas-accepted on PLAIN sm_120
// (tests/_probe_warpspec_ptx.cu). The forward header carries the same mbarrier / zfill /
// x4.trans spellings; they are copied here because that header is read-only for the backward
// and no new symbol of it may be imported. Semantics: PTX ISA "Parallel Synchronization and
// Communication Instructions" (mbarrier.try_wait defaults to .acquire.cta, mbarrier.arrive to
// .release.cta; cp.async is a generic-proxy weak operation -> no proxy fence is needed between
// the ldmatrix reads of a stage and its cp.async refill).
// ============================================================================
__device__ __forceinline__ void bw_mbar_init(uint32_t a, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(a), "r"(count) : "memory");
}
__device__ __forceinline__ void bw_mbar_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void bw_mbar_arrive(uint32_t a) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" :: "r"(a) : "memory");
}
// Block until the phase with the given parity has completed (try_wait suspends the thread with a
// hardware time limit between polls).
__device__ __forceinline__ void bw_mbar_wait(uint32_t a, uint32_t parity) {
    uint32_t done;
    do {
        asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                     : "=r"(done) : "r"(a), "r"(parity) : "memory");
    } while (!done);
}
// 16-B cp.async.cg with an explicit source size: src_bytes == 0 reads nothing and ZERO-FILLS the
// 16 B (the LDGSTS zfill predicate); the source address must still be valid (callers clamp it).
__device__ __forceinline__ void bw_cp_cg16_zfill(uint32_t sa, const void* g, uint32_t src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sa), "l"(g), "r"(src_bytes) : "memory");
}
// 4-B cp.async.ca (the .ca form admits 4/8/16-B copies) with the same zero-fill, for lse/delta rows.
// CFG=5: 4-B shared store at a precomputed shared byte address (+ a compile-time immediate folded
// by ptxas into the STS address operand).
__device__ __forceinline__ void bw_sts_b32(uint32_t sa, uint32_t v) {
    asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(sa), "r"(v) : "memory");
}
__device__ __forceinline__ void bw_cp_ca4_zfill(uint32_t sa, const void* g, uint32_t src_bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(sa), "l"(g), "r"(src_bytes) : "memory");
}
// Arrive-on triggered by the hardware once ALL prior cp.async of this thread have landed. .noinc:
// the arrival counts against the mbarrier's init count (one per issuing thread -> count 256).
__device__ __forceinline__ void bw_cp_async_mbar_arrive_noinc(uint32_t a) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(a) : "memory");
}
__device__ __forceinline__ void bw_ldm_x4_trans(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
// mma_16x8x16 with the accumulator in place and the two B registers as scalars (the halves of
// an x4 / x4.trans fragment pair). Same instruction, same numerics as mma_16x8x16(d, a, b, d).
__device__ __forceinline__ void bw_mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ---- CFG=4 tiles and smem map (bytes; design sect. 5 + the two EMPTY mbarriers per stage) ----
//   192/128 (BM=32)                          128/128 (BM=64)
//   Ks    [64][192]        0   24,576        Ks    [64][128]        0   16,384
//   Vs    [64][128]   24,576   16,384 -> sP[0] @24,576, sdS[0] @28,672, sP[1] @32,768, sdS[1] @36,864
//                                            Vs    [64][128]   16,384   16,384 -> sP[0] @16,384, sdS[0] @24,576
//                                            sP[1] @32,768, sdS[1] @40,960 (dedicated)      16,384
//   Qbuf  [2][32][192] 40,960   24,576       Qbuf  [2][64][128] 49,152   32,768
//   dObuf [2][32][128] 65,536   16,384       dObuf [1][64][128] 81,920   16,384  (SINGLE stage)
//   lseS  [2][32] f32  81,920      256       lseS  [2][64]      98,304      512
//   deltaS[2][32] f32  82,176      256       deltaS[2][64]      98,816      512
//   mbar  full[2] empty[2]  82,432   32      mbar  full_q[2] empty_q[2] full_do empty_do 99,328  48
//   TOTAL                        82,464      TOTAL                                        99,376
// Slabs are bf16 [BM][64] at swz(r, c, 64): the C-fragment bf162 stores and the x4.trans / x4
// reads are both conflict-free. Slab stage 0 aliases the dead Vs (V is in registers after the
// prologue); the 2-stage slabs close their WAR through the next tile's B1 (no second barrier).
template <int DQK, int DVO> struct Bw4 {
    static constexpr int BM  = (DQK == 128) ? 64 : 32;         // q rows per tile (FROST's (128,64,64) / (192,32,64))
    static constexpr int NDO = (DQK == 128) ? 1 : 2;           // dO stages (single at d128: smem)
    static constexpr int kKsB     = BW_BN * DQK * 2;
    static constexpr int kVsB     = BW_BN * DVO * 2;
    static constexpr int kSlabB   = BM * BW_BN * 2;            // one bf16 [BM][64] slab
    static constexpr int kSlabsB  = (4 * kSlabB > kVsB) ? 4 * kSlabB : kVsB;   // sP[2] + sdS[2] (stage 0 over Vs)
    static constexpr int kSlabOff = kKsB;
    static constexpr int kQOff    = kKsB + kSlabsB;
    static constexpr int kQB      = 2 * BM * DQK * 2;
    static constexpr int kDoOff   = kQOff + kQB;
    static constexpr int kDoB     = NDO * BM * DVO * 2;
    static constexpr int kLseOff  = kDoOff + kDoB;             // fp32 [2][BM]
    static constexpr int kDeltaOff = kLseOff + 2 * BM * 4;     // fp32 [2][BM]
    static constexpr int kMbarOff = kDeltaOff + 2 * BM * 4;    // 8-B mbarriers: full[2] @+0, empty[2] @+16, (@128) full_do @+32, empty_do @+40
    static constexpr int kMbarB   = (NDO == 1) ? 48 : 32;
    static constexpr int kSmemBytes = kMbarOff + kMbarB;
    static_assert(kSmemBytes <= 101376, "CFG=4 dKdV smem exceeds the sm120 99KB dynamic-smem opt-in maximum");
    static_assert(kSlabOff % 16 == 0 && kQOff % 16 == 0 && kDoOff % 16 == 0 && kLseOff % 16 == 0 && kMbarOff % 8 == 0,
                  "CFG=4 smem regions must keep 16-B (cp.async / ldmatrix) and 8-B (mbarrier) alignment");
    static_assert(2 * kSlabB <= kVsB, "CFG=4 slab stage 0 must fit inside the dead Vs region");
};
template <int DQK, int DVO> __host__ __device__ constexpr size_t bw4_smem() { return (size_t)Bw4<DQK, DVO>::kSmemBytes; }

// ============================================================================
// CFG=4 kernel body (FROST B-structure on the CFG=2 math; audit/design-bwd-cfg4-frost.md
// sect. 3-7, audit/impl-bwd-cfg4.md). CTA per KV block (64 keys), 8 math warps = 256 threads,
// tile = BM queries (32 @192 / 64 @128). Warp w:
//   S/dP  (2,4): sm = w&1 -> q rows sm*BM/2..+BM/2 ; sn = w>>1 -> keys sn*16..+16 (2 n8 frags);
//                V is register-resident (Vr = this warp's 16 keys x DVO as dP B fragments).
//   dK/dV (2,4): km = w&1 -> keys km*32..+32 (2 m16 tiles) ; kn = w>>1 -> dK cols kn*DQK/4..+DQK/4,
//                dV cols kn*32..+32. A = P^T / dS^T by ldmatrix.x4.trans from the slabs, B = dO^T /
//                Q^T by x4.trans pairs.
//   dQ    @192 (1,8): all 32 rows, d cols w*24..+24 ; @128 (2,4): rows (w&1)*32..+32, cols (w>>1)*32..+32.
// Per tile n (stage s = n&1): [L1] issue tile n+1 one tile ahead (all threads, gated on the
// EMPTY event of tile n-1) -> [W1] wait full(n) -> S -> [W2 @128] wait full_do -> dP -> P/dS ->
// slab stores -> [B1] __syncthreads (slab RAW; the only blocking barrier) -> dV -> [@128: dO
// consumed: elected arrive empty_do, wait, issue dO(n+1)] -> dK -> [elected arrive empty(s)] ->
// dQ -> red.v2 into dq_acc. dK/dV summation order == CFG=2's (ascending 16-q k-tiles inside a
// tile, ascending tiles), one warp per element -> bitwise identical to CFG=2.
// ============================================================================
// kAddr == 1 (CFG=5): the same tile with HOISTED swizzled addressing (see the kAddr arm below).
// kDiagNoRed (diag kCfg 95 only, WRONG dQ): the CFG=5 tile with the dQ red.v2 reductions removed --
// isolates the L2 reduction traffic (8 separate 32-B rows per warp instruction) as a wall candidate.
// Measured: removing the reds = +22-31% (b=8 H=32 S=4096 216 -> 282 TF) -> the reductions ARE the wall.
// kCoal (CFG=6): FROST's FRAGMENT-ORDER dQ workspace. Every red.v2 of a warp writes 256 CONTIGUOUS
// bytes (64-float chunk c = ((warp*2 + m-tile)*QNF + n-frag)*2 + row-half, lane*2 inside) instead of
// 8 separate 32-B row segments; blocks are laid out per (batch, q-tile, head) with every batch padded
// to whole BM-row tiles (rbase6 = the prefix over earlier batches of ceil(sq/BM)*BM rows, computed
// from cu_q), zeroed by the launcher, and un-scrambled into the dense bf16 dq by
// fmha_bwd_mma_dq_unscramble_kernel (same chunk formula). dQ values / red order are CFG=5's.
// kSdp (CFG=7): the S and dP GEMMs interleaved over their shared d range (4 accumulator chains).
// kDiagTiming (diag kCfg 94, env FLASH_MLA_SM120_BWD_MMA_DIAG_TIMING=1): per-phase clock64 attribution of
// the CFG=6 tile (lane 0 of every warp accumulates its own cycles per phase and atomically adds them into
// bw_diag_cycles at kernel end; the launcher prints the breakdown). Correct results, perturbed timing.
__device__ unsigned long long bw_diag_cycles[16];
enum BwDiagPhase { BWD_T_ISSUE = 0, BWD_T_WAITQ, BWD_T_SDP, BWD_T_VJP, BWD_T_B1, BWD_T_DV, BWD_T_DO, BWD_T_DK, BWD_T_DQ, BWD_T_RED, BWD_T_NPHASE, BWD_T_TILES = 12, BWD_T_WARPS = 13 };
// kOpt8 (CFG=8, from the diag-94 attribution): (a) Q(n+1) issued right after B1(n) -- every warp has finished
// dK(n-1) there, so the stage-WAR wait is free instead of a skew wait at the tile top; (b) red.v4 fragment
// order: a lane's 2 floats of n-frags 2j and 2j+1 are adjacent -> one 16-B reduction per pair (8-12 LSU
// instructions per lane per tile instead of 16); (c) dV/dK/dQ: every fragment load of a k-step issued
// back-to-back before its mma group (one exposed ldmatrix latency per k-step, not per n-frag pair).
template <int DQK, int DVO, bool kCausal, int kAddr = 0, bool kDiagNoRed = false, bool kCoal = false, bool kSdp = false, bool kDiagTiming = false, bool kOpt8 = false>
__device__ __forceinline__ void fmha_bwd_mma_dkdv_cfg4_body(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ d_o,
    const float* __restrict__ lse, const float* __restrict__ delta,
    const int* __restrict__ cu_q, const int* __restrict__ cu_kv,
    __nv_bfloat16* __restrict__ dk, __nv_bfloat16* __restrict__ dv,
    int num_heads, float scale, float* __restrict__ dq_acc) {
    using C = Bw4<DQK, DVO>;
    constexpr int BM  = C::BM;                  // q rows per tile: 32 @192 / 64 @128
    constexpr int RB  = BM / 2;                 // S/dP q rows per warp (2,4): 16 / 32
    constexpr int RM  = RB / BM_M;              // m16 tiles per warp in S/dP: 1 / 2
    constexpr int MK  = BM / BM_K;              // q k16 tiles per tile (dV/dK contraction): 2 / 4
    constexpr int DKQ = DQK / BM_K;             // S contraction d-tiles: 12 / 8
    constexpr int DKV = DVO / BM_K;             // dP contraction d-tiles: 8
    constexpr int NFK = DQK / 4 / BM_N;         // dK n8 frags per warp (d quarter): 6 / 4
    constexpr int NFV = DVO / 4 / BM_N;         // dV n8 frags per warp: 4
    constexpr int KKN = BW_BN / BM_K;           // key k16 tiles (dQ contraction): 4
    constexpr int DQW = (DQK == 128) ? 32 : 24; // dQ d columns per warp: (2,4) @128 / (1,8) @192
    constexpr int QNF = DQW / BM_N;             // dQ n8 frags per warp: 4 / 3
    constexpr int kThreads = BW4_THREADS;
    constexpr int kWarps = kThreads / 32;
    static_assert(DVO == 128 && (DQK == 128 || DQK == 192), "CFG=4 partitions are laid out for 128/128 and 192/128");
    static_assert(RM * BM_M == RB && MK * BM_K == BM && NFK % 2 == 0 && NFV % 2 == 0, "CFG=4 fragment pairing");
    static_assert(QNF * BM_N * ((DQK == 128) ? 4 : 8) == DQK, "CFG=4 dQ warp columns must tile DQK");
    static_assert((BM * (DQK / 8)) % kThreads == 0 && (BM * (DVO / 8)) % kThreads == 0,
                  "CFG=4 striped loader: every thread issues the same number of 16-B granules");
    static_assert(2 * BM <= kThreads, "CFG=4 loader: lse/delta rows are issued one per thread");

    const int b = blockIdx.z, h = blockIdx.y, nb = blockIdx.x;
    if (h >= num_heads) return;
    const int q0 = cu_q[b], q1 = cu_q[b + 1], kv0 = cu_kv[b], kv1 = cu_kv[b + 1];
    const int sq = q1 - q0, skv = kv1 - kv0;
    const int n_start = nb * BW_BN;
    if (n_start >= skv) return;
    const int n_size = min(BW_BN, skv - n_start);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    // CFG=6: padded row base of this batch in the fragment-order dQ workspace (see kCoal above).
    [[maybe_unused]] int rbase6 = 0;
    if constexpr (kCoal) {
        for (int bb = 0; bb < b; ++bb) rbase6 += ((cu_q[bb + 1] - cu_q[bb]) + BM - 1) / BM * BM;
    }
    const int sm = warp & 1, sn = warp >> 1;            // S/dP: q half x 16-key quarter
    const int km = warp & 1, kn = warp >> 1;            // dK/dV: 32-key half x d quarter
    const int qm = (DQK == 128) ? (warp & 1) : 0;       // dQ: 32-row half (@128) / all rows (@192)
    const int qn = (DQK == 128) ? (warp >> 1) : warp;   // dQ: d slice
    const int sk_tok = num_heads * DQK, sv_tok = num_heads * DVO;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* Ks    = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* Vs    = reinterpret_cast<__nv_bfloat16*>(smem_raw + C::kSlabOff);   // prologue only; then sP[0]/sdS[0]/sP[1]/sdS[1]
    __nv_bfloat16* Qbuf  = reinterpret_cast<__nv_bfloat16*>(smem_raw + C::kQOff);      // [2][BM][DQK]
    __nv_bfloat16* dObuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + C::kDoOff);     // [NDO][BM][DVO]
    float* lseS   = reinterpret_cast<float*>(smem_raw + C::kLseOff);                   // [2][BM]
    float* deltaS = reinterpret_cast<float*>(smem_raw + C::kDeltaOff);                 // [2][BM]
    const uint32_t mbar = cvta_shared(smem_raw + C::kMbarOff);
    auto full_q  = [&](int s) { return mbar + 8 * s; };          // stage s landed (Q + lse + delta; + dO @192): 256 noinc arrivals
    auto empty_q = [&](int s) { return mbar + 16 + 8 * s; };     // stage s consumed: 8 elected arrivals (after dK; dV precedes it)
    [[maybe_unused]] const uint32_t full_do = mbar + 32, empty_do = mbar + 40;   // @128 only: the single dO stage

    // ---- prologue ---------------------------------------------------------------------
    if (tid == 0) {
        bw_mbar_init(full_q(0), kThreads);  bw_mbar_init(full_q(1), kThreads);
        bw_mbar_init(empty_q(0), kWarps);   bw_mbar_init(empty_q(1), kWarps);
        if constexpr (C::NDO == 1) { bw_mbar_init(full_do, kThreads); bw_mbar_init(empty_do, kWarps); }
        bw_mbar_fence_init();
    }
    __syncthreads();   // [P0] mbarriers initialised and visible before any arrive-on references them

    int mb0 = 0;
    // Exact causal start: the first Q tile with an unmasked (query, key) pair for keys >= n_start
    // is n_start / BM (BM | n_start). CFG<=3's (n_start - (BM-1)) / BM visits one fully-masked
    // tile first; its P = dS = 0 mma are exact no-ops on the +0-initialised accumulators (design
    // sect. 7), so skipping it keeps dK/dV bitwise identical.
    // Causal alignment is BOTTOM-RIGHT (FlashAttention >= 2.1 convention; identical to the
    // forward's, which writes LSE = -inf for the queries that see no key): query row r sees key g
    // iff g <= r + (skv - sq). The first row that sees key n_start is n_start - coff.
    const int coff = skv - sq;
    if (kCausal) mb0 = max(0, n_start - coff) / BM;
    const int nq = (sq + BM - 1) / BM;
    const int nt = nq - mb0;   // tiles to visit; <= 0 -> zero-trip (dK/dV = 0 stored below)

    // Striped loaders: every thread issues 1/256 of a tile with 16-B cp.async (tail rows
    // r >= msz hardware-zero-filled, source clamped to a valid row) and, for tid < 2*BM, one 4-B
    // lse / delta row; then ONE noinc arrive on the stage's full barrier (fires when all of this
    // thread's prior cp.async have landed). Loops stay rolled (an unrolled loader hoists its
    // address invariants across the mainloop and eats registers, fwd rung-8 lesson).
    auto issue_q = [&](int t) {   // Q (+ dO @192) + lse + delta of tile t -> stage t & 1
        const int s = t & 1, m0 = (mb0 + t) * BM, msz = min(BM, sq - m0);
        __nv_bfloat16* Qst = Qbuf + s * (BM * DQK);
        constexpr int VQ = DQK / 8;
        #pragma unroll 1
        for (int j = 0; j < (BM * VQ) / kThreads; ++j) {
            const int i = tid + j * kThreads;
            const int r = i / VQ, c = (i % VQ) * 8;
            const int rs = (r < msz) ? r : (msz - 1);
            const __nv_bfloat16* src = q + (int64_t)(q0 + m0 + rs) * sk_tok + h * DQK + c;
            bw_cp_cg16_zfill(cvta_shared(Qst + swz(r, c, DQK)), src, (r < msz) ? 16u : 0u);
        }
        if constexpr (C::NDO == 2) {
            __nv_bfloat16* dOst = dObuf + s * (BM * DVO);
            constexpr int VO = DVO / 8;
            #pragma unroll 1
            for (int j = 0; j < (BM * VO) / kThreads; ++j) {
                const int i = tid + j * kThreads;
                const int r = i / VO, c = (i % VO) * 8;
                const int rs = (r < msz) ? r : (msz - 1);
                const __nv_bfloat16* src = d_o + (int64_t)(q0 + m0 + rs) * sv_tok + h * DVO + c;
                bw_cp_cg16_zfill(cvta_shared(dOst + swz(r, c, DVO)), src, (r < msz) ? 16u : 0u);
            }
        }
        if (tid < BM) {
            const int r = tid, rs = (r < msz) ? r : (msz - 1);
            bw_cp_ca4_zfill(cvta_shared(lseS + s * BM + r), lse + (int64_t)(q0 + m0 + rs) * num_heads + h, (r < msz) ? 4u : 0u);
        } else if (tid < 2 * BM) {
            const int r = tid - BM, rs = (r < msz) ? r : (msz - 1);
            bw_cp_ca4_zfill(cvta_shared(deltaS + s * BM + r), delta + (int64_t)(q0 + m0 + rs) * num_heads + h, (r < msz) ? 4u : 0u);
        }
        bw_cp_async_mbar_arrive_noinc(full_q(s));
    };
    [[maybe_unused]] auto issue_do = [&](int t) {   // @128: dO of tile t -> the single dO stage
        const int m0 = (mb0 + t) * BM, msz = min(BM, sq - m0);
        constexpr int VO = DVO / 8;
        #pragma unroll 1
        for (int j = 0; j < (BM * VO) / kThreads; ++j) {
            const int i = tid + j * kThreads;
            const int r = i / VO, c = (i % VO) * 8;
            const int rs = (r < msz) ? r : (msz - 1);
            const __nv_bfloat16* src = d_o + (int64_t)(q0 + m0 + rs) * sv_tok + h * DVO + c;
            bw_cp_cg16_zfill(cvta_shared(dObuf + swz(r, c, DVO)), src, (r < msz) ? 16u : 0u);
        }
        bw_cp_async_mbar_arrive_noinc(full_do);
    };

    // tile 0 -> stage 0 (both stages virgin: no empty wait); it lands while K/V are staged below
    if (nt > 0) { issue_q(0); if constexpr (C::NDO == 1) issue_do(0); }
    bg2s<kThreads>(Ks, k, kv0, n_start, n_size, num_heads, h, DQK);
    bg2s<kThreads>(Vs, v, kv0, n_start, n_size, num_heads, h, DVO);
    for (int i = tid; i < (BW_BN - n_size) * DQK; i += kThreads) {
        int r = n_size + i / DQK, c = i % DQK; Ks[swz(r, c, DQK)] = __float2bfloat16(0.f);
    }
    for (int i = tid; i < (BW_BN - n_size) * DVO; i += kThreads) {
        int r = n_size + i / DVO, c = i % DVO; Vs[swz(r, c, DVO)] = __float2bfloat16(0.f);
    }
    __syncthreads();   // [P1] Ks / Vs staged (incl. the zero-filled key tail)
    // V register-resident as the dP B fragments of this warp's 16 keys: Vr[d][0..1] = n-frag
    // keys sn*16 + 0..7, Vr[d][2..3] = keys sn*16 + 8..15 (the K x4 n-pair address formula);
    // the same bf16 values CFG<=3's x2 loads feed (b0 = d 0-7, b1 = d 8-15 of the d-tile).
    uint32_t Vr[DKV][4];
    #pragma unroll
    for (int d = 0; d < DKV; ++d) {
        const int kr = sn * 16 + (lane / 16) * 8 + (lane % 8), kc = d * BM_K + ((lane / 8) % 2) * 8;
        ldm_x4(Vr[d], cvta_shared(Vs + swz(kr, kc, DVO)));
    }
    __syncthreads();   // [P2] every warp's Vr loads done: Vs is dead, slab stage 0 may overwrite it

    float dKr[2][NFK][4], dVr[2][NFV][4];
    #pragma unroll
    for (int mb = 0; mb < 2; ++mb) {
        #pragma unroll
        for (int i = 0; i < NFK; ++i) { dKr[mb][i][0]=dKr[mb][i][1]=dKr[mb][i][2]=dKr[mb][i][3]=0.f; }
        #pragma unroll
        for (int i = 0; i < NFV; ++i) { dVr[mb][i][0]=dVr[mb][i][1]=dVr[mb][i][2]=dVr[mb][i][3]=0.f; }
    }

    if constexpr (kAddr == 0) {
    // ---- mainloop over Q tiles -------------------------------------------------------
    for (int n = 0; n < nt; ++n) {
        const int s = n & 1;
        const int m_start = (mb0 + n) * BM, m_size = min(BM, sq - m_start);
        const __nv_bfloat16* Qs  = Qbuf + s * (BM * DQK);
        const __nv_bfloat16* dOs = dObuf + ((C::NDO == 2) ? s : 0) * (BM * DVO);
        const float* lseSs = lseS + s * BM;
        const float* deltaSs = deltaS + s * BM;
        __nv_bfloat16* sPs  = Vs + (2 * s) * (BM * BW_BN);       // slab stage s
        __nv_bfloat16* sdSs = sPs + BM * BW_BN;

        // [L1] tile n+1 -> stage s^1, ONE TILE AHEAD. WAR: stage s^1 held tile n-1, whose last
        // reader (dK(n-1), after dV(n-1)) completed phase (n-1)>>1 of empty(s^1) -> wait parity
        // ((n-1)>>1)&1; a warp blocks only if it is more than dQ(n-1)+reds ahead of the slowest.
        // n == 0: stage 1 is virgin, no wait.
        if (n + 1 < nt) {
            if (n >= 1) bw_mbar_wait(empty_q(s ^ 1), ((n - 1) >> 1) & 1);
            issue_q(n + 1);
        }
        // [W1] Q (+ dO @192) + lse + delta of tile n landed: the (n>>1)-th phase of full(s).
        bw_mbar_wait(full_q(s), (n >> 1) & 1);

        // [G1] S = scale*Q@K^T (contract DQK): this warp's RB rows x 16 keys; d ascending as CFG<=3.
        float Sr[RM][2][4];
        #pragma unroll
        for (int mm = 0; mm < RM; ++mm) { Sr[mm][0][0]=Sr[mm][0][1]=Sr[mm][0][2]=Sr[mm][0][3]=0.f; Sr[mm][1][0]=Sr[mm][1][1]=Sr[mm][1][2]=Sr[mm][1][3]=0.f; }
        #pragma unroll
        for (int d = 0; d < DKQ; ++d) {
            uint32_t Kd[4];
            { const int kr = sn * 16 + (lane / 16) * 8 + (lane % 8), kc = d * BM_K + ((lane / 8) % 2) * 8; ldm_x4(Kd, cvta_shared(Ks + swz(kr, kc, DQK))); }
            #pragma unroll
            for (int mm = 0; mm < RM; ++mm) {
                uint32_t Qd[4];
                { const int row = sm * RB + mm * BM_M + (lane % 16), col = d * BM_K + (lane / 16) * 8; ldm_x4(Qd, cvta_shared(Qs + swz(row, col, DQK))); }
                bw_mma(Sr[mm][0], Qd, Kd[0], Kd[1]);
                bw_mma(Sr[mm][1], Qd, Kd[2], Kd[3]);
            }
        }
        // [W2] (@128) dO of tile n landed (phase n of the single-stage barrier).
        if constexpr (C::NDO == 1) bw_mbar_wait(full_do, n & 1);
        // [G2] dP = dO@V^T (contract DVO) with the register-resident V fragments.
        float dPr[RM][2][4];
        #pragma unroll
        for (int mm = 0; mm < RM; ++mm) { dPr[mm][0][0]=dPr[mm][0][1]=dPr[mm][0][2]=dPr[mm][0][3]=0.f; dPr[mm][1][0]=dPr[mm][1][1]=dPr[mm][1][2]=dPr[mm][1][3]=0.f; }
        #pragma unroll
        for (int d = 0; d < DKV; ++d) {
            #pragma unroll
            for (int mm = 0; mm < RM; ++mm) {
                uint32_t dOd[4];
                { const int row = sm * RB + mm * BM_M + (lane % 16), col = d * BM_K + (lane / 16) * 8; ldm_x4(dOd, cvta_shared(dOs + swz(row, col, DVO))); }
                bw_mma(dPr[mm][0], dOd, Vr[d][0], Vr[d][1]);
                bw_mma(dPr[mm][1], dOd, Vr[d][2], Vr[d][3]);
            }
        }

        // [P] P = exp(scale*S - LSE); dS = P*(dP - delta)*scale -- CFG<=3's expressions and guards
        // (rows >= m_size, keys >= n_size, causal). Stored ONCE as bf162 into the swizzled slabs
        // (the same __floats2bfloat162_rn rounding as packbf), no movmatrix.
        #pragma unroll
        for (int mm = 0; mm < RM; ++mm) {
            const int ra = sm * RB + mm * BM_M + (lane / 4), rb = ra + 8;
            const float lsea = (ra < m_size) ? lseSs[ra] : INFINITY, lseb = (rb < m_size) ? lseSs[rb] : INFINITY;
            const float da = (ra < m_size) ? deltaSs[ra] : 0.f, db = (rb < m_size) ? deltaSs[rb] : 0.f;
            #pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
                const int c0 = sn * 16 + nn * BM_N + (lane % 4) * 2, gca = c0, gcb = c0 + 1;
                float p0=__expf(scale*Sr[mm][nn][0]-lsea), p1=__expf(scale*Sr[mm][nn][1]-lsea);
                float p2=__expf(scale*Sr[mm][nn][2]-lseb), p3=__expf(scale*Sr[mm][nn][3]-lseb);
                bool va0=(gca<n_size)&&(!kCausal||(n_start+gca)<=(m_start+ra+coff)), va1=(gcb<n_size)&&(!kCausal||(n_start+gcb)<=(m_start+ra+coff));
                bool vb0=(gca<n_size)&&(!kCausal||(n_start+gca)<=(m_start+rb+coff)), vb1=(gcb<n_size)&&(!kCausal||(n_start+gcb)<=(m_start+rb+coff));
                p0=(va0&&ra<m_size)?p0:0.f; p1=(va1&&ra<m_size)?p1:0.f; p2=(vb0&&rb<m_size)?p2:0.f; p3=(vb1&&rb<m_size)?p3:0.f;
                const float s0=p0*(dPr[mm][nn][0]-da)*scale, s1=p1*(dPr[mm][nn][1]-da)*scale;
                const float s2=p2*(dPr[mm][nn][2]-db)*scale, s3=p3*(dPr[mm][nn][3]-db)*scale;
                *reinterpret_cast<__nv_bfloat162*>(sPs  + swz(ra, c0, BW_BN)) = __floats2bfloat162_rn(p0, p1);
                *reinterpret_cast<__nv_bfloat162*>(sPs  + swz(rb, c0, BW_BN)) = __floats2bfloat162_rn(p2, p3);
                *reinterpret_cast<__nv_bfloat162*>(sdSs + swz(ra, c0, BW_BN)) = __floats2bfloat162_rn(s0, s1);
                *reinterpret_cast<__nv_bfloat162*>(sdSs + swz(rb, c0, BW_BN)) = __floats2bfloat162_rn(s2, s3);
            }
        }
        // [B1] slab RAW across the 8 warps -- the only blocking barrier of the tile. It also closes
        // the 2-stage slab WAR (a warp stores into stage s at tile n+2 only after passing B1(n+1),
        // which every warp reaches after finishing tile n).
        __syncthreads();

        // [G3] dV += P^T @ dO over the tile's queries, k-tiles kk ascending (CFG<=3's order).
        // A = P^T: x4.trans of the slab block (q kk*16..+16, keys km*32 + mb*16..+16) -> matrices
        // {a0, a2, a1, a3} (matrix 1 of the .trans x4 is A rows 0-7 / k 8-15 = a2) -> reordered.
        // B = dO^T: x4.trans (q kk*16..+16, d kn*32 + ddp*16..+16) -> r0,r1 = n-frag 2ddp, r2,r3 = 2ddp+1.
        #pragma unroll
        for (int kk = 0; kk < MK; ++kk) {
            uint32_t A[2][4];
            #pragma unroll
            for (int mb = 0; mb < 2; ++mb) {
                uint32_t r[4];
                const int row = kk * BM_K + (lane % 16), col = km * 32 + mb * BM_M + (lane / 16) * 8;
                bw_ldm_x4_trans(r, cvta_shared(sPs + swz(row, col, BW_BN)));
                A[mb][0] = r[0]; A[mb][1] = r[2]; A[mb][2] = r[1]; A[mb][3] = r[3];
            }
            #pragma unroll
            for (int ddp = 0; ddp < NFV / 2; ++ddp) {
                uint32_t B[4];
                const int row = kk * BM_K + (lane % 16), col = kn * (DVO / 4) + ddp * 16 + (lane / 16) * 8;
                bw_ldm_x4_trans(B, cvta_shared(dOs + swz(row, col, DVO)));
                #pragma unroll
                for (int mb = 0; mb < 2; ++mb) {
                    bw_mma(dVr[mb][2 * ddp],     A[mb], B[0], B[1]);
                    bw_mma(dVr[mb][2 * ddp + 1], A[mb], B[2], B[3]);
                }
            }
        }
        if constexpr (C::NDO == 1) {
            // (@128) dO consumed by this warp (dV was its last reader) -> elected arrive; the
            // single dO stage is refilled with tile n+1 as soon as all 8 warps arrived (phase n),
            // landing during dK(n) + dQ(n) + reds + S(n+1) (FROST's d128 window).
            __syncwarp();
            if (lane == 0) bw_mbar_arrive(empty_do);
            if (n + 1 < nt) { bw_mbar_wait(empty_do, n & 1); issue_do(n + 1); }
        }
        // [G4] dK += dS^T @ Q, same structure; B = Q^T x4.trans pairs over the d quarter kn.
        #pragma unroll
        for (int kk = 0; kk < MK; ++kk) {
            uint32_t A[2][4];
            #pragma unroll
            for (int mb = 0; mb < 2; ++mb) {
                uint32_t r[4];
                const int row = kk * BM_K + (lane % 16), col = km * 32 + mb * BM_M + (lane / 16) * 8;
                bw_ldm_x4_trans(r, cvta_shared(sdSs + swz(row, col, BW_BN)));
                A[mb][0] = r[0]; A[mb][1] = r[2]; A[mb][2] = r[1]; A[mb][3] = r[3];
            }
            #pragma unroll
            for (int ddp = 0; ddp < NFK / 2; ++ddp) {
                uint32_t B[4];
                const int row = kk * BM_K + (lane % 16), col = kn * (DQK / 4) + ddp * 16 + (lane / 16) * 8;
                bw_ldm_x4_trans(B, cvta_shared(Qs + swz(row, col, DQK)));
                #pragma unroll
                for (int mb = 0; mb < 2; ++mb) {
                    bw_mma(dKr[mb][2 * ddp],     A[mb], B[0], B[1]);
                    bw_mma(dKr[mb][2 * ddp + 1], A[mb], B[2], B[3]);
                }
            }
        }
        // Stage s consumed by this warp (Q: dK above; dO @192: dV above) -> elected arrive on
        // empty(s); non-blocking. Its waiter is [L1] of tile n+1 (issue of tile n+2 into stage s).
        __syncwarp();
        if (lane == 0) bw_mbar_arrive(empty_q(s));

        // [G5] dQ = dS @ K over the CTA's 64 resident keys (kk ascending per accumulator = CFG=2's
        // per-(row, kv-block) partial). A = dS x4 from the slab (rows qm*32 + mb*16..), B = K^T
        // x4.trans pairs (+ one x2.trans for the odd third frag @192) from the resident Ks.
        float dQp[2][QNF][4];
        #pragma unroll
        for (int mb = 0; mb < 2; ++mb) {
            #pragma unroll
            for (int i = 0; i < QNF; ++i) { dQp[mb][i][0]=dQp[mb][i][1]=dQp[mb][i][2]=dQp[mb][i][3]=0.f; }
        }
        #pragma unroll
        for (int kk = 0; kk < KKN; ++kk) {
            uint32_t A[2][4];
            #pragma unroll
            for (int mb = 0; mb < 2; ++mb) {
                const int row = qm * 32 + mb * BM_M + (lane % 16), col = kk * BM_K + (lane / 16) * 8;
                ldm_x4(A[mb], cvta_shared(sdSs + swz(row, col, BW_BN)));
            }
            #pragma unroll
            for (int j = 0; j < QNF / 2; ++j) {
                uint32_t B[4];
                const int kr = kk * BM_K + (lane % 16), kc = qn * DQW + j * 16 + (lane / 16) * 8;
                bw_ldm_x4_trans(B, cvta_shared(Ks + swz(kr, kc, DQK)));
                #pragma unroll
                for (int mb = 0; mb < 2; ++mb) {
                    bw_mma(dQp[mb][2 * j],     A[mb], B[0], B[1]);
                    bw_mma(dQp[mb][2 * j + 1], A[mb], B[2], B[3]);
                }
            }
            if constexpr (QNF % 2 == 1) {
                uint32_t B2[2];
                const int kr = kk * BM_K + (lane % 16), kc = qn * DQW + (QNF - 1) * BM_N + (lane / 16) * 8;
                ldm_x2_trans(B2, cvta_shared(Ks + swz(kr, kc, DQK)));
                #pragma unroll
                for (int mb = 0; mb < 2; ++mb) bw_mma(dQp[mb][QNF - 1], A[mb], B2[0], B2[1]);
            }
        }
        // [R] CFG=2's register-direct, fire-and-forget red.global.add.v2.f32 into dq_acc (true
        // [token][head][d] layout). Rows >= m_size belong to the next varlen sequence: skipped.
        // Completion is guaranteed at kernel exit (stream order before the convert kernel).
        #pragma unroll
        for (int mb = 0; mb < 2; ++mb) {
            const int ra = qm * 32 + mb * BM_M + (lane / 4), rb = ra + 8;
            float* ga = dq_acc + ((int64_t)(q0 + m_start + ra) * num_heads + h) * DQK + qn * DQW + (lane % 4) * 2;
            float* gb = dq_acc + ((int64_t)(q0 + m_start + rb) * num_heads + h) * DQK + qn * DQW + (lane % 4) * 2;
            #pragma unroll
            for (int nf = 0; nf < QNF; ++nf) {
                if (ra < m_size) red_add_v2f32(ga + nf * BM_N, dQp[mb][nf][0], dQp[mb][nf][1]);
                if (rb < m_size) red_add_v2f32(gb + nf * BM_N, dQp[mb][nf][2], dQp[mb][nf][3]);
            }
        }
    }

    } else {
    // ================= CFG=5: the CFG=4 tile with HOISTED swizzled addressing =================
    // SASS attribution of CFG=4 (audit/SM90_SM100_COVERAGE.md "SASS ATTRIBUTION"): 947 integer /
    // address instructions per 104 HMMA at 192/128 = 111% of the issue budget at 2 warps/SMSP --
    // the kernel is ISSUE-bound, and the instructions are swz(row, col, DIM) recomputed at every
    // ldmatrix / slab store. For a fixed lane every operand's row (hence its swizzle key row & 7)
    // is invariant along the tile's compile-time steps (8/16-row steps keep row & 7), and along a
    // column the XOR only permutes the 16-B granule inside its 128-B span:
    //     swz(r, 64a + x, DIM) == swz(r, x, DIM) + 64a   for 0 <= x < 64.
    // So: one per-lane base register per distinct x (column mod 64) and a compile-time immediate
    // for everything else (64-element column blocks, row steps, the P/dS slab, the STAGE -- the
    // Q-tile loop is unrolled by two so the stage index is a compile-time constant). ptxas folds
    // base + immediate into the LDSM / STS address operand: zero address arithmetic per use.
    // The mma / softmax-VJP instruction order is CFG=4's -> bitwise the same dK/dV/dQ partials.
    auto swzb = [&](uint32_t sbase, int r, int c, int dim) -> uint32_t {   // shared byte address of element (r, c)
        return sbase + 2u * static_cast<uint32_t>(swz(r, c, dim));
    };
    const uint32_t sK5 = cvta_shared(Ks), sQ5 = cvta_shared(Qbuf), sO5 = cvta_shared(dObuf), sL5 = cvta_shared(Vs);
    const int kr5 = sn * 16 + (lane / 16) * 8 + (lane % 8), kcl5 = ((lane / 8) % 2) * 8;   // S: K x4 rows / col lane part
    const int rq5 = sm * RB + (lane % 16), cl5 = (lane / 16) * 8;                          // S/dP: Q, dO x4 rows / col lane part
    const int ra5 = sm * RB + (lane / 4);                                                  // P/dS slab store rows (+8, +16 mm)
    uint32_t aK5[4], aQ5[4], aO5[4], aP5[2], aA5[2], aDA5[KKN], aVB5[NFV / 2], aKB5[NFK / 2], aDB5[(QNF + 1) / 2];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        aK5[j] = swzb(sK5, kr5, 16 * j + kcl5, DQK);     // + 128 B per 64-column block (d >> 2)
        aQ5[j] = swzb(sQ5, rq5, 16 * j + cl5, DQK);      // + stage, + mm * 16 rows, + 128 B per block
        aO5[j] = swzb(sO5, rq5, 16 * j + cl5, DVO);
    }
    #pragma unroll
    for (int nn = 0; nn < 2; ++nn) aP5[nn] = swzb(sL5, ra5, sn * 16 + nn * BM_N + (lane % 4) * 2, BW_BN);   // + slab stage, + dS, + 8/16 rows
    #pragma unroll
    for (int mb = 0; mb < 2; ++mb) aA5[mb] = swzb(sL5, lane % 16, km * 32 + mb * BM_M + cl5, BW_BN);      // P^T / dS^T x4.trans, + 16 kk rows
    #pragma unroll
    for (int kk = 0; kk < KKN; ++kk) aDA5[kk] = swzb(sL5, qm * 32 + (lane % 16), kk * BM_K + cl5, BW_BN); // dS x4 for dQ, + 16 mb rows
    #pragma unroll
    for (int ddp = 0; ddp < NFV / 2; ++ddp) aVB5[ddp] = swzb(sO5, lane % 16, kn * (DVO / 4) + ddp * 16 + cl5, DVO);   // dO^T x4.trans, + 16 kk rows
    #pragma unroll
    for (int ddp = 0; ddp < NFK / 2; ++ddp) aKB5[ddp] = swzb(sQ5, lane % 16, kn * (DQK / 4) + ddp * 16 + cl5, DQK);   // Q^T x4.trans
    #pragma unroll
    for (int j = 0; j < (QNF + 1) / 2; ++j) aDB5[j] = swzb(sK5, lane % 16, qn * DQW + j * 16 + cl5, DQK);            // K^T x4.trans (+ x2.trans tail)

    [[maybe_unused]] unsigned long long tdiag[BWD_T_NPHASE] = {};
    [[maybe_unused]] long long tprev = 0;
    auto tstamp = [&](int phase) {   // diag 94 only: charge the cycles since the previous stamp to `phase`
        if constexpr (kDiagTiming) { const long long t = clock64(); tdiag[phase] += static_cast<unsigned long long>(t - tprev); tprev = t; }
    };
    auto tile5 = [&](auto stg, int n) {
        constexpr int S = decltype(stg)::value;
        constexpr uint32_t QST = static_cast<uint32_t>(S) * static_cast<uint32_t>(BM * DQK * 2);                 // Q stage bytes
        constexpr uint32_t OST = static_cast<uint32_t>((C::NDO == 2) ? S : 0) * static_cast<uint32_t>(BM * DVO * 2);
        constexpr uint32_t LST = static_cast<uint32_t>(2 * S) * static_cast<uint32_t>(BM * BW_BN * 2);           // slab stage (sP)
        constexpr uint32_t DSO = static_cast<uint32_t>(BM * BW_BN * 2);                                          // sdS = sP + one slab
        constexpr uint32_t QROW16 = static_cast<uint32_t>(16 * DQK * 2), OROW16 = static_cast<uint32_t>(16 * DVO * 2), LROW16 = static_cast<uint32_t>(16 * BW_BN * 2), LROW8 = LROW16 / 2u;
        const int m_start = (mb0 + n) * BM, m_size = min(BM, sq - m_start);
        const float* lseSs = lseS + S * BM;
        const float* deltaSs = deltaS + S * BM;

        // [L1] tile n+1 -> stage S^1 one tile ahead (WAR on the empty phase of tile n-1, as CFG=4).
        // kOpt8: moved after B1(n) (see below), where the WAR wait is already satisfied by every warp.
        if constexpr (kDiagTiming) tprev = clock64();
        if constexpr (!kOpt8) {
            if (n + 1 < nt) {
                if (n >= 1) bw_mbar_wait(empty_q(S ^ 1), ((n - 1) >> 1) & 1);
                issue_q(n + 1);
            }
        }
        tstamp(BWD_T_ISSUE);
        // [W1] tile n landed in stage S
        bw_mbar_wait(full_q(S), (n >> 1) & 1);
        tstamp(BWD_T_WAITQ);

        // [G1]/[G2] S = scale*Q@K^T and dP = dO@V^T (register-resident V fragments)
        float Sr[RM][2][4], dPr[RM][2][4];
        #pragma unroll
        for (int mm = 0; mm < RM; ++mm) { Sr[mm][0][0]=Sr[mm][0][1]=Sr[mm][0][2]=Sr[mm][0][3]=0.f; Sr[mm][1][0]=Sr[mm][1][1]=Sr[mm][1][2]=Sr[mm][1][3]=0.f; }
        #pragma unroll
        for (int mm = 0; mm < RM; ++mm) { dPr[mm][0][0]=dPr[mm][0][1]=dPr[mm][0][2]=dPr[mm][0][3]=0.f; dPr[mm][1][0]=dPr[mm][1][1]=dPr[mm][1][2]=dPr[mm][1][3]=0.f; }
        if constexpr (kSdp) {
            // CFG=7: the two independent GEMMs INTERLEAVED over their shared d range so four
            // accumulator chains (S[0], S[1], dP[0], dP[1]) are in flight per warp instead of two
            // (the ILP map: 2 chains at 8 warps/SM = 473 TF, 4 = 540). Per accumulator the d order
            // is unchanged (ascending) -> bitwise the same S / dP as the split loops.
            if constexpr (C::NDO == 1) bw_mbar_wait(full_do, n & 1);   // [W2] before the first dO fragment
            #pragma unroll
            for (int d = 0; d < DKV; ++d) {
                uint32_t Kd[4];
                ldm_x4(Kd, aK5[d & 3] + 128u * static_cast<uint32_t>(d >> 2));
                #pragma unroll
                for (int mm = 0; mm < RM; ++mm) {
                    uint32_t Qd[4], dOd[4];
                    ldm_x4(Qd, aQ5[d & 3] + (QST + static_cast<uint32_t>(mm) * QROW16 + 128u * static_cast<uint32_t>(d >> 2)));
                    ldm_x4(dOd, aO5[d & 3] + (OST + static_cast<uint32_t>(mm) * OROW16 + 128u * static_cast<uint32_t>(d >> 2)));
                    bw_mma(Sr[mm][0], Qd, Kd[0], Kd[1]);
                    bw_mma(dPr[mm][0], dOd, Vr[d][0], Vr[d][1]);
                    bw_mma(Sr[mm][1], Qd, Kd[2], Kd[3]);
                    bw_mma(dPr[mm][1], dOd, Vr[d][2], Vr[d][3]);
                }
            }
            #pragma unroll
            for (int d = DKV; d < DKQ; ++d) {   // @192: the 4 S-only d-tiles
                uint32_t Kd[4];
                ldm_x4(Kd, aK5[d & 3] + 128u * static_cast<uint32_t>(d >> 2));
                #pragma unroll
                for (int mm = 0; mm < RM; ++mm) {
                    uint32_t Qd[4];
                    ldm_x4(Qd, aQ5[d & 3] + (QST + static_cast<uint32_t>(mm) * QROW16 + 128u * static_cast<uint32_t>(d >> 2)));
                    bw_mma(Sr[mm][0], Qd, Kd[0], Kd[1]);
                    bw_mma(Sr[mm][1], Qd, Kd[2], Kd[3]);
                }
            }
        } else {
        #pragma unroll
        for (int d = 0; d < DKQ; ++d) {
            uint32_t Kd[4];
            ldm_x4(Kd, aK5[d & 3] + 128u * static_cast<uint32_t>(d >> 2));
            #pragma unroll
            for (int mm = 0; mm < RM; ++mm) {
                uint32_t Qd[4];
                ldm_x4(Qd, aQ5[d & 3] + (QST + static_cast<uint32_t>(mm) * QROW16 + 128u * static_cast<uint32_t>(d >> 2)));
                bw_mma(Sr[mm][0], Qd, Kd[0], Kd[1]);
                bw_mma(Sr[mm][1], Qd, Kd[2], Kd[3]);
            }
        }
        // [W2] (@128) dO of tile n landed
        if constexpr (C::NDO == 1) bw_mbar_wait(full_do, n & 1);
        // [G2] dP = dO@V^T with the register-resident V fragments
        #pragma unroll
        for (int d = 0; d < DKV; ++d) {
            #pragma unroll
            for (int mm = 0; mm < RM; ++mm) {
                uint32_t dOd[4];
                ldm_x4(dOd, aO5[d & 3] + (OST + static_cast<uint32_t>(mm) * OROW16 + 128u * static_cast<uint32_t>(d >> 2)));
                bw_mma(dPr[mm][0], dOd, Vr[d][0], Vr[d][1]);
                bw_mma(dPr[mm][1], dOd, Vr[d][2], Vr[d][3]);
            }
        }
        }   // end S / dP (split or interleaved)
        tstamp(BWD_T_SDP);
        // [P] P, dS -> the swizzled slabs (CFG=4's expressions, guards and rounding)
        #pragma unroll
        for (int mm = 0; mm < RM; ++mm) {
            const int ra = ra5 + mm * BM_M, rb = ra + 8;
            const float lsea = (ra < m_size) ? lseSs[ra] : INFINITY, lseb = (rb < m_size) ? lseSs[rb] : INFINITY;
            const float da = (ra < m_size) ? deltaSs[ra] : 0.f, db = (rb < m_size) ? deltaSs[rb] : 0.f;
            #pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
                const int c0 = sn * 16 + nn * BM_N + (lane % 4) * 2, gca = c0, gcb = c0 + 1;
                float p0=__expf(scale*Sr[mm][nn][0]-lsea), p1=__expf(scale*Sr[mm][nn][1]-lsea);
                float p2=__expf(scale*Sr[mm][nn][2]-lseb), p3=__expf(scale*Sr[mm][nn][3]-lseb);
                bool va0=(gca<n_size)&&(!kCausal||(n_start+gca)<=(m_start+ra+coff)), va1=(gcb<n_size)&&(!kCausal||(n_start+gcb)<=(m_start+ra+coff));
                bool vb0=(gca<n_size)&&(!kCausal||(n_start+gca)<=(m_start+rb+coff)), vb1=(gcb<n_size)&&(!kCausal||(n_start+gcb)<=(m_start+rb+coff));
                p0=(va0&&ra<m_size)?p0:0.f; p1=(va1&&ra<m_size)?p1:0.f; p2=(vb0&&rb<m_size)?p2:0.f; p3=(vb1&&rb<m_size)?p3:0.f;
                const float s0=p0*(dPr[mm][nn][0]-da)*scale, s1=p1*(dPr[mm][nn][1]-da)*scale;
                const float s2=p2*(dPr[mm][nn][2]-db)*scale, s3=p3*(dPr[mm][nn][3]-db)*scale;
                const uint32_t pa = aP5[nn] + (LST + static_cast<uint32_t>(mm) * LROW16);
                bw_sts_b32(pa,                packbf(p0, p1));
                bw_sts_b32(pa + LROW8,        packbf(p2, p3));
                bw_sts_b32(pa + DSO,          packbf(s0, s1));
                bw_sts_b32(pa + DSO + LROW8,  packbf(s2, s3));
            }
        }
        tstamp(BWD_T_VJP);
        // [B1] slab RAW across the 8 warps (the only blocking barrier of the tile; also the 2-stage slab WAR)
        __syncthreads();
        tstamp(BWD_T_B1);
        if constexpr (kOpt8) {
            // [L1'] Q(n+1) -> stage S^1: after B1(n) every warp has finished dK(n-1) (its last read of that
            // stage), so the empty wait cannot block; the copies land under dV/dK/dQ(n) + reds.
            if (n + 1 < nt) {
                if (n >= 1) bw_mbar_wait(empty_q(S ^ 1), ((n - 1) >> 1) & 1);
                issue_q(n + 1);
            }
            tstamp(BWD_T_ISSUE);
        }

        // [G3] dV += P^T @ dO
        #pragma unroll
        for (int kk = 0; kk < MK; ++kk) {
            uint32_t A[2][4];
            #pragma unroll
            for (int mb = 0; mb < 2; ++mb) {
                uint32_t r[4];
                bw_ldm_x4_trans(r, aA5[mb] + (LST + static_cast<uint32_t>(kk) * LROW16));
                A[mb][0] = r[0]; A[mb][1] = r[2]; A[mb][2] = r[1]; A[mb][3] = r[3];
            }
            if constexpr (kOpt8) {
                uint32_t Bv[NFV / 2][4];
                #pragma unroll
                for (int ddp = 0; ddp < NFV / 2; ++ddp) bw_ldm_x4_trans(Bv[ddp], aVB5[ddp] + (OST + static_cast<uint32_t>(kk) * OROW16));
                #pragma unroll
                for (int ddp = 0; ddp < NFV / 2; ++ddp) {
                    #pragma unroll
                    for (int mb = 0; mb < 2; ++mb) {
                        bw_mma(dVr[mb][2 * ddp],     A[mb], Bv[ddp][0], Bv[ddp][1]);
                        bw_mma(dVr[mb][2 * ddp + 1], A[mb], Bv[ddp][2], Bv[ddp][3]);
                    }
                }
            } else {
            #pragma unroll
            for (int ddp = 0; ddp < NFV / 2; ++ddp) {
                uint32_t B[4];
                bw_ldm_x4_trans(B, aVB5[ddp] + (OST + static_cast<uint32_t>(kk) * OROW16));
                #pragma unroll
                for (int mb = 0; mb < 2; ++mb) {
                    bw_mma(dVr[mb][2 * ddp],     A[mb], B[0], B[1]);
                    bw_mma(dVr[mb][2 * ddp + 1], A[mb], B[2], B[3]);
                }
            }
            }
        }
        tstamp(BWD_T_DV);
        if constexpr (C::NDO == 1) {
            __syncwarp();
            if (lane == 0) bw_mbar_arrive(empty_do);
            if (n + 1 < nt) { bw_mbar_wait(empty_do, n & 1); issue_do(n + 1); }
        }
        tstamp(BWD_T_DO);
        // [G4] dK += dS^T @ Q
        #pragma unroll
        for (int kk = 0; kk < MK; ++kk) {
            uint32_t A[2][4];
            #pragma unroll
            for (int mb = 0; mb < 2; ++mb) {
                uint32_t r[4];
                bw_ldm_x4_trans(r, aA5[mb] + (LST + DSO + static_cast<uint32_t>(kk) * LROW16));
                A[mb][0] = r[0]; A[mb][1] = r[2]; A[mb][2] = r[1]; A[mb][3] = r[3];
            }
            if constexpr (kOpt8) {
                uint32_t Bk[NFK / 2][4];
                #pragma unroll
                for (int ddp = 0; ddp < NFK / 2; ++ddp) bw_ldm_x4_trans(Bk[ddp], aKB5[ddp] + (QST + static_cast<uint32_t>(kk) * QROW16));
                #pragma unroll
                for (int ddp = 0; ddp < NFK / 2; ++ddp) {
                    #pragma unroll
                    for (int mb = 0; mb < 2; ++mb) {
                        bw_mma(dKr[mb][2 * ddp],     A[mb], Bk[ddp][0], Bk[ddp][1]);
                        bw_mma(dKr[mb][2 * ddp + 1], A[mb], Bk[ddp][2], Bk[ddp][3]);
                    }
                }
            } else {
            #pragma unroll
            for (int ddp = 0; ddp < NFK / 2; ++ddp) {
                uint32_t B[4];
                bw_ldm_x4_trans(B, aKB5[ddp] + (QST + static_cast<uint32_t>(kk) * QROW16));
                #pragma unroll
                for (int mb = 0; mb < 2; ++mb) {
                    bw_mma(dKr[mb][2 * ddp],     A[mb], B[0], B[1]);
                    bw_mma(dKr[mb][2 * ddp + 1], A[mb], B[2], B[3]);
                }
            }
            }
        }
        __syncwarp();
        if (lane == 0) bw_mbar_arrive(empty_q(S));
        tstamp(BWD_T_DK);

        // [G5] dQ = dS @ K over the resident keys, then [R] red.v2 into dq_acc (as CFG=4)
        float dQp[2][QNF][4];
        #pragma unroll
        for (int mb = 0; mb < 2; ++mb) {
            #pragma unroll
            for (int i = 0; i < QNF; ++i) { dQp[mb][i][0]=dQp[mb][i][1]=dQp[mb][i][2]=dQp[mb][i][3]=0.f; }
        }
        #pragma unroll
        for (int kk = 0; kk < KKN; ++kk) {
            uint32_t A[2][4];
            #pragma unroll
            for (int mb = 0; mb < 2; ++mb) ldm_x4(A[mb], aDA5[kk] + (LST + DSO + static_cast<uint32_t>(mb) * LROW16));
            if constexpr (kOpt8) {
                uint32_t Bq[QNF / 2][4];
                [[maybe_unused]] uint32_t B2[2];
                #pragma unroll
                for (int j = 0; j < QNF / 2; ++j) bw_ldm_x4_trans(Bq[j], aDB5[j] + static_cast<uint32_t>(kk) * QROW16);
                if constexpr (QNF % 2 == 1) ldm_x2_trans(B2, aDB5[QNF / 2] + static_cast<uint32_t>(kk) * QROW16);
                #pragma unroll
                for (int j = 0; j < QNF / 2; ++j) {
                    #pragma unroll
                    for (int mb = 0; mb < 2; ++mb) {
                        bw_mma(dQp[mb][2 * j],     A[mb], Bq[j][0], Bq[j][1]);
                        bw_mma(dQp[mb][2 * j + 1], A[mb], Bq[j][2], Bq[j][3]);
                    }
                }
                if constexpr (QNF % 2 == 1) {
                    #pragma unroll
                    for (int mb = 0; mb < 2; ++mb) bw_mma(dQp[mb][QNF - 1], A[mb], B2[0], B2[1]);
                }
            } else {
            #pragma unroll
            for (int j = 0; j < QNF / 2; ++j) {
                uint32_t B[4];
                bw_ldm_x4_trans(B, aDB5[j] + static_cast<uint32_t>(kk) * QROW16);
                #pragma unroll
                for (int mb = 0; mb < 2; ++mb) {
                    bw_mma(dQp[mb][2 * j],     A[mb], B[0], B[1]);
                    bw_mma(dQp[mb][2 * j + 1], A[mb], B[2], B[3]);
                }
            }
            if constexpr (QNF % 2 == 1) {
                uint32_t B2[2];
                ldm_x2_trans(B2, aDB5[QNF / 2] + static_cast<uint32_t>(kk) * QROW16);
                #pragma unroll
                for (int mb = 0; mb < 2; ++mb) bw_mma(dQp[mb][QNF - 1], A[mb], B2[0], B2[1]);
            }
            }
        }
        tstamp(BWD_T_DQ);
        #pragma unroll
        for (int mb = 0; mb < 2; ++mb) {
            const int ra = qm * 32 + mb * BM_M + (lane / 4), rb = ra + 8;
            float* ga = dq_acc + ((int64_t)(q0 + m_start + ra) * num_heads + h) * DQK + qn * DQW + (lane % 4) * 2;
            float* gb = dq_acc + ((int64_t)(q0 + m_start + rb) * num_heads + h) * DQK + qn * DQW + (lane % 4) * 2;
            if constexpr (kDiagNoRed) {
                // diag 95: keep the dQ GEMM alive (fold into a register the compiler cannot drop), no reds
                float acc = 0.f;
                #pragma unroll
                for (int nf = 0; nf < QNF; ++nf) acc += dQp[mb][nf][0] + dQp[mb][nf][1] + dQp[mb][nf][2] + dQp[mb][nf][3];
                if (acc == 12345.678f && ra < m_size) ga[0] = acc;   // never true for real data; defeats DCE
            } else if constexpr (kCoal && kOpt8) {
                // CFG=8: red.v4 fragment order. Run of this (warp, m-tile, half) = QNF*64 floats: pairs of n-frags
                // (2j, 2j+1) as 128-float chunks (lane*4 + {nf 2j regs, nf 2j+1 regs}), then (QNF odd) a 64-float
                // chunk (lane*2) for the last n-frag. One 16-B reduction per pair.
                float* blk = dq_acc + ((int64_t)(rbase6 + m_start) * num_heads + (int64_t)h * BM) * DQK;
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int r0 = half * 2, r1 = half * 2 + 1;
                    const bool ok = (half == 0) ? (ra < m_size) : (rb < m_size);
                    float* run = blk + (((warp * 2 + mb) * 2 + half) * QNF * 64);
                    #pragma unroll
                    for (int j = 0; j < QNF / 2; ++j) {
                        if (ok) red_add_v4f32(run + (j * 128 + lane * 4), dQp[mb][2 * j][r0], dQp[mb][2 * j][r1], dQp[mb][2 * j + 1][r0], dQp[mb][2 * j + 1][r1]);
                    }
                    if constexpr (QNF % 2 == 1) {
                        if (ok) red_add_v2f32(run + ((QNF / 2) * 128 + lane * 2), dQp[mb][QNF - 1][r0], dQp[mb][QNF - 1][r1]);
                    }
                }
            } else if constexpr (kCoal) {
                // CFG=6: 256 contiguous bytes per warp red.v2 (fragment-order block of this (batch, tile, head)).
                // The tile's row range owns BM*num_heads*DQK floats = num_heads blocks of BM*DQK: head h's
                // block starts h*BM rows in (NOT h*DQK: that is the dense per-row head interleave and would
                // overlap the neighbouring heads' blocks).
                float* blk = dq_acc + ((int64_t)(rbase6 + m_start) * num_heads + (int64_t)h * BM) * DQK;
                #pragma unroll
                for (int nf = 0; nf < QNF; ++nf) {
                    const int c = ((warp * 2 + mb) * QNF + nf) * 2;
                    if (ra < m_size) red_add_v2f32(blk + (c * 64 + lane * 2),       dQp[mb][nf][0], dQp[mb][nf][1]);
                    if (rb < m_size) red_add_v2f32(blk + ((c + 1) * 64 + lane * 2), dQp[mb][nf][2], dQp[mb][nf][3]);
                }
            } else {
            #pragma unroll
            for (int nf = 0; nf < QNF; ++nf) {
                if (ra < m_size) red_add_v2f32(ga + nf * BM_N, dQp[mb][nf][0], dQp[mb][nf][1]);
                if (rb < m_size) red_add_v2f32(gb + nf * BM_N, dQp[mb][nf][2], dQp[mb][nf][3]);
            }
            }
        }
        tstamp(BWD_T_RED);
    };
    for (int n = 0; n < nt; n += 2) {
        tile5(std::integral_constant<int, 0>{}, n);
        if (n + 1 < nt) tile5(std::integral_constant<int, 1>{}, n + 1);
    }
    if constexpr (kDiagTiming) {
        __syncwarp();
        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < BWD_T_NPHASE; ++i) atomicAdd(&bw_diag_cycles[i], tdiag[i]);
            atomicAdd(&bw_diag_cycles[BWD_T_TILES], static_cast<unsigned long long>(nt > 0 ? nt : 0));
            atomicAdd(&bw_diag_cycles[BWD_T_WARPS], 1ull);
        }
    }
    }   // end of the kAddr arms

    // ---- epilogue: dK / dV of this warp's 32 keys x d quarter, bf16 pairs, no merge, no barrier.
    // Every issued tile was waited on by every thread, so no cp.async is in flight at exit.
    #pragma unroll
    for (int mb = 0; mb < 2; ++mb) {
        const int ka = km * 32 + mb * BM_M + (lane / 4), kb = ka + 8;
        #pragma unroll
        for (int nf = 0; nf < NFK; ++nf) {
            const int dc = kn * (DQK / 4) + nf * BM_N + (lane % 4) * 2;
            if (ka < n_size) *reinterpret_cast<__nv_bfloat162*>(dk + (int64_t)(kv0 + n_start + ka) * sk_tok + h * DQK + dc) = __floats2bfloat162_rn(dKr[mb][nf][0], dKr[mb][nf][1]);
            if (kb < n_size) *reinterpret_cast<__nv_bfloat162*>(dk + (int64_t)(kv0 + n_start + kb) * sk_tok + h * DQK + dc) = __floats2bfloat162_rn(dKr[mb][nf][2], dKr[mb][nf][3]);
        }
        #pragma unroll
        for (int nf = 0; nf < NFV; ++nf) {
            const int dc = kn * (DVO / 4) + nf * BM_N + (lane % 4) * 2;
            if (ka < n_size) *reinterpret_cast<__nv_bfloat162*>(dv + (int64_t)(kv0 + n_start + ka) * sv_tok + h * DVO + dc) = __floats2bfloat162_rn(dVr[mb][nf][0], dVr[mb][nf][1]);
            if (kb < n_size) *reinterpret_cast<__nv_bfloat162*>(dv + (int64_t)(kv0 + n_start + kb) * sv_tok + h * DVO + dc) = __floats2bfloat162_rn(dVr[mb][nf][2], dVr[mb][nf][3]);
        }
    }
}

// ============================================================================
// Kernel A: dK + dV.  CTA per KV block; warps own keys; loop Q; no atomics.
//   kCfg >= 1 (FLASH_MLA_SM120_BWD_MMA_CFG=1): + a dQ PHASE per Q block -- dQ = dS @ K over
//   the CTA's 64 resident keys, staged through dedicated smem and reduced into the fp32
//   dq_acc[total_q][num_heads][DQK] with cp.reduce.async.bulk (per-row bulk adds on the
//   async proxy). kCfg == 0 is byte-identical to the original kernel: the dQ phase is
//   compiled out and dq_acc is unused.
//   kCfg == 2: dQ reduced register-direct with red.global.add.v2.f32 (smem = base + sDS).
//   kCfg == 3: CFG=2 with 8 warps (BwCta<3>::kThreads = 256): warp = key group (w & 3) x
//   query half (w >> 2); per-warp PARTIAL dK/dV over 16 queries, merged at kernel end
//   (deterministic); dQ d-quarter per warp, still red.v2. CFG 0-2 code paths are untouched.
//   kCfg == 4: the self-contained FROST-structured body fmha_bwd_mma_dkdv_cfg4_body (8 math
//   warps, BwCta<4>::kThreads = 256, its own prologue / smem map Bw4<> / mbarrier stages);
//   nothing of the kCfg <= 3 body below is shared with it.
// ============================================================================
template <int DQK, int DVO, bool kCausal, int kCfg>
__global__ void __launch_bounds__(BwCta<kCfg>::kThreads)
fmha_bwd_mma_dkdv_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ o,
    const __nv_bfloat16* __restrict__ d_o, const float* __restrict__ lse,
    const float* __restrict__ delta,
    const int* __restrict__ cu_q, const int* __restrict__ cu_kv,
    __nv_bfloat16* __restrict__ dk, __nv_bfloat16* __restrict__ dv,
    int num_heads, float scale, int max_sq, int max_skv,
    float* __restrict__ dq_acc) {
    if constexpr (kCfg >= 4) {
        fmha_bwd_mma_dkdv_cfg4_body<DQK, DVO, kCausal, (kCfg >= 5) ? 1 : 0, (kCfg == 95), (kCfg >= 6 && kCfg != 95),
                                    (kCfg == 7), (kCfg == 94), ((kCfg >= 8 && kCfg != 95) || kCfg == 94)>(q, k, v, d_o, lse, delta, cu_q, cu_kv, dk, dv,
                                                       num_heads, scale, dq_acc);
    } else {
    // BM=32 (query loop) so the cp.async double-buffered Q/dO stages fit the delta-freed
    // smem at both MHA and MLA head dims.
    constexpr int BM = 32;
    constexpr int PT_STRIDE = BM > 64 ? BM : 64;
    constexpr int DKQ = DQK / BM_K;      // S contraction d-tiles (12 for 192)
    constexpr int DKV = DVO / BM_K;      // dP contraction d-tiles (8)
    constexpr int DNQ = DQK / BM_N;      // dK output n-tiles (24 for 192)
    constexpr int DNV = DVO / BM_N;      // dV output n-tiles (16)
    constexpr int WNN = BW_WN / BM_N;    // n8 tiles over a warp's 16 keys (2)
    constexpr int MM = BM / BM_M;        // m16 tiles over BM queries
    constexpr int MK = BM / BM_K;        // k16 tiles over BM queries
    constexpr int kThreads = BwCta<kCfg>::kThreads;   // CTA threads: 128 (CFG 0-2) / 256 (CFG=3)

    const int b = blockIdx.z, h = blockIdx.y, nb = blockIdx.x;
    if (h >= num_heads) return;
    const int q0 = cu_q[b], q1 = cu_q[b + 1], kv0 = cu_kv[b], kv1 = cu_kv[b + 1];
    const int sq = q1 - q0, skv = kv1 - kv0;
    const int n_start = nb * BW_BN;
    if (n_start >= skv) return;
    const int n_size = min(BW_BN, skv - n_start);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int kw0 = warp * BW_WN;
    // CFG=3 warp map (8 warps): key group kg3 = warp & 3 (keys kw3 = kg3*16..+16, exactly the
    // CFG<=2 key map of warp kg3) x query half mh3 = warp >> 2 (m-tile mh3 = queries
    // mh3*16..+16 of the BM=32 Q block). Dead (unused) at kCfg < 3.
    [[maybe_unused]] const int kg3 = (kCfg >= 3) ? (warp & 3) : warp;
    [[maybe_unused]] const int mh3 = (kCfg >= 3) ? (warp >> 2) : 0;
    [[maybe_unused]] const int kw3 = kg3 * BW_WN;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* Ks    = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* Vs    = Ks    + BW_BN * DQK;
    __nv_bfloat16* Qbuf  = Vs    + BW_BN * DVO;     // [2][BM*DQK] cp.async double-buffer
    __nv_bfloat16* dObuf = Qbuf  + 2 * BM * DQK;    // [2][BM*DVO]
    // O tile dropped (delta precomputed); P^T/dS^T transpose staging dropped (movmatrix).
    float* lseS   = reinterpret_cast<float*>(dObuf + 2 * BM * DVO);
    float* deltaS = lseS + BM;
    // CFG>=1 dQ-phase regions, DEDICATED (appended after deltaS; nothing aliases them):
    // sDS bf16 [BM][BW_BN] swizzled, then sDQh fp32 [BM][DQK/2 + BW_DQH_PAD]. Both start
    // 16B-aligned: the base region is 82,176 B @192 / 65,792 B @128 and sDS is 4,096 B.
    // Totals: 99,584 B @192 / 79,104 B @128 (sm120 opt-in maximum 101,376 B).
    static_assert(kCfg == 0 || bw_dkdv_base_smem<DQK, DVO, BM>() + bw_dq_phase_smem<DQK, BM, kCfg>() <= 101376,
                  "CFG>=1 dKdV smem exceeds the sm120 99KB dynamic-smem opt-in maximum");
    static_assert(bw_dkdv_base_smem<DQK, DVO, BM>() % 16 == 0 && (BM * BW_BN * 2) % 16 == 0,
                  "sDS / sDQh must start 16B-aligned (cp.reduce.async.bulk source rows)");
    [[maybe_unused]] __nv_bfloat16* sDS  = reinterpret_cast<__nv_bfloat16*>(deltaS + BM);
    [[maybe_unused]] float*         sDQh = reinterpret_cast<float*>(sDS + BM * BW_BN);

    bg2s<kThreads>(Ks, k, kv0, n_start, n_size, num_heads, h, DQK);
    bg2s<kThreads>(Vs, v, kv0, n_start, n_size, num_heads, h, DVO);
    for (int i = tid; i < (BW_BN - n_size) * DQK; i += kThreads) {
        int r = n_size + i / DQK, c = i % DQK; Ks[swz(r, c, DQK)] = __float2bfloat16(0.f);
    }
    for (int i = tid; i < (BW_BN - n_size) * DVO; i += kThreads) {
        int r = n_size + i / DVO, c = i % DVO; Vs[swz(r, c, DVO)] = __float2bfloat16(0.f);
    }
    __syncthreads();

    float dKr[DNQ][4], dVr[DNV][4];
    #pragma unroll
    for (int i = 0; i < DNQ; ++i) { dKr[i][0]=dKr[i][1]=dKr[i][2]=dKr[i][3]=0.f; }
    #pragma unroll
    for (int i = 0; i < DNV; ++i) { dVr[i][0]=dVr[i][1]=dVr[i][2]=dVr[i][3]=0.f; }

    int mb0 = 0;
    // Bottom-right causal alignment (= the forward's): row r sees key g iff g <= r + (skv - sq).
    // For sq == skv this is exactly the old (n_start - (BM-1)) / BM start (identical numerics).
    const int coff = skv - sq;
    if (kCausal) mb0 = max(0, n_start - coff - (BM - 1)) / BM;
    int nq = (sq + BM - 1) / BM;
    const int sk_tok = num_heads * DQK, sv_tok = num_heads * DVO;

    // cp.async pipeline over Q blocks: prefetch mb0 into stage 0, then prefetch block mb+1
    // while computing mb. After mb0 the block range is contiguous (the leading mb0 block may
    // be fully causal-masked -> zero contribution, harmless), so a 2-stage buffer applies.
    {
        int ms0 = min(BM, sq - mb0 * BM);
        g2s_cp_t<kThreads>(Qbuf,  q,   q0, mb0 * BM, ms0, num_heads, h, DQK);
        g2s_cp_t<kThreads>(dObuf, d_o, q0, mb0 * BM, ms0, num_heads, h, DVO);
        cp_commit();
    }
    for (int mb = mb0; mb < nq; ++mb) {
        const int cur = (mb - mb0) & 1;
        __nv_bfloat16* Qs  = Qbuf  + cur * (BM * DQK);
        __nv_bfloat16* dOs = dObuf + cur * (BM * DVO);
        const int m_start = mb * BM;
        const int m_size = min(BM, sq - m_start);

        if (mb + 1 < nq) {
            const int nxt = (mb + 1 - mb0) & 1;
            int ms2 = min(BM, sq - (mb + 1) * BM);
            g2s_cp_t<kThreads>(Qbuf  + nxt * (BM * DQK), q,   q0, (mb + 1) * BM, ms2, num_heads, h, DQK);
            g2s_cp_t<kThreads>(dObuf + nxt * (BM * DVO), d_o, q0, (mb + 1) * BM, ms2, num_heads, h, DVO);
            cp_commit();
            cp_wait_group<1>();
        } else {
            cp_wait_group<0>();
        }
        for (int m = tid; m < m_size; m += kThreads) {
            lseS[m]   = lse[(q0 + m_start + m) * num_heads + h];
            deltaS[m] = delta[(q0 + m_start + m) * num_heads + h];   // precomputed (no O tile)
        }
        for (int i = tid; i < (BM - m_size) * DQK; i += kThreads) {
            int r = m_size + i / DQK, c = i % DQK; Qs[swz(r, c, DQK)] = __float2bfloat16(0.f);
        }
        for (int i = tid; i < (BM - m_size) * DVO; i += kThreads) {
            int r = m_size + i / DVO, c = i % DVO; dOs[swz(r, c, DVO)] = __float2bfloat16(0.f);
        }
        __syncthreads();

        if constexpr (kCfg >= 3) {
            // ================================================================
            // CFG=3 Q-block body (8 warps, 256 threads; same smem -> 1 CTA/SM, 2 warps per SM
            // sub-partition). This warp = key group kg3 (keys kw3..+16, WNN=2 n-tiles; the
            // CFG<=2 key map) x query half mh3 (m-tile mh3 of the BM=32 block): a 16-key x
            // 16-query slab of S/dP/P/dS = HALF of the CFG<=2 per-warp temporaries. Same
            // per-element operands and order as CFG=2; the ONLY numeric difference is the
            // dV/dK contraction over the block's 32 queries, which is SPLIT: this warp
            // accumulates its 16 queries (k-tile kk = mh3) into its own PARTIAL dKr/dVr and
            // the two halves of a key group are merged once after the loop (fixed fp32
            // order -> deterministic). dQ keeps CFG=2's register-direct red.v2 path.
            // ================================================================
            static_assert(MM == 2 && MK == 2 && BW_NW == 4, "CFG=3 warp map: 8 warps = 4 key groups x 2 query halves of BM=32");
            static_assert(DNQ % 4 == 0, "CFG=3 dQ phase splits the head dim into d-quarters");
            // S = scale*Q@K^T (contract DQK), m-tile mh3 only (Qd: one A fragment per d-tile).
            float Sr[WNN][4];
            #pragma unroll
            for (int nn = 0; nn < WNN; ++nn) { Sr[nn][0]=Sr[nn][1]=Sr[nn][2]=Sr[nn][3]=0.f; }
            #pragma unroll
            for (int d = 0; d < DKQ; ++d) {
                uint32_t Qd[4], Kd[WNN][2];
                { int row=mh3*BM_M+(lane%16), col=d*BM_K+(lane/16)*8; ldm_x4(Qd, cvta_shared(Qs + swz(row, col, DQK))); }
                #pragma unroll
                for (int nn = 0; nn < WNN; ++nn) { int kr=kw3+nn*BM_N+(lane%8), kc=d*BM_K+(lane/8%2)*8; ldm_x2(Kd[nn], cvta_shared(Ks + swz(kr, kc, DQK))); }
                #pragma unroll
                for (int nn = 0; nn < WNN; ++nn) mma_16x8x16(Sr[nn], Qd, Kd[nn], Sr[nn]);
            }
            // dP = dO@V^T (contract DVO), m-tile mh3 only.
            float dPr[WNN][4];
            #pragma unroll
            for (int nn = 0; nn < WNN; ++nn) { dPr[nn][0]=dPr[nn][1]=dPr[nn][2]=dPr[nn][3]=0.f; }
            #pragma unroll
            for (int d = 0; d < DKV; ++d) {
                uint32_t dOd[4], Vd[WNN][2];
                { int row=mh3*BM_M+(lane%16), col=d*BM_K+(lane/16)*8; ldm_x4(dOd, cvta_shared(dOs + swz(row, col, DVO))); }
                #pragma unroll
                for (int nn = 0; nn < WNN; ++nn) { int kr=kw3+nn*BM_N+(lane%8), kc=d*BM_K+(lane/8%2)*8; ldm_x2(Vd[nn], cvta_shared(Vs + swz(kr, kc, DVO))); }
                #pragma unroll
                for (int nn = 0; nn < WNN; ++nn) mma_16x8x16(dPr[nn], dOd, Vd[nn], dPr[nn]);
            }

            // P = exp(scale*S - LSE); dS = P*(dP - delta)*scale -- CFG<=2's expressions and
            // guards (rows >= m_size, keys >= n_size, causal) for the rows of m-tile mh3.
            float Pr[WNN][4], dSr[WNN][4];
            const int ra3 = mh3 * BM_M + (lane / 4), rb3 = ra3 + 8;   // this warp's query rows
            {
                float lsea = (ra3 < m_size) ? lseS[ra3] : INFINITY, lseb = (rb3 < m_size) ? lseS[rb3] : INFINITY;
                float da = (ra3 < m_size) ? deltaS[ra3] : 0.f, db = (rb3 < m_size) ? deltaS[rb3] : 0.f;
                #pragma unroll
                for (int nn = 0; nn < WNN; ++nn) {
                    int c0 = kw3 + nn * BM_N + (lane % 4) * 2, gca = c0, gcb = c0 + 1;
                    float p0=__expf(scale*Sr[nn][0]-lsea), p1=__expf(scale*Sr[nn][1]-lsea);
                    float p2=__expf(scale*Sr[nn][2]-lseb), p3=__expf(scale*Sr[nn][3]-lseb);
                    bool va0=(gca<n_size)&&(!kCausal||(n_start+gca)<=(m_start+ra3+coff)), va1=(gcb<n_size)&&(!kCausal||(n_start+gcb)<=(m_start+ra3+coff));
                    bool vb0=(gca<n_size)&&(!kCausal||(n_start+gca)<=(m_start+rb3+coff)), vb1=(gcb<n_size)&&(!kCausal||(n_start+gcb)<=(m_start+rb3+coff));
                    p0=(va0&&ra3<m_size)?p0:0.f; p1=(va1&&ra3<m_size)?p1:0.f; p2=(vb0&&rb3<m_size)?p2:0.f; p3=(vb1&&rb3<m_size)?p3:0.f;
                    Pr[nn][0]=p0; Pr[nn][1]=p1; Pr[nn][2]=p2; Pr[nn][3]=p3;
                    dSr[nn][0]=p0*(dPr[nn][0]-da)*scale; dSr[nn][1]=p1*(dPr[nn][1]-da)*scale;
                    dSr[nn][2]=p2*(dPr[nn][2]-db)*scale; dSr[nn][3]=p3*(dPr[nn][3]-db)*scale;
                }
            }

            // dV += P^T @ dO over THIS warp's 16 queries only (k-tile kk = mh3). A = P^T built
            // in-register via movmatrix (a0/a2 = keys 0-7 = n-tile 0, a1/a3 = keys 8-15 =
            // n-tile 1; c0c1 -> query-lo, c2c3 -> query-hi), B = dO rows mh3*16..+16 x2.trans.
            // dV before dK so the P fragments die first (one transient A fragment live).
            {
                uint32_t Ar[4];
                Ar[0] = movmatrix_t(packbf(Pr[0][0], Pr[0][1]));
                Ar[1] = movmatrix_t(packbf(Pr[1][0], Pr[1][1]));
                Ar[2] = movmatrix_t(packbf(Pr[0][2], Pr[0][3]));
                Ar[3] = movmatrix_t(packbf(Pr[1][2], Pr[1][3]));
                #pragma unroll
                for (int dd = 0; dd < DNV; ++dd) {
                    uint32_t Br[2]; int vrow=mh3*BM_K+(lane%16), vcol=dd*BM_N+(lane/16)*8;
                    ldm_x2_trans(Br, cvta_shared(dOs + swz(vrow, vcol, DVO)));
                    mma_16x8x16(dVr[dd], Ar, Br, dVr[dd]);
                }
            }
            // dK += dS^T @ Q over the same 16 queries. A = dS^T via movmatrix; B = Q rows
            // mh3*16..+16 x2.trans.
            {
                uint32_t Ar[4];
                Ar[0] = movmatrix_t(packbf(dSr[0][0], dSr[0][1]));
                Ar[1] = movmatrix_t(packbf(dSr[1][0], dSr[1][1]));
                Ar[2] = movmatrix_t(packbf(dSr[0][2], dSr[0][3]));
                Ar[3] = movmatrix_t(packbf(dSr[1][2], dSr[1][3]));
                #pragma unroll
                for (int dd = 0; dd < DNQ; ++dd) {
                    uint32_t Br[2]; int qrow=mh3*BM_K+(lane%16), qcol=dd*BM_N+(lane/16)*8;
                    ldm_x2_trans(Br, cvta_shared(Qs + swz(qrow, qcol, DQK)));
                    mma_16x8x16(dKr[dd], Ar, Br, dKr[dd]);
                }
            }

            // (1) dS exchange: this warp's 16x16 slab -> sDS rows ra3/rb3, cols kw3 + nn*8 +
            // (lane%4)*2 (+1), swz(row, col, 64); the full [32][64] tile is complete after
            // BARRIER [B]. No barrier BEFORE the stores: the only readers of sDS are the
            // previous iteration's dQ ldmatrix (step 2), ordered before this iteration by that
            // iteration's trailing __syncthreads. dSr already carries `scale`.
            #pragma unroll
            for (int nn = 0; nn < WNN; ++nn) {
                const int c = kw3 + nn * BM_N + (lane % 4) * 2;
                *reinterpret_cast<__nv_bfloat162*>(sDS + swz(ra3, c, BW_BN)) = __floats2bfloat162_rn(dSr[nn][0], dSr[nn][1]);
                *reinterpret_cast<__nv_bfloat162*>(sDS + swz(rb3, c, BW_BN)) = __floats2bfloat162_rn(dSr[nn][2], dSr[nn][3]);
            }
            // BARRIER [B]: RAW on sDS across the 8 warps -- the dQ GEMM below contracts over
            // all 64 keys, i.e. over the four key groups' columns.
            __syncthreads();

            // (2) dQ GEMM: warp -> m-tile dq_mm = mh3, d-QUARTER dq_nq = kg3 (n-tiles
            // dq_nq*DNQ4..+DNQ4: 6 @192 / 4 @128) over the KKN=4 key k-tiles. A = dS[dq_mm][kk]
            // via ldmatrix.x4 from sDS (non-trans), B = K^T via ldmatrix.x2.trans from the
            // RESIDENT Ks -- CFG=2's exact operand pattern (padded keys contribute 0: dS = 0
            // there, Ks rows zeroed). kk outer / ddl inner: one A fragment live at a time,
            // consecutive mmas hit independent accumulators, each dQp[ddl] accumulates
            // kk = 0..3 ascending. No partial sums across warps.
            constexpr int DNQ4 = DNQ / 4;
            constexpr int KKN = BW_BN / BM_K;
            const int dq_mm = mh3, dq_nq = kg3;
            float dQp[DNQ4][4];
            #pragma unroll
            for (int i = 0; i < DNQ4; ++i) { dQp[i][0]=dQp[i][1]=dQp[i][2]=dQp[i][3]=0.f; }
            #pragma unroll
            for (int kk = 0; kk < KKN; ++kk) {
                uint32_t Ar[4];
                { int row = dq_mm * BM_M + (lane % 16), col = kk * BM_K + (lane / 16) * 8; ldm_x4(Ar, cvta_shared(sDS + swz(row, col, BW_BN))); }
                #pragma unroll
                for (int ddl = 0; ddl < DNQ4; ++ddl) {
                    uint32_t Br[2]; int kr = kk * BM_K + (lane % 16), kc = (dq_nq * DNQ4 + ddl) * BM_N + (lane / 16) * 8;
                    ldm_x2_trans(Br, cvta_shared(Ks + swz(kr, kc, DQK)));
                    mma_16x8x16(dQp[ddl], Ar, Br, dQp[ddl]);
                }
            }
            // (3') CFG=2's register-direct, fire-and-forget red.global.add.v2.f32 of this
            // warp's dQ quarter: rows dq_mm*16 + lane/4 (+8) = ra3/rb3, cols dq_nq*DQK/4 +
            // ddl*8 + (lane%4)*2. Rows >= m_size belong to the next varlen sequence and are
            // never reduced. Completion is guaranteed at kernel exit (stream order before the
            // convert kernel).
            {
                float* ga = dq_acc + ((int64_t)(q0 + m_start + ra3) * num_heads + h) * DQK + dq_nq * (DQK / 4) + (lane % 4) * 2;
                float* gb = dq_acc + ((int64_t)(q0 + m_start + rb3) * num_heads + h) * DQK + dq_nq * (DQK / 4) + (lane % 4) * 2;
                #pragma unroll
                for (int ddl = 0; ddl < DNQ4; ++ddl) {
                    if (ra3 < m_size) red_add_v2f32(ga + ddl * BM_N, dQp[ddl][0], dQp[ddl][1]);
                    if (rb3 < m_size) red_add_v2f32(gb + ddl * BM_N, dQp[ddl][2], dQp[ddl][3]);
                }
            }
            // The trailing __syncthreads below closes the Q-block body: WAR on the Qs/dOs stage
            // against the next cp.async prefetch, and on sDS against the next iteration's
            // exchange stores (every warp's step-2 ldmatrix precedes it).
        } else {
        // S = scale*Q@K^T (contract DQK)
        float Sr[MM][WNN][4];
        #pragma unroll
        for (int mm = 0; mm < MM; ++mm) for (int nn = 0; nn < WNN; ++nn) { Sr[mm][nn][0]=Sr[mm][nn][1]=Sr[mm][nn][2]=Sr[mm][nn][3]=0.f; }
        #pragma unroll
        for (int d = 0; d < DKQ; ++d) {
            uint32_t Qd[MM][4], Kd[WNN][2];
            #pragma unroll
            for (int mm = 0; mm < MM; ++mm) { int row=mm*BM_M+(lane%16), col=d*BM_K+(lane/16)*8; ldm_x4(Qd[mm], cvta_shared(Qs + swz(row, col, DQK))); }
            #pragma unroll
            for (int nn = 0; nn < WNN; ++nn) { int kr=kw0+nn*BM_N+(lane%8), kc=d*BM_K+(lane/8%2)*8; ldm_x2(Kd[nn], cvta_shared(Ks + swz(kr, kc, DQK))); }
            #pragma unroll
            for (int mm = 0; mm < MM; ++mm) for (int nn = 0; nn < WNN; ++nn) mma_16x8x16(Sr[mm][nn], Qd[mm], Kd[nn], Sr[mm][nn]);
        }
        // dP = dO@V^T (contract DVO)
        float dPr[MM][WNN][4];
        #pragma unroll
        for (int mm = 0; mm < MM; ++mm) for (int nn = 0; nn < WNN; ++nn) { dPr[mm][nn][0]=dPr[mm][nn][1]=dPr[mm][nn][2]=dPr[mm][nn][3]=0.f; }
        #pragma unroll
        for (int d = 0; d < DKV; ++d) {
            uint32_t dOd[MM][4], Vd[WNN][2];
            #pragma unroll
            for (int mm = 0; mm < MM; ++mm) { int row=mm*BM_M+(lane%16), col=d*BM_K+(lane/16)*8; ldm_x4(dOd[mm], cvta_shared(dOs + swz(row, col, DVO))); }
            #pragma unroll
            for (int nn = 0; nn < WNN; ++nn) { int kr=kw0+nn*BM_N+(lane%8), kc=d*BM_K+(lane/8%2)*8; ldm_x2(Vd[nn], cvta_shared(Vs + swz(kr, kc, DVO))); }
            #pragma unroll
            for (int mm = 0; mm < MM; ++mm) for (int nn = 0; nn < WNN; ++nn) mma_16x8x16(dPr[mm][nn], dOd[mm], Vd[nn], dPr[mm][nn]);
        }

        // P = exp(scale*S - LSE); dS = P*(dP - delta)*scale.
        float Pr[MM][WNN][4], dSr[MM][WNN][4];
        #pragma unroll
        for (int mm = 0; mm < MM; ++mm) {
            int ra = mm * BM_M + (lane / 4), rb = ra + 8;
            float lsea = (ra < m_size) ? lseS[ra] : INFINITY, lseb = (rb < m_size) ? lseS[rb] : INFINITY;
            float da = (ra < m_size) ? deltaS[ra] : 0.f, db = (rb < m_size) ? deltaS[rb] : 0.f;
            #pragma unroll
            for (int nn = 0; nn < WNN; ++nn) {
                int c0 = kw0 + nn * BM_N + (lane % 4) * 2, gca = c0, gcb = c0 + 1;
                float p0=__expf(scale*Sr[mm][nn][0]-lsea), p1=__expf(scale*Sr[mm][nn][1]-lsea);
                float p2=__expf(scale*Sr[mm][nn][2]-lseb), p3=__expf(scale*Sr[mm][nn][3]-lseb);
                bool va0=(gca<n_size)&&(!kCausal||(n_start+gca)<=(m_start+ra+coff)), va1=(gcb<n_size)&&(!kCausal||(n_start+gcb)<=(m_start+ra+coff));
                bool vb0=(gca<n_size)&&(!kCausal||(n_start+gca)<=(m_start+rb+coff)), vb1=(gcb<n_size)&&(!kCausal||(n_start+gcb)<=(m_start+rb+coff));
                p0=(va0&&ra<m_size)?p0:0.f; p1=(va1&&ra<m_size)?p1:0.f; p2=(vb0&&rb<m_size)?p2:0.f; p3=(vb1&&rb<m_size)?p3:0.f;
                Pr[mm][nn][0]=p0; Pr[mm][nn][1]=p1; Pr[mm][nn][2]=p2; Pr[mm][nn][3]=p3;
                dSr[mm][nn][0]=p0*(dPr[mm][nn][0]-da)*scale; dSr[mm][nn][1]=p1*(dPr[mm][nn][1]-da)*scale;
                dSr[mm][nn][2]=p2*(dPr[mm][nn][2]-db)*scale; dSr[mm][nn][3]=p3*(dPr[mm][nn][3]-db)*scale;
            }
        }

        // dV += P^T @ dO (out DVO). A=P^T built IN-REGISTER via movmatrix (no smem round-trip);
        // B=dO x2.trans. Per query k-tile kk: a0/a2 = keys 0-7 (n-tile 0), a1/a3 = keys 8-15
        // (n-tile 1); c0c1 -> query-lo (a0/a1), c2c3 -> query-hi (a2/a3).
        {
            uint32_t Ar[MK][4];
            #pragma unroll
            for (int kk = 0; kk < MK; ++kk) {
                Ar[kk][0] = movmatrix_t(packbf(Pr[kk][0][0], Pr[kk][0][1]));
                Ar[kk][1] = movmatrix_t(packbf(Pr[kk][1][0], Pr[kk][1][1]));
                Ar[kk][2] = movmatrix_t(packbf(Pr[kk][0][2], Pr[kk][0][3]));
                Ar[kk][3] = movmatrix_t(packbf(Pr[kk][1][2], Pr[kk][1][3]));
            }
            #pragma unroll
            for (int dd = 0; dd < DNV; ++dd) for (int kk = 0; kk < MK; ++kk) {
                uint32_t Br[2]; int vrow=kk*BM_K+(lane%16), vcol=dd*BM_N+(lane/16)*8;
                ldm_x2_trans(Br, cvta_shared(dOs + swz(vrow, vcol, DVO)));
                mma_16x8x16(dVr[dd], Ar[kk], Br, dVr[dd]);
            }
        }
        // dK += dS^T @ Q (out DQK). A=dS^T via movmatrix; B=Q x2.trans.
        {
            uint32_t Ar[MK][4];
            #pragma unroll
            for (int kk = 0; kk < MK; ++kk) {
                Ar[kk][0] = movmatrix_t(packbf(dSr[kk][0][0], dSr[kk][0][1]));
                Ar[kk][1] = movmatrix_t(packbf(dSr[kk][1][0], dSr[kk][1][1]));
                Ar[kk][2] = movmatrix_t(packbf(dSr[kk][0][2], dSr[kk][0][3]));
                Ar[kk][3] = movmatrix_t(packbf(dSr[kk][1][2], dSr[kk][1][3]));
            }
            #pragma unroll
            for (int dd = 0; dd < DNQ; ++dd) for (int kk = 0; kk < MK; ++kk) {
                uint32_t Br[2]; int qrow=kk*BM_K+(lane%16), qcol=dd*BM_N+(lane/16)*8;
                ldm_x2_trans(Br, cvta_shared(Qs + swz(qrow, qcol, DQK)));
                mma_16x8x16(dKr[dd], Ar[kk], Br, dKr[dd]);
            }
        }
        if constexpr (kCfg >= 1) {
            // ================================================================
            // CFG>=1 dQ PHASE: dQ[32 x DQK] = dS[32 x 64] @ K[64 x DQK] for this Q block,
            // reduced into the fp32 accumulator dq_acc by the bulk async engine. Runs after
            // the last ldmatrix of Qs/dOs (dV, dK GEMMs above); Sr/dPr/Pr are dead here, so
            // the phase's temporaries (dQp + one A + one B fragment) reuse their registers.
            // ================================================================
            static_assert(MM == 2 && BW_NW == 4, "dQ phase maps warp -> (m-tile = warp>>1, d-half = warp&1)");
            static_assert(DNQ % 2 == 0, "d-half split needs an even n-tile count");
            constexpr int DNH = DNQ / 2;                      // n8 tiles per d-half (12 @192, 8 @128)
            constexpr int KKN = BW_BN / BM_K;                 // k16 tiles over the 64 resident keys (4)
            constexpr int DQH_STRIDE = bw_dqh_stride<DQK>();  // padded fp32 row stride of sDQh
            constexpr int ROW_BYTES = (DQK / 2) * 4;          // bulk-reduced bytes per row (384 / 256)
            static_assert(ROW_BYTES % 16 == 0 && (DQH_STRIDE * 4) % 16 == 0, "bulk rows must be 16B multiples");

            // (1) dS exchange through sDS (bf16 [32][64], swizzled). Each warp owns 16 key
            // COLUMNS of dS; the dQ GEMM contracts over all 64 keys, so every warp needs every
            // other warp's columns. No barrier is needed BEFORE these stores: the only readers
            // of sDS are the ldmatrix loads of the previous iteration's dQ GEMM (step 2), and
            // those are ordered before this iteration by that iteration's trailing
            // __syncthreads (plus the two staging barriers that follow the GEMM). dSr already
            // carries `scale`, so dQ = dS @ K exactly as the dQ kernel computes it. The bf162
            // stores are conflict-free: the XOR swizzle spreads the 8 rows (lane/4) of one
            // store over 8 distinct 16B granules.
            #pragma unroll
            for (int mm = 0; mm < MM; ++mm) {
                const int ra = mm * BM_M + (lane / 4), rb = ra + 8;
                #pragma unroll
                for (int nn = 0; nn < WNN; ++nn) {
                    const int c = kw0 + nn * BM_N + (lane % 4) * 2;
                    *reinterpret_cast<__nv_bfloat162*>(sDS + swz(ra, c, BW_BN)) = __floats2bfloat162_rn(dSr[mm][nn][0], dSr[mm][nn][1]);
                    *reinterpret_cast<__nv_bfloat162*>(sDS + swz(rb, c, BW_BN)) = __floats2bfloat162_rn(dSr[mm][nn][2], dSr[mm][nn][3]);
                }
            }
            // BARRIER [B]: RAW on sDS across warps -- every warp's 16 columns must be visible
            // before any warp ldmatrix-reads a full 64-key row.
            __syncthreads();

            // (2) dQ GEMM: warp w -> m-tile dq_mm = w>>1 (queries dq_mm*16..+16), d-half
            // dq_nh = w&1 (n-tiles dq_nh*DNH..+DNH). A = dS[dq_mm][kk] via ldmatrix.x4 from
            // sDS (non-trans: keys are contiguous = the k dim); B = K^T via ldmatrix.x2.trans
            // from the RESIDENT Ks -- the dQ kernel's exact B-operand pattern. Padded keys
            // contribute 0 (dS = 0 there, Ks rows zeroed). kk outer / ddl inner: one A fragment
            // (4 regs) is live at a time, consecutive mmas hit independent accumulators, and
            // each dQp[ddl] still accumulates kk = 0..3 in ascending order.
            const int dq_mm = warp >> 1, dq_nh = warp & 1;
            float dQp[DNH][4];
            #pragma unroll
            for (int i = 0; i < DNH; ++i) { dQp[i][0]=dQp[i][1]=dQp[i][2]=dQp[i][3]=0.f; }
            #pragma unroll
            for (int kk = 0; kk < KKN; ++kk) {
                uint32_t Ar[4];
                { int row = dq_mm * BM_M + (lane % 16), col = kk * BM_K + (lane / 16) * 8; ldm_x4(Ar, cvta_shared(sDS + swz(row, col, BW_BN))); }
                #pragma unroll
                for (int ddl = 0; ddl < DNH; ++ddl) {
                    uint32_t Br[2]; int kr = kk * BM_K + (lane % 16), kc = (dq_nh * DNH + ddl) * BM_N + (lane / 16) * 8;
                    ldm_x2_trans(Br, cvta_shared(Ks + swz(kr, kc, DQK)));
                    mma_16x8x16(dQp[ddl], Ar, Br, dQp[ddl]);
                }
            }

            if constexpr (kCfg == 1) {
            // (3) Stage + reduce, one d-half per pass p (sDQh holds [32 x DQK/2] fp32; the
            // full [32 x DQK] tile would not fit the 99KB budget at DQK=192).
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                // (i) WAR across proxies: sDQh is about to be overwritten while the bulk engine
                // may still be READING it for the previous reduce (pass p-1 of this Q block, or
                // pass 1 of the previous Q block). Bulk async-groups are per-thread, so only the
                // issuing lanes (warp 0) can wait; .read completes as soon as the smem source
                // has been consumed (the global adds may still be in flight -- irrelevant here).
                if (warp == 0) bulk_wait_group_read<0>();
                // BARRIER [C_p]: publishes "sDQh is free" from warp 0 to the writer warps; no
                // writer may store before every lane of warp 0 has passed its wait.
                __syncthreads();
                // (ii) the two warps of this d-half (dq_mm = 0 and 1 -> rows 0-15 and 16-31)
                // store their C-fragments: row = dq_mm*16 + lane/4 (+8), col = ddl*8 +
                // (lane%4)*2 (+1), as float2 into the padded row-major tile (conflict-free, see
                // BW_DQH_PAD). Columns are LOCAL to the half; the global offset adds p*DQK/2.
                if (dq_nh == p) {
                    #pragma unroll
                    for (int ddl = 0; ddl < DNH; ++ddl) {
                        const int ra = dq_mm * BM_M + (lane / 4), rb = ra + 8, c = ddl * BM_N + (lane % 4) * 2;
                        *reinterpret_cast<float2*>(sDQh + ra * DQH_STRIDE + c) = make_float2(dQp[ddl][0], dQp[ddl][1]);
                        *reinterpret_cast<float2*>(sDQh + rb * DQH_STRIDE + c) = make_float2(dQp[ddl][2], dQp[ddl][3]);
                    }
                    // (iii) proxy fence by the WRITER threads: st.shared goes through the
                    // generic proxy while cp.reduce.async.bulk reads through the async proxy;
                    // the PTX memory model orders them only via fence.proxy.async executed by
                    // the thread that performed the writes (CUTLASS: fence_view_async_shared()
                    // before every TMA/bulk store from smem). A fence by warp 0 alone would NOT
                    // cover the other warps' stores.
                    fence_proxy_async_shared();
                }
                // BARRIER [D_p]: RAW across threads -- the fenced stores of every writer lane
                // happen-before warp 0's bulk issue below.
                __syncthreads();
                // (iv) warp 0, lane r = staging row r = query m_start + r: ONE bulk reduce of
                // the row's DQK/2 fp32 into dq_acc[(q0 + m_start + r)][h][p*DQK/2 ..]. Rows
                // r >= m_size are NOT issued: they belong to the next varlen sequence (their dS
                // is 0, but their global rows are another sequence's dQ). Each lane commits its
                // own bulk async-group so the waits in (i) and at kernel end can track it.
                if (warp == 0 && lane < m_size) {
                    float* g = dq_acc + ((int64_t)(q0 + m_start + lane) * num_heads + h) * DQK + p * (DQK / 2);
                    bulk_reduce_add_f32<ROW_BYTES>(g, cvta_shared(sDQh + lane * DQH_STRIDE));
                    bulk_commit_group();
                }
            }
            } else {
                // CFG=2 (3'): register-direct, fire-and-forget reduction of this warp's dQ
                // partial straight from the C-fragments (rows dq_mm*16 + lane/4 (+8), cols
                // dq_nh*DQK/2 + ddl*8 + (lane%4)*2): no staging, no proxy fence, no extra
                // barriers, and every warp issues in parallel -- the FROST sm120 backward's
                // dQ path (red.add.f32x2 into fp32 dq_accum). Rows >= m_size belong to the
                // next varlen sequence and are skipped. Completion is guaranteed at kernel
                // exit (stream order before the convert kernel).
                const int ra = dq_mm * BM_M + (lane / 4), rb = ra + 8;
                float* ga = dq_acc + ((int64_t)(q0 + m_start + ra) * num_heads + h) * DQK + dq_nh * (DQK / 2) + (lane % 4) * 2;
                float* gb = dq_acc + ((int64_t)(q0 + m_start + rb) * num_heads + h) * DQK + dq_nh * (DQK / 2) + (lane % 4) * 2;
                #pragma unroll
                for (int ddl = 0; ddl < DNH; ++ddl) {
                    if (ra < m_size) red_add_v2f32(ga + ddl * BM_N, dQp[ddl][0], dQp[ddl][1]);
                    if (rb < m_size) red_add_v2f32(gb + ddl * BM_N, dQp[ddl][2], dQp[ddl][3]);
                }
            }
            // The existing trailing __syncthreads below closes the Q-block body (it orders this
            // iteration's Qs/dOs reads before the next cp.async prefetch into that stage).
            // sDS/sDQh are dedicated, so only the wait_group.read in (i) protects their reuse.
        }
        }   // kCfg < 3 body
        __syncthreads();
    }

    if constexpr (kCfg >= 3) {
        // ================================================================
        // CFG=3 END-OF-KERNEL MERGE of the per-warp PARTIAL dK/dV (once per CTA, negligible vs
        // the Q loop). The two warps of key group kg3 (mh3 = 0 and 1) hold partial sums over
        // the two query halves. Staging = the Q/dO cp.async stage buffers, DEAD after the
        // loop: the last iteration waited for cp.async group 0, every ldmatrix of Qs/dOs
        // precedes the loop's trailing __syncthreads, and the zero-trip loop (mb0 >= nq)
        // issues no copy and reads nothing. Reinterpreted as fp32:
        //   STG_FLOATS = 2*BM*(DQK+DVO)*2/4 = 10,240 floats (40,960 B) @192 / 8,192 (32,768 B) @128.
        // One n8 tile from a writer half = 4 warps x 32 lanes x float4 = 512 floats, so a
        // sub-round carries at most NT_MAX = STG_FLOATS/512 tiles (20 @192 / 16 @128):
        //   round A (dK, DNQ tiles): mh3=1 writes, mh3=0 adds + stores bf16 dK
        //                            -> SUB_K sub-rounds of NT_K tiles (2 x 12 @192 = 24,576 B each;
        //                               1 x 16 @128 = 32,768 B)
        //   round B (dV, DNV tiles): mh3=0 writes, mh3=1 adds + stores bf16 dV
        //                            -> SUB_V sub-rounds of NT_V tiles (1 x 16 = 32,768 B at both)
        // Lane slot (kg3, tile t, lane) -> float4 index (kg3*NT + t)*32 + lane: one tile of one
        // warp is 512 contiguous bytes (16 B per lane, conflict-free); the reader lane loads
        // exactly the fragment its writer twin (same kg3, same lane, same C-layout coordinates)
        // stored. Barriers: [M0] before the first write (WAR against the loop's stage reads --
        // redundant with the loop's trailing barrier, kept so the proof is local), then per
        // sub-round: writers store -> [Mw] (RAW) -> readers add -> [Mr] (WAR) before the staging
        // is rewritten; the last sub-round of round B needs no [Mr] (nothing rewrites the
        // staging afterwards). fp32 sum = (mh3=0 partial) + (mh3=1 partial), a fixed pairing
        // of two deterministic per-warp sums -> dK/dV are bitwise reproducible run to run.
        // ================================================================
        constexpr int STG_FLOATS = (2 * BM * DQK + 2 * BM * DVO) * (int)sizeof(__nv_bfloat16) / (int)sizeof(float);
        constexpr int TILE_FLOATS = BW_NW * 32 * 4;                       // one n8 tile from the 4 writer warps
        constexpr int NT_MAX = STG_FLOATS / TILE_FLOATS;                  // tiles per sub-round that fit
        constexpr int SUB_K = (DNQ + NT_MAX - 1) / NT_MAX, NT_K = (DNQ + SUB_K - 1) / SUB_K;
        constexpr int SUB_V = (DNV + NT_MAX - 1) / NT_MAX, NT_V = (DNV + SUB_V - 1) / SUB_V;
        static_assert(NT_K * TILE_FLOATS <= STG_FLOATS && NT_V * TILE_FLOATS <= STG_FLOATS,
                      "CFG=3 merge sub-round exceeds the dead Q/dO stage bytes");
        static_assert(((BW_BN * DQK + BW_BN * DVO) * (int)sizeof(__nv_bfloat16)) % 16 == 0,
                      "CFG=3 merge staging (Qbuf) must start 16B-aligned for float4 traffic");
        float4* stg4 = reinterpret_cast<float4*>(Qbuf);
        const int ka = kw3 + (lane / 4), kb = ka + 8;
        __syncthreads();                                                  // [M0]
        #pragma unroll
        for (int s = 0; s < SUB_K; ++s) {                                 // round A: dK
            if (mh3 == 1) {
                #pragma unroll
                for (int t = 0; t < NT_K; ++t) {
                    const int dd = s * NT_K + t;
                    if (dd < DNQ) stg4[(kg3 * NT_K + t) * 32 + lane] = make_float4(dKr[dd][0], dKr[dd][1], dKr[dd][2], dKr[dd][3]);
                }
            }
            __syncthreads();                                              // [Mw]: the partial is visible to the reader twin
            if (mh3 == 0) {
                #pragma unroll
                for (int t = 0; t < NT_K; ++t) {
                    const int dd = s * NT_K + t;
                    if (dd < DNQ) {
                        const float4 p = stg4[(kg3 * NT_K + t) * 32 + lane];
                        dKr[dd][0] += p.x; dKr[dd][1] += p.y; dKr[dd][2] += p.z; dKr[dd][3] += p.w;
                        int dc = dd * BM_N + (lane % 4) * 2;
                        if (ka < n_size) { dk[(kv0+n_start+ka)*sk_tok + h*DQK + dc]=__float2bfloat16(dKr[dd][0]); dk[(kv0+n_start+ka)*sk_tok + h*DQK + dc+1]=__float2bfloat16(dKr[dd][1]); }
                        if (kb < n_size) { dk[(kv0+n_start+kb)*sk_tok + h*DQK + dc]=__float2bfloat16(dKr[dd][2]); dk[(kv0+n_start+kb)*sk_tok + h*DQK + dc+1]=__float2bfloat16(dKr[dd][3]); }
                    }
                }
            }
            __syncthreads();                                              // [Mr]: staging free before it is rewritten (next dK sub-round / round B)
        }
        #pragma unroll
        for (int s = 0; s < SUB_V; ++s) {                                 // round B: dV
            if (mh3 == 0) {
                #pragma unroll
                for (int t = 0; t < NT_V; ++t) {
                    const int dd = s * NT_V + t;
                    if (dd < DNV) stg4[(kg3 * NT_V + t) * 32 + lane] = make_float4(dVr[dd][0], dVr[dd][1], dVr[dd][2], dVr[dd][3]);
                }
            }
            __syncthreads();                                              // [Mw]
            if (mh3 == 1) {
                #pragma unroll
                for (int t = 0; t < NT_V; ++t) {
                    const int dd = s * NT_V + t;
                    if (dd < DNV) {
                        const float4 p = stg4[(kg3 * NT_V + t) * 32 + lane];
                        dVr[dd][0] = p.x + dVr[dd][0]; dVr[dd][1] = p.y + dVr[dd][1]; dVr[dd][2] = p.z + dVr[dd][2]; dVr[dd][3] = p.w + dVr[dd][3];
                        int dc = dd * BM_N + (lane % 4) * 2;
                        if (ka < n_size) { dv[(kv0+n_start+ka)*sv_tok + h*DVO + dc]=__float2bfloat16(dVr[dd][0]); dv[(kv0+n_start+ka)*sv_tok + h*DVO + dc+1]=__float2bfloat16(dVr[dd][1]); }
                        if (kb < n_size) { dv[(kv0+n_start+kb)*sv_tok + h*DVO + dc]=__float2bfloat16(dVr[dd][2]); dv[(kv0+n_start+kb)*sv_tok + h*DVO + dc+1]=__float2bfloat16(dVr[dd][3]); }
                    }
                }
            }
            if (s + 1 < SUB_V) __syncthreads();                           // [Mr] only if the staging is rewritten
        }
    } else {
    // write dK (DQK) and dV (DVO) for this warp's 16 keys -- no atomics
    const int ka = kw0 + (lane / 4), kb = ka + 8;
    #pragma unroll
    for (int dd = 0; dd < DNQ; ++dd) {
        int dc = dd * BM_N + (lane % 4) * 2;
        if (ka < n_size) { dk[(kv0+n_start+ka)*sk_tok + h*DQK + dc]=__float2bfloat16(dKr[dd][0]); dk[(kv0+n_start+ka)*sk_tok + h*DQK + dc+1]=__float2bfloat16(dKr[dd][1]); }
        if (kb < n_size) { dk[(kv0+n_start+kb)*sk_tok + h*DQK + dc]=__float2bfloat16(dKr[dd][2]); dk[(kv0+n_start+kb)*sk_tok + h*DQK + dc+1]=__float2bfloat16(dKr[dd][3]); }
    }
    #pragma unroll
    for (int dd = 0; dd < DNV; ++dd) {
        int dc = dd * BM_N + (lane % 4) * 2;
        if (ka < n_size) { dv[(kv0+n_start+ka)*sv_tok + h*DVO + dc]=__float2bfloat16(dVr[dd][0]); dv[(kv0+n_start+ka)*sv_tok + h*DVO + dc+1]=__float2bfloat16(dVr[dd][1]); }
        if (kb < n_size) { dv[(kv0+n_start+kb)*sv_tok + h*DVO + dc]=__float2bfloat16(dVr[dd][2]); dv[(kv0+n_start+kb)*sv_tok + h*DVO + dc+1]=__float2bfloat16(dVr[dd][3]); }
    }
    }   // kCfg < 3 epilogue
    if constexpr (kCfg == 1) {
        // Drain the issuing lanes' bulk async-groups COMPLETELY (smem reads AND global adds)
        // before the CTA exits: exiting with pending reads would release sDQh while the engine
        // is still reading it, and the stream-ordered convert kernel must observe every add.
        // Placed after the dK/dV stores so the drain overlaps those global writes.
        if (warp == 0) bulk_wait_group<0>();
    }
    }   // kCfg < 4 body
}

// ============================================================================
// Kernel B: dQ.  CTA per Q block; warps own queries; loop KV; no atomics.
// ============================================================================
template <int DQK, int DVO, bool kCausal>
__global__ void __launch_bounds__(BW_THREADS)
fmha_bwd_mma_dq_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ o,
    const __nv_bfloat16* __restrict__ d_o, const float* __restrict__ lse,
    const float* __restrict__ delta,
    const int* __restrict__ cu_q, const int* __restrict__ cu_kv,
    __nv_bfloat16* __restrict__ dq, int num_heads, float scale, int max_sq, int max_skv) {
    // Queries are warp-partitioned (16/warp) -> BM MUST be NW*16 = 64. For MLA the smem is
    // kept under 99KB by shrinking the KV LOOP block (BNB) instead of the query tile.
    constexpr int BM = BW_NW * BM_M;     // 64 queries per CTA (4 warps x 16)
    constexpr int BNB = 32;              // KV loop block (cp.async double-buffer; fits smem)
    constexpr int DS_STRIDE = BNB > 64 ? BNB : 64;    // 64; DsS stride >= 64 for swz validity
    constexpr int BW_WM = BM / BW_NW;    // 16 queries per warp
    constexpr int DKQ = DQK / BM_K;      // S contraction d-tiles (12 for 192)
    constexpr int DKV = DVO / BM_K;      // dP contraction d-tiles (8)
    constexpr int DNQ = DQK / BM_N;      // dQ output n-tiles (24 for 192)
    constexpr int NN = BNB / BM_N;       // n8 key tiles in a KV block
    constexpr int NK = BNB / BM_K;       // k16 key tiles for dQ contraction

    const int b = blockIdx.z, h = blockIdx.y, mb = blockIdx.x;
    if (h >= num_heads) return;
    const int q0 = cu_q[b], q1 = cu_q[b + 1], kv0 = cu_kv[b], kv1 = cu_kv[b + 1];
    const int sq = q1 - q0, skv = kv1 - kv0;
    const int m_start = mb * BM;
    if (m_start >= sq) return;
    const int m_size = min(BM, sq - m_start);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int qw0 = warp * BW_WM;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* Qs    = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* dOs   = Qs    + BM * DQK;        // O tile dropped (delta precomputed);
    __nv_bfloat16* Kbuf  = dOs   + BM * DVO;        // dS staging dropped (direct C-layout pack)
    __nv_bfloat16* Vbuf  = Kbuf  + 2 * BNB * DQK;   // [2][BNB*DQK],[2][BNB*DVO] cp.async
    float* lseS   = reinterpret_cast<float*>(Vbuf + 2 * BNB * DVO);
    float* deltaS = lseS + BM;

    bg2s(Qs,  q,   q0, m_start, m_size, num_heads, h, DQK);
    bg2s(dOs, d_o, q0, m_start, m_size, num_heads, h, DVO);
    for (int m = tid; m < m_size; m += BW_THREADS) {
        lseS[m]   = lse[(q0 + m_start + m) * num_heads + h];
        deltaS[m] = delta[(q0 + m_start + m) * num_heads + h];   // precomputed (no O tile)
    }
    for (int i = tid; i < (BM - m_size) * DQK; i += BW_THREADS) { int r=m_size+i/DQK, c=i%DQK; Qs[swz(r,c,DQK)]=__float2bfloat16(0.f); }
    for (int i = tid; i < (BM - m_size) * DVO; i += BW_THREADS) { int r=m_size+i/DVO, c=i%DVO; dOs[swz(r,c,DVO)]=__float2bfloat16(0.f); }
    __syncthreads();

    // Q (DQK) and dO (DVO) A-fragments for this warp's 16 queries, reused across KV loop.
    uint32_t Qrq[DKQ][4], dOrq[DKV][4];
    #pragma unroll
    for (int d = 0; d < DKQ; ++d) { int row=qw0+(lane%16), col=d*BM_K+(lane/16)*8; ldm_x4(Qrq[d], cvta_shared(Qs + swz(row, col, DQK))); }
    #pragma unroll
    for (int d = 0; d < DKV; ++d) { int row=qw0+(lane%16), col=d*BM_K+(lane/16)*8; ldm_x4(dOrq[d], cvta_shared(dOs + swz(row, col, DVO))); }
    int qa = qw0 + (lane / 4), qb = qa + 8;
    float lsea = (qa < m_size) ? lseS[qa] : INFINITY, lseb = (qb < m_size) ? lseS[qb] : INFINITY;
    float da = (qa < m_size) ? deltaS[qa] : 0.f, db = (qb < m_size) ? deltaS[qb] : 0.f;

    float dQr[DNQ][4];
    #pragma unroll
    for (int i = 0; i < DNQ; ++i) { dQr[i][0]=dQr[i][1]=dQr[i][2]=dQr[i][3]=0.f; }

    int nb_max = (skv + BNB - 1) / BNB;
    // Bottom-right causal alignment (= the forward's): the last query of this tile sees keys up to
    // last_q + (skv - sq); a negative last key means the tile sees nothing (zero-trip, dQ = 0).
    const int coff = skv - sq;
    if (kCausal) { int last_key = m_start + m_size - 1 + coff; nb_max = (last_key < 0) ? 0 : min(nb_max, (last_key + 1 + BNB - 1) / BNB); }
    const int sq_tok = num_heads * DQK;

    {   // cp.async prologue: prefetch KV block 0
        int ns0 = min(BNB, skv);
        g2s_cp(Kbuf, k, kv0, 0, ns0, num_heads, h, DQK);
        g2s_cp(Vbuf, v, kv0, 0, ns0, num_heads, h, DVO);
        cp_commit();
    }
    for (int nb = 0; nb < nb_max; ++nb) {
        const int cur = nb & 1;
        __nv_bfloat16* Ks = Kbuf + cur * (BNB * DQK);
        __nv_bfloat16* Vs = Vbuf + cur * (BNB * DVO);
        const int n_start = nb * BNB;
        const int n_size = min(BNB, skv - n_start);
        if (nb + 1 < nb_max) {
            const int nxt = (nb + 1) & 1;
            int ns2 = min(BNB, skv - (nb + 1) * BNB);
            g2s_cp(Kbuf + nxt * (BNB * DQK), k, kv0, (nb + 1) * BNB, ns2, num_heads, h, DQK);
            g2s_cp(Vbuf + nxt * (BNB * DVO), v, kv0, (nb + 1) * BNB, ns2, num_heads, h, DVO);
            cp_commit();
            cp_wait_group<1>();
        } else {
            cp_wait_group<0>();
        }
        for (int i = tid; i < (BNB - n_size) * DQK; i += BW_THREADS) { int r=n_size+i/DQK, c=i%DQK; Ks[swz(r,c,DQK)]=__float2bfloat16(0.f); }
        for (int i = tid; i < (BNB - n_size) * DVO; i += BW_THREADS) { int r=n_size+i/DVO, c=i%DVO; Vs[swz(r,c,DVO)]=__float2bfloat16(0.f); }
        __syncthreads();

        // S = scale*Q@K^T (contract DQK) ; dP = dO@V^T (contract DVO).
        float Sr[NN][4], dPr[NN][4];
        #pragma unroll
        for (int nn = 0; nn < NN; ++nn) { Sr[nn][0]=Sr[nn][1]=Sr[nn][2]=Sr[nn][3]=0.f; dPr[nn][0]=dPr[nn][1]=dPr[nn][2]=dPr[nn][3]=0.f; }
        #pragma unroll
        for (int d = 0; d < DKQ; ++d) {
            uint32_t Kd[NN][2];
            #pragma unroll
            for (int nn = 0; nn < NN; ++nn) { int kr=nn*BM_N+(lane%8), kc=d*BM_K+(lane/8%2)*8; ldm_x2(Kd[nn], cvta_shared(Ks + swz(kr, kc, DQK))); }
            #pragma unroll
            for (int nn = 0; nn < NN; ++nn) mma_16x8x16(Sr[nn], Qrq[d], Kd[nn], Sr[nn]);
        }
        #pragma unroll
        for (int d = 0; d < DKV; ++d) {
            uint32_t Vd[NN][2];
            #pragma unroll
            for (int nn = 0; nn < NN; ++nn) { int kr=nn*BM_N+(lane%8), kc=d*BM_K+(lane/8%2)*8; ldm_x2(Vd[nn], cvta_shared(Vs + swz(kr, kc, DVO))); }
            #pragma unroll
            for (int nn = 0; nn < NN; ++nn) mma_16x8x16(dPr[nn], dOrq[d], Vd[nn], dPr[nn]);
        }

        // dS = P*(dP - delta)*scale -> kept IN REGISTERS (no DsS smem round-trip).
        float dSr[NN][4];
        #pragma unroll
        for (int nn = 0; nn < NN; ++nn) {
            int c0 = nn * BM_N + (lane % 4) * 2, gca = c0, gcb = c0 + 1;
            float p0=__expf(scale*Sr[nn][0]-lsea), p1=__expf(scale*Sr[nn][1]-lsea);
            float p2=__expf(scale*Sr[nn][2]-lseb), p3=__expf(scale*Sr[nn][3]-lseb);
            bool va0=(gca<n_size)&&(!kCausal||(n_start+gca)<=(m_start+qa+coff)), va1=(gcb<n_size)&&(!kCausal||(n_start+gcb)<=(m_start+qa+coff));
            bool vb0=(gca<n_size)&&(!kCausal||(n_start+gca)<=(m_start+qb+coff)), vb1=(gcb<n_size)&&(!kCausal||(n_start+gcb)<=(m_start+qb+coff));
            p0=(va0&&qa<m_size)?p0:0.f; p1=(va1&&qa<m_size)?p1:0.f; p2=(vb0&&qb<m_size)?p2:0.f; p3=(vb1&&qb<m_size)?p3:0.f;
            dSr[nn][0]=p0*(dPr[nn][0]-da)*scale; dSr[nn][1]=p1*(dPr[nn][1]-da)*scale;
            dSr[nn][2]=p2*(dPr[nn][2]-db)*scale; dSr[nn][3]=p3*(dPr[nn][3]-db)*scale;
        }

        // dQ += dS @ K (out DQK). A=dS packed DIRECTLY from the mma C-layout (dS is the
        // un-transposed operand here): k-tile kk = key n-tiles 2kk,2kk+1; a0/a1=keys 0-7
        // (query lo/hi), a2/a3=keys 8-15. B=K x2.trans.
        uint32_t dSar[NK][4];
        #pragma unroll
        for (int kk = 0; kk < NK; ++kk) {
            dSar[kk][0] = packbf(dSr[2*kk][0],   dSr[2*kk][1]);
            dSar[kk][1] = packbf(dSr[2*kk][2],   dSr[2*kk][3]);
            dSar[kk][2] = packbf(dSr[2*kk+1][0], dSr[2*kk+1][1]);
            dSar[kk][3] = packbf(dSr[2*kk+1][2], dSr[2*kk+1][3]);
        }
        #pragma unroll
        for (int dd = 0; dd < DNQ; ++dd) for (int kk = 0; kk < NK; ++kk) {
            uint32_t Br[2]; int kr=kk*BM_K+(lane%16), kc=dd*BM_N+(lane/16)*8;
            ldm_x2_trans(Br, cvta_shared(Ks + swz(kr, kc, DQK)));
            mma_16x8x16(dQr[dd], dSar[kk], Br, dQr[dd]);
        }
        __syncthreads();
    }

    // write dQ (DQK) for this warp's 16 queries -- no atomics
    #pragma unroll
    for (int dd = 0; dd < DNQ; ++dd) {
        int dc = dd * BM_N + (lane % 4) * 2;
        if (qa < m_size) { dq[(q0+m_start+qa)*sq_tok + h*DQK + dc]=__float2bfloat16(dQr[dd][0]); dq[(q0+m_start+qa)*sq_tok + h*DQK + dc+1]=__float2bfloat16(dQr[dd][1]); }
        if (qb < m_size) { dq[(q0+m_start+qb)*sq_tok + h*DQK + dc]=__float2bfloat16(dQr[dd][2]); dq[(q0+m_start+qb)*sq_tok + h*DQK + dc+1]=__float2bfloat16(dQr[dd][3]); }
    }
}

// ---- CFG>=1: fp32 dQ accumulator -> bf16 dq (grid-stride; float4 -> 2x bf16x2) --------
// Both tensors are the flat [total_q][num_heads][DQK]; n = total_q*num_heads*DQK. `vec` is
// set by the host only when both base pointers are 16B-aligned (float4 / 2x bf162 path,
// with a scalar tail for n % 4); otherwise the scalar path handles any alignment
// (agnostic, no error-out).
__global__ void fmha_bwd_mma_dq_convert_kernel(const float* __restrict__ acc,
                                               __nv_bfloat16* __restrict__ dq,
                                               int64_t n, bool vec) {
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t start = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (vec) {
        const int64_t n4 = n >> 2;
        for (int64_t i = start; i < n4; i += stride) {
            const float4 v = reinterpret_cast<const float4*>(acc)[i];
            __nv_bfloat162* d = reinterpret_cast<__nv_bfloat162*>(dq) + 2 * i;
            d[0] = __floats2bfloat162_rn(v.x, v.y);
            d[1] = __floats2bfloat162_rn(v.z, v.w);
        }
        for (int64_t i = (n4 << 2) + start; i < n; i += stride) dq[i] = __float2bfloat16(acc[i]);
    } else {
        for (int64_t i = start; i < n; i += stride) dq[i] = __float2bfloat16(acc[i]);
    }
}

// ---- CFG=6: fragment-order fp32 dQ workspace -> dense bf16 dq --------------------------
// Block (batch b, q-tile mb, head h) of the workspace = BM x DQK floats at
// ((rbase_b + mb*BM) * num_heads + h*BM) * DQK, rbase_b = sum over earlier batches of ceil(sq/BM)*BM
// (every batch padded to whole tiles -> blocks never overlap). Inside a block, 64-float chunk
// c = ((warp*2 + mt)*QNF + nf)*2 + half holds lane*2 + {0,1} = dQ[row][col] with
// row = qm(warp)*32 + mt*16 + half*8 + lane/4, col = qn(warp)*DQW + nf*8 + (lane%4)*2 -- the exact
// red.v2 pattern of fmha_bwd_mma_dkdv_cfg4_body<kCoal>. Coalesced float2 reads, bf162 writes.
template <int DQK, int BM, bool kV4 = false>
__global__ void fmha_bwd_mma_dq_unscramble_kernel(const float* __restrict__ acc,
                                                  __nv_bfloat16* __restrict__ dq,
                                                  const int* __restrict__ cu_q, int num_heads) {
    const int b = blockIdx.z, h = blockIdx.y, mb = blockIdx.x;
    if (h >= num_heads) return;
    const int q0 = cu_q[b], sq = cu_q[b + 1] - q0;
    const int m_start = mb * BM;
    if (m_start >= sq) return;
    int rbase = 0;
    for (int bb = 0; bb < b; ++bb) rbase += ((cu_q[bb + 1] - cu_q[bb]) + BM - 1) / BM * BM;
    const float* blk = acc + ((int64_t)(rbase + m_start) * num_heads + (int64_t)h * BM) * DQK;   // head block = h*BM rows in
    constexpr int DQW = (DQK == 128) ? 32 : 24, QNF = DQW / 8;
    constexpr int NCH = 8 * 2 * QNF * 2;          // 64-float chunks per block: (warp, m-tile, n-frag, half)
    static_assert(NCH * 64 == BM * DQK, "fragment-order block must tile the dense BM x DQK tile exactly");
    const int m_size = sq - m_start;
    if constexpr (kV4) {
        // CFG=8 layout: run (w, mt, half) of QNF*64 floats = (QNF/2) 128-float pair chunks + (QNF odd) a 64-float chunk.
        constexpr int RUN = QNF * 64, NRUN = 8 * 2 * 2;
        for (int i = threadIdx.x; i < NRUN * (RUN / 2); i += blockDim.x) {   // one float2 per iteration
            const int r = i / (RUN / 2), t = (i % (RUN / 2)) * 2;           // run index, float offset in the run
            const int half = r & 1, mt = (r >> 1) & 1, w = r >> 2;
            const int qm = (DQK == 128) ? (w & 1) : 0, qn = (DQK == 128) ? (w >> 1) : w;
            int nf, lane;
            if (t < 128 * (QNF / 2)) { const int j = t / 128, u = t % 128; lane = u / 4; nf = 2 * j + ((u % 4) >> 1); }
            else { const int u = t - 128 * (QNF / 2); lane = u / 2; nf = QNF - 1; }
            const int row = qm * 32 + mt * 16 + half * 8 + (lane >> 2);
            const int col = qn * DQW + nf * 8 + (lane & 3) * 2;
            if (row < m_size) {
                const float2 v = *reinterpret_cast<const float2*>(blk + r * RUN + t);
                *reinterpret_cast<__nv_bfloat162*>(dq + ((int64_t)(q0 + m_start + row) * num_heads + h) * DQK + col) =
                    __floats2bfloat162_rn(v.x, v.y);
            }
        }
        return;
    }
    for (int i = threadIdx.x; i < NCH * 32; i += blockDim.x) {
        const int c = i >> 5, lane = i & 31;
        const int half = c & 1, t = c >> 1;
        const int nf = t % QNF, t2 = t / QNF;
        const int mt = t2 & 1, w = t2 >> 1;
        const int qm = (DQK == 128) ? (w & 1) : 0, qn = (DQK == 128) ? (w >> 1) : w;
        const int row = qm * 32 + mt * 16 + half * 8 + (lane >> 2);
        const int col = qn * DQW + nf * 8 + (lane & 3) * 2;
        if (row < m_size) {
            const float2 v = *reinterpret_cast<const float2*>(blk + c * 64 + lane * 2);
            *reinterpret_cast<__nv_bfloat162*>(dq + ((int64_t)(q0 + m_start + row) * num_heads + h) * DQK + col) =
                __floats2bfloat162_rn(v.x, v.y);
        }
    }
}

// ---- Launchers --------------------------------------------------------------
// kCfg = FLASH_MLA_SM120_BWD_MMA_CFG: 0 = two-kernel split (byte-identical legacy path);
// 1 = single-kernel dQ via cp.reduce.async.bulk into the fp32 workspace + convert kernel
// (workspace_buffer is only touched at kCfg >= 1); 2 = single-kernel, dQ via red.v2;
// 3 = CFG=2 with the 8-warp CTA (block dim BwCta<3>::kThreads = 256; grid and smem unchanged);
// 4 = FROST-structured single kernel (block dim BwCta<4>::kThreads = 256, smem bw4_smem<>():
// 82,464 B @192 / 99,376 B @128; same grid, same dq_acc / memset / convert contract as 2 and 3).
template <int DQK, int DVO, bool kCausal, int kCfg>
void launch_fmha_bwd_mma(const c10::cuda::CUDAStream& stream,
                         at::Tensor d_o, at::Tensor q, at::Tensor k, at::Tensor v,
                         at::Tensor o, at::Tensor lse, at::Tensor cu_q, at::Tensor cu_kv,
                         at::Tensor dq, at::Tensor dk, at::Tensor dv,
                         float scale, int max_sq, int max_skv, at::Tensor workspace_buffer) {
    constexpr int BM = 32;   // kernel A query loop (matches the kernel; cp.async double-buf)
    constexpr int PT_STRIDE = BM > 64 ? BM : 64;
    const int batch = cu_q.size(0) - 1, num_heads = q.size(1);
    const int total_q = q.size(0);
    const auto qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    const auto kp = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
    const auto vp = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
    const auto op = reinterpret_cast<const __nv_bfloat16*>(o.data_ptr());
    const auto dop = reinterpret_cast<const __nv_bfloat16*>(d_o.data_ptr());
    const auto lp = lse.data_ptr<float>();
    const auto cq = cu_q.data_ptr<int>(); const auto ck = cu_kv.data_ptr<int>();

    // CFG>=1: fp32 dQ accumulator = the caller's per-call workspace (flash_mla_interface.py
    // allocates >= 4*bs*max_seqlen_qo_aligned*H*DQK bytes per call as a fresh torch.empty ->
    // cudagraph_trees-safe), laid out [total_q][num_heads][DQK] from offset 0 -- the same
    // flat indexing the bf16 dq uses -- and zeroed here (memset nodes are capture-legal).
    // Stream order: memset -> delta -> dKdV+dQ kernel -> convert.
    float* dq_acc = nullptr;
    [[maybe_unused]] int64_t acc_elems = 0;
    // CFG=6 (kCoalWs): fragment-order workspace, every batch padded to whole BM-row tiles; the
    // capacity needed is bounded by batch * ceil(max_sq/BM) * BM rows (no host sync on cu_seqlens).
    [[maybe_unused]] constexpr bool kCoalWs = (kCfg >= 6 && kCfg != 95);
    [[maybe_unused]] constexpr int BM6 = Bw4<DQK, DVO>::BM;
    if constexpr (kCfg >= 1) {
        acc_elems = kCoalWs ? (int64_t)batch * (((int64_t)max_sq + BM6 - 1) / BM6) * BM6 * num_heads * DQK
                            : (int64_t)total_q * num_heads * DQK;
        const int64_t acc_bytes = acc_elems * (int64_t)sizeof(float);
        TORCH_CHECK(workspace_buffer.is_cuda() && workspace_buffer.device() == q.device(),
                    "flash_mla sm120 bwd CFG=1: workspace_buffer must live on q's CUDA device");
        TORCH_CHECK(workspace_buffer.numel() * workspace_buffer.element_size() >= acc_bytes,
                    "flash_mla sm120 bwd CFG=1: workspace_buffer too small for the fp32 dQ accumulator: need ",
                    acc_bytes, " bytes (total_q=", total_q, " x heads=", num_heads, " x DQK=", DQK,
                    " x 4), got ", workspace_buffer.numel() * workspace_buffer.element_size());
        TORCH_CHECK(dq.is_contiguous(),
                    "flash_mla sm120 bwd CFG=1: dq must be a contiguous [total_q, heads, DQK] tensor");
        dq_acc = reinterpret_cast<float*>(workspace_buffer.data_ptr());
        TORCH_CHECK((reinterpret_cast<uintptr_t>(dq_acc) & 15) == 0,
                    "flash_mla sm120 bwd CFG=1: workspace_buffer must be 16B-aligned (cp.reduce.async.bulk destination)");
        C10_CUDA_CHECK(cudaMemsetAsync(dq_acc, 0, acc_bytes, stream.stream()));
    }

    // Preprocess: delta[token,head] = sum_d O*dO. One pass; frees the O smem tile in the
    // two main kernels so they can cp.async-pipeline the looped operand.
    auto delta = at::empty({total_q, num_heads}, q.options().dtype(at::kFloat));
    const auto dp = delta.data_ptr<float>();
    {
        const int n_th = total_q * num_heads;
        const int tpb = 256;                          // 8 warps -> 8 tokens/block
        const int gb = (n_th + (tpb / 32) - 1) / (tpb / 32);
        fmha_bwd_mma_delta_kernel<<<gb, tpb, 0, stream.stream()>>>(op, dop, dp, n_th, DVO);
    }

    {   // Kernel A: dK/dV, grid over KV blocks
        const int nblocks = (max_skv + BW_BN - 1) / BW_BN;
        dim3 grid(nblocks, num_heads, batch);
        size_t smem = (BW_BN * DQK + BW_BN * DVO + 2 * BM * DQK + 2 * BM * DVO) * sizeof(__nv_bfloat16) + (2 * BM) * sizeof(float);
        if constexpr (kCfg >= 1) smem += bw_dq_phase_smem<DQK, BM, kCfg>();   // + sDS (+ sDQh at CFG=1)
        if constexpr (kCfg >= 4) smem = bw4_smem<DQK, DVO>();                  // CFG=4: its own map (Bw4<>)
        auto kern = fmha_bwd_mma_dkdv_kernel<DQK, DVO, kCausal, kCfg>;
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kern<<<grid, dim3(BwCta<kCfg>::kThreads), smem, stream.stream()>>>(qp, kp, vp, op, dop, lp, dp, cq, ck,
            reinterpret_cast<__nv_bfloat16*>(dk.data_ptr()), reinterpret_cast<__nv_bfloat16*>(dv.data_ptr()), num_heads, scale, max_sq, max_skv, dq_acc);
        if constexpr (kCfg >= 1) C10_CUDA_CHECK(cudaGetLastError());
        if constexpr (kCfg == 94) {   // timing diag: sync, read the per-phase cycle sums, print, reset
            C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            unsigned long long c[16] = {};
            C10_CUDA_CHECK(cudaMemcpyFromSymbol(c, bw_diag_cycles, sizeof(c)));
            static const char* names[BWD_T_NPHASE] = {"issue(L1)", "wait_full_q", "S+dP", "VJP+slabs", "B1_barrier", "dV", "dO_handoff", "dK", "dQ_gemm", "reds"};
            unsigned long long tot = 0;
            for (int i = 0; i < BWD_T_NPHASE; ++i) tot += c[i];
            const double per_tile_warp = (c[BWD_T_TILES] > 0) ? static_cast<double>(tot) / static_cast<double>(c[BWD_T_TILES]) : 0.0;
            fprintf(stderr, "[BWD-TIMING DQK=%d causal=%d] warps=%llu tiles/warp=%.1f cycles/tile/warp=%.0f |", DQK, static_cast<int>(kCausal), c[BWD_T_WARPS],
                    c[BWD_T_WARPS] ? static_cast<double>(c[BWD_T_TILES]) / static_cast<double>(c[BWD_T_WARPS]) : 0.0, per_tile_warp);
            for (int i = 0; i < BWD_T_NPHASE; ++i) fprintf(stderr, " %s %.1f%%", names[i], tot ? 100.0 * static_cast<double>(c[i]) / static_cast<double>(tot) : 0.0);
            fprintf(stderr, "%s", "\n");
            unsigned long long z[16] = {};
            C10_CUDA_CHECK(cudaMemcpyToSymbol(bw_diag_cycles, z, sizeof(z)));
        }
    }
    if constexpr (kCoalWs) {     // CFG=6: un-scramble the fragment-order workspace into bf16 dq
        const int qtiles = (max_sq + BM6 - 1) / BM6;
        dim3 grid(qtiles, num_heads, batch);
        constexpr bool kV4Ws = ((kCfg >= 8 && kCfg != 95) || kCfg == 94);
        fmha_bwd_mma_dq_unscramble_kernel<DQK, BM6, kV4Ws><<<grid, dim3(256), 0, stream.stream()>>>(
            dq_acc, reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()), cq, num_heads);
        C10_CUDA_CHECK(cudaGetLastError());
    } else if constexpr (kCfg >= 1) {   // Convert: bf16 dq = fp32 accumulator. Kernel B is NOT launched.
        const bool vec = (reinterpret_cast<uintptr_t>(dq.data_ptr()) & 15) == 0;   // dq_acc checked above
        const int64_t work = vec ? (acc_elems >> 2) : acc_elems;
        const int tpb = 256;
        const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>((work + tpb - 1) / tpb, 65535)));
        fmha_bwd_mma_dq_convert_kernel<<<blocks, tpb, 0, stream.stream()>>>(
            dq_acc, reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()), acc_elems, vec);
        C10_CUDA_CHECK(cudaGetLastError());
    } else
    {   // Kernel B: dQ, grid over Q blocks (queries warp-partitioned -> BMB=64; KV loop BNB).
        constexpr int BMB = BW_NW * BM_M;            // 64
        constexpr int BNB = 32;
        const int nblocks = (max_sq + BMB - 1) / BMB;
        dim3 grid(nblocks, num_heads, batch);
        size_t smem = (BMB * DQK + BMB * DVO + 2 * BNB * DQK + 2 * BNB * DVO) * sizeof(__nv_bfloat16) + (2 * BMB) * sizeof(float);
        auto kern = fmha_bwd_mma_dq_kernel<DQK, DVO, kCausal>;
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kern<<<grid, dim3(BW_THREADS), smem, stream.stream()>>>(qp, kp, vp, op, dop, lp, dp, cq, ck,
            reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()), num_heads, scale, max_sq, max_skv);
    }
}

}  // namespace mma_bwd
}  // namespace detail
}  // namespace flash
