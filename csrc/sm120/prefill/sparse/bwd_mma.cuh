#pragma once
// ============================================================================
// SPARSE PREFILL BACKWARD via raw mma.sync + ldmatrix + cp.async (SM120) [CFG>=1]
// ============================================================================
// The mma tier of the sparse backward: same math as the green WMMA kernel in
// bwd.cu (FA-2 formulation, arXiv:2307.08691), same CTA decomposition
// (one CTA = one query position x 64-head block), rebuilt on the proven
// fwd_mma.cuh machinery. Per topk block of 64 gathered tokens:
//
//   S  = Q K^T                  A = Q   (resident sQ, x4)      B = K  (x2)
//   P  = exp2(S*scale2 - lse2)  (stored GLOBAL 2-based LSE, valid-masked)
//   dP = dO V^T                 A = dO  (sDO, x4)              B = V  (x2)
//   dV = P^T dO   (scatter)     A = P^T (sPS, x4 TRANS)        B = dO (x2 trans)
//   dS = P (dP - D) * sm_scale  (D = rowsum(dO*O), once per CTA)
//   dQ = dS K     (exclusive)   A = dS  (sPS, x4)              B = K  (x2 trans)
//   dK = dS^T Q   (scatter)     A = dS^T(sPS, x4 TRANS)        B = Q  (resident, x2 trans)
//
// Key structural wins over the WMMA kernel:
//   - sQ RESIDENT [64 x 576] swizzled: the WMMA kernel reloaded Q 18x per block.
//   - dQ REGISTER-RESIDENT across the whole topk loop (dQr[9][4][4] = 144 fp32),
//     written ONCE at the end as a PURE STORE (rows are CTA-exclusive and the
//     n-half warp split makes every element single-writer) -- the WMMA kernel
//     did a gmem fp32 RMW round trip EVERY block.
//   - dP and dV fused into one V/dO streaming phase (shared tile loads).
//   - P and dS share one smem tile (sPS): P is consumed before dS overwrites it.
//   - dK/dV remain fp32 atomicAdd scatter (inherent: tokens are shared across
//     query positions), with the zero-skip filter kept.
//
// Smem: sQ 73,728 + bufA 8,192 + bufB 8,192 + sPS 8,192 + stats/ptrs/idx/pad 1,408
//       -> sizeof = 99,712 B <= 101,376 (99KB opt-in), 1 CTA/SM.
// ============================================================================

#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

namespace sm120 {
namespace sparse_bwd_mma {

constexpr int BM_MMA_M = 16, BM_MMA_N = 8, BM_MMA_K = 16;
constexpr int BM_NW = 8;
constexpr int BM_THREADS = BM_NW * 32;      // 256
constexpr int BM_BH = 64;                   // head rows per CTA
constexpr int BM_BTOPK = 64;                // gathered tokens per block
constexpr int BM_DQK = 576, BM_DV = 512, BM_DTILE = 64;
constexpr int BM_NKTILES = BM_DQK / BM_DTILE;   // 9
constexpr int BM_NVTILES = BM_DV / BM_DTILE;    // 8
constexpr float BM_LOG2E = 1.4426950408889634f;

// ---- PTX primitives (identical to fwd_mma.cuh) ------------------------------
__device__ __forceinline__ uint32_t b_cvta(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void b_ldm_x4(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void b_ldm_x4_trans(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void b_ldm_x2(uint32_t (&r)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void b_ldm_x2_trans(uint32_t (&r)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void b_mma(float (&d)[4], const uint32_t (&a)[4],
                                      const uint32_t (&b)[2], const float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ void b_cp16(uint32_t sa, const void* g, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 ::"r"(sa), "l"(g), "r"(src_bytes));
}
// Vectorized fire-and-forget global reduction (Blackwell-class; ptxas-verified on
// plain sm_120 under CUDA 12.9 AND 13.0 -- see tests/_probe_blackwell_reducers.sh).
// Same fp32-add reduction semantics as atomicAdd, half the transactions, and the SM
// never waits on a returned prior value. Target must be 8B-aligned (col even).
// NO "memory" clobber: dk/dv are write-only reduction sinks the kernel never reads
// back, and `volatile` alone pins ordering among the other volatile asm (ldmatrix/
// cp.async). A clobber here serializes scheduling across every scatter call and was
// measured to push the 255-reg kernel into a 20 B local-memory spill.
__device__ __forceinline__ void b_red_v2(float* addr, float a, float b) {
    asm volatile("red.global.add.v2.f32 [%0], {%1, %2};\n"
                 ::"l"(addr), "f"(a), "f"(b));
}
// 16B variant (CFG>=3): a lane PAIR fuses its two adjacent col-pairs into one
// red.global.add.v4.f32 (ptxas-verified on plain sm_120, same probe). Target
// must be 16B-aligned -- even lanes' (lane%4)*2 col offset is 0 or 4 floats
// and every other address term is a multiple of 8 floats, so even-lane
// addresses are 16B-clean by construction. Same no-clobber rationale as v2.
__device__ __forceinline__ void b_red_v4(float* addr, float a, float b, float c, float d) {
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n"
                 ::"l"(addr), "f"(a), "f"(b), "f"(c), "f"(d));
}
__device__ __forceinline__ void b_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void b_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ int b_swz(int r, int c, int dim) {
    return r * dim + (c ^ ((r & 7) << 3));
}

// ---- smem plan --------------------------------------------------------------
// bufA/bufB: the two 64x64 bf16 streaming tiles. QK and dK/dQ phases use them as
// the 2-stage gathered-K pipeline; the fused dP+dV phase uses bufA for the
// (single-staged) gathered V tile and bufB for the dense dO tile.
struct SmemPlanBwdMma {
    __align__(128) __nv_bfloat16 sQ[BM_BH * BM_DQK];         // 73,728 B, swz(576)
    __align__(128) __nv_bfloat16 bufA[BM_BTOPK * BM_DTILE];  // 8,192 B, swz(64)
    __align__(128) __nv_bfloat16 bufB[BM_BTOPK * BM_DTILE];  // 8,192 B, swz(64)
    __align__(128) __nv_bfloat16 sPS[BM_BH * BM_BTOPK];      // 8,192 B, swz(64): P then dS
    __align__(128) float sLse[BM_BH];                        // 2-based LSE (forward)
    __align__(128) float sD[BM_BH];                          // FA-2 D-term
    __align__(128) const __nv_bfloat16* sTokPtr[BM_BTOPK];
    __align__(128) int sTok[BM_BTOPK];                       // token index (for scatter)
    __align__(128) int8_t sValid[BM_BTOPK];
};
static_assert(sizeof(SmemPlanBwdMma) <= 99 * 1024, "sparse bwd mma smem exceeds 99KB");

// Gather one 64-token x 64-col bf16 tile from the KV pool rows into `dst`,
// swizzled, predicated (src-size 0 zero-fill for invalid). Same as fwd_mma.
__device__ __forceinline__ void b_gather_tile(SmemPlanBwdMma& sm, __nv_bfloat16* dst,
                                              int col_start) {
    const int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int g = tid + i * BM_THREADS;
        const int r = g >> 3;
        const int c = (g & 7) * 8;
        const __nv_bfloat16* tp = sm.sTokPtr[r];
        b_cp16(b_cvta(&dst[b_swz(r, c, BM_DTILE)]), tp + col_start + c,
               sm.sValid[r] ? 16 : 0);
    }
}

// Load a dense 64-row x 64-col bf16 tile (rows = heads: dO) into `dst`, swizzled.
__device__ __forceinline__ void b_load_rows_tile(SmemPlanBwdMma& sm, __nv_bfloat16* dst,
                                                 const __nv_bfloat16* gptr, int stride_h,
                                                 int col_start) {
    const int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int g = tid + i * BM_THREADS;
        const int r = g >> 3;
        const int c = (g & 7) * 8;
        *reinterpret_cast<int4*>(&dst[b_swz(r, c, BM_DTILE)]) =
            *reinterpret_cast<const int4*>(gptr + (int64_t)r * stride_h + col_start + c);
    }
}

// ---- kernel -----------------------------------------------------------------
// VRED (CFG>=2): dK/dV scatter via red.global.add.v2.f32 instead of scalar
// atomicAdd pairs -- identical reduction semantics, vectorized transactions.
// VRED4 (CFG>=3): lane pairs (l, l^1) exchange their col-pairs via shfl and the
// even lane issues ONE red.global.add.v4.f32 per row -- half the reduction
// transactions again. The quad shares trow_a/trow_b (lane/4 equal across the
// pair) so va/vb/ta/tb are pair-uniform and the shfl is convergence-safe.
template <bool VRED, bool VRED4>
__global__ void __launch_bounds__(BM_THREADS, 1)
sparse_prefill_bwd_mma_kernel(const SparsePrefillBwdParams params) {
    extern __shared__ char smem_raw[];
    SmemPlanBwdMma& sm = *reinterpret_cast<SmemPlanBwdMma*>(smem_raw);

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int num_h_blocks = params.h_q / BM_BH;
    const int h_block = blockIdx.x % num_h_blocks;
    const int s_q_idx = blockIdx.x / num_h_blocks;

    const __nv_bfloat16* q_ptr = reinterpret_cast<const __nv_bfloat16*>(params.q)
        + (int64_t)s_q_idx * params.stride_q_s_q + (int64_t)h_block * BM_BH * params.stride_q_h_q;
    const __nv_bfloat16* do_ptr = reinterpret_cast<const __nv_bfloat16*>(params.d_o)
        + (int64_t)s_q_idx * params.stride_do_s_q + (int64_t)h_block * BM_BH * params.stride_do_h_q;
    const __nv_bfloat16* o_ptr = reinterpret_cast<const __nv_bfloat16*>(params.o)
        + (int64_t)s_q_idx * params.stride_o_s_q + (int64_t)h_block * BM_BH * params.stride_o_h_q;
    const __nv_bfloat16* kv_base = reinterpret_cast<const __nv_bfloat16*>(params.kv);
    const int* idx_row = params.indices + (int64_t)s_q_idx * params.stride_indices_s_q;
    float* dq_ptr = params.dq + (int64_t)s_q_idx * params.stride_dq_s_q
        + (int64_t)h_block * BM_BH * params.stride_dq_h_q;

    // ---- prologue: resident swizzled Q, LSE, D-term ------------------------
    {
        #pragma unroll
        for (int i = 0; i < 18; ++i) {
            const int g = tid + i * BM_THREADS;
            const int r = g / (BM_DQK / 8);
            const int c = (g % (BM_DQK / 8)) * 8;
            *reinterpret_cast<int4*>(&sm.sQ[b_swz(r, c, BM_DQK)]) =
                *reinterpret_cast<const int4*>(q_ptr + (int64_t)r * params.stride_q_h_q + c);
        }
    }
    if (tid < BM_BH) {
        sm.sLse[tid] = params.lse[(int64_t)s_q_idx * params.h_q + h_block * BM_BH + tid];
    }
    {   // D[h] = sum_v dO[h,v] * O[h,v], fp32, 4 threads per head row (shfl-reduced)
        const int row = tid / 4, sub = tid % 4;
        float part = 0.f;
        for (int v = sub * 8; v < BM_DV; v += 32) {
            #pragma unroll
            for (int u = 0; u < 8; ++u)
                part += __bfloat162float(do_ptr[(int64_t)row * params.stride_do_h_q + v + u])
                      * __bfloat162float(o_ptr[(int64_t)row * params.stride_o_h_q + v + u]);
        }
        part += __shfl_xor_sync(0xffffffff, part, 1);
        part += __shfl_xor_sync(0xffffffff, part, 2);
        if (sub == 0) sm.sD[row] = part;
    }
    __syncthreads();

    const int ms = warp & 3, nh = warp >> 2;
    const int wrow = ms * BM_MMA_M;

    // dQ accumulator, register-resident across the whole topk loop:
    // dQr[dt][n][.] covers rows {wrow+lane/4, +8} x cols {dt*64 + nh*32 + n*8 + (lane%4)*2, +1}
    float dQr[BM_NKTILES][4][4];
    #pragma unroll
    for (int t = 0; t < BM_NKTILES; ++t)
        #pragma unroll
        for (int n = 0; n < 4; ++n) { dQr[t][n][0] = dQr[t][n][1] = dQr[t][n][2] = dQr[t][n][3] = 0.f; }

    const int num_topk_blocks = (params.topk + BM_BTOPK - 1) / BM_BTOPK;

    for (int kb = 0; kb < num_topk_blocks; ++kb) {
        // ---- token pointers + validity -------------------------------------
        if (tid < BM_BTOPK) {
            const int k = kb * BM_BTOPK + tid;
            const int token = (k < params.topk) ? idx_row[k] : -1;
            const bool ok = (token >= 0 && token < params.s_kv);
            sm.sValid[tid] = ok ? 1 : 0;
            sm.sTok[tid] = token;
            sm.sTokPtr[tid] = ok ? (kv_base + (int64_t)token * params.stride_kv_s_kv)
                                 : kv_base;
        }
        __syncthreads();

        // ---- S = Q K^T over 9 pipelined gathered K d-tiles (fwd-identical) --
        float Cr[4][4];   // transient GEMM accumulator (S here, dP/dV/dK later)
        #pragma unroll
        for (int n = 0; n < 4; ++n) { Cr[n][0] = Cr[n][1] = Cr[n][2] = Cr[n][3] = 0.f; }

        b_gather_tile(sm, sm.bufA, 0);
        b_cp_commit();
        for (int dt = 0; dt < BM_NKTILES; ++dt) {
            __nv_bfloat16* cur = (dt & 1) ? sm.bufB : sm.bufA;
            __nv_bfloat16* nxt = (dt & 1) ? sm.bufA : sm.bufB;
            if (dt + 1 < BM_NKTILES) {
                b_gather_tile(sm, nxt, (dt + 1) * BM_DTILE);
                b_cp_commit();
                b_cp_wait<1>();
            } else {
                b_cp_wait<0>();
            }
            __syncthreads();
            #pragma unroll
            for (int dl = 0; dl < 4; ++dl) {
                uint32_t Ar[4];
                {
                    const int row = wrow + (lane % 16);
                    const int col = dt * BM_DTILE + dl * BM_MMA_K + (lane / 16) * 8;
                    b_ldm_x4(Ar, b_cvta(&sm.sQ[b_swz(row, col, BM_DQK)]));
                }
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    uint32_t Br[2];
                    const int krow = (nh * 4 + n) * BM_MMA_N + (lane % 8);
                    const int kcol = dl * BM_MMA_K + ((lane / 8) % 2) * 8;
                    b_ldm_x2(Br, b_cvta(&cur[b_swz(krow, kcol, BM_DTILE)]));
                    b_mma(Cr[n], Ar, Br, Cr[n]);
                }
            }
            __syncthreads();   // stage consumed before dt+2 overwrites it
        }

        // ---- P = exp2(S*scale2 - lse2), keep in regs AND write sPS ----------
        // lse2 is the forward's GLOBAL 2-based LSE; !isfinite(lse2) (all-invalid
        // row, lse = -inf) forces P = 0 for the whole row (WMMA-identical).
        float Pr[4][4];
        {
            const float scale2 = params.sm_scale * BM_LOG2E;
            const int row_a = wrow + lane / 4, row_b = row_a + 8;
            const float lse_a = sm.sLse[row_a], lse_b = sm.sLse[row_b];
            const bool fa = isfinite(lse_a), fb = isfinite(lse_b);
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int col0 = (nh * 4 + n) * BM_MMA_N + (lane % 4) * 2;
                const bool v0 = sm.sValid[col0] != 0, v1 = sm.sValid[col0 + 1] != 0;
                Pr[n][0] = (v0 && fa) ? exp2f(Cr[n][0] * scale2 - lse_a) : 0.f;
                Pr[n][1] = (v1 && fa) ? exp2f(Cr[n][1] * scale2 - lse_a) : 0.f;
                Pr[n][2] = (v0 && fb) ? exp2f(Cr[n][2] * scale2 - lse_b) : 0.f;
                Pr[n][3] = (v1 && fb) ? exp2f(Cr[n][3] * scale2 - lse_b) : 0.f;
                *reinterpret_cast<__nv_bfloat162*>(&sm.sPS[b_swz(row_a, col0, BM_BTOPK)]) =
                    __floats2bfloat162_rn(Pr[n][0], Pr[n][1]);
                *reinterpret_cast<__nv_bfloat162*>(&sm.sPS[b_swz(row_b, col0, BM_BTOPK)]) =
                    __floats2bfloat162_rn(Pr[n][2], Pr[n][3]);
            }
        }
        // (the first fused-phase __syncthreads below publishes sPS before its use)

        // ---- fused dP + dV over 8 V/dO tiles --------------------------------
        // dP (accumulated in Cr): A = dO x4, B = V x2 (non-trans).
        // dV (transient Dr per tile): A = P^T via x4 TRANS on sPS, B = dO x2 trans;
        //   output rows = TOKENS (warp strip ms), cols = this tile's nh 32-col half;
        //   scattered to global dv by atomicAdd (zero-skip kept).
        #pragma unroll
        for (int n = 0; n < 4; ++n) { Cr[n][0] = Cr[n][1] = Cr[n][2] = Cr[n][3] = 0.f; }

        b_gather_tile(sm, sm.bufA, 0);        // V tile 0 (cols [0,64) of the KV rows)
        b_cp_commit();
        for (int vt = 0; vt < BM_NVTILES; ++vt) {
            b_load_rows_tile(sm, sm.bufB, do_ptr, params.stride_do_h_q, vt * BM_DTILE);
            b_cp_wait<0>();
            __syncthreads();                   // V + dO + sPS all visible
            // dP partial: contract this 64-wide v-slice
            #pragma unroll
            for (int dl = 0; dl < 4; ++dl) {
                uint32_t Ar[4];
                {
                    const int row = wrow + (lane % 16);
                    const int col = dl * BM_MMA_K + (lane / 16) * 8;
                    b_ldm_x4(Ar, b_cvta(&sm.bufB[b_swz(row, col, BM_DTILE)]));
                }
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    uint32_t Br[2];
                    const int vrow = (nh * 4 + n) * BM_MMA_N + (lane % 8);
                    const int vcol = dl * BM_MMA_K + ((lane / 8) % 2) * 8;
                    b_ldm_x2(Br, b_cvta(&sm.bufA[b_swz(vrow, vcol, BM_DTILE)]));
                    b_mma(Cr[n], Ar, Br, Cr[n]);
                }
            }
            // dV tile: rows = tokens [ms*16, +16), cols = [nh*32, nh*32+32) of vt
            {
                float Dr[4][4];
                #pragma unroll
                for (int n = 0; n < 4; ++n) { Dr[n][0] = Dr[n][1] = Dr[n][2] = Dr[n][3] = 0.f; }
                #pragma unroll
                for (int hk = 0; hk < 4; ++hk) {          // contract 64 heads = 4 k16
                    uint32_t Ar[4];
                    {
                        // A = P^T: x4 TRANS on sPS (stored [head][token]).
                        // Tile t-rows wrow..+16, k-heads hk*16..+16. Addresses walk
                        // stored rows (heads); trans distribution transposes each 8x8.
                        const int prow = hk * BM_MMA_K + (lane / 16) * 8 + (lane % 8);
                        const int pcol = wrow + ((lane / 8) % 2) * 8;
                        b_ldm_x4_trans(Ar, b_cvta(&sm.sPS[b_swz(prow, pcol, BM_BTOPK)]));
                    }
                    #pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        uint32_t Br[2];
                        // B[k=h][n=v] = dO[h][v]: stored rows = k -> x2 trans
                        const int drow = hk * BM_MMA_K + (lane % 16);
                        const int dcol = nh * 32 + n * BM_MMA_N + (lane / 16) * 8;
                        b_ldm_x2_trans(Br, b_cvta(&sm.bufB[b_swz(drow, dcol, BM_DTILE)]));
                        b_mma(Dr[n], Ar, Br, Dr[n]);
                    }
                }
                const int trow_a = wrow + lane / 4, trow_b = trow_a + 8;
                const bool va = sm.sValid[trow_a] != 0, vb = sm.sValid[trow_b] != 0;
                const int64_t ta = va ? (int64_t)sm.sTok[trow_a] * params.stride_dv_s_kv : 0;
                const int64_t tb = vb ? (int64_t)sm.sTok[trow_b] * params.stride_dv_s_kv : 0;
                // VRED: hoisted bases + NO per-pair zero-skip (va/vb already gate the
                // masked-row traffic the skip existed for; +0.0 from a valid token is
                // a no-op) -- keeps the 255-reg kernel out of local-memory spill.
                float* const dva = params.dv + ta + vt * BM_DTILE + nh * 32 + (lane % 4) * 2;
                float* const dvb = params.dv + tb + vt * BM_DTILE + nh * 32 + (lane % 4) * 2;
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const int vcol = vt * BM_DTILE + nh * 32 + n * BM_MMA_N + (lane % 4) * 2;
                    if constexpr (VRED4) {
                        const float oa0 = __shfl_xor_sync(0xffffffff, Dr[n][0], 1);
                        const float oa1 = __shfl_xor_sync(0xffffffff, Dr[n][1], 1);
                        const float ob0 = __shfl_xor_sync(0xffffffff, Dr[n][2], 1);
                        const float ob1 = __shfl_xor_sync(0xffffffff, Dr[n][3], 1);
                        if ((lane & 1) == 0) {
                            if (va) b_red_v4(dva + n * BM_MMA_N, Dr[n][0], Dr[n][1], oa0, oa1);
                            if (vb) b_red_v4(dvb + n * BM_MMA_N, Dr[n][2], Dr[n][3], ob0, ob1);
                        }
                    } else if constexpr (VRED) {
                        if (va) b_red_v2(dva + n * BM_MMA_N, Dr[n][0], Dr[n][1]);
                        if (vb) b_red_v2(dvb + n * BM_MMA_N, Dr[n][2], Dr[n][3]);
                    } else {
                        if (va) {
                            if (Dr[n][0] != 0.f) atomicAdd(params.dv + ta + vcol, Dr[n][0]);
                            if (Dr[n][1] != 0.f) atomicAdd(params.dv + ta + vcol + 1, Dr[n][1]);
                        }
                        if (vb) {
                            if (Dr[n][2] != 0.f) atomicAdd(params.dv + tb + vcol, Dr[n][2]);
                            if (Dr[n][3] != 0.f) atomicAdd(params.dv + tb + vcol + 1, Dr[n][3]);
                        }
                    }
                }
            }
            __syncthreads();                   // bufA/bufB consumed
            if (vt + 1 < BM_NVTILES) {
                b_gather_tile(sm, sm.bufA, (vt + 1) * BM_DTILE);
                b_cp_commit();
            }
        }

        // ---- dS = P * (dP - D[h]) * sm_scale -> overwrite sPS ---------------
        {
            const int row_a = wrow + lane / 4, row_b = row_a + 8;
            const float da = sm.sD[row_a], db = sm.sD[row_b];
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int col0 = (nh * 4 + n) * BM_MMA_N + (lane % 4) * 2;
                const float s0 = Pr[n][0] * (Cr[n][0] - da) * params.sm_scale;
                const float s1 = Pr[n][1] * (Cr[n][1] - da) * params.sm_scale;
                const float s2 = Pr[n][2] * (Cr[n][2] - db) * params.sm_scale;
                const float s3 = Pr[n][3] * (Cr[n][3] - db) * params.sm_scale;
                *reinterpret_cast<__nv_bfloat162*>(&sm.sPS[b_swz(row_a, col0, BM_BTOPK)]) =
                    __floats2bfloat162_rn(s0, s1);
                *reinterpret_cast<__nv_bfloat162*>(&sm.sPS[b_swz(row_b, col0, BM_BTOPK)]) =
                    __floats2bfloat162_rn(s2, s3);
            }
        }
        // (the first dK/dQ-phase __syncthreads publishes the new sPS)

        // ---- dQ += dS K and dK = dS^T Q over 9 pipelined K d-tiles ----------
        // FULL unroll REQUIRED: dQr[dt] must be a compile-time register index
        // (a runtime index sends the 144-reg accumulator to .local -- the
        // established sm120 spill rule). Syncs inside the unrolled body are
        // legal: the loop is uniform across all 256 threads.
        b_gather_tile(sm, sm.bufA, 0);
        b_cp_commit();
        #pragma unroll
        for (int dt = 0; dt < BM_NKTILES; ++dt) {
            __nv_bfloat16* cur = (dt & 1) ? sm.bufB : sm.bufA;
            __nv_bfloat16* nxt = (dt & 1) ? sm.bufA : sm.bufB;
            if (dt + 1 < BM_NKTILES) {
                b_gather_tile(sm, nxt, (dt + 1) * BM_DTILE);
                b_cp_commit();
                b_cp_wait<1>();
            } else {
                b_cp_wait<0>();
            }
            __syncthreads();                   // K tile + (first iter) new sPS visible

            // (a) dQ tile: A = dS x4 (rows = heads), B = K x2 trans (k = tokens)
            #pragma unroll
            for (int tk = 0; tk < 4; ++tk) {   // contract 64 tokens = 4 k16
                uint32_t Ar[4];
                {
                    const int row = wrow + (lane % 16);
                    const int col = tk * BM_MMA_K + (lane / 16) * 8;
                    b_ldm_x4(Ar, b_cvta(&sm.sPS[b_swz(row, col, BM_BTOPK)]));
                }
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    uint32_t Br[2];
                    const int krow = tk * BM_MMA_K + (lane % 16);
                    const int kcol = nh * 32 + n * BM_MMA_N + (lane / 16) * 8;
                    b_ldm_x2_trans(Br, b_cvta(&cur[b_swz(krow, kcol, BM_DTILE)]));
                    b_mma(dQr[dt][n], Ar, Br, dQr[dt][n]);
                }
            }

            // (b) dK tile: A = dS^T x4 trans, B = Q x2 trans (resident sQ, col dt*64)
            {
                float Kr4[4][4];
                #pragma unroll
                for (int n = 0; n < 4; ++n) { Kr4[n][0] = Kr4[n][1] = Kr4[n][2] = Kr4[n][3] = 0.f; }
                #pragma unroll
                for (int hk = 0; hk < 4; ++hk) {          // contract 64 heads
                    uint32_t Ar[4];
                    {
                        const int prow = hk * BM_MMA_K + (lane / 16) * 8 + (lane % 8);
                        const int pcol = wrow + ((lane / 8) % 2) * 8;
                        b_ldm_x4_trans(Ar, b_cvta(&sm.sPS[b_swz(prow, pcol, BM_BTOPK)]));
                    }
                    #pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        uint32_t Br[2];
                        const int qrow = hk * BM_MMA_K + (lane % 16);
                        const int qcol = dt * BM_DTILE + nh * 32 + n * BM_MMA_N + (lane / 16) * 8;
                        b_ldm_x2_trans(Br, b_cvta(&sm.sQ[b_swz(qrow, qcol, BM_DQK)]));
                        b_mma(Kr4[n], Ar, Br, Kr4[n]);
                    }
                }
                const int trow_a = wrow + lane / 4, trow_b = trow_a + 8;
                const bool va = sm.sValid[trow_a] != 0, vb = sm.sValid[trow_b] != 0;
                const int64_t ta = va ? (int64_t)sm.sTok[trow_a] * params.stride_dk_s_kv : 0;
                const int64_t tb = vb ? (int64_t)sm.sTok[trow_b] * params.stride_dk_s_kv : 0;
                float* const dka = params.dk + ta + dt * BM_DTILE + nh * 32 + (lane % 4) * 2;
                float* const dkb = params.dk + tb + dt * BM_DTILE + nh * 32 + (lane % 4) * 2;
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const int dcol = dt * BM_DTILE + nh * 32 + n * BM_MMA_N + (lane % 4) * 2;
                    if constexpr (VRED4) {
                        const float oa0 = __shfl_xor_sync(0xffffffff, Kr4[n][0], 1);
                        const float oa1 = __shfl_xor_sync(0xffffffff, Kr4[n][1], 1);
                        const float ob0 = __shfl_xor_sync(0xffffffff, Kr4[n][2], 1);
                        const float ob1 = __shfl_xor_sync(0xffffffff, Kr4[n][3], 1);
                        if ((lane & 1) == 0) {
                            if (va) b_red_v4(dka + n * BM_MMA_N, Kr4[n][0], Kr4[n][1], oa0, oa1);
                            if (vb) b_red_v4(dkb + n * BM_MMA_N, Kr4[n][2], Kr4[n][3], ob0, ob1);
                        }
                    } else if constexpr (VRED) {
                        if (va) b_red_v2(dka + n * BM_MMA_N, Kr4[n][0], Kr4[n][1]);
                        if (vb) b_red_v2(dkb + n * BM_MMA_N, Kr4[n][2], Kr4[n][3]);
                    } else {
                        if (va) {
                            if (Kr4[n][0] != 0.f) atomicAdd(params.dk + ta + dcol, Kr4[n][0]);
                            if (Kr4[n][1] != 0.f) atomicAdd(params.dk + ta + dcol + 1, Kr4[n][1]);
                        }
                        if (vb) {
                            if (Kr4[n][2] != 0.f) atomicAdd(params.dk + tb + dcol, Kr4[n][2]);
                            if (Kr4[n][3] != 0.f) atomicAdd(params.dk + tb + dcol + 1, Kr4[n][3]);
                        }
                    }
                }
            }
            __syncthreads();                   // stage + sPS consumed before reuse
        }
    }

    // ---- epilogue: dQ pure store (rows CTA-exclusive, cols warp-disjoint) ---
    {
        const int row_a = wrow + lane / 4, row_b = row_a + 8;
        #pragma unroll
        for (int t = 0; t < BM_NKTILES; ++t)
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int col = t * BM_DTILE + nh * 32 + n * BM_MMA_N + (lane % 4) * 2;
                float* pa = dq_ptr + (int64_t)row_a * params.stride_dq_h_q + col;
                float* pb = dq_ptr + (int64_t)row_b * params.stride_dq_h_q + col;
                pa[0] = dQr[t][n][0]; pa[1] = dQr[t][n][1];
                pb[0] = dQr[t][n][2]; pb[1] = dQr[t][n][3];
            }
    }
}

inline void launch_sparse_bwd_mma(const SparsePrefillBwdParams& params, int red_tier) {
    const int num_h_blocks = params.h_q / BM_BH;
    const dim3 grid(num_h_blocks * params.s_q);
    constexpr size_t smem = sizeof(SmemPlanBwdMma);
    static_assert(smem <= 99 * 1024, "sparse bwd mma smem exceeds 99KB");
    if (red_tier >= 2) {
        CHECK_CUDA(cudaFuncSetAttribute(sparse_prefill_bwd_mma_kernel<true, true>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        sparse_prefill_bwd_mma_kernel<true, true><<<grid, dim3(BM_THREADS), smem, params.stream>>>(params);
    } else if (red_tier == 1) {
        CHECK_CUDA(cudaFuncSetAttribute(sparse_prefill_bwd_mma_kernel<true, false>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        sparse_prefill_bwd_mma_kernel<true, false><<<grid, dim3(BM_THREADS), smem, params.stream>>>(params);
    } else {
        CHECK_CUDA(cudaFuncSetAttribute(sparse_prefill_bwd_mma_kernel<false, false>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        sparse_prefill_bwd_mma_kernel<false, false><<<grid, dim3(BM_THREADS), smem, params.stream>>>(params);
    }
}

}  // namespace sparse_bwd_mma
}  // namespace sm120
