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
#pragma once

#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

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

// vectorized synchronous global->shared load of [rows, dim] bf16, swizzled (dim mult 8)
__device__ __forceinline__ void bg2s(__nv_bfloat16* dst, const __nv_bfloat16* src,
                                     int seq_start, int start, int rows, int num_heads,
                                     int head_idx, int dim) {
    const int tid = threadIdx.x;
    const int stride_token = num_heads * dim;
    const int vec = dim / 8;
    for (int i = tid; i < rows * vec; i += BW_THREADS) {
        int r = i / vec, c = (i % vec) * 8;
        const __nv_bfloat16* s = src + (seq_start + start + r) * stride_token + head_idx * dim + c;
        *reinterpret_cast<int4*>(dst + swz(r, c, dim)) = *reinterpret_cast<const int4*>(s);
    }
}

// ============================================================================
// Kernel A: dK + dV.  CTA per KV block; warps own keys; loop Q; no atomics.
// ============================================================================
template <int DQK, int DVO, bool kCausal>
__global__ void __launch_bounds__(BW_THREADS)
fmha_bwd_mma_dkdv_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ o,
    const __nv_bfloat16* __restrict__ d_o, const float* __restrict__ lse,
    const float* __restrict__ delta,
    const int* __restrict__ cu_q, const int* __restrict__ cu_kv,
    __nv_bfloat16* __restrict__ dk, __nv_bfloat16* __restrict__ dv,
    int num_heads, float scale, int max_sq, int max_skv) {
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

    const int b = blockIdx.z, h = blockIdx.y, nb = blockIdx.x;
    if (h >= num_heads) return;
    const int q0 = cu_q[b], q1 = cu_q[b + 1], kv0 = cu_kv[b], kv1 = cu_kv[b + 1];
    const int sq = q1 - q0, skv = kv1 - kv0;
    const int n_start = nb * BW_BN;
    if (n_start >= skv) return;
    const int n_size = min(BW_BN, skv - n_start);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int kw0 = warp * BW_WN;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* Ks    = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* Vs    = Ks    + BW_BN * DQK;
    __nv_bfloat16* Qbuf  = Vs    + BW_BN * DVO;     // [2][BM*DQK] cp.async double-buffer
    __nv_bfloat16* dObuf = Qbuf  + 2 * BM * DQK;    // [2][BM*DVO]
    __nv_bfloat16* PtS   = dObuf + 2 * BM * DVO;    // O tile dropped: delta is precomputed
    float* lseS   = reinterpret_cast<float*>(PtS + BW_BN * PT_STRIDE);
    float* deltaS = lseS + BM;

    bg2s(Ks, k, kv0, n_start, n_size, num_heads, h, DQK);
    bg2s(Vs, v, kv0, n_start, n_size, num_heads, h, DVO);
    for (int i = tid; i < (BW_BN - n_size) * DQK; i += BW_THREADS) {
        int r = n_size + i / DQK, c = i % DQK; Ks[swz(r, c, DQK)] = __float2bfloat16(0.f);
    }
    for (int i = tid; i < (BW_BN - n_size) * DVO; i += BW_THREADS) {
        int r = n_size + i / DVO, c = i % DVO; Vs[swz(r, c, DVO)] = __float2bfloat16(0.f);
    }
    __syncthreads();

    float dKr[DNQ][4], dVr[DNV][4];
    #pragma unroll
    for (int i = 0; i < DNQ; ++i) { dKr[i][0]=dKr[i][1]=dKr[i][2]=dKr[i][3]=0.f; }
    #pragma unroll
    for (int i = 0; i < DNV; ++i) { dVr[i][0]=dVr[i][1]=dVr[i][2]=dVr[i][3]=0.f; }

    int mb0 = 0;
    if (kCausal) mb0 = max(0, (n_start - (BM - 1)) / BM);
    int nq = (sq + BM - 1) / BM;
    const int sk_tok = num_heads * DQK, sv_tok = num_heads * DVO;

    // cp.async pipeline over Q blocks: prefetch mb0 into stage 0, then prefetch block mb+1
    // while computing mb. After mb0 the block range is contiguous (the leading mb0 block may
    // be fully causal-masked -> zero contribution, harmless), so a 2-stage buffer applies.
    {
        int ms0 = min(BM, sq - mb0 * BM);
        g2s_cp(Qbuf,  q,   q0, mb0 * BM, ms0, num_heads, h, DQK);
        g2s_cp(dObuf, d_o, q0, mb0 * BM, ms0, num_heads, h, DVO);
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
            g2s_cp(Qbuf  + nxt * (BM * DQK), q,   q0, (mb + 1) * BM, ms2, num_heads, h, DQK);
            g2s_cp(dObuf + nxt * (BM * DVO), d_o, q0, (mb + 1) * BM, ms2, num_heads, h, DVO);
            cp_commit();
            cp_wait_group<1>();
        } else {
            cp_wait_group<0>();
        }
        for (int m = tid; m < m_size; m += BW_THREADS) {
            lseS[m]   = lse[(q0 + m_start + m) * num_heads + h];
            deltaS[m] = delta[(q0 + m_start + m) * num_heads + h];   // precomputed (no O tile)
        }
        for (int i = tid; i < (BM - m_size) * DQK; i += BW_THREADS) {
            int r = m_size + i / DQK, c = i % DQK; Qs[swz(r, c, DQK)] = __float2bfloat16(0.f);
        }
        for (int i = tid; i < (BM - m_size) * DVO; i += BW_THREADS) {
            int r = m_size + i / DVO, c = i % DVO; dOs[swz(r, c, DVO)] = __float2bfloat16(0.f);
        }
        __syncthreads();

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
                bool va0=(gca<n_size)&&(!kCausal||(n_start+gca)<=(m_start+ra)), va1=(gcb<n_size)&&(!kCausal||(n_start+gcb)<=(m_start+ra));
                bool vb0=(gca<n_size)&&(!kCausal||(n_start+gca)<=(m_start+rb)), vb1=(gcb<n_size)&&(!kCausal||(n_start+gcb)<=(m_start+rb));
                p0=(va0&&ra<m_size)?p0:0.f; p1=(va1&&ra<m_size)?p1:0.f; p2=(vb0&&rb<m_size)?p2:0.f; p3=(vb1&&rb<m_size)?p3:0.f;
                Pr[mm][nn][0]=p0; Pr[mm][nn][1]=p1; Pr[mm][nn][2]=p2; Pr[mm][nn][3]=p3;
                dSr[mm][nn][0]=p0*(dPr[mm][nn][0]-da)*scale; dSr[mm][nn][1]=p1*(dPr[mm][nn][1]-da)*scale;
                dSr[mm][nn][2]=p2*(dPr[mm][nn][2]-db)*scale; dSr[mm][nn][3]=p3*(dPr[mm][nn][3]-db)*scale;
            }
        }

        // dV += P^T @ dO (out DVO). store P^T to PtS[key,query], A x4 nt, B=dO x2.trans.
        #pragma unroll
        for (int mm = 0; mm < MM; ++mm) {
            int ra = mm * BM_M + (lane / 4), rb = ra + 8;
            #pragma unroll
            for (int nn = 0; nn < WNN; ++nn) {
                int lk = kw0 + nn * BM_N + (lane % 4) * 2;
                PtS[swz(lk,   ra, PT_STRIDE)] = __float2bfloat16(Pr[mm][nn][0]);
                PtS[swz(lk,   rb, PT_STRIDE)] = __float2bfloat16(Pr[mm][nn][2]);
                PtS[swz(lk+1, ra, PT_STRIDE)] = __float2bfloat16(Pr[mm][nn][1]);
                PtS[swz(lk+1, rb, PT_STRIDE)] = __float2bfloat16(Pr[mm][nn][3]);
            }
        }
        __syncwarp();
        {
            uint32_t Ar[MK][4];
            #pragma unroll
            for (int kk = 0; kk < MK; ++kk) { int row=kw0+(lane%16), col=kk*BM_K+(lane/16)*8; ldm_x4(Ar[kk], cvta_shared(PtS + swz(row, col, PT_STRIDE))); }
            #pragma unroll
            for (int dd = 0; dd < DNV; ++dd) for (int kk = 0; kk < MK; ++kk) {
                uint32_t Br[2]; int vrow=kk*BM_K+(lane%16), vcol=dd*BM_N+(lane/16)*8;
                ldm_x2_trans(Br, cvta_shared(dOs + swz(vrow, vcol, DVO)));
                mma_16x8x16(dVr[dd], Ar[kk], Br, dVr[dd]);
            }
        }
        __syncwarp();

        // dK += dS^T @ Q (out DQK). store dS^T to PtS[key,query], A x4 nt, B=Q x2.trans.
        #pragma unroll
        for (int mm = 0; mm < MM; ++mm) {
            int ra = mm * BM_M + (lane / 4), rb = ra + 8;
            #pragma unroll
            for (int nn = 0; nn < WNN; ++nn) {
                int lk = kw0 + nn * BM_N + (lane % 4) * 2;
                PtS[swz(lk,   ra, PT_STRIDE)] = __float2bfloat16(dSr[mm][nn][0]);
                PtS[swz(lk,   rb, PT_STRIDE)] = __float2bfloat16(dSr[mm][nn][2]);
                PtS[swz(lk+1, ra, PT_STRIDE)] = __float2bfloat16(dSr[mm][nn][1]);
                PtS[swz(lk+1, rb, PT_STRIDE)] = __float2bfloat16(dSr[mm][nn][3]);
            }
        }
        __syncwarp();
        {
            uint32_t Ar[MK][4];
            #pragma unroll
            for (int kk = 0; kk < MK; ++kk) { int row=kw0+(lane%16), col=kk*BM_K+(lane/16)*8; ldm_x4(Ar[kk], cvta_shared(PtS + swz(row, col, PT_STRIDE))); }
            #pragma unroll
            for (int dd = 0; dd < DNQ; ++dd) for (int kk = 0; kk < MK; ++kk) {
                uint32_t Br[2]; int qrow=kk*BM_K+(lane%16), qcol=dd*BM_N+(lane/16)*8;
                ldm_x2_trans(Br, cvta_shared(Qs + swz(qrow, qcol, DQK)));
                mma_16x8x16(dKr[dd], Ar[kk], Br, dKr[dd]);
            }
        }
        __syncthreads();
    }

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
    __nv_bfloat16* dOs   = Qs    + BM * DQK;        // O tile dropped: delta is precomputed
    __nv_bfloat16* Kbuf  = dOs   + BM * DVO;        // [2][BNB*DQK] cp.async double-buffer
    __nv_bfloat16* Vbuf  = Kbuf  + 2 * BNB * DQK;   // [2][BNB*DVO]
    __nv_bfloat16* DsS   = Vbuf  + 2 * BNB * DVO;
    float* lseS   = reinterpret_cast<float*>(DsS + BM * DS_STRIDE);
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
    if (kCausal) { int last_q = m_start + m_size - 1; nb_max = min(nb_max, (last_q + 1 + BNB - 1) / BNB); }
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

        // dS = P*(dP - delta)*scale ; store natural into DsS[query, key].
        #pragma unroll
        for (int nn = 0; nn < NN; ++nn) {
            int c0 = nn * BM_N + (lane % 4) * 2, gca = c0, gcb = c0 + 1;
            float p0=__expf(scale*Sr[nn][0]-lsea), p1=__expf(scale*Sr[nn][1]-lsea);
            float p2=__expf(scale*Sr[nn][2]-lseb), p3=__expf(scale*Sr[nn][3]-lseb);
            bool va0=(gca<n_size)&&(!kCausal||(n_start+gca)<=(m_start+qa)), va1=(gcb<n_size)&&(!kCausal||(n_start+gcb)<=(m_start+qa));
            bool vb0=(gca<n_size)&&(!kCausal||(n_start+gca)<=(m_start+qb)), vb1=(gcb<n_size)&&(!kCausal||(n_start+gcb)<=(m_start+qb));
            p0=(va0&&qa<m_size)?p0:0.f; p1=(va1&&qa<m_size)?p1:0.f; p2=(vb0&&qb<m_size)?p2:0.f; p3=(vb1&&qb<m_size)?p3:0.f;
            DsS[swz(qa, c0,   DS_STRIDE)] = __float2bfloat16(p0 * (dPr[nn][0] - da) * scale);
            DsS[swz(qa, c0+1, DS_STRIDE)] = __float2bfloat16(p1 * (dPr[nn][1] - da) * scale);
            DsS[swz(qb, c0,   DS_STRIDE)] = __float2bfloat16(p2 * (dPr[nn][2] - db) * scale);
            DsS[swz(qb, c0+1, DS_STRIDE)] = __float2bfloat16(p3 * (dPr[nn][3] - db) * scale);
        }
        __syncwarp();

        // dQ += dS @ K (out DQK). A=dS x4 nt (key contiguous) ; B=K x2.trans.
        uint32_t dSar[NK][4];
        #pragma unroll
        for (int kk = 0; kk < NK; ++kk) { int row=qw0+(lane%16), col=kk*BM_K+(lane/16)*8; ldm_x4(dSar[kk], cvta_shared(DsS + swz(row, col, DS_STRIDE))); }
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

// ---- Launchers --------------------------------------------------------------
template <int DQK, int DVO, bool kCausal>
void launch_fmha_bwd_mma(const c10::cuda::CUDAStream& stream,
                         at::Tensor d_o, at::Tensor q, at::Tensor k, at::Tensor v,
                         at::Tensor o, at::Tensor lse, at::Tensor cu_q, at::Tensor cu_kv,
                         at::Tensor dq, at::Tensor dk, at::Tensor dv,
                         float scale, int max_sq, int max_skv) {
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
        size_t smem = (BW_BN * DQK + BW_BN * DVO + 2 * BM * DQK + 2 * BM * DVO + BW_BN * PT_STRIDE) * sizeof(__nv_bfloat16) + (2 * BM) * sizeof(float);
        auto kern = fmha_bwd_mma_dkdv_kernel<DQK, DVO, kCausal>;
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kern<<<grid, dim3(BW_THREADS), smem, stream.stream()>>>(qp, kp, vp, op, dop, lp, dp, cq, ck,
            reinterpret_cast<__nv_bfloat16*>(dk.data_ptr()), reinterpret_cast<__nv_bfloat16*>(dv.data_ptr()), num_heads, scale, max_sq, max_skv);
    }
    {   // Kernel B: dQ, grid over Q blocks (queries warp-partitioned -> BMB=64; KV loop BNB).
        constexpr int BMB = BW_NW * BM_M;            // 64
        constexpr int BNB = 32;
        constexpr int DSST = BNB > 64 ? BNB : 64;
        const int nblocks = (max_sq + BMB - 1) / BMB;
        dim3 grid(nblocks, num_heads, batch);
        size_t smem = (BMB * DQK + BMB * DVO + 2 * BNB * DQK + 2 * BNB * DVO + BMB * DSST) * sizeof(__nv_bfloat16) + (2 * BMB) * sizeof(float);
        auto kern = fmha_bwd_mma_dq_kernel<DQK, DVO, kCausal>;
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kern<<<grid, dim3(BW_THREADS), smem, stream.stream()>>>(qp, kp, vp, op, dop, lp, dp, cq, ck,
            reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()), num_heads, scale, max_sq, max_skv);
    }
}

}  // namespace mma_bwd
}  // namespace detail
}  // namespace flash
