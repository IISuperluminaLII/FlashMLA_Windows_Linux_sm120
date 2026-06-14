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

#include <cuda_bf16.h>
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

// ---- Kernel ----------------------------------------------------------------
// Split head dims: Q,K (and S contraction) use DQK; V,O use DVO. DQK==DVO==128 is the MHA
// case; DQK=192/DVO=128 is MLA. Smem: Q[BQ,DQK], K[BKV,DQK] (x2 staged), V[BKV,DVO] (x2).
template <int DQK, int DVO, bool kCausal>
__global__ void __launch_bounds__(FM_THREADS)
fmha_fwd_mma_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int* __restrict__ cu_q,
    const int* __restrict__ cu_kv,
    __nv_bfloat16* __restrict__ o,
    float* __restrict__ lse,
    int num_heads, float scale, int max_sq, int max_skv, int lse_head_stride) {
    constexpr int DK = DQK / MMA_K;     // d-tiles for QK contraction (DQK=192 -> 12)
    constexpr int NKV = FM_BKV / MMA_N; // n8 tiles in a KV block (4)
    constexpr int DN = DVO / MMA_N;     // n8 output tiles for O (DVO=128 -> 16)
    constexpr int PK = FM_BKV / MMA_K;  // k16 P tiles (2)

    // Double-buffered K/V for the cp.async software pipeline: K stage = FM_BKV*DQK,
    // V stage = FM_BKV*DVO. Q is single-buffered.
    extern __shared__ char smem_raw[];
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* Kbuf = Qs + FM_BQ * DQK;
    __nv_bfloat16* Vbuf = Kbuf + 2 * FM_BKV * DQK;

    const int b = blockIdx.z, h = blockIdx.y, qb = blockIdx.x;
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
        int last_q = qtile + min(FM_BQ, sq - qtile) - 1;
        nb_max = min(nb_max, (last_q + 1 + FM_BKV - 1) / FM_BKV);
    }

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

        // causal mask + online softmax (C layout: c0,c1 row=lane/4; c2,c3 row=lane/4+8)
        const int row_a = qtile + wrow + (lane / 4);
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
        float nmax0 = fmaxf(rmax[0], tmax[0]), nmax1 = fmaxf(rmax[1], tmax[1]);
        float rs0 = (rmax[0] == -INFINITY) ? 0.f : __expf(rmax[0] - nmax0);
        float rs1 = (rmax[1] == -INFINITY) ? 0.f : __expf(rmax[1] - nmax1);
        rmax[0] = nmax0; rmax[1] = nmax1;
        #pragma unroll
        for (int i = 0; i < DN; ++i) {
            Or[i][0] *= rs0; Or[i][1] *= rs0; Or[i][2] *= rs1; Or[i][3] *= rs1;
        }

        // P = exp(S - max); pack to bf16x2 A-fragments for P@V
        uint32_t Pr[PK][4];
        float tsum[2] = {0.f, 0.f};
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

        // O += P @ V  (contract over kv; V via ldmatrix.x2.trans)
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
        __syncthreads();
    }

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
        if (orow_a < sq) lse[(q0 + orow_a) + h * lse_head_stride] = (rsum[0] > 0.f) ? (rmax[0] + __logf(rsum[0])) : -INFINITY;
        if (orow_b < sq) lse[(q0 + orow_b) + h * lse_head_stride] = (rsum[1] > 0.f) ? (rmax[1] + __logf(rsum[1])) : -INFINITY;
    }
}

template <int DQK, int DVO, bool kCausal>
void launch_fmha_fwd_mma(const c10::cuda::CUDAStream& stream,
                         at::Tensor q, at::Tensor k, at::Tensor v,
                         at::Tensor cu_q, at::Tensor cu_kv, at::Tensor o, at::Tensor lse,
                         float scale, int max_sq, int max_skv) {
    const int batch = cu_q.size(0) - 1, num_heads = q.size(1);
    const int qblocks = (max_sq + FM_BQ - 1) / FM_BQ;
    dim3 grid(qblocks, num_heads, batch);
    // Q[BQ,DQK] + 2x K[BKV,DQK] + 2x V[BKV,DVO]
    size_t smem = (FM_BQ * DQK + 2 * FM_BKV * DQK + 2 * FM_BKV * DVO) * sizeof(__nv_bfloat16);
    auto kern = fmha_fwd_mma_kernel<DQK, DVO, kCausal>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    const int lse_head_stride = static_cast<int>(lse.stride(1));
    kern<<<grid, dim3(FM_THREADS), smem, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        cu_q.data_ptr<int>(), cu_kv.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(o.data_ptr()), lse.data_ptr<float>(),
        num_heads, scale, max_sq, max_skv, lse_head_stride);
}

}  // namespace mma_fwd
}  // namespace detail
}  // namespace flash
