// ============================================================================
// FUSED PREFILL FORWARD KERNEL (SM120, WMMA, FlashAttention-2 online softmax)
// ============================================================================
// Replaces the pure-ATen fp32 forward fallback (run_fmha_fwd_sm120_fallback) with a
// real fused attention forward for consumer Blackwell (sm_120, no TMEM/tcgen05).
// Templated on <DQK, DVO> so it serves BOTH the MLA head dims (192/128) and the
// symmetric 128/128 case. Additive + gated (default-ON, opt-out) at the dispatcher
// so the prior fallback stays available and byte-identical when disabled.
//
// Algorithm (per Q-tile, looping K-tiles): FlashAttention-2
//   m_i = -inf, l_i = 0, acc = 0
//   for each K-tile j:
//     S   = scale * Q @ K_j^T            (WMMA, contract DQK)   [BM, BN]
//     causal-mask S (top-left: key > query -> -inf)
//     m_new = max(m_i, rowmax(S))
//     P   = exp(S - m_new)                                       [BM, BN]
//     corr = exp(m_i - m_new)
//     l_i = l_i*corr + rowsum(P);  acc = acc*corr + P @ V_j      (WMMA, contract BN)
//     m_i = m_new
//   O = acc / l_i ;  LSE = m_i + log(l_i)
// LSE is natural-log logsumexp of the scaled, causally-masked scores -- exactly what
// the backward (fused or fallback) consumes via expf(score - lse).
#pragma once

#include <cuda_bf16.h>
#include <mma.h>
#include <c10/cuda/CUDAStream.h>

// NOTE: intentionally does NOT include fmha_bwd_kernel_sm120.cuh. That header defines
// many NON-inline __device__ backward functions; pulling it into the forward translation
// unit too produced multiply-defined symbols (LNK2005) at link. The forward uses
// synchronous smem loads (no cp.async) and only needs these tile constants, defined
// locally with unique names to avoid any ODR clash with the backward header.
namespace flash {
namespace detail {

constexpr int FWD_WMMA = 16;                          // WMMA m=n=k tile
constexpr int FWD_NUM_WARPS = 8;
constexpr int FWD_NUM_THREADS = FWD_NUM_WARPS * 32;   // 256

// Forward tile sizes. BN must be 32 (one lane per key column in the row reductions);
// DVO must be a multiple of 32 (4 acc elements per lane). Both hold for 192/128 & 128/128.
constexpr int FWD_BM = 32;
constexpr int FWD_BN = 32;

// --- cp.async helpers (local, __forceinline__ => no external linkage; unique names so
//     this header stays independent of fmha_bwd_kernel_sm120.cuh). sm_120 has NO TMA,
//     so cp.async (LDGSTS) is the correct async-copy path. ---
__device__ __forceinline__ void fwd_cp_async_cg16(void* smem_ptr, const void* gmem_ptr) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void fwd_cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void fwd_cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }
template <int N> __device__ __forceinline__ void fwd_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <int DQK, int DVO>
struct alignas(256) FwdSmemLayout {
    static constexpr size_t q_off = 0;
    static constexpr size_t q_sz  = FWD_BM * DQK * sizeof(__nv_bfloat16);
    // K/V are DOUBLE-buffered for cp.async pipelining.
    static constexpr size_t k0_off = q_off + ((q_sz + 255) / 256) * 256;
    static constexpr size_t k_sz  = FWD_BN * DQK * sizeof(__nv_bfloat16);
    static constexpr size_t k1_off = k0_off + ((k_sz + 255) / 256) * 256;
    static constexpr size_t v0_off = k1_off + ((k_sz + 255) / 256) * 256;
    static constexpr size_t v_sz  = FWD_BN * DVO * sizeof(__nv_bfloat16);
    static constexpr size_t v1_off = v0_off + ((v_sz + 255) / 256) * 256;
    static constexpr size_t s_off  = v1_off + ((v_sz + 255) / 256) * 256;
    static constexpr size_t s_sz   = FWD_BM * FWD_BN * sizeof(float);
    static constexpr size_t p_off  = s_off + ((s_sz + 255) / 256) * 256;
    static constexpr size_t p_sz   = FWD_BM * FWD_BN * sizeof(__nv_bfloat16);
    static constexpr size_t acc_off = p_off + ((p_sz + 255) / 256) * 256;
    static constexpr size_t acc_sz  = FWD_BM * DVO * sizeof(float);
    static constexpr size_t mi_off = acc_off + ((acc_sz + 255) / 256) * 256;
    static constexpr size_t mi_sz  = FWD_BM * sizeof(float);
    static constexpr size_t li_off = mi_off + ((mi_sz + 255) / 256) * 256;
    static constexpr size_t li_sz  = FWD_BM * sizeof(float);
    static constexpr size_t stg_off = li_off + ((li_sz + 255) / 256) * 256;
    static constexpr size_t stg_sz  = FWD_NUM_WARPS * FWD_WMMA * FWD_WMMA * sizeof(float);
    static constexpr size_t total_size = stg_off + ((stg_sz + 255) / 256) * 256;
};

template <int DQK, int DVO>
struct FwdSmem {
    using L = FwdSmemLayout<DQK, DVO>;
    char* base;
    int cur_buf;
    __device__ __forceinline__ void init(char* b) { base = b; cur_buf = 0; }
    __device__ __forceinline__ void set_buffer(int c) { cur_buf = c; }
    __device__ __forceinline__ __nv_bfloat16* q()  { return reinterpret_cast<__nv_bfloat16*>(base + L::q_off); }
    __device__ __forceinline__ __nv_bfloat16* k_buf(int i) { return reinterpret_cast<__nv_bfloat16*>(base + (i == 0 ? L::k0_off : L::k1_off)); }
    __device__ __forceinline__ __nv_bfloat16* v_buf(int i) { return reinterpret_cast<__nv_bfloat16*>(base + (i == 0 ? L::v0_off : L::v1_off)); }
    __device__ __forceinline__ __nv_bfloat16* k()  { return k_buf(cur_buf); }
    __device__ __forceinline__ __nv_bfloat16* v()  { return v_buf(cur_buf); }
    __device__ __forceinline__ float* s()          { return reinterpret_cast<float*>(base + L::s_off); }
    __device__ __forceinline__ __nv_bfloat16* p()  { return reinterpret_cast<__nv_bfloat16*>(base + L::p_off); }
    __device__ __forceinline__ float* acc()        { return reinterpret_cast<float*>(base + L::acc_off); }
    __device__ __forceinline__ float* mi()         { return reinterpret_cast<float*>(base + L::mi_off); }
    __device__ __forceinline__ float* li()         { return reinterpret_cast<float*>(base + L::li_off); }
    __device__ __forceinline__ float* stg()        { return reinterpret_cast<float*>(base + L::stg_off); }
    __device__ __forceinline__ float* stg_warp(int w) { return stg() + w * FWD_WMMA * FWD_WMMA; }
    __device__ __forceinline__ __nv_bfloat16* q_row(int m)  { return q() + m * DQK; }
    __device__ __forceinline__ __nv_bfloat16* k_row(int n)  { return k() + n * DQK; }
    __device__ __forceinline__ __nv_bfloat16* v_row(int n)  { return v() + n * DVO; }
};

// Load `count` rows of `dim` bf16 from global [seq, H, dim] into a smem tile (stride dim).
__device__ __forceinline__ void fwd_load_rows(
    __nv_bfloat16* dst, const __nv_bfloat16* src,
    int seq_start, int start, int count, int num_heads, int head_idx, int dim) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = FWD_NUM_THREADS / 32;
    const int stride_token = num_heads * dim;
    for (int r = warp_id; r < count; r += num_warps) {
        const __nv_bfloat16* sr = src + (seq_start + start + r) * stride_token + head_idx * dim;
        __nv_bfloat16* dr = dst + r * dim;
        for (int d = lane_id * 2; d < dim; d += 64) {
            if (d + 1 < dim) *reinterpret_cast<__nv_bfloat162*>(dr + d) = *reinterpret_cast<const __nv_bfloat162*>(sr + d);
            else if (d < dim) dr[d] = sr[d];
        }
    }
}

// Async cp.async.cg (16B = 8 bf16) load of `count` rows of `dim` bf16 (dim multiple of 8).
__device__ __forceinline__ void fwd_async_load_rows(
    __nv_bfloat16* dst, const __nv_bfloat16* src,
    int seq_start, int start, int count, int num_heads, int head_idx, int dim) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = FWD_NUM_THREADS / 32;
    const int stride_token = num_heads * dim;
    for (int r = warp_id; r < count; r += num_warps) {
        const __nv_bfloat16* sr = src + (seq_start + start + r) * stride_token + head_idx * dim;
        __nv_bfloat16* dr = dst + r * dim;
        for (int d = lane_id * 8; d < dim; d += 256) {
            if (d + 8 <= dim) fwd_cp_async_cg16(dr + d, sr + d);
            else for (int dd = d; dd < dim && dd < d + 8; ++dd) dr[dd] = sr[dd];
        }
    }
}

// S = scale * Q @ K^T  (contract over DQK) -> s() buffer [BM, BN]
template <int DQK, int DVO>
__device__ __forceinline__ void fwd_qk(FwdSmem<DQK, DVO>& smem, int m_size, int n_size, float scale) {
    using namespace nvcuda::wmma;
    const int warp_id = threadIdx.x / 32;
    const int m_tiles = (m_size + FWD_WMMA - 1) / FWD_WMMA;
    const int n_tiles = (n_size + FWD_WMMA - 1) / FWD_WMMA;
    const int d_tiles = DQK / FWD_WMMA;
    for (int mn = warp_id; mn < m_tiles * n_tiles; mn += FWD_NUM_WARPS) {
        int mt = mn / n_tiles, nt = mn % n_tiles;
        fragment<accumulator, FWD_WMMA, FWD_WMMA, FWD_WMMA, float> acc;
        fill_fragment(acc, 0.0f);
        #pragma unroll
        for (int kt = 0; kt < d_tiles; ++kt) {
            fragment<matrix_a, FWD_WMMA, FWD_WMMA, FWD_WMMA, __nv_bfloat16, row_major> a;
            fragment<matrix_b, FWD_WMMA, FWD_WMMA, FWD_WMMA, __nv_bfloat16, col_major> b;
            load_matrix_sync(a, smem.q_row(mt * FWD_WMMA) + kt * FWD_WMMA, DQK);
            load_matrix_sync(b, smem.k_row(nt * FWD_WMMA) + kt * FWD_WMMA, DQK);
            mma_sync(acc, a, b, acc);
        }
        #pragma unroll
        for (int i = 0; i < acc.num_elements; ++i) acc.x[i] *= scale;
        store_matrix_sync(smem.s() + mt * FWD_WMMA * FWD_BN + nt * FWD_WMMA, acc, FWD_BN, mem_row_major);
    }
    __syncthreads();
}

// Online-softmax update for one K-tile: warp-per-row over BN=32 columns.
// Updates m_i, l_i, writes P (bf16), and rescales acc[m, :] *= corr in place.
template <int DQK, int DVO>
__device__ __forceinline__ void fwd_online_softmax(
    FwdSmem<DQK, DVO>& smem, int m_size, int n_size, bool is_causal, int m_start_g, int n_start_g,
    int causal_off) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    for (int m = warp_id; m < m_size; m += FWD_NUM_WARPS) {
        // each lane owns key column n = lane (BN=32)
        float s = -INFINITY;
        if (lane < n_size) {
            // Bottom-right causal alignment: query causal pos = (m_start_g + m) + causal_off,
            // with causal_off = seq_kv - seq_q. =0 for square (prefill/training); for a single
            // decode query (seq_q=1) it lets that query attend to all cached keys.
            bool masked = is_causal && ((m_start_g + m + causal_off) < (n_start_g + lane));
            if (!masked) s = smem.s()[m * FWD_BN + lane];
        }
        // row max
        float rmax = s;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffff, rmax, o));
        float m_old = smem.mi()[m];
        float m_new = fmaxf(m_old, rmax);
        float corr = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
        float p = (s == -INFINITY) ? 0.0f : __expf(s - m_new);
        // row sum of p
        float rsum = p;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) rsum += __shfl_xor_sync(0xffffffff, rsum, o);
        // write P (full BN width; lanes >= n_size already have p=0)
        smem.p()[m * FWD_BN + lane] = __float2bfloat16(p);
        if (lane == 0) {
            smem.li()[m] = smem.li()[m] * corr + rsum;
            smem.mi()[m] = m_new;
        }
        // rescale acc[m, :] *= corr  (DVO elements, 4 per lane for DVO=128)
        #pragma unroll
        for (int d = lane; d < DVO; d += 32) smem.acc()[m * DVO + d] *= corr;
    }
    __syncthreads();
}

// acc += P @ V  (P [BM,BN] bf16, V [BN,DVO] bf16, contract over BN) -> acc()
template <int DQK, int DVO>
__device__ __forceinline__ void fwd_pv(FwdSmem<DQK, DVO>& smem, int m_size, int n_size) {
    using namespace nvcuda::wmma;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int m_tiles = (m_size + FWD_WMMA - 1) / FWD_WMMA;
    const int d_tiles = DVO / FWD_WMMA;
    const int k_tiles = (n_size + FWD_WMMA - 1) / FWD_WMMA;
    for (int md = warp_id; md < m_tiles * d_tiles; md += FWD_NUM_WARPS) {
        int mt = md / d_tiles, dt = md % d_tiles;
        fragment<accumulator, FWD_WMMA, FWD_WMMA, FWD_WMMA, float> acc;
        fill_fragment(acc, 0.0f);
        for (int kt = 0; kt < k_tiles; ++kt) {
            fragment<matrix_a, FWD_WMMA, FWD_WMMA, FWD_WMMA, __nv_bfloat16, row_major> pa;  // P[M,N]
            fragment<matrix_b, FWD_WMMA, FWD_WMMA, FWD_WMMA, __nv_bfloat16, row_major> vb;  // V[N,D]
            load_matrix_sync(pa, smem.p() + mt * FWD_WMMA * FWD_BN + kt * FWD_WMMA, FWD_BN);
            load_matrix_sync(vb, smem.v_row(kt * FWD_WMMA) + dt * FWD_WMMA, DVO);
            mma_sync(acc, pa, vb, acc);
        }
        float* stg = smem.stg_warp(warp_id);
        store_matrix_sync(stg, acc, FWD_WMMA, mem_row_major);
        __syncwarp();
        for (int i = lane; i < FWD_WMMA * FWD_WMMA; i += 32) {
            int r = i / FWD_WMMA, c = i % FWD_WMMA;
            int gr = mt * FWD_WMMA + r, gc = dt * FWD_WMMA + c;
            if (gr < m_size) smem.acc()[gr * DVO + gc] += stg[i];
        }
        __syncwarp();
    }
    __syncthreads();
}

template <int DQK, int DVO, bool kIsCausal>
__global__ void __launch_bounds__(FWD_NUM_THREADS)
fmha_fwd_sm120_mla_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ cu_seqlens_kv,
    __nv_bfloat16* __restrict__ o,   // [total_q, H, DVO] contiguous
    float* __restrict__ lse,         // head-major: lse[token + head*lse_head_stride] (the .T view)
    int num_heads,
    float scale,
    int max_seqlen_q,
    int max_seqlen_kv,
    int lse_head_stride) {
    extern __shared__ char smem_base[];
    FwdSmem<DQK, DVO> smem;
    smem.init(smem_base);

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int m_block = blockIdx.x;
    if (head_idx >= num_heads) return;

    const int q_start = cu_seqlens_q[batch_idx];
    const int q_end   = cu_seqlens_q[batch_idx + 1];
    const int kv_start = cu_seqlens_kv[batch_idx];
    const int kv_end   = cu_seqlens_kv[batch_idx + 1];
    const int seq_q = q_end - q_start;
    const int seq_kv = kv_end - kv_start;

    const int m_start = m_block * FWD_BM;
    if (m_start >= seq_q) return;
    const int m_end = min(m_start + FWD_BM, seq_q);
    const int m_size = m_end - m_start;
    const int tid = threadIdx.x;

    // init running stats + acc
    for (int i = tid; i < FWD_BM; i += FWD_NUM_THREADS) { smem.mi()[i] = -INFINITY; smem.li()[i] = 0.0f; }
    for (int i = tid; i < FWD_BM * DVO; i += FWD_NUM_THREADS) smem.acc()[i] = 0.0f;
    // load Q once
    fwd_load_rows(smem.q(), q, q_start, m_start, m_size, num_heads, head_idx, DQK);
    __syncthreads();

    // causal: only K-tiles up to this Q-tile's last row matter (bottom-right aligned)
    int n_block_max = (seq_kv + FWD_BN - 1) / FWD_BN;
    if (kIsCausal) {
        int last_q = m_start + m_size - 1;          // global query index
        int last_key = last_q + (seq_kv - seq_q);   // bottom-right: furthest visible key (=last_q if square)
        n_block_max = min(n_block_max, (last_key + 1 + FWD_BN - 1) / FWD_BN);
    }

    // cp.async double-buffered K/V pipeline: prefetch block n+1 while computing block n.
    int cur = 0;
    if (n_block_max > 0) {
        const int ns0 = min(FWD_BN, seq_kv);
        fwd_async_load_rows(smem.k_buf(0), k, kv_start, 0, ns0, num_heads, head_idx, DQK);
        fwd_async_load_rows(smem.v_buf(0), v, kv_start, 0, ns0, num_heads, head_idx, DVO);
        fwd_cp_async_commit();
    }
    for (int n_block = 0; n_block < n_block_max; ++n_block) {
        const int n_start = n_block * FWD_BN;
        const int n_end = min(n_start + FWD_BN, seq_kv);
        const int n_size = n_end - n_start;

        const bool has_next = (n_block + 1 < n_block_max);
        if (has_next) {
            const int nb = 1 - cur;
            const int ns1_start = (n_block + 1) * FWD_BN;
            const int ns1 = min(FWD_BN, seq_kv - ns1_start);
            fwd_async_load_rows(smem.k_buf(nb), k, kv_start, ns1_start, ns1, num_heads, head_idx, DQK);
            fwd_async_load_rows(smem.v_buf(nb), v, kv_start, ns1_start, ns1, num_heads, head_idx, DVO);
            fwd_cp_async_commit();
            fwd_cp_async_wait_group<1>();  // 2 groups in flight -> wait for the current (older) one
        } else {
            fwd_cp_async_wait_all();       // last block: drain the remaining group
        }
        __syncthreads();
        smem.set_buffer(cur);

        fwd_qk<DQK, DVO>(smem, m_size, n_size, scale);
        fwd_online_softmax<DQK, DVO>(smem, m_size, n_size, kIsCausal, m_start, n_start, seq_kv - seq_q);
        fwd_pv<DQK, DVO>(smem, m_size, n_size);
        __syncthreads();  // compute done; buffer `cur` is free for a future prefetch

        if (has_next) cur = 1 - cur;
    }

    // finalize: O = acc / l_i ; LSE = m_i + log(l_i)
    const int stride_o_tok = num_heads * DVO;
    for (int idx = tid; idx < m_size * DVO; idx += FWD_NUM_THREADS) {
        int m = idx / DVO, d = idx % DVO;
        float li = smem.li()[m];
        float inv = (li > 0.0f) ? (1.0f / li) : 0.0f;
        int gt = q_start + m_start + m;
        o[gt * stride_o_tok + head_idx * DVO + d] = __float2bfloat16(smem.acc()[m * DVO + d] * inv);
    }
    for (int m = tid; m < m_size; m += FWD_NUM_THREADS) {
        float li = smem.li()[m];
        int gt = q_start + m_start + m;
        lse[gt + head_idx * lse_head_stride] = (li > 0.0f) ? (smem.mi()[m] + __logf(li)) : (-INFINITY);
    }
}

// ---- Launcher (templated on head dims) --------------------------------------
template <int DQK, int DVO, bool kIsCausal>
void launch_fmha_fwd_sm120_mla(
    const c10::cuda::CUDAStream& stream,
    at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor cu_seqlens_q, at::Tensor cu_seqlens_kv,
    at::Tensor o, at::Tensor lse,
    float scale, int max_seqlen_q, int max_seqlen_kv) {
    const int batch_size = cu_seqlens_q.size(0) - 1;
    const int num_heads = q.size(1);
    const int m_blocks = (max_seqlen_q + FWD_BM - 1) / FWD_BM;
    dim3 grid(m_blocks, num_heads, batch_size);
    dim3 block(FWD_NUM_THREADS);
    size_t smem_size = FwdSmemLayout<DQK, DVO>::total_size;
    // LSE is the .T view [total_q, H] with stride (1, total_q): head stride = lse.stride(1).
    const int lse_head_stride = static_cast<int>(lse.stride(1));
    auto kern = fmha_fwd_sm120_mla_kernel<DQK, DVO, kIsCausal>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kern<<<grid, block, smem_size, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        cu_seqlens_q.data_ptr<int>(), cu_seqlens_kv.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(o.data_ptr()),
        lse.data_ptr<float>(),
        num_heads, scale, max_seqlen_q, max_seqlen_kv, lse_head_stride);
}

}  // namespace detail
}  // namespace flash
