// ============================================================================
// FUSED 192/128 MLA BACKWARD KERNEL (SM120, WMMA)
// ============================================================================
// Single K-major kernel computing dQ (atomic) + dK/dV (smem-accumulated) for the
// MLA head dims head_dim_qk=192, head_dim_vo=128. This is the fused replacement
// for the slow ATen fallback on the 192/128 path.
//
// Design notes / why this is separate from the 128 K-major kernel:
//   * MLA has SPLIT head dims: Q,K,dQ,dK use D_QK=192; V,O,dO,dV use D_VO=128.
//     The 128 kernel hardcodes a single BWD_BLOCK_D stride; reusing it would read
//     garbage for the 192-strided tiles. This file keeps the 128 path byte-identical
//     by being purely additive and gated default-OFF at the dispatcher.
//   * 99KB smem wall (the reason MLA fell back): a 192/128 K-major tile with M=32
//     and double-buffering is ~125KB. We use MLA_BLOCK_M=16, MLA_BLOCK_N=16 which
//     fits in ~81KB WITH double-buffered Q/dO (cp.async overlap retained).
//   * Causal correctness is by construction: softmax uses the forward's saved
//     (causal) LSE AND applies the top-left causal mask (prob=0 for key>query),
//     matching the forward. This is unlike the old ATen fallback bug.
//
// All compute is bf16 inputs -> fp32 WMMA accumulate -> fp32 dQ/dK/dV (atomics /
// smem), matching the 128 kernel's numerics.
#pragma once

#include <cuda_bf16.h>
#include <mma.h>
#include <c10/cuda/CUDAStream.h>

#include "sm120/prefill/dense/fmha_bwd_kernel_sm120.cuh"  // WMMA_*, BWD_NUM_*, cp_async_* , warp consts

namespace flash {
namespace detail {

// ---- MLA tile constants -----------------------------------------------------
constexpr int MLA_DQK = 192;        // head_dim_qk
constexpr int MLA_DVO = 128;        // head_dim_vo
constexpr int MLA_BLOCK_M = 16;     // Q rows per block (small to fit 99KB at D=192)
constexpr int MLA_BLOCK_N = 16;     // KV per block

// 256-byte alignment helper for the smem layout below.
__host__ __device__ constexpr size_t mla_al256(size_t s) { return ((s + 255) / 256) * 256; }

// ---- Shared-memory layout (split DQK / DVO strides) -------------------------
struct alignas(256) KMajorMlaSmemLayout {
    // Q tile - DOUBLE buffered (D_QK stride)
    static constexpr size_t q_tile_0_offset = 0;
    static constexpr size_t q_tile_size = MLA_BLOCK_M * MLA_DQK * sizeof(__nv_bfloat16);
    static constexpr size_t q_tile_1_offset = q_tile_0_offset + mla_al256(q_tile_size);

    // K tile - single (D_QK stride)
    static constexpr size_t k_tile_offset = q_tile_1_offset + mla_al256(q_tile_size);
    static constexpr size_t k_tile_size = MLA_BLOCK_N * MLA_DQK * sizeof(__nv_bfloat16);

    // V tile - single (D_VO stride)
    static constexpr size_t v_tile_offset = k_tile_offset + mla_al256(k_tile_size);
    static constexpr size_t v_tile_size = MLA_BLOCK_N * MLA_DVO * sizeof(__nv_bfloat16);

    // dO tile - DOUBLE buffered (D_VO stride)
    static constexpr size_t do_tile_0_offset = v_tile_offset + mla_al256(v_tile_size);
    static constexpr size_t do_tile_size = MLA_BLOCK_M * MLA_DVO * sizeof(__nv_bfloat16);
    static constexpr size_t do_tile_1_offset = do_tile_0_offset + mla_al256(do_tile_size);

    // O tile - single (D_VO stride)
    static constexpr size_t o_tile_offset = do_tile_1_offset + mla_al256(do_tile_size);
    static constexpr size_t o_tile_size = MLA_BLOCK_M * MLA_DVO * sizeof(__nv_bfloat16);

    // Scores / probs / dscores - float [M x N]
    static constexpr size_t scores_offset = o_tile_offset + mla_al256(o_tile_size);
    static constexpr size_t scores_size = MLA_BLOCK_M * MLA_BLOCK_N * sizeof(float);
    static constexpr size_t probs_offset = scores_offset + mla_al256(scores_size);
    static constexpr size_t probs_size = MLA_BLOCK_M * MLA_BLOCK_N * sizeof(float);
    static constexpr size_t dscores_offset = probs_offset + mla_al256(probs_size);
    static constexpr size_t dscores_size = MLA_BLOCK_M * MLA_BLOCK_N * sizeof(float);

    // LSE + delta
    static constexpr size_t lse_offset = dscores_offset + mla_al256(dscores_size);
    static constexpr size_t lse_size = MLA_BLOCK_M * sizeof(float);
    static constexpr size_t delta_offset = lse_offset + mla_al256(lse_size);
    static constexpr size_t delta_size = MLA_BLOCK_M * sizeof(float);

    // dQ accumulator (D_QK stride), float
    static constexpr size_t dq_acc_offset = delta_offset + mla_al256(delta_size);
    static constexpr size_t dq_acc_size = MLA_BLOCK_M * MLA_DQK * sizeof(float);

    // WMMA staging (one 16x16 fp32 tile per warp)
    static constexpr size_t wmma_staging_offset = dq_acc_offset + mla_al256(dq_acc_size);
    static constexpr size_t wmma_staging_size = BWD_NUM_WARPS * WMMA_M * WMMA_N * sizeof(float);

    // temp bf16 buffer (reused for dscores->bf16 and transposes), [N x M] worst case
    static constexpr size_t temp_bf16_offset = wmma_staging_offset + mla_al256(wmma_staging_size);
    static constexpr size_t temp_bf16_size = MLA_BLOCK_M * MLA_BLOCK_N * sizeof(__nv_bfloat16);

    // dK accumulator (D_QK stride) + dV accumulator (D_VO stride), float
    static constexpr size_t dk_acc_offset = temp_bf16_offset + mla_al256(temp_bf16_size);
    static constexpr size_t dk_acc_size = MLA_BLOCK_N * MLA_DQK * sizeof(float);
    static constexpr size_t dv_acc_offset = dk_acc_offset + mla_al256(dk_acc_size);
    static constexpr size_t dv_acc_size = MLA_BLOCK_N * MLA_DVO * sizeof(float);

    static constexpr size_t total_size = dv_acc_offset + mla_al256(dv_acc_size);
};

struct KMajorMlaSmemAccessor {
    char* base;
    int cur_buf;
    __device__ __forceinline__ void init(char* b, int buf = 0) { base = b; cur_buf = buf; }
    __device__ __forceinline__ void set_buffer(int b) { cur_buf = b; }

    __device__ __forceinline__ __nv_bfloat16* q_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base +
            (cur_buf == 0 ? KMajorMlaSmemLayout::q_tile_0_offset : KMajorMlaSmemLayout::q_tile_1_offset));
    }
    __device__ __forceinline__ __nv_bfloat16* q_tile_buf(int b) {
        return reinterpret_cast<__nv_bfloat16*>(base +
            (b == 0 ? KMajorMlaSmemLayout::q_tile_0_offset : KMajorMlaSmemLayout::q_tile_1_offset));
    }
    __device__ __forceinline__ __nv_bfloat16* k_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base + KMajorMlaSmemLayout::k_tile_offset);
    }
    __device__ __forceinline__ __nv_bfloat16* v_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base + KMajorMlaSmemLayout::v_tile_offset);
    }
    __device__ __forceinline__ __nv_bfloat16* do_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base +
            (cur_buf == 0 ? KMajorMlaSmemLayout::do_tile_0_offset : KMajorMlaSmemLayout::do_tile_1_offset));
    }
    __device__ __forceinline__ __nv_bfloat16* do_tile_buf(int b) {
        return reinterpret_cast<__nv_bfloat16*>(base +
            (b == 0 ? KMajorMlaSmemLayout::do_tile_0_offset : KMajorMlaSmemLayout::do_tile_1_offset));
    }
    __device__ __forceinline__ __nv_bfloat16* o_tile() {
        return reinterpret_cast<__nv_bfloat16*>(base + KMajorMlaSmemLayout::o_tile_offset);
    }
    __device__ __forceinline__ float* scores() { return reinterpret_cast<float*>(base + KMajorMlaSmemLayout::scores_offset); }
    __device__ __forceinline__ float* probs()  { return reinterpret_cast<float*>(base + KMajorMlaSmemLayout::probs_offset); }
    __device__ __forceinline__ float* dscores(){ return reinterpret_cast<float*>(base + KMajorMlaSmemLayout::dscores_offset); }
    __device__ __forceinline__ float* lse()    { return reinterpret_cast<float*>(base + KMajorMlaSmemLayout::lse_offset); }
    __device__ __forceinline__ float* delta()  { return reinterpret_cast<float*>(base + KMajorMlaSmemLayout::delta_offset); }
    __device__ __forceinline__ float* dq_acc() { return reinterpret_cast<float*>(base + KMajorMlaSmemLayout::dq_acc_offset); }
    __device__ __forceinline__ float* wmma_staging() { return reinterpret_cast<float*>(base + KMajorMlaSmemLayout::wmma_staging_offset); }
    __device__ __forceinline__ float* wmma_staging_warp(int w) { return wmma_staging() + w * WMMA_M * WMMA_N; }
    __device__ __forceinline__ __nv_bfloat16* temp_bf16() { return reinterpret_cast<__nv_bfloat16*>(base + KMajorMlaSmemLayout::temp_bf16_offset); }
    __device__ __forceinline__ float* dk_acc() { return reinterpret_cast<float*>(base + KMajorMlaSmemLayout::dk_acc_offset); }
    __device__ __forceinline__ float* dv_acc() { return reinterpret_cast<float*>(base + KMajorMlaSmemLayout::dv_acc_offset); }

    // Row accessors with the correct per-tensor stride
    __device__ __forceinline__ __nv_bfloat16* q_row(int m)  { return q_tile() + m * MLA_DQK; }
    __device__ __forceinline__ __nv_bfloat16* q_row_buf(int m, int b) { return q_tile_buf(b) + m * MLA_DQK; }
    __device__ __forceinline__ __nv_bfloat16* k_row(int n)  { return k_tile() + n * MLA_DQK; }
    __device__ __forceinline__ __nv_bfloat16* v_row(int n)  { return v_tile() + n * MLA_DVO; }
    __device__ __forceinline__ __nv_bfloat16* do_row(int m) { return do_tile() + m * MLA_DVO; }
    __device__ __forceinline__ __nv_bfloat16* do_row_buf(int m, int b) { return do_tile_buf(b) + m * MLA_DVO; }
    __device__ __forceinline__ __nv_bfloat16* o_row(int m)  { return o_tile() + m * MLA_DVO; }
    __device__ __forceinline__ float* scores_row(int m)  { return scores() + m * MLA_BLOCK_N; }
    __device__ __forceinline__ float* probs_row(int m)   { return probs() + m * MLA_BLOCK_N; }
    __device__ __forceinline__ float* dscores_row(int m) { return dscores() + m * MLA_BLOCK_N; }
};

// ---- Load helpers (warp-per-row, vectorized bf16x2) -------------------------
// Generic row loader: copies `dim` bf16 elements per row, dst stride = dim.
__device__ __forceinline__ void mla_load_rows(
    __nv_bfloat16* dst_base, const __nv_bfloat16* src,
    int seq_start, int start, int count, int num_heads, int head_idx, int dim) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int stride_token = num_heads * dim;
    for (int r = warp_id; r < count; r += num_warps) {
        const __nv_bfloat16* src_row = src + (seq_start + start + r) * stride_token + head_idx * dim;
        __nv_bfloat16* dst_row = dst_base + r * dim;
        for (int d = lane_id * 2; d < dim; d += 64) {
            if (d + 1 < dim) {
                *reinterpret_cast<__nv_bfloat162*>(dst_row + d) = *reinterpret_cast<const __nv_bfloat162*>(src_row + d);
            } else if (d < dim) {
                dst_row[d] = src_row[d];
            }
        }
    }
}

// Async (cp.async) row loader for double-buffered Q/dO. dim must be multiple of 8.
__device__ __forceinline__ void mla_async_load_rows(
    __nv_bfloat16* dst_base, const __nv_bfloat16* src,
    int seq_start, int start, int count, int num_heads, int head_idx, int dim) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BWD_NUM_THREADS / 32;
    const int stride_token = num_heads * dim;
    for (int r = warp_id; r < count; r += num_warps) {
        const __nv_bfloat16* src_row = src + (seq_start + start + r) * stride_token + head_idx * dim;
        __nv_bfloat16* dst_row = dst_base + r * dim;
        for (int d = lane_id * 8; d < dim; d += 256) {
            if (d + 8 <= dim) {
                cp_async_cg_16(dst_row + d, src_row + d);
            } else {
                for (int dd = d; dd < dim && dd < d + 8; ++dd) dst_row[dd] = src_row[dd];
            }
        }
    }
}

// ---- delta[m] = sum_d O[m,d]*dO[m,d]  (over D_VO) ---------------------------
__device__ __forceinline__ void mla_compute_delta(KMajorMlaSmemAccessor& smem, int m_size) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    for (int m = warp_id; m < m_size; m += BWD_NUM_WARPS) {
        __nv_bfloat16* o_row = smem.o_row(m);
        __nv_bfloat16* do_row = smem.do_row(m);
        float sum = 0.0f;
        for (int d = lane_id * 2; d < MLA_DVO; d += 64) {
            __nv_bfloat162 ov = *reinterpret_cast<__nv_bfloat162*>(o_row + d);
            __nv_bfloat162 dv = *reinterpret_cast<__nv_bfloat162*>(do_row + d);
            sum += __bfloat162float(__low2bfloat16(ov)) * __bfloat162float(__low2bfloat16(dv))
                 + __bfloat162float(__high2bfloat16(ov)) * __bfloat162float(__high2bfloat16(dv));
        }
        for (int off = 16; off > 0; off /= 2) sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (lane_id == 0) smem.delta()[m] = sum;
    }
    __syncthreads();
}

// ---- S[m,n] = scale * Q[m,:] . K[n,:]  (contract over D_QK=192) -------------
__device__ __forceinline__ void mla_compute_qk_scores(KMajorMlaSmemAccessor& smem, int m_size, int n_size, float scale) {
    using namespace nvcuda::wmma;
    const int warp_id = threadIdx.x / 32;
    const int m_tiles = (m_size + WMMA_M - 1) / WMMA_M;
    const int n_tiles = (n_size + WMMA_N - 1) / WMMA_N;
    const int d_tiles = MLA_DQK / WMMA_K;  // 12
    for (int mn = warp_id; mn < m_tiles * n_tiles; mn += BWD_NUM_WARPS) {
        int m_tile = mn / n_tiles, n_tile = mn % n_tiles;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        fill_fragment(acc, 0.0f);
        #pragma unroll
        for (int k_tile = 0; k_tile < d_tiles; ++k_tile) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;
            load_matrix_sync(a_frag, smem.q_row(m_tile * WMMA_M) + k_tile * WMMA_K, MLA_DQK);
            load_matrix_sync(b_frag, smem.k_row(n_tile * WMMA_N) + k_tile * WMMA_K, MLA_DQK);
            mma_sync(acc, a_frag, b_frag, acc);
        }
        #pragma unroll
        for (int i = 0; i < acc.num_elements; ++i) acc.x[i] *= scale;
        store_matrix_sync(smem.scores_row(m_tile * WMMA_M) + n_tile * WMMA_N, acc, MLA_BLOCK_N, mem_row_major);
    }
    __syncthreads();
}

// ---- P = causal-masked softmax(S) using forward LSE -------------------------
__device__ __forceinline__ void mla_recompute_softmax(
    KMajorMlaSmemAccessor& smem, int m_size, int n_size, bool is_causal, int m_start_g, int n_start_g) {
    const int tid = threadIdx.x;
    #pragma unroll 4
    for (int idx = tid; idx < MLA_BLOCK_M * MLA_BLOCK_N; idx += BWD_NUM_THREADS) {
        int m = idx / MLA_BLOCK_N;
        int n = idx % MLA_BLOCK_N;
        float prob = 0.0f;
        if (m < m_size && n < n_size) {
            bool masked = is_causal && ((m_start_g + m) < (n_start_g + n));
            prob = masked ? 0.0f : expf(smem.scores()[idx] - smem.lse()[m]);
        }
        smem.probs()[idx] = prob;
    }
    __syncthreads();
}

// ---- dP[m,n] = dO[m,:] . V[n,:]  (contract over D_VO=128) -> dscores buf ----
__device__ __forceinline__ void mla_compute_dp(KMajorMlaSmemAccessor& smem, int m_size, int n_size) {
    using namespace nvcuda::wmma;
    const int warp_id = threadIdx.x / 32;
    const int m_tiles = (m_size + WMMA_M - 1) / WMMA_M;
    const int n_tiles = (n_size + WMMA_N - 1) / WMMA_N;
    const int d_tiles = MLA_DVO / WMMA_K;  // 8
    for (int mn = warp_id; mn < m_tiles * n_tiles; mn += BWD_NUM_WARPS) {
        int m_tile = mn / n_tiles, n_tile = mn % n_tiles;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        fill_fragment(acc, 0.0f);
        #pragma unroll
        for (int k_tile = 0; k_tile < d_tiles; ++k_tile) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> do_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> v_frag;
            load_matrix_sync(do_frag, smem.do_row(m_tile * WMMA_M) + k_tile * WMMA_K, MLA_DVO);
            load_matrix_sync(v_frag, smem.v_row(n_tile * WMMA_N) + k_tile * WMMA_K, MLA_DVO);
            mma_sync(acc, do_frag, v_frag, acc);
        }
        store_matrix_sync(smem.dscores_row(m_tile * WMMA_M) + n_tile * WMMA_N, acc, MLA_BLOCK_N, mem_row_major);
    }
    __syncthreads();
}

// ---- dScores = (dP - delta) * P * scale  (in place in dscores buf) ----------
__device__ __forceinline__ void mla_compute_dscores(KMajorMlaSmemAccessor& smem, int m_size, int n_size, float scale) {
    const int tid = threadIdx.x;
    #pragma unroll 4
    for (int idx = tid; idx < MLA_BLOCK_M * MLA_BLOCK_N; idx += BWD_NUM_THREADS) {
        int m = idx / MLA_BLOCK_N;
        int n = idx % MLA_BLOCK_N;
        float ds = 0.0f;
        if (m < m_size && n < n_size) {
            ds = (smem.dscores()[idx] - smem.delta()[m]) * smem.probs()[idx] * scale;
        }
        smem.dscores()[idx] = ds;
    }
    __syncthreads();
}

// ---- dQ[m,:] += dScores[m,:] @ K  (over n; out D_QK) -> dq_acc --------------
__device__ __forceinline__ void mla_compute_dq(KMajorMlaSmemAccessor& smem, int m_size, int n_size) {
    using namespace nvcuda::wmma;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid = threadIdx.x;

    // dscores -> bf16 (full MLA_BLOCK_M x MLA_BLOCK_N tile, zero invalid)
    __nv_bfloat16* ds_bf16 = smem.temp_bf16();
    #pragma unroll 4
    for (int idx = tid; idx < MLA_BLOCK_M * MLA_BLOCK_N; idx += BWD_NUM_THREADS) {
        ds_bf16[idx] = __float2bfloat16(smem.dscores()[idx]);
    }
    __syncthreads();

    const int m_tiles = (m_size + WMMA_M - 1) / WMMA_M;
    const int d_tiles = MLA_DQK / WMMA_N;   // 12
    const int k_tiles = (n_size + WMMA_K - 1) / WMMA_K;
    for (int md = warp_id; md < m_tiles * d_tiles; md += BWD_NUM_WARPS) {
        int m_tile = md / d_tiles, d_tile = md % d_tiles;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        fill_fragment(acc, 0.0f);
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> ds_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> k_frag;
            load_matrix_sync(ds_frag, ds_bf16 + m_tile * WMMA_M * MLA_BLOCK_N + k_tile * WMMA_K, MLA_BLOCK_N);
            load_matrix_sync(k_frag, smem.k_row(k_tile * WMMA_K) + d_tile * WMMA_N, MLA_DQK);
            mma_sync(acc, ds_frag, k_frag, acc);
        }
        float* staging = smem.wmma_staging_warp(warp_id);
        store_matrix_sync(staging, acc, WMMA_N, mem_row_major);
        __syncwarp();
        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int row = i / WMMA_N, col = i % WMMA_N;
            int gr = m_tile * WMMA_M + row, gc = d_tile * WMMA_N + col;
            if (gr < m_size && gc < MLA_DQK) {
                smem.dq_acc()[gr * MLA_DQK + gc] += staging[i];
            }
        }
        __syncwarp();
    }
    __syncthreads();
}

// ---- dK[n,:] += dScores^T[n,:] @ Q  (over m; out D_QK) -> dk_acc ------------
__device__ __forceinline__ void mla_accumulate_dk(KMajorMlaSmemAccessor& smem, int m_size, int n_size) {
    using namespace nvcuda::wmma;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid = threadIdx.x;

    // transpose dscores [M,N] -> ds_t [N,M] bf16 (stride MLA_BLOCK_M), zero invalid
    __nv_bfloat16* ds_t = smem.temp_bf16();
    #pragma unroll 4
    for (int dst = tid; dst < MLA_BLOCK_N * MLA_BLOCK_M; dst += BWD_NUM_THREADS) {
        int n = dst / MLA_BLOCK_M, m = dst % MLA_BLOCK_M;
        float v = (m < m_size && n < n_size) ? smem.dscores()[m * MLA_BLOCK_N + n] : 0.0f;
        ds_t[dst] = __float2bfloat16(v);
    }
    __syncthreads();

    const int n_tiles = (n_size + WMMA_M - 1) / WMMA_M;
    const int d_tiles = MLA_DQK / WMMA_N;   // 12
    const int k_tiles = (m_size + WMMA_K - 1) / WMMA_K;
    for (int nd = warp_id; nd < n_tiles * d_tiles; nd += BWD_NUM_WARPS) {
        int n_tile = nd / d_tiles, d_tile = nd % d_tiles;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        fill_fragment(acc, 0.0f);
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;  // ds_t[N,M]
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> q_frag;  // Q[M,D]
            load_matrix_sync(a_frag, ds_t + n_tile * WMMA_M * MLA_BLOCK_M + k_tile * WMMA_K, MLA_BLOCK_M);
            load_matrix_sync(q_frag, smem.q_row(k_tile * WMMA_K) + d_tile * WMMA_N, MLA_DQK);
            mma_sync(acc, a_frag, q_frag, acc);
        }
        float* staging = smem.wmma_staging_warp(warp_id);
        store_matrix_sync(staging, acc, WMMA_N, mem_row_major);
        __syncwarp();
        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int ln = i / WMMA_N, ld = i % WMMA_N;
            int gn = n_tile * WMMA_M + ln, gd = d_tile * WMMA_N + ld;
            if (gn < n_size && gd < MLA_DQK) {
                smem.dk_acc()[gn * MLA_DQK + gd] += staging[i];
            }
        }
        __syncwarp();
    }
    __syncthreads();
}

// ---- dV[n,:] += P^T[n,:] @ dO  (over m; out D_VO) -> dv_acc -----------------
__device__ __forceinline__ void mla_accumulate_dv(KMajorMlaSmemAccessor& smem, int m_size, int n_size) {
    using namespace nvcuda::wmma;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid = threadIdx.x;

    // transpose probs [M,N] -> p_t [N,M] bf16 (stride MLA_BLOCK_M), zero invalid
    __nv_bfloat16* p_t = smem.temp_bf16();
    #pragma unroll 4
    for (int dst = tid; dst < MLA_BLOCK_N * MLA_BLOCK_M; dst += BWD_NUM_THREADS) {
        int n = dst / MLA_BLOCK_M, m = dst % MLA_BLOCK_M;
        float v = (m < m_size && n < n_size) ? smem.probs()[m * MLA_BLOCK_N + n] : 0.0f;
        p_t[dst] = __float2bfloat16(v);
    }
    __syncthreads();

    const int n_tiles = (n_size + WMMA_M - 1) / WMMA_M;
    const int d_tiles = MLA_DVO / WMMA_N;   // 8
    const int k_tiles = (m_size + WMMA_K - 1) / WMMA_K;
    for (int nd = warp_id; nd < n_tiles * d_tiles; nd += BWD_NUM_WARPS) {
        int n_tile = nd / d_tiles, d_tile = nd % d_tiles;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        fill_fragment(acc, 0.0f);
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;  // p_t[N,M]
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> do_frag; // dO[M,D]
            load_matrix_sync(a_frag, p_t + n_tile * WMMA_M * MLA_BLOCK_M + k_tile * WMMA_K, MLA_BLOCK_M);
            load_matrix_sync(do_frag, smem.do_row(k_tile * WMMA_K) + d_tile * WMMA_N, MLA_DVO);
            mma_sync(acc, a_frag, do_frag, acc);
        }
        float* staging = smem.wmma_staging_warp(warp_id);
        store_matrix_sync(staging, acc, WMMA_N, mem_row_major);
        __syncwarp();
        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int ln = i / WMMA_N, ld = i % WMMA_N;
            int gn = n_tile * WMMA_M + ln, gd = d_tile * WMMA_N + ld;
            if (gn < n_size && gd < MLA_DVO) {
                smem.dv_acc()[gn * MLA_DVO + gd] += staging[i];
            }
        }
        __syncwarp();
    }
    __syncthreads();
}

// ============================================================================
// MLA K-major kernel: dQ (atomic) + dK/dV (smem) for head_dim_qk=192, vo=128
// ============================================================================
template<bool kIsCausal>
__global__ void __launch_bounds__(BWD_NUM_THREADS)
fmha_bwd_sm120_mla_kernel(
    const __nv_bfloat16* __restrict__ d_o,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ o,
    const float* __restrict__ lse,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ cu_seqlens_kv,
    float* __restrict__ dq,   // [total_q, num_heads, 192] float
    float* __restrict__ dk,   // [total_kv, num_heads, 192] float
    float* __restrict__ dv,   // [total_kv, num_heads, 128] float
    int num_heads,
    float scale,
    int max_seqlen_q,
    int max_seqlen_kv
) {
    extern __shared__ char smem_base[];
    KMajorMlaSmemAccessor smem;
    smem.init(smem_base, 0);

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int n_block_idx = blockIdx.x;
    if (head_idx >= num_heads) return;

    const int q_start = cu_seqlens_q[batch_idx];
    const int q_end   = cu_seqlens_q[batch_idx + 1];
    const int kv_start = cu_seqlens_kv[batch_idx];
    const int kv_end   = cu_seqlens_kv[batch_idx + 1];
    const int seq_len_q = q_end - q_start;
    const int seq_len_kv = kv_end - kv_start;

    const int n_start = n_block_idx * MLA_BLOCK_N;
    if (n_start >= seq_len_kv) return;
    const int n_end = min(n_start + MLA_BLOCK_N, seq_len_kv);
    const int n_size = n_end - n_start;
    const int tid = threadIdx.x;

    // zero dk_acc / dv_acc
    for (int i = tid; i < MLA_BLOCK_N * MLA_DQK; i += BWD_NUM_THREADS) smem.dk_acc()[i] = 0.0f;
    for (int i = tid; i < MLA_BLOCK_N * MLA_DVO; i += BWD_NUM_THREADS) smem.dv_acc()[i] = 0.0f;
    __syncthreads();

    // load K (D_QK), V (D_VO) once
    mla_load_rows(smem.k_tile(), k, kv_start, n_start, n_size, num_heads, head_idx, MLA_DQK);
    mla_load_rows(smem.v_tile(), v, kv_start, n_start, n_size, num_heads, head_idx, MLA_DVO);
    __syncthreads();

    const int num_q_blocks = (seq_len_q + MLA_BLOCK_M - 1) / MLA_BLOCK_M;

    // find first non-skipped Q block (causal)
    int first_m = 0;
    if (kIsCausal) {
        for (int m = 0; m < num_q_blocks; ++m) {
            int ms = m * MLA_BLOCK_M, me = min(ms + MLA_BLOCK_M, seq_len_q);
            if ((ms + (me - ms) - 1) >= n_start) { first_m = m; break; }
        }
    }
    int cur_buf = 0;
    if (first_m < num_q_blocks) {
        int ms = first_m * MLA_BLOCK_M, me = min(ms + MLA_BLOCK_M, seq_len_q);
        mla_async_load_rows(smem.q_row_buf(0, 0), q, q_start, ms, me - ms, num_heads, head_idx, MLA_DQK);
        mla_async_load_rows(smem.do_row_buf(0, 0), d_o, q_start, ms, me - ms, num_heads, head_idx, MLA_DVO);
        cp_async_commit_group();
    }

    const int stride_q_tok = num_heads * MLA_DQK;

    for (int m_block = first_m; m_block < num_q_blocks; ++m_block) {
        const int m_start = m_block * MLA_BLOCK_M;
        const int m_end = min(m_start + MLA_BLOCK_M, seq_len_q);
        const int m_size = m_end - m_start;
        if (kIsCausal && (m_start + m_size - 1) < n_start) continue;

        // prefetch next non-skipped block
        int next_m = -1;
        for (int nm = m_block + 1; nm < num_q_blocks; ++nm) {
            int s = nm * MLA_BLOCK_M, e = min(s + MLA_BLOCK_M, seq_len_q);
            if (!kIsCausal || (s + (e - s) - 1) >= n_start) { next_m = nm; break; }
        }
        if (next_m >= 0) {
            int nb = 1 - cur_buf;
            int s = next_m * MLA_BLOCK_M, e = min(s + MLA_BLOCK_M, seq_len_q);
            mla_async_load_rows(smem.q_row_buf(0, nb), q, q_start, s, e - s, num_heads, head_idx, MLA_DQK);
            mla_async_load_rows(smem.do_row_buf(0, nb), d_o, q_start, s, e - s, num_heads, head_idx, MLA_DVO);
            cp_async_commit_group();
        }
        cp_async_wait_all();
        __syncthreads();
        smem.set_buffer(cur_buf);

        // zero dq_acc for this Q-block
        for (int i = tid; i < MLA_BLOCK_M * MLA_DQK; i += BWD_NUM_THREADS) smem.dq_acc()[i] = 0.0f;

        // load O (D_VO) and LSE (sync)
        mla_load_rows(smem.o_tile(), o, q_start, m_start, m_size, num_heads, head_idx, MLA_DVO);
        for (int m = tid; m < m_size; m += BWD_NUM_THREADS)
            smem.lse()[m] = lse[(q_start + m_start + m) * num_heads + head_idx];
        __syncthreads();

        mla_compute_delta(smem, m_size);
        mla_compute_qk_scores(smem, m_size, n_size, scale);
        mla_recompute_softmax(smem, m_size, n_size, kIsCausal, m_start, n_start);
        mla_compute_dp(smem, m_size, n_size);
        mla_compute_dscores(smem, m_size, n_size, scale);

        mla_compute_dq(smem, m_size, n_size);
        // atomicAdd dq_acc -> global dq (D_QK=192 is NOT power of 2 -> div/mod)
        #pragma unroll 4
        for (int idx = tid; idx < m_size * MLA_DQK; idx += BWD_NUM_THREADS) {
            int m = idx / MLA_DQK, d = idx % MLA_DQK;
            float val = smem.dq_acc()[m * MLA_DQK + d];
            if (val != 0.0f) {
                int gt = q_start + m_start + m;
                atomicAdd(&dq[gt * stride_q_tok + head_idx * MLA_DQK + d], val);
            }
        }
        __syncthreads();

        mla_accumulate_dk(smem, m_size, n_size);
        mla_accumulate_dv(smem, m_size, n_size);

        if (next_m >= 0) cur_buf = 1 - cur_buf;
    }

    // write dK (D_QK) and dV (D_VO) to global (single non-atomic write)
    const int stride_k_tok = num_heads * MLA_DQK;
    const int stride_v_tok = num_heads * MLA_DVO;
    for (int idx = tid; idx < n_size * MLA_DQK; idx += BWD_NUM_THREADS) {
        int n = idx / MLA_DQK, d = idx % MLA_DQK;
        int gt = kv_start + n_start + n;
        dk[gt * stride_k_tok + head_idx * MLA_DQK + d] = smem.dk_acc()[n * MLA_DQK + d];
    }
    for (int idx = tid; idx < n_size * MLA_DVO; idx += BWD_NUM_THREADS) {
        int n = idx / MLA_DVO, d = idx % MLA_DVO;
        int gt = kv_start + n_start + n;
        dv[gt * stride_v_tok + head_idx * MLA_DVO + d] = smem.dv_acc()[n * MLA_DVO + d];
    }
}

// ---- Launcher ---------------------------------------------------------------
template<bool kIsCausal>
void launch_fmha_bwd_sm120_mla(
    const c10::cuda::CUDAStream& stream,
    at::Tensor d_o, at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o, at::Tensor lse,
    at::Tensor cu_seqlens_q, at::Tensor cu_seqlens_kv,
    at::Tensor dq, at::Tensor dk, at::Tensor dv,
    float scale, int max_seqlen_q, int max_seqlen_kv) {
    const int batch_size = cu_seqlens_q.size(0) - 1;
    const int num_heads = q.size(1);
    const int n_blocks = (max_seqlen_kv + MLA_BLOCK_N - 1) / MLA_BLOCK_N;
    dim3 grid(n_blocks, num_heads, batch_size);
    dim3 block(BWD_NUM_THREADS);
    size_t smem_size = KMajorMlaSmemLayout::total_size;
    cudaFuncSetAttribute(fmha_bwd_sm120_mla_kernel<kIsCausal>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    fmha_bwd_sm120_mla_kernel<kIsCausal><<<grid, block, smem_size, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(d_o.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()),
        lse.data_ptr<float>(),
        cu_seqlens_q.data_ptr<int>(), cu_seqlens_kv.data_ptr<int>(),
        dq.data_ptr<float>(), dk.data_ptr<float>(), dv.data_ptr<float>(),
        num_heads, scale, max_seqlen_q, max_seqlen_kv);
}

}  // namespace detail
}  // namespace flash
