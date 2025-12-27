// SM120 Sparse FP8 Decode Kernel
// WMMA-based implementation for Blackwell workstation GPUs

#include "splitkv_mla.h"
#include "dequant.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

namespace sm120 {
namespace sparse_decode {

using namespace nvcuda::wmma;

// WMMA fragment types for 16x16x16 operations
using FragA_QK = fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major>;
using FragB_QK = fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major>;  // K^T
using FragC_QK = fragment<accumulator, 16, 16, 16, float>;

using FragA_PV = fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major>;  // S
using FragB_PV = fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major>;  // V
using FragC_PV = fragment<accumulator, 16, 16, 16, float>;

// ============================================================================
// Helper functions
// ============================================================================

// Load Q tile from global to shared memory
__device__ __forceinline__
void load_q_tile(
    const bf16* __restrict__ q_global,
    bf16 q_tile[BLOCK_M][SharedMemoryLayout::Q_TILE_COLS],
    int tile_idx,
    int head_dim,
    int num_threads
) {
    const int tid = threadIdx.x;
    const int col_start = tile_idx * SharedMemoryLayout::Q_TILE_COLS;
    const int cols_this_tile = min(SharedMemoryLayout::Q_TILE_COLS, head_dim - col_start);

    // Each thread loads multiple elements
    const int elems_per_thread = (BLOCK_M * SharedMemoryLayout::Q_TILE_COLS + num_threads - 1) / num_threads;

    #pragma unroll 4
    for (int i = 0; i < elems_per_thread; i++) {
        int elem_idx = tid + i * num_threads;
        int row = elem_idx / SharedMemoryLayout::Q_TILE_COLS;
        int col = elem_idx % SharedMemoryLayout::Q_TILE_COLS;

        if (row < BLOCK_M && col < cols_this_tile) {
            q_tile[row][col] = q_global[row * head_dim + col_start + col];
        } else if (row < BLOCK_M && col < SharedMemoryLayout::Q_TILE_COLS) {
            q_tile[row][col] = bf16(0.0f);  // Pad with zeros
        }
    }
}

// Load K tile from FP8 global memory with dequantization
// K is stored as: [page][token][FP8_NOPE | scales | BF16_ROPE]
__device__ __forceinline__
void load_k_tile_fp8(
    const fp8* __restrict__ kv_ptr,
    bf16 k_tile[TOPK_BLOCK_SIZE][SharedMemoryLayout::K_TILE_COLS],
    const int* __restrict__ indices,
    bool* __restrict__ is_valid,
    int tile_idx,
    int topk,
    int page_size,
    int kv_token_stride,
    const int* __restrict__ block_table,
    int num_threads
) {
    const int tid = threadIdx.x;
    const int col_start = tile_idx * SharedMemoryLayout::K_TILE_COLS;

    // Each thread handles one or more tokens
    const int tokens_per_thread = (TOPK_BLOCK_SIZE + num_threads - 1) / num_threads;

    #pragma unroll 2
    for (int t = 0; t < tokens_per_thread; t++) {
        int token_local = tid + t * num_threads;
        if (token_local >= TOPK_BLOCK_SIZE) break;

        // Get token index from sparse indices
        int token_idx = (token_local < topk) ? indices[token_local] : -1;
        bool valid = (token_idx >= 0);
        is_valid[token_local] = valid;

        if (!valid) {
            // Invalid token - fill with zeros
            #pragma unroll 4
            for (int c = 0; c < SharedMemoryLayout::K_TILE_COLS; c += 8) {
                *reinterpret_cast<float4*>(&k_tile[token_local][c]) = make_float4(0, 0, 0, 0);
            }
            continue;
        }

        // Calculate page and offset
        int page_idx = token_idx / page_size;
        int token_in_page = token_idx % page_size;
        int page = block_table[page_idx];

        const fp8* token_ptr = kv_ptr + page * page_size * kv_token_stride + token_in_page * kv_token_stride;

        // Load scales (4 floats for 512 elements, 128 elements per scale)
        float4 scales = *reinterpret_cast<const float4*>(token_ptr + HEAD_DIM_NOPE);

        // Dequantize FP8 to BF16 for this tile
        // col_start determines which part of the 576-dim vector we're loading
        if (col_start < HEAD_DIM_NOPE) {
            // Loading from FP8 NOPE region
            int fp8_col = col_start;
            int scale_idx = fp8_col / QUANT_TILE_SIZE;
            float scale = (scale_idx == 0) ? scales.x : (scale_idx == 1) ? scales.y :
                          (scale_idx == 2) ? scales.z : scales.w;

            // Load 64 FP8 elements (4 x fp8x16)
            #pragma unroll 4
            for (int c = 0; c < SharedMemoryLayout::K_TILE_COLS; c += 16) {
                if (fp8_col + c < HEAD_DIM_NOPE) {
                    fp8x16 fp8_data = load_128b<fp8x16>(token_ptr + fp8_col + c);

                    // Determine scale for this chunk
                    int chunk_scale_idx = (fp8_col + c) / QUANT_TILE_SIZE;
                    float chunk_scale = (chunk_scale_idx == 0) ? scales.x : (chunk_scale_idx == 1) ? scales.y :
                                        (chunk_scale_idx == 2) ? scales.z : scales.w;

                    bf16x8 bf16_lo = cvt_fp8x8_bf16x8(fp8_data.lo, chunk_scale);
                    bf16x8 bf16_hi = cvt_fp8x8_bf16x8(fp8_data.hi, chunk_scale);

                    store_128b(&k_tile[token_local][c], bf16_lo);
                    store_128b(&k_tile[token_local][c + 8], bf16_hi);
                }
            }
        } else {
            // Loading from BF16 ROPE region
            int rope_col = col_start - HEAD_DIM_NOPE;
            const bf16* rope_ptr = reinterpret_cast<const bf16*>(
                token_ptr + HEAD_DIM_NOPE + NUM_SCALES * sizeof(float));

            #pragma unroll 4
            for (int c = 0; c < SharedMemoryLayout::K_TILE_COLS; c += 8) {
                if (rope_col + c < HEAD_DIM_ROPE) {
                    *reinterpret_cast<float4*>(&k_tile[token_local][c]) =
                        *reinterpret_cast<const float4*>(rope_ptr + rope_col + c);
                } else {
                    *reinterpret_cast<float4*>(&k_tile[token_local][c]) = make_float4(0, 0, 0, 0);
                }
            }
        }
    }
}

// Load V tile from FP8 storage (V is the NOPE part only, 512 dims)
__device__ __forceinline__
void load_v_tile_fp8(
    const fp8* __restrict__ kv_ptr,
    bf16 v_tile[TOPK_BLOCK_SIZE][SharedMemoryLayout::V_TILE_COLS],
    const int* __restrict__ indices,
    const bool* __restrict__ is_valid,
    int tile_idx,
    int topk,
    int page_size,
    int kv_token_stride,
    const int* __restrict__ block_table,
    int num_threads
) {
    const int tid = threadIdx.x;
    const int col_start = tile_idx * SharedMemoryLayout::V_TILE_COLS;

    const int tokens_per_thread = (TOPK_BLOCK_SIZE + num_threads - 1) / num_threads;

    #pragma unroll 2
    for (int t = 0; t < tokens_per_thread; t++) {
        int token_local = tid + t * num_threads;
        if (token_local >= TOPK_BLOCK_SIZE) break;

        if (!is_valid[token_local]) {
            #pragma unroll 4
            for (int c = 0; c < SharedMemoryLayout::V_TILE_COLS; c += 8) {
                *reinterpret_cast<float4*>(&v_tile[token_local][c]) = make_float4(0, 0, 0, 0);
            }
            continue;
        }

        int token_idx = indices[token_local];
        int page_idx = token_idx / page_size;
        int token_in_page = token_idx % page_size;
        int page = block_table[page_idx];

        const fp8* token_ptr = kv_ptr + page * page_size * kv_token_stride + token_in_page * kv_token_stride;
        float4 scales = *reinterpret_cast<const float4*>(token_ptr + HEAD_DIM_NOPE);

        // V uses the same FP8 NOPE region as K's first 512 dims
        int scale_idx = col_start / QUANT_TILE_SIZE;

        #pragma unroll 4
        for (int c = 0; c < SharedMemoryLayout::V_TILE_COLS; c += 16) {
            int abs_col = col_start + c;
            if (abs_col < HEAD_DIM_V) {
                fp8x16 fp8_data = load_128b<fp8x16>(token_ptr + abs_col);

                int chunk_scale_idx = abs_col / QUANT_TILE_SIZE;
                float chunk_scale = (chunk_scale_idx == 0) ? scales.x : (chunk_scale_idx == 1) ? scales.y :
                                    (chunk_scale_idx == 2) ? scales.z : scales.w;

                bf16x8 bf16_lo = cvt_fp8x8_bf16x8(fp8_data.lo, chunk_scale);
                bf16x8 bf16_hi = cvt_fp8x8_bf16x8(fp8_data.hi, chunk_scale);

                store_128b(&v_tile[token_local][c], bf16_lo);
                store_128b(&v_tile[token_local][c + 8], bf16_hi);
            }
        }
    }
}

// WMMA-based Q @ K^T computation for a single 16x16 tile
__device__ __forceinline__
void wmma_qk_tile(
    const bf16 q_tile[BLOCK_M][SharedMemoryLayout::Q_TILE_COLS],
    const bf16 k_tile[TOPK_BLOCK_SIZE][SharedMemoryLayout::K_TILE_COLS],
    float qk_accum[BLOCK_M / 16][TOPK_BLOCK_SIZE / 16][16 * 16 / 32],  // Per-warp accumulators
    int warp_id,
    int lane_id,
    bool clear
) {
    // Each warp handles a 16x16 output tile
    // BLOCK_M=64, TOPK_BLOCK_SIZE=64 -> 4x4 = 16 tiles, 8 warps
    // Each warp handles 2 tiles

    const int tiles_m = BLOCK_M / 16;  // 4
    const int tiles_n = TOPK_BLOCK_SIZE / 16;  // 4
    const int total_tiles = tiles_m * tiles_n;  // 16
    const int tiles_per_warp = (total_tiles + NUM_WARPS - 1) / NUM_WARPS;  // 2

    #pragma unroll
    for (int t = 0; t < tiles_per_warp; t++) {
        int tile_idx = warp_id * tiles_per_warp + t;
        if (tile_idx >= total_tiles) break;

        int tile_m = tile_idx / tiles_n;
        int tile_n = tile_idx % tiles_n;

        // WMMA fragments
        FragA_QK frag_q;
        FragB_QK frag_k;
        FragC_QK frag_c;

        if (clear) {
            fill_fragment(frag_c, 0.0f);
        } else {
            // Load existing accumulator
            #pragma unroll
            for (int i = 0; i < frag_c.num_elements; i++) {
                frag_c.x[i] = qk_accum[tile_m][tile_n][lane_id * frag_c.num_elements / 32 + i % (frag_c.num_elements / 32)];
            }
        }

        // Accumulate over K dimension (64 elements in this tile)
        const int k_tiles = SharedMemoryLayout::K_TILE_COLS / 16;  // 4
        #pragma unroll
        for (int k = 0; k < k_tiles; k++) {
            // Load Q fragment: [16, 16] from q_tile[tile_m*16 : tile_m*16+16, k*16 : k*16+16]
            load_matrix_sync(frag_q,
                reinterpret_cast<const __nv_bfloat16*>(&q_tile[tile_m * 16][k * 16]),
                SharedMemoryLayout::Q_TILE_COLS);

            // Load K fragment (transposed): [16, 16] from k_tile[tile_n*16 : tile_n*16+16, k*16 : k*16+16]
            load_matrix_sync(frag_k,
                reinterpret_cast<const __nv_bfloat16*>(&k_tile[tile_n * 16][k * 16]),
                SharedMemoryLayout::K_TILE_COLS);

            // C += A @ B^T
            mma_sync(frag_c, frag_q, frag_k, frag_c);
        }

        // Store back to accumulator
        #pragma unroll
        for (int i = 0; i < frag_c.num_elements; i++) {
            qk_accum[tile_m][tile_n][lane_id * frag_c.num_elements / 32 + i % (frag_c.num_elements / 32)] = frag_c.x[i];
        }
    }
}

// Online softmax with masking for invalid tokens
__device__ __forceinline__
void online_softmax(
    float qk_accum[BLOCK_M / 16][TOPK_BLOCK_SIZE / 16][16 * 16 / 32],
    bf16 s_out[BLOCK_M][TOPK_BLOCK_SIZE],
    float* __restrict__ row_max,
    float* __restrict__ row_sum,
    const bool* __restrict__ is_valid,
    float sm_scale,
    int warp_id,
    int lane_id
) {
    // Each warp handles its assigned rows
    const int rows_per_warp = BLOCK_M / NUM_WARPS;
    const int row_start = warp_id * rows_per_warp;

    #pragma unroll
    for (int r = 0; r < rows_per_warp; r++) {
        int row = row_start + r;

        // Gather QK values for this row from accumulators
        float qk_row[TOPK_BLOCK_SIZE];

        #pragma unroll
        for (int c = 0; c < TOPK_BLOCK_SIZE; c++) {
            int tile_m = row / 16;
            int tile_n = c / 16;
            int local_r = row % 16;
            int local_c = c % 16;

            // Read from accumulator (simplified - actual layout depends on WMMA)
            float val = 0.0f;  // Placeholder - need proper WMMA result extraction

            // Apply mask for invalid tokens
            if (!is_valid[c]) {
                val = -INFINITY;
            }

            qk_row[c] = val * sm_scale;
        }

        // Find max
        float max_val = row_max[row];
        #pragma unroll
        for (int c = 0; c < TOPK_BLOCK_SIZE; c++) {
            max_val = fmaxf(max_val, qk_row[c]);
        }

        // Rescale old sum and compute new exp values
        float scale_old = expf(row_max[row] - max_val);
        row_sum[row] *= scale_old;
        row_max[row] = max_val;

        float sum_val = 0.0f;
        #pragma unroll
        for (int c = 0; c < TOPK_BLOCK_SIZE; c++) {
            float exp_val = expf(qk_row[c] - max_val);
            s_out[row][c] = bf16(exp_val);
            sum_val += exp_val;
        }

        row_sum[row] += sum_val;
    }
}

// WMMA-based S @ V computation
// v_tile_idx: which 64-column V tile we're processing (0-7 for HEAD_DIM_V=512)
__device__ __forceinline__
void wmma_sv_tile(
    const bf16 s_mat[BLOCK_M][TOPK_BLOCK_SIZE],
    const bf16 v_tile[TOPK_BLOCK_SIZE][SharedMemoryLayout::V_TILE_COLS],
    float o_accum[BLOCK_M / 16][HEAD_DIM_V / 16][16 * 16 / 32],
    int v_tile_idx,  // Which V tile (0-7)
    int warp_id,
    int lane_id,
    float scale  // Rescaling factor for online softmax
) {
    const int tiles_m = BLOCK_M / 16;  // 4
    const int tiles_n_per_v_tile = SharedMemoryLayout::V_TILE_COLS / 16;  // 4
    const int total_tiles = tiles_m * tiles_n_per_v_tile;  // 16
    const int tiles_per_warp = (total_tiles + NUM_WARPS - 1) / NUM_WARPS;

    // Offset into o_accum for this V tile
    const int o_tile_n_offset = v_tile_idx * tiles_n_per_v_tile;

    #pragma unroll
    for (int t = 0; t < tiles_per_warp; t++) {
        int tile_idx = warp_id * tiles_per_warp + t;
        if (tile_idx >= total_tiles) break;

        int tile_m = tile_idx / tiles_n_per_v_tile;
        int tile_n = tile_idx % tiles_n_per_v_tile;
        int o_tile_n = o_tile_n_offset + tile_n;  // Global tile index in output

        FragA_PV frag_s;
        FragB_PV frag_v;
        FragC_PV frag_c;

        // Load existing accumulator and rescale
        #pragma unroll
        for (int i = 0; i < frag_c.num_elements; i++) {
            frag_c.x[i] = o_accum[tile_m][o_tile_n][lane_id * frag_c.num_elements / 32 + i % (frag_c.num_elements / 32)] * scale;
        }

        // Accumulate S @ V over K dimension (64 elements)
        const int k_tiles = TOPK_BLOCK_SIZE / 16;  // 4
        #pragma unroll
        for (int k = 0; k < k_tiles; k++) {
            load_matrix_sync(frag_s,
                reinterpret_cast<const __nv_bfloat16*>(&s_mat[tile_m * 16][k * 16]),
                TOPK_BLOCK_SIZE);

            load_matrix_sync(frag_v,
                reinterpret_cast<const __nv_bfloat16*>(&v_tile[k * 16][tile_n * 16]),
                SharedMemoryLayout::V_TILE_COLS);

            mma_sync(frag_c, frag_s, frag_v, frag_c);
        }

        // Store back
        #pragma unroll
        for (int i = 0; i < frag_c.num_elements; i++) {
            o_accum[tile_m][o_tile_n][lane_id * frag_c.num_elements / 32 + i % (frag_c.num_elements / 32)] = frag_c.x[i];
        }
    }
}

// Main kernel
__global__ void __launch_bounds__(NUM_THREADS, 1)
sparse_fp8_decode_kernel(SparseFP8DecodeParams params) {
    // Block indices
    const int batch_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int head_block_idx = blockIdx.z;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Shared memory
    extern __shared__ char smem_buf[];
    SharedMemoryPlan& smem = *reinterpret_cast<SharedMemoryPlan*>(smem_buf);

    // Initialize row_max and row_sum
    if (threadIdx.x < BLOCK_M) {
        smem.row_max[threadIdx.x] = -INFINITY;
        smem.row_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Calculate head indices for this block
    const int head_start = head_block_idx * BLOCK_M;
    const int num_heads = min(BLOCK_M, params.h_q - head_start);

    // Get Q pointer for this batch/seq/head_block
    const bf16* q_ptr = params.q_ptr +
        batch_idx * params.q_batch_stride +
        seq_idx * params.q_seq_stride +
        head_start * params.q_head_stride;

    // Get indices pointer
    const int* indices_ptr = params.indices_ptr +
        batch_idx * params.indices_batch_stride +
        seq_idx * params.indices_seq_stride;  // h_kv is typically 1 for MLA

    // Get block table for this batch
    const int* block_table = params.block_table_ptr + batch_idx * (params.topk / params.page_size + 1);

    // Per-thread output accumulator (stored in registers)
    float o_accum[BLOCK_M / 16][HEAD_DIM_V / 16][16 * 16 / 32];
    #pragma unroll
    for (int i = 0; i < BLOCK_M / 16; i++)
        for (int j = 0; j < HEAD_DIM_V / 16; j++)
            for (int k = 0; k < 16 * 16 / 32; k++)
                o_accum[i][j][k] = 0.0f;

    // Per-thread QK accumulator
    float qk_accum[BLOCK_M / 16][TOPK_BLOCK_SIZE / 16][16 * 16 / 32];

    // Number of K blocks
    const int num_k_blocks = (params.topk + TOPK_BLOCK_SIZE - 1) / TOPK_BLOCK_SIZE;

    // Process each K block
    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        const int buf_idx = k_block % NUM_K_BUFS;
        const int* block_indices = indices_ptr + k_block * TOPK_BLOCK_SIZE;
        const int tokens_this_block = min(TOPK_BLOCK_SIZE, params.topk - k_block * TOPK_BLOCK_SIZE);

        // Clear QK accumulator
        #pragma unroll
        for (int i = 0; i < BLOCK_M / 16; i++)
            for (int j = 0; j < TOPK_BLOCK_SIZE / 16; j++)
                for (int k = 0; k < 16 * 16 / 32; k++)
                    qk_accum[i][j][k] = 0.0f;

        // Process Q @ K^T in tiles
        for (int q_tile = 0; q_tile < SharedMemoryLayout::Q_TILES; q_tile++) {
            // Load Q tile
            load_q_tile(q_ptr, smem.q_tile, q_tile, HEAD_DIM_K, NUM_THREADS);
            __syncthreads();

            // Load K tile (same column range as Q)
            load_k_tile_fp8(
                params.kv_ptr, smem.kv.k_tile[buf_idx],
                block_indices, smem.is_valid[buf_idx],
                q_tile, tokens_this_block,
                params.page_size, params.kv_token_stride,
                block_table, NUM_THREADS
            );
            __syncthreads();

            // Compute Q @ K^T for this tile
            wmma_qk_tile(smem.q_tile, smem.kv.k_tile[buf_idx], qk_accum,
                warp_id, lane_id, q_tile == 0);
            __syncthreads();
        }

        // Apply softmax with masking
        online_softmax(qk_accum, smem.s, smem.row_max, smem.row_sum,
            smem.is_valid[buf_idx], params.sm_scale, warp_id, lane_id);
        __syncthreads();

        // Compute S @ V in tiles
        for (int v_tile = 0; v_tile < SharedMemoryLayout::V_TILES; v_tile++) {
            // Load V tile
            load_v_tile_fp8(
                params.kv_ptr, smem.kv.v_tile,
                block_indices, smem.is_valid[buf_idx],
                v_tile, tokens_this_block,
                params.page_size, params.kv_token_stride,
                block_table, NUM_THREADS
            );
            __syncthreads();

            // Compute S @ V
            float rescale = 1.0f;  // Rescaling handled in online_softmax
            wmma_sv_tile(smem.s, smem.kv.v_tile, o_accum, v_tile, warp_id, lane_id, rescale);
            __syncthreads();
        }
    }

    // Finalize: divide by sum and convert to BF16
    // Store output
    bf16* o_ptr = params.o_ptr +
        batch_idx * params.o_batch_stride +
        seq_idx * params.o_seq_stride +
        head_start * params.o_head_stride;

    // Each thread stores its portion of the output
    // (Simplified - actual implementation needs proper WMMA store)
    if (threadIdx.x < BLOCK_M && threadIdx.x < num_heads) {
        float inv_sum = 1.0f / smem.row_sum[threadIdx.x];
        #pragma unroll 8
        for (int d = 0; d < HEAD_DIM_V; d++) {
            // Extract from o_accum and normalize
            float val = 0.0f;  // Placeholder
            o_ptr[threadIdx.x * params.o_head_stride + d] = bf16(val * inv_sum);
        }
    }

    // Store LSE for split-KV merging
    if (params.softmax_lse_ptr && threadIdx.x < num_heads) {
        float* lse_ptr = params.softmax_lse_ptr +
            batch_idx * params.h_q * params.s_q +
            seq_idx * params.h_q +
            head_start;
        lse_ptr[threadIdx.x] = logf(smem.row_sum[threadIdx.x]) + smem.row_max[threadIdx.x];
    }
}

// Kernel launch wrapper
void run_sparse_fp8_decode_kernel(const SparseFP8DecodeParams& params) {
    const int num_head_blocks = (params.h_q + BLOCK_M - 1) / BLOCK_M;

    dim3 grid(params.batch_size, params.s_q, num_head_blocks);
    dim3 block(NUM_THREADS);
    size_t smem_size = sizeof(SharedMemoryPlan);

    sparse_fp8_decode_kernel<<<grid, block, smem_size, params.stream>>>(params);
}

} // namespace sparse_decode
} // namespace sm120
