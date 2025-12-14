#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace sm120 {

//==============================================================================
// Decode Parameters - MSVC-safe header (no CUTLASS includes)
//
// This header can be included from both MSVC host code (pybind.cpp) and
// NVCC device code (splitkv_mla.cu)
//==============================================================================

struct DecodingParams {
    // Batch and head configuration
    int b;              // batch size
    int h_k;            // number of KV heads
    int h_q;            // number of Q heads
    int q_head_per_hk;  // Q heads per KV head (for GQA)
    int q_seq_per_hk;   // Q sequence length per KV head
    int s_q;            // Q sequence length per sample
    int d;              // head dimension K (576)
    int d_v;            // head dimension V (512)
    int num_blocks;     // total KV cache blocks

    // Pointers
    void* q_ptr;
    void* k_ptr;
    void* o_ptr;
    void* softmax_lse_ptr;
    int* seqlens_k_ptr;

    // Strides
    int q_batch_stride;
    int q_row_stride;
    int q_head_stride;
    int k_batch_stride;
    int k_row_stride;
    int k_head_stride;
    int o_batch_stride;
    int o_row_stride;
    int o_head_stride;

    // Block table for paged attention
    int* block_table;
    int block_table_batch_stride;
    int page_block_size;

    // Tile scheduler
    int* tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int* num_splits_ptr;

    // Accumulators for split-K
    int total_num_splits;
    void* softmax_lseaccum_ptr;
    void* oaccum_ptr;

    // Softmax scale
    float scale_softmax;
    float scale_softmax_log2;

    // Mask mode
    bool is_causal;

    // Sparse attention (not supported on SM120)
    int* indices_ptr;
    int indices_batch_stride;
    int indices_row_stride;
    int topk;
};

//==============================================================================
// Forward declarations for kernel launch functions
//==============================================================================

// Launch SM120 decode kernel with bf16/fp16 data
void run_flash_splitkv_mla_kernel_bf16(DecodingParams& params, cudaStream_t stream);
void run_flash_splitkv_mla_kernel_fp16(DecodingParams& params, cudaStream_t stream);

}  // namespace sm120
