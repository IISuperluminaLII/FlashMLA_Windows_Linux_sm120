#pragma once
// SM120 Sparse-FP8 Decode - MSVC-safe POD params (no CUTLASS, no mma.h, no cuda_bf16.h)
// Mirrors csrc/sm120/decode/dense/params.h so pybind.cpp can include this from a host TU.
//
// Design: audit/design-sparse-decode.md section 4.1. Deliberately DELETED relative to the
// old struct (all dead or actively harmful):
//   - block_table_ptr / page_size: the authors' sparse decode NEVER touches a block table
//     (README.md "the kernel does not require the block_table parameter"); indices address
//     the paged cache directly via page = idx/64, offset = idx%64.
//   - seq_lens_ptr: sparse attendance is defined entirely by -1 indices.
//   - q_seq_stride/q_head_stride/o_seq_stride/o_head_stride: values off the RESHAPED
//     [b, q_seq_per_hk, h_kv, d] tensor that do not mean what their names say; replaced by
//     the honest q_row_stride/o_row_stride over the folded (s_q x head) axis.
//   - indices_head_stride: stride(2)==1 misnomer.
// Split-KV fields (CFG>=2) re-added 2026-08-27 per audit/design-sparse-splitkv.md:
// the mma splitkv tier consumes the authors' tile-scheduler metadata and writes
// partials the (already-unconditional) combine kernel merges. CFG<=1 kernels
// ignore every one of these fields.

#include <cuda_runtime.h>
#include <cstdint>

#include "../sched_meta.h"   // sm120::TileSchedulerMetaDataSize (shared w/ dense)

namespace sm120 {
namespace sparse_decode {

struct SparseFP8DecodeParams {
    // shape
    int b;                  // batch size
    int s_q;                // query positions per request
    int h_q;                // total query heads
    int h_kv;               // KV heads (MUST be 1; host-enforced)
    int q_head_per_hk;      // h_q / h_kv          -- rows per s_q position on the folded axis
    int q_seq_per_hk;       // s_q * q_head_per_hk -- extent of the folded axis
    int topk;

    // softmax
    float sm_scale;         // 1/sqrt(576)
    float sm_scale_log2;    // sm_scale * log2(e)

    // pointers (void*/const void* keeps this header CUTLASS-free)
    void*        q_ptr;             // bf16 [b, q_seq_per_hk, h_kv, 576]
    const void*  kv_ptr;            // fp8  [num_pages, 64, 1, 656 bytes/token]
    const int*   indices_ptr;       // i32  [b, s_q, topk]
    void*        o_ptr;             // bf16 [b, q_seq_per_hk, h_kv, 512]
    float*       softmax_lse_ptr;   // f32  [b, h_kv, q_seq_per_hk] contiguous

    // strides IN ELEMENTS (for the fp8 cache 1 element == 1 byte)
    int q_batch_stride;             // q.stride(0)
    int q_row_stride;               // q.stride(-3) over the folded axis (= 576)
    int o_batch_stride;             // out.stride(0)
    int o_row_stride;               // out.stride(-3) (= 512)
    int kv_page_stride;             // kcache.stride(0)
    int kv_token_stride;            // kcache.stride(1) == 656 (host-enforced)
    int indices_batch_stride;       // indices.stride(0)
    int indices_seq_stride;         // indices.stride(1)

    // split-KV (CFG>=2 only; the batch-parallel tiers never read these)
    const int* tile_scheduler_metadata_ptr;  // i32 [num_sm_parts, TileSchedulerMetaDataSize]
    int        num_sm_parts;                 // tile_scheduler_metadata.size(0)
    const int* num_splits_ptr;               // i32 [b+1] prefix sums (combine's skip predicate)
    float*     softmax_lseaccum_ptr;         // f32 [b+num_sm_parts, h_kv, q_seq_per_hk], 2-BASED partials
    float*     oaccum_ptr;                   // f32 [b+num_sm_parts, h_kv, q_seq_per_hk, 512], NORMALIZED

    cudaStream_t stream;
};

void run_sparse_fp8_decode_kernel(const SparseFP8DecodeParams& params);

}  // namespace sparse_decode
}  // namespace sm120
