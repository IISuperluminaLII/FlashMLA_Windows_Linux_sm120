#pragma once

#include "params.h"

namespace sm120 {

// Sparse prefill backward parameters
struct SparsePrefillBwdParams {
    int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
    float sm_scale, sm_scale_log2;

    // Input tensors (from forward)
    cutlass::bfloat16_t* __restrict__ q;      // [s_q, h_q, d_qk]
    cutlass::bfloat16_t* __restrict__ kv;     // [s_kv, h_kv, d_qk]
    int* __restrict__ indices;                 // [s_q, h_kv, topk]
    cutlass::bfloat16_t* __restrict__ o;      // [s_q, h_q, d_v]
    float* __restrict__ lse;                   // [s_q, h_q]

    // Gradient input
    cutlass::bfloat16_t* __restrict__ d_o;    // [s_q, h_q, d_v]

    // Strides
    int stride_q_s_q, stride_q_h_q;
    int stride_kv_s_kv, stride_kv_h_kv;
    int stride_indices_s_q, stride_indices_h_kv;
    int stride_o_s_q, stride_o_h_q;
    int stride_do_s_q, stride_do_h_q;

    // Gradient outputs
    // dq is bf16 (written once per query, no atomics needed)
    cutlass::bfloat16_t* __restrict__ dq;     // [s_q, h_q, d_qk]
    // dk and dv use float32 for atomic accumulation (multiple queries write to same KV)
    float* __restrict__ dk;                    // [s_kv, h_kv, d_qk] - float32 for atomics
    float* __restrict__ dv;                    // [s_kv, h_kv, d_v] - float32 for atomics

    int stride_dq_s_q, stride_dq_h_q;
    int stride_dk_s_kv, stride_dk_h_kv;
    int stride_dv_s_kv, stride_dv_h_kv;

    cudaStream_t stream;
};

void run_sparse_bwd_kernel(const SparsePrefillBwdParams& params);

}  // namespace sm120
