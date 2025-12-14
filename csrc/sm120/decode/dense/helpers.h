#pragma once

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>

using namespace cute;

namespace sm120 {

//==============================================================================
// SM120/SM80 MMA Helper Functions
//
// These helpers provide similar functionality to SM90/SM100 helpers but use
// SM80-compatible mma.sync operations instead of GMMA/UMMA.
//==============================================================================

// GEMM with shared memory operands (A from smem, B from smem)
// Uses SM80 mma.sync.aligned.m16n8k16
template<bool zero_init, typename TiledMma, typename TensorA, typename TensorB, typename TensorC>
__forceinline__ __device__ void gemm_smem_smem(
    TiledMma& tiled_mma,
    TensorA const& sA,
    TensorB const& sB,
    TensorC& rC,
    int thread_idx
) {
    auto thr_mma = tiled_mma.get_slice(thread_idx);
    auto tArA = thr_mma.partition_fragment_A(sA);  // Partition A for this thread
    auto tBrB = thr_mma.partition_fragment_B(sB);  // Partition B for this thread

    // Zero-init or accumulate
    if constexpr (zero_init) {
        clear(rC);
    }

    // Iterate over K dimension
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tArA); ++k) {
        cute::gemm(tiled_mma, tArA(_, _, k), tBrB(_, _, k), rC);
    }
}

// Convert MMA output fragment to linear row-major storage for softmax
// SM80_16x8x16 MMA: each thread holds 4 floats from 2x2 submatrix
template<typename TensorFrag, typename TensorLinear>
__forceinline__ __device__ void fragment_to_linear(
    TensorFrag const& frag,
    TensorLinear& linear,
    int thread_idx,
    int M,
    int N
) {
    // For SM80_16x8x16_F32BF16BF16F32_TN:
    // - Each warp (32 threads) computes 16x8 output
    // - Thread layout within warp: 4 rows x 8 threads
    // - Each thread holds 4 consecutive elements (2 per row, 2 rows)

    const int lane_id = thread_idx % 32;
    const int warp_id = thread_idx / 32;

    // Thread position within 16x8 output
    const int row_offset = (lane_id / 4) * 2;  // 0, 2, 4, 6, 8, 10, 12, 14
    const int col_offset = (lane_id % 4) * 2;  // 0, 2, 4, 6

    // Fragment stores 4 values: (row, col), (row, col+1), (row+1, col), (row+1, col+1)
    CUTE_UNROLL
    for (int i = 0; i < size(frag); ++i) {
        int local_row = (i / 2);
        int local_col = (i % 2);
        int global_row = row_offset + local_row;
        int global_col = col_offset + local_col;

        // Apply warp offset
        int warp_row = (warp_id / 2) * 32;
        int warp_col = (warp_id % 2) * 32;

        int final_row = warp_row + global_row;
        int final_col = warp_col + global_col;

        if (final_row < M && final_col < N) {
            linear(final_row, final_col) = frag(i);
        }
    }
}

// Online softmax row-max for register fragment
template<typename TensorFrag>
__forceinline__ __device__ void fragment_row_max(
    TensorFrag const& frag,
    float* row_max,
    int thread_idx,
    int M
) {
    // Each thread contributes to multiple rows' max values
    // Need warp-level reduction for each row

    const int lane_id = thread_idx % 32;
    const int warp_id = thread_idx / 32;

    const int row_offset = (lane_id / 4) * 2;
    const int warp_row = (warp_id / 2) * 32;

    // Each thread holds values for 2 consecutive rows
    float local_max[2] = {-INFINITY, -INFINITY};

    CUTE_UNROLL
    for (int i = 0; i < size(frag); ++i) {
        int local_row = (i / 2);  // 0 or 1
        local_max[local_row] = fmaxf(local_max[local_row], frag(i));
    }

    // Warp reduction for each row
    CUTE_UNROLL
    for (int r = 0; r < 2; ++r) {
        int global_row = warp_row + row_offset + r;
        if (global_row < M) {
            // Reduce across threads that share the same row
            float val = local_max[r];
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));

            // Thread 0 of each group writes the result
            if ((lane_id % 4) == 0) {
                atomicMax(reinterpret_cast<int*>(row_max + global_row), __float_as_int(val));
            }
        }
    }
}

// Scale factor computation for online softmax
__forceinline__ __device__ float compute_rescale(float old_max, float new_max) {
    if (old_max == -INFINITY) return 1.0f;
    return exp2f((old_max - new_max) * 1.4426950408889634f);  // ln(2) conversion
}

// Apply exp and compute row sum in register fragment
template<typename TensorFrag>
__forceinline__ __device__ void fragment_exp_sum(
    TensorFrag& frag,
    float const* row_max,
    float* row_sum,
    int thread_idx,
    int M
) {
    const int lane_id = thread_idx % 32;
    const int warp_id = thread_idx / 32;

    const int row_offset = (lane_id / 4) * 2;
    const int warp_row = (warp_id / 2) * 32;

    float local_sum[2] = {0.0f, 0.0f};

    CUTE_UNROLL
    for (int i = 0; i < size(frag); ++i) {
        int local_row = (i / 2);
        int global_row = warp_row + row_offset + local_row;
        if (global_row < M) {
            float max_val = row_max[global_row];
            float exp_val = exp2f((frag(i) - max_val) * 1.4426950408889634f);
            frag(i) = exp_val;
            local_sum[local_row] += exp_val;
        }
    }

    // Warp reduction for row sums
    CUTE_UNROLL
    for (int r = 0; r < 2; ++r) {
        int global_row = warp_row + row_offset + r;
        if (global_row < M) {
            float val = local_sum[r];
            val += __shfl_xor_sync(0xffffffff, val, 1);
            val += __shfl_xor_sync(0xffffffff, val, 2);

            if ((lane_id % 4) == 0) {
                atomicAdd(row_sum + global_row, val);
            }
        }
    }
}

// Rescale output fragment by per-row factor
template<typename TensorFrag>
__forceinline__ __device__ void fragment_rescale(
    TensorFrag& frag,
    float const* scale,
    int thread_idx,
    int M
) {
    const int lane_id = thread_idx % 32;
    const int warp_id = thread_idx / 32;

    const int row_offset = (lane_id / 4) * 2;
    const int warp_row = (warp_id / 2) * 32;

    CUTE_UNROLL
    for (int i = 0; i < size(frag); ++i) {
        int local_row = (i / 2);
        int global_row = warp_row + row_offset + local_row;
        if (global_row < M) {
            frag(i) *= scale[global_row];
        }
    }
}

// Normalize output fragment and store to global memory
template<typename InputT, typename TensorFrag>
__forceinline__ __device__ void fragment_normalize_store(
    TensorFrag const& frag,
    InputT* out_ptr,
    float const* row_sum,
    int out_stride,
    int thread_idx,
    int M,
    int N
) {
    const int lane_id = thread_idx % 32;
    const int warp_id = thread_idx / 32;

    const int row_offset = (lane_id / 4) * 2;
    const int col_offset = (lane_id % 4) * 2;
    const int warp_row = (warp_id / 2) * 32;
    const int warp_col = (warp_id % 2) * 32;

    CUTE_UNROLL
    for (int i = 0; i < size(frag); ++i) {
        int local_row = (i / 2);
        int local_col = (i % 2);
        int global_row = warp_row + row_offset + local_row;
        int global_col = warp_col + col_offset + local_col;

        if (global_row < M && global_col < N) {
            float inv_sum = 1.0f / (row_sum[global_row] + 1e-6f);
            out_ptr[global_row * out_stride + global_col] = InputT(frag(i) * inv_sum);
        }
    }
}

}  // namespace sm120
