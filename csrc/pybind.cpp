// Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

// Minimal torch includes to avoid compiled_autograd.h on MSVC
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>
#include <torch/library.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Critical: provides type_caster<at::Tensor> for pybind11 to convert Python tensors
#include <torch/csrc/utils/pybind.h>

// Import necessary symbols from torch namespace
using torch::Tensor;
using c10::IntArrayRef;
using c10::ScalarType;
namespace py = pybind11;

#include <cutlass/fast_math.h>

#include "params.h"
#include "smxx/get_mla_metadata.h"
#include "smxx/mla_combine.h"

#ifndef FLASH_MLA_DISABLE_SM90
#include "sm90/decode/dense/splitkv_mla.h"
#include "sm90/decode/sparse_fp8/splitkv_mla.h"
#include "sm90/prefill/sparse/fwd.h"
#endif

// Dense prefill interface depends on build target
#if defined(FLASH_MLA_BUILD_SM100)
#include "sm100/prefill/dense/interface.h"
#elif defined(FLASH_MLA_BUILD_SM120)
#include "sm120/prefill/dense/interface.h"
#else
#error "A FlashMLA build target must define FLASH_MLA_BUILD_SM100 or FLASH_MLA_BUILD_SM120"
#endif

#if defined(FLASH_MLA_BUILD_SM100)
#define FLASH_MLA_DENSE_FWD_RUN FMHACutlassSM100FwdRun
#define FLASH_MLA_DENSE_BWD_RUN FMHACutlassSM100BwdRun
static constexpr const char* kFlashMlaVariantName = "sm100";
#elif defined(FLASH_MLA_BUILD_SM120)
#define FLASH_MLA_DENSE_FWD_RUN FMHACutlassSM120FwdRun
#define FLASH_MLA_DENSE_BWD_RUN FMHACutlassSM120BwdRun
static constexpr const char* kFlashMlaVariantName = "sm120";
#endif

// Provide stub implementation when backward is disabled
#ifdef FLASH_MLA_SM120_DISABLE_BWD
void FLASH_MLA_DENSE_BWD_RUN(at::Tensor workspace_buffer, at::Tensor d_o, at::Tensor q,
                             at::Tensor k, at::Tensor v, at::Tensor o, at::Tensor lse,
                             at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                             at::Tensor dq, at::Tensor dk, at::Tensor dv,
                             int mask_mode_code, float softmax_scale, int max_seqlen_q,
                             int max_seqlen_kv, bool is_varlen) {
    TORCH_CHECK(
        false,
        "SM120 backward pass is disabled in this build. Please rebuild without "
        "FLASH_MLA_SM120_DISABLE_BWD flag to enable backward pass.");
}
#endif

#ifndef FLASH_MLA_DISABLE_SM100
#include "sm100/decode/sparse_fp8/splitkv_mla.h"
#include "sm100/prefill/sparse/fwd.h"
#endif

// SM120 decode kernel (CUTLASS with SM80 MMA atoms)
#if defined(FLASH_MLA_BUILD_SM120)
#include "sm120/decode/dense/splitkv_mla.h"
#endif

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == c10::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// SM120 decode fallback - ATen-based implementation for workstation GPUs
// that lack TMA/TMEM support required by native CUTLASS kernels
#if defined(FLASH_MLA_BUILD_SM120)
namespace sm120_fallback {

// Gather KV cache from paged blocks into contiguous tensors
inline std::tuple<at::Tensor, at::Tensor> gather_kv_from_paged_cache(
    const at::Tensor& kcache,
    const at::Tensor& block_table,
    const at::Tensor& seqlens_k,
    int batch_size,
    int head_size_k,
    int head_size_v,
    int page_block_size,
    int num_heads_k) {
  // kcache shape: [num_blocks, page_block_size, num_heads_k, head_size_k]
  // block_table: [batch_size, max_num_blocks_per_seq]
  // Returns: K [batch, max_seqlen, num_heads_k, head_size_k]
  //          V [batch, max_seqlen, num_heads_k, head_size_v]

  auto seqlens_cpu = seqlens_k.to(at::kCPU);
  int* seqlens_ptr = seqlens_cpu.data_ptr<int>();
  int max_seqlen = 0;
  for (int b = 0; b < batch_size; b++) {
    max_seqlen = std::max(max_seqlen, seqlens_ptr[b]);
  }

  auto block_table_cpu = block_table.to(at::kCPU);
  int* block_table_ptr = block_table_cpu.data_ptr<int>();
  int max_blocks_per_seq = block_table.size(1);

  auto opts = kcache.options();
  at::Tensor k_out = at::zeros({batch_size, max_seqlen, num_heads_k, head_size_k}, opts);
  at::Tensor v_out = at::zeros({batch_size, max_seqlen, num_heads_k, head_size_v}, opts);

  // Gather K and V from paged cache
  for (int b = 0; b < batch_size; b++) {
    int seqlen = seqlens_ptr[b];
    int num_blocks = (seqlen + page_block_size - 1) / page_block_size;

    for (int blk_idx = 0; blk_idx < num_blocks; blk_idx++) {
      int block_id = block_table_ptr[b * max_blocks_per_seq + blk_idx];
      int start_pos = blk_idx * page_block_size;
      int end_pos = std::min(start_pos + page_block_size, seqlen);
      int tokens_in_block = end_pos - start_pos;

      // kcache[block_id, :tokens_in_block, :, :head_size_k] -> k_out[b, start_pos:end_pos, :, :]
      at::Tensor k_block = kcache.index({block_id}).slice(0, 0, tokens_in_block);
      k_out.index_put_({b, at::indexing::Slice(start_pos, end_pos)},
                       k_block.slice(-1, 0, head_size_k));

      // For V, use last head_size_v dimensions (MLA layout)
      v_out.index_put_({b, at::indexing::Slice(start_pos, end_pos)},
                       k_block.slice(-1, head_size_k - head_size_v, head_size_k));
    }
  }

  return {k_out, v_out};
}

// ATen-based scaled dot-product attention fallback
inline std::tuple<at::Tensor, at::Tensor> run_decode_fallback(
    at::Tensor& q,                    // [batch, q_seq_per_hk, num_heads, head_size_k]
    const at::Tensor& kcache,         // paged cache
    const at::Tensor& seqlens_k,
    const at::Tensor& block_table,
    float softmax_scale,
    bool is_causal,
    int head_size_v,
    int page_block_size,
    int num_heads_k) {

  int batch_size = q.size(0);
  int q_seq = q.size(1);
  int num_heads = q.size(2);
  int head_size_k = q.size(3);

  // Gather K and V from paged cache
  auto [k_gathered, v_gathered] = gather_kv_from_paged_cache(
      kcache, block_table, seqlens_k, batch_size,
      head_size_k, head_size_v, page_block_size, num_heads_k);

  // k_gathered: [batch, max_seqlen, num_heads_k, head_size_k]
  // v_gathered: [batch, max_seqlen, num_heads_k, head_size_v]

  auto seqlens_cpu = seqlens_k.to(at::kCPU);
  int* seqlens_ptr = seqlens_cpu.data_ptr<int>();
  int max_seqlen = k_gathered.size(1);

  auto opts = q.options();
  at::Tensor out = at::zeros({batch_size, q_seq, num_heads, head_size_v}, opts);
  at::Tensor lse = at::zeros({batch_size, num_heads, q_seq}, opts.dtype(at::kFloat));

  // Convert to float for numerical stability
  at::Tensor q_float = q.to(at::kFloat);
  at::Tensor k_float = k_gathered.to(at::kFloat);
  at::Tensor v_float = v_gathered.to(at::kFloat);

  int num_heads_per_kv = num_heads / num_heads_k;

  for (int b = 0; b < batch_size; b++) {
    int kv_len = seqlens_ptr[b];
    if (kv_len == 0) continue;

    // q: [q_seq, num_heads, head_size_k]
    at::Tensor q_b = q_float.index({b});

    // k, v: [kv_len, num_heads_k, head_size]
    at::Tensor k_b = k_float.index({b}).slice(0, 0, kv_len);
    at::Tensor v_b = v_float.index({b}).slice(0, 0, kv_len);

    for (int h = 0; h < num_heads; h++) {
      int kv_head = h / num_heads_per_kv;

      // q_h: [q_seq, head_size_k]
      at::Tensor q_h = q_b.select(1, h);

      // k_h: [kv_len, head_size_k], v_h: [kv_len, head_size_v]
      at::Tensor k_h = k_b.select(1, kv_head);
      at::Tensor v_h = v_b.select(1, kv_head);

      // scores: [q_seq, kv_len]
      at::Tensor scores = at::matmul(q_h, k_h.transpose(0, 1)) * softmax_scale;

      // Apply causal mask if needed
      if (is_causal && q_seq > 1) {
        at::Tensor row_idx = at::arange(0, q_seq, scores.options().dtype(at::kLong)).unsqueeze(1);
        at::Tensor col_idx = at::arange(0, kv_len, scores.options().dtype(at::kLong)).unsqueeze(0);
        // For decode, typically q_seq==1, but handle general case
        at::Tensor mask = col_idx > (row_idx + kv_len - q_seq);
        scores.masked_fill_(mask, -std::numeric_limits<float>::infinity());
      }

      // Compute softmax
      at::Tensor lse_h = at::logsumexp(scores, -1, true);
      at::Tensor probs = (scores - lse_h).exp();

      // Output: [q_seq, head_size_v]
      at::Tensor out_h = at::matmul(probs, v_h);

      out.index_put_({b, at::indexing::Slice(), h}, out_h.to(opts.dtype()));
      lse.index_put_({b, h}, lse_h.squeeze(-1));
    }
  }

  return {out, lse};
}

}  // namespace sm120_fallback
#endif  // FLASH_MLA_BUILD_SM120

struct Arch {
    int major;
    int minor;

    bool is_sm90() const {
        return major == 9 && minor == 0;
    }

    bool is_sm100() const {
        return major == 10;
    }

    bool is_sm120() const {
        return major == 12 && minor == 0;
    }

    bool is_blackwell() const {
        return is_sm100() || is_sm120();
    }

    void assert_is_supported() const {
#if defined(FLASH_MLA_BUILD_SM100)
        TORCH_CHECK(
            is_sm100(),
            "flash_mla_sm100 was compiled for Blackwell server GPUs (SM100). Detected sm_",
            major,
            minor,
            ". Please install the SM120 build for workstation-class devices.");
#elif defined(FLASH_MLA_BUILD_SM120)
        TORCH_CHECK(
            is_sm120(),
            "flash_mla_sm120 was compiled for Blackwell workstation GPUs (SM120). Detected sm_",
            major,
            minor,
            ". Please install the SM100 build for server-class devices.");
#else
        TORCH_CHECK(
            is_sm90() || is_sm100() || is_sm120(),
            "Only SM90 (Hopper) and SM100/SM120 (Blackwell) are supported. Your GPU architecture sm_",
            major,
            minor,
            " is not supported.");
#endif
    }
};

// DecodingAttnImplMeta - A struct to hold metadata for Decoding Attention Implementation (i.e. SM90 Dense BF16, SM90 Sparse FP8, etc.)
struct DecodingAttnImplMeta {
    int num_sm_parts;
    int fixed_overhead_num_blocks;
    int k_block_size;
};

DecodingAttnImplMeta get_attn_impl_meta(
    Arch arch,
    int sm_count,
    int num_q_tokens_per_head_k,
    int h_k,
    std::optional<int> h_q_,
    bool is_fp8_kvcache,
    bool is_sparse_attn
) {
#ifndef FLASH_MLA_DISABLE_SM90
    if (arch.is_sm90()) {
        if (is_sparse_attn) {
            if (is_fp8_kvcache) {
                TORCH_CHECK(h_q_.has_value());
                int h_q = h_q_.value();
                TORCH_CHECK(h_q % h_k == 0);
                int s_q = num_q_tokens_per_head_k * h_k / h_q;
                // FP8 + Sparse MLA
                return {
                    std::max((sm_count/2) / h_k / (cutlass::ceil_div(h_q/h_k, 2*64) * s_q), 1),
                    5,
                    64
                };
            } else {
                // Sparse BF16 MLA
                TORCH_CHECK(false, "Sparse BF16 MLA is not supported on SM90");
            }
        } else {
            if (is_fp8_kvcache) {
                // Dense FP8 MLA
                TORCH_CHECK(false, "Dense FP8 MLA is not supported on SM90");
            } else {
                // Dense BF16 MLA
                return {
                    std::max(sm_count / h_k / cutlass::ceil_div(num_q_tokens_per_head_k, 64), 1),
                    5,
                    64
                };
            }
        }
    } else
#endif
#ifndef FLASH_MLA_DISABLE_SM100
    if (arch.is_blackwell()) {
        if (is_sparse_attn) {
            if (is_fp8_kvcache) {
                TORCH_CHECK(h_q_.has_value());
                int h_q = h_q_.value();
                TORCH_CHECK(h_q % h_k == 0);
                int s_q = num_q_tokens_per_head_k * h_k / h_q;
                // FP8 + Sparse MLA
                return {
                    std::max(sm_count / h_k / (cutlass::ceil_div(h_q/h_k, 64) * s_q), 1),
                    5,
                    64
                };
            } else {
                // Sparse BF16 MLA
                TORCH_CHECK(false, "Sparse BF16 MLA is not supported on SM100/SM120");
            }
        } else {
            if (is_fp8_kvcache) {
                // FP8 MLA
                TORCH_CHECK(false, "FP8 Dence MLA is not supported on SM100/SM120");
            } else {
                // Normal BF16 MLA
                TORCH_CHECK(false, "BF16 Dence MLA is not supported on SM100/SM120");
            }
        }
    } else
#endif
#if defined(FLASH_MLA_BUILD_SM120)
    // SM120 CUTLASS path - single partition (no split-K) for now
    // TODO: Implement split-K support in SM120 kernel to enable proper parallelization.
    // The kernel currently writes directly to output buffers, not accumulator buffers.
    // Split-K would require: (1) writing to oaccum_ptr/softmax_lseaccum_ptr, (2) proper
    // partition index handling for batch/block assignment.
    if (arch.is_sm120()) {
        // SM120 CUTLASS kernel supports dense BF16/FP16 (no FP8, no sparse)
        TORCH_CHECK(!is_fp8_kvcache, "SM120 CUTLASS does not support FP8 KV cache.");
        TORCH_CHECK(!is_sparse_attn, "SM120 CUTLASS does not support sparse attention.");
        // Single partition mode - kernel handles all batches sequentially
        return {
            1,   // num_sm_parts (single partition - no split-K support yet)
            5,   // fixed_overhead_num_blocks
            64   // k_block_size
        };
    } else
#endif
    {
        TORCH_CHECK(false, "Unsupported GPU architecture");
    }
}


std::vector<at::Tensor>
get_mla_decoding_metadata(
    at::Tensor &seqlens_k,
    const int num_q_tokens_per_head_k,
    const int h_k,
    const std::optional<int> h_q,
    const bool is_fp8_kvcache,
    const std::optional<int> topk
) {
    bool is_sparse_attn = topk.has_value();
    CHECK_DEVICE(seqlens_k);
    TORCH_CHECK(seqlens_k.is_contiguous());
    TORCH_CHECK(seqlens_k.dtype() == c10::kInt);
    if (is_sparse_attn)
        TORCH_CHECK(h_q.has_value(), "num_heads_q must be provided when topk is provided");

    int batch_size = seqlens_k.size(0);
    int *seqlens_k_ptr = seqlens_k.data_ptr<int>();
    auto options = seqlens_k.options();

    auto dprops = at::cuda::getCurrentDeviceProperties();
    int sm_count = dprops->multiProcessorCount;
    Arch arch = {dprops->major, dprops->minor};
    arch.assert_is_supported();
    DecodingAttnImplMeta attn_impl_meta = get_attn_impl_meta(arch, sm_count, num_q_tokens_per_head_k, h_k, h_q, is_fp8_kvcache, is_sparse_attn);

    auto tile_scheduler_metadata = at::empty({attn_impl_meta.num_sm_parts, TileSchedulerMetaDataSize}, options);
    auto num_splits = at::empty({batch_size + 1}, options);
    int *tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    int *num_splits_ptr = num_splits.data_ptr<int>();

    at::cuda::CUDAGuard device_guard{(char)seqlens_k.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    GetDecodingMetadataParams params = {};
    params.seqlens_k_ptr = seqlens_k_ptr;
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
    params.num_splits_ptr = num_splits_ptr;
    params.batch_size = batch_size;
    params.block_size_n = attn_impl_meta.k_block_size;
    params.fixed_overhead_num_blocks = attn_impl_meta.fixed_overhead_num_blocks;
    params.num_sm_parts = attn_impl_meta.num_sm_parts;
    params.topk = is_sparse_attn ? topk.value() : -1;
    run_get_mla_metadata_kernel(params, stream);

    return {tile_scheduler_metadata, num_splits};
}

std::vector<at::Tensor>
fwd_kvcache_mla(
    at::Tensor &q,                               // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,                    // num_blocks x page_block_size x num_heads_k x head_size (when is_fp8 is False) or num_blocks x num_heads_k x (page_block_size*656) (when is_fp8 is True)
    const int head_size_v,
    const at::Tensor &seqlens_k,                 // batch_size
    const at::Tensor &block_table,               // batch_size x max_num_blocks_per_seq
    const float softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,   // num_sm_parts x TileSchedulerMetaDataSize
    const at::Tensor &num_splits,                // batch_size + 1
    const bool &is_fp8,
    const std::optional<at::Tensor> &indices     // None, or batch_size x seqlen_q x topk
) {
    bool is_sparse_attn = indices.has_value();
    int topk = is_sparse_attn ? indices->size(-1) : -1;

    // Check the architecture
    auto dprops = at::cuda::getCurrentDeviceProperties();
    Arch arch = {dprops->major, dprops->minor};
    arch.assert_is_supported();

    // Check data types
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == c10::kBFloat16 || q_dtype == c10::kHalf);
    
    if (!is_fp8) {
        TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
    } else {
        TORCH_CHECK(kcache.dtype() == c10::kFloat8_e4m3fn || kcache.dtype() == c10::kChar || kcache.dtype() == c10::kByte, "key must have dtype fp8_e4m3fn or int8 or uint8");
    }
    TORCH_CHECK(seqlens_k.dtype() == c10::kInt, "seqlens_k must have dtype int32");
    TORCH_CHECK(block_table.dtype() == c10::kInt, "block_table must have dtype torch.int32");
    TORCH_CHECK(tile_scheduler_metadata.dtype() == c10::kInt, "tile_scheduler_metadata must have dtype int32");
    TORCH_CHECK(num_splits.dtype() == c10::kInt, "num_splits must have dtype int32");
    TORCH_CHECK(!is_sparse_attn || indices->dtype() == c10::kInt, "indices must have dtype int32");

    // Check device
    CHECK_DEVICE(q);
    CHECK_DEVICE(kcache);
    CHECK_DEVICE(seqlens_k);
    CHECK_DEVICE(block_table);
    CHECK_DEVICE(tile_scheduler_metadata);
    CHECK_DEVICE(num_splits);
    if (is_sparse_attn) CHECK_DEVICE(indices.value());

    // Check layout
    TORCH_CHECK(q.stride(-1) == 1, "q must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "kcache must have contiguous last dimension");
    CHECK_CONTIGUOUS(seqlens_k);
    TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    CHECK_CONTIGUOUS(tile_scheduler_metadata);
    CHECK_CONTIGUOUS(num_splits);
    TORCH_CHECK(!is_sparse_attn || indices->stride(-1) == 1, "indices must have contiguous last dimension");

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int seqlen_q_ori = sizes[1];
    const int num_heads_q = sizes[2];
    const int head_size_k = sizes[3];
    TORCH_CHECK(head_size_k == 576, "Only head_size_k == 576 is supported");
    TORCH_CHECK(head_size_v == 512, "Only head_size_v == 576 is supported");

    const int max_num_blocks_per_seq = block_table.size(1);
    const int num_blocks = kcache.size(0);
    const int page_block_size = kcache.size(1);
    const int num_heads_k = kcache.size(2);
    TORCH_CHECK(page_block_size == 64, "Currently page_block_size must be 64");
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(num_heads_q % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (seqlen_q_ori == 1) { is_causal = false; }

    const int num_q_heads_per_hk = num_heads_q / num_heads_k;
    const int q_seq_per_hk = seqlen_q_ori * num_q_heads_per_hk;
    const int num_heads = num_heads_k;
    q = q.view({batch_size, seqlen_q_ori, num_heads_k, num_q_heads_per_hk, head_size_k}).transpose(2, 3)
            .reshape({batch_size, q_seq_per_hk, num_heads, head_size_k});

    CHECK_SHAPE(q, batch_size, q_seq_per_hk, num_heads, head_size_k);
    if (!is_fp8) {
        CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_k);
    } else {
        int bytes_per_token = 512 + 64*2 + (512/128)*4;
        CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, bytes_per_token);
        TORCH_CHECK(num_heads_k == 1, "Currently the number of k heads must be 1 when is_fp8_kvcache is True");
        TORCH_CHECK(kcache.stride(1) == bytes_per_token, "The whole block must be contiguous when is_fp8_cache is True");
    }
    CHECK_SHAPE(seqlens_k, batch_size);
    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    TORCH_CHECK(tile_scheduler_metadata.size(1) == TileSchedulerMetaDataSize);
    CHECK_SHAPE(num_splits, batch_size+1);
    if (is_sparse_attn) CHECK_SHAPE(indices.value(), batch_size, seqlen_q_ori, topk);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    at::Tensor out = at::empty({batch_size, q_seq_per_hk, num_heads, head_size_v}, opts);
    at::Tensor softmax_lse = at::empty({batch_size, num_heads, q_seq_per_hk}, opts.dtype(at::kFloat));
    CHECK_CONTIGUOUS(softmax_lse);

    DecodingParams params = {};
    // Set the sizes.
    params.b = batch_size;
    params.s_q = seqlen_q_ori;
    params.q_seq_per_hk = q_seq_per_hk;
    params.seqlens_k_ptr = seqlens_k.data_ptr<int>();
    params.h_q = num_heads_q;
    params.h_k = num_heads_k;
    params.num_blocks = num_blocks;
    params.q_head_per_hk = num_q_heads_per_hk;
    params.is_causal = is_causal;
    params.d = head_size_k;
    params.d_v = head_size_v;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = float(softmax_scale * M_LOG2E);
    params.topk = topk;
    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = kcache.data_ptr();
    params.o_ptr = out.data_ptr();
    params.indices_ptr = is_sparse_attn ? indices->data_ptr<int>() : nullptr;
    params.softmax_lse_ptr = softmax_lse.data_ptr();
    // All stride are in elements, not bytes.
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = kcache.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = kcache.stride(1);
    params.o_row_stride = out.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = kcache.stride(2);
    params.o_head_stride = out.stride(-2);
    params.indices_batch_stride = is_sparse_attn ? indices->stride(0) : 0;
    params.indices_row_stride = is_sparse_attn ? indices->stride(1) : 0;

    params.block_table = block_table.data_ptr<int>();
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size = page_block_size;
    
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    params.num_sm_parts = tile_scheduler_metadata.size(0);
    params.num_splits_ptr = num_splits.data_ptr<int>();

    const int total_num_splits = batch_size + params.num_sm_parts;
    at::Tensor softmax_lse_accum = at::empty({total_num_splits, num_heads, q_seq_per_hk}, opts.dtype(at::kFloat));
    at::Tensor out_accum = at::empty({total_num_splits, num_heads, q_seq_per_hk, head_size_v}, opts.dtype(at::kFloat));
    CHECK_CONTIGUOUS(softmax_lse_accum);
    CHECK_CONTIGUOUS(out_accum);
    params.total_num_splits = total_num_splits;
    params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
    params.oaccum_ptr = out_accum.data_ptr();

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(head_size_k == 576);

    if (q_dtype == c10::kHalf) {
#ifdef FLASH_MLA_DISABLE_FP16
        TORCH_CHECK(false, "FlashMLA is compiled with -DFLASH_MLA_DISABLE_FP16. Please remove this flag from your environment and re-compile FlashMLA.");
#endif
    }

#ifdef FLASH_MLA_FORCE_FALLBACK
    TORCH_CHECK(false, "FlashMLA sparse decode kernels are disabled in the fallback-only Windows build.");
#else
#ifndef FLASH_MLA_DISABLE_SM90
    if (arch.is_sm90()) {
        if (is_sparse_attn) {
            if (is_fp8) {
                TORCH_CHECK(q_dtype == c10::kBFloat16, "Sparse FP8 MLA only supports BFloat16 on SM90");
                sm90::run_flash_splitkv_mla_fp8_sparse_kernel(params, stream);
            } else {
                TORCH_CHECK(false, "Only FP8 kvcahe is supported for sparse MLA on SM90");
            }
        } else {
            if (is_fp8) {
                TORCH_CHECK(false, "Dense FP8 MLA is not supported on SM90");
            } else {
                if (q_dtype == c10::kBFloat16) {
                    sm90::run_flash_splitkv_mla_kernel<cutlass::bfloat16_t>(params, stream);
                } else if (q_dtype == c10::kHalf) {
#ifndef FLASH_MLA_DISABLE_FP16
                    sm90::run_flash_splitkv_mla_kernel<cutlass::half_t>(params, stream);
#endif
                } else {
                    TORCH_CHECK(false, "Unsupported dtype for dense MLA on SM90");
                }
            }
        }
    } else
#endif
#ifndef FLASH_MLA_DISABLE_SM100
    if (arch.is_blackwell()) {
        TORCH_CHECK(is_fp8 && is_sparse_attn, "Only FP8 + Sparse attention is supported on SM100/SM120");
        sm100::run_flash_splitkv_mla_fp8_sparse_kernel(params, stream);
    } else
#endif
#if defined(FLASH_MLA_BUILD_SM120)
    // SM120 CUTLASS path - uses SM80 MMA atoms for decode
    if (arch.is_sm120()) {
        // SM120 CUTLASS kernel supports BF16/FP16 dense attention
        TORCH_CHECK(!is_fp8, "SM120 CUTLASS does not support FP8 KV cache. Use BF16/FP16.");
        TORCH_CHECK(!is_sparse_attn, "SM120 CUTLASS does not support sparse attention.");
        TORCH_CHECK(q_dtype == c10::kBFloat16 || q_dtype == c10::kHalf,
                    "SM120 CUTLASS requires BF16 or FP16 query tensor.");

        // Populate SM120 decode params
        sm120::DecodingParams sm120_params;
        sm120_params.b = batch_size;
        sm120_params.h_k = num_heads_k;
        sm120_params.h_q = num_heads_q;
        sm120_params.q_head_per_hk = num_q_heads_per_hk;
        sm120_params.q_seq_per_hk = q_seq_per_hk;
        sm120_params.s_q = seqlen_q_ori;
        sm120_params.d = head_size_k;
        sm120_params.d_v = head_size_v;
        sm120_params.num_blocks = kcache.size(0);

        sm120_params.q_ptr = q.data_ptr();
        sm120_params.k_ptr = kcache.data_ptr();
        sm120_params.o_ptr = out.data_ptr();
        sm120_params.softmax_lse_ptr = softmax_lse.data_ptr();
        sm120_params.seqlens_k_ptr = seqlens_k.data_ptr<int>();

        // Note: Q has been reshaped from [batch, s_q, h_q, d] to [batch, q_seq_per_hk, h_kv, d]
        // by the view/transpose/reshape on lines 504-505 above. Use strides of reshaped tensor.
        sm120_params.q_batch_stride = q.stride(0);
        sm120_params.q_row_stride = q.stride(-3);   // stride for q_seq_per_hk dim
        sm120_params.q_head_stride = q.stride(-2);  // stride for h_kv dim
        sm120_params.k_batch_stride = kcache.stride(0);
        sm120_params.k_row_stride = kcache.stride(1);
        sm120_params.k_head_stride = kcache.stride(2);
        // Output has shape [batch, q_seq_per_hk, h_kv, d_v], same layout as reshaped Q
        sm120_params.o_batch_stride = out.stride(0);
        sm120_params.o_row_stride = out.stride(-3);
        sm120_params.o_head_stride = out.stride(-2);

        sm120_params.block_table = block_table.data_ptr<int>();
        sm120_params.block_table_batch_stride = block_table.stride(0);
        sm120_params.page_block_size = page_block_size;

        sm120_params.tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
        sm120_params.num_sm_parts = tile_scheduler_metadata.size(0);
        sm120_params.num_splits_ptr = num_splits.data_ptr<int>();

        sm120_params.total_num_splits = total_num_splits;
        sm120_params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
        sm120_params.oaccum_ptr = out_accum.data_ptr();

        sm120_params.scale_softmax = softmax_scale;
        sm120_params.scale_softmax_log2 = softmax_scale * float(M_LOG2E);
        sm120_params.is_causal = is_causal;

        sm120_params.indices_ptr = nullptr;
        sm120_params.indices_batch_stride = 0;
        sm120_params.indices_row_stride = 0;
        sm120_params.topk = 0;

        // Run SM120 CUTLASS decode kernel (non-templated wrappers for MSVC compatibility)
        if (q_dtype == c10::kBFloat16) {
            sm120::run_flash_splitkv_mla_kernel_bf16(sm120_params, stream);
        } else if (q_dtype == c10::kHalf) {
#ifndef FLASH_MLA_DISABLE_FP16
            sm120::run_flash_splitkv_mla_kernel_fp16(sm120_params, stream);
#endif
        }
        // Continue to combine kernel below
    } else
#endif
    {
        TORCH_CHECK(false, "Unsupported GPU architecture");
    }
#endif

    if (q_dtype == c10::kBFloat16) {
        run_flash_mla_combine_kernel<cutlass::bfloat16_t>(params, stream);
    } else if (q_dtype == c10::kHalf) {
#ifndef FLASH_MLA_DISABLE_FP16
        run_flash_mla_combine_kernel<cutlass::half_t>(params, stream);
#endif
    } else {
        TORCH_CHECK(false, "Unsupported tensor dtype for query");
    }

    out = out.view({batch_size, seqlen_q_ori, num_q_heads_per_hk, num_heads_k, head_size_v}).transpose(2, 3)
            .reshape({batch_size, seqlen_q_ori, num_heads_q, head_size_v});
    softmax_lse = softmax_lse.view({batch_size, num_heads_k, seqlen_q_ori, num_q_heads_per_hk}).transpose(2, 3)
            .reshape({batch_size, num_heads_q, seqlen_q_ori});

    return {out, softmax_lse};
}


inline int int64_stride_to_int(int64_t orig_stride) {
    if (orig_stride > std::numeric_limits<int>::max()) {
        TORCH_CHECK(false, "[Sparse TopK Attention] Stride exceeds int32 limit: ", orig_stride);
    }
    return static_cast<int>(orig_stride);
}

std::vector<at::Tensor> sparse_prefill_fwd(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    float sm_scale,
    int d_v
) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm90 = dprops->major == 9;
    bool is_sm100 = dprops->major == 10;
    TORCH_CHECK(is_sm90 || is_sm100, "Sparse Attention Forward Kernel (sparse_prefill_fwd) is only supported on SM90 or SM100 architectures");

    CHECK_DEVICE(q);
    CHECK_DEVICE(kv);
    CHECK_DEVICE(indices);

    TORCH_CHECK(q.dtype() == c10::kBFloat16);
    TORCH_CHECK(kv.dtype() == c10::kBFloat16);
    TORCH_CHECK(indices.dtype() == c10::kInt);

    int s_q = q.size(0);
    int s_kv = kv.size(0);
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int d_qk = q.size(2);
    int topk = indices.size(2);

    CHECK_SHAPE(q, s_q, h_q, d_qk);
    CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    CHECK_SHAPE(indices, s_q, h_kv, topk);

    TORCH_CHECK(q.stride(-1) == 1);
    TORCH_CHECK(kv.stride(-1) == 1);
    TORCH_CHECK(indices.stride(-1) == 1);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();
    at::Tensor out = at::empty({s_q, h_q, d_v}, opts);
    CHECK_CONTIGUOUS(out);
    
    at::Tensor buf_attn_score, max_logits, lse, p_sum;
    max_logits = at::empty({s_q, h_q}, opts.dtype(at::kFloat));
    lse = at::empty({s_q, h_q}, opts.dtype(at::kFloat));
    CHECK_CONTIGUOUS(max_logits);
    CHECK_CONTIGUOUS(lse);

    SparsePrefillParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * 1.44269504f,

        (cutlass::bfloat16_t*)q.data_ptr(),
        (cutlass::bfloat16_t*)kv.data_ptr(),
        (int*)indices.data_ptr(),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        (cutlass::bfloat16_t*)out.data_ptr(),
        (float*)max_logits.data_ptr(),
        (float*)lse.data_ptr(),

        at::cuda::getCurrentCUDAStream().stream()
    };

#ifdef FLASH_MLA_FORCE_FALLBACK
    TORCH_CHECK(false, "FlashMLA dense prefill kernels are disabled in the fallback-only Windows build.");
#else
#ifndef FLASH_MLA_DISABLE_SM90
    if (is_sm90) {
        sm90::run_fwd_kernel(params);
    } else
#endif
#ifndef FLASH_MLA_DISABLE_SM100
    if (is_sm100) {
        sm100::run_fwd_kernel(params);
    } else
#endif
    {
        TORCH_CHECK(false, "Unknown architecture or architecture not supported in this build");
    }
#endif

    return {out, max_logits, lse};
}

// Wrapper functions for dense kernels (support SM100a + SM120) with explicit Python bindings
void dense_prefill_fwd_wrapper(
    at::Tensor workspace_buffer,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor cumulative_seqlen_q,
    at::Tensor cumulative_seqlen_kv,
    at::Tensor o,
    at::Tensor lse,
    int mask_mode_code,
    float softmax_scale,
    int max_seqlen_q,
    int max_seqlen_kv,
    bool is_varlen
) {
    FLASH_MLA_DENSE_FWD_RUN(
        workspace_buffer, q, k, v,
        cumulative_seqlen_q, cumulative_seqlen_kv,
        o, lse,
        mask_mode_code, softmax_scale,
        max_seqlen_q, max_seqlen_kv, is_varlen
    );
}

void dense_prefill_bwd_wrapper(
    at::Tensor workspace_buffer,
    at::Tensor d_o,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor o,
    at::Tensor lse,
    at::Tensor cumulative_seqlen_q,
    at::Tensor cumulative_seqlen_kv,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    int mask_mode_code,
    float softmax_scale,
    int max_seqlen_q,
    int max_seqlen_kv,
    bool is_varlen
) {
    FLASH_MLA_DENSE_BWD_RUN(
        workspace_buffer, d_o, q, k, v, o, lse,
        cumulative_seqlen_q, cumulative_seqlen_kv,
        dq, dk, dv,
        mask_mode_code, softmax_scale,
        max_seqlen_q, max_seqlen_kv, is_varlen
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("get_mla_decoding_metadata", &get_mla_decoding_metadata);
    m.def("fwd_kvcache_mla", &fwd_kvcache_mla);
    // Dense prefill kernels support BOTH SM100a and SM120
    m.def("dense_prefill_fwd", &dense_prefill_fwd_wrapper,
          py::arg("workspace_buffer"),
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("cumulative_seqlen_q"),
          py::arg("cumulative_seqlen_kv"),
          py::arg("o"),
          py::arg("lse"),
          py::arg("mask_mode_code"),
          py::arg("softmax_scale"),
          py::arg("max_seqlen_q"),
          py::arg("max_seqlen_kv"),
          py::arg("is_varlen")
    );
    m.def("dense_prefill_bwd", &dense_prefill_bwd_wrapper,
          py::arg("workspace_buffer"),
          py::arg("d_o"),
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("o"),
          py::arg("lse"),
          py::arg("cumulative_seqlen_q"),
          py::arg("cumulative_seqlen_kv"),
          py::arg("dq"),
          py::arg("dk"),
          py::arg("dv"),
          py::arg("mask_mode_code"),
          py::arg("softmax_scale"),
          py::arg("max_seqlen_q"),
          py::arg("max_seqlen_kv"),
          py::arg("is_varlen")
    );
    m.def("sparse_prefill_fwd", &sparse_prefill_fwd);
    m.attr("__flash_mla_variant__") = kFlashMlaVariantName;
}
