import importlib
import math
import os
from types import ModuleType
from typing import Optional, Tuple

import torch


def _load_flash_mla_extension() -> ModuleType:
    prefer_arch = os.getenv("FLASH_MLA_RUNTIME_ARCH")
    candidates = []
    if prefer_arch:
        prefer_arch = prefer_arch.lower()
        if prefer_arch == "sm120":
            candidates.append(("flash_mla.cuda_sm120", "sm120"))
        elif prefer_arch == "sm100":
            candidates.append(("flash_mla.cuda_sm100", "sm100"))
        elif prefer_arch == "legacy":
            candidates.append(("flash_mla.cuda", "legacy"))

    candidates.extend(
        [
            ("flash_mla.cuda_sm120", "sm120"),
            ("flash_mla.cuda_sm100", "sm100"),
            ("flash_mla.cuda", "legacy"),
        ]
    )

    seen = set()
    last_error: Optional[Exception] = None
    for module_name, variant in candidates:
        if module_name in seen:
            continue
        seen.add(module_name)
        try:
            module = importlib.import_module(module_name)
            setattr(module, "__flash_mla_variant__", variant)
            return module
        except ImportError as exc:
            last_error = exc
            continue

    raise ImportError(
        "Unable to import FlashMLA CUDA extension. Tried: "
        + ", ".join(seen)
    ) from last_error


flash_mla_cuda = _load_flash_mla_extension()
FLASH_MLA_LOADED_VARIANT = getattr(flash_mla_cuda, "__flash_mla_variant__", "legacy")

TILE_SCHEDULER_METADATA_SIZE = 8


def _legacy_flash_mla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: Optional[torch.Tensor] = None,
    k_new: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    if kv_cache is not None or k_new is not None or v_new is not None:
        raise NotImplementedError("KV cache updates are not supported in the SM120 fallback build.")

    batch_size, seqlen, num_heads_q, head_dim_qk = q.shape
    num_heads_k = k.shape[2]
    head_dim_v = v.shape[-1]

    if cu_seqlens is None:
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seqlen,
            seqlen,
            device=q.device,
            dtype=torch.int32,
        )
        is_varlen = False
    else:
        cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32, non_blocking=True)
        is_varlen = True

    if max_seqlen is None:
        # Default to the maximum span in cu_seqlens when not provided
        spans = torch.diff(cu_seqlens)
        max_seqlen = int(spans.max().item())

    if k.size(0) != q.size(0) or k.size(1) != q.size(1):
        raise NotImplementedError("GQA with differing sequence lengths is not supported in the fallback path.")

    q_varlen = q.reshape(-1, num_heads_q, head_dim_qk)
    k_varlen = k.reshape(-1, num_heads_k, head_dim_qk)
    v_varlen = v.reshape(-1, num_heads_k, head_dim_v)

    out_varlen, _ = flash_attn_varlen_func(
        q_varlen,
        k_varlen,
        v_varlen,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        softmax_scale=softmax_scale,
        causal=causal,
        is_varlen=is_varlen,
    )

    out = out_varlen.reshape(batch_size, seqlen, num_heads_q, head_dim_v)
    return out

def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
    num_heads_q: Optional[int] = None,
    is_fp8_kvcache: bool = False,
    topk: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_q_tokens_per_head_k: Equals to num_q_tokens_per_q_seq * num_heads_q // num_heads_k.
        num_heads_k: The number of k heads.
        num_heads_q: The number of q heads. This argument is optional when sparse attention is not enabled
        is_fp8_kvcache: Whether the k_cache and v_cache are in fp8 format.
        topk: If not None, sparse attention will be enabled, and only tokens in the `indices` array passed to `flash_mla_with_kvcache_sm90` will be attended to.

    Returns:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    def _fallback_metadata() -> Tuple[torch.Tensor, torch.Tensor]:
        device = cache_seqlens.device
        tile_scheduler_metadata = torch.zeros(
            (1, TILE_SCHEDULER_METADATA_SIZE), dtype=torch.int32, device=device
        )
        num_splits = torch.arange(
            0, cache_seqlens.size(0) + 1, dtype=torch.int32, device=device
        )
        return tile_scheduler_metadata, num_splits

    force_fallback = os.getenv("FLASH_MLA_FORCE_FALLBACK", "0").lower() in {"1", "true", "yes"}
    if force_fallback or FLASH_MLA_LOADED_VARIANT != "sm120":
        return _fallback_metadata()

    try:
        return flash_mla_cuda.get_mla_decoding_metadata(
            cache_seqlens,
            num_q_tokens_per_head_k,
            num_heads_k,
            num_heads_q,
            is_fp8_kvcache,
            topk,
        )
    except RuntimeError as exc:
        if "Unsupported GPU architecture" in str(exc):
            return _fallback_metadata()
        raise


def _fallback_flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    softmax_scale: Optional[float],
    causal: bool,
    indices: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = q.device
    dtype = q.dtype
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    b, s_q, h_q, d = q.shape
    block_size = k_cache.size(1)
    h_kv = k_cache.size(2)
    total_heads_k = h_kv
    out = torch.empty(b, s_q, h_q, head_dim_v, device=device, dtype=dtype)
    lse = torch.empty(b, h_q, s_q, device=device, dtype=torch.float32)

    k_cache_fp32 = k_cache.to(torch.float32)
    block_table_device = block_table.to(device)
    cache_seqlens_cpu = cache_seqlens.to("cpu")

    if indices is not None:
        indices = indices.to(device)

    def _get_topk_mask(local_indices: torch.Tensor, seq_len_k: int) -> torch.Tensor:
        mask = torch.zeros(s_q, seq_len_k, dtype=torch.bool, device=device)
        for qi in range(s_q):
            row = local_indices[qi]
            valid = row[row != -1]
            if valid.numel() > 0:
                mask[qi, valid] = True
        return mask

    for batch_idx in range(b):
        cur_len = int(cache_seqlens_cpu[batch_idx].item())
        if cur_len == 0:
            out[batch_idx].zero_()
            lse[batch_idx].fill_(float("inf"))
            continue

        num_blocks = math.ceil(cur_len / block_size)
        block_ids = block_table_device[batch_idx, :num_blocks]
        kv_tiles = k_cache_fp32[block_ids]  # [num_blocks, block_size, h_kv, d]
        kv_tokens = kv_tiles.view(-1, h_kv, k_cache_fp32.size(-1))[:cur_len]
        kv_tokens = kv_tokens.transpose(0, 1).contiguous()  # [h_kv, cur_len, d]

        query = q[batch_idx].transpose(0, 1).to(torch.float32)  # [h_q, s_q, d]
        value = kv_tokens[:, :, :head_dim_v]
        key = kv_tokens

        if total_heads_k != 1:
            key = key.repeat_interleave(h_q // total_heads_k, dim=0)
            value = value.repeat_interleave(h_q // total_heads_k, dim=0)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * softmax_scale

        if indices is not None:
            mask = _get_topk_mask(indices[batch_idx], key.size(-2))
            attn_scores = attn_scores.masked_fill(mask.logical_not(), float("-inf"))
        elif causal and query.size(1) > 1:
            rows = torch.arange(0, s_q, device=device).unsqueeze(1)
            cols = torch.arange(0, key.size(-2), device=device).unsqueeze(0)
            causal_mask = cols > rows
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        lse_batch = torch.logsumexp(attn_scores, dim=-1)
        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32)
        output = torch.matmul(attn_weights, value)

        lonely_mask = (lse_batch == float("-inf"))
        if lonely_mask.any():
            output = output.masked_fill(lonely_mask.unsqueeze(-1), 0.0)
            lse_batch = lse_batch.masked_fill(lonely_mask, float("inf"))

        out[batch_idx] = output.transpose(0, 1).to(dtype)
        lse[batch_idx] = lse_batch

    return out, lse


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head dimension of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, returned by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, returned by get_mla_metadata.
        softmax_scale: float. The scale of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.
        is_fp8_kvcache: bool. Whether the k_cache and v_cache are in fp8 format. For the format of FP8 KV cache, please refer to README.md
        indices: (batch_size, seq_len_q, topk), torch.int32. If not None, sparse attention will be enabled, and only tokens in the `indices` array will be attended to. Invalid indices should be set to -1 or numbers >= total_seq_len_kv. For details about how to set up `indices`, please refer to README.md.

    Returns:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if indices is not None:
        assert causal == False, "causal must be `false` if sparse attention is enabled."
    force_fallback = os.getenv("FLASH_MLA_FORCE_FALLBACK", "0").lower() in {"1", "true", "yes"}
    if force_fallback or FLASH_MLA_LOADED_VARIANT != "sm120":
        return _fallback_flash_mla_with_kvcache(
            q,
            k_cache,
            block_table,
            cache_seqlens,
            head_dim_v,
            softmax_scale,
            causal,
            indices,
        )

    out, softmax_lse = flash_mla_cuda.fwd_kvcache_mla(
        q,
        k_cache,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        is_fp8_kvcache,
        indices,
    )
    return out, softmax_lse


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512

    Returns:
        (output, max_logits, lse)
        About the definition of output, max_logits and lse, please refer to README.md
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, 2-based log-sum-exp
    """
    results = flash_mla_cuda.sparse_prefill_fwd(
        q, kv, indices, sm_scale, d_v
    )
    return results


def flash_mla_sparse_bwd(
    d_o: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill backward kernel (SM120 only)

    Args:
        d_o: [s_q, h_q, d_v], bfloat16 - gradient of output
        q: [s_q, h_q, d_qk], bfloat16 - query tensor from forward
        kv: [s_kv, h_kv, d_qk], bfloat16 - key-value tensor from forward
        o: [s_q, h_q, d_v], bfloat16 - output tensor from forward
        lse: [s_q, h_q], float32 - log-sum-exp from forward
        indices: [s_q, h_kv, topk], int32 - sparse attention indices
        sm_scale: float - softmax scale
        d_v: The dimension of value vectors. Must be 512

    Returns:
        (dq, dk, dv)
        - dq: [s_q, h_q, d_qk], bfloat16 - gradient of query
        - dk: [s_kv, h_kv, d_qk], bfloat16 - gradient of key
        - dv: [s_kv, h_kv, d_v], bfloat16 - gradient of value
    """
    results = flash_mla_cuda.sparse_prefill_bwd(
        d_o, q, kv, o, lse, indices, sm_scale, d_v
    )
    return results


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    qo_total_len, num_qo_heads, head_dim_qk = q.shape
    kv_total_len, num_kv_heads, head_dim_vo = v.shape

    mask_mode_code = 1 if causal else 0
    if softmax_scale is None:
        softmax_scale = head_dim_qk ** (-0.5)

    if out is None:
        out = torch.empty(qo_total_len, num_qo_heads, head_dim_vo, device=q.device, dtype=q.dtype)
    if lse is None:
        # Make lse contiguous on seqlen dim
        lse = torch.empty(num_qo_heads, qo_total_len, device=q.device, dtype=torch.float32).T

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    flash_mla_cuda.dense_prefill_fwd(
        workspace_buffer,
        q,
        k,
        v,
        cu_seqlens_qo,
        cu_seqlens_kv,
        out,
        lse,
        mask_mode_code,
        softmax_scale,
        max_seqlen_qo,
        max_seqlen_kv,
        is_varlen,
    )

    return out, lse


def _flash_attn_varlen_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # CRITICAL: All input tensors must be contiguous for correct memory layout
    # The kernel expects [total_tokens, num_heads, head_dim] with strides [H*D, D, 1]
    # Non-contiguous tensors (e.g., from permute()) have wrong strides and cause
    # the kernel to read incorrect data (mixing heads and tokens)
    do = do.contiguous()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = out.contiguous()
    lse = lse.contiguous()

    qo_total_len, num_qo_heads, head_dim_qk = q.shape
    kv_total_len, num_kv_heads, head_dim_vo = v.shape

    # TODO: fix bwd GQA
    if num_qo_heads != num_kv_heads:
        raise ValueError(f"SM100 bwd doesn't support GQA now. num_qo_heads: {num_qo_heads}, num_kv_heads: {num_kv_heads}.")

    mask_mode_code = 1 if causal else 0
    if softmax_scale is None:
        softmax_scale = head_dim_qk ** (-0.5)

    if dq is None:
        dq = torch.empty(qo_total_len, num_qo_heads, head_dim_qk, device=q.device, dtype=q.dtype)
    if dk is None:
        dk = torch.empty(kv_total_len, num_kv_heads, head_dim_qk, device=q.device, dtype=q.dtype)
    if dv is None:
        dv = torch.empty(kv_total_len, num_kv_heads, head_dim_vo, device=q.device, dtype=q.dtype)

    max_seqlen_qo_aligned = (max_seqlen_qo + 7) // 8 * 8
    bs = cu_seqlens_qo.shape[0] - 1
    workspace_bytes = 0
    workspace_bytes += 4 * bs * max_seqlen_qo_aligned * num_qo_heads * head_dim_qk  # dQ_acc
    workspace_bytes += 4 * max_seqlen_qo_aligned * bs * num_qo_heads * 2  # sum_OdO and scaled_lse
    if num_qo_heads != num_kv_heads:
        workspace_bytes += 2 * kv_total_len * num_qo_heads * (head_dim_qk + head_dim_vo)  # dKV_acc
    workspace_buffer = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
    flash_mla_cuda.dense_prefill_bwd(
        workspace_buffer,
        do,
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_qo,
        cu_seqlens_kv,
        dq,
        dk,
        dv,
        mask_mode_code,
        softmax_scale,
        max_seqlen_qo,
        max_seqlen_kv,
        is_varlen,
    )

    return dq, dk, dv


class FlashAttnVarlenFunc(torch.autograd.Function):
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_qo: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_qo: int,
        max_seqlen_kv: int,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        is_varlen: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, lse = _flash_attn_varlen_forward(
            q, k, v,
            cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
            causal=causal, softmax_scale=softmax_scale,
            is_varlen=is_varlen,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_qo, cu_seqlens_kv)
        ctx.max_seqlen_qo = max_seqlen_qo
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.is_varlen = is_varlen
        return out, lse

    def backward(
        ctx,
        do: torch.Tensor,
        dlse: torch.Tensor,
    ):
        del dlse  # LSE doesn't support backward currently
        q, k, v, out, lse, cu_seqlens_qo, cu_seqlens_kv = ctx.saved_tensors
        # Contiguity is now handled in _flash_attn_varlen_backward
        dq, dk, dv = _flash_attn_varlen_backward(
            do, q, k, v, out, lse,
            cu_seqlens_qo, cu_seqlens_kv, ctx.max_seqlen_qo, ctx.max_seqlen_kv,
            causal=ctx.causal, softmax_scale=ctx.softmax_scale,
            is_varlen=ctx.is_varlen,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        q, k, v,
        cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
        causal, softmax_scale, is_varlen,
    )


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    head_dim_qk: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        qkv[:, :, :head_dim_qk], qkv[:, :, head_dim_qk:head_dim_qk * 2], qkv[:, :, head_dim_qk * 2:],
        cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        causal, softmax_scale, is_varlen,
    )


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    head_dim_qk: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        q, kv[:, :, :head_dim_qk], kv[:, :, head_dim_qk:],
        cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
        causal, softmax_scale, is_varlen,
    )


if not hasattr(flash_mla_cuda, "fwd"):
    flash_mla_cuda.fwd = _legacy_flash_mla_fwd
