"""
Golden Accuracy Test for FlashMLA SM120 Decode Kernel

This test validates the numerical correctness of FlashMLA against a
PyTorch reference implementation of Multi-head Latent Attention (MLA).

MLA Reference:
- DeepSeek-V2/V3 Paper: https://arxiv.org/abs/2405.04434
- Machine Learning Mastery: https://machinelearningmastery.com/a-gentle-introduction-to-multi-head-latent-attention-mla/
- HuggingFace Implementation: https://huggingface.co/bird-of-paradise/deepseek-mla

MLA Key Dimensions (DeepSeek-V3):
- d_qk = 576 (Q/K head dimension = d_c + d_rope = 512 + 64)
- d_v = 512 (V head dimension, latent compressed)
- h_q = 128 (query heads)
- h_kv = 1 (key-value heads, MQA-style)
- GQA ratio = h_q / h_kv = 128

The attention computation:
    scores = Q @ K^T / sqrt(d_qk)    # [batch, h_q, seq_q, seq_k]
    probs = softmax(scores, dim=-1)
    output = probs @ V               # [batch, h_q, seq_q, d_v]

Test uses deterministic seeding for reproducibility.
"""

import math
import sys
import torch
import argparse
from typing import Tuple, Optional

# Ensure flash_mla is importable
sys.path.insert(0, str(__file__).replace("tests/test_golden_accuracy.py", ""))


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def set_deterministic_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reference_mla_attention_pytorch(
    q: torch.Tensor,           # [batch, seq_q, h_q, d_qk]
    k: torch.Tensor,           # [batch, seq_k, h_kv, d_qk]
    v: torch.Tensor,           # [batch, seq_k, h_kv, d_v]
    cache_seqlens: torch.Tensor,  # [batch]
    is_causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference PyTorch implementation of MLA attention.

    This follows the standard scaled dot-product attention with GQA expansion:
    1. Expand K/V heads to match Q heads (GQA: h_kv=1 -> h_q=128)
    2. Compute attention scores: Q @ K^T / sqrt(d)
    3. Apply softmax
    4. Compute output: probs @ V
    5. Return output and log-sum-exp (LSE) for numerical stability checks

    Based on DeepSeek-V3 MLA specification and FlashMLA reference_torch().
    """
    batch, seq_q, h_q, d_qk = q.shape
    _, seq_k, h_kv, d_v = v.shape

    # Convert to float32 for numerical stability in reference
    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()

    # Expand KV heads for GQA (h_kv=1 -> h_q=128)
    # K: [batch, seq_k, h_kv, d_qk] -> [batch, h_q, seq_k, d_qk]
    # V: [batch, seq_k, h_kv, d_v] -> [batch, h_q, seq_k, d_v]
    gqa_ratio = h_q // h_kv
    k_expanded = k_f32.transpose(1, 2).repeat_interleave(gqa_ratio, dim=1)  # [batch, h_q, seq_k, d_qk]
    v_expanded = v_f32.transpose(1, 2).repeat_interleave(gqa_ratio, dim=1)  # [batch, h_q, seq_k, d_v]

    # Q: [batch, seq_q, h_q, d_qk] -> [batch, h_q, seq_q, d_qk]
    q_transposed = q_f32.transpose(1, 2)

    # Compute attention scores: Q @ K^T
    # [batch, h_q, seq_q, d_qk] @ [batch, h_q, d_qk, seq_k] -> [batch, h_q, seq_q, seq_k]
    scores = torch.matmul(q_transposed, k_expanded.transpose(-2, -1))

    # Scale by sqrt(d_qk)
    scale = 1.0 / math.sqrt(d_qk)
    scores = scores * scale

    # Apply causal mask and sequence length mask
    cache_seqlens_cpu = cache_seqlens.cpu()
    for b in range(batch):
        valid_len = int(cache_seqlens_cpu[b].item())
        # Mask out positions beyond valid sequence length
        if valid_len < seq_k:
            scores[b, :, :, valid_len:] = float("-inf")
        # Apply causal mask if needed
        if is_causal and seq_q > 1:
            causal_mask = torch.triu(
                torch.ones(seq_q, seq_k, device=scores.device, dtype=torch.bool),
                diagonal=seq_k - seq_q + 1
            )
            scores[b, :, causal_mask] = float("-inf")

    # Compute log-sum-exp for numerical stability verification
    lse = torch.logsumexp(scores, dim=-1)  # [batch, h_q, seq_q]

    # Softmax
    probs = torch.softmax(scores, dim=-1)

    # Compute output: probs @ V
    # [batch, h_q, seq_q, seq_k] @ [batch, h_q, seq_k, d_v] -> [batch, h_q, seq_q, d_v]
    output = torch.matmul(probs, v_expanded)

    # Transpose back: [batch, h_q, seq_q, d_v] -> [batch, seq_q, h_q, d_v]
    output = output.transpose(1, 2)

    # Convert to bfloat16 to match FlashMLA output dtype
    output_bf16 = output.to(torch.bfloat16)

    return output_bf16, lse


def generate_deterministic_test_data(
    batch: int,
    seq_q: int,
    seq_k: int,
    h_q: int,
    h_kv: int,
    d_qk: int,
    d_v: int,
    block_size: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate deterministic test data for MLA attention.

    Returns:
        cache_seqlens: [batch] - sequence lengths
        q: [batch, seq_q, h_q, d_qk] - queries
        block_table: [batch, num_blocks] - block indices for paged KV cache
        blocked_k: [total_blocks, block_size, h_kv, d_qk] - blocked KV cache
        (K and V are stored together, V is blocked_k[..., :d_v])
    """
    set_deterministic_seed(seed)

    # Fixed sequence lengths for determinism
    cache_seqlens = torch.full((batch,), seq_k, dtype=torch.int32, device=device)

    # Generate Q with controlled range
    q = torch.randn(batch, seq_q, h_q, d_qk, device=device, dtype=torch.bfloat16)
    q = q.clamp(-1.0, 1.0)

    # Generate block table (sequential for simplicity)
    num_blocks_per_seq = cdiv(seq_k, block_size)
    total_blocks = batch * num_blocks_per_seq
    block_table = torch.arange(total_blocks, dtype=torch.int32, device=device).view(batch, num_blocks_per_seq)

    # Generate blocked KV cache
    # In MLA, K has dim d_qk (576), V has dim d_v (512)
    # FlashMLA stores them together in blocked_k with shape [blocks, block_size, h_kv, d_qk]
    # V is the first d_v dimensions
    blocked_k = torch.randn(total_blocks, block_size, h_kv, d_qk, device=device, dtype=torch.bfloat16)
    blocked_k = blocked_k.clamp(-0.1, 0.1)  # Smaller range for numerical stability

    return cache_seqlens, q, block_table, blocked_k


def extract_kv_from_blocked(
    blocked_k: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_size: int,
    d_v: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract K and V tensors from blocked KV cache for reference computation.

    Returns:
        k: [batch, max_seq_k, h_kv, d_qk]
        v: [batch, max_seq_k, h_kv, d_v]
    """
    batch = block_table.shape[0]
    h_kv = blocked_k.shape[2]
    d_qk = blocked_k.shape[3]
    max_seq_k = int(cache_seqlens.max().item())

    k = torch.zeros(batch, max_seq_k, h_kv, d_qk, device=blocked_k.device, dtype=blocked_k.dtype)
    v = torch.zeros(batch, max_seq_k, h_kv, d_v, device=blocked_k.device, dtype=blocked_k.dtype)

    for b in range(batch):
        seq_len = int(cache_seqlens[b].item())
        num_blocks = cdiv(seq_len, block_size)
        for blk_idx in range(num_blocks):
            block_id = block_table[b, blk_idx].item()
            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, seq_len)
            actual_len = end_pos - start_pos

            k[b, start_pos:end_pos] = blocked_k[block_id, :actual_len]
            v[b, start_pos:end_pos] = blocked_k[block_id, :actual_len, :, :d_v]

    return k, v


def check_allclose(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-3,
    rtol: float = 1e-2,
) -> bool:
    """
    Check if two tensors are close within tolerance.

    Returns True if pass, False if fail.
    """
    actual_f32 = actual.float()
    expected_f32 = expected.float()

    # Compute differences
    abs_diff = (actual_f32 - expected_f32).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    # Relative difference (avoid division by zero)
    rel_diff = abs_diff / (expected_f32.abs() + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    # Cosine similarity
    actual_flat = actual_f32.flatten()
    expected_flat = expected_f32.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        actual_flat.unsqueeze(0),
        expected_flat.unsqueeze(0)
    ).item()

    # Check if within tolerance
    is_close = torch.allclose(actual_f32, expected_f32, atol=atol, rtol=rtol)

    status = "PASSED" if is_close else "FAILED"
    print(f"  {name}: {status}")
    print(f"    Max abs diff: {max_abs_diff:.6e}")
    print(f"    Mean abs diff: {mean_abs_diff:.6e}")
    print(f"    Max rel diff: {max_rel_diff:.6e}")
    print(f"    Mean rel diff: {mean_rel_diff:.6e}")
    print(f"    Cosine similarity: {cos_sim:.8f}")

    if not is_close:
        # Print sample of mismatched values
        mismatch_mask = abs_diff > atol
        mismatch_indices = mismatch_mask.nonzero()[:5]
        print(f"    Sample mismatches (first 5):")
        for idx in mismatch_indices:
            idx_tuple = tuple(idx.tolist())
            print(f"      [{idx_tuple}]: actual={actual_f32[idx_tuple]:.6f}, expected={expected_f32[idx_tuple]:.6f}")

    return is_close


@torch.inference_mode()
def test_golden_accuracy_sm120(
    batch: int = 4,
    seq_q: int = 1,
    seq_k: int = 512,
    h_q: int = 128,
    h_kv: int = 1,
    d_qk: int = 576,
    d_v: int = 512,
    block_size: int = 64,
    seed: int = 42,
    is_causal: bool = False,
    atol: float = 8e-3,
    rtol: float = 5e-2,
) -> bool:
    """
    Golden accuracy test comparing FlashMLA SM120 against PyTorch reference.

    Args:
        batch: Batch size
        seq_q: Query sequence length (typically 1 for decode)
        seq_k: Key/Value sequence length
        h_q: Number of query heads
        h_kv: Number of KV heads (1 for MQA)
        d_qk: Q/K head dimension (576 for MLA)
        d_v: V head dimension (512 for MLA)
        block_size: KV cache block size
        seed: Random seed for reproducibility
        is_causal: Whether to apply causal masking
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        True if test passes, False otherwise
    """
    import flash_mla

    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)

    print("=" * 70)
    print("FlashMLA SM120 Golden Accuracy Test")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  batch={batch}, seq_q={seq_q}, seq_k={seq_k}")
    print(f"  h_q={h_q}, h_kv={h_kv}, d_qk={d_qk}, d_v={d_v}")
    print(f"  block_size={block_size}, seed={seed}, is_causal={is_causal}")
    print(f"  atol={atol}, rtol={rtol}")
    print("-" * 70)

    # Generate deterministic test data
    print("[1/4] Generating deterministic test data...")
    cache_seqlens, q, block_table, blocked_k = generate_deterministic_test_data(
        batch, seq_q, seq_k, h_q, h_kv, d_qk, d_v, block_size, seed, device
    )

    # Extract K, V for reference computation
    print("[2/4] Extracting K, V from blocked cache...")
    k, v = extract_kv_from_blocked(blocked_k, block_table, cache_seqlens, block_size, d_v)

    # Compute reference output using PyTorch
    print("[3/4] Computing PyTorch reference...")
    torch.cuda.synchronize()
    out_ref, lse_ref = reference_mla_attention_pytorch(q, k, v, cache_seqlens, is_causal)
    torch.cuda.synchronize()

    # Compute FlashMLA output
    print("[4/4] Computing FlashMLA output...")
    torch.cuda.synchronize()

    # Get metadata for tile scheduler
    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
        cache_seqlens,
        seq_q * h_q // h_kv,
        h_kv,
        h_q,
        False,  # is_fp8
        None    # topk
    )

    out_flash, lse_flash = flash_mla.flash_mla_with_kvcache(
        q.cuda(),
        blocked_k.cuda(),
        block_table.cuda(),
        cache_seqlens.cuda(),
        d_v,
        tile_scheduler_metadata,
        num_splits,
        causal=is_causal,
        is_fp8_kvcache=False,
        indices=None
    )
    torch.cuda.synchronize()

    # Compare results
    print("-" * 70)
    print("Comparing results:")

    output_pass = check_allclose("Output", out_flash, out_ref, atol=atol, rtol=rtol)

    # LSE comparison (more lenient tolerance)
    lse_pass = check_allclose("LSE", lse_flash, lse_ref, atol=1e-3, rtol=1e-1)

    print("-" * 70)
    all_pass = output_pass and lse_pass
    final_status = "PASSED" if all_pass else "FAILED"
    print(f"Final Result: {final_status}")
    print("=" * 70)

    assert all_pass, f"Golden accuracy test failed: output_pass={output_pass}, lse_pass={lse_pass}"


def run_test_suite():
    """Run a suite of golden accuracy tests with various configurations."""
    print("\n" + "=" * 70)
    print("FlashMLA SM120 Golden Accuracy Test Suite")
    print("Based on DeepSeek-V3 MLA specification")
    print("=" * 70 + "\n")

    test_configs = [
        # Basic decode test (seq_q=1)
        {"batch": 4, "seq_q": 1, "seq_k": 256, "name": "Basic decode (small)"},
        {"batch": 4, "seq_q": 1, "seq_k": 512, "name": "Basic decode (medium)"},
        {"batch": 8, "seq_q": 1, "seq_k": 1024, "name": "Basic decode (large)"},

        # Multi-token decode (MTP/speculative decoding)
        {"batch": 4, "seq_q": 2, "seq_k": 512, "name": "Multi-token decode (2 tokens)"},
        {"batch": 4, "seq_q": 4, "seq_k": 512, "name": "Multi-token decode (4 tokens)"},

        # Different batch sizes
        {"batch": 1, "seq_q": 1, "seq_k": 256, "name": "Single batch"},
        {"batch": 16, "seq_q": 1, "seq_k": 256, "name": "Large batch"},
        {"batch": 32, "seq_q": 1, "seq_k": 512, "name": "Very large batch"},

        # Different sequence lengths
        {"batch": 4, "seq_q": 1, "seq_k": 64, "name": "Short sequence"},
        {"batch": 4, "seq_q": 1, "seq_k": 2048, "name": "Long sequence"},

        # With causal masking
        {"batch": 4, "seq_q": 4, "seq_k": 512, "is_causal": True, "name": "Causal masking"},
    ]

    results = []
    for config in test_configs:
        name = config.pop("name")
        print(f"\n>>> Test: {name}")
        try:
            passed = test_golden_accuracy_sm120(**config)
            results.append((name, "PASSED" if passed else "FAILED"))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((name, f"ERROR: {e}"))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)
    passed_count = sum(1 for _, status in results if status == "PASSED")
    total_count = len(results)

    for name, status in results:
        status_str = "[OK]" if status == "PASSED" else "[FAIL]"
        print(f"  {status_str} {name}")

    print("-" * 70)
    print(f"Results: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("All tests PASSED!")
        return True
    else:
        print("Some tests FAILED!")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashMLA SM120 Golden Accuracy Test")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-q", type=int, default=1, help="Query sequence length")
    parser.add_argument("--seq-k", type=int, default=512, help="Key/Value sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--suite", action="store_true", help="Run full test suite")
    parser.add_argument("--atol", type=float, default=8e-3, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=5e-2, help="Relative tolerance")

    args = parser.parse_args()

    if args.suite:
        success = run_test_suite()
    else:
        success = test_golden_accuracy_sm120(
            batch=args.batch,
            seq_q=args.seq_q,
            seq_k=args.seq_k,
            seed=args.seed,
            atol=args.atol,
            rtol=args.rtol,
        )

    sys.exit(0 if success else 1)
