"""
SM120 Backward Kernel Profiler

Profiles the SM120 backward kernel to identify time spent in each phase,
tests different configurations to understand scaling behavior, and compares
FlashMLA SM120 vs PyTorch SDPA performance.

This module provides:
1. Precise timing using torch.cuda.Event
2. Forward vs backward pass isolation
3. Scaling analysis with sequence length (expected O(n^2) for attention)
4. Occupancy estimation and bottleneck identification
5. Detailed performance reports with optimization recommendations

Usage:
    python -m tests.profile_backward_kernel

References:
    - FlashAttention-2 paper (Dao et al., 2023) for tiling strategy
    - NVIDIA CUDA Best Practices Guide for occupancy estimation
"""

import math
import statistics
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# =============================================================================
# Configuration and Data Classes
# =============================================================================

@dataclass
class ProfilingConfig:
    """Configuration for a single profiling run."""
    batch_size: int
    seq_len: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype = torch.bfloat16
    causal: bool = False
    warmup_iterations: int = 10
    benchmark_iterations: int = 100

    @property
    def total_tokens(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def theoretical_flops_fwd(self) -> int:
        """Theoretical FLOPs for forward pass: 4 * N^2 * d (2 matmuls)"""
        # Q @ K^T: 2 * N * N * d
        # P @ V: 2 * N * N * d
        return 4 * self.seq_len * self.seq_len * self.head_dim * self.num_heads * self.batch_size

    @property
    def theoretical_flops_bwd(self) -> int:
        """Theoretical FLOPs for backward pass: ~8 * N^2 * d"""
        # dV = P^T @ dO: 2 * N * N * d
        # dP = dO @ V^T: 2 * N * N * d
        # dScores @ K: 2 * N * N * d
        # dScores^T @ Q: 2 * N * N * d
        return 8 * self.seq_len * self.seq_len * self.head_dim * self.num_heads * self.batch_size

    @property
    def memory_footprint_bytes(self) -> int:
        """Memory footprint estimate for Q, K, V, dQ, dK, dV, O, dO"""
        elem_size = 2 if self.dtype == torch.bfloat16 else 4
        # Q, K, V, O for forward
        # dQ, dK, dV, dO for backward
        num_tensors = 8
        return num_tensors * self.total_tokens * self.num_heads * self.head_dim * elem_size


@dataclass
class TimingResult:
    """Timing results for a single operation."""
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    iterations: int
    config: ProfilingConfig

    @property
    def throughput_tflops(self) -> float:
        """Compute throughput in TFLOPs."""
        if "forward" in self.name.lower():
            flops = self.config.theoretical_flops_fwd
        elif "backward" in self.name.lower():
            flops = self.config.theoretical_flops_bwd
        else:
            flops = self.config.theoretical_flops_fwd + self.config.theoretical_flops_bwd
        return flops / (self.mean_ms * 1e-3) / 1e12


@dataclass
class ComparisonResult:
    """Comparison between FlashMLA and PyTorch SDPA."""
    flashmla_result: TimingResult
    sdpa_result: TimingResult

    @property
    def speedup(self) -> float:
        return self.sdpa_result.mean_ms / self.flashmla_result.mean_ms


@dataclass
class ScalingAnalysis:
    """Analysis of scaling behavior with sequence length."""
    seq_lengths: List[int]
    times_ms: List[float]
    expected_complexity: str  # "O(n)", "O(n^2)", etc.
    actual_exponent: float
    r_squared: float


@dataclass
class ProfilingReport:
    """Complete profiling report."""
    device_info: Dict[str, any]
    timing_results: List[TimingResult]
    comparisons: List[ComparisonResult]
    scaling_analysis: Optional[ScalingAnalysis]
    bottleneck_analysis: Dict[str, any]
    recommendations: List[str]


# =============================================================================
# Utility Functions
# =============================================================================

def get_device_info() -> Dict[str, any]:
    """Get CUDA device information."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_version": props.major * 10 + props.minor,
        "multiprocessors": props.multi_processor_count,
        "max_threads_per_sm": props.max_threads_per_multi_processor,
        "shared_memory_per_block": props.shared_memory_per_block,
        "shared_memory_per_sm": getattr(props, 'shared_memory_per_multiprocessor', 102400),
        "total_memory_gb": props.total_memory / 1024**3,
        "clock_rate_mhz": props.clock_rate / 1000,
        "memory_clock_rate_mhz": props.memory_clock_rate / 1000,
        "memory_bus_width": props.memory_bus_width,
        "l2_cache_size": getattr(props, 'l2_cache_size', 0),
    }


def estimate_occupancy(config: ProfilingConfig, shared_mem_bytes: int = 43520) -> Dict[str, float]:
    """
    Estimate kernel occupancy based on configuration.

    For SM120 backward kernel:
    - Tile sizes: M=16 (Q-block), N=32 (KV-block), D=128
    - Warps: 4
    - Shared memory: ~43.5KB per block
    """
    props = torch.cuda.get_device_properties(0)

    # SM120 specifics
    max_threads_per_sm = props.max_threads_per_multi_processor
    max_blocks_per_sm = 32  # SM120 limit
    shared_mem_per_sm = getattr(props, 'shared_memory_per_multiprocessor', 102400)

    # Our kernel configuration
    warps_per_block = 4
    threads_per_block = warps_per_block * 32  # 128 threads

    # Occupancy limits
    # 1. Thread limit
    blocks_by_threads = max_threads_per_sm // threads_per_block

    # 2. Shared memory limit
    blocks_by_smem = shared_mem_per_sm // shared_mem_bytes if shared_mem_bytes > 0 else max_blocks_per_sm

    # 3. Block limit
    blocks_by_limit = max_blocks_per_sm

    # Effective blocks per SM
    blocks_per_sm = min(blocks_by_threads, blocks_by_smem, blocks_by_limit)
    active_warps = blocks_per_sm * warps_per_block
    max_warps_per_sm = max_threads_per_sm // 32

    return {
        "blocks_per_sm": blocks_per_sm,
        "active_warps_per_sm": active_warps,
        "occupancy_percent": 100.0 * active_warps / max_warps_per_sm,
        "limiting_factor": "threads" if blocks_per_sm == blocks_by_threads else
        "shared_memory" if blocks_per_sm == blocks_by_smem else "block_limit",
    }


def compute_time_per_block(total_time_ms: float, config: ProfilingConfig,
                           m_block_size: int = 16, n_block_size: int = 32) -> Dict[str, float]:
    """Compute estimated time per M-block and KV-block iteration."""
    num_m_blocks = math.ceil(config.seq_len / m_block_size)
    num_kv_blocks = math.ceil(config.seq_len / n_block_size)

    # Total blocks = batch * heads * m_blocks * kv_blocks (for backward)
    total_blocks = config.batch_size * config.num_heads * num_m_blocks * num_kv_blocks

    return {
        "num_m_blocks": num_m_blocks,
        "num_kv_blocks": num_kv_blocks,
        "total_block_pairs": total_blocks,
        "time_per_block_pair_us": (total_time_ms * 1000) / total_blocks,
        "time_per_m_block_us": (total_time_ms * 1000) / (config.batch_size * config.num_heads * num_m_blocks),
        "time_per_kv_iter_us": (total_time_ms * 1000) / (config.batch_size * config.num_heads * num_m_blocks * num_kv_blocks),
    }


def fit_power_law(x_vals: List[float], y_vals: List[float]) -> Tuple[float, float]:
    """
    Fit y = a * x^b using log-linear regression.
    Returns (exponent b, r_squared).
    """
    import math

    n = len(x_vals)
    if n < 2:
        return 0.0, 0.0

    log_x = [math.log(x) for x in x_vals]
    log_y = [math.log(y) for y in y_vals]

    mean_log_x = sum(log_x) / n
    mean_log_y = sum(log_y) / n

    numerator = sum((lx - mean_log_x) * (ly - mean_log_y) for lx, ly in zip(log_x, log_y))
    denominator = sum((lx - mean_log_x) ** 2 for lx in log_x)

    if denominator == 0:
        return 0.0, 0.0

    b = numerator / denominator

    # R-squared
    ss_tot = sum((ly - mean_log_y) ** 2 for ly in log_y)
    ss_res = sum((ly - (mean_log_y + b * (lx - mean_log_x))) ** 2 for lx, ly in zip(log_x, log_y))

    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return b, r_squared


# =============================================================================
# Timing Functions
# =============================================================================

class CUDATimer:
    """Precise CUDA timing using events."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self) -> float:
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)


def time_operation(fn, warmup: int = 10, iterations: int = 100) -> List[float]:
    """
    Time a CUDA operation with warmup and multiple iterations.
    Returns list of times in milliseconds.
    """
    timer = CUDATimer()
    times = []

    # Warmup
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    # Benchmark
    for _ in range(iterations):
        timer.start()
        fn()
        elapsed = timer.stop()
        times.append(elapsed)

    return times


def benchmark_flashmla_forward(config: ProfilingConfig) -> TimingResult:
    """Benchmark FlashMLA forward pass."""
    try:
        from flash_mla.flash_mla_interface import flash_attn_varlen_func, FLASH_MLA_LOADED_VARIANT
    except ImportError as e:
        raise RuntimeError(f"Could not import FlashMLA: {e}")

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    L = config.seq_len
    H = config.num_heads
    D = config.head_dim
    B = config.batch_size

    # Create inputs in varlen format: [total_tokens, heads, dim]
    total_tokens = B * L
    q = torch.randn(total_tokens, H, D, device=device, dtype=config.dtype)
    k = torch.randn(total_tokens, H, D, device=device, dtype=config.dtype)
    v = torch.randn(total_tokens, H, D, device=device, dtype=config.dtype)

    # cu_seqlens for B sequences of length L
    cu_seqlens = torch.arange(0, (B + 1) * L, L, device=device, dtype=torch.int32)

    def forward_fn():
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens, cu_seqlens,
            L, L,
            softmax_scale=None,
            causal=config.causal,
            is_varlen=True,
        )

    times = time_operation(forward_fn, config.warmup_iterations, config.benchmark_iterations)

    return TimingResult(
        name=f"FlashMLA Forward (variant={FLASH_MLA_LOADED_VARIANT})",
        mean_ms=statistics.mean(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_ms=min(times),
        max_ms=max(times),
        iterations=len(times),
        config=config,
    )


def benchmark_flashmla_backward(config: ProfilingConfig) -> TimingResult:
    """Benchmark FlashMLA backward pass (isolated from forward)."""
    try:
        from flash_mla.flash_mla_interface import _flash_attn_varlen_forward, _flash_attn_varlen_backward
    except ImportError as e:
        raise RuntimeError(f"Could not import FlashMLA: {e}")

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    L = config.seq_len
    H = config.num_heads
    D = config.head_dim
    B = config.batch_size

    total_tokens = B * L
    q = torch.randn(total_tokens, H, D, device=device, dtype=config.dtype)
    k = torch.randn(total_tokens, H, D, device=device, dtype=config.dtype)
    v = torch.randn(total_tokens, H, D, device=device, dtype=config.dtype)
    cu_seqlens = torch.arange(0, (B + 1) * L, L, device=device, dtype=torch.int32)

    # Run forward once to get outputs needed for backward
    out, lse = _flash_attn_varlen_forward(
        q, k, v,
        cu_seqlens, cu_seqlens,
        L, L,
        causal=config.causal,
        softmax_scale=None,
        is_varlen=True,
    )
    torch.cuda.synchronize()

    d_o = torch.randn_like(out)

    def backward_fn():
        return _flash_attn_varlen_backward(
            d_o, q, k, v, out, lse,
            cu_seqlens, cu_seqlens,
            L, L,
            causal=config.causal,
            softmax_scale=None,
            is_varlen=True,
        )

    times = time_operation(backward_fn, config.warmup_iterations, config.benchmark_iterations)

    return TimingResult(
        name="FlashMLA Backward",
        mean_ms=statistics.mean(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_ms=min(times),
        max_ms=max(times),
        iterations=len(times),
        config=config,
    )


def benchmark_flashmla_full(config: ProfilingConfig) -> TimingResult:
    """Benchmark FlashMLA full forward+backward pass."""
    try:
        from flash_mla.flash_mla_interface import flash_attn_varlen_func
    except ImportError as e:
        raise RuntimeError(f"Could not import FlashMLA: {e}")

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    L = config.seq_len
    H = config.num_heads
    D = config.head_dim
    B = config.batch_size

    total_tokens = B * L

    def full_pass():
        q = torch.randn(total_tokens, H, D, device=device, dtype=config.dtype, requires_grad=True)
        k = torch.randn(total_tokens, H, D, device=device, dtype=config.dtype, requires_grad=True)
        v = torch.randn(total_tokens, H, D, device=device, dtype=config.dtype, requires_grad=True)
        cu_seqlens = torch.arange(0, (B + 1) * L, L, device=device, dtype=torch.int32)

        out, lse = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens, cu_seqlens,
            L, L,
            softmax_scale=None,
            causal=config.causal,
            is_varlen=True,
        )

        d_o = torch.randn_like(out)
        out.backward(d_o)

        return q.grad, k.grad, v.grad

    times = time_operation(full_pass, config.warmup_iterations, config.benchmark_iterations)

    return TimingResult(
        name="FlashMLA Full (Fwd+Bwd)",
        mean_ms=statistics.mean(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_ms=min(times),
        max_ms=max(times),
        iterations=len(times),
        config=config,
    )


def benchmark_sdpa_forward(config: ProfilingConfig) -> TimingResult:
    """Benchmark PyTorch SDPA forward pass."""
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    B = config.batch_size
    H = config.num_heads
    L = config.seq_len
    D = config.head_dim

    # SDPA expects [batch, heads, seq, dim]
    q = torch.randn(B, H, L, D, device=device, dtype=config.dtype)
    k = torch.randn(B, H, L, D, device=device, dtype=config.dtype)
    v = torch.randn(B, H, L, D, device=device, dtype=config.dtype)

    def forward_fn():
        return F.scaled_dot_product_attention(q, k, v, is_causal=config.causal)

    times = time_operation(forward_fn, config.warmup_iterations, config.benchmark_iterations)

    return TimingResult(
        name="PyTorch SDPA Forward",
        mean_ms=statistics.mean(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_ms=min(times),
        max_ms=max(times),
        iterations=len(times),
        config=config,
    )


def benchmark_sdpa_backward(config: ProfilingConfig) -> TimingResult:
    """Benchmark PyTorch SDPA backward pass (isolated)."""
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    B = config.batch_size
    H = config.num_heads
    L = config.seq_len
    D = config.head_dim

    q = torch.randn(B, H, L, D, device=device, dtype=config.dtype, requires_grad=True)
    k = torch.randn(B, H, L, D, device=device, dtype=config.dtype, requires_grad=True)
    v = torch.randn(B, H, L, D, device=device, dtype=config.dtype, requires_grad=True)

    # Run forward once
    out = F.scaled_dot_product_attention(q, k, v, is_causal=config.causal)
    torch.cuda.synchronize()

    d_o = torch.randn_like(out)

    def backward_fn():
        # Clear gradients
        if q.grad is not None:
            q.grad.zero_()
        if k.grad is not None:
            k.grad.zero_()
        if v.grad is not None:
            v.grad.zero_()

        # Recompute forward (SDPA doesn't cache intermediates by default)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=config.causal)
        out.backward(d_o, retain_graph=True)

    times = time_operation(backward_fn, config.warmup_iterations, config.benchmark_iterations)

    return TimingResult(
        name="PyTorch SDPA Backward",
        mean_ms=statistics.mean(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_ms=min(times),
        max_ms=max(times),
        iterations=len(times),
        config=config,
    )


def benchmark_sdpa_full(config: ProfilingConfig) -> TimingResult:
    """Benchmark PyTorch SDPA full forward+backward pass."""
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    B = config.batch_size
    H = config.num_heads
    L = config.seq_len
    D = config.head_dim

    def full_pass():
        q = torch.randn(B, H, L, D, device=device, dtype=config.dtype, requires_grad=True)
        k = torch.randn(B, H, L, D, device=device, dtype=config.dtype, requires_grad=True)
        v = torch.randn(B, H, L, D, device=device, dtype=config.dtype, requires_grad=True)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=config.causal)

        d_o = torch.randn_like(out)
        out.backward(d_o)

        return q.grad, k.grad, v.grad

    times = time_operation(full_pass, config.warmup_iterations, config.benchmark_iterations)

    return TimingResult(
        name="PyTorch SDPA Full (Fwd+Bwd)",
        mean_ms=statistics.mean(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_ms=min(times),
        max_ms=max(times),
        iterations=len(times),
        config=config,
    )


# =============================================================================
# Profiling Routines
# =============================================================================

def profile_sequence_length_scaling(
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 128,
    seq_lengths: Optional[List[int]] = None,
    warmup: int = 5,
    iterations: int = 50,
) -> ScalingAnalysis:
    """
    Profile how kernel time scales with sequence length.
    Expected: O(n^2) for standard attention.
    """
    if seq_lengths is None:
        seq_lengths = [64, 128, 256, 512, 1024, 2048]

    times_ms = []

    print("Profiling sequence length scaling...")
    print("-" * 60)

    for seq_len in seq_lengths:
        config = ProfilingConfig(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            warmup_iterations=warmup,
            benchmark_iterations=iterations,
        )

        try:
            result = benchmark_flashmla_backward(config)
            times_ms.append(result.mean_ms)
            print(f"  seq_len={seq_len:5d}: {result.mean_ms:8.3f} ms (+/- {result.std_ms:.3f} ms)")
        except Exception as e:
            print(f"  seq_len={seq_len:5d}: FAILED - {e}")
            times_ms.append(float('nan'))

    # Filter out failed runs
    valid_data = [(s, t) for s, t in zip(seq_lengths, times_ms) if not math.isnan(t)]

    if len(valid_data) >= 2:
        valid_seqs, valid_times = zip(*valid_data)
        exponent, r_squared = fit_power_law(list(valid_seqs), list(valid_times))

        print(f"\nScaling analysis:")
        print(f"  Fitted exponent: {exponent:.2f} (expected: 2.0 for O(n^2))")
        print(f"  R-squared: {r_squared:.4f}")

        return ScalingAnalysis(
            seq_lengths=list(valid_seqs),
            times_ms=list(valid_times),
            expected_complexity="O(n^2)",
            actual_exponent=exponent,
            r_squared=r_squared,
        )
    else:
        return ScalingAnalysis(
            seq_lengths=seq_lengths,
            times_ms=times_ms,
            expected_complexity="O(n^2)",
            actual_exponent=0.0,
            r_squared=0.0,
        )


def profile_batch_size_scaling(
    seq_len: int = 512,
    num_heads: int = 8,
    head_dim: int = 128,
    batch_sizes: Optional[List[int]] = None,
    warmup: int = 5,
    iterations: int = 50,
) -> List[TimingResult]:
    """Profile how kernel time scales with batch size."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16]

    results = []

    print("Profiling batch size scaling...")
    print("-" * 60)

    for batch_size in batch_sizes:
        config = ProfilingConfig(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            warmup_iterations=warmup,
            benchmark_iterations=iterations,
        )

        try:
            result = benchmark_flashmla_backward(config)
            results.append(result)
            print(f"  batch_size={batch_size:3d}: {result.mean_ms:8.3f} ms (+/- {result.std_ms:.3f} ms)")
        except Exception as e:
            print(f"  batch_size={batch_size:3d}: FAILED - {e}")

    return results


def profile_head_count_scaling(
    batch_size: int = 1,
    seq_len: int = 512,
    head_dim: int = 128,
    head_counts: Optional[List[int]] = None,
    warmup: int = 5,
    iterations: int = 50,
) -> List[TimingResult]:
    """Profile how kernel time scales with number of heads."""
    if head_counts is None:
        head_counts = [1, 2, 4, 8, 16, 32]

    results = []

    print("Profiling head count scaling...")
    print("-" * 60)

    for num_heads in head_counts:
        config = ProfilingConfig(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            warmup_iterations=warmup,
            benchmark_iterations=iterations,
        )

        try:
            result = benchmark_flashmla_backward(config)
            results.append(result)
            print(f"  num_heads={num_heads:3d}: {result.mean_ms:8.3f} ms (+/- {result.std_ms:.3f} ms)")
        except Exception as e:
            print(f"  num_heads={num_heads:3d}: FAILED - {e}")

    return results


def compare_flashmla_vs_sdpa(configs: List[ProfilingConfig]) -> List[ComparisonResult]:
    """Compare FlashMLA and PyTorch SDPA for given configurations."""
    results = []

    print("Comparing FlashMLA SM120 vs PyTorch SDPA...")
    print("-" * 60)

    for config in configs:
        print(f"\nConfig: B={config.batch_size}, L={config.seq_len}, H={config.num_heads}, D={config.head_dim}")

        try:
            flashmla_fwd = benchmark_flashmla_forward(config)
            sdpa_fwd = benchmark_sdpa_forward(config)
            print(f"  Forward:  FlashMLA={flashmla_fwd.mean_ms:.3f}ms, SDPA={sdpa_fwd.mean_ms:.3f}ms, "
                  f"Speedup={sdpa_fwd.mean_ms/flashmla_fwd.mean_ms:.2f}x")

            flashmla_bwd = benchmark_flashmla_backward(config)
            sdpa_bwd = benchmark_sdpa_backward(config)
            print(f"  Backward: FlashMLA={flashmla_bwd.mean_ms:.3f}ms, SDPA={sdpa_bwd.mean_ms:.3f}ms, "
                  f"Speedup={sdpa_bwd.mean_ms/flashmla_bwd.mean_ms:.2f}x")

            results.append(ComparisonResult(flashmla_fwd, sdpa_fwd))
            results.append(ComparisonResult(flashmla_bwd, sdpa_bwd))

        except Exception as e:
            print(f"  FAILED: {e}")

    return results


# =============================================================================
# Report Generation
# =============================================================================

def analyze_bottlenecks(config: ProfilingConfig, bwd_result: TimingResult) -> Dict[str, any]:
    """Analyze potential performance bottlenecks."""
    device_info = get_device_info()
    occupancy = estimate_occupancy(config)
    block_times = compute_time_per_block(bwd_result.mean_ms, config)

    # Compute-bound vs memory-bound analysis
    # Memory bandwidth: device_info['memory_bus_width'] * device_info['memory_clock_rate_mhz'] * 2 / 8 * 1e6 bytes/s
    memory_bw_gbps = device_info.get('memory_bus_width', 384) * device_info.get('memory_clock_rate_mhz', 1000) * 2 / 8

    # Achieved memory bandwidth
    achieved_bw_gbps = config.memory_footprint_bytes / (bwd_result.mean_ms * 1e-3) / 1e9

    # Theoretical peak compute (estimate for SM120)
    # RTX 5090: ~2600 TFLOPS FP8, ~320 TFLOPS BF16
    peak_tflops_bf16 = 320.0  # Approximate for consumer Blackwell

    achieved_tflops = bwd_result.throughput_tflops

    return {
        "occupancy": occupancy,
        "block_times": block_times,
        "memory_bandwidth": {
            "theoretical_gbps": memory_bw_gbps,
            "achieved_gbps": achieved_bw_gbps,
            "efficiency_percent": 100 * achieved_bw_gbps / memory_bw_gbps if memory_bw_gbps > 0 else 0,
        },
        "compute": {
            "peak_tflops": peak_tflops_bf16,
            "achieved_tflops": achieved_tflops,
            "efficiency_percent": 100 * achieved_tflops / peak_tflops_bf16 if peak_tflops_bf16 > 0 else 0,
        },
        "limiting_resource": "memory" if achieved_bw_gbps / memory_bw_gbps > achieved_tflops / peak_tflops_bf16 else "compute",
    }


def generate_recommendations(bottleneck_analysis: Dict[str, any]) -> List[str]:
    """Generate optimization recommendations based on bottleneck analysis."""
    recommendations = []

    # Occupancy recommendations
    occupancy = bottleneck_analysis["occupancy"]
    if occupancy["occupancy_percent"] < 50:
        if occupancy["limiting_factor"] == "shared_memory":
            recommendations.append(
                f"Low occupancy ({occupancy['occupancy_percent']:.1f}%) limited by shared memory. "
                f"Consider reducing tile size or using shared memory more efficiently."
            )
        elif occupancy["limiting_factor"] == "threads":
            recommendations.append(
                f"Low occupancy ({occupancy['occupancy_percent']:.1f}%) limited by threads. "
                f"Consider increasing warps per block if register pressure allows."
            )

    # Memory bandwidth recommendations
    mem_bw = bottleneck_analysis["memory_bandwidth"]
    if mem_bw["efficiency_percent"] < 30:
        recommendations.append(
            f"Low memory bandwidth utilization ({mem_bw['efficiency_percent']:.1f}%). "
            f"Consider improving data locality or increasing arithmetic intensity."
        )

    # Compute recommendations
    compute = bottleneck_analysis["compute"]
    if compute["efficiency_percent"] < 20:
        recommendations.append(
            f"Low compute utilization ({compute['efficiency_percent']:.1f}%). "
            f"Consider using more tensor core operations or reducing synchronization overhead."
        )

    # Block-level recommendations
    block_times = bottleneck_analysis["block_times"]
    if block_times["time_per_block_pair_us"] > 100:
        recommendations.append(
            f"High time per block pair ({block_times['time_per_block_pair_us']:.1f} us). "
            f"Consider optimizing inner loop or reducing per-block overhead."
        )

    # Limiting resource
    if bottleneck_analysis["limiting_resource"] == "memory":
        recommendations.append(
            "Kernel appears memory-bound. Focus on data reuse and cache optimization."
        )
    else:
        recommendations.append(
            "Kernel appears compute-bound. Focus on maximizing tensor core utilization."
        )

    if not recommendations:
        recommendations.append("Kernel performance appears well-balanced. Profile at kernel level for further optimization.")

    return recommendations


def print_report(report: ProfilingReport):
    """Print a formatted profiling report."""
    print("")
    print("=" * 80)
    print("SM120 BACKWARD KERNEL PROFILING REPORT")
    print("=" * 80)

    # Device info
    print("\n[DEVICE INFORMATION]")
    print("-" * 40)
    for key, value in report.device_info.items():
        if "gb" in key.lower() or "mhz" in key.lower():
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Timing results
    print("\n[TIMING RESULTS]")
    print("-" * 40)
    print(f"{'Operation':<35} {'Mean (ms)':<12} {'Std (ms)':<12} {'TFLOPs':<10}")
    print("-" * 70)
    for result in report.timing_results:
        print(f"{result.name:<35} {result.mean_ms:<12.3f} {result.std_ms:<12.3f} {result.throughput_tflops:<10.2f}")

    # Comparisons
    if report.comparisons:
        print("\n[FLASHMLA vs SDPA COMPARISON]")
        print("-" * 40)
        for comp in report.comparisons:
            print(f"  {comp.flashmla_result.name} vs {comp.sdpa_result.name}: {comp.speedup:.2f}x speedup")

    # Scaling analysis
    if report.scaling_analysis:
        print("\n[SCALING ANALYSIS]")
        print("-" * 40)
        sa = report.scaling_analysis
        print(f"  Expected complexity: {sa.expected_complexity}")
        print(f"  Measured exponent: {sa.actual_exponent:.2f}")
        print(f"  R-squared: {sa.r_squared:.4f}")
        if abs(sa.actual_exponent - 2.0) < 0.2:
            print("  Status: [OK] Scaling matches expected O(n^2)")
        else:
            print(f"  Status: [WARN] Scaling deviates from expected O(n^2)")

    # Bottleneck analysis
    if report.bottleneck_analysis:
        print("\n[BOTTLENECK ANALYSIS]")
        print("-" * 40)
        ba = report.bottleneck_analysis

        occ = ba.get("occupancy", {})
        print(f"  Occupancy: {occ.get('occupancy_percent', 0):.1f}% ({occ.get('limiting_factor', 'unknown')} limited)")

        mem = ba.get("memory_bandwidth", {})
        print(f"  Memory BW: {mem.get('achieved_gbps', 0):.1f} GB/s ({mem.get('efficiency_percent', 0):.1f}% of theoretical)")

        comp = ba.get("compute", {})
        print(f"  Compute: {comp.get('achieved_tflops', 0):.2f} TFLOPs ({comp.get('efficiency_percent', 0):.1f}% of peak)")

        bt = ba.get("block_times", {})
        print(f"  Time per M-block: {bt.get('time_per_m_block_us', 0):.2f} us")
        print(f"  Time per KV-iter: {bt.get('time_per_kv_iter_us', 0):.2f} us")

        print(f"  Limiting resource: {ba.get('limiting_resource', 'unknown')}")

    # Recommendations
    if report.recommendations:
        print("\n[OPTIMIZATION RECOMMENDATIONS]")
        print("-" * 40)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)


# =============================================================================
# Main Entry Point
# =============================================================================

def run_full_profile():
    """Run complete profiling suite."""
    print("=" * 80)
    print("SM120 Backward Kernel Profiler")
    print("=" * 80)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available")
        return None

    device_info = get_device_info()
    print(f"\nDevice: {device_info.get('name', 'Unknown')}")
    print(f"SM Version: {device_info.get('sm_version', 'Unknown')}")

    # Check FlashMLA availability
    try:
        from flash_mla.flash_mla_interface import FLASH_MLA_LOADED_VARIANT
        print(f"FlashMLA variant: {FLASH_MLA_LOADED_VARIANT}")
    except ImportError as e:
        print(f"[FAIL] Could not import FlashMLA: {e}")
        return None

    all_results = []
    all_comparisons = []

    # 1. Profile sequence length scaling
    print("\n" + "=" * 60)
    print("PHASE 1: Sequence Length Scaling")
    print("=" * 60)
    scaling_analysis = profile_sequence_length_scaling(
        batch_size=1,
        num_heads=8,
        head_dim=128,
        seq_lengths=[64, 128, 256, 512, 1024, 2048],
        warmup=5,
        iterations=50,
    )

    # 2. Profile batch size scaling
    print("\n" + "=" * 60)
    print("PHASE 2: Batch Size Scaling")
    print("=" * 60)
    batch_results = profile_batch_size_scaling(
        seq_len=512,
        num_heads=8,
        head_dim=128,
        batch_sizes=[1, 2, 4, 8],
        warmup=5,
        iterations=50,
    )
    all_results.extend(batch_results)

    # 3. Profile head count scaling
    print("\n" + "=" * 60)
    print("PHASE 3: Head Count Scaling")
    print("=" * 60)
    head_results = profile_head_count_scaling(
        batch_size=1,
        seq_len=512,
        head_dim=128,
        head_counts=[1, 2, 4, 8, 16],
        warmup=5,
        iterations=50,
    )
    all_results.extend(head_results)

    # 4. Compare FlashMLA vs SDPA
    print("\n" + "=" * 60)
    print("PHASE 4: FlashMLA vs SDPA Comparison")
    print("=" * 60)
    comparison_configs = [
        ProfilingConfig(batch_size=1, seq_len=256, num_heads=8, head_dim=128, warmup_iterations=5, benchmark_iterations=50),
        ProfilingConfig(batch_size=1, seq_len=512, num_heads=8, head_dim=128, warmup_iterations=5, benchmark_iterations=50),
        ProfilingConfig(batch_size=1, seq_len=1024, num_heads=8, head_dim=128, warmup_iterations=5, benchmark_iterations=50),
        ProfilingConfig(batch_size=4, seq_len=512, num_heads=8, head_dim=128, warmup_iterations=5, benchmark_iterations=50),
    ]
    all_comparisons = compare_flashmla_vs_sdpa(comparison_configs)

    # 5. Detailed analysis on a representative config
    print("\n" + "=" * 60)
    print("PHASE 5: Detailed Bottleneck Analysis")
    print("=" * 60)
    detail_config = ProfilingConfig(
        batch_size=1,
        seq_len=512,
        num_heads=8,
        head_dim=128,
        warmup_iterations=10,
        benchmark_iterations=100,
    )

    print(f"\nRunning detailed analysis on: B={detail_config.batch_size}, L={detail_config.seq_len}, "
          f"H={detail_config.num_heads}, D={detail_config.head_dim}")

    try:
        bwd_result = benchmark_flashmla_backward(detail_config)
        all_results.append(bwd_result)

        fwd_result = benchmark_flashmla_forward(detail_config)
        all_results.append(fwd_result)

        full_result = benchmark_flashmla_full(detail_config)
        all_results.append(full_result)

        bottleneck_analysis = analyze_bottlenecks(detail_config, bwd_result)
        recommendations = generate_recommendations(bottleneck_analysis)

    except Exception as e:
        print(f"[FAIL] Detailed analysis failed: {e}")
        bottleneck_analysis = {}
        recommendations = []

    # Generate report
    report = ProfilingReport(
        device_info=device_info,
        timing_results=all_results,
        comparisons=all_comparisons,
        scaling_analysis=scaling_analysis,
        bottleneck_analysis=bottleneck_analysis,
        recommendations=recommendations,
    )

    # Print report
    print_report(report)

    return report


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SM120 Backward Kernel Profiler")
    parser.add_argument("--quick", action="store_true", help="Run quick profile with fewer iterations")
    parser.add_argument("--seq-only", action="store_true", help="Only profile sequence length scaling")
    parser.add_argument("--compare-only", action="store_true", help="Only run FlashMLA vs SDPA comparison")

    args = parser.parse_args()

    if args.quick:
        # Quick profile for testing
        print("Running quick profile...")
        config = ProfilingConfig(
            batch_size=1,
            seq_len=256,
            num_heads=4,
            head_dim=128,
            warmup_iterations=3,
            benchmark_iterations=10,
        )
        try:
            result = benchmark_flashmla_backward(config)
            print(f"Backward: {result.mean_ms:.3f} ms (+/- {result.std_ms:.3f} ms)")
            print(f"Throughput: {result.throughput_tflops:.2f} TFLOPs")
        except Exception as e:
            print(f"[FAIL] {e}")

    elif args.seq_only:
        profile_sequence_length_scaling()

    elif args.compare_only:
        configs = [
            ProfilingConfig(batch_size=1, seq_len=512, num_heads=8, head_dim=128,
                            warmup_iterations=5, benchmark_iterations=50),
        ]
        compare_flashmla_vs_sdpa(configs)

    else:
        run_full_profile()


if __name__ == "__main__":
    main()
