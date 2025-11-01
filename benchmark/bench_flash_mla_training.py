import argparse
import time

import torch
import torch.nn.functional as F

from flash_mla.flash_mla_interface import flash_attn_varlen_func


@torch.inference_mode(False)
def benchmark_step(fn, warmup=10, iters=40):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def make_inputs(batch, seqlen, heads, head_dim, dtype, device):
    q = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def flash_iteration_factory(q0, k0, v0, causal):
    batch, seqlen, heads, head_dim = q0.shape
    total_tokens = batch * seqlen
    cu_seqlens = torch.arange(0, (batch + 1) * seqlen, seqlen, device=q0.device, dtype=torch.int32)

    def run():
        q = q0.reshape(total_tokens, heads, head_dim).clone().detach().requires_grad_(True)
        k = k0.reshape(total_tokens, heads, head_dim).clone().detach().requires_grad_(True)
        v = v0.reshape(total_tokens, heads, head_dim).clone().detach().requires_grad_(True)

        out, _ = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            seqlen,
            seqlen,
            causal=causal,
            softmax_scale=None,
            is_varlen=True,
        )
        grad = torch.randn_like(out)
        out.backward(grad)
        # clear for next iteration
        q.grad = None
        k.grad = None
        v.grad = None

    return run


def torch_iteration_factory(q0, k0, v0, causal):
    q_t = q0.clone().detach()
    k_t = k0.clone().detach()
    v_t = v0.clone().detach()

    def run():
        q = q_t.clone().detach().requires_grad_(True)
        k = k_t.clone().detach().requires_grad_(True)
        v = v_t.clone().detach().requires_grad_(True)

        q_heads = q.permute(0, 2, 1, 3)
        k_heads = k.permute(0, 2, 1, 3)
        v_heads = v.permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, dropout_p=0.0, is_causal=causal)
        grad = torch.randn_like(attn)
        attn.backward(grad)
        q.grad = None
        k.grad = None
        v.grad = None

    return run


def compute_tflops(batch, seqlen, heads, head_dim, latency_ms):
    flops = 2.0 * batch * heads * seqlen * seqlen * head_dim
    return flops / (latency_ms / 1000.0) / 1e12


def run_case(batch, seqlen, heads, head_dim, dtype, causal, device):
    q, k, v = make_inputs(batch, seqlen, heads, head_dim, dtype, device)
    flash_iter = flash_iteration_factory(q, k, v, causal)
    torch_iter = torch_iteration_factory(q, k, v, causal)

    flash_ms = benchmark_step(flash_iter)
    torch_ms = benchmark_step(torch_iter)

    flash_tflops = compute_tflops(batch, seqlen, heads, head_dim, flash_ms)
    torch_tflops = compute_tflops(batch, seqlen, heads, head_dim, torch_ms)
    tokens = batch * seqlen
    flash_tokens = tokens / (flash_ms / 1000.0)
    torch_tokens = tokens / (torch_ms / 1000.0)

    return {
        "batch": batch,
        "seqlen": seqlen,
        "heads": heads,
        "head_dim": head_dim,
        "flash_ms": flash_ms,
        "torch_ms": torch_ms,
        "flash_tflops": flash_tflops,
        "torch_tflops": torch_tflops,
        "flash_tok_s": flash_tokens,
        "torch_tok_s": torch_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seqlen", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=128)
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    seqlens = [512, 1024, 2048, args.max_seqlen]

    results = []
    for seqlen in seqlens:
        print(f"Benchmarking batch={args.batch}, seqlen={seqlen}, heads={args.heads}, head_dim={args.head_dim}, dtype={args.dtype}, causal={args.causal}")
        res = run_case(args.batch, seqlen, args.heads, args.head_dim, dtype, args.causal, device)
        results.append(res)
        print(f"  FlashMLA: {res['flash_ms']:.3f} ms, {res['flash_tflops']:.2f} TFLOPS, {res['flash_tok_s']:.0f} tok/s")
        print(f"  Torch    : {res['torch_ms']:.3f} ms, {res['torch_tflops']:.2f} TFLOPS, {res['torch_tok_s']:.0f} tok/s")

    return results


if __name__ == "__main__":
    main()
