import math
import time
import torch
from flash_mla.flash_mla_interface import flash_attn_varlen_func

seqlens = [1024, 2048, 4096, 8192]
batch = 128
heads = 128
head_dim = 128
device = torch.device('cuda:0')
torch.set_default_device(device)
torch.set_default_dtype(torch.bfloat16)
torch.cuda.set_device(device)

def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters

results_torch = []
results_flash = []

for seqlen in seqlens:
    total_tokens = batch * seqlen
    q = torch.randn(total_tokens, heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    grad = torch.randn_like(q)

    cu_seqlens = torch.arange(0, (batch + 1) * seqlen, seqlen, dtype=torch.int32, device=device)
    max_seqlen = seqlen

    q_t = q.clone().detach().requires_grad_(True)
    k_t = k.clone().detach().requires_grad_(True)
    v_t = v.clone().detach().requires_grad_(True)

    def torch_step():
        q_t.grad = k_t.grad = v_t.grad = None
        out = torch.nn.functional.scaled_dot_product_attention(
            q_t.view(batch, seqlen, heads, head_dim).transpose(1, 2),
            k_t.view(batch, seqlen, heads, head_dim).transpose(1, 2),
            v_t.view(batch, seqlen, heads, head_dim).transpose(1, 2),
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(total_tokens, heads, head_dim)
        out.backward(grad)
        return out

    torch_time = bench(torch_step)

    q_f = q.clone().detach().requires_grad_(True)
    k_f = k.clone().detach().requires_grad_(True)
    v_f = v.clone().detach().requires_grad_(True)

    def flash_step():
        q_f.grad = k_f.grad = v_f.grad = None
        out, _ = flash_attn_varlen_func(
            q_f,
            k_f,
            v_f,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=False,
            softmax_scale=None,
            is_varlen=True,
        )
        out.backward(grad)
        return out

    flash_time = bench(flash_step)

    bytes_rw = (q.numel() + k.numel() + v.numel() + grad.numel()) * q.element_size()
    gb_torch = bytes_rw / (torch_time / 1000.0) / 1e9
    gb_flash = bytes_rw / (flash_time / 1000.0) / 1e9

    results_torch.append((seqlen, gb_torch, torch_time))
    results_flash.append((seqlen, gb_flash, flash_time))

with open('bwd_torch_perf.csv', 'w') as f:
    f.write('name,batch,seqlen,head,bw,time_ms\n')
    for seqlen, gb, ms in results_torch:
        f.write(f'torch_bwd,{batch},{seqlen},{heads},{gb:.2f},{ms:.3f}\n')

with open('bwd_flash_mla_perf.csv', 'w') as f:
    f.write('name,batch,seqlen,head,bw,time_ms\n')
    for seqlen, gb, ms in results_flash:
        f.write(f'flash_mla_bwd,{batch},{seqlen},{heads},{gb:.2f},{ms:.3f}\n')
