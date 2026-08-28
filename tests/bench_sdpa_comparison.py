"""
FlashMLA vs torch SDPA on the three paths that matter, same GPU, same instant.

1. DENSE TRAINING PATH (192/128, causal, the 7b model's 22 heads):
   flash_attn_varlen_func vs F.scaled_dot_product_attention. Unequal qk/v head dims force
   SDPA onto the MATH backend -- exactly what the training stack falls back to without
   FlashMLA. fwd and fwd+bwd.

2. SPARSE PREFILL (576/512, topk=2048, h_q=128, DeepSeek DSA):
   flash_mla_sparse_fwd/_bwd vs the SDPA-equivalent a torch user would write:
   index_select the topk KV rows then math-SDPA over them (gather included in the timing --
   it is part of the job; FlashMLA gathers in-kernel).

3. SPARSE FP8 DECODE (b=128, s_q=2, topk=2048):
   flash_mla_with_kvcache (fp8 cache) vs gather+math-SDPA over the dequantized bf16 pool.

Each pair is numerically cross-checked (cos diff) before timing so both sides provably
compute the same attention.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/bench_sdpa_comparison.py
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import math
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import triton

import flash_mla
from flash_mla import flash_mla_sparse_fwd
from flash_mla.flash_mla_interface import (
    flash_attn_varlen_func, flash_mla_sparse_bwd, FLASH_MLA_LOADED_VARIANT,
)
import quant
from test_flash_mla_decoding import TestParam, generate_test_data

torch.manual_seed(0)
DEV = "cuda"


def bench_ms(fn):
    return triton.testing.do_bench(fn, warmup=10, rep=25)


def cosdiff(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (1 - (a @ b) / (a.norm() * b.norm() + 1e-30)).item()


def fmt(name, flash_ms, sdpa_ms, note=""):
    print(f"{name:44s} FlashMLA {flash_ms:9.3f} ms   SDPA {sdpa_ms:9.3f} ms   "
          f"speedup {sdpa_ms / flash_ms:5.2f}x  {note}")


print(f"[INFO] variant={FLASH_MLA_LOADED_VARIANT} dev={torch.cuda.get_device_name(0)} "
      f"CFG={os.environ.get('FLASH_MLA_SM120_SPARSE_FWD_CFG', '0')}")

# ============================================================================
# 1. DENSE TRAINING PATH: 192/128, 22 heads (the 7b model shape), causal
# ============================================================================
print("\n== 1. dense training path (qk=192, v=128, H=22, causal; SDPA=math backend) ==")
for S in (1024, 4096, 8192):
    H, Dqk, Dvo = 22, 192, 128
    scale = Dqk ** -0.5
    q = torch.randn(S, H, Dqk, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(S, H, Dqk, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(S, H, Dvo, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    cu = torch.tensor([0, S], device=DEV, dtype=torch.int32)
    qs = q.detach().transpose(0, 1).unsqueeze(0).requires_grad_(True)   # [1,H,S,192]
    ks = k.detach().transpose(0, 1).unsqueeze(0).requires_grad_(True)
    vs = v.detach().transpose(0, 1).unsqueeze(0).requires_grad_(True)

    def flash_fwd():
        return flash_attn_varlen_func(q, k, v, cu, cu, S, S, softmax_scale=scale, causal=True)[0]

    def sdpa_fwd():
        return F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, scale=scale)

    cd = cosdiff(flash_fwd(), sdpa_fwd().squeeze(0).transpose(0, 1))
    fmt(f"  fwd S={S}", bench_ms(flash_fwd), bench_ms(sdpa_fwd), f"(cos_diff {cd:.1e})")

    go = torch.randn(S, H, Dvo, device=DEV, dtype=torch.bfloat16)
    gos = go.transpose(0, 1).unsqueeze(0).contiguous()

    def flash_fb():
        out = flash_attn_varlen_func(q, k, v, cu, cu, S, S, softmax_scale=scale, causal=True)[0]
        torch.autograd.grad(out, (q, k, v), go, retain_graph=False)

    def sdpa_fb():
        out = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, scale=scale)
        torch.autograd.grad(out, (qs, ks, vs), gos, retain_graph=False)

    fmt(f"  fwd+bwd S={S}", bench_ms(flash_fb), bench_ms(sdpa_fb))
    del q, k, v, qs, ks, vs, go, gos
    torch.cuda.empty_cache()

# ============================================================================
# 2. SPARSE PREFILL: 576/512, topk=2048, h_q=128 (gather + math-SDPA baseline)
# ============================================================================
print("\n== 2. sparse prefill (d_qk=576, d_v=512, topk=2048, h_q=128; SDPA=gather+math) ==")
s_q, s_kv, topk, h_q = 512, 8192, 2048, 128
sm_scale = 1.0 / math.sqrt(576)
q = (torch.randn(s_q, h_q, 576, device=DEV, dtype=torch.bfloat16) / 10)
kv = (torch.randn(s_kv, 1, 576, device=DEV, dtype=torch.bfloat16) / 10)
idx = torch.stack([torch.randperm(s_kv, device=DEV)[:topk] for _ in range(s_q)]
                  ).to(torch.int32).view(s_q, 1, topk)


def flash_sparse_fwd():
    return flash_mla_sparse_fwd(q, kv, idx, sm_scale)


def sdpa_sparse_fwd():
    # MQA trick: all h_q heads share this position's KV, so heads become the L_q dim
    # (batch=s_q, H=1, L_q=h_q, L_k=topk) -- no broadcast materialization.
    gathered = kv[:, 0, :].index_select(0, idx.view(-1).long()).view(s_q, topk, 576)
    qq = q.view(s_q, 1, h_q, 576)
    kk = gathered.view(s_q, 1, topk, 576)
    vv = gathered[:, :, :512].view(s_q, 1, topk, 512)
    return F.scaled_dot_product_attention(qq, kk, vv, scale=sm_scale)


cd = cosdiff(flash_sparse_fwd()[0], sdpa_sparse_fwd().squeeze(1))
fmt(f"  fwd s_q={s_q} s_kv={s_kv}", bench_ms(flash_sparse_fwd), bench_ms(sdpa_sparse_fwd),
    f"(cos_diff {cd:.1e})")

out, _, lse = flash_sparse_fwd()
d_o = torch.randn_like(out) / 10
qg = q.detach().requires_grad_(True)
kvg = kv.detach().requires_grad_(True)


def flash_sparse_fb():
    o, _, l = flash_mla_sparse_fwd(q, kv, idx, sm_scale)
    flash_mla_sparse_bwd(d_o, q, kv, o, l, idx, sm_scale)


def sdpa_sparse_fb():
    gathered = kvg[:, 0, :].index_select(0, idx.view(-1).long()).view(s_q, topk, 576)
    o = F.scaled_dot_product_attention(
        qg.view(s_q, 1, h_q, 576), gathered.view(s_q, 1, topk, 576),
        gathered[:, :, :512].view(s_q, 1, topk, 512), scale=sm_scale)
    torch.autograd.grad(o, (qg, kvg), d_o.view(s_q, 1, h_q, 512), retain_graph=False)


fmt(f"  fwd+bwd s_q={s_q}", bench_ms(flash_sparse_fb), bench_ms(sdpa_sparse_fb))
del q, kv, idx, out, d_o, qg, kvg
torch.cuda.empty_cache()

# ============================================================================
# 3. SPARSE FP8 DECODE: b=128, s_q=2, topk=2048 (authors' perf shape)
# ============================================================================
print("\n== 3. sparse fp8 decode (b=128, s_q=2, h_q=128, topk=2048; SDPA=gather+math) ==")
t = TestParam(128, 2, 4096, is_varlen=True, is_causal=False, is_fp8=True,
              topk=2048, test_performance=False)
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(torch.device("cuda:0"))
cache_seqlens, dq, block_table, blocked_k, abs_indices, indices_in_kvcache = generate_test_data(t)
blocked_k_quantized = quant.quantize_k_cache(blocked_k, t.dv, 128)
blocked_k_dequant = quant.dequantize_k_cache(blocked_k_quantized)
tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
    cache_seqlens, t.s_q * t.h_q // t.h_kv, t.h_kv, t.h_q, True, t.topk)

# The oracle NaN-poisons every pool row no index references, so invalid (-1) lanes
# must NOT gather an arbitrary row (clamp-to-0 can hit a poisoned row; the math
# backend adds the mask's -inf AFTER QK^T, and NaN + -inf = NaN kills the whole
# softmax row). Gather invalids from an appended all-zero sentinel row instead:
# finite scores, then the attn_mask removes them exactly.
flat_pool = torch.cat([blocked_k_dequant.view(-1, 576),
                       torch.zeros(1, 576, dtype=blocked_k_dequant.dtype)], dim=0)
zero_row = flat_pool.size(0) - 1
safe_idx = indices_in_kvcache.long().view(-1).clone()
safe_idx[safe_idx < 0] = zero_row
invalid = (indices_in_kvcache < 0).view(t.b, t.s_q, 1, t.topk)


def flash_decode():
    return flash_mla.flash_mla_with_kvcache(
        dq, blocked_k_quantized, block_table, cache_seqlens, t.dv,
        tile_scheduler_metadata, num_splits, causal=False,
        is_fp8_kvcache=True, indices=indices_in_kvcache)


def sdpa_decode():
    # Same MQA trick: (batch = b*s_q, H = 1, L_q = h_q, L_k = topk).
    gathered = flat_pool.index_select(0, safe_idx).view(t.b, t.s_q, t.topk, 576)
    qq = dq.reshape(t.b * t.s_q, 1, t.h_q, 576)
    kk = gathered.view(t.b * t.s_q, 1, t.topk, 576)
    vv = gathered[..., :512].view(t.b * t.s_q, 1, t.topk, 512)
    mask = (~invalid).view(t.b * t.s_q, 1, 1, t.topk)
    return F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask,
                                          scale=1.0 / math.sqrt(576))


fa = flash_decode()[0]
sd = sdpa_decode().view(t.b, t.s_q, t.h_q, 512)
# Rows with ZERO valid indices are softmax(all -inf) = NaN under SDPA; the kernel's
# convention is O = 0 there (L == 0 sentinel). Align the baseline, then the finite
# assert makes the cross-check real (a NaN on either side must fail, not print nan).
empty_rows = invalid.view(t.b, t.s_q, 1, t.topk).all(dim=-1)      # [b, s_q, 1]
sd = torch.where(empty_rows.unsqueeze(-1), torch.zeros_like(sd), sd)
assert torch.isfinite(fa.float()).all(), "FlashMLA decode produced non-finite output"
assert torch.isfinite(sd.float()).all(), "SDPA decode baseline produced non-finite output"
cd = cosdiff(fa, sd)
fmt(f"  decode b=128 s_q=2 topk=2048", bench_ms(flash_decode), bench_ms(sdpa_decode),
    f"(cos_diff {cd:.1e})")
print("\nBENCH_SDPA_DONE")
