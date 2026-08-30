"""
Decode-case parity probe (NOT a pytest; underscore-prefixed). The needle eval generates
autoregressively with a KV cache: after prefill, every step calls attention with ONE query
token (seq_q=1) against N cached keys (seq_kv=N). The single most-recent query must attend
to ALL N keys (causal bottom-right alignment, the FlashAttention/KV-cache convention).

The model's FlashMLA path hardcodes causal=True and reshapes to varlen with seq_q=1,
seq_kv=N. If the sm_120 forward kernel aligns causal TOP-LEFT (query local index 0 -> sees
only key 0), the decode query sees almost nothing -> generation is garbage -> needle fails,
while SDPA works. This probe isolates exactly that, which the square-only accuracy suite
never exercised.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/_diag_decode.py
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn.functional as F
from flash_mla.flash_mla_interface import flash_attn_varlen_func, FLASH_MLA_LOADED_VARIANT

DEV = "cuda"


def relerr(a, b):
    a, b = a.float(), b.float()
    return (a - b).abs().mean().item() / (b.abs().mean().item() + 1e-8)


def ref_full(q1, k, v, scale):
    """Single query attends to ALL keys (correct decode semantics, causal bottom-right)."""
    qf, kf, vf = q1.float(), k.float(), v.float()
    scores = torch.einsum("qhd,khd->hqk", qf, kf) * scale     # [H,1,N]
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", p, vf)                # [1,H,Dvo]


def ref_topleft(q1, k, v, scale):
    """BUGGY top-left: query at local index 0 attends to key 0 only."""
    qf, kf, vf = q1.float(), k[:1].float(), v[:1].float()
    scores = torch.einsum("qhd,khd->hqk", qf, kf) * scale     # [H,1,1]
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", p, vf)


def main():
    print(f"[INFO] variant={FLASH_MLA_LOADED_VARIANT} dev={torch.cuda.get_device_name(0)}")
    torch.manual_seed(0)
    H, Dqk, Dvo = 2, 192, 128
    scale = Dqk ** -0.5

    print("\n=== DECODE case: seq_q=1, seq_kv=N, causal=True ===")
    print("(correct: single query attends to ALL N keys)")
    for N in [16, 64, 200, 512]:
        k = torch.randn(N, H, Dqk, device=DEV, dtype=torch.bfloat16) * 0.5
        v = torch.randn(N, H, Dvo, device=DEV, dtype=torch.bfloat16) * 0.5
        q1 = torch.randn(1, H, Dqk, device=DEV, dtype=torch.bfloat16) * 0.5
        cu_q = torch.tensor([0, 1], device=DEV, dtype=torch.int32)
        cu_k = torch.tensor([0, N], device=DEV, dtype=torch.int32)
        with torch.no_grad():
            o_flash, _ = flash_attn_varlen_func(q1, k, v, cu_q, cu_k, 1, N,
                                                softmax_scale=scale, causal=True, is_varlen=True)
        nan = torch.isnan(o_flash.float()).any().item()
        e_full = relerr(o_flash, ref_full(q1, k, v, scale))
        e_tl = relerr(o_flash, ref_topleft(q1, k, v, scale))
        verdict = "OK(full)" if e_full < 0.05 else ("TOPLEFT-BUG" if e_tl < 0.05 else "WRONG")
        print(f"  N={N:4d}: nan={nan!s:5s} rel_vs_full={e_full:.3e} rel_vs_topleft={e_tl:.3e}  [{verdict}]")

    print("\n=== CONTROL prefill: seq_q=seq_kv=N, causal=True (square, already tested) ===")
    for N in [64, 200]:
        q = torch.randn(N, H, Dqk, device=DEV, dtype=torch.bfloat16) * 0.5
        k = torch.randn(N, H, Dqk, device=DEV, dtype=torch.bfloat16) * 0.5
        v = torch.randn(N, H, Dvo, device=DEV, dtype=torch.bfloat16) * 0.5
        cu = torch.tensor([0, N], device=DEV, dtype=torch.int32)
        with torch.no_grad():
            o_flash, _ = flash_attn_varlen_func(q, k, v, cu, cu, N, N,
                                                softmax_scale=scale, causal=True, is_varlen=True)
        # ref: causal square, read the LAST query row (its full causal context = all N keys)
        qf, kf, vf = q.float(), k.float(), v.float()
        scores = torch.einsum("ihd,jhd->hij", qf, kf) * scale
        i = torch.arange(N, device=DEV).view(1, N, 1); j = torch.arange(N, device=DEV).view(1, 1, N)
        scores = scores.masked_fill(j > i, float("-inf"))
        p = torch.softmax(scores, dim=-1)
        o_ref = torch.einsum("hij,jhd->ihd", p, vf)
        e = relerr(o_flash, o_ref)
        print(f"  N={N:4d}: rel_vs_causal_ref={e:.3e}  [{'OK' if e < 0.05 else 'WRONG'}]")


if __name__ == "__main__":
    main()
