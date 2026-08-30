"""Quick verification test for K-major kernel."""
import torch
import math
from flash_mla.flash_mla_interface import flash_attn_varlen_func, FLASH_MLA_LOADED_VARIANT

def main():
    print(f"FlashMLA variant: {FLASH_MLA_LOADED_VARIANT}")

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    # Test multiple sizes
    test_configs = [
        (64, 2, 128, "small"),
        (128, 4, 128, "medium"),
        (256, 8, 128, "large"),
        (512, 4, 128, "longer"),
    ]

    for L, H, D, name in test_configs:
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(L, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(L, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(L, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        cu = torch.tensor([0, L], device=device, dtype=torch.int32)

        out, lse = flash_attn_varlen_func(q, k, v, cu, cu, L, L, softmax_scale=scale, causal=False, is_varlen=True)

        d_o = torch.randn_like(out)
        out.backward(d_o)
        torch.cuda.synchronize()

        # Verify gradients exist and are non-zero
        assert q.grad is not None and k.grad is not None and v.grad is not None, f"Gradients None for {name}"
        dq_norm = float(q.grad.float().norm())
        dk_norm = float(k.grad.float().norm())
        dv_norm = float(v.grad.float().norm())

        assert dq_norm > 0 and dk_norm > 0 and dv_norm > 0, f"Zero gradients for {name}"
        print(f"[OK] {name} (L={L}, H={H}): dq={dq_norm:.4f}, dk={dk_norm:.4f}, dv={dv_norm:.4f}")

    # Test causal attention
    print("\nTesting causal attention:")
    for L, H, D, name in [(128, 4, 128, "causal_medium"), (256, 4, 128, "causal_large")]:
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(L, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(L, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(L, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        cu = torch.tensor([0, L], device=device, dtype=torch.int32)

        out, lse = flash_attn_varlen_func(q, k, v, cu, cu, L, L, softmax_scale=scale, causal=True, is_varlen=True)

        d_o = torch.randn_like(out)
        out.backward(d_o)
        torch.cuda.synchronize()

        assert q.grad is not None and k.grad is not None and v.grad is not None
        dq_norm = float(q.grad.float().norm())
        dk_norm = float(k.grad.float().norm())
        dv_norm = float(v.grad.float().norm())

        assert dq_norm > 0 and dk_norm > 0 and dv_norm > 0
        print(f"[OK] {name} (L={L}, H={H}): dq={dq_norm:.4f}, dk={dk_norm:.4f}, dv={dv_norm:.4f}")

    print("\n[DONE] All K-major kernel tests passed!")

if __name__ == "__main__":
    main()
