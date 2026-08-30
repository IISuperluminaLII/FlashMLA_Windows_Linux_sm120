"""
Verify FlashMLA under Inductor cudagraph_trees (mode='reduce-overhead' == the cudagraph path of
max-autotune). This is the layer raw torch.cuda.graph can't test: cudagraph_trees does memory-pool
ACCOUNTING and rejects persistent tensors left in the pool ("live storage data ptrs ... not
accounted for"). Runs fwd+bwd several iters to force warmup -> capture -> replay.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/_diag_cgtrees.py
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.9")
os.environ["PATH"] = "/usr/local/cuda-12.9/bin:" + os.environ.get("PATH", "")
import torch
from flash_mla.flash_mla_interface import flash_attn_varlen_func

DEV = "cuda"
S, H, Dqk, Dvo = 1024, 22, 192, 128
cu = torch.tensor([0, S], device=DEV, dtype=torch.int32)


def mk(seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(S, H, Dqk, device=DEV, dtype=torch.bfloat16, generator=g, requires_grad=True),
            torch.randn(S, H, Dqk, device=DEV, dtype=torch.bfloat16, generator=g, requires_grad=True),
            torch.randn(S, H, Dvo, device=DEV, dtype=torch.bfloat16, generator=g, requires_grad=True))


def fn(q, k, v):
    # 4 custom-op calls in one graph (mimics multiple layers -> multiple per-call workspaces that
    # must allocate+free correctly in the SINGLE cudagraph_trees pool).
    acc = 0
    for _ in range(4):
        o, _ = flash_attn_varlen_func(q, k, v, cu, cu, S, S, softmax_scale=Dqk ** -0.5, causal=True)
        acc = acc + o
    return acc


if __name__ == "__main__":
    print(f"[INFO] torch={torch.__version__} dev={torch.cuda.get_device_name(0)}")
    cfn = torch.compile(fn, mode="max-autotune")      # the user's exact mode (Inductor + cudagraph_trees)
    try:
        for i in range(6):                            # warmup -> capture (~3rd) -> replay
            q, k, v = mk(i)
            o = cfn(q, k, v)
            o.float().pow(2).mean().backward()
            torch.cuda.synchronize()
        print("RESULT: OK -- fwd+bwd ran 6 iters under cudagraph_trees (capture+replay, no 'live storage' error)")
        raise SystemExit(0)
    except Exception as e:
        msg = str(e)
        live = "live storage data ptrs" in msg
        print(f"RESULT: FAILED {type(e).__name__}: {'LIVE-STORAGE (persistent tensor in pool)' if live else msg[:200]}")
        raise SystemExit(1)
