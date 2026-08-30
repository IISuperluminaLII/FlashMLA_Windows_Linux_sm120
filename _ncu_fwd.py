import torch
import flash_mla
from flash_mla.flash_mla_interface import flash_attn_varlen_func
torch.manual_seed(0)
S, H, D = 2048, 8, 128
cu = torch.arange(0, 2 * S + 1, S, device='cuda', dtype=torch.int32)
q = torch.randn(2 * S, H, D, device='cuda', dtype=torch.bfloat16)
k = torch.randn(2 * S, H, D, device='cuda', dtype=torch.bfloat16)
v = torch.randn(2 * S, H, D, device='cuda', dtype=torch.bfloat16)
with torch.no_grad():
    for _ in range(3):
        flash_attn_varlen_func(q, k, v, cu, cu, S, S, softmax_scale=D ** -0.5, causal=True, is_varlen=True)
torch.cuda.synchronize()
