"""
Perf-only probe for the sparse fp8 decode tiers (authors' perf shapes + metric line verbatim, NO
oracle assert -- the fp8 tensor-math tiers 5/6 are certified by test_sparse_decode_fp8mma.py, and the
authors' cos 5e-6 tolerance would raise before timing). The CFG env is latched per process; the chain
runs this once per tier.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 FLASH_MLA_SM120_SPARSE_DECODE_CFG=<n> \
      python tests/_diag_sparse_fp8mma_bench.py
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import triton
import flash_mla
import quant
from test_flash_mla_decoding import TestParam, generate_test_data


@torch.inference_mode()
def bench(name: str, t: TestParam):
    cache_seqlens, q, block_table, blocked_k, _abs, indices = generate_test_data(t)
    kq = quant.quantize_k_cache(blocked_k, t.dv, 128)
    meta, num_splits = flash_mla.get_mla_metadata(cache_seqlens, t.s_q * t.h_q // t.h_kv, t.h_kv, t.h_q, True, t.topk)

    def run():
        return flash_mla.flash_mla_with_kvcache(q, kq, block_table, cache_seqlens, t.dv, meta, num_splits,
                                                causal=False, is_fp8_kvcache=True, indices=indices)
    out, lse = run(); torch.cuda.synchronize()
    assert torch.isfinite(out).all(), "non-finite output"
    ms = triton.testing.do_bench(run, warmup=25, rep=200)
    flop = t.b * t.h_q * t.s_q * (2 * t.d * t.topk + 2 * t.topk * t.dv)
    byte = t.b * (t.s_q * t.h_q * t.d * 2 + t.s_q * t.topk * t.h_kv * 656 + t.s_q * t.h_q * t.dv * 2)
    print(f"[BENCH] CFG={os.environ.get('FLASH_MLA_SM120_SPARSE_DECODE_CFG', '0')} {name}: {ms:.3f} ms, "
          f"{flop / ms / 1e9:.0f} TFLOPS, {byte / ms / 1e6:.0f} GB/s")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    bench("bench-b128-sq2-topk2048", TestParam(128, 2, 4096, True, False, True, 2048, test_performance=True))
    bench("serving-b4-sq1-topk2048", TestParam(4, 1, 4096, False, False, True, 2048, test_performance=True))
    bench("serving-b32-sq1-topk2048", TestParam(32, 1, 4096, False, False, True, 2048, test_performance=True))
    print("BENCH_DONE")
