"""
Sparse fp8 decode perf probe for the split-KV A/B (authors' harness verbatim,
correctness asserted in the same run). The CFG env is latched per process --
the runner script executes this once per tier.

Shapes:
  - bench   : b=128, s_q=2, s_k=4096, topk=2048  (the 0.800 ms CFG=1 row;
              512 CTAs / 3 waves batch-parallel vs 188 CTAs / 1 wave split)
  - serving : b=4, s_q=1, s_k=4096, topk=2048    (batch-parallel: 8 CTAs on
              188 SMs = 4.3% occupancy; split-KV: 128 CTAs)

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  FLASH_MLA_SM120_SPARSE_DECODE_CFG=<n> python tests/_diag_sparse_splitkv_bench.py
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from test_flash_mla_decoding import TestParam, test_flash_mla

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    cfg = os.environ.get("FLASH_MLA_SM120_SPARSE_DECODE_CFG", "0")
    cases = [
        ("bench-b128-sq2", TestParam(128, 2, 4096, True, False, True, 2048, test_performance=True)),
        ("serving-b4-sq1", TestParam(4, 1, 4096, False, False, True, 2048, test_performance=True)),
    ]
    failed = 0
    for name, t in cases:
        print(f"[CASE] CFG={cfg} {name}")
        try:
            test_flash_mla(t)   # prints "x.xxx ms, N TFLOPS, N GB/s"; asserts correctness
        except Exception as e:
            failed += 1
            print(f"[FAILED] {name}: {type(e).__name__}: {str(e)[:140]}")
    print(f"[RESULT] {len(cases) - failed} passed, {failed} failed of {len(cases)}")
    raise SystemExit(1 if failed else 0)
