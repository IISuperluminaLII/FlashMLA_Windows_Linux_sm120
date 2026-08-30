"""
Authors' sparse prefill PERFORMANCE cases (test_flash_mla_prefill.py performance_cases,
run_test reused verbatim: correctness check + triton do_bench TFlops print) on sm120.
Produces the sm120 column for the final authors-vs-sm120 benchmark comparison.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/_diag_sparse_perf.py
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from test_flash_mla_prefill import TestParam, run_test

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    performance_cases = [
        TestParam(1, s_q, s_kv, topk, h_q=128)
        for s_q in [4096]
        for s_kv in [4096, 8192, 16384, 32768, 49152, 65536, 81920, 98304, 114688, 131072]
        for topk in [2048]
    ]

    failed = []
    for t in performance_cases:
        time.sleep(0.2)
        try:
            ok = run_test(t)
        except Exception as e:
            print(f"[FAILED] {t}: {type(e).__name__}: {str(e)[:140]}")
            ok = False
        if not ok:
            failed.append(t)

    print(f"\n[RESULT] {len(performance_cases) - len(failed)} passed, {len(failed)} failed of {len(performance_cases)}")
    raise SystemExit(1 if failed else 0)
