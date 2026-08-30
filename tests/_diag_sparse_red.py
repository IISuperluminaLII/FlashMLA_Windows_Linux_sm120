"""
Red/green probe for sm120 sparse prefill fwd: runs the AUTHORS' correctness + corner
cases from test_flash_mla_prefill.py (run_test reused verbatim, benchmarks off).
Expected RED on the unfixed kernel: fwd.cu loads V at offset d_qk (OOB / next token's
K data) instead of the first d_v latent elements, and max_logits uses the wrong scale
convention (raw max instead of max of sm_scale*log2e-scaled scores).

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/_diag_sparse_red.py
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from test_flash_mla_prefill import TestParam, run_test

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    correctness_cases = [
        TestParam(1, s_q, s_kv, topk, h_q=128, benchmark=False)
        for s_kv, topk in [
            (128, 128), (256, 256), (512, 512),
            (592, 128), (1840, 256), (1592, 384), (1521, 512),
            (95, 128), (153, 256), (114, 384),
        ]
        for s_q in [1, 62]
    ]
    corner_cases = [
        TestParam(1, s_q, s_kv, topk, h_q=128, benchmark=False)
        for s_kv, topk in [(32, 2048), (64, 8192)]
        for s_q in [1, 1024]
    ]
    testcases = correctness_cases + corner_cases

    failed = []
    for t in testcases:
        try:
            ok = run_test(t)
        except Exception as e:
            print(f"[FAILED] {t}: {type(e).__name__}: {str(e)[:120]}")
            ok = False
        if not ok:
            failed.append(t)

    print(f"\n[RESULT] {len(testcases) - len(failed)} passed, {len(failed)} failed of {len(testcases)}")
    raise SystemExit(1 if failed else 0)
