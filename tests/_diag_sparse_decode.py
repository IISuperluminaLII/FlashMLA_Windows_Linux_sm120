"""
First-ever exercise of the sm120 sparse-FP8 decode path (pybind wires it to
sm120::sparse_decode::run_sparse_fp8_decode_kernel, but no test has ever run it).
Reuses the AUTHORS' test_flash_mla_decoding.py runner (test_flash_mla) verbatim on the
sparse (is_fp8=True, topk!=None) correctness + corner subset.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/_diag_sparse_decode.py
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

    cases = [
        TestParam(b, s_q, s_k, is_varlen, False, True, topk, test_performance=False)
        for b in [1, 64]
        for s_q in [1, 2]
        for s_k in [140, 4096]
        for is_varlen in [False, True]
        for topk in [128, 2048]
    ] + [
        TestParam(128, 2, 4096, is_varlen=True, is_causal=False, is_fp8=True,
                  topk=2048, test_performance=False, is_all_indices_invalid=True),
        TestParam(128, 2, 4096, is_varlen=True, is_causal=False, is_fp8=True,
                  topk=128, test_performance=False, have_zero_seqlen_k=True),
    ]

    passed, failed = 0, []
    for t in cases:
        try:
            test_flash_mla(t)
            passed += 1
            print(f"[OK] {t}")
        except Exception as e:
            failed.append((t, f"{type(e).__name__}: {str(e)[:140]}"))
            print(f"[FAILED] {t}\n    {type(e).__name__}: {str(e)[:140]}")
            torch.cuda.synchronize() if "illegal" not in str(e).lower() else None

    print(f"\n[RESULT] {passed} passed, {len(failed)} failed of {len(cases)}")
    raise SystemExit(1 if failed else 0)
