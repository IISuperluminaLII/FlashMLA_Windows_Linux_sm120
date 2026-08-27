"""
Authors' DENSE decode performance cases (test_flash_mla_decoding.py performance_cases,
dense subset: is_fp8=False, topk=None) on sm120 -- runner test_flash_mla reused verbatim
(it asserts correctness AND prints bandwidth/TFlops for test_performance=True).

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/_diag_dense_decode_perf.py
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
        TestParam(128, s_q, s_k, is_varlen=True, is_causal=is_causal, is_fp8=False,
                  topk=None, test_performance=True)
        for is_causal in [False, True]
        for s_q in [1, 2]
        for s_k in [4096, 8192, 16384, 32768]
    ]

    passed, failed = 0, []
    for t in cases:
        try:
            test_flash_mla(t)
            passed += 1
        except Exception as e:
            failed.append((t, f"{type(e).__name__}: {str(e)[:120]}"))
            print(f"[FAILED] {t}\n    {type(e).__name__}: {str(e)[:120]}")

    print(f"\n[RESULT] {passed} passed, {len(failed)} failed of {len(cases)}")
    raise SystemExit(1 if failed else 0)
