#!/usr/bin/env bash
# Compile-check the three TUs the agents left mid-flight, with the per-kernel ptxas lines that matter.
cd "$(dirname "$0")/.."
echo "=== FWD (rung 11 / 97)"; bash tests/_ptxas_probe_tu.sh csrc/sm120/prefill/dense/fmha_cutlass_fwd_sm120.cu 'Li11E|Li97E' 2>&1 | grep -vE 'pynvml|FutureWarning' | sed 's/_ZN5flash6detail7mma_fwd[0-9]*fmha_fwd_mma_//; s/EEvPK13__nv_bfloat16S5_S5_PKiS7_PS3_Pfifiii//; s/EEv14CUtensorMap_stS3_PK13__nv_bfloat16S6_S6_PKiS8_PS4_Pfifi//' | grep -E 'OK|FAILED|error|Function|PTXAS' | head -14
echo "=== BWD (CFG=4)"; bash tests/_ptxas_probe_tu.sh csrc/sm120/prefill/dense/fmha_cutlass_bwd_sm120.cu 'dkdv_kernelILi[0-9]*ELi128ELb[01]ELi4E' 2>&1 | grep -vE 'pynvml|FutureWarning' | grep -E 'OK|FAILED|error|Function|PTXAS' | sed 's/_ZN5flash6detail7mma_bwd[0-9]*//' | head -12
echo "=== SPARSE FWD (ws)"; bash tests/_ptxas_probe_tu.sh csrc/sm120/prefill/sparse/fwd.cu 'sparse_prefill_fwd_ws|fwd_ws' 2>&1 | grep -vE 'pynvml|FutureWarning' | grep -E 'OK|FAILED|error|Function|PTXAS' | head -12
echo THREE_TU_CHECK_DONE
