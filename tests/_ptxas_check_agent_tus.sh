#!/usr/bin/env bash
# Compile-check the TUs currently under agent edit (bwd dense + sparse decode) so the
# coordinator knows whether a source snapshot would build. CPU only.
set -u
cd "$(dirname "$0")/.."
for tu in csrc/sm120/prefill/dense/fmha_cutlass_bwd_sm120.cu csrc/sm120/decode/sparse_fp8/splitkv_mla.cu; do
  echo "=== $tu"
  bash tests/_ptxas_probe_tu.sh "$tu" 'NOMATCH_REGEX_ONLY_STATUS' 2>&1 | grep -E '^\[OK\]|^\[FAILED\]|error|PTXAS_PROBE_DONE' | head -20
done
echo AGENT_TU_CHECK_DONE
