#!/usr/bin/env bash
# Single-TU compile check + ptxas resource stats WITHOUT touching the shared build.
#   bash tests/_ptxas_probe_tu.sh <path/to/tu.cu> [kernel-name-regex]
# Resolves torch/python include dirs from the env python, compiles with the same
# nvcc flags setup.py uses (-O3 -std=c++17 sm_120 relaxed-constexpr), prints
# REG/STACK/spill lines for kernels matching the regex. CPU-only.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
TU="${1:?usage: _ptxas_probe_tu.sh <tu.cu> [kernel-regex]}"
PAT="${2:-.}"
NVCC="$CUDA_HOME/bin/nvcc"
TORCH_INC="$("$PY" -c 'import torch.utils.cpp_extension as c; print(" ".join("-I"+p for p in c.include_paths()))')"
PY_INC="$("$PY" -c 'import sysconfig; print("-I"+sysconfig.get_paths()["include"])')"
OUT="/tmp/_ptxas_probe_$(basename "$TU" .cu).o"
LOG="/tmp/_ptxas_probe_$(basename "$TU" .cu).log"
"$NVCC" -std=c++17 -O3 -arch=sm_120 -DFLASH_MLA_BUILD_SM120 -DNDEBUG \
  --expt-relaxed-constexpr --expt-extended-lambda -Wno-deprecated-declarations ${EXTRA_NVCC_FLAGS:-} \
  -I csrc -I csrc/cutlass/include -I csrc/cutlass/tools/util/include -I "$CUDA_HOME/include/cccl" $TORCH_INC $PY_INC \
  -D_GLIBCXX_USE_CXX11_ABI=1 -DTORCH_API_INCLUDE_EXTENSION_H \
  --ptxas-options=-v -c "$TU" -o "$OUT" > "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  echo "[FAILED] compile rc=$RC (first errors):"; grep -m8 -E "error|Error" "$LOG"; exit $RC
fi
echo "[OK] compiled $TU"
grep -B1 -A3 "ptxas info    : Function properties for" "$LOG" | grep -E "Function properties|spill|Used [0-9]+ registers" \
  | paste - - - 2>/dev/null | grep -E "$PAT" | sed 's/ptxas info    : //g' | cut -c1-220
echo "PTXAS_PROBE_DONE"
