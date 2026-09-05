#!/usr/bin/env bash
# Print the live toolchain: torch / CUDA runtime / cuDNN / triton, env nvcc, system toolkits,
# the resolver's CUDA_HOME, and the toolchain stamp of the built extension.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
echo "PY=$PY"
"$PY" - <<'PYEOF' 2>/dev/null
import torch, triton
print(f"torch {torch.__version__} | torch.version.cuda {torch.version.cuda} | cudnn {torch.backends.cudnn.version()} | triton {triton.__version__}")
print(f"torch cuda available {torch.cuda.is_available()} | device0 {torch.cuda.get_device_name(0)} | cc {torch.cuda.get_device_capability(0)}")
PYEOF
echo "env nvcc: $(command -v nvcc) -> $(nvcc --version | tail -1)"
echo "resolver CUDA_HOME=$CUDA_HOME -> $("$CUDA_HOME/bin/nvcc" --version | tail -1)"
echo "toolkits: $(ls -d /usr/local/cuda-* 2>/dev/null | tr '\n' ' ') | /usr/local/cuda -> $(readlink -f /usr/local/cuda)"
echo "conda cuda pkgs:"; "$PY" -m pip list 2>/dev/null | grep -iE "^(torch|nvidia-cuda|nvidia-cudnn|cuda-nvcc|triton|transformer.engine|flash)" | head -12
echo "built .so stamp: $(cat "$FMLA_ROOT/build/.toolchain_stamp" 2>/dev/null)"
echo "ENV_VERSIONS_DONE"
