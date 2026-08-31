#!/usr/bin/env bash
set -e
# Canonical WSL build for the sm120 extension. Machine-agnostic: repo root,
# python and CUDA_HOME resolve via env_sm120.sh (override with FLASHMLA_PYTHON /
# FLASHMLA_CONDA_ENV / CUDA_HOME).
source "$(dirname "${BASH_SOURCE[0]}")/env_sm120.sh"
cd "$FMLA_ROOT"
export MAX_JOBS="${MAX_JOBS:-4}"   # ninja TU concurrency (each nvcc also runs --threads 32)
export FLASH_MLA_ARCH=sm120

# The .so name is python-ABI-derived, never hardcoded (cp312 today, anything later).
SO="flash_mla/cuda_sm120$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

# The extension MUST build with an nvcc matching torch's bundled CUDA runtime
# major.minor (a mismatched .so misreads the cudaDeviceProp ABI and links the
# wrong libtorch ABI). Verified live, warned loudly, never assumed.
TORCH_CUDA="$("$PY" -c 'import torch; print(torch.version.cuda)' 2>/dev/null || true)"
NVCC_VER="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\),.*/\1/p')"
if [ -n "$TORCH_CUDA" ] && [ -n "$NVCC_VER" ] && [ "$NVCC_VER" != "$TORCH_CUDA" ]; then
    echo "[WARN] nvcc $NVCC_VER != torch bundled CUDA $TORCH_CUDA -- ABI skew risk; set CUDA_HOME to the matching toolkit"
fi

# Toolchain stamp: objects compiled under a DIFFERENT nvcc/torch must never be
# reused (incremental builds skip recompilation by mtime and would silently link
# stale-ABI objects into a "fresh" .so). Wipe build/ on any toolchain change.
STAMP_FILE=build/.toolchain_stamp
STAMP="$(command -v nvcc) $(nvcc --version | tail -1) torch=$("$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>/dev/null)"
if [ ! -f "$STAMP_FILE" ] || [ "$(cat "$STAMP_FILE" 2>/dev/null)" != "$STAMP" ]; then
  echo "TOOLCHAIN_CHANGED -> clean rebuild"
  rm -rf build
  mkdir -p build
  printf "%s" "$STAMP" > "$STAMP_FILE"
fi

# Rename the (possibly live-mapped) old .so aside BEFORE building. rename(2) keeps the
# inode alive for any running training process; setuptools then writes a fresh file.
if [ -f "$SO" ]; then
  mv -f "$SO" "${SO}.prebuild_bak"
  echo "MOVED_OLD_SO_ASIDE -> ${SO}.prebuild_bak"
fi
echo "BUILD_START $("$PY" --version) nvcc=$(command -v nvcc)"
"$PY" setup.py build_ext --inplace

# Closed-loop staleness guard: the produced .so must postdate EVERY source file
# (csrc incl. vendored cutlass, setup.py). A dependency-tracking hole upstream
# must fail the build loudly here, never ship a stale binary as "done".
STALE="$(find csrc setup.py -newer "$SO" -print -quit 2>/dev/null)"
if [ -n "$STALE" ]; then
  echo "BUILD_STALE: source newer than built .so: $STALE"
  exit 1
fi
echo "WSL_BUILD_DONE"
ls -la "$SO"
