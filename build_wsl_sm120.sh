#!/usr/bin/env bash
set -e
cd /mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
# Live env moved to torch 2.13.0+cu130 (bundled CUDA runtime 13.0) -> the extension
# MUST build with nvcc 13.0 (exact major.minor match; 12.9-built .so misreads the
# cu13 cudaDeviceProp ABI and links the old libtorch ABI).
export CUDA_HOME=/usr/local/cuda-13.0
# Conda env bin FIRST: torch's BuildExtension only tracks HEADER dependencies
# when the `ninja` executable is on PATH (pip installs it into the env bin, but
# these scripts never activate the env). Without it torch falls back to
# distutils' mtime-on-.cu-only check, so .cuh edits silently reuse stale
# objects and relink/copy the OLD .so (observed: a "successful" rebuild shipped
# a binary predating the fix it was meant to contain).
export PATH="/home/shashankm/miniconda3/envs/150BLLM/bin:/usr/local/cuda-13.0/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-4}"   # ninja TU concurrency (each nvcc also runs --threads 32)
export FLASH_MLA_ARCH=sm120
PY=/home/shashankm/miniconda3/envs/150BLLM/bin/python
SO=flash_mla/cuda_sm120.cpython-312-x86_64-linux-gnu.so

# Toolchain stamp: objects compiled under a DIFFERENT nvcc/torch must never be
# reused (incremental builds skip recompilation by mtime and would silently link
# stale-ABI objects into a "fresh" .so). Wipe build/ on any toolchain change.
STAMP_FILE=build/.toolchain_stamp
STAMP="$(command -v nvcc) $(nvcc --version | tail -1) torch=$($PY -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>/dev/null)"
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
echo "BUILD_START $($PY --version) nvcc=$(command -v nvcc)"
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
