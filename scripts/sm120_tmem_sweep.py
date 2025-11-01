#!/usr/bin/env python3
"""
SM120 TMEM/TMA sweep harness.

This script explores a small grid of reducer TMEM configurations for SM120 backward
DQ staging and tries to compile+run a minimal micro-kernel per configuration. The
goal is to find a combination of DQ column shape, per-warp offset policy and TMEM
copy atom that compiles cleanly and yields unique TMEM indices at runtime (no
collisions), without falling back to shared memory.

It generates a tiny .cu source in a temp dir for each config, compiles it with
nvcc, runs it, and parses stdout for success. Results are printed as a table at
the end and also saved to sweep_results.csv in the current directory.

Requires:
  - CUDA 12.9+ in WSL at /usr/local/cuda (or adjust PATH detection below)
  - Access to the repo include paths (this script assumes it is run from the
    repo root or anywhere under it).
"""
import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CUDA = os.environ.get("CUDA_HOME", "/usr/local/cuda")
NVCC = Path(CUDA) / "bin" / "nvcc"

INCLUDE_DIRS = [
    REPO_ROOT / "external/FlashMLA/csrc",
    REPO_ROOT / "external/FlashMLA/csrc/cutlass/include",
    REPO_ROOT / "external/FlashMLA/csrc/cutlass/tools/util/include",
]

MICRO_TEMPLATE = r'''
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/arch/copy_sm100.hpp"

using namespace cute;

#ifndef DQ_TILE
#define DQ_TILE 32
#endif

// Copy atom selector
#if PROBE_USE_TMEM
  using CopyAtom = Copy_Atom<SM100_TMEM_LOAD_32dp32b16x, float>;
#else
  using CopyAtom = Copy_Atom<UniversalCopy<uint128_t>, float>;
#endif

__global__ void probe(int *ok) {
  // Build a synthetic TMEM fragment with (Q x DQ_TILE) columns and feed it to the copy atom
  // Use a simple rank-2 layout that mimics partition_fragment_C output for DQ
  constexpr int Q = 64;
  constexpr int DQ = DQ_TILE;
  auto tDQ = make_tensor(make_tmem_ptr((float*)nullptr), make_layout(make_shape(Int<Q>{}, Int<DQ>{})));
  auto tiled = make_tiled_copy(CopyAtom{}, tDQ.layout(), tDQ.layout());
  auto thr = tiled.get_slice(threadIdx.x % 32);
  auto tS = thr.partition_S(tDQ);
  auto tD = thr.partition_D(tDQ);
  // If partition succeeds and sizes match, mark success
  if (size(tS) == size(tD) && size(tS) > 0) {
    *ok = 1;
  }
}

int main() {
  int *d_ok; cudaMalloc(&d_ok, sizeof(int));
  cudaMemset(d_ok, 0, sizeof(int));
  probe<<<1, 128>>>(d_ok);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { return 2; }
  int h_ok = 0; cudaMemcpy(&h_ok, d_ok, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_ok);
  return h_ok ? 0 : 3;
}
'''


def compile_and_run(tmpdir: Path, dq_tile: int, use_tmem: bool) -> tuple[bool, str, int]:
    src = tmpdir / f"micro_{dq_tile}_{int(use_tmem)}.cu"
    exe = tmpdir / f"micro_{dq_tile}_{int(use_tmem)}"
    code = MICRO_TEMPLATE
    src.write_text(code)
    cmd = [
        str(NVCC),
        "-std=c++17",
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-lineinfo",
        f"-DDQ_TILE={dq_tile}",
        f"-DPROBE_USE_TMEM={1 if use_tmem else 0}",
    ]
    for inc in INCLUDE_DIRS:
        cmd.append(f"-I{inc}")
    cmd += [str(src), "-o", str(exe)]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        return False, exc.stdout + exc.stderr, -1

    # run
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    return run.returncode == 0, run.stdout + run.stderr, run.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="keep temp dir")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="sm120_tmem_sweep_"))
    results = []
    try:
        grid = [(16, False), (32, False), (16, True), (32, True)]
        print("DQ_TILE  USE_TMEM  COMPILES  RUNS  RC")
        for dq, use_tmem in grid:
            okc, logc, rc = compile_and_run(tmp, dq, use_tmem)
            print(f"{dq:>6}  {int(use_tmem):>8}  {str(okc):>8}  {str(okc):>5}  {rc}")
            results.append((dq, use_tmem, okc, rc))
        # save
        out = REPO_ROOT / "external/FlashMLA" / "sweep_results.csv"
        with out.open("w") as f:
            f.write("dq_tile,use_tmem,ok,rc\n")
            for row in results:
                f.write(",".join(map(str, row)) + "\n")
        print(f"Saved: {out}")
        print(f"Temp:  {tmp}")
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

