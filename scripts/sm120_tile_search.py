#!/usr/bin/env python
"""
Utility for probing SM120 CUTLASS shared-memory usage.

The script iteratively instantiates the FlashMLA forward/backward kernel
templates with different tile shapes, compiles a tiny NVCC harness, and
reports the required shared-memory footprint.  It keeps shrinking the tile
dimensions until the shared-memory requirement fits within the SM120 limit
(~99 KB).  This gives quick feedback on which tile shapes are viable before
changing the production kernel traits.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Tuple


CPP_SNIPPET = r"""
#include <iostream>

#define TORCH_CHECK(...) (void)0

namespace at {
struct Tensor {
  int size(int) const { return 0; }
  int size(int, int) const { return 0; }
  int stride(int) const { return 1; }
  Tensor slice(int, int, int) const { return Tensor(); }
  Tensor select(int, int) const { return Tensor(); }
  Tensor contiguous() const { return Tensor(); }
  Tensor to(int) const { return Tensor(); }
  Tensor clone() const { return Tensor(); }
  Tensor& zero_() { return *this; }
  Tensor matmul(const Tensor&) const { return Tensor(); }
  Tensor transpose(int, int) const { return Tensor(); }
  Tensor mul_(float) { return Tensor(); }
  Tensor mul(const Tensor&) const { return Tensor(); }
  Tensor exp() const { return Tensor(); }
  Tensor logsumexp(int, bool, std::nullptr_t = nullptr) const { return Tensor(); }
  Tensor logsumexp(const Tensor&, bool = false, std::nullptr_t = nullptr) const { return Tensor(); }
  Tensor masked_fill_(const Tensor&, float) { return Tensor(); }
  Tensor unsqueeze(int) const { return Tensor(); }
  Tensor squeeze(int) const { return Tensor(); }
  Tensor sum(int, bool, int) const { return Tensor(); }
  Tensor sum(const Tensor&, bool = false, std::nullptr_t = nullptr) const { return Tensor(); }
  Tensor diff() const { return Tensor(); }
  Tensor options() const { return Tensor(); }
  Tensor dtype(int) const { return Tensor(); }
  Tensor& copy_(const Tensor&) { return *this; }
  Tensor& copy_(const Tensor&, bool) { return *this; }
  Tensor to(int, bool) const { return Tensor(); }
  Tensor detach() const { return Tensor(); }
  Tensor mul_(const Tensor&) { return Tensor(); }
  Tensor& masked_fill_(float) { return *this; }
  Tensor squeeze() const { return Tensor(); }
  int dim() const { return 0; }
  int device() const { return 0; }
  template <typename T> T* data_ptr() const { return nullptr; }
};
}

namespace c10 { namespace cuda {
struct CUDAStream {};
class CUDAStreamGuard { public: explicit CUDAStreamGuard(const CUDAStream&) {} };
class OptionalCUDAGuard { public: explicit OptionalCUDAGuard(int) {} };
inline CUDAStream getCurrentCUDAStream() { return CUDAStream(); }
}}

#include "sm100/prefill/dense/sm100_kernel_traits.hpp"
#include "sm100/prefill/dense/collective/fmha_fusion.hpp"
#include "sm100/prefill/dense/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "sm100/prefill/dense/collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "sm100/prefill/dense/collective/sm100_fmha_mla_fwd_mainloop_tma_warpspecialized.hpp"
#include "sm100/prefill/dense/kernel/fmha_tile_scheduler.hpp"
#include "sm100/prefill/dense/kernel/fmha_causal_tile_scheduler.hpp"
#include "sm100/prefill/dense/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
#include "sm100/prefill/dense/device/fmha.hpp"
#include "cute/tensor.hpp"

#if defined(PROBE_BWD)

#include "sm100/prefill/dense/device/fmha_device_bwd.hpp"
#include "sm100/prefill/dense/kernel/sm100_fmha_bwd_kernel_tma_warpspecialized.hpp"
#include "sm100/prefill/dense/kernel/sm100_fmha_bwd_mla_kernel_tma_warpspecialized.hpp"

#ifndef BWD_DQ
#define BWD_DQ 128
#endif
#ifndef BWD_DV
#define BWD_DV 128
#endif
#ifndef MLA_DQ
#define MLA_DQ 192
#endif
#ifndef MLA_DV
#define MLA_DV 128
#endif

template <int BM, int BN>
struct CustomBwdTraits : flash::Sm120WorkstationConfig {
  using TileShapeFmhaBwd =
      cute::Shape<cute::Int<BM>, cute::Int<BN>, cute::Int<BWD_DQ>, cute::Int<BWD_DV>>;
  using TileShapeMlaBwd =
      cute::Shape<cute::Int<BM>, cute::Int<BN>, cute::Int<MLA_DQ>, cute::Int<MLA_DV>>;
};

template <class KernelTraits>
struct ProbeBwdMha {
  using Element = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using ProblemShape = cute::tuple<int, int, int, int, cute::tuple<int, int>>;
  using TileShape = typename KernelTraits::TileShapeFmhaBwd;
  using Mask = cutlass::fmha::collective::ResidualMask;

  using Kernel = cutlass::fmha::kernel::Sm100FmhaBwdKernelTmaWarpSpecialized<
      KernelTraits, ProblemShape, Element, ElementAccumulator, TileShape, Mask>;

  static constexpr size_t Shared = Kernel::SharedStorageSize;
};

template <class KernelTraits>
struct ProbeBwdMla {
  using Element = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using ProblemShape = cute::tuple<int, int, int, int, cute::tuple<int, int>>;
  using TileShape = typename KernelTraits::TileShapeMlaBwd;
  using Mask = cutlass::fmha::collective::ResidualMask;

  using Kernel = cutlass::fmha::kernel::Sm100FmhaBwdMlaKernelTmaWarpSpecialized<
      KernelTraits, ProblemShape, Element, ElementAccumulator, TileShape, Mask>;

  static constexpr size_t Shared = Kernel::SharedStorageSize;
};

int main() {
  using Traits = CustomBwdTraits<BWD_M, BWD_N>;
  constexpr size_t shared_mha = ProbeBwdMha<Traits>::Shared;
  constexpr size_t shared_mla = ProbeBwdMla<Traits>::Shared;
  constexpr size_t shared = shared_mha > shared_mla ? shared_mha : shared_mla;
  constexpr size_t limit = flash::Sm120WorkstationConfig::kSharedMemLimit;
  std::cout << shared << std::endl;
  return shared <= limit ? 0 : 2;
}

#else

template <int FM, int FN>
struct CustomFwdTraits : flash::Sm120WorkstationConfig {
  using TileShapeFmhaFwd = cute::Shape<cute::Int<FM>, cute::Int<FN>, cute::Int<128>>;
  using TileShapeMlaFwd = cute::Shape<cute::Int<FM>, cute::Int<FN>, typename flash::Sm120WorkstationConfig::HeadDim>;
};

template <class KernelTraits>
struct ProbeFwd {
  using Element = cutlass::bfloat16_t;
  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;
  using ElementOut = cutlass::bfloat16_t;
  using TileShapeFmhaFwd = typename KernelTraits::TileShapeFmhaFwd;
  using TileShapeMlaFwd = typename KernelTraits::TileShapeMlaFwd;
  using ThreadShape = typename KernelTraits::ThreadShape;

  using StrideQ = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<int, int>, int>>;
  using StrideK = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<cute::_0, int>, int>>;
  using StrideV = StrideK;
  using StrideO = StrideQ;
  using StrideLSE = cute::tuple<cute::_1, cute::tuple<cute::tuple<int, int>, int>>;

  using OrderLoadEpilogue = cute::false_type;
  using TileScheduler = cutlass::fmha::kernel::IndividualTileScheduler;

  using MainloopFmha = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
      Element, ElementAccumulatorQK, ElementAccumulatorPV,
      TileShapeFmhaFwd, StrideQ, StrideK, StrideV,
      cutlass::fmha::collective::ResidualMask,
      ThreadShape, OrderLoadEpilogue>;

  using MainloopMla = cutlass::fmha::collective::Sm100MlaFwdMainloopTmaWarpspecialized<
      Element, ElementAccumulatorQK, ElementAccumulatorPV,
      TileShapeMlaFwd, StrideQ, StrideK, StrideV,
      cutlass::fmha::collective::ResidualMask,
      ThreadShape, OrderLoadEpilogue>;

  using EpilogueFmha = cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
      ElementOut, ElementAccumulatorPV, TileShapeFmhaFwd,
      StrideO, StrideLSE, cute::false_type>;

  using EpilogueMla = cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
      ElementOut, ElementAccumulatorPV, TileShapeMlaFwd,
      StrideO, StrideLSE, cute::false_type>;

  using OperationFmha = cutlass::fmha::device::FMHA<
      cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
          KernelTraits,
          cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>,
          MainloopFmha,
          EpilogueFmha,
          TileScheduler,
          cutlass::fmha::kernel::Sm100FmhaCtxKernelWarpspecializedSchedule>>;

  using OperationMla = cutlass::fmha::device::FMHA<
      cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
          KernelTraits,
          cute::tuple<int, int, cute::tuple<int, int>, cute::tuple<cute::tuple<int, int>, int>>,
          MainloopMla,
          EpilogueMla,
          TileScheduler,
          cutlass::fmha::kernel::Sm100MlaFwdCtxKernelWarpspecializedSchedule>>;

  static constexpr size_t SharedFmha = OperationFmha::Kernel::SharedStorageSize;
  static constexpr size_t SharedMla = OperationMla::Kernel::SharedStorageSize;
};

int main() {
  using Traits = CustomFwdTraits<TILE_M, TILE_N>;
  constexpr size_t shared_fmha = ProbeFwd<Traits>::SharedFmha;
  constexpr size_t shared_mla = ProbeFwd<Traits>::SharedMla;
  constexpr size_t shared = shared_fmha > shared_mla ? shared_fmha : shared_mla;
  constexpr size_t limit = flash::Sm120WorkstationConfig::kSharedMemLimit;
  std::cout << shared << std::endl;
  return shared <= limit ? 0 : 2;
}

#endif

"""


def find_nvcc() -> Path:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("nvcc not found in PATH. Please ensure CUDA is installed.")
    return Path(nvcc)


def gather_include_dirs(repo_root: Path) -> Iterable[Path]:
    includes = [
        repo_root / "csrc",
        repo_root / "csrc" / "sm90",
        repo_root / "csrc" / "cutlass" / "include",
        repo_root / "csrc" / "cutlass" / "tools" / "util" / "include",
    ]

    import sysconfig

    python_include = Path(sysconfig.get_paths()["include"])
    includes.append(python_include)
    plat_include = sysconfig.get_paths().get("platinclude")
    if plat_include:
        includes.append(Path(plat_include))

    try:
        import torch  # type: ignore

        torch_inc = Path(torch.__file__).resolve().parent / "include"
        includes.append(torch_inc)
        includes.append(torch_inc / "torch" / "csrc" / "api" / "include")
        from torch.utils.cpp_extension import CUDA_HOME  # type: ignore

        if CUDA_HOME:
            includes.append(Path(CUDA_HOME) / "include")
    except Exception as exc:
        raise RuntimeError("Unable to import torch to determine include paths.") from exc

    return includes


def find_msbuild_host() -> Optional[Path]:
    if os.name != "nt":
        return None
    cl_path = shutil.which("cl.exe")
    if cl_path:
        return Path(cl_path).resolve()
    # Fallback to common VS Build Tools location
    default = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC")
    if default.exists():
        versions = sorted(default.glob("*"), reverse=True)
        for version_dir in versions:
            candidate = version_dir / "bin" / "Hostx64" / "x64" / "cl.exe"
            if candidate.exists():
                return candidate
    return None


def nvcc_compile(
    cpp_path: Path,
    output_path: Path,
    nvcc_path: Path,
    include_dirs: Iterable[Path],
    macros: Iterable[str],
    cl_path: Optional[Path],
) -> Tuple[bool, str]:
    cmd = [
        str(nvcc_path),
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-lineinfo",
    ]

    is_windows = os.name == "nt"
    if is_windows:
        cmd += ["-Xcompiler", "/MD", "-Xcompiler", "/Zc:__cplusplus", "-Xcompiler", "/EHsc"]
        cmd += ["-include", "msvc_compat.h"]
        if cl_path:
            cmd += ["-ccbin", str(cl_path.parent)]

    for inc in include_dirs:
        cmd.append(f"-I{inc}")

    cmd += [
        "-DFLASH_MLA_DISABLE_SM90",
        "-DFLASH_MLA_DISABLE_SM100",
        "-DFLASH_MLA_BUILD_SM120",
        "-DFLASH_MLA_DISABLE_SMEM_ASSERT",
    ]
    cmd.extend(macros)
    cmd += [str(cpp_path), "-o", str(output_path)]

    try:
        print("[NVCC]", " ".join(cmd))
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return True, result.stdout
    except subprocess.CalledProcessError as exc:
        return False, exc.stdout


def run_probe(exe_path: Path) -> Tuple[Optional[int], str, bool]:
    try:
        result = subprocess.run(
            [str(exe_path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        return None, str(exc), False

    output = result.stdout.strip()
    shared_bytes: Optional[int] = None
    try:
        shared_bytes = int(output.splitlines()[-1])
    except ValueError:
        pass
    return shared_bytes, output, result.returncode == 0


def search_forward(
    repo_root: Path,
    nvcc_path: Path,
    include_dirs: Iterable[Path],
    cl_path: Optional[Path],
    m_values: Iterable[int],
    n_values: Iterable[int],
) -> None:
    print("=== Forward kernel shared-memory sweep ===")
    for m in m_values:
        for n in n_values:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                cpp_path = tmp / "probe.cu"
                exe_path = tmp / ("probe.exe" if os.name == "nt" else "probe")
                cpp_path.write_text(CPP_SNIPPET)
                ok, log = nvcc_compile(
                    cpp_path,
                    exe_path,
                    nvcc_path,
                    include_dirs,
                    [
                        "-DPROBE_FWD=1",
                        f"-DTILE_M={m}",
                        f"-DTILE_N={n}",
                    ],
                    cl_path,
                )
                if not ok:
                    print(f"[FWD] M={m:3d}, N={n:3d} -> compile failed")
                    print(log)
                    continue
                shared, output, within = run_probe(exe_path)
                if shared is None:
                    print(f"[FWD] M={m:3d}, N={n:3d} -> runtime parse error")
                    print(output)
                    continue
                status = "OK " if within else "OVER"
                print(f"[FWD] M={m:3d}, N={n:3d} -> {shared:6d} bytes ({status})")
                if within:
                    return
    print("No forward configuration satisfied the SM120 shared-memory budget.")


def search_backward(
    repo_root: Path,
    nvcc_path: Path,
    include_dirs: Iterable[Path],
    cl_path: Optional[Path],
    m_values: Iterable[int],
    n_values: Iterable[int],
    dq_values: Iterable[int],
    dv_values: Iterable[int],
    mla_dq_values: Iterable[int],
    mla_dv_values: Iterable[int],
) -> None:
    print("\n=== Backward kernel shared-memory sweep ===")
    for m in m_values:
        for n in n_values:
            for dq in dq_values:
                for dv in dv_values:
                    for mla_dq in mla_dq_values:
                        for mla_dv in mla_dv_values:
                            with tempfile.TemporaryDirectory() as tmpdir:
                                tmp = Path(tmpdir)
                                cpp_path = tmp / "probe.cu"
                                exe_path = tmp / ("probe.exe" if os.name == "nt" else "probe")
                                cpp_path.write_text(CPP_SNIPPET)
                                ok, log = nvcc_compile(
                                    cpp_path,
                                    exe_path,
                                    nvcc_path,
                                    include_dirs,
                                    [
                                        "-DPROBE_BWD=1",
                                        f"-DBWD_M={m}",
                                        f"-DBWD_N={n}",
                                        f"-DBWD_DQ={dq}",
                                        f"-DBWD_DV={dv}",
                                        f"-DMLA_DQ={mla_dq}",
                                        f"-DMLA_DV={mla_dv}",
                                    ],
                                    cl_path,
                                )
                                if not ok:
                                    print(
                                        f"[BWD] M={m:3d}, N={n:3d}, "
                                        f"DQ={dq:3d}, DV={dv:3d}, MLA_DQ={mla_dq:3d}, MLA_DV={mla_dv:3d} "
                                        "-> compile failed"
                                    )
                                    print(log)
                                    continue
                                shared, output, within = run_probe(exe_path)
                                if shared is None:
                                    print(
                                        f"[BWD] M={m:3d}, N={n:3d}, "
                                        f"DQ={dq:3d}, DV={dv:3d}, MLA_DQ={mla_dq:3d}, MLA_DV={mla_dv:3d} "
                                        "-> runtime parse error"
                                    )
                                    print(output)
                                    continue
                                status = "OK " if within else "OVER"
                                print(
                                    f"[BWD] M={m:3d}, N={n:3d}, "
                                    f"DQ={dq:3d}, DV={dv:3d}, MLA_DQ={mla_dq:3d}, MLA_DV={mla_dv:3d} "
                                    f"-> {shared:6d} bytes ({status})"
                                )
                                if within:
                                    return
    print("No backward configuration satisfied the SM120 shared-memory budget.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe SM120 kernel shared-memory usage.")
    parser.add_argument(
        "--forward-m",
        nargs="+",
        type=int,
        default=[64],
        help="Candidate forward M tile sizes (default tuned for SM120).",
    )
    parser.add_argument(
        "--forward-n",
        nargs="+",
        type=int,
        default=[64],
        help="Candidate forward N tile sizes.",
    )
    parser.add_argument(
        "--backward-m",
        nargs="+",
        type=int,
        default=[64],
        help="Candidate backward M tile sizes.",
    )
    parser.add_argument(
        "--backward-n",
        nargs="+",
        type=int,
        default=[64],
        help="Candidate backward N tile sizes (SM100 kernels require 128; SM120 allows 64).",
    )
    parser.add_argument(
        "--backward-dq",
        nargs="+",
        type=int,
        default=[64],
        help="Candidate backward TileShapeDQK sizes.",
    )
    parser.add_argument(
        "--backward-dv",
        nargs="+",
        type=int,
        default=[64],
        help="Candidate backward TileShapeDVO sizes.",
    )
    parser.add_argument(
        "--backward-mla-dq",
        nargs="+",
        type=int,
        default=[64],
        help="Candidate MLA backward TileShapeDQK sizes.",
    )
    parser.add_argument(
        "--backward-mla-dv",
        nargs="+",
        type=int,
        default=[64],
        help="Candidate MLA backward TileShapeDVO sizes.",
    )
    parser.add_argument(
        "--skip-backward",
        action="store_true",
        help="Only search forward kernel configurations.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    nvcc_path = find_nvcc()
    include_dirs = list(gather_include_dirs(repo_root))
    cl_path = find_msbuild_host()
    if os.name == "nt" and cl_path is None:
        raise RuntimeError(
            "Unable to locate cl.exe. Please open a Visual Studio Build Tools prompt "
            "or install the MSVC toolchain."
        )

    search_forward(repo_root, nvcc_path, include_dirs, cl_path, args.forward_m, args.forward_n)
    if not args.skip_backward:
        search_backward(
            repo_root,
            nvcc_path,
            include_dirs,
            cl_path,
            args.backward_m,
            args.backward_n,
            args.backward_dq,
            args.backward_dv,
            args.backward_mla_dq,
            args.backward_mla_dv,
        )


if __name__ == "__main__":
    main()
