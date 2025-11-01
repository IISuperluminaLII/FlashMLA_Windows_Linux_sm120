import os
import subprocess
from datetime import datetime
from pathlib import Path

from setuptools import setup, find_packages

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_WINDOWS,
    CUDA_HOME,
)

SUPPORTED_ARCHES = {"sm100", "sm120"}


def is_flag_set(flag: str) -> bool:
    return os.getenv(flag, "FALSE").lower() in {"true", "1", "y", "yes"}


def resolve_target_arch() -> str:
    target = os.getenv("FLASH_MLA_ARCH", "sm100").lower()
    if target not in SUPPORTED_ARCHES:
        raise ValueError(
            f"Unsupported FLASH_MLA_ARCH='{target}'. Expected one of {sorted(SUPPORTED_ARCHES)}."
        )
    print(f"[build] Targeting FlashMLA variant: {target}")
    return target


def get_features_args(for_msvc: bool = False, extra_defines=None):
    """Gather feature flags for either MSVC (/D) or NVCC (-D)."""
    prefix = "/D" if for_msvc else "-D"
    defines = []

    def append_unique(define: str):
        if define not in defines:
            defines.append(define)

    if is_flag_set("FLASH_MLA_DISABLE_FP16"):
        append_unique("FLASH_MLA_DISABLE_FP16")
    if is_flag_set("FLASH_MLA_DISABLE_SM90"):
        append_unique("FLASH_MLA_DISABLE_SM90")
    if is_flag_set("FLASH_MLA_DISABLE_SM100"):
        append_unique("FLASH_MLA_DISABLE_SM100")
    if is_flag_set("FLASH_MLA_SM120_DISABLE_BWD"):
        append_unique("FLASH_MLA_SM120_DISABLE_BWD")
    if is_flag_set("FLASH_MLA_FORCE_FALLBACK"):
        append_unique("FLASH_MLA_FORCE_FALLBACK")

    for define in extra_defines or []:
        append_unique(define)

    return [f"{prefix}{define}" for define in defines]


def query_nvcc_version():
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
    nvcc_path = Path(CUDA_HOME) / "bin" / "nvcc"
    output = subprocess.check_output([str(nvcc_path), "--version"], stderr=subprocess.STDOUT)
    text = output.decode("utf-8")
    version_token = text.split("release ")[1].split(",")[0].strip()
    major, minor = map(int, version_token.split("."))
    print(f"[build] Compiling with NVCC {major}.{minor}")
    return major, minor


def get_arch_flags(target_arch: str):
    major, minor = query_nvcc_version()

    if target_arch == "sm100":
        if major < 12 or (major == 12 and minor <= 8):
            raise RuntimeError(
                "sm100 compilation for FlashMLA requires NVCC 12.9 or higher. "
                "Set FLASH_MLA_ARCH=sm120 to build the workstation variant or upgrade CUDA."
            )
        return [
            "-gencode",
            "arch=compute_100a,code=sm_100a",
            "-gencode",
            "arch=compute_100a,code=compute_100a",
        ]

    if target_arch == "sm120":
        return [
            "-gencode",
            "arch=compute_120,code=sm_120",
            "-gencode",
            "arch=compute_120,code=compute_120",
        ]

    raise ValueError(f"Unhandled arch {target_arch}")


def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]


def get_nvcc_cxx_flags():
    if IS_WINDOWS:
        return ["-Xcompiler", "/Zc:__cplusplus"]
    return []


VARIANTS = {
    "sm100": {
        "extension": "flash_mla.cuda_sm100",
        "sources": [
            "csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu",
            "csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu",
        ],
        "defines": ["FLASH_MLA_BUILD_SM100"],
        "feature_defines": ["FLASH_MLA_DISABLE_SM90"],
    },
    "sm120": {
        "extension": "flash_mla.cuda_sm120",
        "sources": [
            "csrc/sm120/prefill/dense/fmha_cutlass_fwd_sm120.cu",
            "csrc/sm120/prefill/dense/fmha_cutlass_bwd_sm120.cu",
        ],
        # Temporarily force fallback for forward while backward TMEM/TMA is validated.
        "defines": [
            "FLASH_MLA_BUILD_SM120",
            
            
            # Limit SM120 BWD surface during iteration
            
            
        ],
        "feature_defines": ["FLASH_MLA_DISABLE_SM90", "FLASH_MLA_DISABLE_SM100"],
    },
}


this_dir = Path(__file__).resolve().parent
target_arch = resolve_target_arch()
variant = VARIANTS[target_arch]

base_sources = [
    "csrc/pybind.cpp",
    "csrc/smxx/get_mla_metadata.cu",
    "csrc/smxx/mla_combine.cu",
]

sources = base_sources + variant["sources"]

if IS_WINDOWS:
    cxx_args = [
        "/O2",
        "/std:c++17",
        "/Zc:__cplusplus",
        "/EHsc",
        "/permissive-",
        "/DNOMINMAX",
        "/DWIN32_LEAN_AND_MEAN",
        "/D_HAS_EXCEPTIONS=1",
        "/utf-8",
        "/DNDEBUG",
        "/W0",
        "/FImsvc_compat.h",
    ]
else:
    cxx_args = ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"]

extra_defines = variant["feature_defines"] + variant["defines"]

nvcc_common_flags = [
    "-include",
    "msvc_compat.h" if IS_WINDOWS else "cuda_runtime.h",
    "-O3",
    "-std=c++17",
    "-DNDEBUG",
    "-D_USE_MATH_DEFINES",
    "-Wno-deprecated-declarations",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--ptxas-options=-v,--register-usage-level=10",
]

if IS_WINDOWS:
    nvcc_common_flags.extend(["-Xcompiler", "/Zc:__cplusplus", "-Xcompiler", "/permissive-"])

ext_modules = [
    CUDAExtension(
        name=variant["extension"],
        sources=sources,
        extra_compile_args={
            "cxx": cxx_args + get_features_args(for_msvc=IS_WINDOWS, extra_defines=extra_defines),
            "nvcc": nvcc_common_flags
            + get_nvcc_cxx_flags()
            + get_features_args(for_msvc=False, extra_defines=extra_defines)
            + get_arch_flags(target_arch)
            + get_nvcc_thread_args(),
        },
        include_dirs=[
            this_dir / "csrc",
            this_dir / "csrc" / "sm90",
            this_dir / "csrc" / "cutlass" / "include",
            this_dir / "csrc" / "cutlass" / "tools" / "util" / "include",
            
        ],
    )
]

try:
    rev = "+" + subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").rstrip()
except Exception:
    now = datetime.now()
    rev = "+" + now.strftime("%Y-%m-%d-%H-%M-%S")

setup(
    name="flash_mla",
    version="1.0.0" + rev,
    packages=find_packages(include=["flash_mla"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
