import os
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

from setuptools import setup, find_packages

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_WINDOWS,
    CUDA_HOME,
)


def apply_patches():
    """Apply patches to CUTLASS and other dependencies at build time.

    SM120 workstation GPUs (RTX 6000 Pro, RTX 50 series) do not have TMEM
    (Tensor Memory), which is a datacenter-only feature (SM100/101/103).
    CUTLASS 4.2.1 crashes when TMEM is used on SM120, so we patch it to
    no-op the TMEM allocator calls.
    """
    this_dir = Path(__file__).resolve().parent
    patches_dir = this_dir / "patches"

    if not patches_dir.exists():
        return

    for patch_file in patches_dir.glob("*.patch"):
        print(f"[build] Checking patch: {patch_file.name}")

        # Parse patch to get target file and apply in-memory replacement
        if patch_file.name == "cutlass_tmem_sm120_fallback.patch":
            target_file = this_dir / "csrc" / "cutlass" / "include" / "cute" / "arch" / "tmem_allocator_sm100.hpp"
            if not target_file.exists():
                print(f"[build] WARNING: Patch target not found: {target_file}")
                continue

            content = target_file.read_text(encoding="utf-8")

            # Check if already patched
            if "SM120 workstation GPUs do not have TMEM" in content:
                print(f"[build] Patch already applied: {patch_file.name}")
                continue

            # Apply the patch via string replacement
            # Replace CUTE_INVALID_CONTROL_PATH in Allocator1Sm::allocate
            old_allocate1 = 'CUTE_INVALID_CONTROL_PATH("Attempting to use TMEM allocation PTX without CUTE_ARCH_TCGEN05_TMEM_ENABLED");'
            new_allocate1 = '''// SM120 workstation GPUs do not have TMEM (only SM100/101/103 datacenter)
    (void)num_columns;
    if (dst_ptr) *dst_ptr = 0;'''

            # First occurrence is in allocate(), replace it
            content = content.replace(old_allocate1, new_allocate1, 1)

            # For free() and release_allocation_lock(), replace remaining occurrences
            old_free = 'CUTE_INVALID_CONTROL_PATH("Attempting to use TMEM allocation PTX without CUTE_ARCH_TCGEN05_TMEM_ENABLED");'

            # Count remaining occurrences
            if old_free in content:
                # Replace for free() - needs (void) casts
                new_free = '''// SM120 workstation GPUs do not have TMEM - no-op
    (void)tmem_ptr;
    (void)num_columns;'''
                content = content.replace(old_free, new_free, 1)

            if old_free in content:
                # Replace for release_allocation_lock() - simple no-op
                new_release = "// SM120 workstation GPUs do not have TMEM - no-op"
                content = content.replace(old_free, new_release, 1)

            # Repeat for Allocator2Sm class (same pattern)
            if old_allocate1 in content:
                content = content.replace(old_allocate1, new_allocate1, 1)
            if old_free in content:
                content = content.replace(old_free, new_free, 1)
            if old_free in content:
                content = content.replace(old_free, new_release, 1)

            target_file.write_text(content, encoding="utf-8")
            print(f"[build] Applied patch: {patch_file.name}")


# Apply patches before build starts
apply_patches()


SUPPORTED_ARCHES = {"sm100", "sm120"}


def is_flag_set(flag: str) -> bool:
    return os.getenv(flag, "FALSE").lower() in {"true", "1", "y", "yes"}


def resolve_target_arch() -> str:
    target = os.getenv("FLASH_MLA_ARCH", "sm100").strip().lower()
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
    if is_flag_set("FLASH_MLA_FORCE_BWD_FALLBACK"):
        append_unique("FLASH_MLA_FORCE_BWD_FALLBACK")

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
    nvcc_threads = (os.getenv("NVCC_THREADS") or "32").strip()
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
            "csrc/sm120/prefill/sparse/fwd.cu",
            "csrc/sm120/decode/dense/splitkv_mla.cu",
        ],
        "defines": [
            "FLASH_MLA_BUILD_SM120",
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
