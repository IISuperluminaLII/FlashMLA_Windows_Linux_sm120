#!/usr/bin/env bash
# Shared environment resolver for every FlashMLA sm120 build/test/bench script.
# SOURCE this file (do not execute it). Safe under `set -u` and `set -e`.
#
# Inputs (all optional, all overridable per-invocation):
#   FLASHMLA_PYTHON     absolute python interpreter to use (highest priority)
#   FLASHMLA_CONDA_ENV  conda env name to resolve (default: 150BLLM)
#   CUDA_HOME           CUDA toolkit root (default: newest matching preference list)
# Outputs:
#   FMLA_ROOT  FlashMLA repo root, derived from THIS file's location (works from
#              any clone/mount -- never a hardcoded machine path)
#   PY         resolved python interpreter
#   CUDA_HOME  exported toolkit root
#   PATH       prefixed with the env's bin (REQUIRED: torch BuildExtension only
#              tracks HEADER dependencies when the `ninja` executable is on PATH;
#              without it .cuh edits silently reuse stale objects and relink the
#              OLD .so) and with $CUDA_HOME/bin.
#
# Resolution is conda-first (project law: python is always the conda env, never
# system python). Fallbacks WARN loudly instead of erroring so the scripts stay
# machine-agnostic; a wrong interpreter fails visibly at the torch import.

FMLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_fmla_env="${FLASHMLA_CONDA_ENV:-150BLLM}"
if [ -n "${FLASHMLA_PYTHON:-}" ]; then
    PY="$FLASHMLA_PYTHON"
elif [ -x "$HOME/miniconda3/envs/$_fmla_env/bin/python" ]; then
    PY="$HOME/miniconda3/envs/$_fmla_env/bin/python"
elif [ -x "$HOME/anaconda3/envs/$_fmla_env/bin/python" ]; then
    PY="$HOME/anaconda3/envs/$_fmla_env/bin/python"
elif [ -n "${CONDA_EXE:-}" ] && [ -x "$(dirname "$(dirname "$CONDA_EXE")")/envs/$_fmla_env/bin/python" ]; then
    PY="$(dirname "$(dirname "$CONDA_EXE")")/envs/$_fmla_env/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PY="$CONDA_PREFIX/bin/python"
    echo "[WARN] conda env '$_fmla_env' not found; using ACTIVE conda env: $CONDA_PREFIX"
else
    PY="$(command -v python3 || command -v python || true)"
    echo "[WARN] no conda env '$_fmla_env' and no active conda; falling back to: ${PY:-<none>}"
fi
if [ -z "${PY:-}" ] || [ ! -x "$PY" ]; then
    echo "[FAILED] no usable python interpreter found (set FLASHMLA_PYTHON)"
    return 1 2>/dev/null || exit 1
fi

# CUDA toolkit: the extension must build with an nvcc whose major.minor matches
# torch's bundled CUDA runtime (a 12.9-built .so misreads the cu13 cudaDeviceProp
# ABI). Preference order favors the exact-match toolkit over the rolling symlink.
if [ -z "${CUDA_HOME:-}" ]; then
    for _c in /usr/local/cuda-13.0 /usr/local/cuda; do
        if [ -d "$_c" ]; then CUDA_HOME="$_c"; break; fi
    done
fi
export CUDA_HOME="${CUDA_HOME:-}"
if [ -z "$CUDA_HOME" ]; then
    echo "[WARN] no CUDA toolkit found under /usr/local (set CUDA_HOME for builds)"
fi

export PATH="$(dirname "$PY")${CUDA_HOME:+:$CUDA_HOME/bin}:$PATH"
