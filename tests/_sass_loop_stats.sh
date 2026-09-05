#!/usr/bin/env bash
# Usage: bash tests/_sass_loop_stats.sh <mangled-name-regex> [so]   (default so = the snapshot's)
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
SO="${2:-$HOME/flashmla_snap/flash_mla/cuda_sm120$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')}"
export PATH="$CUDA_HOME/bin:$PATH"
"$PY" "$(dirname "${BASH_SOURCE[0]}")/_sass_loop_stats.py" "$SO" "$1"
