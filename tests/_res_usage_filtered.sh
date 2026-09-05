#!/usr/bin/env bash
# Usage: bash tests/_res_usage_filtered.sh <so> <kernel-regex>   -> name-tag REG STACK SHARED per kernel
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
"$CUDA_HOME/bin/cuobjdump" --dump-resource-usage "$1" 2>/dev/null | grep -A1 -E "$2" | grep -oE 'ILi[0-9]+ELi128ELb[01]ELi[0-9]+E|REG:[0-9]+|STACK:[0-9]+|SHARED:[0-9]+' | paste - - - -
