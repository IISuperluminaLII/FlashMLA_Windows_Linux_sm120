#!/usr/bin/env bash
# Isolated snapshot build: copy the FlashMLA tree (minus build artefacts) to the WSL native
# disk and build there, so a coordinator can build+measure one edit while agents keep editing
# other TUs in the main tree. Usage: bash tests/_snapshot_build.sh [snapshot_dir]
set -u
SRC="$(cd "$(dirname "$0")/.." && pwd)"
SNAP="${1:-$HOME/flashmla_snap}"
mkdir -p "$SNAP"
rsync -a --delete --exclude build/ --exclude .git/ --exclude '*.so' --exclude '*.prebuild_bak' \
      --exclude __pycache__/ --exclude '*.egg-info/' --exclude audit/ "$SRC/" "$SNAP/"
echo "SNAPSHOT_AT $SNAP ($(date +%H:%M:%S))"
cd "$SNAP"
bash build_wsl_sm120.sh > "$SNAP/build.log" 2>&1
RC=$?
tail -n 5 "$SNAP/build.log"
echo "SNAPSHOT_BUILD_RC=$RC"
ls -la "$SNAP"/flash_mla/*.so 2>/dev/null
