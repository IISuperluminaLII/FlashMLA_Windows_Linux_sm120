#!/usr/bin/env bash
echo "--- chain_final8.log:"; head -c 1200 ~/flashmla_snap/chain_final8.log; echo; wc -l ~/flashmla_snap/chain_final8.log
echo "--- procs:"; pgrep -af 'snap_chain_final_cert|run_battery_both|build_wsl_sm120|ninja|nvcc|cicc|ptxas' | cut -c1-120 | head -8
echo "--- main build log lines:"; wc -l /mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA/build_main_final.log; grep -cE 'nvcc' /mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA/build_main_final.log
ls -la --time-style=+%H:%M:%S /mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA/flash_mla/ | grep -E 'so|bak'
date +%H:%M:%S
