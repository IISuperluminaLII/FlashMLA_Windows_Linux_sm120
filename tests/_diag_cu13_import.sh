#!/usr/bin/env bash
# Diagnose flash_mla import resolution in the cu13 env from multiple cwds.
set -u
PY=/home/shashankm/miniconda3/envs/150BLLM_cu13/bin/python
echo "--- python ---"
"$PY" -c "import sys; print(sys.version.split()[0], sys.executable)"
echo "--- from HOME ---"
cd ~
"$PY" -c "import flash_mla; print('IMPORTED:', flash_mla.__file__)" 2>&1 | tail -2
echo "--- from cu13 repo ---"
cd ~/flashmla_cu13
"$PY" -c "import flash_mla; print('IMPORTED:', flash_mla.__file__)" 2>&1 | tail -2
echo "--- as script-style (sys.path[0] = tests/) ---"
"$PY" tests/_diag_cu13_probe_import.py 2>&1 | tail -2 || true
echo "--- site-packages listing ---"
ls "$("$PY" -c "import site; print(site.getsitepackages()[0])")" | grep -i -E "flash|editable|\.pth" || echo "NO flash/pth entries"
echo "--- PYTHONPATH ---"
echo "PYTHONPATH=${PYTHONPATH:-<unset>}"
echo "--- home dir flash_mla? ---"
ls -d ~/flash_mla 2>/dev/null || echo "no ~/flash_mla"
echo "DIAG_DONE"
