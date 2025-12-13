import os
import subprocess
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['FLASH_MLA_ARCH'] = 'sm120'
os.environ['MAX_JOBS'] = '1'

result = subprocess.run(
    [sys.executable, 'setup.py', 'build_ext', '--inplace'],
    capture_output=False
)
sys.exit(result.returncode)
