"""Run pytest with a watchdog: if anything hangs, dump every thread's stack and stop.

    python scripts/run_tests.py            # writes outputs/pytest_mac.log
"""
import faulthandler
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
log = open(ROOT / "outputs" / "pytest_mac.log", "w", buffering=1)
faulthandler.dump_traceback_later(180, exit=True, file=log)   # 3 min without finishing → stack dump
start = time.time()
sys.stdout = sys.stderr = log
import pytest  # noqa: E402

code = pytest.main(["-q", "-x", "--durations=8", "-p", "no:cacheprovider", str(ROOT / "tests")])
print(f"exit code {code} in {time.time() - start:.0f}s")
sys.exit(code)
