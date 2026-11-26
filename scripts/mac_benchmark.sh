#!/bin/bash
# Mac (Apple Silicon): install the package, run the tests, benchmark the 13 configurations on MPS.
# Usage from the repository root:  bash scripts/mac_benchmark.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p outputs
export PYTHONPATH="$PWD/src"
.venv/bin/python -m pytest -q 2>&1 | tail -1 | tee outputs/pytest_mac.log
caffeinate -i .venv/bin/python scripts/benchmark_device.py 2>&1 | tee outputs/benchmark.log
echo "DONE: results in outputs/benchmark.json"
