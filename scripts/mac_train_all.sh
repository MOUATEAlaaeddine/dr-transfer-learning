#!/bin/bash
# Mac (Apple Silicon): tests → data preparation → full 65-run grid in the background.
# Usage from the repository root:  bash scripts/mac_train_all.sh
# Follow progress: tail -f outputs/grid.log   ·   status: outputs/grid_status.json
set -e
cd "$(dirname "$0")/.."
mkdir -p outputs
PY=.venv/bin/python
$PY -m pip install -q -e .
$PY -m pytest -q 2>&1 | tail -1 | tee outputs/pytest_mac.log
if [ ! -f data/splits/folds.csv ]; then
  $PY scripts/prepare_data.py --raw data/raw 2>&1 | tee outputs/prepare_data.log
fi
# 6 loader workers on the M5; the grid is fold-major and resumable (re-run this script to continue).
nohup caffeinate -i $PY scripts/run_grid.py --overrides '{"training": {"num_workers": 6}}' > outputs/grid.log 2>&1 &
echo "Grid started in the background (PID $!). You can close this terminal; keep the Mac plugged in."
