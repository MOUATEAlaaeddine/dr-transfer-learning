#!/bin/bash
# Mac (Apple Silicon): tests → data preparation → full 65-run grid in the background.
# Usage: bash scripts/mac_train_all.sh [--test] [--models ...] [--folds ...]
# Follow progress: tail -f outputs/grid.log   ·   status: outputs/grid_status.json
cd "$(dirname "$0")/.."
mkdir -p outputs
export PYTHONPATH="$PWD/src"      # no install needed: the package is imported from src/
PY=${DRTL_PYTHON:-.venv/bin/python}
exec > >(tee -a outputs/launcher.log) 2>&1
echo "== $(date) · $($PY --version) · torch $($PY -c 'import torch;print(torch.__version__, "mps:", torch.backends.mps.is_available())')"
# Tests take long on CPU; run them only with:  bash scripts/mac_train_all.sh --test
if [ "$1" = "--test" ]; then
  "$PY" -m pytest -q || exit 1
  shift
fi
if [ ! -f data/splits/folds.csv ]; then
  $PY scripts/prepare_data.py --raw data/raw > outputs/prepare_data.log 2>&1 || { echo "prepare_data failed: see outputs/prepare_data.log"; exit 1; }
  tail -3 outputs/prepare_data.log
fi
if pgrep -f 'scripts/[r]un_grid.py' >/dev/null; then
  echo "A grid is already running. See outputs/grid.log."
  exit 1
fi
# Match the existing float32 runs and avoid separate loader processes on this Mac.
nohup caffeinate -i "$PY" -u scripts/run_grid.py --overrides '{"training": {"num_workers": 0, "mixed_precision": false}}' "$@" >> outputs/grid.log 2>&1 </dev/null &
echo "Grid started in the background (PID $!). You can close this terminal; keep the Mac plugged in."
if ! pgrep -f "scripts/dashboard.py" >/dev/null; then
  nohup $PY scripts/dashboard.py > outputs/dashboard.log 2>&1 &
  sleep 2
fi
open http://127.0.0.1:8765
