#!/bin/bash
# Mac (Apple Silicon): tests → data preparation → full 65-run grid in the background.
# Usage from the repository root:  bash scripts/mac_train_all.sh
# Follow progress: tail -f outputs/grid.log   ·   status: outputs/grid_status.json
cd "$(dirname "$0")/.."
mkdir -p outputs
export PYTHONPATH="$PWD/src"      # no install needed: the package is imported from src/
PY=.venv/bin/python
exec > >(tee -a outputs/launcher.log) 2>&1
echo "== $(date) · $($PY --version) · torch $($PY -c 'import torch;print(torch.__version__, "mps:", torch.backends.mps.is_available())')"
# Tests take long on CPU; run them only with:  bash scripts/mac_train_all.sh --test
if [ "$1" = "--test" ]; then $PY -X faulthandler scripts/run_tests.py; tail -1 outputs/pytest_mac.log; fi
if [ ! -f data/splits/folds.csv ]; then
  $PY scripts/prepare_data.py --raw data/raw > outputs/prepare_data.log 2>&1 || { echo "prepare_data failed: see outputs/prepare_data.log"; exit 1; }
  tail -3 outputs/prepare_data.log
fi
pkill -f "scripts/run_grid.py" 2>/dev/null && echo "stopped the previous grid (it resumes from its checkpoints)" && sleep 3
# 4 loader workers on the M5 (6 crashed under memory pressure with DenseNet full fine-tuning); the grid is fold-major and resumable (re-run this script to continue).
nohup caffeinate -i env PYTHONPATH="$PYTHONPATH" $PY scripts/run_grid.py --overrides '{"training": {"num_workers": 4}}' > outputs/grid.log 2>&1 &
echo "Grid started in the background (PID $!). You can close this terminal; keep the Mac plugged in."
if ! pgrep -f "scripts/dashboard.py" >/dev/null; then
  nohup $PY scripts/dashboard.py > outputs/dashboard.log 2>&1 &
  sleep 2
fi
open http://127.0.0.1:8765
