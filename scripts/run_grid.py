"""Run the whole cross-validation grid unattended (65 runs), resumable, with a status file.

    caffeinate -i python scripts/run_grid.py            # Mac: keeps the machine awake
    python scripts/run_grid.py --models vit_b16 --folds 0 1

Order is fold-major (fold 0 of all 13 configurations, then fold 1, ...): if time runs short, the
grid stops with the same folds for every configuration, which keeps the comparison fair (report
§11 fallback: k = 3). Completed runs are skipped; an interrupted run resumes from its last.pt.
A run that runs out of GPU memory is retried once with batch 16 + gradient accumulation 2 (same
effective batch of 32), and the override is written in its record.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import time
import traceback
from pathlib import Path

import torch

from drtl.config import PROJECT_ROOT, load_config
from drtl.models import strategies_for
from drtl.results import run_id
from drtl.run import run_experiment

STATUS = PROJECT_ROOT / "outputs" / "grid_status.json"
LOW_MEMORY = {"training": {"batch_size": 16, "accumulation_steps": 2}}


def _free_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom(err: Exception) -> bool:
    text = str(err).lower()
    return "out of memory" in text or "mps backend out of memory" in text


def plan(config, models=None, folds=None):
    order = []
    for fold in folds if folds is not None else range(config["data"]["n_folds"]):
        for model in models or list(config["models"]):
            for strategy in strategies_for(model):
                order.append((model, strategy, fold))
    return order


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*")
    p.add_argument("--folds", nargs="*", type=int)
    p.add_argument("--overrides", default="{}", help='JSON, e.g. {"training": {"num_workers": 6}}')
    a = p.parse_args()
    config = load_config()
    base_overrides = json.loads(a.overrides)
    status = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    status.setdefault("runs", {})
    queue = plan(config, a.models, a.folds)
    status.update(started=dt.datetime.now().isoformat(timespec="seconds"), total=len(queue))
    STATUS.parent.mkdir(parents=True, exist_ok=True)

    def save():
        done = [r for r in status["runs"].values() if r.get("state") == "completed"]
        status["completed"] = len(done)
        status["updated"] = dt.datetime.now().isoformat(timespec="seconds")
        STATUS.write_text(json.dumps(status, indent=2))

    for model, strategy, fold in queue:
        rid = run_id(model, strategy, fold)
        if status["runs"].get(rid, {}).get("state") == "completed":
            continue
        status["runs"][rid] = {"state": "running", "since": dt.datetime.now().isoformat(timespec="seconds")}
        save()
        start = time.time()
        overrides = dict(base_overrides)
        for attempt in (1, 2):
            try:
                record = run_experiment(model, strategy, fold, overrides=overrides)
                status["runs"][rid] = {"state": "completed", "val_f1_macro": record["val_f1_macro"],
                                       "epochs": record["epochs_run"], "minutes": round((time.time() - start) / 60, 1),
                                       "overrides": overrides}
                break
            except Exception as err:  # keep the grid going; the error is in the status file
                _free_memory()
                if attempt == 1 and _is_oom(err):
                    overrides = {**overrides, **LOW_MEMORY}
                    continue
                status["runs"][rid] = {"state": "failed", "error": str(err).splitlines()[0][:300],
                                       "trace": traceback.format_exc()[-1500:]}
                break
        _free_memory()
        save()
        print(f"[{status['completed']}/{len(queue)}] {rid}: {status['runs'][rid]['state']}", flush=True)
    print("Grid finished:", status["completed"], "/", len(queue))


if __name__ == "__main__":
    main()
