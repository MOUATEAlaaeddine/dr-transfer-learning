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
from drtl.results import REAL_STATUS, file_hash, run_id
from drtl.run import run_experiment
from drtl.train import amp_enabled
from drtl.utils import get_device

STATUS = PROJECT_ROOT / "outputs" / "grid_status.json"
LOW_MEMORY = {"batch_size": 16, "accumulation_steps": 2}


def _free_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom(err: Exception) -> bool:
    text = str(err).lower()
    # A DataLoader worker killed by macOS under memory pressure is the same problem as an OOM.
    return "out of memory" in text or "exited unexpectedly" in text


def plan(config, models=None, folds=None):
    order = []
    for fold in folds if folds is not None else range(config["data"]["n_folds"]):
        for model in models or list(config["models"]):
            for strategy in strategies_for(model):
                order.append((model, strategy, fold))
    return order


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+")
    p.add_argument("--folds", nargs="+", type=int)
    p.add_argument("--overrides", default="{}", help='JSON, e.g. {"training": {"num_workers": 6}}')
    a = p.parse_args()
    base_overrides = json.loads(a.overrides)
    config = load_config(overrides=base_overrides)
    unknown = set(a.models or ()) - set(config["models"])
    if unknown:
        p.error(f"unknown model(s): {', '.join(sorted(unknown))}")
    queue = plan(config, a.models, a.folds)
    if not queue or any(not 0 <= fold < config["data"]["n_folds"] for _, _, fold in queue):
        p.error("select at least one run and use fold numbers from the configured split")
    if len(queue) != len(set(queue)):
        p.error("select each model and fold once")
    status = {"runs": {}, "started": dt.datetime.now().isoformat(timespec="seconds"), "total": len(queue)}
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    results = PROJECT_ROOT / config["outputs"]["results_dir"]
    checkpoints = PROJECT_ROOT / config["outputs"]["checkpoints_dir"]

    def save():
        done = [r for r in status["runs"].values() if r.get("state") == "completed"]
        status["completed"] = len(done)
        status["updated"] = dt.datetime.now().isoformat(timespec="seconds")
        temporary = STATUS.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(STATUS)

    want_amp = amp_enabled(config, get_device())
    split_hash = file_hash(PROJECT_ROOT / config["data"]["splits_file"])
    config_hash = file_hash(PROJECT_ROOT / "configs" / "default.yaml")

    def completed(record):
        return {"state": "completed", "val_f1_macro": record["val_f1_macro"],
                "epochs": record["epochs_run"], "minutes": round(record["duration_s"] / 60, 1)}

    for model, strategy, fold in queue:
        rid = run_id(model, strategy, fold)
        path = results / f"{rid}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record.get("status") != REAL_STATUS:
            continue
        if (record["amp"], record["splits_hash"], record["config_hash"]) != (want_amp, split_hash, config_hash):
            p.error(f"{rid} was completed with a different precision, split or config. "
                    "Use matching settings or separate output and results directories.")
        status["runs"][rid] = completed(record)

    def train(model, strategy, fold):
        rid = run_id(model, strategy, fold)
        status["runs"][rid] = {"state": "running", "since": dt.datetime.now().isoformat(timespec="seconds")}
        save()
        overrides = dict(base_overrides)
        for attempt in (1, 2, 3):
            try:
                record = run_experiment(model, strategy, fold, overrides=overrides)
                status["runs"][rid] = completed(record)
                break
            except Exception as err:  # keep the grid going; the error is in the status file
                _free_memory()
                if _is_oom(err):
                    training = {**overrides.get("training", {}), "num_workers": 0}
                    if not (checkpoints / rid / "last.pt").exists():
                        training.update(LOW_MEMORY)
                    overrides = {**overrides, "training": training}
                elif attempt < 3:
                    time.sleep(60)  # network or transient error: resume from last.pt after a pause
                status["runs"][rid] = {"state": "failed", "error": str(err).splitlines()[0][:300],
                                       "trace": traceback.format_exc()[-1500:], "attempts": attempt}
        _free_memory()
        save()
        print(f"[{status['completed']}/{len(queue)}] {rid}: {status['runs'][rid]['state']}", flush=True)

    save()

    for model, strategy, fold in queue:
        if status["runs"].get(run_id(model, strategy, fold), {}).get("state") != "completed":
            train(model, strategy, fold)
    for model, strategy, fold in queue:  # last chance for runs that failed three times
        if status["runs"].get(run_id(model, strategy, fold), {}).get("state") == "failed":
            train(model, strategy, fold)
    print("Grid finished:", status["completed"], "/", len(queue))
    if status["completed"] != len(queue):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
