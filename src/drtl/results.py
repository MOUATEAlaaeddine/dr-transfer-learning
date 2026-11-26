"""Run records: one small JSON per run, merged into results/runs.csv.

Each teammate's runs write their own files (results/runs/<run_id>.json plus the validation
predictions), so four people training in parallel never edit the same file. `runs.csv` is
regenerated from those files with `python -m drtl.results`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from drtl.config import PROJECT_ROOT, load_config
from drtl.fixtures import SYNTHETIC_SOURCE
from drtl.models import strategies_for

SCHEMA_VERSION = 1

# Column order of results/runs.csv. Agreed with evaluation (Ikram) in docs/team/INTERFACES.md.
RUN_COLUMNS = [
    "run_id", "model", "timm_name", "strategy", "fold", "status", "data_source",
    "seed", "config_hash", "splits_hash", "git_commit", "overrides",
    "device", "gpu_name", "amp", "batch_size", "accumulation_steps",
    "params_total", "params_trainable",
    "epochs_run", "best_epoch", "stopped_early", "duration_s", "sec_per_epoch",
    "val_loss", "val_accuracy", "val_precision_macro", "val_recall_macro", "val_f1_macro",
    "checkpoint_ref", "predictions_file", "started_at", "schema_version",
]

REAL_STATUS = "completed"


def run_id(model: str, strategy: str, fold: int) -> str:
    """Stable identifier, e.g. vit_b16__feature_extraction__fold0."""
    return f"{model}__{strategy}__fold{int(fold)}"


def file_hash(path: Path) -> str:
    """First 12 hex digits of SHA-256, with CRLF normalised so Windows clones hash the same."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()[:12]


def expected_run_ids(config: Dict[str, Any]) -> List[str]:
    """The 65 cross-validation runs of the grid (13 configurations × n_folds)."""
    return [run_id(model, strategy, fold)
            for model in config["models"] for strategy in strategies_for(model)
            for fold in range(config["data"]["n_folds"])]


def write_run(record: Dict[str, Any], probs: np.ndarray, labels: np.ndarray,
              class_names: Iterable[str], results_dir: Path) -> Path:
    """Save <run_id>.json and <run_id>_val_predictions.csv (best checkpoint, validation fold)."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rid = record["run_id"]
    preds = pd.DataFrame(probs, columns=[f"prob_{i}" for i in range(probs.shape[1])])
    preds.insert(0, "label", labels)
    preds.insert(1, "pred", probs.argmax(1))
    pred_file = results_dir / f"{rid}_val_predictions.csv"
    preds.to_csv(pred_file, index=False, float_format="%.6f")
    record = {**{k: None for k in RUN_COLUMNS}, **record, "predictions_file": pred_file.name,
              "class_names": list(class_names), "schema_version": SCHEMA_VERSION}
    path = results_dir / f"{rid}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def load_runs(results_dir: Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(results_dir).glob("*.json"))]
    return pd.DataFrame(rows, columns=RUN_COLUMNS) if not rows else pd.DataFrame(rows)[RUN_COLUMNS]


def merge_runs(results_dir: Path, config: Dict[str, Any], out_csv: Path) -> pd.DataFrame:
    """Write runs.csv: every expected run, with status "missing" where no real record exists.

    Smoke and SIMULATED runs are never merged; a missing run is shown as missing, never filled.
    """
    runs = load_runs(results_dir)
    runs = runs[(runs["status"] == REAL_STATUS) & (runs["data_source"] != SYNTHETIC_SOURCE)]
    present = set(runs["run_id"])
    missing = []
    for rid in expected_run_ids(config):
        if rid not in present:
            model, strategy, fold = rid.split("__")
            missing.append({"run_id": rid, "model": model, "strategy": strategy,
                            "fold": int(fold.replace("fold", "")), "status": "missing"})
    table = pd.concat([runs, pd.DataFrame(missing, columns=RUN_COLUMNS)], ignore_index=True)
    table = table.sort_values(["model", "strategy", "fold"]).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge results/runs/*.json into results/runs.csv")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results" / "runs"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "results" / "runs.csv"))
    args = parser.parse_args()
    table = merge_runs(Path(args.results_dir), load_config(), Path(args.out))
    done = (table["status"] == REAL_STATUS).sum()
    print(f"{done}/{len(table)} runs completed; written to {args.out}")


if __name__ == "__main__":
    main()
