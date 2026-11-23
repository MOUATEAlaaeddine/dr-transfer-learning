"""Run one configuration of the grid: model × strategy × fold.

Notebook 02 and the command line both call `run_experiment`, so every teammate trains with the
same code. Command line (from the repository root):

    python -m drtl.run --model vit_b16 --strategy feature_extraction --fold 0
    python -m drtl.run --model mobilenetv4 --strategy feature_extraction --fold 0 --smoke --synthetic
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from drtl.config import PROJECT_ROOT, load_config, merge
from drtl.fixtures import SYNTHETIC_SOURCE, make_synthetic_fold_loaders, synthetic_splits
from drtl.models import build_for_run, count_parameters, strategy_settings
from drtl.results import REAL_STATUS, file_hash, run_id, write_run
from drtl.train import BEST, LAST, fit
from drtl.utils import get_device, set_seed

KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_INPUT = Path("/kaggle/input")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Smoke runs check the plumbing in minutes: two epochs on a few batches. Never reported.
SMOKE_OVERRIDES = {"training": {"max_epochs": 2, "batch_size": 8, "warmup_epochs": 1}}
SMOKE_BATCHES = {"max_train_batches": 4, "max_val_batches": 2}


def on_kaggle() -> bool:
    return KAGGLE_WORKING.exists()


def git_commit(root: Path = PROJECT_ROOT) -> str:
    """Current commit, suffixed -dirty when tracked files changed (record exactly what ran)."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], cwd=root, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def find_kaggle_dataset(slug: str) -> Optional[Path]:
    """Folder where Kaggle mounted the attached dataset (the path layout changed over time)."""
    name = slug.split("/")[-1]
    for candidate in [KAGGLE_INPUT / name, *KAGGLE_INPUT.glob(f"**/{name}")]:
        if candidate.is_dir():
            return candidate
    return None


def _real_loaders(config, splits_file: Path, fold: int, image_root: Optional[Path], batch_size: int):
    try:  # owned by the data branch (feat/data-splits); interface in docs/team/INTERFACES.md
        from drtl.data import load_splits, make_fold_loaders
    except ImportError as err:
        raise RuntimeError("drtl.data is not merged yet (feat/data-splits). Use synthetic=True for a "
                           "SIMULATED pipeline check, or wait for the data PR.") from err
    if not Path(splits_file).exists():
        raise FileNotFoundError(f"{splits_file} not found: the frozen split is produced in Step 2 (Abdelati)")
    root = image_root or find_kaggle_dataset(config["data"]["kaggle_dataset"]) or PROJECT_ROOT / config["data"]["root"]
    splits = load_splits(splits_file)
    train_loader, val_loader = make_fold_loaders(config, splits, fold, Path(root), batch_size=batch_size)
    return splits, train_loader, val_loader


def run_experiment(model: str, strategy: str, fold: int, *, overrides: Optional[Dict[str, Any]] = None,
                   synthetic: bool = False, smoke: bool = False, resume: bool = True, force: bool = False,
                   pretrained: Optional[bool] = None, image_root: Optional[Path] = None,
                   output_root: Optional[Path] = None, results_dir: Optional[Path] = None,
                   splits_file: Optional[Path] = None, device: Optional[str] = None) -> Dict[str, Any]:
    """Train one configuration on one fold, save checkpoints and write its run record.

    Real runs (Kaggle data, not smoke) write to results/runs/. Smoke and synthetic runs are
    marked as such and written next to their checkpoints instead, so they can never reach the
    results table.
    """
    overrides = dict(overrides or {})
    if smoke:
        overrides = merge(SMOKE_OVERRIDES, overrides)
    config = load_config(overrides=overrides)
    set_seed(config["seed"])
    rid = run_id(model, strategy, fold)
    lrs = strategy_settings(config, model, strategy)
    real = not (smoke or synthetic)

    base_out = Path(output_root or (KAGGLE_WORKING / "outputs" if on_kaggle()
                                    else PROJECT_ROOT / config["outputs"]["checkpoints_dir"]))
    out_dir = base_out / ("smoke" if not real else "") / rid
    if real:
        results = Path(results_dir or (KAGGLE_WORKING / "results" / "runs" if on_kaggle()
                                       else PROJECT_ROOT / config["outputs"]["results_dir"]))
    else:
        results = out_dir / "record"
    if real and not force and (results / f"{rid}.json").exists():
        record = json.loads((results / f"{rid}.json").read_text(encoding="utf-8"))
        if record.get("status") == REAL_STATUS:
            print(f"{rid} already completed; pass force=True to retrain.")
            return {**record, "record_file": str(results / f"{rid}.json"),
                    "predictions_path": str(results / record["predictions_file"])}

    batch_size = int(config["training"]["batch_size"])
    splits_file = Path(splits_file or PROJECT_ROOT / config["data"]["splits_file"])
    if synthetic:
        splits = synthetic_splits(num_classes=len(config["data"]["class_names"]),
                                  n_folds=config["data"]["n_folds"], seed=config["seed"])
        train_loader, val_loader = make_synthetic_fold_loaders(config, splits, fold, batch_size)
        splits_hash, data_source = "SIMULATED", SYNTHETIC_SOURCE
    else:
        splits, train_loader, val_loader = _real_loaders(config, splits_file, fold, image_root, batch_size)
        splits_hash, data_source = file_hash(splits_file), "kaggle"

    # Guard against leakage: validation must be exactly fold k of the CV part, test never used.
    cv = splits[splits["split"] == "cv"]
    if len(val_loader.dataset) != int((cv["fold"] == fold).sum()):
        raise RuntimeError("validation loader does not match fold rows in the split table")
    train_labels = cv.loc[cv["fold"] != fold, "label"].to_numpy()

    torch_device = torch.device(device) if device else get_device()
    use_pretrained = (not synthetic) if pretrained is None else pretrained
    net = build_for_run(config, model, strategy, pretrained=use_pretrained)
    counts = count_parameters(net)
    metadata = {
        "run_id": rid, "model": model, "timm_name": config["models"][model], "strategy": strategy,
        "fold": fold, "class_names": config["data"]["class_names"], "image_size": config["data"]["image_size"],
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}, "seed": config["seed"],
        "config_hash": file_hash(PROJECT_ROOT / "configs" / "default.yaml"), "splits_hash": splits_hash,
        "git_commit": git_commit(), "overrides": overrides, "data_source": data_source,
        "pretrained": use_pretrained,
    }
    print(f"{rid} | {data_source} | device {torch_device} | trainable {counts['trainable']:,}/{counts['total']:,}")

    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    summary = fit(net, train_loader, val_loader, train_labels=train_labels, config=config, lrs=lrs,
                  device=torch_device, out_dir=out_dir, metadata=metadata, resume=resume,
                  **(SMOKE_BATCHES if smoke else {}))

    import numpy as np  # predictions of the best epoch, saved by fit
    best_preds = np.load(out_dir / "best_val_predictions.npz")
    best = summary["best"]
    record = {
        "run_id": rid, "model": model, "timm_name": metadata["timm_name"], "strategy": strategy,
        "fold": fold, "status": REAL_STATUS if real else ("smoke" if smoke else "synthetic"),
        "data_source": data_source, "seed": config["seed"], "config_hash": metadata["config_hash"],
        "splits_hash": splits_hash, "git_commit": metadata["git_commit"], "overrides": json.dumps(overrides),
        "device": torch_device.type,
        "gpu_name": torch.cuda.get_device_name(0) if torch_device.type == "cuda" else "",
        "amp": summary["amp"], "batch_size": batch_size, "accumulation_steps": summary["accumulation_steps"],
        "params_total": counts["total"], "params_trainable": counts["trainable"],
        "epochs_run": summary["epochs_run"], "best_epoch": summary["best_epoch"],
        "stopped_early": summary["stopped_early"], "duration_s": summary["train_seconds"],
        "sec_per_epoch": summary["sec_per_epoch"], "val_loss": round(best["val_loss"], 6),
        "val_accuracy": round(best["val_accuracy"], 6), "val_precision_macro": round(best["val_precision_macro"], 6),
        "val_recall_macro": round(best["val_recall_macro"], 6), "val_f1_macro": round(best["val_f1_macro"], 6),
        "checkpoint_ref": str(out_dir / BEST), "started_at": started,
    }
    path = write_run(record, best_preds["probs"], best_preds["labels"], config["data"]["class_names"], results)
    print(f"Record: {path}\nBest checkpoint: {out_dir / BEST}\nResumable state: {out_dir / LAST}")
    # Local paths of this session's files (not stored in the JSON, which must stay portable).
    return {**record, "record_file": str(path), "predictions_path": str(path.parent / f"{rid}_val_predictions.csv")}


def main() -> None:
    p = argparse.ArgumentParser(description="Train one model × strategy × fold")
    p.add_argument("--model", required=True)
    p.add_argument("--strategy", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--smoke", action="store_true", help="2 epochs on 4 batches; never reported")
    p.add_argument("--synthetic", action="store_true", help="SIMULATED data, no dataset needed")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--force", action="store_true", help="retrain even if a completed record exists")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--accumulation-steps", type=int)
    p.add_argument("--device")
    a = p.parse_args()
    training = {k: v for k, v in {"batch_size": a.batch_size, "accumulation_steps": a.accumulation_steps}.items() if v}
    run_experiment(a.model, a.strategy, a.fold, overrides={"training": training} if training else None,
                   synthetic=a.synthetic, smoke=a.smoke, resume=not a.no_resume, force=a.force, device=a.device)


if __name__ == "__main__":
    main()
