"""Training loop shared by every configuration of the grid (Step 5).

One call to `fit` trains one model on one fold: AdamW with a head and a backbone learning rate,
linear warm-up then cosine decay, class-weighted cross-entropy computed on the training folds
only, mixed precision on CUDA, early stopping on validation loss, and the checkpoint kept at
the best validation macro F1. State is saved after every epoch so an interrupted Kaggle
session resumes where it stopped.
"""

from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn

from drtl.models import parameter_groups, set_train_mode

LAST = "last.pt"   # full resumable state, overwritten every epoch
BEST = "best.pt"   # weights + metadata at the best validation macro F1 (used by eval and demo)
HISTORY = "history.csv"


# --------------------------------------------------------------------------- building blocks


def class_weights(labels: Iterable[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights n / (K · n_c), mean ≈ 1 (scikit-learn's "balanced").

    Called with training-fold labels only, so validation and test never influence the loss.
    A class absent from the training folds gets weight 0 (it cannot be learned anyway).
    """
    counts = np.bincount(np.asarray(list(labels), dtype=int), minlength=num_classes).astype(float)
    weights = np.where(counts > 0, counts.sum() / (num_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def warmup_cosine(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    """LR multiplier per optimizer step: linear warm-up, then cosine decay to 0."""
    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return factor


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Accuracy and macro precision / recall / F1 over all five grades.

    `labels=range(K)` keeps a grade missing from the fold in the macro average (it scores 0)
    instead of silently dropping it. Full evaluation (AUC, QWK, confusion matrices) lives in
    drtl.evaluate and is recomputed from the saved predictions.
    """
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision_macro": float(p),
            "recall_macro": float(r), "f1_macro": float(f1)}


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def amp_enabled(config: Dict[str, Any], device: torch.device) -> bool:
    """Mixed precision (float16 + loss scaling) on CUDA and Apple MPS GPUs, never on CPU."""
    return bool(config["training"].get("mixed_precision")) and device.type in ("cuda", "mps")


def _grad_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError, RuntimeError, ValueError):  # old torch or no MPS scaler
        if device.type == "cuda":
            return torch.cuda.amp.GradScaler(enabled=enabled)
        return None


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device, criterion: Optional[nn.Module] = None,
            amp: bool = False, max_batches: Optional[int] = None) -> Dict[str, Any]:
    """Probabilities, labels and mean loss for a whole loader (model in eval mode)."""
    model.eval()
    probs, labels, loss_sum, n = [], [], 0.0, 0
    for i, (images, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, targets = images.to(device, non_blocking=True), targets.to(device)
        with _autocast(device, amp):
            logits = model(images)
        logits = logits.float()
        if criterion is not None:
            loss_sum += criterion(logits, targets).item() * len(targets)
        probs.append(torch.softmax(logits, dim=1).cpu())
        labels.append(targets.cpu())
        n += len(targets)
    return {"probs": torch.cat(probs).numpy(), "labels": torch.cat(labels).numpy(),
            "loss": loss_sum / max(n, 1)}


def _rng_state() -> Dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# --------------------------------------------------------------------------- the loop


def fit(model: nn.Module, train_loader, val_loader, *, train_labels: Iterable[int],
        config: Dict[str, Any], lrs: Dict[str, float], device: torch.device, out_dir: Path,
        metadata: Dict[str, Any], resume: bool = True, max_train_batches: Optional[int] = None,
        max_val_batches: Optional[int] = None, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Train one configuration on one fold and return a summary of the best epoch.

    `metadata` (run id, hashes, class names, ...) is stored in every checkpoint; resuming
    refuses a checkpoint written for another run, config or split.
    """
    tcfg = config["training"]
    num_classes = len(config["data"]["class_names"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    amp = amp_enabled(config, device)
    accum = int(tcfg.get("accumulation_steps", 1))
    patience = int(tcfg["early_stopping_patience"])
    max_epochs = int(tcfg["max_epochs"])

    weights = class_weights(train_labels, num_classes) if tcfg.get("class_weighted_loss") else None
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, lrs["head_lr"], lrs.get("backbone_lr", 0.0), tcfg["weight_decay"]))
    batches = len(train_loader) if max_train_batches is None else min(len(train_loader), max_train_batches)
    steps_per_epoch = max(1, math.ceil(batches / accum))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(steps_per_epoch * max_epochs, steps_per_epoch * int(tcfg["warmup_epochs"])))
    scaler = _grad_scaler(device, amp)
    if scaler is None:  # no loss scaler for this device: stay in float32 rather than risk underflow
        amp, scaler = False, _grad_scaler(torch.device("cpu"), False)

    state = {"epoch": 0, "best_f1": -1.0, "best_epoch": 0, "best_loss": math.inf,
             "epochs_without_improvement": 0, "history": [], "train_seconds": 0.0, "stopped_early": False}
    last_path = out_dir / LAST
    if resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        for key in ("run_id", "config_hash", "splits_hash"):
            if ckpt["metadata"].get(key) != metadata.get(key):
                raise RuntimeError(f"{last_path} belongs to another run ({key} differs); "
                                   "use resume=False or a different output folder")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        try:  # a checkpoint saved without mixed precision has an empty scaler state
            scaler.load_state_dict(ckpt["scaler_state"])
        except RuntimeError:
            pass
        state = ckpt["train_state"]
        _set_rng_state(ckpt["rng_state"])
        log(f"Resumed from epoch {state['epoch']} (best macro F1 {state['best_f1']:.4f} at epoch {state['best_epoch']})")

    while state["epoch"] < max_epochs and not state["stopped_early"]:
        epoch = state["epoch"] + 1
        start = time.time()
        set_train_mode(model)
        optimizer.zero_grad(set_to_none=True)
        train_loss, seen = 0.0, 0
        for i, (images, targets) in enumerate(train_loader):
            if i >= batches:
                break
            images, targets = images.to(device, non_blocking=True), targets.to(device)
            with _autocast(device, amp):
                loss = criterion(model(images).float(), targets)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or i + 1 == batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            train_loss += loss.item() * len(targets)
            seen += len(targets)

        val = predict(model, val_loader, device, criterion, amp, max_val_batches)
        metrics = basic_metrics(val["labels"], val["probs"].argmax(1), num_classes)
        seconds = time.time() - start
        state["train_seconds"] += seconds
        row = {"epoch": epoch, "train_loss": train_loss / max(seen, 1), "val_loss": val["loss"],
               **{f"val_{k}": v for k, v in metrics.items()},
               "lr_head": optimizer.param_groups[0]["lr"], "seconds": round(seconds, 2)}
        state["history"].append(row)

        improved_f1 = metrics["f1_macro"] > state["best_f1"]
        if improved_f1:  # checkpoint selection: validation macro F1 (never the test set)
            state.update(best_f1=metrics["f1_macro"], best_epoch=epoch)
            torch.save({"model_state": model.state_dict(), "metadata": metadata, "epoch": epoch,
                        "val_metrics": {**metrics, "loss": val["loss"]}},
                       out_dir / BEST)
            np.savez_compressed(out_dir / "best_val_predictions.npz", probs=val["probs"], labels=val["labels"])
        if val["loss"] < state["best_loss"] - 1e-6:   # early stopping: validation loss
            state.update(best_loss=val["loss"], epochs_without_improvement=0)
        else:
            state["epochs_without_improvement"] += 1
            if state["epochs_without_improvement"] >= patience:
                state["stopped_early"] = True
        state["epoch"] = epoch

        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                    "train_state": state, "rng_state": _rng_state(), "metadata": metadata},
                   last_path)
        _write_history(out_dir / HISTORY, state["history"])
        log(f"epoch {epoch:>2}/{max_epochs}  train_loss {row['train_loss']:.4f}  val_loss {val['loss']:.4f}  "
            f"val_f1 {metrics['f1_macro']:.4f}{'  *best*' if improved_f1 else ''}  {seconds:.1f}s")

    best = next(r for r in state["history"] if r["epoch"] == state["best_epoch"])
    return {"epochs_run": state["epoch"], "best_epoch": state["best_epoch"],
            "stopped_early": state["stopped_early"], "train_seconds": round(state["train_seconds"], 1),
            "sec_per_epoch": round(state["train_seconds"] / max(state["epoch"], 1), 2),
            "best": best, "amp": amp, "accumulation_steps": accum}


def _write_history(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_best(path: Path, device: str = "cpu", pretrained_fallback: bool = False) -> Dict[str, Any]:
    """Rebuild the model stored in a best.pt checkpoint (for evaluation and the demo).

    Returns {"model", "metadata", "val_metrics"}; the metadata carries class names, image size
    and normalisation so the caller preprocesses images exactly as in training.
    """
    from drtl.models import build_model

    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["metadata"]
    model = build_model(meta["timm_name"], len(meta["class_names"]), pretrained=pretrained_fallback)
    model.load_state_dict(ckpt["model_state"])
    return {"model": model.eval(), "metadata": meta, "val_metrics": ckpt.get("val_metrics", {})}
