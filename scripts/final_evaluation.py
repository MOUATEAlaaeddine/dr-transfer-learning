"""Step 7: select, then evaluate once on the frozen test set. Run after the CV grid.

    python scripts/final_evaluation.py

1. results/summary.csv from the validation folds (drtl.evaluate).
2. Selection on validation only: best strategy of each model by mean macro F1.
3. For each selected configuration, the 5 fold checkpoints predict the test set; their averaged
   probabilities are the declared final model (a 5-fold ensemble). Single-fold test scores are also
   reported (mean ± std) to show the spread.
4. Robustness: macro F1 on test images degraded by blur, under-exposure, noise and JPEG (levels 1–4).
5. Cost: parameters, weight file size, CPU inference time per image.
Outputs: results/final_test.csv, results/robustness.csv, results/cost.csv, results/figures/*.png
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from torchvision import transforms as T

from drtl import evaluate
from drtl.config import PROJECT_ROOT, load_config
from drtl.data import FundusDataset, load_splits, make_test_loader
from drtl.results import run_id
from drtl.robustness import KINDS, degrade
from drtl.run import on_kaggle
from drtl.train import load_best, predict
from drtl.transforms import build_eval_transform
from drtl.utils import get_device

RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"


def checkpoint(rid: str) -> Path:
    base = Path("/kaggle/working/outputs") if on_kaggle() else PROJECT_ROOT / "outputs"
    return base / rid / "best.pt"


def image_root(config) -> Path:
    cache = PROJECT_ROOT / "data" / "cache" / str(config["data"]["image_size"])
    return cache if cache.exists() else PROJECT_ROOT / config["data"]["root"]


def main():
    config = load_config()
    device = get_device()
    FIGURES.mkdir(parents=True, exist_ok=True)
    evaluate.main([])
    summary = pd.read_csv(RESULTS / "summary.csv")
    final = evaluate.select_final_configurations(summary)
    if final.empty:
        raise SystemExit("No configuration has all its folds: finish the grid first.")
    splits = load_splits(PROJECT_ROOT / config["data"]["splits_file"])
    root = image_root(config)
    test_loader = make_test_loader(config, splits, root)
    test_rows = splits[splits["split"] == "test"]
    names = config["data"]["class_names"]
    n_folds = config["data"]["n_folds"]

    rows, robust, cost = [], [], []
    for _, cfg in final.iterrows():
        model_key, strategy = cfg["model"], cfg["strategy"]
        models = [load_best(checkpoint(run_id(model_key, strategy, k)))["model"].to(device) for k in range(n_folds)]
        fold_probs = [predict(m, test_loader, device)["probs"] for m in models]
        labels = predict(models[0], test_loader, device)["labels"]
        ensemble = np.mean(fold_probs, axis=0)
        m = evaluate.compute_metrics(labels, ensemble, len(names))
        single = pd.DataFrame([{k: evaluate.compute_metrics(labels, p, len(names))[k] for k in evaluate.METRICS}
                               for p in fold_probs])
        rows.append({"model": model_key, "strategy": strategy, "test_images": m["n"],
                     **{f"test_{k}": m[k] for k in evaluate.METRICS},
                     **{f"single_fold_{k}_mean": single[k].mean() for k in evaluate.METRICS},
                     **{f"single_fold_{k}_std": single[k].std() for k in evaluate.METRICS},
                     "val_f1_macro_mean": cfg["f1_macro_mean"], "absent_classes": m["absent_classes"]})
        pd.DataFrame(ensemble, columns=[f"prob_{i}" for i in range(len(names))]).assign(
            label=labels, image_id=test_rows["image_id"].to_numpy()).to_csv(
            RESULTS / f"final_test_predictions_{model_key}.csv", index=False, float_format="%.6f")
        _figures(model_key, strategy, labels, ensemble, names)

        # Robustness: same 5-fold ensemble on degraded test images.
        for kind in KINDS:
            for level in range(1, 5):
                tf = T.Compose([T.Lambda(lambda im, k=kind, l=level: degrade(im, k, l)),
                                build_eval_transform(config["data"]["image_size"])])
                loader = torch.utils.data.DataLoader(FundusDataset(test_rows, root, tf), batch_size=32)
                probs = np.mean([predict(mm, loader, device)["probs"] for mm in models], axis=0)
                f1 = evaluate.compute_metrics(labels, probs, len(names))["f1_macro"]
                robust.append({"model": model_key, "strategy": strategy, "kind": kind, "level": level,
                               "f1_macro": f1, "f1_drop": m["f1_macro"] - f1})

        # Cost: one fold checkpoint on CPU, batch of 1 (what a laptop demo sees).
        cpu_model = load_best(checkpoint(run_id(model_key, strategy, 0)))["model"].cpu()
        x = torch.randn(1, 3, config["data"]["image_size"], config["data"]["image_size"])
        with torch.no_grad():
            for _ in range(3):
                cpu_model(x)
            t = time.time()
            for _ in range(20):
                cpu_model(x)
        cost.append({"model": model_key, "strategy": strategy,
                     "params_total": int(cfg["params_total"]), "params_trainable": int(cfg["params_trainable"]),
                     "weights_mb": round(checkpoint(run_id(model_key, strategy, 0)).stat().st_size / 1e6, 1),
                     "cpu_ms_per_image": round((time.time() - t) / 20 * 1000, 1),
                     "train_minutes_per_fold": round(cfg["train_minutes_mean"], 1)})
        print(f"{model_key}/{strategy}: test macro F1 {m['f1_macro']:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(RESULTS / "final_test.csv", index=False, float_format="%.4f")
    rob = pd.DataFrame(robust)
    rob.to_csv(RESULTS / "robustness.csv", index=False, float_format="%.4f")
    pd.DataFrame(cost).to_csv(RESULTS / "cost.csv", index=False)
    _robustness_figure(rob)
    print(json.dumps({"final": [r["model"] + "/" + r["strategy"] for r in rows]}, indent=2))


def _figures(model_key, strategy, labels, probs, names):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ConfusionMatrixDisplay.from_predictions(labels, probs.argmax(1), labels=range(len(names)), display_labels=names,
                                            ax=axes[0], colorbar=False, xticks_rotation=30, cmap="Blues")
    axes[0].set_title("Confusion matrix (test)")
    for c, name in enumerate(names):
        if 0 < (labels == c).sum() < len(labels):
            fpr, tpr, _ = roc_curve(labels == c, probs[:, c])
            axes[1].plot(fpr, tpr, label=name)
    axes[1].plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC one-vs-rest (test)")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{model_key} · {strategy} · 5-fold ensemble")
    fig.tight_layout(); fig.savefig(FIGURES / f"test_{model_key}.png", dpi=150); plt.close(fig)


def _robustness_figure(rob):
    fig, axes = plt.subplots(1, len(KINDS), figsize=(4 * len(KINDS), 3.4), sharey=True)
    for ax, kind in zip(axes, KINDS):
        for model, g in rob[rob["kind"] == kind].groupby("model"):
            ax.plot(g["level"], g["f1_macro"], marker="o", label=model)
        ax.set(title=kind, xlabel="level", xticks=[1, 2, 3, 4])
    axes[0].set_ylabel("test macro F1"); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "robustness.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
