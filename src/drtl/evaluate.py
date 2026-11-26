"""Metrics, cross-validation summary and grid checks (Step 7).

Every number reported in the paper is recomputed here from the saved predictions, never typed
by hand. Command line:  python -m drtl.evaluate   → results/runs.csv + results/summary.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from drtl.config import PROJECT_ROOT, load_config
from drtl.fixtures import SYNTHETIC_SOURCE
from drtl.results import REAL_STATUS, load_runs, merge_runs

METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_macro_ovr", "qwk"]


def compute_metrics(y_true, y_prob, num_classes: int = 5) -> Dict[str, Any]:
    """All classification metrics of report §6.1 from labels and class probabilities.

    Precision / recall / F1 are macro averages over all `num_classes` grades (a grade the model never
    predicts scores 0). ROC-AUC is one-vs-rest per grade; a grade absent from `y_true` has no ROC
    curve, so it is excluded from the macro AUC and listed in `absent_classes` (never set to 0.5).
    QWK penalises errors by the squared distance between grades.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = y_prob.argmax(axis=1)
    labels = list(range(num_classes))
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    absent = [c for c in labels if support[c] == 0]
    aucs = {}
    for c in labels:
        if c not in absent and support[c] < len(y_true):
            aucs[c] = float(roc_auc_score((y_true == c).astype(int), y_prob[:, c]))
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p.mean()), "recall_macro": float(r.mean()), "f1_macro": float(f1.mean()),
        "auc_macro_ovr": float(np.mean(list(aucs.values()))) if aucs else float("nan"),
        "qwk": float(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {c: {"precision": float(p[c]), "recall": float(r[c]), "f1": float(f1[c]),
                          "support": int(support[c]), "auc": aucs.get(c)} for c in labels},
        "absent_classes": absent,
    }


def run_metrics(results_dir: Path, runs: pd.DataFrame) -> pd.DataFrame:
    """Recompute every metric of every run from its saved validation predictions."""
    rows = []
    for _, run in runs.iterrows():
        preds = pd.read_csv(Path(results_dir) / run["predictions_file"])
        probs = preds[[c for c in preds.columns if c.startswith("prob_")]].to_numpy()
        m = compute_metrics(preds["label"], probs, probs.shape[1])
        if abs(m["f1_macro"] - float(run["val_f1_macro"])) > 1e-4:
            raise ValueError(f"{run['run_id']}: recomputed F1 {m['f1_macro']:.6f} differs from record")
        rows.append({"run_id": run["run_id"], "model": run["model"], "strategy": run["strategy"],
                     "fold": run["fold"], **{k: m[k] for k in METRICS},
                     "params_total": run["params_total"], "params_trainable": run["params_trainable"],
                     "duration_s": run["duration_s"], "epochs_run": run["epochs_run"]})
    return pd.DataFrame(rows)


def check_grid(runs: pd.DataFrame) -> List[str]:
    """Problems that would make the comparison unfair (report Step 6 control)."""
    issues = []
    for column in ("config_hash", "splits_hash", "seed"):
        values = runs[column].astype(str).unique()
        if len(values) > 1:
            issues.append(f"{column} differs between runs: {sorted(values)}")
    return issues


def summarize(per_run: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Mean ± standard deviation over folds for each configuration (results table, assignment p.3)."""
    grouped = per_run.groupby(["model", "strategy"])
    table = grouped[METRICS].agg(["mean", "std"])
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    table["folds_done"] = grouped.size()
    table["folds_missing"] = n_folds - table["folds_done"]
    table["params_total"] = grouped["params_total"].first()
    table["params_trainable"] = grouped["params_trainable"].first()
    table["train_minutes_mean"] = grouped["duration_s"].mean() / 60
    return table.reset_index().sort_values("f1_macro_mean", ascending=False)


def select_final_configurations(summary: pd.DataFrame) -> pd.DataFrame:
    """Best strategy of each model by mean validation macro F1 (the test set plays no part)."""
    complete = summary[summary["folds_missing"] == 0]
    return complete.sort_values("f1_macro_mean", ascending=False).groupby("model").head(1)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Recompute metrics and write results/summary.csv")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results" / "runs"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    args = parser.parse_args(argv)
    config = load_config()
    out = Path(args.out_dir)
    merge_runs(Path(args.results_dir), config, out / "runs.csv")
    runs = load_runs(Path(args.results_dir))
    runs = runs[(runs["status"] == REAL_STATUS) & (runs["data_source"] != SYNTHETIC_SOURCE)]
    if runs.empty:
        print("No completed real runs yet.")
        return
    for issue in check_grid(runs):
        print("WARNING:", issue)
    per_run = run_metrics(Path(args.results_dir), runs)
    per_run.to_csv(out / "per_run_metrics.csv", index=False)
    summary = summarize(per_run, config["data"]["n_folds"])
    summary.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    print(summary[["model", "strategy", "folds_done", "f1_macro_mean", "f1_macro_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
