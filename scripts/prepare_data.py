"""Step 2: explore the dataset, control duplicates, freeze the split, build the image cache.

    python scripts/prepare_data.py --raw data/raw

Writes data/splits/folds.csv (committed, frozen), results/eda/*.csv|json and figures (committed),
data/cache/224/ (ignored by Git). Refuses to overwrite an existing folds.csv unless --force.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from drtl.config import PROJECT_ROOT, load_config
from drtl.data import (SPLIT_COLUMNS, build_cache, discover_images, duplicate_groups, inspect_images,
                       make_splits, split_report)
from drtl.results import file_hash
from drtl.transforms import CropRetina


def _inspect_chunk(args):
    chunk, root = args
    return inspect_images(chunk, Path(root))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=str(PROJECT_ROOT / "data" / "raw"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    config = load_config()
    raw = Path(a.raw)
    names = config["data"]["class_names"]
    out = PROJECT_ROOT / "results" / "eda"
    out.mkdir(parents=True, exist_ok=True)
    splits_file = PROJECT_ROOT / config["data"]["splits_file"]
    if splits_file.exists() and not a.force:
        raise SystemExit(f"{splits_file} exists and is frozen; use --force only before any real run")

    table = discover_images(raw)
    print(f"{len(table)} image files found under {raw}")
    unlabeled = table[table["label"].isna()]
    if len(unlabeled):
        print(f"WARNING: {len(unlabeled)} files in folders without a grade name, ignored:",
              unlabeled["path"].head(5).tolist())
    table = table.dropna(subset=["label"]).astype({"label": int})

    chunks = [(table.iloc[idx], str(raw)) for idx in np.array_split(np.arange(len(table)), max(1, a.workers * 4)) if len(idx)]
    with ProcessPoolExecutor(a.workers) as pool:
        table = pd.concat(pool.map(_inspect_chunk, chunks), ignore_index=True)
    bad = table[~table["readable"]]
    table = table[table["readable"]].reset_index(drop=True)
    table["group_id"] = duplicate_groups(table)
    group_sizes = table["group_id"].value_counts()
    dup = table[table["group_id"].map(group_sizes) > 1]
    conflicting = dup.groupby("group_id")["label"].nunique()
    conflicting = conflicting[conflicting > 1]
    # A duplicate pair with two different grades cannot be graded consistently: drop both copies.
    dropped_conflicts = table[table["group_id"].isin(conflicting.index)]
    table = table[~table["group_id"].isin(conflicting.index)].reset_index(drop=True)

    splits = make_splits(table, config["data"]["test_fraction"], config["data"]["n_folds"], config["seed"])
    splits_file.parent.mkdir(parents=True, exist_ok=True)
    splits[SPLIT_COLUMNS].to_csv(splits_file, index=False, lineterminator="\n")

    counts = table["label"].value_counts().sort_index()
    report = {
        "images_found": int(len(table) + len(bad) + len(dropped_conflicts)),
        "unreadable": bad["path"].tolist(),
        "class_folders": sorted(table["class_name"].unique().tolist()),
        "source_splits": sorted(table["source_split"].unique().tolist()),
        "per_class": {names[i]: int(counts.get(i, 0)) for i in range(len(names))},
        "duplicate_groups": int((group_sizes > 1).sum()),
        "images_in_duplicate_groups": int(len(dup)),
        "largest_duplicate_group": int(group_sizes.max()),
        "conflicting_label_duplicates_dropped": dropped_conflicts["path"].tolist(),
        "patient_ids": "not available in this dataset: group_id = duplicate cluster",
        "size_px": {"width": table["width"].describe().round(1).to_dict(),
                    "height": table["height"].describe().round(1).to_dict()},
        "modes": table["mode"].value_counts().to_dict(),
        "test_fraction_actual": round(float((splits["split"] == "test").mean()), 4),
        "splits_hash": file_hash(splits_file),
    }
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2))
    split_report(splits, names).to_csv(out / "split_counts.csv")
    table.drop(columns=["phash"]).to_csv(out / "images_inspected.csv", index=False)
    _figures(table, raw, names, out)
    print(json.dumps({k: v for k, v in report.items() if k not in ("unreadable", "conflicting_label_duplicates_dropped")}, indent=2))

    cache = build_cache(splits, raw, PROJECT_ROOT / "data" / "cache" / str(config["data"]["image_size"]),
                        config["data"]["image_size"])
    print(f"Frozen split: {splits_file} (hash {report['splits_hash']}) · cache: {cache}")


def _figures(table, raw, names, out):
    counts = table["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar([names[i] for i in counts.index], counts.values, color="#3b6ea5")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("images"); ax.set_title("Images per ICDR grade"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "class_distribution.png", dpi=150); plt.close(fig)

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(len(names), 4, figsize=(8, 2.1 * len(names)))
    for label in range(len(names)):
        rows = table[table["label"] == label]
        picks = rows.sample(min(4, len(rows)), random_state=42) if len(rows) else rows
        for j in range(4):
            ax = axes[label, j]; ax.axis("off")
            if j < len(picks):
                with Image.open(raw / picks["path"].iloc[j]) as im:
                    ax.imshow(CropRetina()(im).resize((200, 200)))
            if j == 0:
                ax.set_title(names[label], loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "gallery_by_grade.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
