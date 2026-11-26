"""SIMULATED data for pipeline tests and CPU smoke runs. Never used for reported results.

The images are random noise with a class-dependent bright square, so a model can learn them
and the training loop can be checked end to end before the real split exists. Every run
trained on this data is recorded with `data_source = "synthetic-SIMULATED"` and is excluded
from the merged results table.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SYNTHETIC_SOURCE = "synthetic-SIMULATED"


def synthetic_splits(n_per_class: int = 15, num_classes: int = 5, n_folds: int = 5,
                     test_fraction: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """A table with the same columns as data/splits/folds.csv (docs/team/INTERFACES.md)."""
    rng = np.random.default_rng(seed)
    rows = []
    for label in range(num_classes):
        ids = [f"sim_{label}_{i:03d}" for i in range(n_per_class)]
        rng.shuffle(ids)
        n_test = max(1, round(test_fraction * n_per_class))
        for j, image_id in enumerate(ids):
            test = j < n_test
            rows.append({"image_id": image_id, "path": f"SIMULATED/{label}/{image_id}.png",
                         "label": label, "class_name": f"SIMULATED_{label}", "group_id": image_id,
                         "split": "test" if test else "cv", "fold": -1 if test else (j - n_test) % n_folds})
    return pd.DataFrame(rows)


class SyntheticFundus(Dataset):
    """Deterministic fake images: image i is generated from its row index, not stored."""

    def __init__(self, table: pd.DataFrame, image_size: int, seed: int = 42):
        self.table = table.reset_index(drop=True)
        self.image_size = image_size
        self.seed = seed
        self.labels = self.table["label"].to_numpy()

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        label = int(self.labels[i])
        g = torch.Generator().manual_seed(self.seed * 1000 + i)
        image = torch.randn(3, self.image_size, self.image_size, generator=g) * 0.5
        s = self.image_size // 5
        image[:, label * s:(label + 1) * s, label * s:(label + 1) * s] += 2.0  # the class "lesion"
        return image, label


def make_synthetic_fold_loaders(config: Dict[str, Any], splits: pd.DataFrame, fold: int,
                                batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Same signature and batch format as drtl.data.make_fold_loaders, on SIMULATED images."""
    cv = splits[splits["split"] == "cv"]
    size = config["data"]["image_size"]
    g = torch.Generator().manual_seed(config["seed"])
    train = DataLoader(SyntheticFundus(cv[cv["fold"] != fold], size), batch_size=batch_size,
                       shuffle=True, generator=g)
    val = DataLoader(SyntheticFundus(cv[cv["fold"] == fold], size), batch_size=batch_size)
    return train, val
