"""Dataset discovery, duplicate control, the frozen split and the PyTorch loaders (Steps 2–3).

Split protocol (report §4.2): a stratified 15 % test set frozen once, then 5 stratified folds on the
remaining 85 %, seed 42. Exact and near-duplicate images are grouped (`group_id`) and always land
in the same part, so the same photo can never be in training and test at once.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from drtl.transforms import CropRetina, build_eval_transform, build_train_transform

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
CACHE_INFO = "CACHE_INFO.json"

# Folder name → ICDR grade. Checked in order; "no dr"/"healthy" before "dr" alone.
_GRADE_PATTERNS = [
    (0, r"(^|[^a-z])(no[ _-]?dr|healthy|normal|nodr|grade[ _-]?0|^0$)"),
    (4, r"(prolif|grade[ _-]?4|^4$)"),
    (3, r"(severe|grade[ _-]?3|^3$)"),
    (2, r"(moderate|grade[ _-]?2|^2$)"),
    (1, r"(mild|grade[ _-]?1|^1$)"),
]


def grade_from_folder(name: str) -> Optional[int]:
    """ICDR grade from a class folder name such as 'Mild DR', 'Proliferate_DR' or '0'."""
    lowered = name.lower().strip()
    for grade, pattern in _GRADE_PATTERNS:
        if re.search(pattern, lowered):
            return grade
    return None


# --------------------------------------------------------------------------- discovery & checks


def discover_images(root: Path) -> pd.DataFrame:
    """Every image under `root`, labelled by the nearest parent folder that names a grade."""
    root = Path(root)
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
        rel = path.relative_to(root)
        label, class_name = None, None
        for part in reversed(rel.parts[:-1]):
            label = grade_from_folder(part)
            if label is not None:
                class_name = part
                break
        rows.append({"image_id": rel.with_suffix("").as_posix().replace("/", "__"),
                     "path": rel.as_posix(), "label": label, "class_name": class_name,
                     "source_split": rel.parts[0] if len(rel.parts) > 2 else ""})
    return pd.DataFrame(rows)


def perceptual_hash(image: Image.Image, size: int = 16) -> int:
    """Difference hash of the inside of the retinal disc (256 bits).

    The disc outline is the same on every photo, so the hash looks only at the central 60 % of the
    cropped retina, where vessels, optic disc and lesions differ. Robust to resizing, JPEG
    recompression and global brightness changes.
    """
    disc = CropRetina()(image).convert("L")
    side = disc.size[0]
    m = int(side * 0.2)
    center = disc.crop((m, m, side - m, side - m)).resize((size + 1, size), Image.BILINEAR)
    pixels = np.asarray(center, dtype=np.float32)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def inspect_images(table: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Add width, height, mode, md5 and perceptual hash; mark unreadable files."""
    info = []
    for rel in table["path"]:
        try:
            raw = (Path(root) / rel).read_bytes()
            with Image.open(Path(root) / rel) as im:
                im.load()
                info.append({"width": im.width, "height": im.height, "mode": im.mode, "readable": True,
                             "md5": hashlib.md5(raw).hexdigest(), "phash": perceptual_hash(im)})
        except Exception:  # corrupted or not an image
            info.append({"width": None, "height": None, "mode": None, "readable": False, "md5": None, "phash": None})
    return pd.concat([table.reset_index(drop=True), pd.DataFrame(info)], axis=1)


def duplicate_groups(table: pd.DataFrame, max_distance: int = 10) -> pd.Series:
    """Group id per image: identical files and near-identical retinas share one group.

    Near-duplicates = perceptual hashes within `max_distance` bits of 256 (union-find).
    """
    parent = list(range(len(table)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for _, idx in table.groupby("md5").groups.items():
        idx = list(idx)
        for j in idx[1:]:
            union(idx[0], j)
    hashes = table["phash"].tolist()
    valid = [i for i, h in enumerate(hashes) if h is not None and not pd.isna(h)]
    arr = np.array([[int(b) for b in format(int(hashes[i]), "0256b")] for i in valid], dtype=np.uint8)
    for a in range(len(valid)):  # O(n²) on bit arrays: fine for a few thousand images
        dist = (arr[a + 1:] != arr[a]).sum(axis=1)
        for b in np.where(dist <= max_distance)[0]:
            union(valid[a], valid[a + 1 + b])
    roots = [find(i) for i in range(len(table))]
    return pd.Series([table["image_id"].iloc[r] for r in roots], index=table.index)


def make_splits(table: pd.DataFrame, test_fraction: float = 0.15, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Add `split` (cv/test) and `fold` (0..n-1, −1 for test), stratified by label, grouped by `group_id`.

    The test part is the first fold of a stratified group split with round(1/test_fraction) parts
    (≈ 15 %), so groups never cross the test boundary either.
    """
    table = table.reset_index(drop=True).copy()
    n_parts = round(1 / test_fraction)
    outer = StratifiedGroupKFold(n_splits=n_parts, shuffle=True, random_state=seed)
    _, test_idx = next(outer.split(table, table["label"], table["group_id"]))
    table["split"] = "cv"
    table.loc[test_idx, "split"] = "test"
    table["fold"] = -1
    cv = table[table["split"] == "cv"]
    inner = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(inner.split(cv, cv["label"], cv["group_id"])):
        table.loc[cv.index[val_idx], "fold"] = k
    return table


SPLIT_COLUMNS = ["image_id", "path", "label", "class_name", "group_id", "split", "fold"]


def load_splits(path: Path) -> pd.DataFrame:
    """The frozen split table (docs/team/INTERFACES.md §1)."""
    table = pd.read_csv(path)
    missing = set(SPLIT_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    return table


# --------------------------------------------------------------------------- cache


def build_cache(splits: pd.DataFrame, image_root: Path, cache_root: Path, image_size: int) -> Path:
    """Write every image once, retina-cropped and resized, as PNG (lossless).

    Decoding multi-megapixel JPEGs every epoch would make the GPU wait on the CPU. The cached
    image is exactly what `build_eval_transform` would produce before normalisation, and the
    crop/resize steps are idempotent on it, so training and the demo still share one pipeline.
    """
    cache_root = Path(cache_root)
    crop = CropRetina()
    for rel in splits["path"]:
        target = cache_root / (rel + ".png")
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(Path(image_root) / rel) as im:
            crop(im).resize((image_size, image_size), Image.BICUBIC).save(target)
    (cache_root / CACHE_INFO).write_text(json.dumps({"image_size": image_size, "count": len(splits),
                                                     "source": str(image_root)}))
    return cache_root


# --------------------------------------------------------------------------- PyTorch


class FundusDataset(Dataset):
    """Images listed in a split table, read from `root` (raw dataset or cache)."""

    def __init__(self, table: pd.DataFrame, root: Path, transform):
        self.table = table.reset_index(drop=True)
        self.root = Path(root)
        self.suffix = ".png" if (self.root / CACHE_INFO).exists() else ""
        self.transform = transform
        self.labels = self.table["label"].to_numpy(dtype=int)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        with Image.open(self.root / (self.table["path"].iloc[i] + self.suffix)) as im:
            image = im.convert("RGB")
        return self.transform(image), int(self.labels[i])


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _loader(dataset, config, batch_size, shuffle) -> DataLoader:
    workers = int(config["training"].get("num_workers", 2))
    generator = torch.Generator().manual_seed(int(config["seed"]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      worker_init_fn=_seed_worker, generator=generator,
                      persistent_workers=workers > 0, pin_memory=torch.cuda.is_available())


def make_fold_loaders(config: Dict[str, Any], splits: pd.DataFrame, fold: int, image_root: Path,
                      batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """Train = CV rows outside `fold` (augmented, shuffled); val = CV rows of `fold`."""
    size = config["data"]["image_size"]
    batch_size = batch_size or config["training"]["batch_size"]
    cv = splits[splits["split"] == "cv"]
    train = FundusDataset(cv[cv["fold"] != fold], image_root, build_train_transform(size))
    val = FundusDataset(cv[cv["fold"] == fold], image_root, build_eval_transform(size))
    return _loader(train, config, batch_size, True), _loader(val, config, batch_size, False)


def make_test_loader(config: Dict[str, Any], splits: pd.DataFrame, image_root: Path,
                     batch_size: Optional[int] = None) -> DataLoader:
    """The frozen test set. Called only by the final evaluation, after model selection."""
    test = FundusDataset(splits[splits["split"] == "test"], image_root,
                         build_eval_transform(config["data"]["image_size"]))
    return _loader(test, config, batch_size or config["training"]["batch_size"], False)


def split_report(splits: pd.DataFrame, class_names: List[str]) -> pd.DataFrame:
    """Images per class in test and in each fold (counts and %), for the paper's Dataset table."""
    part = np.where(splits["split"] == "test", "test", "fold " + splits["fold"].astype(str))
    counts = pd.crosstab(splits["label"].map(dict(enumerate(class_names))), part)
    return counts
