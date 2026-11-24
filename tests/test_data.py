"""Step 2–3 checks on generated fundus-like images (SIMULATED, written to a temp folder)."""

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from drtl.config import load_config
from drtl.data import (build_cache, discover_images, duplicate_groups, grade_from_folder, inspect_images,
                       make_fold_loaders, make_splits, make_test_loader)
from drtl.transforms import CropRetina, build_eval_transform, build_train_transform

CONFIG = load_config(overrides={"training": {"num_workers": 0}})
FOLDERS = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]


def fake_fundus(seed, size=(300, 260)):
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", size, (0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((30, 10, 270, 250), fill=tuple(int(v) for v in rng.integers(90, 200, 3)))
    for _ in range(12):  # vessel-like lines: what makes real retinas different from each other
        pts = [tuple(int(v) for v in rng.integers(40, 260, 2)) for _ in range(3)]
        d.line(pts, fill=tuple(int(v) for v in rng.integers(20, 90, 3)), width=3)
    for _ in range(int(rng.integers(3, 12))):
        x, y = rng.integers(60, 240, 2)
        d.ellipse((x, y, x + 8, y + 8), fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    return im


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    for label, folder in enumerate(FOLDERS):
        (root / folder).mkdir()
        for i in range(14):
            fake_fundus(label * 100 + i).save(root / folder / f"img{label}_{i}.jpg", quality=95)
    fake_fundus(0).save(root / "No_DR" / "copy_of_img0_0.jpg", quality=95)  # a duplicate
    (root / "Mild" / "broken.jpg").write_bytes(b"not an image")
    return root


def test_grade_names():
    assert [grade_from_folder(f) for f in FOLDERS] == [0, 1, 2, 3, 4]
    assert grade_from_folder("Healthy") == 0 and grade_from_folder("Mild DR") == 1


def test_crop_removes_black_border_and_squares():
    out = CropRetina()(fake_fundus(1))
    assert out.size[0] == out.size[1] and out.size[0] < 300


def test_eval_transform_shape_normalisation_and_determinism():
    t = build_eval_transform(224)
    x = t(fake_fundus(2))
    assert x.shape == (3, 224, 224) and torch.equal(x, t(fake_fundus(2)))
    batch = torch.stack([t(fake_fundus(s)) for s in range(8)])
    assert abs(batch.mean().item()) < 1.5


def test_train_transform_is_random():
    t = build_train_transform(224)
    torch.manual_seed(0)
    assert not torch.equal(t(fake_fundus(3)), t(fake_fundus(3)))


def test_pipeline_splits_without_leakage(dataset, tmp_path):
    table = inspect_images(discover_images(dataset), dataset)
    assert (~table["readable"]).sum() == 1
    table = table[table["readable"]].reset_index(drop=True)
    table["group_id"] = duplicate_groups(table)
    a = table.loc[table["path"] == "No_DR/img0_0.jpg", "group_id"].item()
    b = table.loc[table["path"] == "No_DR/copy_of_img0_0.jpg", "group_id"].item()
    assert a == b
    splits = make_splits(table, 0.15, 5, 42)
    test_groups = set(splits.loc[splits.split == "test", "group_id"])
    assert not test_groups & set(splits.loc[splits.split == "cv", "group_id"])
    assert set(splits.loc[splits.split == "cv", "fold"]) == set(range(5))
    assert 0.1 < (splits.split == "test").mean() < 0.2
    assert splits.equals(make_splits(table, 0.15, 5, 42))   # same seed → same split

    cache = build_cache(splits, dataset, tmp_path / "cache", 224)
    train, val = make_fold_loaders(CONFIG, splits, 0, cache, batch_size=4)
    images, labels = next(iter(val))
    assert images.shape == (4, 3, 224, 224) and labels.dtype == torch.long
    assert len(val.dataset) == ((splits.split == "cv") & (splits.fold == 0)).sum()
    assert len(make_test_loader(CONFIG, splits, cache).dataset) == (splits.split == "test").sum()
    # the cached image goes through the same eval transform as the raw one
    raw = build_eval_transform(224)(Image.open(dataset / "Mild/img1_0.jpg"))
    cached = build_eval_transform(224)(Image.open(cache / "Mild/img1_0.jpg.png"))
    assert (raw - cached).abs().mean() < 0.05
