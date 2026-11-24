"""The contract between the shared pipeline and drtl.data (docs/team/INTERFACES.md).

Until feat/data-splits is merged, a stand-in module with the agreed signatures (backed by
SIMULATED images) checks that a real-mode run writes a complete record to results/runs.
When the real drtl.data exists, the same checks run against it on a SIMULATED split table.
"""

import json
import sys
import types

import pytest

from drtl.fixtures import make_synthetic_fold_loaders, synthetic_splits
from drtl.results import RUN_COLUMNS, file_hash
from drtl.run import run_experiment

SPLIT_COLUMNS = {"image_id", "path", "label", "class_name", "split", "fold"}


@pytest.fixture
def stand_in_data(monkeypatch, tmp_path):
    import pandas as pd

    module = types.ModuleType("drtl.data")
    module.load_splits = lambda path: pd.read_csv(path)
    module.make_fold_loaders = lambda config, splits, fold, image_root, batch_size: \
        make_synthetic_fold_loaders(config, splits, fold, batch_size)
    monkeypatch.setitem(sys.modules, "drtl.data", module)
    splits_file = tmp_path / "folds.csv"
    synthetic_splits().to_csv(splits_file, index=False)
    return splits_file


def test_split_table_has_the_agreed_columns():
    table = synthetic_splits()
    assert SPLIT_COLUMNS <= set(table.columns)
    assert set(table.loc[table.split == "test", "fold"]) == {-1}
    assert set(table.loc[table.split == "cv", "fold"]) == set(range(5))


def test_real_mode_run_writes_a_complete_record(stand_in_data, tmp_path):
    record = run_experiment("scratch_cnn", "scratch", 0, splits_file=stand_in_data,
                            overrides={"training": {"max_epochs": 1, "batch_size": 8}},
                            output_root=tmp_path / "outputs", results_dir=tmp_path / "runs", device="cpu")
    saved = json.loads((tmp_path / "runs" / "scratch_cnn__scratch__fold0.json").read_text())
    assert set(RUN_COLUMNS) <= set(saved)
    assert saved["status"] == "completed" and saved["data_source"] == "jockeroika/diabetic-retinopathy"
    assert saved["splits_hash"] == file_hash(stand_in_data) and len(saved["config_hash"]) == 12
    assert (tmp_path / "runs" / saved["predictions_file"]).exists()
    again = run_experiment("scratch_cnn", "scratch", 0, splits_file=stand_in_data,
                           output_root=tmp_path / "outputs", results_dir=tmp_path / "runs", device="cpu")
    assert again["started_at"] == saved["started_at"]   # completed runs are not retrained


def test_test_images_never_reach_the_validation_loader(stand_in_data, monkeypatch, tmp_path):
    leaky = lambda config, splits, fold, image_root, batch_size: make_synthetic_fold_loaders(
        config, splits.assign(split="cv", fold=splits.fold.clip(lower=0)), fold, batch_size)
    monkeypatch.setattr(sys.modules["drtl.data"], "make_fold_loaders", leaky)
    with pytest.raises(RuntimeError, match="does not match"):
        run_experiment("scratch_cnn", "scratch", 0, splits_file=stand_in_data, smoke=True,
                       output_root=tmp_path, device="cpu")


def test_missing_data_module_gives_a_clear_message(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "drtl.data", None)
    with pytest.raises(RuntimeError, match="feat/data-splits"):
        run_experiment("scratch_cnn", "scratch", 0, output_root=tmp_path, device="cpu")
