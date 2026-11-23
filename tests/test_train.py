"""Step 5 checks on SIMULATED data: the loop learns, resumes and writes the agreed run record."""

import json

import numpy as np
import pytest
import torch

from drtl.config import load_config
from drtl.fixtures import SYNTHETIC_SOURCE, make_synthetic_fold_loaders, synthetic_splits
from drtl.models import build_for_run
from drtl.results import RUN_COLUMNS, expected_run_ids, file_hash, merge_runs, run_id
from drtl.run import run_experiment
from drtl.train import class_weights, fit, warmup_cosine


def test_class_weights_use_inverse_frequency():
    w = class_weights([0, 0, 0, 1], num_classes=3)
    assert torch.allclose(w, torch.tensor([4 / 9, 4 / 3, 0.0]))


def test_schedule_warms_up_then_decays_to_zero():
    f = warmup_cosine(total_steps=10, warmup_steps=2)
    assert f(0) == 0.5 and f(1) == 1.0 and f(2) == 1.0
    assert f(9) < 0.1 and f(10) == 0.0


def test_hash_ignores_windows_line_endings(tmp_path):
    (tmp_path / "a.csv").write_bytes(b"x,y\n1,2\n")
    (tmp_path / "b.csv").write_bytes(b"x,y\r\n1,2\r\n")
    assert file_hash(tmp_path / "a.csv") == file_hash(tmp_path / "b.csv")


def test_grid_expects_65_runs():
    ids = expected_run_ids(load_config())
    assert len(ids) == len(set(ids)) == 65
    assert run_id("vit_b16", "feature_extraction", 0) in ids


def test_overfits_32_simulated_images(tmp_path):
    """Planning report Step 5: on 32 images the training loss must go towards 0."""
    config = load_config(overrides={"data": {"image_size": 64}, "training": {
        "max_epochs": 25, "early_stopping_patience": 100, "batch_size": 8, "warmup_epochs": 0}})
    splits = synthetic_splits(n_per_class=8)
    splits = splits.assign(split="cv", fold=np.where(np.arange(len(splits)) % 5 == 0, 0, 1))
    train, val = make_synthetic_fold_loaders(config, splits, fold=0, batch_size=8)
    assert len(train.dataset) == 32
    torch.manual_seed(0)
    net = build_for_run(config, "scratch_cnn", "scratch")
    summary = fit(net, train, val, train_labels=train.dataset.labels, config=config,
                  lrs={"head_lr": 3e-3, "backbone_lr": 3e-3}, device=torch.device("cpu"), out_dir=tmp_path,
                  metadata={"run_id": "overfit"}, resume=False, log=lambda _: None)
    history = summary["best"]
    first = json.loads(json.dumps(summary))  # summary is serialisable
    losses = [float(l.split(",")[1]) for l in (tmp_path / "history.csv").read_text().splitlines()[1:]]
    assert losses[-1] < 0.1 * losses[0] and losses[-1] < 0.15
    assert first["epochs_run"] == 25 and history["val_f1_macro"] > 0.5


def test_smoke_run_resumes_and_writes_the_record(tmp_path):
    kwargs = dict(synthetic=True, smoke=True, output_root=tmp_path, device="cpu")
    first = run_experiment("scratch_cnn", "scratch", 0, overrides={"training": {"max_epochs": 1}}, **kwargs)
    assert first["epochs_run"] == 1 and first["status"] == "smoke"
    second = run_experiment("scratch_cnn", "scratch", 0, **kwargs)   # resumes at epoch 2
    assert second["epochs_run"] == 2
    record_dir = tmp_path / "smoke" / "scratch_cnn__scratch__fold0" / "record"
    record = json.loads((record_dir / "scratch_cnn__scratch__fold0.json").read_text())
    assert set(RUN_COLUMNS) <= set(record) and record["data_source"] == SYNTHETIC_SOURCE
    assert record["splits_hash"] == "SIMULATED"
    preds = (record_dir / record["predictions_file"]).read_text().splitlines()
    assert preds[0] == "label,pred,prob_0,prob_1,prob_2,prob_3,prob_4"


def test_resume_refuses_a_checkpoint_from_another_run(tmp_path):
    run_experiment("scratch_cnn", "scratch", 0, synthetic=True, smoke=True, output_root=tmp_path, device="cpu")
    out = tmp_path / "smoke" / "scratch_cnn__scratch__fold0"
    state = torch.load(out / "last.pt", weights_only=False)
    state["metadata"]["config_hash"] = "other"
    torch.save(state, out / "last.pt")
    with pytest.raises(RuntimeError, match="another run"):
        run_experiment("scratch_cnn", "scratch", 0, synthetic=True, smoke=True, output_root=tmp_path, device="cpu")


def test_merge_ignores_simulated_runs_and_marks_missing(tmp_path):
    run_experiment("scratch_cnn", "scratch", 1, synthetic=True, smoke=True, output_root=tmp_path, device="cpu")
    record_dir = tmp_path / "smoke" / "scratch_cnn__scratch__fold1" / "record"
    table = merge_runs(record_dir, load_config(), tmp_path / "runs.csv")
    assert len(table) == 65 and set(table["status"]) == {"missing"}
