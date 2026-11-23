"""Step 4 checks: freezing strategies and optimizer groups (fiche ALAA EDDINE)."""

import pytest
import torch

from drtl.config import load_config
from drtl.models import (TRANSFER_STRATEGIES, build_for_run, count_parameters, parameter_groups,
                         parameter_table, set_train_mode, strategies_for, strategy_settings)

CONFIG = load_config()
BACKBONES = [m for m in CONFIG["models"] if m != "scratch_cnn"]


def test_grid_has_13_configurations():
    assert sum(len(strategies_for(m)) for m in CONFIG["models"]) == 13
    assert strategies_for("scratch_cnn") == ["scratch"]


@pytest.mark.parametrize("model", BACKBONES)
def test_trainable_parameters_grow_with_each_strategy(model):
    counts = [count_parameters(build_for_run(CONFIG, model, s, pretrained=False))["trainable"]
              for s in TRANSFER_STRATEGIES]
    assert 0 < counts[0] < counts[1] < counts[2]


def test_scratch_cnn_trains_everything():
    counts = count_parameters(build_for_run(CONFIG, "scratch_cnn", "scratch"))
    assert counts["trainable"] == counts["total"]


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        strategy_settings(CONFIG, "scratch_cnn", "feature_extraction")


@pytest.mark.parametrize("model", ["mobilenetv4", "vit_b16"])
@pytest.mark.parametrize("strategy", ["feature_extraction", "partial_finetuning"])
def test_optimizer_step_leaves_frozen_weights_and_batchnorm_statistics_unchanged(model, strategy):
    torch.manual_seed(0)
    net = build_for_run(CONFIG, model, strategy, pretrained=False)
    settings = strategy_settings(CONFIG, model, strategy)
    frozen = {n: p.detach().clone() for n, p in net.named_parameters() if not p.requires_grad}
    frozen_bn = {n: b.detach().clone() for n, b in net.named_buffers()
                 if "running_mean" in n and n.rsplit(".", 1)[0] + ".weight" in frozen}
    trainable_before = {n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW(parameter_groups(net, settings["head_lr"], settings["backbone_lr"], 1e-4))

    set_train_mode(net)
    loss = torch.nn.functional.cross_entropy(net(torch.randn(4, 3, 224, 224)), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    opt.step()

    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())
    assert all(torch.equal(params[n], v) for n, v in frozen.items())
    assert all(torch.equal(buffers[n], v) for n, v in frozen_bn.items())
    assert any(not torch.equal(params[n], v) for n, v in trainable_before.items())


def test_parameter_groups_use_two_learning_rates_only_when_backbone_trains():
    fe = build_for_run(CONFIG, "mobilenetv4", "feature_extraction", pretrained=False)
    assert [g["name"] for g in parameter_groups(fe, 1e-3, 0.0, 1e-4)] == ["head"]
    full = build_for_run(CONFIG, "mobilenetv4", "full_finetuning", pretrained=False)
    groups = parameter_groups(full, 1e-3, 3e-5, 1e-4)
    assert [(g["name"], g["lr"]) for g in groups] == [("head", 1e-3), ("backbone", 3e-5)]


def test_parameter_table_lists_every_configuration():
    rows = parameter_table(CONFIG, ["scratch_cnn", "mobilenetv4"])
    assert [(r["model"], r["strategy"]) for r in rows][:2] == [("scratch_cnn", "scratch"),
                                                                ("mobilenetv4", "feature_extraction")]
