"""Build the five networks compared in the project.

Four are ImageNet-pretrained backbones from timm; the fifth is a small CNN trained
from scratch, which serves as the baseline that shows what transfer learning adds.
Step 4 adds the three transfer strategies (which layers train) and the two learning-rate
parameter groups (new head vs pretrained backbone).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import timm
import torch
from timm.models import group_parameters
from torch import nn

SCRATCH_CNN = "scratch_cnn"
SCRATCH_STRATEGY = "scratch"
TRANSFER_STRATEGIES = ("feature_extraction", "partial_finetuning", "full_finetuning")


class ScratchCNN(nn.Module):
    """Four conv blocks (conv-BN-ReLU-pool) followed by global pooling and a linear head.

    Kept deliberately small (~0.4M parameters): with a few thousand fundus images a
    bigger network trained from zero would mostly overfit.
    """

    def __init__(self, num_classes: int, widths=(32, 64, 128, 256)):
        super().__init__()
        layers = []
        in_channels = 3
        for width in widths:
            layers += [
                nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_channels = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return self.head(x)


    def get_classifier(self) -> nn.Module:
        """Same accessor as timm models, so strategy code treats all five networks alike."""
        return self.head


def build_model(timm_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a model with a fresh `num_classes`-way classification head.

    `pretrained=False` is used by the tests and the setup check, so they run offline
    and fast; real experiments always use the ImageNet weights.
    """
    if timm_name == SCRATCH_CNN:
        return ScratchCNN(num_classes)
    return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Total and trainable parameter counts (reported in the results table, p.3)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# --------------------------------------------------------------------------- strategies


def strategies_for(model_key: str) -> List[str]:
    """The from-scratch CNN has one strategy; every pretrained backbone has three (report §5.2)."""
    return [SCRATCH_STRATEGY] if model_key == SCRATCH_CNN else list(TRANSFER_STRATEGIES)


def strategy_settings(config: Dict[str, Any], model_key: str, strategy: str) -> Dict[str, float]:
    """Learning rates (and unfreeze fraction) for one configuration, read from the shared YAML."""
    if strategy not in strategies_for(model_key):
        raise ValueError(f"{model_key!r} does not support strategy {strategy!r}; "
                         f"choose one of {strategies_for(model_key)}")
    table = config["baseline"] if strategy == SCRATCH_STRATEGY else config["strategies"]
    return dict(table[strategy])


def head_parameter_names(model: nn.Module) -> List[str]:
    """Names of the parameters of the new 5-way classification layer."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    return [name for name, p in model.named_parameters() if id(p) in head_ids]


def backbone_groups(model: nn.Module) -> List[List[str]]:
    """Backbone parameter names, grouped by depth from input to output, head excluded.

    timm already knows the block structure of each architecture (`group_matcher`), so we reuse
    it rather than hard-coding layer names: ViT → patch embedding + 12 blocks, DenseNet-201 →
    one group per dense layer, VGG19 → one group per conv/BN layer and the two FC layers,
    MobileNetV4 → one group per inverted-residual block.
    """
    head = set(head_parameter_names(model))
    if isinstance(model, ScratchCNN):
        names = [n for n, _ in model.named_parameters() if n not in head]
        blocks: Dict[str, List[str]] = {}
        for n in names:  # features.<index>.<param>; four layers per conv block
            blocks.setdefault(str(int(n.split(".")[1]) // 4), []).append(n)
        return list(blocks.values())
    grouped = group_parameters(model, model.group_matcher(coarse=False))
    groups = [[n for n in names if n not in head] for _, names in sorted(grouped.items())]
    return [g for g in groups if g]


def apply_strategy(model: nn.Module, strategy: str, unfreeze_fraction: float = 0.33) -> nn.Module:
    """Set `requires_grad` for one strategy and return the same model.

    feature_extraction: only the new head learns; the ImageNet features stay fixed.
    partial_finetuning: head + the last `unfreeze_fraction` of backbone groups (the most
        task-specific features), early generic filters stay fixed.
    full_finetuning / scratch: every parameter learns.
    """
    if strategy in ("full_finetuning", SCRATCH_STRATEGY):
        for p in model.parameters():
            p.requires_grad = True
        return model
    if strategy not in ("feature_extraction", "partial_finetuning"):
        raise ValueError(f"unknown strategy {strategy!r}")

    for p in model.parameters():
        p.requires_grad = False
    trainable = set(head_parameter_names(model))
    if strategy == "partial_finetuning":
        groups = backbone_groups(model)
        n_open = max(1, round(unfreeze_fraction * len(groups)))
        for group in groups[-n_open:]:
            trainable.update(group)
    for name, p in model.named_parameters():
        if name in trainable:
            p.requires_grad = True
    return model


def set_train_mode(model: nn.Module) -> nn.Module:
    """`model.train()`, except BatchNorm layers whose weights are frozen stay in eval mode.

    Otherwise a frozen backbone would still drift: BatchNorm running statistics update in train
    mode even without gradients, which silently breaks feature extraction.
    """
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            params = list(module.parameters(recurse=False))
            if params and not any(p.requires_grad for p in params):
                module.eval()
    return model


def parameter_groups(model: nn.Module, head_lr: float, backbone_lr: float,
                     weight_decay: float) -> List[Dict[str, Any]]:
    """AdamW parameter groups: new head at `head_lr`, trainable backbone at `backbone_lr`.

    The head starts random and needs a larger step; pretrained layers only need small
    corrections (report §5.2). Frozen parameters are left out of the optimizer entirely.
    """
    head = set(head_parameter_names(model))
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (head_params if name in head else backbone_params).append(p)
    groups = [{"name": "head", "params": head_params, "lr": head_lr, "weight_decay": weight_decay}]
    if backbone_params:
        if backbone_lr <= 0:
            raise ValueError("trainable backbone parameters need backbone_lr > 0")
        groups.append({"name": "backbone", "params": backbone_params, "lr": backbone_lr,
                       "weight_decay": weight_decay})
    return groups


def build_for_run(config: Dict[str, Any], model_key: str, strategy: str,
                  pretrained: bool = True) -> nn.Module:
    """Model for one configuration of the grid, with the strategy already applied."""
    settings = strategy_settings(config, model_key, strategy)
    model = build_model(config["models"][model_key], len(config["data"]["class_names"]),
                        pretrained=pretrained and model_key != SCRATCH_CNN)
    return apply_strategy(model, strategy, settings.get("unfreeze_fraction", 0.33))


def parameter_table(config: Dict[str, Any], model_keys: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Total / trainable parameters for the 13 configurations (table for the paper)."""
    rows = []
    for key in model_keys or list(config["models"]):
        for strategy in strategies_for(key):
            model = build_for_run(config, key, strategy, pretrained=False)
            counts = count_parameters(model)
            rows.append({"model": key, "strategy": strategy, **counts,
                         "trainable_pct": round(100 * counts["trainable"] / counts["total"], 2)})
    return rows
