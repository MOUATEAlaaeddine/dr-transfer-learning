"""Build the five networks compared in the project.

Four are ImageNet-pretrained backbones from timm; the fifth is a small CNN trained
from scratch, which serves as the baseline that shows what transfer learning adds.
Freezing for the three transfer strategies is added in Step 4.
"""

from __future__ import annotations

from typing import Dict

import timm
import torch
from torch import nn

SCRATCH_CNN = "scratch_cnn"


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
