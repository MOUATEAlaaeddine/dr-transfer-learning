"""Grad-CAM attribution maps for the five networks (report §6.3, Step 8).

Grad-CAM weighs each channel of a late feature map by the gradient of the target grade's score,
which shows the image regions that pushed the model towards that grade. CNNs use their last
spatial feature map; ViT-B/16 uses the 196 patch tokens of its last block reshaped to 14 × 14.
Written directly (about 60 lines) rather than through pytorch-grad-cam, to keep the demo's
dependencies small; method after Selvaraju et al., ICCV 2017.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from drtl.models import ScratchCNN


def target_layer(model: nn.Module) -> Tuple[nn.Module, str, bool]:
    """(module, readable name, is_vit) of the feature map used for Grad-CAM."""
    if isinstance(model, ScratchCNN):
        return model.features, "features (last conv block)", False
    if hasattr(model, "blocks") and hasattr(model, "patch_embed"):        # ViT
        return model.blocks[-1].norm1, "blocks[-1].norm1 (14×14 patch tokens)", True
    if hasattr(model, "features") and hasattr(model.features, "norm5"):  # DenseNet
        return model.features.norm5, "features.norm5", False
    if hasattr(model, "features"):                                       # VGG
        return model.features, "features (last conv stage)", False
    if hasattr(model, "blocks"):                                         # MobileNetV4
        return model.blocks[-1], "blocks[-1] (last stage)", False
    raise ValueError(f"no Grad-CAM layer known for {type(model).__name__}")


def gradcam(model: nn.Module, image: torch.Tensor, target_class: int) -> Tuple[np.ndarray, str]:
    """Heat map in [0, 1] with the input's height × width, and the method label.

    `image` is one preprocessed tensor [3, H, W]. Works whatever the training strategy: frozen
    layers still pass gradients back to the feature map.
    """
    layer, name, is_vit = target_layer(model)
    store = {}

    def hook(_, __, output):
        output.retain_grad()
        store["act"] = output

    handle = layer.register_forward_hook(hook)
    model.eval()
    x = image.unsqueeze(0).detach().clone().requires_grad_(True)
    try:
        with torch.enable_grad():
            logits = model(x)
            model.zero_grad(set_to_none=True)
            logits[0, target_class].backward()
    finally:
        handle.remove()
    act, grad = store["act"].detach(), store["act"].grad.detach()
    if is_vit:  # [1, 1 + 196, C] → [1, C, 14, 14]
        prefix = getattr(model, "num_prefix_tokens", 1)
        act, grad = act[:, prefix:], grad[:, prefix:]
        side = int(round(act.shape[1] ** 0.5))
        act = act.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        grad = grad.reshape(1, side, side, -1).permute(0, 3, 1, 2)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[1:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = (cam / cam.max()).cpu().numpy() if cam.max() > 0 else cam.cpu().numpy()
    return cam.astype(np.float32), f"Grad-CAM · {name}"


def overlay(image_01: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heat map (jet colours) over an RGB image in [0, 1]; returns uint8 H×W×3."""
    import matplotlib

    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return ((1 - alpha) * image_01 + alpha * heat).clip(0, 1).__mul__(255).astype(np.uint8)
