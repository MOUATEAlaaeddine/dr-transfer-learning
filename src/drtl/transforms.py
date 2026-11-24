"""Image preprocessing shared by training, evaluation and the demo (report §4.3–4.4).

`build_eval_transform` is the single definition of how a fundus photo becomes a model input; the
Gradio demo imports it as is, so the app sees images exactly as the models were trained on them.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CropRetina:
    """Remove the black border around the retinal disc, then pad to a square.

    Pixels darker than `threshold` (on 0–255) are background. Removing them keeps the disc the
    same size on every camera, so the network does not learn border width instead of lesions.
    If almost nothing passes the threshold (very dark or unusual image) the photo is kept whole.
    """

    def __init__(self, threshold: int = 15, min_fraction: float = 0.2):
        self.threshold = threshold
        self.min_fraction = min_fraction

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        gray = np.asarray(image.convert("L"))
        rows = np.where((gray > self.threshold).mean(axis=1) > 0.01)[0]
        cols = np.where((gray > self.threshold).mean(axis=0) > 0.01)[0]
        if len(rows) < self.min_fraction * gray.shape[0] or len(cols) < self.min_fraction * gray.shape[1]:
            cropped = image
        else:
            cropped = image.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
        w, h = cropped.size
        side = max(w, h)
        square = Image.new("RGB", (side, side), (0, 0, 0))
        square.paste(cropped, ((side - w) // 2, (side - h) // 2))
        return square


def build_eval_transform(image_size: int) -> Callable[[Image.Image], torch.Tensor]:
    """Validation, test and demo: crop, resize, normalise. No randomness."""
    return T.Compose([
        CropRetina(),
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_train_transform(image_size: int) -> Callable[[Image.Image], torch.Tensor]:
    """Training only: the evaluation pipeline plus plausible acquisition variations (report §4.4).

    No strong warps and no hue shifts: they can erase or imitate lesions.
    """
    return T.Compose([
        CropRetina(),
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomRotation(20),
        T.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0), antialias=True),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Model input → H×W×3 image in [0, 1], for figures and Grad-CAM overlays."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
