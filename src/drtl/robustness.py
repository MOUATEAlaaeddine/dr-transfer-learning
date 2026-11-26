"""Image degradations for the robustness test (report §6.2) and the demo's quality slider.

Applied to the raw photo **before** `build_eval_transform`, exactly as a worse camera or capture
would. Level 0 returns the image unchanged; levels 1–4 get progressively worse.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

KINDS = ("blur", "underexposure", "noise", "jpeg")
LEVELS = (0, 1, 2, 3, 4)

# Strength per level, chosen so level 4 is clearly poor but the retina is still recognisable.
_BLUR_RADIUS = (0, 1.0, 2.0, 3.5, 5.0)          # Gaussian radius, in pixels of a 224-px image
_EXPOSURE = (1.0, 0.8, 0.6, 0.45, 0.3)           # brightness factor
_NOISE_STD = (0, 8, 16, 28, 40)                  # Gaussian noise on 0–255
_JPEG_QUALITY = (None, 50, 25, 12, 5)


def degrade(image: Image.Image, kind: str, level: int, seed: int = 0) -> Image.Image:
    """Return a degraded copy of `image` (RGB)."""
    if kind not in KINDS or level not in LEVELS:
        raise ValueError(f"kind must be in {KINDS} and level in {LEVELS}")
    image = image.convert("RGB")
    if level == 0:
        return image.copy()
    scale = max(image.size) / 224  # same visual strength whatever the input resolution
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(_BLUR_RADIUS[level] * scale))
    if kind == "underexposure":
        return ImageEnhance.Brightness(image).enhance(_EXPOSURE[level])
    if kind == "noise":
        rng = np.random.default_rng(seed)
        arr = np.asarray(image, dtype=np.float32) + rng.normal(0, _NOISE_STD[level], (image.size[1], image.size[0], 3))
        return Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=_JPEG_QUALITY[level])
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")
