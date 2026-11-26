"""RetinaLens Transfer Lab: compare the three transfer strategies of one backbone on one image.

    python app/app.py            # then open http://127.0.0.1:7860

Educational research demo, not a medical device. Checkpoints are read from outputs/<run_id>/best.pt
(trained by scripts/run_grid.py). Without real checkpoints the app runs in example mode on smoke
checkpoints trained on synthetic data, and every output is labelled SIMULATED.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

from drtl.config import load_config  # noqa: E402
from drtl.gradcam import gradcam, overlay  # noqa: E402
from drtl.models import TRANSFER_STRATEGIES  # noqa: E402
from drtl.robustness import KINDS, degrade  # noqa: E402
from drtl.train import load_best  # noqa: E402
from drtl.transforms import build_eval_transform, denormalize  # noqa: E402

CONFIG = load_config()
BACKBONES = {"MobileNetV4": "mobilenetv4", "DenseNet-201": "densenet201", "VGG19": "vgg19", "ViT-B/16": "vit_b16"}
STRATEGY_LABELS = {"feature_extraction": "Feature extraction", "partial_finetuning": "Partial fine-tuning",
                   "full_finetuning": "Full fine-tuning"}
DISCLAIMER = ("Educational research demo (Master ISI, FSSM). Not a medical device: it gives no diagnosis "
              "and no recommendation.")


def find_checkpoint(model: str, strategy: str):
    """(path, simulated) of the best fold for a configuration; real runs before smoke runs."""
    real = sorted((ROOT / "outputs").glob(f"{model}__{strategy}__fold*/best.pt"))
    if real:
        scores = [(torch.load(p, map_location="cpu", weights_only=False).get("val_metrics", {}).get("f1_macro", 0), p)
                  for p in real]
        return max(scores)[1], False
    smoke = sorted((ROOT / "outputs" / "smoke").glob(f"{model}__{strategy}__fold*/best.pt"))
    return (smoke[0], True) if smoke else (None, False)


@lru_cache(maxsize=12)
def load(path: str):
    return load_best(Path(path))


def validate(image) -> Image.Image:
    if image is None:
        raise gr.Error("Upload a JPG or PNG fundus photograph first.")
    if isinstance(image, str):
        try:
            image = Image.open(image)
        except (UnidentifiedImageError, OSError):
            raise gr.Error("This file is not a readable image. Use a JPG or PNG fundus photograph.")
    if min(image.size) < 64:
        raise gr.Error("Image too small (under 64 px). Use the original fundus photograph.")
    return image.convert("RGB")


def analyze(image, backbone, kind, level):
    image = validate(image)
    model_key = BACKBONES[backbone]
    shown = degrade(image, kind, int(level)) if int(level) > 0 else image
    outputs, notes = [], []
    for strategy in TRANSFER_STRATEGIES:
        path, simulated = find_checkpoint(model_key, strategy)
        title = STRATEGY_LABELS[strategy]
        if path is None:
            outputs += [None, None, f"**{title}** · Checkpoint not available: "
                                    f"`{model_key}__{strategy}` (not trained yet)."]
            continue
        bundle = load(str(path))
        meta, model = bundle["metadata"], bundle["model"]
        x = build_eval_transform(meta["image_size"])(shown)
        with torch.no_grad():
            probs = torch.softmax(model(x.unsqueeze(0)), dim=1)[0].tolist()
        names = meta["class_names"]
        tag = " · SIMULATED" if simulated else ""
        scores = {f"{n}{tag}": p for n, p in zip(names, probs)}
        target = int(torch.tensor(probs).argmax())
        try:
            cam, method = gradcam(model, x, target)
            heat = overlay(denormalize(x), cam)
            caption = f"{method} · {backbone} · {title} · target grade: {names[target]}{tag}"
        except Exception:  # noqa: BLE001 - attribution must never block the scores
            heat, caption = None, "Attribution unavailable for this model."
        outputs += [scores, heat, f"**{title}**{tag} · {meta['run_id']}  \n{caption}"]
        notes.append(simulated)
    banner = ("⚠ **SIMULATED example mode**: smoke checkpoints trained on synthetic images. Scores mean nothing."
              if notes and any(notes) else "")
    return [banner, shown] + outputs


def compare_table():
    path = ROOT / "results" / "summary.csv"
    if not path.exists():
        return pd.DataFrame({"status": ["No measured results yet: the cross-validation grid has not finished."]})
    table = pd.read_csv(path)
    cols = ["model", "strategy", "folds_done", "f1_macro_mean", "f1_macro_std", "auc_macro_ovr_mean",
            "qwk_mean", "params_total", "train_minutes_mean"]
    return table[[c for c in cols if c in table.columns]].round(4)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RetinaLens Transfer Lab") as app:
        gr.Markdown("# RetinaLens · Transfer Lab\nSame image, same backbone, three transfer strategies. "
                    "Scores for the five ICDR grades and a Grad-CAM map of the regions behind the top grade.")
        with gr.Tab("Analyze an image"):
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    upload = gr.Image(type="pil", label="Fundus photograph (JPG or PNG)", height=260,
                                      sources=["upload"])
                    backbone = gr.Radio(list(BACKBONES), value="MobileNetV4", label="Backbone (ImageNet-pretrained)")
                    kind = gr.Radio(list(KINDS), value="blur", label="Capture quality: degradation")
                    level = gr.Slider(0, 4, value=0, step=1, label="Level (0 = original)")
                    run = gr.Button("Analyze", variant="primary")
                with gr.Column(scale=1, min_width=220):
                    banner = gr.Markdown()
                    seen = gr.Image(label="Image given to the models", height=220, interactive=False)
            columns = []
            with gr.Row():
                for strategy in TRANSFER_STRATEGIES:
                    with gr.Column(min_width=240):
                        note = gr.Markdown(f"**{STRATEGY_LABELS[strategy]}**")
                        label = gr.Label(num_top_classes=5, label="Scores (5 grades)")
                        heat = gr.Image(label="Attribution overlay", height=220, interactive=False)
                        columns += [label, heat, note]
            run.click(analyze, [upload, backbone, kind, level], [banner, seen] + columns)
        with gr.Tab("Compare models"):
            gr.Markdown("Cross-validation results measured on the validation folds (mean ± std over 5 folds).")
            gr.Dataframe(compare_table(), interactive=False)
        gr.Markdown(f"<small>{DISCLAIMER}</small>")
    return app


if __name__ == "__main__":
    build_app().launch()
