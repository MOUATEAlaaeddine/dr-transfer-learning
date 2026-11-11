"""Fast checks that the shared code works before anyone spends GPU time.

Run with: pytest -q
Models are built without pretrained weights so the tests need no internet.
"""

import pytest
import torch

from drtl.config import load_config, merge
from drtl.models import build_model, count_parameters
from drtl.utils import get_device, set_seed

CONFIG = load_config()
NUM_CLASSES = len(CONFIG["data"]["class_names"])
SIZE = CONFIG["data"]["image_size"]


def test_config_has_the_five_models_and_three_strategies():
    assert set(CONFIG["models"]) == {"mobilenetv4", "densenet201", "vgg19", "vit_b16", "scratch_cnn"}
    assert set(CONFIG["strategies"]) == {"feature_extraction", "partial_finetuning", "full_finetuning"}
    assert NUM_CLASSES == 5


def test_overrides_change_only_the_given_key():
    config = load_config(overrides={"training": {"batch_size": 16}})
    assert config["training"]["batch_size"] == 16
    assert config["training"]["max_epochs"] == CONFIG["training"]["max_epochs"]
    assert merge({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}


def test_seed_makes_random_numbers_repeatable():
    set_seed(42)
    first = torch.rand(3)
    set_seed(42)
    assert torch.equal(first, torch.rand(3))


def test_device_is_detected():
    assert get_device().type in {"cuda", "mps", "cpu"}


@pytest.mark.parametrize("key", sorted(CONFIG["models"]))
def test_every_model_outputs_one_score_per_grade(key):
    model = build_model(CONFIG["models"][key], NUM_CLASSES, pretrained=False).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, SIZE, SIZE))
    assert logits.shape == (2, NUM_CLASSES)
    counts = count_parameters(model)
    assert counts["trainable"] == counts["total"] > 0
