"""Step 7–8 checks: metrics against scikit-learn, degradations, Grad-CAM."""

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score

from drtl.config import load_config
from drtl.evaluate import compute_metrics
from drtl.gradcam import gradcam, overlay
from drtl.models import build_for_run
from drtl.robustness import KINDS, degrade

CONFIG = load_config()


def test_metrics_match_scikit_learn():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(5), 20)
    probs = rng.dirichlet(np.ones(5), size=100)
    probs[np.arange(100), y] += 0.3
    m = compute_metrics(y, probs)
    assert m["f1_macro"] == pytest.approx(f1_score(y, probs.argmax(1), average="macro"))
    assert m["auc_macro_ovr"] == pytest.approx(roc_auc_score(y, probs / probs.sum(1, keepdims=True), multi_class="ovr"), abs=1e-9)
    assert sum(map(sum, m["confusion_matrix"])) == 100 and m["absent_classes"] == []


def test_absent_class_is_reported_not_faked():
    y = np.array([0, 0, 1, 1, 2, 2])
    probs = np.eye(5)[[0, 0, 1, 2, 2, 2]] * 0.8 + 0.04
    m = compute_metrics(y, probs)
    assert m["absent_classes"] == [3, 4]
    assert m["per_class"][3]["auc"] is None and not np.isnan(m["auc_macro_ovr"])


@pytest.mark.parametrize("kind", KINDS)
def test_degradations_keep_size_and_get_worse(kind):
    img = Image.fromarray((np.random.default_rng(1).random((224, 224, 3)) * 255).astype(np.uint8))
    assert np.array_equal(np.asarray(degrade(img, kind, 0)), np.asarray(img))
    diffs = [np.abs(np.asarray(degrade(img, kind, l), float) - np.asarray(img, float)).mean() for l in (1, 4)]
    assert degrade(img, kind, 4).size == img.size and diffs[1] > diffs[0]


@pytest.mark.parametrize("model,strategy", [("scratch_cnn", "scratch"), ("mobilenetv4", "feature_extraction"),
                                            ("densenet201", "feature_extraction"), ("vgg19", "feature_extraction"),
                                            ("vit_b16", "feature_extraction")])
def test_gradcam_shape_range_and_target_dependence(model, strategy):
    torch.manual_seed(0)
    net = build_for_run(CONFIG, model, strategy, pretrained=False)
    x = torch.randn(3, 224, 224)
    cams = [gradcam(net, x, c)[0] for c in (0, 4)]
    assert cams[0].shape == (224, 224) and 0 <= cams[0].min() and cams[0].max() <= 1
    assert not np.allclose(cams[0], cams[1])
    assert overlay(np.zeros((224, 224, 3)), cams[0]).dtype == np.uint8
