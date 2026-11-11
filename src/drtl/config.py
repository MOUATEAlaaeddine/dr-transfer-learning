"""Load the shared YAML configuration, with optional overrides for one experiment."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: Optional[Path] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read a YAML config and apply overrides on top of it.

    Overrides use the same nesting as the file, e.g. {"training": {"batch_size": 16}}.
    Only the keys you pass change; everything else keeps the shared value.
    """
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if overrides:
        config = merge(config, overrides)
    return config


def merge(base: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `base` with `changes` merged in, key by key."""
    result = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result
