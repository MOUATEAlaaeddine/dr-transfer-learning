"""Measure training speed and memory of the 13 configurations on this machine.

No dataset and no internet needed: random 224×224 images, ImageNet weights not loaded (speed does
not depend on weight values). Used once to decide where the grid runs (Apple MPS vs Kaggle CUDA).

    python scripts/benchmark_device.py               # all 13 configurations
    python scripts/benchmark_device.py --models mobilenetv4 --precision fp32 fp16
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from drtl.config import PROJECT_ROOT, load_config
from drtl.models import build_for_run, parameter_groups, set_train_mode, strategies_for, strategy_settings
from drtl.utils import get_device


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _memory_gb(device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return float("nan")


def bench(config, model_key, strategy, device, batch_size, precision, steps=8, warmup=3):
    lrs = strategy_settings(config, model_key, strategy)
    model = build_for_run(config, model_key, strategy, pretrained=False).to(device)
    opt = torch.optim.AdamW(parameter_groups(model, lrs["head_lr"], lrs.get("backbone_lr", 0.0), 1e-4))
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, 5, (batch_size,), device=device)
    use_amp = precision == "fp16" and device.type in ("cuda", "mps")
    set_train_mode(model)
    for i in range(warmup + steps):
        if i == warmup:
            _sync(device)
            start = time.time()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = torch.nn.functional.cross_entropy(model(x).float(), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    _sync(device)
    train_step = (time.time() - start) / steps
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        _sync(device)
        t = time.time()
        for _ in range(steps):
            model(x)
        _sync(device)
    eval_step = (time.time() - t) / steps
    mem = _memory_gb(device)
    del model, opt, x
    if device.type == "mps":
        torch.mps.empty_cache()
    return {"model": model_key, "strategy": strategy, "precision": precision, "batch_size": batch_size,
            "train_s_per_1000_img": round(1000 / batch_size * train_step, 2),
            "eval_s_per_1000_img": round(1000 / batch_size * eval_step, 2), "memory_gb": round(mem, 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*")
    p.add_argument("--precision", nargs="*", default=["fp32", "fp16"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "benchmark.json"))
    a = p.parse_args()
    config = load_config()
    device = get_device()
    info = {"device": device.type, "machine": platform.machine(), "platform": platform.platform(),
            "torch": torch.__version__}
    print(info, flush=True)
    rows = []
    for key in a.models or list(config["models"]):
        for strategy in strategies_for(key):
            for precision in a.precision:
                try:
                    row = bench(config, key, strategy, device, a.batch_size, precision)
                except RuntimeError as err:  # out of memory or unsupported op
                    row = {"model": key, "strategy": strategy, "precision": precision,
                           "batch_size": a.batch_size, "error": str(err).splitlines()[0][:160]}
                rows.append(row)
                print(json.dumps(row), flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"info": info, "rows": rows}, indent=2))
    print("written", out)


if __name__ == "__main__":
    main()
