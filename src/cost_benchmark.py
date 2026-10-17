"""Inference cost of the four benchmark variants: latency, throughput, peak GPU memory, checkpoint size.

Run from the project folder with the .venv active (close other GPU-heavy programs first):

    python D:\\PORTFOLIO_2026\\P1_defect_detection\\src\\cost_benchmark.py screw cable capsule

What is measured: the model's forward pass only (no image loading, resizing or score normalisation),
on random input of the model's native size. Latency uses batch size 1 (median of 50 timed runs after
10 warm-up runs); throughput uses batch size 16. Peak memory is the highest GPU memory allocated while
running one batch-1 forward pass, including the model's own weights and statistics.
Results go to metrics/cost_table.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from failure_analysis import METRICS_DIR, _np, find_checkpoint, get_field, update_csv

ALL_VARIANTS = ["patchcore", "patchcore_r18", "padim", "padim_wrn50"]
DEFAULT_CATEGORIES = ["screw", "cable", "capsule"]


def forward(net, x):
    return net(x)


def timed_median_ms(net, x, runs, warmup, torch):
    for _ in range(warmup):
        forward(net, x)
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        forward(net, x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def measure(variant, category):
    import torch
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from lightning import seed_everything
    from run_benchmark import OUT, ROOT, VARIANTS

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this measurement.")

    model_cls, kwargs = VARIANTS[variant]
    try:
        model = model_cls(**kwargs, visualizer=False)
    except TypeError:
        model = model_cls(**kwargs)

    ckpt = find_checkpoint(OUT, variant, category)
    print(f"[cost] {variant} / {category}  checkpoint: {ckpt}")
    seed_everything(42, workers=True)
    datamodule = MVTecAD(root=ROOT, category=category, train_batch_size=32, eval_batch_size=16, num_workers=0)
    root_dir = OUT / "_predict" / variant / category
    try:
        engine = Engine(default_root_dir=root_dir, limit_predict_batches=1)
    except TypeError:
        engine = Engine(default_root_dir=root_dir)

    # A one-batch predict loads the trained weights into `model` and tells us the native input size.
    batches = engine.predict(model=model, datamodule=datamodule, ckpt_path=ckpt)
    amap = _np(get_field(batches[0], "anomaly_map"))
    height, width = int(amap.shape[-2]), int(amap.shape[-1])

    net = model.model.eval().to("cuda")
    device = "cuda"
    with torch.inference_mode():
        x1 = torch.randn(1, 3, height, width, device=device)
        x16 = torch.randn(16, 3, height, width, device=device)
        latency_ms = timed_median_ms(net, x1, runs=50, warmup=10, torch=torch)
        batch_ms = timed_median_ms(net, x16, runs=10, warmup=3, torch=torch)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        forward(net, x1)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1e6

    row = {
        "variant": variant, "category": category, "input_hw": f"{height}x{width}",
        "ckpt_mb": round(ckpt.stat().st_size / 1e6, 1),
        "latency_ms_bs1": round(latency_ms, 2),
        "throughput_img_s_bs16": round(16 / (batch_ms / 1000), 1),
        "peak_vram_mb": round(peak_mb, 0),
    }
    del model, net, engine
    torch.cuda.empty_cache()
    return row


def main(categories):
    rows = []
    for category in categories:
        for variant in ALL_VARIANTS:
            row = measure(variant, category)
            print(row)
            rows.append(row)
    new = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    update_csv(METRICS_DIR / "cost_table.csv", new, ["variant", "category"])
    print("\nMedian over the measured categories:")
    cols = ["ckpt_mb", "latency_ms_bs1", "throughput_img_s_bs16", "peak_vram_mb"]
    print(new.groupby("variant")[cols].median().round(1).to_string())
    print("\nSaved", METRICS_DIR / "cost_table.csv")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_CATEGORIES)
