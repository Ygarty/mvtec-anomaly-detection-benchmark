import csv
import sys
from pathlib import Path

import torch
from lightning import seed_everything

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore

ROOT = Path(r"D:\PORTFOLIO_2026\datasets\mvtec_ad")
OUT = Path(r"D:\PORTFOLIO_2026\P1_defect_detection\results")
METRICS_CSV = Path(r"D:\PORTFOLIO_2026\P1_defect_detection\metrics\mvtec_metrics.csv")

# Leave CPU headroom for other programs (the CPU has 24 threads).
CPU_THREADS = 8

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]

# Each variant name is written to the "model" column of the CSV, so the CSV
# schema stays unchanged and the skip logic works per variant.
#   patchcore / padim      -> library defaults (WRN50 / ResNet18 backbones)
#   padim_wrn50            -> PaDiM with PatchCore's backbone
#   patchcore_r18          -> PatchCore with PaDiM's backbone
VARIANTS = {
    "patchcore": (Patchcore, {}),
    "padim": (Padim, {}),
    "padim_wrn50": (Padim, {"backbone": "wide_resnet50_2", "n_features": 100}),
    "patchcore_r18": (Patchcore, {"backbone": "resnet18"}),
}


def completed_runs() -> set[tuple[str, str]]:
    """Return (variant, category) pairs already saved in the metrics CSV."""
    if not METRICS_CSV.exists():
        return set()
    with METRICS_CSV.open(newline="") as f:
        return {(row["model"], row["category"]) for row in csv.DictReader(f)}


def main(variant: str) -> None:
    if variant not in VARIANTS:
        raise SystemExit(f"Unknown variant '{variant}'. Choose from: {list(VARIANTS)}")
    model_cls, model_kwargs = VARIANTS[variant]

    torch.set_num_threads(CPU_THREADS)
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    done = completed_runs()

    for category in CATEGORIES:
        if (variant, category) in done:
            print(f"[skip] {variant} / {category} already in {METRICS_CSV.name}")
            continue

        print(f"[run ] {variant} / {category}  kwargs={model_kwargs}")
        seed_everything(42, workers=True)
        datamodule = MVTecAD(
            root=ROOT,
            category=category,
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=0,
        )
        model = model_cls(**model_kwargs)
        engine = Engine(default_root_dir=OUT / variant / category)
        engine.fit(model=model, datamodule=datamodule)
        metrics = engine.test(model=model, datamodule=datamodule)[0]

        # The row is written only after the category fully finishes,
        # so an interrupted category is simply redone on the next run.
        row = {"model": variant, "category": category, **metrics}
        write_header = not METRICS_CSV.exists()
        with METRICS_CSV.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python run_benchmark.py <{'|'.join(VARIANTS)}>")
    main(sys.argv[1])
