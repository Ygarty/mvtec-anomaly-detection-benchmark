"""Failure analysis for the MVTec AD benchmark (PatchCore vs PaDiM, both Wide-ResNet50).

Run from the project folder with the .venv active:

    python D:\\PORTFOLIO_2026\\P1_defect_detection\\src\\failure_analysis.py screw cable capsule

For every category it
  1. loads the saved checkpoints of both variants (written by run_benchmark.py) and scores the test images,
  2. writes per-image scores and a detection-rate-by-defect-type table to metrics/,
  3. saves two figures per category to figures/: missed defects, and false alarms.

No thresholds are tuned on the test set. A defect counts as "caught" if its score is above the
95th percentile of the NORMAL test images' scores (i.e. 5% false-alarm rate).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT / "metrics"
FIG_DIR = PROJECT / "figures"

VARIANT_A, VARIANT_B = "patchcore", "padim_wrn50"
VARIANTS_CMP = (VARIANT_A, VARIANT_B)
LABELS = {VARIANT_A: "PatchCore (WRN50)", VARIANT_B: "PaDiM (WRN50, 100 features)"}
FPR = 0.05
DEFAULT_CATEGORIES = ["screw", "cable", "capsule"]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 9.5,
    "text.color": INK, "axes.labelcolor": INK2,
})


# --------------------------------------------------------------------------- prediction (needs anomalib)
def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def get_field(batch, *names):
    """Read a field from a prediction batch whether it is a dataclass or a dict."""
    for name in names:
        if isinstance(batch, dict) and name in batch:
            return batch[name]
        value = getattr(batch, name, None)
        if value is not None:
            return value
    return None


def find_checkpoint(out_dir, variant, category):
    found = sorted((out_dir / variant / category).rglob("model.ckpt"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"No model.ckpt under {out_dir / variant / category}. Run run_benchmark.py for '{variant}' first.")
    return found[-1]


def predict_variant(variant, category):
    """Score every test image of one category with a trained variant. Returns paths, scores, maps."""
    import torch
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from lightning import seed_everything
    from run_benchmark import OUT, ROOT, VARIANTS

    model_cls, kwargs = VARIANTS[variant]
    try:
        model = model_cls(**kwargs, visualizer=False)
    except TypeError:
        model = model_cls(**kwargs)

    ckpt = find_checkpoint(OUT, variant, category)
    print(f"[predict] {variant} / {category}  checkpoint: {ckpt}")
    seed_everything(42, workers=True)
    datamodule = MVTecAD(root=ROOT, category=category, train_batch_size=32, eval_batch_size=16, num_workers=0)
    engine = Engine(default_root_dir=OUT / "_predict" / variant / category)
    try:
        batches = engine.predict(model=model, datamodule=datamodule, ckpt_path=ckpt)
    except Exception as err:
        print(f"[predict] datamodule route failed ({type(err).__name__}: {err}); retrying with the test dataloader")
        datamodule.setup("test")
        batches = engine.predict(model=model, dataloaders=datamodule.test_dataloader(), ckpt_path=ckpt)

    paths, scores, maps = [], [], []
    for b in batches:
        p = get_field(b, "image_path")
        s = get_field(b, "pred_score")
        m = get_field(b, "anomaly_map")
        if p is None or s is None or m is None:
            fields = [a for a in dir(b) if not a.startswith("_")]
            raise RuntimeError(f"A prediction batch lacks image_path / pred_score / anomaly_map. Fields found: {fields}")
        paths += [str(x) for x in p]
        scores.append(_np(s).astype(np.float32).reshape(-1))
        m = _np(m).astype(np.float32)
        maps.append(m.reshape(-1, m.shape[-2], m.shape[-1]))

    del model, engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"paths": paths, "scores": np.concatenate(scores), "maps": np.concatenate(maps)}


# --------------------------------------------------------------------------- analysis (pure, testable)
def defect_type_of(path):
    return Path(path).parent.name


def build_table(A, B):
    """One row per test image with both variants' scores and threshold-free ranks."""
    if A["paths"] != B["paths"]:
        raise ValueError("The two variants scored the images in a different order.")
    df = pd.DataFrame({"image_path": A["paths"],
                       f"score_{VARIANT_A}": A["scores"], f"score_{VARIANT_B}": B["scores"]})
    df["defect_type"] = df["image_path"].map(defect_type_of)
    df["is_defect"] = df["defect_type"] != "good"
    for v in VARIANTS_CMP:
        s = df[f"score_{v}"].to_numpy()
        normal, defect = s[~df["is_defect"].to_numpy()], s[df["is_defect"].to_numpy()]
        df[f"pct_normal_{v}"] = [(normal < x).mean() * 100 for x in s]   # share of normal images outscored
        df[f"pct_defect_{v}"] = [(defect < x).mean() * 100 for x in s]   # share of defective images outscored
        df[f"detected_{v}"] = s > np.quantile(normal, 1 - FPR)
    return df


def detection_by_type(df, category):
    rows = []
    for dtype, g in df[df["is_defect"]].groupby("defect_type"):
        rows.append({"category": category, "defect_type": dtype, "n_images": len(g),
                     **{f"tpr_{v}": round(float(g[f"detected_{v}"].mean()), 3) for v in VARIANTS_CMP}})
    d = df[df["is_defect"]]
    rows.append({"category": category, "defect_type": "ALL DEFECTS", "n_images": len(d),
                 **{f"tpr_{v}": round(float(d[f"detected_{v}"].mean()), 3) for v in VARIANTS_CMP}})
    return pd.DataFrame(rows)


def pick_cases(df, k=3):
    d = df[df["is_defect"]].copy()
    d["adv"] = d[f"pct_normal_{VARIANT_A}"] - d[f"pct_normal_{VARIANT_B}"]
    d["best"] = d[[f"pct_normal_{VARIANT_A}", f"pct_normal_{VARIANT_B}"]].max(axis=1)
    only_a = d.sort_values("adv", ascending=False).head(k).assign(tag="PatchCore ahead")
    both = d.drop(only_a.index).sort_values("best").head(k).assign(tag="both weak")
    misses = pd.concat([only_a, both])
    n = df[~df["is_defect"]].copy()
    n["worst"] = n[[f"pct_defect_{VARIANT_A}", f"pct_defect_{VARIANT_B}"]].mean(axis=1)
    false_alarms = n.sort_values("worst", ascending=False).head(k).assign(tag="false alarm")
    return misses, false_alarms


def load_gt_mask(root, category, path, shape):
    p = Path(path)
    if p.parent.name == "good":
        return None
    mask_path = Path(root) / category / "ground_truth" / p.parent.name / f"{p.stem}_mask.png"
    if not mask_path.exists():
        return None
    m = Image.open(mask_path).convert("L").resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(m) > 127


def draw_cases(out_path, title, rows, mode, category, root, A, B, ranges):
    """mode 'miss': annotate with caught/missed; mode 'false_alarm': annotate with rank among defects."""
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(7.6, 2.55 * n + 0.8), squeeze=False)
    maps = {VARIANT_A: A["maps"], VARIANT_B: B["maps"]}
    for r, (idx, row) in enumerate(rows.iterrows()):
        shape = maps[VARIANT_A][idx].shape
        img = Image.open(row["image_path"]).convert("RGB").resize((shape[1], shape[0]), Image.BILINEAR)
        gt = load_gt_mask(root, category, row["image_path"], shape)
        panels = [("image", None)] + [("map", v) for v in VARIANTS_CMP]
        for c, (kind, v) in enumerate(panels):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if kind == "image":
                ax.imshow(img)
                if gt is not None and gt.any():
                    ax.contour(gt, levels=[0.5], colors="#00e5ff", linewidths=1.2)
                label = f"{row['tag']}\n{row['defect_type']}\n{Path(row['image_path']).name}"
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8.5, color=INK2, labelpad=10)
            else:
                vmin, vmax = ranges[v]
                ax.imshow(maps[v][idx], cmap="magma", vmin=vmin, vmax=vmax)
                if gt is not None and gt.any():
                    ax.contour(gt, levels=[0.5], colors="white", linewidths=1.0)
                score = row[f"score_{v}"]
                if mode == "miss":
                    note = f"{score:.2f}  {'caught' if row[f'detected_{v}'] else 'missed'}"
                else:
                    note = f"{score:.2f}  above {row[f'pct_defect_{v}']:.0f}% of defects"
                ax.set_xlabel(note, fontsize=8.5, color=INK2, labelpad=4)
            if r == 0:
                ax.set_title("Image (cyan = ground truth)" if kind == "image" else LABELS[v],
                             fontsize=9.5, color=INK, pad=6)
    fig_h = 2.55 * n + 0.8
    fig.suptitle(title, x=0.02, y=1 - 0.08 / fig_h, ha="left", fontsize=11.5, color=INK, weight="medium")
    fig.subplots_adjust(left=0.2, right=0.99, top=1 - 0.75 / fig_h, wspace=0.05, hspace=0.45)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def update_csv(path, new_df, key_cols):
    """Replace the rows with the same keys, keep everything else."""
    if Path(path).exists():
        old = pd.read_csv(path)
        keys = new_df[key_cols].drop_duplicates()
        old = old.merge(keys, on=key_cols, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


def analyze_category(category, A, B, root, metrics_dir=METRICS_DIR, fig_dir=FIG_DIR):
    metrics_dir, fig_dir = Path(metrics_dir), Path(fig_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = build_table(A, B)

    long_rows = []
    for v in VARIANTS_CMP:
        part = df[["image_path", "defect_type", "is_defect"]].copy()
        part["image_path"] = part["image_path"].map(lambda p: str(Path(p).relative_to(root)) if str(p).startswith(str(root)) else p)
        part["variant"], part["category"] = v, category
        part["score"] = df[f"score_{v}"]
        part["pct_of_normals_outscored"] = df[f"pct_normal_{v}"].round(1)
        part["detected_at_5pct_fpr"] = df[f"detected_{v}"]
        long_rows.append(part)
    update_csv(metrics_dir / "per_image_scores.csv", pd.concat(long_rows), ["variant", "category"])

    det = detection_by_type(df, category)
    update_csv(metrics_dir / "defect_type_detection.csv", det, ["category"])
    print(f"\n{category}: share of defects caught at {FPR:.0%} false-alarm rate")
    print(det.to_string(index=False))

    ranges = {}
    for v, M in ((VARIANT_A, A["maps"]), (VARIANT_B, B["maps"])):
        ranges[v] = (float(M.min()), float(np.percentile(M, 99.9)))
    misses, false_alarms = pick_cases(df)
    out = []
    if len(misses):
        p = fig_dir / f"failure_{category}_misses.png"
        draw_cases(p, f"{category}: defects PaDiM misses but PatchCore catches, and defects both find hard",
                   misses, "miss", category, root, A, B, ranges)
        out.append(p)
    if len(false_alarms):
        p = fig_dir / f"failure_{category}_false_alarms.png"
        draw_cases(p, f"{category}: normal images that look most defective to the models",
                   false_alarms, "false_alarm", category, root, A, B, ranges)
        out.append(p)
    return det, out


def main(categories):
    from run_benchmark import ROOT

    for category in categories:
        A = predict_variant(VARIANT_A, category)
        B = predict_variant(VARIANT_B, category)
        _, figures = analyze_category(category, A, B, ROOT)
        for f in figures:
            print("saved", f)
    print("\nDone. Tables are in", METRICS_DIR, "and figures in", FIG_DIR)


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_CATEGORIES)
