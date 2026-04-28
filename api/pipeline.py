"""
pipeline.py — Unified Satellite Deep-Learning Evaluation Pipeline
Registry-based for easy expansion.

Supported tasks
---------------
sr             – super-resolution   (Proba-V)
cloud_removal  – cloud removal      (AllClear)
"""

import os
import csv
import numpy as np

# ── dataset imports ────────────────────────────────────────────────────────────
from datasets.probav.probav import ProbaVDataset
from datasets.allclear.allclear import AllClearDataset

# ── model imports ──────────────────────────────────────────────────────────────
from models.probav_baselines.baseline_upscale import BicubicModel
from models.probav_baselines.central_tendency import MedianModel
from models.allclear_baselines.least_cloudy import LeastCloudy
from models.allclear_baselines.mosaicing import Mosaicing
from models.allclear_baselines.uncrtaints import UnCRtainTS

# ── metric imports ─────────────────────────────────────────────────────────────
from metrics.metrics import evaluate, evaluate_allclear

# ─────────────────────────────────────────────────────────────────────────────
# Registries — add new datasets / models here
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "probav":   ProbaVDataset,
    "allclear": AllClearDataset,
}

# Maps each dataset name to its evaluation task
DATASET_TASK = {
    "probav":   "sr",
    "allclear": "cloud_removal",
}

MODELS = {
    # super-resolution
    "bicubic":     lambda: BicubicModel(),
    "median":      lambda: MedianModel(),
    "rams":        lambda: __import__('models.RAMS.model', fromlist=['RAMSModel']).RAMSModel(band='NIR'),
    # cloud removal
    "leastcloudy": lambda: LeastCloudy(),
    "mosaicing":   lambda: Mosaicing(),
    "uncrtaints":  lambda **kw: UnCRtainTS(**kw),
}

SR_METRICS       = ["psnr", "ssim", "cpsnr", "sam", "ergas"]
ALLCLEAR_METRICS = ["mae", "rmse", "psnr", "sam", "ssim", "lpips"]

# Convenience alias kept for backwards-compat with existing callers
METRICS = SR_METRICS

# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _model_label(model_name, model_kwargs):
    """Human-readable label including checkpoint name when present."""
    ckpt = (model_kwargs or {}).get("exp_name")
    return f"{model_name}/{ckpt}" if ckpt else model_name


def _write_metrics_csv(all_scores, metric_keys, save_dir):
    """
    Write per-scene metrics and an aggregated MEAN row to metrics.csv
    inside save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "metrics.csv")

    # Only write columns that exist in at least one score dict
    available = [k for k in metric_keys if any(k in s for s in all_scores)]
    fieldnames = ["scene"] + available

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_scores:
            writer.writerow(row)

        # ── aggregated MEAN row ───────────────────────────────────────────────
        mean_row = {"scene": "MEAN"}
        for k in available:
            vals = [s[k] for s in all_scores if k in s]
            mean_row[k] = round(sum(vals) / len(vals), 6) if vals else ""
        writer.writerow(mean_row)

    print(f"\n  Metrics saved → {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Logic
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(dataset_name, base_path, model_name, metrics=None, n_scenes=1,
                 save_dir=None, show=False, model_kwargs=None, **dataset_kwargs):
    """
    Run the evaluation pipeline for *dataset_name* / *model_name*.

    Args:
        dataset_name:    key in DATASETS  (e.g. "probav", "allclear")
        base_path:       root data path passed to the dataset constructor.
                         For AllClear this is the JSON metadata file path.
        model_name:      key in MODELS    (e.g. "bicubic", "leastcloudy")
        metrics:         list of metric names to report; None = task default
        n_scenes:        number of samples to evaluate
        save_dir:        directory to save visualizations; None = skip
        show:            whether to display visualizations interactively
        **dataset_kwargs extra keyword arguments forwarded to the dataset
                         constructor (e.g. aux_sensors, tx, target_mode …)
    """
    task = DATASET_TASK[dataset_name]

    if metrics is None:
        metrics = SR_METRICS if task == "sr" else ALLCLEAR_METRICS

    # ── build dataset ──────────────────────────────────────────────────────────
    dataset_cls = DATASETS[dataset_name]
    if task == "sr":
        dataset = dataset_cls(base_path=base_path)
    else:
        dataset = dataset_cls(json_path=base_path, **dataset_kwargs)

    # ── resolve n_scenes: None or -1 means the full dataset ──────────────────
    if n_scenes is None or n_scenes < 0:
        n_scenes = len(dataset)

    # ── build model ───────────────────────────────────────────────────────────
    model = MODELS[model_name](**(model_kwargs or {}))

    # ── structured save directory: <base>/<dataset>/<model>[/<checkpoint>] ────
    effective_save_dir = None
    if save_dir is not None:
        parts = [save_dir, dataset_name, model_name]
        checkpoint = (model_kwargs or {}).get("exp_name")
        if checkpoint:
            parts.append(checkpoint)
        effective_save_dir = os.path.join(*parts)

    visualize = (effective_save_dir is not None) or show
    if visualize:
        from visualize import visualize_results

    print(f"\n{'═' * 50}")
    print(f" Dataset : {dataset_name}  [{task}]")
    print(f" Model   : {model_name}")
    if (model_kwargs or {}).get("exp_name"):
        print(f" Checkpoint: {model_kwargs['exp_name']}")
    print(f" Scenes  : {n_scenes}")
    if effective_save_dir:
        print(f" Save dir: {effective_save_dir}")
    print(f"{'═' * 50}\n")

    all_scores = []   # collect per-scene dicts for CSV

    for i in range(n_scenes):
        # load_sample() always returns a 5-tuple:
        #   (lr_or_sample_dict, hr_np, lr_mask_np, hr_mask_np, name)
        lr, hr, _, hr_mask, name = dataset.load_sample(i)

        print(f"[{i+1}/{n_scenes}] Scene: {name}")

        # ── predict ───────────────────────────────────────────────────────────
        predictions = model.predict(lr)

        # ── evaluate ──────────────────────────────────────────────────────────
        if task == "sr":
            scores = evaluate(predictions, hr, hr_mask=hr_mask)
        else:
            scores = evaluate_allclear(predictions, hr, hr_mask=hr_mask)

        all_scores.append({"scene": name, **scores})

        # ── print requested metrics ───────────────────────────────────────────
        for k, v in scores.items():
            if k in metrics:
                print(f"    {k.upper():<6}: {v:.4f}")

        # ── visualize ─────────────────────────────────────────────────────────
        if visualize:
            visualize_results(
                dataset_name = dataset_name,
                lr           = lr,
                predictions  = predictions,
                target       = hr,
                hr_mask      = hr_mask,
                scores       = scores,
                name         = name,
                model_name   = model_name,
                save_dir     = effective_save_dir,
                show         = show,
            )

    # ── CSV logging ───────────────────────────────────────────────────────────
    if effective_save_dir and all_scores:
        _write_metrics_csv(all_scores, metrics, effective_save_dir)


def _write_comparison_csv(scenes_a, scenes_b, label_a, label_b, metric_keys, save_dir):
    """
    Write a side-by-side comparison CSV with per-scene rows for both models,
    a MEAN row for each, and a DELTA row (B − A, positive = B is larger).
    """
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "comparison.csv")

    available = [k for k in metric_keys
                 if any(k in s for s in scenes_a + scenes_b)]

    col_a = [f"{label_a}_{k}" for k in available]
    col_b = [f"{label_b}_{k}" for k in available]
    fieldnames = ["scene"] + col_a + col_b

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for sa, sb in zip(scenes_a, scenes_b):
            row = {"scene": sa["scene"]}
            for k, col in zip(available, col_a):
                row[col] = sa.get(k, "")
            for k, col in zip(available, col_b):
                row[col] = sb.get(k, "")
            writer.writerow(row)

        def _mean_row(scenes, cols, label):
            row = {"scene": f"MEAN ({label})"}
            for k, col in zip(available, cols):
                vals = [s[k] for s in scenes if k in s]
                row[col] = round(sum(vals) / len(vals), 6) if vals else ""
            return row

        writer.writerow(_mean_row(scenes_a, col_a, label_a))
        writer.writerow(_mean_row(scenes_b, col_b, label_b))

        # DELTA row: mean_B − mean_A  (positive = B larger)
        delta = {"scene": f"DELTA ({label_b} - {label_a})"}
        for k, ca, cb in zip(available, col_a, col_b):
            vals_a = [s[k] for s in scenes_a if k in s]
            vals_b = [s[k] for s in scenes_b if k in s]
            if vals_a and vals_b:
                delta[ca] = ""
                delta[cb] = round(
                    sum(vals_b) / len(vals_b) - sum(vals_a) / len(vals_a), 6
                )
        writer.writerow(delta)

    print(f"\n  Comparison CSV saved → {csv_path}")


def compare_models(
    dataset_name, base_path,
    model_name_a, model_name_b,
    model_kwargs_a=None, model_kwargs_b=None,
    metrics=None, n_scenes=None,
    save_dir=None, show=False,
    **dataset_kwargs,
):
    """
    Run two models on the same scenes and produce side-by-side metrics,
    a comparison visualization, and a comparison CSV.

    Args:
        dataset_name:   key in DATASETS
        base_path:      root data path or JSON metadata file
        model_name_a:   first model  — key in MODELS
        model_name_b:   second model — key in MODELS
        model_kwargs_a: constructor kwargs for model A (e.g. {"exp_name": "noSAR_1"})
        model_kwargs_b: constructor kwargs for model B
        metrics:        list of metric names; None = task default
        n_scenes:       number of scenes; None / -1 = full dataset
        save_dir:       base output directory; structured as
                        <save_dir>/<dataset>/compare_<A>_vs_<B>/
        show:           display figures interactively
        **dataset_kwargs: forwarded to the dataset constructor
    """
    task = DATASET_TASK[dataset_name]

    if metrics is None:
        metrics = SR_METRICS if task == "sr" else ALLCLEAR_METRICS

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset_cls = DATASETS[dataset_name]
    dataset = (
        dataset_cls(base_path=base_path)
        if task == "sr"
        else dataset_cls(json_path=base_path, **dataset_kwargs)
    )

    if n_scenes is None or n_scenes < 0:
        n_scenes = len(dataset)

    # ── models ────────────────────────────────────────────────────────────────
    model_a = MODELS[model_name_a](**(model_kwargs_a or {}))
    model_b = MODELS[model_name_b](**(model_kwargs_b or {}))

    label_a = _model_label(model_name_a, model_kwargs_a)
    label_b = _model_label(model_name_b, model_kwargs_b)

    # ── save directory ────────────────────────────────────────────────────────
    effective_save_dir = None
    if save_dir is not None:
        safe_a = label_a.replace("/", "_")
        safe_b = label_b.replace("/", "_")
        effective_save_dir = os.path.join(
            save_dir, dataset_name, f"compare_{safe_a}_vs_{safe_b}"
        )

    visualize = (effective_save_dir is not None) or show
    if visualize:
        from visualize.visualize import visualize_comparison

    print(f"\n{'═' * 56}")
    print(f" Dataset  : {dataset_name}  [{task}]")
    print(f" Model A  : {label_a}")
    print(f" Model B  : {label_b}")
    print(f" Scenes   : {n_scenes}")
    if effective_save_dir:
        print(f" Save dir : {effective_save_dir}")
    print(f"{'═' * 56}\n")

    all_a, all_b = [], []

    for i in range(n_scenes):
        lr, hr, _, hr_mask, name = dataset.load_sample(i)
        print(f"[{i+1}/{n_scenes}] Scene: {name}")

        pred_a = model_a.predict(lr)
        pred_b = model_b.predict(lr)

        _eval = evaluate if task == "sr" else evaluate_allclear
        scores_a = _eval(pred_a, hr, hr_mask=hr_mask)
        scores_b = _eval(pred_b, hr, hr_mask=hr_mask)

        all_a.append({"scene": name, **scores_a})
        all_b.append({"scene": name, **scores_b})

        # ── print side-by-side ────────────────────────────────────────────────
        header = f"    {'METRIC':<8}  {label_a:>18}  {label_b:>18}"
        print(header)
        print("    " + "─" * (len(header) - 4))
        for k in metrics:
            va = scores_a.get(k)
            vb = scores_b.get(k)
            if va is None and vb is None:
                continue
            winner = ""
            if va is not None and vb is not None:
                higher_better = k in ("psnr", "ssim", "cpsnr")
                winner = "◀ A" if (va > vb) == higher_better else "◀ B"
            print(f"    {k.upper():<8}  {va:>18.4f}  {vb:>18.4f}  {winner}")
        print()

        if visualize:
            visualize_comparison(
                dataset_name = dataset_name,
                lr           = lr,
                pred_a       = pred_a,
                pred_b       = pred_b,
                target       = hr,
                hr_mask      = hr_mask,
                scores_a     = scores_a,
                scores_b     = scores_b,
                name         = name,
                label_a      = label_a,
                label_b      = label_b,
                save_dir     = effective_save_dir,
                show         = show,
            )

    if effective_save_dir and all_a:
        _write_comparison_csv(all_a, all_b, label_a, label_b,
                              metrics, effective_save_dir)


if __name__ == "__main__":
    # Example — Proba-V SR
    run_pipeline(
        dataset_name="probav",
        base_path="/path/to/probav_data/train",
        model_name="bicubic",
        n_scenes=1,
    )
