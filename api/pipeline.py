"""
pipeline.py — Unified Satellite Deep-Learning Evaluation Pipeline
Registry-based for easy expansion.

Supported tasks
---------------
sr             – super-resolution   (Proba-V)
cloud_removal  – cloud removal      (AllClear)
"""

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
ALLCLEAR_METRICS = ["mae", "rmse", "psnr", "sam", "ssim"]

# Convenience alias kept for backwards-compat with existing callers
METRICS = SR_METRICS

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

    # ── build model ───────────────────────────────────────────────────────────
    model = MODELS[model_name](**(model_kwargs or {}))

    visualize = (save_dir is not None) or show
    if visualize:
        from visualize import visualize_results

    print(f"\n{'═' * 50}")
    print(f" Dataset : {dataset_name}  [{task}]")
    print(f" Model   : {model_name}")
    print(f" Scenes  : {n_scenes}")
    print(f"{'═' * 50}\n")

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
                save_dir     = save_dir,
                show         = show,
            )


if __name__ == "__main__":
    # Example — Proba-V SR
    run_pipeline(
        dataset_name="probav",
        base_path="/path/to/probav_data/train",
        model_name="bicubic",
        n_scenes=1,
    )
