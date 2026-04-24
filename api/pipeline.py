"""
pipeline.py — Unified SR Evaluation Pipeline
Registry-based for easy expansion.
"""

import numpy as np

# Imports
from datasets.probav.probav import ProbaVDataset
from models.baselines.baseline_upscale import BicubicModel
from models.baselines.central_tendency import MedianModel
from models.RAMS.model import RAMSModel
from metrics.metrics import evaluate

# ─────────────────────────────────────────────────────────────────────────────
# Registry — Add new datasets / models here
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "probav": ProbaVDataset,
    # "new_dataset": NewDatasetClass,
}

MODELS = {
    "bicubic": lambda: BicubicModel(),
    "median":  lambda: MedianModel(),
    "rams":    lambda: RAMSModel(band='NIR'),
    # "new_model": lambda: NewModelClass(),
}

METRICS = ["psnr", "ssim", "cpsnr", "sam", "ergas"]

# ─────────────────────────────────────────────────────────────────────────────
# Logic
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(dataset_name, base_path, model_name, metrics=None, n_scenes=1):
    if metrics is None: metrics = METRICS

    # Get the dataset and model classes from the Registry
    dataset = DATASETS[dataset_name](base_path=base_path)
    model   = MODELS[model_name]()

    print(f"\n{'═' * 50}")
    print(f" Dataset : {dataset_name}")
    print(f" Model   : {model_name}")
    print(f" Scenes  : {n_scenes}")
    print(f"{'═' * 50}\n")

    for i in range(n_scenes):
        # 1. Unpack the 5 items returned by the updated dataset
        lr, hr, _, hr_mask, name = dataset.load_sample(i)

        print(f"[{i+1}/{n_scenes}] Scene: {name}")

        # 2. Run prediction
        predictions = model.predict(lr)

        # 3. Evaluate with the HR mask
        scores = evaluate(predictions, hr, hr_mask=hr_mask)
        
        # 4. Print metrics
        for k, v in scores.items():
            if k in metrics:
                print(f"    {k.upper():<6}: {v:.4f}")

if __name__ == "__main__":
    # Example direct run
    run_pipeline(
        dataset_name="probav",
        base_path="/Users/royana/Desktop/probav_data/train",
        model_name="bicubic",
        n_scenes=1
    )