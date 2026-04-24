"""
pipeline.py — Unified SR Evaluation Pipeline
"""

import numpy as np
from datasets.probav import ProbaVDataset
from models.baselines.baseline_upscale import BicubicModel
from models.baselines.central_tendency import MedianModel
from models.RAMS.model import RAMSModel
from metrics.metrics import evaluate

DATASETS = {"probav": ProbaVDataset}
MODELS   = {
    "bicubic": lambda: BicubicModel(),
    "median":  lambda: MedianModel(),
    "rams":    lambda: RAMSModel(band='NIR'),
}
METRICS = ["psnr", "ssim", "cpsnr", "sam", "ergas"]

def get_dataset(name: str, base_path: str):
    return DATASETS[name](base_path=base_path)

def get_model(name: str):
    return MODELS[name]()

def filter_metrics(results: dict, selected: list) -> dict:
    return {k: v for k, v in results.items() if k in selected}

def run_pipeline(dataset_name="probav", base_path="path", model_name="bicubic", metrics=None, n_scenes=1):
    if metrics is None: metrics = METRICS
    
    dataset = get_dataset(dataset_name, base_path)
    model   = get_model(model_name)
    all_results = []

    for i in range(n_scenes):
        # Unpack 5 items now: lr, hr, lr_mask, hr_mask, name
        lr, hr, lr_mask, hr_mask, name = dataset.load_sample(i)

        print(f"[{i+1}/{n_scenes}] Scene: {name}")
        
        # Run prediction
        predictions = model.predict(lr)

        # Pass the HR mask (384x384) to evaluate() instead of the LR mask
        scores = evaluate(predictions, hr, hr_mask=hr_mask)
        scores = filter_metrics(scores, metrics)

        print("  Metrics:")
        for k, v in scores.items():
            print(f"    {k.upper():<6}: {v:.4f}")

        all_results.append({"scene": name, **scores})

    return all_results