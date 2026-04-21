"""
Full evaluation pipeline for Proba-V SR benchmark.

Pipeline per scene:
  1. Load LR frames + HR ground truth
  2. Run model (bicubic baseline by default)
  3. Compute metrics on model output vs HR
  4. Visualize and save results
  5. Print summary table

Usage:
    python -m benchmark.run_evaluation \
        --data_path /Users/royana/Desktop/probav_data/train \
        --channel RED \
        --n_scenes 10 \
        --save_dir ./evaluation_outputs
"""
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.io import highres_image, lowres_image_iterator
from evaluation.aggregate import baseline_upscale, central_tendency
from data.transforms import bicubic_upscaling
from evaluation.metrics import evaluate, print_scores
from evaluation.visualize_outputs import visualize_output


# ─────────────────────────────────────────────────────────────────────────────
# Model registry — add new models here
# ─────────────────────────────────────────────────────────────────────────────

def get_model(model_name: str):
    """
    Returns a callable: scene_path -> sr (np.ndarray float64)
    Add new models here as the benchmark grows.
    """
    if model_name == 'bicubic':
        return lambda path: baseline_upscale(path)

    elif model_name == 'median':
        def median_model(path):
            frames = list(lowres_image_iterator(path, img_as_float=True))
            median_lr = central_tendency(frames, agg_with='median',
                                         only_clear=True, fill_obscured=True)
            return bicubic_upscaling(median_lr)
        return median_model

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: bicubic, median")



# Core evaluation loop


def run_evaluation(data_path: str,
                   channel: str = 'RED',
                   model_name: str = 'bicubic',
                   n_scenes: int = 10,
                   save_dir: str = None,
                   show: bool = False,
                   scale: int = 3) -> pd.DataFrame:
    """
    Run full evaluation pipeline on n_scenes scenes.

    Returns:
        DataFrame with one row per scene and columns for each metric.
    """
    channel_path = os.path.join(data_path, channel)
    if not os.path.exists(channel_path):
        raise FileNotFoundError(f"Channel folder not found: {channel_path}")

    scenes = sorted([
        os.path.join(channel_path, f)
        for f in os.listdir(channel_path)
        if not f.startswith('.') and
        os.path.isdir(os.path.join(channel_path, f))
    ])[:n_scenes]

    model_fn   = get_model(model_name)
    model_label = model_name.capitalize() + ' Baseline'

    print(f"\n{'═'*55}")
    print(f"  Proba-V SR Evaluation Benchmark")
    print(f"  Channel : {channel}")
    print(f"  Model   : {model_label}")
    print(f"  Scenes  : {len(scenes)}")
    print(f"{'═'*55}\n")

    results = []

    for scene_path in tqdm(scenes, desc='Evaluating'):
        scene_name = os.path.basename(scene_path)

        # ── 1. Load data ──────────────────────────────────────────────────────
        hr, hr_mask = highres_image(scene_path, img_as_float=True)
        lr_frames   = [(lr, mask) for lr, mask in
                       lowres_image_iterator(scene_path, img_as_float=True)]
        lr_stack    = np.stack([f[0] for f in lr_frames], axis=0)

        # ── 2. Run model ──────────────────────────────────────────────────────
        sr = model_fn(scene_path)

        # ── 3. Compute metrics AFTER model output ─────────────────────────────
        scores = evaluate(sr, hr, hr_mask, scale=scale)

        # ── 4. Visualize ──────────────────────────────────────────────────────
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, model_name, f'{scene_name}.png')

        visualize_output(
            lr_stack   = lr_stack,
            sr         = sr,
            hr         = hr,
            hr_mask    = hr_mask,
            scores     = scores,
            scene_name = scene_name,
            model_name = model_label,
            save_path  = save_path,
            show       = show,
        )

        results.append({'scene': scene_name, **scores})

    # ── 5. Summary table ──────────────────────────────────────────────────────
    df = pd.DataFrame(results)

    print(f"\n{'═'*55}")
    print(f"  Results — {model_label} on {channel} channel")
    print(f"{'═'*55}")
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print(f"\n{'─'*55}")
    print(f"  Averages")
    print(f"{'─'*55}")
    avgs = df.drop(columns='scene').mean()
    for metric, val in avgs.items():
        direction = '↑ higher better' if metric in ['psnr', 'ssim', 'cpsnr'] else '↓ lower better'
        print(f"  {metric.upper():<8}: {val:.4f}   ({direction})")
    print(f"{'═'*55}\n")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f'{model_name}_{channel}_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"  Results saved → {csv_path}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Run Proba-V SR evaluation benchmark.')
    p.add_argument('--data_path',  type=str, required=True,
                   help='Path to probav_data/train folder')
    p.add_argument('--channel',    type=str, default='RED',
                   choices=['RED', 'NIR'])
    p.add_argument('--model',      type=str, default='bicubic',
                   choices=['bicubic', 'median'],
                   help='Model to evaluate')
    p.add_argument('--n_scenes',   type=int, default=10,
                   help='Number of scenes to evaluate')
    p.add_argument('--save_dir',   type=str, default='./evaluation_outputs',
                   help='Directory to save results and visualizations')
    p.add_argument('--show',       action='store_true',
                   help='Show matplotlib windows')
    return p.parse_args()


def main():
    args = parse_args()
    run_evaluation(
        data_path  = args.data_path,
        channel    = args.channel,
        model_name = args.model,
        n_scenes   = args.n_scenes,
        save_dir   = args.save_dir,
        show       = args.show,
    )


if __name__ == '__main__':
    main()