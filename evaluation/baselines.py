"""
run_baselines.py
────────────────
Runs the two baselines from aggregate.py on the Proba-V dataset:
  1. baseline_upscale  — ESA competition baseline (bicubic of clearest frames)
  2. central_tendency  — median aggregation with mask-aware NaN handling

Usage:
    python -m benchmark.run_baselines \
        --data_path /Users/royana/Desktop/probav_data/train \
        --channel RED \
        --n_scenes 10 \
        --save_dir ./baseline_outputs
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.io import highres_image, lowres_image_iterator, all_scenes_paths
from evaluation.aggregate import baseline_upscale, central_tendency
from data.transforms import bicubic_upscaling


def run_scene(scene_path: str, save_dir: str = None, show: bool = False) -> dict:
    """
    Run both baselines on a single scene and return results.

    Returns dict with keys:
        scene_path, baseline_sr, median_sr, hr, hr_mask
    """
    scene_name = os.path.basename(scene_path)

    # ── Load HR ground truth ──────────────────────────────────────────────────
    hr, hr_mask = highres_image(scene_path, img_as_float=True)

    # ── Baseline 1: ESA bicubic baseline ─────────────────────────────────────
    baseline_sr = baseline_upscale(scene_path)          # (384, 384) float64

    # ── Baseline 2: Median aggregation (mask-aware) ───────────────────────────
    lr_frames = list(lowres_image_iterator(scene_path, img_as_float=True))
    median_lr  = central_tendency(lr_frames, agg_with='median',
                                  only_clear=True, fill_obscured=True)
    median_sr  = bicubic_upscaling(median_lr)           # (384, 384) float64

    # ── Optional visualisation ────────────────────────────────────────────────
    if save_dir or show:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor='#0f1117')
        for ax in axes:
            ax.set_facecolor('#0f1117')

        # Best LR frame for reference
        best_lr = max(lr_frames, key=lambda x: x[1].sum())[0]
        axes[0].imshow(best_lr, cmap='gray')
        axes[0].set_title('Best LR frame', color='white', fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(baseline_sr, cmap='gray')
        axes[1].set_title('ESA Bicubic Baseline', color='#e8a838', fontsize=10)
        axes[1].axis('off')

        axes[2].imshow(median_sr, cmap='gray')
        axes[2].set_title('Median Aggregation', color='#5b8dee', fontsize=10)
        axes[2].axis('off')

        axes[3].imshow(hr, cmap='gray')
        axes[3].set_title('HR Ground Truth', color='#4caf7d', fontsize=10)
        axes[3].axis('off')

        plt.suptitle(f'Scene: {scene_name}', color='white', fontsize=12)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{scene_name}.png')
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"  Saved → {save_path}")

        if show:
            plt.show()

        plt.close(fig)

    return {
        'scene':       scene_name,
        'baseline_sr': baseline_sr,
        'median_sr':   median_sr,
        'hr':          hr,
        'hr_mask':     hr_mask,
    }


def run_all(data_path: str,
            channel: str = 'RED',
            n_scenes: int = 10,
            save_dir: str = None,
            show: bool = False) -> list:
    """
    Run baselines on the first n_scenes scenes and return all results.
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

    print(f"\n{'─'*55}")
    print(f"  Proba-V Baseline Runner")
    print(f"  Channel: {channel}  |  Scenes: {len(scenes)}")
    print(f"{'─'*55}\n")

    results = []
    for scene_path in tqdm(scenes, desc='Running baselines'):
        result = run_scene(scene_path, save_dir=save_dir, show=show)
        results.append(result)

    print(f"\n✓ Done. Processed {len(results)} scenes.")
    if save_dir:
        print(f"  Outputs saved to: {save_dir}")

    return results


def parse_args():
    p = argparse.ArgumentParser(description='Run Proba-V baselines.')
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--channel',   type=str, default='RED', choices=['RED', 'NIR'])
    p.add_argument('--n_scenes',  type=int, default=10)
    p.add_argument('--save_dir',  type=str, default='./baseline_outputs')
    p.add_argument('--show',      action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    run_all(
        data_path=args.data_path,
        channel=args.channel,
        n_scenes=args.n_scenes,
        save_dir=args.save_dir,
        show=args.show,
    )


if __name__ == '__main__':
    main()