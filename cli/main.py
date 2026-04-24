"""
cli/main.py — Command Line Interface for the Unified Satellite DL Evaluation Pipeline

Usage examples:

  Proba-V super-resolution
    python -m cli.main --dataset probav --model bicubic --data_path /path/to/probav/train
    python -m cli.main --dataset probav --model median  --metrics cpsnr ssim
    python -m cli.main --dataset probav --model rams    --n_scenes 10

  AllClear cloud removal
    python -m cli.main --dataset allclear --model leastcloudy \\
        --data_path external/metadata/datasets/test_tx3_s2-s1_100pct_1proi_local.json
    python -m cli.main --dataset allclear --model mosaicing \\
        --data_path external/metadata/datasets/test_tx3_s2-s1_100pct_1proi_local.json \\
        --aux_sensors s1 --n_scenes 5

  Visualization
    python -m cli.main --dataset allclear --model leastcloudy \\
        --data_path external/metadata/datasets/test_tx3_s2-s1_100pct_1proi_local.json \\
        --save_dir results/vis
    python -m cli.main --dataset allclear --model leastcloudy \\
        --data_path external/metadata/datasets/test_tx3_s2-s1_100pct_1proi_local.json \\
        --show
"""

import argparse
from api.pipeline import (
    run_pipeline, DATASETS, MODELS, SR_METRICS, ALLCLEAR_METRICS, DATASET_TASK,
)

ALL_METRICS = list(dict.fromkeys(SR_METRICS + ALLCLEAR_METRICS))  # deduped, order preserved


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified Satellite DL Evaluation Benchmark",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument(
        "--dataset",
        type=str,
        default="probav",
        choices=list(DATASETS),
        help="Dataset to evaluate on (default: probav)",
    )
    p.add_argument(
        "--data_path",
        type=str,
        default="path_to_your_data",
        help=(
            "Path to dataset root folder (Proba-V) or JSON metadata file (AllClear).\n"
            "AllClear example: external/metadata/datasets/test_tx3_s2-s1_100pct_1proi_local.json"
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default="bicubic",
        choices=list(MODELS),
        help="Model to run (default: bicubic)",
    )
    p.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        choices=ALL_METRICS,
        help=(
            "Metrics to report (default: task-specific set).\n"
            f"  SR metrics      : {SR_METRICS}\n"
            f"  AllClear metrics: {ALLCLEAR_METRICS}"
        ),
    )
    p.add_argument(
        "--n_scenes",
        type=int,
        default=1,
        help="Number of scenes / samples to evaluate (default: 1)",
    )

    # ── AllClear-specific options ──────────────────────────────────────────────
    ac = p.add_argument_group("AllClear options")
    ac.add_argument(
        "--aux_sensors",
        type=str,
        nargs="*",
        default=[],
        help="Auxiliary sensors to include, e.g. --aux_sensors s1 landsat8",
    )
    ac.add_argument(
        "--tx",
        type=int,
        default=3,
        help="Number of input timesteps for AllClear (default: 3)",
    )
    ac.add_argument(
        "--target_mode",
        type=str,
        default="s2p",
        choices=["s2p", "s2s"],
        help="AllClear target mode: s2p (seq2point) or s2s (seq2seq) (default: s2p)",
    )
    ac.add_argument(
        "--center_crop_size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("W", "H"),
        help="Center crop dimensions for AllClear images (default: 256 256)",
    )

    # ── UnCRtainTS-specific options ────────────────────────────────────────────
    uc = p.add_argument_group("UnCRtainTS options (only used when --model uncrtaints)")
    uc.add_argument(
        "--uc_baseline_dir",
        type=str,
        default="models/allclear_baselines/UnCRtainTS/model",
        help="Path to UnCRtainTS model directory (added to sys.path)",
    )
    uc.add_argument(
        "--uc_checkpoints_dir",
        type=str,
        default="models/allclear_baselines/UnCRtainTS/model/checkpoints",
        help="Directory containing UnCRtainTS experiment subfolders",
    )
    uc.add_argument(
        "--uc_exp_name",
        type=str,
        default="noSAR_1",
        help="UnCRtainTS experiment name (subfolder with conf.json + model.pth.tar)",
    )
    uc.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for deep-learning models, e.g. cpu or cuda (default: cpu)",
    )

    # ── visualization options ──────────────────────────────────────────────────
    vis = p.add_argument_group("Visualization options")
    vis.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save PNG visualizations (e.g. results/vis). Skipped if not set.",
    )
    vis.add_argument(
        "--show",
        action="store_true",
        help="Display visualizations interactively (requires a display).",
    )

    return p.parse_args()


def main():
    args = parse_args()

    task = DATASET_TASK[args.dataset]

    dataset_kwargs = {}
    if task == "cloud_removal":
        dataset_kwargs = {
            "aux_sensors":      args.aux_sensors,
            "tx":               args.tx,
            "target_mode":      args.target_mode,
            "center_crop_size": tuple(args.center_crop_size),
        }

    model_kwargs = None
    if args.model == "uncrtaints":
        model_kwargs = {
            "baseline_dir":    args.uc_baseline_dir,
            "checkpoints_dir": args.uc_checkpoints_dir,
            "exp_name":        args.uc_exp_name,
            "device":          args.device,
        }

    run_pipeline(
        dataset_name = args.dataset,
        base_path    = args.data_path,
        model_name   = args.model,
        metrics      = args.metrics,
        n_scenes     = args.n_scenes,
        save_dir     = args.save_dir,
        show         = args.show,
        model_kwargs = model_kwargs,
        **dataset_kwargs,
    )


if __name__ == "__main__":
    main()
