"""
cli/main.py — Command Line Interface for the SR Evaluation Pipeline

Usage examples:
    python -m cli.main --dataset probav --model bicubic
    python -m cli.main --dataset probav --model median --metrics cpsnr ssim
    python -m cli.main --dataset probav --model rams --n_scenes 10
    python -m cli.main --dataset probav --model bicubic --metrics cpsnr --n_scenes 5
"""

import argparse
from api.pipeline import run_pipeline, DATASETS, MODELS, METRICS


def parse_args():
    p = argparse.ArgumentParser(
        description="Proba-V SR Evaluation Benchmark",
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
        help="Path to dataset root folder",
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
        default=METRICS,
        choices=METRICS,
        help=f"Metrics to report (default: all). Choose from: {METRICS}",
    )
    p.add_argument(
        "--n_scenes",
        type=int,
        default=1,
        help="Number of scenes to evaluate (default: 1)",
    )

    return p.parse_args()


def main():
    args = parse_args()

    run_pipeline(
        dataset_name = args.dataset,
        base_path    = args.data_path,
        model_name   = args.model,
        metrics      = args.metrics,
        n_scenes     = args.n_scenes,
    )


if __name__ == "__main__":
    main()