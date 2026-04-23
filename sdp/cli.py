"""
Terminal interface for the SDP-2026 unified API.

Run via:
    python -m sdp --help
    python -m sdp <subcommand> --help

Subcommands
-----------
    describe         Print supported datasets, types, models and known limitations.
    set-data-path    Persist a default data path in $SDP_DATA_PATH for this shell.
    load-dataset     Validate a dataset config (does not run training).
    benchmark        Train / inference for AllClear (wraps allclear.benchmark).
    visualize        Render side-by-side panels for a finished run.
    metrics          Compute quality metrics for a finished run.
    pipeline         load-dataset -> benchmark -> metrics -> visualize, end-to-end.

Argument conventions
--------------------
* All paths accept relative or absolute values; relative paths resolve
  against the repository root.
* Required arguments are flagged in --help with "(required)".
* Limitations specific to each dataset are surfaced via `describe`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import config
from api import (
    DatasetHandle,
    describe,
    inference,
    load_dataset,
    metrics,
    set_data_path,
    train,
    visualize,
)


# --------------------------------------------------------------------- #
# Common pieces
# --------------------------------------------------------------------- #
EPILOG = """\
Examples
--------
  # 1) Inspect what's supported
  python -m sdp describe
  python -m sdp describe allclear

  # 2) Set a default data path
  python -m sdp set-data-path D:\\satellite\\data

  # 3) Run AllClear benchmark on CPU
  python -m sdp benchmark \\
      --dataset allclear \\
      --variant uncrtaints \\
      --device cpu \\
      --dataset-fpath external/metadata/datasets/test_tx3_s2-s1_100pct_1proi_local.json \\
      --aux-sensors s1 \\
      --aux-data cld_shdw dw

  # 4) Visualize the run
  python -m sdp visualize \\
      --run-dir results/baseline/uncrtaints/utae/AllClear/test_tx3_s2-s1_100pct_1proi_local \\
      --json   external/metadata/datasets/test_tx3_s2-s1_100pct_1proi_local.json \\
      --model  uncrtaints --num 100

  # 5) Compute metrics
  python -m sdp metrics --run-dir <run_dir> --json <json> --model uncrtaints

  # 6) End-to-end pipeline
  python -m sdp pipeline \\
      --dataset allclear --variant uncrtaints --device cpu \\
      --dataset-fpath external/metadata/datasets/test_tx3_s2-s1_100pct_1proi_local.json
"""


def _add_common_dataset_args(p: argparse.ArgumentParser, *, require_dataset: bool = True) -> None:
    p.add_argument(
        "--dataset",
        choices=config.SUPPORTED_DATASETS,
        required=require_dataset,
        help="Which dataset to use (required).",
    )
    p.add_argument(
        "--variant",
        default=None,
        help="Model / training variant. AllClear: uncrtaints, ctgan, utilise, "
             "leastcloudy, mosaicing, dae, pmaa, diffcr. "
             "Proba-V: highresnet, deepsum, sar_sr.",
    )
    p.add_argument(
        "--type",
        choices=config.SUPPORTED_TYPES,
        default="cloud_removal",
        help="Task type. Default: cloud_removal.",
    )
    p.add_argument(
        "--data-path",
        default=None,
        help="Root data directory (overrides $SDP_DATA_PATH).",
    )
    # AllClear specifics
    p.add_argument(
        "--dataset-fpath",
        default=None,
        help="AllClear dataset metadata JSON. Required for AllClear runs.",
    )
    p.add_argument("--main-sensor", default="s2_toa", help="AllClear main sensor.")
    p.add_argument(
        "--aux-sensors",
        nargs="*",
        default=[],
        help="AllClear auxiliary sensors (e.g. s1 landsat8).",
    )
    p.add_argument(
        "--aux-data",
        nargs="*",
        default=["cld_shdw", "dw"],
        help="AllClear auxiliary data layers (default: cld_shdw dw).",
    )
    p.add_argument("--tx", type=int, default=3, help="AllClear temporal length (default: 3).")
    p.add_argument(
        "--target-mode",
        choices=("s2p", "s2s"),
        default="s2p",
        help="AllClear target mode (default: s2p).",
    )
    p.add_argument(
        "--selected-rois",
        default="all",
        help="AllClear selected ROIs ('all' or comma-separated list).",
    )
    # Proba-V specifics
    p.add_argument("--split", default="test", help="Proba-V split: train/test.")
    p.add_argument("--scale", type=int, default=3, help="Proba-V SR scale factor.")
    p.add_argument("--band", default="NIR", choices=("NIR", "RED"), help="Proba-V band.")


def _add_benchmark_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="cpu", help="cpu / cuda / cuda:0.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--draw-vis", type=int, default=0, choices=(0, 1))
    p.add_argument(
        "--experiment-output-path",
        default=None,
        help="Where to write predictions/metadata. Default: results/baseline/<model>/<exp>",
    )
    p.add_argument(
        "--uc-baseline-base-path",
        default="baselines/UnCRtainTS/model",
        help="UnCRtainTS baseline path (relative to external/).",
    )
    p.add_argument(
        "--uc-weight-folder",
        default="checkpoints",
        help="UnCRtainTS weight folder (relative to baseline path).",
    )
    p.add_argument("--uc-exp-name", default="utae", help="UnCRtainTS experiment name.")
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Pass additional raw flags through to allclear.benchmark "
             "(everything after --extra is forwarded verbatim).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the resolved command without executing it.",
    )


def _selected_rois_arg(value: str):
    if value == "all":
        return "all"
    return [v.strip() for v in value.split(",") if v.strip()]


# --------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------- #
def _cmd_describe(args: argparse.Namespace) -> int:
    info = describe(args.name) if args.name else describe()
    print(json.dumps(info, indent=2))
    return 0


def _cmd_set_data_path(args: argparse.Namespace) -> int:
    resolved = set_data_path(args.path)
    print(f"SDP_DATA_PATH = {resolved}")
    print("Note: the env var is set for this Python process. To persist it in "
          "your shell, export it manually.")
    return 0


def _build_handle(args: argparse.Namespace) -> DatasetHandle:
    extras = {}
    handle = load_dataset(
        name=args.dataset,
        variant=args.variant,
        type=args.type,
        data_path=args.data_path,
        dataset_fpath=args.dataset_fpath,
        selected_rois=_selected_rois_arg(args.selected_rois),
        main_sensor=args.main_sensor,
        aux_sensors=args.aux_sensors,
        aux_data=args.aux_data,
        tx=args.tx,
        target_mode=args.target_mode,
        split=args.split,
        scale=args.scale,
        band=args.band,
        **extras,
    )
    return handle


def _cmd_load_dataset(args: argparse.Namespace) -> int:
    handle = _build_handle(args)
    print("Loaded dataset handle:")
    print(json.dumps(handle.as_kwargs(), indent=2, default=str))
    if handle.torch_dataset is not None:
        try:
            print(f"  len(torch_dataset) = {len(handle.torch_dataset)}")
        except Exception as e:
            print(f"  torch_dataset present but len() failed: {e}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    handle = _build_handle(args)
    handle = train(
        handle,
        model_name=args.variant or handle.variant,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        draw_vis=args.draw_vis,
        experiment_output_path=Path(args.experiment_output_path) if args.experiment_output_path else None,
        uc_baseline_base_path=Path(args.uc_baseline_base_path),
        uc_weight_folder=Path(args.uc_weight_folder),
        uc_exp_name=args.uc_exp_name,
        extra_args=args.extra,
        dry_run=args.dry_run,
    )
    print(f"\n[sdp] run_dir = {handle.run_dir}")
    return 0


def _cmd_visualize(args: argparse.Namespace) -> int:
    # Build a thin handle so we can reuse the wrapper.
    handle = DatasetHandle(
        name="allclear",
        type="cloud_removal",
        variant=args.model,
        dataset_fpath=Path(args.json),
        run_dir=Path(args.run_dir),
        model_name=args.model,
    )
    out = visualize(
        handle,
        num=args.num,
        start=args.start,
        out=Path(args.out) if args.out else None,
        no_stretch=args.no_stretch,
        dry_run=args.dry_run,
    )
    print(f"[sdp] visualization output: {out}")
    return 0


def _cmd_metrics(args: argparse.Namespace) -> int:
    handle = DatasetHandle(
        name="allclear",
        type="cloud_removal",
        variant=args.model,
        dataset_fpath=Path(args.json),
        run_dir=Path(args.run_dir),
        model_name=args.model,
    )
    metrics(
        handle,
        out_prefix=args.out_prefix,
        device=args.device,
        dry_run=args.dry_run,
    )
    return 0


def _cmd_pipeline(args: argparse.Namespace) -> int:
    handle = _build_handle(args)
    handle = train(
        handle,
        model_name=args.variant or handle.variant,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        draw_vis=args.draw_vis,
        experiment_output_path=Path(args.experiment_output_path) if args.experiment_output_path else None,
        uc_baseline_base_path=Path(args.uc_baseline_base_path),
        uc_weight_folder=Path(args.uc_weight_folder),
        uc_exp_name=args.uc_exp_name,
        extra_args=args.extra,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        metrics(handle)
        visualize(handle, num=args.num)
    print(f"[sdp] pipeline complete: {handle.run_dir}")
    return 0


# --------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdp",
        description="SDP-2026 unified API: dataset loading, benchmarking, "
                    "visualization and metrics for AllClear and Proba-V.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0",
    )

    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    # describe
    p = sub.add_parser("describe", help="List supported datasets / models / limitations.")
    p.add_argument("name", nargs="?", default=None,
                   help="Optional dataset name to drill into.")
    p.set_defaults(func=_cmd_describe)

    # set-data-path
    p = sub.add_parser("set-data-path", help="Set the global data path.")
    p.add_argument("path", help="Absolute or relative path to the data root.")
    p.set_defaults(func=_cmd_set_data_path)

    # load-dataset
    p = sub.add_parser("load-dataset", help="Build and validate a dataset handle.")
    _add_common_dataset_args(p)
    p.set_defaults(func=_cmd_load_dataset)

    # benchmark (train + inference for AllClear)
    p = sub.add_parser(
        "benchmark",
        help="Run training/inference (wraps allclear.benchmark).",
    )
    _add_common_dataset_args(p)
    _add_benchmark_args(p)
    p.set_defaults(func=_cmd_benchmark)

    # visualize
    p = sub.add_parser("visualize", help="Render visualization panels.")
    p.add_argument("--run-dir", required=True, help="Run directory from a benchmark (required).")
    p.add_argument("--json", required=True, help="Dataset JSON used for the run (required).")
    p.add_argument("--model", required=True,
                   choices=["ctgan", "uncrtaints"],
                   help="Model whose predictions to visualize (required).")
    p.add_argument("--num", type=int, default=20, help="Number of samples to render.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--out", default=None, help="Output folder for PNGs.")
    p.add_argument("--no-stretch", action="store_true", help="Disable RGB stretching.")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_cmd_visualize)

    # metrics
    p = sub.add_parser("metrics", help="Compute MAE/RMSE/PSNR/SAM/SSIM/LPIPS/FID.")
    p.add_argument("--run-dir", required=True, help="Run directory from a benchmark (required).")
    p.add_argument("--json", required=True, help="Dataset JSON used for the run (required).")
    p.add_argument("--model", required=True,
                   choices=["ctgan", "uncrtaints"],
                   help="Model whose predictions to score (required).")
    p.add_argument("--out-prefix", default=None)
    p.add_argument("--device", default=None, help="cpu / cuda / cuda:0.")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_cmd_metrics)

    # pipeline (end-to-end convenience)
    p = sub.add_parser(
        "pipeline",
        help="Run load-dataset -> benchmark -> metrics -> visualize end-to-end.",
    )
    _add_common_dataset_args(p)
    _add_benchmark_args(p)
    p.add_argument("--num", type=int, default=20,
                   help="Visualization sample count (default 20).")
    p.set_defaults(func=_cmd_pipeline)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
