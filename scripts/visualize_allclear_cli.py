#!/usr/bin/env python3
"""
Command-line entry point for AllClear visualization.
"""

import argparse
from pathlib import Path

from utils import ensure_deps, show_grid
from data.allclear import load_batch, save_pngs


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize AllClear dataset.")
    parser.add_argument(
        "--data",
        type=str,
        default="allclear_dataset",
        help="Path to dataset folder (default: allclear_dataset)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="1-based index of the first image to process",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of images to process starting from --start",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Show on-screen grid preview instead of saving PNGs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output folder for PNGs "
             "(default: <data_parent>/truecolor_previews_universal)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Make sure deps exist
    ensure_deps()

    # 2) Load batch
    imgs, paths, indices, skipped_empty, total_files = load_batch(
        data_root=args.data,
        start=args.start,
        count=args.count,
    )

    if not imgs:
        print("❌ No non-empty tiles found in the requested range.")
        return

    print(f"🛰 Found {total_files} total TIFFs under {args.data}")
    print(
        f"▶ Loaded {len(imgs)} non-empty tiles "
        f"in range [{args.start}–{args.start + args.count - 1}]"
    )
    if skipped_empty:
        print(f"ℹ️ Skipped {skipped_empty} empty/nodata tiles.")

    # 3) Either show grid or save PNGs
    if args.grid:
        titles = [p.name for p in paths]
        show_grid(imgs, titles)
    else:
        data_root = Path(args.data)
        if args.output is None:
            out_dir = data_root.parent / "truecolor_previews_universal"
        else:
            out_dir = Path(args.output)

        save_pngs(imgs, paths, indices, out_dir)


if __name__ == "__main__":
    main()
