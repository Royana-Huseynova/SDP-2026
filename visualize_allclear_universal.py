#!/usr/bin/env python3
"""
Universal AllClear Dataset Visualizer
Author: Nijat Alisoy, Huseyn Sadatkhanov, Pasha Zulfugarli, Royana Huseynova
Works on: Windows, macOS, Linux
"""
import os, sys, subprocess, platform, numpy as np
from pathlib import Path

# ============ 1. Ensure dependencies ============
REQUIRED = [
    "numpy", "matplotlib", "pillow", "tifffile",
    "rasterio", "imagecodecs", "zstandard"
]

def ensure_deps():
    import importlib
    for pkg in REQUIRED:
        try:
            importlib.import_module(pkg)
        except ImportError:
            print(f"📦 Installing {pkg} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])

ensure_deps()

import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import rasterio as rio
import argparse

# ============ 2. Utility functions ============
def stretch(x):
    x = x.astype("float32")
    pos = x[x > 0]
    if pos.size < 100:
        lo, hi = np.percentile(x, (1, 99))
    else:
        lo, hi = np.percentile(pos, (1, 99))
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)

def read_rgb(path):
    """Reads a .tif and returns normalized RGB array (float32, 0–1)."""
    with rio.open(path) as ds:
        C = ds.count
        if C >= 4:
            bands = (4, 3, 2)  # Sentinel-2 true color
        elif C == 3:
            bands = (1, 2, 3)
        else:
            x = ds.read(1).astype("float32")
            x = stretch(x)
            return np.stack([x]*3, axis=-1)

        r, g, b = [ds.read(b).astype("float32") for b in bands]
        if max(r.max(), g.max(), b.max()) > 1.5:
            r, g, b = r/10000.0, g/10000.0, b/10000.0
        rgb = np.stack([stretch(r), stretch(g), stretch(b)], axis=-1)
        return np.nan_to_num(rgb, nan=0.0)

# ============ 3. Main function ============
def main():
    parser = argparse.ArgumentParser(description="Visualize AllClear dataset.")
    parser.add_argument("--data", type=str, default="allclear_dataset", help="Path to dataset folder")
    parser.add_argument("--start", type=int, default=1, help="Start index (1-based)")
    parser.add_argument("--count", type=int, default=10, help="Number of images to export")
    parser.add_argument("--grid", action="store_true", help="Show grid preview instead of saving PNGs")
    args = parser.parse_args()

    root = Path(args.data)
    if not root.exists():
        print(f"❌ Dataset folder not found: {root}")
        sys.exit(1)

    files = sorted(root.rglob("*.tif"))
    if not files:
        print("❌ No .tif files found.")
        sys.exit(1)

    start = max(0, args.start - 1)
    end = min(len(files), start + args.count)
    batch = files[start:end]
    out = root.parent / "truecolor_previews_universal"
    out.mkdir(exist_ok=True)

    print(f"🛰 Found {len(files)} total TIFFs")
    print(f"▶ Processing {len(batch)} files ({args.start}–{args.start+len(batch)-1})\n")

    imgs = []
    for i, f in enumerate(batch, start + 1):
        try:
            rgb = read_rgb(f)
            img8 = (np.clip(rgb, 0, 1) * 255).astype("uint8")
            if args.grid:
                imgs.append(img8)
            else:
                Image.fromarray(img8).save(out / f"{i:05d}_{f.stem}.png")
                print(f"✅ Saved: {out / f'{i:05d}_{f.stem}.png'}")
        except Exception as e:
            print(f"⚠️ Skipped {f.name}: {e}")

    # ============ Optional grid display ============
    if args.grid and imgs:
        cols = min(3, len(imgs))
        rows = int(np.ceil(len(imgs) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        axes = np.array(axes).ravel()
        for i, (ax, img) in enumerate(zip(axes, imgs)):
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(batch[i].name, fontsize=8)
        for ax in axes[len(imgs):]:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    print(f"\n🎉 Done — {len(imgs) if args.grid else len(batch)} images processed successfully.")

if __name__ == "__main__":
    main()