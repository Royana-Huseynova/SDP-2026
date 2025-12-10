#!/usr/bin/env python3
"""
Shared utilities for SDP-2026:
- Dependency management
- Image normalization (stretch)
- Generic .tif -> RGB conversion for Sentinel-style data
- Grid visualization
"""

import sys
import subprocess
import importlib
from pathlib import Path

# Packages we need for visualization
REQUIRED_PACKAGES = [
    "numpy",
    "matplotlib",
    "pillow",
    "tifffile",
    "rasterio",
    "imagecodecs",
    "zstandard",
]


def ensure_deps():
    """
    Ensure all REQUIRED_PACKAGES are installed.
    Safe to call from CLI or notebook.
    """
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            print(f"📦 Installing {pkg} ...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", pkg]
            )


def stretch(x):
    """
    Contrast-stretch a single band using 1st-99th percentiles.
    Returns float32 in [0, 1].
    """
    import numpy as np

    x = x.astype("float32")

    # Ignore NaNs when computing percentiles
    vals = x[~np.isnan(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype="float32")

    lo, hi = np.percentile(vals, (1, 99))

    # If there's basically no contrast, treat as empty
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype="float32")

    y = (x - lo) / (hi - lo + 1e-6)
    return np.clip(y, 0.0, 1.0)


def read_rgb(path):
    """
    Generic .tif → normalized RGB (float32, 0–1).

    - For 4+ bands: use (4, 3, 2) as Sentinel-2 true color
    - For 3 bands: assume RGB
    - For 1 band: stretch to grayscale and replicate to RGB
    - Returns None for tiles that are basically empty/nodata
    """
    import numpy as np
    import rasterio as rio

    path = Path(path)

    with rio.open(path) as ds:
        C = ds.count
        nodata = ds.nodata

        # Single-band case → grayscale
        if C < 3:
            x = ds.read(1).astype("float32")

            if nodata is not None:
                mask = (x == nodata)
            else:
                mask = (x == 0)

            if (~mask).sum() < 100:
                # basically empty tile
                return None

            x[mask] = np.nan
            x = stretch(x)
            rgb = np.stack([x] * 3, axis=-1)
            return np.nan_to_num(rgb, nan=0.0)

        # Multi-band case
        if C >= 4:
            bands = (4, 3, 2)  # Sentinel-style true color
        else:  # C == 3
            bands = (1, 2, 3)

        r, g, b = [ds.read(b).astype("float32") for b in bands]

        # Build validity mask: nodata or pure-zero triplets
        if nodata is not None:
            mask = (r == nodata) | (g == nodata) | (b == nodata)
        else:
            mask = (r == 0) & (g == 0) & (b == 0)

        if (~mask).sum() < 100:
            return None

        # Sentinel-2 reflectance scaling if necessary
        if max(r.max(), g.max(), b.max()) > 1.5:
            r, g, b = r / 10000.0, g / 10000.0, b / 10000.0

        # Mask invalid pixels as NaN so stretch() ignores them
        r = np.where(mask, np.nan, r)
        g = np.where(mask, np.nan, g)
        b = np.where(mask, np.nan, b)

        rgb = np.stack([stretch(r), stretch(g), stretch(b)], axis=-1)
        return np.nan_to_num(rgb, nan=0.0)


def show_grid(imgs, titles=None, cols=3, figsize=(12, 12)):
    """
    Show a grid of RGB images using matplotlib.
    imgs: list of H×W×3 uint8 arrays
    titles: list of strings (same length as imgs) or None
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if not imgs:
        print("⚠️ No images to display in grid.")
        return

    cols = min(cols, len(imgs))
    rows = int(np.ceil(len(imgs) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).ravel()

    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img)
        ax.axis("off")
        if titles is not None and i < len(titles):
            ax.set_title(str(titles[i]), fontsize=8)

    for ax in axes[len(imgs):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
