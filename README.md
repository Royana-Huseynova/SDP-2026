# SDP-2026





# A Unified Deep Learning Benchmark for Satellite Image Restoration and Generation 

This repository contains a small, cross-platform utility script for quickly visualizing tiles from the **AllClear** (or similar) satellite image dataset. It reads `.tif` files, builds true-color RGB previews, and either saves them as PNGs or shows a grid preview for manual inspection.

---

## 1. Visualization Features

- Works on **Windows, macOS, and Linux**
- Automatically checks and installs required Python packages on first run
- Supports:
  - Multi-band Sentinel-2 style `.tif` (uses bands 4–3–2 for true color)
  - 3-band `.tif` (RGB)
  - Single-band `.tif` (stretches to grayscale and replicates to RGB)
- Automatically handles:
  - `nodata` values and pure-zero tiles
  - Empty tiles (skips them instead of saving black images)
  - Sentinel-2 reflectance scaling (divides by 10,000 when needed)
  - Contrast stretching using 1st–99th percentiles (ignoring NaNs)
- Exports:
  - Individual PNG files to a `truecolor_previews_universal/` folder  
  - Or an on-screen grid preview (`--grid`)

---

## 2. Script Overview

Main file:

- `visualize_allclear_universal.py`  
  Universal visualizer script using:
  - `rasterio` for reading `.tif` tiles
  - `numpy` for numerical operations and normalization
  - `Pillow` (`PIL.Image`) for saving PNGs
  - `matplotlib` for grid previews

Core logic:

- Recursively scans the dataset folder for all `.tif` files
- Selects a slice from the list using `--start` and `--count`
- For each tile:
  - Reads the image bands
  - Applies scaling and percentile stretching
  - Skips tiles that are essentially empty / nodata
  - Either:
    - Saves as PNG into `truecolor_previews_universal/`, or
    - Adds to an in-memory list for grid visualization (`--grid`)

---

## 3. Requirements

The script will automatically try to install these packages if they are missing:

- `numpy`
- `matplotlib`
- `pillow`
- `tifffile`
- `rasterio`
- `imagecodecs`
- `zstandard`

### Recommended environment (Windows / VS Code)

1. Install **Python 3.9+**
2. Inside the project folder, create a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

(Optional but recommended) Upgrade pip:

python -m pip install --upgrade pip


You do not need to manually install the listed packages; the script will attempt to install them on the first run. But if you want to pre-install them:

pip install numpy matplotlib pillow tifffile rasterio imagecodecs zstandard

4. Dataset Structure

By default, the script expects a folder like:

project_root/
├─ visualize_allclear_universal.py
├─ allclear_dataset/          # <-- default dataset folder
│   ├─ tile_0001.tif
│   ├─ tile_0002.tif
│   ├─ ...


Important notes:

The script recursively searches all subfolders under --data for .tif files.

You can point --data to any root folder containing .tif tiles; it does not have to be named allclear_dataset.

5. Usage

From the repository root, in a terminal (PowerShell on Windows):

python visualize_allclear_universal.py [OPTIONS]

Command-line arguments

--data PATH
Path to the dataset folder (default: allclear_dataset).

--start N
1-based index of the first image to process (default: 1).

--count K
Number of images to process starting from --start (default: 10).

--grid
If set, show an on-screen grid preview (using matplotlib) instead of saving PNGs.

Examples
5.1. Basic: save first 10 tiles as PNGs
python visualize_allclear_universal.py --data allclear_dataset --start 1 --count 10


This will:

Search allclear_dataset/ for .tif tiles

Process tiles 1–10 in sorted order

Save PNGs to:

truecolor_previews_universal/
    00001_<original_name>.png
    00002_<original_name>.png
    ...

5.2. Show a grid of previews (no files saved)
python visualize_allclear_universal.py --data allclear_dataset --start 1 --count 12 --grid


Opens a matplotlib window with up to 3 columns and multiple rows

Shows image titles with the original .tif filenames

Does not write PNGs to disk

5.3. Use a custom dataset path
python visualize_allclear_universal.py --data D:\datasets\AllClear --start 50 --count 20

6. Output

When not using --grid:

PNGs are saved under:

<dataset_parent>/
└─ truecolor_previews_universal/
    ├─ 00001_tile_0001.png
    ├─ 00002_tile_0002.png
    └─ ...


The script prints a line for each saved file:

✅ Saved: ...\truecolor_previews_universal\00001_tile_0001.png


When using --grid:

No files are written.

A single grid window pops up, and the script reports how many images were shown.
