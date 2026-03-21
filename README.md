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

# Run Instructions

## 1. Repository structure

This project uses:

- `external/` for the AllClear codebase and baselines
- `results/` for generated outputs
- `scripts/` for helper launch scripts
- `src/` for custom visualization and utility scripts

Example structure:

```text
SDP-2026-1/
├─ external/
├─ results/
├─ scripts/
├─ src/
├─ README.md
└─ .gitignore
```

---

## 2. Environment setup

Open a terminal in the repository root:

```bash
cd SDP-2026-1
```

Activate the environment:

### Conda
```bash
conda activate allclear
```

If needed, install dependencies according to the AllClear project requirements inside `external/`.

---

## 3. Run a benchmark

Benchmarks are run from the `external/` directory because `allclear` is imported as a Python module.

### Example: UnCRtainTS on CPU

```bash
cd external
python -m allclear.benchmark ^
  --dataset-fpath "metadata\datasets\test_on_dataset_root_EXISTING_DW.json" ^
  --model-name uncrtaints ^
  --device cpu ^
  --main-sensor s2_toa ^
  --aux-sensors s1 ^
  --aux-data cld_shdw dw ^
  --target-mode s2p ^
  --tx 3 ^
  --batch-size 1 ^
  --num-workers 0 ^
  --draw-vis 0 ^
  --experiment-output-path "..\results\baseline" ^
  --uc-baseline-base-path baselines\UnCRtainTS\model ^
  --uc-weight-folder checkpoints ^
  --uc-exp-name multitemporalL2
```

### Output

Results will be saved under a folder like:

```text
results/baseline/<model_name>/<run_name>/AllClear/<dataset_name>/
```

Typical files include:

- `uncrtaints_predictions.pt`
- `uncrtaints_metadata.csv`
- `uncrtaints_lulc_metrics.csv`
- `uncrtaints_aggregated_metrics.csv`

---

## 4. Visualize predictions

Return to the repo root and run the visualization script.

### Example

```bash
cd ..
python src\visualize_allclear.py ^
  --run_dir "results\baseline\uncrtaints\multitemporalL2\AllClear\test_on_dataset_root_EXISTING_DW" ^
  --json "external\metadata\datasets\test_on_dataset_root_EXISTING_DW.json" ^
  --num 10
```

### Output

PNG panels will be saved in:

```text
<run_dir>/vis_json/
```

Each panel includes:

- Input image
- Prediction
- Ground truth target
- Absolute difference
- Cloud/shadow mask
- Dynamic World map

---

## 5. Using helper `.cmd` files

If you use Windows helper scripts, place them in `scripts/`.

### Run benchmark
```bash
scripts\run_benchmark.cmd
```

### Run visualization
```bash
scripts\run_visualize.cmd
```

These scripts should be launched from the repository root.

---

## 6. Important notes

### Module path
`benchmark.py` must be run from `external/`, not from `external/allclear/`, because it is executed as:

```bash
python -m allclear.benchmark
```

### Outputs
Do not save generated outputs inside `external/`, since it is a submodule.  
All generated files should go into `results/`.

### Predictions
During inference, the model sees the cloudy input image only.  
The target image is used only for evaluation.

---

## 7. Common errors

### `No module named 'allclear'`
Cause: running benchmark from the wrong directory.  
Fix: run it from `external/`.

### `WinError 123`
Cause: bad Windows path handling in output folder creation.  
Fix: use `Path(...)` instead of splitting paths manually.

### OpenMP error
If visualization crashes with an OpenMP duplication error, use:

```bash
set KMP_DUPLICATE_LIB_OK=TRUE
```

before running the script.

---

## 8. Minimal workflow

From repo root:

```bash
conda activate allclear
cd external
python -m allclear.benchmark ...
cd ..
python src\visualize_allclear.py ...
```

---

## 9. Recommended helper script layout

### `scripts/run_benchmark.cmd`
```bat
@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d %~dp0external

python -m allclear.benchmark ^
  --dataset-fpath "%~dp0..\external\metadata\datasets\test_tx3_s2-s1_100pct_1proi_LOCAL_READY.json" ^
  --model-name uncrtaints ^
  --device cpu ^
  --main-sensor s2_toa ^
  --aux-sensors s1 ^
  --aux-data cld_shdw dw ^
  --target-mode s2p ^
  --tx 3 ^
  --batch-size 1 ^
  --num-workers 0 ^
  --draw-vis 0 ^
  --experiment-output-path "%~dp0..\results\baseline\uncrtaints" ^
  --uc-baseline-base-path baselines\UnCRtainTS\model ^
  --uc-weight-folder checkpoints ^
  --uc-exp-name noSAR_1 ^

pause
```

### `scripts/run_visualize.cmd`
```bat
@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d %~dp0..

python src\visualize_allclear.py ^
  --run_dir "results\baseline\uncrtaints\multitemporalL2\AllClear\test_on_dataset_root_EXISTING_DW" ^
  --json "external\metadata\datasets\test_on_dataset_root_EXISTING_DW.json" ^
  --num 10

pause
```

---

