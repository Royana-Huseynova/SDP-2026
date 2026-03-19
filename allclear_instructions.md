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
cd /d %~dp0..\external

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

For the README, the most important thing is to make every command copy-pasteable exactly as written.
