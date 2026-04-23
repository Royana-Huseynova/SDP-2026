# `sdp/` — unified API

Single Python + CLI entry point for both datasets in this repo.

| Component | Source | Wrapped by |
|---|---|---|
| AllClear dataset | `external/allclear/dataset.py` | `sdp.datasets` |
| AllClear benchmark | `external/allclear/benchmark.py` | `sdp.runner` (subprocess) |
| AllClear visualization | `src/visualize_allclear.py` | `sdp.visualize` (subprocess) |
| AllClear metrics | `src/Metrics.py` | `sdp.metrics` (subprocess) |
| Proba-V dataset | `sdp/_probav.py:ProbaVDataset` | `sdp.datasets` |
| Proba-V baseline | `sdp/_probav.py:run_probav_baseline` | `sdp.runner` (in-process) |
| Proba-V metrics | `sdp/_probav.py:compute_probav_metrics` | `sdp.metrics` (in-process) |
| Proba-V visualization | `sdp/_probav.py:visualize_probav` | `sdp.visualize` (in-process) |
| Proba-V submodule runner | `probav/run_inference.py` (optional) | `sdp.runner` (subprocess) |

---

## Install

```bash
conda env create -f environment.yml
conda activate sdp-2026
pip install -e .
```

Both entry points work after install:

```bash
sdp --help
python -m sdp --help
```

---

## AllClear

### Python API

```python
import sdp

sdp.set_data_path("D:/satellite/data")

handle = sdp.load_dataset(
    "allclear",
    variant="uncrtaints",
    type="cloud_removal",
    dataset_fpath="external/metadata/datasets/test_tx3_s2-s1_100pct_1proi_local.json",
    aux_sensors=["s1"],
    aux_data=["cld_shdw", "dw"],
    tx=3,
)

handle = sdp.train(handle, device="cpu")   # runs allclear.benchmark in subprocess
sdp.metrics(handle)
sdp.visualize(handle, num=20)
```

### CLI

```bash
# Run benchmark
sdp benchmark \
    --dataset allclear \
    --variant uncrtaints \
    --device cpu \
    --dataset-fpath external/metadata/datasets/test_tx3_s2-s1_100pct_1proi_local.json \
    --aux-sensors s1 \
    --aux-data cld_shdw dw

# Compute metrics for a finished run
sdp metrics \
    --run-dir results/baseline/uncrtaints/utae/AllClear/test_tx3_s2-s1_100pct_1proi_local \
    --json external/metadata/datasets/test_tx3_s2-s1_100pct_1proi_local.json \
    --model uncrtaints

# Visualize
sdp visualize \
    --run-dir results/baseline/uncrtaints/utae/AllClear/test_tx3_s2-s1_100pct_1proi_local \
    --json external/metadata/datasets/test_tx3_s2-s1_100pct_1proi_local.json \
    --model uncrtaints --num 20

# End-to-end
sdp pipeline \
    --dataset allclear --variant uncrtaints --device cpu \
    --dataset-fpath external/metadata/datasets/test_tx3_s2-s1_100pct_1proi_local.json
```

### Supported models

`uncrtaints`, `ctgan`, `utilise`, `leastcloudy`, `mosaicing`, `dae`, `pmaa`, `diffcr`

### Limitations

- `tx` is fixed per dataset JSON; changing it requires regenerating metadata.
- Center-crop is hard-coded to 256×256 in `AllClearDataset`.
- Must be invoked from the repo root so `external/allclear` resolves as a module.
- GPU strongly recommended; CPU inference is slow.

---

## Proba-V

### On-disk layout

The loader expects the standard ESA Proba-V super-resolution challenge layout:

```
<data_path>/probav/<split>/<band>/
    imgset000001/
        LR000.png  LR001.png  ...   (128×128, uint16, 14-bit)
        QM000.png  QM001.png  ...   (128×128 binary mask, 1=clear)
        HR.png                      (384×384, train split only)
        SM.png                      (384×384 binary mask, train only)
    imgset000002/
        ...
```

`split` is `train` or `test`; `band` is `NIR` or `RED`.

### Python API

```python
import sdp

sdp.set_data_path("D:/satellite/data")

handle = sdp.load_dataset(
    "probav",
    type="super_resolution",
    variant="probav_baseline",  # or "highresnet", "deepsum", "sar_sr"
    split="train",
    band="NIR",
    scale=3,
)

handle = sdp.train(handle)     # runs baseline in-process; or submodule if available
sdp.metrics(handle)            # cPSNR / RMSE / SSIM → results/probav/.../metrics.json
sdp.visualize(handle, num=5)   # best-LR | SR | HR | diff → results/probav/.../vis/
```

### CLI

```bash
# Run the built-in baseline (clear-pixel median composite + bicubic ×3)
sdp benchmark \
    --dataset probav \
    --type super_resolution \
    --variant probav_baseline \
    --data-path D:/satellite/data \
    --split train --band NIR

# Compute cPSNR / RMSE / SSIM for a finished run
sdp metrics \
    --dataset probav \
    --run-dir results/probav/probav_baseline \
    --data-path D:/satellite/data --split train --band NIR

# Visualize SR results
sdp visualize \
    --dataset probav \
    --run-dir results/probav/probav_baseline \
    --data-path D:/satellite/data --split train --band NIR --num 10

# End-to-end convenience command
sdp probav-pipeline \
    --variant probav_baseline \
    --data-path D:/satellite/data \
    --split train --band NIR
```

### Supported models

| Model | Source | Notes |
|---|---|---|
| `probav_baseline` | `sdp/_probav.py` | Clear-pixel median composite + bicubic ×3. Built-in, no submodule needed. |
| `highresnet` | `probav/run_inference.py` | Requires initialized submodule. |
| `deepsum` | `probav/run_inference.py` | Requires initialized submodule. |
| `sar_sr` | `probav/run_inference.py` | Requires initialized submodule. |

When `probav/run_inference.py` exists and the variant is not `probav_baseline`, the runner delegates to it via:

```
python probav/run_inference.py \
    --data-path <path> --split <split> --band <band> \
    --scale 3 --model <variant> --run-dir <run_dir> --device <device>
```

### Metrics

The official challenge metric is **cPSNR** — PSNR computed only on clear pixels as defined by the HR clearance mask (`SM.png`). The built-in `compute_probav_metrics` also reports RMSE (on clear pixels) and SSIM (whole image, requires `scikit-image`).

### Limitations

- Scale is fixed at ×3 (128 → 384 pixels).
- Temporal stack length T varies per scene; models must handle variable T.
- Test split has no HR ground truth; metrics only run on train split.
- Training split has ~566 scenes; use k-fold CV for small-scale experiments.

---

## CLI reference

```
sdp describe [allclear|probav]       List models, types, limitations.
sdp set-data-path <path>             Set SDP_DATA_PATH for this process.
sdp load-dataset --dataset ...       Validate config (no training).
sdp benchmark    --dataset ...       Run inference / training.
sdp metrics      --dataset ...       Compute quality metrics.
sdp visualize    --dataset ...       Render output panels.
sdp pipeline     --dataset ...       benchmark → metrics → visualize (AllClear).
sdp probav-pipeline ...              benchmark → metrics → visualize (Proba-V).
```

Use `sdp <subcommand> --help` for the full argument list.
