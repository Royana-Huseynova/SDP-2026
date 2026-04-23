# SDP-2026 unified API (`sdp/`)

A single Python + Terminal entry point that ties together the moving
parts of this repository:

| piece                 | lives in                              | wrapped by         |
| --------------------- | ------------------------------------- | ------------------ |
| AllClear dataset      | `external/allclear/dataset.py`        | `sdp.datasets`     |
| AllClear benchmark    | `external/allclear/benchmark.py`      | `sdp.runner`       |
| Visualization         | `src/visualize_allclear.py`           | `sdp.visualize`    |
| Metrics               | `src/Metrics.py`                      | `sdp.metrics`      |
| Proba-V loader (stub) | `sdp/_probav.py` (to be implemented)  | `sdp.datasets`     |

## Install

```bash
conda env create -f environment.yml
conda activate sdp-2026
pip install -e .
```

Once installed you can use either entry point:

```bash
sdp --help                # console script (after pip install -e .)
python -m sdp --help      # works without install
```

## Python API

```python
import sdp as sat

# Set data path once per process (or export SDP_DATA_PATH).
sat.set_data_path("D:/satellite/data")

model = sat.load_dataset(
    "allclear",
    variant="uncrtaints",
    type="cloud_removal",
    dataset_fpath="external/metadata/datasets/test_tx3_s2-s1_100pct_1proi_local.json",
    aux_sensors=["s1"],
)

sat.train(model)        # for AllClear, this is benchmark/inference
imgs = sat.inference(model)
sat.visualize(model)
sat.metrics(model)
```

## CLI

```text
sdp describe                     List datasets, types, models, limitations.
sdp set-data-path <path>         Pin SDP_DATA_PATH for this shell.
sdp load-dataset --dataset ...   Validate config without running training.
sdp benchmark    --dataset ...   Train / inference (wraps allclear.benchmark).
sdp visualize    --run-dir ...   Render side-by-side panels.
sdp metrics      --run-dir ...   Compute MAE/RMSE/PSNR/SAM/SSIM/LPIPS/FID.
sdp pipeline     --dataset ...   load -> benchmark -> metrics -> visualize.
```

Use `--help` on any subcommand to see required arguments and limits, e.g.:

```bash
sdp benchmark --help
```

## Limitations

Run `sdp describe allclear` (or `probav`) to see the full list. Highlights:

- AllClear `tx` is fixed per dataset JSON — changing it requires
  regenerating metadata.
- AllClear is hard-coded to 256x256 center crops.
- Proba-V scale is fixed at x3 (128 → 384).
- The benchmark must run from `external/`; the runner takes care of that.
- Use `KMP_DUPLICATE_LIB_OK=TRUE` on Windows to avoid OpenMP collisions
  (the runner sets this for you).
