"""
SDP-2026 — Satellite Data Processing unified API.

A single Python entry point that ties together the separate pieces of the
repository (AllClear benchmark engine, Proba-V super-resolution loader,
visualization and metrics) behind one dataset-agnostic surface.

Notebook usage
--------------
    >>> import sdp as sat
    >>> model = sat.load_dataset("allclear", variant="uncrtaints",
    ...                          type="cloud_removal")
    >>> sat.train(model)           # (stub — calls benchmark in test mode)
    >>> preds = sat.inference(model)
    >>> sat.visualize(model)
    >>> sat.metrics(model)

Terminal usage
--------------
    $ python -m sdp --help
    $ python -m sdp load-dataset allclear --type cloud_removal \
          --dataset-fpath external/metadata/datasets/test_tx3_s2-s1_100pct_1proi_local.json
    $ python -m sdp benchmark --model uncrtaints --device cpu ...
    $ python -m sdp visualize --run-dir ... --json ...
    $ python -m sdp metrics   --run-dir ... --json ...
"""

from .config import (
    REPO_ROOT,
    EXTERNAL_DIR,
    RESULTS_DIR,
    METADATA_DIR,
    SUPPORTED_DATASETS,
    SUPPORTED_TYPES,
    SUPPORTED_MODELS,
)
from .api import (
    load_dataset,
    train,
    inference,
    visualize,
    metrics,
    set_data_path,
    describe,
)

__all__ = [
    "REPO_ROOT",
    "EXTERNAL_DIR",
    "RESULTS_DIR",
    "METADATA_DIR",
    "SUPPORTED_DATASETS",
    "SUPPORTED_TYPES",
    "SUPPORTED_MODELS",
    "load_dataset",
    "train",
    "inference",
    "visualize",
    "metrics",
    "set_data_path",
    "describe",
]

__version__ = "0.1.0"
