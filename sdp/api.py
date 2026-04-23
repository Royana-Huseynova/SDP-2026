"""
Public Python API surface — what `sdp.<verb>` resolves to.

Keeps every verb in one place so tab-completion in a notebook is short
and discoverable.
"""

from __future__ import annotations

from typing import Any

import config
from datasets import load_dataset, DatasetHandle
from runner import train, inference
from visualize import visualize
from metrics import metrics


def set_data_path(path: str) -> str:
    """Set the global SDP_DATA_PATH used by all loaders."""
    return str(config.set_data_path(path))


def describe(name: str | None = None) -> dict[str, Any]:
    """
    Return a dict describing supported datasets / types / models /
    limitations. Useful in notebooks and for the `sdp describe` CLI.
    """
    if name is None:
        return {
            "datasets": list(config.SUPPORTED_DATASETS),
            "types": list(config.SUPPORTED_TYPES),
            "models": {k: list(v) for k, v in config.SUPPORTED_MODELS.items()},
            "type_matrix": {k: list(v) for k, v in config.DATASET_TYPE_MATRIX.items()},
            "limitations": {k: list(v) for k, v in config.LIMITATIONS.items()},
        }
    name = name.lower()
    if name not in config.SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset {name!r}. Supported: {config.SUPPORTED_DATASETS}"
        )
    return {
        "name": name,
        "types": list(config.DATASET_TYPE_MATRIX[name]),
        "models": [
            m for t in config.DATASET_TYPE_MATRIX[name]
            for m in config.SUPPORTED_MODELS.get(t, ())
        ],
        "limitations": list(config.LIMITATIONS.get(name, ())),
    }


__all__ = [
    "load_dataset",
    "DatasetHandle",
    "train",
    "inference",
    "visualize",
    "metrics",
    "set_data_path",
    "describe",
]
