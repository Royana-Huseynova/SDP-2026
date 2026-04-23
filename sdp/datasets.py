"""
Unified dataset loaders for SDP-2026.

This module wraps the AllClear dataset (in `external/allclear/dataset.py`)
and exposes a placeholder Proba-V loader so that the same `load_dataset`
entry point covers both. It is designed so that adding a new dataset only
requires a new subclass of `BaseDatasetHandle` and a registration line.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from . import config


# --------------------------------------------------------------------- #
# Dataset handle — what `load_dataset()` returns
# --------------------------------------------------------------------- #
@dataclass
class DatasetHandle:
    """
    A lightweight, framework-agnostic handle.

    It carries the resolved arguments needed by every downstream step
    (train / inference / visualize / metrics) so callers don't have to
    keep passing them around.
    """

    name: str                      # "allclear" or "probav"
    type: str                      # task type, e.g. "cloud_removal"
    variant: Optional[str] = None  # e.g. "uncrtaints", "highresnet"
    data_path: Path = field(default_factory=config.get_data_path)
    dataset_fpath: Optional[Path] = None    # AllClear: dataset JSON
    split: str = "test"
    extras: dict[str, Any] = field(default_factory=dict)
    torch_dataset: Any = None      # the underlying torch.utils.data.Dataset

    # Filled in after train()/inference() so visualize/metrics can reuse it.
    run_dir: Optional[Path] = None
    model_name: Optional[str] = None

    def as_kwargs(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "type": self.type,
            "variant": self.variant,
            "data_path": str(self.data_path),
            "split": self.split,
            "model_name": self.model_name,
            "run_dir": str(self.run_dir) if self.run_dir else None,
        }
        if self.dataset_fpath:
            d["dataset_fpath"] = str(self.dataset_fpath)
        d.update(self.extras)
        return d


# --------------------------------------------------------------------- #
# AllClear loader
# --------------------------------------------------------------------- #
def _load_allclear(
    type: str,
    variant: Optional[str],
    data_path: Path,
    dataset_fpath: Optional[Path],
    selected_rois: str | Sequence[str] = "all",
    main_sensor: str = "s2_toa",
    aux_sensors: Optional[Sequence[str]] = None,
    aux_data: Optional[Sequence[str]] = None,
    tx: int = 3,
    target_mode: str = "s2p",
    **extras: Any,
) -> DatasetHandle:
    """Wrap `external/allclear/dataset.py:AllClearDataset`."""

    if type not in config.DATASET_TYPE_MATRIX["allclear"]:
        raise ValueError(
            f"AllClear does not support type={type!r}. "
            f"Allowed: {config.DATASET_TYPE_MATRIX['allclear']}"
        )

    if dataset_fpath is None:
        dataset_fpath = config.METADATA_DIR / "test_tx3_s2-s1_100pct_1proi_local.json"
    dataset_fpath = Path(dataset_fpath)
    if not dataset_fpath.exists():
        raise FileNotFoundError(
            f"AllClear dataset JSON not found: {dataset_fpath}\n"
            f"Pass --dataset-fpath or place a JSON under {config.METADATA_DIR}."
        )

    # Make sure `external/` is importable so `from allclear ...` works.
    sys.path.insert(0, str(config.EXTERNAL_DIR))

    try:
        from allclear.dataset import AllClearDataset  # type: ignore
    except Exception as exc:  # pragma: no cover - import errors are env-specific
        raise ImportError(
            "Could not import AllClearDataset. Make sure the external/ "
            "submodule is present and the conda environment is active."
        ) from exc

    with open(dataset_fpath, "r") as f:
        meta = json.load(f)
    # Convert string keys back to int if needed (matches benchmark.py behavior).
    meta = {int(k) if str(k).isdigit() else k: v for k, v in meta.items()}

    torch_ds = AllClearDataset(
        dataset=meta,
        selected_rois=selected_rois,
        main_sensor=main_sensor,
        aux_sensors=list(aux_sensors) if aux_sensors else [],
        aux_data=list(aux_data) if aux_data else ["cld_shdw", "dw"],
        tx=tx,
        target_mode=target_mode,
    )

    return DatasetHandle(
        name="allclear",
        type=type,
        variant=variant,
        data_path=Path(data_path),
        dataset_fpath=dataset_fpath,
        torch_dataset=torch_ds,
        extras={
            "selected_rois": selected_rois,
            "main_sensor": main_sensor,
            "aux_sensors": list(aux_sensors) if aux_sensors else [],
            "aux_data": list(aux_data) if aux_data else ["cld_shdw", "dw"],
            "tx": tx,
            "target_mode": target_mode,
            **extras,
        },
    )


# --------------------------------------------------------------------- #
# Proba-V loader (placeholder skeleton)
# --------------------------------------------------------------------- #
def _load_probav(
    type: str,
    variant: Optional[str],
    data_path: Path,
    split: str = "train",
    scale: int = 3,
    band: str = "NIR",
    **extras: Any,
) -> DatasetHandle:
    """
    Skeleton loader for the ESA Proba-V super-resolution challenge.

    The on-disk layout this assumes is the standard challenge layout:

        <data_path>/probav/{train,test}/<band>/imgsetXXXXXX/
            ├─ LR000.png ... LR<NN>.png
            ├─ QM000.png ... QM<NN>.png    (LR clearance mask)
            ├─ HR.png                      (only for train)
            └─ SM.png                      (HR clearance mask)

    The actual torch Dataset implementation lives in `sdp/_probav.py` —
    add it there when wiring up real training.
    """
    if type not in config.DATASET_TYPE_MATRIX["probav"]:
        raise ValueError(
            f"Proba-V does not support type={type!r}. "
            f"Allowed: {config.DATASET_TYPE_MATRIX['probav']}"
        )

    root = Path(data_path) / "probav" / split / band
    if not root.exists():
        # Don't hard-fail in --help/dry-run flows; just warn.
        sys.stderr.write(
            f"[sdp] WARNING: Proba-V root not found at {root}. "
            f"Pass --data-path to set the dataset location.\n"
        )

    try:
        from ._probav import ProbaVDataset  # type: ignore
        torch_ds = ProbaVDataset(root=root, scale=scale, split=split)
    except Exception:
        torch_ds = None  # tolerated until the loader is implemented

    return DatasetHandle(
        name="probav",
        type=type,
        variant=variant,
        data_path=Path(data_path),
        split=split,
        torch_dataset=torch_ds,
        extras={"scale": scale, "band": band, **extras},
    )


# --------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------- #
_LOADERS = {
    "allclear": _load_allclear,
    "probav":   _load_probav,
}


def load_dataset(
    name: str,
    variant: Optional[str] = None,
    *,
    type: str = "cloud_removal",
    data_path: Optional[str | os.PathLike] = None,
    **kwargs: Any,
) -> DatasetHandle:
    """
    Public, dataset-agnostic loader.

    Parameters
    ----------
    name : {"allclear", "probav"}
        Which dataset to load.
    variant : str, optional
        Model / training variant (e.g. ``"uncrtaints"``,
        ``"uncertants"`` -> uncrtaints alias).
    type : {"cloud_removal", "super_resolution", "generation"}
        Task type the dataset will be used for.
    data_path : path-like, optional
        Root data directory. Falls back to ``$SDP_DATA_PATH`` then
        ``<repo>/data``.
    **kwargs
        Forwarded to the dataset-specific loader (e.g. ``dataset_fpath``,
        ``aux_sensors``, ``tx``, ``scale``, ``split``).

    Returns
    -------
    DatasetHandle
        Pass this to :func:`train`, :func:`inference`, :func:`visualize`,
        or :func:`metrics`.
    """
    name = name.lower().strip()
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset {name!r}. Supported: {config.SUPPORTED_DATASETS}"
        )

    # Tolerate the handwritten alias 'uncertants' -> 'uncrtaints'.
    if variant and variant.lower() in ("uncertants", "uncertainty", "uncertainties"):
        variant = "uncrtaints"

    if data_path is None:
        data_path = config.get_data_path()
    else:
        config.set_data_path(data_path)

    return _LOADERS[name](
        type=type,
        variant=variant,
        data_path=Path(data_path),
        **kwargs,
    )
