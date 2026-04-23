"""
Central configuration for the SDP-2026 unified API.

All paths are resolved relative to the repository root (the directory
containing this `sdp/` package). Paths are the single source of truth —
every other module imports from here rather than hard-coding.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------- #
# Repository layout
# --------------------------------------------------------------------- #
# sdp/config.py  ->  sdp/  ->  <REPO_ROOT>
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

EXTERNAL_DIR: Path = REPO_ROOT / "external"
RESULTS_DIR:  Path = REPO_ROOT / "results"
SCRIPTS_DIR:  Path = REPO_ROOT / "scripts"
SRC_DIR:      Path = REPO_ROOT / "src"
METADATA_DIR: Path = EXTERNAL_DIR / "metadata" / "datasets"


# --------------------------------------------------------------------- #
# User-overridable data path
# --------------------------------------------------------------------- #
# Users can:
#   1. export SDP_DATA_PATH=/path/to/data     (env var), or
#   2. call sdp.set_data_path("/path/to/data") at runtime.
# The CLI `--data-path` flag also feeds into this.
DATA_PATH: Path = Path(os.environ.get("SDP_DATA_PATH", REPO_ROOT / "data"))


# --------------------------------------------------------------------- #
# Supported datasets / task types / models
# --------------------------------------------------------------------- #
SUPPORTED_DATASETS = ("allclear", "probav")

SUPPORTED_TYPES = (
    "cloud_removal",   # AllClear
    "super_resolution",  # Proba-V
    "generation",      # diffusion / GAN baselines (experimental)
)

# Maps high-level task types to the models that can handle them.
SUPPORTED_MODELS: dict[str, tuple[str, ...]] = {
    "cloud_removal": (
        "uncrtaints",
        "ctgan",
        "utilise",
        "leastcloudy",
        "mosaicing",
        "dae",
        "pmaa",
        "diffcr",
    ),
    "super_resolution": (
        "probav_baseline",
        "highresnet",
        "deepsum",
        "sar_sr",
    ),
    "generation": (
        "diffcr",
        "ctgan",
    ),
}

# Which dataset provides which task types.
DATASET_TYPE_MATRIX: dict[str, tuple[str, ...]] = {
    "allclear": ("cloud_removal", "generation"),
    "probav":   ("super_resolution",),
}


# --------------------------------------------------------------------- #
# Limitations that the CLI surfaces via --help / describe()
# --------------------------------------------------------------------- #
LIMITATIONS: dict[str, tuple[str, ...]] = {
    "allclear": (
        "tx (temporal length) is fixed per dataset JSON; changing it requires regenerating metadata.",
        "Center-crop is hard-coded to 256x256 in AllClearDataset.",
        "Only target-mode 's2p' and 's2s' are supported.",
        "GPU strongly recommended — CPU inference is supported but slow.",
        "Must be launched from repo root so that `external/allclear` resolves as a module.",
    ),
    "probav": (
        "Dataset is 128x128 LR -> 384x384 HR (fixed scale x3).",
        "Temporal stacks vary in length; models should handle variable T.",
        "Clearance masks are 1-bit and must be aligned per LR frame.",
        "Training split has ~566 scenes; use k-fold CV for small-scale experiments.",
    ),
}


def set_data_path(path: str | os.PathLike) -> Path:
    """Update the global DATA_PATH used by dataset loaders."""
    global DATA_PATH
    DATA_PATH = Path(path).expanduser().resolve()
    os.environ["SDP_DATA_PATH"] = str(DATA_PATH)
    return DATA_PATH


def get_data_path() -> Path:
    """Current data path, honoring runtime env var changes."""
    return Path(os.environ.get("SDP_DATA_PATH", DATA_PATH))
